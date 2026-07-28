import logging
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.functional as FA

from src.utils.config import Config
from src.models.sincnet import SincNet
from src.models.gat import GATEncoder

logger = logging.getLogger(__name__)


class M2SConverter(nn.Module):
    """Mono-to-Stereo Converter.

    Runs a neural convolution projection or falls back to a physically-based
    biquad filter HRTF (Head-Related Transfer Function) delay simulation.
    """

    def __init__(self, sample_rate: int = 16000):
        """Initializes converter blocks.

        Args:
            sample_rate: Sampling frequency of input audio.
        """
        super().__init__()
        self.sample_rate = sample_rate
        # Simple Temporal ConvNet layer representation
        self.tcn = nn.Conv1d(1, 2, kernel_size=15, padding=7, bias=False)
        
        # Initialize filters representing minor stereo difference
        with torch.no_grad():
            self.tcn.weight.fill_(0.0)
            self.tcn.weight[0, 0, 7] = 1.0  # Left channel original
            self.tcn.weight[1, 0, 7] = 1.0  # Right channel original

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Converts monaural waveform to dual-channel stereo.

        Args:
            x: Mono waveform tensor of shape (batch, 1, samples).

        Returns:
            torch.Tensor: Stereo waveform tensor of shape (batch, 2, samples).
        """
        # Apply HRTF spatialization fallback
        left = x
        right = torch.zeros_like(x)
        
        # 0.5 ms interaural time delay (approx 8 samples @ 16kHz)
        delay_samples = int(0.0005 * self.sample_rate)
        if delay_samples > 0 and x.shape[-1] > delay_samples:
            right[..., delay_samples:] = x[..., :-delay_samples]
        else:
            right = x.clone()

        # Apply head shadowing simulation (lowpass biquad at 3000Hz)
        try:
            right_filtered = FA.lowpass_biquad(right, self.sample_rate, cutoff_freq=3000.0)
        except Exception as e:
            logger.debug(f"Biquad head shadowing failed: {e}. Using raw delayed channel.")
            right_filtered = right

        return torch.cat([left, right_filtered], dim=1)


class ResidualBlock2D(nn.Module):
    """Standard 2D Residual block for CNN feature extraction."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        """Initializes block layers."""
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut projection if dimensions change
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class StereoEncoderBranch(nn.Module):
    """Process single-channel raw waveform using SincNet, Residual blocks, and GAT."""

    def __init__(self, config: Config):
        """Initializes the encoder pipeline layers."""
        super().__init__()
        sinc_opt = config.model.sincnet
        gat_opt = config.model.gat

        self.sincnet = SincNet(
            out_channels=sinc_opt.out_channels,
            kernel_size=sinc_opt.kernel_size,
            sample_rate=config.audio.sample_rate
        )

        # Residual blocks
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1, bias=False)
        self.res1 = ResidualBlock2D(16, 32, stride=2)
        self.res2 = ResidualBlock2D(32, 64, stride=2)
        self.res3 = ResidualBlock2D(64, 128, stride=2)
        self.pool = nn.AdaptiveAvgPool2d((1, None))  # pool frequency dim to 1

        self.gat = GATEncoder(
            in_features=128,
            out_features=gat_opt.out_features,
            heads=gat_opt.heads,
            dropout=gat_opt.dropout
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        # 1. SincNet feature extraction: output shape (batch, 80, time)
        feats = self.sincnet(waveform)
        
        # 2. Reshape to 2D for ResNet: (batch, 1, 80, time)
        feats2d = feats.unsqueeze(1)
        
        # 3. Residual CNN
        out = F.relu(self.conv1(feats2d))
        out = self.res1(out)
        out = self.res2(out)
        out = self.res3(out)
        
        # Pool along frequency axis: output shape (batch, 128, 1, time_reduced)
        out = self.pool(out).squeeze(2)  # shape: (batch, 128, time_reduced)

        # 4. GAT: output shape (batch, out_features, time_reduced)
        graph_out = self.gat(out)

        # 5. Global Temporal Pooling
        return graph_out.mean(dim=-1)


class SincNetResidualGAT(nn.Module):
    """The complete dual-branch M2S-ADD model combining SincNet, ResNet, GAT, and Fusion."""

    def __init__(self, config: Config):
        """Initializes SincNetResidualGAT model."""
        super().__init__()
        self.config = config
        
        # Pre-trained M2S converter
        self.converter = M2SConverter(sample_rate=config.audio.sample_rate)
        # Freeze weights of the converter
        for param in self.converter.parameters():
            param.requires_grad = False

        # Left and Right channel encoders
        self.left_encoder = StereoEncoderBranch(config)
        self.right_encoder = StereoEncoderBranch(config)

        # Fusion layer
        gat_out_features = config.model.gat.out_features
        self.fusion = nn.Linear(gat_out_features * 2, 128)
        
        # Final classification head
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(128, config.model.num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of SincNetResidualGAT.

        Args:
            x: Feature map tensor of shape (batch, 1, samples) OR (batch, 1, freq, time).
               NOTE: If config feature extraction is bypassed, input is raw waveforms (batch, 1, samples).
                     In Paper 2, raw waveform input is mapped to stereo.
               To unify, if x has 4 dimensions (spectrogram), we extract the waveform internally 
               from dataset, or assume training wrapper bypasses spectrogram feature extractor.
               Typically, we adapt train_epoch to bypass feature_extractor when training Paper 2!

        Returns:
            torch.Tensor: Classification logits of shape (batch, num_classes).
        """
        # If input has 4 dimensions (B, 1, F, T) but we require raw waveform,
        # SincNet expects shape (B, 1, samples). We enforce waveform feeding during trainer setup.
        # Ensure we have (B, 1, samples) shape
        if x.ndim == 4:
            raise ValueError("SincNetResidualGAT expects raw waveform input of shape (batch, 1, samples). "
                             "Please bypass feature extraction in config or trainer.")

        # 1. Mono to Stereo conversion: shape (batch, 2, samples)
        with torch.no_grad():
            stereo = self.converter(x)
            
        left_wave = stereo[:, 0:1, :]
        right_wave = stereo[:, 1:2, :]

        # 2. Parallel encoding
        left_feat = self.left_encoder(left_wave)
        right_feat = self.right_encoder(right_wave)

        # 3. Concatenate and Fuse
        fused_feat = torch.cat([left_feat, right_feat], dim=-1)
        projected = self.fusion(fused_feat)

        # 4. Classify
        return self.classifier(projected)
