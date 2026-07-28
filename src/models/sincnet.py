import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class SincConv1d(nn.Module):
    """Parameterised Sinc-convolution 1D layer.

    Coded faithfully following the original SincNet paper:
    "Speaker Recognition from Raw Waveform with SincNet" (Mirco Ravanelli et al.).
    """

    def __init__(
        self,
        out_channels: int,
        kernel_size: int,
        sample_rate: int = 16000,
        min_low_hz: float = 50.0,
        min_band_hz: float = 50.0,
        stride: int = 1,
        padding: int = 0
    ):
        """Initializes SincConv1d parameters.

        Args:
            out_channels: Number of learnable bandpass filters.
            kernel_size: Length of filter kernel (must be odd).
            sample_rate: Sampling frequency of input audio.
            min_low_hz: Minimum low cutoff frequency.
            min_band_hz: Minimum bandwidth frequency.
            stride: Stride of the 1D convolution.
            padding: Padding of the 1D convolution.
        """
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError(f"SincConv1d requires odd kernel_size, got {kernel_size}")

        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.sample_rate = sample_rate
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz
        self.stride = stride
        self.padding = padding

        # Initialize cutoff frequencies linearly on a mel-like scale
        low_hz = 30.0
        high_hz = sample_rate / 2 - (min_low_hz + min_band_hz)
        
        hz = np.linspace(low_hz, high_hz, out_channels)
        self.low_hz_ = nn.Parameter(torch.from_numpy(hz).float())
        self.band_hz_ = nn.Parameter(torch.ones(out_channels).float() * min_band_hz)

        # Register time grid vector for right-half filter (t > 0)
        t_right = torch.linspace(1, (kernel_size - 1) // 2, steps=(kernel_size - 1) // 2) / sample_rate
        self.register_buffer("t_right", t_right)

        # Register Hamming window
        window = 0.54 - 0.46 * torch.cos(2 * np.pi * torch.arange(kernel_size) / (kernel_size - 1))
        self.register_buffer("window", window.float())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Convolves raw waveform input with parameterized bandpass filters.

        Args:
            x: Raw input waveforms of shape (batch, 1, samples).

        Returns:
            torch.Tensor: Convolved output of shape (batch, out_channels, time).
        """
        min_low = self.min_low_hz / self.sample_rate
        min_band = self.min_band_hz / self.sample_rate

        f1 = torch.abs(self.low_hz_) / self.sample_rate + min_low
        f2 = f1 + torch.abs(self.band_hz_) / self.sample_rate + min_band

        # Time grids (t > 0)
        n = 2 * np.pi * self.t_right  # shape: (half_len,)
        
        f1_t = f1.unsqueeze(1)  # shape: (out_channels, 1)
        f2_t = f2.unsqueeze(1)  # shape: (out_channels, 1)

        # Compute filter weights: (sin(2pi f2 t) - sin(2pi f1 t)) / (pi t)
        # Using 2 * (sin(f2 * n) - sin(f1 * n)) / n since n = 2 * pi * t
        right_side = 2 * (torch.sin(f2_t * n) - torch.sin(f1_t * n)) / n
        center = 2 * (f2_t - f1_t)
        left_side = torch.flip(right_side, dims=[1])
        
        filters = torch.cat([left_side, center, right_side], dim=1)  # shape: (out_channels, kernel_size)
        filters = filters * self.window  # Apply Hamming window smoothing

        # Convolve input
        return F.conv1d(
            x,
            filters.unsqueeze(1),
            stride=self.stride,
            padding=self.padding
        )


class SincNet(nn.Module):
    """SincNet feature extraction block."""

    def __init__(self, out_channels: int = 80, kernel_size: int = 251, sample_rate: int = 16000):
        """Initializes SincNet wrapper block."""
        super().__init__()
        # SincConv1d layer
        self.sinc_conv = SincConv1d(
            out_channels=out_channels,
            kernel_size=kernel_size,
            sample_rate=sample_rate,
            stride=1,
            padding=kernel_size // 2
        )
        self.ln = nn.LayerNorm([out_channels, 1])  # LayerNorm across filterbanks
        self.pool = nn.MaxPool1d(kernel_size=3, stride=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies SincConv1d, LayerNorm, LeakyReLU, and MaxPool1d.

        Args:
            x: Raw input waveforms of shape (batch, 1, samples).

        Returns:
            torch.Tensor: Feature map of shape (batch, out_channels, time_reduced).
        """
        out = self.sinc_conv(x)
        
        # Layer Normalization expects shapes like (batch, channels, time)
        # We normalize along channel axis
        # LayerNorm in PyTorch requires normalizing dimensions. Let's do instance norm-like scaling:
        mean = out.mean(dim=1, keepdim=True)
        std = out.std(dim=1, keepdim=True)
        out = (out - mean) / (std + 1e-8)
        
        out = F.leaky_relu(out, negative_slope=0.2)
        return self.pool(out)
