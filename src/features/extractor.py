import logging
from pathlib import Path
from typing import Optional, Union
import torch
import torch.nn as nn
import torchaudio

logger = logging.getLogger(__name__)


def make_double_sided(spectrogram: torch.Tensor) -> torch.Tensor:
    """Converts a single-sided spectrogram to a double-sided centered spectrogram.

    Places DC (0 Hz) in the center and Nyquist (high frequencies) at the edges.

    Args:
        spectrogram: Tensor of shape (..., freq_bins, time_steps).

    Returns:
        torch.Tensor: Double-sided centered spectrogram of shape 
                      (..., 2 * freq_bins - 1, time_steps).
    """
    flipped = torch.flip(spectrogram, dims=[-2])
    num_bins = spectrogram.shape[-2]
    
    # Exclude DC bin from the flipped tensor to avoid duplication at the center
    # narrow(-2, 0, num_bins - 1) slices along the freq axis from index 0 to num_bins-1
    sliced_flipped = flipped.narrow(-2, 0, num_bins - 1)
    
    return torch.cat([sliced_flipped, spectrogram], dim=-2)


class FeatureExtractor(nn.Module):
    """Module for robust speech feature extraction (LogSpec, LFCC, Double-sided Spec)."""

    def __init__(
        self,
        feature_type: str = "lfcc",
        sample_rate: int = 16000,
        n_fft: int = 512,
        win_length: int = 400,
        hop_length: int = 160,
        n_lfcc: int = 60,
        n_mels: int = 80,
        f_min: float = 0.0,
        f_max: Optional[float] = 8000.0,
    ):
        """Initializes feature extraction modules.

        Args:
            feature_type: "spectrogram", "lfcc", or "double_spectrogram".
            sample_rate: Sampling rate of input audio.
            n_fft: FFT window size.
            win_length: Window analysis length.
            hop_length: Frame advance size.
            n_lfcc: Number of LFCC coefficients.
            n_mels: Number of Mel/Linear filterbanks.
            f_min: Minimum analysis frequency.
            f_max: Maximum analysis frequency.
        """
        super().__init__()
        self.feature_type = feature_type.lower()
        self.sample_rate = sample_rate

        # Base spectrogram transform
        self.spectrogram_transform = torchaudio.transforms.Spectrogram(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            power=2.0
        )

        # Amplitude-to-Decibel converter
        self.db_transform = torchaudio.transforms.AmplitudeToDB(top_db=80.0)

        # LFCC transform
        speckwargs = {
            "n_fft": n_fft,
            "win_length": win_length,
            "hop_length": hop_length,
            "power": 2.0
        }
        self.lfcc_transform = torchaudio.transforms.LFCC(
            sample_rate=sample_rate,
            n_lfcc=n_lfcc,
            speckwargs=speckwargs,
            n_filter=n_mels,
            f_min=f_min,
            f_max=f_max
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Extracts features from the raw input waveform.

        Args:
            waveform: Raw audio waveform tensor of shape (..., samples).

        Returns:
            torch.Tensor: Feature map of shape (..., features, time).
        """
        if self.feature_type == "lfcc":
            return self.lfcc_transform(waveform)

        # Compute power spectrogram
        spec = self.spectrogram_transform(waveform)
        log_spec = self.db_transform(spec)

        if self.feature_type == "double_spectrogram":
            log_spec = make_double_sided(log_spec)

        return log_spec

    def normalize(self, feature: torch.Tensor) -> torch.Tensor:
        """Applies instance-level Cepstral Mean and Variance Normalization (CMVN).

        Normalizes features across the time dimension.

        Args:
            feature: Unnormalized feature map.

        Returns:
            torch.Tensor: Normalized feature map.
        """
        mean = feature.mean(dim=-1, keepdim=True)
        std = feature.std(dim=-1, keepdim=True)
        return (feature - mean) / (std + 1e-8)

    def extract_and_cache(
        self,
        waveform: torch.Tensor,
        audio_id: str,
        cache_dir: Optional[Union[Path, str]] = None,
        normalize_feature: bool = True
    ) -> torch.Tensor:
        """Handles on-disk feature extraction and caching.

        Args:
            waveform: Raw audio waveform tensor.
            audio_id: Unique string identifier for the audio.
            cache_dir: Optional path to cache directories.
            normalize_feature: Whether to run CMVN.

        Returns:
            torch.Tensor: Extracted and potentially cached feature tensor.
        """
        if cache_dir is not None:
            cache_dir = Path(cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path = cache_dir / f"{audio_id}_{self.feature_type}.pt"
            
            if cache_path.exists():
                try:
                    return torch.load(cache_path, weights_only=True)
                except Exception as e:
                    logger.warning(f"Failed to read cache {cache_path}: {e}. Recomputing.")

        # Recompute
        feature = self.forward(waveform)
        if normalize_feature:
            feature = self.normalize(feature)

        if cache_dir is not None:
            try:
                torch.save(feature, cache_path)
            except Exception as e:
                logger.warning(f"Failed to write cache {cache_path}: {e}")

        return feature
