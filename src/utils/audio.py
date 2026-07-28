import logging
from pathlib import Path
from typing import Optional, Tuple
import torch
import torchaudio

logger = logging.getLogger(__name__)


def load_audio(
    file_path: Path | str, 
    target_sr: Optional[int] = None
) -> Tuple[torch.Tensor, int]:
    """Loads an audio file, converts it to mono, and resamples if target_sr is set.

    Args:
        file_path: Path to the audio file.
        target_sr: Optional target sample rate.

    Returns:
        Tuple[torch.Tensor, int]: Waveform tensor of shape (1, num_samples)
                                  and the sample rate.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    try:
        waveform, sr = torchaudio.load(str(file_path))
    except Exception as e:
        logger.debug(f"torchaudio.load failed for {file_path}, trying soundfile: {e}")
        try:
            import soundfile as sf
            data, sr = sf.read(str(file_path))
            waveform = torch.from_numpy(data).float()
            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)
            else:
                waveform = waveform.transpose(0, 1)
        except Exception as sf_err:
            logger.error(f"Failed to load audio {file_path} with both torchaudio and soundfile: {sf_err}")
            raise ValueError(f"Corrupted or invalid audio file: {file_path}") from sf_err

    if waveform.numel() == 0:
        raise ValueError(f"Empty audio file: {file_path}")

    if torch.isnan(waveform).any() or torch.isinf(waveform).any():
        logger.warning(f"Audio contains NaN or Inf values: {file_path}. Sanitizing.")
        waveform = torch.nan_to_num(waveform, nan=0.0, posinf=1.0, neginf=-1.0)

    # Convert to mono by averaging across channel dimension
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Resample if needed
    if target_sr is not None and sr != target_sr:
        waveform = resample_audio(waveform, sr, target_sr)
        sr = target_sr

    return waveform, sr


def resample_audio(
    waveform: torch.Tensor, 
    orig_sr: int, 
    target_sr: int
) -> torch.Tensor:
    """Resamples the waveform to target sample rate.

    Args:
        waveform: Waveform tensor of shape (channels, num_samples).
        orig_sr: Original sample rate.
        target_sr: Target sample rate.

    Returns:
        torch.Tensor: Resampled waveform.
    """
    if orig_sr == target_sr:
        return waveform
    resampler = torchaudio.transforms.Resample(orig_freq=orig_sr, new_freq=target_sr)
    return resampler(waveform)


def pad_crop_audio(
    waveform: torch.Tensor,
    sample_rate: int,
    target_duration: float,
    padding_type: str = "wrap",
    crop_type: str = "random"
) -> torch.Tensor:
    """Pads or crops waveform to match exactly target_duration.

    Args:
        waveform: Waveform tensor of shape (1, num_samples).
        sample_rate: The sample rate of the audio.
        target_duration: Target duration in seconds.
        padding_type: How to pad ("zero" or "wrap").
        crop_type: How to crop ("random" or "center").

    Returns:
        torch.Tensor: Adjusted waveform of shape (1, target_len).
    """
    target_len = int(target_duration * sample_rate)
    current_len = waveform.shape[1]

    if current_len == target_len:
        return waveform

    if current_len > target_len:
        # Crop
        if crop_type == "random":
            max_start = current_len - target_len
            start = torch.randint(0, max_start, (1,)).item()
        else:  # center
            start = (current_len - target_len) // 2
        return waveform[:, start : start + target_len]

    # Pad
    if padding_type == "zero":
        pad_len = target_len - current_len
        return torch.nn.functional.pad(waveform, (0, pad_len))
    
    # Wrap (repeat waveform)
    repeats = (target_len // current_len) + 1
    return waveform.repeat(1, repeats)[:, :target_len]


def normalize_audio(waveform: torch.Tensor) -> torch.Tensor:
    """Normalizes waveform to peak amplitude in range [-1.0, 1.0].

    Args:
        waveform: Waveform tensor of shape (1, num_samples).

    Returns:
        torch.Tensor: Normalized waveform.
    """
    max_val = torch.max(torch.abs(waveform))
    if max_val > 0:
        return waveform / max_val
    return waveform
