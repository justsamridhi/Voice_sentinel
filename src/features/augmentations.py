from io import BytesIO
import logging
from pathlib import Path
import random
from typing import Optional, Union
import numpy as np
import torch
import torchaudio

logger = logging.getLogger(__name__)


def compress_audio(
    waveform: torch.Tensor,
    sample_rate: int,
    codec: str = "mp3",
    bitrate: str = "64k"
) -> torch.Tensor:
    """In-memory lossy audio compression augmentation using pydub.

    Args:
        waveform: Waveform tensor of shape (1, samples).
        sample_rate: Audio sampling rate.
        codec: Targeting codec ("mp3", "ogg").
        bitrate: Codec bit rate (e.g. "64k", "32k").

    Returns:
        torch.Tensor: Compressed and re-loaded waveform.
    """
    try:
        from pydub import AudioSegment
    except ImportError:
        logger.warning("pydub not installed. Skipping compression augmentation.")
        return waveform

    # Scale to 16-bit PCM integer values
    samples = waveform.squeeze(0).cpu().numpy()
    samples_int = (np.clip(samples, -1.0, 1.0) * 32767.0).astype(np.int16)

    try:
        # Create AudioSegment from raw PCM bytes
        audio_segment = AudioSegment(
            data=samples_int.tobytes(),
            sample_width=2,
            frame_rate=sample_rate,
            channels=1
        )
        
        # Export to in-memory byte buffer
        buffer = BytesIO()
        audio_segment.export(buffer, format=codec, bitrate=bitrate)
        buffer.seek(0)

        # Load back
        compressed_segment = AudioSegment.from_file(buffer, format=codec)
        compressed_samples = np.array(compressed_segment.get_array_of_samples(), dtype=np.float32) / 32767.0
        
        # Format back as PyTorch tensor
        compressed_tensor = torch.from_numpy(compressed_samples).unsqueeze(0)
        
        # Ensure identical length
        if compressed_tensor.shape[1] != waveform.shape[1]:
            # Crop/pad if format changes duration slightly
            from src.utils.audio import pad_crop_audio
            compressed_tensor = pad_crop_audio(compressed_tensor, sample_rate, waveform.shape[1] / sample_rate)
            
        return compressed_tensor

    except Exception as e:
        logger.warning(f"In-memory compression failed (FFmpeg missing?): {e}. Returning original.")
        return waveform


def apply_channel_effects(
    waveform: torch.Tensor,
    sample_rate: int,
    ir_path: Optional[Union[Path, str]] = None
) -> torch.Tensor:
    """Simulates channel distortions or room impulse response (RIR) convolutions.

    Args:
        waveform: Waveform tensor of shape (1, samples).
        sample_rate: Audio sampling rate.
        ir_path: Optional path to an impulse response file.

    Returns:
        torch.Tensor: Distorted audio waveform.
    """
    if ir_path is not None:
        ir_path = Path(ir_path)
        if ir_path.exists():
            try:
                from src.utils.audio import load_audio, normalize_audio
                # Load impulse response
                rir, rir_sr = load_audio(ir_path, target_sr=sample_rate)
                rir = normalize_audio(rir)
                
                # Convolve (using torchaudio's fftconvolve)
                convolved = torchaudio.functional.fftconvolve(waveform, rir)
                
                # Settle length back to original size
                return convolved[:, :waveform.shape[1]]
            except Exception as e:
                logger.warning(f"Impulse response convolution failed: {e}. Falling back to filters.")

    # Telephony filter bandpass simulation ([100-400Hz] to [3000-4000Hz])
    try:
        low_cut = random.uniform(100.0, 400.0)
        high_cut = random.uniform(3000.0, 4000.0)
        
        # Apply highpass and lowpass filters
        filtered = torchaudio.functional.highpass_biquad(waveform, sample_rate, low_cut)
        filtered = torchaudio.functional.lowpass_biquad(filtered, sample_rate, high_cut)
        return filtered
    except Exception as e:
        logger.warning(f"Filter channel simulation failed: {e}. Returning original.")
        return waveform


def apply_spec_average(
    spectrogram: torch.Tensor,
    time_mask_max: int = 30,
    freq_mask_max: int = 15,
    num_time_masks: int = 2,
    num_freq_masks: int = 2
) -> torch.Tensor:
    """Masks time and frequency bands with the average amplitude of the spectrogram.

    Proposed in Ariel Cohen et al. (Paper 4) to replace standard SpecAugment zero-out.

    Args:
        spectrogram: Spectrogram tensor of shape (..., freq_bins, time_steps).
        time_mask_max: Maximum width of time masks.
        freq_mask_max: Maximum width of frequency masks.
        num_time_masks: Number of time masks to apply.
        num_freq_masks: Number of frequency masks to apply.

    Returns:
        torch.Tensor: SpecAverage augmented spectrogram.
    """
    spec = spectrogram.clone()
    avg_val = spec.mean()

    num_freq = spec.shape[-2]
    num_time = spec.shape[-1]

    # Apply frequency masks
    for _ in range(num_freq_masks):
        w = random.randint(1, freq_mask_max)
        f0 = random.randint(0, num_freq - w)
        spec[..., f0 : f0 + w, :] = avg_val

    # Apply time masks
    for _ in range(num_time_masks):
        t = random.randint(1, time_mask_max)
        t0 = random.randint(0, num_time - t)
        spec[..., :, t0 : t0 + t] = avg_val

    return spec
