import tempfile
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
import torchaudio
from src.utils.audio import load_audio, resample_audio, pad_crop_audio, normalize_audio
from src.utils.visualization import plot_waveform, plot_fft, plot_spectrogram


@pytest.fixture
def synthetic_wav():
    """Generates a 1-second, 440Hz synthetic sine wave at 16000Hz sample rate."""
    import soundfile as sf
    sr = 16000
    duration = 1.0
    t = torch.linspace(0, duration, int(sr * duration))
    # 440 Hz sine wave, shape (1, 16000)
    waveform = 0.5 * torch.sin(2 * np.pi * 440.0 * t).unsqueeze(0)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        temp_path = Path(temp_file.name)

    # Use soundfile to save, as torchaudio.save may try to load torchcodec on Windows
    sf.write(str(temp_path), waveform.squeeze(0).numpy(), sr)
    yield temp_path, sr, waveform
    if temp_path.exists():
        temp_path.unlink()



def test_load_audio(synthetic_wav):
    temp_path, sr, orig_waveform = synthetic_wav
    waveform, loaded_sr = load_audio(temp_path, target_sr=sr)
    assert loaded_sr == sr
    assert waveform.shape == orig_waveform.shape
    assert torch.allclose(waveform, orig_waveform, atol=1e-4)


def test_resample_audio(synthetic_wav):
    temp_path, sr, orig_waveform = synthetic_wav
    target_sr = 8000
    resampled = resample_audio(orig_waveform, sr, target_sr)
    assert resampled.shape[1] == orig_waveform.shape[1] // 2


def test_pad_crop_audio(synthetic_wav):
    temp_path, sr, orig_waveform = synthetic_wav

    # Test padding (zero)
    padded_zero = pad_crop_audio(orig_waveform, sr, target_duration=1.5, padding_type="zero")
    assert padded_zero.shape[1] == 24000
    assert torch.all(padded_zero[:, 16000:] == 0)

    # Test padding (wrap)
    padded_wrap = pad_crop_audio(orig_waveform, sr, target_duration=1.5, padding_type="wrap")
    assert padded_wrap.shape[1] == 24000
    assert torch.allclose(padded_wrap[:, 16000:], orig_waveform[:, :8000], atol=1e-4)

    # Test cropping (center)
    cropped_center = pad_crop_audio(orig_waveform, sr, target_duration=0.5, crop_type="center")
    assert cropped_center.shape[1] == 8000
    assert torch.allclose(cropped_center, orig_waveform[:, 4000:12000], atol=1e-4)

    # Test cropping (random)
    cropped_random = pad_crop_audio(orig_waveform, sr, target_duration=0.5, crop_type="random")
    assert cropped_random.shape[1] == 8000


def test_normalize_audio():
    waveform = torch.tensor([[0.5, -0.2, 0.1, -0.8]])
    normalized = normalize_audio(waveform)
    assert torch.allclose(normalized, torch.tensor([[0.625, -0.25, 0.125, -1.0]]))
    assert torch.max(torch.abs(normalized)) == 1.0


def test_visualization_plots(synthetic_wav):
    temp_path, sr, orig_waveform = synthetic_wav

    # Test waveform plotting
    fig_wav = plot_waveform(orig_waveform, sr)
    assert isinstance(fig_wav, plt.Figure)
    plt.close(fig_wav)

    # Test FFT plotting
    fig_fft = plot_fft(orig_waveform, sr)
    assert isinstance(fig_fft, plt.Figure)
    plt.close(fig_fft)

    # Test spectrogram plotting (mock spectrogram input of shape (60, 100))
    mock_spec = torch.randn(60, 100)
    fig_spec = plot_spectrogram(mock_spec)
    assert isinstance(fig_spec, plt.Figure)
    plt.close(fig_spec)
