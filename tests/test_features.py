import tempfile
from pathlib import Path
import pytest
import torch

from src.features.extractor import FeatureExtractor, make_double_sided


def test_make_double_sided():
    # Shape: (batch, channels, freq_bins, time_steps)
    # Mock single-sided spectrogram of shape (1, 1, 5, 4)
    # Frequencies 0 to 4 are represented as 0.0 to 4.0
    spec = torch.arange(5).view(1, 1, 5, 1).repeat(1, 1, 1, 4).float()
    
    double_sided = make_double_sided(spec)
    # Expected shape along freq axis is (2 * 5) - 1 = 9
    assert double_sided.shape == (1, 1, 9, 4)
    
    # Flipped: 4, 3, 2, 1 (up to index 3)
    # Original: 0, 1, 2, 3, 4 (index 4 to 8)
    expected_order = [4.0, 3.0, 2.0, 1.0, 0.0, 1.0, 2.0, 3.0, 4.0]
    
    for i, val in enumerate(expected_order):
        assert torch.allclose(double_sided[0, 0, i], torch.tensor(val))


def test_spectrogram_extractor():
    extractor = FeatureExtractor(feature_type="spectrogram", sample_rate=16000)
    
    # 1 second of white noise at 16kHz
    waveform = torch.randn(1, 16000)
    feature = extractor(waveform)
    
    # With n_fft=512, standard n_bins = n_fft // 2 + 1 = 257
    # hop_length = 160 -> frames = 16000 / 160 + 1 = 101 frames
    assert feature.ndim == 3
    assert feature.shape[-2] == 257
    assert feature.shape[-1] == 101


def test_lfcc_extractor():
    extractor = FeatureExtractor(feature_type="lfcc", sample_rate=16000, n_lfcc=60)
    waveform = torch.randn(1, 16000)
    feature = extractor(waveform)
    
    assert feature.ndim == 3
    assert feature.shape[-2] == 60  # n_lfcc
    assert feature.shape[-1] == 101


def test_double_spectrogram_extractor():
    extractor = FeatureExtractor(feature_type="double_spectrogram", sample_rate=16000)
    waveform = torch.randn(1, 16000)
    feature = extractor(waveform)
    
    # expected frequency bins: 257 * 2 - 1 = 513
    assert feature.shape[-2] == 513
    assert feature.shape[-1] == 101


def test_normalization():
    extractor = FeatureExtractor(feature_type="spectrogram")
    feature = torch.randn(1, 257, 100) * 10.0 + 5.0  # arbitrary mean and std
    
    normalized = extractor.normalize(feature)
    
    # Mean and std should be ~0.0 and ~1.0 respectively across the time axis (dim=-1)
    mean = normalized.mean(dim=-1)
    std = normalized.std(dim=-1)
    
    assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-5)
    assert torch.allclose(std, torch.ones_like(std), atol=1e-4)


def test_disk_caching():
    extractor = FeatureExtractor(feature_type="lfcc", sample_rate=16000, n_lfcc=40)
    waveform = torch.randn(1, 16000)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        cache_dir = Path(temp_dir)
        audio_id = "test_sample"
        
        # Initial extraction saves feature to disk
        feature1 = extractor.extract_and_cache(waveform, audio_id, cache_dir=cache_dir)
        cache_file = cache_dir / f"{audio_id}_lfcc.pt"
        assert cache_file.exists()
        
        # Second extraction reads from disk
        feature2 = extractor.extract_and_cache(waveform, audio_id, cache_dir=cache_dir)
        assert torch.allclose(feature1, feature2)
