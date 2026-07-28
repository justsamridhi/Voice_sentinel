import pytest
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset

from src.features.augmentations import compress_audio, apply_channel_effects, apply_spec_average
from src.training.trainer import Trainer
from src.utils.config import Config
from src.models.baseline import CNNBaseline
from src.features.extractor import FeatureExtractor


class SimpleDataset(Dataset):
    def __init__(self):
        # 4 waveforms of 1 second at 16000Hz
        self.data = [torch.randn(1, 16000) for _ in range(4)]
        self.labels = [0, 1, 0, 1]

    def __len__(self) -> int:
        return 4

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        return self.data[idx], self.labels[idx]


def test_compress_audio():
    waveform = torch.randn(1, 16000)
    compressed = compress_audio(waveform, 16000, codec="mp3", bitrate="32k")
    
    assert isinstance(compressed, torch.Tensor)
    assert compressed.shape == waveform.shape
    assert not torch.isnan(compressed).any()


def test_apply_channel_effects():
    waveform = torch.randn(1, 16000)
    filtered = apply_channel_effects(waveform, 16000)
    
    assert isinstance(filtered, torch.Tensor)
    assert filtered.shape == waveform.shape
    assert not torch.isnan(filtered).any()
    # Filtered waveform should be different from original
    assert not torch.allclose(filtered, waveform)


def test_apply_spec_average():
    # Spectrogram shape: (batch, channels, freq, time)
    # E.g. (1, 1, 40, 100)
    spec = torch.ones(1, 1, 40, 100) * 5.0
    avg_val = spec.mean().item()
    assert avg_val == 5.0
    
    # We alter one corner to have a different mean
    spec[0, 0, 0, 0] = 105.0  # mean becomes 5.0 + 100.0/4000 = 5.025
    new_mean = spec.mean().item()
    
    masked = apply_spec_average(spec, time_mask_max=5, freq_mask_max=5, num_time_masks=1, num_freq_masks=1)
    
    assert masked.shape == spec.shape
    # Check that some values equal the mean
    assert torch.any(torch.isclose(masked, torch.tensor(new_mean)))


def test_bce_loss_trainer():
    # Configure config with loss_type = "w_bce"
    config_dict = {
        "experiment_name": "test_bce",
        "mode": "B",  # Baseline + Paper 4
        "device": "cpu",
        "seed": 42,
        "paths": {
            "data_dir": "test_data",
            "asvspoof_root": "test_data/ASVspoof",
            "checkpoint_dir": "checkpoints_test_bce",
            "output_dir": "outputs_test_bce",
            "tb_log_dir": "outputs_test_bce/runs"
        },
        "audio": {
            "sample_rate": 16000,
            "duration": 1.0,
            "padding_type": "wrap",
            "crop_type": "random"
        },
        "features": {
            "type": "lfcc",
            "n_fft": 512,
            "win_length": 400,
            "hop_length": 160,
            "n_lfcc": 20,
            "n_mels": 40,
            "f_min": 0.0,
            "f_max": 8000.0
        },
        "augmentation": {
            "enabled": True,  # Enable SpecAverage
            "compression": {"enabled": True, "codec": "mp3", "bitrate": "64k"},
            "channel": {"enabled": True, "impulse_response_path": None},
            "spec_average": {"enabled": True, "time_mask_max": 5, "freq_mask_max": 5}
        },
        "model": {
            "name": "cnn_baseline",
            "sincnet": {"out_channels": 80, "kernel_size": 251},
            "gat": {"in_features": 128, "out_features": 64, "heads": 2, "dropout": 0.5},
            "num_classes": 2
        },
        "training": {
            "epochs": 1,
            "batch_size": 2,
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "optimizer": "adam",
            "scheduler": {"type": "none", "step_size": 1, "gamma": 0.5, "patience": 1},
            "loss_type": "w_bce",  # Test Weighted BCE Loss
            "class_weights": [1.0, 3.0],  # Weighted BCE
            "early_stopping": {"enabled": False, "patience": 1, "min_delta": 0.0}
        }
    }
    
    config = Config.from_dict(config_dict)
    
    # Models, Datasets
    model = CNNBaseline(num_classes=2)
    feat_extractor = FeatureExtractor(
        feature_type="lfcc", sample_rate=16000, n_lfcc=20, n_mels=40
    )
    
    train_dataset = SimpleDataset()
    val_dataset = SimpleDataset()
    
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    trainer = Trainer(
        config=config,
        model=model,
        feature_extractor=feat_extractor,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        class_weights=torch.tensor(config.training.class_weights)
    )
    
    metrics = trainer.fit()
    assert "eer" in metrics
    
    # Check that model weights can be loaded back using BCE configuration
    from src.inference.pipeline import load_model_from_config
    model_eval = load_model_from_config(config)
    assert isinstance(model_eval, CNNBaseline)

    # Clean test files
    import shutil
    shutil.rmtree("checkpoints_test_bce", ignore_errors=True)
    shutil.rmtree("outputs_test_bce", ignore_errors=True)
