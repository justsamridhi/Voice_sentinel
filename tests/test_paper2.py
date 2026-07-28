import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from src.models.sincnet import SincConv1d, SincNet
from src.models.gat import GATEncoder
from src.models.stereo_net import M2SConverter, SincNetResidualGAT
from src.utils.config import Config
from src.features.extractor import FeatureExtractor
from src.training.trainer import Trainer


class SimpleWaveformDataset(Dataset):
    def __init__(self):
        # 2 raw audio samples of 1.0 second duration @ 16kHz
        self.data = [torch.randn(1, 16000) for _ in range(2)]
        self.labels = [0, 1]

    def __len__(self) -> int:
        return 2

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        return self.data[idx], self.labels[idx]


def test_sinc_conv():
    # SincConv1d parameters: out_channels=40, kernel_size=121
    sinc = SincConv1d(out_channels=40, kernel_size=121, sample_rate=16000)
    x = torch.randn(2, 1, 16000)
    
    out = sinc(x)
    # Output length: 16000 - 121 + 1 = 15880 (without padding)
    assert out.shape == (2, 40, 15880)


def test_m2s_converter():
    converter = M2SConverter(sample_rate=16000)
    x = torch.randn(2, 1, 16000)
    
    stereo = converter(x)
    # Target shape: (batch, channels=2, samples)
    assert stereo.shape == (2, 2, 16000)
    # Verify channels differ due to time delay
    assert not torch.allclose(stereo[:, 0], stereo[:, 1])


def test_gat_encoder():
    # in_features=32, out_features=16, heads=2
    encoder = GATEncoder(in_features=32, out_features=16, heads=2, dropout=0.0)
    # Shape: (batch, channels, time)
    x = torch.randn(2, 32, 50)
    
    out = encoder(x)
    assert out.shape == (2, 16, 50)


def test_stereo_net():
    config_dict = {
        "experiment_name": "test_stereo",
        "mode": "C",
        "device": "cpu",
        "seed": 42,
        "paths": {
            "data_dir": "test",
            "asvspoof_root": "test",
            "checkpoint_dir": "test",
            "output_dir": "test",
            "tb_log_dir": "test"
        },
        "audio": {
            "sample_rate": 16000,
            "duration": 1.0,
            "padding_type": "wrap",
            "crop_type": "random"
        },
        "features": {
            "type": "spectrogram", "n_fft": 512, "win_length": 400, "hop_length": 160,
            "n_lfcc": 60, "n_mels": 80, "f_min": 0.0, "f_max": 8000.0
        },
        "augmentation": {
            "enabled": False,
            "compression": {"enabled": False, "codec": "mp3", "bitrate": "64k"},
            "channel": {"enabled": False, "impulse_response_path": None},
            "spec_average": {"enabled": False, "time_mask_max": 2, "freq_mask_max": 2}
        },
        "model": {
            "name": "sincnet_gat",
            "sincnet": {"out_channels": 40, "kernel_size": 121},
            "gat": {"in_features": 128, "out_features": 32, "heads": 2, "dropout": 0.0},
            "num_classes": 2
        },
        "training": {
            "epochs": 1, "batch_size": 2, "learning_rate": 0.001, "weight_decay": 0.0, "optimizer": "adam",
            "scheduler": {"type": "none", "step_size": 1, "gamma": 0.5, "patience": 1},
            "loss_type": "ce", "class_weights": [1.0, 1.0], "early_stopping": {"enabled": False, "patience": 1, "min_delta": 0.0}
        }
    }
    
    config = Config.from_dict(config_dict)
    model = SincNetResidualGAT(config)
    x = torch.randn(2, 1, 16000)
    
    logits = model(x)
    assert logits.shape == (2, 2)


def test_stereo_training():
    # Setup minimal config
    config_dict = {
        "experiment_name": "test_stereo_train",
        "mode": "C",
        "device": "cpu",
        "seed": 12,
        "paths": {
            "data_dir": "test", "asvspoof_root": "test", "checkpoint_dir": "test_check", "output_dir": "test_out", "tb_log_dir": "test_out/runs"
        },
        "audio": {
            "sample_rate": 16000, "duration": 1.0, "padding_type": "wrap", "crop_type": "random"
        },
        "features": {
            "type": "spectrogram", "n_fft": 512, "win_length": 400, "hop_length": 160,
            "n_lfcc": 60, "n_mels": 80, "f_min": 0.0, "f_max": 8000.0
        },
        "augmentation": {
            "enabled": False,
            "compression": {"enabled": False, "codec": "mp3", "bitrate": "64k"},
            "channel": {"enabled": False, "impulse_response_path": None},
            "spec_average": {"enabled": False, "time_mask_max": 2, "freq_mask_max": 2}
        },
        "model": {
            "name": "sincnet_gat",
            "sincnet": {"out_channels": 20, "kernel_size": 121},  # small filters for speed
            "gat": {"in_features": 128, "out_features": 16, "heads": 2, "dropout": 0.0},
            "num_classes": 2
        },
        "training": {
            "epochs": 1, "batch_size": 2, "learning_rate": 0.001, "weight_decay": 0.0, "optimizer": "adam",
            "scheduler": {"type": "none", "step_size": 1, "gamma": 0.5, "patience": 1},
            "loss_type": "ce", "class_weights": [1.0, 1.0], "early_stopping": {"enabled": False, "patience": 1, "min_delta": 0.0}
        }
    }
    
    config = Config.from_dict(config_dict)
    
    # Models, Datasets
    model = SincNetResidualGAT(config)
    feat_extractor = FeatureExtractor(feature_type="spectrogram")
    
    train_dataset = SimpleWaveformDataset()
    val_dataset = SimpleWaveformDataset()
    
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    trainer = Trainer(
        config=config,
        model=model,
        feature_extractor=feat_extractor,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer
    )
    
    metrics = trainer.fit()
    assert "eer" in metrics

    # Cleanup test files
    import shutil
    shutil.rmtree("test_check", ignore_errors=True)
    shutil.rmtree("test_out", ignore_errors=True)
