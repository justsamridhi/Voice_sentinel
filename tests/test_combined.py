import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from src.utils.config import Config
from src.features.extractor import FeatureExtractor
from src.models.baseline import ResNet34Baseline
from src.models.stereo_net import M2SConverter
from src.training.trainer import Trainer


class SimpleWaveformDataset(Dataset):
    def __init__(self):
        # 2 waveforms of 1.0 second duration @ 16kHz
        self.data = [torch.randn(1, 16000) for _ in range(2)]
        self.labels = [0, 1]

    def __len__(self) -> int:
        return 2

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        return self.data[idx], self.labels[idx]


def test_combined_mode_shapes():
    config_dict = {
        "experiment_name": "test_combined",
        "mode": "D",  # Combined Mode D
        "device": "cpu",
        "seed": 42,
        "paths": {
            "data_dir": "test", "asvspoof_root": "test", "checkpoint_dir": "test", "output_dir": "test", "tb_log_dir": "test"
        },
        "audio": {
            "sample_rate": 16000, "duration": 1.0, "padding_type": "wrap", "crop_type": "random"
        },
        "features": {
            "type": "spectrogram", "n_fft": 512, "win_length": 400, "hop_length": 160,
            "n_lfcc": 60, "n_mels": 80, "f_min": 0.0, "f_max": 8000.0
        },
        "augmentation": {
            "enabled": True,
            "compression": {"enabled": False, "codec": "mp3", "bitrate": "64k"},
            "channel": {"enabled": False, "impulse_response_path": None},
            "spec_average": {"enabled": True, "time_mask_max": 2, "freq_mask_max": 2}
        },
        "model": {
            "name": "resnet34",
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
    
    # 1. Instantiate 2-channel model
    from src.inference.pipeline import load_model_from_config
    model = load_model_from_config(config)
    assert isinstance(model, ResNet34Baseline)
    assert model.resnet.conv1.in_channels == 2
    
    # 2. Extract 2-channel features
    converter = M2SConverter(sample_rate=16000)
    feat_extractor = FeatureExtractor(feature_type="spectrogram")
    
    x = torch.randn(2, 1, 16000)
    stereo = converter(x)
    
    left_feat = feat_extractor(stereo[:, 0:1, :])
    right_feat = feat_extractor(stereo[:, 1:2, :])
    combined_feat = torch.cat([left_feat, right_feat], dim=1)
    
    assert combined_feat.shape[1] == 2  # 2 channels
    
    # Forward pass
    logits = model(combined_feat)
    assert logits.shape == (2, 2)


def test_combined_mode_training():
    config_dict = {
        "experiment_name": "test_combined_train",
        "mode": "D",  # Mode D
        "device": "cpu",
        "seed": 12,
        "paths": {
            "data_dir": "test", "asvspoof_root": "test", "checkpoint_dir": "test_check_comb", "output_dir": "test_out_comb", "tb_log_dir": "test_out_comb/runs"
        },
        "audio": {
            "sample_rate": 16000, "duration": 1.0, "padding_type": "wrap", "crop_type": "random"
        },
        "features": {
            "type": "spectrogram", "n_fft": 512, "win_length": 400, "hop_length": 160,
            "n_lfcc": 60, "n_mels": 80, "f_min": 0.0, "f_max": 8000.0
        },
        "augmentation": {
            "enabled": True,
            "compression": {"enabled": False, "codec": "mp3", "bitrate": "64k"},
            "channel": {"enabled": False, "impulse_response_path": None},
            "spec_average": {"enabled": True, "time_mask_max": 2, "freq_mask_max": 2}
        },
        "model": {
            "name": "resnet34",
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
    
    from src.inference.pipeline import load_model_from_config
    model = load_model_from_config(config)
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

    # Clean test files
    import shutil
    shutil.rmtree("test_check_comb", ignore_errors=True)
    shutil.rmtree("test_out_comb", ignore_errors=True)
