from pathlib import Path
import pytest
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.models.baseline import CNNBaseline, ResNet34Baseline
from src.evaluation.metrics import compute_eer, calculate_metrics
from src.training.trainer import EarlyStopping, Trainer
from src.features.extractor import FeatureExtractor
from src.utils.config import Config


# Mock Dataset for trainer test
class SyntheticWaveformDataset(Dataset):
    def __init__(self, size: int = 10, samples: int = 16000):
        self.size = size
        self.samples = samples

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        waveform = torch.randn(1, self.samples)
        label = idx % 2  # alterning 0 and 1
        return waveform, label


def test_baseline_models():
    # Input shape: (batch, channels, freq_bins, time_steps)
    x = torch.randn(2, 1, 60, 100)

    # 1. CNNBaseline
    cnn = CNNBaseline(num_classes=2)
    logits_cnn = cnn(x)
    assert logits_cnn.shape == (2, 2)

    # 2. ResNet34Baseline
    resnet = ResNet34Baseline(num_classes=2, pretrained=False)
    logits_resnet = resnet(x)
    assert logits_resnet.shape == (2, 2)


def test_eer_metric():
    # Labels: 0 (bonafide), 1 (spoof)
    labels = np.array([0, 0, 1, 1])
    
    # Perfect predictions: lower scores for bonafide, higher for spoof
    scores_perfect = np.array([0.1, 0.2, 0.8, 0.9])
    eer_perfect, threshold_perfect = compute_eer(scores_perfect, labels)
    assert eer_perfect == 0.0
    
    # Inverse predictions (perfectly wrong)
    scores_wrong = np.array([0.9, 0.8, 0.2, 0.1])
    eer_wrong, _ = compute_eer(scores_wrong, labels)
    assert eer_wrong == 1.0


def test_early_stopping():
    # Patience of 2
    es = EarlyStopping(patience=2, mode="min")
    
    # Score gets worse (EER increases)
    assert es(0.1) is False
    assert es(0.2) is False  # Counter = 1
    assert es(0.3) is True   # Counter = 2, stop!


def test_trainer_fit():
    # Configure simple test config
    config_dict = {
        "experiment_name": "test_fit",
        "mode": "A",
        "device": "cpu",
        "seed": 42,
        "paths": {
            "data_dir": "test_data",
            "asvspoof_root": "test_data/ASVspoof",
            "checkpoint_dir": "checkpoints_test",
            "output_dir": "outputs_test",
            "tb_log_dir": "outputs_test/runs"
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
            "n_lfcc": 20,  # small features for fast testing
            "n_mels": 40,
            "f_min": 0.0,
            "f_max": 8000.0
        },
        "augmentation": {
            "enabled": False,
            "compression": {"enabled": False, "codec": "mp3", "bitrate": "64k"},
            "channel": {"enabled": False, "impulse_response_path": None},
            "spec_average": {"enabled": False, "time_mask_max": 2, "freq_mask_max": 2}
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
            "loss_type": "ce",
            "class_weights": [1.0, 1.0],
            "early_stopping": {"enabled": False, "patience": 1, "min_delta": 0.0}
        }
    }
    
    config = Config.from_dict(config_dict)
    
    # Models and datasets
    model = CNNBaseline(num_classes=2)
    feat_extractor = FeatureExtractor(
        feature_type="lfcc", sample_rate=16000, n_lfcc=20, n_mels=40
    )
    
    train_dataset = SyntheticWaveformDataset(size=4)
    val_dataset = SyntheticWaveformDataset(size=2)
    
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Run fit
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
    assert "accuracy" in metrics

    # Cleanup checkpoints created by test
    checkpoint_dir = Path("checkpoints_test")
    if checkpoint_dir.exists():
        for f in checkpoint_dir.glob("*.pt"):
            f.unlink()
        checkpoint_dir.rmdir()
        
    outputs_dir = Path("outputs_test")
    if outputs_dir.exists():
        # clean files inside runs and outputs_test
        for f in outputs_dir.rglob("*"):
            if f.is_file():
                f.unlink()
        # remove directories
        for d in sorted(outputs_dir.rglob("*"), key=lambda x: len(str(x)), reverse=True):
            if d.is_dir():
                d.rmdir()
        outputs_dir.rmdir()
