import tempfile
from pathlib import Path
import yaml
import pytest
from src.utils.config import Config
from src.utils.logging import setup_logging, get_logger


def test_config_loading():
    # Define a valid configuration mapping matching the YAML schema
    mock_config_data = {
        "experiment_name": "test_run",
        "mode": "A",
        "device": "cpu",
        "seed": 123,
        "paths": {
            "data_dir": "test_data",
            "asvspoof_root": "test_data/ASVspoof2019",
            "checkpoint_dir": "test_checkpoints",
            "output_dir": "test_outputs",
            "tb_log_dir": "test_outputs/runs"
        },
        "audio": {
            "sample_rate": 16000,
            "duration": 4.0,
            "padding_type": "wrap",
            "crop_type": "random"
        },
        "features": {
            "type": "lfcc",
            "n_fft": 512,
            "win_length": 400,
            "hop_length": 160,
            "n_lfcc": 60,
            "n_mels": 80,
            "f_min": 0.0,
            "f_max": 8000.0
        },
        "augmentation": {
            "enabled": False,
            "compression": {
                "enabled": False,
                "codec": "mp3",
                "bitrate": "64k"
            },
            "channel": {
                "enabled": False,
                "impulse_response_path": None
            },
            "spec_average": {
                "enabled": False,
                "time_mask_max": 30,
                "freq_mask_max": 15
            }
        },
        "model": {
            "name": "resnet34",
            "sincnet": {
                "out_channels": 80,
                "kernel_size": 251
            },
            "gat": {
                "in_features": 128,
                "out_features": 64,
                "heads": 4,
                "dropout": 0.5
            },
            "num_classes": 2
        },
        "training": {
            "epochs": 10,
            "batch_size": 32,
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "optimizer": "adam",
            "scheduler": {
                "type": "none",
                "step_size": 5,
                "gamma": 0.1,
                "patience": 2
            },
            "loss_type": "ce",
            "class_weights": [1.0, 1.0],
            "early_stopping": {
                "enabled": False,
                "patience": 2,
                "min_delta": 0.0
            }
        }
    }

    with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
        yaml.dump(mock_config_data, f)
        temp_path = Path(f.name)

    try:
        config = Config.load_from_yaml(temp_path)
        assert config.experiment_name == "test_run"
        assert config.paths.data_dir == Path("test_data")
        assert config.audio.sample_rate == 16000
        assert config.features.type == "lfcc"
        assert config.augmentation.enabled is False
        assert config.training.batch_size == 32
        assert config.model.name == "resnet34"
    finally:
        temp_path.unlink()


def test_logging_setup():
    with tempfile.TemporaryDirectory() as temp_dir:
        log_file = Path(temp_dir) / "test.log"
        setup_logging(log_file=log_file)
        
        logger = get_logger("TestModule")
        test_message = "Verifying setup logging execution."
        logger.info(test_message)
        
        assert log_file.exists()
        with open(log_file, "r") as f:
            log_content = f.read()
            assert test_message in log_content
            assert "TestModule" in log_content

        # Clean up logging handlers to release file lock on Windows
        import logging
        for handler in logging.root.handlers[:]:
            handler.close()
            logging.root.removeHandler(handler)

