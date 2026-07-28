import argparse
import logging
import os
import shutil
from pathlib import Path
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for Windows file locks
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.utils.config import Config
from src.data.dataset import ASVspoofDataset, collate_fn
from src.features.extractor import FeatureExtractor
from src.inference.pipeline import load_model_from_config
from src.training.trainer import Trainer

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("AblationStudy")


def generate_mock_ablation_dataset(root_dir: Path, num_samples: int = 12):
    """Generates a small mock ASVspoof protocol and wave folder for speed."""
    root_dir.mkdir(parents=True, exist_ok=True)
    
    # Create wav subdirectories
    for folder in ["ASVspoof2019_LA_train", "ASVspoof2019_LA_dev"]:
        (root_dir / folder / "flac").mkdir(parents=True, exist_ok=True)

    # Create synthetic protocols
    protocols = {
        "train": root_dir / "ASVspoof2019_LA_protocols" / "ASVspoof2019.LA.cm.train.trn.txt",
        "dev": root_dir / "ASVspoof2019_LA_protocols" / "ASVspoof2019.LA.cm.dev.asl.txt"
    }
    protocols["train"].parent.mkdir(parents=True, exist_ok=True)

    sr = 16000
    duration = 1.0
    samples = int(sr * duration)
    t = np.linspace(0, duration, samples, endpoint=False)

    for split, proto_path in protocols.items():
        lines = []
        folder_name = f"ASVspoof2019_LA_{split}"
        for i in range(num_samples):
            audio_id = f"LA_{split}_{i:04d}"
            label = "bonafide" if i % 2 == 0 else "spoof"
            sys_id = "-" if label == "bonafide" else f"A{i % 19 + 1:02d}"
            lines.append(f"LA_0000 {audio_id} - {sys_id} {label}\n")

            # Write raw sine waves to simulate audio
            import soundfile as sf
            wav_path = root_dir / folder_name / "flac" / f"{audio_id}.wav"
            freq = 440.0 if label == "bonafide" else 880.0
            wav_data = 0.5 * np.sin(2 * np.pi * freq * t)
            sf.write(str(wav_path), wav_data, sr)

        with open(proto_path, "w") as f:
            f.writelines(lines)

    return protocols["train"], protocols["dev"]


def run_ablation_for_mode(mode: str, data_dir: Path, train_proto: Path, dev_proto: Path):
    """Runs a single training epoch and evaluates for the given mode."""
    logger.info(f"=== Starting Ablation for Mode {mode} ===")
    
    # Configure directories for this mode
    checkpoints_dir = Path(f"outputs_ablation/checkpoints_mode_{mode}")
    runs_dir = Path(f"outputs_ablation/runs_mode_{mode}")
    shutil.rmtree(checkpoints_dir, ignore_errors=True)
    shutil.rmtree(runs_dir, ignore_errors=True)

    # Base configuration dictionary
    config_dict = {
        "experiment_name": f"ablation_mode_{mode}",
        "mode": mode,
        "device": "cpu",
        "seed": 42,
        "paths": {
            "data_dir": str(data_dir),
            "asvspoof_root": str(data_dir),
            "checkpoint_dir": str(checkpoints_dir),
            "output_dir": "outputs_ablation",
            "tb_log_dir": str(runs_dir)
        },
        "audio": {
            "sample_rate": 16000,
            "duration": 1.0,
            "padding_type": "wrap",
            "crop_type": "random"
        },
        "features": {
            "type": "spectrogram" if mode in ["B", "D"] else "lfcc",
            "n_fft": 512,
            "win_length": 400,
            "hop_length": 160,
            "n_lfcc": 40,
            "n_mels": 60,
            "f_min": 0.0,
            "f_max": 8000.0
        },
        "augmentation": {
            "enabled": True if mode in ["B", "D"] else False,
            "compression": {"enabled": True if mode in ["B", "D"] else False, "codec": "mp3", "bitrate": "64k"},
            "channel": {"enabled": True if mode in ["B", "D"] else False, "impulse_response_path": None},
            "spec_average": {"enabled": True if mode in ["B", "D"] else False, "time_mask_max": 2, "freq_mask_max": 2}
        },
        "model": {
            "name": "sincnet_gat" if mode == "C" else "resnet34",
            "sincnet": {"out_channels": 20, "kernel_size": 121},
            "gat": {"in_features": 128, "out_features": 16, "heads": 2, "dropout": 0.0},
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

    # Initialize datasets & loaders
    train_dataset = ASVspoofDataset(
        protocol_file=train_proto,
        asvspoof_root=config.paths.asvspoof_root,
        partition="train",
        target_sr=config.audio.sample_rate,
        duration=config.audio.duration,
        padding_type=config.audio.padding_type,
        crop_type=config.audio.crop_type,
        config=config
    )
    
    dev_dataset = ASVspoofDataset(
        protocol_file=dev_proto,
        asvspoof_root=config.paths.asvspoof_root,
        partition="dev",
        target_sr=config.audio.sample_rate,
        duration=config.audio.duration,
        padding_type=config.audio.padding_type,
        crop_type="center"
    )

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
    dev_loader = DataLoader(dev_dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)

    # Setup feature extractor and model
    model = load_model_from_config(config)
    feat_extractor = FeatureExtractor(
        feature_type=config.features.type,
        sample_rate=config.audio.sample_rate,
        n_fft=config.features.n_fft,
        win_length=config.features.win_length,
        hop_length=config.features.hop_length,
        n_lfcc=config.features.n_lfcc,
        n_mels=config.features.n_mels,
        f_min=config.features.f_min,
        f_max=config.features.f_max
    )

    # Train for 1 epoch
    optimizer = torch.optim.Adam(model.parameters(), lr=config.training.learning_rate)
    trainer = Trainer(
        config=config,
        model=model,
        feature_extractor=feat_extractor,
        train_loader=train_loader,
        val_loader=dev_loader,
        optimizer=optimizer
    )

    metrics = trainer.fit()
    logger.info(f"Mode {mode} Finished. Metrics: EER={metrics['eer']:.4f}, AUC={metrics['auc']:.4f}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="VoiceSentinel Ablation Study Script")
    parser.add_argument("--samples", type=int, default=12, help="Number of synthetic samples to generate per split")
    args = parser.parse_args()

    # Paths
    temp_data = Path("outputs_ablation/ablation_mock_data")
    shutil.rmtree("outputs_ablation", ignore_errors=True)
    
    # 1. Generate synthetic data
    train_proto, dev_proto = generate_mock_ablation_dataset(temp_data, num_samples=args.samples)
    
    modes = ["A", "B", "C", "D"]
    results = {}

    # 2. Run ablation for each mode
    for m in modes:
        try:
            res = run_ablation_for_mode(m, temp_data, train_proto, dev_proto)
            results[m] = res
        except Exception as e:
            logger.error(f"Failed running ablation for mode {m}: {e}", exc_info=True)

    # 3. Clean up raw synthetic data
    shutil.rmtree(temp_data, ignore_errors=True)

    # 4. Generate report
    print("\n" + "=" * 50)
    print("           ABLATION STUDY COMPARISON REPORT           ")
    print("=" * 50)
    print("| Mode | Configuration Name          | Val Loss | Val EER | Val AUC |")
    print("|------|-----------------------------|----------|---------|---------|")
    
    names = {
        "A": "Baseline (Mono LFCC)",
        "B": "Paper 4 (LogSpec+Augment)",
        "C": "Paper 2 (Stereo Raw Waveform)",
        "D": "Mode D (Combined Stereo Spec)"
    }
    
    for m in modes:
        if m in results:
            val_loss = results[m]["loss"]
            eer = results[m]["eer"]
            auc = results[m]["auc"]
            print(f"|  {m}   | {names[m]:<27} | {val_loss:.4f}   | {eer:.4f}  | {auc:.4f}  |")
        else:
            print(f"|  {m}   | {names[m]:<27} | Failed   | Failed  | Failed  |")
    print("=" * 50)

    # 5. Plot comparison
    try:
        modes_present = [m for m in modes if m in results]
        eers = [results[m]["eer"] for m in modes_present]
        aucs = [results[m]["auc"] for m in modes_present]
        names_present = [names[m] for m in modes_present]

        x = np.arange(len(modes_present))
        width = 0.35

        fig, ax = plt.subplots(figsize=(8, 5))
        rects1 = ax.bar(x - width/2, eers, width, label="EER (Lower is better)", color="#e06666")
        rects2 = ax.bar(x + width/2, aucs, width, label="AUC (Higher is better)", color="#6fa8dc")

        ax.set_ylabel("Metric Value")
        ax.set_title("Ablation Study Model Comparisons")
        ax.set_xticks(x)
        ax.set_xticklabels(names_present, rotation=15)
        ax.legend()
        ax.set_ylim(0, 1.1)

        fig.tight_layout()
        plot_path = Path("outputs_ablation/ablation_comparison.png")
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_path, dpi=150)
        logger.info(f"Saved comparison chart to {plot_path.resolve()}")
    except Exception as e:
        logger.error(f"Failed to plot ablation chart: {e}")


if __name__ == "__main__":
    main()
