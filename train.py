import argparse
import logging
from pathlib import Path
import random
import numpy as np
import soundfile as sf
import torch
from typing import Tuple

from src.utils.config import Config
from src.utils.logging import setup_logging
from src.data.dataset import ASVspoofDataset, collate_fn
from src.data.utils import compute_dataset_stats, verify_dataset_integrity
from src.features.extractor import FeatureExtractor
from src.inference.pipeline import load_model_from_config
from src.training.trainer import Trainer

logger = logging.getLogger(__name__)


def generate_synthetic_dataset(asvspoof_root: Path) -> None:
    """Creates a small mock ASVspoof2019 Logical Access dataset for dry-runs."""
    logger.info(f"Generating synthetic mock dataset under {asvspoof_root}...")
    
    # Create structure
    for part in ["train", "dev", "eval"]:
        flac_dir = asvspoof_root / "LA" / f"ASVspoof2019_LA_{part}" / "flac"
        flac_dir.mkdir(parents=True, exist_ok=True)
        
        # Write 5 mock audio files (0.5s duration)
        sr = 16000
        t = np.linspace(0, 0.5, int(sr * 0.5))
        wav_data = 0.3 * np.sin(2 * np.pi * 440.0 * t)
        
        audio_ids = [f"LA_{part[0].upper()}_100000{i}" for i in range(1, 6)]
        for aid in audio_ids:
            sf.write(str(flac_dir / f"{aid}.wav"), wav_data, sr)

        # Write protocol files
        protocol_dir = asvspoof_root / "LA" / "ASVspoof2019_LA_protocols"
        protocol_dir.mkdir(parents=True, exist_ok=True)
        protocol_file = protocol_dir / f"ASVspoof2019.LA.protocol.{part}.txt"
        
        with open(protocol_file, "w", encoding="utf-8") as f:
            f.write(f"LA_0001 {audio_ids[0]} - - bonafide\n")
            for i, aid in enumerate(audio_ids[1:]):
                f.write(f"LA_0002 {aid} - A{i+1:02d} spoof\n")


def resolve_dataset(config: Config) -> Tuple[Path, Path]:
    """Ensures ASVspoof protocol files exist, triggering synthetic generation if missing.

    Returns:
        Tuple[Path, Path]: Paths to (train_protocol, dev_protocol).
    """
    root = config.paths.asvspoof_root
    proto_dir = root / "LA" / "ASVspoof2019_LA_protocols"
    train_proto = proto_dir / "ASVspoof2019.LA.protocol.train.txt"
    dev_proto = proto_dir / "ASVspoof2019.LA.protocol.dev.txt"

    if not train_proto.exists() or not dev_proto.exists():
        logger.warning("ASVspoof2019 Logical Access protocol files not found.")
        generate_synthetic_dataset(root)

    return train_proto, dev_proto


def set_seed(seed: int) -> None:
    """Sets standard seeds for reproducible PyTorch experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main() -> None:
    parser = argparse.ArgumentParser(description="Train VoiceSentinel Deepfake Detector")
    parser.add_argument("--config", type=str, default="configs/default_config.yaml", help="Path to config")
    args = parser.parse_args()

    # Load configuration
    config = Config.load_from_yaml(args.config)

    # Set logging and seeds
    setup_logging(log_file=config.paths.output_dir / "voicesentinel_train.log")
    set_seed(config.seed)
    
    logger.info(f"--- Training VoiceSentinel in Mode {config.mode} ---")

    # Resolve dataset paths
    train_proto, dev_proto = resolve_dataset(config)

    # Load Train & Dev datasets
    train_dataset = ASVspoofDataset(
        protocol_file=train_proto,
        asvspoof_root=config.paths.asvspoof_root,
        partition="train",
        target_sr=config.audio.sample_rate,
        duration=config.audio.duration,
        padding_type=config.audio.padding_type,
        crop_type=config.audio.crop_type,
        cache=False,
        config=config
    )
    
    dev_dataset = ASVspoofDataset(
        protocol_file=dev_proto,
        asvspoof_root=config.paths.asvspoof_root,
        partition="dev",
        target_sr=config.audio.sample_rate,
        duration=config.audio.duration,
        padding_type=config.audio.padding_type,
        crop_type="center",  # Evaluate deterministically
        cache=False
    )

    # Run checks & stats
    verify_dataset_integrity(train_dataset, num_checks=5)
    stats = compute_dataset_stats(train_dataset)
    
    # Dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.training.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=0
    )
    dev_loader = torch.utils.data.DataLoader(
        dev_dataset, batch_size=config.training.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0
    )

    # Model & Feature Extractor
    model = load_model_from_config(config)
    
    feat_opt = config.features
    feature_extractor = FeatureExtractor(
        feature_type=feat_opt.type,
        sample_rate=config.audio.sample_rate,
        n_fft=feat_opt.n_fft,
        win_length=feat_opt.win_length,
        hop_length=feat_opt.hop_length,
        n_lfcc=feat_opt.n_lfcc,
        n_mels=feat_opt.n_mels,
        f_min=feat_opt.f_min,
        f_max=feat_opt.f_max
    )

    # Loss weights, Optimizer, Scheduler
    weights = torch.tensor(stats["recommended_weights"])
    
    if config.training.optimizer.lower() == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.learning_rate, weight_decay=config.training.weight_decay)
    elif config.training.optimizer.lower() == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=config.training.learning_rate, momentum=0.9, weight_decay=config.training.weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=config.training.learning_rate, weight_decay=config.training.weight_decay)

    # Scheduler setup
    sched_opt = config.training.scheduler
    if sched_opt.type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.training.epochs)
    elif sched_opt.type == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=sched_opt.step_size, gamma=sched_opt.gamma)
    else:
        scheduler = None

    # Run Trainer fit
    trainer = Trainer(
        config=config,
        model=model,
        feature_extractor=feature_extractor,
        train_loader=train_loader,
        val_loader=dev_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        class_weights=weights
    )

    best_metrics = trainer.fit()
    logger.info(f"Training completed. Best Val EER: {best_metrics['eer']:.4f}")


if __name__ == "__main__":
    main()
