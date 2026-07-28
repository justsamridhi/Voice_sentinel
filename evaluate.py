import argparse
import logging
from pathlib import Path
import numpy as np
import torch

from src.utils.config import Config
from src.utils.logging import setup_logging
from src.data.dataset import ASVspoofDataset, collate_fn
from src.data.utils import verify_dataset_integrity
from src.features.extractor import FeatureExtractor
from src.inference.pipeline import load_model_from_config
from src.evaluation.metrics import calculate_metrics

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate VoiceSentinel model checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .pt file")
    parser.add_argument("--protocol", type=str, default=None, help="Path to evaluation protocol file")
    parser.add_argument("--partition", type=str, default="dev", choices=["train", "dev", "eval"], help="Target partition")
    parser.add_argument("--device", type=str, default="cpu", help="Computing device (cpu or cuda)")
    args = parser.parse_args()

    # Configure logging
    setup_logging()
    
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    logger.info(f"Loading checkpoint: {args.checkpoint} on {device}")
    
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = checkpoint["config"]

    # Reconstruct model and load weights
    model = load_model_from_config(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # Reconstruct feature extractor
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
    ).to(device)

    # Determine protocol path
    if args.protocol is not None:
        protocol_path = Path(args.protocol)
    else:
        # Fallback to config path for the chosen partition
        proto_dir = config.paths.asvspoof_root / "LA" / "ASVspoof2019_LA_protocols"
        protocol_path = proto_dir / f"ASVspoof2019.LA.protocol.{args.partition}.txt"

    logger.info(f"Using protocol file: {protocol_path}")
    if not protocol_path.exists():
        raise FileNotFoundError(f"Protocol file not found: {protocol_path}")

    # Dataset loader
    dataset = ASVspoofDataset(
        protocol_file=protocol_path,
        asvspoof_root=config.paths.asvspoof_root,
        partition=args.partition,
        target_sr=config.audio.sample_rate,
        duration=config.audio.duration,
        padding_type=config.audio.padding_type,
        crop_type="center",  # Evaluate deterministically
        cache=False
    )

    verify_dataset_integrity(dataset, num_checks=5)

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=config.training.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0
    )

    # Evaluate
    logger.info("Running forward passes...")
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for waveforms, labels in loader:
            waveforms = waveforms.to(device)
            features = feature_extractor(waveforms)
            logits = model(features)
            
            probs = torch.softmax(logits, dim=-1)
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    metrics = calculate_metrics(np.array(all_labels), np.array(all_probs))

    # Print results
    logger.info("=== Evaluation Results ===")
    logger.info(f"Accuracy : {metrics['accuracy']:.4f}")
    logger.info(f"Precision: {metrics['precision']:.4f}")
    logger.info(f"Recall   : {metrics['recall']:.4f}")
    logger.info(f"F1 Score : {metrics['f1']:.4f}")
    logger.info(f"ROC AUC  : {metrics['auc']:.4f}")
    logger.info(f"EER      : {metrics['eer']:.4f} (at threshold {metrics['eer_threshold']:.4f})")
    logger.info(f"Confusion Matrix:\n{np.array(metrics['confusion_matrix'])}")


if __name__ == "__main__":
    main()
