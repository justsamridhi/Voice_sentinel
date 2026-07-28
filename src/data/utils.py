import logging
import random
from typing import Dict, Any
from pathlib import Path
from src.data.dataset import ASVspoofDataset, get_audio_path

logger = logging.getLogger(__name__)


def compute_dataset_stats(dataset: ASVspoofDataset) -> Dict[str, Any]:
    """Computes and logs class balance metrics and recommended loss weights.

    Args:
        dataset: Instantiated ASVspoofDataset object.

    Returns:
        Dict[str, Any]: Dictionary containing counts, percentages, and weights.
    """
    df = dataset.df
    total_samples = len(df)
    
    # Value counts of label column
    label_counts = df["label"].value_counts().to_dict()
    bonafide_count = label_counts.get("bonafide", 0)
    spoof_count = label_counts.get("spoof", 0)

    # Percentage composition
    bonafide_pct = (bonafide_count / total_samples * 100) if total_samples > 0 else 0.0
    spoof_pct = (spoof_count / total_samples * 100) if total_samples > 0 else 0.0

    # Compute inverse frequency class weights: weight_c = N / (C * N_c)
    # class 0: bonafide, class 1: spoof
    weight_bonafide = total_samples / (2.0 * bonafide_count) if bonafide_count > 0 else 1.0
    weight_spoof = total_samples / (2.0 * spoof_count) if spoof_count > 0 else 1.0

    stats = {
        "total_samples": total_samples,
        "counts": {
            "bonafide": bonafide_count,
            "spoof": spoof_count
        },
        "percentages": {
            "bonafide": bonafide_pct,
            "spoof": spoof_pct
        },
        "recommended_weights": [weight_bonafide, weight_spoof]
    }

    logger.info(f"--- Dataset Statistics: {dataset.partition.upper()} partition ---")
    logger.info(f"Total Samples: {total_samples}")
    logger.info(f"Bonafide (Real): {bonafide_count} ({bonafide_pct:.2f}%)")
    logger.info(f"Spoof (Fake): {spoof_count} ({spoof_pct:.2f}%)")
    logger.info(f"Recommended weights [bonafide, spoof]: {stats['recommended_weights']}")
    
    return stats


def verify_dataset_integrity(
    dataset: ASVspoofDataset, 
    num_checks: int = 100
) -> bool:
    """Verifies physical presence of a subset of files referenced in the protocol.

    Checks a random sample of rows from the protocol, checking that paths exist
    and can be successfully resolved.

    Args:
        dataset: Instantiated ASVspoofDataset object.
        num_checks: Number of random checks to perform.

    Returns:
        bool: True if all checked files exist, False if any check fails.
    """
    df = dataset.df
    total_samples = len(df)
    
    if total_samples == 0:
        logger.error("Dataset protocol is empty. Integrity check failed.")
        return False

    sample_size = min(num_checks, total_samples)
    checked_indices = random.sample(range(total_samples), sample_size)
    
    missing_count = 0
    for idx in checked_indices:
        row = df.iloc[idx]
        audio_id = row["audio_id"]
        audio_path = get_audio_path(dataset.asvspoof_root, dataset.partition, audio_id)
        
        if not audio_path.exists():
            logger.error(f"Missing file for audio ID {audio_id}: {audio_path}")
            missing_count += 1

    if missing_count > 0:
        logger.error(f"Integrity check failed. {missing_count}/{sample_size} checked files are missing.")
        return False
        
    logger.info(f"Integrity check passed! Verified {sample_size}/{total_samples} files.")
    return True
