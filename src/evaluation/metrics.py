import logging
from typing import Dict, Any, Tuple
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
)

logger = logging.getLogger(__name__)


def compute_eer(scores: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
    """Computes the Equal Error Rate (EER) and its corresponding threshold.

    EER is the point where the False Acceptance Rate (FAR) equals the
    False Rejection Rate (FRR). Assuming 'spoof' (1) is the positive class
    and 'bonafide' (0) is the negative class:
    - FAR is FPR (False Positive Rate)
    - FRR is FNR (False Negative Rate = 1 - True Positive Rate)

    Args:
        scores: Predicted probabilities of the positive class (spoof).
        labels: True binary labels (0 for bonafide, 1 for spoof).

    Returns:
        Tuple[float, float]: (EER value, threshold at EER).
    """
    if len(np.unique(labels)) < 2:
        logger.warning("Only one class present in labels. EER calculation is invalid.")
        return 0.0, 0.5

    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1.0 - tpr

    # Find the index where FPR and FNR are closest
    idx = np.nanargmin(np.abs(fpr - fnr))
    
    # EER is the average of FPR and FNR at this point
    eer = (fpr[idx] + fnr[idx]) / 2.0
    threshold = thresholds[idx]

    return float(eer), float(threshold)


def calculate_metrics(y_true: np.ndarray, y_pred_probs: np.ndarray) -> Dict[str, Any]:
    """Calculates evaluation metrics for spoof detection.

    Args:
        y_true: Ground truth binary labels (0 = bonafide, 1 = spoof).
        y_pred_probs: Predicted probability of being spoof (shape (N, 2) or (N,)).

    Returns:
        Dict[str, Any]: Dictionary containing acc, precision, recall, f1, auc, cm, eer.
    """
    # Extract spoof class probability
    if y_pred_probs.ndim == 2:
        scores = y_pred_probs[:, 1]
    else:
        scores = y_pred_probs

    y_pred = (scores >= 0.5).astype(int)

    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )

    try:
        auc = roc_auc_score(y_true, scores)
    except Exception as e:
        logger.warning(f"Failed to calculate ROC AUC: {e}")
        auc = 0.5

    cm = confusion_matrix(y_true, y_pred)
    eer, eer_threshold = compute_eer(scores, y_true)

    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "auc": float(auc),
        "confusion_matrix": cm.tolist(),
        "eer": float(eer),
        "eer_threshold": float(eer_threshold),
    }
