import logging
from pathlib import Path
from typing import Dict, Any, Union
import torch
import torch.nn as nn

from src.utils.config import Config
from src.utils.audio import load_audio, pad_crop_audio, normalize_audio
from src.features.extractor import FeatureExtractor

logger = logging.getLogger(__name__)


def load_model_from_config(config: Config) -> nn.Module:
    """Instantiates the correct model architecture based on Config settings.

    Args:
        config: The parsed Config object.

    Returns:
        nn.Module: Instantiated PyTorch model.
    """
    model_name = config.model.name.lower()
    num_classes = config.model.num_classes

    if model_name in ["resnet34", "resnet"]:
        from src.models.baseline import ResNet34Baseline
        return ResNet34Baseline(num_classes=num_classes)
    elif model_name in ["cnn_baseline", "cnn"]:
        from src.models.baseline import CNNBaseline
        return CNNBaseline(num_classes=num_classes)
    elif model_name == "sincnet_gat":
        # Lazy load to avoid circular dependencies when Paper 2 is introduced
        from src.models.stereo_net import SincNetResidualGAT
        return SincNetResidualGAT(config=config)
    else:
        raise ValueError(f"Unsupported model architecture: {config.model.name}")


class InferencePipeline:
    """Predicts deepfake spoof probability on single audio files."""

    def __init__(self, checkpoint_path: Union[str, Path], device: str = "cpu"):
        """Loads a model checkpoint and instantiates the pipeline.

        Args:
            checkpoint_path: Path to a saved PyTorch model checkpoint (.pt).
            device: Computing device ("cuda" or "cpu").
        """
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        
        # Load state dictionary and configurations
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.config = checkpoint["config"]

        # Reconstruct model
        self.model = load_model_from_config(self.config)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        # Reconstruct feature extractor
        feat_opt = self.config.features
        self.feature_extractor = FeatureExtractor(
            feature_type=feat_opt.type,
            sample_rate=self.config.audio.sample_rate,
            n_fft=feat_opt.n_fft,
            win_length=feat_opt.win_length,
            hop_length=feat_opt.hop_length,
            n_lfcc=feat_opt.n_lfcc,
            n_mels=feat_opt.n_mels,
            f_min=feat_opt.f_min,
            f_max=feat_opt.f_max
        ).to(self.device)

    @torch.no_grad()
    def predict(self, audio_path: Union[str, Path]) -> Dict[str, Any]:
        """Loads audio, extracts features, and outputs prediction.

        Args:
            audio_path: Path to the target audio file (.wav, .flac).

        Returns:
            Dict[str, Any]: Contains label, spoof probability, and confidence.
        """
        # Load and preprocess waveform (shape: 1, samples)
        waveform, sr = load_audio(audio_path, target_sr=self.config.audio.sample_rate)
        waveform = normalize_audio(waveform)
        waveform = pad_crop_audio(
            waveform,
            sample_rate=sr,
            target_duration=self.config.audio.duration,
            padding_type=self.config.audio.padding_type,
            crop_type="center"  # Center crop for deterministic inference
        )
        
        # Format as batch: (1, 1, samples)
        waveform_batch = waveform.unsqueeze(0).to(self.device)

        # Extract features and predict
        features = self.feature_extractor(waveform_batch)
        logits = self.model(features)
        probs = torch.softmax(logits, dim=-1).squeeze(0)

        spoof_prob = probs[1].item()
        bonafide_prob = probs[0].item()
        
        pred_label = "spoof" if spoof_prob > 0.5 else "bonafide"
        confidence = spoof_prob if pred_label == "spoof" else bonafide_prob

        return {
            "label": pred_label,
            "spoof_probability": spoof_prob,
            "bonafide_probability": bonafide_prob,
            "confidence": confidence
        }
