import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.utils.config import Config
from src.features.extractor import FeatureExtractor
from src.evaluation.metrics import calculate_metrics

logger = logging.getLogger(__name__)


class EarlyStopping:
    """Early stopping utility to terminate training when validation metric stalls."""

    def __init__(self, patience: int = 5, min_delta: float = 0.0, mode: str = "min"):
        """Initializes early stopping parameters.

        Args:
            patience: Number of epochs to wait for improvement.
            min_delta: Minimum change to qualify as an improvement.
            mode: "min" (e.g. loss, EER) or "max" (e.g. accuracy).
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score: Optional[float] = None
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        """Checks if metric improved, updates counter.

        Args:
            score: Current epoch metric score.

        Returns:
            bool: True if training should stop.
        """
        val_score = -score if self.mode == "min" else score
        
        if self.best_score is None:
            self.best_score = val_score
        elif val_score < self.best_score + self.min_delta:
            self.counter += 1
            logger.info(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_score
            self.counter = 0
        return self.early_stop


class Trainer:
    """Manages the training and validation execution of PyTorch models."""

    def __init__(
        self,
        config: Config,
        model: nn.Module,
        feature_extractor: FeatureExtractor,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None,
        class_weights: Optional[torch.Tensor] = None
    ):
        """Initializes trainer components."""
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() and config.device == "cuda" else "cpu")
        self.model = model.to(self.device)
        self.feature_extractor = feature_extractor.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler

        # Set loss function
        loss_type = self.config.training.loss_type.lower()
        if loss_type in ["bce", "w_bce"]:
            if class_weights is not None:
                # class_weights: [weight_real, weight_spoof] -> pos_weight = weight_spoof / weight_real
                pos_weight = torch.tensor([class_weights[1] / class_weights[0]])
                self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(self.device))
            else:
                self.criterion = nn.BCEWithLogitsLoss()
        else:
            if class_weights is not None:
                weights = class_weights.to(self.device)
                self.criterion = nn.CrossEntropyLoss(weight=weights)
            else:
                self.criterion = nn.CrossEntropyLoss()

        # Tensorboard writer
        log_dir = self.config.paths.tb_log_dir / self.config.experiment_name
        self.writer = SummaryWriter(log_dir=str(log_dir))
        
        # Setup early stopping
        es_opt = self.config.training.early_stopping
        self.early_stopping = EarlyStopping(
            patience=es_opt["patience"],
            min_delta=es_opt["min_delta"],
            mode="min"  # Stop based on validation loss/EER minimization
        )

    def train_epoch(self) -> float:
        """Trains the model for one epoch.

        Returns:
            float: Average training loss.
        """
        self.model.train()
        total_loss = 0.0

        for waveforms, labels in self.train_loader:
            waveforms = waveforms.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            # Model forward pass
            if self.config.model.name.lower() == "sincnet_gat":
                logits = self.model(waveforms)
            else:
                # Extract features on-the-fly (features: B, channels, freq, time)
                features = self.feature_extractor(waveforms)
                
                # Apply SpecAverage feature augmentation during training
                if self.config.augmentation.enabled and self.config.augmentation.spec_average.enabled:
                    from src.features.augmentations import apply_spec_average
                    features = apply_spec_average(
                        features,
                        time_mask_max=self.config.augmentation.spec_average.time_mask_max,
                        freq_mask_max=self.config.augmentation.spec_average.freq_mask_max
                    )
                logits = self.model(features)
            
            if self.config.training.loss_type.lower() in ["bce", "w_bce"]:
                log_odds = logits[:, 1] - logits[:, 0]
                loss = self.criterion(log_odds, labels.float())
            else:
                loss = self.criterion(logits, labels)
            
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * waveforms.size(0)

        return total_loss / len(self.train_loader.dataset)

    @torch.no_grad()
    def validate(self) -> Tuple[float, Dict[str, Any]]:
        """Evaluates the model on the validation set.

        Returns:
            Tuple[float, Dict[str, Any]]: Average validation loss and dict of metrics.
        """
        self.model.eval()
        total_loss = 0.0
        
        all_labels = []
        all_probs = []

        for waveforms, labels in self.val_loader:
            waveforms = waveforms.to(self.device)
            labels = labels.to(self.device)

            if self.config.model.name.lower() == "sincnet_gat":
                logits = self.model(waveforms)
            else:
                features = self.feature_extractor(waveforms)
                logits = self.model(features)
            
            if self.config.training.loss_type.lower() in ["bce", "w_bce"]:
                log_odds = logits[:, 1] - logits[:, 0]
                loss = self.criterion(log_odds, labels.float())
            else:
                loss = self.criterion(logits, labels)

            total_loss += loss.item() * waveforms.size(0)
            
            # Predict probabilities
            probs = torch.softmax(logits, dim=-1)
            
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

        avg_loss = total_loss / len(self.val_loader.dataset)
        metrics = calculate_metrics(np.array(all_labels), np.array(all_probs))
        
        return avg_loss, metrics

    def fit(self) -> Dict[str, Any]:
        """Runs the full training loop over multiple epochs.

        Returns:
            Dict[str, Any]: Validation metrics of the best model checkpoint.
        """
        best_eer = float("inf")
        best_metrics = {}
        checkpoint_dir = self.config.paths.checkpoint_dir
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(1, self.config.training.epochs + 1):
            train_loss = self.train_epoch()
            val_loss, val_metrics = self.validate()
            
            # Scheduler step
            if self.scheduler:
                if self.config.training.scheduler.type == "plateau":
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # Logging to standard logger
            val_eer = val_metrics["eer"]
            logger.info(
                f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val EER: {val_eer:.4f}"
            )

            # TensorBoard logging
            self.writer.add_scalar("Loss/Train", train_loss, epoch)
            self.writer.add_scalar("Loss/Val", val_loss, epoch)
            self.writer.add_scalar("Metrics/EER", val_eer, epoch)
            self.writer.add_scalar("Metrics/Accuracy", val_metrics["accuracy"], epoch)

            # Checkpoint saving
            if val_eer < best_eer:
                best_eer = val_eer
                best_metrics = val_metrics
                self.save_checkpoint(epoch, val_loss, val_eer, is_best=True)

            # Periodic checkpoint
            if epoch % 5 == 0:
                self.save_checkpoint(epoch, val_loss, val_eer, is_best=False)

            # Early stopping check
            if self.config.training.early_stopping["enabled"]:
                if self.early_stopping(val_eer):  # Evaluate EER improvement
                    logger.info("Early stopping triggered. Training stopped.")
                    break

        self.writer.close()
        return best_metrics

    def save_checkpoint(self, epoch: int, loss: float, eer: float, is_best: bool = False) -> None:
        """Saves a model checkpoint dictionary.

        Args:
            epoch: Current epoch number.
            loss: Current validation loss.
            eer: Current validation EER score.
            is_best: True to save as the best overall model weights.
        """
        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss": loss,
            "val_eer": eer,
            "config": self.config
        }
        
        prefix = "best_model" if is_best else f"checkpoint_epoch_{epoch}"
        save_path = self.config.paths.checkpoint_dir / f"{prefix}.pt"
        torch.save(state, save_path)
        logger.info(f"Saved checkpoint: {save_path.name}")
