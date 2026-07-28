from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml


@dataclass
class PathsConfig:
    """Paths for data, checkpoints, and logging."""
    data_dir: Path
    asvspoof_root: Path
    checkpoint_dir: Path
    output_dir: Path
    tb_log_dir: Path


@dataclass
class AudioConfig:
    """Basic audio loading and cropping properties."""
    sample_rate: int
    duration: float
    padding_type: str
    crop_type: str


@dataclass
class FeaturesConfig:
    """Feature extraction parameters (LFCC, Spectrogram, etc.)."""
    type: str
    n_fft: int
    win_length: int
    hop_length: int
    n_lfcc: int
    n_mels: int
    f_min: float
    f_max: float


@dataclass
class CompressionConfig:
    """Compression augmentation settings (Paper 4)."""
    enabled: bool
    codec: str
    bitrate: str


@dataclass
class ChannelConfig:
    """Channel impulse response augmentation settings (Paper 4)."""
    enabled: bool
    impulse_response_path: Optional[Path]


@dataclass
class SpecAverageConfig:
    """SpecAverage spectrogram masking settings (Paper 4)."""
    enabled: bool
    time_mask_max: int
    freq_mask_max: int


@dataclass
class AugmentationConfig:
    """Consolidated audio augmentations."""
    enabled: bool
    compression: CompressionConfig
    channel: ChannelConfig
    spec_average: SpecAverageConfig


@dataclass
class SincNetConfig:
    """SincNet parameters for raw waveform modeling (Paper 2)."""
    out_channels: int
    kernel_size: int


@dataclass
class GATConfig:
    """Graph Attention Network parameters (Paper 2)."""
    in_features: int
    out_features: int
    heads: int
    dropout: float


@dataclass
class ModelConfig:
    """Overall model configuration."""
    name: str
    sincnet: SincNetConfig
    gat: GATConfig
    num_classes: int


@dataclass
class SchedulerConfig:
    """Learning rate scheduler settings."""
    type: str
    step_size: int
    gamma: float
    patience: int


@dataclass
class TrainingConfig:
    """Hyperparameters for model training."""
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    optimizer: str
    scheduler: SchedulerConfig
    loss_type: str
    class_weights: List[float]
    early_stopping: Dict[str, Any]


@dataclass
class Config:
    """Top-level configuration mapping the yaml structure."""
    experiment_name: str
    mode: str
    device: str
    seed: int
    paths: PathsConfig
    audio: AudioConfig
    features: FeaturesConfig
    augmentation: AugmentationConfig
    model: ModelConfig
    training: TrainingConfig

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        """Build Config object from a parsed dictionary."""
        paths_data = data["paths"]
        paths = PathsConfig(
            data_dir=Path(paths_data["data_dir"]),
            asvspoof_root=Path(paths_data["asvspoof_root"]),
            checkpoint_dir=Path(paths_data["checkpoint_dir"]),
            output_dir=Path(paths_data["output_dir"]),
            tb_log_dir=Path(paths_data["tb_log_dir"])
        )

        audio_data = data["audio"]
        audio = AudioConfig(
            sample_rate=int(audio_data["sample_rate"]),
            duration=float(audio_data["duration"]),
            padding_type=str(audio_data["padding_type"]),
            crop_type=str(audio_data["crop_type"])
        )

        feat_data = data["features"]
        features = FeaturesConfig(
            type=str(feat_data["type"]),
            n_fft=int(feat_data["n_fft"]),
            win_length=int(feat_data["win_length"]),
            hop_length=int(feat_data["hop_length"]),
            n_lfcc=int(feat_data["n_lfcc"]),
            n_mels=int(feat_data["n_mels"]),
            f_min=float(feat_data["f_min"]),
            f_max=float(feat_data["f_max"])
        )

        aug_data = data["augmentation"]
        comp_data = aug_data["compression"]
        compression = CompressionConfig(
            enabled=bool(comp_data["enabled"]),
            codec=str(comp_data["codec"]),
            bitrate=str(comp_data["bitrate"])
        )

        chan_data = aug_data["channel"]
        ir_path = chan_data.get("impulse_response_path")
        channel = ChannelConfig(
            enabled=bool(chan_data["enabled"]),
            impulse_response_path=Path(ir_path) if ir_path else None
        )

        spec_avg_data = aug_data["spec_average"]
        spec_average = SpecAverageConfig(
            enabled=bool(spec_avg_data["enabled"]),
            time_mask_max=int(spec_avg_data["time_mask_max"]),
            freq_mask_max=int(spec_avg_data["freq_mask_max"])
        )

        augmentation = AugmentationConfig(
            enabled=bool(aug_data["enabled"]),
            compression=compression,
            channel=channel,
            spec_average=spec_average
        )

        model_data = data["model"]
        sincnet_data = model_data["sincnet"]
        sincnet = SincNetConfig(
            out_channels=int(sincnet_data["out_channels"]),
            kernel_size=int(sincnet_data["kernel_size"])
        )

        gat_data = model_data["gat"]
        gat = GATConfig(
            in_features=int(gat_data["in_features"]),
            out_features=int(gat_data["out_features"]),
            heads=int(gat_data["heads"]),
            dropout=float(gat_data["dropout"])
        )

        model = ModelConfig(
            name=str(model_data["name"]),
            sincnet=sincnet,
            gat=gat,
            num_classes=int(model_data["num_classes"])
        )

        train_data = data["training"]
        sched_data = train_data["scheduler"]
        scheduler = SchedulerConfig(
            type=str(sched_data["type"]),
            step_size=int(sched_data["step_size"]),
            gamma=float(sched_data["gamma"]),
            patience=int(sched_data["patience"])
        )

        training = TrainingConfig(
            epochs=int(train_data["epochs"]),
            batch_size=int(train_data["batch_size"]),
            learning_rate=float(train_data["learning_rate"]),
            weight_decay=float(train_data["weight_decay"]),
            optimizer=str(train_data["optimizer"]),
            scheduler=scheduler,
            loss_type=str(train_data["loss_type"]),
            class_weights=list(map(float, train_data["class_weights"])),
            early_stopping=dict(train_data["early_stopping"])
        )

        return cls(
            experiment_name=str(data["experiment_name"]),
            mode=str(data["mode"]),
            device=str(data["device"]),
            seed=int(data["seed"]),
            paths=paths,
            audio=audio,
            features=features,
            augmentation=augmentation,
            model=model,
            training=training
        )

    @classmethod
    def load_from_yaml(cls, path: str | Path) -> "Config":
        """Load YAML configuration from a path."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)
