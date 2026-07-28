import logging
import random
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.utils.audio import load_audio, pad_crop_audio, normalize_audio
from src.utils.config import Config
from src.features.augmentations import compress_audio, apply_channel_effects

logger = logging.getLogger(__name__)

# Map string labels to numeric targets: bonafide is 0 (real), spoof is 1 (fake)
LABEL_MAPPING = {"bonafide": 0, "spoof": 1}
INV_LABEL_MAPPING = {v: k for k, v in LABEL_MAPPING.items()}


class ASVspoofProtocolParser:
    """Parses ASVspoof2019 Logical Access protocol files."""

    @staticmethod
    def parse(protocol_file_path: Union[Path, str]) -> pd.DataFrame:
        """Parses the protocol file and returns a pandas DataFrame.

        Args:
            protocol_file_path: Path to the protocol text file.

        Returns:
            pd.DataFrame: Columns: [speaker_id, audio_id, system_id, attack_type, label]
        """
        path = Path(protocol_file_path)
        if not path.exists():
            raise FileNotFoundError(f"Protocol file not found at: {path}")

        records = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    records.append({
                        "speaker_id": parts[0],
                        "audio_id": parts[1],
                        "system_id": parts[2],
                        "attack_type": parts[3],
                        "label": parts[4]
                    })
                elif len(parts) == 2:  # Safe fallback for custom evaluation keys
                    records.append({
                        "speaker_id": "unknown",
                        "audio_id": parts[0],
                        "system_id": "-",
                        "attack_type": "-",
                        "label": parts[1]
                    })

        df = pd.DataFrame(records)
        if df.empty:
            raise ValueError(f"Protocol file is empty or formatted incorrectly: {path}")
        
        logger.info(f"Successfully parsed {len(df)} records from {path.name}")
        return df


def get_audio_path(asvspoof_root: Path, partition: str, audio_id: str) -> Path:
    """Finds the absolute path of an audio file under ASVspoof2019 structure.

    Args:
        asvspoof_root: The root path of the ASVspoof2019 dataset.
        partition: "train", "dev", or "eval".
        audio_id: The identifier of the audio file.

    Returns:
        Path: Resolved absolute path to the audio file.
    """
    folder_name = f"ASVspoof2019_LA_{partition}"
    # Standard candidates with and without inner 'LA' folder, support both flac and wav
    candidates = [
        asvspoof_root / "LA" / folder_name / "flac" / f"{audio_id}.flac",
        asvspoof_root / "LA" / folder_name / "flac" / f"{audio_id}.wav",
        asvspoof_root / folder_name / "flac" / f"{audio_id}.flac",
        asvspoof_root / folder_name / "flac" / f"{audio_id}.wav",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    # Default to the first candidate if file does not exist (will raise error in loader)
    return candidates[0]


class ASVspoofDataset(Dataset):
    """Dataset class for ASVspoof2019 Logical Access."""

    def __init__(
        self,
        protocol_file: Union[Path, str],
        asvspoof_root: Union[Path, str],
        partition: str,
        target_sr: int = 16000,
        duration: float = 4.0,
        padding_type: str = "wrap",
        crop_type: str = "random",
        cache: bool = False,
        config: Optional[Config] = None
    ):
        """Initializes the dataset and checks file presence.

        Args:
            protocol_file: Path to the protocol txt file.
            asvspoof_root: Root directory of ASVspoof2019.
            partition: Dataset subset: "train", "dev", or "eval".
            target_sr: Target sample rate for features.
            duration: Normalized audio duration in seconds.
            padding_type: Waveform padding mode ("zero" or "wrap").
            crop_type: Waveform cropping mode ("random" or "center").
            cache: Whether to cache loaded waveforms in memory.
        """
        self.protocol_file = Path(protocol_file)
        self.asvspoof_root = Path(asvspoof_root)
        self.partition = partition
        self.target_sr = target_sr
        self.duration = duration
        self.padding_type = padding_type
        self.crop_type = crop_type
        self.cache = cache
        self.config = config

        # Parse protocol
        self.df = ASVspoofProtocolParser.parse(self.protocol_file)
        
        # Waveform memory caching
        self.waveform_cache: Dict[str, torch.Tensor] = {}

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Gets processed audio waveform and its label at index.

        Args:
            idx: Index of sample.

        Returns:
            Tuple[torch.Tensor, int]: (waveform tensor of shape (1, samples), label integer)
        """
        row = self.df.iloc[idx]
        audio_id = row["audio_id"]
        label_str = row["label"]
        label = LABEL_MAPPING.get(label_str, 1)  # Default to spoof if label unknown

        # Check in cache
        if self.cache and audio_id in self.waveform_cache:
            return self.waveform_cache[audio_id], label

        audio_path = get_audio_path(self.asvspoof_root, self.partition, audio_id)
        
        # Load waveform
        waveform, sr = load_audio(audio_path, target_sr=self.target_sr)
        
        # Apply raw audio augmentations if in training partition
        if self.partition == "train" and self.config is not None and self.config.augmentation.enabled:
            # Compression augmentation
            comp_opt = self.config.augmentation.compression
            if comp_opt.enabled and random.random() < 0.5:
                waveform = compress_audio(
                    waveform, 
                    sample_rate=sr, 
                    codec=comp_opt.codec, 
                    bitrate=comp_opt.bitrate
                )
            
            # Channel augmentation
            chan_opt = self.config.augmentation.channel
            if chan_opt.enabled and random.random() < 0.5:
                waveform = apply_channel_effects(
                    waveform, 
                    sample_rate=sr, 
                    ir_path=chan_opt.impulse_response_path
                )
        
        # Normalize and pad/crop
        waveform = normalize_audio(waveform)
        waveform = pad_crop_audio(
            waveform,
            sample_rate=sr,
            target_duration=self.duration,
            padding_type=self.padding_type,
            crop_type=self.crop_type
        )

        if self.cache:
            self.waveform_cache[audio_id] = waveform

        return waveform, label


def collate_fn(batch: list) -> Tuple[torch.Tensor, torch.Tensor]:
    """Custom collate function for DataLoader batches.

    Args:
        batch: List of tuples (waveform, label) from ASVspoofDataset.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Waveforms batch (B, 1, samples),
                                           Labels batch (B,)
    """
    waveforms = [item[0] for item in batch]
    labels = [item[1] for item in batch]

    waveforms_stacked = torch.stack(waveforms, dim=0)
    labels_tensor = torch.tensor(labels, dtype=torch.long)

    return waveforms_stacked, labels_tensor
