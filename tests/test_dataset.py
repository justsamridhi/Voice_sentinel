import tempfile
from pathlib import Path
import numpy as np
import pytest
import soundfile as sf
import torch
from torch.utils.data import DataLoader

from src.data.dataset import ASVspoofDataset, ASVspoofProtocolParser, collate_fn
from src.data.utils import compute_dataset_stats, verify_dataset_integrity


@pytest.fixture
def mock_dataset_env():
    """Sets up a temporary directory containing mock ASVspoof2019 protocols and files."""
    temp_dir = tempfile.TemporaryDirectory()
    root_path = Path(temp_dir.name)

    # Create directories
    flac_dir = root_path / "LA" / "ASVspoof2019_LA_train" / "flac"
    flac_dir.mkdir(parents=True, exist_ok=True)
    protocol_dir = root_path / "LA" / "ASVspoof2019_LA_protocols"
    protocol_dir.mkdir(parents=True, exist_ok=True)

    # Write synthetic audio files (0.5 second wavs)
    sr = 16000
    t = np.linspace(0, 0.5, int(sr * 0.5))
    waveform_data = 0.5 * np.sin(2 * np.pi * 440.0 * t)

    audio_ids = ["LA_T_1000001", "LA_T_1000002", "LA_T_1000003", "LA_T_1000004"]
    for aid in audio_ids:
        sf.write(str(flac_dir / f"{aid}.wav"), waveform_data, sr)

    # Write protocol file
    protocol_lines = [
        "LA_0079 LA_T_1000001 - - bonafide\n",
        "LA_0079 LA_T_1000002 - A01 spoof\n",
        "LA_0080 LA_T_1000003 - A02 spoof\n",
        "LA_0081 LA_T_1000004 - A03 spoof\n",
    ]
    protocol_file = protocol_dir / "ASVspoof2019.LA.protocol.train.txt"
    with open(protocol_file, "w", encoding="utf-8") as f:
        f.writelines(protocol_lines)

    yield root_path, protocol_file

    temp_dir.cleanup()


def test_protocol_parser(mock_dataset_env):
    root_path, protocol_file = mock_dataset_env
    df = ASVspoofProtocolParser.parse(protocol_file)

    assert len(df) == 4
    assert list(df.columns) == ["speaker_id", "audio_id", "system_id", "attack_type", "label"]
    assert df.iloc[0]["audio_id"] == "LA_T_1000001"
    assert df.iloc[0]["label"] == "bonafide"
    assert df.iloc[1]["label"] == "spoof"


def test_dataset_loading(mock_dataset_env):
    root_path, protocol_file = mock_dataset_env
    dataset = ASVspoofDataset(
        protocol_file=protocol_file,
        asvspoof_root=root_path,
        partition="train",
        target_sr=16000,
        duration=1.0,  # 1.0 second target duration
        padding_type="wrap",
        cache=True
    )

    assert len(dataset) == 4
    waveform, label = dataset[0]

    # Target duration 1.0s @ 16000Hz = 16000 samples
    assert waveform.shape == (1, 16000)
    assert label == 0  # bonafide

    waveform_fake, label_fake = dataset[1]
    assert label_fake == 1  # spoof


def test_dataset_caching(mock_dataset_env):
    root_path, protocol_file = mock_dataset_env
    dataset = ASVspoofDataset(
        protocol_file=protocol_file,
        asvspoof_root=root_path,
        partition="train",
        duration=1.0,
        cache=True
    )

    # Initial load will populate cache
    assert len(dataset.waveform_cache) == 0
    w1, l1 = dataset[0]
    assert "LA_T_1000001" in dataset.waveform_cache

    # Subsequent load retrieves from cache
    w2, l2 = dataset[0]
    assert torch.allclose(w1, w2)


def test_dataset_stats_and_integrity(mock_dataset_env):
    root_path, protocol_file = mock_dataset_env
    dataset = ASVspoofDataset(
        protocol_file=protocol_file,
        asvspoof_root=root_path,
        partition="train"
    )

    stats = compute_dataset_stats(dataset)
    assert stats["total_samples"] == 4
    assert stats["counts"]["bonafide"] == 1
    assert stats["counts"]["spoof"] == 3
    # Check that recommended weights balance the classes
    assert stats["recommended_weights"][0] > stats["recommended_weights"][1]

    # Verify integrity passes
    assert verify_dataset_integrity(dataset, num_checks=2) is True


def test_collate_and_dataloader(mock_dataset_env):
    root_path, protocol_file = mock_dataset_env
    dataset = ASVspoofDataset(
        protocol_file=protocol_file,
        asvspoof_root=root_path,
        partition="train",
        duration=1.0
    )
    
    loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)
    
    batches = list(loader)
    assert len(batches) == 2
    
    waveforms, labels = batches[0]
    # batch_size=2, channels=1, samples=16000
    assert waveforms.shape == (2, 1, 16000)
    assert labels.shape == (2,)
    assert labels[0] == 0
    assert labels[1] == 1
