# VoiceSentinel

### Robust & Stereo-Aware Audio Deepfake Detection

VoiceSentinel is a modular, research-grade, and production-ready PyTorch framework for robust audio deepfake and voice spoofing detection. It unifies two key research papers:
1. **Paper 4**: *"A Study on Data Augmentation in Voice Anti-Spoofing"* (robust augmentation, feature engineering).
2. **Paper 2**: *"Betray Oneself: M2S-ADD"* (mono-to-stereo conversion and stereo-aware representation learning).

---

## Project Structure

The project has the following directory structure:
```text
VoiceSentinel/
├── configs/          # Configuration files (YAML)
├── data/             # Dataset directory (ASVspoof2019 logical access)
├── docs/             # Documentation and diagrams
├── notebooks/        # Jupyter notebooks for exploration
├── src/              # Source code
│   ├── data/         # Dataset loaders and parsing logic
│   ├── features/     # Feature extraction (LFCC, Spectrogram, etc.)
│   ├── models/       # Models (Baseline, Paper2, Paper4, Combined)
│   ├── training/     # PyTorch training loops, losses, optimization
│   ├── evaluation/   # EER calculation and assessment scripts
│   ├── inference/    # Audio inference pipeline
│   └── utils/        # Utilities (logging, configuration, plotting)
├── checkpoints/      # Saved weights and models
├── outputs/          # Experiment logs, results, and plots
├── app.py            # Gradio Web UI
├── train.py          # Main training execution script
├── evaluate.py       # Main evaluation script
├── inference.py      # CLI inference script
├── requirements.txt  # Python requirements
├── README.md         # This readme file
└── LICENSE           # MIT License
```

---

## Setup Instructions

### Prerequisites
- Python 3.11+
- FFmpeg (installed on system path for audio loading and pydub/torchaudio conversion)

### Installation
1. Clone the repository and navigate to the directory:
   ```bash
   cd VoiceSentinel
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # Linux/macOS:
   source .venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
