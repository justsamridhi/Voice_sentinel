# VoiceSentinel

### Robust & Stereo-Aware Audio Deepfake Detection System

VoiceSentinel is a modular, research-grade, and production-ready PyTorch framework for robust audio deepfake and voice spoofing detection. It unifies state-of-the-art architectures, robust spectral features, and stereo-aware representation learning into an intuitive, high-tech dashboard.

---

## Detection Pipelines

VoiceSentinel features four modular detection pipelines:

- **Mode A (Standard Acoustic Analysis)**: Mono Linear Frequency Cepstral Coefficients (LFCC) paired with a ResNet34 backbone.
- **Mode B (Robust Data Augmentation)**: Log-Spectrogram representation trained with SpecAverage data augmentation and BCE loss.
- **Mode C (Spatial-Graph Attention Pipeline)**: Mono-to-Stereo (M2S) conversion combined with raw waveform SincNet filters, ResNet feature extractors, and Graph Attention Networks (GAT).
- **Mode D (Comprehensive Stereo Analysis)**: Dual-channel stereo Log-Spectrogram features analyzed using a stereo-aware ResNet34 architecture.

---

## Project Structure

```text
VoiceSentinel/
├── configs/          # Yaml configuration files per detection mode
├── data/             # ASVspoof dataset directory
├── docs/             # Diagrams and documentation
├── src/              # Core Source Code
│   ├── data/         # Dataset parsing and PyTorch loaders
│   ├── features/     # Feature extractors (LFCC, Spectrogram, etc.)
│   ├── models/       # Neural network architectures (ResNet, SincNet, GAT, StereoNet)
│   ├── training/     # Training loops, loss functions, and optimizers
│   ├── evaluation/   # Metric calculations (EER, min t-DCF)
│   ├── inference/    # Real-time audio inference pipeline
│   └── utils/        # Visualization helpers, logging, audio utils
├── outputs/          # Model checkpoints, ablation logs, and results
├── app.py            # Gradio Web Dashboard (Cyber-Sentinel UI)
├── train.py          # Model training execution script
├── evaluate.py       # Evaluation script
├── inference.py      # Command-line audio inference script
├── requirements.txt  # Python package dependencies
└── README.md         # Documentation
```

---

## Quick Start & Setup Guide

### Prerequisites
- **Python 3.11+**
- **FFmpeg** installed on system path (required for `pydub` and audio loading)

---

### Step 1: Environment Setup & Virtual Environment Activation

#### Windows (Command Prompt / PowerShell)
```cmd
# Create virtual environment if not already created
python -m venv .venv

# Activate the virtual environment (.venv)
.venv\Scripts\activate

# (Alternatively, if using 'venv')
venv\Scripts\activate
```

#### Linux / macOS
```bash
# Create virtual environment if not already created
python3 -m venv .venv

# Activate the virtual environment (.venv)
source .venv/bin/activate
```

---

### Step 2: Install Dependencies
With your virtual environment activated, install the required packages:
```bash
pip install -r requirements.txt
```

---

### Step 3: Run the Application

#### A. Web Dashboard (Gradio UI)
To launch the interactive **Cyber-Sentinel Command Center** dashboard:
```bash
python app.py
```
Once launched, open your browser and navigate to:
👉 **`http://127.0.0.1:7860`**

#### B. Command Line Inference (CLI)
To run quick inference on a single audio file via the terminal:
```bash
python inference.py --audio path/to/sample.wav --mode D
```

#### C. Train / Evaluate Models
- **Train a model pipeline**:
  ```bash
  python train.py --config configs/modes/mode_d_combined.yaml
  ```
- **Evaluate model performance**:
  ```bash
  python evaluate.py --config configs/modes/mode_d_combined.yaml --checkpoint outputs/checkpoints/best_model.pt
  ```

---

## License
Distributed under the MIT License. See `LICENSE` for more information.
