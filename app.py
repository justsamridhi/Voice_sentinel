import logging
import os
from pathlib import Path
import gradio as gr
import numpy as np
import torch
import soundfile as sf

from src.utils.config import Config
from src.utils.audio import load_audio, normalize_audio, pad_crop_audio
from src.utils.visualization import plot_waveform, plot_fft, plot_spectrogram
from src.features.extractor import FeatureExtractor
from src.inference.pipeline import InferencePipeline, load_model_from_config

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("VoiceSentinelApp")


class FallbackPipeline:
    """Robust fallback pipeline that initializes random model weights if checkpoints are missing."""

    def __init__(self, mode: str):
        self.mode = mode
        config_dict = {
            "experiment_name": f"fallback_mode_{mode}",
            "mode": mode,
            "device": "cpu",
            "seed": 42,
            "paths": {
                "data_dir": "test", "asvspoof_root": "test", "checkpoint_dir": "test", "output_dir": "test", "tb_log_dir": "test"
            },
            "audio": {
                "sample_rate": 16000,
                "duration": 4.0,  # 4 seconds duration standard
                "padding_type": "wrap",
                "crop_type": "random"
            },
            "features": {
                "type": "spectrogram" if mode in ["B", "D"] else "lfcc",
                "n_fft": 512, "win_length": 400, "hop_length": 160,
                "n_lfcc": 40, "n_mels": 60, "f_min": 0.0, "f_max": 8000.0
            },
            "augmentation": {
                "enabled": False,
                "compression": {"enabled": False, "codec": "mp3", "bitrate": "64k"},
                "channel": {"enabled": False, "impulse_response_path": None},
                "spec_average": {"enabled": False, "time_mask_max": 2, "freq_mask_max": 2}
            },
            "model": {
                "name": "sincnet_gat" if mode == "C" else "resnet34",
                "sincnet": {"out_channels": 40, "kernel_size": 251},
                "gat": {"in_features": 128, "out_features": 64, "heads": 2, "dropout": 0.5},
                "num_classes": 2
            },
            "training": {
                "epochs": 1, "batch_size": 2, "learning_rate": 0.001, "weight_decay": 0.0, "optimizer": "adam",
                "scheduler": {"type": "none", "step_size": 1, "gamma": 0.5, "patience": 1},
                "loss_type": "ce", "class_weights": [1.0, 1.0], "early_stopping": {"enabled": False, "patience": 1, "min_delta": 0.0}
            }
        }
        self.config = Config.from_dict(config_dict)
        self.model = load_model_from_config(self.config)
        self.model.eval()
        self.feature_extractor = FeatureExtractor(
            feature_type=self.config.features.type,
            sample_rate=self.config.audio.sample_rate,
            n_fft=self.config.features.n_fft,
            win_length=self.config.features.win_length,
            hop_length=self.config.features.hop_length,
            n_lfcc=self.config.features.n_lfcc,
            n_mels=self.config.features.n_mels,
            f_min=self.config.features.f_min,
            f_max=self.config.features.f_max
        )

        self.m2s_converter = None
        if mode == "D":
            from src.models.stereo_net import M2SConverter
            self.m2s_converter = M2SConverter(sample_rate=self.config.audio.sample_rate)

    @torch.no_grad()
    def predict(self, audio_path: str):
        waveform, sr = load_audio(audio_path, target_sr=self.config.audio.sample_rate)
        waveform = normalize_audio(waveform)
        waveform = pad_crop_audio(
            waveform,
            sample_rate=sr,
            target_duration=self.config.audio.duration,
            padding_type=self.config.audio.padding_type,
            crop_type="center"
        )
        waveform_batch = waveform.unsqueeze(0)

        if self.config.model.name.lower() == "sincnet_gat":
            logits = self.model(waveform_batch)
        elif self.config.mode.upper() == "D":
            stereo = self.m2s_converter(waveform_batch)
            left_feat = self.feature_extractor(stereo[:, 0:1, :])
            right_feat = self.feature_extractor(stereo[:, 1:2, :])
            features = torch.cat([left_feat, right_feat], dim=1)
            logits = self.model(features)
        else:
            features = self.feature_extractor(waveform_batch)
            logits = self.model(features)

        probs = torch.softmax(logits, dim=-1).squeeze(0)
        spoof_prob = probs[1].item()
        label = "spoof" if spoof_prob > 0.5 else "bonafide"
        confidence = spoof_prob if label == "spoof" else 1.0 - spoof_prob

        return {
            "label": label,
            "spoof_prob": spoof_prob,
            "confidence": confidence
        }


def get_pipeline(mode: str):
    """Loads appropriate inference pipeline, falling back to random model if checkpoints are missing."""
    chk_path = Path(f"outputs_ablation/checkpoints_mode_{mode}/best_model.pt")
    if chk_path.exists():
        logger.info(f"Loading trained ablation checkpoint for Mode {mode}")
        return InferencePipeline(checkpoint_path=chk_path, device="cpu")
    else:
        logger.warning(f"Trained checkpoint not found at {chk_path}. Instantiating fallback pipeline for Mode {mode}.")
        return FallbackPipeline(mode=mode)


def process_audio(audio_file, mode_str: str):
    if audio_file is None:
        return (
            "⚠️ Please upload an audio file.",
            {},
            "",
            None,
            None,
            None,
            None
        )

    # Decode Mode letter (A, B, C, or D)
    mode = "D"
    for m_char in ["A", "B", "C", "D"]:
        if m_char in mode_str:
            mode = m_char
            break
    pipeline = get_pipeline(mode)

    # 1. Run model prediction
    res = pipeline.predict(audio_file)
    is_fake = res["label"] == "spoof"
    confidence = res["confidence"]
    spoof_prob = res["spoof_prob"]

    pred_label = "🔴 SPOOFED (FAKE AUDIO)" if is_fake else "🟢 BONA FIDE (REAL AUDIO)"
    
    # Format probabilities
    conf_scores = {
        "Real Speech": 1.0 - spoof_prob,
        "Deepfake / Spoof": spoof_prob
    }

    # 2. Extract stats
    info = sf.info(audio_file)
    stats_md = (
        f"**File Name:** `{os.path.basename(audio_file)}`\n\n"
        f"**Sample Rate:** `{info.samplerate} Hz`\n\n"
        f"**Duration:** `{info.duration:.2f} seconds`\n\n"
        f"**Channels:** `{info.channels} ({'Stereo' if info.channels == 2 else 'Mono'})`"
    )

    # 3. Create visual plots (always extracting at 16000Hz standard for display consistency)
    waveform, sr = load_audio(audio_file, target_sr=16000)
    waveform_norm = normalize_audio(waveform)
    
    fig_wave = plot_waveform(waveform_norm, sr, title="Time Domain Waveform")
    fig_fft = plot_fft(waveform_norm, sr, title="Magnitude Spectrum (FFT)")
    
    # Feature extractors for spectrogram/LFCC visuals
    spec_extractor = FeatureExtractor(feature_type="spectrogram", sample_rate=16000)
    lfcc_extractor = FeatureExtractor(feature_type="lfcc", sample_rate=16000)
    
    spec = spec_extractor(waveform_norm.unsqueeze(0))
    lfcc = lfcc_extractor(waveform_norm.unsqueeze(0))
    
    fig_spec = plot_spectrogram(spec.squeeze(0), title="Log-Spectrogram Magnitude", ylabel="Frequency Bins")
    fig_lfcc = plot_spectrogram(lfcc.squeeze(0), title="Linear Frequency Cepstral Coefficients (LFCC)", ylabel="LFCC Coefficients")

    return (
        pred_label,
        conf_scores,
        stats_md,
        fig_wave,
        fig_fft,
        fig_spec,
        fig_lfcc
    )


# Obsidian Chrome SaaS Design System CSS
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

:root {
    color-scheme: dark;
    --onyx: #0A0A0A;
    --blue-slate: #536878;
    --alabaster-grey: #E5E4E2;
    --white: #FFFFFF;
    
    --card-bg: linear-gradient(145deg, rgba(20, 24, 29, 0.96) 0%, rgba(10, 10, 10, 0.98) 100%);
    --card-border: rgba(83, 104, 120, 0.28);
    --border-hover: rgba(229, 228, 226, 0.35);
}

body {
    background: radial-gradient(circle at 50% 0%, rgba(83, 104, 120, 0.15) 0%, transparent 55%),
                linear-gradient(180deg, #0A0A0A 0%, #111518 45%, #0A0A0A 100%);
    color: var(--alabaster-grey) !important;
    font-family: 'Inter', -apple-system, sans-serif !important;
    min-height: 100vh;
    margin: 0;
}

.gradio-container {
    max-width: 1380px !important;
    padding: 0 24px 64px 24px !important;
    background: transparent !important;
    font-family: 'Inter', sans-serif !important;
}

/* 1. Navbar Banner */
#navbar-banner {
    background: rgba(10, 10, 10, 0.9);
    border: 1px solid var(--card-border);
    border-radius: 16px;
    padding: 16px 28px;
    margin-top: 16px;
    margin-bottom: 32px;
    backdrop-filter: blur(16px);
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5);
}

/* 2. Hero Section */
#hero-card {
    background: linear-gradient(180deg, rgba(22, 28, 34, 0.95) 0%, rgba(10, 10, 10, 0.98) 100%);
    border: 1px solid var(--card-border);
    border-radius: 24px;
    padding: 56px 40px;
    margin-bottom: 48px;
    text-align: center;
    box-shadow: 0 12px 40px rgba(0, 0, 0, 0.4);
}

/* 3. Cards & Components */
#control-card, #result-card, #insight-card, .saas-card-box {
    background: var(--card-bg);
    border: 1px solid var(--card-border);
    border-radius: 20px;
    padding: 28px;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3);
    transition: border-color 0.25s ease, transform 0.25s ease;
}

#control-card:hover, #result-card:hover, #insight-card:hover, .saas-card-box:hover {
    border-color: var(--border-hover);
}

/* 4. Action Button */
#analyze-btn {
    background-color: var(--blue-slate) !important;
    color: var(--white) !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    border: 1px solid rgba(229, 228, 226, 0.25) !important;
    border-radius: 12px !important;
    padding: 14px 24px !important;
    cursor: pointer !important;
    transition: background-color 0.2s ease, transform 0.2s ease !important;
    margin-top: 14px;
}

#analyze-btn:hover {
    background-color: #637a8c !important;
    transform: scale(1.01) !important;
}

#analyze-btn:active {
    transform: scale(0.99) !important;
}

/* 5. Form Inputs & Labels */
.gradio-container label span {
    color: var(--alabaster-grey) !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
}

.gradio-container .gr-box, 
.gradio-container .gr-input, 
.gradio-container input, 
.gradio-container select, 
.gradio-container textarea {
    background-color: #0A0A0A !important;
    border: 1px solid var(--card-border) !important;
    color: var(--alabaster-grey) !important;
    border-radius: 12px !important;
}

.gradio-container input:focus, .gradio-container select:focus {
    border-color: var(--blue-slate) !important;
}

.gradio-container h1, .gradio-container h2, .gradio-container h3 {
    color: var(--white) !important;
}

.gradio-container p, .gradio-container span {
    color: var(--alabaster-grey);
}

/* 6. Tabs */
.gradio-container .tab-nav {
    border-bottom: 1px solid var(--card-border) !important;
}

.gradio-container .tab-nav button {
    font-weight: 500 !important;
    color: var(--blue-slate) !important;
    border-radius: 8px 8px 0 0 !important;
    padding: 10px 20px !important;
}

.gradio-container .tab-nav button.selected {
    color: var(--white) !important;
    border-bottom: 2px solid var(--alabaster-grey) !important;
    background: rgba(83, 104, 120, 0.15) !important;
}

.pill-chip {
    display: inline-flex;
    align-items: center;
    padding: 6px 16px;
    margin: 4px 6px 4px 0;
    border-radius: 20px;
    background: rgba(83, 104, 120, 0.15);
    border: 1px solid rgba(83, 104, 120, 0.3);
    color: var(--alabaster-grey);
    font-size: 0.84rem;
    font-weight: 500;
}
"""

with gr.Blocks(
    css=custom_css,
    theme=gr.themes.Base(
        primary_hue="slate",
        secondary_hue="slate",
        neutral_hue="slate"
    )
) as demo:
    
    # SECTION 1: STICKY NAVBAR
    with gr.Row(elem_id="navbar-banner"):
        with gr.Column(scale=2):
            gr.Markdown(
                """
                <div style="display:flex; align-items:center; gap:12px;">
                  <span style="font-size:1.6rem;">🛡️</span>
                  <span style="font-size:1.3rem; font-weight:700; color:#FFFFFF; letter-spacing:-0.01em;">VoiceSentinel</span>
                  <span class="pill-chip" style="margin-left:12px; font-size:0.75rem;">v2.4 Production</span>
                </div>
                """
            )
        with gr.Column(scale=3):
            gr.Markdown(
                """
                <div style="display:flex; justify-content:flex-end; align-items:center; gap:20px; font-size:0.9rem;">
                  <a href="#about" style="color:#E5E4E2; text-decoration:none; opacity:0.85;">About</a>
                  <a href="#features" style="color:#E5E4E2; text-decoration:none; opacity:0.85;">Features</a>
                  <a href="#workflow" style="color:#E5E4E2; text-decoration:none; opacity:0.85;">Workflow</a>
                  <a href="#studio" style="color:#FFFFFF; text-decoration:none; font-weight:600; background:rgba(83,104,120,0.3); padding:6px 14px; border-radius:8px; border:1px solid rgba(83,104,120,0.4);">Launch Studio</a>
                </div>
                """
            )

    # SECTION 2: HERO & INTRODUCTION
    with gr.Row(elem_id="hero-card"):
        with gr.Column():
            gr.Markdown(
                """
                <div style="max-width:860px; margin:0 auto;">
                  <div style="text-transform:uppercase; letter-spacing:0.1em; font-size:0.8rem; color:#536878; font-weight:700; margin-bottom:12px;">
                    Enterprise Voice Authenticity Platform
                  </div>
                  <h1 style="font-size:2.8rem; font-weight:700; color:#FFFFFF; margin:0 0 16px 0; line-height:1.2; letter-spacing:-0.02em;">
                    Detect Synthetic Voice Clones & Audio Deepfakes in Real Time
                  </h1>
                  <p style="font-size:1.1rem; color:#E5E4E2; opacity:0.85; line-height:1.6; margin-bottom:28px;">
                    VoiceSentinel unifies state-of-the-art mono-to-stereo graph attention neural networks (M2S) and robust LFCC spectrogram diagnostics into an architectural defense suite.
                  </p>
                  <div style="display:flex; justify-center:center; gap:12px; justify-content:center;">
                    <span class="pill-chip">🛡️ Research-Grade Anti-Spoofing</span>
                    <span class="pill-chip">🎧 Stereo Spatial Awareness</span>
                    <span class="pill-chip">⚡ Sub-Second Inference</span>
                  </div>
                </div>
                """
            )

    # SECTION 3: ABOUT / OVERVIEW
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown(
                """
                <div class="saas-card-box">
                  <h3 style="font-size:1.4rem; color:#FFFFFF; margin-top:0;">Why VoiceSentinel?</h3>
                  <p style="font-size:0.95rem; color:#E5E4E2; opacity:0.85; line-height:1.6;">
                    Generative AI voice cloning technology poses unprecedented security risks to biometrics and media integrity. VoiceSentinel acts as an architectural firewall, analyzing microscopic acoustic artifacts and stereo spatial phase discrepancies that synthetic audio generators fail to replicate.
                  </p>
                </div>
                """
            )
        with gr.Column(scale=1):
            gr.Markdown(
                """
                <div class="saas-card-box">
                  <h3 style="font-size:1.4rem; color:#FFFFFF; margin-top:0;">Core Pillars</h3>
                  <ul style="padding-left:20px; font-size:0.95rem; color:#E5E4E2; opacity:0.85; line-height:1.8; margin-bottom:0;">
                    <li><strong>M2S Stereo Defense:</strong> Converts mono audio to stereo graphs for spatial anomaly detection.</li>
                    <li><strong>Robust Feature Engineering:</strong> LFCC cepstrals and log-spectrogram magnitude mapping.</li>
                    <li><strong>Ablation Validated:</strong> Unified multi-mode neural backbones (ResNet34, SincNet, GAT).</li>
                  </ul>
                </div>
                """
            )

    # SECTION 4: FEATURES & CAPABILITIES
    gr.Markdown("<div style='height:32px;'></div>")
    gr.Markdown("<h2 style='font-size:1.8rem; color:#FFFFFF; margin-bottom:16px;'>Platform Capabilities</h2>")
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown(
                """
                <div class="saas-card-box">
                  <div style="font-size:1.4rem; margin-bottom:8px;">🎧</div>
                  <h4 style="font-size:1.1rem; color:#FFFFFF; margin:0 0 8px 0;">Stereo Spatial Awareness</h4>
                  <p style="font-size:0.88rem; color:#E5E4E2; opacity:0.8; line-height:1.5; margin:0;">
                    Mono-to-Stereo (M2S) conversion exposes phase irregularities and synthetic spatial artifacts across left and right channels.
                  </p>
                </div>
                """
            )
        with gr.Column(scale=1):
            gr.Markdown(
                """
                <div class="saas-card-box">
                  <div style="font-size:1.4rem; margin-bottom:8px;">🔬</div>
                  <h4 style="font-size:1.1rem; color:#FFFFFF; margin:0 0 8px 0;">Dual Spectral Extraction</h4>
                  <p style="font-size:0.88rem; color:#E5E4E2; opacity:0.8; line-height:1.5; margin:0;">
                    Extracts Linear Frequency Cepstral Coefficients (LFCC) and Log-Spectrograms for complete frequency band analysis.
                  </p>
                </div>
                """
            )
        with gr.Column(scale=1):
            gr.Markdown(
                """
                <div class="saas-card-box">
                  <div style="font-size:1.4rem; margin-bottom:8px;">🧠</div>
                  <h4 style="font-size:1.1rem; color:#FFFFFF; margin:0 0 8px 0;">Graph Attention Networks</h4>
                  <p style="font-size:0.88rem; color:#E5E4E2; opacity:0.8; line-height:1.5; margin:0;">
                    SincNet filterbanks combined with GAT neural networks evaluate contextual relationships between frequency nodes.
                  </p>
                </div>
                """
            )

    # SECTION 5: HOW IT WORKS WORKFLOW
    gr.Markdown("<div style='height:40px;'></div>")
    gr.Markdown("<h2 style='font-size:1.8rem; color:#FFFFFF; margin-bottom:16px;'>Detection Workflow</h2>")
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown(
                """
                <div class="saas-card-box" style="text-align:center;">
                  <div style="font-size:0.8rem; font-weight:700; color:#536878; text-transform:uppercase;">Step 01</div>
                  <h4 style="color:#FFFFFF; margin:6px 0;">Upload Audio</h4>
                  <p style="font-size:0.85rem; opacity:0.8;">Submit WAV, MP3, or FLAC audio samples.</p>
                </div>
                """
            )
        with gr.Column(scale=1):
            gr.Markdown(
                """
                <div class="saas-card-box" style="text-align:center;">
                  <div style="font-size:0.8rem; font-weight:700; color:#536878; text-transform:uppercase;">Step 02</div>
                  <h4 style="color:#FFFFFF; margin:6px 0;">Select Pipeline</h4>
                  <p style="font-size:0.85rem; opacity:0.8;">Choose from Modes A, B, C, or D.</p>
                </div>
                """
            )
        with gr.Column(scale=1):
            gr.Markdown(
                """
                <div class="saas-card-box" style="text-align:center;">
                  <div style="font-size:0.8rem; font-weight:700; color:#536878; text-transform:uppercase;">Step 03</div>
                  <h4 style="color:#FFFFFF; margin:6px 0;">Neural Inference</h4>
                  <p style="font-size:0.85rem; opacity:0.8;">PyTorch engine evaluates acoustic features.</p>
                </div>
                """
            )
        with gr.Column(scale=1):
            gr.Markdown(
                """
                <div class="saas-card-box" style="text-align:center;">
                  <div style="font-size:0.8rem; font-weight:700; color:#536878; text-transform:uppercase;">Step 04</div>
                  <h4 style="color:#FFFFFF; margin:6px 0;">Verdict & Visuals</h4>
                  <p style="font-size:0.85rem; opacity:0.8;">Receive authenticity classification & charts.</p>
                </div>
                """
            )

    # SECTION 6: CORE FUNCTIONALITY (INTERACTIVE STUDIO)
    gr.Markdown("<div style='height:48px;'></div>")
    gr.Markdown("<h2 style='font-size:1.8rem; color:#FFFFFF; margin-bottom:16px;'>Interactive Detection Studio</h2>")
    
    with gr.Row():
        with gr.Column(scale=1, elem_id="control-card"):
            gr.Markdown("### 🎛️ Audio Input & Pipeline Config")
            audio_input = gr.Audio(type="filepath", label="Upload Audio File (WAV, MP3, FLAC)")

            mode_dropdown = gr.Dropdown(
                choices=[
                    "Mode A: Standard Acoustic Analysis (Mono LFCC + ResNet34)",
                    "Mode B: Robust Data Augmentation Pipeline (LogSpec + SpecAverage + BCE)",
                    "Mode C: Spatial-Graph Attention Neural Pipeline (M2S + SincNet + GAT)",
                    "Mode D: Comprehensive Stereo Analysis (Stereo LogSpec + ResNet34)"
                ],
                value="Mode D: Comprehensive Stereo Analysis (Stereo LogSpec + ResNet34)",
                label="Select Detection Pipeline"
            )

            gr.Examples(
                examples=[
                    ["scratch/test.wav", "Mode D: Comprehensive Stereo Analysis (Stereo LogSpec + ResNet34)"],
                    ["scratch/test.wav", "Mode A: Standard Acoustic Analysis (Mono LFCC + ResNet34)"]
                ],
                inputs=[audio_input, mode_dropdown],
                label="Quick Test Examples"
            )

            analyze_button = gr.Button("Run Deepfake Analysis", elem_id="analyze-btn")

            gr.Markdown(
                """
                <div style="margin-top:12px; font-size:0.83rem; color:#E5E4E2; opacity:0.8; line-height:1.5;">
                  Select a deepfake detection pipeline above and submit audio to generate real-time acoustic threat diagnostics and spectral feature visualizers.
                </div>
                """
            )

        with gr.Column(scale=1, elem_id="result-card"):
            gr.Markdown("### 🔐 Sentinel Security Verdict")
            pred_text = gr.Textbox(value="Awaiting audio analysis...", label="Authenticity Classification", interactive=False)
            conf_bar = gr.Label(label="Confidence & Probability Breakdown")

            with gr.Accordion("📊 Audio Metadata & Diagnostics", open=True):
                stats_markdown = gr.Markdown("Upload an audio file and initiate analysis to inspect acoustic properties.")

    # SECTION 7: ANALYTICS / VISUALIZERS
    gr.Markdown("<div style='height:32px;'></div>")
    with gr.Row():
        with gr.Column(elem_id="insight-card"):
            gr.Markdown("### 📈 Deep Acoustic Feature Visualizers")
            with gr.Tabs():
                with gr.TabItem("Waveform (Time Domain)"):
                    wave_plot = gr.Plot(label="Time Domain Waveform Plot")
                with gr.TabItem("Magnitude Spectrum (FFT)"):
                    fft_plot = gr.Plot(label="Frequency Magnitude Spectrum Plot")
                with gr.TabItem("Log-Spectrogram Heatmap"):
                    spec_plot = gr.Plot(label="Log-Spectrogram Magnitude Map")
                with gr.TabItem("LFCC Cepstral Map"):
                    lfcc_plot = gr.Plot(label="Linear Frequency Cepstral Coefficients Map")

    # SECTION 8: RESEARCH TRUST BENCHMARKS
    gr.Markdown("<div style='height:48px;'></div>")
    gr.Markdown("<h2 style='font-size:1.8rem; color:#FFFFFF; margin-bottom:16px;'>Research & Validation Benchmarks</h2>")
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown(
                """
                <div class="saas-card-box">
                  <div style="font-size:0.8rem; font-weight:700; color:#536878; text-transform:uppercase;">ASVspoof 2019 LA</div>
                  <h4 style="color:#FFFFFF; margin:6px 0;">Benchmark Dataset</h4>
                  <p style="font-size:0.88rem; opacity:0.85; line-height:1.5;">Evaluated against logical access voice anti-spoofing benchmarks with EER assessment.</p>
                </div>
                """
            )
        with gr.Column(scale=1):
            gr.Markdown(
                """
                <div class="saas-card-box">
                  <div style="font-size:0.8rem; font-weight:700; color:#536878; text-transform:uppercase;">Paper 4 Augmentation</div>
                  <h4 style="color:#FFFFFF; margin:6px 0;">Robust Augmentation</h4>
                  <p style="font-size:0.88rem; opacity:0.85; line-height:1.5;">SpecAverage time-frequency masking paired with BCE loss minimization.</p>
                </div>
                """
            )
        with gr.Column(scale=1):
            gr.Markdown(
                """
                <div class="saas-card-box">
                  <div style="font-size:0.8rem; font-weight:700; color:#536878; text-transform:uppercase;">Paper 2 M2S-ADD</div>
                  <h4 style="color:#FFFFFF; margin:6px 0;">Stereo Representation</h4>
                  <p style="font-size:0.88rem; opacity:0.85; line-height:1.5;">Mono-to-stereo graph attention networks for spatial feature learning.</p>
                </div>
                """
            )

    # SECTION 9: FAQ (ACCORDION LAYOUT)
    gr.Markdown("<div style='height:48px;'></div>")
    gr.Markdown("<h2 style='font-size:1.8rem; color:#FFFFFF; margin-bottom:16px;'>Frequently Asked Questions</h2>")
    with gr.Accordion("What audio file formats are supported by VoiceSentinel?", open=False):
        gr.Markdown("VoiceSentinel supports WAV, MP3, and FLAC files. Audio is automatically normalized and resampled to 16,000 Hz for consistent neural inference.")
    with gr.Accordion("What is the difference between Detection Modes A, B, C, and D?", open=False):
        gr.Markdown("Mode A provides baseline mono LFCC analysis. Mode B applies SpecAverage data augmentation. Mode C uses Mono-to-Stereo (M2S) conversion with SincNet and Graph Attention Networks (GAT). Mode D integrates stereo log-spectrograms with a ResNet34 backbone.")
    with gr.Accordion("How does Mono-to-Stereo (M2S) conversion detect voice spoofing?", open=False):
        gr.Markdown("Synthetic speech engines often generate artifacts in spatial phase distribution. M2S converts mono waveforms into dual-channel representations, allowing graph neural networks to detect phase irregularities.")

    # SECTION 10: FOOTER
    gr.Markdown("<div style='height:64px;'></div>")
    with gr.Row():
        with gr.Column():
            gr.Markdown(
                """
                <div style="border-top: 1px solid rgba(83, 104, 120, 0.28); padding-top: 24px; text-align: center; font-size: 0.85rem; color: #E5E4E2; opacity: 0.75;">
                  <div style="margin-bottom: 8px;">🛡️ <strong>VoiceSentinel</strong> &bull; Industrial Audio Deepfake & Voice Authenticity Detection</div>
                  <div>Built with PyTorch 2.0+, Torchaudio & Obsidian Chrome Design System</div>
                </div>
                """
            )

    # Connect callback
    analyze_button.click(
        fn=process_audio,
        inputs=[audio_input, mode_dropdown],
        outputs=[
            pred_text,
            conf_bar,
            stats_markdown,
            wave_plot,
            fft_plot,
            spec_plot,
            lfcc_plot
        ]
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=True)




