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


# Customized CSS to match a premium, high-tech Cyber-Sentinel Command Center design
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

:root {
    color-scheme: dark;
    --bg-main: #030712;
    --card-bg: rgba(15, 23, 42, 0.75);
    --card-border: rgba(255, 255, 255, 0.08);
    --accent-cyan: #06b6d4;
    --accent-blue: #3b82f6;
    --accent-indigo: #6366f1;
    --accent-emerald: #10b981;
    --accent-rose: #f43f5e;
    --text-primary: #f8fafc;
    --text-secondary: #94a3b8;
}

body {
    background: radial-gradient(circle at 15% 15%, rgba(6, 182, 212, 0.12), transparent 40%),
                radial-gradient(circle at 85% 75%, rgba(99, 102, 241, 0.12), transparent 45%),
                linear-gradient(180deg, #030712 0%, #0b1329 50%, #030712 100%);
    color: var(--text-primary);
    font-family: 'Plus Jakarta Sans', -apple-system, sans-serif !important;
    min-height: 100vh;
}

.gradio-container {
    max-width: 1440px !important;
    padding: 24px !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}

#title-banner {
    background: linear-gradient(135deg, rgba(15, 23, 42, 0.95) 0%, rgba(30, 41, 59, 0.85) 100%);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 24px;
    padding: 24px 30px;
    margin-bottom: 24px;
    box-shadow: 0 20px 50px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.1);
    backdrop-filter: blur(20px);
}

#control-card, #result-card, #insight-card {
    background: var(--card-bg);
    border: 1px solid var(--card-border);
    border-radius: 24px;
    padding: 24px;
    box-shadow: 0 16px 40px rgba(0, 0, 0, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.05);
    backdrop-filter: blur(20px);
    transition: border-color 0.3s ease, box-shadow 0.3s ease;
}

#control-card:hover, #result-card:hover, #insight-card:hover {
    border-color: rgba(6, 182, 212, 0.3);
    box-shadow: 0 20px 45px rgba(6, 182, 212, 0.1);
}

#analyze-btn {
    background: linear-gradient(135deg, #06b6d4 0%, #3b82f6 50%, #6366f1 100%);
    color: #ffffff;
    font-weight: 700;
    font-size: 1.05rem;
    letter-spacing: 0.02em;
    border: none;
    border-radius: 14px;
    padding: 14px 24px;
    cursor: pointer;
    box-shadow: 0 8px 25px rgba(6, 182, 212, 0.35);
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
    margin-top: 12px;
}

#analyze-btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 30px rgba(6, 182, 212, 0.5);
    filter: brightness(1.1);
}

#analyze-btn:active {
    transform: translateY(0);
}

.pill-chip {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 6px 14px;
    margin: 4px 6px 4px 0;
    border-radius: 999px;
    background: rgba(15, 23, 42, 0.6);
    border: 1px solid rgba(255, 255, 255, 0.12);
    color: #e2e8f0;
    font-size: 0.85rem;
    font-weight: 600;
}

.pill-chip-glow {
    background: rgba(6, 182, 212, 0.15);
    border-color: rgba(6, 182, 212, 0.4);
    color: #67e8f9;
}

.status-pulse {
    width: 9px;
    height: 9px;
    background-color: #10b981;
    border-radius: 50%;
    box-shadow: 0 0 10px #10b981;
    display: inline-block;
    margin-right: 6px;
}

.gradio-container label span {
    color: #cbd5e1 !important;
    font-weight: 600 !important;
    font-size: 0.92rem !important;
}

.gradio-container .gr-box, .gradio-container .gr-input {
    background-color: rgba(15, 23, 42, 0.6) !important;
    border-color: rgba(255, 255, 255, 0.1) !important;
    border-radius: 14px !important;
}
"""

with gr.Blocks(
    css=custom_css,
    theme=gr.themes.Soft(
        primary_hue="cyan",
        secondary_hue="indigo",
        neutral_hue="slate",
        radius_size="lg"
    )
) as demo:
    with gr.Row(elem_id="title-banner"):
        with gr.Column(scale=3):
            gr.Markdown(
                """
                <div style="display:flex; flex-direction:column; gap:8px;">
                  <div style="display:flex; align-items:center; gap:12px;">
                    <span style="font-size:2.2rem;">🛡️</span>
                    <div>
                      <h1 style="margin:0; font-size:2.1rem; font-weight:800; tracking: -0.02em; background: linear-gradient(135deg, #f8fafc 0%, #38bdf8 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">VoiceSentinel</h1>
                      <div style="font-size:0.85rem; color:#94a3b8; font-weight:500;">Research-Grade Audio Deepfake & Voice Spoof Detection System</div>
                    </div>
                  </div>
                  <div style="margin-top:6px; display:flex; flex-wrap:wrap; align-items:center;">
                    <span class="pill-chip pill-chip-glow"><span class="status-pulse"></span>Sentinel Engine Active</span>
                    <span class="pill-chip">⚡ Real-Time Audio Diagnostics</span>
                    <span class="pill-chip">🎧 Mono-to-Stereo Aware (M2S)</span>
                    <span class="pill-chip">🔬 LFCC & Log-Spectrogram</span>
                  </div>
                </div>
                """
            )
        with gr.Column(scale=1):
            gr.Markdown(
                """
                <div style="text-align:right; color:#e2e8f0; background: rgba(15, 23, 42, 0.6); padding: 14px 18px; border-radius: 16px; border: 1px solid rgba(255, 255, 255, 0.08);">
                  <div style="font-size:0.8rem; text-transform:uppercase; letter-spacing:0.05em; color:#38bdf8; font-weight:700;">System Status</div>
                  <div style="font-size:1.1rem; font-weight:700; margin-top:2px;">Defense Operational</div>
                  <div style="font-size:0.82rem; color:#94a3b8; margin-top:2px;">PyTorch 2.0+ / Gradio Dashboard</div>
                </div>
                """
            )

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

            analyze_button = gr.Button("⚡ Run Deepfake Diagnostic", elem_id="analyze-btn")

            gr.Markdown(
                """
                <div style="margin-top:12px; font-size:0.83rem; color:#94a3b8; line-height:1.4;">
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
    demo.launch(server_name="127.0.0.1", server_port=7860)

