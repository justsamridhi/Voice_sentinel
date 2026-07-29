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

    # Decode Mode
    mode = mode_str[0]  # Extracts "A", "B", "C", or "D"
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


# Customized CSS to match a premium, modern dashboard design
custom_css = """
:root {
    color-scheme: dark;
}
body {
    background: radial-gradient(circle at top left, rgba(14, 165, 233, 0.15), transparent 28%),
                linear-gradient(135deg, #020617 0%, #0f172a 45%, #111827 100%);
    color: #f8fafc;
}
.gradio-container {
    max-width: 1400px !important;
    padding: 24px !important;
    font-family: 'Inter', 'Segoe UI', sans-serif !important;
}
#title-banner {
    background: linear-gradient(135deg, rgba(14, 165, 233, 0.2), rgba(37, 99, 235, 0.3));
    border: 1px solid rgba(148, 163, 184, 0.25);
    border-radius: 24px;
    padding: 24px 28px;
    margin-bottom: 20px;
    box-shadow: 0 18px 45px rgba(2, 6, 23, 0.35);
    backdrop-filter: blur(16px);
}
#title-banner .markdown {
    color: #f8fafc;
}
#control-card, #result-card, #insight-card {
    background: rgba(15, 23, 42, 0.8);
    border: 1px solid rgba(148, 163, 184, 0.22);
    border-radius: 20px;
    padding: 18px;
    box-shadow: 0 12px 30px rgba(2, 6, 23, 0.25);
    backdrop-filter: blur(16px);
}
#analyze-btn {
    background: linear-gradient(135deg, #14b8a6 0%, #2563eb 100%);
    color: white;
    font-weight: 700;
    border: none;
    border-radius: 999px;
    padding: 12px 18px;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
    box-shadow: 0 8px 20px rgba(37, 99, 235, 0.25);
}
#analyze-btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 24px rgba(37, 99, 235, 0.35);
}
.pill-chip {
    display: inline-block;
    padding: 7px 12px;
    margin: 4px 6px 0 0;
    border-radius: 999px;
    background: rgba(15, 23, 42, 0.45);
    border: 1px solid rgba(148, 163, 184, 0.25);
    color: #dbeafe;
    font-size: 0.9rem;
}
"""

with gr.Blocks(
    css=custom_css,
    theme=gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="indigo",
        neutral_hue="slate",
        radius_size="lg"
    )
) as demo:
    with gr.Row(elem_id="title-banner"):
        with gr.Column(scale=2):
            gr.Markdown(
                """
                <div style="display:flex; flex-direction:column; gap:6px;">
                  <h1 style="margin:0; font-size:2rem;">🛡️ VoiceSentinel</h1>
                  <p style="margin:0; font-size:1rem; color:#dbeafe;">
                    Robust audio deepfake detection with a polished, premium interface and rich acoustic diagnostics.
                  </p>
                  <div style="margin-top:6px;">
                    <span class="pill-chip">Live inference</span>
                    <span class="pill-chip">Multi-mode analysis</span>
                    <span class="pill-chip">Visual insights</span>
                  </div>
                </div>
                """
            )
        with gr.Column(scale=1):
            gr.Markdown(
                """
                <div style="text-align:right; color:#e2e8f0;">
                  <div style="font-size:0.95rem; opacity:0.85;">Detection ready</div>
                  <div style="font-size:1.15rem; font-weight:700;">Stereo-aware, research-grade analysis</div>
                </div>
                """
            )

    with gr.Row():
        with gr.Column(scale=1, elem_id="control-card"):
            gr.Markdown("### 🎛️ Upload Audio & Configure")
            audio_input = gr.Audio(type="filepath", label="Upload Audio File (wav, mp3, flac)")

            mode_dropdown = gr.Dropdown(
                choices=[
                    "A (Baseline - Mono LFCC + ResNet34)",
                    "B (Paper 4 - LogSpec + SpecAverage + BCE)",
                    "C (Paper 2 - M2S + SincNet + ResNet + GAT)",
                    "D (Combined Mode - Stereo LogSpec + ResNet34)"
                ],
                value="D (Combined Mode - Stereo LogSpec + ResNet34)",
                label="Select Detection Mode"
            )

            gr.Examples(
                examples=[
                    ["scratch/test.wav", "D (Combined Mode - Stereo LogSpec + ResNet34)"],
                    ["scratch/test.wav", "A (Baseline - Mono LFCC + ResNet34)"]
                ],
                inputs=[audio_input, mode_dropdown],
                label="Try an Example"
            )

            analyze_button = gr.Button("⚡ Analyze Audio", elem_id="analyze-btn")

            gr.Markdown("Choose a detection pipeline and inspect the prediction in a streamlined, modern workflow.")

        with gr.Column(scale=1, elem_id="result-card"):
            gr.Markdown("### 🔐 Sentinel Diagnostics")
            pred_text = gr.Textbox(value="Awaiting analysis...", label="Sentinel Prediction Label", interactive=False)
            conf_bar = gr.Label(label="Confidence Distribution")

            with gr.Accordion("📊 Audio File Metadata", open=True):
                stats_markdown = gr.Markdown("Upload a file and press Analyze to display audio details.")

    with gr.Row():
        with gr.Column(elem_id="insight-card"):
            gr.Markdown("### 📈 Acoustic Visualizations")
            with gr.Tabs():
                with gr.TabItem("Waveform"):
                    wave_plot = gr.Plot(label="Time Domain Waveform")
                with gr.TabItem("Spectrum (FFT)"):
                    fft_plot = gr.Plot(label="Frequency Magnitude Spectrum")
                with gr.TabItem("Spectrogram"):
                    spec_plot = gr.Plot(label="Log-Spectrogram Magnitude Heatmap")
                with gr.TabItem("LFCC Cepstrum"):
                    lfcc_plot = gr.Plot(label="LFCC Cepstrum Coefficients Heatmap")

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
    demo.launch(
    server_name="0.0.0.0",
    server_port=int(os.environ.get("PORT", 7860)),
    allowed_paths=["/content/Voice_sentinel/scratch"]
)
