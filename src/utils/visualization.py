import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server/CLI safety

import matplotlib.pyplot as plt
import numpy as np
import torch
from typing import Optional


def plot_waveform(
    waveform: torch.Tensor, 
    sample_rate: int, 
    title: str = "Waveform"
) -> plt.Figure:
    """Creates a matplotlib figure plotting the waveform in the time domain.

    Args:
        waveform: Waveform tensor of shape (1, num_samples) or (num_samples,).
        sample_rate: Sample rate of the audio.
        title: Title of the plot.

    Returns:
        plt.Figure: The matplotlib figure object.
    """
    waveform_np = waveform.detach().cpu().squeeze().numpy()
    num_samples = len(waveform_np)
    time_axis = np.arange(num_samples) / sample_rate

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(time_axis, waveform_np, color="#1F6E7A", alpha=0.8)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Time (seconds)", fontsize=10)
    ax.set_ylabel("Amplitude", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_ylim(-1.05, 1.05)
    fig.tight_layout()
    return fig


def plot_fft(
    waveform: torch.Tensor, 
    sample_rate: int, 
    title: str = "FFT Spectrum"
) -> plt.Figure:
    """Computes the Fast Fourier Transform (FFT) and plots the magnitude spectrum.

    Args:
        waveform: Waveform tensor of shape (1, num_samples) or (num_samples,).
        sample_rate: Sample rate of the audio.
        title: Title of the plot.

    Returns:
        plt.Figure: The matplotlib figure object.
    """
    waveform_np = waveform.detach().cpu().squeeze().numpy()
    num_samples = len(waveform_np)
    
    # Compute rfft (real FFT)
    fft_vals = np.fft.rfft(waveform_np)
    fft_magnitude = np.abs(fft_vals)
    freqs = np.fft.rfftfreq(num_samples, d=1 / sample_rate)

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(freqs, fft_magnitude, color="#B07A2C", alpha=0.8)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Frequency (Hz)", fontsize=10)
    ax.set_ylabel("Magnitude", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    return fig


def plot_spectrogram(
    spectrogram: torch.Tensor, 
    title: str = "Spectrogram", 
    ylabel: str = "Frequency Bin"
) -> plt.Figure:
    """Generates a heat map plot of a log-spectrogram or LFCC representation.

    Args:
        spectrogram: Spectrogram tensor of shape (freq_bins, time_steps) 
                     or (1, freq_bins, time_steps).
        title: Title of the plot.
        ylabel: Label for the y-axis (e.g. "LFCC Coefficients").

    Returns:
        plt.Figure: The matplotlib figure object.
    """
    spec_np = spectrogram.detach().cpu().squeeze().numpy()

    fig, ax = plt.subplots(figsize=(10, 4))
    img = ax.imshow(spec_np, aspect="auto", origin="lower", cmap="magma")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Time Frames", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    
    # Add colorbar
    cbar = fig.colorbar(img, ax=ax)
    cbar.set_label("Magnitude (dB)", fontsize=10)
    
    fig.tight_layout()
    return fig
