import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server/CLI safety

import matplotlib.pyplot as plt
import numpy as np
import torch
from typing import Optional


def _apply_dark_theme(fig: plt.Figure, ax: plt.Axes, title: str, xlabel: str, ylabel: str):
    """Helper function to apply consistent Obsidian Chrome styling to matplotlib figures."""
    fig.patch.set_facecolor("#0A0A0A")  # Onyx
    ax.set_facecolor("#12161A")        # Soft Metallic Slate
    
    ax.set_title(title, fontsize=11, fontweight="600", pad=10, color="#FFFFFF")   # White
    ax.set_xlabel(xlabel, fontsize=9, fontweight="500", color="#E5E4E2")        # Alabaster Grey
    ax.set_ylabel(ylabel, fontsize=9, fontweight="500", color="#E5E4E2")        # Alabaster Grey
    
    ax.tick_params(colors="#E5E4E2", labelsize=8)
    ax.grid(True, linestyle=":", alpha=0.2, color="#536878")  # Blue Slate
    
    for spine in ax.spines.values():
        spine.set_color("#536878")
        spine.set_linewidth(0.8)


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

    fig, ax = plt.subplots(figsize=(10, 3.2), dpi=120)
    ax.plot(time_axis, waveform_np, color="#E5E4E2", linewidth=1.1)  # Alabaster Grey
    ax.fill_between(time_axis, waveform_np, 0, color="#536878", alpha=0.25)  # Blue Slate
    
    _apply_dark_theme(fig, ax, title, "Time (seconds)", "Amplitude")
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

    fig, ax = plt.subplots(figsize=(10, 3.2), dpi=120)
    ax.plot(freqs, fft_magnitude, color="#FFFFFF", linewidth=1.1)  # White
    ax.fill_between(freqs, fft_magnitude, 0, color="#536878", alpha=0.2)  # Blue Slate
    
    _apply_dark_theme(fig, ax, title, "Frequency (Hz)", "Magnitude")
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

    fig, ax = plt.subplots(figsize=(10, 3.8), dpi=120)
    cmap_name = "magma"
    try:
        plt.get_cmap(cmap_name)
    except ValueError:
        cmap_name = "viridis"
    img = ax.imshow(spec_np, aspect="auto", origin="lower", cmap=cmap_name)
    
    _apply_dark_theme(fig, ax, title, "Time Frames", ylabel)
    
    # Add colorbar with Obsidian Chrome styling
    cbar = fig.colorbar(img, ax=ax)
    cbar.set_label("Magnitude (dB)", fontsize=9, fontweight="500", color="#E5E4E2")
    cbar.ax.tick_params(colors="#E5E4E2", labelsize=8)
    cbar.outline.set_edgecolor("#536878")
    
    fig.tight_layout()
    return fig



