"""Multi-signal overlay and before/after comparison visualizations."""

from __future__ import annotations

import numpy as np

from audio_samples import AudioSamples

from ._utils import (
    _apply_style,
    _compute_stft,
    _ensure_axes,
    _palette_colors,
    _resolve_style,
    _to_db,
)

from matplotlib.axes import Axes
from matplotlib.figure import Figure, SubFigure
from .style import MplStyle


def plot_waveform_overlay(
    signals: list[AudioSamples],
    labels: list[str],
    *,
    channel: int | None = None,
    alpha: float = 0.7,
    title: str = "Waveform Overlay",
    ax: Axes | None = None,
    style: MplStyle | None = None,
) -> tuple[Figure | SubFigure, Axes]:
    """Multiple waveforms on one axes using a seaborn palette.

    Args:
        signals: List of AudioSamples objects.
        labels: Label per signal.
        channel: Channel index (default: 0).
        alpha: Line transparency.
        title: Axes title.
        ax: Existing axes.
        style: Visual style.

    Returns:
        ``(fig, ax)``.
    """
    style = _resolve_style(style)
    fig, ax = _ensure_axes(ax, style)
    colors = _palette_colors(len(signals), style.palette)

    for sig, lbl, col in zip(signals, labels, colors):
        t = sig.time_axis()
        data = sig[channel]
        ax.plot(t, data, color=col, linewidth=style.linewidth, alpha=alpha, label=lbl)

    ax.set_xlabel("Time (s)", fontsize=style.label_size)
    ax.set_ylabel("Amplitude", fontsize=style.label_size)
    ax.set_title(title, fontsize=style.title_size)
    ax.legend(fontsize=style.legend_size)
    assert fig is not None  # for type checker
    _apply_style(fig, ax, style)
    return fig, ax


def plot_hpss_components(
    audio: AudioSamples,
    *,
    win_size: int = 2048,
    hop_size: int = 512,
    title: str = "HPSS Components",
    style: MplStyle | None = None,
) -> tuple[Figure, list[Axes]]:
    """Three-panel: original / harmonic / percussive waveforms.

    Args:
        audio: AudioSamples object.
        win_size: HPSS window size.
        hop_size: HPSS hop size.
        title: Figure suptitle.
        style: Visual style.

    Returns:
        ``(fig, [orig_ax, harm_ax, perc_ax])``.
    """
    import matplotlib.pyplot as plt
    import spectrograms as sp

    style = _resolve_style(style)
    stft_p = sp.StftParams(
        window=sp.WindowType.hanning, hop_size=hop_size, n_fft=win_size, centre=True
    )
    harmonic, percussive = audio.hpss(stft_p)
    signals = [audio, harmonic, percussive]
    subtitles = ["Original", "Harmonic", "Percussive"]
    colors = [style.waveform_color, "#27ae60", "#e74c3c"]

    fig, axes = plt.subplots(
        3,
        1,
        sharex=True,
        figsize=(style.fig_width, style.fig_height * 3),
        dpi=style.dpi,
    )

    for ax, sig, stitle, col in zip(axes, signals, subtitles, colors):
        t = sig.time_axis()
        ax.plot(t, sig, color=col, linewidth=style.linewidth, alpha=style.alpha)
        ax.set_ylabel("Amplitude", fontsize=style.label_size)
        ax.set_title(stitle, fontsize=style.label_size)

    axes[-1].set_xlabel("Time (s)", fontsize=style.label_size)
    fig.suptitle(title, fontsize=style.title_size)
    _apply_style(fig, list(axes), style)
    return fig, list(axes)


def plot_snr_comparison(
    signal: AudioSamples,
    noise: AudioSamples,
    *,
    title: str = "SNR Comparison",
    style: MplStyle | None = None,
) -> tuple[Figure, list[Axes]]:
    """Side-by-side waveforms with SNR / MSE annotations.

    Args:
        signal: Clean AudioSamples.
        noise: Noisy / reference AudioSamples.
        title: Figure suptitle.
        style: Visual style.

    Returns:
        ``(fig, [signal_ax, noise_ax])``.
    """
    import matplotlib.pyplot as plt

    style = _resolve_style(style)
    colors = _palette_colors(2, style.palette)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(style.fig_width, style.fig_height),
        dpi=style.dpi,
    )

    for ax, sig, lbl, col in zip(axes, [signal, noise], ["Signal", "Noise"], colors):
        t = sig.time_axis()
        ax.plot(t, sig, color=col, linewidth=style.linewidth, alpha=style.alpha)
        ax.set_xlabel("Time (s)", fontsize=style.label_size)
        ax.set_ylabel("Amplitude", fontsize=style.label_size)
        ax.set_title(lbl, fontsize=style.label_size)

    # Compute SNR / MSE
    s, n = signal.to_mono(), noise.to_mono()
    min_len = min(len(s), len(n))
    s, n = s[:min_len], n[:min_len]
    mse = float(np.mean((s - n) ** 2))
    sig_power = float(np.mean(s**2))
    if mse > 0 and sig_power > 0:
        snr = 10.0 * np.log10(sig_power / mse)
        fig.text(
            0.5,
            0.01,
            f"SNR: {snr:.2f} dB   MSE: {mse:.2e}",
            ha="center",
            fontsize=style.label_size,
        )

    fig.suptitle(title, fontsize=style.title_size)
    _apply_style(fig, list(axes), style)
    return fig, list(axes)


def plot_signal_vs_filtered(
    original: AudioSamples,
    filtered: AudioSamples,
    *,
    n_fft: int = 2048,
    label_original: str = "Original",
    label_filtered: str = "Filtered",
    title: str = "Signal vs Filtered",
    style: MplStyle | None = None,
) -> tuple[Figure, list[Axes]]:
    """2-column: waveform comparison left, spectrum comparison right.

    Args:
        original: Original AudioSamples.
        filtered: Filtered AudioSamples.
        n_fft: FFT size for spectrum comparison.
        label_original: Label for original signal.
        label_filtered: Label for filtered signal.
        title: Figure suptitle.
        style: Visual style.

    Returns:
        ``(fig, [waveform_ax, spectrum_ax])``.
    """
    import matplotlib.pyplot as plt

    style = _resolve_style(style)
    colors = _palette_colors(2, style.palette)
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(style.fig_width * 1.5, style.fig_height),
        dpi=style.dpi,
    )

    # Left: waveform comparison
    ax_w = axes[0]
    for sig, lbl, col in zip(
        [original, filtered], [label_original, label_filtered], colors
    ):
        t = sig.time_axis()
        ax_w.plot(
            t, sig, color=col, linewidth=style.linewidth, alpha=style.alpha, label=lbl
        )
    ax_w.set_xlabel("Time (s)", fontsize=style.label_size)
    ax_w.set_ylabel("Amplitude", fontsize=style.label_size)
    ax_w.set_title("Waveform", fontsize=style.label_size)
    ax_w.legend(fontsize=style.legend_size)

    # Right: spectrum comparison
    ax_s = axes[1]
    for sig, lbl, col in zip(
        [original, filtered], [label_original, label_filtered], colors
    ):
        src = sig.to_mono()
        stft_mag, freqs, _ = _compute_stft(src, n_fft=n_fft, hop_size=n_fft // 4)
        mag = np.mean(stft_mag, axis=1)
        db = _to_db(mag, style.db_vmin)
        ax_s.plot(
            freqs,
            db,
            color=col,
            linewidth=style.linewidth,
            alpha=style.alpha,
            label=lbl,
        )
    ax_s.set_xlabel("Frequency (Hz)", fontsize=style.label_size)
    ax_s.set_ylabel("Magnitude (dB)", fontsize=style.label_size)
    ax_s.set_title("Spectrum", fontsize=style.label_size)
    ax_s.legend(fontsize=style.legend_size)

    fig.suptitle(title, fontsize=style.title_size)
    _apply_style(fig, list(axes), style)
    return fig, list(axes)
