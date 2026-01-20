"""Frequency-domain visualizations: magnitude spectrum, PSD, comparisons."""

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


def plot_magnitude_spectrum(
    audio: AudioSamples,
    *,
    n_fft: int = 2048,
    channel: int | None = None,
    db_scale: bool = True,
    freq_max: float | None = None,
    log_freq: bool = False,
    show_centroid: bool = True,
    title: str = "Magnitude Spectrum",
    ax: Axes | None = None,
    style: MplStyle | None = None,
) -> tuple[Figure, Axes]:
    """Plot magnitude spectrum, optionally in dB.

    Args:
        audio: AudioSamples object.
        n_fft: FFT size.
        channel: Channel index (default: 0 for multi-channel).
        db_scale: Convert to dB scale.
        freq_max: Upper frequency limit in Hz.
        log_freq: Use logarithmic frequency axis.
        show_centroid: Mark the spectral centroid.
        title: Axes title.
        ax: Existing axes.
        style: Visual style.

    Returns:
        ``(fig, ax)``.
    """
    style = _resolve_style(style)
    fig, ax = _ensure_axes(ax, style)

    src = (
        audio.extract_channel(channel if channel is not None else 0)
        if not audio.is_mono()
        else audio
    )
    stft_mag, freqs, _ = _compute_stft(src, n_fft=n_fft, hop_size=n_fft // 4)
    mag = np.mean(stft_mag, axis=1)
    freqs = np.array(freqs)
    if freq_max is not None:
        mask = freqs <= freq_max
        freqs = freqs[mask]
        mag = mag[mask]

    y = _to_db(mag, style.db_vmin) if db_scale else mag
    ylabel = "Magnitude (dB)" if db_scale else "Magnitude"

    if log_freq:
        ax.semilogx(
            freqs,
            y,
            color=style.waveform_color,
            linewidth=style.linewidth,
            alpha=style.alpha,
        )
    else:
        ax.plot(
            freqs,
            y,
            color=style.waveform_color,
            linewidth=style.linewidth,
            alpha=style.alpha,
        )

    if show_centroid:
        try:
            centroid = float(audio.spectral_centroid())
            ax.axvline(
                centroid,
                color="red",
                linestyle="--",
                linewidth=1.0,
                alpha=0.7,
                label=f"Centroid {centroid:.0f} Hz",
            )
            ax.legend(fontsize=style.legend_size)
        except Exception:
            pass

    ax.set_xlabel("Frequency (Hz)", fontsize=style.label_size)
    ax.set_ylabel(ylabel, fontsize=style.label_size)
    ax.set_title(title, fontsize=style.title_size)
    assert fig is not None  # for type checker
    assert not isinstance(fig, SubFigure)
    _apply_style(fig, ax, style)
    return fig, ax


def plot_power_spectral_density(
    audio: AudioSamples,
    *,
    window_size: int = 2048,
    overlap: float = 0.5,
    channel: int | None = None,
    freq_max: float | None = None,
    title: str = "Power Spectral Density",
    ax: Axes | None = None,
    style: MplStyle | None = None,
) -> tuple[Figure, Axes]:
    """Welch PSD estimate in dB/Hz.

    Args:
        audio: AudioSamples object.
        window_size: Window length in samples.
        overlap: Fractional overlap (0--1).
        channel: Channel index.
        freq_max: Upper frequency limit in Hz.
        title: Axes title.
        ax: Existing axes.
        style: Visual style.

    Returns:
        ``(fig, ax)``.
    """
    style = _resolve_style(style)
    fig, ax = _ensure_axes(ax, style)

    psd = audio.power_spectral_density(window_size, overlap)
    sr = audio.sample_rate
    n_freqs = psd.shape[-1]
    freqs = np.arange(n_freqs) * (sr / window_size)

    if psd.ndim > 1:
        ch = channel if channel is not None else 0
        psd = psd[ch]

    if freq_max is not None:
        mask = freqs <= freq_max
        freqs = freqs[mask]
        psd = psd[mask]

    with np.errstate(divide="ignore", invalid="ignore"):
        psd_db = 10.0 * np.log10(np.maximum(psd, 1e-12))

    ax.plot(
        freqs,
        psd_db,
        color=style.waveform_color,
        linewidth=style.linewidth,
        alpha=style.alpha,
    )
    ax.set_xlabel("Frequency (Hz)", fontsize=style.label_size)
    ax.set_ylabel("Power (dB/Hz)", fontsize=style.label_size)
    ax.set_title(title, fontsize=style.title_size)

    assert fig is not None  # for type checker
    assert not isinstance(fig, SubFigure)
    _apply_style(fig, ax, style)
    return fig, ax


def plot_spectrum_comparison(
    signals: list[AudioSamples],
    labels: list[str],
    *,
    n_fft: int = 2048,
    db_scale: bool = True,
    freq_max: float | None = None,
    fill_under: bool = True,
    title: str = "Spectrum Comparison",
    ax: Axes | None = None,
    style: MplStyle | None = None,
) -> tuple[Figure, Axes]:
    """Overlay multiple magnitude spectra with seaborn palette.

    Args:
        signals: List of AudioSamples objects.
        labels: Label for each signal.
        n_fft: FFT size.
        db_scale: Convert to dB.
        freq_max: Upper frequency limit in Hz.
        fill_under: Fill area under each curve.
        title: Axes title.
        ax: Existing axes.
        style: Visual style.

    Returns:
        ``(fig, ax)``.
    """
    style = _resolve_style(style)
    fig, ax = _ensure_axes(ax, style)
    colors = _palette_colors(len(signals), style.palette)
    ylabel = "Magnitude (dB)" if db_scale else "Magnitude"

    for sig, lbl, col in zip(signals, labels, colors):
        src = sig.to_mono() if not sig.is_mono() else sig
        stft_mag, freqs, _ = _compute_stft(src, n_fft=n_fft, hop_size=n_fft // 4)
        mag = np.mean(stft_mag, axis=1)
        freqs = np.array(freqs)
        if freq_max is not None:
            mask = freqs <= freq_max
            freqs = freqs[mask]
            mag = mag[mask]

        y = _to_db(mag, style.db_vmin) if db_scale else mag
        ax.plot(
            freqs, y, color=col, linewidth=style.linewidth, alpha=style.alpha, label=lbl
        )
        if fill_under:
            ax.fill_between(freqs, y, alpha=0.15, color=col)

    ax.set_xlabel("Frequency (Hz)", fontsize=style.label_size)
    ax.set_ylabel(ylabel, fontsize=style.label_size)
    ax.set_title(title, fontsize=style.title_size)
    ax.legend(fontsize=style.legend_size)

    assert fig is not None  # for type checker
    assert not isinstance(fig, SubFigure)
    _apply_style(fig, ax, style)
    return fig, ax
