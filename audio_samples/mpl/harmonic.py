"""Harmonic series, autocorrelation and cross-correlation visualizations."""

from __future__ import annotations

import numpy as np

from audio_samples import AudioSamples

from ._utils import (
    _apply_style,
    _compute_stft,
    _ensure_axes,
    _resolve_style,
    _to_db,
)
from matplotlib.axes import Axes
from matplotlib.figure import Figure, SubFigure
from .style import MplStyle


def plot_harmonic_analysis(
    audio: AudioSamples,
    *,
    fundamental_freq: float,
    num_harmonics: int = 8,
    tolerance: float = 0.05,
    n_fft: int = 4096,
    channel: int | None = None,
    title: str = "Harmonic Analysis",
    ax: Axes | None = None,
    style: MplStyle | None = None,
) -> tuple[Figure, Axes]:
    """Magnitude spectrum with harmonic peaks highlighted and labelled H1…Hn.

    Args:
        audio: AudioSamples object.
        fundamental_freq: Fundamental frequency in Hz.
        num_harmonics: Number of harmonics to annotate.
        tolerance: Frequency tolerance as a fraction.
        n_fft: FFT size.
        channel: Channel index.
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
    db = _to_db(mag, style.db_vmin)

    ax.plot(
        freqs,
        db,
        color=style.waveform_color,
        linewidth=style.linewidth,
        alpha=style.alpha,
    )

    # Compute harmonic magnitudes from the averaged spectrum
    harmonic_mags = []
    freq_resolution = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0
    for h_idx in range(1, num_harmonics + 1):
        h_freq = fundamental_freq * h_idx
        bin_idx = int(round(h_freq / freq_resolution))
        half_width = max(1, int(tolerance * h_freq / freq_resolution))
        lo = max(0, bin_idx - half_width)
        hi = min(len(mag) - 1, bin_idx + half_width)
        harmonic_mags.append(float(np.max(mag[lo : hi + 1])) if lo <= hi else 0.0)

    for h_idx, h_mag in enumerate(harmonic_mags, start=1):
        h_freq = fundamental_freq * h_idx
        h_db = _to_db(np.array([h_mag]), style.db_vmin)[0]
        ax.axvline(h_freq, color="red", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.annotate(
            f"H{h_idx}",
            xy=(h_freq, h_db),
            xytext=(h_freq, h_db + 3),
            fontsize=style.tick_size,
            color="red",
            ha="center",
        )

    ax.set_xlabel("Frequency (Hz)", fontsize=style.label_size)
    ax.set_ylabel("Magnitude (dB)", fontsize=style.label_size)
    ax.set_title(title, fontsize=style.title_size)
    ax.set_xlim(0, fundamental_freq * (num_harmonics + 1) * 1.2)
    assert fig is not None  # for type checker
    assert not isinstance(fig, SubFigure)
    _apply_style(fig, ax, style)
    return fig, ax


def plot_harmonics_bar(
    audio: AudioSamples,
    *,
    fundamental_freq: float,
    num_harmonics: int = 8,
    tolerance: float = 0.05,
    db_scale: bool = True,
    n_fft: int = 4096,
    title: str = "Harmonic Magnitudes",
    ax: Axes | None = None,
    style: MplStyle | None = None,
) -> tuple[Figure, Axes]:
    """Bar chart of harmonic magnitudes by index.

    Args:
        audio: AudioSamples object.
        fundamental_freq: Fundamental frequency in Hz.
        num_harmonics: Number of harmonics to display.
        tolerance: Frequency tolerance as a fraction.
        db_scale: Show magnitudes in dB.
        n_fft: FFT size.
        title: Axes title.
        ax: Existing axes.
        style: Visual style.

    Returns:
        ``(fig, ax)``.
    """
    style = _resolve_style(style)
    fig, ax = _ensure_axes(ax, style)

    src = audio.to_mono()
    stft_mag, freqs, _ = _compute_stft(src, n_fft=n_fft, hop_size=n_fft // 4)
    mag = np.mean(stft_mag, axis=1)
    freq_resolution = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0
    raw_mags = []
    for h_idx in range(1, num_harmonics + 1):
        h_freq = fundamental_freq * h_idx
        bin_idx = int(round(h_freq / freq_resolution))
        half_width = max(1, int(tolerance * h_freq / freq_resolution))
        lo = max(0, bin_idx - half_width)
        hi = min(len(mag) - 1, bin_idx + half_width)
        raw_mags.append(float(np.max(mag[lo : hi + 1])) if lo <= hi else 0.0)
    harmonic_mags = np.asarray(raw_mags, dtype=np.float64)

    h_labels = [f"H{i + 1}" for i in range(len(harmonic_mags))]
    y = _to_db(harmonic_mags, style.db_vmin) if db_scale else harmonic_mags
    ylabel = "Magnitude (dB)" if db_scale else "Magnitude"

    ax.bar(h_labels, y, color=style.waveform_color, alpha=style.alpha)
    ax.set_xlabel("Harmonic", fontsize=style.label_size)
    ax.set_ylabel(ylabel, fontsize=style.label_size)
    ax.set_title(title, fontsize=style.title_size)
    assert fig is not None  # for type checker
    assert not isinstance(fig, SubFigure)
    _apply_style(fig, ax, style)
    return fig, ax


def plot_autocorrelation(
    audio: AudioSamples,
    *,
    max_lag: int = 4096,
    title: str = "Autocorrelation",
    ax: Axes | None = None,
    style: MplStyle | None = None,
) -> tuple[Figure, Axes]:
    """Autocorrelation vs lag in seconds; annotates dominant peak.

    Args:
        audio: AudioSamples object.
        max_lag: Maximum lag in samples.
        title: Axes title.
        ax: Existing axes.
        style: Visual style.

    Returns:
        ``(fig, ax)``.
    """
    style = _resolve_style(style)
    fig, ax = _ensure_axes(ax, style)

    sr = audio.sample_rate
    acf = audio.autocorrelation(max_lag)
    if acf is None:
        ax.set_title(f"{title} (unavailable)", fontsize=style.title_size)
        assert fig is not None  # for type checker
        assert not isinstance(fig, SubFigure)
        _apply_style(fig, ax, style)
        return fig, ax

    acf_arr = np.asarray(acf, dtype=np.float64)
    lags_s = np.arange(len(acf_arr)) / sr

    ax.plot(
        lags_s,
        acf_arr,
        color=style.waveform_color,
        linewidth=style.linewidth,
        alpha=style.alpha,
    )

    # Annotate dominant peak (skip lag=0)
    if len(acf_arr) > 1:
        peak_idx = int(np.argmax(np.abs(acf_arr[1:]))) + 1
        peak_lag = lags_s[peak_idx]
        ax.axvline(
            peak_lag,
            color="red",
            linestyle="--",
            linewidth=1.0,
            alpha=0.7,
            label=f"Peak @ {peak_lag * 1000:.1f} ms",
        )
        ax.legend(fontsize=style.legend_size)

    ax.set_xlabel("Lag (s)", fontsize=style.label_size)
    ax.set_ylabel("Autocorrelation", fontsize=style.label_size)
    ax.set_title(title, fontsize=style.title_size)
    assert fig is not None  # for type checker
    assert not isinstance(fig, SubFigure)
    return fig, ax


def plot_cross_correlation(
    audio: AudioSamples,
    other: AudioSamples,
    *,
    max_lag: int = 4096,
    title: str = "Cross-Correlation",
    ax: Axes | None = None,
    style: MplStyle | None = None,
) -> tuple[Figure, Axes]:
    """Normalised cross-correlation; annotates lag at peak.

    Args:
        audio: Reference AudioSamples object.
        other: Second AudioSamples object.
        max_lag: Maximum lag in samples.
        title: Axes title.
        ax: Existing axes.
        style: Visual style.

    Returns:
        ``(fig, ax)``.
    """
    style = _resolve_style(style)
    fig, ax = _ensure_axes(ax, style)

    sr = audio.sample_rate
    xcorr = np.asarray(audio.cross_correlation(other, max_lag), dtype=np.float64)
    lags = np.arange(-max_lag, max_lag + 1)[: len(xcorr)]
    lags_s = lags / sr

    ax.plot(
        lags_s,
        xcorr,
        color=style.waveform_color,
        linewidth=style.linewidth,
        alpha=style.alpha,
    )

    peak_idx = int(np.argmax(np.abs(xcorr)))
    peak_lag = lags_s[peak_idx]
    ax.axvline(
        peak_lag,
        color="red",
        linestyle="--",
        linewidth=1.0,
        alpha=0.7,
        label=f"Peak @ {peak_lag * 1000:.1f} ms",
    )
    ax.axvline(0, color="gray", linewidth=0.5, linestyle=":")
    ax.legend(fontsize=style.legend_size)

    ax.set_xlabel("Lag (s)", fontsize=style.label_size)
    ax.set_ylabel("Cross-correlation", fontsize=style.label_size)
    ax.set_title(title, fontsize=style.title_size)
    assert fig is not None  # for type checker
    assert not isinstance(fig, SubFigure)
    _apply_style(fig, ax, style)
    return fig, ax
