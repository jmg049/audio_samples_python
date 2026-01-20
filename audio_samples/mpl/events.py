"""Onset, beat, and pitch tracking visualizations."""

from __future__ import annotations

import numpy as np
from spectrograms import CqtParams

from audio_samples import (
    AudioSamples,
    BeatTrackingData,
    OnsetDetectionConfig,
    SpectralFluxMethod,
    PitchDetectionMethod,
)

from ._utils import (
    _apply_style,
    _beat_vlines,
    _ensure_axes,
    _resolve_style,
)

from matplotlib.axes import Axes
from matplotlib.figure import Figure, SubFigure
from .style import MplStyle


def plot_onsets(
    audio: AudioSamples,
    *,
    config: OnsetDetectionConfig | None = None,
    channel: int | None = None,
    show_odf: bool = False,
    title: str = "Onset Detection",
    style: MplStyle | None = None,
) -> tuple[Figure, list[Axes]]:
    """Waveform with onset vlines; optional ODF panel below.

    Args:
        audio: AudioSamples object.
        config: OnsetDetectionConfig. Uses default when None.
        channel: Channel index.
        show_odf: Add a second panel showing the onset detection function.
        title: Figure suptitle.
        style: Visual style.

    Returns:
        ``(fig, axes)`` — 1 or 2 axes.
    """
    import matplotlib.pyplot as plt
    from audio_samples import OnsetDetectionConfig

    style = _resolve_style(style)

    if config is None:
        config = OnsetDetectionConfig.default()

    onsets = audio.detect_onsets(config)
    t = audio.time_axis()
    data = audio[channel]

    n_rows = 2 if show_odf else 1
    fig, axes = plt.subplots(
        n_rows,
        1,
        sharex=True,
        figsize=(style.fig_width, style.fig_height * n_rows),
        dpi=style.dpi,
    )
    if n_rows == 1:
        axes = [axes]

    axes[0].plot(
        t,
        data,
        color=style.waveform_color,
        linewidth=style.linewidth,
        alpha=style.alpha,
    )
    _beat_vlines(axes[0], onsets, color="#e74c3c", linewidth=1.2, label="Onset")
    axes[0].legend(fontsize=style.legend_size)
    axes[0].set_ylabel("Amplitude", fontsize=style.label_size)

    if show_odf:
        odf_values, odf_times = audio.onset_detection_function(config)
        axes[1].plot(
            odf_times,
            odf_values,
            color="#8e44ad",
            linewidth=style.linewidth,
            alpha=style.alpha,
        )
        axes[1].set_ylabel("ODF", fontsize=style.label_size)
        axes[1].set_xlabel("Time (s)", fontsize=style.label_size)
    else:
        axes[0].set_xlabel("Time (s)", fontsize=style.label_size)

    fig.suptitle(title, fontsize=style.title_size)
    assert fig is not None  # for type checker
    assert not isinstance(fig, SubFigure)
    _apply_style(fig, list(axes), style)
    return fig, list(axes)


def plot_beats(
    audio: AudioSamples,
    *,
    beat_data: BeatTrackingData | None = None,
    channel: int | None = None,
    show_tempo: bool = True,
    title: str = "Beat Tracking",
    ax: Axes | None = None,
    style: MplStyle | None = None,
) -> tuple[Figure, Axes]:
    """Waveform with beat vlines and optional tempo annotation.

    Args:
        audio: AudioSamples object.
        beat_data: BeatTrackingData. If None, detect_beats with default config.
        channel: Channel index.
        show_tempo: Annotate tempo in the title.
        title: Axes title.
        ax: Existing axes.
        style: Visual style.

    Returns:
        ``(fig, ax)``.
    """
    from audio_samples import BeatTrackingConfig, OnsetDetectionConfig

    style = _resolve_style(style)
    fig, ax = _ensure_axes(ax, style)
    t = audio.time_axis()
    data = audio[channel]

    if beat_data is None:
        onset_cfg = OnsetDetectionConfig.default()
        beat_cfg = BeatTrackingConfig(tempo_bpm=120.0, onset_config=onset_cfg)
        beat_data = audio.detect_beats(beat_cfg)

    ax.plot(
        t,
        data,
        color=style.waveform_color,
        linewidth=style.linewidth,
        alpha=style.alpha,
    )
    _beat_vlines(ax, beat_data.beat_times, color="#e67e22", linewidth=1.2, label="Beat")
    ax.legend(fontsize=style.legend_size)

    full_title = title
    if show_tempo:
        full_title = f"{title} — {beat_data.tempo_bpm:.1f} BPM"
    ax.set_xlabel("Time (s)", fontsize=style.label_size)
    ax.set_ylabel("Amplitude", fontsize=style.label_size)
    ax.set_title(full_title, fontsize=style.title_size)
    assert fig is not None  # for type checker
    assert not isinstance(fig, SubFigure)
    _apply_style(fig, ax, style)
    return fig, ax


def plot_beats_and_onsets(
    audio: AudioSamples,
    *,
    onset_config: OnsetDetectionConfig | None = None,
    beat_data: BeatTrackingData | None = None,
    channel: int | None = None,
    title: str = "Beats & Onsets",
    ax: Axes | None = None,
    style: MplStyle | None = None,
) -> tuple[Figure, Axes]:
    """Waveform with onset and beat layers in different colours.

    Args:
        audio: AudioSamples object.
        onset_config: OnsetDetectionConfig.
        beat_data: BeatTrackingData.
        channel: Channel index.
        title: Axes title.
        ax: Existing axes.
        style: Visual style.

    Returns:
        ``(fig, ax)``.
    """
    from audio_samples import BeatTrackingConfig, OnsetDetectionConfig

    style = _resolve_style(style)
    fig, ax = _ensure_axes(ax, style)
    t = audio.time_axis()
    data = audio[channel]

    if onset_config is None:
        onset_config = OnsetDetectionConfig.default()
    if beat_data is None:
        beat_cfg = BeatTrackingConfig(tempo_bpm=120.0, onset_config=onset_config)
        beat_data = audio.detect_beats(beat_cfg)

    onsets = audio.detect_onsets(onset_config)

    ax.plot(
        t,
        data,
        color=style.waveform_color,
        linewidth=style.linewidth,
        alpha=style.alpha,
    )
    _beat_vlines(
        ax, onsets, color="#e74c3c", linewidth=1.0, label="Onset", linestyle="--"
    )
    _beat_vlines(
        ax,
        beat_data.beat_times,
        color="#e67e22",
        linewidth=1.5,
        label="Beat",
        linestyle="-",
    )
    ax.legend(fontsize=style.legend_size)
    ax.set_xlabel("Time (s)", fontsize=style.label_size)
    ax.set_ylabel("Amplitude", fontsize=style.label_size)
    ax.set_title(title, fontsize=style.title_size)
    assert fig is not None  # for type checker
    assert not isinstance(fig, SubFigure)
    _apply_style(fig, ax, style)
    return fig, ax


def plot_spectral_flux(
    audio: AudioSamples,
    *,
    cqt_params: CqtParams | None = None,
    window_size: int = 2048,
    hop_size: int = 512,
    method: SpectralFluxMethod | None = None,
    title: str = "Spectral Flux",
    ax: Axes | None = None,
    style: MplStyle | None = None,
) -> tuple[Figure, Axes]:
    """Spectral flux curve over time.

    Args:
        audio: AudioSamples object.
        cqt_params: CqtParams. Uses default when None.
        window_size: Window size in samples.
        hop_size: Hop size in samples.
        method: SpectralFluxMethod. Uses energy() when None.
        title: Axes title.
        ax: Existing axes.
        style: Visual style.

    Returns:
        ``(fig, ax)``.
    """
    from audio_samples import SpectralFluxMethod
    import spectrograms as sp

    style = _resolve_style(style)
    fig, ax = _ensure_axes(ax, style)

    if cqt_params is None:
        cqt_params = sp.CqtParams(84, 7, 32.7)
    if method is None:
        method = SpectralFluxMethod.energy

    flux_values, flux_times = audio.spectral_flux(
        cqt_params, window_size, hop_size, method
    )
    ax.plot(
        flux_times,
        flux_values,
        color=style.waveform_color,
        linewidth=style.linewidth,
        alpha=style.alpha,
    )
    ax.set_xlabel("Time (s)", fontsize=style.label_size)
    ax.set_ylabel("Spectral flux", fontsize=style.label_size)
    ax.set_title(title, fontsize=style.title_size)
    assert fig is not None  # for type checker
    assert not isinstance(fig, SubFigure)
    _apply_style(fig, ax, style)
    return fig, ax


def plot_onset_detection_function(
    audio: AudioSamples,
    *,
    config: OnsetDetectionConfig | None = None,
    show_peaks: bool = True,
    threshold: float | None = None,
    title: str = "Onset Detection Function",
    ax: Axes | None = None,
    style: MplStyle | None = None,
) -> tuple[Figure, Axes]:
    """ODF curve with peak markers and optional threshold line.

    Args:
        audio: AudioSamples object.
        config: OnsetDetectionConfig.
        show_peaks: Mark detected onset positions.
        threshold: If given, draw a horizontal threshold line.
        title: Axes title.
        ax: Existing axes.
        style: Visual style.

    Returns:
        ``(fig, ax)``.
    """
    from audio_samples import OnsetDetectionConfig

    style = _resolve_style(style)
    fig, ax = _ensure_axes(ax, style)

    if config is None:
        config = OnsetDetectionConfig.default()

    odf_values, odf_times = audio.onset_detection_function(config)
    ax.plot(
        odf_times,
        odf_values,
        color=style.waveform_color,
        linewidth=style.linewidth,
        alpha=style.alpha,
    )

    if threshold is not None:
        ax.axhline(
            threshold,
            color="red",
            linestyle="--",
            linewidth=1.0,
            alpha=0.7,
            label=f"Threshold {threshold:.3f}",
        )
        ax.legend(fontsize=style.legend_size)

    if show_peaks:
        onsets = audio.detect_onsets(config)
        odf_arr = np.asarray(odf_values)
        t_arr = np.asarray(odf_times)
        peak_vals = []
        for onset in onsets:
            idx = int(np.argmin(np.abs(t_arr - onset)))
            peak_vals.append(odf_arr[idx] if idx < len(odf_arr) else 0.0)
        ax.scatter(
            onsets,
            peak_vals,
            color="red",
            s=style.marker_size**2,
            zorder=5,
            label="Peaks",
        )
        ax.legend(fontsize=style.legend_size)

    ax.set_xlabel("Time (s)", fontsize=style.label_size)
    ax.set_ylabel("ODF", fontsize=style.label_size)
    ax.set_title(title, fontsize=style.title_size)
    assert fig is not None  # for type checker
    assert not isinstance(fig, SubFigure)
    _apply_style(fig, ax, style)
    return fig, ax


def plot_pitch_track(
    audio: AudioSamples,
    *,
    window_size: int = 2048,
    hop_size: int = 512,
    pitch_method: PitchDetectionMethod | None = None,
    show_notes: bool = False,
    show_waveform: bool = False,
    title: str = "Pitch Track",
    style: MplStyle | None = None,
) -> tuple[Figure, list[Axes]]:
    """Pitch (Hz) over time; voiced frames shown as dots, unvoiced as gaps.

    Args:
        audio: AudioSamples object.
        window_size: Analysis window in samples.
        hop_size: Hop between windows.
        pitch_method: PitchDetectionMethod. Uses YIN when None.
        show_notes: Add note-name labels on the y-axis.
        show_waveform: Add a waveform panel above the pitch track.
        title: Figure suptitle.
        style: Visual style.

    Returns:
        ``(fig, axes)``.
    """
    import matplotlib.pyplot as plt
    from audio_samples import PitchDetectionMethod

    style = _resolve_style(style)

    if pitch_method is None:
        pitch_method = PitchDetectionMethod.yin

    track = audio.track_pitch(window_size, hop_size, pitch_method)
    times = np.asarray([pt[0] for pt in track])
    pitches = np.asarray([pt[1] if pt[1] is not None else float("nan") for pt in track])

    n_rows = 2 if show_waveform else 1
    fig, axes = plt.subplots(
        n_rows,
        1,
        sharex=True if show_waveform else False,
        figsize=(style.fig_width, style.fig_height * n_rows),
        dpi=style.dpi,
    )
    if n_rows == 1:
        axes = [axes]

    if show_waveform:
        t = audio.time_axis()
        axes[0].plot(t, audio, color=style.waveform_color, linewidth=style.linewidth)
        axes[0].set_ylabel("Amplitude", fontsize=style.label_size)
        pitch_ax = axes[1]
    else:
        pitch_ax = axes[0]

    pitch_ax.scatter(
        times, pitches, s=style.marker_size**2, color=style.waveform_color, alpha=0.8
    )
    pitch_ax.set_xlabel("Time (s)", fontsize=style.label_size)
    pitch_ax.set_ylabel("Frequency (Hz)", fontsize=style.label_size)

    fig.suptitle(title, fontsize=style.title_size)
    assert fig is not None  # for type checker
    assert not isinstance(fig, SubFigure)
    _apply_style(fig, list(axes), style)
    return fig, list(axes)


def plot_pitch_histogram(
    audio: AudioSamples,
    *,
    window_size: int = 2048,
    hop_size: int = 512,
    pitch_method: PitchDetectionMethod | None = None,
    bins: int = 50,
    title: str = "Pitch Histogram",
    ax: Axes | None = None,
    style: MplStyle | None = None,
) -> tuple[Figure, Axes]:
    """Distribution of voiced pitch frequencies.

    Args:
        audio: AudioSamples object.
        window_size: Analysis window in samples.
        hop_size: Hop between windows.
        pitch_method: PitchDetectionMethod.
        bins: Number of histogram bins.
        title: Axes title.
        ax: Existing axes.
        style: Visual style.

    Returns:
        ``(fig, ax)``.
    """
    from audio_samples import PitchDetectionMethod

    style = _resolve_style(style)
    fig, ax = _ensure_axes(ax, style)

    if pitch_method is None:
        pitch_method = PitchDetectionMethod.yin

    track = audio.track_pitch(window_size, hop_size, pitch_method)
    pitches = np.asarray([pt[1] for pt in track if pt[1] is not None], dtype=np.float64)

    if len(pitches) == 0:
        ax.set_title(f"{title} (no voiced frames)", fontsize=style.title_size)

        assert fig is not None  # for type checker
        assert not isinstance(fig, SubFigure)
        _apply_style(fig, ax, style)
        return fig, ax

    ax.hist(pitches, bins=bins, color=style.waveform_color, alpha=style.alpha)
    ax.set_xlabel("Frequency (Hz)", fontsize=style.label_size)
    ax.set_ylabel("Count", fontsize=style.label_size)
    ax.set_title(title, fontsize=style.title_size)
    assert fig is not None  # for type checker
    assert not isinstance(fig, SubFigure)
    _apply_style(fig, ax, style)
    return fig, ax
