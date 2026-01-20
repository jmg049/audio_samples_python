"""Envelope and dynamics visualizations."""

from __future__ import annotations

import numpy as np

from audio_samples import AudioSamples, DynamicRangeMethod, EnvelopeFollower

from ._utils import (
    _apply_style,
    _ensure_axes,
    _resolve_style,
    _to_db,
)

from matplotlib.axes import Axes
from matplotlib.figure import Figure, SubFigure
from .style import MplStyle


def plot_amplitude_envelope(
    audio: AudioSamples,
    *,
    channel: int | None = None,
    show_waveform: bool = True,
    title: str = "Amplitude Envelope",
    ax: Axes | None = None,
    style: MplStyle | None = None,
) -> tuple[Figure, Axes]:
    """Rectified amplitude envelope, optionally overlaid on the waveform.

    Args:
        audio: AudioSamples object.
        channel: Channel index.
        show_waveform: Draw the raw waveform in gray beneath the envelope.
        title: Axes title.
        ax: Existing axes.
        style: Visual style.

    Returns:
        ``(fig, ax)``.
    """
    style = _resolve_style(style)
    fig, ax = _ensure_axes(ax, style)
    t = audio.time_axis()
    data = audio[channel]

    if show_waveform:
        ax.plot(
            t,
            data,
            color="lightgray",
            linewidth=style.linewidth * 0.6,
            alpha=0.6,
            label="Waveform",
        )

    env = audio.amplitude_envelope()
    if env.ndim > 1:
        ch = channel if channel is not None else 0
        env = env[ch]

    ax.plot(
        t,
        env,
        color=style.waveform_color,
        linewidth=style.linewidth,
        alpha=style.alpha,
        label="Amplitude env.",
    )
    ax.set_xlabel("Time (s)", fontsize=style.label_size)
    ax.set_ylabel("Amplitude", fontsize=style.label_size)
    ax.set_title(title, fontsize=style.title_size)
    ax.legend(fontsize=style.legend_size)
    assert fig is not None  # for type checker
    assert not isinstance(fig, SubFigure)
    _apply_style(fig, ax, style)
    return fig, ax


def plot_rms_envelope(
    audio: AudioSamples,
    *,
    window_size: int = 2048,
    hop_size: int = 512,
    db_scale: bool = False,
    show_waveform: bool = True,
    channel: int | None = None,
    title: str = "RMS Envelope",
    ax: Axes | None = None,
    style: MplStyle | None = None,
) -> tuple[Figure, Axes]:
    """RMS over a sliding window.

    Args:
        audio: AudioSamples object.
        window_size: Window length in samples.
        hop_size: Hop between windows.
        db_scale: Convert RMS to dB.
        show_waveform: Draw raw waveform in gray.
        channel: Channel index.
        title: Axes title.
        ax: Existing axes.
        style: Visual style.

    Returns:
        ``(fig, ax)``.
    """
    style = _resolve_style(style)
    fig, ax = _ensure_axes(ax, style)
    t = audio.time_axis()
    data = audio[channel]

    if show_waveform:
        ax.plot(
            t,
            data,
            color="lightgray",
            linewidth=style.linewidth * 0.6,
            alpha=0.6,
            label="Waveform",
        )

    rms = audio.rms_envelope(window_size, hop_size)
    if rms.ndim > 1:
        ch = channel if channel is not None else 0
        rms = rms[ch]

    t_rms = np.linspace(0, t[-1], len(rms))
    y = _to_db(rms) if db_scale else rms
    ylabel = "RMS (dB)" if db_scale else "RMS"

    ax.plot(
        t_rms,
        y,
        color=style.waveform_color,
        linewidth=style.linewidth,
        alpha=style.alpha,
        label="RMS",
    )
    ax.set_xlabel("Time (s)", fontsize=style.label_size)
    ax.set_ylabel(ylabel, fontsize=style.label_size)
    ax.set_title(title, fontsize=style.title_size)
    ax.legend(fontsize=style.legend_size)
    assert fig is not None  # for type checker
    assert not isinstance(fig, SubFigure)
    _apply_style(fig, ax, style)
    return fig, ax


def plot_analytic_envelope(
    audio: AudioSamples,
    *,
    channel: int | None = None,
    show_waveform: bool = True,
    title: str = "Analytic Envelope",
    ax: Axes | None = None,
    style: MplStyle | None = None,
) -> tuple[Figure, Axes]:
    """Hilbert instantaneous amplitude.

    Args:
        audio: AudioSamples object.
        channel: Channel index.
        show_waveform: Draw raw waveform in gray.
        title: Axes title.
        ax: Existing axes.
        style: Visual style.

    Returns:
        ``(fig, ax)``.
    """
    style = _resolve_style(style)
    fig, ax = _ensure_axes(ax, style)
    t = audio.time_axis()
    data = audio[channel]

    if show_waveform:
        ax.plot(
            t,
            data,
            color="lightgray",
            linewidth=style.linewidth * 0.6,
            alpha=0.6,
            label="Waveform",
        )

    env = audio.analytic_envelope()
    if env.ndim > 1:
        ch = channel if channel is not None else 0
        env = env[ch]

    ax.plot(
        t,
        env,
        color=style.waveform_color,
        linewidth=style.linewidth,
        alpha=style.alpha,
        label="Analytic env.",
    )
    ax.set_xlabel("Time (s)", fontsize=style.label_size)
    ax.set_ylabel("Amplitude", fontsize=style.label_size)
    ax.set_title(title, fontsize=style.title_size)
    ax.legend(fontsize=style.legend_size)
    assert fig is not None  # for type checker
    assert not isinstance(fig, SubFigure)
    _apply_style(fig, ax, style)
    return fig, ax


def plot_attack_decay_envelope(
    audio: AudioSamples,
    *,
    follower: EnvelopeFollower | None = None,
    method: DynamicRangeMethod | None = None,
    channel: int | None = None,
    title: str = "Attack / Decay Envelope",
    ax: Axes | None = None,
    style: MplStyle | None = None,
) -> tuple[Figure, Axes]:
    """Attack and decay curves with different colours.

    Args:
        audio: AudioSamples object.
        follower: EnvelopeFollower instance. Uses default when None.
        method: DynamicRangeMethod. Uses peak when None.
        channel: Channel index (used for multi-channel audio selection before
            passing to the Rust method).
        title: Axes title.
        ax: Existing axes.
        style: Visual style.

    Returns:
        ``(fig, ax)``.
    """
    from audio_samples import DynamicRangeMethod, EnvelopeFollower

    style = _resolve_style(style)
    fig, ax = _ensure_axes(ax, style)
    t = audio.time_axis()

    if follower is None:
        follower = EnvelopeFollower.default()
    if method is None:
        method = DynamicRangeMethod.peak

    attack, decay = audio.attack_decay_envelope(follower, method)

    if attack.ndim > 1:
        ch = channel if channel is not None else 0
        attack = attack[ch]
        decay = decay[ch]

    t_env = np.linspace(0, t[-1], len(attack))
    ax.plot(
        t_env,
        attack,
        color="#e74c3c",
        linewidth=style.linewidth,
        alpha=style.alpha,
        label="Attack",
    )
    ax.plot(
        t_env,
        decay,
        color="#2980b9",
        linewidth=style.linewidth,
        alpha=style.alpha,
        label="Decay",
    )

    ax.set_xlabel("Time (s)", fontsize=style.label_size)
    ax.set_ylabel("Amplitude", fontsize=style.label_size)
    ax.set_title(title, fontsize=style.title_size)
    ax.legend(fontsize=style.legend_size)
    assert fig is not None  # for type checker
    assert not isinstance(fig, SubFigure)
    _apply_style(fig, ax, style)
    return fig, ax


def plot_moving_average_envelope(
    audio: AudioSamples,
    *,
    window_size: int = 2048,
    hop_size: int = 512,
    channel: int | None = None,
    show_waveform: bool = True,
    title: str = "Moving Average Envelope",
    ax: Axes | None = None,
    style: MplStyle | None = None,
) -> tuple[Figure, Axes]:
    """Smoothed rectified envelope using a moving average.

    Args:
        audio: AudioSamples object.
        window_size: Window length in samples.
        hop_size: Hop between windows.
        channel: Channel index.
        show_waveform: Draw raw waveform in gray.
        title: Axes title.
        ax: Existing axes.
        style: Visual style.

    Returns:
        ``(fig, ax)``.
    """
    style = _resolve_style(style)
    fig, ax = _ensure_axes(ax, style)
    t = audio.time_axis()
    data = audio[channel]

    if show_waveform:
        ax.plot(
            t,
            data,
            color="lightgray",
            linewidth=style.linewidth * 0.6,
            alpha=0.6,
            label="Waveform",
        )

    env = audio.moving_average_envelope(window_size, hop_size)

    if env.ndim > 1:
        ch = channel if channel is not None else 0
        env = env[ch]

    t_env = np.linspace(0, t[-1], len(env))
    ax.plot(
        t_env,
        env,
        color=style.waveform_color,
        linewidth=style.linewidth,
        alpha=style.alpha,
        label="Moving avg. env.",
    )
    ax.set_xlabel("Time (s)", fontsize=style.label_size)
    ax.set_ylabel("Amplitude", fontsize=style.label_size)
    ax.set_title(title, fontsize=style.title_size)
    ax.legend(fontsize=style.legend_size)
    assert fig is not None  # for type checker
    assert not isinstance(fig, SubFigure)
    _apply_style(fig, ax, style)
    return fig, ax


def plot_envelope_comparison(
    audio: AudioSamples,
    *,
    channel: int | None = None,
    window_size: int = 2048,
    hop_size: int = 512,
    title: str = "Envelope Comparison",
    ax: Axes | None = None,
    style: MplStyle | None = None,
) -> tuple[Figure, Axes]:
    """Overlay amplitude, RMS, analytic and moving-average envelopes.

    Args:
        audio: AudioSamples object.
        channel: Channel index.
        window_size: Window length in samples (RMS / moving average).
        hop_size: Hop between windows.
        title: Axes title.
        ax: Existing axes.
        style: Visual style.

    Returns:
        ``(fig, ax)``.
    """
    style = _resolve_style(style)
    fig, ax = _ensure_axes(ax, style)
    t = audio.time_axis()

    colors = ["#2c7bb6", "#d7191c", "#1a9641", "#fdae61"]
    labels = ["Amplitude", "RMS", "Analytic", "Moving avg."]

    # 1. Amplitude envelope (sample-level)
    amp = audio.amplitude_envelope()
    if amp.ndim > 1:
        ch = channel if channel is not None else 0
        amp = amp[ch]
    ax.plot(
        t, amp, color=colors[0], linewidth=style.linewidth, alpha=0.8, label=labels[0]
    )

    # 2. RMS envelope (frame-level)
    rms = audio.rms_envelope(window_size, hop_size)
    if rms.ndim > 1:
        ch = channel if channel is not None else 0
        rms = rms[ch]
    t_rms = np.linspace(0, t[-1], len(rms))
    ax.plot(
        t_rms,
        rms,
        color=colors[1],
        linewidth=style.linewidth,
        alpha=0.8,
        label=labels[1],
    )

    # 3. Analytic envelope
    anl = audio.analytic_envelope()
    if anl.ndim > 1:
        ch = channel if channel is not None else 0
        anl = anl[ch]
    ax.plot(
        t, anl, color=colors[2], linewidth=style.linewidth, alpha=0.8, label=labels[2]
    )

    # 4. Moving average envelope
    mav = audio.moving_average_envelope(window_size, hop_size)
    if mav.ndim > 1:
        ch = channel if channel is not None else 0
        mav = mav[ch]
    t_mav = np.linspace(0, t[-1], len(mav))
    ax.plot(
        t_mav,
        mav,
        color=colors[3],
        linewidth=style.linewidth,
        alpha=0.8,
        label=labels[3],
    )

    ax.set_xlabel("Time (s)", fontsize=style.label_size)
    ax.set_ylabel("Amplitude", fontsize=style.label_size)
    ax.set_title(title, fontsize=style.title_size)
    ax.legend(fontsize=style.legend_size)
    assert fig is not None  # for type checker
    assert not isinstance(fig, SubFigure)
    _apply_style(fig, ax, style)
    return fig, ax
