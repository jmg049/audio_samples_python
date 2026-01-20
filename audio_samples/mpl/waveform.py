"""Time-domain waveform visualizations."""

from __future__ import annotations

import numpy as np

from audio_samples import AudioSamples

from ._utils import (
    _apply_style,
    _ensure_axes,
    _palette_colors,
    _resolve_style,
)
from matplotlib.axes import Axes
from matplotlib.figure import Figure, SubFigure
from .style import MplStyle


def plot_waveform(
    audio: AudioSamples,
    *,
    channel: int | None = None,
    show_rms: bool = False,
    show_envelope: bool = False,
    title: str = "Waveform",
    ax: Axes | None = None,
    style: MplStyle | None = None,
) -> tuple[Figure, Axes]:
    """Plot a single-channel (or all-channels overlaid) time-domain waveform.

    Args:
        audio: AudioSamples object.
        channel: Channel index to plot. If None and the audio is multi-channel,
            all channels are overlaid.
        show_rms: Overlay the RMS envelope.
        show_envelope: Overlay the analytic (Hilbert) envelope.
        title: Axes title.
        ax: Existing axes to draw into.
        style: Visual style; uses the global default when None.

    Returns:
        ``(fig, ax)`` tuple.
    """
    style = _resolve_style(style)
    fig, ax = _ensure_axes(ax, style)
    t = audio.time_axis()
    arr = np.asarray(audio, dtype=np.float64)

    if arr.ndim == 1 or channel is not None:
        data = audio[channel]
        ax.plot(
            t,
            data,
            color=style.waveform_color,
            linewidth=style.linewidth,
            alpha=style.alpha,
        )
    else:
        colors = _palette_colors(arr.shape[0], style.palette)
        for ch_idx in range(arr.shape[0]):
            ax.plot(
                t,
                arr[ch_idx],
                color=colors[ch_idx],
                linewidth=style.linewidth,
                alpha=style.alpha,
                label=f"Ch {ch_idx}",
            )
        ax.legend(fontsize=style.legend_size)

    if show_rms:
        window = max(1, int(0.02 * audio.sample_rate))
        hop = max(1, window // 4)
        rms = np.asarray(audio.rms_envelope(window, hop), dtype=np.float64)
        if rms.ndim > 1:
            rms = rms[0]
        t_rms = np.linspace(0, t[-1], len(rms))
        ax.plot(
            t_rms,
            rms,
            color="orange",
            linewidth=style.linewidth * 0.8,
            alpha=0.8,
            label="RMS",
        )
        ax.legend(fontsize=style.legend_size)

    if show_envelope:
        env = np.asarray(audio.analytic_envelope(), dtype=np.float64)
        if env.ndim > 1:
            env = env[0]
        ax.plot(
            t,
            env,
            color="red",
            linewidth=style.linewidth * 0.8,
            alpha=0.8,
            label="Envelope",
        )
        ax.legend(fontsize=style.legend_size)

    ax.set_xlabel("Time (s)", fontsize=style.label_size)
    ax.set_ylabel("Amplitude", fontsize=style.label_size)
    ax.set_title(title, fontsize=style.title_size)
    assert fig is not None  # for type checker
    assert not isinstance(fig, SubFigure)
    _apply_style(fig, ax, style)
    return fig, ax


def plot_waveform_channels(
    audio: AudioSamples,
    *,
    share_y: bool = True,
    title: str = "Channels",
    style: MplStyle | None = None,
) -> tuple[Figure, list[Axes]]:
    """Stacked subplots, one per channel, with a shared x-axis.

    Args:
        audio: AudioSamples object.
        share_y: Share the y-axis across all channel subplots.
        title: Figure suptitle.
        style: Visual style.

    Returns:
        ``(fig, axes)`` where axes is a list of length ``num_channels``.
    """
    import matplotlib.pyplot as plt

    style = _resolve_style(style)
    arr = np.asarray(audio, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[np.newaxis, :]
    n_ch = arr.shape[0]
    t = audio.time_axis()

    fig, axes = plt.subplots(
        n_ch,
        1,
        sharex=True,
        sharey=share_y,
        figsize=(style.fig_width, style.fig_height * n_ch),
        dpi=style.dpi,
    )
    if n_ch == 1:
        axes = [axes]

    colors = _palette_colors(n_ch, style.palette)
    for idx, ax in enumerate(axes):
        ax.plot(
            t, arr[idx], color=colors[idx], linewidth=style.linewidth, alpha=style.alpha
        )
        ax.set_ylabel(f"Ch {idx}\nAmp.", fontsize=style.label_size)
        if style.show_grid:
            ax.grid(True, alpha=style.grid_alpha)

    axes[-1].set_xlabel("Time (s)", fontsize=style.label_size)
    fig.suptitle(title, fontsize=style.title_size)
    _apply_style(fig, axes, style)
    return fig, list(axes)


def plot_waveform_comparison(
    signals: list[AudioSamples],
    labels: list[str],
    *,
    share_y: bool = True,
    title: str = "Waveform Comparison",
    style: MplStyle | None = None,
) -> tuple[Figure, list[Axes]]:
    """Multiple signals stacked vertically for before/after comparison.

    Args:
        signals: List of AudioSamples objects.
        labels: Label for each signal.
        share_y: Share y-axis across subplots.
        title: Figure suptitle.
        style: Visual style.

    Returns:
        ``(fig, axes)``.
    """
    import matplotlib.pyplot as plt

    style = _resolve_style(style)
    n = len(signals)
    colors = _palette_colors(n, style.palette)

    fig, axes = plt.subplots(
        n,
        1,
        sharex=True,
        sharey=share_y,
        figsize=(style.fig_width, style.fig_height * n),
        dpi=style.dpi,
    )
    if n == 1:
        axes = [axes]

    for idx, (sig, lbl, col) in enumerate(zip(signals, labels, colors)):
        t = sig.time_axis()
        axes[idx].plot(
            t, sig, color=col, linewidth=style.linewidth, alpha=style.alpha, label=lbl
        )
        axes[idx].set_ylabel("Amplitude", fontsize=style.label_size)
        axes[idx].legend(fontsize=style.legend_size, loc="upper right")
        if style.show_grid:
            axes[idx].grid(True, alpha=style.grid_alpha)

    axes[-1].set_xlabel("Time (s)", fontsize=style.label_size)
    fig.suptitle(title, fontsize=style.title_size)
    _apply_style(fig, list(axes), style)
    return fig, list(axes)


def plot_zero_crossings(
    audio: AudioSamples,
    *,
    channel: int | None = None,
    window_seconds: float | None = None,
    title: str = "Zero Crossings",
    ax: Axes | None = None,
    style: MplStyle | None = None,
) -> tuple[Figure, Axes]:
    """Waveform with zero-crossing markers.

    Args:
        audio: AudioSamples object.
        channel: Channel index (default: 0).
        window_seconds: If given, restrict the x-axis to this duration.
        title: Axes title.
        ax: Existing axes.
        style: Visual style.

    Returns:
        ``(fig, ax)``.
    """
    style = _resolve_style(style)
    fig, ax = _ensure_axes(ax, style)
    data = audio[channel]
    t = audio.time_axis()

    if window_seconds is not None:
        mask = t <= window_seconds
        data = data[mask]
        t = t[mask]

    ax.plot(
        t,
        data,
        color=style.waveform_color,
        linewidth=style.linewidth,
        alpha=style.alpha,
    )

    # Detect zero crossings (sign changes)
    signs = np.sign(data)
    zc_indices = np.where(np.diff(signs) != 0)[0]
    ax.scatter(
        t[zc_indices],
        np.zeros(len(zc_indices)),
        color="red",
        s=style.marker_size**2,
        zorder=5,
        label="ZC",
    )

    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.set_xlabel("Time (s)", fontsize=style.label_size)
    ax.set_ylabel("Amplitude", fontsize=style.label_size)
    ax.set_title(title, fontsize=style.title_size)
    ax.legend(fontsize=style.legend_size)
    assert fig is not None  # for type checker
    assert not isinstance(fig, SubFigure)
    _apply_style(fig, ax, style)
    return fig, ax
