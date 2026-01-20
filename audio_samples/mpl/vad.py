"""Voice activity detection visualizations."""

from __future__ import annotations

import numpy as np

from audio_samples import AudioSamples, VadConfig

from ._utils import (
    _apply_style,
    _ensure_axes,
    _palette_colors,
    _resolve_style,
    _shade_regions,
)

from matplotlib.axes import Axes
from matplotlib.figure import Figure, SubFigure
from .style import MplStyle


def plot_vad_mask(
    audio: AudioSamples,
    *,
    config: VadConfig | None = None,
    channel: int | None = None,
    title: str = "VAD Mask",
    style: MplStyle | None = None,
) -> tuple[Figure, list[Axes]]:
    """Two-panel: waveform + binary VAD mask as a step function.

    Args:
        audio: AudioSamples object.
        config: VadConfig. Uses default when None.
        channel: Channel index for waveform display.
        title: Figure suptitle.
        style: Visual style.

    Returns:
        ``(fig, [waveform_ax, mask_ax])``.
    """
    import matplotlib.pyplot as plt
    from audio_samples import VadConfig

    style = _resolve_style(style)

    if config is None:
        config = VadConfig.default()

    mask = audio.voice_activity_mask(config)
    t = audio.time_axis()
    data = audio[channel]

    fig, axes = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(style.fig_width, style.fig_height * 1.5),
        dpi=style.dpi,
    )

    axes[0].plot(
        t,
        data,
        color=style.waveform_color,
        linewidth=style.linewidth,
        alpha=style.alpha,
    )
    axes[0].set_ylabel("Amplitude", fontsize=style.label_size)

    # Binary mask — one value per frame
    n_frames = len(mask)
    t_mask = np.linspace(0, t[-1], n_frames)
    axes[1].step(t_mask, mask, where="post", color="#27ae60", linewidth=style.linewidth)
    axes[1].set_ylim(-0.1, 1.1)
    axes[1].set_yticks([0, 1])
    axes[1].set_yticklabels(["Silence", "Speech"], fontsize=style.tick_size)
    axes[1].set_xlabel("Time (s)", fontsize=style.label_size)
    axes[1].set_ylabel("VAD", fontsize=style.label_size)

    fig.suptitle(title, fontsize=style.title_size)
    _apply_style(fig, list(axes), style)
    return fig, list(axes)


def plot_speech_regions(
    audio: AudioSamples,
    *,
    config: VadConfig | None = None,
    channel: int | None = None,
    title: str = "Speech Regions",
    ax: Axes | None = None,
    style: MplStyle | None = None,
) -> tuple[Figure, Axes]:
    """Waveform with green shaded spans for detected speech regions.

    Args:
        audio: AudioSamples object.
        config: VadConfig.
        channel: Channel index.
        title: Axes title.
        ax: Existing axes.
        style: Visual style.

    Returns:
        ``(fig, ax)``.
    """
    from audio_samples import VadConfig

    style = _resolve_style(style)
    fig, ax = _ensure_axes(ax, style)
    t = audio.time_axis()
    data = audio[channel]

    if config is None:
        config = VadConfig.default()

    ax.plot(
        t,
        data,
        color=style.waveform_color,
        linewidth=style.linewidth,
        alpha=style.alpha,
    )

    sr = audio.sample_rate
    regions = audio.speech_regions(config)
    regions_seconds = [(start / sr, end / sr) for start, end in regions]
    _shade_regions(ax, regions_seconds, color="#27ae60", alpha=0.25)

    ax.set_xlabel("Time (s)", fontsize=style.label_size)
    ax.set_ylabel("Amplitude", fontsize=style.label_size)
    ax.set_title(title, fontsize=style.title_size)
    assert fig is not None  # for type checker
    assert not isinstance(fig, SubFigure)
    _apply_style(fig, ax, style)
    return fig, ax


def plot_vad_comparison(
    audio: AudioSamples,
    configs: list[VadConfig],
    labels: list[str],
    *,
    channel: int | None = None,
    title: str = "VAD Comparison",
    style: MplStyle | None = None,
) -> tuple[Figure, list[Axes]]:
    """Waveform + horizontal VAD bars for each config.

    Args:
        audio: AudioSamples object.
        configs: List of VadConfig instances.
        labels: Label per config.
        channel: Channel index.
        title: Figure suptitle.
        style: Visual style.

    Returns:
        ``(fig, axes)`` — waveform + one bar per config.
    """
    import matplotlib.pyplot as plt

    style = _resolve_style(style)
    n = len(configs)
    colors = _palette_colors(n, style.palette)

    fig, axes = plt.subplots(
        n + 1,
        1,
        sharex=True,
        figsize=(style.fig_width, style.fig_height * (n + 1)),
        dpi=style.dpi,
    )

    t = audio.time_axis()
    data = audio[channel]
    axes[0].plot(
        t,
        data,
        color=style.waveform_color,
        linewidth=style.linewidth,
        alpha=style.alpha,
    )
    axes[0].set_ylabel("Amplitude", fontsize=style.label_size)

    for idx, (cfg, lbl, col) in enumerate(zip(configs, labels, colors)):
        mask = audio.voice_activity_mask(cfg)
        n_frames = len(mask)
        t_mask = np.linspace(0, t[-1], n_frames)
        axes[idx + 1].step(
            t_mask, mask, where="post", color=col, linewidth=style.linewidth, label=lbl
        )
        axes[idx + 1].set_ylim(-0.1, 1.1)
        axes[idx + 1].set_yticks([0, 1])
        axes[idx + 1].set_yticklabels(["Sil.", "Sp."], fontsize=style.tick_size)
        axes[idx + 1].set_ylabel(lbl, fontsize=style.label_size)

    axes[-1].set_xlabel("Time (s)", fontsize=style.label_size)
    fig.suptitle(title, fontsize=style.title_size)
    assert fig is not None  # for type checker
    assert not isinstance(fig, SubFigure)
    _apply_style(fig, list(axes), style)
    return fig, list(axes)
