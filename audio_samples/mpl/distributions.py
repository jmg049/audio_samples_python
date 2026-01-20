"""Statistical and distribution visualizations."""

from __future__ import annotations

import numpy as np

from audio_samples import AudioSamples

from ._utils import (
    _apply_style,
    _ensure_axes,
    _resolve_style,
)
from matplotlib.axes import Axes
from matplotlib.figure import Figure, SubFigure
from .style import MplStyle


def plot_amplitude_histogram(
    audio: AudioSamples,
    *,
    channel: int | None = None,
    bins: int = 100,
    kde: bool = True,
    log_y: bool = False,
    title: str = "Amplitude Distribution",
    ax: Axes | None = None,
    style: MplStyle | None = None,
) -> tuple[Figure, Axes]:
    """Amplitude histogram with optional KDE and statistical markers.

    Args:
        audio: AudioSamples object.
        channel: Channel index.
        bins: Number of histogram bins.
        kde: Overlay a kernel density estimate.
        log_y: Use log scale for count axis.
        title: Axes title.
        ax: Existing axes.
        style: Visual style.

    Returns:
        ``(fig, ax)``.
    """
    style = _resolve_style(style)
    fig, ax = _ensure_axes(ax, style)
    data = audio[channel]

    try:
        import seaborn as sns

        sns.histplot(
            data.to_numpy(),
            bins=bins,
            kde=kde,
            ax=ax,
            color=style.waveform_color,
            alpha=0.7,
        )
    except ImportError:
        ax.hist(data, bins=bins, color=style.waveform_color, alpha=0.7)

    # Statistical markers
    peak_val = float(np.max(np.abs(data)))
    rms_val = float(np.sqrt(np.mean(data**2)))
    mean_val = float(np.mean(data))

    for val, lbl, col in [
        (peak_val, f"Peak {peak_val:.3f}", "#e74c3c"),
        (rms_val, f"RMS {rms_val:.3f}", "#e67e22"),
        (mean_val, f"Mean {mean_val:.3f}", "#27ae60"),
    ]:
        ax.axvline(val, color=col, linestyle="--", linewidth=1.0, alpha=0.8, label=lbl)

    if log_y:
        ax.set_yscale("log")

    ax.set_xlabel("Amplitude", fontsize=style.label_size)
    ax.set_ylabel("Count (log)" if log_y else "Count", fontsize=style.label_size)
    ax.set_title(title, fontsize=style.title_size)
    ax.legend(fontsize=style.legend_size)
    assert fig is not None  # for type checker
    assert not isinstance(fig, SubFigure)
    _apply_style(fig, ax, style)
    return fig, ax


def plot_sample_distribution(
    audio: AudioSamples,
    *,
    plot_type: str = "violin",
    title: str = "Sample Distribution by Channel",
    ax: Axes | None = None,
    style: MplStyle | None = None,
) -> tuple[Figure, Axes]:
    """Violin or box plot of amplitudes per channel.

    Args:
        audio: AudioSamples object.
        plot_type: ``"violin"`` or ``"box"``.
        title: Axes title.
        ax: Existing axes.
        style: Visual style.

    Returns:
        ``(fig, ax)``.
    """
    style = _resolve_style(style)
    fig, ax = _ensure_axes(ax, style)

    arr = audio.to_mono()
    data_list = [arr[i] for i in range(arr.shape()[0])]
    labels = [f"Ch {i}" for i in range(arr.shape()[0])]

    try:
        import seaborn as sns
        import pandas as pd

        records = []
        for ch_idx, ch_data in enumerate(data_list):
            for val in ch_data:
                records.append({"channel": labels[ch_idx], "amplitude": val})
        df = pd.DataFrame(records)

        if plot_type == "violin":
            sns.violinplot(
                data=df, x="channel", y="amplitude", ax=ax, palette=style.palette
            )
        else:
            sns.boxplot(
                data=df, x="channel", y="amplitude", ax=ax, palette=style.palette
            )
    except ImportError:
        ax.boxplot(data_list)

    ax.set_xlabel("Channel", fontsize=style.label_size)
    ax.set_ylabel("Amplitude", fontsize=style.label_size)
    ax.set_title(title, fontsize=style.title_size)
    assert fig is not None  # for type checker
    assert not isinstance(fig, SubFigure)
    _apply_style(fig, ax, style)
    return fig, ax


def plot_statistics_summary(
    audio: AudioSamples,
    *,
    title: str = "Audio Statistics",
    ax: Axes | None = None,
    style: MplStyle | None = None,
) -> tuple[Figure, Axes]:
    """Horizontal bar chart of normalised audio statistics.

    Args:
        audio: AudioSamples object.
        title: Axes title.
        ax: Existing axes.
        style: Visual style.

    Returns:
        ``(fig, ax)``.
    """
    style = _resolve_style(style)
    fig, ax = _ensure_axes(ax, style)

    stats = {
        "Peak": float(audio.peak()),
        "RMS": float(audio.rms()),
        "Mean": float(audio.mean()),
        "Std Dev": float(audio.std_dev()),
        "Zero Cross. Rate": float(audio.zero_crossing_rate()),
        "Spectral Centroid": float(audio.spectral_centroid()) / (audio.sample_rate / 2),
    }

    labels = list(stats.keys())
    values = list(stats.values())

    ax.barh(labels, values, color=style.waveform_color, alpha=style.alpha)
    ax.set_xlabel("Value (normalised)", fontsize=style.label_size)
    ax.set_title(title, fontsize=style.title_size)
    assert fig is not None  # for type checker
    assert not isinstance(fig, SubFigure)
    _apply_style(fig, ax, style)
    return fig, ax


def plot_dynamic_range(
    audio: AudioSamples,
    *,
    channel: int | None = None,
    title: str = "Dynamic Range",
    ax: Axes | None = None,
    style: MplStyle | None = None,
) -> tuple[Figure, Axes]:
    """Waveform with peak/RMS dB annotations; clips in red, silence in gray.

    Args:
        audio: AudioSamples object.
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

    ax.plot(
        t,
        data,
        color=style.waveform_color,
        linewidth=style.linewidth,
        alpha=style.alpha,
    )

    peak_val = float(np.max(np.abs(data)))
    rms_val = float(np.sqrt(np.mean(data**2)))

    with np.errstate(divide="ignore", invalid="ignore"):
        peak_db = 20.0 * np.log10(peak_val + 1e-12)
        rms_db = 20.0 * np.log10(rms_val + 1e-12)

    ax.axhline(
        peak_val,
        color="#e74c3c",
        linestyle="--",
        linewidth=1.0,
        alpha=0.7,
        label=f"Peak {peak_db:.1f} dB",
    )
    ax.axhline(-peak_val, color="#e74c3c", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.axhline(
        rms_val,
        color="#e67e22",
        linestyle=":",
        linewidth=1.0,
        alpha=0.7,
        label=f"RMS {rms_db:.1f} dB",
    )
    ax.axhline(-rms_val, color="#e67e22", linestyle=":", linewidth=1.0, alpha=0.7)

    ax.set_xlabel("Time (s)", fontsize=style.label_size)
    ax.set_ylabel("Amplitude", fontsize=style.label_size)
    ax.set_title(title, fontsize=style.title_size)
    ax.legend(fontsize=style.legend_size)
    assert fig is not None  # for type checker
    assert not isinstance(fig, SubFigure)
    _apply_style(fig, ax, style)
    return fig, ax


def plot_clipping_detection(
    audio: AudioSamples,
    *,
    threshold: float = 0.99,
    channel: int | None = None,
    title: str = "Clipping Detection",
    ax: Axes | None = None,
    style: MplStyle | None = None,
) -> tuple[Figure, Axes]:
    """Waveform with clipped regions highlighted in red and threshold lines.

    Args:
        audio: AudioSamples object.
        threshold: Clipping threshold (absolute amplitude).
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

    ax.plot(
        t,
        data,
        color=style.waveform_color,
        linewidth=style.linewidth,
        alpha=style.alpha,
    )

    clipped = np.abs(data) >= threshold
    if clipped.any():
        ax.scatter(
            t[clipped],
            data[clipped],
            color="red",
            s=style.marker_size**2,
            zorder=5,
            label="Clipped",
        )
        ax.legend(fontsize=style.legend_size)

    ax.axhline(threshold, color="red", linestyle="--", linewidth=1.0, alpha=0.5)
    ax.axhline(-threshold, color="red", linestyle="--", linewidth=1.0, alpha=0.5)

    ax.set_xlabel("Time (s)", fontsize=style.label_size)
    ax.set_ylabel("Amplitude", fontsize=style.label_size)
    ax.set_title(title, fontsize=style.title_size)
    assert fig is not None  # for type checker
    assert not isinstance(fig, SubFigure)
    _apply_style(fig, ax, style)
    return fig, ax


def plot_silence_regions(
    audio: AudioSamples,
    *,
    threshold: float = 0.01,
    channel: int | None = None,
    title: str = "Silence Regions",
    ax: Axes | None = None,
    style: MplStyle | None = None,
) -> tuple[Figure, Axes]:
    """Waveform with silence spans shaded in gray.

    Args:
        audio: AudioSamples object.
        threshold: Amplitude threshold below which samples are silent.
        channel: Channel index.
        title: Axes title.
        ax: Existing axes.
        style: Visual style.

    Returns:
        ``(fig, ax)``.
    """
    from ._utils import _shade_regions

    style = _resolve_style(style)
    fig, ax = _ensure_axes(ax, style)
    t = audio.time_axis()
    data = audio[channel]

    ax.plot(
        t,
        data,
        color=style.waveform_color,
        linewidth=style.linewidth,
        alpha=style.alpha,
    )

    # Find contiguous silence regions
    silent = np.abs(data) < threshold
    changes = np.diff(silent.astype(int), prepend=0, append=0)
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]

    regions_seconds = []
    for s, e in zip(starts, ends):
        if s < len(t) and e <= len(t):
            regions_seconds.append((t[s], t[min(e, len(t) - 1)]))

    _shade_regions(ax, regions_seconds, color="gray", alpha=0.3)

    ax.set_xlabel("Time (s)", fontsize=style.label_size)
    ax.set_ylabel("Amplitude", fontsize=style.label_size)
    ax.set_title(title, fontsize=style.title_size)
    assert fig is not None  # for type checker
    assert not isinstance(fig, SubFigure)
    _apply_style(fig, ax, style)
    return fig, ax
