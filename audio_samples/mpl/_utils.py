"""Internal helpers for audio_samples.mpl visualizations."""

from __future__ import annotations
from typing import Any

import numpy as np
from numpy.typing import NDArray
from matplotlib.axes import Axes
from matplotlib.figure import Figure, SubFigure
from spectrograms import Spectrogram

from audio_samples import AudioSamples
from .style import MplStyle


def _ensure_axes(
    ax: Axes | None,
    style: MplStyle,
    figsize: tuple[float, float] | None = None,
) -> tuple[Figure | SubFigure | None, Axes]:
    """Return (fig, ax). Creates a new figure if *ax* is None."""
    import matplotlib.pyplot as plt

    if ax is not None:
        return ax.get_figure(), ax
    w = style.fig_width
    h = style.fig_height
    if figsize is not None:
        w, h = figsize
    fig, ax = plt.subplots(figsize=(w, h), dpi=style.dpi)
    return fig, ax


def _to_ax_list(axes: Axes | list[Axes]) -> list[Axes]:
    """Flatten axes (single Axes, list) to a flat list."""
    if hasattr(axes, "__iter__"):
        flat = list(np.asarray(axes).ravel())
        return flat

    return [axes]  # type: ignore


def _apply_style(
    fig: Figure | SubFigure,
    axes: Axes | list[Axes],
    style: MplStyle,
) -> None:
    """Apply seaborn theme, despine, tight layout to a figure."""
    try:
        import seaborn as sns

        sns.set_theme(
            style="ticks" if style.despine else "whitegrid",
            font=style.font_family,
            font_scale=style.font_size / 11.0,
            rc={
                "axes.titlesize": style.title_size,
                "axes.labelsize": style.label_size,
                "xtick.labelsize": style.tick_size,
                "ytick.labelsize": style.tick_size,
                "legend.fontsize": style.legend_size,
                "lines.linewidth": style.linewidth,
            },
        )
        if style.despine:
            for ax in _to_ax_list(axes):
                sns.despine(ax=ax)
    except ImportError:
        # seaborn not available; apply minimal styling via rcParams
        import matplotlib as mpl

        mpl.rcParams.update(
            {
                "font.family": style.font_family,
                "font.size": style.font_size,
                "axes.titlesize": style.title_size,
                "axes.labelsize": style.label_size,
                "xtick.labelsize": style.tick_size,
                "ytick.labelsize": style.tick_size,
                "legend.fontsize": style.legend_size,
                "lines.linewidth": style.linewidth,
            }
        )

    if style.show_grid:
        for ax in _to_ax_list(axes):
            ax.grid(True, alpha=style.grid_alpha)

    if style.tight_layout:
        try:
            assert not isinstance(fig, SubFigure)
            fig.tight_layout()
        except Exception:
            pass


def _add_colorbar(
    mappable: Any,
    ax: "Axes",
    label: str = "dB",
) -> None:
    """Attach a colorbar to *ax* for *mappable*."""
    fig = ax.get_figure()
    if fig is not None:
        fig.colorbar(mappable, ax=ax, label=label)
    else:
        raise RuntimeError("Unable to add colorbar: Axes has no associated Figure")


def _to_db(arr: NDArray, vmin: float = -80.0) -> NDArray:
    """Convert a magnitude array to dB, floored at *vmin*."""
    arr = np.asarray(arr, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        db = 20.0 * np.log10(np.abs(arr) + 1e-12)
    return np.maximum(db, vmin)


def _beat_vlines(ax: "Axes", times: list[float], **kwargs: Any) -> None:
    """Draw vertical lines at each beat/onset time."""
    defaults = {"color": "r", "alpha": 0.7, "linewidth": 1.0, "linestyle": "--"}
    defaults.update(kwargs)
    for t in times:
        ax.axvline(t, **defaults)


def _shade_regions(
    ax: "Axes",
    regions_seconds: list[tuple[float, float]],
    **kwargs: Any,
) -> None:
    """Fill between (start, end) time spans on *ax*."""
    defaults = {"alpha": 0.25, "color": "green"}
    defaults.update(kwargs)
    ylim = ax.get_ylim()
    for start, end in regions_seconds:
        ax.axvspan(start, end, **defaults)
    ax.set_ylim(ylim)


def _palette_colors(n: int, palette: str = "muted") -> list[tuple]:
    """Return *n* RGB tuples from a seaborn palette (falls back to tab10)."""
    try:
        import seaborn as sns

        return list(sns.color_palette(palette, n))
    except ImportError:
        import matplotlib.pyplot as plt

        cmap = plt.get_cmap("tab10")
        return [cmap(i / max(n - 1, 1))[:3] for i in range(n)]


def _resolve_style(style: "MplStyle | None") -> "MplStyle":
    from .style import get_default_style

    return style if style is not None else get_default_style()


def _compute_stft(
    audio: AudioSamples,
    n_fft: int = 2048,
    hop_size: int = 512,
) -> tuple[Spectrogram, list[float], list[float]]:
    """Compute a magnitude STFT via the spectrograms package.

    Returns ``(magnitude, freqs, times)`` where magnitude has shape
    ``(n_freqs, n_frames)``, freqs has shape ``(n_freqs,)`` and times
    has shape ``(n_frames,)``.
    """
    import spectrograms as sp

    stft_p = sp.StftParams(
        window=sp.WindowType.hanning, hop_size=hop_size, n_fft=n_fft, centre=True
    )
    spec_p = sp.SpectrogramParams(stft_p, audio.sample_rate)
    magnitude = sp.compute_linear_magnitude_spectrogram(audio, spec_p)
    freqs = magnitude.frequencies
    times = magnitude.times
    return magnitude, freqs, times
