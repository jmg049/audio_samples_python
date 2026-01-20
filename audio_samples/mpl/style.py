"""MplStyle dataclass and style management for audio_samples matplotlib visualizations."""

from __future__ import annotations

import dataclasses
from contextlib import contextmanager
from typing import Generator

_DEFAULT_STYLE: MplStyle | None = None


@dataclasses.dataclass
class MplStyle:
    """Style configuration for matplotlib-based audio visualizations.

    All visualization functions accept an ``MplStyle`` instance via the ``style``
    keyword argument. When ``style=None`` the global default (set via
    ``set_default_style``) is used; if no global default has been set the
    instance-level defaults apply.
    """

    # Figure geometry
    fig_width: float = 8.0
    fig_height: float = 4.0
    dpi: int = 150

    # Typography
    font_family: str = "serif"
    font_size: float = 11.0
    title_size: float = 13.0
    label_size: float = 11.0
    tick_size: float = 9.0
    legend_size: float = 9.0

    # Lines / markers
    linewidth: float = 1.2
    alpha: float = 1.0
    marker_size: float = 4.0

    # Colour
    palette: str = "muted"
    waveform_color: str = "#2c7bb6"
    colormap: str = "magma"
    heatmap_colormap: str = "coolwarm"

    # Grid / spines
    show_grid: bool = True
    grid_alpha: float = 0.3
    despine: bool = True

    # Spectrogram dB range
    db_vmin: float = -80.0
    db_vmax: float = 0.0
    show_colorbar: bool = True
    colorbar_label: str = "dB"

    tight_layout: bool = False

    # ------------------------------------------------------------------
    # Preset factories
    # ------------------------------------------------------------------

    @classmethod
    def paper(cls) -> MplStyle:
        """Publication-ready style: serif fonts, 300 DPI, no grid, despined."""
        return cls(
            fig_width=6.5,
            fig_height=3.5,
            dpi=300,
            font_family="serif",
            font_size=10.0,
            title_size=11.0,
            label_size=10.0,
            tick_size=8.0,
            legend_size=8.0,
            linewidth=1.0,
            show_grid=False,
            despine=True,
        )

    @classmethod
    def notebook(cls) -> MplStyle:
        """Notebook-friendly style: sans-serif, 150 DPI, grid on."""
        return cls(
            fig_width=9.0,
            fig_height=4.5,
            dpi=150,
            font_family="sans-serif",
            font_size=11.0,
            title_size=13.0,
            label_size=11.0,
            tick_size=9.0,
            legend_size=9.0,
            linewidth=1.2,
            show_grid=True,
            despine=False,
        )

    @classmethod
    def presentation(cls) -> MplStyle:
        """Presentation style: large fonts, thick lines, high contrast."""
        return cls(
            fig_width=12.0,
            fig_height=5.0,
            dpi=150,
            font_family="sans-serif",
            font_size=14.0,
            title_size=16.0,
            label_size=14.0,
            tick_size=12.0,
            legend_size=12.0,
            linewidth=2.0,
            marker_size=6.0,
            show_grid=True,
            grid_alpha=0.25,
            despine=True,
        )

    @classmethod
    def default(cls) -> MplStyle:
        """Balanced defaults."""
        return cls()

    def __repr__(self) -> str:
        fields = dataclasses.fields(self)
        kv = ", ".join(f"{f.name}={getattr(self, f.name)!r}" for f in fields)
        return f"MplStyle({kv})"


# ------------------------------------------------------------------
# Global default helpers
# ------------------------------------------------------------------


def set_default_style(style: MplStyle) -> None:
    """Set the process-wide default ``MplStyle`` used when ``style=None``."""
    global _DEFAULT_STYLE
    _DEFAULT_STYLE = style


def get_default_style() -> MplStyle:
    """Return the current process-wide default ``MplStyle``."""
    if _DEFAULT_STYLE is None:
        return MplStyle()
    return _DEFAULT_STYLE


@contextmanager
def style_context(style: MplStyle) -> Generator[None, None, None]:
    """Context manager that temporarily sets the default style.

    Example::

        with style_context(MplStyle.paper()):
            fig, ax = plot_waveform(audio)
    """
    global _DEFAULT_STYLE
    previous = _DEFAULT_STYLE
    _DEFAULT_STYLE = style
    try:
        yield
    finally:
        _DEFAULT_STYLE = previous
