"""Type stubs for audio_samples.mpl.style."""

from __future__ import annotations
from contextlib import AbstractContextManager

class MplStyle:
    fig_width: float
    fig_height: float
    dpi: int
    font_family: str
    font_size: float
    title_size: float
    label_size: float
    tick_size: float
    legend_size: float
    linewidth: float
    alpha: float
    marker_size: float
    palette: str
    waveform_color: str
    colormap: str
    heatmap_colormap: str
    show_grid: bool
    grid_alpha: float
    despine: bool
    db_vmin: float
    db_vmax: float
    show_colorbar: bool
    colorbar_label: str
    tight_layout: bool

    def __init__(
        self,
        *,
        fig_width: float = ...,
        fig_height: float = ...,
        dpi: int = ...,
        font_family: str = ...,
        font_size: float = ...,
        title_size: float = ...,
        label_size: float = ...,
        tick_size: float = ...,
        legend_size: float = ...,
        linewidth: float = ...,
        alpha: float = ...,
        marker_size: float = ...,
        palette: str = ...,
        waveform_color: str = ...,
        colormap: str = ...,
        heatmap_colormap: str = ...,
        show_grid: bool = ...,
        grid_alpha: float = ...,
        despine: bool = ...,
        db_vmin: float = ...,
        db_vmax: float = ...,
        show_colorbar: bool = ...,
        colorbar_label: str = ...,
        tight_layout: bool = ...,
    ) -> None: ...
    @classmethod
    def paper(cls) -> MplStyle: ...
    @classmethod
    def notebook(cls) -> MplStyle: ...
    @classmethod
    def presentation(cls) -> MplStyle: ...
    @classmethod
    def default(cls) -> MplStyle: ...

def set_default_style(style: MplStyle) -> None: ...
def get_default_style() -> MplStyle: ...
def style_context(style: MplStyle) -> AbstractContextManager[None]: ...
