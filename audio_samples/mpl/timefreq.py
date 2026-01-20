"""Time-frequency visualizations: STFT, mel, MFCC, chroma, gammatone."""

from __future__ import annotations


import numpy as np
from spectrograms import (
    ChromaParams,
    ErbParams,
    LogParams,
    MelParams,
    MfccParams,
    SpectrogramParams,
    StftParams,
    WindowType,
)

from audio_samples import AudioSamples

from ._utils import (
    _add_colorbar,
    _apply_style,
    _compute_stft,
    _ensure_axes,
    _resolve_style,
)

from matplotlib.axes import Axes
from matplotlib.figure import Figure, SubFigure
from .style import MplStyle


def plot_spectrogram(
    audio: AudioSamples,
    *,
    stft_params: StftParams | None = None,
    channel: int | None = None,
    db_range: tuple[float, float] | None = None,
    freq_max: float | None = None,
    title: str = "Spectrogram",
    ax: Axes | None = None,
    style: MplStyle | None = None,
) -> tuple[Figure, Axes]:
    """STFT spectrogram via imshow with time x frequency axes and dB colorbar.

    Args:
        audio: AudioSamples object.
        stft_params: StftParams instance. If None, uses sensible defaults.
        channel: Channel index to analyse.
        db_range: ``(vmin, vmax)`` dB values for the colormap.
        freq_max: Upper frequency limit in Hz.
        title: Axes title.
        ax: Existing axes.
        style: Visual style.

    Returns:
        ``(fig, ax)``.
    """
    style = _resolve_style(style)
    fig, ax = _ensure_axes(ax, style)

    n_fft = 2048
    hop = 512

    src = audio[channel] if channel is not None else audio
    magnitude, freq_axis, _times = _compute_stft(src, n_fft=n_fft, hop_size=hop)

    with np.errstate(divide="ignore", invalid="ignore"):
        db = 20.0 * np.log10(np.maximum(magnitude, 1e-12))

    vmin = db_range[0] if db_range else style.db_vmin
    vmax = db_range[1] if db_range else style.db_vmax

    duration = audio.duration_seconds
    freq_axis = np.array(freq_axis)
    if freq_max is not None:
        freq_mask = freq_axis <= freq_max
        db = db[freq_mask, :]
        freq_axis = freq_axis[freq_mask]

    extent = [0, duration, freq_axis[0], freq_axis[-1]]
    im = ax.imshow(
        db,
        aspect="auto",
        origin="lower",
        extent=tuple(extent),
        cmap=style.colormap,
        vmin=vmin,
        vmax=vmax,
    )

    if style.show_colorbar:
        _add_colorbar(im, ax, label=style.colorbar_label)

    ax.set_xlabel("Time (s)", fontsize=style.label_size)
    ax.set_ylabel("Frequency (Hz)", fontsize=style.label_size)
    ax.set_title(title, fontsize=style.title_size)
    assert fig is not None  # for type checker
    assert not isinstance(fig, SubFigure)
    _apply_style(fig, ax, style)
    return fig, ax


def plot_mel_spectrogram(
    audio: AudioSamples,
    *,
    spec_params: SpectrogramParams | None = None,
    mel_params: MelParams | None = None,
    log_params: LogParams | None = None,
    channel: int | None = None,
    title: str = "Mel Spectrogram",
    ax: Axes | None = None,
    style: MplStyle | None = None,
) -> tuple[Figure, Axes]:
    """Mel spectrogram with frequency axis in Hz.

    Args:
        audio: AudioSamples object.
        spec_params: SpectrogramParams. Defaults used when None.
        mel_params: MelParams. Defaults used when None.
        log_params: Optional LogParams for dB conversion.
        channel: Channel index.
        title: Axes title.
        ax: Existing axes.
        style: Visual style.

    Returns:
        ``(fig, ax)``.
    """
    style = _resolve_style(style)
    fig, ax = _ensure_axes(ax, style)
    sr = audio.sample_rate
    if spec_params is None:
        stft = StftParams(
            window=WindowType.hanning, hop_size=512, n_fft=2048, centre=True
        )
        spec_params = SpectrogramParams(stft=stft, sample_rate=sr)
    if mel_params is None:
        mel_params = MelParams(n_mels=128, f_min=0.0, f_max=sr / 2)

    mel = audio.mel_spectrogram(spec_params, mel_params, log_params)

    if mel.ndim == 3:
        ch = channel if channel is not None else 0
        mel = mel[ch]

    duration = audio.duration_seconds
    extent = [0, duration, mel_params.f_min, mel_params.f_max]

    im = ax.imshow(
        mel,
        aspect="auto",
        origin="lower",
        extent=tuple(extent),
        cmap=style.colormap,
        vmin=style.db_vmin if log_params is not None else None,
        vmax=style.db_vmax if log_params is not None else None,
    )

    if style.show_colorbar:
        label = style.colorbar_label if log_params is not None else "Magnitude"
        _add_colorbar(im, ax, label=label)

    ax.set_xlabel("Time (s)", fontsize=style.label_size)
    ax.set_ylabel("Frequency (Hz)", fontsize=style.label_size)
    ax.set_title(title, fontsize=style.title_size)
    assert fig is not None  # for type checker
    assert not isinstance(fig, SubFigure)
    _apply_style(fig, ax, style)
    return fig, ax


def plot_mfcc(
    audio: AudioSamples,
    *,
    stft_params: StftParams | None = None,
    n_mels: int = 128,
    mfcc_params: MfccParams | None = None,
    show_delta: bool = False,
    show_delta2: bool = False,
    title: str = "MFCC",
    style: MplStyle | None = None,
) -> tuple[Figure, list[Axes]]:
    """MFCC heatmap with optional delta panels.

    Args:
        audio: AudioSamples object.
        stft_params: StftParams. Defaults used when None.
        n_mels: Number of mel bands.
        mfcc_params: MfccParams. Defaults used when None.
        show_delta: Include a delta MFCC panel.
        show_delta2: Include a delta-delta MFCC panel.
        title: Figure suptitle.
        style: Visual style.

    Returns:
        ``(fig, axes)`` where axes has 1--3 elements.
    """
    import matplotlib.pyplot as plt

    style = _resolve_style(style)

    if stft_params is None:
        stft_params = StftParams(
            window=WindowType.hanning, hop_size=512, n_fft=2048, centre=True
        )
    if mfcc_params is None:
        mfcc_params = MfccParams(n_mfcc=13)

    mfcc = audio.mfcc(stft_params, n_mels, mfcc_params)
    if mfcc.ndim == 3:
        mfcc = mfcc[0]

    panels = [mfcc]
    panel_titles = ["MFCC"]
    if show_delta:
        delta = np.diff(mfcc, axis=1, prepend=mfcc[:, :1])
        panels.append(delta)
        panel_titles.append("Δ MFCC")
    if show_delta2:
        if show_delta:
            delta2 = np.diff(panels[1], axis=1, prepend=panels[1][:, :1])
        else:
            d = np.diff(mfcc, axis=1, prepend=mfcc[:, :1])
            delta2 = np.diff(d, axis=1, prepend=d[:, :1])
        panels.append(delta2)
        panel_titles.append("Δ² MFCC")

    n_panels = len(panels)
    duration = audio.duration_seconds
    fig, axes = plt.subplots(
        1,
        n_panels,
        figsize=(style.fig_width * n_panels, style.fig_height),
        dpi=style.dpi,
    )
    if n_panels == 1:
        axes = [axes]

    for ax, data, ptitle in zip(axes, panels, panel_titles):
        extent = [0, duration, 0, data.shape[0]]
        im = ax.imshow(
            data,
            aspect="auto",
            origin="lower",
            extent=extent,
            cmap=style.heatmap_colormap,
        )
        if style.show_colorbar:
            _add_colorbar(im, ax, label="Coefficient")
        ax.set_xlabel("Time (s)", fontsize=style.label_size)
        ax.set_ylabel("MFCC index", fontsize=style.label_size)
        ax.set_title(ptitle, fontsize=style.title_size)

    fig.suptitle(title, fontsize=style.title_size)
    _apply_style(fig, list(axes), style)
    return fig, list(axes)


def plot_chroma(
    audio: AudioSamples,
    *,
    stft_params: StftParams | None = None,
    chroma_params: ChromaParams | None = None,
    note_labels: bool = True,
    title: str = "Chroma",
    ax: Axes | None = None,
    style: MplStyle | None = None,
) -> tuple[Figure, Axes]:
    """Chromagram with note-name y-axis labels.

    Args:
        audio: AudioSamples object.
        stft_params: StftParams. Defaults used when None.
        chroma_params: ChromaParams. Defaults used when None.
        note_labels: Show C/C#/.../B labels on y-axis.
        title: Axes title.
        ax: Existing axes.
        style: Visual style.

    Returns:
        ``(fig, ax)``.
    """
    style = _resolve_style(style)
    fig, ax = _ensure_axes(ax, style)

    if stft_params is None:
        stft_params = StftParams(
            window=WindowType.hanning, hop_size=512, n_fft=2048, centre=True
        )
    if chroma_params is None:
        chroma_params = ChromaParams.music_standard()

    chroma = audio.chroma(stft_params, chroma_params)
    if chroma.ndim == 3:
        chroma = chroma[0]

    duration = audio.duration_seconds
    extent = [0, duration, 0, 12]
    im = ax.imshow(
        chroma,
        aspect="auto",
        origin="lower",
        extent=tuple(extent),
        cmap=style.colormap,
        vmin=0,
    )

    if style.show_colorbar:
        _add_colorbar(im, ax, label="Chroma energy")

    if note_labels:
        notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        ax.set_yticks(np.arange(12) + 0.5)
        ax.set_yticklabels(notes, fontsize=style.tick_size)
    else:
        ax.set_ylabel("Chroma", fontsize=style.label_size)

    ax.set_xlabel("Time (s)", fontsize=style.label_size)
    ax.set_title(title, fontsize=style.title_size)

    assert fig is not None  # for type checker
    assert not isinstance(fig, SubFigure)
    _apply_style(fig, ax, style)
    return fig, ax


def plot_gammatone_spectrogram(
    audio: AudioSamples,
    *,
    spec_params: SpectrogramParams | None = None,
    erb_params: ErbParams | None = None,
    log_params: LogParams | None = None,
    channel: int | None = None,
    title: str = "Gammatone Spectrogram",
    ax: "Axes | None" = None,
    style: "MplStyle | None" = None,
) -> "tuple[Figure, Axes]":
    """ERB-scale gammatone spectrogram.

    Args:
        audio: AudioSamples object.
        spec_params: SpectrogramParams. Defaults used when None.
        erb_params: ErbParams. Defaults used when None.
        log_params: Optional LogParams for dB conversion.
        channel: Channel index.
        title: Axes title.
        ax: Existing axes.
        style: Visual style.

    Returns:
        ``(fig, ax)``.
    """
    style = _resolve_style(style)
    fig, ax = _ensure_axes(ax, style)
    sr = audio.sample_rate
    if spec_params is None:
        stft = StftParams(
            window=WindowType.hanning, hop_size=512, n_fft=2048, centre=True
        )
        spec_params = SpectrogramParams(stft=stft, sample_rate=sr)
    if erb_params is None:
        erb_params = ErbParams(n_filters=64, f_min=50.0, f_max=sr / 2)

    gt = audio.gammatone_spectrogram(spec_params, erb_params, log_params)

    if gt.ndim == 3:
        ch = channel if channel is not None else 0
        gt = gt[ch]

    duration = audio.duration_seconds
    extent = [0, duration, erb_params.f_min, erb_params.f_max]
    im = ax.imshow(
        gt,
        aspect="auto",
        origin="lower",
        extent=tuple(extent),
        cmap=style.colormap,
        vmin=style.db_vmin if log_params is not None else None,
        vmax=style.db_vmax if log_params is not None else None,
    )

    if style.show_colorbar:
        label = style.colorbar_label if log_params is not None else "Magnitude"
        _add_colorbar(im, ax, label=label)

    ax.set_xlabel("Time (s)", fontsize=style.label_size)
    ax.set_ylabel("Frequency (Hz)", fontsize=style.label_size)
    ax.set_title(title, fontsize=style.title_size)

    assert fig is not None  # for type checker
    assert not isinstance(fig, SubFigure)
    _apply_style(fig, ax, style)
    return fig, ax
