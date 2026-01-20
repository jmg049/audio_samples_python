"""Composite multi-panel dashboard figures."""

from __future__ import annotations


import numpy as np

from audio_samples import AudioSamples, VadConfig

from ._utils import (
    _apply_style,
    _compute_stft,
    _resolve_style,
    _to_db,
)

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from .style import MplStyle


def plot_audio_overview(
    audio: AudioSamples,
    *,
    channel: int | None = None,
    title: str = "Audio Overview",
    style: MplStyle | None = None,
) -> tuple[Figure, dict[str, Axes]]:
    """Three-row overview: waveform+RMS, STFT spectrogram, magnitude spectrum.

    Args:
        audio: AudioSamples object.
        channel: Channel index.
        title: Figure suptitle.
        style: Visual style.

    Returns:
        ``(fig, axes_dict)`` with keys ``"waveform"``, ``"spectrogram"``,
        ``"spectrum"``.
    """
    import matplotlib.pyplot as plt

    style = _resolve_style(style)
    fig = plt.figure(figsize=(style.fig_width, style.fig_height * 3), dpi=style.dpi)
    gs = fig.add_gridspec(3, 1, hspace=0.45)
    ax_waveform = fig.add_subplot(gs[0])
    ax_spec = fig.add_subplot(gs[1])
    ax_spectrum = fig.add_subplot(gs[2])

    t = audio.time_axis()
    data = audio[channel]
    sr = audio.sample_rate
    duration = audio.duration_seconds

    # Row 1: Waveform + RMS
    ax_waveform.plot(
        t,
        data,
        color=style.waveform_color,
        linewidth=style.linewidth,
        alpha=style.alpha,
        label="Waveform",
    )
    window = max(1, int(0.02 * sr))
    hop = max(1, window // 4)
    rms = audio.rms_envelope(window, hop)
    if rms.ndim > 1:
        rms = rms[0]
    t_rms = np.linspace(0, t[-1], len(rms))
    ax_waveform.plot(
        t_rms,
        rms,
        color="orange",
        linewidth=style.linewidth * 0.8,
        alpha=0.8,
        label="RMS",
    )
    ax_waveform.set_ylabel("Amplitude", fontsize=style.label_size)
    ax_waveform.set_title("Waveform", fontsize=style.label_size)
    ax_waveform.legend(fontsize=style.legend_size)

    # Row 2: STFT spectrogram
    src = (
        audio.extract_channel(channel if channel is not None else 0)
        if not audio.is_mono()
        else audio
    )
    magnitude, freq_axis, _ = _compute_stft(src, n_fft=2048, hop_size=512)
    with np.errstate(divide="ignore", invalid="ignore"):
        db = 20.0 * np.log10(np.maximum(magnitude, 1e-12))
    extent = [0, duration, freq_axis[0], freq_axis[-1]]
    im = ax_spec.imshow(
        db,
        aspect="auto",
        origin="lower",
        extent=tuple(extent),
        cmap=style.colormap,
        vmin=style.db_vmin,
        vmax=style.db_vmax,
    )
    if style.show_colorbar:
        fig.colorbar(im, ax=ax_spec, label=style.colorbar_label)
    ax_spec.set_ylabel("Frequency (Hz)", fontsize=style.label_size)
    ax_spec.set_title("Spectrogram", fontsize=style.label_size)

    # Row 3: Magnitude spectrum (averaged across STFT frames)
    mag = np.mean(magnitude, axis=1)
    freqs = freq_axis
    db_spec = _to_db(mag, style.db_vmin)
    ax_spectrum.plot(
        freqs, db_spec, color=style.waveform_color, linewidth=style.linewidth
    )
    ax_spectrum.set_xlabel("Frequency (Hz)", fontsize=style.label_size)
    ax_spectrum.set_ylabel("Magnitude (dB)", fontsize=style.label_size)
    ax_spectrum.set_title("Spectrum", fontsize=style.label_size)

    fig.suptitle(title, fontsize=style.title_size)
    if style.show_grid:
        for ax in [ax_waveform, ax_spectrum]:
            ax.grid(True, alpha=style.grid_alpha)

    axes_dict: dict[str, Axes] = {
        "waveform": ax_waveform,
        "spectrogram": ax_spec,
        "spectrum": ax_spectrum,
    }
    _apply_style(fig, list(axes_dict.values()), style)
    return fig, axes_dict


def plot_speech_analysis(
    audio: AudioSamples,
    *,
    vad_config: VadConfig | None = None,
    channel: int | None = None,
    title: str = "Speech Analysis",
    style: MplStyle | None = None,
) -> tuple[Figure, dict[str, Axes]]:
    """Four-panel speech analysis: waveform+VAD, spectrogram, MFCC, VAD mask.

    Args:
        audio: AudioSamples object.
        vad_config: VadConfig. Uses default when None.
        channel: Channel index.
        title: Figure suptitle.
        style: Visual style.

    Returns:
        ``(fig, axes_dict)`` with keys ``"waveform"``, ``"spectrogram"``,
        ``"mfcc"``, ``"vad"``.
    """
    import matplotlib.pyplot as plt
    from audio_samples import VadConfig
    import spectrograms as sp

    style = _resolve_style(style)

    if vad_config is None:
        vad_config = VadConfig.default()

    sr = audio.sample_rate
    duration = audio.duration_seconds

    fig = plt.figure(figsize=(style.fig_width, style.fig_height * 4), dpi=style.dpi)
    gs = fig.add_gridspec(4, 1, hspace=0.5)
    ax_w = fig.add_subplot(gs[0])
    ax_s = fig.add_subplot(gs[1])
    ax_m = fig.add_subplot(gs[2])
    ax_v = fig.add_subplot(gs[3])

    t = audio.time_axis()
    data = audio[channel]

    # Waveform + VAD shading
    ax_w.plot(t, data, color=style.waveform_color, linewidth=style.linewidth)
    regions = audio.speech_regions(vad_config)
    for start, end in regions:
        ax_w.axvspan(start / sr, end / sr, alpha=0.25, color="#27ae60")
    ax_w.set_ylabel("Amplitude", fontsize=style.label_size)
    ax_w.set_title("Waveform (speech shaded)", fontsize=style.label_size)

    # Spectrogram
    src = (
        audio.extract_channel(channel if channel is not None else 0)
        if not audio.is_mono()
        else audio
    )
    magnitude, freq_axis, _ = _compute_stft(src, n_fft=2048, hop_size=512)
    with np.errstate(divide="ignore", invalid="ignore"):
        db_mat = 20.0 * np.log10(np.maximum(magnitude, 1e-12))
    extent = [0, duration, freq_axis[0], freq_axis[-1]]
    im_s = ax_s.imshow(
        db_mat,
        aspect="auto",
        origin="lower",
        extent=tuple(extent),
        cmap=style.colormap,
        vmin=style.db_vmin,
        vmax=style.db_vmax,
    )
    if style.show_colorbar:
        fig.colorbar(im_s, ax=ax_s, label=style.colorbar_label)
    ax_s.set_ylabel("Frequency (Hz)", fontsize=style.label_size)
    ax_s.set_title("Spectrogram", fontsize=style.label_size)

    # MFCC
    stft_p = sp.StftParams(
        window=sp.WindowType.hanning, hop_size=512, n_fft=2048, centre=True
    )
    mfcc = audio.mfcc(stft_p, 128, sp.MfccParams(n_mfcc=13))
    if mfcc.ndim == 3:
        mfcc = mfcc[0]
    extent = [0, duration, 0, mfcc.shape[0]]
    im_m = ax_m.imshow(
        mfcc,
        aspect="auto",
        origin="lower",
        extent=tuple(extent),
        cmap=style.heatmap_colormap,
    )
    if style.show_colorbar:
        fig.colorbar(im_m, ax=ax_m, label="Coefficient")
    ax_m.set_ylabel("MFCC", fontsize=style.label_size)
    ax_m.set_title("MFCC", fontsize=style.label_size)

    # VAD binary mask
    mask = audio.voice_activity_mask(vad_config)
    t_mask = np.linspace(0, t[-1], len(mask))
    ax_v.step(t_mask, mask, where="post", color="#27ae60", linewidth=style.linewidth)
    ax_v.set_ylim(-0.1, 1.1)
    ax_v.set_yticks([0, 1])
    ax_v.set_yticklabels(["Silence", "Speech"], fontsize=style.tick_size)
    ax_v.set_xlabel("Time (s)", fontsize=style.label_size)
    ax_v.set_title("VAD Mask", fontsize=style.label_size)

    fig.suptitle(title, fontsize=style.title_size)
    axes_dict: dict[str, Axes] = {
        "waveform": ax_w,
        "spectrogram": ax_s,
        "mfcc": ax_m,
        "vad": ax_v,
    }
    _apply_style(fig, list(axes_dict.values()), style)
    return fig, axes_dict


def plot_music_analysis(
    audio: AudioSamples,
    *,
    channel: int | None = None,
    title: str = "Music Analysis",
    style: MplStyle | None = None,
) -> tuple[Figure, dict[str, Axes]]:
    """Four-panel music analysis: waveform+beats+onsets, mel, chroma, pitch.

    Args:
        audio: AudioSamples object.
        channel: Channel index.
        title: Figure suptitle.
        style: Visual style.

    Returns:
        ``(fig, axes_dict)`` with keys ``"waveform"``, ``"mel"``,
        ``"chroma"``, ``"pitch"``.
    """
    import matplotlib.pyplot as plt
    from audio_samples import (
        BeatTrackingConfig,
        OnsetDetectionConfig,
        PitchDetectionMethod,
    )
    import spectrograms as sp

    style = _resolve_style(style)
    sr = audio.sample_rate
    duration = audio.duration_seconds

    fig = plt.figure(figsize=(style.fig_width, style.fig_height * 4), dpi=style.dpi)
    gs = fig.add_gridspec(4, 1, hspace=0.5)
    ax_w = fig.add_subplot(gs[0])
    ax_mel = fig.add_subplot(gs[1])
    ax_chr = fig.add_subplot(gs[2])
    ax_pit = fig.add_subplot(gs[3])

    t = audio.time_axis()
    data = audio[channel]

    # Waveform + beats + onsets
    ax_w.plot(
        t,
        data,
        color=style.waveform_color,
        linewidth=style.linewidth,
        alpha=style.alpha,
    )
    try:
        onset_cfg = OnsetDetectionConfig.default()
        onsets = audio.detect_onsets(onset_cfg)
        for ot in onsets:
            ax_w.axvline(ot, color="#e74c3c", linewidth=0.8, alpha=0.6)
        beat_cfg = BeatTrackingConfig(tempo_bpm=120.0, onset_config=onset_cfg)
        beat_data = audio.detect_beats(beat_cfg)
        for bt in beat_data.beat_times:
            ax_w.axvline(bt, color="#e67e22", linewidth=1.2, alpha=0.8)
    except Exception:
        pass
    ax_w.set_ylabel("Amplitude", fontsize=style.label_size)
    ax_w.set_title("Waveform (red=onsets, orange=beats)", fontsize=style.label_size)

    # Mel spectrogram
    stft_p = sp.StftParams(
        window=sp.WindowType.hanning, hop_size=512, n_fft=2048, centre=True
    )
    spec_params = sp.SpectrogramParams(stft=stft_p, sample_rate=sr)
    mel_params = sp.MelParams(n_mels=128, f_min=0.0, f_max=sr / 2)
    mel = audio.mel_spectrogram(spec_params, mel_params)
    if mel.ndim == 3:
        mel = mel[0]
    extent = [0, duration, 0, sr / 2]
    im_mel = ax_mel.imshow(
        mel,
        aspect="auto",
        origin="lower",
        extent=tuple(extent),
        cmap=style.colormap,
    )
    if style.show_colorbar:
        fig.colorbar(im_mel, ax=ax_mel, label="Magnitude")
    ax_mel.set_ylabel("Frequency (Hz)", fontsize=style.label_size)
    ax_mel.set_title("Mel Spectrogram", fontsize=style.label_size)

    # Chroma
    chroma = audio.chroma(stft_p, sp.ChromaParams.music_standard())
    if chroma.ndim == 3:
        chroma = chroma[0]
    extent = [0, duration, 0, 12]
    im_chr = ax_chr.imshow(
        chroma,
        aspect="auto",
        origin="lower",
        extent=tuple(extent),
        cmap=style.colormap,
        vmin=0,
    )
    if style.show_colorbar:
        fig.colorbar(im_chr, ax=ax_chr, label="Energy")
    notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    ax_chr.set_yticks(np.arange(12) + 0.5)
    ax_chr.set_yticklabels(notes, fontsize=style.tick_size)
    ax_chr.set_title("Chroma", fontsize=style.label_size)

    # Pitch track
    try:
        track = audio.track_pitch(2048, 512, PitchDetectionMethod.yin)
        times_p = np.asarray([pt[0] for pt in track])
        pitches_p = np.asarray(
            [pt[1] if pt[1] is not None else float("nan") for pt in track]
        )
        ax_pit.scatter(
            times_p,
            pitches_p,
            s=style.marker_size**2,
            color=style.waveform_color,
            alpha=0.8,
        )
    except Exception:
        pass
    ax_pit.set_xlabel("Time (s)", fontsize=style.label_size)
    ax_pit.set_ylabel("Frequency (Hz)", fontsize=style.label_size)
    ax_pit.set_title("Pitch Track", fontsize=style.label_size)

    fig.suptitle(title, fontsize=style.title_size)
    axes_dict: dict[str, Axes] = {
        "waveform": ax_w,
        "mel": ax_mel,
        "chroma": ax_chr,
        "pitch": ax_pit,
    }
    _apply_style(fig, list(axes_dict.values()), style)
    return fig, axes_dict


def plot_hpss_dashboard(
    audio: AudioSamples,
    *,
    win_size: int = 2048,
    hop_size: int = 512,
    title: str = "HPSS Dashboard",
    style: MplStyle | None = None,
) -> tuple[Figure, dict[str, Axes]]:
    """3x2 grid: waveforms (orig/harm/perc) on left, spectrograms on right.

    Args:
        audio: AudioSamples object.
        win_size: HPSS window size.
        hop_size: HPSS hop size.
        title: Figure suptitle.
        style: Visual style.

    Returns:
        ``(fig, axes_dict)`` with keys ``"waveform_orig"``,
        ``"waveform_harm"``, ``"waveform_perc"``,
        ``"spectrogram_orig"``, ``"spectrogram_harm"``,
        ``"spectrogram_perc"``.
    """
    import matplotlib.pyplot as plt
    import spectrograms as sp

    style = _resolve_style(style)
    stft_p = sp.StftParams(
        window=sp.WindowType.hanning, hop_size=hop_size, n_fft=win_size, centre=True
    )
    harmonic, percussive = audio.hpss(stft_p)
    signals = [audio, harmonic, percussive]
    row_labels = ["Original", "Harmonic", "Percussive"]
    colors = [style.waveform_color, "#27ae60", "#e74c3c"]
    duration = audio.duration_seconds

    fig, axes = plt.subplots(
        3,
        2,
        figsize=(style.fig_width * 1.5, style.fig_height * 3),
        dpi=style.dpi,
    )

    axes_dict: dict[str, Axes] = {}
    for row, (sig, lbl, col) in enumerate(zip(signals, row_labels, colors)):
        ax_w = axes[row, 0]
        ax_s = axes[row, 1]

        t = sig.time_axis()
        ax_w.plot(t, sig, color=col, linewidth=style.linewidth, alpha=style.alpha)
        ax_w.set_ylabel("Amplitude", fontsize=style.label_size)
        ax_w.set_title(f"{lbl} Waveform", fontsize=style.label_size)

        magnitude, freq_axis, _ = _compute_stft(
            sig.to_mono() if not sig.is_mono() else sig, n_fft=2048, hop_size=512
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            db_mat = 20.0 * np.log10(np.maximum(magnitude, 1e-12))
        im = ax_s.imshow(
            db_mat,
            aspect="auto",
            origin="lower",
            extent=[0, duration, freq_axis[0], freq_axis[-1]],
            cmap=style.colormap,
            vmin=style.db_vmin,
            vmax=style.db_vmax,
        )
        if style.show_colorbar:
            fig.colorbar(im, ax=ax_s, label=style.colorbar_label)
        ax_s.set_ylabel("Frequency (Hz)", fontsize=style.label_size)
        ax_s.set_title(f"{lbl} Spectrogram", fontsize=style.label_size)

        key_suffix = lbl.lower()
        axes_dict[f"waveform_{key_suffix}"] = ax_w
        axes_dict[f"spectrogram_{key_suffix}"] = ax_s

    for ax in axes[-1, :]:
        ax.set_xlabel("Time (s)", fontsize=style.label_size)

    fig.suptitle(title, fontsize=style.title_size)
    _apply_style(fig, list(axes_dict.values()), style)
    return fig, axes_dict


def plot_filter_analysis(
    original: AudioSamples,
    filtered: AudioSamples,
    *,
    n_fft: int = 2048,
    label_original: str = "Original",
    label_filtered: str = "Filtered",
    title: str = "Filter Analysis",
    style: MplStyle | None = None,
) -> tuple[Figure, dict[str, Axes]]:
    """2x2 grid: original waveform, filtered waveform, original spectrum, filtered spectrum.

    Args:
        original: Original AudioSamples.
        filtered: Filtered AudioSamples.
        n_fft: FFT size.
        label_original: Label for original.
        label_filtered: Label for filtered.
        title: Figure suptitle.
        style: Visual style.

    Returns:
        ``(fig, axes_dict)`` with keys ``"waveform_orig"``,
        ``"waveform_filt"``, ``"spectrum_orig"``, ``"spectrum_filt"``.
    """
    import matplotlib.pyplot as plt

    style = _resolve_style(style)
    colors = [style.waveform_color, "#e74c3c"]
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(style.fig_width * 1.4, style.fig_height * 2),
        dpi=style.dpi,
    )

    for col_idx, (sig, lbl, col) in enumerate(
        zip([original, filtered], [label_original, label_filtered], colors)
    ):
        t = sig.time_axis()
        axes[0, col_idx].plot(
            t, sig, color=col, linewidth=style.linewidth, alpha=style.alpha
        )
        axes[0, col_idx].set_ylabel("Amplitude", fontsize=style.label_size)
        axes[0, col_idx].set_xlabel("Time (s)", fontsize=style.label_size)
        axes[0, col_idx].set_title(f"{lbl} Waveform", fontsize=style.label_size)

        stft_mag, freqs, _ = _compute_stft(
            sig.to_mono() if not sig.is_mono() else sig,
            n_fft=n_fft,
            hop_size=n_fft // 4,
        )
        mag = np.mean(stft_mag, axis=1)
        db = _to_db(mag, style.db_vmin)
        axes[1, col_idx].plot(
            freqs, db, color=col, linewidth=style.linewidth, alpha=style.alpha
        )
        axes[1, col_idx].set_ylabel("Magnitude (dB)", fontsize=style.label_size)
        axes[1, col_idx].set_xlabel("Frequency (Hz)", fontsize=style.label_size)
        axes[1, col_idx].set_title(f"{lbl} Spectrum", fontsize=style.label_size)

    fig.suptitle(title, fontsize=style.title_size)
    axes_dict: dict[str, Axes] = {
        "waveform_orig": axes[0, 0],
        "waveform_filt": axes[0, 1],
        "spectrum_orig": axes[1, 0],
        "spectrum_filt": axes[1, 1],
    }
    _apply_style(fig, list(axes_dict.values()), style)
    return fig, axes_dict
