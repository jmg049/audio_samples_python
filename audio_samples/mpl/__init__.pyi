"""Type stubs for audio_samples.mpl."""

from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Any, Optional

from matplotlib.axes import Axes
from matplotlib.figure import Figure

from audio_samples import AudioSamples
from audio_samples.spectrograms import (
    ChromaParams,
    CqtParams,
    ErbParams,
    LogParams,
    MelParams,
    MfccParams,
    SpectrogramParams,
    StftParams,
)
from audio_samples.types import (
    BeatTrackingData,
    DynamicRangeMethod,
    EnvelopeFollower,
    OnsetDetectionConfig,
    PitchDetectionMethod,
    SpectralFluxMethod,
    VadConfig,
)

from .style import MplStyle as MplStyle
from .display import (
    plot_matrix as plot_matrix,
    plot_waveform_adaptive as plot_waveform_adaptive,
    plot_waveform_bars as plot_waveform_bars,
    plot_waveform_f0 as plot_waveform_f0,
    plot_grid as plot_grid,
    infer_colormap as infer_colormap,
    add_db_colorbar as add_db_colorbar,
    add_phase_colorbar as add_phase_colorbar,
    legend_for_axes as legend_for_axes,
    highlight_effects as highlight_effects,
    AdaptiveWaveform as AdaptiveWaveform,
    F0Transform as F0Transform,
    TimeFormatter as TimeFormatter,
    NoteFormatter as NoteFormatter,
    SvaraFormatter as SvaraFormatter,
    FJSFormatter as FJSFormatter,
    LogHzFormatter as LogHzFormatter,
    EngFormatter as EngFormatter,
    ChromaFormatter as ChromaFormatter,
    ChromaSvaraFormatter as ChromaSvaraFormatter,
    ChromaFJSFormatter as ChromaFJSFormatter,
    TonnetzFormatter as TonnetzFormatter,
    AXIS_TYPES as AXIS_TYPES,
)

# ---------------------------------------------------------------------------
# Style helpers
# ---------------------------------------------------------------------------

def set_default_style(style: MplStyle) -> None: ...
def get_default_style() -> MplStyle: ...
def style_context(style: MplStyle) -> AbstractContextManager[None]: ...

# ---------------------------------------------------------------------------
# Waveform
# ---------------------------------------------------------------------------

def plot_waveform(
    audio: AudioSamples,
    *,
    channel: Optional[int] = None,
    show_rms: bool = False,
    show_envelope: bool = False,
    title: str = "Waveform",
    ax: Optional[Axes] = None,
    style: Optional[MplStyle] = None,
) -> tuple[Figure, Axes]: ...
def plot_waveform_channels(
    audio: AudioSamples,
    *,
    share_y: bool = True,
    title: str = "Channels",
    style: Optional[MplStyle] = None,
) -> tuple[Figure, list[Axes]]: ...
def plot_waveform_comparison(
    signals: list[AudioSamples],
    labels: list[str],
    *,
    share_y: bool = True,
    title: str = "Waveform Comparison",
    style: Optional[MplStyle] = None,
) -> tuple[Figure, list[Axes]]: ...
def plot_zero_crossings(
    audio: AudioSamples,
    *,
    channel: Optional[int] = None,
    window_seconds: Optional[float] = None,
    title: str = "Zero Crossings",
    ax: Optional[Axes] = None,
    style: Optional[MplStyle] = None,
) -> tuple[Figure, Axes]: ...

# ---------------------------------------------------------------------------
# Spectral
# ---------------------------------------------------------------------------

def plot_magnitude_spectrum(
    audio: AudioSamples,
    *,
    n_fft: int = 2048,
    channel: Optional[int] = None,
    db_scale: bool = True,
    freq_max: Optional[float] = None,
    log_freq: bool = False,
    show_centroid: bool = True,
    title: str = "Magnitude Spectrum",
    ax: Optional[Axes] = None,
    style: Optional[MplStyle] = None,
) -> tuple[Figure, Axes]: ...
def plot_power_spectral_density(
    audio: AudioSamples,
    *,
    window_size: int = 2048,
    overlap: float = 0.5,
    channel: Optional[int] = None,
    freq_max: Optional[float] = None,
    title: str = "Power Spectral Density",
    ax: Optional[Axes] = None,
    style: Optional[MplStyle] = None,
) -> tuple[Figure, Axes]: ...
def plot_spectrum_comparison(
    signals: list[AudioSamples],
    labels: list[str],
    *,
    n_fft: int = 2048,
    db_scale: bool = True,
    freq_max: Optional[float] = None,
    fill_under: bool = True,
    title: str = "Spectrum Comparison",
    ax: Optional[Axes] = None,
    style: Optional[MplStyle] = None,
) -> tuple[Figure, Axes]: ...

# ---------------------------------------------------------------------------
# Time-frequency
# ---------------------------------------------------------------------------

def plot_spectrogram(
    audio: AudioSamples,
    *,
    stft_params: StftParams | None = None,
    channel: Optional[int] = None,
    db_range: Optional[tuple[float, float]] = None,
    freq_max: Optional[float] = None,
    title: str = "Spectrogram",
    ax: Optional[Axes] = None,
    style: Optional[MplStyle] = None,
) -> tuple[Figure, Axes]: ...
def plot_mel_spectrogram(
    audio: AudioSamples,
    *,
    spec_params: SpectrogramParams | None = None,
    mel_params: MelParams | None = None,
    log_params: LogParams | None = None,
    channel: Optional[int] = None,
    title: str = "Mel Spectrogram",
    ax: Optional[Axes] = None,
    style: Optional[MplStyle] = None,
) -> tuple[Figure, Axes]: ...
def plot_mfcc(
    audio: AudioSamples,
    *,
    stft_params: StftParams | None = None,
    n_mels: int = 128,
    mfcc_params: MfccParams | None = None,
    show_delta: bool = False,
    show_delta2: bool = False,
    title: str = "MFCC",
    style: Optional[MplStyle] = None,
) -> tuple[Figure, list[Axes]]: ...
def plot_chroma(
    audio: AudioSamples,
    *,
    stft_params: StftParams | None = None,
    chroma_params: ChromaParams | None = None,
    note_labels: bool = True,
    title: str = "Chroma",
    ax: Optional[Axes] = None,
    style: Optional[MplStyle] = None,
) -> tuple[Figure, Axes]: ...
def plot_gammatone_spectrogram(
    audio: AudioSamples,
    *,
    spec_params: SpectrogramParams | None = None,
    erb_params: ErbParams | None = None,
    log_params: LogParams | None = None,
    channel: Optional[int] = None,
    title: str = "Gammatone Spectrogram",
    ax: Optional[Axes] = None,
    style: Optional[MplStyle] = None,
) -> tuple[Figure, Axes]: ...

# ---------------------------------------------------------------------------
# Envelopes
# ---------------------------------------------------------------------------

def plot_amplitude_envelope(
    audio: AudioSamples,
    *,
    channel: Optional[int] = None,
    show_waveform: bool = True,
    title: str = "Amplitude Envelope",
    ax: Optional[Axes] = None,
    style: Optional[MplStyle] = None,
) -> tuple[Figure, Axes]: ...
def plot_rms_envelope(
    audio: AudioSamples,
    *,
    window_size: int = 2048,
    hop_size: int = 512,
    db_scale: bool = False,
    show_waveform: bool = True,
    channel: Optional[int] = None,
    title: str = "RMS Envelope",
    ax: Optional[Axes] = None,
    style: Optional[MplStyle] = None,
) -> tuple[Figure, Axes]: ...
def plot_analytic_envelope(
    audio: AudioSamples,
    *,
    channel: Optional[int] = None,
    show_waveform: bool = True,
    title: str = "Analytic Envelope",
    ax: Optional[Axes] = None,
    style: Optional[MplStyle] = None,
) -> tuple[Figure, Axes]: ...
def plot_attack_decay_envelope(
    audio: AudioSamples,
    *,
    follower: EnvelopeFollower | None = None,
    method: DynamicRangeMethod | None = None,
    channel: Optional[int] = None,
    title: str = "Attack / Decay Envelope",
    ax: Optional[Axes] = None,
    style: Optional[MplStyle] = None,
) -> tuple[Figure, Axes]: ...
def plot_moving_average_envelope(
    audio: AudioSamples,
    *,
    window_size: int = 2048,
    hop_size: int = 512,
    channel: Optional[int] = None,
    show_waveform: bool = True,
    title: str = "Moving Average Envelope",
    ax: Optional[Axes] = None,
    style: Optional[MplStyle] = None,
) -> tuple[Figure, Axes]: ...
def plot_envelope_comparison(
    audio: AudioSamples,
    *,
    channel: Optional[int] = None,
    window_size: int = 2048,
    hop_size: int = 512,
    title: str = "Envelope Comparison",
    ax: Optional[Axes] = None,
    style: Optional[MplStyle] = None,
) -> tuple[Figure, Axes]: ...

# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------

def plot_onsets(
    audio: AudioSamples,
    *,
    config: OnsetDetectionConfig | None = None,
    channel: Optional[int] = None,
    show_odf: bool = False,
    title: str = "Onset Detection",
    style: Optional[MplStyle] = None,
) -> tuple[Figure, list[Axes]]: ...
def plot_beats(
    audio: AudioSamples,
    *,
    beat_data: BeatTrackingData | None = None,
    channel: Optional[int] = None,
    show_tempo: bool = True,
    title: str = "Beat Tracking",
    ax: Optional[Axes] = None,
    style: Optional[MplStyle] = None,
) -> tuple[Figure, Axes]: ...
def plot_beats_and_onsets(
    audio: AudioSamples,
    *,
    onset_config: OnsetDetectionConfig | None = None,
    beat_data: BeatTrackingData | None = None,
    channel: Optional[int] = None,
    title: str = "Beats & Onsets",
    ax: Optional[Axes] = None,
    style: Optional[MplStyle] = None,
) -> tuple[Figure, Axes]: ...
def plot_spectral_flux(
    audio: AudioSamples,
    *,
    cqt_params: CqtParams | None = None,
    window_size: int = 2048,
    hop_size: int = 512,
    method: SpectralFluxMethod | None = None,
    title: str = "Spectral Flux",
    ax: Optional[Axes] = None,
    style: Optional[MplStyle] = None,
) -> tuple[Figure, Axes]: ...
def plot_onset_detection_function(
    audio: AudioSamples,
    *,
    config: OnsetDetectionConfig | None = None,
    show_peaks: bool = True,
    threshold: Optional[float] = None,
    title: str = "Onset Detection Function",
    ax: Optional[Axes] = None,
    style: Optional[MplStyle] = None,
) -> tuple[Figure, Axes]: ...
def plot_pitch_track(
    audio: AudioSamples,
    *,
    window_size: int = 2048,
    hop_size: int = 512,
    pitch_method: PitchDetectionMethod | None = None,
    show_notes: bool = False,
    show_waveform: bool = False,
    title: str = "Pitch Track",
    style: Optional[MplStyle] = None,
) -> tuple[Figure, list[Axes]]: ...
def plot_pitch_histogram(
    audio: AudioSamples,
    *,
    window_size: int = 2048,
    hop_size: int = 512,
    pitch_method: PitchDetectionMethod | None = None,
    bins: int = 50,
    title: str = "Pitch Histogram",
    ax: Optional[Axes] = None,
    style: Optional[MplStyle] = None,
) -> tuple[Figure, Axes]: ...

# ---------------------------------------------------------------------------
# VAD
# ---------------------------------------------------------------------------

def plot_vad_mask(
    audio: AudioSamples,
    *,
    config: VadConfig | None = None,
    channel: Optional[int] = None,
    title: str = "VAD Mask",
    style: Optional[MplStyle] = None,
) -> tuple[Figure, list[Axes]]: ...
def plot_speech_regions(
    audio: AudioSamples,
    *,
    config: VadConfig | None = None,
    channel: Optional[int] = None,
    title: str = "Speech Regions",
    ax: Optional[Axes] = None,
    style: Optional[MplStyle] = None,
) -> tuple[Figure, Axes]: ...
def plot_vad_comparison(
    audio: AudioSamples,
    configs: list[Any],
    labels: list[str],
    *,
    channel: Optional[int] = None,
    title: str = "VAD Comparison",
    style: Optional[MplStyle] = None,
) -> tuple[Figure, list[Axes]]: ...

# ---------------------------------------------------------------------------
# Harmonic
# ---------------------------------------------------------------------------

def plot_harmonic_analysis(
    audio: AudioSamples,
    *,
    fundamental_freq: float,
    num_harmonics: int = 8,
    tolerance: float = 0.05,
    n_fft: int = 4096,
    channel: Optional[int] = None,
    title: str = "Harmonic Analysis",
    ax: Optional[Axes] = None,
    style: Optional[MplStyle] = None,
) -> tuple[Figure, Axes]: ...
def plot_harmonics_bar(
    audio: AudioSamples,
    *,
    fundamental_freq: float,
    num_harmonics: int = 8,
    tolerance: float = 0.05,
    db_scale: bool = True,
    n_fft: int = 4096,
    title: str = "Harmonic Magnitudes",
    ax: Optional[Axes] = None,
    style: Optional[MplStyle] = None,
) -> tuple[Figure, Axes]: ...
def plot_autocorrelation(
    audio: AudioSamples,
    *,
    max_lag: int = 4096,
    title: str = "Autocorrelation",
    ax: Optional[Axes] = None,
    style: Optional[MplStyle] = None,
) -> tuple[Figure, Axes]: ...
def plot_cross_correlation(
    audio: AudioSamples,
    other: AudioSamples,
    *,
    max_lag: int = 4096,
    title: str = "Cross-Correlation",
    ax: Optional[Axes] = None,
    style: Optional[MplStyle] = None,
) -> tuple[Figure, Axes]: ...

# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def plot_waveform_overlay(
    signals: list[AudioSamples],
    labels: list[str],
    *,
    channel: Optional[int] = None,
    alpha: float = 0.7,
    title: str = "Waveform Overlay",
    ax: Optional[Axes] = None,
    style: Optional[MplStyle] = None,
) -> tuple[Figure, Axes]: ...
def plot_hpss_components(
    audio: AudioSamples,
    *,
    win_size: int = 2048,
    hop_size: int = 512,
    title: str = "HPSS Components",
    style: Optional[MplStyle] = None,
) -> tuple[Figure, list[Axes]]: ...
def plot_snr_comparison(
    signal: AudioSamples,
    noise: AudioSamples,
    *,
    title: str = "SNR Comparison",
    style: Optional[MplStyle] = None,
) -> tuple[Figure, list[Axes]]: ...
def plot_signal_vs_filtered(
    original: AudioSamples,
    filtered: AudioSamples,
    *,
    n_fft: int = 2048,
    label_original: str = "Original",
    label_filtered: str = "Filtered",
    title: str = "Signal vs Filtered",
    style: Optional[MplStyle] = None,
) -> tuple[Figure, list[Axes]]: ...

# ---------------------------------------------------------------------------
# Distributions
# ---------------------------------------------------------------------------

def plot_amplitude_histogram(
    audio: AudioSamples,
    *,
    channel: Optional[int] = None,
    bins: int = 100,
    kde: bool = True,
    log_y: bool = False,
    title: str = "Amplitude Distribution",
    ax: Optional[Axes] = None,
    style: Optional[MplStyle] = None,
) -> tuple[Figure, Axes]: ...
def plot_sample_distribution(
    audio: AudioSamples,
    *,
    plot_type: str = "violin",
    title: str = "Sample Distribution by Channel",
    ax: Optional[Axes] = None,
    style: Optional[MplStyle] = None,
) -> tuple[Figure, Axes]: ...
def plot_statistics_summary(
    audio: AudioSamples,
    *,
    title: str = "Audio Statistics",
    ax: Optional[Axes] = None,
    style: Optional[MplStyle] = None,
) -> tuple[Figure, Axes]: ...
def plot_dynamic_range(
    audio: AudioSamples,
    *,
    channel: Optional[int] = None,
    title: str = "Dynamic Range",
    ax: Optional[Axes] = None,
    style: Optional[MplStyle] = None,
) -> tuple[Figure, Axes]: ...
def plot_clipping_detection(
    audio: AudioSamples,
    *,
    threshold: float = 0.99,
    channel: Optional[int] = None,
    title: str = "Clipping Detection",
    ax: Optional[Axes] = None,
    style: Optional[MplStyle] = None,
) -> tuple[Figure, Axes]: ...
def plot_silence_regions(
    audio: AudioSamples,
    *,
    threshold: float = 0.01,
    channel: Optional[int] = None,
    title: str = "Silence Regions",
    ax: Optional[Axes] = None,
    style: Optional[MplStyle] = None,
) -> tuple[Figure, Axes]: ...

# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

def plot_audio_overview(
    audio: AudioSamples,
    *,
    channel: Optional[int] = None,
    title: str = "Audio Overview",
    style: Optional[MplStyle] = None,
) -> tuple[Figure, dict[str, Axes]]: ...
def plot_speech_analysis(
    audio: AudioSamples,
    *,
    vad_config: VadConfig | None = None,
    channel: Optional[int] = None,
    title: str = "Speech Analysis",
    style: Optional[MplStyle] = None,
) -> tuple[Figure, dict[str, Axes]]: ...
def plot_music_analysis(
    audio: AudioSamples,
    *,
    channel: Optional[int] = None,
    title: str = "Music Analysis",
    style: Optional[MplStyle] = None,
) -> tuple[Figure, dict[str, Axes]]: ...
def plot_hpss_dashboard(
    audio: AudioSamples,
    *,
    win_size: int = 2048,
    hop_size: int = 512,
    title: str = "HPSS Dashboard",
    style: Optional[MplStyle] = None,
) -> tuple[Figure, dict[str, Axes]]: ...
def plot_filter_analysis(
    original: AudioSamples,
    filtered: AudioSamples,
    *,
    n_fft: int = 2048,
    label_original: str = "Original",
    label_filtered: str = "Filtered",
    title: str = "Filter Analysis",
    style: Optional[MplStyle] = None,
) -> tuple[Figure, dict[str, Axes]]: ...

__all__ = [
    "plot_matrix",
    "plot_waveform_adaptive",
    "plot_waveform_bars",
    "plot_waveform_f0",
    "plot_grid",
    "infer_colormap",
    "add_db_colorbar",
    "add_phase_colorbar",
    "legend_for_axes",
    "highlight_effects",
    "AdaptiveWaveform",
    "F0Transform",
    "TimeFormatter",
    "NoteFormatter",
    "SvaraFormatter",
    "FJSFormatter",
    "LogHzFormatter",
    "EngFormatter",
    "ChromaFormatter",
    "ChromaSvaraFormatter",
    "ChromaFJSFormatter",
    "TonnetzFormatter",
    "AXIS_TYPES",

    "MplStyle",
    "get_default_style",
    "plot_amplitude_envelope",
    "plot_amplitude_histogram",
    "plot_analytic_envelope",
    "plot_attack_decay_envelope",
    "plot_audio_overview",
    "plot_autocorrelation",
    "plot_beats",
    "plot_beats_and_onsets",
    "plot_chroma",
    "plot_clipping_detection",
    "plot_cross_correlation",
    "plot_dynamic_range",
    "plot_envelope_comparison",
    "plot_filter_analysis",
    "plot_gammatone_spectrogram",
    "plot_harmonic_analysis",
    "plot_harmonics_bar",
    "plot_hpss_components",
    "plot_hpss_dashboard",
    "plot_magnitude_spectrum",
    "plot_mel_spectrogram",
    "plot_mfcc",
    "plot_moving_average_envelope",
    "plot_music_analysis",
    "plot_onset_detection_function",
    "plot_onsets",
    "plot_pitch_histogram",
    "plot_pitch_track",
    "plot_power_spectral_density",
    "plot_rms_envelope",
    "plot_sample_distribution",
    "plot_signal_vs_filtered",
    "plot_silence_regions",
    "plot_snr_comparison",
    "plot_spectral_flux",
    "plot_spectrogram",
    "plot_spectrum_comparison",
    "plot_speech_analysis",
    "plot_speech_regions",
    "plot_statistics_summary",
    "plot_vad_comparison",
    "plot_vad_mask",
    "plot_waveform",
    "plot_waveform_channels",
    "plot_waveform_comparison",
    "plot_waveform_overlay",
    "plot_zero_crossings",
    "set_default_style",
    "style_context",
]
