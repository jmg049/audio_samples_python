"""audio_samples.mpl — Matplotlib/seaborn visualization extension.

Provides publication-quality static figures (PDF/SVG/PNG at 300 DPI) with a
compositional API compatible with existing matplotlib workflows.

Style configuration is handled via the :class:`MplStyle` dataclass.  Use the
preset factories for common use cases::

    from audio_samples import mpl

    style = mpl.MplStyle.paper()
    fig, ax = mpl.plot_waveform(audio, style=style)

All plotting functions accept ``ax=<existing Axes>`` to draw into user-managed
figures, and ``style=None`` to use the process-wide default style set via
:func:`set_default_style`.
"""

from .style import MplStyle, get_default_style, set_default_style, style_context

from .waveform import (
    plot_waveform,
    plot_waveform_channels,
    plot_waveform_comparison,
    plot_zero_crossings,
)

from .spectral import (
    plot_magnitude_spectrum,
    plot_power_spectral_density,
    plot_spectrum_comparison,
)

from .timefreq import (
    plot_spectrogram,
    plot_mel_spectrogram,
    plot_mfcc,
    plot_chroma,
    plot_gammatone_spectrogram,
)

from .envelopes import (
    plot_amplitude_envelope,
    plot_rms_envelope,
    plot_analytic_envelope,
    plot_attack_decay_envelope,
    plot_moving_average_envelope,
    plot_envelope_comparison,
)

from .events import (
    plot_onsets,
    plot_beats,
    plot_beats_and_onsets,
    plot_spectral_flux,
    plot_onset_detection_function,
    plot_pitch_track,
    plot_pitch_histogram,
)

from .vad import (
    plot_vad_mask,
    plot_speech_regions,
    plot_vad_comparison,
)

from .harmonic import (
    plot_harmonic_analysis,
    plot_harmonics_bar,
    plot_autocorrelation,
    plot_cross_correlation,
)

from .comparison import (
    plot_waveform_overlay,
    plot_hpss_components,
    plot_snr_comparison,
    plot_signal_vs_filtered,
)

from .distributions import (
    plot_amplitude_histogram,
    plot_sample_distribution,
    plot_statistics_summary,
    plot_dynamic_range,
    plot_clipping_detection,
    plot_silence_regions,
)

from .dashboard import (
    plot_audio_overview,
    plot_speech_analysis,
    plot_music_analysis,
    plot_hpss_dashboard,
    plot_filter_analysis,
)


__all__ = [
    # Style
    "MplStyle",
    "set_default_style",
    "get_default_style",
    "style_context",
    # Waveform
    "plot_waveform",
    "plot_waveform_channels",
    "plot_waveform_comparison",
    "plot_zero_crossings",
    # Spectral
    "plot_magnitude_spectrum",
    "plot_power_spectral_density",
    "plot_spectrum_comparison",
    # Time-frequency
    "plot_spectrogram",
    "plot_mel_spectrogram",
    "plot_mfcc",
    "plot_chroma",
    "plot_gammatone_spectrogram",
    # Envelopes
    "plot_amplitude_envelope",
    "plot_rms_envelope",
    "plot_analytic_envelope",
    "plot_attack_decay_envelope",
    "plot_moving_average_envelope",
    "plot_envelope_comparison",
    # Events
    "plot_onsets",
    "plot_beats",
    "plot_beats_and_onsets",
    "plot_spectral_flux",
    "plot_onset_detection_function",
    "plot_pitch_track",
    "plot_pitch_histogram",
    # VAD
    "plot_vad_mask",
    "plot_speech_regions",
    "plot_vad_comparison",
    # Harmonic
    "plot_harmonic_analysis",
    "plot_harmonics_bar",
    "plot_autocorrelation",
    "plot_cross_correlation",
    # Comparison
    "plot_waveform_overlay",
    "plot_hpss_components",
    "plot_snr_comparison",
    "plot_signal_vs_filtered",
    # Distributions
    "plot_amplitude_histogram",
    "plot_sample_distribution",
    "plot_statistics_summary",
    "plot_dynamic_range",
    "plot_clipping_detection",
    "plot_silence_regions",
    # Dashboard
    "plot_audio_overview",
    "plot_speech_analysis",
    "plot_music_analysis",
    "plot_hpss_dashboard",
    "plot_filter_analysis",
]
