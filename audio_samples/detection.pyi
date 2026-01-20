"""
Type stubs for audio_samples.utils.detection.

Audio signal analysis and detection utilities.
"""

from __future__ import annotations

from typing import Iterable, Optional, TypeVar

from . import AudioSamples, VadConfig

TNum = TypeVar("TNum", int, float)

def detect_sample_rate(audio: AudioSamples) -> Optional[int]:
    """
    Attempt to infer the effective sample rate of an audio signal from its content.

    The function analyses spectral structure to detect aliasing or band-limiting
    artefacts that may indicate resampling or incorrect metadata.

    Parameters
    ----------
    audio
        Audio signal to analyse.

    Returns
    -------
    int | None
        Detected sample rate in Hz, or ``None`` if no reliable estimate can be made.
    """
    ...

def detect_fundamental_frequency(audio: AudioSamples) -> Optional[float]:
    """
    Detect the fundamental frequency of an audio signal using spectral analysis.

    Parameters
    ----------
    audio
        Audio signal to analyse.

    Returns
    -------
    float | None
        Estimated fundamental frequency in Hz, or ``None`` if no dominant
        fundamental is detected.
    """
    ...

def detect_silence_regions(
    audio: AudioSamples,
    threshold: TNum,
) -> list[tuple[float, float]]:
    """
    Detect contiguous silence regions in an audio signal.

    A region is considered silent when the signal amplitude remains below the
    given threshold.

    Parameters
    ----------
    audio
        Audio signal to analyse.
    threshold
        Amplitude threshold below which samples are considered silent.

    Returns
    -------
    list[tuple[float, float]]
        List of ``(start_time, end_time)`` tuples in seconds.
    """
    ...

def detect_voice_activity_mask(
    audio: AudioSamples,
    config: VadConfig,
) -> list[bool]:
    """
    Compute a per-frame voice activity mask using the configured VAD.

    This is a convenience wrapper around
    ``AudioVoiceActivityDetection.voice_activity_mask``.

    Returns one boolean decision per analysis frame.
    """
    ...

def detect_speech_regions(
    audio: AudioSamples,
    config: VadConfig,
) -> list[tuple[int, int]]:
    """
    Compute contiguous speech regions as ``(start_sample, end_sample)`` pairs.

    This is a convenience wrapper around
    ``AudioVoiceActivityDetection.speech_regions``.
    ``end_sample`` is exclusive.
    """
    ...

def detect_dynamic_range(audio: AudioSamples) -> tuple[float, float, float]:
    """
    Estimate the dynamic range of an audio signal.

    Parameters
    ----------
    audio
        Audio signal to analyse.

    Returns
    -------
    tuple[float, float, float]
        ``(peak_amplitude, rms_amplitude, dynamic_range_db)``.
    """
    ...

def detect_clipping(
    audio: AudioSamples,
    threshold: float,
) -> list[tuple[float, float]]:
    """
    Detect clipped regions in an audio signal.

    A region is considered clipped when the signal remains close to the extrema
    defined by the threshold.

    Parameters
    ----------
    audio
        Audio signal to analyse.
    threshold
        Absolute or normalised amplitude threshold defining clipping.

    Returns
    -------
    list[tuple[float, float]]
        List of ``(start_time, end_time)`` tuples in seconds.
    """
    ...

def estimate_noise_floor(audio: AudioSamples) -> Optional[float]:
    """
    Estimate the noise floor of an audio signal.

    Returns
    -------
    float | None
        Estimated noise floor level, or ``None`` if it cannot be reliably inferred.
    """
    ...

def estimate_frequency_range(
    audio: AudioSamples,
) -> Optional[tuple[float, float]]:
    """
    Estimate the effective frequency response range of the signal.

    Returns
    -------
    tuple[float, float] | None
        ``(low_freq, high_freq)`` in Hz, or ``None`` if undefined.
    """
    ...

def compute_spectrum(audio: AudioSamples) -> list[float]:
    """
    Compute the magnitude spectrum of the audio signal using an FFT.
    """
    ...

def analyze_spectrum_for_cutoff(
    spectrum: Iterable[float],
    nyquist_freq: float,
) -> Optional[int]:
    """
    Analyse a frequency spectrum for potential cutoff frequencies that may
    indicate resampling or bandwidth limitation.

    Returns
    -------
    int | None
        Index (or bin) of the detected cutoff, or ``None`` if no clear cutoff
        is detected.
    """
    ...

def detect_fundamental_autocorrelation(
    data: Iterable[TNum],
    sample_rate: float,
) -> Optional[float]:
    """
    Detect the fundamental frequency using autocorrelation analysis.

    Parameters
    ----------
    data
        Time-domain samples.
    sample_rate
        Sampling rate in Hz.

    Returns
    -------
    float | None
        Estimated fundamental frequency in Hz, or ``None`` if undetectable.
    """
    ...
