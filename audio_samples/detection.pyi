"""
Type stubs for audio_samples.utils.detection.

Audio signal analysis and detection utilities.
"""

from __future__ import annotations

from typing import Optional

import numpy

from . import AudioSamples

def detect_sample_rate(audio: AudioSamples) -> Optional[int]:
    """
    Heuristically detect the original sample rate of a signal from its spectral content.

    Analyses the power spectrum for sharp high-frequency cutoffs characteristic of
    anti-aliasing filters and matches them against common sample rates. Only the first
    channel is used. Returns ``None`` when no candidate rate can be identified.

    Parameters
    ----------
    audio
        The audio signal to analyse.

    Returns
    -------
    int | None
        The detected original sample rate in Hz, or ``None``.
    """
    ...

def detect_fundamental_frequency(audio: AudioSamples) -> Optional[float]:
    """
    Estimate the fundamental frequency of a signal using autocorrelation.

    Parameters
    ----------
    audio
        The audio signal to analyse.

    Returns
    -------
    float | None
        The estimated fundamental frequency in Hz, or ``None`` if no periodic
        component is found.
    """
    ...

def detect_silence_regions(
    audio: AudioSamples,
    threshold: float,
) -> list[tuple[float, float]]:
    """
    Detect time intervals where the signal amplitude falls below a threshold.

    For mono signals each sample is checked directly. For multi-channel signals a
    position is considered silent only when **all** channels are below the threshold.

    Parameters
    ----------
    audio
        The audio signal to analyse.
    threshold
        Amplitude threshold in the signal's native sample scale (e.g. 0..1 for float
        audio). Samples with absolute value below this are considered silent.

    Returns
    -------
    list[tuple[float, float]]
        A list of ``(start_time, end_time)`` tuples in seconds, one per silent region.
    """
    ...

def detect_clipping(
    audio: AudioSamples,
    threshold_ratio: float = 0.99,
) -> list[tuple[float, float]]:
    """
    Detect time intervals where the signal reaches or exceeds the full-scale value.

    A sample is considered clipped when it reaches or exceeds ``threshold_ratio`` of the
    sample type's positive full scale, or falls at or below ``threshold_ratio`` of its
    negative full scale. For multi-channel signals a position is clipped when **any**
    channel is clipped.

    Parameters
    ----------
    audio
        The audio signal to analyse.
    threshold_ratio
        Fraction of full scale that constitutes clipping, in (0, 1] (default: 0.99).

    Returns
    -------
    list[tuple[float, float]]
        A list of ``(start_time, end_time)`` tuples in seconds, one per clipped region.
    """
    ...

def detect_dynamic_range(audio: AudioSamples) -> tuple[float, float, float]:
    """
    Compute the dynamic-range characteristics of a signal.

    All samples across all channels are considered together.

    Parameters
    ----------
    audio
        The audio signal to analyse.

    Returns
    -------
    tuple[float, float, float]
        A tuple ``(peak_amplitude, rms_amplitude, dynamic_range_db)`` where the dynamic
        range is the crest factor ``20 * log10(peak / rms)`` in decibels (0.0 when rms
        is 0).
    """
    ...

def estimate_noise_floor(audio: AudioSamples) -> Optional[float]:
    """
    Estimate the noise floor of a signal in dBFS.

    Computes the noise floor from the quietest 10th percentile of sample magnitudes.
    Only the first channel is used for multi-channel signals.

    Parameters
    ----------
    audio
        The audio signal to analyse.

    Returns
    -------
    float | None
        The estimated noise floor in dBFS (always below 0), or ``None`` if it cannot be
        estimated (e.g. the signal is too short or entirely silent).
    """
    ...

def estimate_frequency_range(
    audio: AudioSamples,
) -> Optional[tuple[float, float]]:
    """
    Estimate the active frequency range of a signal.

    Computes the power spectrum of the first channel and returns the lowest and highest
    frequencies carrying more than 1% of the peak spectral energy.

    Parameters
    ----------
    audio
        The audio signal to analyse.

    Returns
    -------
    tuple[float, float] | None
        A ``(low_hz, high_hz)`` tuple, or ``None`` when the signal is shorter than 1024
        samples or no bin exceeds the threshold.
    """
    ...

def analyze_spectrum_for_cutoff(
    spectrum: numpy.ndarray,
    nyquist_freq: float,
) -> Optional[int]:
    """
    Analyse a power spectrum for a spectral cutoff indicating prior resampling.

    Checks candidate Nyquist frequencies (derived from common sample rates) for a 2x or
    greater energy drop, and returns the first (lowest-frequency) matching sample rate.

    Parameters
    ----------
    spectrum
        A non-empty 1-D power spectrum (FFT magnitude-squared) as a NumPy float array.
        Only the lower half of the bins is examined.
    nyquist_freq
        The Nyquist frequency of the audio that produced ``spectrum`` in Hz.

    Returns
    -------
    int | None
        The first candidate sample rate in Hz with a significant energy drop, or
        ``None``.

    Raises
    ------
    ValueError
        If ``spectrum`` is empty or ``nyquist_freq`` is non-finite.
    """
    ...

def detect_fundamental_autocorrelation(
    data: numpy.ndarray,
    sample_rate: float,
) -> Optional[float]:
    """
    Estimate the fundamental frequency of a raw sample buffer using autocorrelation.

    Searches candidate periods corresponding to fundamentals in the range 50..2000 Hz.

    Parameters
    ----------
    data
        A non-empty 1-D array of mono f64 samples.
    sample_rate
        The sample rate in Hz (must be finite and positive).

    Returns
    -------
    float | None
        The estimated fundamental frequency in Hz, or ``None`` if no periodic component
        is found or the signal is too short.

    Raises
    ------
    ValueError
        If ``data`` is empty or ``sample_rate`` is not finite and positive.
    """
    ...
