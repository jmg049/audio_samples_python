"""Utility functions for audio signal processing.

This module provides comparison, audio math, and other utility functions.
"""

from typing import Tuple
import numpy as np
from . import AudioSamples

# =============================================================================
# AUDIO COMPARISON FUNCTIONS
# =============================================================================

def correlation(a: AudioSamples, b: AudioSamples) -> float:
    """Compute Pearson correlation coefficient between two audio signals.

    This function measures linear similarity between two signals on a per-sample
    basis. Returns a scalar correlation coefficient in the range [-1, 1].

    For mono signals, the correlation is computed directly over the single channel.
    For multi-channel signals, the correlation is computed independently for each
    channel and the results are averaged.

    Args:
        a: The first audio signal.
        b: The second audio signal.

    Returns:
        A scalar correlation coefficient in the range [-1, 1].

    Raises:
        ValueError: If the signals have different dimensions or channel configurations.
        TypeError: If the signals have different data types.

    Example:
        >>> from audio_samples import AudioSamples
        >>> from audio_samples.utils import correlation
        >>> audio1 = AudioSamples.zeros_mono_f32(44100, 44100)
        >>> corr = correlation(audio1, audio1)  # Returns 1.0 for identical signals
    """
    ...

def mse(a: AudioSamples, b: AudioSamples) -> float:
    """Compute mean squared error (MSE) between two audio signals.

    This function measures the average squared per-sample difference between two
    signals. Lower values indicate higher similarity, with 0.0 indicating identical
    signals.

    For mono signals, the MSE is computed directly over the single channel.
    For multi-channel signals, the MSE is computed independently for each channel
    and the results are averaged.

    Args:
        a: The first audio signal.
        b: The second audio signal.

    Returns:
        The mean squared error as a non-negative scalar value.

    Raises:
        ValueError: If the signals have different dimensions or channel configurations.
        TypeError: If the signals have different data types.

    Example:
        >>> from audio_samples import AudioSamples
        >>> from audio_samples.utils import mse
        >>> audio1 = AudioSamples.zeros_mono_f32(44100, 44100)
        >>> error = mse(audio1, audio1)  # Returns 0.0 for identical signals
    """
    ...

def snr(signal: AudioSamples, noise: AudioSamples) -> float:
    """Compute signal-to-noise ratio (SNR) in decibels.

    This function measures the ratio between the average power of a signal and the
    average power of a noise signal, expressed in decibels. Higher values indicate
    greater dominance of the signal relative to noise.

    For mono inputs, power is computed over the single channel.
    For multi-channel inputs, power is computed over all samples across all channels.

    Args:
        signal: The signal component.
        noise: The noise component.

    Returns:
        The signal-to-noise ratio in decibels. Returns positive infinity if noise
        power is zero.

    Raises:
        ValueError: If the signals have different dimensions or channel configurations.
        TypeError: If the signals have different data types.

    Example:
        >>> from audio_samples import AudioSamples
        >>> from audio_samples.utils import snr
        >>> signal = AudioSamples.from_mono_f32([1.0, 2.0, 3.0], 44100)
        >>> noise = AudioSamples.from_mono_f32([0.1, 0.2, 0.1], 44100)
        >>> ratio = snr(signal, noise)  # Returns positive value in dB
    """
    ...

def align_signals(
    reference: AudioSamples, signal: AudioSamples
) -> Tuple[AudioSamples, int]:
    """Align two audio signals by maximizing correlation.

    This function temporally aligns `signal` to `reference` by searching for a
    non-negative sample offset that maximizes cross-correlation. Returns a shifted
    version of the signal padded at the start, together with the estimated offset.

    For mono signals, alignment is computed directly on the single channel.
    For multi-channel signals, all channels are averaged to a single signal before
    alignment. The estimated offset is then applied uniformly to all channels.

    Only non-negative offsets are considered (signal is shifted forward in time only).
    The maximum offset searched is half of the shorter signal length.

    Args:
        reference: The reference signal that defines the target alignment.
        signal: The signal to be aligned to the reference.

    Returns:
        A tuple (aligned_signal, offset_samples) where aligned_signal is the shifted
        signal and offset_samples is the number of samples by which it was shifted.

    Raises:
        ValueError: If the signals have different channel counts or configurations.
        TypeError: If the signals have different data types.

    Example:
        >>> from audio_samples import AudioSamples
        >>> from audio_samples.utils import align_signals
        >>> reference = AudioSamples.from_mono_f32([1.0, 2.0, 3.0, 4.0], 44100)
        >>> signal = AudioSamples.from_mono_f32([2.0, 3.0, 4.0, 5.0], 44100)
        >>> aligned, offset = align_signals(reference, signal)
        >>> print(f"Offset: {offset} samples")
    """
    ...

def psnr(reference: AudioSamples, test: AudioSamples) -> float:
    """Compute the peak signal-to-noise ratio (PSNR) in decibels between two signals.

    PSNR relates the peak amplitude of the reference signal to the mean squared error
    between the two signals. Higher values indicate greater similarity; identical
    signals yield positive infinity.

    Args:
        reference: The reference (clean) signal.
        test: The signal to compare against the reference.

    Returns:
        The PSNR in decibels. Returns positive infinity for identical signals.

    Raises:
        ValueError: If the signals have different dimensions or channel configurations.
        TypeError: If the signals have different data types.
    """
    ...

def segmental_snr(
    signal: AudioSamples, noise: AudioSamples, segment_len: int = 256
) -> float:
    """Compute the segmental signal-to-noise ratio (segmental SNR) in decibels.

    The signals are divided into fixed-length segments; the SNR of each segment is
    computed, clamped to a perceptually motivated range, and then averaged. This often
    correlates better with perceived quality than a single global SNR.

    Args:
        signal: The signal component.
        noise: The noise component.
        segment_len: Number of samples per segment (must be positive, default: 256).

    Returns:
        The mean of the per-segment SNR values in decibels.

    Raises:
        ValueError: If the signals have different dimensions/channels, or `segment_len` is 0.
        TypeError: If the signals have different data types.
    """
    ...

def log_spectral_distance(a: AudioSamples, b: AudioSamples) -> float:
    """Compute the log-spectral distance (LSD) between two audio signals.

    LSD measures the average difference between the log-power spectra of the two
    signals. Lower values indicate greater spectral similarity; identical signals
    yield 0.0.

    Args:
        a: The first audio signal.
        b: The second audio signal.

    Returns:
        The log-spectral distance as a non-negative scalar value.

    Raises:
        ValueError: If the signals have different dimensions or channel configurations.
        TypeError: If the signals have different data types.
    """
    ...

def correlation_per_channel(a: AudioSamples, b: AudioSamples) -> np.ndarray:
    """Compute the Pearson correlation coefficient for each channel independently.

    Mirrors :func:`correlation` but returns one value per channel instead of an average.
    For mono input the returned array has a single element.

    Args:
        a: The first audio signal.
        b: The second audio signal.

    Returns:
        A 1-D NumPy array of per-channel correlation coefficients, in channel order.

    Raises:
        ValueError: If the signals have different dimensions or channel configurations.
        TypeError: If the signals have different data types.
    """
    ...

def mse_per_channel(a: AudioSamples, b: AudioSamples) -> np.ndarray:
    """Compute the mean squared error (MSE) for each channel independently.

    Mirrors :func:`mse` but returns one value per channel instead of an average.
    For mono input the returned array has a single element.

    Args:
        a: The first audio signal.
        b: The second audio signal.

    Returns:
        A 1-D NumPy array of per-channel MSE values, in channel order.

    Raises:
        ValueError: If the signals have different dimensions or channel configurations.
        TypeError: If the signals have different data types.
    """
    ...

def snr_per_channel(signal: AudioSamples, noise: AudioSamples) -> np.ndarray:
    """Compute the signal-to-noise ratio (SNR) in decibels for each channel independently.

    Mirrors :func:`snr` but returns one value per channel instead of aggregating across
    all channels. A channel with zero noise power yields positive infinity. For mono
    input the returned array has a single element.

    Args:
        signal: The signal component.
        noise: The noise component.

    Returns:
        A 1-D NumPy array of per-channel SNR values in decibels, in channel order.

    Raises:
        ValueError: If the signals have different dimensions or channel configurations.
        TypeError: If the signals have different data types.
    """
    ...

# Re-export audio_math functions for convenience
from .audio_math import (
    # Frequency conversions
    hz_to_mel as hz_to_mel,
    mel_to_hz as mel_to_hz,
    mel_scale as mel_scale,
    hz_to_midi as hz_to_midi,
    midi_to_hz as midi_to_hz,
    # Amplitude conversions
    amplitude_to_db as amplitude_to_db,
    db_to_amplitude as db_to_amplitude,
    power_to_db as power_to_db,
    db_to_power as db_to_power,
    # Time/frame conversions
    frames_to_time as frames_to_time,
    time_to_frames as time_to_frames,
    samples_to_time as samples_to_time,
    seconds_to_samples as seconds_to_samples,
    ms_to_samples as ms_to_samples,
    # Musical theory
    note_to_midi as note_to_midi,
    midi_to_note as midi_to_note,
    note_to_frequency as note_to_frequency,
    frequency_to_note as frequency_to_note,
    cents_to_ratio as cents_to_ratio,
    ratio_to_cents as ratio_to_cents,
    # Spectral utilities
    fft_frequencies as fft_frequencies,
    mel_frequencies as mel_frequencies,
    linspace as linspace,
)

__all__ = [
    # Comparison functions
    "correlation",
    "mse",
    "snr",
    "align_signals",
    "psnr",
    "segmental_snr",
    "log_spectral_distance",
    "correlation_per_channel",
    "mse_per_channel",
    "snr_per_channel",
    # Audio math - frequency conversions
    "hz_to_mel",
    "mel_to_hz",
    "mel_scale",
    "hz_to_midi",
    "midi_to_hz",
    # Audio math - amplitude conversions
    "amplitude_to_db",
    "db_to_amplitude",
    "power_to_db",
    "db_to_power",
    # Audio math - time/frame conversions
    "frames_to_time",
    "time_to_frames",
    "samples_to_time",
    "seconds_to_samples",
    "ms_to_samples",
    # Audio math - musical theory
    "note_to_midi",
    "midi_to_note",
    "note_to_frequency",
    "frequency_to_note",
    "cents_to_ratio",
    "ratio_to_cents",
    # Audio math - spectral utilities
    "fft_frequencies",
    "mel_frequencies",
    "linspace",
]
