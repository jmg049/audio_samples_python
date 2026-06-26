"""
Type stubs for audio_python.generation - Audio signal generation functions.

This module provides functions to generate various audio waveforms and noise types.
"""

from typing import Optional, Any
from . import AudioSamples

def sine_wave(
    frequency: float,
    duration_secs: float,
    sample_rate: int = 44100,
    amplitude: float = 1.0,
    dtype: Optional[Any] = None,
) -> AudioSamples:
    """
    Generate a sine wave audio signal.

    Args:
        frequency: Frequency of the sine wave in Hz
        duration_secs: Duration of the signal in seconds
        sample_rate: Sample rate in samples per second (default: 44100)
        amplitude: Peak amplitude of the wave (default: 1.0)
        dtype: NumPy dtype for the output array (default: f64)

    Returns:
        AudioSamples: Generated sine wave audio data
    """
    ...

def cosine_wave(
    frequency: float,
    duration_secs: float,
    sample_rate: int = 44100,
    amplitude: float = 1.0,
    dtype: Optional[Any] = None,
) -> AudioSamples:
    """
    Generate a cosine wave audio signal.

    Args:
        frequency: Frequency of the cosine wave in Hz
        duration_secs: Duration of the signal in seconds
        sample_rate: Sample rate in samples per second (default: 44100)
        amplitude: Peak amplitude of the wave (default: 1.0)
        dtype: NumPy dtype for the output array (default: f64)

    Returns:
        AudioSamples: Generated cosine wave audio data
    """
    ...

def sawtooth_wave(
    frequency: float,
    duration_secs: float,
    sample_rate: int = 44100,
    amplitude: float = 1.0,
    dtype: Optional[Any] = None,
) -> AudioSamples:
    """
    Generate a sawtooth wave audio signal.

    Args:
        frequency: Frequency of the sawtooth wave in Hz
        duration_secs: Duration of the signal in seconds
        sample_rate: Sample rate in samples per second (default: 44100)
        amplitude: Peak amplitude of the wave (default: 1.0)
        dtype: NumPy dtype for the output array (default: f64)

    Returns:
        AudioSamples: Generated sawtooth wave audio data
    """
    ...

def square_wave(
    frequency: float,
    duration_secs: float,
    sample_rate: int = 44100,
    amplitude: float = 1.0,
    dtype: Optional[Any] = None,
) -> AudioSamples:
    """
    Generate a square wave audio signal.

    Args:
        frequency: Frequency of the square wave in Hz
        duration_secs: Duration of the signal in seconds
        sample_rate: Sample rate in samples per second (default: 44100)
        amplitude: Peak amplitude of the wave (default: 1.0)
        dtype: NumPy dtype for the output array (default: f64)

    Returns:
        AudioSamples: Generated square wave audio data
    """
    ...

def triangle_wave(
    frequency: float,
    duration_secs: float,
    sample_rate: int = 44100,
    amplitude: float = 1.0,
    dtype: Optional[Any] = None,
) -> AudioSamples:
    """
    Generate a triangle wave audio signal.

    Args:
        frequency: Frequency of the triangle wave in Hz
        duration_secs: Duration of the signal in seconds
        sample_rate: Sample rate in samples per second (default: 44100)
        amplitude: Peak amplitude of the wave (default: 1.0)
        dtype: NumPy dtype for the output array (default: f64)

    Returns:
        AudioSamples: Generated triangle wave audio data
    """
    ...

def chirp(
    f0: float,
    f1: float,
    duration_secs: float,
    sample_rate: int = 44100,
    amplitude: float = 1.0,
    dtype: Optional[Any] = None,
) -> AudioSamples:
    """
    Generate a frequency chirp (sweep) audio signal.

    Args:
        f0: Starting frequency in Hz
        f1: Ending frequency in Hz
        duration_secs: Duration of the signal in seconds
        sample_rate: Sample rate in samples per second (default: 44100)
        amplitude: Peak amplitude of the wave (default: 1.0)
        dtype: NumPy dtype for the output array (default: f64)

    Returns:
        AudioSamples: Generated chirp audio data
    """
    ...

def white_noise(
    duration_secs: float,
    sample_rate: int = 44100,
    amplitude: float = 1.0,
    dtype: Optional[Any] = None,
    seed: Optional[int] = None,
) -> AudioSamples:
    """
    Generate white noise audio signal.

    Args:
        duration_secs: Duration of the signal in seconds
        sample_rate: Sample rate in samples per second (default: 44100)
        amplitude: Peak amplitude of the noise (default: 1.0)
        dtype: NumPy dtype for the output array (default: f64)
        seed: Optional seed for reproducible noise (default: None)

    Returns:
        AudioSamples: Generated white noise audio data
    """
    ...

def pink_noise(
    duration_secs: float,
    sample_rate: int = 44100,
    amplitude: float = 1.0,
    dtype: Optional[Any] = None,
) -> AudioSamples:
    """
    Generate pink noise audio signal.

    Args:
        duration_secs: Duration of the signal in seconds
        sample_rate: Sample rate in samples per second (default: 44100)
        amplitude: Peak amplitude of the noise (default: 1.0)
        dtype: NumPy dtype for the output array (default: f64)

    Returns:
        AudioSamples: Generated pink noise audio data
    """
    ...

def brown_noise(
    duration_secs: float,
    sample_rate: int = 44100,
    step: float = 0.01,
    amplitude: float = 1.0,
    dtype: Optional[Any] = None,
) -> AudioSamples:
    """
    Generate brown noise (Brownian/red noise) audio signal.

    Args:
        duration_secs: Duration of the signal in seconds
        sample_rate: Sample rate in samples per second (default: 44100)
        step: Step size for the random walk (default: 0.01)
        amplitude: Peak amplitude of the noise (default: 1.0)
        dtype: NumPy dtype for the output array (default: f64)

    Returns:
        AudioSamples: Generated brown noise audio data
    """
    ...

def impulse(
    duration_secs: float,
    sample_rate: int = 44100,
    amplitude: float = 1.0,
    position: float = 0.5,
    dtype: Optional[Any] = None,
) -> AudioSamples:
    """
    Generate an impulse (delta function) audio signal.

    Args:
        duration_secs: Duration of the signal in seconds
        sample_rate: Sample rate in samples per second (default: 44100)
        amplitude: Peak amplitude of the impulse (default: 1.0)
        position: Position of the impulse as fraction of duration (default: 0.5)
        dtype: NumPy dtype for the output array (default: f64)

    Returns:
        AudioSamples: Generated impulse audio data
    """
    ...

def silence(
    duration_secs: float, sample_rate: int = 44100, dtype: Optional[Any] = None
) -> AudioSamples:
    """
    Generate silence (zero amplitude) audio signal.

    Args:
        duration_secs: Duration of the signal in seconds
        sample_rate: Sample rate in samples per second (default: 44100)
        dtype: NumPy dtype for the output array (default: f64)

    Returns:
        AudioSamples: Generated silence audio data
    """
    ...

def square_wave_bandlimited(
    frequency: float,
    duration_secs: float,
    sample_rate: int = 44100,
    amplitude: float = 1.0,
    dtype: Optional[Any] = None,
) -> AudioSamples:
    """
    Generate a band-limited (alias-free) square wave via additive synthesis.

    Unlike a hard-clipped square wave, this sums only the odd harmonics that lie
    strictly below the Nyquist frequency, producing an exactly band-limited signal
    with no aliasing.

    Args:
        frequency: Fundamental frequency of the square wave in Hz
        duration_secs: Duration of the signal in seconds
        sample_rate: Sample rate in samples per second (default: 44100)
        amplitude: Peak amplitude of the underlying ideal square wave (default: 1.0)
        dtype: NumPy dtype for the output array (default: f32)

    Returns:
        AudioSamples: Generated band-limited square wave audio data
    """
    ...

def sawtooth_wave_bandlimited(
    frequency: float,
    duration_secs: float,
    sample_rate: int = 44100,
    amplitude: float = 1.0,
    dtype: Optional[Any] = None,
) -> AudioSamples:
    """
    Generate a band-limited (alias-free) sawtooth wave via additive synthesis.

    Sums only the harmonics that lie strictly below the Nyquist frequency, producing
    an exactly band-limited signal with no aliasing.

    Args:
        frequency: Fundamental frequency of the sawtooth wave in Hz
        duration_secs: Duration of the signal in seconds
        sample_rate: Sample rate in samples per second (default: 44100)
        amplitude: Peak amplitude of the underlying ideal sawtooth wave (default: 1.0)
        dtype: NumPy dtype for the output array (default: f32)

    Returns:
        AudioSamples: Generated band-limited sawtooth wave audio data
    """
    ...

def triangle_wave_bandlimited(
    frequency: float,
    duration_secs: float,
    sample_rate: int = 44100,
    amplitude: float = 1.0,
    dtype: Optional[Any] = None,
) -> AudioSamples:
    """
    Generate a band-limited (alias-free) triangle wave via additive synthesis.

    Sums only the odd harmonics that lie strictly below the Nyquist frequency,
    producing an exactly band-limited signal with no aliasing.

    Args:
        frequency: Fundamental frequency of the triangle wave in Hz
        duration_secs: Duration of the signal in seconds
        sample_rate: Sample rate in samples per second (default: 44100)
        amplitude: Peak amplitude of the underlying ideal triangle wave (default: 1.0)
        dtype: NumPy dtype for the output array (default: f32)

    Returns:
        AudioSamples: Generated band-limited triangle wave audio data
    """
    ...

def exponential_chirp(
    start_freq: float,
    end_freq: float,
    duration_secs: float,
    sample_rate: int = 44100,
    amplitude: float = 1.0,
    dtype: Optional[Any] = None,
) -> AudioSamples:
    """
    Generate an exponential (logarithmic) frequency chirp (sweep) audio signal.

    The instantaneous frequency sweeps geometrically from ``start_freq`` to
    ``end_freq``. Both frequencies must be strictly positive.

    Args:
        start_freq: Starting frequency in Hz (must be > 0)
        end_freq: Ending frequency in Hz (must be > 0)
        duration_secs: Duration of the signal in seconds
        sample_rate: Sample rate in samples per second (default: 44100)
        amplitude: Peak amplitude of the wave (default: 1.0)
        dtype: NumPy dtype for the output array (default: f32)

    Returns:
        AudioSamples: Generated exponential chirp audio data

    Raises:
        ValueError: If ``start_freq`` or ``end_freq`` is not strictly positive.
    """
    ...

def fm_signal(
    carrier_freq: float,
    modulator_freq: float,
    modulation_index: float,
    duration_secs: float,
    sample_rate: int = 44100,
    amplitude: float = 1.0,
    dtype: Optional[Any] = None,
) -> AudioSamples:
    """
    Generate a frequency-modulated (FM) signal.

    A sinusoidal carrier whose instantaneous phase is modulated by a sinusoidal
    modulator. With ``modulation_index == 0`` this reduces to a pure sine at the
    carrier frequency.

    Args:
        carrier_freq: Carrier frequency in Hz
        modulator_freq: Modulator frequency in Hz
        modulation_index: Modulation index (peak phase deviation in radians)
        duration_secs: Duration of the signal in seconds
        sample_rate: Sample rate in samples per second (default: 44100)
        amplitude: Peak amplitude (default: 1.0)
        dtype: NumPy dtype for the output array (default: f32)

    Returns:
        AudioSamples: Generated FM signal audio data
    """
    ...

def am_signal(
    carrier_freq: float,
    modulator_freq: float,
    modulation_depth: float,
    duration_secs: float,
    sample_rate: int = 44100,
    amplitude: float = 1.0,
    dtype: Optional[Any] = None,
) -> AudioSamples:
    """
    Generate an amplitude-modulated (AM) signal.

    A carrier signal modulated by a low-frequency envelope.

    Args:
        carrier_freq: Frequency of the carrier signal in Hz
        modulator_freq: Frequency of the modulating envelope in Hz
        modulation_depth: Depth of modulation (0.0 = none, 1.0 = full)
        duration_secs: Duration of the signal in seconds
        sample_rate: Sample rate in samples per second (default: 44100)
        amplitude: Overall amplitude (default: 1.0)
        dtype: NumPy dtype for the output array (default: f32)

    Returns:
        AudioSamples: Generated AM signal audio data
    """
    ...

def compound_tone(
    components: list[tuple[float, float]],
    duration_secs: float,
    sample_rate: int = 44100,
    dtype: Optional[Any] = None,
) -> AudioSamples:
    """
    Generate a compound tone from multiple frequency components.

    Useful for creating signals with harmonics or multiple simultaneous tones. Each
    component is given as a ``(frequency_hz, amplitude)`` tuple.

    Args:
        components: List of ``(frequency_hz, amplitude)`` tuples. Must be non-empty.
        duration_secs: Duration of the signal in seconds
        sample_rate: Sample rate in samples per second (default: 44100)
        dtype: NumPy dtype for the output array (default: f32)

    Returns:
        AudioSamples: Generated compound tone audio data

    Raises:
        ValueError: If ``components`` is empty or ``duration_secs`` is not positive.
    """
    ...

def exponential_bursts(
    burst_rate: float,
    decay_rate: float,
    duration_secs: float,
    sample_rate: int = 44100,
    amplitude: float = 1.0,
    dtype: Optional[Any] = None,
) -> AudioSamples:
    """
    Generate a signal with periodic exponential-decay bursts.

    Creates percussive-like transients useful for testing onset detection.

    Args:
        burst_rate: Number of bursts per second
        decay_rate: Exponential decay rate (higher = faster decay)
        duration_secs: Duration of the signal in seconds
        sample_rate: Sample rate in samples per second (default: 44100)
        amplitude: Peak amplitude of bursts (default: 1.0)
        dtype: NumPy dtype for the output array (default: f32)

    Returns:
        AudioSamples: Generated burst signal audio data
    """
    ...

def stereo_sine_wave(
    frequency: float,
    duration_secs: float,
    sample_rate: int = 44100,
    amplitude: float = 1.0,
    dtype: Optional[Any] = None,
) -> AudioSamples:
    """
    Generate a stereo sine wave audio signal (same frequency on both channels).

    Args:
        frequency: Frequency of the sine wave in Hz
        duration_secs: Duration of the signal in seconds
        sample_rate: Sample rate in samples per second (default: 44100)
        amplitude: Peak amplitude of the wave (default: 1.0)
        dtype: NumPy dtype for the output array (default: f64)

    Returns:
        AudioSamples: Generated stereo sine wave audio data (2 channels)
    """
    ...

def stereo_chirp(
    start_freq: float,
    end_freq: float,
    duration_secs: float,
    sample_rate: int = 44100,
    amplitude: float = 1.0,
    dtype: Optional[Any] = None,
) -> AudioSamples:
    """
    Generate a stereo chirp signal (frequency sweep on both channels).

    Args:
        start_freq: Starting frequency in Hz
        end_freq: Ending frequency in Hz
        duration_secs: Duration of the signal in seconds
        sample_rate: Sample rate in samples per second (default: 44100)
        amplitude: Peak amplitude of the chirp (default: 1.0)
        dtype: NumPy dtype for the output array (default: f64)

    Returns:
        AudioSamples: Generated stereo chirp audio data (2 channels)
    """
    ...

def stereo_silence(
    duration_secs: float, sample_rate: int = 44100, dtype: Optional[Any] = None
) -> AudioSamples:
    """
    Generate stereo silence (zero amplitude on both channels).

    Args:
        duration_secs: Duration of the signal in seconds
        sample_rate: Sample rate in samples per second (default: 44100)
        dtype: NumPy dtype for the output array (default: f64)

    Returns:
        AudioSamples: Generated stereo silence audio data (2 channels)
    """
    ...
