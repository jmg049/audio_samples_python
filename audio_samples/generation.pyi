"""
Type stubs for audio_python.generation - Audio signal generation functions.

This module provides functions to generate various audio waveforms and noise types.
"""

from typing import Optional, Any
from . import AudioSamples


def sine_wave(frequency: float, duration_secs: float, sample_rate: int = 44100,
              amplitude: float = 1.0, dtype: Optional[Any] = None) -> AudioSamples:
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

def cosine_wave(frequency: float, duration_secs: float, sample_rate: int = 44100,
                amplitude: float = 1.0, dtype: Optional[Any] = None) -> AudioSamples:
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

def sawtooth_wave(frequency: float, duration_secs: float, sample_rate: int = 44100,
                  amplitude: float = 1.0, dtype: Optional[Any] = None) -> AudioSamples:
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

def square_wave(frequency: float, duration_secs: float, sample_rate: int = 44100,
                amplitude: float = 1.0, dtype: Optional[Any] = None) -> AudioSamples:
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

def triangle_wave(frequency: float, duration_secs: float, sample_rate: int = 44100,
                  amplitude: float = 1.0, dtype: Optional[Any] = None) -> AudioSamples:
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

def chirp(f0: float, f1: float, duration_secs: float, sample_rate: int = 44100,
          amplitude: float = 1.0, dtype: Optional[Any] = None) -> AudioSamples:
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

def white_noise(duration_secs: float, sample_rate: int = 44100,
                amplitude: float = 1.0, dtype: Optional[Any] = None, seed: Optional[int] = None) -> AudioSamples:
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

def pink_noise(duration_secs: float, sample_rate: int = 44100,
               amplitude: float = 1.0, dtype: Optional[Any] = None) -> AudioSamples:
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

def brown_noise(duration_secs: float, sample_rate: int = 44100, step: float = 0.01,
                amplitude: float = 1.0, dtype: Optional[Any] = None) -> AudioSamples:
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

def impulse(duration_secs: float, sample_rate: int = 44100, amplitude: float = 1.0,
            position: float = 0.5, dtype: Optional[Any] = None) -> AudioSamples:
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

def silence(duration_secs: float, sample_rate: int = 44100,
            dtype: Optional[Any] = None) -> AudioSamples:
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
