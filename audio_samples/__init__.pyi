"""
Type stubs for audio_python - A Rust-based Python extension for audio processing.

This module provides high-performance audio processing capabilities with seamless numpy integration.
Supports multiple audio sample formats (i16, I24, i32, f32, f64) with zero-copy numpy integration.
"""

from typing import Optional, Any
import numpy as np
from numpy.typing import NDArray

from . import generation as generation
from . import io as io

# Expose generation functions at the top level for convenience
from .generation import (
    sine_wave, cosine_wave, sawtooth_wave, square_wave, triangle_wave,
    chirp, white_noise, pink_noise, brown_noise, impulse, silence
)

class AudioSamples:
    """
    Main audio processing class supporting multiple sample formats and numpy integration.

    Provides comprehensive audio processing including statistics, filtering, equalization,
    dynamic range processing, and frequency analysis.
    """

    # Constructors
    @staticmethod
    def new_mono(arr: NDArray[Any], sample_rate: int) -> 'AudioSamples':
        """Create mono audio from numpy array (auto-detects dtype)."""
        ...

    @staticmethod
    def new_multi(arr: NDArray[Any], sample_rate: int) -> 'AudioSamples':
        """Create multi-channel audio from numpy array (auto-detects dtype)."""
        ...

    # Factory methods - zeros
    @staticmethod
    def zeros_mono(length: int, sample_rate: int) -> 'AudioSamples':
        """Create zero-filled mono f32 audio."""
        ...

    @staticmethod
    def zeros_mono_i16(length: int, sample_rate: int) -> 'AudioSamples':
        """Create zero-filled mono i16 audio."""
        ...

    @staticmethod
    def zeros_mono_i32(length: int, sample_rate: int) -> 'AudioSamples':
        """Create zero-filled mono i32 audio."""
        ...

    @staticmethod
    def zeros_mono_f64(length: int, sample_rate: int) -> 'AudioSamples':
        """Create zero-filled mono f64 audio."""
        ...

    @staticmethod
    def zeros_multi(channels: int, length: int, sample_rate: int) -> 'AudioSamples':
        """Create zero-filled multi-channel f32 audio."""
        ...

    @staticmethod
    def zeros_multi_i16(channels: int, length: int, sample_rate: int) -> 'AudioSamples':
        """Create zero-filled multi-channel i16 audio."""
        ...

    @staticmethod
    def zeros_multi_i32(channels: int, length: int, sample_rate: int) -> 'AudioSamples':
        """Create zero-filled multi-channel i32 audio."""
        ...

    @staticmethod
    def zeros_multi_f64(channels: int, length: int, sample_rate: int) -> 'AudioSamples':
        """Create zero-filled multi-channel f64 audio."""
        ...

    # Factory methods - ones
    @staticmethod
    def ones_mono(length: int, sample_rate: int) -> 'AudioSamples':
        """Create ones-filled mono f32 audio."""
        ...

    @staticmethod
    def ones_mono_i16(length: int, sample_rate: int) -> 'AudioSamples':
        """Create ones-filled mono i16 audio."""
        ...

    @staticmethod
    def ones_mono_i32(length: int, sample_rate: int) -> 'AudioSamples':
        """Create ones-filled mono i32 audio."""
        ...

    @staticmethod
    def ones_mono_f64(length: int, sample_rate: int) -> 'AudioSamples':
        """Create ones-filled mono f64 audio."""
        ...

    @staticmethod
    def ones_multi(channels: int, length: int, sample_rate: int) -> 'AudioSamples':
        """Create ones-filled multi-channel f32 audio."""
        ...

    @staticmethod
    def ones_multi_i16(channels: int, length: int, sample_rate: int) -> 'AudioSamples':
        """Create ones-filled multi-channel i16 audio."""
        ...

    @staticmethod
    def ones_multi_i32(channels: int, length: int, sample_rate: int) -> 'AudioSamples':
        """Create ones-filled multi-channel i32 audio."""
        ...

    @staticmethod
    def ones_multi_f64(channels: int, length: int, sample_rate: int) -> 'AudioSamples':
        """Create ones-filled multi-channel f64 audio."""
        ...

    # Factory methods - uniform
    @staticmethod
    def uniform_mono(length: int, sample_rate: int, value: float) -> 'AudioSamples':
        """Create uniform-value mono f32 audio."""
        ...

    @staticmethod
    def uniform_mono_i16(length: int, sample_rate: int, value: int) -> 'AudioSamples':
        """Create uniform-value mono i16 audio."""
        ...

    @staticmethod
    def uniform_mono_i32(length: int, sample_rate: int, value: int) -> 'AudioSamples':
        """Create uniform-value mono i32 audio."""
        ...

    @staticmethod
    def uniform_mono_f64(length: int, sample_rate: int, value: float) -> 'AudioSamples':
        """Create uniform-value mono f64 audio."""
        ...

    @staticmethod
    def uniform_multi(channels: int, length: int, sample_rate: int, value: float) -> 'AudioSamples':
        """Create uniform-value multi-channel f32 audio."""
        ...

    @staticmethod
    def uniform_multi_i16(channels: int, length: int, sample_rate: int, value: int) -> 'AudioSamples':
        """Create uniform-value multi-channel i16 audio."""
        ...

    @staticmethod
    def uniform_multi_i32(channels: int, length: int, sample_rate: int, value: int) -> 'AudioSamples':
        """Create uniform-value multi-channel i32 audio."""
        ...

    @staticmethod
    def uniform_multi_f64(channels: int, length: int, sample_rate: int, value: float) -> 'AudioSamples':
        """Create uniform-value multi-channel f64 audio."""
        ...

    # Audio editing operations
    @classmethod
    def concatenate(cls, segments: list['AudioSamples']) -> 'AudioSamples':
        """
        Concatenate multiple audio segments end-to-end.

        All segments must have the same sample rate, channel count, and dtype.

        Args:
            segments: List of AudioSamples to concatenate

        Returns:
            AudioSamples: A new AudioSamples instance with all segments joined

        Raises:
            ValueError: If segments list is empty
            TypeError: If segments have different dtypes
        """
        ...

    @classmethod
    def stack(cls, sources: list['AudioSamples']) -> 'AudioSamples':
        """
        Stack multiple mono audio sources into multi-channel audio.

        All sources must be mono, have the same sample rate, length, and dtype.

        Args:
            sources: List of mono AudioSamples to stack as channels

        Returns:
            AudioSamples: A new multi-channel AudioSamples instance

        Raises:
            ValueError: If sources list is empty or sources are not mono
            TypeError: If sources have different dtypes
        """
        ...

    # AudioEditing methods
    def repeat(self, count: int) -> 'AudioSamples':
        """Repeat the audio samples count times."""
        ...

    def trim_silence(self, threshold_db: float) -> 'AudioSamples':
        """Trim silence from the beginning and end of the audio based on a dB threshold."""
        ...

    def pad(self, pad_start_seconds: float, pad_end_seconds: float, pad_value: float = 0.0) -> 'AudioSamples':
        """Pad audio with silence or a specific value at the start and/or end."""
        ...

    def split(self, segment_duration_seconds: float) -> list['AudioSamples']:
        """Split audio into segments of a specified duration."""
        ...

    @classmethod
    def mix(cls, sources: list['AudioSamples'], weights: Optional[list[float]] = None) -> 'AudioSamples':
        """Mix multiple audio sources together with optional weights."""
        ...

    def fade_in(self, duration_seconds: float, curve: str = 'linear') -> None:
        """Apply a fade-in effect. Curve options: 'linear', 'exponential', 'logarithmic', 'smoothstep'."""
        ...

    def fade_out(self, duration_seconds: float, curve: str = 'linear') -> None:
        """Apply a fade-out effect. Curve options: 'linear', 'exponential', 'logarithmic', 'smoothstep'."""
        ...

    # AudioChannelOps methods
    def pan(self, pan_value: float) -> None:
        """Pan stereo audio. -1.0 = full left, 0.0 = center, 1.0 = full right."""
        ...

    def balance(self, balance: float) -> None:
        """Adjust stereo balance. -1.0 = left only, 0.0 = equal, 1.0 = right only."""
        ...

    # AudioTransforms methods
    def stft(
        self,
        window_size: int,
        hop_size: int,
        window_type: str = 'hann'
    ) -> NDArray[numpy.complexfloating]:
        """Compute Short-Time Fourier Transform. Returns complex STFT matrix."""
        ...

    @classmethod
    def istft(
        cls,
        stft_matrix: NDArray[numpy.complexfloating],
        hop_size: int,
        window_type: str = 'hann',
        sample_rate: int = 44100,
        center: bool = True
    ) -> 'AudioSamples':
        """Compute inverse STFT to reconstruct time-domain signal."""
        ...

    def spectrogram(
        self,
        window_size: int,
        hop_size: int,
        window_type: str = 'hann',
        scale: str = 'linear',
        normalize: bool = False
    ) -> NDArray[numpy.floating]:
        """Compute magnitude spectrogram. Scale options: 'linear', 'log', 'mel'."""
        ...

    # AudioProcessing methods
    def resample(
        self,
        target_sample_rate: int,
        quality: str = 'medium'
    ) -> 'AudioSamples':
        """Resample audio to target sample rate. Quality: 'fast', 'medium', 'high'."""
        ...

    def resample_by_ratio(
        self,
        ratio: float,
        quality: str = 'medium'
    ) -> 'AudioSamples':
        """Resample audio by a given ratio. Quality: 'fast', 'medium', 'high'."""
        ...

    def apply_window(self, window: NDArray[numpy.floating]) -> None:
        """Apply a window function to the audio samples in-place."""
        ...

    # AudioPitchAnalysis methods
    def detect_pitch_yin(
        self,
        threshold: float = 0.1,
        min_frequency: float = 50.0,
        max_frequency: float = 2000.0
    ) -> Optional[float]:
        """Detect pitch using the YIN algorithm. Returns frequency in Hz or None."""
        ...

    def track_pitch(
        self,
        window_size: int,
        hop_size: int,
        method: str = 'yin',
        threshold: float = 0.1,
        min_frequency: float = 50.0,
        max_frequency: float = 2000.0
    ) -> list[tuple[float, Optional[float]]]:
        """Track pitch over time. Returns list of (time, frequency) tuples.

        Method options: 'yin', 'autocorrelation', 'cepstrum', 'harmonic_product'.
        """
        ...

    # AudioDecomposition methods
    def hpss(
        self,
        win_size: int = 2048,
        hop_size: int = 512,
        median_filter_harmonic: int = 17,
        median_filter_percussive: int = 17,
        mask_softness: float = 0.5
    ) -> tuple['AudioSamples', 'AudioSamples']:
        """Separate harmonic and percussive components using HPSS.

        Returns:
            tuple: (harmonic_audio, percussive_audio)
        """
        ...

    # IIR Filtering methods
    def butterworth_lowpass(self, order: int, cutoff_frequency: float) -> None:
        """Apply a Butterworth low-pass filter. Higher order = steeper rolloff."""
        ...

    def butterworth_highpass(self, order: int, cutoff_frequency: float) -> None:
        """Apply a Butterworth high-pass filter. Higher order = steeper rolloff."""
        ...

    def butterworth_bandpass(self, order: int, low_frequency: float, high_frequency: float) -> None:
        """Apply a Butterworth band-pass filter. Higher order = steeper rolloff."""
        ...

    # Simple filter methods
    def low_pass_filter(self, cutoff_hz: float) -> None:
        """Apply a simple low-pass filter. Attenuates frequencies above cutoff."""
        ...

    def high_pass_filter(self, cutoff_hz: float) -> None:
        """Apply a simple high-pass filter. Attenuates frequencies below cutoff."""
        ...

    def band_pass_filter(self, low_cutoff_hz: float, high_cutoff_hz: float) -> None:
        """Apply a simple band-pass filter. Passes frequencies between cutoffs."""
        ...

    # Statistical methods
    def peak(self) -> int | float:
        """Get the peak (absolute maximum) sample value."""
        ...

    def min_sample(self) -> int | float:
        """Get the minimum sample value."""
        ...

    def max_sample(self) -> int | float:
        """Get the maximum sample value."""
        ...

    def mean(self) -> Optional[float]:
        """Calculate the mean sample value."""
        ...

    def rms(self) -> Optional[float]:
        """Calculate the RMS (root mean square) value."""
        ...

    def variance(self) -> Optional[float]:
        """Calculate the variance of sample values."""
        ...

    def std_dev(self) -> Optional[float]:
        """Calculate the standard deviation of sample values."""
        ...

    def zero_crossings(self) -> int:
        """Count the number of zero crossings."""
        ...

    def zero_crossing_rate(self) -> float:
        """Calculate the zero crossing rate."""
        ...

    def autocorrelation(self, max_lag: int) -> Optional[list[float]]:
        """Calculate autocorrelation with given maximum lag."""
        ...

    def spectral_centroid(self) -> float:
        """Calculate the spectral centroid."""
        ...

    def spectral_rolloff(self, rolloff_percent: float) -> float:
        """Calculate spectral rolloff at given percentage."""
        ...

    # Amplitude processing (in-place)
    def scale(self, factor: float) -> None:
        """Scale audio amplitude by factor (in-place)."""
        ...

    def normalize(self, min_val: float, max_val: float, method: str) -> None:
        """Normalize audio to given range using specified method (in-place)."""
        ...

    def clip(self, min_val: float, max_val: float) -> None:
        """Clip audio values to given range (in-place)."""
        ...

    def remove_dc_offset(self) -> None:
        """Remove DC offset from audio (in-place)."""
        ...

    # Time-domain processing
    def reverse(self) -> 'AudioSamples':
        """Return a reversed copy of the audio."""
        ...

    def trim(self, start_seconds: float, end_seconds: float) -> 'AudioSamples':
        """Return a trimmed copy of the audio."""
        ...

    # Dynamic range processing (in-place)
    def apply_compressor(self, threshold_db: float, ratio: float, attack_ms: float,
                        release_ms: float, makeup_gain_db: float, sample_rate: float) -> None:
        """Apply compression (in-place)."""
        ...

    def apply_limiter(self, ceiling_db: float, release_ms: float,
                     lookahead_ms: float, sample_rate: float) -> None:
        """Apply limiting (in-place)."""
        ...

    def apply_gate(self, threshold_db: float, ratio: float, attack_ms: float,
                   release_ms: float, sample_rate: float) -> None:
        """Apply noise gate (in-place)."""
        ...

    def apply_expander(self, threshold_db: float, ratio: float, attack_ms: float,
                       release_ms: float, sample_rate: float) -> None:
        """Apply expander (in-place)."""
        ...

    # Filtering (in-place)
    def apply_iir_filter(self, design: 'IirFilterDesign', sample_rate: float) -> None:
        """Apply IIR filter using filter design (in-place)."""
        ...

    def apply_butterworth_lowpass(self, order: int, cutoff_frequency: float, sample_rate: float) -> None:
        """Apply Butterworth lowpass filter (in-place)."""
        ...

    def apply_butterworth_highpass(self, order: int, cutoff_frequency: float, sample_rate: float) -> None:
        """Apply Butterworth highpass filter (in-place)."""
        ...

    def apply_butterworth_bandpass(self, order: int, low_frequency: float,
                                  high_frequency: float, sample_rate: float) -> None:
        """Apply Butterworth bandpass filter (in-place)."""
        ...

    def apply_chebyshev_i(self, order: int, cutoff_frequency: float, passband_ripple: float,
                         sample_rate: float, response: str) -> None:
        """Apply Chebyshev Type I filter (in-place)."""
        ...

    # Equalization (in-place)
    def apply_parametric_eq(self, eq: 'ParametricEq', sample_rate: float) -> None:
        """Apply parametric equalizer (in-place)."""
        ...

    def apply_eq_band(self, band: 'EqBand', sample_rate: float) -> None:
        """Apply single EQ band (in-place)."""
        ...

    def apply_peak_filter(self, frequency: float, gain_db: float,
                         q_factor: float, sample_rate: float) -> None:
        """Apply peak filter (in-place)."""
        ...

    def apply_low_shelf(self, frequency: float, gain_db: float,
                       q_factor: float, sample_rate: float) -> None:
        """Apply low shelf filter (in-place)."""
        ...

    def apply_high_shelf(self, frequency: float, gain_db: float,
                        q_factor: float, sample_rate: float) -> None:
        """Apply high shelf filter (in-place)."""
        ...

    def apply_three_band_eq(self, low_freq: float, low_gain: float, mid_freq: float,
                           mid_gain: float, mid_q: float, high_freq: float,
                           high_gain: float, sample_rate: float) -> None:
        """Apply three-band equalizer (in-place)."""
        ...

    # Frequency analysis
    def frequency_response(self, frequencies: list[float], sample_rate: float) -> tuple[list[float], list[float]]:
        """Calculate frequency response at given frequencies."""
        ...

    def fft(self) -> tuple[list[float], list[float]]:
        """Calculate FFT returning (real, imaginary) parts."""
        ...

    def power_spectral_density(self, window_size: int, overlap: float) -> tuple[list[float], list[float]]:
        """Calculate power spectral density."""
        ...

    def mel_spectrogram(self, n_mels: int, fmin: float, fmax: float,
                       window_size: int, hop_size: int) -> NDArray[np.float64]:
        """Calculate mel spectrogram."""
        ...

    def mfcc(self, n_mfcc: int, n_mels: int, fmin: float, fmax: float) -> NDArray[np.float64]:
        """Calculate MFCC (Mel-Frequency Cepstral Coefficients)."""
        ...

    def chroma(self, n_chroma: int) -> NDArray[np.float64]:
        """Calculate chroma features."""
        ...

    # Channel manipulation
    def to_mono(self, method: str, weights: Optional[list[float]] = None) -> 'AudioSamples':
        """Convert to mono using specified method."""
        ...

    def to_stereo(self, method: str, pan: Optional[float] = None) -> 'AudioSamples':
        """Convert to stereo using specified method."""
        ...

    def extract_channel(self, channel_index: int) -> 'AudioSamples':
        """Extract a single channel."""
        ...

    def swap_channels(self, channel1: int, channel2: int) -> None:
        """Swap two channels (in-place)."""
        ...

    # Metadata and properties
    @property
    def dtype(self) -> np.dtype[Any]:
        """Get the numpy dtype of the audio samples."""
        ...

    def sample_rate(self) -> int:
        """Get the sample rate in Hz."""
        ...

    def num_channels(self) -> int:
        """Get the number of audio channels."""
        ...

    def __len__(self) -> int:
        """Get the length of the audio (total number of samples)."""
        ...

    def samples_per_channel(self) -> int:
        """Get the number of samples per channel."""
        ...

    def total_samples(self) -> int:
        """Get the total number of samples across all channels."""
        ...

    def shape(self) -> list[int]:
        """Get the shape of the audio data."""
        ...

    def is_mono(self) -> bool:
        """Check if audio is mono (single channel)."""
        ...

    def is_multi_channel(self) -> bool:
        """Check if audio is multi-channel."""
        ...

    def is_empty(self) -> bool:
        """Check if audio data is empty."""
        ...

    def duration_seconds(self) -> float:
        """Get the duration in seconds."""
        ...

    def __str__(self) -> str:
        """String representation of the audio samples."""
        ...


class IirFilterDesign:
    """IIR (Infinite Impulse Response) filter design."""

    def __init__(self, filter_type: str, response: str, order: int,
                 cutoff_frequency: Optional[float] = None,
                 low_frequency: Optional[float] = None,
                 high_frequency: Optional[float] = None) -> None:
        """
        Create IIR filter design.

        Args:
            filter_type: 'butterworth', 'chebyshev1', 'chebyshev2', or 'elliptic'
            response: Filter response type
            order: Filter order
            cutoff_frequency: Cutoff frequency (for lowpass/highpass)
            low_frequency: Low frequency (for bandpass/bandstop)
            high_frequency: High frequency (for bandpass/bandstop)
        """
        ...


class EqBand:
    """Equalizer band for parametric equalization."""

    def __init__(self, band_type: str, frequency: float, gain_db: float, q_factor: float) -> None:
        """
        Create EQ band.

        Args:
            band_type: 'peak', 'lowshelf', 'highshelf', 'lowpass', 'highpass', or 'bandpass'
            frequency: Center/cutoff frequency in Hz
            gain_db: Gain in dB
            q_factor: Q factor (bandwidth control)
        """
        ...


class ParametricEq:
    """Parametric equalizer supporting multiple bands."""

    def __init__(self) -> None:
        """Create empty parametric equalizer."""
        ...

    def add_band(self, band: EqBand) -> None:
        """Add an EQ band to the equalizer."""
        ...


# Re-export generation functions at top level for backwards compatibility
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
                amplitude: float = 1.0, dtype: Optional[Any] = None) -> AudioSamples:
    """
    Generate white noise audio signal.

    Args:
        duration_secs: Duration of the signal in seconds
        sample_rate: Sample rate in samples per second (default: 44100)
        amplitude: Peak amplitude of the noise (default: 1.0)
        dtype: NumPy dtype for the output array (default: f64)

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
