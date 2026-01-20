"""
Python bindings for the AudioSamples Rust ecosystem.
AudioSamples eliminates the manual metadata coordination
burden that exists in audio processing libraries by treating
audio as a first-class data type with intrinsically embedded
properties.
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional, Any
import numpy as np
from numpy.typing import NDArray, DTypeLike
import numpy.typing as npt

from spectrograms import (
    ChromaParams,
    CqtParams,
    LogParams,
    MelParams,
    MfccParams,
    SpectrogramParams,
    StftParams,
    StftResult,
    ErbParams,
)

from audio_samples.types import (
    EqBandType,
    FilterResponse,
    PeakPickingConfig,
    SampleType,
    PadSide,
    PerturbationConfig,
    FadeCurve,
    MonoConversionMethod,
    StereoConversionMethod,
    CompressorConfig,
    LimiterConfig,
    PitchDetectionMethod,
    ResamplingQuality,
    NormalizationMethod,
)

from . import generation as generation
from . import io as io
from . import plotting as plotting
from . import utils as utils
from . import audio_math as audio_math
from . import mpl as mpl

# Expose generation functions at the top level for convenience
from .generation import (
    sine_wave,
    cosine_wave,
    sawtooth_wave,
    square_wave,
    triangle_wave,
    chirp,
    white_noise,
    pink_noise,
    brown_noise,
    impulse,
    silence,
    stereo_sine_wave,
    stereo_chirp,
    stereo_silence,
)

__all__ = [
    "mpl",
    "sine_wave",
    "cosine_wave",
    "sawtooth_wave",
    "square_wave",
    "triangle_wave",
    "chirp",
    "white_noise",
    "pink_noise",
    "brown_noise",
    "impulse",
    "silence",
    "stereo_sine_wave",
    "stereo_chirp",
    "stereo_silence",
    "AudioSamples",
    "concatenate",
    "stack",
    "IirFilterDesign",
    "EqBand",
    "ParametricEq",
    "OnsetDetectionConfig",
    "ComplexOnsetConfig",
    "SpectralFluxConfig",
    "SpectralFluxMethod",
    "DynamicRangeMethod",
    "EnvelopeFollower",
    "BeatTrackingConfig",
    "BeatTrackingData",
    "VadConfig",
    "VadMethod",
    "VadChannelPolicy",
    "Layout",
    "ChannelManagementStrategy",
    "WaveformPlotParams",
    "SpectrogramPlotParams",
    "MagnitudeSpectrumParams",
    "WaveformPlot",
    "SpectrogramPlot",
    "MagnitudeSpectrumPlot",
    "PadSide",
    "PerturbationConfig",
    "FadeCurve",
    "MonoConversionMethod",
    "StereoConversionMethod",
    "CompressorConfig",
    "LimiterConfig",
    "PitchDetectionMethod",
    "ResamplingQuality",
    "NormalizationMethod",
]

class AudioSamples:
    """
    Main audio processing class supporting multiple sample formats and numpy integration.

    Provides comprehensive audio processing including statistics, filtering, equalization,
    dynamic range processing, and frequency analysis.
    """

    def __array__(self, dtype: DTypeLike = np.float64) -> npt.NDArray[np.float64]:
        """Convert the spectrogram to a NumPy array.

        :param dtype: Desired data type (default: np.float64)
        :return: Spectrogram data as a NumPy array of specified dtype
        """
        ...

    @staticmethod
    def new_mono(arr: NDArray[Any], sample_rate: int) -> AudioSamples:
        """Create mono audio from numpy array (auto-detects dtype)."""
        ...

    @staticmethod
    def new_multi(arr: NDArray[Any], sample_rate: int) -> AudioSamples:
        """Create multi-channel audio from numpy array (auto-detects dtype)."""
        ...

    # Factory methods - zeros
    @staticmethod
    def zeros_mono(length: int, sample_rate: int) -> AudioSamples:
        """Create zero-filled mono f32 audio."""
        ...

    @staticmethod
    def zeros_mono_i16(length: int, sample_rate: int) -> AudioSamples:
        """Create zero-filled mono i16 audio."""
        ...

    @staticmethod
    def zeros_mono_i32(length: int, sample_rate: int) -> AudioSamples:
        """Create zero-filled mono i32 audio."""
        ...

    @staticmethod
    def zeros_mono_f64(length: int, sample_rate: int) -> AudioSamples:
        """Create zero-filled mono f64 audio."""
        ...

    @staticmethod
    def zeros_multi(channels: int, length: int, sample_rate: int) -> AudioSamples:
        """Create zero-filled multi-channel f32 audio."""
        ...

    @staticmethod
    def zeros_multi_i16(channels: int, length: int, sample_rate: int) -> AudioSamples:
        """Create zero-filled multi-channel i16 audio."""
        ...

    @staticmethod
    def zeros_multi_i32(channels: int, length: int, sample_rate: int) -> AudioSamples:
        """Create zero-filled multi-channel i32 audio."""
        ...

    @staticmethod
    def zeros_multi_f64(channels: int, length: int, sample_rate: int) -> AudioSamples:
        """Create zero-filled multi-channel f64 audio."""
        ...

    # Factory methods - ones
    @staticmethod
    def ones_mono(length: int, sample_rate: int) -> AudioSamples:
        """Create ones-filled mono f32 audio."""
        ...

    @staticmethod
    def ones_mono_i16(length: int, sample_rate: int) -> AudioSamples:
        """Create ones-filled mono i16 audio."""
        ...

    @staticmethod
    def ones_mono_i32(length: int, sample_rate: int) -> AudioSamples:
        """Create ones-filled mono i32 audio."""
        ...

    @staticmethod
    def ones_mono_f64(length: int, sample_rate: int) -> AudioSamples:
        """Create ones-filled mono f64 audio."""
        ...

    @staticmethod
    def ones_multi(channels: int, length: int, sample_rate: int) -> AudioSamples:
        """Create ones-filled multi-channel f32 audio."""
        ...

    @staticmethod
    def ones_multi_i16(channels: int, length: int, sample_rate: int) -> AudioSamples:
        """Create ones-filled multi-channel i16 audio."""
        ...

    @staticmethod
    def ones_multi_i32(channels: int, length: int, sample_rate: int) -> AudioSamples:
        """Create ones-filled multi-channel i32 audio."""
        ...

    @staticmethod
    def ones_multi_f64(channels: int, length: int, sample_rate: int) -> AudioSamples:
        """Create ones-filled multi-channel f64 audio."""
        ...

    # Factory methods - uniform
    @staticmethod
    def uniform_mono(length: int, sample_rate: int, value: float) -> AudioSamples:
        """Create uniform-value mono f32 audio."""
        ...

    @staticmethod
    def uniform_mono_i16(length: int, sample_rate: int, value: int) -> AudioSamples:
        """Create uniform-value mono i16 audio."""
        ...

    @staticmethod
    def uniform_mono_i32(length: int, sample_rate: int, value: int) -> AudioSamples:
        """Create uniform-value mono i32 audio."""
        ...

    @staticmethod
    def uniform_mono_f64(length: int, sample_rate: int, value: float) -> AudioSamples:
        """Create uniform-value mono f64 audio."""
        ...

    @staticmethod
    def uniform_multi(
        channels: int, length: int, sample_rate: int, value: float
    ) -> AudioSamples:
        """Create uniform-value multi-channel f32 audio."""
        ...

    @staticmethod
    def uniform_multi_i16(
        channels: int, length: int, sample_rate: int, value: int
    ) -> AudioSamples:
        """Create uniform-value multi-channel i16 audio."""
        ...

    @staticmethod
    def uniform_multi_i32(
        channels: int, length: int, sample_rate: int, value: int
    ) -> AudioSamples:
        """Create uniform-value multi-channel i32 audio."""
        ...

    @staticmethod
    def uniform_multi_f64(
        channels: int, length: int, sample_rate: int, value: float
    ) -> AudioSamples:
        """Create uniform-value multi-channel f64 audio."""
        ...

    # Audio editing operations
    @classmethod
    def concatenate(cls, segments: list[AudioSamples]) -> AudioSamples:
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
    def stack(cls, sources: list[AudioSamples]) -> AudioSamples:
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
    def repeat(self, count: int) -> AudioSamples:
        """Repeat the audio samples count times."""
        ...

    def trim_silence(self, threshold_db: float) -> AudioSamples:
        """Trim silence from the beginning and end of the audio based on a dB threshold."""
        ...

    def perturb(self, config: "PerturbationConfig") -> AudioSamples:
        """Apply stochastic perturbations and return a new buffer.

        Args:
            config: Configuration describing perturbations to apply

        Returns:
            Perturbed buffer
        """
        ...

    def perturb_in_place(self, config: "PerturbationConfig") -> None:
        """Apply stochastic perturbations in place.

        Args:
            config: Configuration describing perturbations to apply
        """
        ...

    def pad(
        self, pad_start_seconds: float, pad_end_seconds: float, pad_value: float = 0.0
    ) -> AudioSamples:
        """Pad audio with silence or a specific value at the start and/or end."""
        ...

    def pad_samples_right(
        self, target_num_samples: int, pad_value: float
    ) -> AudioSamples:
        """Right-pad the buffer to a fixed sample count.

        Args:
            target_num_samples: Desired total number of samples
            pad_value: Value used for padding

        Returns:
            New buffer with the requested length
        """
        ...

    def pad_to_duration(
        self, target_duration_seconds: float, pad_value: float, pad_side: "PadSide"
    ) -> AudioSamples:
        """Pad the buffer to reach a target duration.

        Args:
            target_duration_seconds: Desired total duration
            pad_value: Value used for padding
            pad_side: Side to apply the padding

        Returns:
            New buffer matching the requested duration
        """
        ...

    def split(self, segment_duration_seconds: float) -> list[AudioSamples]:
        """Split audio into segments of a specified duration."""
        ...

    @classmethod
    def mix(
        cls, sources: list[AudioSamples], weights: Optional[list[float]] = None
    ) -> AudioSamples:
        """Mix multiple audio sources together with optional weights."""
        ...

    def fade_in(self, duration_seconds: float, curve: "FadeCurve") -> None:
        """Apply a fade-in effect in place.

        Args:
            duration_seconds: Fade duration
            curve: Curve controlling the fade shape
        """
        ...

    def fade_out(self, duration_seconds: float, curve: "FadeCurve") -> None:
        """Apply a fade-out effect in place.

        Args:
            duration_seconds: Fade duration
            curve: Curve controlling the fade shape
        """
        ...

    # AudioChannelOps methods
    def pan(self, pan_value: float) -> None:
        """Pan stereo audio. -1.0 = full left, 0.0 = center, 1.0 = full right."""
        ...

    def balance(self, balance: float) -> None:
        """Adjust stereo balance. -1.0 = left only, 0.0 = equal, 1.0 = right only."""
        ...

    # AudioTransforms methods
    def stft(self, params: StftParams) -> StftResult:
        """Compute Short-Time Fourier Transform.

        Args:
            params: Configuration controlling window length, hop size, and window type

        Returns:
            StftResult object containing complex spectrum, frequency bins, and time stamps
        """
        ...

    @classmethod
    def istft(cls, stft_result: "StftResult") -> AudioSamples:
        """Invert a Short-Time Fourier Transform result.

        Args:
            stft_result: Output from `stft` to be reconstructed

        Returns:
            Reconstructed time-domain signal
        """
        ...

    # AudioProcessing methods
    def resample(
        self, target_sample_rate: int, quality: "ResamplingQuality" = ...
    ) -> AudioSamples:
        """Resample audio to target sample rate.

        Args:
            target_sample_rate: Desired output sample rate in hertz
            quality: Quality preset controlling the resampler

        Returns:
            New buffer rendered at the requested sample rate
        """
        ...

    def resample_by_ratio(
        self, ratio: float, quality: "ResamplingQuality" = ...
    ) -> AudioSamples:
        """Resample audio by a multiplicative ratio.

        Args:
            ratio: Factor applied to the original sample rate
            quality: Quality preset controlling the resampler

        Returns:
            New buffer rendered at the scaled sample rate
        """
        ...

    def apply_window(self, window: NDArray[np.floating]) -> AudioSamples:
        """Apply a window function to the audio samples.

        Args:
            window: Sequence of coefficients matching the audio length

        Returns:
            New buffer with the window applied
        """
        ...

    # AudioPitchAnalysis methods
    def detect_pitch_yin(
        self,
        threshold: float = 0.1,
        min_frequency: float = 50.0,
        max_frequency: float = 2000.0,
    ) -> Optional[float]:
        """Detect pitch using the YIN algorithm. Returns frequency in Hz or None."""
        ...

    def detect_pitch_autocorr(
        self,
        min_frequency: float = 50.0,
        max_frequency: float = 2000.0,
    ) -> Optional[float]:
        """Detect pitch using autocorrelation. Returns frequency in Hz or None."""
        ...

    def track_pitch(
        self,
        window_size: int,
        hop_size: int,
        pitch_method: PitchDetectionMethod = ...,
        threshold: float = 0.1,
        min_frequency: float = 50.0,
        max_frequency: float = 2000.0,
    ) -> list[tuple[float, Optional[float]]]:
        """Track pitch over time. Returns list of (time, frequency) tuples.

        Args:
            window_size: Size of each analysis window in samples
            hop_size: Step between windows in samples
            pitch_method: Detection strategy to apply per window
            threshold: Detection threshold interpreted by the selected method
            min_frequency: Minimum expected frequency in Hz
            max_frequency: Maximum expected frequency in Hz

        Returns:
            Sequence of (time_seconds, pitch_hz) pairs
        """
        ...

    def harmonic_to_noise_ratio(
        self,
        fundamental_freq: float,
        num_harmonics: int,
        n_fft: Optional[int] = None,
        window_type: Optional[str] = None,
    ) -> float:
        """Estimate the harmonic-to-noise ratio around a fundamental frequency.

        Args:
            fundamental_freq: Fundamental frequency in Hz
            num_harmonics: Number of harmonics to evaluate
            n_fft: Optional FFT size in samples
            window_type: Optional window type applied before the FFT

        Returns:
            Harmonic-to-noise ratio value
        """
        ...

    def harmonic_analysis(
        self,
        fundamental_freq: float,
        num_harmonics: int,
        tolerance: float,
        n_fft: Optional[int] = None,
        window_type: Optional[str] = None,
    ) -> list[float]:
        """Measure harmonic amplitudes relative to a fundamental frequency.

        Args:
            fundamental_freq: Fundamental frequency in Hz
            num_harmonics: Number of harmonics to report
            tolerance: Frequency tolerance as a fraction of the harmonic
            n_fft: Optional FFT size in samples
            window_type: Optional window type applied before the FFT

        Returns:
            List of harmonic magnitudes ordered by harmonic index
        """
        ...

    def estimate_key(self, stft_params: StftParams) -> tuple[int, float]:
        """Estimate the musical key of the audio signal.

        Args:
            stft_params: Parameters controlling the underlying STFT analysis

        Returns:
            Tuple of (key_index, confidence_score)
        """
        ...

    # AudioDecomposition methods
    def hpss(
        self,
        stft_params: StftParams,
        median_filter_harmonic: int = 17,
        median_filter_percussive: int = 17,
        mask_softness: float = 0.5,
    ) -> tuple[AudioSamples, AudioSamples]:
        """Separate harmonic and percussive components using HPSS.

        Returns:
            tuple: (harmonic_audio, percussive_audio)
        """
        ...

    # AudioOnsetDetection methods
    def detect_onsets(self, config: OnsetDetectionConfig) -> list[float]:
        """Detect onsets in the audio signal using spectral flux method.

        Args:
            config: Onset detection configuration parameters

        Returns:
            List of onset times in seconds
        """
        ...

    def onset_detection_function(
        self, config: OnsetDetectionConfig
    ) -> tuple[list[float], list[float]]:
        """Compute onset detection function (energy-based).

        Returns the onset detection function values and corresponding timestamps.

        Args:
            config: Onset detection configuration parameters

        Returns:
            Tuple of (odf_values, timestamps) where both are lists of floats
        """
        ...

    def detect_onsets_spectral_flux(self, config: SpectralFluxConfig) -> list[float]:
        """Detect onsets using spectral flux.

        Spectral flux measures the rate of change in the spectrum over time.

        Args:
            config: Spectral flux configuration parameters

        Returns:
            List of onset times in seconds
        """
        ...

    def spectral_flux(
        self,
        cqt_params: CqtParams,  # CqtParams from spectrograms module
        window_size: int,
        hop_size: int,
        method: SpectralFluxMethod,
    ) -> tuple[list[float], list[float]]:
        """Compute spectral flux only.

        Returns the spectral flux values without performing onset detection.

        Args:
            cqt_params: CQT parameters for spectral analysis
            window_size: Window size for analysis in samples
            hop_size: Hop size between successive frames in samples
            method: Method for computing spectral flux

        Returns:
            Tuple of (flux_values, timestamps) where both are lists of floats
        """
        ...

    def complex_onset_detection(self, config: "ComplexOnsetConfig") -> list[float]:
        """Complex domain onset detection.

        Uses both magnitude and phase information for more robust onset detection.

        Args:
            config: Complex onset detection configuration parameters

        Returns:
            List of onset times in seconds
        """
        ...

    def onset_detection_function_complex(
        self, config: ComplexOnsetConfig
    ) -> list[float]:
        """Compute complex domain onset detection function.

        Returns the complex ODF values without performing peak picking.

        Args:
            config: Complex onset detection configuration parameters

        Returns:
            List of complex onset detection function values
        """
        ...

    def magnitude_difference_matrix(
        self, config: ComplexOnsetConfig
    ) -> NDArray[np.float64]:
        """Compute magnitude difference matrix for complex domain onset detection.

        Returns a 2D matrix showing magnitude changes across time and frequency.

        Args:
            config: Complex onset detection configuration parameters

        Returns:
            2D numpy array representing the magnitude difference matrix
        """
        ...

    def phase_deviation_matrix(self, config: ComplexOnsetConfig) -> NDArray[np.float64]:
        """Compute phase deviation matrix for complex domain onset detection.

        Returns a 2D matrix showing phase changes across time and frequency.

        Args:
            config: Complex onset detection configuration parameters

        Returns:
            2D numpy array representing the phase deviation matrix
        """
        ...

    # AudioEnvelopes methods
    def amplitude_envelope(self) -> NDArray[np.floating]:
        """Compute a rectified amplitude envelope.

        Takes the absolute value of each sample to create an amplitude envelope.

        Returns:
            Amplitude envelope (1D for mono, 2D for multi-channel)
        """
        ...

    def rms_envelope(self, window_size: int, hop_size: int) -> NDArray[np.floating]:
        """Compute RMS (root mean square) envelope using a sliding window.

        Args:
            window_size: Number of samples per analysis window
            hop_size: Number of samples to advance between successive windows

        Returns:
            RMS envelope (1D for mono, 2D for multi-channel)
        """
        ...

    def attack_decay_envelope(
        self, follower: EnvelopeFollower, method: DynamicRangeMethod
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Compute attack and decay envelopes using an envelope follower.

        Args:
            follower: Configured envelope follower defining attack/release behavior
            method: Detection strategy (Peak, Rms, or Hybrid)

        Returns:
            Tuple of (attack_envelope, decay_envelope) as numpy arrays
        """
        ...

    def analytic_envelope(self) -> NDArray[np.floating]:
        """Compute analytic amplitude envelope using Hilbert transform.

        Provides the instantaneous amplitude for each sample.

        Returns:
            Analytic envelope (1D for mono, 2D for multi-channel)
        """
        ...

    def moving_average_envelope(
        self, window_size: int, hop_size: int
    ) -> NDArray[np.floating]:
        """Compute moving-average envelope over the rectified signal.

        Args:
            window_size: Number of samples included in each moving-average window
            hop_size: Number of samples between successive window starts

        Returns:
            Moving-average envelope (1D for mono, 2D for multi-channel)
        """
        ...

    # AudioBeatTracking methods
    def detect_beats(self, config: BeatTrackingConfig) -> BeatTrackingData:
        """Detect beats in the audio signal.

        Uses onset detection and dynamic programming to identify beat positions
        based on a target tempo.

        Args:
            config: Beat tracking configuration parameters

        Returns:
            BeatTrackingData containing tempo, beat times, and configuration
        """
        ...

    # AudioVoiceActivityDetection methods
    def voice_activity_mask(self, config: VadConfig) -> list[bool]:
        """Compute a per-frame speech activity mask.

        Returns a boolean mask where True indicates speech/voice activity
        and False indicates silence/non-speech.

        Args:
            config: Voice activity detection configuration parameters

        Returns:
            List of boolean values, one per frame

        Examples:
            >>> from audio_samples import AudioSamples, VadConfig
            >>> audio = AudioSamples.read("speech.wav")
            >>> config = VadConfig.default()
            >>> mask = audio.voice_activity_mask(config)
            >>> speech_frames = sum(mask)
            >>> print(f"{speech_frames}/{len(mask)} frames contain speech")
        """
        ...

    def speech_regions(self, config: VadConfig) -> list[tuple[int, int]]:
        """Compute contiguous speech regions as (start_sample, end_sample) pairs.

        Analyzes the audio to find continuous regions of speech/voice activity.
        Each region is represented as a tuple of (start_sample, end_sample) where
        indices are in samples-per-channel units and end_sample is exclusive.

        Args:
            config: Voice activity detection configuration parameters

        Returns:
            List of (start, end) tuples representing speech regions in samples

        Examples:
            >>> from audio_samples import AudioSamples, VadConfig
            >>> audio = AudioSamples.read("speech.wav")
            >>> config = VadConfig.default()
            >>> regions = audio.speech_regions(config)
            >>> for start, end in regions:
            >>>     duration = (end - start) / audio.sample_rate_hz()
            >>>     print(f"Speech: {start}-{end} samples ({duration:.2f}s)")
        """
        ...

    # AudioPlotting methods
    def plot_waveform(
        self, params: Optional[WaveformPlotParams] = None
    ) -> WaveformPlot:
        """Plot the waveform of the audio signal.

        Creates an interactive time-domain waveform visualization using Plotly.
        The resulting plot can be saved as HTML or static images (PNG, SVG, etc.).

        Args:
            params: Optional waveform plot configuration parameters

        Returns:
            WaveformPlot object with html(), save(), and show() methods

        Examples:
            >>> from audio_samples import AudioSamples, WaveformPlotParams
            >>> audio = AudioSamples.read("audio.wav")
            >>> # Default plot
            >>> plot = audio.plot_waveform()
            >>> plot.save("waveform.html")
            >>> # Custom plot
            >>> params = WaveformPlotParams(title="My Audio", color="blue")
            >>> plot = audio.plot_waveform(params)
            >>> plot.save("waveform.png")
        """
        ...

    def plot_spectrogram(
        self, params: Optional[SpectrogramPlotParams] = None
    ) -> "SpectrogramPlot":
        """Plot the spectrogram of the audio signal.

        Creates an interactive time-frequency spectrogram visualization using Plotly.
        Shows how the frequency content of the audio changes over time.

        Args:
            params: Optional spectrogram plot configuration parameters

        Returns:
            SpectrogramPlot object with html(), save(), and show() methods

        Examples:
            >>> from audio_samples import AudioSamples, SpectrogramPlotParams
            >>> audio = AudioSamples.read("audio.wav")
            >>> # Default plot
            >>> plot = audio.plot_spectrogram()
            >>> plot.save("spectrogram.html")
            >>> # Custom plot
            >>> params = SpectrogramPlotParams(title="Audio Spectrogram")
            >>> plot = audio.plot_spectrogram(params)
        """
        ...

    def plot_magnitude_spectrum(
        self, params: Optional["MagnitudeSpectrumParams"] = None
    ) -> "MagnitudeSpectrumPlot":
        """Plot the magnitude spectrum of the audio signal.

        Creates an interactive frequency-domain magnitude spectrum visualization.
        Shows the frequency content (amplitude vs frequency) of the audio.

        Args:
            params: Optional magnitude spectrum configuration parameters

        Returns:
            MagnitudeSpectrumPlot object with html(), save(), and show() methods

        Examples:
            >>> from audio_samples import AudioSamples, MagnitudeSpectrumParams
            >>> audio = AudioSamples.read("audio.wav")
            >>> # Default plot
            >>> plot = audio.plot_magnitude_spectrum()
            >>> plot.save("spectrum.html")
            >>> # Custom plot with higher resolution
            >>> params = MagnitudeSpectrumParams(title="Frequency Spectrum", n_fft=4096)
            >>> plot = audio.plot_magnitude_spectrum(params)
        """
        ...

    def time_axis(self) -> NDArray[np.floating]:
        """Get the time axis values corresponding to each sample.

        A linspace from 0 to the total duration of the audio, with one value per sample. Useful for plotting or time-based analysis. Returns: 1D numpy array of time values in seconds"""
        ...

    # IIR Filtering methods
    def butterworth_lowpass(self, order: int, cutoff_frequency: float) -> None:
        """Apply a Butterworth low-pass filter. Higher order = steeper rolloff."""
        ...

    def butterworth_highpass(self, order: int, cutoff_frequency: float) -> None:
        """Apply a Butterworth high-pass filter. Higher order = steeper rolloff."""
        ...

    def butterworth_bandpass(
        self, order: int, low_frequency: float, high_frequency: float
    ) -> None:
        """Apply a Butterworth band-pass filter. Higher order = steeper rolloff."""
        ...

    # Simple filter methods
    def low_pass_filter(self, cutoff_hz: float) -> AudioSamples:
        """Apply a simple low-pass filter. Attenuates frequencies above cutoff.

        Args:
            cutoff_hz: Frequency threshold above which content is attenuated

        Returns:
            New buffer containing the filtered signal
        """
        ...

    def high_pass_filter(self, cutoff_hz: float) -> AudioSamples:
        """Apply a simple high-pass filter. Attenuates frequencies below cutoff.

        Args:
            cutoff_hz: Frequency threshold below which content is attenuated

        Returns:
            New buffer containing the filtered signal
        """
        ...

    def band_pass_filter(
        self, low_cutoff_hz: float, high_cutoff_hz: float
    ) -> AudioSamples:
        """Apply a simple band-pass filter. Passes frequencies between cutoffs.

        Args:
            low_cutoff_hz: Lower frequency bound to retain
            high_cutoff_hz: Upper frequency bound to retain

        Returns:
            New buffer containing the filtered signal
        """
        ...

    # Statistical methods
    def peak(self) -> int | float:
        """Get the peak (absolute maximum) sample value."""
        ...

    def min(self) -> int | float:
        """Get the minimum sample value."""
        ...

    def max(self) -> int | float:
        """Get the maximum sample value."""
        ...

    def mean(self) -> float:
        """Calculate the mean sample value."""
        ...

    def median(self) -> Optional[float]:
        """Calculate the median sample value."""
        ...

    def rms(self) -> float:
        """Calculate the RMS (root mean square) value."""
        ...

    def variance(self) -> float:
        """Calculate the variance of sample values."""
        ...

    def std_dev(self) -> float:
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

    def cross_correlation(self, other: AudioSamples, max_lag: int) -> list[float]:
        """Calculate normalized cross-correlation with another audio buffer.

        Args:
            other: Audio buffer to correlate against
            max_lag: Maximum lag to include

        Returns:
            Cross-correlation coefficients up to the requested lag

        Raises:
            ValueError: If max_lag is zero or sample rates/layouts differ
            TypeError: If buffers have incompatible dtypes
        """
        ...

    def spectral_centroid(self) -> float:
        """Calculate the spectral centroid."""
        ...

    def spectral_rolloff(self, rolloff_percent: float) -> float:
        """Calculate spectral rolloff at given percentage."""
        ...

    # Amplitude processing
    def scale(self, factor: float) -> AudioSamples:
        """Scale audio amplitude by factor.

        Args:
            factor: Multiplier applied to every sample

        Returns:
            New buffer with scaled amplitudes
        """
        ...

    def normalize(
        self, min_val: float, max_val: float, method: "NormalizationMethod"
    ) -> AudioSamples:
        """Normalize audio to given range using specified method.

        Args:
            min_val: Lower bound for the normalized output
            max_val: Upper bound for the normalized output
            method: Strategy controlling the normalization

        Returns:
            New buffer containing the normalized samples
        """
        ...

    def clip(self, min_val: float, max_val: float) -> AudioSamples:
        """Clip audio values to given range.

        Args:
            min_val: Lower bound for clamping
            max_val: Upper bound for clamping

        Returns:
            New buffer with clamped amplitudes
        """
        ...

    def remove_dc_offset(self) -> AudioSamples:
        """Remove DC offset from audio.

        Returns:
            New buffer with the mean offset removed
        """
        ...

    def mu_compress(self, mu: float) -> AudioSamples:
        """Apply mu-law compression to the signal.

        Args:
            mu: Mu parameter controlling the compression curve

        Returns:
            New buffer containing the compressed samples
        """
        ...

    def mu_expand(self, mu: float) -> AudioSamples:
        """Apply mu-law expansion to the signal.

        Args:
            mu: Mu parameter used during compression

        Returns:
            New buffer containing the expanded samples
        """
        ...

    # Time-domain processing
    def reverse(self) -> AudioSamples:
        """Return a reversed copy of the audio."""
        ...

    def reverse_in_place(self) -> None:
        """Reverse the audio samples in place."""
        ...

    def trim(self, start_seconds: float, end_seconds: float) -> AudioSamples:
        """Return a trimmed copy of the audio."""
        ...

    # Dynamic range processing (in-place)
    def apply_compressor(
        self,
        threshold_db: float,
        ratio: float,
        attack_ms: float,
        release_ms: float,
        makeup_gain_db: float,
    ) -> None:
        """Apply compression (in-place)."""
        ...

    def apply_limiter(
        self,
        ceiling_db: float,
        release_ms: float,
        lookahead_ms: float,
    ) -> None:
        """Apply limiting (in-place)."""
        ...

    def apply_gate(
        self,
        threshold_db: float,
        ratio: float,
        attack_ms: float,
        release_ms: float,
    ) -> None:
        """Apply noise gate (in-place)."""
        ...

    def apply_expander(
        self,
        threshold_db: float,
        ratio: float,
        attack_ms: float,
        release_ms: float,
    ) -> None:
        """Apply expander (in-place)."""
        ...

    # Filtering (in-place)
    def apply_iir_filter(self, design: IirFilterDesign) -> None:
        """Apply IIR filter using filter design (in-place)."""
        ...

    def apply_butterworth_lowpass(self, order: int, cutoff_frequency: float) -> None:
        """Apply Butterworth lowpass filter (in-place)."""
        ...

    def apply_butterworth_highpass(self, order: int, cutoff_frequency: float) -> None:
        """Apply Butterworth highpass filter (in-place)."""
        ...

    def apply_butterworth_bandpass(
        self,
        order: int,
        low_frequency: float,
        high_frequency: float,
    ) -> None:
        """Apply Butterworth bandpass filter (in-place)."""
        ...

    def apply_chebyshev_i(
        self,
        order: int,
        cutoff_frequency: float,
        passband_ripple: float,
        response: FilterResponse,
    ) -> None:
        """Apply Chebyshev Type I filter (in-place)."""
        ...

    # Equalization (in-place)
    def apply_parametric_eq(self, eq: ParametricEq) -> None:
        """Apply parametric equalizer (in-place)."""
        ...

    def apply_eq_band(self, band: EqBand) -> None:
        """Apply single EQ band (in-place)."""
        ...

    def apply_peak_filter(
        self, frequency: float, gain_db: float, q_factor: float
    ) -> None:
        """Apply peak filter (in-place)."""
        ...

    def apply_low_shelf(
        self, frequency: float, gain_db: float, q_factor: float
    ) -> None:
        """Apply low shelf filter (in-place)."""
        ...

    def apply_high_shelf(
        self, frequency: float, gain_db: float, q_factor: float
    ) -> None:
        """Apply high shelf filter (in-place)."""
        ...

    def apply_three_band_eq(
        self,
        low_freq: float,
        low_gain: float,
        mid_freq: float,
        mid_gain: float,
        mid_q: float,
        high_freq: float,
        high_gain: float,
    ) -> None:
        """Apply three-band equalizer (in-place)."""
        ...

    # Frequency analysis
    def frequency_response(
        self, frequencies: list[float]
    ) -> tuple[list[float], list[float]]:
        """Calculate frequency response at given frequencies."""
        ...

    def fft(self, n_fft: int) -> np.typing.NDArray[np.complexfloating]:
        """Calculate FFT.

        Args:
            n_fft: FFT size in samples

        Returns:
            Complex spectrum shaped (channels, frequency_bins)
        """
        ...

    def rfft(self, n_fft: int) -> np.typing.NDArray[np.float64]:
        """Calculate real-valued FFT.

        Args:
            n_fft: FFT size in samples

        Returns:
            Real-valued spectrum shaped (channels, frequency_bins)
        """
        ...

    def power_spectral_density(
        self, window_size: int = 2048, overlap: float = 0.5
    ) -> np.typing.NDArray[np.float64]:
        """Calculate power spectral density."""
        ...

    def mel_spectrogram(
        self,
        spec_params: SpectrogramParams,
        mel_params: MelParams,
        log_params: Optional[LogParams] = None,
    ) -> NDArray[np.float64]:
        """
        Compute mel-frequency spectrogram.

        Args
        ----
            :param spec_params (SpectrogramParams): Spectrogram parameters (STFT config, sample rate)
            :param mel_params (MelParams): Mel filterbank parameters
            :param log_params (Optional[LogParams]): Optional log scale parameters for dB conversion
            :return numpy.ndarray: Mel spectrogram array
        """
        ...

    def gammatone_spectrogram(
        self,
        spec_params: SpectrogramParams,
        erb_params: ErbParams,
        log_params: Optional[LogParams] = None,
    ) -> NDArray[np.float64]:
        """
        Compute a gammatone spectrogram using ERB-scaled filters.

        Args
        ----
            :param spec_params (SpectrogramParams): Spectrogram parameters (STFT config, sample rate)
            :param erb_params (ErbParams): Configuration for the ERB filterbank
            :param log_params (Optional[LogParams]): Optional log scale parameters for dB conversion
            :return numpy.ndarray: Gammatone spectrogram array
        """
        ...

    def mfcc(
        self, stft_params: StftParams, n_mels: int, mfcc_params: MfccParams
    ) -> NDArray[np.float64]:
        """Calculate MFCC (Mel-Frequency Cepstral Coefficients).

        Args
        ----
            :param stft_params (StftParams): STFT parameters
            :param n_mels (int): Number of mel bands
            :param mfcc_params (MfccParams): MFCC computation parameters
            :return numpy.ndarray: MFCC feature array
        """
        ...

    def chroma(
        self, stft_params: StftParams, chroma_params: ChromaParams
    ) -> NDArray[np.float64]:
        """Calculate chroma features.

        Args
        ----
            :param stft_params (StftParams): STFT parameters
            :param chroma_params (ChromaParams): Chroma feature parameters
            :return numpy.ndarray: Chroma feature array (12 x n_frames for 12 pitch classes)
        """
        ...

    def to_format(self, dtype: SampleType) -> AudioSamples:
        """Convert audio samples to specified dtype (e.g. i16, f32)."""
        ...

    def as_u8(self) -> AudioSamples:
        """Convert audio samples to unsigned 8-bit integer format."""
        ...

    def as_i16(self) -> AudioSamples:
        """Convert audio samples to signed 16-bit integer format."""
        ...

    def as_i32(self) -> AudioSamples:
        """Convert audio samples to signed 32-bit integer format."""
        ...

    def as_f32(self) -> AudioSamples:
        """Convert audio samples to 32-bit floating point format."""
        ...

    def as_f64(self) -> AudioSamples:
        """Convert audio samples to 64-bit floating point format."""
        ...

    def cast_as(self, dtype: SampleType) -> AudioSamples:
        """Casts audio samples to specified dtype (e.g. i16, f32)."""
        ...

    def cast_as_u8(self) -> AudioSamples:
        """Cast audio samples to unsigned 8-bit integer format."""
        ...

    def cast_as_i16(self) -> AudioSamples:
        """Cast audio samples to signed 16-bit integer format."""
        ...

    def cast_as_i32(self) -> AudioSamples:
        """Cast audio samples to signed 32-bit integer format."""
        ...

    def cast_as_f32(self) -> AudioSamples:
        """Cast audio samples to 32-bit floating point format."""
        ...

    def cast_as_f64(self) -> AudioSamples:
        """Cast audio samples to 64-bit floating point format."""
        ...

    # Channel manipulation
    def to_mono(
        self, method: MonoConversionMethod = MonoConversionMethod.average
    ) -> AudioSamples:
        """Convert to mono using specified method.

        Args:
            method: Conversion method describing how to combine channels

        Returns:
            New buffer with a single channel
        """
        ...

    def to_stereo(
        self, method: StereoConversionMethod = StereoConversionMethod.duplicate
    ) -> AudioSamples:
        """Convert to stereo using specified method.

        Args:
            method: Conversion method describing how to expand channels

        Returns:
            New buffer with two channels
        """
        ...

    def duplicate_to_channels(self, n_channels: int) -> AudioSamples:
        """Duplicate the buffer into the requested number of channels.

        Args:
            n_channels: Target channel count

        Returns:
            New buffer with the duplicated channels
        """
        ...

    def extract_channel(self, channel_index: int) -> AudioSamples:
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
    @property
    def sample_rate(self) -> int:
        """Get the sample rate in Hz."""
        ...

    @property
    def num_channels(self) -> int:
        """Get the number of audio channels."""
        ...

    def copy(self) -> AudioSamples:
        """Create a deep copy of the AudioSamples."""
        ...

    def __len__(self) -> int:
        """Get the length of the audio (total number of samples)."""
        ...

    def __getitem__(self, index: Any) -> AudioSamples:
        """Get a slice of the audio samples."""
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

    duration_seconds: float
    """Get the duration in seconds."""

    def to_numpy(self) -> np.typing.NDArray:
        """Convert AudioSamples to a numpy ndarray."""

    def to_tensor(self, device: Optional[str] = None) -> "torch.Tensor":  # type: ignore # noqa: F821
        """Convert AudioSamples to a PyTorch tensor."""

    def to_gpu_tensor(self, gpu_id: Optional[str] = None) -> "torch.Tensor":  # type: ignore # noqa: F821
        """Convert AudioSamples to a PyTorch tensor on the GPU if possible.

        If passing a ``gpu_id`` argument provide the full device name and id. e.g. cuda:1
        """

    # Arithmetic operators
    def __add__(self, other: AudioSamples | NDArray[Any] | float) -> AudioSamples:
        """Add AudioSamples with another AudioSamples, numpy array, or scalar."""
        ...

    def __sub__(self, other: AudioSamples | NDArray[Any] | float) -> AudioSamples:
        """Subtract AudioSamples, numpy array, or scalar from AudioSamples."""
        ...

    def __mul__(self, other: NDArray[Any] | float | AudioSamples) -> AudioSamples:
        """Multiply AudioSamples by a numpy array or scalar."""
        ...

    def __truediv__(self, other: NDArray[Any] | float) -> AudioSamples:
        """Divide AudioSamples by a numpy array or scalar."""
        ...

    def __pow__(self, exponent: float, modulo: Optional[float] = None) -> AudioSamples:
        """Raise AudioSamples to a power (float types only)."""
        ...

    # Reverse arithmetic operators (for scalar on left side)
    def __radd__(self, scalar: float) -> AudioSamples:
        """Add scalar to AudioSamples (scalar + audio)."""
        ...

    def __rmul__(self, scalar: float) -> AudioSamples:
        """Multiply scalar with AudioSamples (scalar * audio)."""
        ...

    def __rsub__(self, scalar: float) -> AudioSamples:
        """Subtract AudioSamples from scalar (scalar - audio)."""
        ...

    def __rtruediv__(self, scalar: float) -> AudioSamples:
        """Divide scalar by AudioSamples (scalar / audio) - not implemented."""
        ...

    # In-place arithmetic operators
    def __iadd__(self, other: AudioSamples | NDArray[Any] | float) -> None:
        """In-place addition with AudioSamples, numpy array, or scalar."""
        ...

    def __isub__(self, other: AudioSamples | NDArray[Any] | float) -> None:
        """In-place subtraction with AudioSamples, numpy array, or scalar."""
        ...

    def __imul__(self, other: NDArray[Any] | float) -> None:
        """In-place multiplication with numpy array or scalar."""
        ...

    def __itruediv__(self, other: NDArray[Any] | float) -> None:
        """In-place division by numpy array or scalar."""
        ...

    def __str__(self) -> str:
        """String representation of the audio samples."""
        ...

class IirFilterDesign:
    """IIR (Infinite Impulse Response) filter design."""

    def __init__(
        self,
        filter_type: str,
        response: str,
        order: int,
        cutoff_frequency: Optional[float] = None,
        low_frequency: Optional[float] = None,
        high_frequency: Optional[float] = None,
    ) -> None:
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

    def __init__(
        self, band_type: EqBandType, frequency: float, gain_db: float, q_factor: float
    ) -> None:
        """
        Create EQ band.

        Args:
            band_type: defines how gain is applied across the frequency spectrum relative to a centre or cutoff frequency.
            frequency: Center/cutoff frequency in Hz
            gain_db: Gain in dB
            q_factor: Q factor (bandwidth control)

        Available band types:
            - ``EqBandType.peak``
            - ``EqBandType.low_shelf``
            - ``EqBandType.high_shelf``
            - ``EqBandType.low_pass``
            - ``EqBandType.high_pass``
            - ``EqBandType.band_pass``
            - ``EqBandType.band_stop``
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

class OnsetDetectionConfig:
    """Configuration for onset detection."""

    def __init__(
        self,
        cqt_params: CqtParams,
        hop_size: int,
        window_size: int,
        threshold: float,
        min_onset_interval_seconds: float,
        pre_emphasis: float,
        adaptive_threshold: bool,
        median_filter_length: int,
        adapative_threshold_multiplier: float,
        peak_picking: PeakPickingConfig,
    ) -> None:
        """
        Create onset detection configuration.

        Args:
        """
        ...

    @classmethod
    def default(cls) -> OnsetDetectionConfig:
        """Create default onset detection configuration."""
        ...

    @classmethod
    def musical(cls) -> OnsetDetectionConfig:
        """Preset optimized for musical content."""
        ...

    @classmethod
    def percussive(cls) -> OnsetDetectionConfig:
        """Preset optimized for percussive content."""
        ...

    @classmethod
    def speech(cls) -> OnsetDetectionConfig:
        """Preset optimized for speech."""
        ...

class ComplexOnsetConfig:
    """Configuration for complex domain onset detection."""

    def __init__(
        self,
        window_size: int = 2048,
        hop_size: int = 512,
        threshold: float = 0.3,
        alpha: float = 0.5,
    ) -> None:
        """
        Create complex onset detection configuration.

        Args:
            window_size: FFT window size in samples
            hop_size: Hop size between analysis frames in samples
            threshold: Peak picking threshold
            alpha: Weight between magnitude and phase (0.0-1.0)
        """
        ...

    @classmethod
    def default(cls) -> ComplexOnsetConfig:
        """Create default configuration."""
        ...

    @classmethod
    def music(cls) -> ComplexOnsetConfig:
        """Preset optimized for music."""
        ...

    @classmethod
    def speech(cls) -> ComplexOnsetConfig:
        """Preset optimized for speech."""
        ...

class SpectralFluxConfig:
    """Configuration for spectral flux onset detection."""

    def __init__(
        self,
        cqt_params: CqtParams,
        hop_size: int,
        window_size: int,
        flux_method: SpectralFluxMethod,
        peak_picking: PeakPickingConfig,
        rectify: bool,
        log_compression: float,
    ) -> None:
        """
        Create spectral flux configuration.

        Args:
            cqt_params: CQT parameters for spectral analysis
            hop_size: Hop size between analysis frames in samples
            window_size: FFT window size in samples
            flux_method: Method for computing spectral flux
            peak_picking: Configuration for peak picking after flux calculation
            rectify: Whether to rectify the spectral flux values before peak picking
            log_compression: Amount of log compression to apply to the flux values (0 = none)
        """
        ...

    @classmethod
    def default(cls) -> SpectralFluxConfig:
        """Create default configuration."""
        ...

class SpectralFluxMethod:
    """Method for computing spectral flux."""

    energy: SpectralFluxMethod
    """Use energy-based spectral flux."""

    magnitude: SpectralFluxMethod
    """Use magnitude-based spectral flux."""

    complex_domain: SpectralFluxMethod
    """Use complex domain spectral flux."""

class DynamicRangeMethod:
    """Detection method for dynamic range processing."""

    peak: DynamicRangeMethod
    """Use peak detection (absolute maximum)."""

    rms: DynamicRangeMethod
    """Use RMS (root mean square) detection."""

    @staticmethod
    def hybrid(peak_weight: float = 0.5) -> DynamicRangeMethod:
        """Use hybrid detection combining peak and RMS.

        Args:
            peak_weight: Weight for peak vs RMS (0.0-1.0)
        """
        ...

class EnvelopeFollower:
    """Envelope follower for attack/release envelope tracking."""

    def __init__(
        self,
        attack_ms: float,
        release_ms: float,
        sample_rate: float,
        detection_method: DynamicRangeMethod,
    ) -> None:
        """
        Create envelope follower.

        Args:
            attack_ms: Attack time in milliseconds
            release_ms: Release time in milliseconds
            sample_rate: Sample rate in Hz
            detection_method: Detection method (Peak, Rms, or Hybrid)
        """
        ...

    @classmethod
    def default(cls) -> "EnvelopeFollower":
        """Create default envelope follower (10ms attack, 100ms release at 44100 Hz)."""
        ...

class BeatTrackingConfig:
    """Configuration for beat detection."""

    def __init__(
        self,
        tempo_bpm: float,
        onset_config: OnsetDetectionConfig,
        tolerance: Optional[float] = None,
    ) -> None:
        """
        Create beat tracking configuration.

        Args:
            tempo_bpm: Target tempo in beats per minute
            onset_config: Configuration for onset detection
            tolerance: Beat timing tolerance in seconds (defaults to 10% of inter-beat interval)
        """
        ...

class BeatTrackingData:
    """Beat tracking results containing tempo and beat timestamps."""

    def __init__(
        self, tempo_bpm: float, beat_times: list[float], config: BeatTrackingConfig
    ) -> None:
        """
        Create beat tracking data.

        Args:
            tempo_bpm: Estimated tempo in beats per minute
            beat_times: Beat timestamps in seconds
            config: Configuration used for beat detection
        """
        ...

    @property
    def tempo_bpm(self) -> float:
        """Estimated tempo in BPM."""
        ...

    @property
    def beat_times(self) -> list[float]:
        """Beat timestamps in seconds."""
        ...

    @property
    def config(self) -> BeatTrackingConfig:
        """Configuration used."""
        ...

class VadMethod:
    """VAD method for voice activity detection."""

    energy: VadMethod
    """Simple energy-based detection using RMS threshold."""

    zero_crossing: VadMethod
    """Zero crossing rate based detection."""

    combined: VadMethod
    """Combined energy and zero crossing rate."""

    spectral: VadMethod
    """Spectral-based detection using spectral features."""

    default: VadMethod
    """Default VAD method (Energy)."""

class VadChannelPolicy:
    """Multi-channel handling policy for Voice Activity Detection."""

    average_to_mono: VadChannelPolicy
    """Average all channels to a mono signal and run VAD once (default)."""

    any_channel: VadChannelPolicy
    """Run VAD per-channel and mark speech if any channel is active."""

    all_channels: VadChannelPolicy
    """Run VAD per-channel and mark speech only if all channels are active."""

    default: VadChannelPolicy
    """Default channel policy (AverageToMono)."""

    @staticmethod
    def channel(index: int) -> VadChannelPolicy:
        """Run VAD on a specific channel index."""
        ...

class VadConfig:
    """Configuration for Voice Activity Detection (VAD).

    The VAD implementation is frame-based: it produces a boolean decision per frame
    of length `frame_size` with step `hop_size`.

    Defaults are chosen to work reasonably well for general audio, but you should
    tune thresholds for your content and sample format.
    """

    def __init__(
        self,
        method: VadMethod,
        frame_size: int,
        hop_size: int,
        pad_end: bool,
        channel_policy: VadChannelPolicy,
        energy_threshold_db: float,
        zcr_min: float,
        zcr_max: float,
        min_speech_frames: int,
        min_silence_frames: int,
        hangover_frames: int,
        smooth_frames: int,
        speech_band_low_hz: float,
        speech_band_high_hz: float,
        spectral_ratio_threshold: float,
    ) -> None:
        """
        Create VAD configuration.

        Args:
            method: VAD method to use
            frame_size: Frame size in samples
            hop_size: Hop size in samples (frame step)
            pad_end: Whether to include a final partial frame (zero-padded)
            channel_policy: Policy for multi-channel audio
            energy_threshold_db: Energy threshold in dBFS (RMS). Typical: -60.0 to -30.0
            zcr_min: Minimum acceptable zero crossing rate in [0, 1]
            zcr_max: Maximum acceptable zero crossing rate in [0, 1]
            min_speech_frames: Minimum consecutive speech frames to keep a speech region
            min_silence_frames: Minimum consecutive non-speech frames to keep silence
            hangover_frames: Keep speech active for this many frames after energy drops
            smooth_frames: Majority-vote smoothing window in frames (1 = no smoothing)
            speech_band_low_hz: Lower bound of speech band (for Spectral method)
            speech_band_high_hz: Upper bound of speech band (for Spectral method)
            spectral_ratio_threshold: Threshold on speech-band energy ratio (for Spectral method)
        """
        ...

    @classmethod
    def default(cls) -> VadConfig:
        """Create default VAD configuration (Energy method, 1024 frame size, 512 hop)."""
        ...

    @classmethod
    def energy_only(cls) -> VadConfig:
        """Create a VAD configuration using only energy-based detection."""
        ...

    def validate(self) -> VadConfig:
        """Validate configuration parameters.

        Returns:
            Self if valid

        Raises:
            ValueError: If parameters are invalid
        """
        ...

class Layout:
    """Plot layout direction for multi-channel visualizations."""

    vertical: Layout
    """Vertical layout (stacked plots)."""

    horizontal: Layout
    """Horizontal layout (side-by-side plots)."""

    @staticmethod
    def default() -> Layout:
        """Default layout (Vertical)."""
        ...

class ChannelManagementStrategy:
    """Strategy for handling multi-channel audio in plots."""

    average: ChannelManagementStrategy
    """Average all channels together into a single plot."""

    separate: ChannelManagementStrategy
    """Plot channels separately with specified layout."""

    first: ChannelManagementStrategy
    """Plot only the first channel."""

    last: ChannelManagementStrategy
    """Plot only the last channel."""

    overlap: ChannelManagementStrategy
    """Overlay all channels on the same plot."""

    @classmethod
    def default(cls) -> ChannelManagementStrategy:
        """Default strategy (Separate with Vertical layout)."""
        ...

class WaveformPlotParams:
    """Configuration parameters for waveform plots."""

    def __init__(
        self,
        title: Optional[str] = None,
        channel_strategy: Optional[ChannelManagementStrategy] = None,
        color: Optional[str] = None,
        line_width: Optional[float] = None,
        markers: bool = False,
    ) -> None:
        """
        Create waveform plot parameters.

        Args:
            title: Plot title
            channel_strategy: How to handle multi-channel audio
            color: Line color (CSS color string)
            line_width: Line width in pixels
            markers: Whether to show markers at each sample point
        """
        ...

    @classmethod
    def default(cls) -> WaveformPlotParams:
        """Create default waveform plot parameters."""
        ...

class SpectrogramPlotParams:
    """Configuration parameters for spectrogram plots."""

    def __init__(self, title: Optional[str] = None) -> None:
        """
        Create spectrogram plot parameters.

        Args:
            title: Plot title
        """
        ...

    @staticmethod
    def default() -> SpectrogramPlotParams:
        """Create default spectrogram plot parameters."""
        ...

class MagnitudeSpectrumParams:
    """Configuration parameters for magnitude spectrum plots."""

    def __init__(
        self, title: Optional[str] = None, n_fft: Optional[int] = None
    ) -> None:
        """
        Create magnitude spectrum parameters.

        Args:
            title: Plot title
            n_fft: FFT size (default: 2048)
        """
        ...

    @classmethod
    def default(cls) -> MagnitudeSpectrumParams:
        """Create default magnitude spectrum parameters."""
        ...

class WaveformPlot:
    """Waveform plot result with methods for display and export."""

    def html(self) -> str:
        """Get HTML representation of the plot.

        Returns:
            HTML string containing the interactive Plotly plot
        """
        ...

    def save(self, path: str | Path) -> None:
        """Save plot to file.

        Supports multiple formats based on file extension:
        - .html: Interactive HTML
        - .png: Static PNG image
        - .svg: Static SVG image
        - .jpg/.jpeg: Static JPEG image
        - .webp: Static WebP image

        Args:
            path: Output file path with extension
        """
        ...

    def show(self) -> None:
        """Show plot in browser .

        Opens the plot.
        """
        ...

class SpectrogramPlot:
    """Spectrogram plot result with methods for display and export."""

    def html(self) -> str:
        """Get HTML representation of the plot.

        Returns:
            HTML string containing the interactive Plotly plot
        """
        ...

    def save(self, path: str) -> None:
        """Save plot to file.

        Supports multiple formats based on file extension:
        - .html: Interactive HTML
        - .png: Static PNG image
        - .svg: Static SVG image
        - .jpg/.jpeg: Static JPEG image
        - .webp: Static WebP image

        Args:
            path: Output file path with extension
        """
        ...

    def show(self) -> None:
        """Show plot."""
        ...

class MagnitudeSpectrumPlot:
    """Magnitude spectrum plot result with methods for display and export."""

    def html(self) -> str:
        """Get HTML representation of the plot.

        Returns:
            HTML string containing the interactive Plotly plot
        """
        ...

    def save(self, path: str | Path) -> None:
        """Save plot to file.

        Supports multiple formats based on file extension:
        - .html: Interactive HTML
        - .png: Static PNG image
        - .svg: Static SVG image
        - .jpg/.jpeg: Static JPEG image
        - .webp: Static WebP image

        Args:
            path: Output file path with extension
        """
        ...

    def show(self) -> None:
        """Show plot."""
        ...

def concatenate(signals: list[AudioSamples]) -> AudioSamples:
    """Concatenates multiple audio segments into one signal"""
    ...

def stack(signals: list[AudioSamples]) -> AudioSamples:
    """Concatenates multiple mono audio segments into one stacked multi-channel signal"""
    ...
