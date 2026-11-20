"""
Python bindings for the AudioSamples Rust crates.
An ``AudioSamples`` object stores its sample rate, channel
layout and sample type together with the samples, so these
properties do not have to be passed separately to each
function.
"""

from __future__ import annotations

from typing import Sequence
from typing import final
from pathlib import Path
from typing import Generic, NoReturn, Optional, Any, Literal, Callable, Union, ClassVar, overload, type_check_only
from types import GenericAlias
from typing_extensions import Self, TypeAlias, TypeVar
import numpy as np
from numpy.typing import NDArray, DTypeLike
import numpy.typing as npt

# Spectrogram parameter classes come from the spectrograms crate compiled into
# this extension (``audio_samples.spectrograms``), not from the separate
# ``spectrograms`` pip package, whose objects are rejected at runtime.
from audio_samples.spectrograms import (
    ChromaParams as ChromaParams,
    CqtParams as CqtParams,
    LogParams as LogParams,
    MelParams as MelParams,
    MelNorm as MelNorm,
    MfccParams as MfccParams,
    SpectrogramParams as SpectrogramParams,
    StftParams as StftParams,
    StftResult as StftResult,
    ErbParams as ErbParams,
    WindowType as WindowType,
)
from audio_samples.exceptions import (
    AudioError as AudioError,
    AudioParameterError as AudioParameterError,
    AudioLayoutError as AudioLayoutError,
    AudioConversionError as AudioConversionError,
    AudioRuntimeError as AudioRuntimeError,
    AudioProcessingError as AudioProcessingError,
    AudioFeatureError as AudioFeatureError,
    AudioSpectrogramError as AudioSpectrogramError,
    AudioIOError as AudioIOError,
    AudioFileNotFoundError as AudioFileNotFoundError,
    AudioFormatError as AudioFormatError,
    AudioDecodeError as AudioDecodeError,
)
from audio_samples.types import (
    EqBandType,
    HpssConfig,
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
    ChannelReduction,
    GateConfig,
    ExpanderConfig,
    ThreeBandEqConfig,
    Psd,
    Key,
    Mode,
    PitchClass,
    PitchContour,
    PitchFrame,
    SosFilter,
)

# Runtime top-level re-exports of audio_samples.types and audio_samples.io
from audio_samples.types import (
    IirFilterDesign as IirFilterDesign,
    EqBand as EqBand,
    ParametricEq as ParametricEq,
    OnsetDetectionConfig as OnsetDetectionConfig,
    ComplexOnsetConfig as ComplexOnsetConfig,
    SpectralFluxConfig as SpectralFluxConfig,
    SpectralFluxMethod as SpectralFluxMethod,
    DynamicRangeMethod as DynamicRangeMethod,
    EnvelopeFollower as EnvelopeFollower,
    BeatTrackingConfig as BeatTrackingConfig,
    BeatTrackingData as BeatTrackingData,
    VadMethod as VadMethod,
    VadChannelPolicy as VadChannelPolicy,
    VadConfig as VadConfig,
)

from audio_samples.types import (
    AdaptiveThresholdConfig as AdaptiveThresholdConfig,
    AdaptiveThresholdMethod as AdaptiveThresholdMethod,
    HpssConfig as HpssConfig,
    IirFilterType as IirFilterType,
    KneeType as KneeType,
    NoiseColor as NoiseColor,
    PerturbationMethod as PerturbationMethod,
    SideChainConfig as SideChainConfig,
    SpectrogramScale as SpectrogramScale,
)
from audio_samples.io import (
    AudioInfo as AudioInfo,
    CompressionLevel as CompressionLevel,
    CuePoint as CuePoint,
    InfoMetadata as InfoMetadata,
    StreamInfo as StreamInfo,
    StreamedAudioReader as StreamedAudioReader,
    StreamedAudioWriter as StreamedAudioWriter,
    StreamedFrameIterator as StreamedFrameIterator,
    VorbisComment as VorbisComment,
    WavMetadata as WavMetadata,
    WavSink as WavSink,
    WriteOptions as WriteOptions,
    info as info,
    peek_native_type as peek_native_type,
    read as read,
    read_and_resample as read_and_resample,
    read_flac_stream_info as read_flac_stream_info,
    read_wav_cue_points as read_wav_cue_points,
    read_wav_info_tags as read_wav_info_tags,
    read_with_info as read_with_info,
    save as save,
    write_with_metadata as write_with_metadata,
    write_with_options as write_with_options,
)

plotting: Any
"""Rust plotly plotting module, or ``None`` when built without the plotting feature."""

from . import generation as generation
from . import io as io
from . import utils as utils
from . import mpl as mpl
from . import types as types
from . import spectrograms as spectrograms
from . import exceptions as exceptions
from . import explain as explain
from audio_samples.explain import ExplainDoc as ExplainDoc
from audio_samples.explain import _Explainable as _Explainable
from . import spectral as spectral
from . import sequence as sequence
from . import rhythm as rhythm
from . import effects as effects
from . import forms as forms
from . import linalg as linalg
from . import inverse as inverse
from . import factorization as factorization
from . import pitch_filterbank as pitch_filterbank
from . import segment as segment
from . import arrays as arrays
from . import notation as notation
from . import features as features
from .utils import mu_law_encode as mu_law_encode, mu_law_decode as mu_law_decode

__version__: str

_S = TypeVar("_S", bound=np.generic)
#: The sample type of an AudioSamples. Unknown (Any) for audio read from a file and for int24 audio,
#: which has no NumPy dtype.
#: A scalar operand. A bool is an int to a type checker but raises TypeError at run time.
_Scalar: TypeAlias = float | np.integer | np.floating
_S_co = TypeVar("_S_co", bound=np.generic, covariant=True, default=Any)
#: Sample types whose analysis results are float32 / complex64.
_SingleSample: TypeAlias = np.uint8 | np.int16 | np.float32
#: Matches only a sample type unknown to the type checker (Any). The overloads use it so that audio of
#: unknown sample type, such as audio read from a file, gives results of unknown precision.
@type_check_only
class _UnknownDtype(np.generic):
    def __new__(cls) -> _UnknownDtype: ...
#: Sample types whose analysis results are float64 / complex128.
_DoubleSample: TypeAlias = np.int32 | np.float64

#: A window specification accepted by the frame-wise analysis functions:
#: a SciPy window name, ``(name, parameter)``, a number (Kaiser beta) or an array.
_WindowSpec = Union[str, tuple[str, float], float, npt.ArrayLike]
#: Mel filter normalisation for the frame-wise analysis functions: ``"slaney"``,
#: None, or a p-norm exponent. (``MelNorm`` is the ``MelParams`` enum class.)
_MelNormSpec = Union[Literal["slaney"], float, None]
#: dB reference: a number, ``"max"`` or ``numpy.max``.
_DbRef = Union[float, Literal["max"], Callable[..., Any]]
#: A seed: an integer, None, or a NumPy generator (one seed is drawn from it).
_RngLike = Union[int, None, np.random.Generator, np.random.RandomState, np.random.SeedSequence, np.random.BitGenerator, Sequence[int]]

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
    square_wave_bandlimited,
    sawtooth_wave_bandlimited,
    triangle_wave_bandlimited,
    exponential_chirp,
    fm_signal,
    am_signal,
    compound_tone,
    exponential_bursts,
)


@final
class AudioSamples(Generic[_S_co]):
    """
    Main audio processing class supporting multiple sample formats and numpy integration.

    Provides statistics, filtering, equalization, dynamic range processing and
    frequency analysis.

    The frame-wise analysis methods return float32 / complex64 arrays
    for uint8, int16, int24 and float32 audio, and float64 / complex128 arrays
    for int32 and float64 audio. Audio-returning methods keep the sample type.
    """


    @staticmethod
    def new_mono(arr: NDArray[_S], sample_rate: int) -> AudioSamples[_S]:
        """Create mono audio from a 1-D numpy array (auto-detects dtype).

        Zero-copy: the AudioSamples views ``arr``'s memory, so in-place
        methods (``apply_compressor``, ``apply_iir_filter``, ``filtfilt``, ...)
        modify ``arr``. Pass ``arr.copy()`` to keep the original. No value
        conversion is performed.
        """
        ...

    @staticmethod
    def new_multi(arr: NDArray[_S], sample_rate: int) -> AudioSamples[_S]:
        """Create multi-channel audio from a 2-D (channels, samples) numpy array.

        Zero-copy: the AudioSamples views ``arr``'s memory, so in-place
        methods modify ``arr``. Pass ``arr.copy()`` to keep the original.
        """
        ...

    # Factory methods - zeros
    @staticmethod
    def zeros_mono(length: int, sample_rate: int) -> AudioSamples[np.float32]:
        """Create zero-filled mono f32 audio."""
        ...

    @staticmethod
    def zeros_mono_i16(length: int, sample_rate: int) -> AudioSamples[np.int16]:
        """Create zero-filled mono i16 audio."""
        ...

    @staticmethod
    def zeros_mono_i32(length: int, sample_rate: int) -> AudioSamples[np.int32]:
        """Create zero-filled mono i32 audio."""
        ...

    @staticmethod
    def zeros_mono_f64(length: int, sample_rate: int) -> AudioSamples[np.float64]:
        """Create zero-filled mono f64 audio."""
        ...

    @staticmethod
    def zeros_multi(channels: int, length: int, sample_rate: int) -> AudioSamples[np.float32]:
        """Create zero-filled multi-channel f32 audio."""
        ...

    @staticmethod
    def zeros_multi_i16(channels: int, length: int, sample_rate: int) -> AudioSamples[np.int16]:
        """Create zero-filled multi-channel i16 audio."""
        ...

    @staticmethod
    def zeros_multi_i32(channels: int, length: int, sample_rate: int) -> AudioSamples[np.int32]:
        """Create zero-filled multi-channel i32 audio."""
        ...

    @staticmethod
    def zeros_multi_f64(channels: int, length: int, sample_rate: int) -> AudioSamples[np.float64]:
        """Create zero-filled multi-channel f64 audio."""
        ...

    # Factory methods - ones
    @staticmethod
    def ones_mono(length: int, sample_rate: int) -> AudioSamples[np.float32]:
        """Create ones-filled mono f32 audio."""
        ...

    @staticmethod
    def ones_mono_i16(length: int, sample_rate: int) -> AudioSamples[np.int16]:
        """Create ones-filled mono i16 audio."""
        ...

    @staticmethod
    def ones_mono_i32(length: int, sample_rate: int) -> AudioSamples[np.int32]:
        """Create ones-filled mono i32 audio."""
        ...

    @staticmethod
    def ones_mono_f64(length: int, sample_rate: int) -> AudioSamples[np.float64]:
        """Create ones-filled mono f64 audio."""
        ...

    @staticmethod
    def ones_multi(channels: int, length: int, sample_rate: int) -> AudioSamples[np.float32]:
        """Create ones-filled multi-channel f32 audio."""
        ...

    @staticmethod
    def ones_multi_i16(channels: int, length: int, sample_rate: int) -> AudioSamples[np.int16]:
        """Create ones-filled multi-channel i16 audio."""
        ...

    @staticmethod
    def ones_multi_i32(channels: int, length: int, sample_rate: int) -> AudioSamples[np.int32]:
        """Create ones-filled multi-channel i32 audio."""
        ...

    @staticmethod
    def ones_multi_f64(channels: int, length: int, sample_rate: int) -> AudioSamples[np.float64]:
        """Create ones-filled multi-channel f64 audio."""
        ...

    # Factory methods - uniform
    @staticmethod
    def uniform_mono(length: int, sample_rate: int, value: float) -> AudioSamples[np.float32]:
        """Create uniform-value mono f32 audio."""
        ...

    @staticmethod
    def uniform_mono_i16(length: int, sample_rate: int, value: int) -> AudioSamples[np.int16]:
        """Create uniform-value mono i16 audio."""
        ...

    @staticmethod
    def uniform_mono_i32(length: int, sample_rate: int, value: int) -> AudioSamples[np.int32]:
        """Create uniform-value mono i32 audio."""
        ...

    @staticmethod
    def uniform_mono_f64(length: int, sample_rate: int, value: float) -> AudioSamples[np.float64]:
        """Create uniform-value mono f64 audio."""
        ...

    @staticmethod
    def uniform_multi(
        channels: int, length: int, sample_rate: int, value: float
    ) -> AudioSamples[np.float32]:
        """Create uniform-value multi-channel f32 audio."""
        ...

    @staticmethod
    def uniform_multi_i16(
        channels: int, length: int, sample_rate: int, value: int
    ) -> AudioSamples[np.int16]:
        """Create uniform-value multi-channel i16 audio."""
        ...

    @staticmethod
    def uniform_multi_i32(
        channels: int, length: int, sample_rate: int, value: int
    ) -> AudioSamples[np.int32]:
        """Create uniform-value multi-channel i32 audio."""
        ...

    @staticmethod
    def uniform_multi_f64(
        channels: int, length: int, sample_rate: int, value: float
    ) -> AudioSamples[np.float64]:
        """Create uniform-value multi-channel f64 audio."""
        ...

    # Audio editing operations
    @classmethod
    def concatenate(cls, segments: list[AudioSamples[_S]]) -> AudioSamples[_S]:
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
    def stack(cls, sources: list[AudioSamples[_S]]) -> AudioSamples[_S]:
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

    @classmethod
    def concatenate_owned(cls, segments: list[AudioSamples[_S]]) -> AudioSamples[_S]:
        """Concatenate multiple owned buffers end to end.

        Behaves identically to :meth:`concatenate`; it mirrors the crate's
        owned-segment variant. All segments must share the same dtype, sample
        rate, and channel count.

        Args:
            segments: Non-empty list of buffers to join.

        Returns:
            AudioSamples: Combined buffer spanning all segments.

        Raises:
            ValueError: If the segment list is empty.
            TypeError: If segments differ in dtype or are not AudioSamples.
            AudioError: If concatenation fails.
        """
        ...

    # AudioEditing methods
    def repeat(self, count: int) -> Self:
        """Repeat the audio samples count times."""
        ...

    def trim_silence(self, threshold_db: float) -> Self:
        """Trim silence from the beginning and end of the audio based on a dB threshold."""
        ...

    def trim_all_silence(
        self, threshold_db: float, min_silence_duration_seconds: float
    ) -> Self:
        """Remove every silent region longer than a minimum duration.

        Unlike :meth:`trim_silence`, which only trims leading and trailing
        silence, this removes interior silence runs as well. A region is treated
        as silent when its level stays below ``threshold_db`` for at least
        ``min_silence_duration_seconds``.

        Args:
            threshold_db: Silence threshold in decibels.
            min_silence_duration_seconds: Minimum duration, in seconds, a silent
                run must reach before it is removed.

        Returns:
            AudioSamples: Buffer with qualifying silence removed.

        Raises:
            AudioError: If trimming fails.
        """
        ...

    def perturb(self, config: "PerturbationConfig") -> Self:
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
        self, pad_start_seconds: float, pad_end_seconds: float, pad_value: float
    ) -> Self:
        """Pad audio with silence or a specific value at the start and/or end."""
        ...

    def pad_samples_right(
        self, target_num_samples: int, pad_value: float
    ) -> Self:
        """Right-pad the buffer to a fixed sample count.

        Args:
            target_num_samples: Desired total number of samples
            pad_value: Value used for padding

        Returns:
            New buffer of ``target_num_samples`` samples. If the buffer is
            already at least that long it is returned unchanged (it is not
            truncated and no error is raised).
        """
        ...

    def pad_to_duration(
        self, target_duration_seconds: float, pad_value: float, pad_side: "PadSide"
    ) -> Self:
        """Pad the buffer to reach a target duration.

        Args:
            target_duration_seconds: Desired total duration
            pad_value: Value used for padding
            pad_side: Side to apply the padding

        Returns:
            New buffer matching the requested duration
        """
        ...

    def split(self, segment_duration_seconds: float) -> list[Self]:
        """Split audio into segments of a specified duration."""
        ...

    @classmethod
    def mix(
        cls, sources: list[AudioSamples[_S]], weights: Optional[list[float]] = None
    ) -> AudioSamples[_S]:
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

    # Explanations (audio_samples.explain)
    def explain(
        self,
        obj: _Explainable,
        *,
        f0_range: Optional[tuple[float, float]] = None,
        n_fft: Optional[int] = None,
        n_mels: Optional[int] = None,
    ) -> ExplainDoc:
        """Explain a configuration in the context of this signal.

        Uses this signal's sample rate, length (samples per channel) and
        channel count as context.

        Args:
            obj: The configuration to explain (``StftParams``,
                ``SpectrogramParams``, ``MelParams``, ``MfccParams``,
                ``WindowType``, ``IirFilterDesign``, ``EqBand``,
                ``CompressorConfig``, ``LimiterConfig``, ``VadConfig`` or
                ``OnsetDetectionConfig``).
            f0_range: Expected fundamental-frequency range ``(lowest, highest)`` in Hz.
            n_fft: FFT size a ``MelParams`` will be combined with.
            n_mels: Number of mel bands an ``MfccParams`` will be combined with.

        Returns:
            The explanation.

        Raises:
            TypeError: If ``obj`` is not a supported type.

        Examples:
            >>> audio = sine_wave(440.0, 1.0, sample_rate=16000)
            >>> params = StftParams(n_fft=512, hop_size=256, window=WindowType.hanning)
            >>> audio.explain(params).quantity("Frequency resolution")
            31.25
        """
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
    def istft(cls, stft_result: "StftResult") -> AudioSamples[Any]:
        """Invert a Short-Time Fourier Transform result.

        Args:
            stft_result: Output from `stft` to be reconstructed

        Returns:
            Reconstructed time-domain signal
        """
        ...

    # AudioProcessing methods
    def resample(
        self,
        target_sample_rate: int,
        quality: "ResamplingQuality | Literal['fast', 'medium', 'high', 'very_high']" = "fast",
    ) -> Self:
        """Resample to ``target_sample_rate``.

        Delay-free windowed-sinc interpolation: the output is time-aligned with
        the input, the tail is flushed, and the result has exactly
        ``ceil(n * target_sample_rate / sample_rate)`` samples per channel.
        ``"high"`` is 0.913 x Nyquist / 125 dB stopband; ``"medium"`` is
        0.90 x Nyquist / 100 dB; ``"fast"`` (default) uses a short
        0.80 x Nyquist / 70 dB filter when downsampling and linear
        interpolation when upsampling.

        Args:
            target_sample_rate: Output sample rate in Hz (> 0).
            quality: A ``ResamplingQuality`` or its name. Default ``"fast"``.

        Returns:
            The resampled audio (input sample type kept).

        Raises:
            ValueError: Non-positive ``target_sample_rate`` or unknown quality name.
        """
        ...

    def resample_by_ratio(
        self,
        ratio: float,
        quality: "ResamplingQuality | Literal['fast', 'medium', 'high', 'very_high']" = "fast",
    ) -> Self:
        """Resample to ``round(sample_rate * ratio)`` Hz; otherwise as ``resample``.

        Args:
            ratio: Output rate / input rate (> 0).
            quality: A ``ResamplingQuality`` or its name. Default ``"fast"``.

        Returns:
            The resampled audio.

        Raises:
            ValueError: Non-positive ratio or a target rate that rounds to 0.
        """
        ...

    def apply_filter(self, coefficients: NDArray[np.float64]) -> Self:
        """FIR filter by valid convolution: ``numpy.convolve(x, h, 'valid')`` per channel.

        ``y[i] = sum_k h[k] * x[i + M - 1 - k]``; the output has ``N - M + 1``
        samples (``convolve`` returns the full ``N + M - 1``). Computed in
        float64 with float64 taps; integer audio is filtered on its raw scale
        and rounded to nearest and saturated, so taps outside ``[-1, 1]`` work.
        """
        ...

    def apply_filter_in_place(self, coefficients: NDArray[np.float64]) -> None:
        """In-place ``apply_filter``; the buffer becomes ``M - 1`` samples shorter."""
        ...

    def apply_window(self, window: NDArray[np.floating]) -> Self:
        """Apply a window function to the audio samples.

        Each sample is multiplied by its float64 coefficient; integer audio is
        rounded to nearest and saturated, so fractional windows work for every
        sample type.

        Args:
            window: Sequence of coefficients matching the audio length

        Returns:
            New buffer of the same sample type with the window applied
        """
        ...

    # AudioPitchAnalysis methods
    def detect_pitch_yin(
        self,
        threshold: float = 0.1,
        min_frequency: float = 50.0,
        max_frequency: float = 2000.0,
    ) -> Optional[float]:
        """Detect pitch using the YIN algorithm. Returns frequency in Hz or None.

        ``threshold`` is the absolute threshold on YIN's normalised difference
        d'(tau) in [0, 1]; lower values are stricter (fewer detections, more
        ``None``), higher values more permissive.
        """
        ...

    def detect_pitch_autocorr(
        self,
        min_frequency: float,
        max_frequency: float,
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
    ) -> PitchContour:
        """Track the detected pitch over successive analysis windows.

        Args:
            window_size: Size of each analysis window in samples; must be greater than zero.
            hop_size: Step between windows in samples; must be greater than zero.
            pitch_method: Detection strategy to apply per window.
            threshold: Detection threshold interpreted by the selected method.
            min_frequency: Minimum expected frequency in Hz.
            max_frequency: Maximum expected frequency in Hz.

        Returns:
            PitchContour: Time-ordered pitch track of per-window frames.

        Raises:
            ValueError: If window_size or hop_size is zero.
            AudioError: If pitch tracking fails.
        """
        ...

    def harmonic_to_noise_ratio(
        self,
        fundamental_freq: float,
        num_harmonics: int,
        n_fft: Optional[int],
        window_type: Optional[WindowType],
    ) -> float:
        """Estimate the harmonic-to-noise ratio around a fundamental frequency.

        Args:
            fundamental_freq: Fundamental frequency in Hz
            num_harmonics: Number of harmonics to evaluate
            n_fft: FFT size in samples, or ``None`` for the default (required argument)
            window_type: Window applied before the FFT, or ``None`` for the default
                (required argument)

        Returns:
            Harmonic-to-noise ratio value
        """
        ...

    def harmonic_analysis(
        self,
        fundamental_freq: float,
        num_harmonics: int,
        tolerance: float,
        n_fft: Optional[int],
        window_type: Optional[WindowType],
    ) -> list[float]:
        """Measure harmonic amplitudes relative to a fundamental frequency.

        Args:
            fundamental_freq: Fundamental frequency in Hz
            num_harmonics: Number of harmonics to report
            tolerance: Frequency tolerance as a fraction of the harmonic
            n_fft: FFT size in samples, or ``None`` for the default (required argument)
            window_type: Window applied before the FFT, or ``None`` for the default
                (required argument)

        Returns:
            List of harmonic magnitudes ordered by harmonic index
        """
        ...

    def estimate_key(self, stft_params: StftParams) -> Key:
        """Estimate the musical key of the audio signal.

        Args:
            stft_params: Parameters controlling the underlying STFT analysis.

        Returns:
            Key: Detected key, including tonic pitch class, mode (major/minor),
                and confidence score.

        Raises:
            AudioError: If key estimation fails.
        """
        ...

    # AudioDecomposition methods
    def hpss(
        self,
        config: Optional[StftParams | HpssConfig] = None,
        median_filter_harmonic: Optional[int] = None,
        median_filter_percussive: Optional[int] = None,
        mask_softness: Optional[float] = None,
        *,
        stft_params: Optional[StftParams] = None,
    ) -> tuple[Self, Self]:
        """Separate harmonic and percussive components using median-filter HPSS.

        Args:
            config: Either a complete ``HpssConfig`` (then the other arguments
                must be omitted), or the ``StftParams`` of the analysis STFT.
            median_filter_harmonic: Harmonic (time-direction) median filter
                length in frames. Default 17. Only with ``StftParams``.
            median_filter_percussive: Percussive (frequency-direction) median
                filter length in bins. Default 17. Only with ``StftParams``.
            mask_softness: Mask softness in [0, 1] (0 = hard mask). Default 0.5.
                Only with ``StftParams``.
            stft_params: Deprecated keyword alias of ``config``; emits a
                ``DeprecationWarning``. Passing both is a ``TypeError``.

        Returns:
            tuple: ``(harmonic, percussive)`` AudioSamples.

        Raises:
            TypeError: If ``config`` is missing or neither ``StftParams`` nor
                ``HpssConfig``, or filter arguments are combined with an ``HpssConfig``.
            ValueError: If the configuration is invalid.
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

        Returns the frame timestamps and the onset detection function values.

        Args:
            config: Onset detection configuration parameters

        Returns:
            Tuple ``(times, values)``: frame times in seconds first, then the
            detection-function value of each frame.
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
            Tuple ``(times, flux)``: frame times in seconds first, then the
            flux values; both lists have one entry per frame.
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
    def amplitude_envelope(self) -> NDArray[_S_co]:
        """Compute a rectified amplitude envelope.

        Takes the absolute value of each sample to create an amplitude envelope.

        Returns:
            Amplitude envelope (1D for mono, 2D for multi-channel)
        """
        ...

    def rms_envelope(self, window_size: int, hop_size: int) -> NDArray[_S_co]:
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
    ) -> tuple[NDArray[_S_co], NDArray[_S_co]]:
        """Compute attack and decay envelopes using an envelope follower.

        Args:
            follower: Configured envelope follower defining attack/release behavior
            method: Detection strategy (Peak, Rms, or Hybrid)

        Returns:
            Tuple of (attack_envelope, decay_envelope) as numpy arrays
        """
        ...

    def analytic_envelope(self) -> NDArray[_S_co]:
        """Compute analytic amplitude envelope using Hilbert transform.

        Provides the instantaneous amplitude for each sample.

        Returns:
            Analytic envelope (1D for mono, 2D for multi-channel)
        """
        ...

    def moving_average_envelope(
        self, window_size: int, hop_size: int
    ) -> NDArray[_S_co]:
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

        Places beats on a fixed-tempo grid at ``config.tempo_bpm``, anchored at
        the strongest onset and stepped forward/backward by one inter-beat
        interval (snapping to nearby onsets). The tempo is not estimated (the
        returned tempo echoes the configuration). For dynamic-programming beat
        tracking use ``beat_track``.

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

    def estimate_tempo(self, config: BeatTrackingConfig) -> float:
        """Estimate the global tempo of the signal in beats per minute.

        This runs the same onset and tempo analysis as :meth:`detect_beats` but
        returns only the estimated tempo, skipping the per-beat tracking step.

        Args:
            config: Parameters controlling tempo targets and onset detection.

        Returns:
            float: Estimated tempo in beats per minute.

        Raises:
            AudioError: If tempo estimation fails.
        """
        ...

    def windows(
        self, window_size: int, hop_size: int
    ) -> list[NDArray[_S_co]]:
        """Split the signal into fixed-size, optionally overlapping windows.

        Each window covers ``window_size`` samples per channel and successive
        windows start ``hop_size`` samples apart, so windows overlap when
        ``hop_size < window_size``. Trailing data that does not fill a complete
        window is zero-padded.

        Args:
            window_size: Samples per channel in each window; must be > 0.
            hop_size: Samples between successive window starts; must be > 0.

        Returns:
            list[numpy.ndarray]: One array per window, each shaped ``(samples,)``
                for mono audio or ``(channels, samples)`` for multi-channel audio.

        Raises:
            ValueError: If window_size or hop_size is zero.
        """
        ...

    def frames(self) -> list[NDArray[_S_co]]:
        """Iterate over time-aligned frames of the signal.

        Each frame is a snapshot across all channels at a single time index, so
        the number of frames equals the number of samples per channel.

        Returns:
            list[numpy.ndarray]: One array per time index, each shaped ``(1,)``
                for mono audio or ``(channels, 1)`` for multi-channel audio.
        """
        ...

    def iter_channels(self) -> list[NDArray[_S_co]]:
        """Iterate over the complete signal of each channel.

        Each item is the full temporal sequence of one channel, yielded in
        increasing channel-index order.

        Returns:
            list[numpy.ndarray]: One mono array per channel.
        """
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
    def low_pass_filter(self, cutoff_hz: float) -> Self:
        """First-order Butterworth low-pass (-3 dB at ``cutoff_hz``, 6 dB/octave).

        Bilinear-transform design with prewarping, applied as one second-order
        section from zero initial state. Does **not** modify ``self``.

        Args:
            cutoff_hz: Cutoff frequency in Hz, in (0, Nyquist)

        Returns:
            New buffer containing the filtered signal
        """
        ...

    def high_pass_filter(self, cutoff_hz: float) -> Self:
        """First-order Butterworth high-pass (-3 dB at ``cutoff_hz``, 6 dB/octave).

        Bilinear-transform design with prewarping, applied as one second-order
        section from zero initial state. Does **not** modify ``self``.

        Args:
            cutoff_hz: Cutoff frequency in Hz, in (0, Nyquist)

        Returns:
            New buffer containing the filtered signal
        """
        ...

    def band_pass_filter(
        self, low_cutoff_hz: float, high_cutoff_hz: float
    ) -> Self:
        """Band-pass from a first-order Butterworth prototype (one biquad).

        Bilinear-transform design with prewarping of both edges, applied as one
        second-order section from zero initial state. Does **not** modify ``self``.

        Args:
            low_cutoff_hz: Lower -3 dB edge in Hz
            high_cutoff_hz: Upper -3 dB edge in Hz

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
        """Return the value at the temporal midpoint, as ``midpoint_sample`` does (the buffer is not sorted)."""
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
        """Count the number of zero crossings.

        A crossing is counted between consecutive samples when one is ``> 0``
        and the other is ``<= 0``. For multi-channel audio the per-channel
        counts are summed.
        """
        ...

    def zero_crossing_rate(self) -> float:
        """Zero crossings per second of the whole signal, averaged over channels.

        For the per-frame fraction of sign changes use
        ``zero_crossing_rate_frames``.
        """
        ...

    def autocorrelation(self, max_lag: int) -> Optional[list[float]]:
        """Unbiased autocorrelation ``r[k] = sum_j x[j] x[j+k] / (n - k)``.

        Lags ``0..=min(max_lag, n - 1)`` of the first channel; ``r[0]`` is the
        mean square. ``r[k] * (n - k)`` is the raw sum ``sum_j x[j] x[j+k]``.
        """
        ...

    def cross_correlation(self, other: AudioSamples, max_lag: int) -> list[float]:
        """Unbiased, unnormalised cross-correlation r[k] = sum self[n]other[n+k] / (N-k).

        Args:
            other: Audio buffer to correlate against
            max_lag: Maximum lag to include

        Returns:
            Cross-correlation coefficients up to the requested lag

        Raises:
            ValueError: If max_lag is zero
            AudioParameterError: If sample rates or channel counts differ
            TypeError: If buffers have different dtypes
        """
        ...

    def spectral_centroid(
        self, reduction: ChannelReduction = ChannelReduction.average
    ) -> float:
        """Compute the spectral centroid using the current FFT backend.

        Args:
            reduction: Channel-reduction policy applied to multi-channel input.
                Defaults to ``ChannelReduction.average``.

        Returns:
            float: Centroid in Hz of the **power** spectrum ``|X|**2`` of the
                whole signal (one FFT, not frame-wise). Returns 0.0 for silence.

        Raises:
            AudioError: If the FFT computation fails or the reduction is invalid
                for the channel layout.
        """
        ...

    def spectral_rolloff(
        self,
        rolloff_percent: float = 0.85,
        reduction: ChannelReduction = ChannelReduction.average,
    ) -> float:
        """Compute the spectral rolloff point for the provided energy fraction.

        Args:
            rolloff_percent: Target cumulative spectral energy fraction between 0 and 1.
            reduction: Channel-reduction policy applied to multi-channel input.
                Defaults to ``ChannelReduction.average``.

        Returns:
            float: Frequency where the cumulative spectrum reaches the rolloff threshold.

        Raises:
            AudioError: If the FFT computation fails or rolloff_percent is invalid.
        """
        ...

    def spectral_bandwidth(
        self, reduction: ChannelReduction = ChannelReduction.average
    ) -> float:
        """Compute the spectral bandwidth (spectral spread) of the signal.

        The magnitude-weighted standard deviation of the spectrum about its
        spectral centroid. Larger values indicate energy spread across a wider
        frequency range; a pure tone yields a value near zero.

        Args:
            reduction: Channel-reduction policy applied to multi-channel input.
                Defaults to ``ChannelReduction.average``.

        Returns:
            float: Spectral bandwidth in Hz. Returns 0.0 for silence.

        Raises:
            AudioError: If the FFT computation fails or the reduction is invalid
                for the channel layout.
        """
        ...

    def spectral_flatness(
        self, reduction: ChannelReduction = ChannelReduction.average
    ) -> float:
        """Compute the spectral flatness (Wiener entropy) of the signal.

        The ratio of the geometric mean to the arithmetic mean of the power
        spectrum. Values near 1 indicate noise-like (flat) spectra, while values
        near 0 indicate tonal spectra dominated by a few peaks.

        Args:
            reduction: Channel-reduction policy applied to multi-channel input.
                Defaults to ``ChannelReduction.average``.

        Returns:
            float: Spectral flatness in [0, 1]. Returns 0.0 for silence.

        Raises:
            AudioError: If the FFT computation fails or the reduction is invalid
                for the channel layout.
        """
        ...

    def spectral_crest(
        self, reduction: ChannelReduction = ChannelReduction.average
    ) -> float:
        """Compute the spectral crest factor of the signal.

        The ratio of the peak magnitude to the mean magnitude of the spectrum.
        High values indicate a strongly peaked (tonal) spectrum; a flat spectrum
        approaches 1.

        Args:
            reduction: Channel-reduction policy applied to multi-channel input.
                Defaults to ``ChannelReduction.average``.

        Returns:
            float: Spectral crest factor (>= 1 for non-silent signals). Returns
                0.0 for silence.

        Raises:
            AudioError: If the FFT computation fails or the reduction is invalid
                for the channel layout.
        """
        ...

    def spectral_slope(
        self, reduction: ChannelReduction = ChannelReduction.average
    ) -> float:
        """Compute the spectral slope of the signal.

        The slope of an ordinary least-squares linear fit of the linear magnitude
        spectrum against frequency (Hz), in units of magnitude per Hz. A negative
        slope indicates energy concentrated at low frequencies.

        Args:
            reduction: Channel-reduction policy applied to multi-channel input.
                Defaults to ``ChannelReduction.average``.

        Returns:
            float: Least-squares slope (magnitude per Hz). Returns 0.0 for
                silence or a degenerate single-bin spectrum.

        Raises:
            AudioError: If the FFT computation fails or the reduction is invalid
                for the channel layout.
        """
        ...

    def spectral_contrast(
        self,
        n_bands: int,
        reduction: ChannelReduction = ChannelReduction.average,
    ) -> list[float]:
        """Compute spectral contrast across octave-spaced sub-bands.

        The spectrum is partitioned into ``n_bands`` octave-spaced sub-bands.
        Within each band the contrast is the dB difference between the mean of the
        top quantile (peaks) and the mean of the bottom quantile (valleys). High
        contrast indicates clear tonal/harmonic structure.

        Args:
            n_bands: Number of octave-spaced sub-bands; must be greater than zero.
            reduction: Channel-reduction policy applied to multi-channel input.
                Defaults to ``ChannelReduction.average``.

        Returns:
            list[float]: ``n_bands`` contrast values in dB, low band to high band.

        Raises:
            ValueError: If n_bands is zero.
            AudioError: If the FFT computation fails or the reduction is invalid
                for the channel layout.
        """
        ...

    def midpoint_sample(self) -> Optional[float]:
        """Return the value at the temporal midpoint of a mono signal.

        For even-length signals the result is the average of the two central
        samples; for odd-length signals the single central sample is returned.
        Samples are selected by index position; the buffer is not sorted.

        Returns:
            Optional[float]: Midpoint value for mono audio, or None if the signal
                is multi-channel.
        """
        ...

    def rms_and_peak(self) -> tuple[float, int | float]:
        """Compute the RMS and peak absolute value in a single pass.

        Equivalent to calling ``rms`` and ``peak`` separately, but reads the
        sample buffer only once.

        Returns:
            tuple[float, int | float]: A ``(rms, peak)`` pair where ``rms`` is the
                root-mean-square as a float and ``peak`` is the maximum absolute
                sample value in the native sample type.
        """
        ...

    def amplitude(self) -> int | float:
        """Return the peak (maximum absolute value) across all samples and channels.

        Alias for ``peak``, provided to match the conventional term "amplitude"
        used in some audio contexts.

        Returns:
            int | float: Maximum absolute amplitude present in the buffer.
        """
        ...

    # Amplitude processing
    def scale(self, factor: float) -> None:
        """Scale audio amplitude by factor.

        Args:
            factor: Multiplier applied to every sample

        Returns:
            New buffer with scaled amplitudes
        """
        ...

    def normalize(
        self, min: float, max: float, method: "NormalizationMethod"
    ) -> None:
        """Normalize audio to given range using specified method.

        Args:
            min_val: Lower bound for the normalized output
            max_val: Upper bound for the normalized output
            method: Strategy controlling the normalization

        Returns:
            New buffer containing the normalized samples
        """
        ...

    def clip(self, min_val: float, max_val: float) -> None:
        """Clip audio values to given range.

        Args:
            min_val: Lower bound for clamping
            max_val: Upper bound for clamping

        Returns:
            New buffer with clamped amplitudes
        """
        ...

    def remove_dc_offset(self) -> None:
        """Remove DC offset from audio.

        Returns:
            New buffer with the mean offset removed
        """
        ...

    def mu_compress(self, mu: float) -> Self:
        """Apply mu-law compression to the signal.

        Args:
            mu: Mu parameter controlling the compression curve

        Returns:
            New buffer containing the compressed samples
        """
        ...

    def mu_expand(self, mu: float) -> Self:
        """Apply mu-law expansion to the signal.

        Args:
            mu: Mu parameter used during compression

        Returns:
            New buffer containing the expanded samples
        """
        ...

    # Time-domain processing
    def reverse(self) -> Self:
        """Return a reversed copy of the audio."""
        ...

    def reverse_in_place(self) -> None:
        """Reverse the audio samples in place."""
        ...

    def trim(self, start_seconds: float, end_seconds: float) -> Self:
        """Return a trimmed copy of the audio."""
        ...

    # Dynamic range processing (in-place)
    def apply_compressor(self, compressor_config: CompressorConfig) -> None:
        """Apply dynamic-range compression in place.

        Args:
            compressor_config: The compressor settings.
        """
        ...

    def apply_limiter(self, config: LimiterConfig) -> None:
        """Apply peak limiting in place.

        Args:
            config: The limiter settings.
        """
        ...

    def apply_gate(self, config: GateConfig) -> None:
        """Apply a downward noise gate in place.

        Args:
            config: Gate parameters controlling threshold, ratio, and timing.

        Returns:
            None: Operation mutates the current buffer.

        Raises:
            AudioError: If gating fails.
        """
        ...

    def apply_expander(self, config: ExpanderConfig) -> None:
        """Apply a downward expander in place.

        Args:
            config: Expander parameters controlling threshold, ratio, and timing.

        Returns:
            None: Operation mutates the current buffer.

        Raises:
            AudioError: If expansion fails.
        """
        ...

    def apply_compressor_sidechain(
        self, config: CompressorConfig, sidechain: AudioSamples
    ) -> Self:
        """Apply sidechain compression, returning a new buffer.

        The compressor's gain reduction is driven by the level of ``sidechain``
        rather than the main signal. Sidechain processing must be enabled on the
        configuration. Only mono-to-mono sidechain processing is supported.

        Args:
            config: Compressor parameters with sidechain enabled.
            sidechain: External control signal. Must match the main signal's
                dtype and length.

        Returns:
            AudioSamples: A new sidechain-compressed buffer.

        Raises:
            TypeError: If the main and sidechain dtypes differ.
            ValueError: If sidechain is not enabled, lengths differ, or either
                signal is multi-channel.
            AudioError: If compression fails.
        """
        ...

    def apply_limiter_sidechain(
        self, config: LimiterConfig, sidechain: AudioSamples
    ) -> Self:
        """Apply sidechain limiting, returning a new buffer.

        The limiter's gain reduction ceiling is enforced based on the level of
        ``sidechain`` rather than the main signal. Sidechain processing must be
        enabled on the configuration. Only mono-to-mono processing is supported.

        Args:
            config: Limiter parameters with sidechain enabled.
            sidechain: External control signal. Must match the main signal's
                dtype and length.

        Returns:
            AudioSamples: A new sidechain-limited buffer.

        Raises:
            TypeError: If the main and sidechain dtypes differ.
            ValueError: If sidechain is not enabled, lengths differ, or either
                signal is multi-channel.
            AudioError: If limiting fails.
        """
        ...

    def get_compression_curve(
        self, config: CompressorConfig, input_levels_db: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Compute the static compression input/output curve.

        Maps each input level through the compressor's static gain characteristic
        (threshold, ratio, knee) plus makeup gain. This is purely analytical and
        does not modify or use the audio content (no envelope following).

        Args:
            config: Compressor parameters.
            input_levels_db: Non-empty array of input levels in dBFS to evaluate.

        Returns:
            numpy.ndarray: Output levels in dBFS, one per input, in the same order.

        Raises:
            ValueError: If ``input_levels_db`` is empty or the configuration is invalid.
            AudioError: If the computation fails.
        """
        ...

    def get_gain_reduction(self, config: CompressorConfig) -> NDArray[np.float64]:
        """Compute the per-sample gain reduction the compressor would apply.

        Passes the audio through the envelope follower and compression gain
        calculation, collecting the gain reduction (in dB) at every sample
        without modifying the signal. For multi-channel audio, only the first
        channel is analysed.

        Args:
            config: Compressor parameters.

        Returns:
            numpy.ndarray: Gain reduction values in dB (each >= 0.0), one per
                sample (first channel for multi-channel audio).

        Raises:
            AudioError: If the configuration is invalid or processing fails.
        """
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
        response: FilterResponse | str,
    ) -> None:
        """Apply a Chebyshev Type I low-pass or high-pass filter (in-place).

        ``response`` is ``'lowpass'`` / ``'highpass'`` (or a FilterResponse).
        ``passband_ripple`` is in dB; the design is a bilinear transform with
        prewarping, applied as cascaded second-order sections.
        """
        ...

    def apply_chebyshev_ii_lowpass(
        self, order: int, cutoff_frequency: float, stopband_attenuation: float
    ) -> None:
        """Apply a Chebyshev Type II low-pass filter in place.

        Chebyshev Type II (inverse Chebyshev) filters have a maximally flat
        passband and equiripple stopband. They are specified by the stopband
        attenuation rather than passband ripple.

        Args:
            order: Filter order; must be greater than zero.
            cutoff_frequency: Stopband edge frequency in hertz.
            stopband_attenuation: Minimum stopband attenuation in decibels
                (typically 20-80 dB); must be positive.

        Returns:
            None: Operation mutates the current buffer.

        Raises:
            ValueError: If order is zero or a parameter is invalid.
            AudioError: If filter design or application fails.
        """
        ...

    def apply_chebyshev_ii_highpass(
        self, order: int, cutoff_frequency: float, stopband_attenuation: float
    ) -> None:
        """Apply a Chebyshev Type II high-pass filter in place.

        Args:
            order: Filter order; must be greater than zero.
            cutoff_frequency: Stopband edge frequency in hertz.
            stopband_attenuation: Minimum stopband attenuation in decibels;
                must be positive.

        Returns:
            None: Operation mutates the current buffer.

        Raises:
            ValueError: If order is zero or a parameter is invalid.
            AudioError: If filter design or application fails.
        """
        ...

    def apply_chebyshev_ii_bandpass(
        self,
        order: int,
        low_frequency: float,
        high_frequency: float,
        stopband_attenuation: float,
    ) -> None:
        """Apply a Chebyshev Type II band-pass filter in place.

        Args:
            order: Prototype filter order per edge; must be greater than zero.
            low_frequency: Lower band edge in hertz.
            high_frequency: Upper band edge in hertz.
            stopband_attenuation: Minimum stopband attenuation in decibels;
                must be positive.

        Returns:
            None: Operation mutates the current buffer.

        Raises:
            ValueError: If order is zero or a parameter is invalid.
            AudioError: If filter design or application fails.
        """
        ...

    def apply_chebyshev_ii_bandstop(
        self,
        order: int,
        low_frequency: float,
        high_frequency: float,
        stopband_attenuation: float,
    ) -> None:
        """Apply a Chebyshev Type II band-stop filter in place.

        Args:
            order: Prototype filter order per edge; must be greater than zero.
            low_frequency: Lower band edge in hertz.
            high_frequency: Upper band edge in hertz.
            stopband_attenuation: Minimum stopband attenuation in decibels;
                must be positive.

        Returns:
            None: Operation mutates the current buffer.

        Raises:
            ValueError: If order is zero or a parameter is invalid.
            AudioError: If filter design or application fails.
        """
        ...

    def apply_elliptic_lowpass(
        self,
        order: int,
        cutoff_frequency: float,
        passband_ripple: float,
        stopband_attenuation: float,
    ) -> None:
        """Apply an elliptic (Cauer) low-pass filter in place.

        Elliptic filters are equiripple in both passband and stopband, giving the
        steepest transition for a given order. They are specified by both the
        passband ripple and the stopband attenuation.

        Args:
            order: Filter order; must be greater than zero.
            cutoff_frequency: Passband edge frequency in hertz.
            passband_ripple: Peak passband ripple in decibels; must be positive.
            stopband_attenuation: Minimum stopband attenuation in decibels;
                must be positive and greater than ``passband_ripple``.

        Returns:
            None: Operation mutates the current buffer.

        Raises:
            ValueError: If order is zero or a parameter is invalid.
            AudioError: If filter design or application fails.
        """
        ...

    def apply_elliptic_highpass(
        self,
        order: int,
        cutoff_frequency: float,
        passband_ripple: float,
        stopband_attenuation: float,
    ) -> None:
        """Apply an elliptic (Cauer) high-pass filter in place.

        Args:
            order: Filter order; must be greater than zero.
            cutoff_frequency: Passband edge frequency in hertz.
            passband_ripple: Peak passband ripple in decibels; must be positive.
            stopband_attenuation: Minimum stopband attenuation in decibels;
                must be positive and greater than ``passband_ripple``.

        Returns:
            None: Operation mutates the current buffer.

        Raises:
            ValueError: If order is zero or a parameter is invalid.
            AudioError: If filter design or application fails.
        """
        ...

    def apply_elliptic_bandpass(
        self,
        order: int,
        low_frequency: float,
        high_frequency: float,
        passband_ripple: float,
        stopband_attenuation: float,
    ) -> None:
        """Apply an elliptic (Cauer) band-pass filter in place.

        Args:
            order: Prototype filter order per edge; must be greater than zero.
            low_frequency: Lower band edge in hertz.
            high_frequency: Upper band edge in hertz.
            passband_ripple: Peak passband ripple in decibels; must be positive.
            stopband_attenuation: Minimum stopband attenuation in decibels;
                must be positive and greater than ``passband_ripple``.

        Returns:
            None: Operation mutates the current buffer.

        Raises:
            ValueError: If order is zero or a parameter is invalid.
            AudioError: If filter design or application fails.
        """
        ...

    def apply_elliptic_bandstop(
        self,
        order: int,
        low_frequency: float,
        high_frequency: float,
        passband_ripple: float,
        stopband_attenuation: float,
    ) -> None:
        """Apply an elliptic (Cauer) band-stop filter in place.

        Args:
            order: Prototype filter order per edge; must be greater than zero.
            low_frequency: Lower band edge in hertz.
            high_frequency: Upper band edge in hertz.
            passband_ripple: Peak passband ripple in decibels; must be positive.
            stopband_attenuation: Minimum stopband attenuation in decibels;
                must be positive and greater than ``passband_ripple``.

        Returns:
            None: Operation mutates the current buffer.

        Raises:
            ValueError: If order is zero or a parameter is invalid.
            AudioError: If filter design or application fails.
        """
        ...

    def apply_bessel_lowpass(self, order: int, cutoff_frequency: float) -> None:
        """Apply a Bessel low-pass filter in place.

        Bessel filters have a maximally flat group delay in the passband,
        preserving the wave shape of in-band signals. The cutoff is the -3 dB point.

        Args:
            order: Filter order; must be greater than zero.
            cutoff_frequency: -3 dB cutoff frequency in hertz.

        Returns:
            None: Operation mutates the current buffer.

        Raises:
            ValueError: If order is zero or a parameter is invalid.
            AudioError: If filter design or application fails.
        """
        ...

    def apply_bessel_highpass(self, order: int, cutoff_frequency: float) -> None:
        """Apply a Bessel high-pass filter in place.

        Args:
            order: Filter order; must be greater than zero.
            cutoff_frequency: -3 dB cutoff frequency in hertz.

        Returns:
            None: Operation mutates the current buffer.

        Raises:
            ValueError: If order is zero or a parameter is invalid.
            AudioError: If filter design or application fails.
        """
        ...

    def apply_bessel_bandpass(
        self, order: int, low_frequency: float, high_frequency: float
    ) -> None:
        """Apply a Bessel band-pass filter in place.

        Args:
            order: Prototype filter order per edge; must be greater than zero.
            low_frequency: Lower band edge in hertz.
            high_frequency: Upper band edge in hertz.

        Returns:
            None: Operation mutates the current buffer.

        Raises:
            ValueError: If order is zero or a parameter is invalid.
            AudioError: If filter design or application fails.
        """
        ...

    def apply_bessel_bandstop(
        self, order: int, low_frequency: float, high_frequency: float
    ) -> None:
        """Apply a Bessel band-stop filter in place.

        Args:
            order: Prototype filter order per edge; must be greater than zero.
            low_frequency: Lower band edge in hertz.
            high_frequency: Upper band edge in hertz.

        Returns:
            None: Operation mutates the current buffer.

        Raises:
            ValueError: If order is zero or a parameter is invalid.
            AudioError: If filter design or application fails.
        """
        ...

    def filtfilt(self, design: IirFilterDesign) -> None:
        """Apply a zero-phase (forward-backward) IIR filter in place.

        Filters the signal twice -- once forward and once backward -- so the net
        phase response is zero. This doubles the effective filter order and
        squares the magnitude response. Useful when phase distortion must be
        avoided (e.g. analysis or offline processing).

        Args:
            design: Filter definition to apply forward and backward.

        Returns:
            None: Operation mutates the current buffer.

        Raises:
            AudioError: If filter design or application fails.
        """
        ...

    # Equalization (in-place)
    def apply_parametric_eq(self, eq: ParametricEq) -> None:
        """Apply parametric equalizer (in-place)."""
        ...

    def parametric_eq(self, eq: ParametricEq) -> Self:
        """Apply a parametric equalizer configuration, returning a new buffer.

        Non-mutating twin of :meth:`apply_parametric_eq`; leaves the current
        buffer unchanged.

        Args:
            eq: Equalizer definition containing one or more bands.

        Returns:
            AudioSamples: A new equalized buffer.

        Raises:
            AudioError: If filter application fails.
        """
        ...

    def apply_eq_band(self, band: EqBand) -> None:
        """Apply single EQ band (in-place)."""
        ...

    def eq_band(self, band: EqBand) -> Self:
        """Apply a single EQ band, returning a new buffer.

        Non-mutating twin of :meth:`apply_eq_band`; leaves the current buffer
        unchanged.

        Args:
            band: Parametric band definition to apply.

        Returns:
            AudioSamples: A new buffer with the band applied.

        Raises:
            AudioError: If filter application fails.
        """
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

    def apply_three_band_eq(self, config: ThreeBandEqConfig) -> None:
        """Apply a three-band equalizer in place.

        Args:
            config: Three-band EQ configuration describing the low shelf, mid
                peak, and high shelf bands.

        Returns:
            None: Operation mutates the current buffer.

        Raises:
            AudioError: If filter application fails.
        """
        ...

    def eq_frequency_response(
        self, eq: ParametricEq, frequencies: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Compute the combined magnitude and phase response of a parametric EQ.

        Evaluates each enabled band's biquad filter at every requested frequency
        and combines the results (magnitudes multiplied, phases summed), applying
        the EQ's output gain to the combined magnitude. Disabled bands are skipped.
        This is purely analytical and does not modify the audio.

        Args:
            eq: The equalizer whose response to evaluate.
            frequencies: Frequencies in hertz at which to evaluate the response.

        Returns:
            tuple[numpy.ndarray, numpy.ndarray]: Linear magnitude (1.0 = unity
                gain) and phase (radians) arrays, each the same length as
                ``frequencies``.

        Raises:
            AudioError: If any enabled band fails to design a filter (e.g. a
                frequency above the Nyquist limit).
        """
        ...

    # Frequency analysis
    def frequency_response(
        self, frequencies: list[float]
    ) -> tuple[list[float], list[float]]:
        """Deprecated: always returns a flat response (emits DeprecationWarning).

        Use ``IirFilterDesign.frequency_response(frequencies, sample_rate)`` or
        ``SosFilter.frequency_response(frequencies)`` instead.
        """
        ...

    def fft(self, n_fft: int) -> NDArray[np.complex128]:
        """One-sided FFT: ``numpy.fft.rfft(x, n_fft)`` per channel.

        Args:
            n_fft: FFT size in samples (>= signal length; input is zero-padded)

        Returns:
            Complex spectrum shaped (channels, n_fft // 2 + 1)
        """
        ...

    def rfft(self, n_fft: int) -> NDArray[np.float64]:
        """Magnitude of the one-sided FFT: ``abs(numpy.fft.rfft(x, n_fft))``.

        Args:
            n_fft: FFT size in samples

        Returns:
            Magnitude spectrum shaped (channels, n_fft // 2 + 1)
        """
        ...

    def power_spectral_density(
        self, window_size: int = 2048, overlap: float = 0.5
    ) -> Psd:
        """Estimate the power spectral density across overlapping windows.

        Args:
            window_size: STFT window length in samples; must be greater than zero.
            overlap: Fractional overlap between successive windows.

        Returns:
            Psd: Power spectral density carrying both the frequency axis and the
                power-per-Hz density, averaged over time.

        Raises:
            ValueError: If window_size is zero.
            AudioError: If the PSD computation fails.
        """
        ...

    def convolve(self, other: AudioSamples) -> Self:
        """Compute the linear convolution of this signal with another.

        Both signals are converted to f64 internally. The result is mono and
        carries this signal's sample rate, with length
        ``len(self) + len(other) - 1``.

        Args:
            other: Signal to convolve with. Must be mono and share this signal's
                data type.

        Returns:
            AudioSamples: The convolved mono signal.

        Raises:
            TypeError: If the two signals have different data types.
            AudioError: If either signal is multi-channel or the FFT fails.
        """
        ...

    def deconvolve(
        self, other: AudioSamples, regularization: float = 0.0
    ) -> Self:
        """Recover a system response by regularised spectral-division deconvolution.

        Solves ``self = denominator (*) h`` for ``h``. The ``regularization``
        term (>= 0) is scaled by the denominator's peak power to stabilise the
        division near spectral nulls. The result is mono at this signal's sample
        rate.

        Args:
            other: The denominator signal. Must be mono and share this signal's
                data type.
            regularization: Non-negative regularisation factor.

        Returns:
            AudioSamples: The deconvolved mono signal.

        Raises:
            TypeError: If the two signals have different data types.
            AudioError: If either signal is multi-channel or the FFT fails.
        """
        ...

    def constant_q_transform(
        self, params: CqtParams, hop_size: int
    ) -> NDArray[np.complex128]:
        """Compute the Constant-Q Transform (CQT) of a mono signal.

        Args:
            params: CQT configuration (bins per octave, number of octaves,
                minimum frequency).
            hop_size: Hop size in samples between successive frames; must be
                greater than zero.

        Returns:
            numpy.ndarray: Complex CQT coefficients shaped (frequency_bins, time_frames).

        Raises:
            ValueError: If hop_size is zero.
            AudioError: If the signal is multi-channel or the computation fails.
        """
        ...

    @staticmethod
    @overload
    def magphase(
        complex_spect: NDArray[_UnknownDtype],
        power: Optional[float] = None,
    ) -> tuple[NDArray[Any], NDArray[Any]]: ...
    @staticmethod
    @overload
    def magphase(  # type: ignore[overload-overlap]
        complex_spect: NDArray[np.complex64],
        power: Optional[float] = None,
    ) -> tuple[NDArray[np.float32], NDArray[np.complex64]]: ...
    @staticmethod
    @overload
    def magphase(
        complex_spect: NDArray[np.complex128],
        power: Optional[float] = None,
    ) -> tuple[NDArray[np.float64], NDArray[np.complex128]]:
        """Decompose a complex spectrogram into magnitude and phase.

        Given a complex matrix ``D``, returns ``(magnitude, phase)`` such that
        ``D = magnitude * phase`` elementwise, where ``phase`` contains
        unit-magnitude complex factors. Bins where the magnitude is zero are
        assigned a phase of ``1 + 0j``.

        This is a static method that operates on a complex spectrogram array
        (such as the output of ``fft`` or ``constant_q_transform``) rather than on
        an ``AudioSamples`` instance.

        Args:
            complex_spect: Complex 2-D STFT or FFT matrix.
            power: Exponent applied to the magnitude values, any number. Defaults to 1
                (raw magnitude).

        Returns:
            tuple[numpy.ndarray, numpy.ndarray]: The real-valued magnitude matrix
                and the complex unit-magnitude phase matrix.

        Raises:
            ValueError: If power is provided and is zero.
        """
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

        Notes: without log_params returns sqrt(M @ |X|**2) (square root of mel
        power); with log_params returns max(10*log10(M @ |X|**2), floor_db),
        floor_db being an absolute floor in dB re 1 (not relative to the peak).
        Slaney mel scale; filters unnormalised unless MelParams(norm="slaney").
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

        Notes: 10*log10 of unnormalised Slaney mel power, unnormalised DCT-II,
        HTK lifter 22. Equals ``mfcc_frames`` (ortho DCT, no lifter, no dB floor)
        times sqrt(n_mels) (c0) or sqrt(n_mels/2) (q >= 1), times the lifter
        weights.
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

    # ------------------------------------------------------------------
    # Frame-wise spectral analysis (AudioSpectrum)
    #
    # All methods return arrays shaped (channels, n_features, n_frames):
    # float32 / complex64 for uint8, int16, int24 and float32 audio, float64 /
    # complex128 for int32 and float64 audio. Mono audio gives a leading axis
    # of length 1; ``result[0]`` is the (n_features, n_frames) array.
    # ``mfcc_frames`` applies no dB floor unless ``top_db`` is given.
    # ------------------------------------------------------------------

    @overload
    def stft_matrix(
        self: AudioSamples[_UnknownDtype],
        *,
        n_fft: int = 2048,
        hop_length: Optional[int] = None,
        win_length: Optional[int] = None,
        window: _WindowSpec = 'hann',
        center: bool = True,
        pad_mode: Literal['constant', 'reflect', 'edge', 'symmetric'] = 'constant',
    ) -> NDArray[Any]: ...
    @overload
    def stft_matrix(  # type: ignore[overload-overlap]
        self: AudioSamples[_SingleSample],
        *,
        n_fft: int = 2048,
        hop_length: Optional[int] = None,
        win_length: Optional[int] = None,
        window: _WindowSpec = 'hann',
        center: bool = True,
        pad_mode: Literal['constant', 'reflect', 'edge', 'symmetric'] = 'constant',
    ) -> NDArray[np.complex64]: ...
    @overload
    def stft_matrix(
        self: AudioSamples[_DoubleSample],
        *,
        n_fft: int = 2048,
        hop_length: Optional[int] = None,
        win_length: Optional[int] = None,
        window: _WindowSpec = 'hann',
        center: bool = True,
        pad_mode: Literal['constant', 'reflect', 'edge', 'symmetric'] = 'constant',
    ) -> NDArray[np.complex128]:
        """Complex short-time Fourier transform of every channel.

        Args:
            n_fft: FFT length in samples.
            hop_length: Samples between frames; None means ``win_length // 4``.
            win_length: Window length (<= n_fft); None means ``n_fft``.
            window: ``"hann"``, ``"hamming"``, ``"blackman"``, ``"blackmanharris"``,
                ``"boxcar"``/``"rectangular"``, ``"bartlett"``, ``"triang"``,
                ``("kaiser", beta)``, ``("gaussian", std)``, a number (Kaiser beta)
                or an array of ``win_length`` coefficients. Named windows are
                periodic.
            center: Pad ``n_fft // 2`` samples on both sides so frame ``t`` is
                centred on sample ``t * hop_length``.
            pad_mode: Padding used when ``center`` is True.

        Returns:
            numpy.ndarray[complex64 | complex128], shape ``(channels, 1 + n_fft // 2, n_frames)``
            with ``n_frames = 1 + len // hop_length`` when centred.

        Raises:
            ValueError: Invalid parameters, or ``center=False`` with a signal
                shorter than ``n_fft``.
        """
        ...

    @staticmethod
    @overload
    def istft_matrix(
        stft_matrix: NDArray[_UnknownDtype],
        *,
        sample_rate: int,
        hop_length: Optional[int] = None,
        win_length: Optional[int] = None,
        n_fft: Optional[int] = None,
        window: _WindowSpec = 'hann',
        center: bool = True,
        length: Optional[int] = None,
        dtype: None = None,
    ) -> AudioSamples[Any]: ...
    @staticmethod
    @overload
    def istft_matrix(  # type: ignore[overload-overlap]
        stft_matrix: NDArray[np.complex64] | NDArray[np.float32],
        *,
        sample_rate: int,
        hop_length: Optional[int] = None,
        win_length: Optional[int] = None,
        n_fft: Optional[int] = None,
        window: _WindowSpec = 'hann',
        center: bool = True,
        length: Optional[int] = None,
        dtype: None = None,
    ) -> AudioSamples[np.float32]: ...
    @staticmethod
    @overload
    def istft_matrix(
        stft_matrix: npt.ArrayLike,
        *,
        sample_rate: int,
        hop_length: Optional[int] = None,
        win_length: Optional[int] = None,
        n_fft: Optional[int] = None,
        window: _WindowSpec = 'hann',
        center: bool = True,
        length: Optional[int] = None,
        dtype: None = None,
    ) -> AudioSamples[np.float64]: ...
    @staticmethod
    @overload
    def istft_matrix(
        stft_matrix: npt.ArrayLike,
        *,
        sample_rate: int,
        hop_length: Optional[int] = None,
        win_length: Optional[int] = None,
        n_fft: Optional[int] = None,
        window: _WindowSpec = 'hann',
        center: bool = True,
        length: Optional[int] = None,
        dtype: SampleType,
    ) -> AudioSamples[Any]:
        """Inverse STFT to audio by weighted overlap-add.

        Args:
            stft_matrix: Complex array, shape ``(channels, 1 + n_fft // 2, n_frames)``
                or ``(1 + n_fft // 2, n_frames)``.
            sample_rate: Sample rate of the returned audio.
            hop_length: None means ``win_length // 4``.
            win_length: None means ``n_fft``.
            n_fft: None infers ``2 * (n_bins - 1)``.
            window: Analysis window (see ``stft_matrix``).
            center: Whether the analysis frames were centred.
            length: Exact output length per channel (pass the original length to
                undo the centre padding exactly).
        dtype: Sample type of the returned audio. None (default) gives float32
            audio for a complex64 input and float64 audio for any other input.
            The computation runs in the float type of ``dtype`` (float32 for
            U8, I16, I24 and F32; float64 for I32 and F64); integer output is
            rounded and saturated.

        Returns:
            AudioSamples: audio of sample type ``dtype`` (mono for one channel).

        Raises:
            ValueError: Wrong dimensionality, bin count inconsistent with ``n_fft``,
                or invalid parameters.
        """
        ...

    @overload
    def spectrogram_matrix(
        self: AudioSamples[_UnknownDtype],
        *,
        n_fft: int = 2048,
        hop_length: Optional[int] = None,
        win_length: Optional[int] = None,
        window: _WindowSpec = 'hann',
        center: bool = True,
        pad_mode: Literal['constant', 'reflect', 'edge', 'symmetric'] = 'constant',
        power: float = 1.0,
    ) -> NDArray[Any]: ...
    @overload
    def spectrogram_matrix(  # type: ignore[overload-overlap]
        self: AudioSamples[_SingleSample],
        *,
        n_fft: int = 2048,
        hop_length: Optional[int] = None,
        win_length: Optional[int] = None,
        window: _WindowSpec = 'hann',
        center: bool = True,
        pad_mode: Literal['constant', 'reflect', 'edge', 'symmetric'] = 'constant',
        power: float = 1.0,
    ) -> NDArray[np.float32]: ...
    @overload
    def spectrogram_matrix(
        self: AudioSamples[_DoubleSample],
        *,
        n_fft: int = 2048,
        hop_length: Optional[int] = None,
        win_length: Optional[int] = None,
        window: _WindowSpec = 'hann',
        center: bool = True,
        pad_mode: Literal['constant', 'reflect', 'edge', 'symmetric'] = 'constant',
        power: float = 1.0,
    ) -> NDArray[np.float64]:
        """STFT magnitude raised to ``power`` for every channel:
        ``abs(stft_matrix(...)) ** power``.

        Returns:
            numpy.ndarray[float32 | float64], shape ``(channels, 1 + n_fft // 2, n_frames)``.

        Raises:
            ValueError: Invalid parameters or ``power <= 0``.
        """
        ...

    @overload
    def melspectrogram(
        self: AudioSamples[_UnknownDtype],
        *,
        n_fft: int = 2048,
        hop_length: int = 512,
        win_length: Optional[int] = None,
        window: _WindowSpec = 'hann',
        center: bool = True,
        pad_mode: Literal['constant', 'reflect', 'edge', 'symmetric'] = 'constant',
        power: float = 2.0,
        n_mels: int = 128,
        fmin: float = 0.0,
        fmax: Optional[float] = None,
        htk: bool = False,
        norm: _MelNormSpec = 'slaney',
    ) -> NDArray[Any]: ...
    @overload
    def melspectrogram(  # type: ignore[overload-overlap]
        self: AudioSamples[_SingleSample],
        *,
        n_fft: int = 2048,
        hop_length: int = 512,
        win_length: Optional[int] = None,
        window: _WindowSpec = 'hann',
        center: bool = True,
        pad_mode: Literal['constant', 'reflect', 'edge', 'symmetric'] = 'constant',
        power: float = 2.0,
        n_mels: int = 128,
        fmin: float = 0.0,
        fmax: Optional[float] = None,
        htk: bool = False,
        norm: _MelNormSpec = 'slaney',
    ) -> NDArray[np.float32]: ...
    @overload
    def melspectrogram(
        self: AudioSamples[_DoubleSample],
        *,
        n_fft: int = 2048,
        hop_length: int = 512,
        win_length: Optional[int] = None,
        window: _WindowSpec = 'hann',
        center: bool = True,
        pad_mode: Literal['constant', 'reflect', 'edge', 'symmetric'] = 'constant',
        power: float = 2.0,
        n_mels: int = 128,
        fmin: float = 0.0,
        fmax: Optional[float] = None,
        htk: bool = False,
        norm: _MelNormSpec = 'slaney',
    ) -> NDArray[np.float64]:
        """Mel spectrogram of every channel: the mel filterbank applied to the
        STFT magnitude raised to ``power``.

        Args:
            n_fft, hop_length, win_length, window, center, pad_mode: STFT
                parameters (``hop_length`` defaults to 512).
            power: Exponent of the STFT magnitude (2 = power, 1 = magnitude).
            n_mels: Number of mel bands.
            fmin: Lowest band edge in Hz.
            fmax: Highest band edge in Hz; None means ``sample_rate / 2``.
            htk: HTK mel scale instead of Slaney.
            norm: ``"slaney"`` (area normalisation), None, or a p-norm.

        Returns:
            numpy.ndarray[float32 | float64], shape ``(channels, n_mels, n_frames)``.

        Raises:
            ValueError: Invalid parameters.
        """
        ...

    @overload
    def mfcc_frames(
        self: AudioSamples[_UnknownDtype],
        *,
        n_mfcc: int = 20,
        dct_type: Literal[1, 2, 3] = 2,
        norm: Optional[Literal['ortho']] = 'ortho',
        lifter: float = 0.0,
        mel_norm: _MelNormSpec = 'slaney',
        n_fft: int = 2048,
        hop_length: int = 512,
        win_length: Optional[int] = None,
        window: _WindowSpec = 'hann',
        center: bool = True,
        pad_mode: Literal['constant', 'reflect', 'edge', 'symmetric'] = 'constant',
        power: float = 2.0,
        n_mels: int = 128,
        fmin: float = 0.0,
        fmax: Optional[float] = None,
        htk: bool = False,
        ref: _DbRef = 1.0,
        amin: float = 1e-10,
        top_db: Optional[float] = None,
    ) -> NDArray[Any]: ...
    @overload
    def mfcc_frames(  # type: ignore[overload-overlap]
        self: AudioSamples[_SingleSample],
        *,
        n_mfcc: int = 20,
        dct_type: Literal[1, 2, 3] = 2,
        norm: Optional[Literal['ortho']] = 'ortho',
        lifter: float = 0.0,
        mel_norm: _MelNormSpec = 'slaney',
        n_fft: int = 2048,
        hop_length: int = 512,
        win_length: Optional[int] = None,
        window: _WindowSpec = 'hann',
        center: bool = True,
        pad_mode: Literal['constant', 'reflect', 'edge', 'symmetric'] = 'constant',
        power: float = 2.0,
        n_mels: int = 128,
        fmin: float = 0.0,
        fmax: Optional[float] = None,
        htk: bool = False,
        ref: _DbRef = 1.0,
        amin: float = 1e-10,
        top_db: Optional[float] = None,
    ) -> NDArray[np.float32]: ...
    @overload
    def mfcc_frames(
        self: AudioSamples[_DoubleSample],
        *,
        n_mfcc: int = 20,
        dct_type: Literal[1, 2, 3] = 2,
        norm: Optional[Literal['ortho']] = 'ortho',
        lifter: float = 0.0,
        mel_norm: _MelNormSpec = 'slaney',
        n_fft: int = 2048,
        hop_length: int = 512,
        win_length: Optional[int] = None,
        window: _WindowSpec = 'hann',
        center: bool = True,
        pad_mode: Literal['constant', 'reflect', 'edge', 'symmetric'] = 'constant',
        power: float = 2.0,
        n_mels: int = 128,
        fmin: float = 0.0,
        fmax: Optional[float] = None,
        htk: bool = False,
        ref: _DbRef = 1.0,
        amin: float = 1e-10,
        top_db: Optional[float] = None,
    ) -> NDArray[np.float64]:
        """Mel-frequency cepstral coefficients of every channel.

        Pipeline: ``melspectrogram`` -> ``power_to_db(ref, amin, top_db)`` (per
        channel) -> DCT along the mel axis -> first ``n_mfcc`` rows -> liftering.

        Args:
            n_mfcc: Number of coefficients (must not exceed ``n_mels``).
            dct_type: DCT type 1, 2 or 3.
            norm: ``"ortho"`` or None (unnormalised DCT).
            lifter: Liftering coefficient; 0 disables.
            mel_norm: Mel filter normalisation (``norm`` of ``melspectrogram``).
            ref, amin, top_db: dB conversion parameters. ``top_db`` defaults to
                None (no floor).
            Remaining arguments: as ``melspectrogram``.

        Returns:
            numpy.ndarray[float32 | float64], shape ``(channels, n_mfcc, n_frames)``.

        Raises:
            ValueError: Invalid parameters, including ``n_mfcc > n_mels``.
        """
        ...

    # ------------------------------------------------------------------
    # Frame-wise features, tonal, constant-Q and pitch analysis
    #
    # Frame-wise outputs are arrays shaped (channels, n_features, n_frames)
    # (pitch trackers: (channels, n_frames)), float32 / complex64 for uint8,
    # int16, int24 and float32 audio and float64 / complex128 otherwise. The
    # chroma methods default to ``tuning=0.0``; ``tuning=None`` estimates it.
    # Index ``[0]`` for the (n_features, n_frames) array of mono audio.
    # Array-level (S=...) forms live in ``audio_samples.features``.
    # ------------------------------------------------------------------

    @overload
    def spectral_centroid_frames(
        self: AudioSamples[_UnknownDtype],
        *,
        n_fft: int = 2048,
        hop_length: int = 512,
        win_length: Optional[int] = None,
        window: _WindowSpec = 'hann',
        center: bool = True,
        pad_mode: Literal['constant', 'reflect', 'edge', 'symmetric'] = 'constant',
    ) -> NDArray[Any]: ...
    @overload
    def spectral_centroid_frames(  # type: ignore[overload-overlap]
        self: AudioSamples[_SingleSample],
        *,
        n_fft: int = 2048,
        hop_length: int = 512,
        win_length: Optional[int] = None,
        window: _WindowSpec = 'hann',
        center: bool = True,
        pad_mode: Literal['constant', 'reflect', 'edge', 'symmetric'] = 'constant',
    ) -> NDArray[np.float32]: ...
    @overload
    def spectral_centroid_frames(
        self: AudioSamples[_DoubleSample],
        *,
        n_fft: int = 2048,
        hop_length: int = 512,
        win_length: Optional[int] = None,
        window: _WindowSpec = 'hann',
        center: bool = True,
        pad_mode: Literal['constant', 'reflect', 'edge', 'symmetric'] = 'constant',
    ) -> NDArray[np.float64]:
        """Spectral centroid per frame: the magnitude-weighted mean frequency.

        Returns:
            numpy.ndarray[float32 | float64], shape ``(channels, 1, n_frames)`` in Hz.
        """
        ...

    @overload
    def spectral_bandwidth_frames(
        self: AudioSamples[_UnknownDtype],
        *,
        n_fft: int = 2048,
        hop_length: int = 512,
        win_length: Optional[int] = None,
        window: _WindowSpec = 'hann',
        center: bool = True,
        pad_mode: Literal['constant', 'reflect', 'edge', 'symmetric'] = 'constant',
        norm: bool = True,
        p: float = 2.0,
    ) -> NDArray[Any]: ...
    @overload
    def spectral_bandwidth_frames(  # type: ignore[overload-overlap]
        self: AudioSamples[_SingleSample],
        *,
        n_fft: int = 2048,
        hop_length: int = 512,
        win_length: Optional[int] = None,
        window: _WindowSpec = 'hann',
        center: bool = True,
        pad_mode: Literal['constant', 'reflect', 'edge', 'symmetric'] = 'constant',
        norm: bool = True,
        p: float = 2.0,
    ) -> NDArray[np.float32]: ...
    @overload
    def spectral_bandwidth_frames(
        self: AudioSamples[_DoubleSample],
        *,
        n_fft: int = 2048,
        hop_length: int = 512,
        win_length: Optional[int] = None,
        window: _WindowSpec = 'hann',
        center: bool = True,
        pad_mode: Literal['constant', 'reflect', 'edge', 'symmetric'] = 'constant',
        norm: bool = True,
        p: float = 2.0,
    ) -> NDArray[np.float64]:
        """Order-``p`` spectral bandwidth per frame around the spectral centroid.

        Returns:
            numpy.ndarray[float32 | float64], shape ``(channels, 1, n_frames)`` in Hz.
        """
        ...

    @overload
    def spectral_rolloff_frames(
        self: AudioSamples[_UnknownDtype],
        *,
        n_fft: int = 2048,
        hop_length: int = 512,
        win_length: Optional[int] = None,
        window: _WindowSpec = 'hann',
        center: bool = True,
        pad_mode: Literal['constant', 'reflect', 'edge', 'symmetric'] = 'constant',
        roll_percent: float = 0.85,
    ) -> NDArray[Any]: ...
    @overload
    def spectral_rolloff_frames(  # type: ignore[overload-overlap]
        self: AudioSamples[_SingleSample],
        *,
        n_fft: int = 2048,
        hop_length: int = 512,
        win_length: Optional[int] = None,
        window: _WindowSpec = 'hann',
        center: bool = True,
        pad_mode: Literal['constant', 'reflect', 'edge', 'symmetric'] = 'constant',
        roll_percent: float = 0.85,
    ) -> NDArray[np.float32]: ...
    @overload
    def spectral_rolloff_frames(
        self: AudioSamples[_DoubleSample],
        *,
        n_fft: int = 2048,
        hop_length: int = 512,
        win_length: Optional[int] = None,
        window: _WindowSpec = 'hann',
        center: bool = True,
        pad_mode: Literal['constant', 'reflect', 'edge', 'symmetric'] = 'constant',
        roll_percent: float = 0.85,
    ) -> NDArray[np.float64]:
        """Roll-off frequency per frame: the frequency below which
        ``roll_percent`` of the energy lies.

        Returns:
            numpy.ndarray[float32 | float64], shape ``(channels, 1, n_frames)`` in Hz.
        """
        ...

    @overload
    def spectral_flatness_frames(
        self: AudioSamples[_UnknownDtype],
        *,
        n_fft: int = 2048,
        hop_length: int = 512,
        win_length: Optional[int] = None,
        window: _WindowSpec = 'hann',
        center: bool = True,
        pad_mode: Literal['constant', 'reflect', 'edge', 'symmetric'] = 'constant',
        amin: float = 1e-10,
        power: float = 2.0,
    ) -> NDArray[Any]: ...
    @overload
    def spectral_flatness_frames(  # type: ignore[overload-overlap]
        self: AudioSamples[_SingleSample],
        *,
        n_fft: int = 2048,
        hop_length: int = 512,
        win_length: Optional[int] = None,
        window: _WindowSpec = 'hann',
        center: bool = True,
        pad_mode: Literal['constant', 'reflect', 'edge', 'symmetric'] = 'constant',
        amin: float = 1e-10,
        power: float = 2.0,
    ) -> NDArray[np.float32]: ...
    @overload
    def spectral_flatness_frames(
        self: AudioSamples[_DoubleSample],
        *,
        n_fft: int = 2048,
        hop_length: int = 512,
        win_length: Optional[int] = None,
        window: _WindowSpec = 'hann',
        center: bool = True,
        pad_mode: Literal['constant', 'reflect', 'edge', 'symmetric'] = 'constant',
        amin: float = 1e-10,
        power: float = 2.0,
    ) -> NDArray[np.float64]:
        """Spectral flatness per frame: geometric over arithmetic mean of the power spectrum.

        Returns:
            numpy.ndarray[float32 | float64], shape ``(channels, 1, n_frames)``, in ``[0, 1]``.
        """
        ...

    @overload
    def spectral_contrast_frames(
        self: AudioSamples[_UnknownDtype],
        *,
        n_fft: int = 2048,
        hop_length: int = 512,
        win_length: Optional[int] = None,
        window: _WindowSpec = 'hann',
        center: bool = True,
        pad_mode: Literal['constant', 'reflect', 'edge', 'symmetric'] = 'constant',
        fmin: float = 200.0,
        n_bands: int = 6,
        quantile: float = 0.02,
        linear: bool = False,
    ) -> NDArray[Any]: ...
    @overload
    def spectral_contrast_frames(  # type: ignore[overload-overlap]
        self: AudioSamples[_SingleSample],
        *,
        n_fft: int = 2048,
        hop_length: int = 512,
        win_length: Optional[int] = None,
        window: _WindowSpec = 'hann',
        center: bool = True,
        pad_mode: Literal['constant', 'reflect', 'edge', 'symmetric'] = 'constant',
        fmin: float = 200.0,
        n_bands: int = 6,
        quantile: float = 0.02,
        linear: bool = False,
    ) -> NDArray[np.float32]: ...
    @overload
    def spectral_contrast_frames(
        self: AudioSamples[_DoubleSample],
        *,
        n_fft: int = 2048,
        hop_length: int = 512,
        win_length: Optional[int] = None,
        window: _WindowSpec = 'hann',
        center: bool = True,
        pad_mode: Literal['constant', 'reflect', 'edge', 'symmetric'] = 'constant',
        fmin: float = 200.0,
        n_bands: int = 6,
        quantile: float = 0.02,
        linear: bool = False,
    ) -> NDArray[np.float64]:
        """Octave-band spectral contrast: peak-to-valley difference in each band, in dB.

        Returns:
            numpy.ndarray[float32 | float64], shape ``(channels, n_bands + 1, n_frames)``.
        """
        ...

    @overload
    def poly_features(
        self: AudioSamples[_UnknownDtype],
        *,
        n_fft: int = 2048,
        hop_length: int = 512,
        win_length: Optional[int] = None,
        window: _WindowSpec = 'hann',
        center: bool = True,
        pad_mode: Literal['constant', 'reflect', 'edge', 'symmetric'] = 'constant',
        order: int = 1,
    ) -> NDArray[Any]: ...
    @overload
    def poly_features(  # type: ignore[overload-overlap]
        self: AudioSamples[_SingleSample],
        *,
        n_fft: int = 2048,
        hop_length: int = 512,
        win_length: Optional[int] = None,
        window: _WindowSpec = 'hann',
        center: bool = True,
        pad_mode: Literal['constant', 'reflect', 'edge', 'symmetric'] = 'constant',
        order: int = 1,
    ) -> NDArray[np.float32]: ...
    @overload
    def poly_features(
        self: AudioSamples[_DoubleSample],
        *,
        n_fft: int = 2048,
        hop_length: int = 512,
        win_length: Optional[int] = None,
        window: _WindowSpec = 'hann',
        center: bool = True,
        pad_mode: Literal['constant', 'reflect', 'edge', 'symmetric'] = 'constant',
        order: int = 1,
    ) -> NDArray[np.float64]:
        """Polynomial fit to every magnitude frame.

        Returns:
            numpy.ndarray[float32 | float64], shape ``(channels, order + 1, n_frames)``, highest power first.
        """
        ...

    @overload
    def rms_frames(
        self: AudioSamples[_UnknownDtype],
        *,
        frame_length: int = 2048,
        hop_length: int = 512,
        center: bool = True,
        pad_mode: Literal['constant', 'reflect', 'edge', 'symmetric'] = 'constant',
    ) -> NDArray[Any]: ...
    @overload
    def rms_frames(  # type: ignore[overload-overlap]
        self: AudioSamples[_SingleSample],
        *,
        frame_length: int = 2048,
        hop_length: int = 512,
        center: bool = True,
        pad_mode: Literal['constant', 'reflect', 'edge', 'symmetric'] = 'constant',
    ) -> NDArray[np.float32]: ...
    @overload
    def rms_frames(
        self: AudioSamples[_DoubleSample],
        *,
        frame_length: int = 2048,
        hop_length: int = 512,
        center: bool = True,
        pad_mode: Literal['constant', 'reflect', 'edge', 'symmetric'] = 'constant',
    ) -> NDArray[np.float64]:
        """Root-mean-square value per frame (whole signal: ``rms()``).

        Returns:
            numpy.ndarray[float32 | float64], shape ``(channels, 1, n_frames)``.
        """
        ...

    @overload
    def zero_crossing_rate_frames(
        self: AudioSamples[_UnknownDtype],
        *,
        frame_length: int = 2048,
        hop_length: int = 512,
        center: bool = True,
        threshold: float = 1e-10,
        ref_magnitude: Union[None, float, Callable[..., Any]] = None,
        pad: bool = False,
        zero_pos: bool = True,
    ) -> NDArray[Any]: ...
    @overload
    def zero_crossing_rate_frames(  # type: ignore[overload-overlap]
        self: AudioSamples[_SingleSample],
        *,
        frame_length: int = 2048,
        hop_length: int = 512,
        center: bool = True,
        threshold: float = 1e-10,
        ref_magnitude: Union[None, float, Callable[..., Any]] = None,
        pad: bool = False,
        zero_pos: bool = True,
    ) -> NDArray[np.float32]: ...
    @overload
    def zero_crossing_rate_frames(
        self: AudioSamples[_DoubleSample],
        *,
        frame_length: int = 2048,
        hop_length: int = 512,
        center: bool = True,
        threshold: float = 1e-10,
        ref_magnitude: Union[None, float, Callable[..., Any]] = None,
        pad: bool = False,
        zero_pos: bool = True,
    ) -> NDArray[np.float64]:
        """Fraction of sign changes per frame (the scalar
        ``zero_crossing_rate()`` is crossings per second).

        Args:
            ref_magnitude: None, a number, or ``numpy.max``.

        Returns:
            numpy.ndarray[float32 | float64], shape ``(channels, 1, n_frames)``.
        """
        ...

    @overload
    def reassigned_spectrogram(
        self: AudioSamples[_UnknownDtype],
        *,
        n_fft: int = 2048,
        hop_length: Optional[int] = None,
        win_length: Optional[int] = None,
        window: _WindowSpec = 'hann',
        center: bool = True,
        reassign_frequencies: bool = True,
        reassign_times: bool = True,
        ref_power: Union[float, Callable[..., Any]] = 1e-06,
        fill_nan: bool = False,
        clip: bool = True,
        pad_mode: Literal['constant', 'reflect', 'edge', 'symmetric'] = 'constant',
    ) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any]]: ...
    @overload
    def reassigned_spectrogram(  # type: ignore[overload-overlap]
        self: AudioSamples[_SingleSample],
        *,
        n_fft: int = 2048,
        hop_length: Optional[int] = None,
        win_length: Optional[int] = None,
        window: _WindowSpec = 'hann',
        center: bool = True,
        reassign_frequencies: bool = True,
        reassign_times: bool = True,
        ref_power: Union[float, Callable[..., Any]] = 1e-06,
        fill_nan: bool = False,
        clip: bool = True,
        pad_mode: Literal['constant', 'reflect', 'edge', 'symmetric'] = 'constant',
    ) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]: ...
    @overload
    def reassigned_spectrogram(
        self: AudioSamples[_DoubleSample],
        *,
        n_fft: int = 2048,
        hop_length: Optional[int] = None,
        win_length: Optional[int] = None,
        window: _WindowSpec = 'hann',
        center: bool = True,
        reassign_frequencies: bool = True,
        reassign_times: bool = True,
        ref_power: Union[float, Callable[..., Any]] = 1e-06,
        fill_nan: bool = False,
        clip: bool = True,
        pad_mode: Literal['constant', 'reflect', 'edge', 'symmetric'] = 'constant',
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Reassigned spectrogram: instantaneous frequency and group-delay time of every bin.

        Returns:
            ``(freqs, times, mags)``, each ``(channels, 1 + n_fft // 2, n_frames)``,
            float32 or float64 by the sample type.
        """
        ...

    @overload
    def chroma_stft(
        self: AudioSamples[_UnknownDtype],
        *,
        norm: Optional[float] = np.inf,
        n_fft: int = 2048,
        hop_length: int = 512,
        win_length: Optional[int] = None,
        window: _WindowSpec = 'hann',
        center: bool = True,
        pad_mode: Literal['constant', 'reflect', 'edge', 'symmetric'] = 'constant',
        tuning: Optional[float] = 0.0,
        n_chroma: int = 12,
        ctroct: float = 5.0,
        octwidth: Optional[float] = 2.0,
        base_c: bool = True,
    ) -> NDArray[Any]: ...
    @overload
    def chroma_stft(  # type: ignore[overload-overlap]
        self: AudioSamples[_SingleSample],
        *,
        norm: Optional[float] = np.inf,
        n_fft: int = 2048,
        hop_length: int = 512,
        win_length: Optional[int] = None,
        window: _WindowSpec = 'hann',
        center: bool = True,
        pad_mode: Literal['constant', 'reflect', 'edge', 'symmetric'] = 'constant',
        tuning: Optional[float] = 0.0,
        n_chroma: int = 12,
        ctroct: float = 5.0,
        octwidth: Optional[float] = 2.0,
        base_c: bool = True,
    ) -> NDArray[np.float32]: ...
    @overload
    def chroma_stft(
        self: AudioSamples[_DoubleSample],
        *,
        norm: Optional[float] = np.inf,
        n_fft: int = 2048,
        hop_length: int = 512,
        win_length: Optional[int] = None,
        window: _WindowSpec = 'hann',
        center: bool = True,
        pad_mode: Literal['constant', 'reflect', 'edge', 'symmetric'] = 'constant',
        tuning: Optional[float] = 0.0,
        n_chroma: int = 12,
        ctroct: float = 5.0,
        octwidth: Optional[float] = 2.0,
        base_c: bool = True,
    ) -> NDArray[np.float64]:
        """Chromagram from the power STFT.

        ``tuning`` defaults to 0.0; ``tuning=None`` estimates it from the signal.

        Returns:
            numpy.ndarray[float32 | float64], shape ``(channels, n_chroma, n_frames)``.
        """
        ...

    @overload
    def piptrack(
        self: AudioSamples[_UnknownDtype],
        *,
        n_fft: int = 2048,
        hop_length: Optional[int] = None,
        fmin: float = 150.0,
        fmax: float = 4000.0,
        threshold: float = 0.1,
        win_length: Optional[int] = None,
        window: _WindowSpec = 'hann',
        center: bool = True,
        pad_mode: Literal['constant', 'reflect', 'edge', 'symmetric'] = 'constant',
        ref: Union[None, float, Callable[..., Any]] = None,
    ) -> tuple[NDArray[Any], NDArray[Any]]: ...
    @overload
    def piptrack(  # type: ignore[overload-overlap]
        self: AudioSamples[_SingleSample],
        *,
        n_fft: int = 2048,
        hop_length: Optional[int] = None,
        fmin: float = 150.0,
        fmax: float = 4000.0,
        threshold: float = 0.1,
        win_length: Optional[int] = None,
        window: _WindowSpec = 'hann',
        center: bool = True,
        pad_mode: Literal['constant', 'reflect', 'edge', 'symmetric'] = 'constant',
        ref: Union[None, float, Callable[..., Any]] = None,
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]: ...
    @overload
    def piptrack(
        self: AudioSamples[_DoubleSample],
        *,
        n_fft: int = 2048,
        hop_length: Optional[int] = None,
        fmin: float = 150.0,
        fmax: float = 4000.0,
        threshold: float = 0.1,
        win_length: Optional[int] = None,
        window: _WindowSpec = 'hann',
        center: bool = True,
        pad_mode: Literal['constant', 'reflect', 'edge', 'symmetric'] = 'constant',
        ref: Union[None, float, Callable[..., Any]] = None,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Spectral peak tracking with parabolic interpolation.

        Returns:
            ``(pitches, magnitudes)``, each ``(channels, 1 + n_fft // 2, n_frames)``,
            float32 or float64 by the sample type.
        """
        ...

    def estimate_tuning(
        self,
        *,
        n_fft: int = 2048,
        resolution: float = 0.01,
        bins_per_octave: int = 12,
        hop_length: Optional[int] = None,
        win_length: Optional[int] = None,
        window: _WindowSpec = "hann",
        center: bool = True,
        pad_mode: Literal["constant", "reflect", "edge", "symmetric"] = "constant",
        fmin: float = 150.0,
        fmax: float = 4000.0,
        threshold: float = 0.1,
        ref: Union[None, float, Callable[..., Any]] = None,
    ) -> float:
        """Tuning deviation from A440 in fractions of a bin, estimated from
        spectral peaks (channels pooled).

        Returns:
            float in ``[-0.5, 0.5)``.
        """
        ...

    @overload
    def cqt(
        self: AudioSamples[_UnknownDtype],
        *,
        hop_length: int = 512,
        fmin: Optional[float] = None,
        n_bins: Optional[int] = 84,
        bins_per_octave: int = 12,
        tuning: Optional[float] = 0.0,
        filter_scale: float = 1,
        norm: Optional[float] = 1,
        sparsity: float = 0.01,
        window: _WindowSpec = 'hann',
        scale: bool = True,
        pad_mode: Literal['constant', 'reflect', 'edge', 'symmetric'] = 'constant',
        res_type: str = 'soxr_hq',
    ) -> NDArray[Any]: ...
    @overload
    def cqt(  # type: ignore[overload-overlap]
        self: AudioSamples[_SingleSample],
        *,
        hop_length: int = 512,
        fmin: Optional[float] = None,
        n_bins: Optional[int] = 84,
        bins_per_octave: int = 12,
        tuning: Optional[float] = 0.0,
        filter_scale: float = 1,
        norm: Optional[float] = 1,
        sparsity: float = 0.01,
        window: _WindowSpec = 'hann',
        scale: bool = True,
        pad_mode: Literal['constant', 'reflect', 'edge', 'symmetric'] = 'constant',
        res_type: str = 'soxr_hq',
    ) -> NDArray[np.complex64]: ...
    @overload
    def cqt(
        self: AudioSamples[_DoubleSample],
        *,
        hop_length: int = 512,
        fmin: Optional[float] = None,
        n_bins: Optional[int] = 84,
        bins_per_octave: int = 12,
        tuning: Optional[float] = 0.0,
        filter_scale: float = 1,
        norm: Optional[float] = 1,
        sparsity: float = 0.01,
        window: _WindowSpec = 'hann',
        scale: bool = True,
        pad_mode: Literal['constant', 'reflect', 'edge', 'symmetric'] = 'constant',
        res_type: str = 'soxr_hq',
    ) -> NDArray[np.complex128]:
        """Constant-Q transform.

        ``res_type`` selects this library's resampler quality for the octave
        decimation: ``"soxr_hq"``/``"soxr_vhq"`` high, ``"soxr_mq"`` medium,
        ``"soxr_lq"`` fast.

        Returns:
            numpy.ndarray[complex64 | complex128], shape ``(channels, n_bins, n_frames)``.
        """
        ...

    @overload
    def vqt(
        self: AudioSamples[_UnknownDtype],
        *,
        hop_length: int = 512,
        fmin: Optional[float] = None,
        n_bins: Optional[int] = 84,
        intervals: Literal['equal'] = 'equal',
        gamma: Optional[float] = None,
        bins_per_octave: int = 12,
        tuning: Optional[float] = 0.0,
        filter_scale: float = 1,
        norm: Optional[float] = 1,
        sparsity: float = 0.01,
        window: _WindowSpec = 'hann',
        scale: bool = True,
        pad_mode: Literal['constant', 'reflect', 'edge', 'symmetric'] = 'constant',
        res_type: str = 'soxr_hq',
    ) -> NDArray[Any]: ...
    @overload
    def vqt(  # type: ignore[overload-overlap]
        self: AudioSamples[_SingleSample],
        *,
        hop_length: int = 512,
        fmin: Optional[float] = None,
        n_bins: Optional[int] = 84,
        intervals: Literal['equal'] = 'equal',
        gamma: Optional[float] = None,
        bins_per_octave: int = 12,
        tuning: Optional[float] = 0.0,
        filter_scale: float = 1,
        norm: Optional[float] = 1,
        sparsity: float = 0.01,
        window: _WindowSpec = 'hann',
        scale: bool = True,
        pad_mode: Literal['constant', 'reflect', 'edge', 'symmetric'] = 'constant',
        res_type: str = 'soxr_hq',
    ) -> NDArray[np.complex64]: ...
    @overload
    def vqt(
        self: AudioSamples[_DoubleSample],
        *,
        hop_length: int = 512,
        fmin: Optional[float] = None,
        n_bins: Optional[int] = 84,
        intervals: Literal['equal'] = 'equal',
        gamma: Optional[float] = None,
        bins_per_octave: int = 12,
        tuning: Optional[float] = 0.0,
        filter_scale: float = 1,
        norm: Optional[float] = 1,
        sparsity: float = 0.01,
        window: _WindowSpec = 'hann',
        scale: bool = True,
        pad_mode: Literal['constant', 'reflect', 'edge', 'symmetric'] = 'constant',
        res_type: str = 'soxr_hq',
    ) -> NDArray[np.complex128]:
        """Variable-Q transform with equal-tempered bin spacing.

        Returns:
            numpy.ndarray[complex64 | complex128], shape ``(channels, n_bins, n_frames)``.
        """
        ...

    @overload
    def pseudo_cqt(
        self: AudioSamples[_UnknownDtype],
        *,
        hop_length: int = 512,
        fmin: Optional[float] = None,
        n_bins: Optional[int] = 84,
        bins_per_octave: int = 12,
        tuning: Optional[float] = 0.0,
        filter_scale: float = 1,
        norm: Optional[float] = 1,
        sparsity: float = 0.01,
        window: _WindowSpec = 'hann',
        scale: bool = True,
        pad_mode: Literal['constant', 'reflect', 'edge', 'symmetric'] = 'constant',
    ) -> NDArray[Any]: ...
    @overload
    def pseudo_cqt(  # type: ignore[overload-overlap]
        self: AudioSamples[_SingleSample],
        *,
        hop_length: int = 512,
        fmin: Optional[float] = None,
        n_bins: Optional[int] = 84,
        bins_per_octave: int = 12,
        tuning: Optional[float] = 0.0,
        filter_scale: float = 1,
        norm: Optional[float] = 1,
        sparsity: float = 0.01,
        window: _WindowSpec = 'hann',
        scale: bool = True,
        pad_mode: Literal['constant', 'reflect', 'edge', 'symmetric'] = 'constant',
    ) -> NDArray[np.float32]: ...
    @overload
    def pseudo_cqt(
        self: AudioSamples[_DoubleSample],
        *,
        hop_length: int = 512,
        fmin: Optional[float] = None,
        n_bins: Optional[int] = 84,
        bins_per_octave: int = 12,
        tuning: Optional[float] = 0.0,
        filter_scale: float = 1,
        norm: Optional[float] = 1,
        sparsity: float = 0.01,
        window: _WindowSpec = 'hann',
        scale: bool = True,
        pad_mode: Literal['constant', 'reflect', 'edge', 'symmetric'] = 'constant',
    ) -> NDArray[np.float64]:
        """Pseudo constant-Q magnitude: constant-Q filters applied to a single STFT.

        Returns:
            numpy.ndarray[float32 | float64], shape ``(channels, n_bins, n_frames)``.
        """
        ...

    @overload
    def hybrid_cqt(
        self: AudioSamples[_UnknownDtype],
        *,
        hop_length: int = 512,
        fmin: Optional[float] = None,
        n_bins: Optional[int] = 84,
        bins_per_octave: int = 12,
        tuning: Optional[float] = 0.0,
        filter_scale: float = 1,
        norm: Optional[float] = 1,
        sparsity: float = 0.01,
        window: _WindowSpec = 'hann',
        scale: bool = True,
        pad_mode: Literal['constant', 'reflect', 'edge', 'symmetric'] = 'constant',
        res_type: str = 'soxr_hq',
    ) -> NDArray[Any]: ...
    @overload
    def hybrid_cqt(  # type: ignore[overload-overlap]
        self: AudioSamples[_SingleSample],
        *,
        hop_length: int = 512,
        fmin: Optional[float] = None,
        n_bins: Optional[int] = 84,
        bins_per_octave: int = 12,
        tuning: Optional[float] = 0.0,
        filter_scale: float = 1,
        norm: Optional[float] = 1,
        sparsity: float = 0.01,
        window: _WindowSpec = 'hann',
        scale: bool = True,
        pad_mode: Literal['constant', 'reflect', 'edge', 'symmetric'] = 'constant',
        res_type: str = 'soxr_hq',
    ) -> NDArray[np.float32]: ...
    @overload
    def hybrid_cqt(
        self: AudioSamples[_DoubleSample],
        *,
        hop_length: int = 512,
        fmin: Optional[float] = None,
        n_bins: Optional[int] = 84,
        bins_per_octave: int = 12,
        tuning: Optional[float] = 0.0,
        filter_scale: float = 1,
        norm: Optional[float] = 1,
        sparsity: float = 0.01,
        window: _WindowSpec = 'hann',
        scale: bool = True,
        pad_mode: Literal['constant', 'reflect', 'edge', 'symmetric'] = 'constant',
        res_type: str = 'soxr_hq',
    ) -> NDArray[np.float64]:
        """Hybrid CQT magnitude: pseudo-CQT for the high bins, full CQT for the low bins.

        Returns:
            numpy.ndarray[float32 | float64], shape ``(channels, n_bins, n_frames)``.
        """
        ...

    @staticmethod
    @overload
    def icqt(
        C: NDArray[_UnknownDtype],
        *,
        sample_rate: int,
        hop_length: int = 512,
        fmin: Optional[float] = None,
        bins_per_octave: int = 12,
        tuning: float = 0.0,
        filter_scale: float = 1,
        norm: Optional[float] = 1,
        sparsity: float = 0.01,
        window: _WindowSpec = 'hann',
        scale: bool = True,
        length: Optional[int] = None,
        res_type: str = 'soxr_hq',
        dtype: None = None,
    ) -> AudioSamples[Any]: ...
    @staticmethod
    @overload
    def icqt(  # type: ignore[overload-overlap]
        C: NDArray[np.complex64] | NDArray[np.float32],
        *,
        sample_rate: int,
        hop_length: int = 512,
        fmin: Optional[float] = None,
        bins_per_octave: int = 12,
        tuning: float = 0.0,
        filter_scale: float = 1,
        norm: Optional[float] = 1,
        sparsity: float = 0.01,
        window: _WindowSpec = 'hann',
        scale: bool = True,
        length: Optional[int] = None,
        res_type: str = 'soxr_hq',
        dtype: None = None,
    ) -> AudioSamples[np.float32]: ...
    @staticmethod
    @overload
    def icqt(
        C: npt.ArrayLike,
        *,
        sample_rate: int,
        hop_length: int = 512,
        fmin: Optional[float] = None,
        bins_per_octave: int = 12,
        tuning: float = 0.0,
        filter_scale: float = 1,
        norm: Optional[float] = 1,
        sparsity: float = 0.01,
        window: _WindowSpec = 'hann',
        scale: bool = True,
        length: Optional[int] = None,
        res_type: str = 'soxr_hq',
        dtype: None = None,
    ) -> AudioSamples[np.float64]: ...
    @staticmethod
    @overload
    def icqt(
        C: npt.ArrayLike,
        *,
        sample_rate: int,
        hop_length: int = 512,
        fmin: Optional[float] = None,
        bins_per_octave: int = 12,
        tuning: float = 0.0,
        filter_scale: float = 1,
        norm: Optional[float] = 1,
        sparsity: float = 0.01,
        window: _WindowSpec = 'hann',
        scale: bool = True,
        length: Optional[int] = None,
        res_type: str = 'soxr_hq',
        dtype: SampleType,
    ) -> AudioSamples[Any]:
        """Inverse constant-Q transform (static). ``C``: complex ``(n_bins, T)`` or
        ``(channels, n_bins, T)``.

        Args:
        dtype: Sample type of the returned audio. None (default) gives float32
            audio for a complex64 input and float64 audio for any other input.
            The computation runs in the float type of ``dtype`` (float32 for
            U8, I16, I24 and F32; float64 for I32 and F64); integer output is
            rounded and saturated.

        Returns:
            AudioSamples: audio of sample type ``dtype``.
        """
        ...

    @staticmethod
    @overload
    def griffinlim_cqt(
        C: NDArray[_UnknownDtype],
        *,
        sample_rate: int,
        n_iter: int = 32,
        hop_length: int = 512,
        fmin: Optional[float] = None,
        bins_per_octave: int = 12,
        tuning: float = 0.0,
        filter_scale: float = 1,
        norm: Optional[float] = 1,
        sparsity: float = 0.01,
        window: _WindowSpec = 'hann',
        scale: bool = True,
        pad_mode: Literal['constant', 'reflect', 'edge', 'symmetric'] = 'constant',
        res_type: str = 'soxr_hq',
        length: Optional[int] = None,
        momentum: float = 0.99,
        init: Optional[Literal['random']] = None,
        rng: _RngLike = 0,
        random_state: _RngLike = None,
        dtype: None = None,
    ) -> AudioSamples[Any]: ...
    @staticmethod
    @overload
    def griffinlim_cqt(  # type: ignore[overload-overlap]
        C: NDArray[np.complex64] | NDArray[np.float32],
        *,
        sample_rate: int,
        n_iter: int = 32,
        hop_length: int = 512,
        fmin: Optional[float] = None,
        bins_per_octave: int = 12,
        tuning: float = 0.0,
        filter_scale: float = 1,
        norm: Optional[float] = 1,
        sparsity: float = 0.01,
        window: _WindowSpec = 'hann',
        scale: bool = True,
        pad_mode: Literal['constant', 'reflect', 'edge', 'symmetric'] = 'constant',
        res_type: str = 'soxr_hq',
        length: Optional[int] = None,
        momentum: float = 0.99,
        init: Optional[Literal['random']] = None,
        rng: _RngLike = 0,
        random_state: _RngLike = None,
        dtype: None = None,
    ) -> AudioSamples[np.float32]: ...
    @staticmethod
    @overload
    def griffinlim_cqt(
        C: npt.ArrayLike,
        *,
        sample_rate: int,
        n_iter: int = 32,
        hop_length: int = 512,
        fmin: Optional[float] = None,
        bins_per_octave: int = 12,
        tuning: float = 0.0,
        filter_scale: float = 1,
        norm: Optional[float] = 1,
        sparsity: float = 0.01,
        window: _WindowSpec = 'hann',
        scale: bool = True,
        pad_mode: Literal['constant', 'reflect', 'edge', 'symmetric'] = 'constant',
        res_type: str = 'soxr_hq',
        length: Optional[int] = None,
        momentum: float = 0.99,
        init: Optional[Literal['random']] = None,
        rng: _RngLike = 0,
        random_state: _RngLike = None,
        dtype: None = None,
    ) -> AudioSamples[np.float64]: ...
    @staticmethod
    @overload
    def griffinlim_cqt(
        C: npt.ArrayLike,
        *,
        sample_rate: int,
        n_iter: int = 32,
        hop_length: int = 512,
        fmin: Optional[float] = None,
        bins_per_octave: int = 12,
        tuning: float = 0.0,
        filter_scale: float = 1,
        norm: Optional[float] = 1,
        sparsity: float = 0.01,
        window: _WindowSpec = 'hann',
        scale: bool = True,
        pad_mode: Literal['constant', 'reflect', 'edge', 'symmetric'] = 'constant',
        res_type: str = 'soxr_hq',
        length: Optional[int] = None,
        momentum: float = 0.99,
        init: Optional[Literal['random']] = None,
        rng: _RngLike = 0,
        random_state: _RngLike = None,
        dtype: SampleType,
    ) -> AudioSamples[Any]:
        """Griffin-Lim phase retrieval from a CQT magnitude (static).
        ``init=None`` (zero phase) is the default; ``init="random"`` draws its
        phases from this library's RNG as in ``effects.griffinlim``.

        Args:
        dtype: Sample type of the returned audio. None (default) gives float32
            audio for a float32 input and float64 audio for any other input.
            The computation runs in the float type of ``dtype`` (float32 for
            U8, I16, I24 and F32; float64 for I32 and F64); integer output is
            rounded and saturated.

        Returns:
            AudioSamples: audio of sample type ``dtype``.
        """
        ...

    @overload
    def chroma_cqt(
        self: AudioSamples[_UnknownDtype],
        *,
        hop_length: int = 512,
        fmin: Optional[float] = None,
        norm: Optional[float] = np.inf,
        threshold: Optional[float] = 0.0,
        tuning: Optional[float] = 0.0,
        n_chroma: int = 12,
        n_octaves: int = 7,
        window: None = None,
        bins_per_octave: Optional[int] = 36,
        cqt_mode: Literal['full', 'hybrid'] = 'full',
    ) -> NDArray[Any]: ...
    @overload
    def chroma_cqt(  # type: ignore[overload-overlap]
        self: AudioSamples[_SingleSample],
        *,
        hop_length: int = 512,
        fmin: Optional[float] = None,
        norm: Optional[float] = np.inf,
        threshold: Optional[float] = 0.0,
        tuning: Optional[float] = 0.0,
        n_chroma: int = 12,
        n_octaves: int = 7,
        window: None = None,
        bins_per_octave: Optional[int] = 36,
        cqt_mode: Literal['full', 'hybrid'] = 'full',
    ) -> NDArray[np.float32]: ...
    @overload
    def chroma_cqt(
        self: AudioSamples[_DoubleSample],
        *,
        hop_length: int = 512,
        fmin: Optional[float] = None,
        norm: Optional[float] = np.inf,
        threshold: Optional[float] = 0.0,
        tuning: Optional[float] = 0.0,
        n_chroma: int = 12,
        n_octaves: int = 7,
        window: None = None,
        bins_per_octave: Optional[int] = 36,
        cqt_mode: Literal['full', 'hybrid'] = 'full',
    ) -> NDArray[np.float64]:
        """Constant-Q chromagram (``window`` must be None).

        Returns:
            numpy.ndarray[float32 | float64], shape ``(channels, n_chroma, n_frames)``.
        """
        ...

    @overload
    def chroma_cens(
        self: AudioSamples[_UnknownDtype],
        *,
        hop_length: int = 512,
        fmin: Optional[float] = None,
        tuning: Optional[float] = 0.0,
        n_chroma: int = 12,
        n_octaves: int = 7,
        bins_per_octave: int = 36,
        cqt_mode: Literal['full', 'hybrid'] = 'full',
        window: None = None,
        norm: Optional[float] = 2,
        win_len_smooth: Optional[int] = 41,
        smoothing_window: _WindowSpec = 'hann',
    ) -> NDArray[Any]: ...
    @overload
    def chroma_cens(  # type: ignore[overload-overlap]
        self: AudioSamples[_SingleSample],
        *,
        hop_length: int = 512,
        fmin: Optional[float] = None,
        tuning: Optional[float] = 0.0,
        n_chroma: int = 12,
        n_octaves: int = 7,
        bins_per_octave: int = 36,
        cqt_mode: Literal['full', 'hybrid'] = 'full',
        window: None = None,
        norm: Optional[float] = 2,
        win_len_smooth: Optional[int] = 41,
        smoothing_window: _WindowSpec = 'hann',
    ) -> NDArray[np.float32]: ...
    @overload
    def chroma_cens(
        self: AudioSamples[_DoubleSample],
        *,
        hop_length: int = 512,
        fmin: Optional[float] = None,
        tuning: Optional[float] = 0.0,
        n_chroma: int = 12,
        n_octaves: int = 7,
        bins_per_octave: int = 36,
        cqt_mode: Literal['full', 'hybrid'] = 'full',
        window: None = None,
        norm: Optional[float] = 2,
        win_len_smooth: Optional[int] = 41,
        smoothing_window: _WindowSpec = 'hann',
    ) -> NDArray[np.float64]:
        """Chroma energy normalised statistics (CENS) features.

        Returns:
            numpy.ndarray[float32 | float64], shape ``(channels, n_chroma, n_frames)``.
        """
        ...

    @overload
    def chroma_vqt(
        self: AudioSamples[_UnknownDtype],
        *,
        hop_length: int = 512,
        fmin: Optional[float] = None,
        intervals: Literal['equal'] = 'equal',
        norm: Optional[float] = np.inf,
        threshold: Optional[float] = 0.0,
        n_octaves: int = 7,
        bins_per_octave: int = 12,
        gamma: Optional[float] = 0,
    ) -> NDArray[Any]: ...
    @overload
    def chroma_vqt(  # type: ignore[overload-overlap]
        self: AudioSamples[_SingleSample],
        *,
        hop_length: int = 512,
        fmin: Optional[float] = None,
        intervals: Literal['equal'] = 'equal',
        norm: Optional[float] = np.inf,
        threshold: Optional[float] = 0.0,
        n_octaves: int = 7,
        bins_per_octave: int = 12,
        gamma: Optional[float] = 0,
    ) -> NDArray[np.float32]: ...
    @overload
    def chroma_vqt(
        self: AudioSamples[_DoubleSample],
        *,
        hop_length: int = 512,
        fmin: Optional[float] = None,
        intervals: Literal['equal'] = 'equal',
        norm: Optional[float] = np.inf,
        threshold: Optional[float] = 0.0,
        n_octaves: int = 7,
        bins_per_octave: int = 12,
        gamma: Optional[float] = 0,
    ) -> NDArray[np.float64]:
        """Variable-Q chromagram with equal-tempered bin spacing.

        Returns:
            numpy.ndarray[float32 | float64], shape ``(channels, bins_per_octave, n_frames)``.
        """
        ...

    @overload
    def tonnetz(
        self: AudioSamples[_UnknownDtype],
        *,
        hop_length: int = 512,
        fmin: Optional[float] = None,
        norm: Optional[float] = np.inf,
        threshold: Optional[float] = 0.0,
        tuning: Optional[float] = 0.0,
        n_chroma: int = 12,
        n_octaves: int = 7,
        bins_per_octave: Optional[int] = 36,
        cqt_mode: Literal['full', 'hybrid'] = 'full',
    ) -> NDArray[Any]: ...
    @overload
    def tonnetz(  # type: ignore[overload-overlap]
        self: AudioSamples[_SingleSample],
        *,
        hop_length: int = 512,
        fmin: Optional[float] = None,
        norm: Optional[float] = np.inf,
        threshold: Optional[float] = 0.0,
        tuning: Optional[float] = 0.0,
        n_chroma: int = 12,
        n_octaves: int = 7,
        bins_per_octave: Optional[int] = 36,
        cqt_mode: Literal['full', 'hybrid'] = 'full',
    ) -> NDArray[np.float32]: ...
    @overload
    def tonnetz(
        self: AudioSamples[_DoubleSample],
        *,
        hop_length: int = 512,
        fmin: Optional[float] = None,
        norm: Optional[float] = np.inf,
        threshold: Optional[float] = 0.0,
        tuning: Optional[float] = 0.0,
        n_chroma: int = 12,
        n_octaves: int = 7,
        bins_per_octave: Optional[int] = 36,
        cqt_mode: Literal['full', 'hybrid'] = 'full',
    ) -> NDArray[np.float64]:
        """Tonal centroid (tonnetz) features (keywords go to ``chroma_cqt``).

        Returns:
            numpy.ndarray[float32 | float64], shape ``(channels, 6, n_frames)``.
        """
        ...

    @overload
    def yin(
        self: AudioSamples[_UnknownDtype],
        *,
        fmin: float,
        fmax: float,
        frame_length: int = 2048,
        hop_length: Optional[int] = None,
        trough_threshold: float = 0.1,
        center: bool = True,
        pad_mode: Literal['constant', 'reflect', 'edge', 'symmetric'] = 'constant',
    ) -> NDArray[Any]: ...
    @overload
    def yin(  # type: ignore[overload-overlap]
        self: AudioSamples[_SingleSample],
        *,
        fmin: float,
        fmax: float,
        frame_length: int = 2048,
        hop_length: Optional[int] = None,
        trough_threshold: float = 0.1,
        center: bool = True,
        pad_mode: Literal['constant', 'reflect', 'edge', 'symmetric'] = 'constant',
    ) -> NDArray[np.float32]: ...
    @overload
    def yin(
        self: AudioSamples[_DoubleSample],
        *,
        fmin: float,
        fmax: float,
        frame_length: int = 2048,
        hop_length: Optional[int] = None,
        trough_threshold: float = 0.1,
        center: bool = True,
        pad_mode: Literal['constant', 'reflect', 'edge', 'symmetric'] = 'constant',
    ) -> NDArray[np.float64]:
        """YIN f0 per frame (``fmin``/``fmax`` required).

        Returns:
            numpy.ndarray[float32 | float64] in Hz, shape ``(channels, n_frames)``.
        """
        ...

    @overload
    def pyin(
        self: AudioSamples[_UnknownDtype],
        *,
        fmin: float,
        fmax: float,
        frame_length: int = 2048,
        hop_length: Optional[int] = None,
        n_thresholds: int = 100,
        beta_parameters: tuple[float, float] = (2, 18),
        boltzmann_parameter: float = 2,
        resolution: float = 0.1,
        max_transition_rate: float = 35.92,
        switch_prob: float = 0.01,
        no_trough_prob: float = 0.01,
        fill_na: Optional[float] = np.nan,
        center: bool = True,
        pad_mode: Literal['constant', 'reflect', 'edge', 'symmetric'] = 'constant',
        transition_min_prob: Optional[float] = 0.0001,
    ) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any]]: ...
    @overload
    def pyin(  # type: ignore[overload-overlap]
        self: AudioSamples[_SingleSample],
        *,
        fmin: float,
        fmax: float,
        frame_length: int = 2048,
        hop_length: Optional[int] = None,
        n_thresholds: int = 100,
        beta_parameters: tuple[float, float] = (2, 18),
        boltzmann_parameter: float = 2,
        resolution: float = 0.1,
        max_transition_rate: float = 35.92,
        switch_prob: float = 0.01,
        no_trough_prob: float = 0.01,
        fill_na: Optional[float] = np.nan,
        center: bool = True,
        pad_mode: Literal['constant', 'reflect', 'edge', 'symmetric'] = 'constant',
        transition_min_prob: Optional[float] = 0.0001,
    ) -> tuple[NDArray[np.float32], NDArray[np.bool_], NDArray[np.float32]]: ...
    @overload
    def pyin(
        self: AudioSamples[_DoubleSample],
        *,
        fmin: float,
        fmax: float,
        frame_length: int = 2048,
        hop_length: Optional[int] = None,
        n_thresholds: int = 100,
        beta_parameters: tuple[float, float] = (2, 18),
        boltzmann_parameter: float = 2,
        resolution: float = 0.1,
        max_transition_rate: float = 35.92,
        switch_prob: float = 0.01,
        no_trough_prob: float = 0.01,
        fill_na: Optional[float] = np.nan,
        center: bool = True,
        pad_mode: Literal['constant', 'reflect', 'edge', 'symmetric'] = 'constant',
        transition_min_prob: Optional[float] = 0.0001,
    ) -> tuple[NDArray[np.float64], NDArray[np.bool_], NDArray[np.float64]]:
        """Probabilistic YIN f0 tracking with voicing decisions.

        Args:
            fill_na: f0 of unvoiced frames: NaN (default), None (best guess) or a number.

        Returns:
            ``(f0, voiced_flag, voiced_prob)``, each ``(channels, n_frames)``.
        """
        ...

    # ------------------------------------------------------------------
    # Rhythm analysis (AudioRhythm)
    #
    # Frame-wise outputs carry a leading channel axis (mono gives length 1);
    # ``result[0]`` drops it.
    # They follow the sample type's float (float32 for uint8, int16, int24 and
    # float32 audio); tempo is always float64. ``beat_track`` defaults to
    # ``trim=False``.
    # Event detectors (onset_detect, beat_track) need mono audio. ``aggregate``
    # takes numpy reductions, their names or a callable f(x, axis=-1);
    # ``prior`` a scipy.stats distribution or a callable of the BPM.
    # Free-function forms on arrays: ``audio_samples.rhythm``.
    # ------------------------------------------------------------------

    @overload
    def onset_strength(
        self: AudioSamples[_UnknownDtype],
        *,
        lag: int = 1,
        max_size: int = 1,
        detrend: bool = False,
        center: bool = True,
        aggregate: Any = None,
        n_fft: int = 2048,
        hop_length: int = 512,
        win_length: Optional[int] = None,
        window: _WindowSpec = 'hann',
        pad_mode: Literal['constant', 'reflect', 'edge', 'symmetric'] = 'constant',
        power: float = 2.0,
        n_mels: int = 128,
        fmin: float = 0.0,
        fmax: Optional[float] = None,
        htk: bool = False,
        norm: _MelNormSpec = 'slaney',
    ) -> NDArray[Any]: ...
    @overload
    def onset_strength(  # type: ignore[overload-overlap]
        self: AudioSamples[_SingleSample],
        *,
        lag: int = 1,
        max_size: int = 1,
        detrend: bool = False,
        center: bool = True,
        aggregate: Any = None,
        n_fft: int = 2048,
        hop_length: int = 512,
        win_length: Optional[int] = None,
        window: _WindowSpec = 'hann',
        pad_mode: Literal['constant', 'reflect', 'edge', 'symmetric'] = 'constant',
        power: float = 2.0,
        n_mels: int = 128,
        fmin: float = 0.0,
        fmax: Optional[float] = None,
        htk: bool = False,
        norm: _MelNormSpec = 'slaney',
    ) -> NDArray[np.float32]: ...
    @overload
    def onset_strength(
        self: AudioSamples[_DoubleSample],
        *,
        lag: int = 1,
        max_size: int = 1,
        detrend: bool = False,
        center: bool = True,
        aggregate: Any = None,
        n_fft: int = 2048,
        hop_length: int = 512,
        win_length: Optional[int] = None,
        window: _WindowSpec = 'hann',
        pad_mode: Literal['constant', 'reflect', 'edge', 'symmetric'] = 'constant',
        power: float = 2.0,
        n_mels: int = 128,
        fmin: float = 0.0,
        fmax: Optional[float] = None,
        htk: bool = False,
        norm: _MelNormSpec = 'slaney',
    ) -> NDArray[np.float64]:
        """Onset strength envelope (spectral flux of the log-power mel spectrogram).

        Positive increase of the log-power mel spectrogram (``power_to_db``,
        ``ref=1``, ``top_db=80``) from frame ``t - lag``, aggregated across bins.

        Returns:
            numpy.ndarray[float32 | float64], shape ``(channels, n_frames)``,
            ``n_frames = 1 + len // hop_length``.
        """
        ...

    @overload
    def onset_strength_multi(
        self: AudioSamples[_UnknownDtype],
        *,
        channels: Optional[Union[list[int], list[slice]]] = None,
        lag: int = 1,
        max_size: int = 1,
        detrend: bool = False,
        center: bool = True,
        aggregate: Any = None,
        n_fft: int = 2048,
        hop_length: int = 512,
        win_length: Optional[int] = None,
        window: _WindowSpec = 'hann',
        pad_mode: Literal['constant', 'reflect', 'edge', 'symmetric'] = 'constant',
        power: float = 2.0,
        n_mels: int = 128,
        fmin: float = 0.0,
        fmax: Optional[float] = None,
        htk: bool = False,
        norm: _MelNormSpec = 'slaney',
    ) -> NDArray[Any]: ...
    @overload
    def onset_strength_multi(  # type: ignore[overload-overlap]
        self: AudioSamples[_SingleSample],
        *,
        channels: Optional[Union[list[int], list[slice]]] = None,
        lag: int = 1,
        max_size: int = 1,
        detrend: bool = False,
        center: bool = True,
        aggregate: Any = None,
        n_fft: int = 2048,
        hop_length: int = 512,
        win_length: Optional[int] = None,
        window: _WindowSpec = 'hann',
        pad_mode: Literal['constant', 'reflect', 'edge', 'symmetric'] = 'constant',
        power: float = 2.0,
        n_mels: int = 128,
        fmin: float = 0.0,
        fmax: Optional[float] = None,
        htk: bool = False,
        norm: _MelNormSpec = 'slaney',
    ) -> NDArray[np.float32]: ...
    @overload
    def onset_strength_multi(
        self: AudioSamples[_DoubleSample],
        *,
        channels: Optional[Union[list[int], list[slice]]] = None,
        lag: int = 1,
        max_size: int = 1,
        detrend: bool = False,
        center: bool = True,
        aggregate: Any = None,
        n_fft: int = 2048,
        hop_length: int = 512,
        win_length: Optional[int] = None,
        window: _WindowSpec = 'hann',
        pad_mode: Literal['constant', 'reflect', 'edge', 'symmetric'] = 'constant',
        power: float = 2.0,
        n_mels: int = 128,
        fmin: float = 0.0,
        fmax: Optional[float] = None,
        htk: bool = False,
        norm: _MelNormSpec = 'slaney',
    ) -> NDArray[np.float64]:
        """Multi-band onset strength: one envelope per group of mel bins.

        ``channels`` are sub-band boundaries in mel bins; ``aggregate=False``
        keeps every mel bin.

        Returns:
            numpy.ndarray[float32 | float64], shape ``(channels, n_bands, n_frames)``.
        """
        ...

    def onset_detect(
        self,
        *,
        hop_length: int = 512,
        backtrack: bool = False,
        energy: Optional[npt.ArrayLike] = None,
        units: Literal["frames", "samples", "time"] = "frames",
        normalize: bool = True,
        sparse: bool = True,
        pre_max: Optional[int] = None,
        post_max: Optional[int] = None,
        pre_avg: Optional[int] = None,
        post_avg: Optional[int] = None,
        delta: Optional[float] = None,
        wait: Optional[int] = None,
        method: Literal["greedy", "dp_count", "dp_value"] = "greedy",
    ) -> NDArray[Any]:
        """Onset detection by peak picking on the onset strength envelope (mono only).

        Returns:
            1-D numpy.ndarray: int64 frames/samples or float64 times.

        Raises:
            ValueError: Multi-channel audio, invalid parameters or ``sparse=False``.
        """
        ...

    @overload
    def tempogram(
        self: AudioSamples[_UnknownDtype],
        *,
        hop_length: int = 512,
        win_length: int = 384,
        center: bool = True,
        window: _WindowSpec = 'hann',
        norm: Optional[float] = np.inf,
    ) -> NDArray[Any]: ...
    @overload
    def tempogram(  # type: ignore[overload-overlap]
        self: AudioSamples[_SingleSample],
        *,
        hop_length: int = 512,
        win_length: int = 384,
        center: bool = True,
        window: _WindowSpec = 'hann',
        norm: Optional[float] = np.inf,
    ) -> NDArray[np.float32]: ...
    @overload
    def tempogram(
        self: AudioSamples[_DoubleSample],
        *,
        hop_length: int = 512,
        win_length: int = 384,
        center: bool = True,
        window: _WindowSpec = 'hann',
        norm: Optional[float] = np.inf,
    ) -> NDArray[np.float64]:
        """Autocorrelation tempogram: windowed autocorrelation of the onset strength envelope.

        Returns:
            numpy.ndarray[float32 | float64], shape ``(channels, win_length, n_frames)``.
        """
        ...

    @overload
    def fourier_tempogram(
        self: AudioSamples[_UnknownDtype],
        *,
        hop_length: int = 512,
        win_length: int = 384,
        center: bool = True,
        window: _WindowSpec = 'hann',
    ) -> NDArray[Any]: ...
    @overload
    def fourier_tempogram(  # type: ignore[overload-overlap]
        self: AudioSamples[_SingleSample],
        *,
        hop_length: int = 512,
        win_length: int = 384,
        center: bool = True,
        window: _WindowSpec = 'hann',
    ) -> NDArray[np.complex64]: ...
    @overload
    def fourier_tempogram(
        self: AudioSamples[_DoubleSample],
        *,
        hop_length: int = 512,
        win_length: int = 384,
        center: bool = True,
        window: _WindowSpec = 'hann',
    ) -> NDArray[np.complex128]:
        """Fourier tempogram: short-time Fourier transform of the onset strength envelope.

        Returns:
            numpy.ndarray[complex64 | complex128], shape
            ``(channels, win_length // 2 + 1, n_frames + 1)`` (centred).
        """
        ...

    def tempo(
        self,
        *,
        hop_length: int = 512,
        start_bpm: float = 120,
        std_bpm: float = 1.0,
        ac_size: float = 8.0,
        max_tempo: Optional[float] = 320.0,
        aggregate: Union[Callable[..., Any], Literal["mean", "median", "max", "min", "sum"]] = "mean",
        prior: Any = None,
    ) -> NDArray[np.float64]:
        """Tempo estimate in BPM from the autocorrelation tempogram.

        Returns:
            numpy.ndarray[float64], shape ``(channels, 1)`` (``(channels, n_frames)``
            with ``aggregate=None``).
        """
        ...

    def beat_track(
        self,
        *,
        hop_length: int = 512,
        start_bpm: float = 120.0,
        tightness: float = 100,
        trim: bool = False,
        bpm: Optional[Union[float, npt.ArrayLike]] = None,
        prior: Any = None,
        units: Literal["frames", "samples", "time"] = "frames",
        sparse: bool = True,
    ) -> tuple[Any, NDArray[Any]]:
        """Dynamic-programming beat tracking (mono only; median-aggregated onset
        envelope).

        ``trim`` defaults to False, so weak leading and trailing beats are kept.

        Returns:
            ``(tempo, beats)``: ``tempo`` a float64 array of shape
            ``(1,)`` (``bpm`` unchanged when given, ``0.0`` for silence);
            ``beats`` 1-D int64 frames/samples or float64 times.
        """
        ...

    @overload
    def plp(
        self: AudioSamples[_UnknownDtype],
        *,
        hop_length: int = 512,
        win_length: int = 384,
        tempo_min: Optional[float] = 30,
        tempo_max: Optional[float] = 300,
        prior: Any = None,
    ) -> NDArray[Any]: ...
    @overload
    def plp(  # type: ignore[overload-overlap]
        self: AudioSamples[_SingleSample],
        *,
        hop_length: int = 512,
        win_length: int = 384,
        tempo_min: Optional[float] = 30,
        tempo_max: Optional[float] = 300,
        prior: Any = None,
    ) -> NDArray[np.float32]: ...
    @overload
    def plp(
        self: AudioSamples[_DoubleSample],
        *,
        hop_length: int = 512,
        win_length: int = 384,
        tempo_min: Optional[float] = 30,
        tempo_max: Optional[float] = 300,
        prior: Any = None,
    ) -> NDArray[np.float64]:
        """Predominant local pulse from the Fourier tempogram.

        Returns:
            numpy.ndarray[float32 | float64], shape ``(channels, n_frames)``, values in ``[0, 1]``.
        """
        ...

    @overload
    def tempogram_ratio(
        self: AudioSamples[_UnknownDtype],
        *,
        hop_length: int = 512,
        win_length: int = 384,
        start_bpm: float = 120,
        std_bpm: float = 1.0,
        max_tempo: Optional[float] = 320.0,
        factors: Optional[npt.ArrayLike] = None,
        aggregate: Any = None,
        prior: Any = None,
        center: bool = True,
        window: _WindowSpec = 'hann',
        fill_value: float = 0,
        norm: Optional[float] = np.inf,
    ) -> NDArray[Any]: ...
    @overload
    def tempogram_ratio(  # type: ignore[overload-overlap]
        self: AudioSamples[_SingleSample],
        *,
        hop_length: int = 512,
        win_length: int = 384,
        start_bpm: float = 120,
        std_bpm: float = 1.0,
        max_tempo: Optional[float] = 320.0,
        factors: Optional[npt.ArrayLike] = None,
        aggregate: Any = None,
        prior: Any = None,
        center: bool = True,
        window: _WindowSpec = 'hann',
        fill_value: float = 0,
        norm: Optional[float] = np.inf,
    ) -> NDArray[np.float32]: ...
    @overload
    def tempogram_ratio(
        self: AudioSamples[_DoubleSample],
        *,
        hop_length: int = 512,
        win_length: int = 384,
        start_bpm: float = 120,
        std_bpm: float = 1.0,
        max_tempo: Optional[float] = 320.0,
        factors: Optional[npt.ArrayLike] = None,
        aggregate: Any = None,
        prior: Any = None,
        center: bool = True,
        window: _WindowSpec = 'hann',
        fill_value: float = 0,
        norm: Optional[float] = np.inf,
    ) -> NDArray[np.float64]:
        """Tempogram ratio: the tempogram sampled at fixed multiples of the
        per-frame tempo estimate.

        Returns:
            numpy.ndarray[float32 | float64], shape ``(channels, n_factors, n_frames)``
            (``(channels, n_factors, 1)`` when aggregated).
        """
        ...

    @overload
    def hybrid_tempogram(
        self: AudioSamples[_UnknownDtype],
        *,
        hop_length: int = 512,
        win_length: int = 384,
        center: bool = True,
        window: _WindowSpec = 'hann',
    ) -> NDArray[Any]: ...
    @overload
    def hybrid_tempogram(  # type: ignore[overload-overlap]
        self: AudioSamples[_SingleSample],
        *,
        hop_length: int = 512,
        win_length: int = 384,
        center: bool = True,
        window: _WindowSpec = 'hann',
    ) -> NDArray[np.float32]: ...
    @overload
    def hybrid_tempogram(
        self: AudioSamples[_DoubleSample],
        *,
        hop_length: int = 512,
        win_length: int = 384,
        center: bool = True,
        window: _WindowSpec = 'hann',
    ) -> NDArray[np.float64]:
        """Hybrid tempogram: the Fourier and autocorrelation tempograms combined
        on the Fourier tempo grid.

        Returns:
            numpy.ndarray[float32 | float64], shape ``(channels, win_length // 2 + 1, n_frames)``.
        """
        ...

    # ------------------------------------------------------------------
    # Effects (AudioEffects)
    #
    # Audio-to-audio effects return AudioSamples of the input's sample type
    # and rate; each has an ``_in_place`` variant (the length may change).
    # ``trim_db`` / ``split_db`` require ``top_db`` and ``remix`` defaults to
    # ``align_zeros=False``.
    # Array forms: ``audio_samples.effects``.
    # ------------------------------------------------------------------

    def time_stretch(
        self,
        rate: float,
        *,
        n_fft: int = 2048,
        hop_length: Optional[int] = None,
        win_length: Optional[int] = None,
        window: _WindowSpec = "hann",
        center: bool = True,
        pad_mode: Literal["constant", "reflect", "edge", "symmetric"] = "constant",
    ) -> Self:
        """Time stretch by ``rate`` with a phase vocoder: exactly ``round(n / rate)`` samples."""
        ...

    def time_stretch_in_place(
        self,
        rate: float,
        *,
        n_fft: int = 2048,
        hop_length: Optional[int] = None,
        win_length: Optional[int] = None,
        window: _WindowSpec = "hann",
        center: bool = True,
        pad_mode: Literal["constant", "reflect", "edge", "symmetric"] = "constant",
    ) -> None:
        """In-place ``time_stretch``."""
        ...

    def pitch_shift(
        self,
        n_steps: float,
        *,
        bins_per_octave: int = 12,
        res_type: Union[str, ResamplingQuality] = "soxr_hq",
        scale: bool = False,
        n_fft: int = 2048,
        hop_length: Optional[int] = None,
        win_length: Optional[int] = None,
        window: _WindowSpec = "hann",
        center: bool = True,
        pad_mode: Literal["constant", "reflect", "edge", "symmetric"] = "constant",
    ) -> Self:
        """Shift the pitch by ``n_steps`` steps of ``bins_per_octave`` per octave,
        keeping the duration (time stretch, then resample).

        The resampling step uses this library's resampler (``"soxr_hq"`` maps to
        ``ResamplingQuality.high``, ``"soxr_mq"`` to medium, ``"soxr_lq"`` to
        fast). ``scale`` must be False.
        """
        ...

    def pitch_shift_in_place(
        self,
        n_steps: float,
        *,
        bins_per_octave: int = 12,
        res_type: Union[str, ResamplingQuality] = "soxr_hq",
        scale: bool = False,
        n_fft: int = 2048,
        hop_length: Optional[int] = None,
        win_length: Optional[int] = None,
        window: _WindowSpec = "hann",
        center: bool = True,
        pad_mode: Literal["constant", "reflect", "edge", "symmetric"] = "constant",
    ) -> None:
        """In-place ``pitch_shift``."""
        ...

    def preemphasis(self, *, coef: float = 0.97) -> Self:
        """Pre-emphasis filter ``y[n] = x[n] - coef * x[n - 1]`` on every channel
        (default initial state)."""
        ...

    def preemphasis_in_place(self, *, coef: float = 0.97) -> None:
        """In-place ``preemphasis``."""
        ...

    def deemphasis(self, *, coef: float = 0.97) -> Self:
        """De-emphasis filter on every channel (inverse of ``preemphasis``)."""
        ...

    def deemphasis_in_place(self, *, coef: float = 0.97) -> None:
        """In-place ``deemphasis``."""
        ...

    def trim_db(
        self,
        *,
        top_db: float,
        ref: _DbRef = "max",
        frame_length: int = 2048,
        hop_length: int = 512,
        aggregate: Optional[Union[Callable[..., Any], Literal['max', 'min', 'mean']]] = None,
    ) -> tuple[Self, NDArray[np.int64]]:
        """Trim leading and trailing frames quieter than ``top_db`` below the
        reference: returns ``(trimmed, [start, end])``.

        ``top_db`` (dB below the reference) is required.
        ``aggregate`` is ``numpy.max`` (default), ``numpy.min``, ``numpy.mean`` or any
        callable, called as ``aggregate(db, 0)``; ``ref`` may be any callable
        ``ref(rms, axis=..., keepdims=True)``. Callables run with the GIL held.

        Raises:
            ValueError: If everything is silent (an ``AudioSamples`` cannot be empty).
        """
        ...

    def trim_db_in_place(
        self,
        *,
        top_db: float,
        ref: _DbRef = "max",
        frame_length: int = 2048,
        hop_length: int = 512,
        aggregate: Optional[Union[Callable[..., Any], Literal['max', 'min', 'mean']]] = None,
    ) -> NDArray[np.int64]:
        """In-place ``trim_db``; returns ``[start, end]``."""
        ...

    def split_db(
        self,
        *,
        top_db: float,
        ref: _DbRef = "max",
        frame_length: int = 2048,
        hop_length: int = 512,
        aggregate: Optional[Union[Callable[..., Any], Literal['max', 'min', 'mean']]] = None,
    ) -> NDArray[np.int64]:
        """Non-silent intervals: frames within ``top_db`` of the reference, as
        int64 sample intervals of shape ``(n, 2)``.

        ``top_db`` is required."""
        ...

    def remix(self, intervals: npt.ArrayLike, *, align_zeros: bool = False) -> Self:
        """Concatenate ``(n, 2)`` sample intervals in the given order.

        ``align_zeros`` snaps the boundaries to zero crossings; it defaults to
        False."""
        ...

    def remix_in_place(self, intervals: npt.ArrayLike, *, align_zeros: bool = False) -> None:
        """In-place ``remix``."""
        ...

    def harmonic(
        self,
        *,
        kernel_size: Union[int, tuple[int, int]] = 31,
        power: float = 2.0,
        mask: bool = False,
        margin: Union[float, tuple[float, float]] = 1.0,
        n_fft: int = 2048,
        hop_length: Optional[int] = None,
        win_length: Optional[int] = None,
        window: _WindowSpec = "hann",
        center: bool = True,
        pad_mode: Literal["constant", "reflect", "edge", "symmetric"] = "constant",
    ) -> Self:
        """Harmonic component from median-filtering harmonic/percussive separation
        (``mask`` must be False; ``window`` sets the STFT window)."""
        ...

    def harmonic_in_place(
        self,
        *,
        kernel_size: Union[int, tuple[int, int]] = 31,
        power: float = 2.0,
        mask: bool = False,
        margin: Union[float, tuple[float, float]] = 1.0,
        n_fft: int = 2048,
        hop_length: Optional[int] = None,
        win_length: Optional[int] = None,
        window: _WindowSpec = "hann",
        center: bool = True,
        pad_mode: Literal["constant", "reflect", "edge", "symmetric"] = "constant",
    ) -> None:
        """In-place ``harmonic``."""
        ...

    def percussive(
        self,
        *,
        kernel_size: Union[int, tuple[int, int]] = 31,
        power: float = 2.0,
        mask: bool = False,
        margin: Union[float, tuple[float, float]] = 1.0,
        n_fft: int = 2048,
        hop_length: Optional[int] = None,
        win_length: Optional[int] = None,
        window: _WindowSpec = "hann",
        center: bool = True,
        pad_mode: Literal["constant", "reflect", "edge", "symmetric"] = "constant",
    ) -> Self:
        """Percussive component from median-filtering harmonic/percussive
        separation; arguments as ``harmonic``."""
        ...

    def percussive_in_place(
        self,
        *,
        kernel_size: Union[int, tuple[int, int]] = 31,
        power: float = 2.0,
        mask: bool = False,
        margin: Union[float, tuple[float, float]] = 1.0,
        n_fft: int = 2048,
        hop_length: Optional[int] = None,
        win_length: Optional[int] = None,
        window: _WindowSpec = "hann",
        center: bool = True,
        pad_mode: Literal["constant", "reflect", "edge", "symmetric"] = "constant",
    ) -> None:
        """In-place ``percussive``."""
        ...

    def harmonic_percussive(
        self,
        *,
        kernel_size: Union[int, tuple[int, int]] = 31,
        power: float = 2.0,
        mask: bool = False,
        margin: Union[float, tuple[float, float]] = 1.0,
        n_fft: int = 2048,
        hop_length: Optional[int] = None,
        win_length: Optional[int] = None,
        window: _WindowSpec = "hann",
        center: bool = True,
        pad_mode: Literal["constant", "reflect", "edge", "symmetric"] = "constant",
    ) -> tuple[Self, Self]:
        """Median-filtering harmonic/percussive separation: ``(harmonic, percussive)``
        (``hpss`` is a separate method with its own mask blend)."""
        ...

    @overload
    def lpc(self: AudioSamples[_UnknownDtype], *, order: int) -> NDArray[Any]: ...
    @overload
    def lpc(  # type: ignore[overload-overlap]
        self: AudioSamples[_SingleSample],
        *,
        order: int,
    ) -> NDArray[np.float32]: ...
    @overload
    def lpc(self: AudioSamples[_DoubleSample], *, order: int) -> NDArray[np.float64]:
        """Linear prediction coefficients by Burg's method per channel: shape
        ``(channels, order + 1)``."""
        ...

    def iirt(
        self,
        *,
        win_length: int = 2048,
        hop_length: Optional[int] = None,
        center: bool = True,
        tuning: float = 0.0,
        pad_mode: str = "constant",
        flayout: Literal["sos", "ba"] = "sos",
        res_type: str = "soxr_hq",
        **kwargs: Any,
    ) -> NDArray[Any]:
        """The energy of each semitone band per frame: ``(channels, 85, n_frames)``."""
    @staticmethod
    @overload
    def griffinlim_audio(
        S: NDArray[_UnknownDtype],
        *,
        sample_rate: int,
        n_iter: int = 32,
        hop_length: Optional[int] = None,
        win_length: Optional[int] = None,
        n_fft: Optional[int] = None,
        window: _WindowSpec = 'hann',
        center: bool = True,
        length: Optional[int] = None,
        pad_mode: Literal['constant', 'reflect', 'edge', 'symmetric'] = 'constant',
        momentum: float = 0.99,
        init: Optional[Literal['random']] = 'random',
        rng: _RngLike = 0,
        dtype: None = None,
    ) -> AudioSamples[Any]: ...
    @staticmethod
    @overload
    def griffinlim_audio(  # type: ignore[overload-overlap]
        S: NDArray[np.complex64] | NDArray[np.float32],
        *,
        sample_rate: int,
        n_iter: int = 32,
        hop_length: Optional[int] = None,
        win_length: Optional[int] = None,
        n_fft: Optional[int] = None,
        window: _WindowSpec = 'hann',
        center: bool = True,
        length: Optional[int] = None,
        pad_mode: Literal['constant', 'reflect', 'edge', 'symmetric'] = 'constant',
        momentum: float = 0.99,
        init: Optional[Literal['random']] = 'random',
        rng: _RngLike = 0,
        dtype: None = None,
    ) -> AudioSamples[np.float32]: ...
    @staticmethod
    @overload
    def griffinlim_audio(
        S: npt.ArrayLike,
        *,
        sample_rate: int,
        n_iter: int = 32,
        hop_length: Optional[int] = None,
        win_length: Optional[int] = None,
        n_fft: Optional[int] = None,
        window: _WindowSpec = 'hann',
        center: bool = True,
        length: Optional[int] = None,
        pad_mode: Literal['constant', 'reflect', 'edge', 'symmetric'] = 'constant',
        momentum: float = 0.99,
        init: Optional[Literal['random']] = 'random',
        rng: _RngLike = 0,
        dtype: None = None,
    ) -> AudioSamples[np.float64]: ...
    @staticmethod
    @overload
    def griffinlim_audio(
        S: npt.ArrayLike,
        *,
        sample_rate: int,
        n_iter: int = 32,
        hop_length: Optional[int] = None,
        win_length: Optional[int] = None,
        n_fft: Optional[int] = None,
        window: _WindowSpec = 'hann',
        center: bool = True,
        length: Optional[int] = None,
        pad_mode: Literal['constant', 'reflect', 'edge', 'symmetric'] = 'constant',
        momentum: float = 0.99,
        init: Optional[Literal['random']] = 'random',
        rng: _RngLike = 0,
        dtype: SampleType,
    ) -> AudioSamples[Any]:
        """Griffin-Lim phase retrieval returning ``AudioSamples`` (see
        ``audio_samples.effects.griffinlim``; ``rng`` defaults to the seed 0).

        Args:
        dtype: Sample type of the returned audio. None (default) gives float32
            audio for a float32 input and float64 audio for any other input.
            The computation runs in the float type of ``dtype`` (float32 for
            U8, I16, I24 and F32; float64 for I32 and F64); integer output is
            rounded and saturated.

        Returns:
            AudioSamples: audio of sample type ``dtype``.
        """
        ...

    def to_format(self, dtype: SampleType) -> AudioSamples[Any]:
        """Audio-aware conversion to ``dtype``.

        Values are rescaled between full-scale ranges. Integer scaling is
        asymmetric: float -1.0 -> -32768 and +1.0 -> 32767 for int16
        (likewise int24/int32); uint8 is offset binary (-1 -> 0, 0 -> 128,
        1 -> 255). Compare ``cast_as`` (raw numeric cast).
        """
        ...

    def as_u8(self) -> AudioSamples[np.uint8]:
        """Audio-aware conversion to unsigned 8-bit integer; ``to_format(SampleType.U8)``."""
        ...

    def as_i16(self) -> AudioSamples[np.int16]:
        """Audio-aware conversion to signed 16-bit integer; ``to_format(SampleType.I16)``."""
        ...

    def as_i32(self) -> AudioSamples[np.int32]:
        """Audio-aware conversion to signed 32-bit integer; ``to_format(SampleType.I32)``."""
        ...

    def as_f32(self) -> AudioSamples[np.float32]:
        """Audio-aware conversion to 32-bit floating point; ``to_format(SampleType.F32)``."""
        ...

    def as_f64(self) -> AudioSamples[np.float64]:
        """Audio-aware conversion to 64-bit floating point; ``to_format(SampleType.F64)``."""
        ...

    def cast_as(self, dtype: SampleType) -> AudioSamples[Any]:
        """Raw numeric cast to ``dtype`` (no rescaling).

        Float-to-integer truncates toward zero and saturates, so float audio
        in [-1, 1] becomes mostly -1/0/1; int16 16384 becomes 16384.0, not
        0.5. Use ``to_format``/``as_*`` for audio-aware conversion.
        """
        ...

    def cast_as_u8(self) -> AudioSamples[np.uint8]:
        """Cast audio samples to unsigned 8-bit integer format."""
        ...

    def cast_as_i16(self) -> AudioSamples[np.int16]:
        """Cast audio samples to signed 16-bit integer format."""
        ...

    def cast_as_i32(self) -> AudioSamples[np.int32]:
        """Cast audio samples to signed 32-bit integer format."""
        ...

    def cast_as_f32(self) -> AudioSamples[np.float32]:
        """Cast audio samples to 32-bit floating point format."""
        ...

    def cast_as_f64(self) -> AudioSamples[np.float64]:
        """Cast audio samples to 64-bit floating point format."""
        ...

    # Channel manipulation
    def to_mono(
        self, method: MonoConversionMethod = MonoConversionMethod.average
    ) -> Self:
        """Convert to mono using specified method.

        Args:
            method: Conversion method describing how to combine channels

        Returns:
            New buffer with a single channel
        """
        ...

    def to_stereo(
        self, method: StereoConversionMethod = StereoConversionMethod.duplicate
    ) -> Self:
        """Convert to stereo using specified method.

        Args:
            method: Conversion method describing how to expand channels

        Returns:
            New buffer with two channels
        """
        ...

    def duplicate_to_channels(self, n_channels: int) -> Self:
        """Duplicate the buffer into the requested number of channels.

        Args:
            n_channels: Target channel count

        Returns:
            New buffer with the duplicated channels
        """
        ...

    def extract_channel(self, channel_index: int) -> Self:
        """Extract a single channel."""
        ...

    def borrow_channel(self, channel_index: int) -> Self:
        """Return a single channel as a new mono buffer.

        Copies the selected channel into an owned buffer before returning it.

        Args:
            channel_index: Zero-based index of the channel to borrow. Ignored
                for mono audio.

        Returns:
            AudioSamples: An owned mono buffer containing the selected channel.

        Raises:
            ValueError: If the channel index is out of range.
        """
        ...

    @staticmethod
    def interleave_channels(channels: list[AudioSamples[_S]]) -> AudioSamples[_S]:
        """Combine multiple mono buffers into a single multi-channel buffer.

        The first buffer becomes channel 0, the second channel 1, and so on. All
        inputs must share the same dtype and number of samples; the output sample
        rate is taken from the first input.

        Args:
            channels: Non-empty list of mono buffers, all of the same dtype and
                length.

        Returns:
            AudioSamples: A multi-channel buffer with one channel per input.

        Raises:
            ValueError: If the list is empty, the dtypes differ, or the inputs do
                not all have the same sample count.
        """
        ...

    def deinterleave_channels(self) -> list[Self]:
        """Split a multi-channel buffer into individual mono buffers.

        Channel 0 becomes element 0, channel 1 becomes element 1, and so on. For
        mono input a single-element list is returned. This is the inverse of
        :meth:`interleave_channels`.

        Returns:
            list[AudioSamples]: One owned mono buffer per input channel.

        Raises:
            AudioError: If channel separation fails.
        """
        ...

    def swap_channels(self, channel1: int, channel2: int) -> None:
        """Swap two channels (in-place)."""
        ...

    # Metadata and properties
    @property
    def dtype(self) -> str:
        """Sample type as a short string: ``'u8'``, ``'i16'``, ``'I24'``,
        ``'i32'``, ``'f32'`` or ``'f64'`` (use ``to_numpy().dtype`` for a
        ``numpy.dtype``)."""
        ...
    @property
    def sample_rate(self) -> int:
        """Get the sample rate in Hz."""
        ...

    @property
    def num_channels(self) -> int:
        """Get the number of audio channels."""
        ...

    def copy(self) -> Self:
        """Create a deep copy of the AudioSamples."""
        ...

    def __class_getitem__(cls, item: Any) -> GenericAlias:
        """``AudioSamples[numpy.float32]`` names audio of that sample type in annotations."""
        ...

    def __len__(self) -> int:
        """Get the length of the audio (total number of samples)."""
        ...

    def __getitem__(self, index: Any, /) -> Self:
        """Get a slice of the audio samples."""
        ...

    def samples_per_channel(self) -> int:
        """Get the number of samples per channel."""
        ...

    @property
    def total_samples(self) -> int:
        """Get the total number of samples across all channels."""
        ...

    @property
    def shape(self) -> list[int]:
        """Shape of the audio data as a list: ``[samples]`` for mono,
        ``[channels, samples]`` for multi-channel."""
        ...

    @property
    def channels(self) -> int:
        """Alias of ``num_channels``."""
        ...

    @property
    def len(self) -> int:
        """Total number of samples across all channels (same as ``len(audio)``)."""
        ...

    @property
    def size(self) -> int:
        """Total number of samples across all channels (alias of ``len``)."""
        ...

    @property
    def ndim(self) -> int:
        """Number of array dimensions: 1 for mono, 2 for multi-channel."""
        ...

    @property
    def duration_milliseconds(self) -> float:
        """Duration in milliseconds."""
        ...

    def nyquist(self) -> float:
        """Nyquist frequency in Hz (``sample_rate / 2``)."""
        ...

    def info(self) -> _AudioSamplesInfo:
        """Summary of sample rate, channels, frames, duration and layout."""
        ...

    @staticmethod
    def from_array(arr: NDArray[_S], sample_rate: int) -> AudioSamples[_S]:
        """Create an AudioSamples from a 1-D (mono) or 2-D ``(channels, samples)``
        NumPy array without copying it."""
        ...

    def add_numpy_array(self, numpy_array: NDArray[Any]) -> Self:
        """Element-wise ``self + numpy_array`` (helper behind ``__add__``)."""
        ...

    def sub_numpy_array(self, numpy_array: NDArray[Any]) -> Self:
        """Element-wise ``self - numpy_array`` (helper behind ``__sub__``)."""
        ...

    def mul_numpy_array(self, numpy_array: NDArray[Any]) -> Self:
        """Element-wise ``self * numpy_array`` (helper behind ``__mul__``)."""
        ...

    def div_numpy_array(self, numpy_array: NDArray[Any]) -> Self:
        """Element-wise ``self / numpy_array`` (helper behind ``__truediv__``)."""
        ...

    def from_numpy_array_with_metadata(self, numpy_array: NDArray[_S]) -> AudioSamples[_S]:
        """Wrap ``numpy_array`` (same shape as this buffer) with this buffer's
        sample rate."""
        ...

    def time_axis(self) -> NDArray[np.float64]:
        """Time of each sample in seconds: ``arange(samples_per_channel()) / sample_rate``."""
        ...

    def _repr_html_(self) -> Optional[str]:
        """Jupyter HTML view: summary table and an inline audio player
        (mono/stereo clips up to 30 s)."""
        ...

    @overload
    def __array__(self, dtype: None=None, copy: Optional[bool]=None) -> NDArray[_S_co]: ...
    @overload
    def __array__(self, dtype: DTypeLike, copy: Optional[bool]=None) -> NDArray[Any]:
        """NumPy array protocol (``np.asarray(audio)``)."""
        ...

    def __array_ufunc__(
        self, ufunc: Any, method: str, *inputs: Any, **kwargs: Any
    ) -> Any:
        """NumPy ufunc protocol: ``np.sin(audio)`` equals ``np.sin(audio.to_numpy())``.
        ``add``/``subtract``/``multiply``/``divide`` results with this buffer's
        shape are returned as AudioSamples."""
        ...

    def __array_function__(
        self, func: Any, types: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> Any:
        """NumPy function protocol: ``np.mean(audio)`` equals ``np.mean(audio.to_numpy())``."""
        ...

    def is_mono(self) -> bool:
        """Check if audio is mono (single channel)."""
        ...

    def is_multi_channel(self) -> bool:
        """Check if audio is multi-channel."""
        ...

    @property
    def duration_seconds(self) -> float:
        """Get the duration in seconds."""
        ...

    def to_numpy(self) -> NDArray[_S_co]:
        """Convert AudioSamples to a numpy ndarray."""

    def to_tensor(self, device: Optional[str]) -> Any:
        """Convert AudioSamples to a PyTorch tensor."""

    def to_gpu_tensor(self, gpu_id: Optional[str] = None) -> Any:
        """Convert AudioSamples to a PyTorch tensor on the GPU if possible.

        If passing a ``gpu_id`` argument provide the full device name and id. e.g. cuda:1
        """

    # Arithmetic operators
    def __add__(self, other: AudioSamples | NDArray[Any] | _Scalar, /) -> Self:
        """Add AudioSamples with another AudioSamples, numpy array, or scalar.

        The result has this audio's sample type. Integer results saturate at the type's limits.
        On integer audio a float scalar is in the floating-point range, where 1.0 is full scale, and
        must be within -1.0 to 1.0, or ``ValueError`` is raised. An int or NumPy integer scalar is a
        number of sample steps relative to silence, so ``audio + 1`` on int16 adds one step, and
        ``OverflowError`` is raised if it does not fit the sample type: int16 takes -32768 to 32767,
        and uint8 takes -128 to 127. A NumPy uint8 scalar on uint8 audio is a sample instead, so
        adding ``np.uint8(128)`` changes nothing. ``*`` and ``/`` are gains, so ``audio * 2`` equals
        ``audio * 2.0``, and the result saturates. A scalar must be finite on integer audio, or
        ``ValueError`` is raised, and dividing integer audio by a zero scalar raises
        ``ZeroDivisionError``. Dividing by an array or other audio works sample by sample: a divisor
        sample of silence gives the limit with the sign of the dividend, and silence over silence
        gives silence, so ``s / s`` is defined. On float audio an int is the same number as
        the float and the usual float rules apply. A NumPy array holds raw sample values, and audio
        of another sample type is converted to this one first.
        A bool scalar raises ``TypeError``.

        Raises:
            AudioParameterError: (a ``ValueError``) if two AudioSamples have
                different sample rates.
            ValueError: If the two AudioSamples have different lengths.
        """
        ...

    def __sub__(self, other: AudioSamples | NDArray[Any] | _Scalar, /) -> Self:
        """Subtract AudioSamples, numpy array, or scalar from AudioSamples."""
        ...

    def __mul__(self, other: AudioSamples | NDArray[Any] | _Scalar, /) -> Self:
        """Multiply AudioSamples by a numpy array or scalar."""
        ...

    def __truediv__(self, other: AudioSamples | NDArray[Any] | _Scalar, /) -> Self:
        """Divide AudioSamples element-wise by another AudioSamples (same sample
        rate and length), a numpy array, or a scalar."""
        ...

    def __pow__(self, exponent: float, modulo: Optional[float] = None, /) -> Self:
        """Raise AudioSamples to a power (float types only)."""
        ...

    # Comparison operators
    def __eq__(self, other: object, /) -> bool:
        """``audio == other``: ``True`` when ``other`` is an AudioSamples with the same sample type,
        sample rate, channel count and length, and every sample is equal. NaN samples compare
        unequal, as they do for floats. Any other object gives ``False``."""
        ...

    def __ne__(self, other: object, /) -> bool:
        """``audio != other``: the negation of ``==``."""
        ...

    # Unary operators
    def __neg__(self) -> Self:
        """``-audio`` in this sample type. Integer results saturate: ``-(-32768)`` is 32767 on int16.
        uint8 mirrors about silence (128), so a stored 0 gives 255."""
        ...

    def __pos__(self) -> Self:
        """``+audio``: a copy of this audio."""
        ...

    def __abs__(self) -> Self:
        """``abs(audio)`` in this sample type. Integer results saturate: ``abs(-32768)`` is 32767 on
        int16. uint8 is measured from silence (128), so a stored 100 gives 156."""
        ...

    # Reverse arithmetic operators (for scalar on left side)
    def __radd__(self, scalar: _Scalar, /) -> Self:
        """Add scalar to AudioSamples (scalar + audio)."""
        ...

    def __rmul__(self, scalar: _Scalar, /) -> Self:
        """Multiply scalar with AudioSamples (scalar * audio)."""
        ...

    def __rsub__(self, scalar: _Scalar, /) -> Self:
        """Subtract AudioSamples from scalar (scalar - audio)."""
        ...

    def __rtruediv__(self, scalar: float, /) -> NoReturn:
        """Divide scalar by AudioSamples (scalar / audio) - not implemented."""
        ...

    # In-place arithmetic operators
    def __iadd__(self, other: AudioSamples | NDArray[Any] | _Scalar, /) -> Self:
        """In-place addition with AudioSamples, numpy array, or scalar."""
        ...

    def __isub__(self, other: AudioSamples | NDArray[Any] | _Scalar, /) -> Self:
        """In-place subtraction with AudioSamples, numpy array, or scalar."""
        ...

    def __imul__(self, other: AudioSamples | NDArray[Any] | _Scalar, /) -> Self:
        """In-place multiplication with AudioSamples, numpy array, or scalar."""
        ...

    def __itruediv__(self, other: AudioSamples | NDArray[Any] | _Scalar, /) -> Self:
        """In-place division by AudioSamples, numpy array, or scalar."""
        ...

    def __str__(self) -> str:
        """String representation of the audio samples."""
        ...

    @property
    def __array_interface__(self) -> dict[str, Any]:
        """NumPy array interface of the samples, version 3."""
        ...















class _AudioSamplesInfo:
    """Summary returned by ``AudioSamples.info()``."""

    sample_rate: int
    channels: int
    frames: int
    duration_seconds: float
    layout: str

__all__ = [
    "AudioSamples",
    "IirFilterDesign",
    "EqBand",
    "ParametricEq",
    "generation",
    "io",
    "utils",
    "spectral",
    "sequence",
    "features",
    "rhythm",
    "effects",
    "forms",
    "linalg",
    "inverse",
    "factorization",
    "pitch_filterbank",
    "segment",
    "arrays",
    "notation",
    "spectrograms",
    "mpl",
    "spectrograms",
    "exceptions",
    "explain",
    "ExplainDoc",
    "StftParams",
    "SpectrogramParams",
    "MelParams",
    "MfccParams",
    "LogParams",
    "ChromaParams",
    "CqtParams",
    "ErbParams",
    "MelNorm",
    "StftResult",
    "WindowType",
    "AudioError",
    "AudioParameterError",
    "AudioLayoutError",
    "AudioConversionError",
    "AudioRuntimeError",
    "AudioProcessingError",
    "AudioFeatureError",
    "AudioSpectrogramError",
    "AudioIOError",
    "AudioFileNotFoundError",
    "AudioFormatError",
    "AudioDecodeError",
    "mu_law_encode",
    "mu_law_decode",
    "sine_wave",
    "cosine_wave",
    "sawtooth_wave",
    "square_wave",
    "triangle_wave",
    "white_noise",
    "pink_noise",
    "brown_noise",
    "impulse",
    "silence",
    "chirp",
    "stereo_sine_wave",
    "stereo_chirp",
    "stereo_silence",
]
