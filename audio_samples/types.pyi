from __future__ import annotations
from enum import Enum
from typing import Optional
import numpy
from spectrograms import CqtParams

class SampleType(Enum):
    """
    Enumeration of supported audio sample data types.

    `SampleType` describes the numeric representation used to store individual
    audio samples. It is typically used when constructing or converting audio
    buffers, configuring I/O, or selecting internal processing formats.

    The available sample types are:

    - ``SampleType.I16``:
      Signed 16-bit integer samples. Common in PCM WAV files and efficient for
      storage and I/O, but with limited dynamic range.

    - ``SampleType.I24``:
      Signed 24-bit integer samples. Higher dynamic range than 16-bit PCM and
      widely used in professional audio pipelines.

    - ``SampleType.I32``:
      Signed 32-bit integer samples. Rare in interchange formats, but sometimes
      used for intermediate or high-precision processing.

    - ``SampleType.F32`` (default):
      32-bit floating-point samples. The most common format for DSP and machine
      learning workloads due to good numerical stability and performance.

    - ``SampleType.F64``:
      64-bit floating-point samples. Provides maximum numerical precision at the
      cost of higher memory usage and lower throughput.

    Instances of ``SampleType`` are immutable and comparable. They should be
    treated as enum values rather than constructed directly.
    """

    I16: SampleType
    I24: SampleType
    I32: SampleType
    F32: SampleType
    F64: SampleType

    ...

class PadSide:
    """
    Side for padding operations.

    Indicates which side of a signal should be padded when applying
    padding operations.
    """

    @classmethod
    def left(cls) -> PadSide: ...
    @classmethod
    def right(cls) -> PadSide: ...

class NormalizationMethod:
    """
    Normalisation strategy for audio sample data.

    `NormalizationMethod` represents a predefined method for rescaling or
    re-centring audio samples prior to further processing. Instances are
    immutable and should be treated as enum-like values.

    Normalisation methods are accessed via class attributes rather than being
    constructed directly.
    """

    minmax: NormalizationMethod
    """Min-max normalisation.

    Linearly rescales samples so that the minimum and maximum values are mapped
    to a fixed range. Sensitive to outliers.
    """

    zscore: NormalizationMethod
    """Z-score normalisation.

    Subtracts the mean and divides by the standard deviation, producing
    zero-mean, unit-variance data.
    """

    peak: NormalizationMethod
    """Peak normalisation.

    Scales samples so that the maximum absolute amplitude equals 1.
    """

    mean: NormalizationMethod
    """Mean normalisation.

    Recentres samples by subtracting the mean value.
    """

    median: NormalizationMethod
    """Median normalisation.

    Recentres samples by subtracting the median value, providing robustness
    against outliers.
    """

    ...

class FadeCurve:
    """
    Fade curve shape for envelope operations.

    `FadeCurve` represents the shape of an amplitude envelope used when applying
    fades, ramps, or transitions to audio signals. Instances are immutable and
    should be treated as enum-like values and accessed via class attributes.
    """

    linear: FadeCurve
    """Linear fade.

    Constant rate of change over time.
    """

    exponential: FadeCurve
    """Exponential fade.

    Faster change at the beginning and slower towards the end.
    """

    logarithmic: FadeCurve
    """Logarithmic fade.

    Slower change at the beginning and faster towards the end.
    """

    smooth_step: FadeCurve
    """Smooth-step fade.

    S-shaped curve with smooth transitions at both the start and end.
    """

    ...

class MonoConversionMethod:
    """
    Method for converting multi-channel audio to mono.

    `MonoConversionMethod` represents a strategy for collapsing multi-channel
    audio (e.g. stereo or surround) into a single mono channel. Instances are
    immutable and should be treated as enum-like values.

    Zero-parameter methods are accessed via class attributes, while
    parameterised methods are constructed via class methods.
    """

    average: MonoConversionMethod
    """Average all channels equally."""

    left: MonoConversionMethod
    """Use the left channel only."""

    right: MonoConversionMethod
    """Use the right channel only."""

    center: MonoConversionMethod
    """Use the centre channel if available, otherwise average left and right."""

    @classmethod
    def weighted(cls, weights: list[float]) -> MonoConversionMethod:
        """
        Weighted average across channels.

        Parameters
        ----------
        weights : list[float]
            Per-channel weights. The length should match the number of input
            channels.
        """
        ...

    ...

class StereoConversionMethod:
    """
    Method for converting mono audio to stereo.

    `StereoConversionMethod` represents a strategy for expanding mono audio
    into stereo. Instances are immutable and should be treated as enum-like
    values.

    Zero-parameter methods are accessed via class attributes, while
    parameterised methods are constructed via class methods.
    """

    duplicate: StereoConversionMethod
    """Duplicate mono signal to both left and right channels."""

    left: StereoConversionMethod
    """Use as the left channel, filling the right channel with silence."""

    right: StereoConversionMethod
    """Use as the right channel, filling the left channel with silence."""

    @classmethod
    def pan(cls, pan_value: float) -> StereoConversionMethod:
        """
        Pan the mono signal between left and right channels.

        A value of -1 places the signal fully in the left channel, 0 centres the
        signal, and 1 places it fully in the right channel.

        Parameters
        ----------
        pan_value : float
            Pan position in the range [-1, 1]
        """
        ...

    ...

class VadMethod:
    """
    Voice Activity Detection (VAD) method.

    `VadMethod` represents the algorithmic strategy used to detect regions of
    speech or activity within an audio signal. Instances are immutable and
    should be treated as enum-like values and accessed via class attributes.
    """

    energy: VadMethod
    """Energy-based voice activity detection.

    Uses signal energy (e.g. RMS) and a threshold to detect activity.
    """

    zcr: VadMethod
    """Zero crossing rate (ZCR) based detection.

    Uses the rate of sign changes in the waveform as an indicator of activity.
    """

    combined: VadMethod
    """Combined energy and zero crossing rate detection."""

    spectral: VadMethod
    """Spectral-based detection.

    Uses spectral features for improved robustness at higher computational cost.
    """

    ...

class VadChannelPolicy:
    """
    Multi-channel handling policy for Voice Activity Detection (VAD).

    `VadChannelPolicy` defines how voice activity decisions are produced when the
    input audio contains multiple channels. Instances are immutable and should
    be treated as enum-like values.

    Zero-parameter policies are accessed via class attributes, while
    parameterised policies are constructed via class methods.
    """

    average_to_mono: VadChannelPolicy
    """Average all channels to a mono signal and run VAD once."""

    any_channel: VadChannelPolicy
    """Run VAD independently on each channel and mark activity if any channel is active."""

    all_channels: VadChannelPolicy
    """Run VAD independently on each channel and mark activity only if all channels are active."""

    @classmethod
    def channel(cls, ch: int) -> VadChannelPolicy:
        """
        Run VAD on a specific channel index.

        Parameters
        ----------
        ch : int
            Zero-based channel index to use for VAD.
        """
        ...

    ...

class VadConfig:
    """
    Configuration for Voice Activity Detection (VAD).

    `VadConfig` defines all parameters controlling frame-based voice activity
    detection. The VAD operates on overlapping frames of length ``frame_size``
    with step ``hop_size`` and produces a boolean decision per frame.

    Instances are immutable once constructed. Use ``validate()`` to check that a
    configuration is internally consistent.
    """

    def __init__(
        self,
        method: VadMethod,
        channel_policy: VadChannelPolicy,
        /,
        frame_size: int,
        hop_size: int,
        pad_end: bool,
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
        Create a new VAD configuration.

        Parameters
        ----------
        method : VadMethod
            Voice activity detection method to use.
        channel_policy : VadChannelPolicy
            Policy for handling multi-channel audio.
        frame_size : int
            Frame size in samples.
        hop_size : int
            Hop size in samples. Must be less than or equal to ``frame_size``.
        pad_end : bool
            Whether to include a final partial frame padded with zeros.
        energy_threshold_db : float
            Energy threshold in dBFS (RMS).
        zcr_min : float
            Minimum acceptable zero crossing rate in the range [0, 1].
        zcr_max : float
            Maximum acceptable zero crossing rate in the range [0, 1].
        min_speech_frames : int
            Minimum number of consecutive speech frames to keep a speech region.
        min_silence_frames : int
            Minimum number of consecutive non-speech frames to keep a silence region.
        hangover_frames : int
            Number of frames to keep speech active after energy drops.
        smooth_frames : int
            Majority-vote smoothing window in frames.
        speech_band_low_hz : float
            Lower bound of the speech band in Hz.
        speech_band_high_hz : float
            Upper bound of the speech band in Hz.
        spectral_ratio_threshold : float
            Threshold on speech-band energy ratio.
        """
        ...

    energy_only: VadConfig
    """Convenience preset using energy-based detection with default parameters."""

    def validate(self) -> None:
        """
        Validate configuration parameters.

        Raises an exception if any parameter is invalid or internally inconsistent.
        """
        ...

    ...

class ResamplingQuality:
    """
    Quality level for resampling operations.

    `ResamplingQuality` represents a discrete trade-off between computational
    cost, latency, and signal fidelity when resampling audio. Instances are
    immutable and should be treated as enum-like values and accessed via class
    attributes.
    """

    fast: ResamplingQuality
    """Fast resampling.

    Prioritises throughput and low latency at the cost of reduced spectral
    fidelity and increased aliasing.
    """

    medium: ResamplingQuality
    """Medium quality resampling.

    Balanced trade-off between computational cost and signal quality.
    """

    high: ResamplingQuality
    """High quality resampling.

    Prioritises spectral fidelity and phase stability at the cost of higher
    computational cost and latency.
    """

    ...

class SpectrogramScale:
    """
    Scaling method for spectrogram magnitude and frequency representations.

    `SpectrogramScale` represents how spectral magnitudes or frequencies are
    mapped for analysis or visualisation. Instances are immutable and should be
    treated as enum-like values and accessed via class attributes.
    """

    linear: SpectrogramScale
    """Linear power scale.

    Preserves absolute magnitude relationships and is most appropriate for
    quantitative analysis and energy measurements.
    """

    log: SpectrogramScale
    """Logarithmic (decibel) magnitude scale.

    Compresses dynamic range to improve visualisation of low-energy components
    alongside strong spectral peaks.
    """

    mel: SpectrogramScale
    """Mel-frequency scale.

    Applies a perceptually motivated nonlinear mapping of frequency designed to
    better approximate human auditory resolution.
    """

    ...

class PitchDetectionMethod:
    """
    Pitch detection algorithm selection.

    `PitchDetectionMethod` selects the algorithm used to estimate the fundamental
    frequency of a signal. Instances are immutable and should be treated as
    enum-like values and accessed via class attributes.
    """

    yin: PitchDetectionMethod
    """YIN pitch detection algorithm.

    Provides robust and accurate fundamental frequency estimation for both
    speech and musical signals at moderate computational cost.
    """

    autocorrelation: PitchDetectionMethod
    """Autocorrelation-based pitch detection.

    Simple and fast, but sensitive to noise and octave errors for complex or
    weakly periodic signals.
    """

    cepstrum: PitchDetectionMethod
    """Cepstral pitch detection.

    Operates in the frequency domain and performs well for voiced speech, but can
    degrade for dense harmonic or noisy spectra.
    """

    harmonic_product: PitchDetectionMethod
    """Harmonic Product Spectrum (HPS).

    Emphasises harmonic structure and is well-suited to musical signals with
    strong harmonic content.
    """

    ...

class IirFilterType:
    """
    IIR filter family selection for digital signal processing.

    `IirFilterType` selects the analogue prototype used when designing an
    infinite impulse response (IIR) digital filter. Instances are immutable and
    should be treated as enum-like values and accessed via class attributes.
    """

    butterworth: IirFilterType
    """Butterworth filter.

    Maximally flat passband with monotonic magnitude response and no ripple.
    """

    chebyshev_type_i: IirFilterType
    """Chebyshev Type I filter.

    Introduces controlled ripple in the passband to achieve a sharper transition
    region than Butterworth designs.
    """

    chebyshev_type_ii: IirFilterType
    """Chebyshev Type II filter.

    Introduces ripple in the stopband while preserving a monotonic passband
    response.
    """

    elliptic: IirFilterType
    """Elliptic (Cauer) filter.

    Introduces ripple in both passband and stopband, yielding very sharp
    transition regions.
    """

    bessel: IirFilterType
    """Bessel (Bessel--Thomson) filter.

    Maximally flat group delay, preserving waveform shape at the cost of a
    gentler magnitude roll-off than other prototypes.
    """

    ...

class FilterResponse:
    """
    Filter response characteristic.

    `FilterResponse` defines the qualitative frequency response shape of a
    filter, such as whether it attenuates low frequencies, high frequencies, or
    a band of frequencies. Instances are immutable and should be treated as
    enum-like values and accessed via class attributes.
    """

    lowpass: FilterResponse
    """Low-pass filter response.

    Attenuates frequencies above the cutoff frequency while preserving
    lower-frequency components.
    """

    highpass: FilterResponse
    """High-pass filter response.

    Attenuates frequencies below the cutoff frequency while preserving
    higher-frequency components.
    """

    bandpass: FilterResponse
    """Band-pass filter response.

    Preserves frequencies within a specified band while attenuating frequencies
    outside that range.
    """

    bandstop: FilterResponse
    """Band-stop (notch) filter response.

    Attenuates frequencies within a specified band while preserving frequencies
    outside that range.
    """

    ...

class IirFilterDesign:
    """
    IIR filter design parameters.

    `IirFilterDesign` describes the parameters required to construct a digital
    infinite impulse response (IIR) filter, including the filter family, response
    type, order, and frequency specifications.

    Instances are immutable value objects. Fields are exposed via read-only
    properties.
    """

    def __init__(
        self,
        filter_type: IirFilterType,
        response: FilterResponse,
        order: int,
        cutoff_frequency: Optional[float] = None,
        low_frequency: Optional[float] = None,
        high_frequency: Optional[float] = None,
    ) -> None:
        """
        Create a new IIR filter design.

        Parameters
        ----------
        filter_type : IirFilterType
            IIR filter family (e.g. Butterworth, Chebyshev).
        response : FilterResponse
            Desired frequency response shape.
        order : int
            Filter order (number of poles).
        cutoff_frequency : float, optional
            Cutoff frequency in Hz for low-pass and high-pass filters.
        low_frequency : float, optional
            Lower cutoff frequency in Hz for band-pass and band-stop filters.
        high_frequency : float, optional
            Upper cutoff frequency in Hz for band-pass and band-stop filters.
        """
        ...

    @property
    def filter_type(self) -> IirFilterType: ...
    """Filter family used for the design."""

    @property
    def response(self) -> FilterResponse: ...
    """Frequency response shape of the filter."""

    @property
    def order(self) -> int: ...
    """Filter order (number of poles)."""

    @property
    def cutoff_frequency(self) -> Optional[float]: ...
    """Cutoff frequency in Hz for low-pass and high-pass filters."""

    @property
    def low_frequency(self) -> Optional[float]: ...
    """Lower cutoff frequency in Hz for band-pass and band-stop filters."""

    @property
    def high_frequency(self) -> Optional[float]: ...
    """Upper cutoff frequency in Hz for band-pass and band-stop filters."""

    @property
    def passband_ripple(self) -> Optional[float]: ...
    """Passband ripple in dB, if applicable."""

    @property
    def stopband_attenuation(self) -> Optional[float]: ...
    """Stopband attenuation in dB, if applicable."""

    @classmethod
    def butterworth_lowpass(
        cls, order: int, cutoff_frequency: float
    ) -> IirFilterDesign:
        """
        Create a Butterworth low-pass filter design.
        """
        ...

    @classmethod
    def butterworth_highpass(
        cls, order: int, cutoff_frequency: float
    ) -> IirFilterDesign:
        """
        Create a Butterworth high-pass filter design.
        """
        ...

    @classmethod
    def butterworth_bandpass(
        cls,
        order: int,
        low_frequency: float,
        high_frequency: float,
    ) -> IirFilterDesign:
        """
        Create a Butterworth band-pass filter design.
        """
        ...

    ...

class EqBandType:
    """
    Parametric equaliser band type.

    `EqBandType` defines how gain is applied across the frequency spectrum
    relative to a centre or cutoff frequency. Instances are immutable and should
    be treated as enum-like values and accessed via class attributes.
    """

    peak: EqBandType
    """Peaking (bell) filter.

    Boosts or attenuates a narrow frequency region centred at the target
    frequency.
    """

    low_shelf: EqBandType
    """Low-shelf filter.

    Applies a broadband boost or cut to frequencies below the corner frequency.
    """

    high_shelf: EqBandType
    """High-shelf filter.

    Applies a broadband boost or cut to frequencies above the corner frequency.
    """

    low_pass: EqBandType
    """Low-pass filter.

    Attenuates frequencies above the cutoff frequency.
    """

    high_pass: EqBandType
    """High-pass filter.

    Attenuates frequencies below the cutoff frequency.
    """

    band_pass: EqBandType
    """Band-pass filter.

    Preserves frequencies within a specified band while attenuating frequencies
    outside that range.
    """

    band_stop: EqBandType
    """Band-stop (notch) filter.

    Attenuates frequencies within a specified band while preserving frequencies
    outside that range.
    """

    ...

class EqBand:
    """
    Parametric equaliser band configuration.

    `EqBand` represents a single band in a parametric equaliser, defined by a band
    type, centre or cutoff frequency, gain, and quality factor (Q).

    Instances are immutable value objects. Parameters are provided at
    construction time and exposed via read-only properties.
    """

    def __init__(
        self,
        band_type: EqBandType,
        frequency: float,
        gain_db: float,
        q_factor: float,
    ) -> None:
        """
        Create a new EQ band with explicit parameters.

        Parameters
        ----------
        band_type : EqBandType
            Type of EQ band (peak, shelf, pass, etc.).
        frequency : float
            Centre or corner frequency in Hz.
        gain_db : float
            Gain in decibels. Positive values boost, negative values attenuate.
        q_factor : float
            Quality factor controlling bandwidth or slope.
        """
        ...

    @classmethod
    def peak(cls, frequency: float, gain_db: float, q_factor: float) -> EqBand:
        """
        Create a peaking (bell) EQ band.
        """
        ...

    @classmethod
    def low_shelf(cls, frequency: float, gain_db: float, q_factor: float) -> EqBand:
        """
        Create a low-shelf EQ band.
        """
        ...

    @classmethod
    def high_shelf(cls, frequency: float, gain_db: float, q_factor: float) -> EqBand:
        """
        Create a high-shelf EQ band.
        """
        ...

    @property
    def frequency(self) -> float: ...
    """Centre or cutoff frequency in Hz."""

    @property
    def gain_db(self) -> float: ...
    """Gain in decibels."""

    @property
    def q_factor(self) -> float: ...
    """Quality factor controlling bandwidth or slope."""

    @property
    def enabled(self) -> bool: ...
    """Whether this band is enabled."""

    ...

class ParametricEq:
    """
    Parametric equaliser configuration.

    `ParametricEq` represents a complete parametric equaliser composed of an
    ordered collection of EQ bands. Instances are mutable and support adding and
    removing bands dynamically.

    The number of bands can be queried using ``len(eq)``.
    """

    def __init__(self) -> None:
        """Create a new empty parametric equaliser."""
        ...

    def add_band(self, band: EqBand) -> None:
        """
        Add an EQ band to the equaliser.

        Parameters
        ----------
        band : EqBand
            Band configuration to append to the equaliser.
        """
        ...

    def remove_band(self, index: int) -> Optional[EqBand]:
        """
        Remove an EQ band by index.

        Parameters
        ----------
        index : int
            Zero-based index of the band to remove.

        Returns
        -------
        EqBand or None
            The removed band if the index was valid, otherwise ``None``.
        """
        ...

    @property
    def output_gain_db(self) -> float: ...
    """Overall output gain in decibels."""

    @property
    def bypassed(self) -> bool: ...
    """Whether the equaliser is currently bypassed."""

    def __len__(self) -> int: ...
    """Return the number of EQ bands."""

    ...

class KneeType:
    """
    Knee characteristic for dynamic range processing.

    `KneeType` controls how smoothly gain reduction transitions as the signal
    crosses the threshold in dynamics processors such as compressors and
    limiters. Instances are immutable and should be treated as enum-like values
    and accessed via class attributes.
    """

    hard: KneeType
    """Hard knee.

    Applies an abrupt transition at the threshold, yielding precise dynamics
    control at the potential cost of audible artefacts.
    """

    soft: KneeType
    """Soft knee.

    Applies a gradual transition around the threshold, producing smoother and
    more perceptually natural behaviour.
    """

    ...

class DynamicRangeMethod:
    """
    Detection method for dynamic range processing.

    `DynamicRangeMethod` selects how signal level is estimated when driving gain
    reduction in dynamics processors such as compressors, limiters, and gates.
    Instances are immutable and should be treated as enum-like values and
    accessed via class attributes.
    """

    rms: DynamicRangeMethod
    """RMS-based level detection.

    Estimates average signal power over time, producing smoother and more
    perceptually stable gain control.
    """

    peak: DynamicRangeMethod
    """Peak-based level detection.

    Responds to instantaneous signal peaks, providing tight peak control with
    increased sensitivity to transients.
    """

    hybrid: DynamicRangeMethod
    """Hybrid level detection.

    Combines RMS and peak estimation to balance smoothness and transient control.
    """

    ...

class EnvelopeFollower:
    """
    Envelope follower for attack/release envelope tracking.

    `EnvelopeFollower` tracks the amplitude envelope of a signal using configurable
    attack and release time constants, suitable for dynamic range processing and
    envelope extraction.
    """

    def __init__(
        self,
        attack_ms: float,
        release_ms: float,
        sample_rate: float,
        detection_method: DynamicRangeMethod,
    ) -> None:
        """
        Create a new EnvelopeFollower.

        Parameters
        ----------
        attack_ms : float
            Attack time in milliseconds
        release_ms : float
            Release time in milliseconds
        sample_rate : float
            Sample rate in Hz
        detection_method : DynamicRangeMethod
            Detection method (Peak, Rms, or Hybrid)
        """
        ...

    @classmethod
    def default(cls) -> EnvelopeFollower:
        """Create a default EnvelopeFollower with 10ms attack and 100ms release at 44100 Hz."""
        ...

    ...

class SideChainConfig:
    """
    Side-chain configuration for dynamic range processing.

    `SideChainConfig` describes how an external or filtered control signal is used
    to drive gain reduction in dynamics processors such as compressors and
    limiters.

    Instances are immutable value objects. Parameters are provided at
    construction time and exposed via read-only properties.
    """

    def __init__(
        self,
        enabled: bool,
        high_pass_freq: Optional[float] = None,
        low_pass_freq: Optional[float] = None,
        pre_emphasis_db: float = 0.0,
        external_mix: float = 0.0,
    ) -> None:
        """
        Create a new side-chain configuration.

        Parameters
        ----------
        enabled : bool
            Whether side-chain processing is enabled.
        high_pass_freq : float or None, optional
            High-pass filter cutoff frequency in Hz for the side-chain signal.
        low_pass_freq : float or None, optional
            Low-pass filter cutoff frequency in Hz for the side-chain signal.
        pre_emphasis_db : float, optional
            Pre-emphasis applied to the side-chain signal in decibels.
        external_mix : float, optional
            Mix ratio between internal and external side-chain signal in the
            range ``[0.0, 1.0]``.
        """
        ...

    @property
    def enabled(self) -> bool: ...
    """Whether side-chain processing is enabled."""

    @property
    def high_pass_freq(self) -> Optional[float]: ...
    """High-pass filter cutoff frequency in Hz for the side-chain signal."""

    @property
    def low_pass_freq(self) -> Optional[float]: ...
    """Low-pass filter cutoff frequency in Hz for the side-chain signal."""

    @property
    def pre_emphasis_db(self) -> float: ...
    """Pre-emphasis applied to the side-chain signal in decibels."""

    @property
    def external_mix(self) -> float: ...
    """Mix ratio between internal and external side-chain signal."""

    ...

class CompressorConfig:
    """
    Compressor configuration parameters.

    `CompressorConfig` defines how a dynamic range compressor responds to signal
    levels above a threshold, including time constants, knee behaviour, detection
    method, and side-chain configuration.

    Instances are immutable value objects. Validation can be performed explicitly
    using ``validate(sample_rate)``.

    Several common presets are available as class attributes.
    """

    def __init__(
        self,
        *,
        threshold_db: float,
        ratio: float,
        attack_ms: float,
        release_ms: float,
        makeup_gain_db: float,
        knee_type: KneeType,
        knee_width_db: float,
        detection_method: DynamicRangeMethod,
        side_chain: SideChainConfig,
        lookahead_ms: float,
    ) -> None:
        """
        Create a new compressor configuration.
        """
        ...

    vocal: CompressorConfig
    """Vocal compression preset."""

    drum: CompressorConfig
    """Drum compression preset."""

    bus: CompressorConfig
    """Bus compression preset."""

    def validate(self, sample_rate: float) -> None:
        """
        Validate compressor configuration parameters.

        Parameters
        ----------
        sample_rate : float
            Audio sample rate in Hz.
        """
        ...

    ...

class LimiterConfig:
    """
    Limiter configuration parameters.

    `LimiterConfig` defines how a limiter prevents signal levels from exceeding a
    specified ceiling, including time constants, knee behaviour, detection
    method, side-chain configuration, and inter-sample peak (ISP) limiting.

    Instances are immutable value objects. Validation can be performed explicitly
    using ``validate(sample_rate)``.
    """

    def __init__(
        self,
        *,
        ceiling_db: float,
        attack_ms: float,
        release_ms: float,
        knee_type: KneeType,
        knee_width_db: float,
        detection_method: DynamicRangeMethod,
        side_chain: SideChainConfig,
        lookahead_ms: float,
        isp_limiting: bool,
    ) -> None:
        """
        Create a new limiter configuration.
        """
        ...

    @classmethod
    def transparent(cls) -> LimiterConfig:
        """Transparent limiter preset."""
        ...

    @classmethod
    def mastering(cls) -> LimiterConfig:
        """Mastering limiter preset."""
        ...

    @classmethod
    def broadcast(cls) -> LimiterConfig:
        """Broadcast limiter preset."""
        ...

    @property
    def ceiling_db(self) -> float: ...
    """Ceiling level in decibels."""

    @property
    def attack_ms(self) -> float: ...
    """Attack time in milliseconds."""

    @property
    def release_ms(self) -> float: ...
    """Release time in milliseconds."""

    @property
    def knee_type(self) -> KneeType: ...
    """Knee characteristic."""

    @property
    def knee_width_db(self) -> float: ...
    """Knee width in decibels."""

    @property
    def detection_method(self) -> DynamicRangeMethod: ...
    """Detection method used for limiting."""

    @property
    def side_chain(self) -> SideChainConfig: ...
    """Side-chain configuration."""

    @property
    def lookahead_ms(self) -> float: ...
    """Lookahead time in milliseconds."""

    @property
    def isp_limiting(self) -> bool: ...
    """Whether inter-sample peak limiting is enabled."""

    def validate(self, sample_rate: float) -> None:
        """
        Validate limiter configuration parameters.

        Parameters
        ----------
        sample_rate : float
            Audio sample rate in Hz.
        """
        ...

    ...

class AdaptiveThresholdMethod:
    """
    Adaptive thresholding strategy for peak picking.

    Selects how dynamic detection thresholds are estimated from the onset
    strength function over time. Different strategies trade off responsiveness
    against robustness to noise and transient outliers.
    """

    delta: AdaptiveThresholdMethod
    """Delta-based adaptive threshold.

    Tracks local maxima and applies a fixed offset to determine the detection
    threshold. Responds quickly to rapid changes but can be sensitive to noise
    and transient outliers.
    """

    percentile: AdaptiveThresholdMethod
    """Percentile-based adaptive threshold.

    Estimates the threshold from rolling distribution statistics of the onset
    strength function, yielding increased robustness at the cost of slower
    adaptation.
    """

    combined: AdaptiveThresholdMethod
    """Combined adaptive threshold.

    Combines delta-based and percentile-based thresholds to balance
    responsiveness and robustness across a wide range of signals.
    """

    ...

class NoiseColor:
    """
    Noise colour classification for audio perturbation and synthesis.

    Classifies stochastic noise processes by their spectral energy distribution.
    Different noise colours influence perceived brightness, smoothness, and
    temporal correlation in audio synthesis and perturbation tasks.
    """

    white: NoiseColor
    """White noise.

    Exhibits approximately uniform spectral energy density across the frequency
    spectrum, resulting in a bright and broadband character.
    """

    pink: NoiseColor
    """Pink noise.

    Exhibits decreasing spectral energy with increasing frequency, producing a
    perceptually balanced spectrum across octaves.
    """

    brown: NoiseColor
    """Brown (red) noise.

    Exhibits strongly attenuated high-frequency content, yielding a smoother and
    more correlated temporal structure.
    """

    ...

class PerturbationMethod:
    """
    Perturbation methods for audio data augmentation.

    Represents a specific audio perturbation configuration used for data
    augmentation, robustness testing, or creative effects.

    Instances are constructed via class constructors (for example,
    ``PerturbationMethod.gaussian(...)``). Use ``validate(sample_rate)`` to
    check parameter validity for a given sample rate.
    """

    @classmethod
    def white_noise(cls, target_snr_db: float) -> PerturbationMethod:
        """
        White noise injection with a target signal-to-noise ratio.

        Adds white Gaussian noise to achieve the target SNR relative to the
        input signal's RMS level.
        """
        ...

    @classmethod
    def pink_noise(cls, target_snr_db: float) -> PerturbationMethod:
        """
        Pink noise injection with a target signal-to-noise ratio.

        Adds pink Gaussian noise to achieve the target SNR relative to the
        input signal's RMS level.
        """
        ...

    @classmethod
    def brown_noise(cls, target_snr_db: float) -> PerturbationMethod:
        """
        Brown noise injection with a target signal-to-noise ratio.

        Adds brown Gaussian noise to achieve the target SNR relative to the
        input signal's RMS level.
        """
        ...

    @classmethod
    def gaussian(
        cls, target_snr_db: float, noise_color: NoiseColor
    ) -> PerturbationMethod:
        """
        Gaussian noise injection with a target signal-to-noise ratio.

        Adds coloured Gaussian noise to achieve the target SNR relative to the
        input signal's RMS level.
        """
        ...

    @classmethod
    def random_gain(cls, min_gain_db: float, max_gain_db: float) -> PerturbationMethod:
        """
        Random gain perturbation within a specified range.

        Applies a uniform random gain (in dB) to all channels.
        """
        ...

    @classmethod
    def high_pass_filter(cls, cutoff_hz: float) -> PerturbationMethod:
        """High-pass filtering perturbation."""
        ...

    @classmethod
    def high_pass_filter_with_slope(
        cls, cutoff_hz: float, slope_db_per_octave: float
    ) -> PerturbationMethod:
        """High-pass filtering perturbation with custom slope."""
        ...

    @classmethod
    def low_pass_filter(cls, cutoff_hz: float) -> PerturbationMethod:
        """Low-pass filtering perturbation."""
        ...

    @classmethod
    def low_pass_filter_with_slope(
        cls, cutoff_hz: float, slope_db_per_octave: float
    ) -> PerturbationMethod:
        """Low-pass filtering perturbation with custom slope."""
        ...

    @classmethod
    def pitch_shift(
        cls, semitones: float, preserve_formants: bool = False
    ) -> PerturbationMethod:
        """
        Pitch shifting perturbation.

        Shifts pitch by a number of semitones while attempting to preserve
        duration.
        """
        ...

    def validate(self, sample_rate: float) -> PerturbationMethod:
        """
        Validate perturbation parameters for a given sample rate.

        Returns this instance to allow chaining.
        """
        ...

    ...

class PerturbationConfig:
    """
    Configuration for audio perturbation operations.

    Defines how a perturbation method should be applied to audio data, optionally
    including a deterministic random seed. The configuration is immutable once
    created and can be validated explicitly.
    """

    def __init__(
        self,
        method: PerturbationMethod,
        seed: Optional[int] = None,
    ) -> None: ...
    @property
    def seed(self) -> Optional[int]:
        """Random seed used for deterministic perturbation, if specified."""
        ...

    def validate(self, sample_rate: float) -> PerturbationConfig:
        """
        Validate the perturbation configuration for a given sample rate.

        Returns this instance to allow chaining.
        """
        ...

    ...

class HpssConfig:
    """
    Configuration for Harmonic / Percussive Source Separation (HPSS).

    Separates audio into harmonic and percussive components using STFT
    magnitude median filtering.
    """

    def __init__(
        self,
        *,
        n_fft: int = 2048,
        win_size: int = 2048,
        hop_size: int = 512,
        median_filter_harmonic: int = 17,
        median_filter_percussive: int = 17,
        mask_softness: float = 0.3,
    ) -> None: ...

    musical: HpssConfig
    """Optimised for musical content."""

    percussive: HpssConfig
    """Optimised for percussive content."""

    harmonic: HpssConfig
    """Optimised for harmonic content."""

    real_time: HpssConfig
    """Optimised for low-latency real-time processing."""

    def set_stft_params(self, n_fft: int, win_size: int, hop_size: int) -> None:
        """
        Set STFT parameters.
        """
        ...

    def set_filter_sizes(self, harmonic_size: int, percussive_size: int) -> None:
        """
        Set median filter sizes.
        """
        ...

    def set_mask_softness(self, softness: float) -> None:
        """
        Set mask softness parameter.
        """
        ...

    def validate(self, sample_rate: float) -> HpssConfig:
        """
        Validate configuration and return this instance.
        """
        ...

    def num_freq_bins(self) -> int:
        """
        Return the number of frequency bins.
        """
        ...

    def freq_resolution(self, sample_rate: float) -> float:
        """
        Return frequency resolution in Hz.
        """
        ...

    def time_resolution(self, sample_rate: float) -> float:
        """
        Return time resolution in seconds.
        """
        ...

    ...

class AdaptiveThresholdConfig:
    """
    Configuration for adaptive thresholding in peak picking.

    Adaptive thresholding dynamically adjusts the detection threshold
    based on local characteristics of the onset strength function.
    """

    def __init__(
        self,
        method: AdaptiveThresholdMethod,
        *,
        delta: float,
        percentile: float,
        window_size: int,
        min_threshold: float,
        max_threshold: float,
    ) -> None: ...

    # --- Presets -------------------------------------------------

    @classmethod
    def delta(cls, delta: float, window_size: int) -> AdaptiveThresholdConfig: ...
    @classmethod
    def percentile(
        cls,
        percentile: float,
        window_size: int,
    ) -> AdaptiveThresholdConfig: ...
    @classmethod
    def combined(
        cls,
        delta: float,
        percentile: float,
        window_size: int,
    ) -> AdaptiveThresholdConfig: ...

    # --- Mutators ------------------------------------------------

    def set_min_threshold(self, value: float) -> None: ...
    def set_max_threshold(self, value: float) -> None: ...

    # --- Validation ----------------------------------------------

    def validate(self) -> AdaptiveThresholdConfig: ...

class PeakPickingConfig:
    """
    Configuration for peak picking with temporal constraints.

    Peak picking identifies local maxima in the onset strength function that
    exceed a threshold. Temporal constraints ensure detected peaks are
    separated by minimum time intervals and can include smoothing.
    """

    def __init__(
        self,
        *,
        adaptive_threshold_config: AdaptiveThresholdConfig,
        min_peak_separation: int,
        pre_emphasis: bool,
        pre_emphasis_coeff: float,
        median_filter: bool,
        median_filter_length: int,
        normalize_onset_strength: bool,
        normalization_method: NormalizationMethod,
    ) -> None: ...
    @classmethod
    def music(cls) -> PeakPickingConfig: ...
    @classmethod
    def speech(cls) -> PeakPickingConfig: ...
    @classmethod
    def drums(cls) -> PeakPickingConfig: ...
    def set_min_peak_separation(self, value: int) -> None: ...
    def set_min_peak_separation_ms(self, value: float, sample_rate: float) -> None: ...
    def set_pre_emphasis(self, enabled: bool, coeff: float) -> None: ...
    def set_median_filter(self, enabled: bool, length: int) -> None: ...
    def validate(self) -> PeakPickingConfig: ...

class SpectralFluxMethod:
    """
    Spectral flux variant for onset detection.

    Different flux formulations emphasise different types of spectral change
    and are therefore suited to different classes of musical and acoustic events.
    """

    energy: SpectralFluxMethod
    """
    Energy-based spectral flux.

    Measures positive changes in spectral energy between successive frames.
    Performs well for transient and percussive onsets.
    """

    magnitude: SpectralFluxMethod
    """
    Magnitude-based spectral flux.

    Measures positive changes in spectral magnitude and is more sensitive
    to subtle spectral variation, making it effective for tonal material.
    """

    complex: SpectralFluxMethod
    """
    Complex-domain spectral flux.

    Incorporates phase information in addition to magnitude, improving
    robustness to noise and spectral smearing at increased computational cost.
    """

    rectified_complex: SpectralFluxMethod
    """
    Rectified complex-domain spectral flux.

    Suppresses negative phase contributions to balance sensitivity and robustness.
    """

class SpectralFluxConfig:
    """
    Configuration for spectral flux onset detection.

    Spectral flux measures the rate of change of the magnitude spectrum
    between consecutive frames, providing effective onset detection for
    both percussive and tonal instruments.
    """

    def __init__(
        self,
        *,
        cqt_config: CqtParams,
        hop_size: int,
        window_size: Optional[int],
        flux_method: SpectralFluxMethod,
        peak_picking: PeakPickingConfig,
        rectify: bool,
        log_compression: float,
    ) -> None: ...
    percussive: SpectralFluxConfig
    """ Configuration optimised for percussive onsets."""

    musical: SpectralFluxConfig
    """ Configuration optimised for musical onsets."""

    complex: SpectralFluxConfig
    """ Configuration optimised for complex onsets."""

    def validate(self, sample_rate: float) -> SpectralFluxConfig: ...
    @property
    def cqt_config(self) -> CqtParams: ...
    @cqt_config.setter
    def cqt_config(self, value: CqtParams) -> None: ...
    @property
    def hop_size(self) -> int: ...
    @hop_size.setter
    def hop_size(self, value: int) -> None: ...
    @property
    def window_size(self) -> Optional[int]: ...
    @window_size.setter
    def window_size(self, value: Optional[int]) -> None: ...
    @property
    def flux_method(self) -> SpectralFluxMethod: ...
    @flux_method.setter
    def flux_method(self, value: SpectralFluxMethod) -> None: ...
    @property
    def peak_picking(self) -> PeakPickingConfig: ...
    @peak_picking.setter
    def peak_picking(self, value: PeakPickingConfig) -> None: ...
    @property
    def rectify(self) -> bool: ...
    @rectify.setter
    def rectify(self, value: bool) -> None: ...
    @property
    def log_compression(self) -> float: ...
    @log_compression.setter
    def log_compression(self, value: float) -> None: ...

    ...

class ComplexOnsetConfig:
    """
    Configuration for complex domain onset detection.

    Complex domain onset detection uses both magnitude and phase information
    from the CQT to provide more accurate onset detection than magnitude-only
    methods, especially for polyphonic music and complex timbres.
    """

    def __init__(
        self,
        *,
        cqt_config: CqtParams,
        hop_size: int,
        window_size: Optional[int],
        peak_picking: PeakPickingConfig,
        magnitude_weight: float,
        phase_weight: float,
        magnitude_rectify: bool,
        phase_rectify: bool,
        log_compression: float,
    ) -> None: ...

    percussive: ComplexOnsetConfig
    """ Configuration optimised for percussive onsets."""

    musical: ComplexOnsetConfig
    """ Configuration optimised for musical onsets."""

    speech: ComplexOnsetConfig
    """ Configuration optimised for speech onsets."""

    def set_weights(self, magnitude_weight: float, phase_weight: float) -> None: ...
    def validate(self, sample_rate: float) -> ComplexOnsetConfig: ...
    @property
    def cqt_config(self) -> CqtParams: ...
    @cqt_config.setter
    def cqt_config(self, value: CqtParams) -> None: ...
    @property
    def hop_size(self) -> int: ...
    @hop_size.setter
    def hop_size(self, value: int) -> None: ...
    @property
    def window_size(self) -> Optional[int]: ...
    @window_size.setter
    def window_size(self, value: Optional[int]) -> None: ...
    @property
    def peak_picking(self) -> PeakPickingConfig: ...
    @peak_picking.setter
    def peak_picking(self, value: PeakPickingConfig) -> None: ...
    @property
    def magnitude_weight(self) -> float: ...
    @magnitude_weight.setter
    def magnitude_weight(self, value: float) -> None: ...
    @property
    def phase_weight(self) -> float: ...
    @phase_weight.setter
    def phase_weight(self, value: float) -> None: ...
    @property
    def magnitude_rectify(self) -> bool: ...
    @magnitude_rectify.setter
    def magnitude_rectify(self, value: bool) -> None: ...
    @property
    def phase_rectify(self) -> bool: ...
    @phase_rectify.setter
    def phase_rectify(self, value: bool) -> None: ...
    @property
    def log_compression(self) -> float: ...
    @log_compression.setter
    def log_compression(self, value: float) -> None: ...

    ...

class OnsetDetectionConfig:
    """
    Configuration for onset detection.

    Onset detection identifies the start times of musical notes, transients,
    or other acoustic events. This configuration controls the spectral analysis
    parameters, peak picking strategy, and various preprocessing options.
    """

    def __init__(
        self,
        cqt_params: CqtParams,
        hop_size: int,
        window_size: Optional[int] = None,
        threshold: float = 0.3,
        min_onset_interval_secs: float = 0.07,
        pre_emphasis: float = 0.0,
        adaptive_threshold: bool = True,
        median_filter_length: int = 3,
        adaptive_threshold_multiplier: float = 3.0,
        peak_picking: Optional[PeakPickingConfig] = None,
    ) -> None: ...
    @classmethod
    def default(cls) -> OnsetDetectionConfig: ...
    @classmethod
    def musical(cls) -> OnsetDetectionConfig: ...
    @classmethod
    def percussive(cls) -> OnsetDetectionConfig: ...
    @classmethod
    def speech(cls) -> OnsetDetectionConfig: ...
    @property
    def cqt_params(self) -> CqtParams: ...
    @cqt_params.setter
    def cqt_params(self, value: CqtParams) -> None: ...
    @property
    def hop_size(self) -> int: ...
    @hop_size.setter
    def hop_size(self, value: int) -> None: ...

    ...

class BeatTrackingConfig:
    """Configuration for beat detection."""

    def __init__(
        self,
        tempo_bpm: float,
        onset_config: OnsetDetectionConfig,
        tolerance: Optional[float] = None,
    ) -> None: ...
    @property
    def tempo_bpm(self) -> float: ...
    @tempo_bpm.setter
    def tempo_bpm(self, value: float) -> None: ...
    @property
    def tolerance(self) -> Optional[float]: ...
    @tolerance.setter
    def tolerance(self, value: Optional[float]) -> None: ...
    @property
    def onset_config(self) -> OnsetDetectionConfig: ...
    @onset_config.setter
    def onset_config(self, value: OnsetDetectionConfig) -> None: ...

    ...

class BeatTrackingData:
    """Beat tracking results containing tempo and beat timestamps."""

    def __init__(
        self,
        tempo_bpm: float,
        beat_times: list[float],
        config: BeatTrackingConfig,
    ) -> None: ...
    @property
    def tempo_bpm(self) -> float: ...
    @property
    def beat_times(self) -> list[float]: ...
    @property
    def config(self) -> BeatTrackingConfig: ...

    ...

class Layout:
    """Plot layout direction for multi-channel visualizations."""

    vertical: Layout
    horizontal: Layout

    @classmethod
    def default(cls) -> Layout: ...

    ...

class ChannelManagementStrategy:
    """Strategy for handling multi-channel audio in plots."""

    @classmethod
    def average(cls) -> ChannelManagementStrategy: ...
    @classmethod
    def separate(cls, layout: Layout) -> ChannelManagementStrategy: ...
    @classmethod
    def first(cls) -> ChannelManagementStrategy: ...
    @classmethod
    def last(cls) -> ChannelManagementStrategy: ...
    @classmethod
    def overlap(cls) -> ChannelManagementStrategy: ...
    @classmethod
    def default(cls) -> ChannelManagementStrategy: ...

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
    ) -> None: ...
    @staticmethod
    def default() -> WaveformPlotParams: ...

    ...

class SpectrogramPlotParams:
    """Configuration parameters for spectrogram plots."""

    def __init__(self, title: Optional[str] = None) -> None: ...
    @staticmethod
    def default() -> SpectrogramPlotParams: ...

    ...

class MagnitudeSpectrumParams:
    """Configuration parameters for magnitude spectrum plots."""

    def __init__(
        self, title: Optional[str] = None, n_fft: Optional[int] = None
    ) -> None: ...
    @staticmethod
    def default() -> MagnitudeSpectrumParams: ...

    ...

class WaveformPlot:
    """Waveform plot result with methods for display and export."""

    def html(self) -> str: ...
    def save(self, path: str) -> None: ...
    def show(self) -> None: ...

    ...

class SpectrogramPlot:
    """Spectrogram plot result with methods for display and export."""

    def html(self) -> str: ...
    def save(self, path: str) -> None: ...
    def show(self) -> None: ...

    ...

class MagnitudeSpectrumPlot:
    """Magnitude spectrum plot result with methods for display and export."""

    def html(self) -> str: ...
    def save(self, path: str) -> None: ...
    def show(self) -> None: ...

    ...

class ChannelReduction:
    """
    Strategy for reducing a multi-channel signal to a single channel.

    `ChannelReduction` controls how operations that fundamentally require a single
    channel (such as spectral centroid or roll-off) behave when presented with a
    multi-channel signal. It lets the caller choose between failing loudly,
    selecting a single channel, or averaging across channels.

    Instances are immutable and should be treated as enum-like values.
    Zero-parameter strategies are accessed via class attributes, while the
    channel-selection strategy is constructed via :meth:`channel`.
    """

    error: ChannelReduction
    """Return an error when the signal has more than one channel.

    This is the default strategy and preserves the strictest behaviour by
    refusing to silently collapse channels.
    """

    first: ChannelReduction
    """Use the first channel (index 0) and ignore the rest."""

    average: ChannelReduction
    """Average the corresponding samples across all channels."""

    @staticmethod
    def channel(index: int) -> ChannelReduction:
        """
        Use the channel at the given index.

        The index is bounds-checked when the reduction is applied.

        Parameters
        ----------
        index : int
            Zero-based channel index to select.
        """
        ...

    ...

class GateConfig:
    """
    Noise gate configuration parameters.

    `GateConfig` defines how a downward noise gate attenuates a signal that falls
    below a threshold, including the attenuation ratio and the attack and release
    envelope time constants.

    Instances are immutable value objects. Parameters are provided at construction
    time and exposed via read-only properties. Validation can be performed
    explicitly using ``validate()``.

    A general-purpose noise-gating preset is provided via ``GateConfig.noise_gate``.
    """

    def __init__(
        self,
        *,
        threshold_db: float,
        ratio: float,
        attack_ms: float,
        release_ms: float,
    ) -> None:
        """
        Create a new gate configuration.

        Parameters
        ----------
        threshold_db : float
            Gate threshold in dBFS. Signals below this level are attenuated.
            Typical range: [-80.0, 0.0].
        ratio : float
            Attenuation ratio applied below the threshold. Higher values produce
            more aggressive gating; values near 1.0 approach unity gain. Must be
            greater than 0.0.
        attack_ms : float
            Attack time in milliseconds. Controls how quickly the gate opens once
            the signal rises above the threshold. Valid range: [0.01, 1000.0] ms.
        release_ms : float
            Release time in milliseconds. Controls how quickly the gate closes
            once the signal falls below the threshold. Valid range:
            [1.0, 10000.0] ms.
        """
        ...

    noise_gate: GateConfig
    """General-purpose noise-gating preset.

    Moderate threshold and a high ratio for firmly attenuating background noise
    and room tone between phrases.
    """

    @property
    def threshold_db(self) -> float: ...
    """Gate threshold in dBFS."""

    @property
    def ratio(self) -> float: ...
    """Attenuation ratio applied below the threshold."""

    @property
    def attack_ms(self) -> float: ...
    """Attack time in milliseconds."""

    @property
    def release_ms(self) -> float: ...
    """Release time in milliseconds."""

    def validate(self) -> GateConfig:
        """
        Validate gate configuration parameters.

        Returns
        -------
        GateConfig
            The validated gate configuration.

        Raises
        ------
        ValueError
            If any configuration parameter is invalid.
        """
        ...

    ...

class ExpanderConfig:
    """
    Downward expander configuration parameters.

    `ExpanderConfig` defines how a downward expander attenuates a signal that
    falls below a threshold, increasing the dynamic range of low-level material.
    RMS detection is always used for expansion.

    Instances are immutable value objects. Parameters are provided at construction
    time and exposed via read-only properties. Validation can be performed
    explicitly using ``validate()``.

    A gentle expansion preset is provided via ``ExpanderConfig.gentle``.
    """

    def __init__(
        self,
        *,
        threshold_db: float,
        ratio: float,
        attack_ms: float,
        release_ms: float,
    ) -> None:
        """
        Create a new expander configuration.

        Parameters
        ----------
        threshold_db : float
            Expansion threshold in dBFS. Signals below this level are attenuated.
            Typical range: [-80.0, 0.0].
        ratio : float
            Expansion ratio applied below the threshold. Values greater than 1.0
            produce increasing attenuation the further the signal falls below the
            threshold. Must be greater than 0.0.
        attack_ms : float
            Attack time in milliseconds. Valid range: [0.01, 1000.0] ms.
        release_ms : float
            Release time in milliseconds. Valid range: [1.0, 10000.0] ms.
        """
        ...

    gentle: ExpanderConfig
    """Gentle downward expansion preset.

    A low ratio and relaxed envelope times for subtle dynamic-range enhancement
    without obvious pumping.
    """

    @property
    def threshold_db(self) -> float: ...
    """Expansion threshold in dBFS."""

    @property
    def ratio(self) -> float: ...
    """Expansion ratio applied below the threshold."""

    @property
    def attack_ms(self) -> float: ...
    """Attack time in milliseconds."""

    @property
    def release_ms(self) -> float: ...
    """Release time in milliseconds."""

    def validate(self) -> ExpanderConfig:
        """
        Validate expander configuration parameters.

        Returns
        -------
        ExpanderConfig
            The validated expander configuration.

        Raises
        ------
        ValueError
            If any configuration parameter is invalid.
        """
        ...

    ...

class ThreeBandEqConfig:
    """
    Three-band parametric equaliser configuration.

    `ThreeBandEqConfig` describes a simple three-band equaliser composed of a low
    shelf, a mid peaking band, and a high shelf. It is a convenience configuration
    for common tone-shaping tasks without constructing individual EQ bands.

    Instances are immutable value objects. Parameters are provided at construction
    time and exposed via read-only properties. Validation can be performed
    explicitly using ``validate()``.

    A flat (unity-gain) preset is provided via ``ThreeBandEqConfig.flat``.
    """

    def __init__(
        self,
        *,
        low_freq: float,
        low_gain: float,
        mid_freq: float,
        mid_gain: float,
        mid_q: float,
        high_freq: float,
        high_gain: float,
    ) -> None:
        """
        Create a new three-band EQ configuration.

        Parameters
        ----------
        low_freq : float
            Low shelf corner frequency in Hz. Must be greater than 0.0 and less
            than ``mid_freq``.
        low_gain : float
            Low shelf gain in dB.
        mid_freq : float
            Mid peak centre frequency in Hz. Must be greater than ``low_freq`` and
            less than ``high_freq``.
        mid_gain : float
            Mid peak gain in dB.
        mid_q : float
            Mid peak Q factor. Must be greater than 0.0.
        high_freq : float
            High shelf corner frequency in Hz. Must be greater than ``mid_freq``.
        high_gain : float
            High shelf gain in dB.
        """
        ...

    flat: ThreeBandEqConfig
    """Flat (unity-gain) three-band EQ preset.

    Low shelf at 200 Hz, mid peak at 1 kHz (Q = 1.0), high shelf at 4 kHz, all
    gains at 0 dB. A neutral starting point for further adjustment.
    """

    @property
    def low_freq(self) -> float: ...
    """Low shelf corner frequency in Hz."""

    @property
    def low_gain(self) -> float: ...
    """Low shelf gain in dB."""

    @property
    def mid_freq(self) -> float: ...
    """Mid peak centre frequency in Hz."""

    @property
    def mid_gain(self) -> float: ...
    """Mid peak gain in dB."""

    @property
    def mid_q(self) -> float: ...
    """Mid peak Q factor."""

    @property
    def high_freq(self) -> float: ...
    """High shelf corner frequency in Hz."""

    @property
    def high_gain(self) -> float: ...
    """High shelf gain in dB."""

    def validate(self) -> ThreeBandEqConfig:
        """
        Validate the three-band EQ parameters.

        Returns
        -------
        ThreeBandEqConfig
            The validated configuration.

        Raises
        ------
        ValueError
            If any configuration parameter is invalid (non-positive frequency,
            mis-ordered frequencies, or non-positive mid Q).
        """
        ...

    ...

class Psd:
    """
    Power spectral density (PSD) estimate.

    `Psd` pairs a frequency axis with the estimated power-per-Hz at each bin. The
    two arrays are always the same length: ``frequencies[i]`` is the centre
    frequency of bin ``i`` in Hz, and ``density[i]`` is the estimated power
    spectral density at that frequency.

    Instances are produced by the power spectral density transform and are not
    constructed directly from Python.
    """

    @property
    def frequencies(self) -> numpy.ndarray:
        """The frequency axis, in Hz (a 1-D array, same length as ``density``)."""
        ...

    @property
    def density(self) -> numpy.ndarray:
        """The estimated power spectral density (power per Hz) at each frequency bin."""
        ...

    def into_parts(self) -> tuple[numpy.ndarray, numpy.ndarray]:
        """
        Return the ``(frequencies, density)`` arrays as a tuple.

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray]
            A pair of 1-D NumPy arrays containing the frequency axis and the
            density values respectively.
        """
        ...

    def __len__(self) -> int:
        """The number of frequency bins."""
        ...

    def __repr__(self) -> str: ...

class PitchClass:
    """
    One of the twelve pitch classes of the chromatic scale.

    `PitchClass` represents a pitch class independent of octave. Pitch class ``C``
    has index 0, ascending chromatically to ``B`` at index 11.

    Instances are immutable and should be treated as enum-like values. They are
    accessed via class attributes, or constructed from a chromatic index via
    :meth:`from_index`.
    """

    c: PitchClass
    """Pitch class C (index 0)."""

    c_sharp: PitchClass
    """Pitch class C# / Db (index 1)."""

    d: PitchClass
    """Pitch class D (index 2)."""

    d_sharp: PitchClass
    """Pitch class D# / Eb (index 3)."""

    e: PitchClass
    """Pitch class E (index 4)."""

    f: PitchClass
    """Pitch class F (index 5)."""

    f_sharp: PitchClass
    """Pitch class F# / Gb (index 6)."""

    g: PitchClass
    """Pitch class G (index 7)."""

    g_sharp: PitchClass
    """Pitch class G# / Ab (index 8)."""

    a: PitchClass
    """Pitch class A (index 9)."""

    a_sharp: PitchClass
    """Pitch class A# / Bb (index 10)."""

    b: PitchClass
    """Pitch class B (index 11)."""

    @staticmethod
    def from_index(index: int) -> PitchClass:
        """
        Construct a pitch class from a chromatic index in ``0..=11``.

        Parameters
        ----------
        index : int
            Chromatic index where 0 maps to C and 11 maps to B.

        Returns
        -------
        PitchClass
            The pitch class for the given index.

        Raises
        ------
        ValueError
            If ``index`` is greater than 11.
        """
        ...

    def to_index(self) -> int:
        """The chromatic index of this pitch class (C = 0, B = 11)."""
        ...

    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...

class Mode:
    """
    The mode (tonality) of an estimated musical key.

    `Mode` distinguishes between the major and minor tonalities of a key estimate.

    Instances are immutable and should be treated as enum-like values. They are
    accessed via class attributes rather than being constructed directly.
    """

    major: Mode
    """Major mode."""

    minor: Mode
    """Minor mode."""

    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...

class Key:
    """
    A musical key estimate.

    `Key` pairs a tonic pitch class with a mode (major or minor) and a confidence
    score describing how strongly the estimate matched the analysed signal.

    Instances are typically produced by key estimation, but can also be
    constructed directly. Fields are exposed via read-only properties.
    """

    def __init__(self, tonic: PitchClass, mode: Mode, confidence: float) -> None:
        """
        Create a new key estimate.

        Parameters
        ----------
        tonic : PitchClass
            The tonic pitch class of the key.
        mode : Mode
            The mode (major or minor) of the key.
        confidence : float
            Match confidence in [0.0, 1.0]; higher values indicate a stronger
            match.
        """
        ...

    @property
    def tonic(self) -> PitchClass: ...
    """The tonic pitch class of the key."""

    @property
    def mode(self) -> Mode: ...
    """The mode (major or minor) of the key."""

    @property
    def confidence(self) -> float: ...
    """Match confidence in [0.0, 1.0]."""

    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...

class PitchFrame:
    """
    A single frame of a pitch contour.

    `PitchFrame` pairs a frame onset time with the fundamental frequency detected
    in that frame. The frequency is ``None`` for unvoiced frames where no pitch
    was detected.

    Fields are exposed via read-only properties.
    """

    def __init__(self, time: float, frequency: Optional[float] = None) -> None:
        """
        Create a new pitch frame.

        Parameters
        ----------
        time : float
            Frame onset time, in seconds from the start of the signal.
        frequency : float, optional
            Detected fundamental frequency in Hz, or ``None`` if the frame is
            unvoiced.
        """
        ...

    @property
    def time(self) -> float: ...
    """Frame onset time, in seconds from the start of the signal."""

    @property
    def frequency(self) -> Optional[float]: ...
    """Detected fundamental frequency in Hz, or ``None`` if the frame is unvoiced."""

    @property
    def voiced(self) -> bool: ...
    """Whether this frame is voiced (a pitch was detected)."""

    def __repr__(self) -> str: ...

class PitchContour:
    """
    A time-ordered pitch track.

    `PitchContour` is a sequence of :class:`PitchFrame` values produced by pitch
    tracking. Each frame pairs a frame onset time with the pitch detected in that
    frame (or ``None`` when no pitch was found).

    Instances are produced by pitch tracking and are not constructed directly from
    Python.
    """

    def frames(self) -> list[PitchFrame]:
        """
        All frames in time order, voiced and unvoiced alike.

        Returns
        -------
        list[PitchFrame]
            The complete list of pitch frames.
        """
        ...

    def voiced_frames(self) -> list[tuple[float, float]]:
        """
        Voiced frames as ``(time_seconds, frequency_hz)`` pairs.

        Unvoiced frames are skipped.

        Returns
        -------
        list[tuple[float, float]]
            The voiced frames as (time, frequency) tuples.
        """
        ...

    def mean_pitch(self) -> Optional[float]:
        """The mean of all voiced frequencies, or ``None`` when no frame is voiced."""
        ...

    def __len__(self) -> int:
        """The total number of frames (voiced and unvoiced)."""
        ...

    def __repr__(self) -> str: ...

class SosFilter:
    """
    Streaming second-order-sections (SOS) IIR filter.

    A `SosFilter` is a cascade of biquad sections built once from an
    :class:`IirFilterDesign` and then driven sample-by-sample or block-by-block.
    Internal delay-line state persists across calls, so processing consecutive
    blocks produces exactly the same result as processing the whole signal at
    once. This makes it suitable for real-time and streaming use where redesigning
    the filter for every block would be wasteful.

    Build one with :meth:`from_design`; the sample rate is fixed at construction
    time and used for all frequency-dependent computations.
    """

    @staticmethod
    def from_design(design: IirFilterDesign, sample_rate: float) -> SosFilter:
        """
        Build a streaming SOS filter from a filter design.

        Designs the filter once and returns a stateful cascade. The returned
        filter starts with zeroed delay lines.

        Args:
            design (IirFilterDesign): Filter specification (type, order,
                frequencies, ripple/attenuation).
            sample_rate (float): Sample rate of the signal in hertz.

        Returns:
            SosFilter: A freshly-reset streaming filter implementing the design.

        Raises:
            AudioError: If the design is invalid (out-of-range frequency,
                unsupported response, order too high, etc.).
        """
        ...

    def process_sample(self, x: float) -> float:
        """
        Process a single sample through the cascade.

        Feeds ``x`` through each section in order; internal state is updated.

        Args:
            x (float): Input sample.

        Returns:
            float: The filtered output sample.
        """
        ...

    def process_samples(self, samples: numpy.ndarray) -> numpy.ndarray:
        """
        Process an array of samples, returning a new array.

        Equivalent to calling :meth:`process_sample` for each input in order;
        state carries across the whole array.

        Args:
            samples (numpy.ndarray): Input samples.

        Returns:
            numpy.ndarray: Filtered output, same length as input.
        """
        ...

    def process_samples_in_place(self, samples: numpy.ndarray) -> None:
        """
        Process an array of samples in place.

        Overwrites each element of ``samples`` with its filtered value. The input
        array is modified directly.

        Args:
            samples (numpy.ndarray): Input/output buffer; modified in place.
        """
        ...

    def process_block(self, block: numpy.ndarray) -> None:
        """
        Process a block of samples in place, retaining state across calls.

        Alias of :meth:`process_samples_in_place` with a name that signals the
        design-once / stream-many usage. Because delay lines persist, calling
        ``process_block`` on consecutive blocks of a signal yields exactly the
        same result as one call over the whole signal.

        Args:
            block (numpy.ndarray): Input/output block; modified in place.
        """
        ...

    def reset(self) -> None:
        """
        Reset all sections' delay lines to zero.

        After a reset the cascade behaves identically to a freshly built filter
        with the same coefficients.
        """
        ...

    def frequency_response(
        self, frequencies: numpy.ndarray
    ) -> tuple[numpy.ndarray, numpy.ndarray]:
        """
        Compute the frequency response of the cascade.

        Evaluates the combined transfer function of all sections. The magnitude
        is the product of section magnitudes; the phase is the sum of section
        phases. The sample rate fixed at construction is used.

        Args:
            frequencies (numpy.ndarray): Frequencies in hertz.

        Returns:
            tuple[numpy.ndarray, numpy.ndarray]: Magnitude and phase (radians)
            arrays, each the same length as ``frequencies``.
        """
        ...
