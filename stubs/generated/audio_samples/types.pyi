"""
Configuration and result classes of the ``AudioSamples`` methods. Every class is
also available from the top-level package.
"""

from .explain import ExplainDoc
from _typeshed import Incomplete
from collections.abc import Sequence
from typing import Any, Final, final

__all__ = ["AdaptiveThresholdConfig", "AdaptiveThresholdMethod", "BeatTrackingConfig", "BeatTrackingData", "ChannelReduction", "ComplexOnsetConfig", "CompressorConfig", "DynamicRangeMethod", "EnvelopeFollower", "EqBand", "EqBandType", "ExpanderConfig", "FadeCurve", "FilterResponse", "GateConfig", "HpssConfig", "IirFilterDesign", "IirFilterType", "Key", "KneeType", "LimiterConfig", "Mode", "MonoConversionMethod", "NoiseColor", "NormalizationMethod", "OnsetDetectionConfig", "PadSide", "ParametricEq", "PeakPickingConfig", "PerturbationConfig", "PerturbationMethod", "PitchClass", "PitchContour", "PitchDetectionMethod", "PitchFrame", "Psd", "ResamplingQuality", "SampleType", "SideChainConfig", "SosFilter", "SpectralFluxConfig", "SpectralFluxMethod", "SpectrogramScale", "StereoConversionMethod", "ThreeBandEqConfig", "VadChannelPolicy", "VadConfig", "VadMethod", "WindowType"]

@final
class AdaptiveThresholdConfig:
    """
    Configuration for adaptive thresholding in peak picking.

    Adaptive thresholding dynamically adjusts the detection threshold based on
    local characteristics of the onset strength function to improve detection
    accuracy across varying signal conditions.
    """
    def __new__(cls, /, method: "AdaptiveThresholdMethod", *, delta: "float", percentile: "float", window_size: "int", min_threshold: "float", max_threshold: "float") -> AdaptiveThresholdConfig:
        """
        Create a new ``AdaptiveThresholdConfig``.

        Parameters
        ----------
        method : ``AdaptiveThresholdMethod``
            Adaptive thresholding method to use.
        delta : float
            Delta offset for delta-based thresholding.
        percentile : float
           Percentile value for percentile-based thresholding.
        ``window_size`` : int
           Size of the rolling window in frames.
        ``min_threshold`` : float
           Minimum allowable threshold.
        ``max_threshold`` : float
          Maximum allowable threshold.

        Returns
        -------
        AdaptiveThresholdConfig
          A new adaptive threshold configuration.
        """
    def __repr__(self, /) -> str: ...
    @classmethod
    def combined(cls, /, delta: "float", percentile: "float", window_size: "int") -> AdaptiveThresholdConfig:
        """
        Create a combined adaptive threshold configuration.

        Parameters
        ----------
        delta : float
           Delta offset for thresholding.
        percentile : float
           Percentile value for thresholding.
        ``window_size`` : int
           Size of the rolling window in frames.

        Returns
        -------
        ``AdaptiveThresholdConfig``
           A new combined adaptive threshold configuration.
        """
    @staticmethod
    def default() -> AdaptiveThresholdConfig: ...
    @classmethod
    def delta(cls, /, delta: "float", window_size: "int") -> AdaptiveThresholdConfig:
        """
        Create a delta-based adaptive threshold configuration.

        Parameters
        ----------
        delta : float
            Delta offset for thresholding.
        ``window_size`` : int
            Size of the rolling window in frames.

        Returns
        -------
        ``AdaptiveThresholdConfig``
            A new delta-based adaptive threshold configuration.
        """
    @property
    def max_threshold(self, /) -> float:
        """
        Maximum allowable threshold.
        """
    @max_threshold.setter
    def max_threshold(self, /, value: float) -> None:
        """
        Set the maximum allowable threshold.
        """
    @property
    def min_threshold(self, /) -> float:
        """
        Minimum allowable threshold.
        """
    @min_threshold.setter
    def min_threshold(self, /, value: float) -> None:
        """
        Set the minimum allowable threshold.
        """
    @classmethod
    def percentile(cls, /, percentile: "float", window_size: "int") -> AdaptiveThresholdConfig:
        """
        Create a percentile-based adaptive threshold configuration.

        Parameters
        ----------
        percentile : float
            Percentile value for thresholding.
        ``window_size`` : int
            Size of the rolling window in frames.

        Returns
        -------
        ``AdaptiveThresholdConfig``
            A new percentile-based adaptive threshold configuration.
        """
    def validate(self, /) -> AdaptiveThresholdConfig:
        """
        Validate the configuration parameters.

        Raises an error if any parameters are invalid.
        """

@final
class AdaptiveThresholdMethod:
    """
    Adaptive thresholding strategy for peak picking.

    `AdaptiveThresholdMethod` selects how dynamic detection thresholds are
    estimated from the onset strength function over time when performing
    peak picking or onset detection.

    Different strategies trade off responsiveness to rapid changes against
    robustness to noise and transient outliers.

    Instances of `AdaptiveThresholdMethod` are immutable and should be treated
    as enum-like values. They are accessed via class attributes rather than
    being constructed directly.

    Available methods:

    - ``AdaptiveThresholdMethod.delta``
    - ``AdaptiveThresholdMethod.percentile``
    - ``AdaptiveThresholdMethod.combined``
    """
    combined: Final[AdaptiveThresholdMethod]
    """
    Combined adaptive threshold.

    Combines delta-based and percentile-based thresholds to balance
    responsiveness and robustness across a wide range of signals.
    """
    delta: Final[AdaptiveThresholdMethod]
    """
    Delta-based adaptive threshold.

    Tracks local maxima and applies a fixed offset to determine the detection
    threshold. Responds quickly to rapid changes but can be sensitive to
    noise and transient outliers.
    """
    percentile: Final[AdaptiveThresholdMethod]
    """
    Percentile-based adaptive threshold.

    Estimates the threshold from rolling distribution statistics of the
    onset strength function, yielding increased robustness at the cost of
    slower adaptation.
    """
    @staticmethod
    def default() -> AdaptiveThresholdMethod: ...

@final
class BeatTrackingConfig:
    """
    Python wrapper for ``BeatTrackingConfig``.

    Configuration for beat detection.
    """
    def __new__(cls, /, tempo_bpm: float, onset_config: OnsetDetectionConfig, tolerance: float | None = None) -> BeatTrackingConfig:
        """
        Create a new ``BeatTrackingConfig``.

        Parameters
        ----------
        ``tempo_bpm`` : float
            Target tempo in beats per minute
        tolerance : float, optional
            Beat timing tolerance in seconds. If None, defaults to 10% of the inter-beat interval
        ``onset_config`` : ``OnsetDetectionConfig``
            Configuration for onset detection
        """
    def __repr__(self, /) -> str: ...
    @property
    def onset_config(self, /) -> OnsetDetectionConfig:
        """
        Get the onset detection config.
        """
    @onset_config.setter
    def onset_config(self, /, onset_config: OnsetDetectionConfig) -> None:
        """
        Set the onset detection config.
        """
    @property
    def tempo_bpm(self, /) -> float:
        """
        Get the tempo in BPM.
        """
    @tempo_bpm.setter
    def tempo_bpm(self, /, tempo_bpm: float) -> None:
        """
        Set the tempo in BPM.
        """
    @property
    def tolerance(self, /) -> float | None:
        """
        Get the tolerance in seconds.
        """
    @tolerance.setter
    def tolerance(self, /, tolerance: float | None) -> None:
        """
        Set the tolerance in seconds.
        """

@final
class BeatTrackingData:
    """
    Python wrapper for ``BeatTrackingData``.

    Beat tracking results containing tempo and beat timestamps.
    """
    def __new__(cls, /, tempo_bpm: float, beat_times: Sequence[float], config: BeatTrackingConfig) -> BeatTrackingData:
        """
        Create a new ``BeatTrackingData``.

        Parameters
        ----------
        ``tempo_bpm`` : float
            Estimated tempo in beats per minute
        ``beat_times`` : list[float]
            Beat timestamps in seconds
        config : ``BeatTrackingConfig``
            Configuration used for beat detection
        """
    def __str__(self, /) -> str:
        """
        String representation showing tempo and beat count.
        """
    @property
    def beat_times(self, /) -> list[float]:
        """
        Get the beat timestamps in seconds.
        """
    @property
    def config(self, /) -> BeatTrackingConfig:
        """
        Get the configuration used.
        """
    @property
    def tempo_bpm(self, /) -> float:
        """
        Get the estimated tempo in BPM.
        """

@final
class ChannelReduction:
    """
    Strategy for reducing a multi-channel signal to a single channel.

    `ChannelReduction` controls how operations that fundamentally require a single
    channel (such as spectral centroid or roll-off) behave when presented with a
    multi-channel signal. It lets the caller choose between failing loudly,
    selecting a single channel, or averaging across channels.

    Instances of `ChannelReduction` are immutable and should be treated as
    enum-like values. Zero-parameter strategies are accessed via class attributes,
    while the channel-selection strategy is constructed via a static method.

    Available strategies:

    - ``ChannelReduction.error`` (default):
      Return an error when the signal has more than one channel.

    - ``ChannelReduction.first``:
      Use the first channel (index 0) and ignore the rest.

    - ``ChannelReduction.average``:
      Average the corresponding samples across all channels.

    - ``ChannelReduction.channel(index)``:
      Use the channel at the given (bounds-checked) index.
    """
    average: Final[ChannelReduction]
    """
    Average the corresponding samples across all channels.
    """
    error: Final[ChannelReduction]
    """
    Return an error when the signal has more than one channel.

    This is the default strategy and preserves the strictest behaviour by
    refusing to silently collapse channels.
    """
    first: Final[ChannelReduction]
    """
    Use the first channel (index 0) and ignore the rest.
    """
    @staticmethod
    def channel(index: "int") -> ChannelReduction:
        """
        Use the channel at the given index.

        The index is bounds-checked when the reduction is applied.

        Parameters
        ----------
        index : int
            Zero-based channel index to select.
        """
    @staticmethod
    def default() -> ChannelReduction: ...

@final
class ComplexOnsetConfig:
    """
    Configuration for complex domain onset detection.

    Complex domain onset detection uses both magnitude and phase information
    from the CQT to provide more accurate onset detection than magnitude-only
    methods, especially for polyphonic music and complex timbres.
    """
    musical: Final[ComplexOnsetConfig]
    """
    Musical preset configuration.
    """
    percussive: Final[ComplexOnsetConfig]
    """
    Percussive preset configuration.
    """
    speech: Final[ComplexOnsetConfig]
    """
    Speech preset configuration.
    """
    def __new__(cls, /, *, cqt_params: "CqtParams", hop_size: "int", window_size: "Optional[int]", peak_picking: "PeakPickingConfig", magnitude_weight: "float", phase_weight: "float", magnitude_rectify: "bool", phase_rectify: "bool", log_compression: "float") -> ComplexOnsetConfig:
        """
        Create a new ``ComplexOnsetConfig``.

        Parameters
        ----------
        ``cqt_config`` : ``CqtConfig``
            CQT configuration object.
        ``hop_size`` : int
            Hop size in samples.
        ``window_size`` : Optional[int]
            Window size in samples. If ``None``, defaults to CQT's minimum window size.
        ``peak_picking`` : ``PeakPickingConfig``
            Peak picking configuration.
        ``magnitude_weight`` : float
            Weighting factor for magnitude component.
        ``phase_weight`` : float    
            Weighting factor for phase component.
        ``magnitude_rectify`` : bool
            Whether to rectify the magnitude component.
        ``phase_rectify`` : bool
            Whether to rectify the phase component.
        ``log_compression`` : float
            Logarithmic compression factor.

        Returns
        -------
        ComplexOnsetConfig
            A new complex onset detection configuration.
        """
    def __repr__(self, /) -> str: ...
    @property
    def cqt_params(self, /) -> Any:
        """
        Get the CQT parameters.
        """
    @cqt_params.setter
    def cqt_params(self, /, value: Any) -> None:
        """
        Set the CQT parameters.
        """
    @property
    def hop_size(self, /) -> int:
        """
        Get the hop size in samples.
        """
    @hop_size.setter
    def hop_size(self, /, value: int) -> None:
        """
        Set the hop size in samples.
        """
    @property
    def log_compression(self, /) -> float:
        """
        Get the logarithmic compression factor.
        """
    @log_compression.setter
    def log_compression(self, /, value: float) -> None:
        """
        Set the logarithmic compression factor.
        """
    @property
    def magnitude_rectify(self, /) -> bool:
        """
        Get whether to rectify the magnitude component.
        """
    @magnitude_rectify.setter
    def magnitude_rectify(self, /, value: bool) -> None:
        """
        Set whether to rectify the magnitude component.
        """
    @property
    def magnitude_weight(self, /) -> float:
        """
        Get the magnitude weighting factor.
        """
    @magnitude_weight.setter
    def magnitude_weight(self, /, value: float) -> None:
        """
        Set the magnitude weighting factor.
        """
    @property
    def peak_picking(self, /) -> PeakPickingConfig:
        """
        Get the peak picking configuration.

        Returns a **copy**: setting fields on the returned object does not change
        this config. Modify the copy and assign it back:

        ::

            pp = config.peak_picking
            pp.min_peak_separation = 256
            config.peak_picking = pp
        """
    @peak_picking.setter
    def peak_picking(self, /, value: PeakPickingConfig) -> None:
        """
        Set the peak picking configuration.
        """
    @property
    def phase_rectify(self, /) -> bool:
        """
        Get whether to rectify the phase component.
        """
    @phase_rectify.setter
    def phase_rectify(self, /, value: bool) -> None:
        """
        Set whether to rectify the phase component.
        """
    @property
    def phase_weight(self, /) -> float:
        """
        Get the phase weighting factor.
        """
    @phase_weight.setter
    def phase_weight(self, /, value: float) -> None:
        """
        Set the phase weighting factor.
        """
    def set_weights(self, /, magnitude_weight: "float", phase_weight: "float") -> None:
        """
        Set the magnitude and phase weighting factors.
        """
    def validate(self, /) -> ComplexOnsetConfig:
        """
        Validate the configuration parameters.
        """
    @property
    def window_size(self, /) -> int | None:
        """
        Get the window size in samples, or None if it defaults to CQT's minimum window size.
        """
    @window_size.setter
    def window_size(self, /, value: int | None) -> None:
        """
        Set the window size in samples.
        """

@final
class CompressorConfig:
    """
    Compressor configuration parameters.

    `CompressorConfig` defines how a dynamic range compressor responds to signal
    levels above a threshold, including time constants, compression curve shape,
    detection method, and side-chain behaviour.

    Instances of `CompressorConfig` are immutable value objects in the Python API.
    Parameters are provided at construction time and exposed via read-only
    properties internally. Validation can be performed explicitly using
    ``validate(sample_rate)``.

    Several common presets are provided as class attributes, accessed
    **without** parentheses: ``CompressorConfig.vocal``, ``CompressorConfig.drum``,
    ``CompressorConfig.bus``. (By contrast ``CompressorConfig.default()`` and the
    ``LimiterConfig`` presets such as ``LimiterConfig.mastering()`` are methods.)
    Parameters can be read back through read-only properties such as
    ``threshold_db`` and ``ratio``.
    """
    bus: Final[CompressorConfig]
    """
    Bus compression preset (class attribute: ``CompressorConfig.bus``).

    Tuned for gentle dynamic glue on groups or mix buses.
    """
    drum: Final[CompressorConfig]
    """
    Drum compression preset (class attribute: ``CompressorConfig.drum``).

    Tuned for fast transient control and strong dynamic shaping.
    """
    vocal: Final[CompressorConfig]
    """
    Vocal compression preset (class attribute: ``CompressorConfig.vocal``).

    Tuned for moderate dynamic control and natural sounding speech or vocals.
    """
    def __new__(cls, /, *, threshold_db: "float", ratio: "float", attack_ms: "float", release_ms: "float", makeup_gain_db: "float", knee_type: "KneeType", knee_width_db: "float", detection_method: "DynamicRangeMethod", side_chain: "SideChainConfig", lookahead_ms: "float") -> CompressorConfig:
        """
        Create a new compressor configuration.

        Parameters
        ----------
        ``threshold_db`` : float
            Threshold level in decibels. Signal levels above this value will be
            compressed. Typically negative (e.g. -40 to 0 dB).
        ratio : float
            Compression ratio. ``1.0`` means no compression; larger values apply
            stronger compression.
        ``attack_ms`` : float
            Attack time in milliseconds. Controls how quickly compression engages
            once the signal exceeds the threshold.
        ``release_ms`` : float
            Release time in milliseconds. Controls how quickly compression
            disengages once the signal falls below the threshold.
        ``makeup_gain_db`` : float
            Makeup gain in decibels applied after compression.
        ``knee_type`` : ``KneeType``
            Knee characteristic controlling transition smoothness around the
            threshold.
        ``knee_width_db`` : float
            Width of the knee region in decibels for soft-knee behaviour.
        ``detection_method`` : ``DynamicRangeMethod``
            Signal level detection method driving gain reduction.
        ``side_chain`` : ``SideChainConfig``
            Side-chain configuration.
        ``lookahead_ms`` : float
            Lookahead time in milliseconds. Allows the compressor to anticipate
            upcoming peaks.

        Returns
        -------
        ``CompressorConfig``
            A new compressor configuration.
        """
    def __repr__(self, /) -> str: ...
    @property
    def attack_ms(self, /) -> float:
        """
        Attack time in milliseconds.
        """
    @staticmethod
    def default() -> CompressorConfig: ...
    @property
    def detection_method(self, /) -> DynamicRangeMethod:
        """
        Detection method driving gain reduction.
        """
    def explain(self, /, sample_rate: "float | None" = None, *, n_samples: "int | None" = None, n_channels: "int | None" = None, f0_range: "tuple[float, float] | None" = None) -> ExplainDoc:
        """
        Explain what this compressor configuration will do.

        Args:

            sample_rate (float | None): Sample rate in Hz of the signal it
                will be applied to. Without it, quantities that need a
                sample rate are omitted and an ``info`` warning says so.
            n_samples (int | None): Signal length in samples per channel.
            n_channels (int | None): Number of channels.
            f0_range (tuple[float, float] | None): Expected
                fundamental-frequency range ``(lowest, highest)`` in Hz.

        Returns:

            ExplainDoc: The explanation (see ``audio_samples.explain``).
        """
    @property
    def knee_type(self, /) -> KneeType:
        """
        Knee characteristic.
        """
    @property
    def knee_width_db(self, /) -> float:
        """
        Knee width in decibels.
        """
    @property
    def lookahead_ms(self, /) -> float:
        """
        Lookahead time in milliseconds.
        """
    @property
    def makeup_gain_db(self, /) -> float:
        """
        Makeup gain in decibels applied after compression.
        """
    @property
    def ratio(self, /) -> float:
        """
        Compression ratio (``1.0`` = no compression).
        """
    @property
    def release_ms(self, /) -> float:
        """
        Release time in milliseconds.
        """
    @property
    def side_chain(self, /) -> SideChainConfig:
        """
        Side-chain configuration.
        """
    @property
    def threshold_db(self, /) -> float:
        """
        Threshold level in decibels.
        """
    def validate(self, /, sample_rate: "float") -> CompressorConfig:
        """
        Validate compressor configuration parameters.

        Parameters
        ----------
        ``sample_rate`` : float
            Audio sample rate in Hz, used to validate time- and frequency-dependent
            constraints.

        Returns
        -------
        ``CompressorConfig``
           The validated compressor configuration.

        Raises
        ------
        ``ValueError``
            If any configuration parameter is invalid.
        """

@final
class DynamicRangeMethod:
    """
    Detection method for dynamic range processing.

    `DynamicRangeMethod` selects how signal level is estimated when driving gain
    reduction in dynamics processors such as compressors, limiters, and gates.

    Different methods trade off smoothness, transient responsiveness, and
    perceptual stability.

    Instances of `DynamicRangeMethod` are immutable and should be treated as
    enum-like values. They are accessed via class attributes rather than being
    constructed directly.

    Available detection methods:

    - ``DynamicRangeMethod.rms``
    - ``DynamicRangeMethod.peak``
    - ``DynamicRangeMethod.hybrid``
    """
    hybrid: Final[DynamicRangeMethod]
    """
    Hybrid level detection.

    Combines RMS and peak estimation to balance smoothness and transient
    control.
    """
    peak: Final[DynamicRangeMethod]
    """
    Peak-based level detection.

    Responds to instantaneous signal peaks, providing tight peak control with
    increased sensitivity to transients.
    """
    rms: Final[DynamicRangeMethod]
    """
    RMS-based level detection.

    Estimates average signal power over time, producing smoother and more
    perceptually stable gain control.
    """
    @staticmethod
    def default() -> DynamicRangeMethod: ...

@final
class EnvelopeFollower:
    """
    Python wrapper for ``EnvelopeFollower``.

    Envelope follower for attack/release envelope tracking.
    """
    def __new__(cls, /, attack_ms: float, release_ms: float, sample_rate: float, detection_method: DynamicRangeMethod) -> EnvelopeFollower:
        """
        Create a new ``EnvelopeFollower``.

        Parameters
        ----------
        ``attack_ms`` : float
            Attack time in milliseconds
        ``release_ms`` : float
            Release time in milliseconds
        ``sample_rate`` : float
            Sample rate in Hz
        ``detection_method`` : ``DynamicRangeMethod``
            Detection method (Peak, Rms, or Hybrid)
        """
    def __repr__(self, /) -> str: ...
    @classmethod
    def default(cls, /) -> EnvelopeFollower:
        """
        Create a default ``EnvelopeFollower`` with 10ms attack and 100ms release at 44100 Hz.
        """
    def process(self, /, input: float) -> float:
        """
        Process one sample and return the current envelope level (>= 0).

        The follower is stateful: successive calls continue from the previous
        envelope value. Call :meth:`reset` to start again from 0.

        Parameters
        ----------
        input : float
            Input sample value (any numeric scale).

        Returns
        -------
        float
            Smoothed envelope level.
        """
    def process_array(self, /, samples: "NDArray[np.float64]") -> "NDArray[np.float64]":
        """
        Process a 1-D array sample by sample and return the envelope per sample.

        Equivalent to ``np.array([follower.process(v) for v in samples])``; the
        state carries over between calls.

        Parameters
        ----------
        samples : numpy.ndarray
            1-D float64 array of input samples.

        Returns
        -------
        numpy.ndarray
            Envelope level for each input sample.
        """
    def reset(self, /) -> None:
        """
        Reset the envelope to 0 (and clear the RMS window).
        """

@final
class EqBand:
    """
    Parametric equaliser band configuration.

    `EqBand` represents a single band in a parametric equaliser, defined by a band
    type, centre or cutoff frequency, gain, and quality factor (Q). Each band
    describes how a specific region of the frequency spectrum is shaped.

    Instances of `EqBand` are immutable value objects. Parameters are provided at
    construction time and exposed via read-only properties.

    Convenience constructors are available for common band types such as peak
    and shelving filters.
    """
    def __new__(cls, /, band_type: "EqBandType", frequency: "float", gain_db: "float", q_factor: "float") -> EqBand:
        """
        Create a new EQ band with explicit parameters.

        Parameters
        ----------
        ``band_type`` : ``EqBandType``
            Type of EQ band (peak, shelf, pass, etc.).
        frequency : float
            Centre frequency in Hz (for peak / notch) or corner frequency (for
            shelving and pass filters).
        ``gain_db`` : float
            Gain in decibels. Positive values boost, negative values attenuate.
            For purely filtering bands (e.g. low-pass), this is typically zero.
        ``q_factor`` : float
            Quality factor controlling bandwidth or slope. Higher values produce
            narrower bandwidths or steeper transitions.

        Returns
        -------
        EqBand
            A new EQ band instance.

        Notes
        -----
        Parameter validity is not automatically checked at construction time.
        Invalid configurations may fail later during processing.
        """
    def __repr__(self, /) -> str: ...
    @property
    def enabled(self, /) -> bool:
        """
        Whether this band is enabled.
        """
    def explain(self, /, sample_rate: "float | None" = None, *, n_samples: "int | None" = None, n_channels: "int | None" = None, f0_range: "tuple[float, float] | None" = None) -> ExplainDoc:
        """
        Explain what this EQ band will do.

        Args:

            sample_rate (float | None): Sample rate in Hz of the signal it
                will be applied to. Without it, quantities that need a
                sample rate are omitted and an ``info`` warning says so.
            n_samples (int | None): Signal length in samples per channel.
            n_channels (int | None): Number of channels.
            f0_range (tuple[float, float] | None): Expected
                fundamental-frequency range ``(lowest, highest)`` in Hz.

        Returns:

            ExplainDoc: The explanation (see ``audio_samples.explain``).
        """
    @property
    def frequency(self, /) -> float:
        """
        Centre or cutoff frequency in Hz.
        """
    @property
    def gain_db(self, /) -> float:
        """
        Gain in decibels.
        """
    @classmethod
    def high_shelf(cls, /, frequency: "float", gain_db: "float", q_factor: "float") -> EqBand:
        """
        Create a high-shelf EQ band.

        Parameters
        ----------
        frequency : float
            Corner frequency in Hz.
        ``gain_db`` : float
            Gain in decibels.
        ``q_factor`` : float
            Shelf slope control.
        """
    @classmethod
    def low_shelf(cls, /, frequency: "float", gain_db: "float", q_factor: "float") -> EqBand:
        """
        Create a low-shelf EQ band.

        Parameters
        ----------
        frequency : float
            Corner frequency in Hz.
        ``gain_db`` : float
            Gain in decibels.
        ``q_factor`` : float
            Shelf slope control.
        """
    @classmethod
    def peak(cls, /, frequency: "float", gain_db: "float", q_factor: "float") -> EqBand:
        """
        Create a peaking (bell) EQ band.

        Parameters
        ----------
        frequency : float
            Centre frequency in Hz.
        ``gain_db`` : float
            Gain in decibels. Positive values boost, negative values attenuate.
        ``q_factor`` : float
            Quality factor controlling bandwidth.
        """
    @property
    def q_factor(self, /) -> float:
        """
        Quality factor controlling bandwidth or slope.
        """

@final
class EqBandType:
    """
    Parametric equaliser band type.

    `EqBandType` defines how gain is applied across the frequency spectrum
    relative to a centre or cutoff frequency. Different band types emphasise or
    suppress different spectral regions.

    Instances of `EqBandType` are immutable and should be treated as enum-like
    values. They are accessed via class attributes rather than being constructed
    directly.

    Available band types:

    - ``EqBandType.peak``
    - ``EqBandType.low_shelf``
    - ``EqBandType.high_shelf``
    - ``EqBandType.low_pass``
    - ``EqBandType.high_pass``
    - ``EqBandType.band_pass``
    - ``EqBandType.band_stop``
    """
    band_pass: Final[EqBandType]
    """
    Band-pass filter.

    Preserves frequencies within a specified band while attenuating
    frequencies outside that range.
    """
    band_stop: Final[EqBandType]
    """
    Band-stop (notch) filter.

    Attenuates frequencies within a specified band while preserving
    frequencies outside that range.
    """
    high_pass: Final[EqBandType]
    """
    High-pass filter.

    Attenuates frequencies below the cutoff frequency.
    """
    high_shelf: Final[EqBandType]
    """
    High-shelf filter.

    Applies a broadband boost or cut to frequencies above the corner
    frequency.
    """
    low_pass: Final[EqBandType]
    """
    Low-pass filter.

    Attenuates frequencies above the cutoff frequency.
    """
    low_shelf: Final[EqBandType]
    """
    Low-shelf filter.

    Applies a broadband boost or cut to frequencies below the corner
    frequency.
    """
    peak: Final[EqBandType]
    """
    Peaking (bell) filter.

    Boosts or attenuates a narrow frequency region centred at the target
    frequency. Positive gain produces a peak; negative gain produces a notch.
    """
    @staticmethod
    def default() -> EqBandType: ...

@final
class ExpanderConfig:
    """
    Downward expander configuration parameters.

    `ExpanderConfig` defines how a downward expander attenuates a signal that
    falls below a threshold, increasing the dynamic range of low-level material.
    RMS detection is always used for expansion.

    Instances of `ExpanderConfig` are immutable value objects in the Python API.
    Parameters are provided at construction time and exposed via read-only
    properties. Validation can be performed explicitly using ``validate()``.

    A gentle expansion preset is provided via ``ExpanderConfig.gentle``.
    """
    gentle: Final[ExpanderConfig]
    """
    Gentle downward expansion preset.

    A low ratio and relaxed envelope times for subtle dynamic-range
    enhancement without obvious pumping.
    """
    def __new__(cls, /, *, threshold_db: "float", ratio: "float", attack_ms: "float", release_ms: "float") -> ExpanderConfig:
        """
        Create a new expander configuration.

        Parameters
        ----------
        ``threshold_db`` : float
            Expansion threshold in dBFS. Signals below this level are attenuated.
            Typical range: [-80.0, 0.0].
        ratio : float
            Expansion ratio applied below the threshold. Values greater than 1.0
            produce increasing attenuation the further the signal falls below the
            threshold. Must be greater than 0.0.
        ``attack_ms`` : float
            Attack time in milliseconds. Valid range: [0.01, 1000.0] ms.
        ``release_ms`` : float
            Release time in milliseconds. Valid range: [1.0, 10000.0] ms.

        Returns
        -------
        ``ExpanderConfig``
            A new expander configuration.
        """
    def __repr__(self, /) -> str: ...
    @property
    def attack_ms(self, /) -> float:
        """
        Attack time in milliseconds.
        """
    @staticmethod
    def default() -> ExpanderConfig: ...
    @property
    def ratio(self, /) -> float:
        """
        Expansion ratio applied below the threshold.
        """
    @property
    def release_ms(self, /) -> float:
        """
        Release time in milliseconds.
        """
    @property
    def threshold_db(self, /) -> float:
        """
        Expansion threshold in dBFS.
        """
    def validate(self, /) -> ExpanderConfig:
        """
        Validate expander configuration parameters.

        Returns
        -------
        ``ExpanderConfig``
            The validated expander configuration.

        Raises
        ------
        ``ValueError``
            If any configuration parameter is invalid.
        """

@final
class FadeCurve:
    """
    Fade curve shape for envelope operations.

    `FadeCurve` represents the shape of an amplitude envelope used when applying
    fades, ramps, or transitions to audio signals. Different curves produce
    different perceptual characteristics in how the signal level changes over
    time.

    Instances of `FadeCurve` are immutable and should be treated as enum-like
    values. They are accessed via class attributes rather than being constructed
    directly.

    Available fade curves:

    - ``FadeCurve.linear``
    - ``FadeCurve.exponential``
    - ``FadeCurve.logarithmic``
    - ``FadeCurve.smooth_step``
    """
    exponential: Final[FadeCurve]
    """
    Exponential fade.

    Faster change at the beginning, slower towards the end.
    """
    linear: Final[FadeCurve]
    """
    Linear fade.

    Constant rate of change over time.
    """
    logarithmic: Final[FadeCurve]
    """
    Logarithmic fade.

    Slower change at the beginning, faster towards the end.
    """
    smooth_step: Final[FadeCurve]
    """
    Smooth-step fade.

    S-shaped curve with smooth transitions at both the start and end.
    """
    @staticmethod
    def default() -> FadeCurve: ...

@final
class FilterResponse:
    """
    Filter response characteristic.

    `FilterResponse` defines the qualitative frequency response shape of a filter,
    such as whether it attenuates low frequencies, high frequencies, or a band of
    frequencies.

    Instances of `FilterResponse` are immutable and should be treated as
    enum-like values. They are accessed via class attributes rather than being
    constructed directly.

    Available responses:

    - ``FilterResponse.lowpass``
    - ``FilterResponse.highpass``
    - ``FilterResponse.bandpass``
    - ``FilterResponse.bandstop``
    """
    bandpass: Final[FilterResponse]
    """
    Band-pass filter response.

    Preserves frequencies within a specified band while attenuating
    frequencies outside that range.
    """
    bandstop: Final[FilterResponse]
    """
    Band-stop (notch) filter response.

    Attenuates frequencies within a specified band while preserving
    frequencies outside that range.
    """
    highpass: Final[FilterResponse]
    """
    High-pass filter response.

    Attenuates frequencies below the cutoff frequency while preserving
    higher-frequency components.
    """
    lowpass: Final[FilterResponse]
    """
    Low-pass filter response.

    Attenuates frequencies above the cutoff frequency while preserving
    lower-frequency components.
    """
    @staticmethod
    def default() -> FilterResponse: ...

@final
class GateConfig:
    """
    Noise gate configuration parameters.

    `GateConfig` defines how a downward noise gate attenuates a signal that falls
    below a threshold, including the attenuation ratio and the attack and release
    envelope time constants.

    Instances of `GateConfig` are immutable value objects in the Python API.
    Parameters are provided at construction time and exposed via read-only
    properties. Validation can be performed explicitly using ``validate()``.

    A general-purpose noise-gating preset is provided via ``GateConfig.noise_gate``.
    """
    noise_gate: Final[GateConfig]
    """
    General-purpose noise-gating preset.

    Moderate threshold and a high ratio for firmly attenuating background
    noise and room tone between phrases.
    """
    def __new__(cls, /, *, threshold_db: "float", ratio: "float", attack_ms: "float", release_ms: "float") -> GateConfig:
        """
        Create a new gate configuration.

        Parameters
        ----------
        ``threshold_db`` : float
            Gate threshold in dBFS. Signals below this level are attenuated.
            Typical range: [-80.0, 0.0].
        ratio : float
            Attenuation ratio applied below the threshold. Higher values produce
            more aggressive gating; values near 1.0 approach unity gain. Must be
            greater than 0.0.
        ``attack_ms`` : float
            Attack time in milliseconds. Controls how quickly the gate opens once
            the signal rises above the threshold. Valid range: [0.01, 1000.0] ms.
        ``release_ms`` : float
            Release time in milliseconds. Controls how quickly the gate closes once
            the signal falls below the threshold. Valid range: [1.0, 10000.0] ms.

        Returns
        -------
        ``GateConfig``
            A new gate configuration.
        """
    def __repr__(self, /) -> str: ...
    @property
    def attack_ms(self, /) -> float:
        """
        Attack time in milliseconds.
        """
    @staticmethod
    def default() -> GateConfig: ...
    @property
    def ratio(self, /) -> float:
        """
        Attenuation ratio applied below the threshold.
        """
    @property
    def release_ms(self, /) -> float:
        """
        Release time in milliseconds.
        """
    @property
    def threshold_db(self, /) -> float:
        """
        Gate threshold in dBFS.
        """
    def validate(self, /) -> GateConfig:
        """
        Validate gate configuration parameters.

        Returns
        -------
        ``GateConfig``
            The validated gate configuration.

        Raises
        ------
        ``ValueError``
            If any configuration parameter is invalid.
        """

@final
class HpssConfig:
    """
    Configuration for Harmonic/Percussive Source Separation (HPSS).

    HPSS separates audio into harmonic and percussive components using
    STFT magnitude median filtering. Harmonic components are enhanced
    by median filtering along the time axis, while percussive components
    are enhanced by median filtering along the frequency axis.
    """
    harmonic: Final[HpssConfig]
    """
    Create configuration optimized for harmonic content.

    Uses asymmetric filters favoring harmonic separation:
    - Strong harmonic filtering
    - Moderate percussive filtering
    - Harder masking for cleaner tonal isolation
    """
    musical: Final[HpssConfig]
    """
    Create configuration optimized for musical content.

    Uses larger filters for stronger separation, suitable for complex musical material:
    - Larger harmonic kernel for better tonal separation
    - Larger percussive kernel for cleaner transient isolation
    - Softer masking for more musical results
    """
    percussive: Final[HpssConfig]
    """
    Create configuration optimized for percussive content.

    Uses asymmetric filters favoring percussive separation:
    - Moderate harmonic filtering
    - Strong percussive filtering
    - Harder masking for cleaner drum isolation
    """
    realtime: Final[HpssConfig]
    """
    Create configuration for real-time processing.

    Uses smaller window and filter sizes for lower latency:
    - Smaller window for reduced latency
    - Smaller hop size for responsiveness
    - Smaller filters for faster processing
    """
    def __new__(cls, /, *, stft_params: "Optional[StftParams]" = None, median_filter_harmonic: "int" = 17, median_filter_percussive: "int" = 17, mask_softness: "float" = 0.3) -> HpssConfig:
        """
        Create a new HPSS configuration with default settings.

        Default configuration suitable for general harmonic/percussive separation:
        - 2048 sample window (good frequency resolution)
        - 512 sample hop size (good time resolution)
        - Harmonic kernel: 17 (enhances sustained tones)
        - Percussive kernel: 17 (enhances transients)
        - Moderate soft masking (0.3)
        """
    def __repr__(self, /) -> str: ...
    def freq_resolution(self, /, sample_rate: "float") -> float:
        """
        Calculate the frequency resolution in Hz.
        """
    def num_freq_bins(self, /) -> int:
        """
        Calculate the number of frequency bins for this configuration.
        """
    def set_filter_sizes(self, /, harmonic_size: "int", percussive_size: "int") -> None:
        """
        Set median filter sizes.

        Parameters
        ----------
        harmonic: int
            Harmonic filter size (odd numbers recommended)
        percussive: int
            Percussive filter size (odd numbers recommended)
        """
    def set_mask_softness(self, /, softness: "float") -> None:
        """
        Set mask softness parameter.

        Parameters
        ----------
        softness: float
            Softness value (0.0 = hard mask, 1.0 = completely soft)
        """
    def set_stft_params(self, /, n_fft: "int", hop_size: "int") -> None:
        """
        Set STFT parameters.

        Parameters
        ----------
        ``n_fft``: int
            FFT size in samples (should be a power of 2)
        ``hop_size``: int
            Hop size in samples, at most ``n_fft`` (typically ``n_fft/4``)

        Raises
        ------
        ``ValueError``
            If either value is 0 or ``hop_size`` is larger than ``n_fft``. The
            configuration is then unchanged.
        """
    def time_resolution(self, /, sample_rate: "float") -> float:
        """
        Calculate the time resolution in seconds.
        """

@final
class IirFilterDesign:
    """
    IIR filter design parameters.

    `IirFilterDesign` describes the parameters required to construct a digital
    infinite impulse response (IIR) filter, including the filter family, response
    type, order, and frequency specifications.

    Instances of `IirFilterDesign` are immutable value objects. Fields are exposed
    via read-only properties.

    A generic constructor is provided for manual configuration, and convenience
    constructors are available for common Butterworth designs.
    """
    def __new__(cls, /, filter_type: "FilterType", response: "FilterResponse | str", order: "int", cutoff_frequency: float | None = None, low_frequency: float | None = None, high_frequency: float | None = None, passband_ripple: float | None = None, stopband_attenuation: float | None = None, bessel_norm: str | None = None) -> IirFilterDesign:
        """
        Create a new IIR filter design.

        This constructor provides a generic configuration interface. Depending on
        the selected filter type and response, only a subset of parameters may be
        meaningful.

        Parameters
        ----------
        ``filter_type`` : ``IirFilterType``
            IIR filter family (e.g. Butterworth, Chebyshev).
        response : ``FilterResponse``
            Desired frequency response shape (low-pass, high-pass, band-pass, etc.).
        order : int
            Filter order (number of poles).
        ``cutoff_frequency`` : float, optional
            Cutoff frequency in Hz for low-pass and high-pass filters.
        ``low_frequency`` : float, optional
            Lower cutoff frequency in Hz for band-pass and band-stop filters.
        ``high_frequency`` : float, optional
            Upper cutoff frequency in Hz for band-pass and band-stop filters.
        ``passband_ripple`` : float, optional
            Passband ripple in dB (Chebyshev I, elliptic).
        ``stopband_attenuation`` : float, optional
            Stopband attenuation in dB (Chebyshev II, elliptic).
        ``bessel_norm`` : str, optional
            Bessel normalisation: ``"mag"`` (default, -3 dB at the cutoff),
            ``"phase"`` (phase response at its midpoint at the cutoff) or
            ``"delay"`` (passband group delay set by the cutoff).

        Returns
        -------
        ``IirFilterDesign``
            A new filter design description.

        Notes
        -----
        Parameter consistency is not validated automatically. Invalid or
        incompatible combinations may fail later during filter construction.
        """
    def __repr__(self, /) -> str: ...
    @classmethod
    def bessel(cls, /, response: "FilterResponse | str", order: "int", cutoff_frequency: "float", norm: "str" = "mag") -> IirFilterDesign:
        """
        Create a Bessel low-pass / high-pass design.

        ``norm`` (``"mag"``, ``"phase"`` or ``"delay"``) defaults to ``"mag"``:
        -3 dB at the cutoff.
        """
    @classmethod
    def bessel_band(cls, /, response: "FilterResponse | str", order: "int", low_frequency: "float", high_frequency: "float", norm: "str" = "mag") -> IirFilterDesign:
        """
        Create a Bessel band-pass / band-stop design; ``norm`` as for ``bessel``.
        """
    @property
    def bessel_norm(self, /) -> str:
        """
        Bessel normalisation (``"mag"``, ``"phase"`` or ``"delay"``).
        """
    @classmethod
    def butterworth_bandpass(cls, /, order: "int", low_frequency: "float", high_frequency: "float") -> IirFilterDesign:
        """
        Create a Butterworth band-pass filter design.

        Parameters
        ----------
        order : int
            Filter order.
        ``low_frequency`` : float
            Lower cutoff frequency in Hz.
        ``high_frequency`` : float
            Upper cutoff frequency in Hz.
        """
    @classmethod
    def butterworth_bandstop(cls, /, order: "int", low_frequency: "float", high_frequency: "float") -> IirFilterDesign:
        """
        Create a Butterworth band-stop filter design between ``low_frequency``
        and ``high_frequency`` Hz.
        """
    @classmethod
    def butterworth_highpass(cls, /, order: "int", cutoff_frequency: "float") -> IirFilterDesign:
        """
        Create a Butterworth high-pass filter design.

        Parameters
        ----------
        order : int
            Filter order.
        ``cutoff_frequency`` : float
            Cutoff frequency in Hz.
        """
    @classmethod
    def butterworth_lowpass(cls, /, order: "int", cutoff_frequency: "float") -> IirFilterDesign:
        """
        Create a Butterworth low-pass filter design.

        Parameters
        ----------
        order : int
            Filter order.
        ``cutoff_frequency`` : float
            Cutoff frequency in Hz.

        Returns
        -------

        ``IirFilterDesign``
            A new Butterworth low-pass filter design instance.

        Errors
        ------

        If the order argument is less than or equal to zero
        """
    @classmethod
    def chebyshev_i(cls, /, response: "FilterResponse | str", order: "int", cutoff_frequency: "float", passband_ripple: "float") -> IirFilterDesign:
        """
        Create a Chebyshev Type I low-pass / high-pass design.

        The passband ripples between 0 and ``-passband_ripple`` dB.
        """
    @classmethod
    def chebyshev_i_band(cls, /, response: "FilterResponse | str", order: "int", low_frequency: "float", high_frequency: "float", passband_ripple: "float") -> IirFilterDesign:
        """
        Create a Chebyshev Type I band-pass / band-stop design.

        The passband ripples between 0 and ``-passband_ripple`` dB.
        """
    @classmethod
    def chebyshev_ii(cls, /, response: "FilterResponse | str", order: "int", cutoff_frequency: "float", stopband_attenuation: "float") -> IirFilterDesign:
        """
        Create a Chebyshev Type II low-pass / high-pass design.

        The stopband is at least ``stopband_attenuation`` dB down;
        ``cutoff_frequency`` is the stopband edge.
        """
    @classmethod
    def chebyshev_ii_band(cls, /, response: "FilterResponse | str", order: "int", low_frequency: "float", high_frequency: "float", stopband_attenuation: "float") -> IirFilterDesign:
        """
        Create a Chebyshev Type II band-pass / band-stop design.

        The stopband is at least ``stopband_attenuation`` dB down; the band
        edges are the stopband edges.
        """
    @property
    def cutoff_frequency(self, /) -> float | None:
        """
        Cutoff frequency in Hz for low-pass and high-pass filters.
        """
    @classmethod
    def elliptic(cls, /, response: "FilterResponse | str", order: "int", cutoff_frequency: "float", passband_ripple: "float", stopband_attenuation: "float") -> IirFilterDesign:
        """
        Create an elliptic (Cauer) low-pass / high-pass design.

        ``passband_ripple`` dB of passband ripple and at least
        ``stopband_attenuation`` dB of stopband attenuation.
        """
    @classmethod
    def elliptic_band(cls, /, response: "FilterResponse | str", order: "int", low_frequency: "float", high_frequency: "float", passband_ripple: "float", stopband_attenuation: "float") -> IirFilterDesign:
        """
        Create an elliptic (Cauer) band-pass / band-stop design.

        ``passband_ripple`` dB of passband ripple and at least
        ``stopband_attenuation`` dB of stopband attenuation.
        """
    def explain(self, /, sample_rate: "float | None" = None, *, n_samples: "int | None" = None, n_channels: "int | None" = None, f0_range: "tuple[float, float] | None" = None) -> ExplainDoc:
        """
        Explain what this IIR filter design will do.

        Args:

            sample_rate (float | None): Sample rate in Hz of the signal it
                will be applied to. Without it, quantities that need a
                sample rate are omitted and an ``info`` warning says so.
            n_samples (int | None): Signal length in samples per channel.
            n_channels (int | None): Number of channels.
            f0_range (tuple[float, float] | None): Expected
                fundamental-frequency range ``(lowest, highest)`` in Hz.

        Returns:

            ExplainDoc: The explanation (see ``audio_samples.explain``).
        """
    @property
    def filter_type(self, /) -> IirFilterType:
        """
        Filter family used for the design.
        """
    def frequency_response(self, /, frequencies: "Sequence[float] | NDArray[np.floating]", sample_rate: "float") -> tuple[Incomplete, Incomplete]:
        """
        Evaluate the frequency response of the design.

        Args:

            frequencies (np.typing.NDArray[np.float64]): Frequencies in hertz.
            sample_rate (float): Sample rate in hertz.

        Returns:

            tuple[np.typing.NDArray[np.float64], np.typing.NDArray[np.float64]]:
            Linear magnitude and phase (radians, wrapped to (-pi, pi]) of the
            cascaded sections at ``frequencies``.

        Raises:

            ValueError: If the design is invalid for ``sample_rate``.
        """
    @property
    def high_frequency(self, /) -> float | None:
        """
        Upper cutoff frequency in Hz for band-pass and band-stop filters.
        """
    @property
    def low_frequency(self, /) -> float | None:
        """
        Lower cutoff frequency in Hz for band-pass and band-stop filters.
        """
    @property
    def order(self, /) -> int:
        """
        Filter order (number of poles).
        """
    @property
    def passband_ripple(self, /) -> float | None:
        """
        Passband ripple in dB, if applicable.
        """
    @property
    def response(self, /) -> FilterResponse:
        """
        Frequency response shape of the filter.
        """
    def sos(self, /, sample_rate: "float") -> "NDArray[np.float64]":
        """
        Design the filter and return its second-order sections.

        Returns a ``(n_sections, 6)`` float64 array with rows
        ``[b0, b1, b2, a0, a1, a2]`` and ``a0 == 1``, usable directly with
        ``scipy.signal.sosfilt``.

        Args:

            sample_rate (float): Sample rate in hertz.

        Raises:

            ValueError: If the design is invalid for ``sample_rate``.
        """
    @property
    def stopband_attenuation(self, /) -> float | None:
        """
        Stopband attenuation in dB, if applicable.
        """

@final
class IirFilterType:
    """
    IIR filter family selection for digital signal processing.

    `IirFilterType` selects the analogue prototype used when designing an
    infinite impulse response (IIR) digital filter. Different families trade off
    passband ripple, stopband attenuation, transition sharpness, and phase
    behaviour.

    Instances of `IirFilterType` are immutable and should be treated as enum-like
    values. They are accessed via class attributes rather than being constructed
    directly.

    Available filter types:

    - ``IirFilterType.butterworth``
    - ``IirFilterType.chebyshev_type_i``
    - ``IirFilterType.chebyshev_type_ii``
    - ``IirFilterType.elliptic``
    - ``IirFilterType.bessel``
    """
    bessel: Final[IirFilterType]
    """
    Bessel (Bessel--Thomson) filter.

    Exhibits a maximally flat group delay (approximately linear phase) in the
    passband, preserving wave shape at the cost of a gentle, monotonic
    magnitude roll-off.
    """
    butterworth: Final[IirFilterType]
    """
    Butterworth filter.

    Maximally flat passband with monotonic magnitude response and no ripple.
    Provides smooth behaviour and predictable phase characteristics at the
    cost of wider transition bands.
    """
    chebyshev_type_i: Final[IirFilterType]
    """
    Chebyshev Type I filter.

    Introduces controlled ripple in the passband to achieve a sharper
    transition region than Butterworth designs.
    """
    chebyshev_type_ii: Final[IirFilterType]
    """
    Chebyshev Type II filter.

    Introduces ripple in the stopband while preserving a monotonic passband
    response, allowing sharper transitions than Butterworth designs.
    """
    elliptic: Final[IirFilterType]
    """
    Elliptic (Cauer) filter.

    Introduces ripple in both passband and stopband, yielding the steepest
    transition region for a given filter order.
    """
    @staticmethod
    def default() -> IirFilterType: ...

@final
class Key:
    """
    A musical key estimate.

    `Key` pairs a tonic pitch class with a mode (major or minor) and a confidence
    score describing how strongly the estimate matched the analysed signal.

    Instances are typically produced by key estimation, but can also be
    constructed directly. Fields are exposed via read-only properties.
    """
    def __new__(cls, /, tonic: "PitchClass", mode: "Mode", confidence: "float") -> Key:
        """
        Create a new key estimate.

        Parameters
        ----------
        tonic : ``PitchClass``
            The tonic pitch class of the key.
        mode : Mode
            The mode (major or minor) of the key.
        confidence : float
            Match confidence in [0.0, 1.0]; higher values indicate a stronger
            match.

        Returns
        -------
        Key
            A new key estimate.
        """
    def __repr__(self, /) -> str:
        """
        Human-readable representation including the confidence score.
        """
    def __str__(self, /) -> str:
        """
        Human-readable key name (e.g. ``"C# minor"``).
        """
    @property
    def confidence(self, /) -> float:
        """
        Match confidence in [0.0, 1.0].
        """
    @property
    def mode(self, /) -> Mode:
        """
        The mode (major or minor) of the key.
        """
    @property
    def tonic(self, /) -> PitchClass:
        """
        The tonic pitch class of the key.
        """

@final
class KneeType:
    """
    Knee characteristic for dynamic range processing.

    `KneeType` controls how smoothly gain reduction transitions as the signal
    crosses the threshold in dynamics processors such as compressors and
    limiters.

    Instances of `KneeType` are immutable and should be treated as enum-like
    values. They are accessed via class attributes rather than being constructed
    directly.

    Available knee types:

    - ``KneeType.hard``
    - ``KneeType.soft``
    """
    hard: Final[KneeType]
    """
    Hard knee.

    Applies an abrupt transition at the threshold, yielding precise dynamics
    control at the potential cost of audible artefacts.
    """
    soft: Final[KneeType]
    """
    Soft knee.

    Applies a gradual transition around the threshold, producing smoother
    and more perceptually natural behaviour.
    """
    @staticmethod
    def default() -> KneeType: ...

@final
class LimiterConfig:
    """
    Limiter configuration parameters.

    `LimiterConfig` defines how a limiter prevents signal levels from exceeding a
    specified ceiling, including time constants, knee behaviour, detection
    method, side-chain configuration, and inter-sample peak (ISP) limiting.

    Instances of `LimiterConfig` are immutable value objects in the Python API.
    Parameters are provided at construction time and exposed via read-only
    properties internally. Validation can be performed explicitly using
    ``validate(sample_rate)``.

    Several common presets are provided as class methods
    (``LimiterConfig.mastering()``), unlike ``CompressorConfig``'s presets,
    which are class attributes.

    ``isp_limiting`` is a reserved flag: inter-sample (true-peak) limiting is
    not implemented and the flag has no effect.
    """
    def __new__(cls, /, *, ceiling_db: "float", attack_ms: "float", release_ms: "float", knee_type: KneeType, knee_width_db: float, detection_method: DynamicRangeMethod, side_chain: SideChainConfig, lookahead_ms: float, isp_limiting: bool) -> LimiterConfig:
        """
        Create a new limiter configuration.

        Parameters
        ----------
        ``ceiling_db`` : float
            Maximum allowed output level in decibels. Typically negative
            (e.g. -0.1 to -3.0 dB).
        ``attack_ms`` : float
            Attack time in milliseconds. Controls how quickly limiting engages as
            the signal approaches the ceiling.
        ``release_ms`` : float
            Release time in milliseconds. Controls how quickly limiting disengages
            once the signal falls below the ceiling.
        ``knee_type`` : ``KneeType``
            Knee characteristic controlling transition smoothness around the
            ceiling.
        ``knee_width_db`` : float
            Width of the knee region in decibels for soft-knee behaviour.
        ``detection_method`` : ``DynamicRangeMethod``
            Signal level detection method driving limiting.
        ``side_chain`` : ``SideChainConfig``
            Side-chain configuration.
        ``lookahead_ms`` : float
            Lookahead time in milliseconds. Allows the limiter to anticipate
            upcoming peaks.
        ``isp_limiting`` : bool
            Reserved flag for inter-sample peak (ISP) limiting. Not implemented:
            it has no effect (sample peaks only).

        Returns
        -------
        ``LimiterConfig``
            A new limiter configuration.
        """
    def __repr__(self, /) -> str: ...
    @property
    def attack_ms(self, /) -> float:
        """
        Attack time in milliseconds.
        """
    @classmethod
    def broadcast(cls, /) -> LimiterConfig:
        """
        Broadcast limiter preset.

        Tuned for aggressive peak control and regulatory compliance.
        """
    @property
    def ceiling_db(self, /) -> float:
        """
        Ceiling level in decibels.
        """
    @staticmethod
    def default() -> LimiterConfig: ...
    @property
    def detection_method(self, /) -> DynamicRangeMethod:
        """
        Detection method used for limiting.
        """
    def explain(self, /, sample_rate: "float | None" = None, *, n_samples: "int | None" = None, n_channels: "int | None" = None, f0_range: "tuple[float, float] | None" = None) -> ExplainDoc:
        """
        Explain what this limiter configuration will do.

        Args:

            sample_rate (float | None): Sample rate in Hz of the signal it
                will be applied to. Without it, quantities that need a
                sample rate are omitted and an ``info`` warning says so.
            n_samples (int | None): Signal length in samples per channel.
            n_channels (int | None): Number of channels.
            f0_range (tuple[float, float] | None): Expected
                fundamental-frequency range ``(lowest, highest)`` in Hz.

        Returns:

            ExplainDoc: The explanation (see ``audio_samples.explain``).
        """
    @property
    def isp_limiting(self, /) -> bool:
        """
        Whether inter-sample peak limiting was requested.

        Reserved flag: inter-sample (true-peak) limiting is not implemented and
        this flag has no effect; the limiter measures sample peaks.
        """
    @property
    def knee_type(self, /) -> KneeType:
        """
        Knee characteristic.
        """
    @property
    def knee_width_db(self, /) -> float:
        """
        Knee width in decibels.
        """
    @property
    def lookahead_ms(self, /) -> float:
        """
        Lookahead time in milliseconds.
        """
    @classmethod
    def mastering(cls, /) -> LimiterConfig:
        """
        Mastering limiter preset.

        Tuned for loudness maximisation with controlled transparency.
        """
    @property
    def release_ms(self, /) -> float:
        """
        Release time in milliseconds.
        """
    @property
    def side_chain(self, /) -> SideChainConfig:
        """
        Side-chain configuration.
        """
    @classmethod
    def transparent(cls, /) -> LimiterConfig:
        """
        Transparent limiter preset.

        Tuned for minimal audible impact while preventing clipping.
        """
    def validate(self, /, sample_rate: "float") -> LimiterConfig:
        """
        Validate limiter configuration parameters.

        Parameters
        ----------
        ``sample_rate`` : float
            Audio sample rate in Hz, used to validate time- and frequency-dependent
            constraints.

        Returns
        -------
        ``LimiterConfig``
          The validated limiter configuration.

        Raises
        ------
        ``ValueError``
            If any configuration parameter is invalid.
        """

@final
class Mode:
    """
    The mode (tonality) of an estimated musical key.

    `Mode` distinguishes between the major and minor tonalities of a key estimate.

    Instances of `Mode` are immutable and should be treated as enum-like values.
    They are accessed via class attributes rather than being constructed directly.

    Available modes:

    - ``Mode.major``
    - ``Mode.minor``
    """
    major: Final[Mode]
    """
    Major mode.
    """
    minor: Final[Mode]
    """
    Minor mode.
    """
    def __repr__(self, /) -> str:
        """
        Human-readable representation of the mode.
        """
    def __str__(self, /) -> str:
        """
        Human-readable name of the mode.
        """

@final
class MonoConversionMethod:
    """
    Method for converting multi-channel audio to mono.

    `MonoConversionMethod` represents a strategy for collapsing multi-channel
    audio (e.g. stereo or surround) into a single mono channel. Different methods
    trade off simplicity, spatial fidelity, and control over channel weighting.

    Instances of `MonoConversionMethod` are immutable and should be treated as
    value objects constructed via class methods.

    Available conversion methods:

    - ``MonoConversionMethod.average()``
    - ``MonoConversionMethod.left()``
    - ``MonoConversionMethod.right()``
    - ``MonoConversionMethod.weighted(weights: list[float])``
    - ``MonoConversionMethod.center()``
    """
    average: Final[MonoConversionMethod]
    """
    Average all channels equally.

    Each input channel contributes equally to the output mono signal.
    """
    center: Final[MonoConversionMethod]
    """
    Use the centre channel if available, otherwise average left and right.
    """
    left: Final[MonoConversionMethod]
    """
    Use the left channel only.

    Intended primarily for stereo input.
    """
    right: Final[MonoConversionMethod]
    """
    Use the right channel only.

    Intended primarily for stereo input.
    """
    @staticmethod
    def default() -> MonoConversionMethod: ...
    @classmethod
    def weighted(cls, /, weights: "list[float]") -> MonoConversionMethod:
        """
        Weighted average across channels.

        Each channel is multiplied by a corresponding weight prior to summation.

        Parameters
        ----------
        weights : list[float]
            Per-channel weights. The length should match the number of input
            channels.
        """

@final
class NoiseColor:
    """
    Noise colour classification for audio perturbation and synthesis.

    `NoiseColor` classifies stochastic noise processes by their spectral energy
    distribution. Different noise colours influence perceived brightness,
    smoothness, and temporal correlation in audio synthesis and perturbation
    tasks.

    Instances of `NoiseColor` are immutable and should be treated as enum-like
    values. They are accessed via class attributes rather than being constructed
    directly.

    Available noise colours:

    - ``NoiseColor.white``
    - ``NoiseColor.pink``
    - ``NoiseColor.brown``
    """
    brown: Final[NoiseColor]
    """
    Brown (red) noise.

    Exhibits strongly attenuated high-frequency content, yielding a smoother
    and more correlated temporal structure.
    """
    pink: Final[NoiseColor]
    """
    Pink noise.

    Exhibits decreasing spectral energy with increasing frequency, producing
    a perceptually balanced spectrum across octaves.
    """
    white: Final[NoiseColor]
    """
    White noise.

    Exhibits approximately uniform spectral energy density across the
    frequency spectrum, resulting in a bright and broadband character.
    """
    @staticmethod
    def default() -> NoiseColor: ...

@final
class NormalizationMethod:
    """
    Normalisation strategy for audio sample data.

    `NormalizationMethod` represents a predefined method for rescaling or
    re-centring audio samples prior to further processing. Normalisation is
    commonly used to stabilise numerical behaviour, improve comparability
    between signals, or enforce amplitude constraints.

    Instances of `NormalizationMethod` are immutable and should be treated as
    enum-like values. They are accessed via class attributes rather than being
    constructed directly.

    Available normalisation methods:

    - ``NormalizationMethod.minmax``:
      Linearly rescales samples so that the minimum maps to 0 and the maximum
      maps to 1 (or -1 to 1, depending on context).

    - ``NormalizationMethod.zscore``:
      Applies z-score normalisation by subtracting the mean and dividing by the
      standard deviation.

    - ``NormalizationMethod.peak``:
      Scales samples so that the maximum absolute amplitude equals 1.

    - ``NormalizationMethod.mean``:
      Recentres samples by subtracting the mean value.

    - ``NormalizationMethod.median``:
      Recentres samples by subtracting the median value.

    Normalisation methods do not modify data eagerly; they describe how
    normalisation should be applied by downstream operations.
    """
    mean: Final[NormalizationMethod]
    """
    Mean normalisation.

    Recentres samples by subtracting the mean value without rescaling their
    variance.
    """
    median: Final[NormalizationMethod]
    """
    Median normalisation.

    Recentres samples by subtracting the median value. This method is more
    robust to outliers than mean-based normalisation.
    """
    minmax: Final[NormalizationMethod]
    """
    Min--max normalisation.

    Linearly rescales samples so that the minimum and maximum values are
    mapped to a fixed range. This method preserves relative distances but
    is sensitive to outliers.
    """
    peak: Final[NormalizationMethod]
    """
    Peak normalisation.

    Scales samples so that the maximum absolute amplitude equals 1. This
    preserves waveform shape while enforcing a strict amplitude bound.
    """
    zscore: Final[NormalizationMethod]
    """
    Z-score normalisation.

    Subtracts the mean of the samples and divides by their standard
    deviation. This produces zero-mean, unit-variance data and is commonly
    used in statistical and machine learning pipelines.
    """
    @staticmethod
    def default() -> NormalizationMethod: ...

@final
class OnsetDetectionConfig:
    """
    Python wrapper for ``OnsetDetectionConfig``.

    Configuration for onset detection using spectral flux method.

    Notes:

        The keyword defaults of the constructor
        ``OnsetDetectionConfig(cqt_params, hop_size, ...)`` are not those of
        ``OnsetDetectionConfig.default()``: the constructor uses
        ``min_onset_interval_secs=0.07``, ``pre_emphasis=0.0``,
        ``median_filter_length=3``, ``adaptive_threshold_multiplier=3.0``,
        while ``default()`` uses ``0.1``, ``0.1``, ``5``, ``1.5`` (other fields
        agree). Start from ``default()`` for the library defaults.

        ``peak_picking`` and ``cqt_params`` return copies; assign a modified
        copy back (``config.peak_picking = pp``) to change them.
    """
    def __new__(cls, /, cqt_params: Any, hop_size: int, window_size: int | None = None, threshold: float = 0.3, min_onset_interval_secs: float = 0.07, pre_emphasis: float = 0.0, adaptive_threshold: bool = True, median_filter_length: int = 3, adaptive_threshold_multiplier: float = 3.0, peak_picking: PeakPickingConfig | None = None) -> OnsetDetectionConfig:
        """
        Create a new ``OnsetDetectionConfig`` with custom parameters.

        Parameters
        ----------
        ``cqt_params`` : ``PyCqtParams``
            CQT configuration for spectral analysis
        ``hop_size`` : int
            Hop size for frame-based analysis in samples
        ``window_size`` : Optional[int]
            Window size for CQT analysis in samples (None = auto-calculate)
        threshold : float
            Threshold for onset detection
        ``min_onset_interval_secs`` : float
            Minimum time between consecutive onsets in seconds
        ``pre_emphasis`` : float
            Pre-emphasis factor for signal processing
        ``adaptive_threshold`` : bool
            Whether to use adaptive thresholding
        ``median_filter_length`` : int
            Length of median filter for smoothing
        ``adaptive_threshold_multiplier`` : float
            Multiplier for adaptive threshold
        ``peak_picking`` : ``PyPeakPickingConfig``
            Peak picking configuration

        Notes
        -----
        The keyword defaults of this constructor are not those of
        ``OnsetDetectionConfig.default()``: the constructor uses
        ``min_onset_interval_secs=0.07``, ``pre_emphasis=0.0``,
        ``median_filter_length=3`` and ``adaptive_threshold_multiplier=3.0``,
        while ``default()`` uses ``0.1``, ``0.1``, ``5`` and ``1.5`` (other
        fields agree). Use ``default()`` and set fields for the library default.
        """
    def __repr__(self, /) -> str: ...
    @property
    def adaptive_threshold(self, /) -> bool:
        """
        Get whether adaptive thresholding is enabled.
        """
    @adaptive_threshold.setter
    def adaptive_threshold(self, /, value: bool) -> None:
        """
        Set whether adaptive thresholding is enabled.
        """
    @property
    def adaptive_threshold_multiplier(self, /) -> float:
        """
        Get the adaptive threshold multiplier.
        """
    @adaptive_threshold_multiplier.setter
    def adaptive_threshold_multiplier(self, /, value: float) -> None:
        """
        Set the adaptive threshold multiplier.
        """
    @property
    def cqt_params(self, /) -> Any:
        """
        Get the CQT parameters.
        """
    @cqt_params.setter
    def cqt_params(self, /, value: Any) -> None:
        """
        Set the CQT parameters.
        """
    @classmethod
    def default(cls, /) -> OnsetDetectionConfig:
        """
        Create a default ``OnsetDetectionConfig``.
        """
    def explain(self, /, sample_rate: "float | None" = None, *, n_samples: "int | None" = None, n_channels: "int | None" = None, f0_range: "tuple[float, float] | None" = None) -> ExplainDoc:
        """
        Explain what this onset-detection configuration will do.

        Args:

            sample_rate (float | None): Sample rate in Hz of the signal it
                will be applied to. Without it, quantities that need a
                sample rate are omitted and an ``info`` warning says so.
            n_samples (int | None): Signal length in samples per channel.
            n_channels (int | None): Number of channels.
            f0_range (tuple[float, float] | None): Expected
                fundamental-frequency range ``(lowest, highest)`` in Hz.

        Returns:

            ExplainDoc: The explanation (see ``audio_samples.explain``).
        """
    @property
    def hop_size(self, /) -> int:
        """
        Get the hop size in samples.
        """
    @hop_size.setter
    def hop_size(self, /, value: int) -> None:
        """
        Set the hop size in samples.
        """
    @property
    def median_filter_length(self, /) -> int:
        """
        Get the median filter length.
        """
    @median_filter_length.setter
    def median_filter_length(self, /, value: int) -> None:
        """
        Set the median filter length.
        """
    @property
    def min_onset_interval_secs(self, /) -> float:
        """
        Get the minimum onset interval in seconds.
        """
    @min_onset_interval_secs.setter
    def min_onset_interval_secs(self, /, value: float) -> None:
        """
        Set the minimum onset interval in seconds.
        """
    @classmethod
    def musical(cls, /) -> OnsetDetectionConfig:
        """
        Create an ``OnsetDetectionConfig`` optimized for general music.
        """
    @property
    def peak_picking(self, /) -> PeakPickingConfig:
        """
        Get the peak picking configuration.

        Returns a **copy**: setting fields on the returned object does not change
        this config. Modify the copy and assign it back:

        ::

            pp = config.peak_picking
            pp.min_peak_separation = 256
            config.peak_picking = pp
        """
    @peak_picking.setter
    def peak_picking(self, /, value: PeakPickingConfig) -> None:
        """
        Set the peak picking configuration.
        """
    @classmethod
    def percussive(cls, /) -> OnsetDetectionConfig:
        """
        Create an ``OnsetDetectionConfig`` optimized for percussive sounds.
        """
    @property
    def pre_emphasis(self, /) -> float:
        """
        Get the pre-emphasis factor.
        """
    @pre_emphasis.setter
    def pre_emphasis(self, /, value: float) -> None:
        """
        Set the pre-emphasis factor.
        """
    @classmethod
    def speech(cls, /) -> OnsetDetectionConfig:
        """
        Create an ``OnsetDetectionConfig`` optimized for speech.
        """
    @property
    def threshold(self, /) -> float:
        """
        Get the onset detection threshold.
        """
    @threshold.setter
    def threshold(self, /, value: float) -> None:
        """
        Set the onset detection threshold.
        """
    @property
    def window_size(self, /) -> int | None:
        """
        Get the window size in samples, or None if it defaults to CQT's minimum window size.
        """
    @window_size.setter
    def window_size(self, /, value: int | None) -> None:
        """
        Set the window size in samples.
        """

@final
class PadSide:
    """
    Side for padding operations.

    `PadSide` indicates which side of an audio signal should be padded when
    applying padding operations. It is used to specify whether padding should be
    added to the beginning (left) or end (right) of the signal.
    """
    left: Final[PadSide]
    """
    Pad on the left side (beginning) of the signal.
    """
    right: Final[PadSide]
    """
    Pad on the right side (end) of the signal.
    """
    @staticmethod
    def default() -> PadSide: ...

@final
class ParametricEq:
    """
    Parametric equaliser configuration.

    `ParametricEq` represents a complete parametric equaliser composed of an
    ordered collection of EQ bands. Each band shapes a specific region of the
    frequency spectrum, and the combined effect defines the overall frequency
    response.

    Instances of `ParametricEq` are mutable. Bands can be added and removed after
    construction. The number of bands can be queried using ``len(eq)`` in Python.
    """
    def __len__(self, /) -> int:
        """
        Number of EQ bands.

        Enables use of ``len(eq)`` in Python.
        """
    def __new__(cls, /) -> ParametricEq:
        """
        Create a new empty parametric equaliser.
        """
    def __repr__(self, /) -> str: ...
    def add_band(self, /, band: EqBand) -> None:
        """
        Add an EQ band to the equaliser.

        Parameters
        ----------
        band : ``EqBand``
            Band configuration to append to the equaliser.
        """
    @property
    def bypassed(self, /) -> bool:
        """
        Whether the equaliser is currently bypassed.
        """
    @property
    def output_gain_db(self, /) -> float:
        """
        Overall output gain in decibels.
        """
    def remove_band(self, /, index: int) -> EqBand | None:
        """
        Remove an EQ band by index.

        Parameters
        ----------
        index : int
            Zero-based index of the band to remove.

        Returns
        -------
        ``EqBand`` or None
            The removed band if the index was valid, otherwise ``None``.
        """

@final
class PeakPickingConfig:
    """
    Configuration for peak picking with temporal constraints.

    Peak picking identifies local maxima in the onset strength function that
    exceed a threshold. Temporal constraints ensure detected peaks are
    separated by minimum time intervals and can include smoothing.

    Presets are class attributes, accessed **without** parentheses:
    ``PeakPickingConfig.music``, ``PeakPickingConfig.speech``,
    ``PeakPickingConfig.drums``. ``PeakPickingConfig.default()`` is a method.
    """
    drums: Final[PeakPickingConfig]
    """
    Drums preset configuration (class attribute: ``PeakPickingConfig.drums``).
    """
    music: Final[PeakPickingConfig]
    """
    Music preset configuration (class attribute: ``PeakPickingConfig.music``).
    """
    speech: Final[PeakPickingConfig]
    """
    Speech preset configuration (class attribute: ``PeakPickingConfig.speech``).
    """
    def __new__(cls, /, *, adaptive_threshold_config: "AdaptiveThresholdConfig", min_peak_separation: "int", pre_emphasis: "bool", pre_emphasis_coeff: "float", median_filter: "bool", median_filter_length: "int", normalize_onset_strength: "bool", normalization_method: "NormalizationMethod") -> PeakPickingConfig:
        """
        Create a new ``PeakPickingConfig``.

        Parameters
        ----------
        ``adaptive_threshold_config`` : ``AdaptiveThresholdConfig``
            Adaptive thresholding configuration.
        ``min_peak_separation`` : int
            Minimum separation between detected peaks in frames.
        ``pre_emphasis`` : bool
            Whether to apply pre-emphasis filtering.
        ``pre_emphasis_coeff`` : float
            Pre-emphasis filter coefficient.
        ``median_filter`` : bool
            Whether to apply median filtering.
        ``median_filter_length`` : int
            Length of the median filter in frames.
        ``normalize_onset_strength`` : bool
            Whether to normalize the onset strength function.
        ``normalization_method`` : ``NormalizationMethod``
            Method for normalization.
        """
    def __repr__(self, /) -> str: ...
    @staticmethod
    def default() -> PeakPickingConfig: ...
    @property
    def min_peak_separation(self, /) -> int:
        """
        Minimum separation between detected peaks in frames.
        """
    @min_peak_separation.setter
    def min_peak_separation(self, /, value: int) -> None:
        """
        Minimum separation between detected peaks in frames.
        """
    def set_median_filter(self, /, enabled: "bool", length: "int") -> None:
        """
        Enable or disable median filtering.
        """
    def set_min_peak_separation_ms(self, /, value: "float", sample_rate: "float") -> None:
        """
        Set minimum peak separation in milliseconds.
        """
    def set_pre_emphasis(self, /, enabled: "bool", coeff: "float") -> None:
        """
        Enable or disable pre-emphasis filtering.
        """
    def validate(self, /) -> PeakPickingConfig:
        """
        Validate the configuration parameters.
        """

@final
class PerturbationConfig:
    """
    Configuration for audio perturbation operations.

    `PerturbationConfig` defines how a perturbation method should be applied to
    audio data, optionally including a deterministic random seed.

    The configuration is immutable once created. Validation can be performed
    explicitly using ``validate(sample_rate)``.

    Typical usage is to construct a perturbation method first and then wrap it
    in a configuration object:

    ```python
    method = PerturbationMethod.gaussian(-12.0, NoiseColor.white)
    cfg = PerturbationConfig(method, seed=42).validate(sample_rate)
    ```
    """
    def __new__(cls, /, method: "PerturbationMethod", seed: "int | None" = None) -> PerturbationConfig:
        """
        Create a new perturbation configuration.

        Parameters
        ----------
        method : ``PerturbationMethod``
            The perturbation method to apply.
        seed : Optional[int]
            Optional random seed for deterministic behaviour. If ``None``, a
            non-deterministic random generator is used.

        Returns
        -------
        ``PerturbationConfig``
            A new perturbation configuration.
        """
    def __repr__(self, /) -> str: ...
    @property
    def seed(self, /) -> int | None:
        """
        Random seed used for deterministic perturbation, if specified.

        Returns
        -------
        Optional[int]
            The seed value, or ``None`` if no seed is configured.
        """
    def validate(self, /, sample_rate: "float") -> PerturbationConfig:
        """
        Validate the perturbation configuration for a given sample rate.

        Parameters
        ----------
        ``sample_rate`` : float
            Sample rate in Hz, used to validate frequency-dependent parameters.

        Returns
        -------
        ``PerturbationConfig``
            This instance. Returned to allow call chaining.

        Raises
        ------
        ``ValueError``
            If the configuration or underlying perturbation method is invalid.
        """

@final
class PerturbationMethod:
    """
    Perturbation methods for audio data augmentation.

    `PerturbationMethod` represents a specific audio perturbation configuration
    used for data augmentation, robustness testing, or creative effects.

    This type is enum-like in Python. Instances are constructed via class
    constructors such as ``PerturbationMethod.gaussian(...)`` or
    ``PerturbationMethod.pitch_shift(...)`` rather than by calling
    ``PerturbationMethod()`` directly.

    A perturbation configuration can be validated against a sample rate via
    ``validate(sample_rate)``.
    """
    @classmethod
    def brown_noise(cls, /, target_snr_db: "float") -> PerturbationMethod:
        """
        Brown noise injection with a target signal-to-noise ratio.

        Adds coloured Gaussian noise to achieve the target SNR relative to the
        input signal's RMS level.

        Parameters
        ----------
        ``target_snr_db`` : float
            Target signal-to-noise ratio in decibels.

        Returns
        -------

        ``PerturbationMethod``
           A Gaussian-noise perturbation configuration.
        """
    @classmethod
    def gaussian(cls, /, target_snr_db: "float", noise_color: "NoiseColor") -> PerturbationMethod:
        """
        Gaussian noise injection with a target signal-to-noise ratio.

        Adds coloured Gaussian noise to achieve the target SNR relative to the
        input signal's RMS level.

        Parameters
        ----------
        ``target_snr_db`` : float
            Target signal-to-noise ratio in decibels.
        ``noise_color`` : ``NoiseColor``
            Noise colour (spectral distribution) to use.

        Returns
        -------
        ``PerturbationMethod``
            A Gaussian-noise perturbation configuration.
        """
    @classmethod
    def high_pass_filter(cls, /, cutoff_hz: "float") -> PerturbationMethod:
        """
        High-pass filtering perturbation.

        Applies a high-pass filter to reduce low-frequency content, simulating
        effects such as rumble removal.

        Parameters
        ----------
        ``cutoff_hz`` : float
            High-pass cutoff frequency in Hz.

        Returns
        -------
        ``PerturbationMethod``
            A high-pass filter perturbation configuration.
        """
    @classmethod
    def high_pass_filter_with_slope(cls, /, cutoff_hz: "float", slope_db_per_octave: "float") -> PerturbationMethod:
        """
        High-pass filtering perturbation with custom slope.

        Parameters
        ----------
        ``cutoff_hz`` : float
            High-pass cutoff frequency in Hz.
        ``slope_db_per_octave`` : float
            Filter slope in dB per octave.

        Returns
        -------
        ``PerturbationMethod``
            A high-pass filter perturbation configuration.
        """
    @classmethod
    def low_pass_filter(cls, /, cutoff_hz: "float") -> PerturbationMethod:
        """
        Low-pass filtering perturbation.

        Applies a low-pass filter to reduce high-frequency content, simulating
        effects such as telephone bandwidth limitation.

        Parameters
        ----------
        ``cutoff_hz`` : float
            Low-pass cutoff frequency in Hz.

        Returns
        -------
        ``PerturbationMethod``
            A low-pass filter perturbation configuration.
        """
    @classmethod
    def low_pass_filter_with_slope(cls, /, cutoff_hz: "float", slope_db_per_octave: "float") -> PerturbationMethod:
        """
        Low-pass filtering perturbation with custom slope.

        Parameters
        ----------
        ``cutoff_hz`` : float
            Low-pass cutoff frequency in Hz.
        ``slope_db_per_octave`` : float
            Filter slope in dB per octave.

        Returns
        -------
        ``PerturbationMethod``
            A low-pass filter perturbation configuration.
        """
    @classmethod
    def pink_noise(cls, /, target_snr_db: "float") -> PerturbationMethod:
        """
        Pink noise injection with a target signal-to-noise ratio.

        Adds coloured Gaussian noise to achieve the target SNR relative to the
        input signal's RMS level.

        Parameters
        ----------
        ``target_snr_db`` : float
           Target signal-to-noise ratio in decibels.

        Returns
        -------

        ``PerturbationMethod``
            A Gaussian-noise perturbation configuration.
        """
    @classmethod
    def pitch_shift(cls, /, semitones: "float", preserve_formants: "bool" = False) -> PerturbationMethod:
        """
        Pitch shifting perturbation.

        Shifts the pitch of the signal by a number of semitones while attempting
        to maintain duration.

        Parameters
        ----------
        semitones : float
            Pitch shift in semitones. Positive shifts up; negative shifts down.
        ``preserve_formants`` : bool, optional
            Whether to attempt formant preservation.

        Returns
        -------
        ``PerturbationMethod``
            A pitch-shift perturbation configuration.
        """
    @classmethod
    def random_gain(cls, /, min_gain_db: "float", max_gain_db: "float") -> PerturbationMethod:
        """
        Random gain perturbation within a specified range.

        Applies a uniform random gain (in dB) to all channels. Positive values
        boost; negative values attenuate.

        Parameters
        ----------
        ``min_gain_db`` : float
            Minimum gain in decibels.
        ``max_gain_db`` : float
            Maximum gain in decibels.

        Returns
        -------
        ``PerturbationMethod``
            A random-gain perturbation configuration.
        """
    def validate(self, /, sample_rate: "float") -> PerturbationMethod:
        """
        Validate perturbation parameters for a given sample rate.

        Parameters
        ----------
        ``sample_rate`` : float
            Sample rate in Hz, used to validate frequency-dependent parameters.

        Returns
        -------
        ``PerturbationMethod``
            This instance. Returned to allow call chaining.

        Raises
        ------
        ``ValueError``
            If the perturbation parameters are invalid.
        """
    @classmethod
    def white_noise(cls, /, target_snr_db: "float") -> PerturbationMethod:
        """
        White noise injection with a target signal-to-noise ratio.

        Adds coloured Gaussian noise to achieve the target SNR relative to the
        input signal's RMS level.

        Parameters
        ----------
        ``target_snr_db`` : float
            Target signal-to-noise ratio in decibels.

        Returns
        -------
        ``PerturbationMethod``
            A Gaussian-noise perturbation configuration.
        """

@final
class PitchClass:
    """
    One of the twelve pitch classes of the chromatic scale.

    `PitchClass` represents a pitch class independent of octave. Pitch class ``C``
    has index 0, ascending chromatically to ``B`` at index 11.

    Instances of `PitchClass` are immutable and should be treated as enum-like
    values. They are accessed via class attributes, or constructed from a
    chromatic index via ``PitchClass.from_index(index)``.

    Available pitch classes:

    - ``PitchClass.c``, ``PitchClass.c_sharp``, ``PitchClass.d``,
      ``PitchClass.d_sharp``, ``PitchClass.e``, ``PitchClass.f``,
      ``PitchClass.f_sharp``, ``PitchClass.g``, ``PitchClass.g_sharp``,
      ``PitchClass.a``, ``PitchClass.a_sharp``, ``PitchClass.b``.
    """
    a: Final[PitchClass]
    """
    Pitch class A (index 9).
    """
    a_sharp: Final[PitchClass]
    """
    Pitch class A# / Bb (index 10).
    """
    b: Final[PitchClass]
    """
    Pitch class B (index 11).
    """
    c: Final[PitchClass]
    """
    Pitch class C (index 0).
    """
    c_sharp: Final[PitchClass]
    """
    Pitch class C# / Db (index 1).
    """
    d: Final[PitchClass]
    """
    Pitch class D (index 2).
    """
    d_sharp: Final[PitchClass]
    """
    Pitch class D# / Eb (index 3).
    """
    e: Final[PitchClass]
    """
    Pitch class E (index 4).
    """
    f: Final[PitchClass]
    """
    Pitch class F (index 5).
    """
    f_sharp: Final[PitchClass]
    """
    Pitch class F# / Gb (index 6).
    """
    g: Final[PitchClass]
    """
    Pitch class G (index 7).
    """
    g_sharp: Final[PitchClass]
    """
    Pitch class G# / Ab (index 8).
    """
    def __repr__(self, /) -> str:
        """
        Human-readable representation of the pitch class.
        """
    def __str__(self, /) -> str:
        """
        Human-readable name of the pitch class (e.g. ``"C#"``).
        """
    @staticmethod
    def from_index(index: "int") -> PitchClass:
        """
        Construct a pitch class from a chromatic index in ``0..=11``.

        Parameters
        ----------
        index : int
            Chromatic index where 0 maps to C and 11 maps to B.

        Returns
        -------
        ``PitchClass``
            The pitch class for the given index.

        Raises
        ------
        ``ValueError``
            If ``index`` is greater than 11.
        """
    def to_index(self, /) -> int:
        """
        The chromatic index of this pitch class (C = 0, B = 11).
        """

@final
class PitchContour:
    """
    A time-ordered pitch track.

    `PitchContour` is a sequence of [`PitchFrame`] values produced by pitch
    tracking. Each frame pairs a frame onset time with the pitch detected in that
    frame (or ``None`` when no pitch was found).

    Instances are produced by pitch tracking and are not constructed directly from
    Python.
    """
    def __len__(self, /) -> int:
        """
        The total number of frames (voiced and unvoiced).
        """
    def __repr__(self, /) -> str:
        """
        Human-readable representation showing the frame count.
        """
    def frames(self, /) -> list[PitchFrame]:
        """
        All frames in time order, voiced and unvoiced alike.

        Returns
        -------
        list[``PitchFrame``]
            The complete list of pitch frames.
        """
    def mean_pitch(self, /) -> float | None:
        """
        The mean of all voiced frequencies, or ``None`` when no frame is voiced.
        """
    def voiced_frames(self, /) -> list[tuple[float, float]]:
        """
        Voiced frames as ``(time_seconds, frequency_hz)`` pairs.

        Unvoiced frames are skipped.

        Returns
        -------
        list[tuple[float, float]]
            The voiced frames as (time, frequency) tuples.
        """

@final
class PitchDetectionMethod:
    """
    Pitch detection algorithm selection.

    `PitchDetectionMethod` selects the algorithm used to estimate the fundamental
    frequency of a signal. Different algorithms trade off accuracy, robustness to
    noise and inharmonicity, latency, and computational cost.

    Instances of `PitchDetectionMethod` are immutable and should be treated as
    enum-like values. They are accessed via class attributes rather than being
    constructed directly.

    Available methods:

    - ``PitchDetectionMethod.yin``
    - ``PitchDetectionMethod.autocorrelation``
    - ``PitchDetectionMethod.cepstrum`` (not implemented)
    - ``PitchDetectionMethod.harmonic_product`` (not implemented)

    Notes:

        Only ``yin`` and ``autocorrelation`` are implemented. ``track_pitch``
        raises ``AudioParameterError`` for ``cepstrum`` and
        ``harmonic_product``.
    """
    autocorrelation: Final[PitchDetectionMethod]
    """
    Autocorrelation-based pitch detection.

    Simple and fast, but sensitive to noise and octave errors for complex or
    weakly periodic signals.
    """
    cepstrum: Final[PitchDetectionMethod]
    """
    Cepstral pitch detection (not implemented).

    Reserved; ``track_pitch`` raises ``AudioParameterError`` for this method.
    Use ``yin`` or ``autocorrelation``.
    """
    harmonic_product: Final[PitchDetectionMethod]
    """
    Harmonic Product Spectrum (HPS) (not implemented).

    Reserved; ``track_pitch`` raises ``AudioParameterError`` for this method.
    Use ``yin`` or ``autocorrelation``.
    """
    yin: Final[PitchDetectionMethod]
    """
    YIN pitch detection algorithm.

    Accurate fundamental frequency estimation for speech and musical
    signals, at moderate computational cost.
    """
    @staticmethod
    def default() -> PitchDetectionMethod: ...

@final
class PitchFrame:
    """
    A single frame of a pitch contour.

    `PitchFrame` pairs a frame onset time with the fundamental frequency detected
    in that frame. The frequency is ``None`` for unvoiced frames where no pitch
    was detected.

    Fields are exposed via read-only properties.
    """
    def __new__(cls, /, time: "float", frequency: float | None = None) -> PitchFrame:
        """
        Create a new pitch frame.

        Parameters
        ----------
        time : float
            Frame onset time, in seconds from the start of the signal.
        frequency : float, optional
            Detected fundamental frequency in Hz, or ``None`` if the frame is
            unvoiced.

        Returns
        -------
        ``PitchFrame``
            A new pitch frame.
        """
    def __repr__(self, /) -> str:
        """
        Human-readable representation of the frame.
        """
    @property
    def frequency(self, /) -> float | None:
        """
        Detected fundamental frequency in Hz, or ``None`` if the frame is unvoiced.
        """
    @property
    def time(self, /) -> float:
        """
        Frame onset time, in seconds from the start of the signal.
        """
    @property
    def voiced(self, /) -> bool:
        """
        Whether this frame is voiced (a pitch was detected).
        """

@final
class Psd:
    """
    Power spectral density (PSD) estimate.

    `Psd` pairs a frequency axis with the estimated power-per-Hz at each bin. The
    two arrays are always the same length: ``frequencies[i]`` is the centre
    frequency of bin ``i`` in Hz, and ``density[i]`` is the estimated power
    spectral density at that frequency.

    Instances of `Psd` are produced by the power spectral density transform and
    are not constructed directly from Python. The frequency and density data are
    exposed as NumPy arrays.
    """
    def __len__(self, /) -> int:
        """
        The number of frequency bins.
        """
    def __repr__(self, /) -> str:
        """
        String representation showing the number of bins.
        """
    @property
    def density(self, /) -> Incomplete:
        """
        The estimated power spectral density (power per Hz) at each frequency bin.

        Returns
        -------
        numpy.ndarray
            A 1-D array of power-per-Hz values. Same length as ``frequencies``.
        """
    @property
    def frequencies(self, /) -> Incomplete:
        """
        The frequency axis, in Hz.

        Returns
        -------
        numpy.ndarray
            A 1-D array of bin centre frequencies in Hz. Same length as
            ``density``.
        """
    def into_parts(self, /) -> tuple[Incomplete, Incomplete]:
        """
        Return the ``(frequencies, density)`` arrays as a tuple.

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray]
            A pair of 1-D NumPy arrays containing the frequency axis and the
            density values respectively.
        """

@final
class ResamplingQuality:
    """
    Quality level for resampling operations.

    Every tier is delay-free (output sample ``k`` is the input reconstructed
    at time ``k * in_rate / out_rate``) and produces exactly
    ``ceil(n * out_rate / in_rate)`` samples; the tiers differ in the
    anti-aliasing / anti-imaging filter (Nyquist of the lower rate):

    - ``ResamplingQuality.fast`` (the default of ``AudioSamples.resample``):
      short windowed sinc when downsampling (passband 0.80 x Nyquist, 70 dB
      stopband); linear interpolation when upsampling (images not suppressed).
    - ``ResamplingQuality.medium``: windowed sinc, passband 0.90 x Nyquist,
      100 dB stopband.
    - ``ResamplingQuality.high``: windowed sinc, passband 0.913 x Nyquist,
      125 dB stopband.
    - ``ResamplingQuality.very_high``: windowed sinc, 185 dB stopband
      (limited to about -140 dB by rounding for float32 audio); passband
      0.95 x Nyquist for decimation by a power of two, 0.98 for other ratios
      with no prime factor above 7, 0.996 for the rest (such as pitch shifts).

    Instances are immutable enum-like values accessed via the class
    attributes; resampling methods also accept the names ``"fast"``,
    ``"medium"``, ``"high"`` and ``"very_high"``.
    """
    fast: Final[ResamplingQuality]
    """
    Fast resampling: short windowed sinc (passband 0.80 x Nyquist, 70 dB
    stopband) when downsampling, linear interpolation when upsampling.
    """
    high: Final[ResamplingQuality]
    """
    High quality resampling: windowed sinc, passband 0.913 x Nyquist,
    125 dB stopband.
    """
    medium: Final[ResamplingQuality]
    """
    Medium quality resampling: windowed sinc, passband 0.90 x Nyquist,
    100 dB stopband.
    """
    very_high: Final[ResamplingQuality]
    """
    Very high quality resampling: windowed sinc, 185 dB stopband; passband
    0.95 x Nyquist for decimation by a power of two, 0.98 for other ratios
    with no prime factor above 7, 0.996 for the rest (such as pitch shifts).
    """
    @staticmethod
    def default() -> ResamplingQuality: ...

@final
class SampleType:
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

    Instances of `SampleType` are immutable and comparable. They should be
    treated as enum values rather than constructed directly.
    """
    F32: Final[SampleType]
    F64: Final[SampleType]
    I16: Final[SampleType]
    I24: Final[SampleType]
    I32: Final[SampleType]
    U8: Final[SampleType]
    def __int__(self, /) -> int: ...
    def __repr__(self, /) -> str: ...

@final
class SideChainConfig:
    """
    Side-chain configuration for dynamic range processing.

    `SideChainConfig` describes how an external or filtered control signal is used
    to drive gain reduction in dynamics processors such as compressors and
    limiters.

    Side-chain processing can be enabled or disabled, optionally filtered, and
    mixed with the internal detector signal.

    Instances of `SideChainConfig` are immutable value objects in the Python API.
    Parameters are provided at construction time and exposed via read-only
    properties.
    """
    def __new__(cls, /, enabled: "bool", high_pass_freq: "Optional[float]" = None, low_pass_freq: "Optional[float]" = None, pre_emphasis_db: float = 0.0, external_mix: float = 0.0) -> SideChainConfig:
        """
        Create a new side-chain configuration.

        Parameters
        ----------
        enabled : bool
            Whether side-chain processing is enabled.
        ``high_pass_freq`` : float or None, optional
            High-pass filter cutoff frequency in Hz for the side-chain signal.
            Useful for reducing low-frequency pumping effects.
        ``low_pass_freq`` : float or None, optional
            Low-pass filter cutoff frequency in Hz for the side-chain signal.
            Useful for focusing compression on specific frequency ranges.
        ``pre_emphasis_db`` : float, optional
            Pre-emphasis applied to the side-chain signal in decibels.
        ``external_mix`` : float, optional
            Mix ratio between internal and external side-chain signal in the
            range ``[0.0, 1.0]``. ``0.0`` selects internal only, ``1.0`` selects
            external only.

        Returns
        -------
        ``SideChainConfig``
            A new side-chain configuration.

        Notes
        -----
        Parameter validity is not automatically checked at construction time.
        Invalid configurations may fail later during processing.
        """
    def __repr__(self, /) -> str: ...
    @staticmethod
    def default() -> SideChainConfig: ...
    @property
    def enabled(self, /) -> bool:
        """
        Whether side-chain processing is enabled.
        """
    @property
    def external_mix(self, /) -> float:
        """
        Mix ratio between internal and external side-chain signal.

        Values are expected in the range ``[0.0, 1.0]``.
        """
    @property
    def high_pass_freq(self, /) -> float | None:
        """
        High-pass filter cutoff frequency in Hz for the side-chain signal, if set.
        """
    @property
    def low_pass_freq(self, /) -> float | None:
        """
        Low-pass filter cutoff frequency in Hz for the side-chain signal, if set.
        """
    @property
    def pre_emphasis_db(self, /) -> float:
        """
        Pre-emphasis applied to the side-chain signal in decibels.
        """

@final
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
    def filtfilt(self, /, samples: "np.typing.NDArray[np.float64]") -> "NDArray[np.float64]":
        """
        Zero-phase forward-backward filtering of a whole signal.

        The signal is extended at both ends by odd reflection and each pass
        starts from the steady-state initial conditions for its first sample.
        The streaming state is neither used nor modified. Returns a new array.

        Args:

            samples (np.typing.NDArray[np.float64]): The whole signal.

        Returns:

            np.typing.NDArray[np.float64]: Filtered signal, same length.
        """
    def frequency_response(self, /, frequencies: "np.typing.NDArray[np.float64]") -> tuple[Incomplete, Incomplete]:
        """
        Compute the frequency response of the cascade.

        Evaluates the combined transfer function of all sections. The magnitude
        is the product of section magnitudes; the phase is the sum of section
        phases. The sample rate fixed at construction is used.

        Args:

            frequencies (np.typing.NDArray[np.float64]): Frequencies in hertz.

        Returns:

            tuple[np.typing.NDArray[np.float64], np.typing.NDArray[np.float64]]:
            Magnitude and phase (radians) arrays, each the same length as
            ``frequencies``.
        """
    @staticmethod
    def from_design(design: "IirFilterDesign", sample_rate: "float") -> SosFilter:
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
    @staticmethod
    def from_sos(sos: "np.typing.NDArray[np.float64]", sample_rate: "float") -> SosFilter:
        """
        Build a streaming filter from second-order-section coefficients.

        Accepts the ``(n_sections, 6)`` array returned by
        ``scipy.signal.butter(..., output='sos')`` (and friends) or by
        :meth:`IirFilterDesign.sos`, with rows ``[b0, b1, b2, a0, a1, a2]``.

        Args:

            sos (np.typing.NDArray[np.float64]): Section coefficients, shape ``(n, 6)``.
            sample_rate (float): Sample rate in hertz (used by :meth:`frequency_response`).

        Raises:

            ValueError: If the array is not ``(n, 6)``, is empty, contains
                non-finite values, or has a zero ``a0``.
        """
    def process_block(self, /, block: "np.typing.NDArray[np.float64]") -> None:
        """
        Process a block of samples in place, retaining state across calls.

        Alias of :meth:`process_samples_in_place` with a name that signals the
        design-once / stream-many usage. Because delay lines persist, calling
        ``process_block`` on consecutive blocks of a signal yields exactly the
        same result as one call over the whole signal.

        Args:

            block (np.typing.NDArray[np.float64]): Input/output block; modified in place.

        Returns:

            None: The provided array is modified in place.
        """
    def process_sample(self, /, x: "float") -> float:
        """
        Process a single sample through the cascade.

        Feeds ``x`` through each section in order; internal state is updated.

        Args:

            x (float): Input sample.

        Returns:

            float: The filtered output sample.
        """
    def process_samples(self, /, samples: "np.typing.NDArray[np.float64]") -> "NDArray[np.float64]":
        """
        Process an array of samples, returning a new array.

        Equivalent to calling :meth:`process_sample` for each input in order;
        state carries across the whole array.

        Args:

            samples (np.typing.NDArray[np.float64]): Input samples.

        Returns:

            np.typing.NDArray[np.float64]: Filtered output, same length as input.
        """
    def process_samples_in_place(self, /, samples: "np.typing.NDArray[np.float64]") -> None:
        """
        Process an array of samples in place.

        Overwrites each element of ``samples`` with its filtered value. The input
        array is modified directly.

        Args:

            samples (np.typing.NDArray[np.float64]): Input/output buffer; modified in place.

        Returns:

            None: The provided array is modified in place.
        """
    def reset(self, /) -> None:
        """
        Reset all sections' delay lines to zero.

        After a reset the cascade behaves identically to a freshly built filter
        with the same coefficients.

        Returns:

            None.
        """
    def sos(self, /) -> "NDArray[np.float64]":
        """
        The filter's second-order sections as a ``(n_sections, 6)`` array
        (SciPy ``sos`` layout, ``a0`` normalised to 1).
        """

@final
class SpectralFluxConfig:
    """
    Configuration for spectral flux onset detection.

    Spectral flux measures the rate of change of the magnitude spectrum
    between consecutive frames, providing effective onset detection for
    both percussive and tonal instruments.
    """
    complex: Final[SpectralFluxConfig]
    """
    Complex preset configuration.
    """
    musical: Final[SpectralFluxConfig]
    """
    Musical preset configuration.
    """
    percussive: Final[SpectralFluxConfig]
    """
    Percussive preset configuration.
    """
    def __new__(cls, /, *, cqt_params: Any, hop_size: int, window_size: int | None = None, flux_method: SpectralFluxMethod, peak_picking: PeakPickingConfig, rectify: bool, log_compression: float) -> SpectralFluxConfig:
        """
        Create a new ``SpectralFluxConfig``.

        Parameters
        ----------
        ``cqt_params`` : ``CqtParams``
            CQT parameters object from spectrograms module.
        ``hop_size`` : int
            Hop size in samples (must be > 0).
        ``window_size`` : Optional[int]
            Window size in samples (must be > 0 if provided). If ``None``, defaults to CQT's minimum window size.
        ``flux_method`` : ``SpectralFluxMethod``
            Spectral flux computation method.
        ``peak_picking`` : ``PeakPickingConfig``
            Peak picking configuration.
        rectify : bool
            Whether to rectify the onset strength function.
        ``log_compression`` : float
            Logarithmic compression factor.

        Returns
        ------
        SpectralFluxConfig
            A new spectral flux configuration.
        """
    def __repr__(self, /) -> str: ...
    @property
    def cqt_params(self, /) -> Any:
        """
        Get the CQT parameters.
        """
    @cqt_params.setter
    def cqt_params(self, /, value: Any) -> None:
        """
        Set the CQT parameters.
        """
    @property
    def flux_method(self, /) -> SpectralFluxMethod:
        """
        Get the spectral flux computation method.
        """
    @flux_method.setter
    def flux_method(self, /, value: SpectralFluxMethod) -> None:
        """
        Set the spectral flux computation method.
        """
    @property
    def hop_size(self, /) -> int:
        """
        Get the hop size in samples.
        """
    @hop_size.setter
    def hop_size(self, /, value: int) -> None:
        """
        Set the hop size in samples.
        """
    @property
    def log_compression(self, /) -> float:
        """
        Get the logarithmic compression factor.
        """
    @log_compression.setter
    def log_compression(self, /, value: float) -> None:
        """
        Set the logarithmic compression factor.
        """
    @property
    def peak_picking(self, /) -> PeakPickingConfig:
        """
        Get the peak picking configuration.

        Returns a **copy**: setting fields on the returned object does not change
        this config. Modify the copy and assign it back:

        ::

            pp = config.peak_picking
            pp.min_peak_separation = 256
            config.peak_picking = pp
        """
    @peak_picking.setter
    def peak_picking(self, /, value: PeakPickingConfig) -> None:
        """
        Set the peak picking configuration.
        """
    @property
    def rectify(self, /) -> bool:
        """
        Get whether to rectify the onset strength function.
        """
    @rectify.setter
    def rectify(self, /, value: bool) -> None:
        """
        Set whether to rectify the onset strength function.
        """
    def validate(self, /) -> SpectralFluxConfig:
        """
        Validate the configuration parameters.
        """
    @property
    def window_size(self, /) -> int | None:
        """
        Get the window size in samples, or None if it defaults to CQT's minimum window size.
        """
    @window_size.setter
    def window_size(self, /, value: int | None) -> None:
        """
        Set the window size in samples.
        """

@final
class SpectralFluxMethod:
    """
    Spectral flux variant for onset detection.

    Different flux formulations emphasise different types of spectral change
    and are therefore suited to different classes of musical and acoustic
    events.
    """
    complex: Final[SpectralFluxMethod]
    """
    Complex-domain spectral flux.

    Incorporates phase information in addition to magnitude, improving
    robustness to noise and spectral smearing at increased computational
    cost.
    """
    energy: Final[SpectralFluxMethod]
    """
    Energy-based spectral flux.

    Measures positive changes in spectral energy between successive frames.
    Performs well for transient and percussive onsets.
    """
    magnitude: Final[SpectralFluxMethod]
    """
    Magnitude-based spectral flux.

    Measures positive changes in spectral magnitude and is more sensitive
    to subtle spectral variation, making it effective for tonal material.
    """
    rectified_complex: Final[SpectralFluxMethod]
    """
    Rectified complex-domain spectral flux.

    Suppresses negative phase contributions to balance sensitivity and
    robustness.
    """

@final
class SpectrogramScale:
    """
    Scaling method for spectrogram magnitude and frequency representations.

    `SpectrogramScale` represents how spectral magnitudes or frequencies are
    mapped for analysis or visualisation. Different scaling approaches expose
    different structure in spectral content and are appropriate for different
    tasks.

    Instances of `SpectrogramScale` are immutable and should be treated as
    enum-like values. They are accessed via class attributes rather than being
    constructed directly.

    Available scales:

    - ``SpectrogramScale.linear``
    - ``SpectrogramScale.log``
    - ``SpectrogramScale.mel``
    """
    linear: Final[SpectrogramScale]
    """
    Linear power scale.

    Preserves absolute magnitude relationships and is most appropriate for
    quantitative analysis and energy measurements.
    """
    log: Final[SpectrogramScale]
    """
    Logarithmic (decibel) magnitude scale.

    Compresses dynamic range to improve visualisation of low-energy components
    alongside strong spectral peaks.
    """
    mel: Final[SpectrogramScale]
    """
    Mel-frequency scale.

    Applies a perceptually motivated nonlinear mapping of frequency designed
    to better approximate human auditory resolution.
    """
    @staticmethod
    def default() -> SpectrogramScale: ...

@final
class StereoConversionMethod:
    """
    Method for converting mono audio to stereo.

    `StereoConversionMethod` represents a strategy for expanding a mono signal
    into a two-channel stereo signal. Different methods control how the mono
    signal is distributed between the left and right channels.

    Instances of `StereoConversionMethod` are immutable and should be treated as
    enum-like values. Zero-parameter methods are accessed via class attributes,
    while parameterised methods are constructed via class methods.

    Available conversion methods:

    - ``StereoConversionMethod.duplicate``
    - ``StereoConversionMethod.pan(pan_value)``
    - ``StereoConversionMethod.left``
    - ``StereoConversionMethod.right``
    """
    duplicate: Final[StereoConversionMethod]
    """
    Duplicate mono signal to both left and right channels.
    """
    left: Final[StereoConversionMethod]
    """
    Use as the left channel, filling the right channel with silence.
    """
    right: Final[StereoConversionMethod]
    """
    Use as the right channel, filling the left channel with silence.
    """
    @staticmethod
    def default() -> StereoConversionMethod: ...
    @classmethod
    def pan(cls, /, pan_value: "float") -> StereoConversionMethod:
        """
        Pan the mono signal between left and right channels.

        A value of -1 places the signal fully in the left channel, 0 centres the
        signal, and 1 places it fully in the right channel.

        Parameters
        ----------
        ``pan_value`` : float
            Pan position in the range [-1, 1].
        """

@final
class ThreeBandEqConfig:
    """
    Three-band parametric equaliser configuration.

    `ThreeBandEqConfig` describes a simple three-band equaliser composed of a low
    shelf, a mid peaking band, and a high shelf. It is a convenience configuration
    for common tone-shaping tasks without constructing individual EQ bands.

    Instances of `ThreeBandEqConfig` are immutable value objects in the Python
    API. Parameters are provided at construction time and exposed via read-only
    properties. Validation can be performed explicitly using ``validate()``.

    A flat (unity-gain) preset is provided via ``ThreeBandEqConfig.flat``.
    """
    flat: Final[ThreeBandEqConfig]
    """
    Flat (unity-gain) three-band EQ preset.

    Low shelf at 200 Hz, mid peak at 1 kHz (Q = 1.0), high shelf at 4 kHz, all
    gains at 0 dB. A neutral starting point for further adjustment.
    """
    def __new__(cls, /, *, low_freq: "float", low_gain: "float", mid_freq: "float", mid_gain: "float", mid_q: "float", high_freq: "float", high_gain: "float") -> ThreeBandEqConfig:
        """
        Create a new three-band EQ configuration.

        Parameters
        ----------
        ``low_freq`` : float
            Low shelf corner frequency in Hz. Must be greater than 0.0 and less
            than ``mid_freq``.
        ``low_gain`` : float
            Low shelf gain in dB.
        ``mid_freq`` : float
            Mid peak centre frequency in Hz. Must be greater than ``low_freq`` and
            less than ``high_freq``.
        ``mid_gain`` : float
            Mid peak gain in dB.
        ``mid_q`` : float
            Mid peak Q factor. Must be greater than 0.0.
        ``high_freq`` : float
            High shelf corner frequency in Hz. Must be greater than ``mid_freq``.
        ``high_gain`` : float
            High shelf gain in dB.

        Returns
        -------
        ``ThreeBandEqConfig``
            A new three-band EQ configuration.
        """
    def __repr__(self, /) -> str: ...
    @staticmethod
    def default() -> ThreeBandEqConfig: ...
    @property
    def high_freq(self, /) -> float:
        """
        High shelf corner frequency in Hz.
        """
    @property
    def high_gain(self, /) -> float:
        """
        High shelf gain in dB.
        """
    @property
    def low_freq(self, /) -> float:
        """
        Low shelf corner frequency in Hz.
        """
    @property
    def low_gain(self, /) -> float:
        """
        Low shelf gain in dB.
        """
    @property
    def mid_freq(self, /) -> float:
        """
        Mid peak centre frequency in Hz.
        """
    @property
    def mid_gain(self, /) -> float:
        """
        Mid peak gain in dB.
        """
    @property
    def mid_q(self, /) -> float:
        """
        Mid peak Q factor.
        """
    def validate(self, /) -> ThreeBandEqConfig:
        """
        Validate the three-band EQ parameters.

        Returns
        -------
        ``ThreeBandEqConfig``
            The validated configuration.

        Raises
        ------
        ``ValueError``
            If any configuration parameter is invalid (non-positive frequency,
            mis-ordered frequencies, or non-positive mid Q).
        """

@final
class VadChannelPolicy:
    """
    Multi-channel handling policy for Voice Activity Detection (VAD).

    `VadChannelPolicy` defines how voice activity decisions are produced when the
    input audio contains multiple channels. Different policies determine whether
    channels are mixed, evaluated independently, or selected explicitly.

    Instances of `VadChannelPolicy` are immutable and should be treated as
    enum-like values. Zero-parameter policies are accessed via class attributes,
    while parameterised policies are constructed via class methods.

    Available policies:

    - ``VadChannelPolicy.average_to_mono``
    - ``VadChannelPolicy.any_channel``
    - ``VadChannelPolicy.all_channels``
    - ``VadChannelPolicy.channel(ch)``
    """
    all_channels: Final[VadChannelPolicy]
    """
    Run VAD independently on each channel and mark activity only if all channels are active.
    """
    any_channel: Final[VadChannelPolicy]
    """
    Run VAD independently on each channel and mark activity if any channel is active.
    """
    average_to_mono: Final[VadChannelPolicy]
    """
    Average all channels to a mono signal and run VAD once.
    """
    def __repr__(self, /) -> str: ...
    @classmethod
    def channel(cls, /, ch: "int") -> VadChannelPolicy:
        """
        Run VAD on a specific channel index.

        Parameters
        ----------
        ch : int
            Zero-based channel index to use for VAD.
        """
    @staticmethod
    def default() -> VadChannelPolicy: ...

@final
class VadConfig:
    """
    Configuration for Voice Activity Detection (VAD).

    `VadConfig` defines all parameters controlling frame-based voice activity
    detection. The VAD operates on overlapping frames of length ``frame_size``
    with step ``hop_size`` and produces a boolean decision per frame.

    Defaults are chosen to work reasonably well for general audio, but most
    applications should tune thresholds and timing parameters for their content
    and sample format.

    Instances are immutable once constructed. Use ``validate()`` to check that a
    configuration is internally consistent.

    A convenience preset is available via ``VadConfig.energy_only``.
    """
    energy_only: Final[VadConfig]
    """
    Create a configuration using only energy-based detection.

    This is a convenience preset equivalent to selecting
    ``VadMethod.energy`` with default parameters.
    """
    def __new__(cls, method: "VadMethod", channel_policy: "VadChannelPolicy", /, frame_size: "int", hop_size: "int", pad_end: "bool", energy_threshold_db: "float", zcr_min: "float", zcr_max: "float", min_speech_frames: "int", min_silence_frames: "int", hangover_frames: "int", smooth_frames: "int", speech_band_low_hz: "float", speech_band_high_hz: "float", spectral_ratio_threshold: "float") -> VadConfig:
        """
        Create a new VAD configuration.

        Parameters
        ----------
        method : ``VadMethod``
            Voice activity detection method to use (positional-only).
        ``channel_policy`` : ``VadChannelPolicy``
            Policy for handling multi-channel audio (positional-only, second).
        ``frame_size`` : int
            Frame size in samples.
        ``hop_size`` : int
            Hop size in samples (frame step). Must be less than or equal to
            ``frame_size``.
        ``pad_end`` : bool
            Whether to include a final partial frame padded with zeros.
        ``energy_threshold_db`` : float
            Energy threshold in dBFS (RMS). Typical values range from approximately
            -60.0 (very sensitive) to -30.0.
        ``zcr_min`` : float
            Minimum acceptable zero crossing rate, expressed as crossings per
            sample in the range [0, 1].
        ``zcr_max`` : float
            Maximum acceptable zero crossing rate, expressed as crossings per
            sample in the range [0, 1].
        ``min_speech_frames`` : int
            Minimum number of consecutive speech frames required to keep a speech
            region.
        ``min_silence_frames`` : int
            Minimum number of consecutive non-speech frames required to keep a
            silence region. Shorter silence gaps are filled as speech.
        ``hangover_frames`` : int
            Number of frames to keep speech active after energy drops.
        ``smooth_frames`` : int
            Majority-vote smoothing window in frames. A value of 1 disables
            smoothing.
        ``speech_band_low_hz`` : float
            Lower bound of the speech band in Hz (used by spectral VAD).
        ``speech_band_high_hz`` : float
            Upper bound of the speech band in Hz (used by spectral VAD).
        ``spectral_ratio_threshold`` : float
            Threshold on speech-band energy ratio (used by spectral VAD).

        Returns
        -------
        ``VadConfig``
            A new VAD configuration instance.

        Notes
        -----
        This constructor does not automatically validate parameter consistency.
        Call ``validate()`` to explicitly check constraints.
        """
    def __repr__(self, /) -> str: ...
    @property
    def channel_policy(self, /) -> VadChannelPolicy:
        """
        Policy for multi-channel audio.
        """
    @staticmethod
    def default() -> VadConfig: ...
    @property
    def energy_threshold_db(self, /) -> float:
        """
        Energy threshold in dBFS (frame RMS).
        """
    def explain(self, /, sample_rate: "float | None" = None, *, n_samples: "int | None" = None, n_channels: "int | None" = None, f0_range: "tuple[float, float] | None" = None) -> ExplainDoc:
        """
        Explain what this voice-activity-detection configuration will do.

        Args:

            sample_rate (float | None): Sample rate in Hz of the signal it
                will be applied to. Without it, quantities that need a
                sample rate are omitted and an ``info`` warning says so.
            n_samples (int | None): Signal length in samples per channel.
            n_channels (int | None): Number of channels.
            f0_range (tuple[float, float] | None): Expected
                fundamental-frequency range ``(lowest, highest)`` in Hz.

        Returns:

            ExplainDoc: The explanation (see ``audio_samples.explain``).
        """
    @property
    def frame_size(self, /) -> int:
        """
        Analysis frame length in samples.
        """
    @property
    def hangover_frames(self, /) -> int:
        """
        Hangover in frames: speech stays active this many frames after energy drops.
        """
    @property
    def hop_size(self, /) -> int:
        """
        Hop size in samples (frame step).
        """
    @property
    def method(self, /) -> VadMethod:
        """
        VAD method.
        """
    @property
    def min_silence_frames(self, /) -> int:
        """
        Minimum number of consecutive non-speech frames to keep a silence region.
        """
    @property
    def min_speech_frames(self, /) -> int:
        """
        Minimum number of consecutive speech frames to keep a speech region.
        """
    @property
    def pad_end(self, /) -> bool:
        """
        Whether to include a final partial frame (zero-padded).
        """
    @property
    def smooth_frames(self, /) -> int:
        """
        Majority-vote smoothing window in frames (1 = no smoothing).
        """
    @property
    def spectral_ratio_threshold(self, /) -> float:
        """
        Threshold on the speech-band energy ratio (``VadMethod.spectral``).
        """
    @property
    def speech_band_high_hz(self, /) -> float:
        """
        Upper bound of the speech band in Hz (``VadMethod.spectral``).
        """
    @property
    def speech_band_low_hz(self, /) -> float:
        """
        Lower bound of the speech band in Hz (``VadMethod.spectral``).
        """
    def validate(self, /) -> VadConfig:
        """
        Validate configuration parameters.

        Checks internal parameter consistency and value ranges. An exception is
        raised if any constraint is violated.

        Returns
        -------
        ``VadConfig``
            Returns the validated configuration if validation succeeds.

        Raises
        ------
        ``ValueError``
            If any configuration parameter is invalid or inconsistent.
        """
    @property
    def zcr_max(self, /) -> float:
        """
        Maximum zero-crossing rate, in crossings per sample (``[0, 1]``).
        """
    @property
    def zcr_min(self, /) -> float:
        """
        Minimum zero-crossing rate, in crossings per sample (``[0, 1]``).
        """

@final
class VadMethod:
    """
    Voice Activity Detection (VAD) method.

    `VadMethod` represents the algorithmic strategy used to detect regions of
    speech or activity within an audio signal. Different methods trade off
    computational cost, robustness to noise, and detection accuracy.

    Instances of `VadMethod` are immutable and should be treated as enum-like
    values. They are accessed via class attributes rather than being constructed
    directly.

    Available VAD methods:

    - ``VadMethod.energy``
    - ``VadMethod.zcr`` (zero crossing rate)
    - ``VadMethod.combined``
    - ``VadMethod.spectral``
    """
    combined: Final[VadMethod]
    """
    Combined energy and zero crossing rate detection.

    Integrates both energy and ZCR cues for more robust detection.
    """
    energy: Final[VadMethod]
    """
    Energy-based voice activity detection.

    Uses signal energy (e.g. RMS) and a threshold to detect activity.
    """
    spectral: Final[VadMethod]
    """
    Spectral-based detection (not implemented).

    Reserved for speech-band spectral detection. It is **not implemented**:
    ``voice_activity_mask`` / ``speech_regions`` raise an error when this
    method is selected. Use ``energy``, ``zcr`` or ``combined``.
    """
    zcr: Final[VadMethod]
    """
    Zero crossing rate (ZCR) based detection.

    Uses the rate of sign changes in the waveform as an indicator of activity.
    """
    def __repr__(self, /) -> str: ...
    @staticmethod
    def default() -> VadMethod: ...

@final
class WindowType:
    """
    Window function for spectral analysis and filtering.

    `WindowType` represents a predefined window function applied to signals prior
    to FFT-based analysis or filtering. Different window types trade off frequency
    resolution against spectral leakage and side-lobe suppression.

    Instances of `WindowType` are immutable and should be treated as enum-like
    values. They are accessed via class attributes rather than being constructed
    directly.

    The window types are:

    - ``WindowType.hanning``
    - ``WindowType.hamming``
    - ``WindowType.blackman``
    - ``WindowType.rectangular``
    """
    blackman: Final[WindowType]
    """
    Blackman window.

    Low spectral leakage at the cost of a wider main lobe and reduced
    frequency resolution.
    """
    hamming: Final[WindowType]
    """
    Hamming window.

    Similar to the Hanning window, with slightly different coefficients and
    side-lobe behaviour.
    """
    hanning: Final[WindowType]
    """
    Hanning window.

    Good general-purpose window with a balanced trade-off between frequency
    resolution and spectral leakage.
    """
    rectangular: Final[WindowType]
    """
    Rectangular window (no windowing).

    Best frequency resolution but high spectral leakage.
    """
    @staticmethod
    def default() -> WindowType: ...
    @classmethod
    def gaussian(cls, /, std: "float") -> WindowType:
        """
        Gaussian window.

        Smooth window with a parameter controlling the effective width of the
        window.

        Parameters
        ----------
        std : float
            Standard deviation controlling the width of the Gaussian envelope.
        """
    @classmethod
    def kaiser(cls, /, beta: "float") -> WindowType:
        """
        Kaiser window.

        Parameterised window providing a tunable trade-off between main-lobe width
        and side-lobe attenuation.

        Parameters
        ----------
        beta : float
            Controls the trade-off between frequency resolution and spectral
            leakage. Larger values increase side-lobe suppression at the cost of
            wider main lobes.
        """
