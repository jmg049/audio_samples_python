/// Types module for audio processing configurations and enumerations.
use std::hash::Hash;

// Types that stayed in audio_samples::operations::types
use audio_samples::operations::types::{
    AdaptiveThresholdConfig, AdaptiveThresholdMethod, CompressorConfig, DynamicRangeMethod, EqBand,
    EqBandType, FadeCurve, FilterResponse, IirFilterDesign, IirFilterType, KneeType, LimiterConfig,
    MonoConversionMethod, NoiseColor, PadSide, ParametricEq, PeakPickingConfig, PerturbationConfig,
    PerturbationMethod, PitchDetectionMethod, ResamplingQuality, SideChainConfig, SpectrogramScale,
    StereoConversionMethod, VadChannelPolicy, VadConfig, VadMethod,
};

// Types re-exported from operations submodules
use audio_samples::operations::{
    BeatTrackingConfig, BeatTrackingData, ComplexOnsetConfig, HpssConfig, OnsetDetectionConfig,
    SpectralFluxConfig, SpectralFluxMethod,
};

// Types from spectrograms crate
use audio_samples::operations::dynamic_range::EnvelopeFollower;
use audio_samples::{I24, NormalizationMethod, SampleType, nzu};
use numpy::{PyArrayDescr, PyArrayDescrMethods};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::PyType;
use spectrograms::python::{PyCqtParams, PyStftParams};
use spectrograms::{StftParams, WindowType};

use crate::{
    audio_err_to_py, impl_py_default_static, impl_py_repr, impl_py_wrapper_core,
    impl_py_wrapper_fromstr, nzu_or_err, reexport, register_types,
};

#[pymodule]
pub fn types(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    let types_mod = PyModule::new(py, "types")?;

    register_types!(
        py,
        m,
        types_mod,
        [
            (PyAdaptiveThresholdConfig, "AdaptiveThresholdConfig"),
            (PyAdaptiveThresholdMethod, "AdaptiveThresholdMethod"),
            (PyBeatTrackingConfig, "BeatTrackingConfig"),
            (PyBeatTrackingData, "BeatTrackingData"),
            (PyComplexOnsetConfig, "ComplexOnsetConfig"),
            (PyCompressorConfig, "CompressorConfig"),
            (PyOnsetDetectionConfig, "OnsetDetectionConfig"),
            (PyDynamicRangeMethod, "DynamicRangeMethod"),
            (PyEnvelopeFollower, "EnvelopeFollower"),
            (PyEqBand, "EqBand"),
            (PyEqBandType, "EqBandType"),
            (PyFilterResponse, "FilterResponse"),
            (PyIirFilterDesign, "IirFilterDesign"),
            (PyIirFilterType, "IirFilterType"),
            (PyKneeType, "KneeType"),
            (PyLimiterConfig, "LimiterConfig"),
            (PyMonoConversionMethod, "MonoConversionMethod"),
            (PyNormalizationMethod, "NormalizationMethod"),
            (PyPadSide, "PadSide"),
            (PyParametricEq, "ParametricEq"),
            (PyPeakPickingConfig, "PeakPickingConfig"),
            (PyPitchDetectionMethod, "PitchDetectionMethod"),
            (PySideChainConfig, "SideChainConfig"),
            (PySpectralFluxConfig, "SpectralFluxConfig"),
            (PySpectralFluxMethod, "SpectralFluxMethod"),
            (PySpectrogramScale, "SpectrogramScale"),
            (PyStereoConversionMethod, "StereoConversionMethod"),
            (PyVadChannelPolicy, "VadChannelPolicy"),
            (PyVadConfig, "VadConfig"),
            (PyVadMethod, "VadMethod"),
            (PyWindowType, "WindowType"),
            (PyFadeCurve, "FadeCurve"),
            (PySampleType, "SampleType"),
            (PyNoiseColor, "NoiseColor"),
            (PyPerturbationMethod, "PerturbationMethod"),
            (PyPerturbationConfig, "PerturbationConfig"),
            (PyResamplingQuality, "ResamplingQuality"),
        ]
    );

    m.add_submodule(&types_mod)?;
    Ok(())
}

/// Enumeration of supported audio sample data types.
///
/// `SampleType` describes the numeric representation used to store individual
/// audio samples. It is typically used when constructing or converting audio
/// buffers, configuring I/O, or selecting internal processing formats.
///
/// The available sample types are:
///
/// - ``SampleType.I16``:
///   Signed 16-bit integer samples. Common in PCM WAV files and efficient for
///   storage and I/O, but with limited dynamic range.
///
/// - ``SampleType.I24``:
///   Signed 24-bit integer samples. Higher dynamic range than 16-bit PCM and
///   widely used in professional audio pipelines.
///
/// - ``SampleType.I32``:
///   Signed 32-bit integer samples. Rare in interchange formats, but sometimes
///   used for intermediate or high-precision processing.
///
/// - ``SampleType.F32`` (default):
///   32-bit floating-point samples. The most common format for DSP and machine
///   learning workloads due to good numerical stability and performance.
///
/// - ``SampleType.F64``:
///   64-bit floating-point samples. Provides maximum numerical precision at the
///   cost of higher memory usage and lower throughput.
///
/// Instances of `SampleType` are immutable and comparable. They should be
/// treated as enum values rather than constructed directly.
#[pyclass(name = "SampleType", from_py_object, module = "audio_samples.types")]
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum PySampleType {
    U8,
    I16,
    I24,
    I32,
    #[default]
    F32,
    F64,
}

impl From<SampleType> for PySampleType {
    fn from(value: SampleType) -> Self {
        match value {
            SampleType::U8 => PySampleType::U8,
            SampleType::I16 => PySampleType::I16,
            SampleType::I24 => PySampleType::I24,
            SampleType::I32 => PySampleType::I32,
            SampleType::F32 => PySampleType::F32,
            SampleType::F64 => PySampleType::F64,
            _ => unreachable!("unknown SampleType variant"),
        }
    }
}

impl PySampleType {
    pub(crate) fn from_numpy(py: Python<'_>, dt: &Bound<'_, PyArrayDescr>) -> PyResult<Self> {
        if dt.is_equiv_to(&numpy::dtype::<u8>(py)) {
            Ok(Self::U8)
        } else if dt.is_equiv_to(&numpy::dtype::<i16>(py)) {
            Ok(Self::I16)
        } else if dt.is_equiv_to(&numpy::dtype::<I24>(py)) {
            Ok(Self::I24)
        } else if dt.is_equiv_to(&numpy::dtype::<i32>(py)) {
            Ok(Self::I32)
        } else if dt.is_equiv_to(&numpy::dtype::<f32>(py)) {
            Ok(Self::F32)
        } else if dt.is_equiv_to(&numpy::dtype::<f64>(py)) {
            Ok(Self::F64)
        } else {
            Err(PyTypeError::new_err("Unsupported dtype"))
        }
    }

    pub(crate) fn from_native(st: SampleType) -> Option<Self> {
        match st {
            SampleType::U8 => Some(Self::U8),
            SampleType::I16 => Some(Self::I16),
            SampleType::I24 => Some(Self::I24),
            SampleType::I32 => Some(Self::I32),
            SampleType::F32 => Some(Self::F32),
            SampleType::F64 => Some(Self::F64),
            _ => None,
        }
    }
}
/// Side for padding operations.
///
/// `PadSide` indicates which side of an audio signal should be padded when
/// applying padding operations. It is used to specify whether padding should be
/// added to the beginning (left) or end (right) of the signal.
#[pyclass(name = "PadSide", from_py_object, module = "audio_samples.types")]
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
pub struct PyPadSide {
    pub(crate) inner: PadSide,
}

#[pymethods]
impl PyPadSide {
    /// Pad on the left side (beginning) of the signal.
    #[classattr]
    fn left() -> Self {
        Self {
            inner: PadSide::Left,
        }
    }

    /// Pad on the right side (end) of the signal.
    #[classattr]
    fn right() -> Self {
        Self {
            inner: PadSide::Right,
        }
    }
}

impl_py_wrapper_core!(PyPadSide, PadSide);
impl_py_wrapper_fromstr!(PyPadSide, PadSide);
impl_py_default_static!(PyPadSide);
impl_py_repr!(PyPadSide);

/// Normalisation strategy for audio sample data.
///
/// `NormalizationMethod` represents a predefined method for rescaling or
/// re-centring audio samples prior to further processing. Normalisation is
/// commonly used to stabilise numerical behaviour, improve comparability
/// between signals, or enforce amplitude constraints.
///
/// Instances of `NormalizationMethod` are immutable and should be treated as
/// enum-like values. They are accessed via class attributes rather than being
/// constructed directly.
///
/// Available normalisation methods:
///
/// - ``NormalizationMethod.minmax``:
///   Linearly rescales samples so that the minimum maps to 0 and the maximum
///   maps to 1 (or -1 to 1, depending on context).
///
/// - ``NormalizationMethod.zscore``:
///   Applies z-score normalisation by subtracting the mean and dividing by the
///   standard deviation.
///
/// - ``NormalizationMethod.peak``:
///   Scales samples so that the maximum absolute amplitude equals 1.
///
/// - ``NormalizationMethod.mean``:
///   Recentres samples by subtracting the mean value.
///
/// - ``NormalizationMethod.median``:
///   Recentres samples by subtracting the median value.
///
/// Normalisation methods do not modify data eagerly; they describe how
/// normalisation should be applied by downstream operations.
#[pyclass(
    name = "NormalizationMethod",
    from_py_object,
    module = "audio_samples.types"
)]
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PyNormalizationMethod {
    pub(crate) inner: NormalizationMethod,
}
#[pymethods]
impl PyNormalizationMethod {
    /// Min--max normalisation.
    ///
    /// Linearly rescales samples so that the minimum and maximum values are
    /// mapped to a fixed range. This method preserves relative distances but
    /// is sensitive to outliers.
    #[classattr]
    fn minmax() -> Self {
        PyNormalizationMethod {
            inner: NormalizationMethod::MinMax,
        }
    }

    /// Z-score normalisation.
    ///
    /// Subtracts the mean of the samples and divides by their standard
    /// deviation. This produces zero-mean, unit-variance data and is commonly
    /// used in statistical and machine learning pipelines.
    #[classattr]
    fn zscore() -> Self {
        PyNormalizationMethod {
            inner: NormalizationMethod::ZScore,
        }
    }

    /// Peak normalisation.
    ///
    /// Scales samples so that the maximum absolute amplitude equals 1. This
    /// preserves waveform shape while enforcing a strict amplitude bound.
    #[classattr]
    fn peak() -> Self {
        PyNormalizationMethod {
            inner: NormalizationMethod::Peak,
        }
    }

    /// Mean normalisation.
    ///
    /// Recentres samples by subtracting the mean value without rescaling their
    /// variance.
    #[classattr]
    fn mean() -> Self {
        PyNormalizationMethod {
            inner: NormalizationMethod::Mean,
        }
    }

    /// Median normalisation.
    ///
    /// Recentres samples by subtracting the median value. This method is more
    /// robust to outliers than mean-based normalisation.
    #[classattr]
    fn median() -> Self {
        PyNormalizationMethod {
            inner: NormalizationMethod::Median,
        }
    }
}

impl_py_wrapper_core!(PyNormalizationMethod, NormalizationMethod);
impl_py_wrapper_fromstr!(PyNormalizationMethod, NormalizationMethod);
impl_py_default_static!(PyNormalizationMethod);
impl_py_repr!(PyNormalizationMethod);

/// Window function for spectral analysis and filtering.
///
/// `WindowType` represents a predefined window function applied to signals prior
/// to FFT-based analysis or filtering. Different window types trade off frequency
/// resolution against spectral leakage and side-lobe suppression.
///
/// Instances of `WindowType` are immutable and should be treated as enum-like
/// values. They are accessed via class attributes rather than being constructed
/// directly.
///
/// The following window types are currently exposed to Python:
///
/// - ``WindowType.hanning``
/// - ``WindowType.hamming``
/// - ``WindowType.blackman``
/// - ``WindowType.rectangular``
///
#[pyclass(name = "WindowType", from_py_object, module = "audio_samples.types")]
#[derive(Default, Debug, Clone, PartialEq)]
pub struct PyWindowType {
    pub(crate) inner: WindowType,
}

#[pymethods]
impl PyWindowType {
    /// Hanning window.
    ///
    /// Good general-purpose window with a balanced trade-off between frequency
    /// resolution and spectral leakage.
    #[classattr]
    fn hanning() -> Self {
        PyWindowType {
            inner: WindowType::Hanning,
        }
    }

    /// Hamming window.
    ///
    /// Similar to the Hanning window, with slightly different coefficients and
    /// side-lobe behaviour.
    #[classattr]
    fn hamming() -> Self {
        PyWindowType {
            inner: WindowType::Hamming,
        }
    }

    /// Blackman window.
    ///
    /// Low spectral leakage at the cost of a wider main lobe and reduced
    /// frequency resolution.
    #[classattr]
    fn blackman() -> Self {
        PyWindowType {
            inner: WindowType::Blackman,
        }
    }

    /// Rectangular window (no windowing).
    ///
    /// Best frequency resolution but high spectral leakage.
    #[classattr]
    fn rectangular() -> Self {
        PyWindowType {
            inner: WindowType::Rectangular,
        }
    }

    /// Kaiser window.
    ///
    /// Parameterised window providing a tunable trade-off between main-lobe width
    /// and side-lobe attenuation.
    ///
    /// Parameters
    /// ----------
    /// beta : float
    ///     Controls the trade-off between frequency resolution and spectral
    ///     leakage. Larger values increase side-lobe suppression at the cost of
    ///     wider main lobes.
    #[classmethod]
    #[pyo3(signature = (beta: "float"), text_signature="")]
    fn kaiser(_cls: &Bound<'_, PyType>, beta: f64) -> Self {
        PyWindowType {
            inner: WindowType::Kaiser { beta },
        }
    }

    /// Gaussian window.
    ///
    /// Smooth window with a parameter controlling the effective width of the
    /// window.
    ///
    /// Parameters
    /// ----------
    /// std : float
    ///     Standard deviation controlling the width of the Gaussian envelope.
    #[classmethod]
    #[pyo3(signature = (std: "float"), text_signature="")]
    fn gaussian(_cls: &Bound<'_, PyType>, std: f64) -> Self {
        PyWindowType {
            inner: WindowType::Gaussian { std },
        }
    }
}

impl_py_wrapper_core!(PyWindowType, WindowType);
impl_py_wrapper_fromstr!(PyWindowType, WindowType);
impl_py_default_static!(PyWindowType);
impl_py_repr!(PyWindowType);

/// Fade curve shape for envelope operations.
///
/// `FadeCurve` represents the shape of an amplitude envelope used when applying
/// fades, ramps, or transitions to audio signals. Different curves produce
/// different perceptual characteristics in how the signal level changes over
/// time.
///
/// Instances of `FadeCurve` are immutable and should be treated as enum-like
/// values. They are accessed via class attributes rather than being constructed
/// directly.
///
/// Available fade curves:
///
/// - ``FadeCurve.linear``
/// - ``FadeCurve.exponential``
/// - ``FadeCurve.logarithmic``
/// - ``FadeCurve.smooth_step``
#[pyclass(name = "FadeCurve", from_py_object, module = "audio_samples.types")]
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
pub struct PyFadeCurve {
    pub(crate) inner: FadeCurve,
}

#[pymethods]
impl PyFadeCurve {
    /// Linear fade.
    ///
    /// Constant rate of change over time.
    #[classattr]
    fn linear() -> Self {
        PyFadeCurve {
            inner: FadeCurve::Linear,
        }
    }

    /// Exponential fade.
    ///
    /// Faster change at the beginning, slower towards the end.
    #[classattr]
    fn exponential() -> Self {
        PyFadeCurve {
            inner: FadeCurve::Exponential,
        }
    }

    /// Logarithmic fade.
    ///
    /// Slower change at the beginning, faster towards the end.
    #[classattr]
    fn logarithmic() -> Self {
        PyFadeCurve {
            inner: FadeCurve::Logarithmic,
        }
    }

    /// Smooth-step fade.
    ///
    /// S-shaped curve with smooth transitions at both the start and end.
    #[classattr]
    fn smooth_step() -> Self {
        PyFadeCurve {
            inner: FadeCurve::SmoothStep,
        }
    }
}

impl_py_wrapper_core!(PyFadeCurve, FadeCurve);
impl_py_wrapper_fromstr!(PyFadeCurve, FadeCurve);
impl_py_default_static!(PyFadeCurve);
impl_py_repr!(PyFadeCurve);

/// Voice Activity Detection (VAD) method.
///
/// `VadMethod` represents the algorithmic strategy used to detect regions of
/// speech or activity within an audio signal. Different methods trade off
/// computational cost, robustness to noise, and detection accuracy.
///
/// Instances of `VadMethod` are immutable and should be treated as enum-like
/// values. They are accessed via class attributes rather than being constructed
/// directly.
///
/// Available VAD methods:
///
/// - ``VadMethod.energy``
/// - ``VadMethod.zcr`` (zero crossing rate)
/// - ``VadMethod.combined``
/// - ``VadMethod.spectral``
#[pyclass(name = "VadMethod", from_py_object, module = "audio_samples.types")]
#[derive(Default, Debug, Clone, Copy, PartialEq)]
pub struct PyVadMethod {
    pub(crate) inner: VadMethod,
}

#[pymethods]
impl PyVadMethod {
    /// Energy-based voice activity detection.
    ///
    /// Uses signal energy (e.g. RMS) and a threshold to detect activity.
    #[classattr]
    fn energy() -> Self {
        PyVadMethod {
            inner: VadMethod::Energy,
        }
    }

    /// Zero crossing rate (ZCR) based detection.
    ///
    /// Uses the rate of sign changes in the waveform as an indicator of activity.
    #[classattr]
    fn zcr() -> Self {
        PyVadMethod {
            inner: VadMethod::ZeroCrossing,
        }
    }

    /// Combined energy and zero crossing rate detection.
    ///
    /// Integrates both energy and ZCR cues for more robust detection.
    #[classattr]
    fn combined() -> Self {
        PyVadMethod {
            inner: VadMethod::Combined,
        }
    }

    /// Spectral-based detection.
    ///
    /// Uses spectral features for voice activity detection, typically providing
    /// improved robustness in noisy conditions at higher computational cost.
    #[classattr]
    fn spectral() -> Self {
        PyVadMethod {
            inner: VadMethod::Spectral,
        }
    }
}

impl_py_wrapper_core!(PyVadMethod, VadMethod);
impl_py_wrapper_fromstr!(PyVadMethod, VadMethod);
impl_py_default_static!(PyVadMethod);
impl_py_repr!(PyVadMethod);

/// Multi-channel handling policy for Voice Activity Detection (VAD).
///
/// `VadChannelPolicy` defines how voice activity decisions are produced when the
/// input audio contains multiple channels. Different policies determine whether
/// channels are mixed, evaluated independently, or selected explicitly.
///
/// Instances of `VadChannelPolicy` are immutable and should be treated as
/// enum-like values. Zero-parameter policies are accessed via class attributes,
/// while parameterised policies are constructed via class methods.
///
/// Available policies:
///
/// - ``VadChannelPolicy.average_to_mono``
/// - ``VadChannelPolicy.any_channel``
/// - ``VadChannelPolicy.all_channels``
/// - ``VadChannelPolicy.channel(ch)``
#[pyclass(
    name = "VadChannelPolicy",
    from_py_object,
    module = "audio_samples.types"
)]
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
pub struct PyVadChannelPolicy {
    pub(crate) inner: VadChannelPolicy,
}

#[pymethods]
impl PyVadChannelPolicy {
    /// Average all channels to a mono signal and run VAD once.
    #[classattr]
    fn average_to_mono() -> Self {
        PyVadChannelPolicy {
            inner: VadChannelPolicy::AverageToMono,
        }
    }

    /// Run VAD independently on each channel and mark activity if any channel is active.
    #[classattr]
    fn any_channel() -> Self {
        PyVadChannelPolicy {
            inner: VadChannelPolicy::AnyChannel,
        }
    }

    /// Run VAD independently on each channel and mark activity only if all channels are active.
    #[classattr]
    fn all_channels() -> Self {
        PyVadChannelPolicy {
            inner: VadChannelPolicy::AllChannels,
        }
    }

    /// Run VAD on a specific channel index.
    ///
    /// Parameters
    /// ----------
    /// ch : int
    ///     Zero-based channel index to use for VAD.
    #[classmethod]
    #[pyo3(signature = (ch: "int"), text_signature="($cls, ch: int) -> VadChannelPolicy")]
    fn channel(_cls: &Bound<'_, PyType>, ch: usize) -> Self {
        PyVadChannelPolicy {
            inner: VadChannelPolicy::Channel(ch),
        }
    }
}

impl_py_wrapper_core!(PyVadChannelPolicy, VadChannelPolicy);
impl_py_default_static!(PyVadChannelPolicy);
impl_py_repr!(PyVadChannelPolicy);

/// Configuration for Voice Activity Detection (VAD).
///
/// `VadConfig` defines all parameters controlling frame-based voice activity
/// detection. The VAD operates on overlapping frames of length ``frame_size``
/// with step ``hop_size`` and produces a boolean decision per frame.
///
/// Defaults are chosen to work reasonably well for general audio, but most
/// applications should tune thresholds and timing parameters for their content
/// and sample format.
///
/// Instances are immutable once constructed. Use ``validate()`` to check that a
/// configuration is internally consistent.
///
/// A convenience preset is available via ``VadConfig.energy_only``.
#[pyclass(name = "VadConfig", from_py_object, module = "audio_samples.types")]
#[derive(Default, Debug, Clone, Copy, PartialEq)]
pub struct PyVadConfig {
    pub(crate) inner: VadConfig,
}

#[pymethods]
impl PyVadConfig {
    /// Create a new VAD configuration.
    ///
    /// Parameters
    /// ----------
    /// method : VadMethod
    ///     Voice activity detection method to use.
    /// frame_size : int
    ///     Frame size in samples.
    /// hop_size : int
    ///     Hop size in samples (frame step). Must be less than or equal to
    ///     ``frame_size``.
    /// pad_end : bool
    ///     Whether to include a final partial frame padded with zeros.
    /// channel_policy : VadChannelPolicy
    ///     Policy for handling multi-channel audio.
    /// energy_threshold_db : float
    ///     Energy threshold in dBFS (RMS). Typical values range from approximately
    ///     -60.0 (very sensitive) to -30.0.
    /// zcr_min : float
    ///     Minimum acceptable zero crossing rate, expressed as crossings per
    ///     sample in the range [0, 1].
    /// zcr_max : float
    ///     Maximum acceptable zero crossing rate, expressed as crossings per
    ///     sample in the range [0, 1].
    /// min_speech_frames : int
    ///     Minimum number of consecutive speech frames required to keep a speech
    ///     region.
    /// min_silence_frames : int
    ///     Minimum number of consecutive non-speech frames required to keep a
    ///     silence region. Shorter silence gaps are filled as speech.
    /// hangover_frames : int
    ///     Number of frames to keep speech active after energy drops.
    /// smooth_frames : int
    ///     Majority-vote smoothing window in frames. A value of 1 disables
    ///     smoothing.
    /// speech_band_low_hz : float
    ///     Lower bound of the speech band in Hz (used by spectral VAD).
    /// speech_band_high_hz : float
    ///     Upper bound of the speech band in Hz (used by spectral VAD).
    /// spectral_ratio_threshold : float
    ///     Threshold on speech-band energy ratio (used by spectral VAD).
    ///
    /// Returns
    /// -------
    /// VadConfig
    ///     A new VAD configuration instance.
    ///
    /// Notes
    /// -----
    /// This constructor does not automatically validate parameter consistency.
    /// Call ``validate()`` to explicitly check constraints.
    #[new]
    #[pyo3(signature = (method: "VadMethod", channel_policy: "VadChannelPolicy", /, frame_size: "int", hop_size: "int", pad_end: "bool", energy_threshold_db: "float", zcr_min: "float", zcr_max: "float", min_speech_frames: "int", min_silence_frames: "int", hangover_frames: "int", smooth_frames: "int", speech_band_low_hz: "float", speech_band_high_hz: "float", spectral_ratio_threshold: "float"), text_signature="($cls, method: VadMethod, frame_size: int, hop_size: int, pad_end: bool, channel_policy: VadChannelPolicy, energy_threshold_db: float, zcr_min: float, zcr_max: float, min_speech_frames: int, min_silence_frames: int, hangover_frames: int, smooth_frames: int, speech_band_low_hz: float, speech_band_high_hz: float, spectral_ratio_threshold: float) -> VadConfig")]
    fn new(
        method: PyVadMethod,
        channel_policy: PyVadChannelPolicy,
        frame_size: usize,
        hop_size: usize,
        pad_end: bool,
        energy_threshold_db: f64,
        zcr_min: f64,
        zcr_max: f64,
        min_speech_frames: usize,
        min_silence_frames: usize,
        hangover_frames: usize,
        smooth_frames: usize,
        speech_band_low_hz: f64,
        speech_band_high_hz: f64,
        spectral_ratio_threshold: f64,
    ) -> PyResult<Self> {
        Ok(PyVadConfig {
            inner: VadConfig::new(
                method.inner,
                nzu_or_err(frame_size)?,
                nzu_or_err(hop_size)?,
                pad_end,
                channel_policy.inner,
                energy_threshold_db,
                zcr_min,
                zcr_max,
                min_speech_frames,
                min_silence_frames,
                nzu_or_err(hangover_frames)?,
                nzu_or_err(smooth_frames)?,
                speech_band_low_hz,
                speech_band_high_hz,
                spectral_ratio_threshold,
            ),
        })
    }

    /// Create a configuration using only energy-based detection.
    ///
    /// This is a convenience preset equivalent to selecting
    /// ``VadMethod.energy`` with default parameters.
    #[classattr]
    fn energy_only() -> Self {
        PyVadConfig {
            inner: VadConfig::energy_only(),
        }
    }

    /// Validate configuration parameters.
    ///
    /// Checks internal parameter consistency and value ranges. An exception is
    /// raised if any constraint is violated.
    ///
    /// Returns
    /// -------
    /// VadConfig
    ///     Returns the validated configuration if validation succeeds.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If any configuration parameter is invalid or inconsistent.
    #[pyo3(signature = (), text_signature="($self) -> VadConfig")]
    fn validate(&mut self) -> PyResult<Self> {
        self.inner.validate().map_err(audio_err_to_py)?;
        Ok(Self { inner: self.inner })
    }
}

impl_py_wrapper_core!(PyVadConfig, VadConfig);
impl_py_default_static!(PyVadConfig);
impl_py_repr!(PyVadConfig);

/// Quality level for resampling operations.
///
/// `ResamplingQuality` represents a discrete trade-off between computational
/// cost, latency, and signal fidelity when resampling audio. Higher quality
/// levels provide improved spectral accuracy and reduced aliasing at the cost
/// of increased computation.
///
/// Instances of `ResamplingQuality` are immutable and should be treated as
/// enum-like values. They are accessed via class attributes rather than being
/// constructed directly.
///
/// Available quality levels:
///
/// - ``ResamplingQuality.fast``
/// - ``ResamplingQuality.medium``
/// - ``ResamplingQuality.high``
#[pyclass(
    name = "ResamplingQuality",
    from_py_object,
    module = "audio_samples.types"
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct PyResamplingQuality {
    pub(crate) inner: ResamplingQuality,
}

impl PyResamplingQuality {
    #[inline]
    pub fn resample_fast() -> Self {
        Self {
            inner: ResamplingQuality::Fast,
        }
    }

    #[inline]
    pub fn resample_mediuem() -> Self {
        Self {
            inner: ResamplingQuality::Medium,
        }
    }

    #[inline]
    pub fn resample_high() -> Self {
        Self {
            inner: ResamplingQuality::High,
        }
    }
}

#[pymethods]
impl PyResamplingQuality {
    /// Fast resampling.
    ///
    /// Prioritises throughput and low latency at the cost of reduced spectral
    /// fidelity and increased aliasing.
    #[classattr]
    fn fast() -> Self {
        PyResamplingQuality {
            inner: ResamplingQuality::Fast,
        }
    }

    /// Medium quality resampling.
    ///
    /// Balanced trade-off between computational cost and signal quality.
    #[classattr]
    fn medium() -> Self {
        PyResamplingQuality {
            inner: ResamplingQuality::Medium,
        }
    }

    /// High quality resampling.
    ///
    /// Prioritises spectral fidelity and phase stability at the cost of higher
    /// computational cost and latency.
    #[classattr]
    fn high() -> Self {
        PyResamplingQuality {
            inner: ResamplingQuality::High,
        }
    }
}

impl_py_wrapper_core!(PyResamplingQuality, ResamplingQuality);
impl_py_wrapper_fromstr!(PyResamplingQuality, ResamplingQuality);
impl_py_default_static!(PyResamplingQuality);
impl_py_repr!(PyResamplingQuality);

/// Scaling method for spectrogram magnitude and frequency representations.
///
/// `SpectrogramScale` represents how spectral magnitudes or frequencies are
/// mapped for analysis or visualisation. Different scaling approaches expose
/// different structure in spectral content and are appropriate for different
/// tasks.
///
/// Instances of `SpectrogramScale` are immutable and should be treated as
/// enum-like values. They are accessed via class attributes rather than being
/// constructed directly.
///
/// Available scales:
///
/// - ``SpectrogramScale.linear``
/// - ``SpectrogramScale.log``
/// - ``SpectrogramScale.mel``
#[pyclass(
    name = "SpectrogramScale",
    from_py_object,
    module = "audio_samples.types"
)]
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PySpectrogramScale {
    pub(crate) inner: SpectrogramScale,
}

#[pymethods]
impl PySpectrogramScale {
    /// Linear power scale.
    ///
    /// Preserves absolute magnitude relationships and is most appropriate for
    /// quantitative analysis and energy measurements.
    #[classattr]
    fn linear() -> Self {
        PySpectrogramScale {
            inner: SpectrogramScale::Linear,
        }
    }

    /// Logarithmic (decibel) magnitude scale.
    ///
    /// Compresses dynamic range to improve visualisation of low-energy components
    /// alongside strong spectral peaks.
    #[classattr]
    fn log() -> Self {
        PySpectrogramScale {
            inner: SpectrogramScale::Log,
        }
    }

    /// Mel-frequency scale.
    ///
    /// Applies a perceptually motivated nonlinear mapping of frequency designed
    /// to better approximate human auditory resolution.
    #[classattr]
    fn mel() -> Self {
        PySpectrogramScale {
            inner: SpectrogramScale::Mel,
        }
    }
}

impl_py_wrapper_core!(PySpectrogramScale, SpectrogramScale);
impl_py_wrapper_fromstr!(PySpectrogramScale, SpectrogramScale);
impl_py_default_static!(PySpectrogramScale);
impl_py_repr!(PySpectrogramScale);

/// Method for converting multi-channel audio to mono.
///
/// `MonoConversionMethod` represents a strategy for collapsing multi-channel
/// audio (e.g. stereo or surround) into a single mono channel. Different methods
/// trade off simplicity, spatial fidelity, and control over channel weighting.
///
/// Instances of `MonoConversionMethod` are immutable and should be treated as
/// value objects constructed via class methods.
///
/// Available conversion methods:
///
/// - ``MonoConversionMethod.average()``
/// - ``MonoConversionMethod.left()``
/// - ``MonoConversionMethod.right()``
/// - ``MonoConversionMethod.weighted(weights: list[float])``
/// - ``MonoConversionMethod.center()``
#[pyclass(
    name = "MonoConversionMethod",
    from_py_object,
    module = "audio_samples.types"
)]
#[derive(Default, Debug, Clone, PartialEq)]
pub struct PyMonoConversionMethod {
    pub(crate) inner: MonoConversionMethod,
}

#[pymethods]
impl PyMonoConversionMethod {
    /// Average all channels equally.
    ///
    /// Each input channel contributes equally to the output mono signal.
    #[classattr]
    fn average() -> Self {
        PyMonoConversionMethod {
            inner: MonoConversionMethod::Average,
        }
    }

    /// Use the left channel only.
    ///
    /// Intended primarily for stereo input.
    #[classattr]
    fn left() -> Self {
        PyMonoConversionMethod {
            inner: MonoConversionMethod::Left,
        }
    }

    /// Use the right channel only.
    ///
    /// Intended primarily for stereo input.
    #[classattr]
    fn right() -> Self {
        PyMonoConversionMethod {
            inner: MonoConversionMethod::Right,
        }
    }

    /// Weighted average across channels.
    ///
    /// Each channel is multiplied by a corresponding weight prior to summation.
    ///
    /// Parameters
    /// ----------
    /// weights : list[float]
    ///     Per-channel weights. The length should match the number of input
    ///     channels.
    #[classmethod]
    #[pyo3(signature = (weights: "list[float]"), text_signature="($cls, weights: list[float]) -> MonoConversionMethod")]
    fn weighted(_cls: &Bound<'_, PyType>, weights: Vec<f64>) -> Self {
        PyMonoConversionMethod {
            inner: MonoConversionMethod::Weighted(weights),
        }
    }

    /// Use the centre channel if available, otherwise average left and right.
    #[classattr]
    fn center() -> Self {
        PyMonoConversionMethod {
            inner: MonoConversionMethod::Center,
        }
    }
}

impl_py_wrapper_core!(PyMonoConversionMethod, MonoConversionMethod);
impl_py_default_static!(PyMonoConversionMethod);
impl_py_repr!(PyMonoConversionMethod);

/// Method for converting mono audio to stereo.
///
/// `StereoConversionMethod` represents a strategy for expanding a mono signal
/// into a two-channel stereo signal. Different methods control how the mono
/// signal is distributed between the left and right channels.
///
/// Instances of `StereoConversionMethod` are immutable and should be treated as
/// enum-like values. Zero-parameter methods are accessed via class attributes,
/// while parameterised methods are constructed via class methods.
///
/// Available conversion methods:
///
/// - ``StereoConversionMethod.duplicate``
/// - ``StereoConversionMethod.pan(pan_value)``
/// - ``StereoConversionMethod.left``
/// - ``StereoConversionMethod.right``
#[pyclass(
    name = "StereoConversionMethod",
    from_py_object,
    module = "audio_samples.types"
)]
#[derive(Default, Debug, Clone, Copy, PartialEq)]
pub struct PyStereoConversionMethod {
    pub(crate) inner: StereoConversionMethod,
}

#[pymethods]
impl PyStereoConversionMethod {
    /// Duplicate mono signal to both left and right channels.
    #[classattr]
    fn duplicate() -> Self {
        PyStereoConversionMethod {
            inner: StereoConversionMethod::Duplicate,
        }
    }

    /// Pan the mono signal between left and right channels.
    ///
    /// A value of -1 places the signal fully in the left channel, 0 centres the
    /// signal, and 1 places it fully in the right channel.
    ///
    /// Parameters
    /// ----------
    /// pan_value : float
    ///     Pan position in the range [-1, 1].
    #[classmethod]
    #[pyo3(signature = (pan_value: "float"), text_signature="($cls, pan_value: float) -> StereoConversionMethod")]
    fn pan(_cls: &Bound<'_, PyType>, pan_value: f64) -> Self {
        PyStereoConversionMethod {
            inner: StereoConversionMethod::Pan(pan_value),
        }
    }

    /// Use as the left channel, filling the right channel with silence.
    #[classattr]
    fn left() -> Self {
        PyStereoConversionMethod {
            inner: StereoConversionMethod::Left,
        }
    }

    /// Use as the right channel, filling the left channel with silence.
    #[classattr]
    fn right() -> Self {
        PyStereoConversionMethod {
            inner: StereoConversionMethod::Right,
        }
    }
}

impl_py_wrapper_core!(PyStereoConversionMethod, StereoConversionMethod);
impl_py_default_static!(PyStereoConversionMethod);
impl_py_repr!(PyStereoConversionMethod);

/// Pitch detection algorithm selection.
///
/// `PitchDetectionMethod` selects the algorithm used to estimate the fundamental
/// frequency of a signal. Different algorithms trade off accuracy, robustness to
/// noise and inharmonicity, latency, and computational cost.
///
/// Instances of `PitchDetectionMethod` are immutable and should be treated as
/// enum-like values. They are accessed via class attributes rather than being
/// constructed directly.
///
/// Available methods:
///
/// - ``PitchDetectionMethod.yin``
/// - ``PitchDetectionMethod.autocorrelation``
/// - ``PitchDetectionMethod.cepstrum``
/// - ``PitchDetectionMethod.harmonic_product``
#[pyclass(
    name = "PitchDetectionMethod",
    from_py_object,
    module = "audio_samples.types"
)]
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PyPitchDetectionMethod {
    pub(crate) inner: PitchDetectionMethod,
}

impl PyPitchDetectionMethod {
    #[inline]
    pub fn detect_yin() -> Self {
        Self {
            inner: PitchDetectionMethod::Yin,
        }
    }
    #[inline]
    pub fn detect_autocorrelation() -> Self {
        Self {
            inner: PitchDetectionMethod::Autocorrelation,
        }
    }
    #[inline]
    pub fn detect_cepstrum() -> Self {
        Self {
            inner: PitchDetectionMethod::Cepstrum,
        }
    }
    #[inline]
    pub fn detect_harmonic_product() -> Self {
        Self {
            inner: PitchDetectionMethod::HarmonicProduct,
        }
    }
}

#[pymethods]
impl PyPitchDetectionMethod {
    /// YIN pitch detection algorithm.
    ///
    /// Provides robust and accurate fundamental frequency estimation for both
    /// speech and musical signals at moderate computational cost.
    #[classattr]
    fn yin() -> Self {
        PyPitchDetectionMethod {
            inner: PitchDetectionMethod::Yin,
        }
    }

    /// Autocorrelation-based pitch detection.
    ///
    /// Simple and fast, but sensitive to noise and octave errors for complex or
    /// weakly periodic signals.
    #[classattr]
    fn autocorrelation() -> Self {
        PyPitchDetectionMethod {
            inner: PitchDetectionMethod::Autocorrelation,
        }
    }

    /// Cepstral pitch detection.
    ///
    /// Operates in the frequency domain and performs well for voiced speech, but
    /// can degrade for dense harmonic or noisy spectra.
    #[classattr]
    fn cepstrum() -> Self {
        PyPitchDetectionMethod {
            inner: PitchDetectionMethod::Cepstrum,
        }
    }

    /// Harmonic Product Spectrum (HPS).
    ///
    /// Emphasises harmonic structure and is well-suited to musical signals with
    /// strong harmonic content.
    #[classattr]
    fn harmonic_product() -> Self {
        PyPitchDetectionMethod {
            inner: PitchDetectionMethod::HarmonicProduct,
        }
    }
}

impl_py_wrapper_core!(PyPitchDetectionMethod, PitchDetectionMethod);
impl_py_wrapper_fromstr!(PyPitchDetectionMethod, PitchDetectionMethod);
impl_py_default_static!(PyPitchDetectionMethod);
impl_py_repr!(PyPitchDetectionMethod);

/// IIR filter family selection for digital signal processing.
///
/// `IirFilterType` selects the analogue prototype used when designing an
/// infinite impulse response (IIR) digital filter. Different families trade off
/// passband ripple, stopband attenuation, transition sharpness, and phase
/// behaviour.
///
/// Instances of `IirFilterType` are immutable and should be treated as enum-like
/// values. They are accessed via class attributes rather than being constructed
/// directly.
///
/// Available filter types:
///
/// - ``IirFilterType.butterworth``
/// - ``IirFilterType.chebyshev_type_i``
/// - ``IirFilterType.chebyshev_type_ii``
/// - ``IirFilterType.elliptic``
#[pyclass(name = "IirFilterType", from_py_object, module = "audio_samples.types")]
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PyIirFilterType {
    pub(crate) inner: IirFilterType,
}

#[pymethods]
impl PyIirFilterType {
    /// Butterworth filter.
    ///
    /// Maximally flat passband with monotonic magnitude response and no ripple.
    /// Provides smooth behaviour and predictable phase characteristics at the
    /// cost of wider transition bands.
    #[classattr]
    fn butterworth() -> Self {
        PyIirFilterType {
            inner: IirFilterType::Butterworth,
        }
    }

    /// Chebyshev Type I filter.
    ///
    /// Introduces controlled ripple in the passband to achieve a sharper
    /// transition region than Butterworth designs.
    #[classattr]
    fn chebyshev_type_i() -> Self {
        PyIirFilterType {
            inner: IirFilterType::ChebyshevI,
        }
    }

    /// Chebyshev Type II filter.
    ///
    /// Introduces ripple in the stopband while preserving a monotonic passband
    /// response, allowing sharper transitions than Butterworth designs.
    #[classattr]
    fn chebyshev_type_ii() -> Self {
        PyIirFilterType {
            inner: IirFilterType::ChebyshevII,
        }
    }

    /// Elliptic (Cauer) filter.
    ///
    /// Introduces ripple in both passband and stopband, yielding the steepest
    /// transition region for a given filter order.
    #[classattr]
    fn elliptic() -> Self {
        PyIirFilterType {
            inner: IirFilterType::Elliptic,
        }
    }
}

impl_py_wrapper_core!(PyIirFilterType, IirFilterType);
impl_py_wrapper_fromstr!(PyIirFilterType, IirFilterType);
impl_py_default_static!(PyIirFilterType);
impl_py_repr!(PyIirFilterType);

/// Filter response characteristic.
///
/// `FilterResponse` defines the qualitative frequency response shape of a filter,
/// such as whether it attenuates low frequencies, high frequencies, or a band of
/// frequencies.
///
/// Instances of `FilterResponse` are immutable and should be treated as
/// enum-like values. They are accessed via class attributes rather than being
/// constructed directly.
///
/// Available responses:
///
/// - ``FilterResponse.lowpass``
/// - ``FilterResponse.highpass``
/// - ``FilterResponse.bandpass``
/// - ``FilterResponse.bandstop``
#[pyclass(
    name = "FilterResponse",
    from_py_object,
    module = "audio_samples.types"
)]
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
pub struct PyFilterResponse {
    pub(crate) inner: FilterResponse,
}

#[pymethods]
impl PyFilterResponse {
    /// Low-pass filter response.
    ///
    /// Attenuates frequencies above the cutoff frequency while preserving
    /// lower-frequency components.
    #[classattr]
    fn lowpass() -> Self {
        PyFilterResponse {
            inner: FilterResponse::LowPass,
        }
    }

    /// High-pass filter response.
    ///
    /// Attenuates frequencies below the cutoff frequency while preserving
    /// higher-frequency components.
    #[classattr]
    fn highpass() -> Self {
        PyFilterResponse {
            inner: FilterResponse::HighPass,
        }
    }

    /// Band-pass filter response.
    ///
    /// Preserves frequencies within a specified band while attenuating
    /// frequencies outside that range.
    #[classattr]
    fn bandpass() -> Self {
        PyFilterResponse {
            inner: FilterResponse::BandPass,
        }
    }

    /// Band-stop (notch) filter response.
    ///
    /// Attenuates frequencies within a specified band while preserving
    /// frequencies outside that range.
    #[classattr]
    fn bandstop() -> Self {
        PyFilterResponse {
            inner: FilterResponse::BandStop,
        }
    }
}

impl_py_wrapper_core!(PyFilterResponse, FilterResponse);
impl_py_wrapper_fromstr!(PyFilterResponse, FilterResponse);
impl_py_default_static!(PyFilterResponse);
impl_py_repr!(PyFilterResponse);

/// IIR filter design parameters.
///
/// `IirFilterDesign` describes the parameters required to construct a digital
/// infinite impulse response (IIR) filter, including the filter family, response
/// type, order, and frequency specifications.
///
/// Instances of `IirFilterDesign` are immutable value objects. Fields are exposed
/// via read-only properties.
///
/// A generic constructor is provided for manual configuration, and convenience
/// constructors are available for common Butterworth designs.
#[pyclass(
    name = "IirFilterDesign",
    from_py_object,
    module = "audio_samples.types"
)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PyIirFilterDesign {
    pub(crate) inner: IirFilterDesign,
}

#[pymethods]
impl PyIirFilterDesign {
    /// Create a new IIR filter design.
    ///
    /// This constructor provides a generic configuration interface. Depending on
    /// the selected filter type and response, only a subset of parameters may be
    /// meaningful.
    ///
    /// Parameters
    /// ----------
    /// filter_type : IirFilterType
    ///     IIR filter family (e.g. Butterworth, Chebyshev).
    /// response : FilterResponse
    ///     Desired frequency response shape (low-pass, high-pass, band-pass, etc.).
    /// order : int
    ///     Filter order (number of poles).
    /// cutoff_frequency : float, optional
    ///     Cutoff frequency in Hz for low-pass and high-pass filters.
    /// low_frequency : float, optional
    ///     Lower cutoff frequency in Hz for band-pass and band-stop filters.
    /// high_frequency : float, optional
    ///     Upper cutoff frequency in Hz for band-pass and band-stop filters.
    ///
    /// Returns
    /// -------
    /// IirFilterDesign
    ///     A new filter design description.
    ///
    /// Notes
    /// -----
    /// Parameter consistency is not validated automatically. Invalid or
    /// incompatible combinations may fail later during filter construction.
    #[new]
    #[pyo3(signature = (filter_type: "FilterType", response: "FilterResponse", order: "int", cutoff_frequency=None, low_frequency=None, high_frequency=None), text_signature = "($cls, filter_type: PyIirFilterType, response: PyFilterResponse, order: int, cutoff_frequency: Optional[float] = None, low_frequency: Optional[float] = None, high_frequency: Optional[float] = None) -> IirFilterDesign")]
    fn new(
        filter_type: PyIirFilterType,
        response: PyFilterResponse,
        order: usize,
        cutoff_frequency: Option<f64>,
        low_frequency: Option<f64>,
        high_frequency: Option<f64>,
    ) -> PyResult<Self> {
        let order = nzu_or_err(order)?;

        let dummy = cutoff_frequency
            .or(low_frequency)
            .or(high_frequency)
            .unwrap_or(1000.0);
        let mut inner = IirFilterDesign::butterworth_lowpass(order, dummy);
        inner.filter_type = filter_type.into();
        inner.response = response.into();
        inner.cutoff_frequency = cutoff_frequency;
        inner.low_frequency = low_frequency;
        inner.high_frequency = high_frequency;
        inner.passband_ripple = None;
        inner.stopband_attenuation = None;
        Ok(Self { inner })
    }

    /// Filter family used for the design.
    #[getter]
    fn filter_type(&self) -> PyIirFilterType {
        self.inner.filter_type.into()
    }

    /// Frequency response shape of the filter.
    #[getter]
    fn response(&self) -> PyFilterResponse {
        self.inner.response.into()
    }

    /// Filter order (number of poles).
    #[getter]
    fn order(&self) -> usize {
        self.inner.order.get()
    }

    /// Cutoff frequency in Hz for low-pass and high-pass filters.
    #[getter]
    fn cutoff_frequency(&self) -> Option<f64> {
        self.inner.cutoff_frequency
    }

    /// Lower cutoff frequency in Hz for band-pass and band-stop filters.
    #[getter]
    fn low_frequency(&self) -> Option<f64> {
        self.inner.low_frequency
    }

    /// Upper cutoff frequency in Hz for band-pass and band-stop filters.
    #[getter]
    fn high_frequency(&self) -> Option<f64> {
        self.inner.high_frequency
    }

    /// Passband ripple in dB, if applicable.
    #[getter]
    fn passband_ripple(&self) -> Option<f64> {
        self.inner.passband_ripple
    }

    /// Stopband attenuation in dB, if applicable.
    #[getter]
    fn stopband_attenuation(&self) -> Option<f64> {
        self.inner.stopband_attenuation
    }

    /// Create a Butterworth low-pass filter design.
    ///
    /// Parameters
    /// ----------
    /// order : int
    ///     Filter order.
    /// cutoff_frequency : float
    ///     Cutoff frequency in Hz.
    ///
    /// Returns
    /// -------
    ///
    /// IirFilterDesign
    ///     A new Butterworth low-pass filter design instance.
    ///
    /// Errors
    /// ------
    ///
    /// If the order argument is less than or equal to zero
    #[classmethod]
    #[pyo3(signature = (order: "int", cutoff_frequency: "float"), text_signature = "($cls, order: int, cutoff_frequency: float) -> IirFilterDesign")]
    fn butterworth_lowpass(
        _cls: &Bound<'_, PyType>,
        order: usize,
        cutoff_frequency: f64,
    ) -> PyResult<Self> {
        let order = nzu_or_err(order)?;

        Ok(Self {
            inner: IirFilterDesign::butterworth_lowpass(order, cutoff_frequency),
        })
    }

    /// Create a Butterworth high-pass filter design.
    ///
    /// Parameters
    /// ----------
    /// order : int
    ///     Filter order.
    /// cutoff_frequency : float
    ///     Cutoff frequency in Hz.
    #[classmethod]
    #[pyo3(signature = (order: "int", cutoff_frequency: "float"), text_signature = "($cls, order: int, cutoff_frequency: float) -> IirFilterDesign")]
    fn butterworth_highpass(
        _cls: &Bound<'_, PyType>,
        order: usize,
        cutoff_frequency: f64,
    ) -> PyResult<Self> {
        let order = nzu_or_err(order)?;

        Ok(Self {
            inner: IirFilterDesign::butterworth_highpass(order, cutoff_frequency),
        })
    }

    /// Create a Butterworth band-pass filter design.
    ///
    /// Parameters
    /// ----------
    /// order : int
    ///     Filter order.
    /// low_frequency : float
    ///     Lower cutoff frequency in Hz.
    /// high_frequency : float
    ///     Upper cutoff frequency in Hz.
    #[classmethod]
    #[pyo3(signature = (order: "int", low_frequency: "float", high_frequency: "float"), text_signature = "($cls, order: int, low_frequency: float, high_frequency: float) -> IirFilterDesign")]
    fn butterworth_bandpass(
        _cls: &Bound<'_, PyType>,
        order: usize,
        low_frequency: f64,
        high_frequency: f64,
    ) -> PyResult<Self> {
        let order = nzu_or_err(order)?;

        Ok(Self {
            inner: IirFilterDesign::butterworth_bandpass(order, low_frequency, high_frequency),
        })
    }
}

impl_py_wrapper_core!(PyIirFilterDesign, IirFilterDesign);
impl_py_repr!(PyIirFilterDesign);

/// Parametric equaliser band type.
///
/// `EqBandType` defines how gain is applied across the frequency spectrum
/// relative to a centre or cutoff frequency. Different band types emphasise or
/// suppress different spectral regions.
///
/// Instances of `EqBandType` are immutable and should be treated as enum-like
/// values. They are accessed via class attributes rather than being constructed
/// directly.
///
/// Available band types:
///
/// - ``EqBandType.peak``
/// - ``EqBandType.low_shelf``
/// - ``EqBandType.high_shelf``
/// - ``EqBandType.low_pass``
/// - ``EqBandType.high_pass``
/// - ``EqBandType.band_pass``
/// - ``EqBandType.band_stop``
#[pyclass(name = "EqBandType", from_py_object, module = "audio_samples.types")]
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
pub struct PyEqBandType {
    pub(crate) inner: EqBandType,
}
#[pymethods]
impl PyEqBandType {
    /// Peaking (bell) filter.
    ///
    /// Boosts or attenuates a narrow frequency region centred at the target
    /// frequency. Positive gain produces a peak; negative gain produces a notch.
    #[classattr]
    fn peak() -> Self {
        Self {
            inner: EqBandType::Peak,
        }
    }

    /// Low-shelf filter.
    ///
    /// Applies a broadband boost or cut to frequencies below the corner
    /// frequency.
    #[classattr]
    fn low_shelf() -> Self {
        Self {
            inner: EqBandType::LowShelf,
        }
    }

    /// High-shelf filter.
    ///
    /// Applies a broadband boost or cut to frequencies above the corner
    /// frequency.
    #[classattr]
    fn high_shelf() -> Self {
        Self {
            inner: EqBandType::HighShelf,
        }
    }

    /// Low-pass filter.
    ///
    /// Attenuates frequencies above the cutoff frequency.
    #[classattr]
    fn low_pass() -> Self {
        Self {
            inner: EqBandType::LowPass,
        }
    }

    /// High-pass filter.
    ///
    /// Attenuates frequencies below the cutoff frequency.
    #[classattr]
    fn high_pass() -> Self {
        Self {
            inner: EqBandType::HighPass,
        }
    }

    /// Band-pass filter.
    ///
    /// Preserves frequencies within a specified band while attenuating
    /// frequencies outside that range.
    #[classattr]
    fn band_pass() -> Self {
        Self {
            inner: EqBandType::BandPass,
        }
    }

    /// Band-stop (notch) filter.
    ///
    /// Attenuates frequencies within a specified band while preserving
    /// frequencies outside that range.
    #[classattr]
    fn band_stop() -> Self {
        Self {
            inner: EqBandType::BandStop,
        }
    }
}

impl_py_wrapper_core!(PyEqBandType, EqBandType);
impl_py_wrapper_fromstr!(PyEqBandType, EqBandType);
impl_py_default_static!(PyEqBandType);
impl_py_repr!(PyEqBandType);

/// Parametric equaliser band configuration.
///
/// `EqBand` represents a single band in a parametric equaliser, defined by a band
/// type, centre or cutoff frequency, gain, and quality factor (Q). Each band
/// describes how a specific region of the frequency spectrum is shaped.
///
/// Instances of `EqBand` are immutable value objects. Parameters are provided at
/// construction time and exposed via read-only properties.
///
/// Convenience constructors are available for common band types such as peak
/// and shelving filters.
#[pyclass(name = "EqBand", from_py_object, module = "audio_samples.types")]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PyEqBand {
    pub(crate) inner: EqBand,
}

#[pymethods]
impl PyEqBand {
    /// Create a new EQ band with explicit parameters.
    ///
    /// Parameters
    /// ----------
    /// band_type : EqBandType
    ///     Type of EQ band (peak, shelf, pass, etc.).
    /// frequency : float
    ///     Centre frequency in Hz (for peak / notch) or corner frequency (for
    ///     shelving and pass filters).
    /// gain_db : float
    ///     Gain in decibels. Positive values boost, negative values attenuate.
    ///     For purely filtering bands (e.g. low-pass), this is typically zero.
    /// q_factor : float
    ///     Quality factor controlling bandwidth or slope. Higher values produce
    ///     narrower bandwidths or steeper transitions.
    ///
    /// Returns
    /// -------
    /// EqBand
    ///     A new EQ band instance.
    ///
    /// Notes
    /// -----
    /// Parameter validity is not automatically checked at construction time.
    /// Invalid configurations may fail later during processing.
    #[new]
    #[pyo3(signature = (band_type: "EqBandType", frequency: "float", gain_db: "float", q_factor: "float"), text_signature = "($cls, band_type: EqBandType, frequency: float, gain_db: float, q_factor: float) -> EqBand")]
    fn new(band_type: PyEqBandType, frequency: f64, gain_db: f64, q_factor: f64) -> PyResult<Self> {
        let band_type: EqBandType = band_type.into();
        let mut inner = EqBand::peak(frequency, gain_db, q_factor);
        inner.band_type = band_type;
        Ok(Self { inner })
    }

    /// Create a peaking (bell) EQ band.
    ///
    /// Parameters
    /// ----------
    /// frequency : float
    ///     Centre frequency in Hz.
    /// gain_db : float
    ///     Gain in decibels. Positive values boost, negative values attenuate.
    /// q_factor : float
    ///     Quality factor controlling bandwidth.
    #[classmethod]
    #[pyo3(signature = (frequency: "float", gain_db: "float", q_factor: "float"), text_signature = "($cls, frequency: float, gain_db: float, q_factor: float) -> EqBand")]
    fn peak(_cls: &Bound<'_, PyType>, frequency: f64, gain_db: f64, q_factor: f64) -> Self {
        Self {
            inner: EqBand::peak(frequency, gain_db, q_factor),
        }
    }

    /// Create a low-shelf EQ band.
    ///
    /// Parameters
    /// ----------
    /// frequency : float
    ///     Corner frequency in Hz.
    /// gain_db : float
    ///     Gain in decibels.
    /// q_factor : float
    ///     Shelf slope control.
    #[classmethod]
    #[pyo3(signature = (frequency: "float", gain_db: "float", q_factor: "float"), text_signature = "($cls, frequency: float, gain_db: float, q_factor: float) -> EqBand")]
    fn low_shelf(_cls: &Bound<'_, PyType>, frequency: f64, gain_db: f64, q_factor: f64) -> Self {
        Self {
            inner: EqBand::low_shelf(frequency, gain_db, q_factor),
        }
    }

    /// Create a high-shelf EQ band.
    ///
    /// Parameters
    /// ----------
    /// frequency : float
    ///     Corner frequency in Hz.
    /// gain_db : float
    ///     Gain in decibels.
    /// q_factor : float
    ///     Shelf slope control.
    #[classmethod]
    #[pyo3(signature = (frequency: "float", gain_db: "float", q_factor: "float"), text_signature = "($cls, frequency: float, gain_db: float, q_factor: float) -> EqBand")]
    fn high_shelf(_cls: &Bound<'_, PyType>, frequency: f64, gain_db: f64, q_factor: f64) -> Self {
        Self {
            inner: EqBand::high_shelf(frequency, gain_db, q_factor),
        }
    }

    /// Centre or cutoff frequency in Hz.
    #[getter]
    fn frequency(&self) -> f64 {
        self.inner.frequency
    }

    /// Gain in decibels.
    #[getter]
    fn gain_db(&self) -> f64 {
        self.inner.gain_db
    }

    /// Quality factor controlling bandwidth or slope.
    #[getter]
    fn q_factor(&self) -> f64 {
        self.inner.q_factor
    }

    /// Whether this band is enabled.
    #[getter]
    fn enabled(&self) -> bool {
        self.inner.enabled
    }
}

impl_py_wrapper_core!(PyEqBand, EqBand);
impl_py_repr!(PyEqBand);

/// Parametric equaliser configuration.
///
/// `ParametricEq` represents a complete parametric equaliser composed of an
/// ordered collection of EQ bands. Each band shapes a specific region of the
/// frequency spectrum, and the combined effect defines the overall frequency
/// response.
///
/// Instances of `ParametricEq` are mutable. Bands can be added and removed after
/// construction. The number of bands can be queried using ``len(eq)`` in Python.
#[pyclass(name = "ParametricEq", from_py_object, module = "audio_samples.types")]
#[derive(Debug, Clone, PartialEq)]
pub struct PyParametricEq {
    pub(crate) inner: ParametricEq,
}

#[pymethods]
impl PyParametricEq {
    /// Create a new empty parametric equaliser.
    #[new]
    #[pyo3(signature = (), text_signature = "($cls) -> ParametricEq")]
    fn new() -> Self {
        Self {
            inner: ParametricEq::new(),
        }
    }

    /// Add an EQ band to the equaliser.
    ///
    /// Parameters
    /// ----------
    /// band : EqBand
    ///     Band configuration to append to the equaliser.
    #[pyo3(signature = (band), text_signature = "($self, band: EqBand) -> None")]
    fn add_band(&mut self, band: &PyEqBand) {
        self.inner.add_band(band.inner.clone());
    }

    /// Remove an EQ band by index.
    ///
    /// Parameters
    /// ----------
    /// index : int
    ///     Zero-based index of the band to remove.
    ///
    /// Returns
    /// -------
    /// EqBand or None
    ///     The removed band if the index was valid, otherwise ``None``.
    #[pyo3(signature = (index), text_signature = "($self, index: int) -> Optional[EqBand]")]
    fn remove_band(&mut self, index: usize) -> Option<PyEqBand> {
        self.inner
            .remove_band(index)
            .map(|inner| PyEqBand { inner })
    }

    /// Overall output gain in decibels.
    #[getter]
    fn output_gain_db(&self) -> f64 {
        self.inner.output_gain_db
    }

    /// Whether the equaliser is currently bypassed.
    #[getter]
    fn bypassed(&self) -> bool {
        self.inner.bypassed
    }

    /// Number of EQ bands.
    ///
    /// Enables use of ``len(eq)`` in Python.
    fn __len__(&self) -> usize {
        self.inner.bands.len()
    }
}

impl_py_wrapper_core!(PyParametricEq, ParametricEq);
impl_py_repr!(PyParametricEq);

/// Knee characteristic for dynamic range processing.
///
/// `KneeType` controls how smoothly gain reduction transitions as the signal
/// crosses the threshold in dynamics processors such as compressors and
/// limiters.
///
/// Instances of `KneeType` are immutable and should be treated as enum-like
/// values. They are accessed via class attributes rather than being constructed
/// directly.
///
/// Available knee types:
///
/// - ``KneeType.hard``
/// - ``KneeType.soft``
#[pyclass(name = "KneeType", from_py_object, module = "audio_samples.types")]
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
pub struct PyKneeType {
    pub(crate) inner: KneeType,
}

#[pymethods]
impl PyKneeType {
    /// Hard knee.
    ///
    /// Applies an abrupt transition at the threshold, yielding precise dynamics
    /// control at the potential cost of audible artefacts.
    #[classattr]
    fn hard() -> Self {
        PyKneeType {
            inner: KneeType::Hard,
        }
    }

    /// Soft knee.
    ///
    /// Applies a gradual transition around the threshold, producing smoother
    /// and more perceptually natural behaviour.
    #[classattr]
    fn soft() -> Self {
        PyKneeType {
            inner: KneeType::Soft,
        }
    }
}

impl_py_wrapper_core!(PyKneeType, KneeType);
impl_py_wrapper_fromstr!(PyKneeType, KneeType);
impl_py_default_static!(PyKneeType);
impl_py_repr!(PyKneeType);

/// Detection method for dynamic range processing.
///
/// `DynamicRangeMethod` selects how signal level is estimated when driving gain
/// reduction in dynamics processors such as compressors, limiters, and gates.
///
/// Different methods trade off smoothness, transient responsiveness, and
/// perceptual stability.
///
/// Instances of `DynamicRangeMethod` are immutable and should be treated as
/// enum-like values. They are accessed via class attributes rather than being
/// constructed directly.
///
/// Available detection methods:
///
/// - ``DynamicRangeMethod.rms``
/// - ``DynamicRangeMethod.peak``
/// - ``DynamicRangeMethod.hybrid``
#[pyclass(
    name = "DynamicRangeMethod",
    from_py_object,
    module = "audio_samples.types"
)]
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
pub struct PyDynamicRangeMethod {
    pub(crate) inner: DynamicRangeMethod,
}

#[pymethods]
impl PyDynamicRangeMethod {
    /// RMS-based level detection.
    ///
    /// Estimates average signal power over time, producing smoother and more
    /// perceptually stable gain control.
    #[classattr]
    fn rms() -> Self {
        PyDynamicRangeMethod {
            inner: DynamicRangeMethod::Rms,
        }
    }

    /// Peak-based level detection.
    ///
    /// Responds to instantaneous signal peaks, providing tight peak control with
    /// increased sensitivity to transients.
    #[classattr]
    fn peak() -> Self {
        PyDynamicRangeMethod {
            inner: DynamicRangeMethod::Peak,
        }
    }

    /// Hybrid level detection.
    ///
    /// Combines RMS and peak estimation to balance smoothness and transient
    /// control.
    #[classattr]
    fn hybrid() -> Self {
        PyDynamicRangeMethod {
            inner: DynamicRangeMethod::Hybrid,
        }
    }
}

impl_py_wrapper_core!(PyDynamicRangeMethod, DynamicRangeMethod);
impl_py_wrapper_fromstr!(PyDynamicRangeMethod, DynamicRangeMethod);
impl_py_default_static!(PyDynamicRangeMethod);
impl_py_repr!(PyDynamicRangeMethod);

/// Side-chain configuration for dynamic range processing.
///
/// `SideChainConfig` describes how an external or filtered control signal is used
/// to drive gain reduction in dynamics processors such as compressors and
/// limiters.
///
/// Side-chain processing can be enabled or disabled, optionally filtered, and
/// mixed with the internal detector signal.
///
/// Instances of `SideChainConfig` are immutable value objects in the Python API.
/// Parameters are provided at construction time and exposed via read-only
/// properties.
#[pyclass(
    name = "SideChainConfig",
    from_py_object,
    module = "audio_samples.types"
)]
#[derive(Default, Debug, Clone, PartialEq)]
pub struct PySideChainConfig {
    pub(crate) inner: SideChainConfig,
}

#[pymethods]
impl PySideChainConfig {
    /// Create a new side-chain configuration.
    ///
    /// Parameters
    /// ----------
    /// enabled : bool
    ///     Whether side-chain processing is enabled.
    /// high_pass_freq : float or None, optional
    ///     High-pass filter cutoff frequency in Hz for the side-chain signal.
    ///     Useful for reducing low-frequency pumping effects.
    /// low_pass_freq : float or None, optional
    ///     Low-pass filter cutoff frequency in Hz for the side-chain signal.
    ///     Useful for focusing compression on specific frequency ranges.
    /// pre_emphasis_db : float, optional
    ///     Pre-emphasis applied to the side-chain signal in decibels.
    /// external_mix : float, optional
    ///     Mix ratio between internal and external side-chain signal in the
    ///     range ``[0.0, 1.0]``. ``0.0`` selects internal only, ``1.0`` selects
    ///     external only.
    ///
    /// Returns
    /// -------
    /// SideChainConfig
    ///     A new side-chain configuration.
    ///
    /// Notes
    /// -----
    /// Parameter validity is not automatically checked at construction time.
    /// Invalid configurations may fail later during processing.
    #[new]
    #[pyo3(signature = (enabled: "bool", high_pass_freq: "Optional[float]"=None, low_pass_freq: "Optional[float]"=None, pre_emphasis_db=0.0, external_mix=0.0), text_signature="($cls, enabled: bool, high_pass_freq: float | None = None, low_pass_freq: float | None = None, pre_emphasis_db: float = 0.0, external_mix: float = 0.0) -> SideChainConfig")]
    fn new(
        enabled: bool,
        high_pass_freq: Option<f64>,
        low_pass_freq: Option<f64>,
        pre_emphasis_db: f64,
        external_mix: f64,
    ) -> Self {
        let mut inner = SideChainConfig::default();
        inner.enabled = enabled;
        inner.high_pass_freq = high_pass_freq;
        inner.low_pass_freq = low_pass_freq;
        inner.pre_emphasis_db = pre_emphasis_db;
        inner.external_mix = external_mix;
        PySideChainConfig { inner }
    }

    /// Whether side-chain processing is enabled.
    #[getter]
    fn enabled(&self) -> bool {
        self.inner.enabled
    }

    /// High-pass filter cutoff frequency in Hz for the side-chain signal, if set.
    #[getter]
    fn high_pass_freq(&self) -> Option<f64> {
        self.inner.high_pass_freq
    }

    /// Low-pass filter cutoff frequency in Hz for the side-chain signal, if set.
    #[getter]
    fn low_pass_freq(&self) -> Option<f64> {
        self.inner.low_pass_freq
    }

    /// Pre-emphasis applied to the side-chain signal in decibels.
    #[getter]
    fn pre_emphasis_db(&self) -> f64 {
        self.inner.pre_emphasis_db
    }

    /// Mix ratio between internal and external side-chain signal.
    ///
    /// Values are expected in the range ``[0.0, 1.0]``.
    #[getter]
    fn external_mix(&self) -> f64 {
        self.inner.external_mix
    }
}

impl_py_wrapper_core!(PySideChainConfig, SideChainConfig);
impl_py_default_static!(PySideChainConfig);
impl_py_repr!(PySideChainConfig);

/// Compressor configuration parameters.
///
/// `CompressorConfig` defines how a dynamic range compressor responds to signal
/// levels above a threshold, including time constants, compression curve shape,
/// detection method, and side-chain behaviour.
///
/// Instances of `CompressorConfig` are immutable value objects in the Python API.
/// Parameters are provided at construction time and exposed via read-only
/// properties internally. Validation can be performed explicitly using
/// ``validate(sample_rate)``.
///
/// Several common presets are provided as class attributes.
#[pyclass(
    name = "CompressorConfig",
    from_py_object,
    module = "audio_samples.types"
)]
#[derive(Default, Debug, Clone, Copy, PartialEq)]
pub struct PyCompressorConfig {
    pub(crate) inner: CompressorConfig,
}

#[pymethods]
impl PyCompressorConfig {
    /// Create a new compressor configuration.
    ///
    /// Parameters
    /// ----------
    /// threshold_db : float
    ///     Threshold level in decibels. Signal levels above this value will be
    ///     compressed. Typically negative (e.g. -40 to 0 dB).
    /// ratio : float
    ///     Compression ratio. ``1.0`` means no compression; larger values apply
    ///     stronger compression.
    /// attack_ms : float
    ///     Attack time in milliseconds. Controls how quickly compression engages
    ///     once the signal exceeds the threshold.
    /// release_ms : float
    ///     Release time in milliseconds. Controls how quickly compression
    ///     disengages once the signal falls below the threshold.
    /// makeup_gain_db : float
    ///     Makeup gain in decibels applied after compression.
    /// knee_type : KneeType
    ///     Knee characteristic controlling transition smoothness around the
    ///     threshold.
    /// knee_width_db : float
    ///     Width of the knee region in decibels for soft-knee behaviour.
    /// detection_method : DynamicRangeMethod
    ///     Signal level detection method driving gain reduction.
    /// side_chain : SideChainConfig
    ///     Side-chain configuration.
    /// lookahead_ms : float
    ///     Lookahead time in milliseconds. Allows the compressor to anticipate
    ///     upcoming peaks.
    ///
    /// Returns
    /// -------
    /// CompressorConfig
    ///     A new compressor configuration.
    #[new]
    #[pyo3(signature = (*, threshold_db: "float", ratio: "float", attack_ms: "float", release_ms: "float", makeup_gain_db: "float", knee_type: "KneeType", knee_width_db: "float", detection_method: "DynamicRangeMethod", side_chain: "SideChainConfig", lookahead_ms: "float"), text_signature="($cls, *, threshold_db: float, ratio: float, attack_ms: float, release_ms: float, makeup_gain_db: float, knee_type: KneeType, knee_width_db: float, detection_method: DynamicRangeMethod, side_chain: SideChainConfig, lookahead_ms: float) -> CompressorConfig")]
    fn new(
        threshold_db: f64,
        ratio: f64,
        attack_ms: f64,
        release_ms: f64,
        makeup_gain_db: f64,
        knee_type: PyKneeType,
        knee_width_db: f64,
        detection_method: PyDynamicRangeMethod,
        side_chain: PySideChainConfig,
        lookahead_ms: f64,
    ) -> Self {
        let mut inner = CompressorConfig::default();
        inner.threshold_db = threshold_db;
        inner.ratio = ratio;
        inner.attack_ms = attack_ms;
        inner.release_ms = release_ms;
        inner.makeup_gain_db = makeup_gain_db;
        inner.knee_type = knee_type.inner;
        inner.knee_width_db = knee_width_db;
        inner.detection_method = detection_method.inner;
        inner.side_chain = side_chain.inner;
        inner.lookahead_ms = lookahead_ms;
        Self { inner }
    }

    /// Vocal compression preset.
    ///
    /// Tuned for moderate dynamic control and natural sounding speech or vocals.
    #[classattr]
    fn vocal() -> Self {
        Self {
            inner: CompressorConfig::vocal(),
        }
    }

    /// Drum compression preset.
    ///
    /// Tuned for fast transient control and strong dynamic shaping.
    #[classattr]
    fn drum() -> Self {
        Self {
            inner: CompressorConfig::drum(),
        }
    }

    /// Bus compression preset.
    ///
    /// Tuned for gentle dynamic glue on groups or mix buses.
    #[classattr]
    fn bus() -> Self {
        Self {
            inner: CompressorConfig::bus(),
        }
    }

    /// Validate compressor configuration parameters.
    ///
    /// Parameters
    /// ----------
    /// sample_rate : float
    ///     Audio sample rate in Hz, used to validate time- and frequency-dependent
    ///     constraints.
    ///
    /// Returns
    /// -------
    /// CompressorConfig
    ///    The validated compressor configuration.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If any configuration parameter is invalid.
    #[pyo3(signature = (sample_rate: "float"), text_signature="($self, sample_rate: float) -> CompressorConfig")]
    fn validate(&mut self, sample_rate: f64) -> PyResult<Self> {
        self.inner.validate(sample_rate).map_err(audio_err_to_py)?;
        Ok(Self { inner: self.inner })
    }
}

impl_py_wrapper_core!(PyCompressorConfig, CompressorConfig);
impl_py_default_static!(PyCompressorConfig);
impl_py_repr!(PyCompressorConfig);

/// Limiter configuration parameters.
///
/// `LimiterConfig` defines how a limiter prevents signal levels from exceeding a
/// specified ceiling, including time constants, knee behaviour, detection
/// method, side-chain configuration, and inter-sample peak (ISP) limiting.
///
/// Instances of `LimiterConfig` are immutable value objects in the Python API.
/// Parameters are provided at construction time and exposed via read-only
/// properties internally. Validation can be performed explicitly using
/// ``validate(sample_rate)``.
///
/// Several common presets are provided as static constructors.
#[pyclass(name = "LimiterConfig", from_py_object, module = "audio_samples.types")]
#[derive(Default, Debug, Clone, PartialEq)]
pub struct PyLimiterConfig {
    pub(crate) inner: LimiterConfig,
}

#[pymethods]
impl PyLimiterConfig {
    /// Create a new limiter configuration.
    ///
    /// Parameters
    /// ----------
    /// ceiling_db : float
    ///     Maximum allowed output level in decibels. Typically negative
    ///     (e.g. -0.1 to -3.0 dB).
    /// attack_ms : float
    ///     Attack time in milliseconds. Controls how quickly limiting engages as
    ///     the signal approaches the ceiling.
    /// release_ms : float
    ///     Release time in milliseconds. Controls how quickly limiting disengages
    ///     once the signal falls below the ceiling.
    /// knee_type : KneeType
    ///     Knee characteristic controlling transition smoothness around the
    ///     ceiling.
    /// knee_width_db : float
    ///     Width of the knee region in decibels for soft-knee behaviour.
    /// detection_method : DynamicRangeMethod
    ///     Signal level detection method driving limiting.
    /// side_chain : SideChainConfig
    ///     Side-chain configuration.
    /// lookahead_ms : float
    ///     Lookahead time in milliseconds. Allows the limiter to anticipate
    ///     upcoming peaks.
    /// isp_limiting : bool
    ///     Whether to enable inter-sample peak (ISP) limiting.
    ///
    /// Returns
    /// -------
    /// LimiterConfig
    ///     A new limiter configuration.
    #[new]
    #[pyo3(signature = (*, ceiling_db: "float", attack_ms: "float", release_ms: "float", knee_type, knee_width_db, detection_method, side_chain, lookahead_ms, isp_limiting), text_signature="($cls, ceiling_db: float, attack_ms: float, release_ms: float, knee_type: KneeType, knee_width_db: float, detection_method: DynamicRangeMethod, side_chain: SideChainConfig, lookahead_ms: float, isp_limiting: bool) -> LimiterConfig")]
    fn new(
        ceiling_db: f64,
        attack_ms: f64,
        release_ms: f64,
        knee_type: PyKneeType,
        knee_width_db: f64,
        detection_method: PyDynamicRangeMethod,
        side_chain: PySideChainConfig,
        lookahead_ms: f64,
        isp_limiting: bool,
    ) -> PyResult<Self> {
        let mut inner = LimiterConfig::new(
            ceiling_db,
            attack_ms,
            release_ms,
            knee_type.inner,
            knee_width_db,
            detection_method.inner,
            lookahead_ms,
            isp_limiting,
        );
        inner.side_chain = side_chain.inner;
        Ok(PyLimiterConfig { inner })
    }

    /// Ceiling level in decibels.
    #[getter]
    fn ceiling_db(&self) -> f64 {
        self.inner.ceiling_db
    }

    /// Attack time in milliseconds.
    #[getter]
    fn attack_ms(&self) -> f64 {
        self.inner.attack_ms
    }

    /// Release time in milliseconds.
    #[getter]
    fn release_ms(&self) -> f64 {
        self.inner.release_ms
    }

    /// Knee characteristic.
    #[getter]
    fn knee_type(&self) -> PyKneeType {
        PyKneeType {
            inner: self.inner.knee_type,
        }
    }

    /// Knee width in decibels.
    #[getter]
    fn knee_width_db(&self) -> f64 {
        self.inner.knee_width_db
    }

    /// Detection method used for limiting.
    #[getter]
    fn detection_method(&self) -> PyDynamicRangeMethod {
        PyDynamicRangeMethod {
            inner: self.inner.detection_method,
        }
    }

    /// Side-chain configuration.
    #[getter]
    fn side_chain(&self) -> PySideChainConfig {
        PySideChainConfig {
            inner: self.inner.side_chain.clone(), // todo: avoid clone
        }
    }

    /// Lookahead time in milliseconds.
    #[getter]
    fn lookahead_ms(&self) -> f64 {
        self.inner.lookahead_ms
    }

    /// Whether inter-sample peak limiting is enabled.
    #[getter]
    fn isp_limiting(&self) -> bool {
        self.inner.isp_limiting
    }

    /// Transparent limiter preset.
    ///
    /// Tuned for minimal audible impact while preventing clipping.
    #[classmethod]
    #[pyo3(signature=(), text_signature="($cls) -> LimiterConfig")]
    fn transparent(_cls: &Bound<'_, PyType>) -> Self {
        PyLimiterConfig {
            inner: LimiterConfig::transparent(),
        }
    }

    /// Mastering limiter preset.
    ///
    /// Tuned for loudness maximisation with controlled transparency.
    #[classmethod]
    #[pyo3(signature=(), text_signature="($cls) -> LimiterConfig")]
    fn mastering(_cls: &Bound<'_, PyType>) -> Self {
        PyLimiterConfig {
            inner: LimiterConfig::mastering(),
        }
    }

    /// Broadcast limiter preset.
    ///
    /// Tuned for aggressive peak control and regulatory compliance.
    #[classmethod]
    #[pyo3(signature=(), text_signature="($cls) -> LimiterConfig")]
    fn broadcast(_cls: &Bound<'_, PyType>) -> Self {
        PyLimiterConfig {
            inner: LimiterConfig::broadcast(),
        }
    }

    /// Validate limiter configuration parameters.
    ///
    /// Parameters
    /// ----------
    /// sample_rate : float
    ///     Audio sample rate in Hz, used to validate time- and frequency-dependent
    ///     constraints.
    ///
    /// Returns
    /// -------
    /// LimiterConfig
    ///   The validated limiter configuration.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If any configuration parameter is invalid.
    #[pyo3(signature=(sample_rate: "float"), text_signature="($self, sample_rate: float) -> LimiterConfig")]
    fn validate(&mut self, sample_rate: f64) -> PyResult<Self> {
        self.inner.validate(sample_rate).map_err(audio_err_to_py)?;
        Ok(Self { inner: self.inner })
    }
}

impl_py_wrapper_core!(PyLimiterConfig, LimiterConfig);
impl_py_default_static!(PyLimiterConfig);
impl_py_repr!(PyLimiterConfig);

/// Adaptive thresholding strategy for peak picking.
///
/// `AdaptiveThresholdMethod` selects how dynamic detection thresholds are
/// estimated from the onset strength function over time when performing
/// peak picking or onset detection.
///
/// Different strategies trade off responsiveness to rapid changes against
/// robustness to noise and transient outliers.
///
/// Instances of `AdaptiveThresholdMethod` are immutable and should be treated
/// as enum-like values. They are accessed via class attributes rather than
/// being constructed directly.
///
/// Available methods:
///
/// - ``AdaptiveThresholdMethod.delta``
/// - ``AdaptiveThresholdMethod.percentile``
/// - ``AdaptiveThresholdMethod.combined``
#[pyclass(
    name = "AdaptiveThresholdMethod",
    from_py_object,
    module = "audio_samples.types"
)]
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PyAdaptiveThresholdMethod {
    pub(crate) inner: AdaptiveThresholdMethod,
}

#[pymethods]
impl PyAdaptiveThresholdMethod {
    /// Delta-based adaptive threshold.
    ///
    /// Tracks local maxima and applies a fixed offset to determine the detection
    /// threshold. Responds quickly to rapid changes but can be sensitive to
    /// noise and transient outliers.
    #[classattr]
    fn delta() -> Self {
        PyAdaptiveThresholdMethod {
            inner: AdaptiveThresholdMethod::Delta,
        }
    }

    /// Percentile-based adaptive threshold.
    ///
    /// Estimates the threshold from rolling distribution statistics of the
    /// onset strength function, yielding increased robustness at the cost of
    /// slower adaptation.
    #[classattr]
    fn percentile() -> Self {
        PyAdaptiveThresholdMethod {
            inner: AdaptiveThresholdMethod::Percentile,
        }
    }

    /// Combined adaptive threshold.
    ///
    /// Combines delta-based and percentile-based thresholds to balance
    /// responsiveness and robustness across a wide range of signals.
    #[classattr]
    fn combined() -> Self {
        PyAdaptiveThresholdMethod {
            inner: AdaptiveThresholdMethod::Combined,
        }
    }
}

impl_py_wrapper_core!(PyAdaptiveThresholdMethod, AdaptiveThresholdMethod);
impl_py_wrapper_fromstr!(PyAdaptiveThresholdMethod, AdaptiveThresholdMethod);
impl_py_default_static!(PyAdaptiveThresholdMethod);
impl_py_repr!(PyAdaptiveThresholdMethod);

/// Noise colour classification for audio perturbation and synthesis.
///
/// `NoiseColor` classifies stochastic noise processes by their spectral energy
/// distribution. Different noise colours influence perceived brightness,
/// smoothness, and temporal correlation in audio synthesis and perturbation
/// tasks.
///
/// Instances of `NoiseColor` are immutable and should be treated as enum-like
/// values. They are accessed via class attributes rather than being constructed
/// directly.
///
/// Available noise colours:
///
/// - ``NoiseColor.white``
/// - ``NoiseColor.pink``
/// - ``NoiseColor.brown``
#[pyclass(name = "NoiseColor", from_py_object, module = "audio_samples.types")]
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PyNoiseColor {
    pub(crate) inner: NoiseColor,
}

#[pymethods]
impl PyNoiseColor {
    /// White noise.
    ///
    /// Exhibits approximately uniform spectral energy density across the
    /// frequency spectrum, resulting in a bright and broadband character.
    #[classattr]
    fn white() -> Self {
        PyNoiseColor {
            inner: NoiseColor::White,
        }
    }

    /// Pink noise.
    ///
    /// Exhibits decreasing spectral energy with increasing frequency, producing
    /// a perceptually balanced spectrum across octaves.
    #[classattr]
    fn pink() -> Self {
        PyNoiseColor {
            inner: NoiseColor::Pink,
        }
    }

    /// Brown (red) noise.
    ///
    /// Exhibits strongly attenuated high-frequency content, yielding a smoother
    /// and more correlated temporal structure.
    #[classattr]
    fn brown() -> Self {
        PyNoiseColor {
            inner: NoiseColor::Brown,
        }
    }
}

impl_py_wrapper_core!(PyNoiseColor, NoiseColor);
impl_py_wrapper_fromstr!(PyNoiseColor, NoiseColor);
impl_py_default_static!(PyNoiseColor);
impl_py_repr!(PyNoiseColor);

/// Perturbation methods for audio data augmentation.
///
/// `PerturbationMethod` represents a specific audio perturbation configuration
/// used for data augmentation, robustness testing, or creative effects.
///
/// This type is enum-like in Python. Instances are constructed via class
/// constructors such as ``PerturbationMethod.gaussian(...)`` or
/// ``PerturbationMethod.pitch_shift(...)`` rather than by calling
/// ``PerturbationMethod()`` directly.
///
/// A perturbation configuration can be validated against a sample rate via
/// ``validate(sample_rate)``.
#[pyclass(
    name = "PerturbationMethod",
    from_py_object,
    module = "audio_samples.types"
)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PyPerturbationMethod {
    pub(crate) inner: PerturbationMethod,
}

#[pymethods]
impl PyPerturbationMethod {
    /// White noise injection with a target signal-to-noise ratio.
    ///
    /// Adds coloured Gaussian noise to achieve the target SNR relative to the
    /// input signal's RMS level.
    ///
    /// Parameters
    /// ----------
    /// target_snr_db : float
    ///     Target signal-to-noise ratio in decibels.
    ///
    /// Returns
    /// -------
    /// PerturbationMethod
    ///     A Gaussian-noise perturbation configuration.
    #[classmethod]
    #[pyo3(signature=(target_snr_db: "float"), text_signature="($cls, target_snr_db: float) -> PerturbationMethod")]
    fn white_noise(cls: &Bound<'_, PyType>, target_snr_db: f64) -> Self {
        PyPerturbationMethod::gaussian(cls, target_snr_db, PyNoiseColor::white())
    }

    /// Pink noise injection with a target signal-to-noise ratio.
    ///
    /// Adds coloured Gaussian noise to achieve the target SNR relative to the
    /// input signal's RMS level.
    ///
    /// Parameters
    /// ----------
    /// target_snr_db : float
    ///    Target signal-to-noise ratio in decibels.
    ///
    /// Returns
    /// -------
    ///
    /// PerturbationMethod
    ///     A Gaussian-noise perturbation configuration.
    #[classmethod]
    #[pyo3(signature=(target_snr_db: "float"), text_signature="($cls, target_snr_db: float) -> PerturbationMethod")]
    fn pink_noise(cls: &Bound<'_, PyType>, target_snr_db: f64) -> Self {
        PyPerturbationMethod::gaussian(cls, target_snr_db, PyNoiseColor::pink())
    }

    /// Brown noise injection with a target signal-to-noise ratio.
    ///
    /// Adds coloured Gaussian noise to achieve the target SNR relative to the
    /// input signal's RMS level.
    ///
    /// Parameters
    /// ----------
    /// target_snr_db : float
    ///     Target signal-to-noise ratio in decibels.
    ///
    /// Returns
    /// -------
    ///
    /// PerturbationMethod
    ///    A Gaussian-noise perturbation configuration.
    #[classmethod]
    #[pyo3(signature=(target_snr_db: "float"), text_signature="($cls, target_snr_db: float) -> PerturbationMethod")]
    fn brown_noise(cls: &Bound<'_, PyType>, target_snr_db: f64) -> Self {
        PyPerturbationMethod::gaussian(cls, target_snr_db, PyNoiseColor::brown())
    }

    /// Gaussian noise injection with a target signal-to-noise ratio.
    ///
    /// Adds coloured Gaussian noise to achieve the target SNR relative to the
    /// input signal's RMS level.
    ///
    /// Parameters
    /// ----------
    /// target_snr_db : float
    ///     Target signal-to-noise ratio in decibels.
    /// noise_color : NoiseColor
    ///     Noise colour (spectral distribution) to use.
    ///
    /// Returns
    /// -------
    /// PerturbationMethod
    ///     A Gaussian-noise perturbation configuration.
    #[classmethod]
    #[pyo3(signature=(target_snr_db: "float", noise_color: "NoiseColor"), text_signature="($cls, target_snr_db: float, noise_color: NoiseColor) -> PerturbationMethod")]
    fn gaussian(_cls: &Bound<'_, PyType>, target_snr_db: f64, noise_color: PyNoiseColor) -> Self {
        PyPerturbationMethod {
            inner: PerturbationMethod::GaussianNoise {
                target_snr_db,
                noise_color: noise_color.inner,
            },
        }
    }

    /// Random gain perturbation within a specified range.
    ///
    /// Applies a uniform random gain (in dB) to all channels. Positive values
    /// boost; negative values attenuate.
    ///
    /// Parameters
    /// ----------
    /// min_gain_db : float
    ///     Minimum gain in decibels.
    /// max_gain_db : float
    ///     Maximum gain in decibels.
    ///
    /// Returns
    /// -------
    /// PerturbationMethod
    ///     A random-gain perturbation configuration.
    #[classmethod]
    #[pyo3(signature=(min_gain_db: "float", max_gain_db: "float"), text_signature="($cls, min_gain_db: float, max_gain_db: float) -> PerturbationMethod")]
    fn random_gain(_cls: &Bound<'_, PyType>, min_gain_db: f64, max_gain_db: f64) -> Self {
        PyPerturbationMethod {
            inner: PerturbationMethod::RandomGain {
                min_gain_db,
                max_gain_db,
            },
        }
    }

    /// High-pass filtering perturbation.
    ///
    /// Applies a high-pass filter to reduce low-frequency content, simulating
    /// effects such as rumble removal.
    ///
    /// Parameters
    /// ----------
    /// cutoff_hz : float
    ///     High-pass cutoff frequency in Hz.
    ///
    /// Returns
    /// -------
    /// PerturbationMethod
    ///     A high-pass filter perturbation configuration.
    #[classmethod]
    #[pyo3(signature=(cutoff_hz: "float"), text_signature="($cls, cutoff_hz: float) -> PerturbationMethod")]
    fn high_pass_filter(_cls: &Bound<'_, PyType>, cutoff_hz: f64) -> Self {
        PyPerturbationMethod {
            inner: PerturbationMethod::HighPassFilter {
                cutoff_hz,
                slope_db_per_octave: None,
            },
        }
    }

    /// High-pass filtering perturbation with custom slope.
    ///
    /// Parameters
    /// ----------
    /// cutoff_hz : float
    ///     High-pass cutoff frequency in Hz.
    /// slope_db_per_octave : float
    ///     Filter slope in dB per octave.
    ///
    /// Returns
    /// -------
    /// PerturbationMethod
    ///     A high-pass filter perturbation configuration.
    #[classmethod]
    #[pyo3(signature=(cutoff_hz: "float", slope_db_per_octave: "float"), text_signature="($cls, cutoff_hz: float, slope_db_per_octave: float) -> PerturbationMethod")]
    fn high_pass_filter_with_slope(
        _cls: &Bound<'_, PyType>,
        cutoff_hz: f64,
        slope_db_per_octave: f64,
    ) -> Self {
        PyPerturbationMethod {
            inner: PerturbationMethod::HighPassFilter {
                cutoff_hz,
                slope_db_per_octave: Some(slope_db_per_octave),
            },
        }
    }

    /// Low-pass filtering perturbation.
    ///
    /// Applies a low-pass filter to reduce high-frequency content, simulating
    /// effects such as telephone bandwidth limitation.
    ///
    /// Parameters
    /// ----------
    /// cutoff_hz : float
    ///     Low-pass cutoff frequency in Hz.
    ///
    /// Returns
    /// -------
    /// PerturbationMethod
    ///     A low-pass filter perturbation configuration.
    #[classmethod]
    #[pyo3(signature=(cutoff_hz: "float"), text_signature="($cls, cutoff_hz: float) -> PerturbationMethod")]
    fn low_pass_filter(_cls: &Bound<'_, PyType>, cutoff_hz: f64) -> Self {
        PyPerturbationMethod {
            inner: PerturbationMethod::LowPassFilter {
                cutoff_hz,
                slope_db_per_octave: None,
            },
        }
    }

    /// Low-pass filtering perturbation with custom slope.
    ///
    /// Parameters
    /// ----------
    /// cutoff_hz : float
    ///     Low-pass cutoff frequency in Hz.
    /// slope_db_per_octave : float
    ///     Filter slope in dB per octave.
    ///
    /// Returns
    /// -------
    /// PerturbationMethod
    ///     A low-pass filter perturbation configuration.
    #[classmethod]
    #[pyo3(signature=(cutoff_hz: "float", slope_db_per_octave: "float"), text_signature="($cls, cutoff_hz: float, slope_db_per_octave: float) -> PerturbationMethod")]
    fn low_pass_filter_with_slope(
        _cls: &Bound<'_, PyType>,
        cutoff_hz: f64,
        slope_db_per_octave: f64,
    ) -> Self {
        PyPerturbationMethod {
            inner: PerturbationMethod::LowPassFilter {
                cutoff_hz,
                slope_db_per_octave: Some(slope_db_per_octave),
            },
        }
    }

    /// Pitch shifting perturbation.
    ///
    /// Shifts the pitch of the signal by a number of semitones while attempting
    /// to maintain duration.
    ///
    /// Parameters
    /// ----------
    /// semitones : float
    ///     Pitch shift in semitones. Positive shifts up; negative shifts down.
    /// preserve_formants : bool, optional
    ///     Whether to attempt formant preservation.
    ///
    /// Returns
    /// -------
    /// PerturbationMethod
    ///     A pitch-shift perturbation configuration.
    #[classmethod]
    #[pyo3(signature=(semitones: "float", preserve_formants: "bool"=false), text_signature="($cls, semitones: float, preserve_formants: bool = False) -> PerturbationMethod")]
    fn pitch_shift(_cls: &Bound<'_, PyType>, semitones: f64, preserve_formants: bool) -> Self {
        PyPerturbationMethod {
            inner: PerturbationMethod::PitchShift {
                semitones,
                preserve_formants,
            },
        }
    }

    /// Validate perturbation parameters for a given sample rate.
    ///
    /// Parameters
    /// ----------
    /// sample_rate : float
    ///     Sample rate in Hz, used to validate frequency-dependent parameters.
    ///
    /// Returns
    /// -------
    /// PerturbationMethod
    ///     This instance. Returned to allow call chaining.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If the perturbation parameters are invalid.
    #[pyo3(signature=(sample_rate: "float"), text_signature="($self, sample_rate: float) -> PerturbationMethod")]
    fn validate(&self, sample_rate: f64) -> PyResult<Self> {
        self.inner.validate(sample_rate).map_err(audio_err_to_py)?;
        Ok(Self { inner: self.inner })
    }
}

impl_py_wrapper_core!(PyPerturbationMethod, PerturbationMethod);
impl_py_repr!(PyPerturbationMethod);

/// Configuration for audio perturbation operations.
///
/// `PerturbationConfig` defines how a perturbation method should be applied to
/// audio data, optionally including a deterministic random seed.
///
/// The configuration is immutable once created. Validation can be performed
/// explicitly using ``validate(sample_rate)``.
///
/// Typical usage is to construct a perturbation method first and then wrap it
/// in a configuration object:
///
/// ```python
/// method = PerturbationMethod.gaussian(-12.0, NoiseColor.white)
/// cfg = PerturbationConfig(method, seed=42).validate(sample_rate)
/// ```

#[pyclass(
    name = "PerturbationConfig",
    from_py_object,
    module = "audio_samples.types"
)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PyPerturbationConfig {
    pub(crate) inner: PerturbationConfig,
}

#[pymethods]
impl PyPerturbationConfig {
    /// Create a new perturbation configuration.
    ///
    /// Parameters
    /// ----------
    /// method : PerturbationMethod
    ///     The perturbation method to apply.
    /// seed : Optional[int]
    ///     Optional random seed for deterministic behaviour. If ``None``, a
    ///     non-deterministic random generator is used.
    ///
    /// Returns
    /// -------
    /// PerturbationConfig
    ///     A new perturbation configuration.
    #[new]
    #[pyo3(signature=(method: "PerturbationMethod", seed: "Option[u64]"=None), text_signature="($cls, method: PerturbationMethod, seed: Optional[int] = None) -> None")]
    fn new(method: PyPerturbationMethod, seed: Option<u64>) -> Self {
        let mut inner = PerturbationConfig::new(method.inner);
        inner.seed = seed;
        PyPerturbationConfig { inner }
    }

    /// Random seed used for deterministic perturbation, if specified.
    ///
    /// Returns
    /// -------
    /// Optional[int]
    ///     The seed value, or ``None`` if no seed is configured.
    #[getter]
    fn seed(&self) -> Option<u64> {
        self.inner.seed
    }

    /// Validate the perturbation configuration for a given sample rate.
    ///
    /// Parameters
    /// ----------
    /// sample_rate : float
    ///     Sample rate in Hz, used to validate frequency-dependent parameters.
    ///
    /// Returns
    /// -------
    /// PerturbationConfig
    ///     This instance. Returned to allow call chaining.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If the configuration or underlying perturbation method is invalid.
    #[pyo3(signature=(sample_rate: "float"), text_signature="($self, sample_rate: float) -> PerturbationConfig")]
    fn validate(&self, sample_rate: f64) -> PyResult<Self> {
        self.inner.validate(sample_rate).map_err(audio_err_to_py)?;
        Ok(Self { inner: self.inner })
    }
}

impl_py_wrapper_core!(PyPerturbationConfig, PerturbationConfig);
impl_py_repr!(PyPerturbationConfig);

/// Configuration for Harmonic/Percussive Source Separation (HPSS).
///
/// HPSS separates audio into harmonic and percussive components using
/// STFT magnitude median filtering. Harmonic components are enhanced
/// by median filtering along the time axis, while percussive components
/// are enhanced by median filtering along the frequency axis.
#[pyclass(name = "HpssConfig", from_py_object, module = "audio_samples.types")]
#[derive(Debug, Clone, PartialEq)]
pub struct PyHpssConfig {
    pub(crate) inner: HpssConfig,
}

#[pymethods]
impl PyHpssConfig {
    /// Create a new HPSS configuration with default settings.
    ///
    /// Default configuration suitable for general harmonic/percussive separation:
    /// - 2048 sample window (good frequency resolution)
    /// - 512 sample hop size (good time resolution)
    /// - Harmonic kernel: 17 (enhances sustained tones)
    /// - Percussive kernel: 17 (enhances transients)
    /// - Moderate soft masking (0.3)
    #[new]
    #[pyo3(signature=(
        *,
        stft_params: "Optional[StftParams]" = None,
        median_filter_harmonic: "int"=17,
        median_filter_percussive: "int"=17,
        mask_softness: "float"=0.3),
        text_signature="($cls, *, stft_params: Optional[StftParams], median_filter_harmonic: int=17, median_filter_percussive: int=17, mask_softness: float=0.3) -> None")]
    fn new(
        stft_params: Option<PyStftParams>,
        median_filter_harmonic: usize,
        median_filter_percussive: usize,
        mask_softness: f64,
    ) -> PyResult<Self> {
        let stft_params = stft_params.map(|x| x.inner).unwrap_or(StftParams::new(
            nzu!(2048),
            nzu!(512),
            WindowType::Hanning,
            true,
        )?);
        let hpss_config = HpssConfig::new(
            stft_params,
            median_filter_harmonic,
            median_filter_percussive,
            mask_softness,
        );
        Ok(PyHpssConfig { inner: hpss_config })
    }

    /// Create configuration optimized for musical content.
    ///
    /// Uses larger filters for stronger separation, suitable for complex musical material:
    /// - Larger harmonic kernel for better tonal separation
    /// - Larger percussive kernel for cleaner transient isolation
    /// - Softer masking for more musical results
    #[classattr]
    fn musical() -> Self {
        PyHpssConfig {
            inner: HpssConfig::musical(),
        }
    }
    /// Create configuration optimized for percussive content.
    ///
    /// Uses asymmetric filters favoring percussive separation:
    /// - Moderate harmonic filtering
    /// - Strong percussive filtering
    /// - Harder masking for cleaner drum isolation
    #[classattr]
    fn percussive() -> Self {
        PyHpssConfig {
            inner: HpssConfig::percussive(),
        }
    }

    /// Create configuration optimized for harmonic content.
    ///
    /// Uses asymmetric filters favoring harmonic separation:
    /// - Strong harmonic filtering
    /// - Moderate percussive filtering
    /// - Harder masking for cleaner tonal isolation
    #[classattr]
    fn harmonic() -> Self {
        PyHpssConfig {
            inner: HpssConfig::harmonic(),
        }
    }

    /// Create configuration for real-time processing.
    ///
    /// Uses smaller window and filter sizes for lower latency:
    /// - Smaller window for reduced latency
    /// - Smaller hop size for responsiveness
    /// - Smaller filters for faster processing
    #[classattr]
    fn realtime() -> Self {
        PyHpssConfig {
            inner: HpssConfig::realtime(),
        }
    }

    /// Set STFT parameters.
    ///
    /// Parameters
    /// ----------
    /// n_fft: int
    ///     FFT size in samples (should be power of 2 and >= win_size)
    /// win_size: int
    ///     Window size in samples (should be power of 2)
    /// hop_size: int
    ///     Hop size in samples (typically win_size/4)
    #[pyo3(signature=(n_fft: "int",  hop_size: "int"), text_signature="($self, n_fft: int, hop_size: int) -> None")]
    fn set_stft_params(&mut self, n_fft: usize, hop_size: usize) -> PyResult<()> {
        let n_fft = nzu_or_err(n_fft)?;
        let hop_size = nzu_or_err(hop_size)?;
        self.inner.set_stft_params(n_fft, hop_size);
        Ok(())
    }

    /// Set median filter sizes.
    ///
    /// Parameters
    /// ----------
    /// harmonic: int
    ///     Harmonic filter size (odd numbers recommended)
    /// percussive: int
    ///     Percussive filter size (odd numbers recommended)
    #[pyo3(signature=(harmonic_size: "int", percussive_size: "int"), text_signature="($self, harmonic_size: int, percussive_size: int) -> None")]
    fn set_filter_sizes(&mut self, harmonic_size: usize, percussive_size: usize) {
        self.inner.set_filter_sizes(harmonic_size, percussive_size);
    }

    /// Set mask softness parameter.
    ///
    /// Parameters
    /// ----------
    /// softness: float
    ///     Softness value (0.0 = hard mask, 1.0 = completely soft)
    #[pyo3(signature=(softness: "float"), text_signature="($self, softness: float) -> None")]
    fn set_mask_softness(&mut self, softness: f64) {
        self.inner.set_mask_softness(softness);
    }

    // /// Validate HPSS configuration.
    // ///
    // /// Parameters
    // /// ----------
    // /// sample_rate: float
    // ///     Sample rate in Hz
    // ///
    // /// Returns
    // /// -------
    // /// HpssConfig
    // ///    This instance if valid, otherwise raises ValueError.
    // #[pyo3(signature=(sample_rate: "float"), text_signature="($self, sample_rate: float) -> HpssConfig")]
    // fn validate(&self, sample_rate: f64) -> PyResult<Self> {
    //     let inner = self.inner.validate(sample_rate).map_err(audio_err_to_py)?;
    //     Ok(PyHpssConfig { inner })
    // }

    /// Calculate the number of frequency bins for this configuration.
    #[pyo3(signature=(), text_signature="($self) -> int")]
    fn num_freq_bins(&self) -> usize {
        self.inner.num_freq_bins().get()
    }

    /// Calculate the frequency resolution in Hz.
    #[pyo3(signature=(sample_rate: "float"), text_signature="($self, sample_rate: float) -> float")]
    fn freq_resolution(&self, sample_rate: f64) -> f64 {
        self.inner.freq_resolution(sample_rate)
    }

    /// Calculate the time resolution in seconds.
    #[pyo3(signature=(sample_rate: "float"), text_signature="($self, sample_rate: float) -> float")]
    fn time_resolution(&self, sample_rate: f64) -> f64 {
        self.inner.time_resolution(sample_rate)
    }
}

impl_py_wrapper_core!(PyHpssConfig, HpssConfig);
impl_py_repr!(PyHpssConfig);

/// Configuration for adaptive thresholding in peak picking.
///
/// Adaptive thresholding dynamically adjusts the detection threshold based on
/// local characteristics of the onset strength function to improve detection
/// accuracy across varying signal conditions.
#[pyclass(
    name = "AdaptiveThresholdConfig",
    from_py_object,
    module = "audio_samples.types"
)]
#[derive(Default, Debug, Clone, Copy, PartialEq)]
pub struct PyAdaptiveThresholdConfig {
    pub(crate) inner: AdaptiveThresholdConfig,
}

#[pymethods]
impl PyAdaptiveThresholdConfig {
    /// Create a new AdaptiveThresholdConfig.
    ///
    /// Parameters
    /// ----------
    /// method : AdaptiveThresholdMethod
    ///     Adaptive thresholding method to use.
    /// delta : float
    ///     Delta offset for delta-based thresholding.
    /// percentile : float
    ///    Percentile value for percentile-based thresholding.
    /// window_size : int
    ///    Size of the rolling window in frames.
    /// min_threshold : float
    ///    Minimum allowable threshold.
    /// max_threshold : float
    ///   Maximum allowable threshold.
    ///
    /// Returns
    /// -------
    /// AdaptiveThresholdConfig
    ///   A new adaptive threshold configuration.
    #[new]
    #[pyo3(signature = (method: "AdaptiveThresholdMethod", *, delta: "float", percentile: "float", window_size: "int", min_threshold: "float", max_threshold: "float"), text_signature="($cls, method: AdaptiveThresholdMethod, *, delta: float, percentile: float, window_size: int, min_threshold: float, max_threshold: float) -> None")]
    fn new(
        method: PyAdaptiveThresholdMethod,
        delta: f64,
        percentile: f64,
        window_size: usize,
        min_threshold: f64,
        max_threshold: f64,
    ) -> Self {
        PyAdaptiveThresholdConfig {
            inner: AdaptiveThresholdConfig::new(
                method.inner,
                delta,
                percentile,
                window_size,
                min_threshold,
                max_threshold,
            ),
        }
    }

    /// Create a delta-based adaptive threshold configuration.
    ///
    /// Parameters
    /// ----------
    /// delta : float
    ///     Delta offset for thresholding.
    /// window_size : int
    ///     Size of the rolling window in frames.
    ///
    /// Returns
    /// -------
    /// AdaptiveThresholdConfig
    ///     A new delta-based adaptive threshold configuration.
    #[classmethod]
    #[pyo3(signature = (delta: "float", window_size: "int"), text_signature="($cls, delta: float, window_size: int) -> AdaptiveThresholdConfig")]
    fn delta(_cls: &Bound<'_, PyType>, delta: f64, window_size: usize) -> Self {
        PyAdaptiveThresholdConfig {
            inner: AdaptiveThresholdConfig::delta(delta, window_size),
        }
    }

    /// Create a percentile-based adaptive threshold configuration.
    ///
    /// Parameters
    /// ----------
    /// percentile : float
    ///     Percentile value for thresholding.
    /// window_size : int
    ///     Size of the rolling window in frames.
    ///
    /// Returns
    /// -------
    /// AdaptiveThresholdConfig
    ///     A new percentile-based adaptive threshold configuration.
    #[classmethod]
    #[pyo3(signature = (percentile: "float", window_size: "int"), text_signature="($cls, percentile: float, window_size: int) -> AdaptiveThresholdConfig")]
    fn percentile(_cls: &Bound<'_, PyType>, percentile: f64, window_size: usize) -> Self {
        PyAdaptiveThresholdConfig {
            inner: AdaptiveThresholdConfig::percentile(percentile, window_size),
        }
    }

    /// Create a combined adaptive threshold configuration.
    ///
    /// Parameters
    /// ----------
    /// delta : float
    ///    Delta offset for thresholding.
    /// percentile : float
    ///    Percentile value for thresholding.
    /// window_size : int
    ///    Size of the rolling window in frames.
    ///
    /// Returns
    /// -------
    /// AdaptiveThresholdConfig
    ///    A new combined adaptive threshold configuration.
    #[classmethod]
    #[pyo3(signature = (delta: "float", percentile: "float", window_size: "int"), text_signature="($cls, delta: float, percentile: float, window_size: int) -> AdaptiveThresholdConfig")]
    fn combined(_cls: &Bound<'_, PyType>, delta: f64, percentile: f64, window_size: usize) -> Self {
        PyAdaptiveThresholdConfig {
            inner: AdaptiveThresholdConfig::combined(delta, percentile, window_size),
        }
    }

    /// Set the minimum allowable threshold.
    #[setter]
    fn set_min_threshold(&mut self, value: f64) {
        self.inner.min_threshold = value;
    }

    /// Set the maximum allowable threshold.
    #[setter]
    fn set_max_threshold(&mut self, value: f64) {
        self.inner.max_threshold = value;
    }

    /// Validate the configuration parameters.
    ///
    /// Raises an error if any parameters are invalid.
    #[pyo3(signature = (), text_signature="($self) -> AdaptiveThresholdConfig")]
    fn validate(&mut self) -> PyResult<Self> {
        self.inner.validate().map_err(audio_err_to_py)?;
        Ok(Self { inner: self.inner })
    }
}

impl_py_wrapper_core!(PyAdaptiveThresholdConfig, AdaptiveThresholdConfig);
impl_py_default_static!(PyAdaptiveThresholdConfig);
impl_py_repr!(PyAdaptiveThresholdConfig);

/// Configuration for peak picking with temporal constraints.
///
/// Peak picking identifies local maxima in the onset strength function that
/// exceed a threshold. Temporal constraints ensure detected peaks are
/// separated by minimum time intervals and can include smoothing.
#[pyclass(
    name = "PeakPickingConfig",
    from_py_object,
    module = "audio_samples.types"
)]
#[derive(Default, Debug, Clone, Copy, PartialEq)]
pub struct PyPeakPickingConfig {
    pub(crate) inner: PeakPickingConfig,
}

#[pymethods]
impl PyPeakPickingConfig {
    /// Create a new PeakPickingConfig.
    ///
    /// Parameters
    /// ----------
    /// adaptive_threshold_config : AdaptiveThresholdConfig
    ///     Adaptive thresholding configuration.
    /// min_peak_separation : int
    ///     Minimum separation between detected peaks in frames.
    /// pre_emphasis : bool
    ///     Whether to apply pre-emphasis filtering.
    /// pre_emphasis_coeff : float
    ///     Pre-emphasis filter coefficient.
    /// median_filter : bool
    ///     Whether to apply median filtering.
    /// median_filter_length : int
    ///     Length of the median filter in frames.
    /// normalize_onset_strength : bool
    ///     Whether to normalize the onset strength function.
    /// normalization_method : NormalizationMethod
    ///     Method for normalization.
    #[new]
    #[pyo3(signature = (*, adaptive_threshold_config: "AdaptiveThresholdConfig",
        min_peak_separation: "int",
        pre_emphasis: "bool",
        pre_emphasis_coeff: "float",
        median_filter: "bool",
        median_filter_length: "int",
        normalize_onset_strength: "bool",
        normalization_method: "NormalizationMethod"),
        text_signature="($cls, *, adaptive_threshold_config: AdaptiveThresholdConfig, min_peak_separation: int, pre_emphasis: bool, pre_emphasis_coeff: float, median_filter: bool, median_filter_length: int, normalize_onset_strength: bool, normalization_method: NormalizationMethod) -> PeakPickingConfig")]
    fn new(
        adaptive_threshold_config: PyAdaptiveThresholdConfig,
        min_peak_separation: usize,
        pre_emphasis: bool,
        pre_emphasis_coeff: f64,
        median_filter: bool,
        median_filter_length: usize,
        normalize_onset_strength: bool,
        normalization_method: PyNormalizationMethod,
    ) -> PyResult<Self> {
        let min_peak_separation = nzu_or_err(min_peak_separation)?;
        let median_filter_length = nzu_or_err(median_filter_length)?;
        Ok(PyPeakPickingConfig {
            inner: PeakPickingConfig::new(
                adaptive_threshold_config.inner,
                min_peak_separation,
                pre_emphasis,
                pre_emphasis_coeff,
                median_filter,
                median_filter_length,
                normalize_onset_strength,
                normalization_method.inner,
            ),
        })
    }

    /// Music preset configuration.
    #[classattr]
    fn music() -> Self {
        PyPeakPickingConfig {
            inner: PeakPickingConfig::music(),
        }
    }

    /// Speech preset configuration.
    #[classattr]
    fn speech() -> Self {
        PyPeakPickingConfig {
            inner: PeakPickingConfig::speech(),
        }
    }

    /// Drums preset configuration.
    #[classattr]
    fn drums() -> Self {
        PyPeakPickingConfig {
            inner: PeakPickingConfig::drums(),
        }
    }

    /// Minimum separation between detected peaks in frames.
    #[setter]
    fn set_min_peak_separation(&mut self, value: usize) -> PyResult<()> {
        self.inner.min_peak_separation = nzu_or_err(value)?;
        Ok(())
    }

    /// Set minimum peak separation in milliseconds.
    #[pyo3(signature = (value: "float", sample_rate: "float"), text_signature="($self, value: float, sample_rate: float) -> None")]
    fn set_min_peak_separation_ms(&mut self, value: f64, sample_rate: f64) {
        self.inner.set_min_peak_separation_ms(value, sample_rate);
    }

    /// Enable or disable pre-emphasis filtering.
    #[pyo3(signature = (enabled: "bool", coeff: "float"), text_signature="($self, enabled: bool, coeff: float) -> None")]
    fn set_pre_emphasis(&mut self, enabled: bool, coeff: f64) {
        self.inner.pre_emphasis = enabled;
        self.inner.pre_emphasis_coeff = coeff;
    }

    /// Enable or disable median filtering.
    #[pyo3(signature = (enabled: "bool", length: "int"), text_signature="($self, enabled: bool, length: int) -> None")]
    fn set_median_filter(&mut self, enabled: bool, length: usize) -> PyResult<()> {
        self.inner.median_filter = enabled;
        self.inner.median_filter_length = nzu_or_err(length)?;
        Ok(())
    }

    /// Validate the configuration parameters.
    #[pyo3(signature = (), text_signature="($self) -> PeakPickingConfig")]
    fn validate(&mut self) -> PyResult<Self> {
        self.inner.validate().map_err(audio_err_to_py)?;
        Ok(Self { inner: self.inner })
    }
}
impl_py_wrapper_core!(PyPeakPickingConfig, PeakPickingConfig);
impl_py_default_static!(PyPeakPickingConfig);
impl_py_repr!(PyPeakPickingConfig);

/// Spectral flux variant for onset detection.
///
/// Different flux formulations emphasise different types of spectral change
/// and are therefore suited to different classes of musical and acoustic
/// events.
#[pyclass(
    name = "SpectralFluxMethod",
    from_py_object,
    module = "audio_samples.types"
)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PySpectralFluxMethod {
    pub(crate) inner: SpectralFluxMethod,
}

#[pymethods]
impl PySpectralFluxMethod {
    /// Energy-based spectral flux.
    ///
    /// Measures positive changes in spectral energy between successive frames.
    /// Performs well for transient and percussive onsets.
    #[classattr]
    fn energy() -> Self {
        PySpectralFluxMethod {
            inner: SpectralFluxMethod::Energy,
        }
    }

    /// Magnitude-based spectral flux.
    ///
    /// Measures positive changes in spectral magnitude and is more sensitive
    /// to subtle spectral variation, making it effective for tonal material.
    #[classattr]
    fn magnitude() -> Self {
        PySpectralFluxMethod {
            inner: SpectralFluxMethod::Magnitude,
        }
    }

    /// Complex-domain spectral flux.
    ///
    /// Incorporates phase information in addition to magnitude, improving
    /// robustness to noise and spectral smearing at increased computational
    /// cost.
    #[classattr]
    fn complex() -> Self {
        PySpectralFluxMethod {
            inner: SpectralFluxMethod::Complex,
        }
    }

    /// Rectified complex-domain spectral flux.
    ///
    /// Suppresses negative phase contributions to balance sensitivity and
    /// robustness.
    #[classattr]
    fn rectified_complex() -> Self {
        PySpectralFluxMethod {
            inner: SpectralFluxMethod::RectifiedComplex,
        }
    }
}

impl_py_wrapper_core!(PySpectralFluxMethod, SpectralFluxMethod);
impl_py_wrapper_fromstr!(PySpectralFluxMethod, SpectralFluxMethod);
impl_py_repr!(PySpectralFluxMethod);

/// Configuration for spectral flux onset detection.
///
/// Spectral flux measures the rate of change of the magnitude spectrum
/// between consecutive frames, providing effective onset detection for
/// both percussive and tonal instruments.
#[pyclass(
    name = "SpectralFluxConfig",
    from_py_object,
    module = "audio_samples.types"
)]
#[derive(Debug, Clone, PartialEq)]
pub struct PySpectralFluxConfig {
    pub(crate) inner: SpectralFluxConfig,
}

#[pymethods]
impl PySpectralFluxConfig {
    /// Create a new SpectralFluxConfig.
    ///
    /// Parameters
    /// ----------
    /// cqt_params : CqtParams
    ///     CQT parameters object from spectrograms module.
    /// hop_size : int
    ///     Hop size in samples (must be > 0).
    /// window_size : Optional[int]
    ///     Window size in samples (must be > 0 if provided). If ``None``, defaults to CQT's minimum window size.
    /// flux_method : SpectralFluxMethod
    ///     Spectral flux computation method.
    /// peak_picking : PeakPickingConfig
    ///     Peak picking configuration.
    /// rectify : bool
    ///     Whether to rectify the onset strength function.
    /// log_compression : float
    ///     Logarithmic compression factor.
    ///
    /// Returns
    /// ------
    /// SpectralFluxConfig
    ///     A new spectral flux configuration.
    #[new]
    #[pyo3(signature = (*, cqt_params, hop_size, window_size = None, flux_method, peak_picking, rectify, log_compression), text_signature="($cls, *, cqt_params: CqtParams, hop_size: int, window_size: Optional[int], flux_method: SpectralFluxMethod, peak_picking: PeakPickingConfig, rectify: bool, log_compression: float)")]
    fn new(
        cqt_params: PyCqtParams,
        hop_size: usize,
        window_size: Option<usize>,
        flux_method: PySpectralFluxMethod,
        peak_picking: PyPeakPickingConfig,
        rectify: bool,
        log_compression: f64,
    ) -> PyResult<Self> {
        let hop_size_nz = nzu_or_err(hop_size)?;
        let window_size_nz = window_size.map(|w| nzu_or_err(w)).transpose()?;

        Ok(PySpectralFluxConfig {
            inner: SpectralFluxConfig::new(
                cqt_params.into(),
                hop_size_nz,
                window_size_nz,
                flux_method.inner,
                peak_picking.inner,
                rectify,
                log_compression,
            ),
        })
    }

    /// Percussive preset configuration.
    #[classattr]
    fn percussive() -> Self {
        PySpectralFluxConfig {
            inner: SpectralFluxConfig::percussive(),
        }
    }

    /// Musical preset configuration.
    #[classattr]
    fn musical() -> Self {
        PySpectralFluxConfig {
            inner: SpectralFluxConfig::musical(),
        }
    }

    /// Complex preset configuration.
    #[classattr]
    fn complex() -> Self {
        PySpectralFluxConfig {
            inner: SpectralFluxConfig::complex(),
        }
    }

    /// Validate the configuration parameters.
    #[pyo3(signature = (), text_signature="($self) -> SpectralFluxConfig")]
    fn validate(&mut self) -> PyResult<Self> {
        self.inner.validate().map_err(audio_err_to_py)?;
        Ok(Self {
            inner: self.inner.clone(),
        })
    }

    /// Get the CQT parameters.
    #[getter]
    fn cqt_params(&self) -> PyCqtParams {
        self.inner.cqt_params.clone().into()
    }

    /// Set the CQT parameters.
    #[setter]
    fn set_cqt_params(&mut self, value: PyCqtParams) {
        self.inner.cqt_params = value.into();
    }

    /// Get the hop size in samples.
    #[getter]
    fn hop_size(&self) -> usize {
        self.inner.hop_size.get()
    }

    /// Set the hop size in samples.
    #[setter]
    fn set_hop_size(&mut self, value: usize) -> PyResult<()> {
        self.inner.hop_size = nzu_or_err(value)?;
        Ok(())
    }

    /// Get the window size in samples, or None if it defaults to CQT's minimum window size.
    #[getter]
    fn window_size(&self) -> Option<usize> {
        self.inner.window_size.map(|w| w.get())
    }

    /// Set the window size in samples.
    #[setter]
    fn set_window_size(&mut self, value: Option<usize>) -> PyResult<()> {
        self.inner.window_size = value.map(|w| nzu_or_err(w)).transpose()?;
        Ok(())
    }

    /// Get the spectral flux computation method.
    #[getter]
    fn flux_method(&self) -> PySpectralFluxMethod {
        PySpectralFluxMethod {
            inner: self.inner.flux_method,
        }
    }

    /// Set the spectral flux computation method.
    #[setter]
    fn set_flux_method(&mut self, value: PySpectralFluxMethod) {
        self.inner.flux_method = value.inner;
    }

    /// Get the peak picking configuration.
    #[getter]
    fn peak_picking(&self) -> PyPeakPickingConfig {
        PyPeakPickingConfig {
            inner: self.inner.peak_picking,
        }
    }

    /// Set the peak picking configuration.
    #[setter]
    fn set_peak_picking(&mut self, value: PyPeakPickingConfig) {
        self.inner.peak_picking = value.inner;
    }

    /// Get whether to rectify the onset strength function.
    #[getter]
    fn rectify(&self) -> bool {
        self.inner.rectify
    }

    /// Set whether to rectify the onset strength function.
    #[setter]
    fn set_rectify(&mut self, value: bool) {
        self.inner.rectify = value;
    }

    /// Get the logarithmic compression factor.
    #[getter]
    fn log_compression(&self) -> f64 {
        self.inner.log_compression
    }

    /// Set the logarithmic compression factor.
    #[setter]
    fn set_log_compression(&mut self, value: f64) {
        self.inner.log_compression = value;
    }
}

impl_py_wrapper_core!(PySpectralFluxConfig, SpectralFluxConfig);
impl_py_repr!(PySpectralFluxConfig);

/// Configuration for complex domain onset detection.
///
/// Complex domain onset detection uses both magnitude and phase information
/// from the CQT to provide more accurate onset detection than magnitude-only
/// methods, especially for polyphonic music and complex timbres.
#[pyclass(
    name = "ComplexOnsetConfig",
    from_py_object,
    module = "audio_samples.types"
)]
#[derive(Debug, Clone, PartialEq)]
pub struct PyComplexOnsetConfig {
    pub(crate) inner: ComplexOnsetConfig,
}

#[pymethods]
impl PyComplexOnsetConfig {
    /// Create a new ComplexOnsetConfig.
    ///
    /// Parameters
    /// ----------
    /// cqt_config : CqtConfig
    ///     CQT configuration object.
    /// hop_size : int
    ///     Hop size in samples.
    /// window_size : Optional[int]
    ///     Window size in samples. If ``None``, defaults to CQT's minimum window size.
    /// peak_picking : PeakPickingConfig
    ///     Peak picking configuration.
    /// magnitude_weight : float
    ///     Weighting factor for magnitude component.
    /// phase_weight : float    
    ///     Weighting factor for phase component.
    /// magnitude_rectify : bool
    ///     Whether to rectify the magnitude component.
    /// phase_rectify : bool
    ///     Whether to rectify the phase component.
    /// log_compression : float
    ///     Logarithmic compression factor.
    ///
    /// Returns
    /// -------
    /// ComplexOnsetConfig
    ///     A new complex onset detection configuration.
    #[new]
    #[pyo3(signature = (*, cqt_params: "CqtParams", hop_size: "int", window_size: "Optional[int]", peak_picking: "PeakPickingConfig", magnitude_weight: "float", phase_weight: "float", magnitude_rectify: "bool", phase_rectify: "bool", log_compression: "float"), text_signature="($cls, *, cqt_params: CqtParams, hop_size: int, window_size: Optional[int], peak_picking: PeakPickingConfig, magnitude_weight: float, phase_weight: float, magnitude_rectify: bool, phase_rectify: bool, log_compression: float) -> ComplexOnsetConfig")]
    fn new(
        cqt_params: PyCqtParams,
        hop_size: usize,
        window_size: Option<usize>,
        peak_picking: PyPeakPickingConfig,
        magnitude_weight: f64,
        phase_weight: f64,
        magnitude_rectify: bool,
        phase_rectify: bool,
        log_compression: f64,
    ) -> PyResult<Self> {
        let hop_size_nz = nzu_or_err(hop_size)?;
        let window_size_nz = window_size.map(|w| nzu_or_err(w)).transpose()?;

        Ok(PyComplexOnsetConfig {
            inner: ComplexOnsetConfig::new(
                cqt_params.into(),
                hop_size_nz,
                window_size_nz,
                peak_picking.inner,
                magnitude_weight,
                phase_weight,
                magnitude_rectify,
                phase_rectify,
                log_compression,
            ),
        })
    }

    /// Percussive preset configuration.
    #[classattr]
    fn percussive() -> Self {
        PyComplexOnsetConfig {
            inner: ComplexOnsetConfig::percussive(),
        }
    }

    /// Musical preset configuration.
    #[classattr]
    fn musical() -> Self {
        PyComplexOnsetConfig {
            inner: ComplexOnsetConfig::musical(),
        }
    }

    /// Speech preset configuration.
    #[classattr]
    fn speech() -> Self {
        PyComplexOnsetConfig {
            inner: ComplexOnsetConfig::speech(),
        }
    }

    /// Set the magnitude and phase weighting factors.
    #[pyo3(signature = (magnitude_weight: "float", phase_weight: "float"), text_signature="($self, magnitude_weight: float, phase_weight: float) -> None")]
    fn set_weights(&mut self, magnitude_weight: f64, phase_weight: f64) {
        self.inner.magnitude_weight = magnitude_weight;
        self.inner.phase_weight = phase_weight;
    }

    /// Validate the configuration parameters.
    #[pyo3(signature = (), text_signature="($self) -> ComplexOnsetConfig")]
    fn validate(&mut self) -> PyResult<Self> {
        self.inner.validate().map_err(audio_err_to_py)?;
        Ok(Self {
            inner: self.inner.clone(),
        })
    }

    /// Get the CQT parameters.
    #[getter]
    fn cqt_params(&self) -> PyCqtParams {
        self.inner.cqt_config.clone().into()
    }

    /// Set the CQT parameters.
    #[setter]
    fn set_cqt_params(&mut self, value: PyCqtParams) {
        self.inner.cqt_config = value.into();
    }

    /// Get the hop size in samples.
    #[getter]
    fn hop_size(&self) -> usize {
        self.inner.hop_size.get()
    }

    /// Set the hop size in samples.
    #[setter]
    fn set_hop_size(&mut self, value: usize) -> PyResult<()> {
        self.inner.hop_size = nzu_or_err(value)?;
        Ok(())
    }

    /// Get the window size in samples, or None if it defaults to CQT's minimum window size.
    #[getter]
    fn window_size(&self) -> Option<usize> {
        self.inner.window_size.map(|w| w.get())
    }

    /// Set the window size in samples.
    #[setter]
    fn set_window_size(&mut self, value: Option<usize>) -> PyResult<()> {
        self.inner.window_size = value.map(|w| nzu_or_err(w)).transpose()?;
        Ok(())
    }

    /// Get the peak picking configuration.
    #[getter]
    fn peak_picking(&self) -> PyPeakPickingConfig {
        PyPeakPickingConfig {
            inner: self.inner.peak_picking,
        }
    }

    /// Set the peak picking configuration.
    #[setter]
    fn set_peak_picking(&mut self, value: PyPeakPickingConfig) {
        self.inner.peak_picking = value.inner;
    }

    /// Get the magnitude weighting factor.
    #[getter]
    fn magnitude_weight(&self) -> f64 {
        self.inner.magnitude_weight
    }

    /// Set the magnitude weighting factor.
    #[setter]
    fn set_magnitude_weight(&mut self, value: f64) {
        self.inner.magnitude_weight = value;
    }

    /// Get the phase weighting factor.
    #[getter]
    fn phase_weight(&self) -> f64 {
        self.inner.phase_weight
    }

    /// Set the phase weighting factor.
    #[setter]
    fn set_phase_weight(&mut self, value: f64) {
        self.inner.phase_weight = value;
    }

    /// Get whether to rectify the magnitude component.
    #[getter]
    fn magnitude_rectify(&self) -> bool {
        self.inner.magnitude_rectify
    }

    /// Set whether to rectify the magnitude component.
    #[setter]
    fn set_magnitude_rectify(&mut self, value: bool) {
        self.inner.magnitude_rectify = value;
    }

    /// Get whether to rectify the phase component.
    #[getter]
    fn phase_rectify(&self) -> bool {
        self.inner.phase_rectify
    }

    /// Set whether to rectify the phase component.
    #[setter]
    fn set_phase_rectify(&mut self, value: bool) {
        self.inner.phase_rectify = value;
    }

    /// Get the logarithmic compression factor.
    #[getter]
    fn log_compression(&self) -> f64 {
        self.inner.log_compression
    }

    /// Set the logarithmic compression factor.
    #[setter]
    fn set_log_compression(&mut self, value: f64) {
        self.inner.log_compression = value;
    }
}

impl_py_wrapper_core!(PyComplexOnsetConfig, ComplexOnsetConfig);
impl_py_repr!(PyComplexOnsetConfig);

/// Python wrapper for OnsetDetectionConfig.
///
/// Configuration for onset detection using spectral flux method.
#[pyclass(
    name = "OnsetDetectionConfig",
    from_py_object,
    module = "audio_samples.types"
)]
#[derive(Clone, Debug)]
pub struct PyOnsetDetectionConfig {
    pub(crate) inner: OnsetDetectionConfig,
}

#[pymethods]
impl PyOnsetDetectionConfig {
    /// Create a new OnsetDetectionConfig with custom parameters.
    ///
    /// Parameters
    /// ----------
    /// cqt_params : PyCqtParams
    ///     CQT configuration for spectral analysis
    /// hop_size : int
    ///     Hop size for frame-based analysis in samples
    /// window_size : Optional[int]
    ///     Window size for CQT analysis in samples (None = auto-calculate)
    /// threshold : float
    ///     Threshold for onset detection
    /// min_onset_interval_secs : float
    ///     Minimum time between consecutive onsets in seconds
    /// pre_emphasis : float
    ///     Pre-emphasis factor for signal processing
    /// adaptive_threshold : bool
    ///     Whether to use adaptive thresholding
    /// median_filter_length : int
    ///     Length of median filter for smoothing
    /// adaptive_threshold_multiplier : float
    ///     Multiplier for adaptive threshold
    /// peak_picking : PyPeakPickingConfig
    ///     Peak picking configuration
    #[new]
    #[pyo3(signature = (cqt_params, hop_size, window_size=None, threshold=0.3, min_onset_interval_secs=0.07, pre_emphasis=0.0, adaptive_threshold=true, median_filter_length=3, adaptive_threshold_multiplier=3.0, peak_picking=None))]
    fn new(
        cqt_params: PyCqtParams,
        hop_size: usize,
        window_size: Option<usize>,
        threshold: f64,
        min_onset_interval_secs: f64,
        pre_emphasis: f64,
        adaptive_threshold: bool,
        median_filter_length: usize,
        adaptive_threshold_multiplier: f64,
        peak_picking: Option<PyPeakPickingConfig>,
    ) -> PyResult<Self> {
        let hop_size_nz = nzu_or_err(hop_size)?;
        let window_size_nz = window_size.map(nzu_or_err).transpose()?;
        let median_filter_nz = nzu_or_err(median_filter_length)?;
        let peak_picking_inner = peak_picking
            .map(|p| p.inner)
            .unwrap_or_else(PeakPickingConfig::default);

        Ok(PyOnsetDetectionConfig {
            inner: OnsetDetectionConfig::new(
                cqt_params.into(),
                hop_size_nz,
                window_size_nz,
                threshold,
                min_onset_interval_secs,
                pre_emphasis,
                adaptive_threshold,
                median_filter_nz,
                adaptive_threshold_multiplier,
                peak_picking_inner,
            ),
        })
    }

    /// Create a default OnsetDetectionConfig.
    #[classmethod]
    fn default(_cls: &Bound<'_, PyType>) -> Self {
        PyOnsetDetectionConfig {
            inner: OnsetDetectionConfig::default(),
        }
    }

    /// Create an OnsetDetectionConfig optimized for general music.
    #[classmethod]
    fn musical(_cls: &Bound<'_, PyType>) -> Self {
        PyOnsetDetectionConfig {
            inner: OnsetDetectionConfig::musical(),
        }
    }

    /// Create an OnsetDetectionConfig optimized for percussive sounds.
    #[classmethod]
    fn percussive(_cls: &Bound<'_, PyType>) -> Self {
        PyOnsetDetectionConfig {
            inner: OnsetDetectionConfig::percussive(),
        }
    }

    /// Create an OnsetDetectionConfig optimized for speech.
    #[classmethod]
    fn speech(_cls: &Bound<'_, PyType>) -> Self {
        PyOnsetDetectionConfig {
            inner: OnsetDetectionConfig::speech(),
        }
    }

    /// Get the CQT parameters.
    #[getter]
    fn cqt_params(&self) -> PyCqtParams {
        self.inner.cqt_params.clone().into()
    }

    /// Set the CQT parameters.
    #[setter]
    fn set_cqt_params(&mut self, value: PyCqtParams) {
        self.inner.cqt_params = value.into();
    }

    /// Get the hop size in samples.
    #[getter]
    fn hop_size(&self) -> usize {
        self.inner.hop_size.get()
    }

    /// Set the hop size in samples.
    #[setter]
    fn set_hop_size(&mut self, value: usize) -> PyResult<()> {
        self.inner.hop_size = nzu_or_err(value)?;
        Ok(())
    }

    /// Get the window size in samples, or None if it defaults to CQT's minimum window size.
    #[getter]
    fn window_size(&self) -> Option<usize> {
        self.inner.window_size.map(|w| w.get())
    }

    /// Set the window size in samples.
    #[setter]
    fn set_window_size(&mut self, value: Option<usize>) -> PyResult<()> {
        self.inner.window_size = value.map(|w| nzu_or_err(w)).transpose()?;
        Ok(())
    }

    /// Get the onset detection threshold.
    #[getter]
    fn threshold(&self) -> f64 {
        self.inner.threshold
    }

    /// Set the onset detection threshold.
    #[setter]
    fn set_threshold(&mut self, value: f64) {
        self.inner.threshold = value;
    }

    /// Get the minimum onset interval in seconds.
    #[getter]
    fn min_onset_interval_secs(&self) -> f64 {
        self.inner.min_onset_interval_secs
    }

    /// Set the minimum onset interval in seconds.
    #[setter]
    fn set_min_onset_interval_secs(&mut self, value: f64) {
        self.inner.min_onset_interval_secs = value;
    }

    /// Get the pre-emphasis factor.
    #[getter]
    fn pre_emphasis(&self) -> f64 {
        self.inner.pre_emphasis
    }

    /// Set the pre-emphasis factor.
    #[setter]
    fn set_pre_emphasis(&mut self, value: f64) {
        self.inner.pre_emphasis = value;
    }

    /// Get whether adaptive thresholding is enabled.
    #[getter]
    fn adaptive_threshold(&self) -> bool {
        self.inner.adaptive_threshold
    }

    /// Set whether adaptive thresholding is enabled.
    #[setter]
    fn set_adaptive_threshold(&mut self, value: bool) {
        self.inner.adaptive_threshold = value;
    }

    /// Get the median filter length.
    #[getter]
    fn median_filter_length(&self) -> usize {
        self.inner.median_filter_length.get()
    }

    /// Set the median filter length.
    #[setter]
    fn set_median_filter_length(&mut self, value: usize) -> PyResult<()> {
        self.inner.median_filter_length = nzu_or_err(value)?;
        Ok(())
    }

    /// Get the adaptive threshold multiplier.
    #[getter]
    fn adaptive_threshold_multiplier(&self) -> f64 {
        self.inner.adaptive_threshold_multiplier
    }

    /// Set the adaptive threshold multiplier.
    #[setter]
    fn set_adaptive_threshold_multiplier(&mut self, value: f64) {
        self.inner.adaptive_threshold_multiplier = value;
    }

    /// Get the peak picking configuration.
    #[getter]
    fn peak_picking(&self) -> PyPeakPickingConfig {
        PyPeakPickingConfig {
            inner: self.inner.peak_picking,
        }
    }

    /// Set the peak picking configuration.
    #[setter]
    fn set_peak_picking(&mut self, value: PyPeakPickingConfig) {
        self.inner.peak_picking = value.inner;
    }
}

impl_py_wrapper_core!(PyOnsetDetectionConfig, OnsetDetectionConfig);
impl_py_repr!(PyOnsetDetectionConfig);

/// Python wrapper for EnvelopeFollower.
///
/// Envelope follower for attack/release envelope tracking.
#[pyclass(
    name = "EnvelopeFollower",
    from_py_object,
    module = "audio_samples.types"
)]
#[derive(Clone, Debug)]
pub struct PyEnvelopeFollower {
    pub(crate) inner: EnvelopeFollower,
}

#[pymethods]
impl PyEnvelopeFollower {
    /// Create a new EnvelopeFollower.
    ///
    /// Parameters
    /// ----------
    /// attack_ms : float
    ///     Attack time in milliseconds
    /// release_ms : float
    ///     Release time in milliseconds
    /// sample_rate : float
    ///     Sample rate in Hz
    /// detection_method : DynamicRangeMethod
    ///     Detection method (Peak, Rms, or Hybrid)
    #[new]
    #[pyo3(signature = (attack_ms, release_ms, sample_rate, detection_method))]
    fn new(
        attack_ms: f64,
        release_ms: f64,
        sample_rate: f64,
        detection_method: PyDynamicRangeMethod,
    ) -> Self {
        PyEnvelopeFollower {
            inner: EnvelopeFollower::new(
                attack_ms,
                release_ms,
                sample_rate,
                detection_method.inner,
            ),
        }
    }

    /// Create a default EnvelopeFollower with 10ms attack and 100ms release at 44100 Hz.
    #[classmethod]
    fn default(_cls: &Bound<'_, PyType>) -> Self {
        PyEnvelopeFollower {
            inner: EnvelopeFollower::new(10.0, 100.0, 44100.0, DynamicRangeMethod::Peak),
        }
    }
}

impl_py_wrapper_core!(PyEnvelopeFollower, EnvelopeFollower);
impl_py_repr!(PyEnvelopeFollower);

/// Python wrapper for BeatTrackingConfig.
///
/// Configuration for beat detection.
#[pyclass(
    name = "BeatTrackingConfig",
    from_py_object,
    module = "audio_samples.types"
)]
#[derive(Clone, Debug)]
pub struct PyBeatTrackingConfig {
    pub(crate) inner: BeatTrackingConfig,
}

#[pymethods]
impl PyBeatTrackingConfig {
    /// Create a new BeatTrackingConfig.
    ///
    /// Parameters
    /// ----------
    /// tempo_bpm : float
    ///     Target tempo in beats per minute
    /// tolerance : float, optional
    ///     Beat timing tolerance in seconds. If None, defaults to 10% of the inter-beat interval
    /// onset_config : OnsetDetectionConfig
    ///     Configuration for onset detection
    #[new]
    #[pyo3(signature = (tempo_bpm, onset_config, tolerance=None))]
    fn new(tempo_bpm: f64, onset_config: PyOnsetDetectionConfig, tolerance: Option<f64>) -> Self {
        PyBeatTrackingConfig {
            inner: BeatTrackingConfig::new(tempo_bpm, tolerance, onset_config.inner),
        }
    }

    /// Get the tempo in BPM.
    #[getter]
    fn tempo_bpm(&self) -> f64 {
        self.inner.tempo_bpm
    }

    /// Set the tempo in BPM.
    #[setter]
    fn set_tempo_bpm(&mut self, tempo_bpm: f64) {
        self.inner.tempo_bpm = tempo_bpm;
    }

    /// Get the tolerance in seconds.
    #[getter]
    fn tolerance(&self) -> Option<f64> {
        self.inner.tolerance
    }

    /// Set the tolerance in seconds.
    #[setter]
    fn set_tolerance(&mut self, tolerance: Option<f64>) {
        self.inner.tolerance = tolerance;
    }

    /// Get the onset detection config.
    #[getter]
    fn onset_config(&self) -> PyOnsetDetectionConfig {
        PyOnsetDetectionConfig {
            inner: self.inner.onset_config.clone(),
        }
    }

    /// Set the onset detection config.
    #[setter]
    fn set_onset_config(&mut self, onset_config: PyOnsetDetectionConfig) {
        self.inner.onset_config = onset_config.inner;
    }
}

impl_py_wrapper_core!(PyBeatTrackingConfig, BeatTrackingConfig);
impl_py_repr!(PyBeatTrackingConfig);

/// Python wrapper for BeatTrackingData.
///
/// Beat tracking results containing tempo and beat timestamps.
#[pyclass(
    name = "BeatTrackingData",
    from_py_object,
    module = "audio_samples.types"
)]
#[derive(Clone, Debug)]
pub struct PyBeatTrackingData {
    pub(crate) inner: BeatTrackingData,
}

#[pymethods]
impl PyBeatTrackingData {
    /// Create a new BeatTrackingData.
    ///
    /// Parameters
    /// ----------
    /// tempo_bpm : float
    ///     Estimated tempo in beats per minute
    /// beat_times : list[float]
    ///     Beat timestamps in seconds
    /// config : BeatTrackingConfig
    ///     Configuration used for beat detection
    #[new]
    #[pyo3(signature = (tempo_bpm, beat_times, config))]
    fn new(tempo_bpm: f64, beat_times: Vec<f64>, config: PyBeatTrackingConfig) -> Self {
        PyBeatTrackingData {
            inner: BeatTrackingData::new(tempo_bpm, beat_times, config.inner),
        }
    }

    /// Get the estimated tempo in BPM.
    #[getter]
    fn tempo_bpm(&self) -> f64 {
        self.inner.tempo_bpm
    }

    /// Get the beat timestamps in seconds.
    #[getter]
    fn beat_times(&self) -> Vec<f64> {
        self.inner.beat_times.clone()
    }

    /// Get the configuration used.
    #[getter]
    fn config(&self) -> PyBeatTrackingConfig {
        PyBeatTrackingConfig {
            inner: self.inner.config.clone(),
        }
    }

    /// String representation showing tempo and beat count.
    fn __str__(&self) -> String {
        format!(
            "BeatTrackingData(tempo={:.2} BPM, beats={})",
            self.inner.tempo_bpm,
            self.inner.beat_times.len()
        )
    }
}

impl_py_wrapper_core!(PyBeatTrackingData, BeatTrackingData);
impl_py_repr!(PyBeatTrackingData);
