//! Configuration, enumeration and result classes of the `AudioSamples` methods.
use std::hash::Hash;

use audio_samples::operations::types::{
    AdaptiveThresholdConfig, AdaptiveThresholdMethod, BesselNorm, ChannelReduction,
    CompressorConfig, DynamicRangeMethod, EqBand, EqBandType, ExpanderConfig, FadeCurve,
    FilterResponse, GateConfig, IirFilterDesign, IirFilterType, KneeType, LimiterConfig,
    MonoConversionMethod, NoiseColor, PadSide, ParametricEq, PeakPickingConfig, PerturbationConfig,
    PerturbationMethod, PitchDetectionMethod, ResamplingQuality, SideChainConfig, SpectrogramScale,
    StereoConversionMethod, ThreeBandEqConfig, VadChannelPolicy, VadConfig, VadMethod,
};

// Power spectral density result and pitch-analysis result types.
use audio_samples::operations::pitch_analysis::{Key, Mode, PitchClass, PitchContour, PitchFrame};
use audio_samples::operations::transforms::Psd;

// Types re-exported from operations submodules
use audio_samples::operations::{
    BeatTrackingConfig, BeatTrackingData, ComplexOnsetConfig, HpssConfig, OnsetDetectionConfig,
    SpectralFluxConfig, SpectralFluxMethod,
};

use audio_samples::operations::dynamic_range::EnvelopeFollower;
use audio_samples::{I24, NormalizationMethod, SampleType, nzu};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayDescr, PyArrayDescrMethods};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::PyType;
use spectrograms::python::{PyCqtParams, PyStftParams};
use spectrograms::{StftParams, WindowType};

use crate::operations::iir_filtering::PySosFilter;
use crate::{
    audio_err_to_py, impl_py_default_static, impl_py_repr, impl_py_wrapper_core,
    impl_py_wrapper_fromstr, nzu_or_err,
};

/// Configuration and result classes of the ``AudioSamples`` methods. Every class is
/// also available from the top-level package.
#[pymodule(submodule, name = "types")]
pub mod py_module {
    #[pymodule_export]
    use super::{
        PyAdaptiveThresholdConfig, PyAdaptiveThresholdMethod, PyBeatTrackingConfig,
        PyBeatTrackingData, PyChannelReduction, PyComplexOnsetConfig, PyCompressorConfig,
        PyDynamicRangeMethod, PyEnvelopeFollower, PyEqBand, PyEqBandType, PyExpanderConfig,
        PyFadeCurve, PyFilterResponse, PyGateConfig, PyHpssConfig, PyIirFilterDesign,
        PyIirFilterType, PyKey, PyKneeType, PyLimiterConfig, PyMode, PyMonoConversionMethod,
        PyNoiseColor, PyNormalizationMethod, PyOnsetDetectionConfig, PyPadSide, PyParametricEq,
        PyPeakPickingConfig, PyPerturbationConfig, PyPerturbationMethod, PyPitchClass,
        PyPitchContour, PyPitchDetectionMethod, PyPitchFrame, PyPsd, PyResamplingQuality,
        PySampleType, PySideChainConfig, PySosFilter, PySpectralFluxConfig, PySpectralFluxMethod,
        PySpectrogramScale, PyStereoConversionMethod, PyThreeBandEqConfig, PyVadChannelPolicy,
        PyVadConfig, PyVadMethod, PyWindowType,
    };
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
            SampleType::U8 => Self::U8,
            SampleType::I16 => Self::I16,
            SampleType::I24 => Self::I24,
            SampleType::I32 => Self::I32,
            SampleType::F32 => Self::F32,
            SampleType::F64 => Self::F64,
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

    pub(crate) const fn from_native(st: SampleType) -> Option<Self> {
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
    const fn left() -> Self {
        Self {
            inner: PadSide::Left,
        }
    }

    /// Pad on the right side (end) of the signal.
    #[classattr]
    const fn right() -> Self {
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
    const fn minmax() -> Self {
        Self {
            inner: NormalizationMethod::MinMax,
        }
    }

    /// Z-score normalisation.
    ///
    /// Subtracts the mean of the samples and divides by their standard
    /// deviation. This produces zero-mean, unit-variance data and is commonly
    /// used in statistical and machine learning pipelines.
    #[classattr]
    const fn zscore() -> Self {
        Self {
            inner: NormalizationMethod::ZScore,
        }
    }

    /// Peak normalisation.
    ///
    /// Scales samples so that the maximum absolute amplitude equals 1. This
    /// preserves waveform shape while enforcing a strict amplitude bound.
    #[classattr]
    const fn peak() -> Self {
        Self {
            inner: NormalizationMethod::Peak,
        }
    }

    /// Mean normalisation.
    ///
    /// Recentres samples by subtracting the mean value without rescaling their
    /// variance.
    #[classattr]
    const fn mean() -> Self {
        Self {
            inner: NormalizationMethod::Mean,
        }
    }

    /// Median normalisation.
    ///
    /// Recentres samples by subtracting the median value. This method is more
    /// robust to outliers than mean-based normalisation.
    #[classattr]
    const fn median() -> Self {
        Self {
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
/// The window types are:
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
    const fn hanning() -> Self {
        Self {
            inner: WindowType::Hanning,
        }
    }

    /// Hamming window.
    ///
    /// Similar to the Hanning window, with slightly different coefficients and
    /// side-lobe behaviour.
    #[classattr]
    const fn hamming() -> Self {
        Self {
            inner: WindowType::Hamming,
        }
    }

    /// Blackman window.
    ///
    /// Low spectral leakage at the cost of a wider main lobe and reduced
    /// frequency resolution.
    #[classattr]
    const fn blackman() -> Self {
        Self {
            inner: WindowType::Blackman,
        }
    }

    /// Rectangular window (no windowing).
    ///
    /// Best frequency resolution but high spectral leakage.
    #[classattr]
    const fn rectangular() -> Self {
        Self {
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
    #[pyo3(signature = (beta: "float"))]
    const fn kaiser(_cls: &Bound<'_, PyType>, beta: f64) -> Self {
        Self {
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
    #[pyo3(signature = (std: "float"))]
    const fn gaussian(_cls: &Bound<'_, PyType>, std: f64) -> Self {
        Self {
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
    const fn linear() -> Self {
        Self {
            inner: FadeCurve::Linear,
        }
    }

    /// Exponential fade.
    ///
    /// Faster change at the beginning, slower towards the end.
    #[classattr]
    const fn exponential() -> Self {
        Self {
            inner: FadeCurve::Exponential,
        }
    }

    /// Logarithmic fade.
    ///
    /// Slower change at the beginning, faster towards the end.
    #[classattr]
    const fn logarithmic() -> Self {
        Self {
            inner: FadeCurve::Logarithmic,
        }
    }

    /// Smooth-step fade.
    ///
    /// S-shaped curve with smooth transitions at both the start and end.
    #[classattr]
    const fn smooth_step() -> Self {
        Self {
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
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
pub struct PyVadMethod {
    pub(crate) inner: VadMethod,
}

#[pymethods]
impl PyVadMethod {
    /// Energy-based voice activity detection.
    ///
    /// Uses signal energy (e.g. RMS) and a threshold to detect activity.
    #[classattr]
    const fn energy() -> Self {
        Self {
            inner: VadMethod::Energy,
        }
    }

    /// Zero crossing rate (ZCR) based detection.
    ///
    /// Uses the rate of sign changes in the waveform as an indicator of activity.
    #[classattr]
    const fn zcr() -> Self {
        Self {
            inner: VadMethod::ZeroCrossing,
        }
    }

    /// Combined energy and zero crossing rate detection.
    ///
    /// Integrates both energy and ZCR cues for more robust detection.
    #[classattr]
    const fn combined() -> Self {
        Self {
            inner: VadMethod::Combined,
        }
    }

    /// Energy threshold and speech-band energy ratio detection.
    ///
    /// A frame is speech when its RMS level reaches ``energy_threshold_db`` and at
    /// least ``spectral_ratio_threshold`` of its energy (Hann window, real FFT of
    /// length ``frame_size``) lies between ``speech_band_low_hz`` and
    /// ``speech_band_high_hz`` (defaults: 0.6 of the energy in 300-3400 Hz).
    /// Rejects loud broadband noise, hum and hiss that ``energy`` accepts, and
    /// unvoiced fricatives, whose energy lies mostly above the band.
    #[classattr]
    const fn spectral() -> Self {
        Self {
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
    const fn average_to_mono() -> Self {
        Self {
            inner: VadChannelPolicy::AverageToMono,
        }
    }

    /// Run VAD independently on each channel and mark activity if any channel is active.
    #[classattr]
    const fn any_channel() -> Self {
        Self {
            inner: VadChannelPolicy::AnyChannel,
        }
    }

    /// Run VAD independently on each channel and mark activity only if all channels are active.
    #[classattr]
    const fn all_channels() -> Self {
        Self {
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
    #[pyo3(signature = (ch: "int"))]
    const fn channel(_cls: &Bound<'_, PyType>, ch: usize) -> Self {
        Self {
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
    /// method : ``VadMethod``
    ///     Voice activity detection method to use (positional-only).
    /// ``channel_policy`` : ``VadChannelPolicy``
    ///     Policy for handling multi-channel audio (positional-only, second).
    /// ``frame_size`` : int
    ///     Frame size in samples.
    /// ``hop_size`` : int
    ///     Hop size in samples (frame step). Must be less than or equal to
    ///     ``frame_size``.
    /// ``pad_end`` : bool
    ///     Whether to include a final partial frame padded with zeros.
    /// ``energy_threshold_db`` : float
    ///     Energy threshold in dBFS (RMS). Typical values range from approximately
    ///     -60.0 (very sensitive) to -30.0.
    /// ``zcr_min`` : float
    ///     Minimum acceptable zero crossing rate, expressed as crossings per
    ///     sample in the range [0, 1].
    /// ``zcr_max`` : float
    ///     Maximum acceptable zero crossing rate, expressed as crossings per
    ///     sample in the range [0, 1].
    /// ``min_speech_frames`` : int
    ///     Minimum number of consecutive speech frames required to keep a speech
    ///     region.
    /// ``min_silence_frames`` : int
    ///     Minimum number of consecutive non-speech frames required to keep a
    ///     silence region. Shorter silence gaps are filled as speech.
    /// ``hangover_frames`` : int
    ///     Number of frames to keep speech active after energy drops.
    /// ``smooth_frames`` : int
    ///     Majority-vote smoothing window in frames. A value of 1 disables
    ///     smoothing.
    /// ``speech_band_low_hz`` : float
    ///     Lower edge of the speech band in Hz (``VadMethod.spectral``).
    /// ``speech_band_high_hz`` : float
    ///     Upper edge of the speech band in Hz, at most the Nyquist frequency of
    ///     the audio (``VadMethod.spectral``).
    /// ``spectral_ratio_threshold`` : float
    ///     Minimum fraction, in ``[0, 1]``, of a frame's energy inside the speech
    ///     band for ``VadMethod.spectral`` to classify it as speech.
    ///
    /// Returns
    /// -------
    /// ``VadConfig``
    ///     A new VAD configuration instance.
    ///
    /// Notes
    /// -----
    /// This constructor does not automatically validate parameter consistency.
    /// Call ``validate()`` to explicitly check constraints.
    #[new]
    #[pyo3(signature = (method: "VadMethod", channel_policy: "VadChannelPolicy", /, frame_size: "int", hop_size: "int", pad_end: "bool", energy_threshold_db: "float", zcr_min: "float", zcr_max: "float", min_speech_frames: "int", min_silence_frames: "int", hangover_frames: "int", smooth_frames: "int", speech_band_low_hz: "float", speech_band_high_hz: "float", spectral_ratio_threshold: "float"), text_signature="(method, channel_policy, /, frame_size, hop_size, pad_end, energy_threshold_db, zcr_min, zcr_max, min_speech_frames, min_silence_frames, hangover_frames, smooth_frames, speech_band_low_hz, speech_band_high_hz, spectral_ratio_threshold)")]
    #[allow(
        clippy::too_many_arguments,
        reason = "the Python signature takes these arguments"
    )] // PyO3 constructor mirrors the Python signature
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
        Ok(Self {
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
        Self {
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
    /// ``VadConfig``
    ///     Returns the validated configuration if validation succeeds.
    ///
    /// Raises
    /// ------
    /// ``ValueError``
    ///     If any configuration parameter is invalid or inconsistent.
    #[pyo3(signature = ())]
    fn validate(&mut self) -> PyResult<Self> {
        self.inner.validate().map_err(audio_err_to_py)?;
        Ok(Self { inner: self.inner })
    }

    /// VAD method.
    #[getter]
    const fn method(&self) -> PyVadMethod {
        PyVadMethod {
            inner: self.inner.method,
        }
    }

    /// Policy for multi-channel audio.
    #[getter]
    const fn channel_policy(&self) -> PyVadChannelPolicy {
        PyVadChannelPolicy {
            inner: self.inner.channel_policy,
        }
    }

    /// Whether to include a final partial frame (zero-padded).
    #[getter]
    const fn pad_end(&self) -> bool {
        self.inner.pad_end
    }

    /// Analysis frame length in samples.
    #[getter]
    const fn frame_size(&self) -> usize {
        self.inner.frame_size.get()
    }

    /// Hop size in samples (frame step).
    #[getter]
    const fn hop_size(&self) -> usize {
        self.inner.hop_size.get()
    }

    /// Hangover in frames: speech stays active this many frames after energy drops.
    #[getter]
    const fn hangover_frames(&self) -> usize {
        self.inner.hangover_frames.get()
    }

    /// Majority-vote smoothing window in frames (1 = no smoothing).
    #[getter]
    const fn smooth_frames(&self) -> usize {
        self.inner.smooth_frames.get()
    }

    /// Minimum number of consecutive speech frames to keep a speech region.
    #[getter]
    const fn min_speech_frames(&self) -> usize {
        self.inner.min_speech_frames
    }

    /// Minimum number of consecutive non-speech frames to keep a silence region.
    #[getter]
    const fn min_silence_frames(&self) -> usize {
        self.inner.min_silence_frames
    }

    /// Energy threshold in dBFS (frame RMS).
    #[getter]
    const fn energy_threshold_db(&self) -> f64 {
        self.inner.energy_threshold_db
    }

    /// Minimum zero-crossing rate, in crossings per sample (``[0, 1]``).
    #[getter]
    const fn zcr_min(&self) -> f64 {
        self.inner.zcr_min
    }

    /// Maximum zero-crossing rate, in crossings per sample (``[0, 1]``).
    #[getter]
    const fn zcr_max(&self) -> f64 {
        self.inner.zcr_max
    }

    /// Lower bound of the speech band in Hz (``VadMethod.spectral``).
    #[getter]
    const fn speech_band_low_hz(&self) -> f64 {
        self.inner.speech_band_low_hz
    }

    /// Upper bound of the speech band in Hz (``VadMethod.spectral``).
    #[getter]
    const fn speech_band_high_hz(&self) -> f64 {
        self.inner.speech_band_high_hz
    }

    /// Threshold on the speech-band energy ratio (``VadMethod.spectral``).
    #[getter]
    const fn spectral_ratio_threshold(&self) -> f64 {
        self.inner.spectral_ratio_threshold
    }
}

impl_py_wrapper_core!(PyVadConfig, VadConfig);
impl_py_default_static!(PyVadConfig);
impl_py_repr!(PyVadConfig);

/// Quality level for resampling operations.
///
/// Every tier is delay-free (output sample ``k`` is the input reconstructed
/// at time ``k * in_rate / out_rate``) and produces exactly
/// ``ceil(n * out_rate / in_rate)`` samples; the tiers differ in the
/// anti-aliasing / anti-imaging filter (Nyquist of the lower rate):
///
/// - ``ResamplingQuality.fast`` (the default of ``AudioSamples.resample``):
///   short windowed sinc when downsampling (passband 0.80 x Nyquist, 70 dB
///   stopband); linear interpolation when upsampling (images not suppressed).
/// - ``ResamplingQuality.medium``: windowed sinc, passband 0.90 x Nyquist,
///   100 dB stopband.
/// - ``ResamplingQuality.high``: windowed sinc, passband 0.913 x Nyquist,
///   125 dB stopband.
/// - ``ResamplingQuality.very_high``: windowed sinc, 185 dB stopband
///   (limited to about -140 dB by rounding for float32 audio); passband
///   0.95 x Nyquist for decimation by a power of two, 0.98 for other ratios
///   with no prime factor above 7, 0.996 for the rest (such as pitch shifts).
///
/// Instances are immutable enum-like values accessed via the class
/// attributes; resampling methods also accept the names ``"fast"``,
/// ``"medium"``, ``"high"`` and ``"very_high"``.
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
    pub const fn resample_fast() -> Self {
        Self {
            inner: ResamplingQuality::Fast,
        }
    }

    #[inline]
    pub const fn resample_mediuem() -> Self {
        Self {
            inner: ResamplingQuality::Medium,
        }
    }

    #[inline]
    pub const fn resample_high() -> Self {
        Self {
            inner: ResamplingQuality::High,
        }
    }

    #[inline]
    pub const fn resample_very_high() -> Self {
        Self {
            inner: ResamplingQuality::VeryHigh,
        }
    }
}

#[pymethods]
impl PyResamplingQuality {
    /// Fast resampling: short windowed sinc (passband 0.80 x Nyquist, 70 dB
    /// stopband) when downsampling, linear interpolation when upsampling.
    #[classattr]
    const fn fast() -> Self {
        Self {
            inner: ResamplingQuality::Fast,
        }
    }

    /// Medium quality resampling: windowed sinc, passband 0.90 x Nyquist,
    /// 100 dB stopband.
    #[classattr]
    const fn medium() -> Self {
        Self {
            inner: ResamplingQuality::Medium,
        }
    }

    /// High quality resampling: windowed sinc, passband 0.913 x Nyquist,
    /// 125 dB stopband.
    #[classattr]
    const fn high() -> Self {
        Self {
            inner: ResamplingQuality::High,
        }
    }

    /// Very high quality resampling: windowed sinc, 185 dB stopband; passband
    /// 0.95 x Nyquist for decimation by a power of two, 0.98 for other ratios
    /// with no prime factor above 7, 0.996 for the rest (such as pitch shifts).
    #[classattr]
    const fn very_high() -> Self {
        Self {
            inner: ResamplingQuality::VeryHigh,
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
    const fn linear() -> Self {
        Self {
            inner: SpectrogramScale::Linear,
        }
    }

    /// Logarithmic (decibel) magnitude scale.
    ///
    /// Compresses dynamic range to improve visualisation of low-energy components
    /// alongside strong spectral peaks.
    #[classattr]
    const fn log() -> Self {
        Self {
            inner: SpectrogramScale::Log,
        }
    }

    /// Mel-frequency scale.
    ///
    /// Applies a perceptually motivated nonlinear mapping of frequency designed
    /// to better approximate human auditory resolution.
    #[classattr]
    const fn mel() -> Self {
        Self {
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
    const fn average() -> Self {
        Self {
            inner: MonoConversionMethod::Average,
        }
    }

    /// Use the left channel only.
    ///
    /// Intended primarily for stereo input.
    #[classattr]
    const fn left() -> Self {
        Self {
            inner: MonoConversionMethod::Left,
        }
    }

    /// Use the right channel only.
    ///
    /// Intended primarily for stereo input.
    #[classattr]
    const fn right() -> Self {
        Self {
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
    #[pyo3(signature = (weights: "list[float]"))]
    const fn weighted(_cls: &Bound<'_, PyType>, weights: Vec<f64>) -> Self {
        Self {
            inner: MonoConversionMethod::Weighted(weights),
        }
    }

    /// Use the centre channel if available, otherwise average left and right.
    #[classattr]
    const fn center() -> Self {
        Self {
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
    const fn duplicate() -> Self {
        Self {
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
    /// ``pan_value`` : float
    ///     Pan position in the range [-1, 1].
    #[classmethod]
    #[pyo3(signature = (pan_value: "float"))]
    const fn pan(_cls: &Bound<'_, PyType>, pan_value: f64) -> Self {
        Self {
            inner: StereoConversionMethod::Pan(pan_value),
        }
    }

    /// Use as the left channel, filling the right channel with silence.
    #[classattr]
    const fn left() -> Self {
        Self {
            inner: StereoConversionMethod::Left,
        }
    }

    /// Use as the right channel, filling the left channel with silence.
    #[classattr]
    const fn right() -> Self {
        Self {
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
/// - ``PitchDetectionMethod.cepstrum`` (not implemented)
/// - ``PitchDetectionMethod.harmonic_product`` (not implemented)
///
/// Notes:
///
///     Only ``yin`` and ``autocorrelation`` are implemented. ``track_pitch``
///     raises ``AudioParameterError`` for ``cepstrum`` and
///     ``harmonic_product``.
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
    pub const fn detect_yin() -> Self {
        Self {
            inner: PitchDetectionMethod::Yin,
        }
    }
    #[inline]
    pub const fn detect_autocorrelation() -> Self {
        Self {
            inner: PitchDetectionMethod::Autocorrelation,
        }
    }
    #[inline]
    pub const fn detect_cepstrum() -> Self {
        Self {
            inner: PitchDetectionMethod::Cepstrum,
        }
    }
    #[inline]
    pub const fn detect_harmonic_product() -> Self {
        Self {
            inner: PitchDetectionMethod::HarmonicProduct,
        }
    }
}

#[pymethods]
impl PyPitchDetectionMethod {
    /// YIN pitch detection algorithm.
    ///
    /// Accurate fundamental frequency estimation for speech and musical
    /// signals, at moderate computational cost.
    #[classattr]
    const fn yin() -> Self {
        Self {
            inner: PitchDetectionMethod::Yin,
        }
    }

    /// Autocorrelation-based pitch detection.
    ///
    /// Simple and fast, but sensitive to noise and octave errors for complex or
    /// weakly periodic signals.
    #[classattr]
    const fn autocorrelation() -> Self {
        Self {
            inner: PitchDetectionMethod::Autocorrelation,
        }
    }

    /// Cepstral pitch detection (not implemented).
    ///
    /// Reserved; ``track_pitch`` raises ``AudioParameterError`` for this method.
    /// Use ``yin`` or ``autocorrelation``.
    #[classattr]
    const fn cepstrum() -> Self {
        Self {
            inner: PitchDetectionMethod::Cepstrum,
        }
    }

    /// Harmonic Product Spectrum (HPS) (not implemented).
    ///
    /// Reserved; ``track_pitch`` raises ``AudioParameterError`` for this method.
    /// Use ``yin`` or ``autocorrelation``.
    #[classattr]
    const fn harmonic_product() -> Self {
        Self {
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
/// - ``IirFilterType.bessel``
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
    const fn butterworth() -> Self {
        Self {
            inner: IirFilterType::Butterworth,
        }
    }

    /// Chebyshev Type I filter.
    ///
    /// Introduces controlled ripple in the passband to achieve a sharper
    /// transition region than Butterworth designs.
    #[classattr]
    const fn chebyshev_type_i() -> Self {
        Self {
            inner: IirFilterType::ChebyshevI,
        }
    }

    /// Chebyshev Type II filter.
    ///
    /// Introduces ripple in the stopband while preserving a monotonic passband
    /// response, allowing sharper transitions than Butterworth designs.
    #[classattr]
    const fn chebyshev_type_ii() -> Self {
        Self {
            inner: IirFilterType::ChebyshevII,
        }
    }

    /// Elliptic (Cauer) filter.
    ///
    /// Introduces ripple in both passband and stopband, yielding the steepest
    /// transition region for a given filter order.
    #[classattr]
    const fn elliptic() -> Self {
        Self {
            inner: IirFilterType::Elliptic,
        }
    }

    /// Bessel (Bessel--Thomson) filter.
    ///
    /// Exhibits a maximally flat group delay (approximately linear phase) in the
    /// passband, preserving wave shape at the cost of a gentle, monotonic
    /// magnitude roll-off.
    #[classattr]
    const fn bessel() -> Self {
        Self {
            inner: IirFilterType::Bessel,
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
    const fn lowpass() -> Self {
        Self {
            inner: FilterResponse::LowPass,
        }
    }

    /// High-pass filter response.
    ///
    /// Attenuates frequencies below the cutoff frequency while preserving
    /// higher-frequency components.
    #[classattr]
    const fn highpass() -> Self {
        Self {
            inner: FilterResponse::HighPass,
        }
    }

    /// Band-pass filter response.
    ///
    /// Preserves frequencies within a specified band while attenuating
    /// frequencies outside that range.
    #[classattr]
    const fn bandpass() -> Self {
        Self {
            inner: FilterResponse::BandPass,
        }
    }

    /// Band-stop (notch) filter response.
    ///
    /// Attenuates frequencies within a specified band while preserving
    /// frequencies outside that range.
    #[classattr]
    const fn bandstop() -> Self {
        Self {
            inner: FilterResponse::BandStop,
        }
    }
}

impl_py_wrapper_core!(PyFilterResponse, FilterResponse);
impl_py_wrapper_fromstr!(PyFilterResponse, FilterResponse);
impl_py_default_static!(PyFilterResponse);
impl_py_repr!(PyFilterResponse);

/// Extract a `FilterResponse` from a `FilterResponse` instance or a string
/// such as ``"lowpass"``, ``"highpass"``, ``"bandpass"`` or ``"bandstop"``.
pub fn filter_response_arg(obj: &Bound<'_, PyAny>) -> PyResult<FilterResponse> {
    use std::str::FromStr;
    if let Ok(r) = obj.extract::<PyFilterResponse>() {
        return Ok(r.inner);
    }
    let Ok(s) = obj.extract::<String>() else {
        return Err(PyTypeError::new_err(
            "response must be a FilterResponse or str",
        ));
    };
    FilterResponse::from_str(&s).map_err(|_| {
        pyo3::exceptions::PyValueError::new_err(format!(
            "Unknown filter response: '{s}' (expected 'lowpass', 'highpass', 'bandpass' or 'bandstop')"
        ))
    })
}

/// Parse a Bessel normalisation name (``"mag"``, ``"phase"`` or ``"delay"``).
fn bessel_norm_arg(norm: Option<&str>) -> PyResult<BesselNorm> {
    use std::str::FromStr;
    norm.map_or_else(
        || Ok(BesselNorm::default()),
        |s| {
            BesselNorm::from_str(s).map_err(|_| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "Unknown Bessel normalisation: '{s}' (expected 'mag', 'phase' or 'delay')"
                ))
            })
        },
    )
}

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
    /// ``filter_type`` : ``IirFilterType``
    ///     IIR filter family (e.g. Butterworth, Chebyshev).
    /// response : ``FilterResponse``
    ///     Desired frequency response shape (low-pass, high-pass, band-pass, etc.).
    /// order : int
    ///     Filter order (number of poles).
    /// ``cutoff_frequency`` : float, optional
    ///     Cutoff frequency in Hz for low-pass and high-pass filters.
    /// ``low_frequency`` : float, optional
    ///     Lower cutoff frequency in Hz for band-pass and band-stop filters.
    /// ``high_frequency`` : float, optional
    ///     Upper cutoff frequency in Hz for band-pass and band-stop filters.
    /// ``passband_ripple`` : float, optional
    ///     Passband ripple in dB (Chebyshev I, elliptic).
    /// ``stopband_attenuation`` : float, optional
    ///     Stopband attenuation in dB (Chebyshev II, elliptic).
    /// ``bessel_norm`` : str, optional
    ///     Bessel normalisation: ``"mag"`` (default, -3 dB at the cutoff),
    ///     ``"phase"`` (phase response at its midpoint at the cutoff) or
    ///     ``"delay"`` (passband group delay set by the cutoff).
    ///
    /// Returns
    /// -------
    /// ``IirFilterDesign``
    ///     A new filter design description.
    ///
    /// Notes
    /// -----
    /// Parameter consistency is not validated automatically. Invalid or
    /// incompatible combinations may fail later during filter construction.
    #[new]
    #[pyo3(signature = (filter_type: "FilterType", response: "FilterResponse | str", order: "int", cutoff_frequency=None, low_frequency=None, high_frequency=None, passband_ripple=None, stopband_attenuation=None, bessel_norm=None))]
    #[allow(
        clippy::too_many_arguments,
        reason = "the Python signature takes these arguments"
    )]
    fn new(
        filter_type: PyIirFilterType,
        response: &Bound<'_, PyAny>,
        order: usize,
        cutoff_frequency: Option<f64>,
        low_frequency: Option<f64>,
        high_frequency: Option<f64>,
        passband_ripple: Option<f64>,
        stopband_attenuation: Option<f64>,
        bessel_norm: Option<&str>,
    ) -> PyResult<Self> {
        let response = filter_response_arg(response)?;
        let order = nzu_or_err(order)?;

        let dummy = cutoff_frequency
            .or(low_frequency)
            .or(high_frequency)
            .unwrap_or(1000.0);
        let mut inner = IirFilterDesign::butterworth_lowpass(order, dummy);
        inner.filter_type = filter_type.into();
        inner.response = response;
        inner.cutoff_frequency = cutoff_frequency;
        inner.low_frequency = low_frequency;
        inner.high_frequency = high_frequency;
        inner.passband_ripple = passband_ripple;
        inner.stopband_attenuation = stopband_attenuation;
        inner.bessel_norm = bessel_norm_arg(bessel_norm)?;
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
    const fn order(&self) -> usize {
        self.inner.order.get()
    }

    /// Cutoff frequency in Hz for low-pass and high-pass filters.
    #[getter]
    const fn cutoff_frequency(&self) -> Option<f64> {
        self.inner.cutoff_frequency
    }

    /// Lower cutoff frequency in Hz for band-pass and band-stop filters.
    #[getter]
    const fn low_frequency(&self) -> Option<f64> {
        self.inner.low_frequency
    }

    /// Upper cutoff frequency in Hz for band-pass and band-stop filters.
    #[getter]
    const fn high_frequency(&self) -> Option<f64> {
        self.inner.high_frequency
    }

    /// Passband ripple in dB, if applicable.
    #[getter]
    const fn passband_ripple(&self) -> Option<f64> {
        self.inner.passband_ripple
    }

    /// Stopband attenuation in dB, if applicable.
    #[getter]
    const fn stopband_attenuation(&self) -> Option<f64> {
        self.inner.stopband_attenuation
    }

    /// Create a Butterworth low-pass filter design.
    ///
    /// Parameters
    /// ----------
    /// order : int
    ///     Filter order.
    /// ``cutoff_frequency`` : float
    ///     Cutoff frequency in Hz.
    ///
    /// Returns
    /// -------
    ///
    /// ``IirFilterDesign``
    ///     A new Butterworth low-pass filter design instance.
    ///
    /// Errors
    /// ------
    ///
    /// If the order argument is less than or equal to zero
    #[classmethod]
    #[pyo3(signature = (order: "int", cutoff_frequency: "float"))]
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
    /// ``cutoff_frequency`` : float
    ///     Cutoff frequency in Hz.
    #[classmethod]
    #[pyo3(signature = (order: "int", cutoff_frequency: "float"))]
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
    /// ``low_frequency`` : float
    ///     Lower cutoff frequency in Hz.
    /// ``high_frequency`` : float
    ///     Upper cutoff frequency in Hz.
    #[classmethod]
    #[pyo3(signature = (order: "int", low_frequency: "float", high_frequency: "float"))]
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

#[pymethods]
impl PyIirFilterDesign {
    /// Bessel normalisation (``"mag"``, ``"phase"`` or ``"delay"``).
    #[getter]
    fn bessel_norm(&self) -> String {
        self.inner.bessel_norm.to_string()
    }

    /// Create a Butterworth band-stop filter design between ``low_frequency``
    /// and ``high_frequency`` Hz.
    #[classmethod]
    #[pyo3(signature = (order: "int", low_frequency: "float", high_frequency: "float"))]
    fn butterworth_bandstop(
        _cls: &Bound<'_, PyType>,
        order: usize,
        low_frequency: f64,
        high_frequency: f64,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: IirFilterDesign::butterworth_bandstop(
                nzu_or_err(order)?,
                low_frequency,
                high_frequency,
            ),
        })
    }

    /// Create a Chebyshev Type I low-pass / high-pass design.
    ///
    /// The passband ripples between 0 and ``-passband_ripple`` dB.
    #[classmethod]
    #[pyo3(signature = (response: "FilterResponse | str", order: "int", cutoff_frequency: "float", passband_ripple: "float"))]
    fn chebyshev_i(
        _cls: &Bound<'_, PyType>,
        response: &Bound<'_, PyAny>,
        order: usize,
        cutoff_frequency: f64,
        passband_ripple: f64,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: IirFilterDesign::chebyshev_i(
                filter_response_arg(response)?,
                nzu_or_err(order)?,
                cutoff_frequency,
                passband_ripple,
            ),
        })
    }

    /// Create a Chebyshev Type I band-pass / band-stop design.
    ///
    /// The passband ripples between 0 and ``-passband_ripple`` dB.
    #[classmethod]
    #[pyo3(signature = (response: "FilterResponse | str", order: "int", low_frequency: "float", high_frequency: "float", passband_ripple: "float"))]
    fn chebyshev_i_band(
        _cls: &Bound<'_, PyType>,
        response: &Bound<'_, PyAny>,
        order: usize,
        low_frequency: f64,
        high_frequency: f64,
        passband_ripple: f64,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: IirFilterDesign::chebyshev_i_band(
                filter_response_arg(response)?,
                nzu_or_err(order)?,
                low_frequency,
                high_frequency,
                passband_ripple,
            ),
        })
    }

    /// Create a Chebyshev Type II low-pass / high-pass design.
    ///
    /// The stopband is at least ``stopband_attenuation`` dB down;
    /// ``cutoff_frequency`` is the stopband edge.
    #[classmethod]
    #[pyo3(signature = (response: "FilterResponse | str", order: "int", cutoff_frequency: "float", stopband_attenuation: "float"))]
    fn chebyshev_ii(
        _cls: &Bound<'_, PyType>,
        response: &Bound<'_, PyAny>,
        order: usize,
        cutoff_frequency: f64,
        stopband_attenuation: f64,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: IirFilterDesign::chebyshev_ii(
                filter_response_arg(response)?,
                nzu_or_err(order)?,
                cutoff_frequency,
                stopband_attenuation,
            ),
        })
    }

    /// Create a Chebyshev Type II band-pass / band-stop design.
    ///
    /// The stopband is at least ``stopband_attenuation`` dB down; the band
    /// edges are the stopband edges.
    #[classmethod]
    #[pyo3(signature = (response: "FilterResponse | str", order: "int", low_frequency: "float", high_frequency: "float", stopband_attenuation: "float"))]
    fn chebyshev_ii_band(
        _cls: &Bound<'_, PyType>,
        response: &Bound<'_, PyAny>,
        order: usize,
        low_frequency: f64,
        high_frequency: f64,
        stopband_attenuation: f64,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: IirFilterDesign::chebyshev_ii_band(
                filter_response_arg(response)?,
                nzu_or_err(order)?,
                low_frequency,
                high_frequency,
                stopband_attenuation,
            ),
        })
    }

    /// Create an elliptic (Cauer) low-pass / high-pass design.
    ///
    /// ``passband_ripple`` dB of passband ripple and at least
    /// ``stopband_attenuation`` dB of stopband attenuation.
    #[classmethod]
    #[pyo3(signature = (response: "FilterResponse | str", order: "int", cutoff_frequency: "float", passband_ripple: "float", stopband_attenuation: "float"))]
    fn elliptic(
        _cls: &Bound<'_, PyType>,
        response: &Bound<'_, PyAny>,
        order: usize,
        cutoff_frequency: f64,
        passband_ripple: f64,
        stopband_attenuation: f64,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: IirFilterDesign::elliptic(
                filter_response_arg(response)?,
                nzu_or_err(order)?,
                cutoff_frequency,
                passband_ripple,
                stopband_attenuation,
            ),
        })
    }

    /// Create an elliptic (Cauer) band-pass / band-stop design.
    ///
    /// ``passband_ripple`` dB of passband ripple and at least
    /// ``stopband_attenuation`` dB of stopband attenuation.
    #[classmethod]
    #[pyo3(signature = (response: "FilterResponse | str", order: "int", low_frequency: "float", high_frequency: "float", passband_ripple: "float", stopband_attenuation: "float"))]
    fn elliptic_band(
        _cls: &Bound<'_, PyType>,
        response: &Bound<'_, PyAny>,
        order: usize,
        low_frequency: f64,
        high_frequency: f64,
        passband_ripple: f64,
        stopband_attenuation: f64,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: IirFilterDesign::elliptic_band(
                filter_response_arg(response)?,
                nzu_or_err(order)?,
                low_frequency,
                high_frequency,
                passband_ripple,
                stopband_attenuation,
            ),
        })
    }

    /// Create a Bessel low-pass / high-pass design.
    ///
    /// ``norm`` (``"mag"``, ``"phase"`` or ``"delay"``) defaults to ``"mag"``:
    /// -3 dB at the cutoff.
    #[classmethod]
    #[pyo3(signature = (response: "FilterResponse | str", order: "int", cutoff_frequency: "float", norm: "str" = "mag"))]
    fn bessel(
        _cls: &Bound<'_, PyType>,
        response: &Bound<'_, PyAny>,
        order: usize,
        cutoff_frequency: f64,
        norm: &str,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: IirFilterDesign::bessel(
                filter_response_arg(response)?,
                nzu_or_err(order)?,
                cutoff_frequency,
            )
            .with_bessel_norm(bessel_norm_arg(Some(norm))?),
        })
    }

    /// Create a Bessel band-pass / band-stop design; ``norm`` as for ``bessel``.
    #[classmethod]
    #[pyo3(signature = (response: "FilterResponse | str", order: "int", low_frequency: "float", high_frequency: "float", norm: "str" = "mag"))]
    fn bessel_band(
        _cls: &Bound<'_, PyType>,
        response: &Bound<'_, PyAny>,
        order: usize,
        low_frequency: f64,
        high_frequency: f64,
        norm: &str,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: IirFilterDesign::bessel_band(
                filter_response_arg(response)?,
                nzu_or_err(order)?,
                low_frequency,
                high_frequency,
            )
            .with_bessel_norm(bessel_norm_arg(Some(norm))?),
        })
    }

    /// Design the filter and return its second-order sections.
    ///
    /// Returns a ``(n_sections, 6)`` float64 array with rows
    /// ``[b0, b1, b2, a0, a1, a2]`` and ``a0 == 1``, usable directly with
    /// ``scipy.signal.sosfilt``.
    ///
    /// Args:
    ///
    ///     sample_rate (float): Sample rate in hertz.
    ///
    /// Raises:
    ///
    ///     ValueError: If the design is invalid for ``sample_rate``.
    #[pyo3(signature = (sample_rate: "float") -> "NDArray[np.float64]")]
    fn sos<'py>(&self, py: Python<'py>, sample_rate: f64) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let rows = self
            .inner
            .sos_coefficients(sample_rate)
            .map_err(audio_err_to_py)?;
        let flat: Vec<f64> = rows.iter().flat_map(|r| r.iter().copied()).collect();
        let arr = ndarray::Array2::from_shape_vec((rows.len(), 6), flat)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(arr.into_pyarray(py))
    }

    /// Evaluate the frequency response of the design.
    ///
    /// Args:
    ///
    ///     frequencies (np.typing.NDArray[np.float64]): Frequencies in hertz.
    ///     sample_rate (float): Sample rate in hertz.
    ///
    /// Returns:
    ///
    ///     tuple[np.typing.NDArray[np.float64], np.typing.NDArray[np.float64]]:
    ///     Linear magnitude and phase (radians, wrapped to (-pi, pi]) of the
    ///     cascaded sections at ``frequencies``.
    ///
    /// Raises:
    ///
    ///     ValueError: If the design is invalid for ``sample_rate``.
    #[pyo3(signature = (frequencies: "Sequence[float] | NDArray[np.floating]", sample_rate: "float"))]
    #[allow(clippy::needless_pass_by_value)] // PyO3 extracts the sequence argument by value
    fn frequency_response<'py>(
        &self,
        py: Python<'py>,
        frequencies: Vec<f64>,
        sample_rate: f64,
    ) -> PyResult<MagnitudePhase<'py>> {
        let (mag, phase) = self
            .inner
            .frequency_response(&frequencies, sample_rate)
            .map_err(audio_err_to_py)?;
        Ok((mag.into_pyarray(py), phase.into_pyarray(py)))
    }
}

impl_py_wrapper_core!(PyIirFilterDesign, IirFilterDesign);
impl_py_repr!(PyIirFilterDesign);

/// Magnitude and phase arrays of a frequency response.
type MagnitudePhase<'py> = (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>);

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
    const fn peak() -> Self {
        Self {
            inner: EqBandType::Peak,
        }
    }

    /// Low-shelf filter.
    ///
    /// Applies a broadband boost or cut to frequencies below the corner
    /// frequency.
    #[classattr]
    const fn low_shelf() -> Self {
        Self {
            inner: EqBandType::LowShelf,
        }
    }

    /// High-shelf filter.
    ///
    /// Applies a broadband boost or cut to frequencies above the corner
    /// frequency.
    #[classattr]
    const fn high_shelf() -> Self {
        Self {
            inner: EqBandType::HighShelf,
        }
    }

    /// Low-pass filter.
    ///
    /// Attenuates frequencies above the cutoff frequency.
    #[classattr]
    const fn low_pass() -> Self {
        Self {
            inner: EqBandType::LowPass,
        }
    }

    /// High-pass filter.
    ///
    /// Attenuates frequencies below the cutoff frequency.
    #[classattr]
    const fn high_pass() -> Self {
        Self {
            inner: EqBandType::HighPass,
        }
    }

    /// Band-pass filter.
    ///
    /// Preserves frequencies within a specified band while attenuating
    /// frequencies outside that range.
    #[classattr]
    const fn band_pass() -> Self {
        Self {
            inner: EqBandType::BandPass,
        }
    }

    /// Band-stop (notch) filter.
    ///
    /// Attenuates frequencies within a specified band while preserving
    /// frequencies outside that range.
    #[classattr]
    const fn band_stop() -> Self {
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
    /// ``band_type`` : ``EqBandType``
    ///     Type of EQ band (peak, shelf, pass, etc.).
    /// frequency : float
    ///     Centre frequency in Hz (for peak / notch) or corner frequency (for
    ///     shelving and pass filters).
    /// ``gain_db`` : float
    ///     Gain in decibels. Positive values boost, negative values attenuate.
    ///     For purely filtering bands (e.g. low-pass), this is typically zero.
    /// ``q_factor`` : float
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
    #[pyo3(signature = (band_type: "EqBandType", frequency: "float", gain_db: "float", q_factor: "float"))]
    fn new(band_type: PyEqBandType, frequency: f64, gain_db: f64, q_factor: f64) -> Self {
        let band_type: EqBandType = band_type.into();
        let mut inner = EqBand::peak(frequency, gain_db, q_factor);
        inner.band_type = band_type;
        Self { inner }
    }

    /// Create a peaking (bell) EQ band.
    ///
    /// Parameters
    /// ----------
    /// frequency : float
    ///     Centre frequency in Hz.
    /// ``gain_db`` : float
    ///     Gain in decibels. Positive values boost, negative values attenuate.
    /// ``q_factor`` : float
    ///     Quality factor controlling bandwidth.
    #[classmethod]
    #[pyo3(signature = (frequency: "float", gain_db: "float", q_factor: "float"))]
    const fn peak(_cls: &Bound<'_, PyType>, frequency: f64, gain_db: f64, q_factor: f64) -> Self {
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
    /// ``gain_db`` : float
    ///     Gain in decibels.
    /// ``q_factor`` : float
    ///     Shelf slope control.
    #[classmethod]
    #[pyo3(signature = (frequency: "float", gain_db: "float", q_factor: "float"))]
    const fn low_shelf(
        _cls: &Bound<'_, PyType>,
        frequency: f64,
        gain_db: f64,
        q_factor: f64,
    ) -> Self {
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
    /// ``gain_db`` : float
    ///     Gain in decibels.
    /// ``q_factor`` : float
    ///     Shelf slope control.
    #[classmethod]
    #[pyo3(signature = (frequency: "float", gain_db: "float", q_factor: "float"))]
    const fn high_shelf(
        _cls: &Bound<'_, PyType>,
        frequency: f64,
        gain_db: f64,
        q_factor: f64,
    ) -> Self {
        Self {
            inner: EqBand::high_shelf(frequency, gain_db, q_factor),
        }
    }

    /// Centre or cutoff frequency in Hz.
    #[getter]
    const fn frequency(&self) -> f64 {
        self.inner.frequency
    }

    /// Gain in decibels.
    #[getter]
    const fn gain_db(&self) -> f64 {
        self.inner.gain_db
    }

    /// Quality factor controlling bandwidth or slope.
    #[getter]
    const fn q_factor(&self) -> f64 {
        self.inner.q_factor
    }

    /// Whether this band is enabled.
    #[getter]
    const fn enabled(&self) -> bool {
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
    #[pyo3(signature = ())]
    const fn new() -> Self {
        Self {
            inner: ParametricEq::new(),
        }
    }

    /// Add an EQ band to the equaliser.
    ///
    /// Parameters
    /// ----------
    /// band : ``EqBand``
    ///     Band configuration to append to the equaliser.
    #[pyo3(signature = (band))]
    fn add_band(&mut self, band: &PyEqBand) {
        self.inner.add_band(band.inner);
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
    /// ``EqBand`` or None
    ///     The removed band if the index was valid, otherwise ``None``.
    #[pyo3(signature = (index))]
    fn remove_band(&mut self, index: usize) -> Option<PyEqBand> {
        self.inner
            .remove_band(index)
            .map(|inner| PyEqBand { inner })
    }

    /// Overall output gain in decibels.
    #[getter]
    const fn output_gain_db(&self) -> f64 {
        self.inner.output_gain_db
    }

    /// Whether the equaliser is currently bypassed.
    #[getter]
    const fn bypassed(&self) -> bool {
        self.inner.bypassed
    }

    /// Number of EQ bands.
    ///
    /// Enables use of ``len(eq)`` in Python.
    const fn __len__(&self) -> usize {
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
    const fn hard() -> Self {
        Self {
            inner: KneeType::Hard,
        }
    }

    /// Soft knee.
    ///
    /// Applies a gradual transition around the threshold, producing smoother
    /// and more perceptually natural behaviour.
    #[classattr]
    const fn soft() -> Self {
        Self {
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
    const fn rms() -> Self {
        Self {
            inner: DynamicRangeMethod::Rms,
        }
    }

    /// Peak-based level detection.
    ///
    /// Responds to instantaneous signal peaks, providing tight peak control with
    /// increased sensitivity to transients.
    #[classattr]
    const fn peak() -> Self {
        Self {
            inner: DynamicRangeMethod::Peak,
        }
    }

    /// Hybrid level detection.
    ///
    /// Combines RMS and peak estimation to balance smoothness and transient
    /// control.
    #[classattr]
    const fn hybrid() -> Self {
        Self {
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
/// When ``enabled``, the compressor and limiter build their detector input from
/// the main signal ``x`` and, in the ``*_sidechain`` methods, the side-chain
/// signal ``s`` as ``(1 - external_mix) * x + external_mix * s``, then filter it
/// with a 2nd-order Butterworth high-pass at ``high_pass_freq``, a 2nd-order
/// Butterworth low-pass at ``low_pass_freq`` and a first-order high shelf of
/// ``pre_emphasis_db``. Only the detector is filtered, never the signal path.
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
    /// ``high_pass_freq`` : float or None, optional
    ///     Cutoff in Hz of a 2nd-order Butterworth high-pass on the detector
    ///     input, in ``(0, nyquist)``. Keeps bass from triggering gain reduction
    ///     ("pumping").
    /// ``low_pass_freq`` : float or None, optional
    ///     Cutoff in Hz of a 2nd-order Butterworth low-pass on the detector
    ///     input, in ``(0, nyquist)`` and above ``high_pass_freq``.
    /// ``pre_emphasis_db`` : float, optional
    ///     High-frequency gain of a first-order high shelf on the detector input:
    ///     unity at DC, half the gain (in dB) at 2 kHz (or a quarter of the
    ///     sample rate when lower), the full gain at Nyquist. Positive values make
    ///     high frequencies trigger gain reduction more easily. Within
    ///     ``[-40, 40]``; ``0.0`` disables it.
    /// ``external_mix`` : float, optional
    ///     Mix of the main signal (``0.0``) and the external side-chain signal
    ///     (``1.0``, the default) in the detector input of the ``*_sidechain``
    ///     methods, in ``[0.0, 1.0]``. The other methods have no external
    ///     signal and do not read it.
    ///
    /// Returns
    /// -------
    /// ``SideChainConfig``
    ///     A new side-chain configuration.
    ///
    /// Notes
    /// -----
    /// Parameter validity is not automatically checked at construction time.
    /// Invalid configurations may fail later during processing.
    #[new]
    #[pyo3(signature = (enabled: "bool", high_pass_freq: "Optional[float]"=None, low_pass_freq: "Optional[float]"=None, pre_emphasis_db=0.0, external_mix=1.0))]
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
        Self { inner }
    }

    /// Whether side-chain processing is enabled.
    #[getter]
    const fn enabled(&self) -> bool {
        self.inner.enabled
    }

    /// High-pass filter cutoff frequency in Hz for the side-chain signal, if set.
    #[getter]
    const fn high_pass_freq(&self) -> Option<f64> {
        self.inner.high_pass_freq
    }

    /// Low-pass filter cutoff frequency in Hz for the side-chain signal, if set.
    #[getter]
    const fn low_pass_freq(&self) -> Option<f64> {
        self.inner.low_pass_freq
    }

    /// Pre-emphasis applied to the side-chain signal in decibels.
    #[getter]
    const fn pre_emphasis_db(&self) -> f64 {
        self.inner.pre_emphasis_db
    }

    /// Mix ratio between internal and external side-chain signal.
    ///
    /// Values are expected in the range ``[0.0, 1.0]``.
    #[getter]
    const fn external_mix(&self) -> f64 {
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
/// Several common presets are provided as class attributes, accessed
/// **without** parentheses: ``CompressorConfig.vocal``, ``CompressorConfig.drum``,
/// ``CompressorConfig.bus``. (By contrast ``CompressorConfig.default()`` and the
/// ``LimiterConfig`` presets such as ``LimiterConfig.mastering()`` are methods.)
/// Parameters can be read back through read-only properties such as
/// ``threshold_db`` and ``ratio``.
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
    /// ``threshold_db`` : float
    ///     Threshold level in decibels. Signal levels above this value will be
    ///     compressed. Typically negative (e.g. -40 to 0 dB).
    /// ratio : float
    ///     Compression ratio. ``1.0`` means no compression; larger values apply
    ///     stronger compression.
    /// ``attack_ms`` : float
    ///     Attack time in milliseconds. Controls how quickly compression engages
    ///     once the signal exceeds the threshold.
    /// ``release_ms`` : float
    ///     Release time in milliseconds. Controls how quickly compression
    ///     disengages once the signal falls below the threshold.
    /// ``makeup_gain_db`` : float
    ///     Makeup gain in decibels applied after compression.
    /// ``knee_type`` : ``KneeType``
    ///     Knee characteristic controlling transition smoothness around the
    ///     threshold.
    /// ``knee_width_db`` : float
    ///     Width of the knee region in decibels for soft-knee behaviour.
    /// ``detection_method`` : ``DynamicRangeMethod``
    ///     Signal level detection method driving gain reduction.
    /// ``side_chain`` : ``SideChainConfig``
    ///     Side-chain configuration.
    /// ``lookahead_ms`` : float
    ///     Lookahead time in milliseconds. Allows the compressor to anticipate
    ///     upcoming peaks.
    ///
    /// Returns
    /// -------
    /// ``CompressorConfig``
    ///     A new compressor configuration.
    #[new]
    #[pyo3(signature = (*, threshold_db: "float", ratio: "float", attack_ms: "float", release_ms: "float", makeup_gain_db: "float", knee_type: "KneeType", knee_width_db: "float", detection_method: "DynamicRangeMethod", side_chain: "SideChainConfig", lookahead_ms: "float"))]
    #[allow(
        clippy::too_many_arguments,
        reason = "the Python signature takes these arguments"
    )] // PyO3 constructor mirrors the Python signature
    fn new(
        threshold_db: f64,
        ratio: f64,
        attack_ms: f64,
        release_ms: f64,
        makeup_gain_db: f64,
        knee_type: PyKneeType,
        knee_width_db: f64,
        detection_method: PyDynamicRangeMethod,
        side_chain: &Bound<'_, PySideChainConfig>,
        lookahead_ms: f64,
    ) -> Self {
        let side_chain = side_chain.borrow().clone();
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

    /// Threshold level in decibels.
    #[getter]
    const fn threshold_db(&self) -> f64 {
        self.inner.threshold_db
    }

    /// Compression ratio (``1.0`` = no compression).
    #[getter]
    const fn ratio(&self) -> f64 {
        self.inner.ratio
    }

    /// Attack time in milliseconds.
    #[getter]
    const fn attack_ms(&self) -> f64 {
        self.inner.attack_ms
    }

    /// Release time in milliseconds.
    #[getter]
    const fn release_ms(&self) -> f64 {
        self.inner.release_ms
    }

    /// Makeup gain in decibels applied after compression.
    #[getter]
    const fn makeup_gain_db(&self) -> f64 {
        self.inner.makeup_gain_db
    }

    /// Knee characteristic.
    #[getter]
    const fn knee_type(&self) -> PyKneeType {
        PyKneeType {
            inner: self.inner.knee_type,
        }
    }

    /// Knee width in decibels.
    #[getter]
    const fn knee_width_db(&self) -> f64 {
        self.inner.knee_width_db
    }

    /// Detection method driving gain reduction.
    #[getter]
    const fn detection_method(&self) -> PyDynamicRangeMethod {
        PyDynamicRangeMethod {
            inner: self.inner.detection_method,
        }
    }

    /// Side-chain configuration.
    #[getter]
    const fn side_chain(&self) -> PySideChainConfig {
        PySideChainConfig {
            inner: self.inner.side_chain,
        }
    }

    /// Lookahead time in milliseconds.
    #[getter]
    const fn lookahead_ms(&self) -> f64 {
        self.inner.lookahead_ms
    }

    /// Vocal compression preset (class attribute: ``CompressorConfig.vocal``).
    ///
    /// Tuned for moderate dynamic control and natural sounding speech or vocals.
    #[classattr]
    const fn vocal() -> Self {
        Self {
            inner: CompressorConfig::vocal(),
        }
    }

    /// Drum compression preset (class attribute: ``CompressorConfig.drum``).
    ///
    /// Tuned for fast transient control and strong dynamic shaping.
    #[classattr]
    const fn drum() -> Self {
        Self {
            inner: CompressorConfig::drum(),
        }
    }

    /// Bus compression preset (class attribute: ``CompressorConfig.bus``).
    ///
    /// Tuned for gentle dynamic glue on groups or mix buses.
    #[classattr]
    const fn bus() -> Self {
        Self {
            inner: CompressorConfig::bus(),
        }
    }

    /// Validate compressor configuration parameters.
    ///
    /// Parameters
    /// ----------
    /// ``sample_rate`` : float
    ///     Audio sample rate in Hz, used to validate time- and frequency-dependent
    ///     constraints.
    ///
    /// Returns
    /// -------
    /// ``CompressorConfig``
    ///    The validated compressor configuration.
    ///
    /// Raises
    /// ------
    /// ``ValueError``
    ///     If any configuration parameter is invalid.
    #[pyo3(signature = (sample_rate: "float"))]
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
/// Several common presets are provided as class methods
/// (``LimiterConfig.mastering()``), unlike ``CompressorConfig``'s presets,
/// which are class attributes.
///
/// ``isp_limiting`` makes the peak detector read true (inter-sample) peaks,
/// interpolated at 4x, instead of sample peaks.
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
    /// ``ceiling_db`` : float
    ///     Maximum allowed output level in decibels. Typically negative
    ///     (e.g. -0.1 to -3.0 dB).
    /// ``attack_ms`` : float
    ///     Attack time in milliseconds. Controls how quickly limiting engages as
    ///     the signal approaches the ceiling.
    /// ``release_ms`` : float
    ///     Release time in milliseconds. Controls how quickly limiting disengages
    ///     once the signal falls below the ceiling.
    /// ``knee_type`` : ``KneeType``
    ///     Knee characteristic controlling transition smoothness around the
    ///     ceiling.
    /// ``knee_width_db`` : float
    ///     Width of the knee region in decibels for soft-knee behaviour.
    /// ``detection_method`` : ``DynamicRangeMethod``
    ///     Signal level detection method driving limiting.
    /// ``side_chain`` : ``SideChainConfig``
    ///     Side-chain configuration.
    /// ``lookahead_ms`` : float
    ///     Lookahead time in milliseconds. Allows the limiter to anticipate
    ///     upcoming peaks.
    /// ``isp_limiting`` : bool
    ///     True-peak detection: the detector signal is interpolated at 4x
    ///     (as in ITU-R BS.1770) and each sample is detected at the largest
    ///     magnitude within one sample of it. A full-scale sine at fs/4 sampled
    ///     45 degrees off its peaks reads 0 dBTP instead of -3 dBFS.
    ///
    /// Returns
    /// -------
    /// ``LimiterConfig``
    ///     A new limiter configuration.
    #[new]
    #[pyo3(signature = (*, ceiling_db: "float", attack_ms: "float", release_ms: "float", knee_type, knee_width_db, detection_method, side_chain, lookahead_ms, isp_limiting))]
    #[allow(
        clippy::too_many_arguments,
        reason = "the Python signature takes these arguments"
    )] // PyO3 constructor mirrors the Python signature
    fn new(
        ceiling_db: f64,
        attack_ms: f64,
        release_ms: f64,
        knee_type: PyKneeType,
        knee_width_db: f64,
        detection_method: PyDynamicRangeMethod,
        side_chain: &Bound<'_, PySideChainConfig>,
        lookahead_ms: f64,
        isp_limiting: bool,
    ) -> Self {
        let side_chain = side_chain.borrow().clone();
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
        Self { inner }
    }

    /// Ceiling level in decibels.
    #[getter]
    const fn ceiling_db(&self) -> f64 {
        self.inner.ceiling_db
    }

    /// Attack time in milliseconds.
    #[getter]
    const fn attack_ms(&self) -> f64 {
        self.inner.attack_ms
    }

    /// Release time in milliseconds.
    #[getter]
    const fn release_ms(&self) -> f64 {
        self.inner.release_ms
    }

    /// Knee characteristic.
    #[getter]
    const fn knee_type(&self) -> PyKneeType {
        PyKneeType {
            inner: self.inner.knee_type,
        }
    }

    /// Knee width in decibels.
    #[getter]
    const fn knee_width_db(&self) -> f64 {
        self.inner.knee_width_db
    }

    /// Detection method used for limiting.
    #[getter]
    const fn detection_method(&self) -> PyDynamicRangeMethod {
        PyDynamicRangeMethod {
            inner: self.inner.detection_method,
        }
    }

    /// Side-chain configuration.
    #[getter]
    const fn side_chain(&self) -> PySideChainConfig {
        PySideChainConfig {
            inner: self.inner.side_chain,
        }
    }

    /// Lookahead time in milliseconds.
    #[getter]
    const fn lookahead_ms(&self) -> f64 {
        self.inner.lookahead_ms
    }

    /// Whether inter-sample peak limiting was requested.
    ///
    /// Whether the peak detector reads true (inter-sample) peaks, interpolated
    /// at 4x, instead of sample peaks.
    #[getter]
    const fn isp_limiting(&self) -> bool {
        self.inner.isp_limiting
    }

    /// Transparent limiter preset.
    ///
    /// Tuned for minimal audible impact while preventing clipping.
    #[classmethod]
    #[pyo3(signature=())]
    const fn transparent(_cls: &Bound<'_, PyType>) -> Self {
        Self {
            inner: LimiterConfig::transparent(),
        }
    }

    /// Mastering limiter preset.
    ///
    /// Tuned for loudness maximisation with controlled transparency.
    #[classmethod]
    #[pyo3(signature=())]
    const fn mastering(_cls: &Bound<'_, PyType>) -> Self {
        Self {
            inner: LimiterConfig::mastering(),
        }
    }

    /// Broadcast limiter preset.
    ///
    /// Tuned for aggressive peak control and regulatory compliance.
    #[classmethod]
    #[pyo3(signature=())]
    const fn broadcast(_cls: &Bound<'_, PyType>) -> Self {
        Self {
            inner: LimiterConfig::broadcast(),
        }
    }

    /// Validate limiter configuration parameters.
    ///
    /// Parameters
    /// ----------
    /// ``sample_rate`` : float
    ///     Audio sample rate in Hz, used to validate time- and frequency-dependent
    ///     constraints.
    ///
    /// Returns
    /// -------
    /// ``LimiterConfig``
    ///   The validated limiter configuration.
    ///
    /// Raises
    /// ------
    /// ``ValueError``
    ///     If any configuration parameter is invalid.
    #[pyo3(signature=(sample_rate: "float"))]
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
    const fn delta() -> Self {
        Self {
            inner: AdaptiveThresholdMethod::Delta,
        }
    }

    /// Percentile-based adaptive threshold.
    ///
    /// Estimates the threshold from rolling distribution statistics of the
    /// onset strength function, yielding increased robustness at the cost of
    /// slower adaptation.
    #[classattr]
    const fn percentile() -> Self {
        Self {
            inner: AdaptiveThresholdMethod::Percentile,
        }
    }

    /// Combined adaptive threshold.
    ///
    /// Combines delta-based and percentile-based thresholds to balance
    /// responsiveness and robustness across a wide range of signals.
    #[classattr]
    const fn combined() -> Self {
        Self {
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
    const fn white() -> Self {
        Self {
            inner: NoiseColor::White,
        }
    }

    /// Pink noise.
    ///
    /// Exhibits decreasing spectral energy with increasing frequency, producing
    /// a perceptually balanced spectrum across octaves.
    #[classattr]
    const fn pink() -> Self {
        Self {
            inner: NoiseColor::Pink,
        }
    }

    /// Brown (red) noise.
    ///
    /// Exhibits strongly attenuated high-frequency content, yielding a smoother
    /// and more correlated temporal structure.
    #[classattr]
    const fn brown() -> Self {
        Self {
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
    /// ``target_snr_db`` : float
    ///     Target signal-to-noise ratio in decibels.
    ///
    /// Returns
    /// -------
    /// ``PerturbationMethod``
    ///     A Gaussian-noise perturbation configuration.
    #[classmethod]
    #[pyo3(signature=(target_snr_db: "float"))]
    const fn white_noise(cls: &Bound<'_, PyType>, target_snr_db: f64) -> Self {
        Self::gaussian(cls, target_snr_db, PyNoiseColor::white())
    }

    /// Pink noise injection with a target signal-to-noise ratio.
    ///
    /// Adds coloured Gaussian noise to achieve the target SNR relative to the
    /// input signal's RMS level.
    ///
    /// Parameters
    /// ----------
    /// ``target_snr_db`` : float
    ///    Target signal-to-noise ratio in decibels.
    ///
    /// Returns
    /// -------
    ///
    /// ``PerturbationMethod``
    ///     A Gaussian-noise perturbation configuration.
    #[classmethod]
    #[pyo3(signature=(target_snr_db: "float"))]
    const fn pink_noise(cls: &Bound<'_, PyType>, target_snr_db: f64) -> Self {
        Self::gaussian(cls, target_snr_db, PyNoiseColor::pink())
    }

    /// Brown noise injection with a target signal-to-noise ratio.
    ///
    /// Adds coloured Gaussian noise to achieve the target SNR relative to the
    /// input signal's RMS level.
    ///
    /// Parameters
    /// ----------
    /// ``target_snr_db`` : float
    ///     Target signal-to-noise ratio in decibels.
    ///
    /// Returns
    /// -------
    ///
    /// ``PerturbationMethod``
    ///    A Gaussian-noise perturbation configuration.
    #[classmethod]
    #[pyo3(signature=(target_snr_db: "float"))]
    const fn brown_noise(cls: &Bound<'_, PyType>, target_snr_db: f64) -> Self {
        Self::gaussian(cls, target_snr_db, PyNoiseColor::brown())
    }

    /// Gaussian noise injection with a target signal-to-noise ratio.
    ///
    /// Adds coloured Gaussian noise to achieve the target SNR relative to the
    /// input signal's RMS level.
    ///
    /// Parameters
    /// ----------
    /// ``target_snr_db`` : float
    ///     Target signal-to-noise ratio in decibels.
    /// ``noise_color`` : ``NoiseColor``
    ///     Noise colour (spectral distribution) to use.
    ///
    /// Returns
    /// -------
    /// ``PerturbationMethod``
    ///     A Gaussian-noise perturbation configuration.
    #[classmethod]
    #[pyo3(signature=(target_snr_db: "float", noise_color: "NoiseColor"))]
    const fn gaussian(
        _cls: &Bound<'_, PyType>,
        target_snr_db: f64,
        noise_color: PyNoiseColor,
    ) -> Self {
        Self {
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
    /// ``min_gain_db`` : float
    ///     Minimum gain in decibels.
    /// ``max_gain_db`` : float
    ///     Maximum gain in decibels.
    ///
    /// Returns
    /// -------
    /// ``PerturbationMethod``
    ///     A random-gain perturbation configuration.
    #[classmethod]
    #[pyo3(signature=(min_gain_db: "float", max_gain_db: "float"))]
    const fn random_gain(_cls: &Bound<'_, PyType>, min_gain_db: f64, max_gain_db: f64) -> Self {
        Self {
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
    /// ``cutoff_hz`` : float
    ///     High-pass cutoff frequency in Hz.
    ///
    /// Returns
    /// -------
    /// ``PerturbationMethod``
    ///     A high-pass filter perturbation configuration.
    #[classmethod]
    #[pyo3(signature=(cutoff_hz: "float"))]
    const fn high_pass_filter(_cls: &Bound<'_, PyType>, cutoff_hz: f64) -> Self {
        Self {
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
    /// ``cutoff_hz`` : float
    ///     High-pass cutoff frequency in Hz.
    /// ``slope_db_per_octave`` : float
    ///     Filter slope in dB per octave.
    ///
    /// Returns
    /// -------
    /// ``PerturbationMethod``
    ///     A high-pass filter perturbation configuration.
    #[classmethod]
    #[pyo3(signature=(cutoff_hz: "float", slope_db_per_octave: "float"))]
    const fn high_pass_filter_with_slope(
        _cls: &Bound<'_, PyType>,
        cutoff_hz: f64,
        slope_db_per_octave: f64,
    ) -> Self {
        Self {
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
    /// ``cutoff_hz`` : float
    ///     Low-pass cutoff frequency in Hz.
    ///
    /// Returns
    /// -------
    /// ``PerturbationMethod``
    ///     A low-pass filter perturbation configuration.
    #[classmethod]
    #[pyo3(signature=(cutoff_hz: "float"))]
    const fn low_pass_filter(_cls: &Bound<'_, PyType>, cutoff_hz: f64) -> Self {
        Self {
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
    /// ``cutoff_hz`` : float
    ///     Low-pass cutoff frequency in Hz.
    /// ``slope_db_per_octave`` : float
    ///     Filter slope in dB per octave.
    ///
    /// Returns
    /// -------
    /// ``PerturbationMethod``
    ///     A low-pass filter perturbation configuration.
    #[classmethod]
    #[pyo3(signature=(cutoff_hz: "float", slope_db_per_octave: "float"))]
    const fn low_pass_filter_with_slope(
        _cls: &Bound<'_, PyType>,
        cutoff_hz: f64,
        slope_db_per_octave: f64,
    ) -> Self {
        Self {
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
    /// ``preserve_formants`` : bool, optional
    ///     Whether to attempt formant preservation.
    ///
    /// Returns
    /// -------
    /// ``PerturbationMethod``
    ///     A pitch-shift perturbation configuration.
    #[classmethod]
    #[pyo3(signature=(semitones: "float", preserve_formants: "bool"=false))]
    const fn pitch_shift(
        _cls: &Bound<'_, PyType>,
        semitones: f64,
        preserve_formants: bool,
    ) -> Self {
        Self {
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
    /// ``sample_rate`` : float
    ///     Sample rate in Hz, used to validate frequency-dependent parameters.
    ///
    /// Returns
    /// -------
    /// ``PerturbationMethod``
    ///     This instance. Returned to allow call chaining.
    ///
    /// Raises
    /// ------
    /// ``ValueError``
    ///     If the perturbation parameters are invalid.
    #[pyo3(signature=(sample_rate: "float"))]
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
    /// method : ``PerturbationMethod``
    ///     The perturbation method to apply.
    /// seed : Optional[int]
    ///     Optional random seed for deterministic behaviour. If ``None``, a
    ///     non-deterministic random generator is used.
    ///
    /// Returns
    /// -------
    /// ``PerturbationConfig``
    ///     A new perturbation configuration.
    #[new]
    #[pyo3(signature=(method: "PerturbationMethod", seed: "int | None" = None))]
    const fn new(method: PyPerturbationMethod, seed: Option<u64>) -> Self {
        let mut inner = PerturbationConfig::new(method.inner);
        inner.seed = seed;
        Self { inner }
    }

    /// Random seed used for deterministic perturbation, if specified.
    ///
    /// Returns
    /// -------
    /// Optional[int]
    ///     The seed value, or ``None`` if no seed is configured.
    #[getter]
    const fn seed(&self) -> Option<u64> {
        self.inner.seed
    }

    /// Validate the perturbation configuration for a given sample rate.
    ///
    /// Parameters
    /// ----------
    /// ``sample_rate`` : float
    ///     Sample rate in Hz, used to validate frequency-dependent parameters.
    ///
    /// Returns
    /// -------
    /// ``PerturbationConfig``
    ///     This instance. Returned to allow call chaining.
    ///
    /// Raises
    /// ------
    /// ``ValueError``
    ///     If the configuration or underlying perturbation method is invalid.
    #[pyo3(signature=(sample_rate: "float"))]
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
        mask_softness: "float"=0.3))]
    fn new(
        stft_params: Option<PyStftParams>,
        median_filter_harmonic: usize,
        median_filter_percussive: usize,
        mask_softness: f64,
    ) -> PyResult<Self> {
        let stft_params = match stft_params {
            Some(p) => p.inner,
            None => StftParams::new(nzu!(2048), nzu!(512), WindowType::Hanning, true)?,
        };
        let hpss_config = HpssConfig::new(
            stft_params,
            median_filter_harmonic,
            median_filter_percussive,
            mask_softness,
        );
        Ok(Self { inner: hpss_config })
    }

    /// Create configuration optimized for musical content.
    ///
    /// Uses larger filters for stronger separation, suitable for complex musical material:
    /// - Larger harmonic kernel for better tonal separation
    /// - Larger percussive kernel for cleaner transient isolation
    /// - Softer masking for more musical results
    #[classattr]
    fn musical() -> Self {
        Self {
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
        Self {
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
        Self {
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
        Self {
            inner: HpssConfig::realtime(),
        }
    }

    /// Set STFT parameters.
    ///
    /// Parameters
    /// ----------
    /// ``n_fft``: int
    ///     FFT size in samples (should be a power of 2)
    /// ``hop_size``: int
    ///     Hop size in samples, at most ``n_fft`` (typically ``n_fft/4``)
    ///
    /// Raises
    /// ------
    /// ``ValueError``
    ///     If either value is 0 or ``hop_size`` is larger than ``n_fft``. The
    ///     configuration is then unchanged.
    #[pyo3(signature=(n_fft: "int",  hop_size: "int"))]
    fn set_stft_params(&mut self, n_fft: usize, hop_size: usize) -> PyResult<()> {
        let n_fft = nzu_or_err(n_fft)?;
        let hop_size = nzu_or_err(hop_size)?;
        self.inner
            .set_stft_params(n_fft, hop_size)
            .map_err(audio_err_to_py)
    }

    /// Set median filter sizes.
    ///
    /// Parameters
    /// ----------
    /// harmonic: int
    ///     Harmonic filter size (odd numbers recommended)
    /// percussive: int
    ///     Percussive filter size (odd numbers recommended)
    #[pyo3(signature=(harmonic_size: "int", percussive_size: "int"))]
    const fn set_filter_sizes(&mut self, harmonic_size: usize, percussive_size: usize) {
        self.inner.set_filter_sizes(harmonic_size, percussive_size);
    }

    /// Set mask softness parameter.
    ///
    /// Parameters
    /// ----------
    /// softness: float
    ///     Softness value (0.0 = hard mask, 1.0 = completely soft)
    #[pyo3(signature=(softness: "float"))]
    const fn set_mask_softness(&mut self, softness: f64) {
        self.inner.set_mask_softness(softness);
    }

    /// Calculate the number of frequency bins for this configuration.
    #[pyo3(signature=())]
    const fn num_freq_bins(&self) -> usize {
        self.inner.num_freq_bins().get()
    }

    /// Calculate the frequency resolution in Hz.
    #[pyo3(signature=(sample_rate: "float"))]
    fn freq_resolution(&self, sample_rate: f64) -> f64 {
        self.inner.freq_resolution(sample_rate)
    }

    /// Calculate the time resolution in seconds.
    #[pyo3(signature=(sample_rate: "float"))]
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
    /// Create a new ``AdaptiveThresholdConfig``.
    ///
    /// Parameters
    /// ----------
    /// method : ``AdaptiveThresholdMethod``
    ///     Adaptive thresholding method to use.
    /// delta : float
    ///     Delta offset for delta-based thresholding.
    /// percentile : float
    ///    Percentile value for percentile-based thresholding.
    /// ``window_size`` : int
    ///    Size of the rolling window in frames.
    /// ``min_threshold`` : float
    ///    Minimum allowable threshold.
    /// ``max_threshold`` : float
    ///   Maximum allowable threshold.
    ///
    /// Returns
    /// -------
    /// AdaptiveThresholdConfig
    ///   A new adaptive threshold configuration.
    #[new]
    #[pyo3(signature = (method: "AdaptiveThresholdMethod", *, delta: "float", percentile: "float", window_size: "int", min_threshold: "float", max_threshold: "float"))]
    const fn new(
        method: PyAdaptiveThresholdMethod,
        delta: f64,
        percentile: f64,
        window_size: usize,
        min_threshold: f64,
        max_threshold: f64,
    ) -> Self {
        Self {
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
    /// ``window_size`` : int
    ///     Size of the rolling window in frames.
    ///
    /// Returns
    /// -------
    /// ``AdaptiveThresholdConfig``
    ///     A new delta-based adaptive threshold configuration.
    #[classmethod]
    #[pyo3(signature = (delta: "float", window_size: "int"))]
    const fn delta(_cls: &Bound<'_, PyType>, delta: f64, window_size: usize) -> Self {
        Self {
            inner: AdaptiveThresholdConfig::delta(delta, window_size),
        }
    }

    /// Create a percentile-based adaptive threshold configuration.
    ///
    /// Parameters
    /// ----------
    /// percentile : float
    ///     Percentile value for thresholding.
    /// ``window_size`` : int
    ///     Size of the rolling window in frames.
    ///
    /// Returns
    /// -------
    /// ``AdaptiveThresholdConfig``
    ///     A new percentile-based adaptive threshold configuration.
    #[classmethod]
    #[pyo3(signature = (percentile: "float", window_size: "int"))]
    const fn percentile(_cls: &Bound<'_, PyType>, percentile: f64, window_size: usize) -> Self {
        Self {
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
    /// ``window_size`` : int
    ///    Size of the rolling window in frames.
    ///
    /// Returns
    /// -------
    /// ``AdaptiveThresholdConfig``
    ///    A new combined adaptive threshold configuration.
    #[classmethod]
    #[pyo3(signature = (delta: "float", percentile: "float", window_size: "int"))]
    const fn combined(
        _cls: &Bound<'_, PyType>,
        delta: f64,
        percentile: f64,
        window_size: usize,
    ) -> Self {
        Self {
            inner: AdaptiveThresholdConfig::combined(delta, percentile, window_size),
        }
    }

    /// Minimum allowable threshold.
    #[getter]
    const fn min_threshold(&self) -> f64 {
        self.inner.min_threshold
    }

    /// Maximum allowable threshold.
    #[getter]
    const fn max_threshold(&self) -> f64 {
        self.inner.max_threshold
    }

    /// Set the minimum allowable threshold.
    #[setter]
    const fn set_min_threshold(&mut self, value: f64) {
        self.inner.min_threshold = value;
    }

    /// Set the maximum allowable threshold.
    #[setter]
    const fn set_max_threshold(&mut self, value: f64) {
        self.inner.max_threshold = value;
    }

    /// Validate the configuration parameters.
    ///
    /// Raises an error if any parameters are invalid.
    #[pyo3(signature = ())]
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
///
/// Presets are class attributes, accessed **without** parentheses:
/// ``PeakPickingConfig.music``, ``PeakPickingConfig.speech``,
/// ``PeakPickingConfig.drums``. ``PeakPickingConfig.default()`` is a method.
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
    /// Create a new ``PeakPickingConfig``.
    ///
    /// Parameters
    /// ----------
    /// ``adaptive_threshold_config`` : ``AdaptiveThresholdConfig``
    ///     Adaptive thresholding configuration.
    /// ``min_peak_separation`` : int
    ///     Minimum separation between detected peaks in frames.
    /// ``pre_emphasis`` : bool
    ///     Whether to apply pre-emphasis filtering.
    /// ``pre_emphasis_coeff`` : float
    ///     Pre-emphasis filter coefficient.
    /// ``median_filter`` : bool
    ///     Whether to apply median filtering.
    /// ``median_filter_length`` : int
    ///     Length of the median filter in frames.
    /// ``normalize_onset_strength`` : bool
    ///     Whether to normalize the onset strength function.
    /// ``normalization_method`` : ``NormalizationMethod``
    ///     Method for normalization.
    #[new]
    #[pyo3(signature = (*, adaptive_threshold_config: "AdaptiveThresholdConfig",
        min_peak_separation: "int",
        pre_emphasis: "bool",
        pre_emphasis_coeff: "float",
        median_filter: "bool",
        median_filter_length: "int",
        normalize_onset_strength: "bool",
        normalization_method: "NormalizationMethod"))]
    #[allow(
        clippy::too_many_arguments,
        reason = "the Python signature takes these arguments"
    )] // PyO3 constructor mirrors the Python signature
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
        Ok(Self {
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

    /// Music preset: a 0.2 s combined threshold window and 0.1 s minimum separation.
    ///
    /// The durations are converted to frames of the onset strength function at
    /// ``sample_rate / hop_size`` frames per second, so pass the sample rate and
    /// hop size of the onset function it will be applied to.
    ///
    /// Args:
    ///
    ///     sample_rate (int): Sample rate of the audio in Hz.
    ///     hop_size (int): Hop between onset-function frames, in samples.
    #[staticmethod]
    #[pyo3(signature = (sample_rate: "int", hop_size: "int"))]
    fn music(sample_rate: std::num::NonZeroU32, hop_size: std::num::NonZeroUsize) -> Self {
        Self {
            inner: PeakPickingConfig::music(sample_rate, hop_size),
        }
    }

    /// Speech preset: a 0.2 s delta threshold window and 80 ms minimum separation.
    ///
    /// The durations are converted to frames of the onset strength function at
    /// ``sample_rate / hop_size`` frames per second, so pass the sample rate and
    /// hop size of the onset function it will be applied to.
    ///
    /// Args:
    ///
    ///     sample_rate (int): Sample rate of the audio in Hz.
    ///     hop_size (int): Hop between onset-function frames, in samples.
    #[staticmethod]
    #[pyo3(signature = (sample_rate: "int", hop_size: "int"))]
    fn speech(sample_rate: std::num::NonZeroU32, hop_size: std::num::NonZeroUsize) -> Self {
        Self {
            inner: PeakPickingConfig::speech(sample_rate, hop_size),
        }
    }

    /// Drums preset: a 0.1 s percentile threshold window and 30 ms minimum separation.
    ///
    /// The durations are converted to frames of the onset strength function at
    /// ``sample_rate / hop_size`` frames per second, so pass the sample rate and
    /// hop size of the onset function it will be applied to.
    ///
    /// Args:
    ///
    ///     sample_rate (int): Sample rate of the audio in Hz.
    ///     hop_size (int): Hop between onset-function frames, in samples.
    #[staticmethod]
    #[pyo3(signature = (sample_rate: "int", hop_size: "int"))]
    fn drums(sample_rate: std::num::NonZeroU32, hop_size: std::num::NonZeroUsize) -> Self {
        Self {
            inner: PeakPickingConfig::drums(sample_rate, hop_size),
        }
    }

    /// Minimum separation between detected peaks in frames.
    #[getter]
    const fn min_peak_separation(&self) -> usize {
        self.inner.min_peak_separation.get()
    }

    /// Minimum separation between detected peaks in frames.
    #[setter]
    fn set_min_peak_separation(&mut self, value: usize) -> PyResult<()> {
        self.inner.min_peak_separation = nzu_or_err(value)?;
        Ok(())
    }

    /// Set minimum peak separation in milliseconds.
    #[pyo3(signature = (value: "float", sample_rate: "float"))]
    fn set_min_peak_separation_ms(&mut self, value: f64, sample_rate: f64) {
        self.inner.set_min_peak_separation_ms(value, sample_rate);
    }

    /// Enable or disable pre-emphasis filtering.
    #[pyo3(signature = (enabled: "bool", coeff: "float"))]
    const fn set_pre_emphasis(&mut self, enabled: bool, coeff: f64) {
        self.inner.pre_emphasis = enabled;
        self.inner.pre_emphasis_coeff = coeff;
    }

    /// Enable or disable median filtering.
    #[pyo3(signature = (enabled: "bool", length: "int"))]
    fn set_median_filter(&mut self, enabled: bool, length: usize) -> PyResult<()> {
        self.inner.median_filter = enabled;
        self.inner.median_filter_length = nzu_or_err(length)?;
        Ok(())
    }

    /// Validate the configuration parameters.
    #[pyo3(signature = ())]
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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
    const fn energy() -> Self {
        Self {
            inner: SpectralFluxMethod::Energy,
        }
    }

    /// Magnitude-based spectral flux.
    ///
    /// Measures positive changes in spectral magnitude and is more sensitive
    /// to subtle spectral variation, making it effective for tonal material.
    #[classattr]
    const fn magnitude() -> Self {
        Self {
            inner: SpectralFluxMethod::Magnitude,
        }
    }

    /// Complex-domain spectral flux.
    ///
    /// Incorporates phase information in addition to magnitude, improving
    /// robustness to noise and spectral smearing at increased computational
    /// cost.
    #[classattr]
    const fn complex() -> Self {
        Self {
            inner: SpectralFluxMethod::Complex,
        }
    }

    /// Rectified complex-domain spectral flux.
    ///
    /// Suppresses negative phase contributions to balance sensitivity and
    /// robustness.
    #[classattr]
    const fn rectified_complex() -> Self {
        Self {
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
    /// Create a new ``SpectralFluxConfig``.
    ///
    /// Parameters
    /// ----------
    /// ``cqt_params`` : ``CqtParams``
    ///     CQT parameters object from spectrograms module.
    /// ``hop_size`` : int
    ///     Hop size in samples (must be > 0).
    /// ``window_size`` : Optional[int]
    ///     Window size in samples (must be > 0 if provided). If ``None``, defaults to CQT's minimum window size.
    /// ``flux_method`` : ``SpectralFluxMethod``
    ///     Spectral flux computation method.
    /// ``peak_picking`` : ``PeakPickingConfig``
    ///     Peak picking configuration.
    /// rectify : bool
    ///     Whether to rectify the onset strength function.
    /// ``log_compression`` : float
    ///     Logarithmic compression factor.
    ///
    /// Returns
    /// ------
    /// SpectralFluxConfig
    ///     A new spectral flux configuration.
    #[new]
    #[pyo3(signature = (*, cqt_params, hop_size, window_size = None, flux_method, peak_picking, rectify, log_compression))]
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
        let window_size_nz = window_size.map(nzu_or_err).transpose()?;

        Ok(Self {
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
    ///
    /// Args:
    ///
    ///     sample_rate (int): Sample rate of the audio the configuration will be
    ///         applied to; the peak picker's durations are converted to frames
    ///         at ``sample_rate / hop_size``.
    #[staticmethod]
    #[pyo3(signature = (sample_rate: "int"))]
    fn percussive(sample_rate: std::num::NonZeroU32) -> Self {
        Self {
            inner: SpectralFluxConfig::percussive(sample_rate),
        }
    }

    /// Musical preset configuration.
    ///
    /// Args:
    ///
    ///     sample_rate (int): Sample rate of the audio the configuration will be
    ///         applied to; the peak picker's durations are converted to frames
    ///         at ``sample_rate / hop_size``.
    #[staticmethod]
    #[pyo3(signature = (sample_rate: "int"))]
    fn musical(sample_rate: std::num::NonZeroU32) -> Self {
        Self {
            inner: SpectralFluxConfig::musical(sample_rate),
        }
    }

    /// Complex preset configuration.
    #[classattr]
    fn complex() -> Self {
        Self {
            inner: SpectralFluxConfig::complex(),
        }
    }

    /// Validate the configuration parameters.
    #[pyo3(signature = ())]
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
    const fn hop_size(&self) -> usize {
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
        self.inner.window_size.map(std::num::NonZero::get)
    }

    /// Set the window size in samples.
    #[setter]
    fn set_window_size(&mut self, value: Option<usize>) -> PyResult<()> {
        self.inner.window_size = value.map(nzu_or_err).transpose()?;
        Ok(())
    }

    /// Get the spectral flux computation method.
    #[getter]
    const fn flux_method(&self) -> PySpectralFluxMethod {
        PySpectralFluxMethod {
            inner: self.inner.flux_method,
        }
    }

    /// Set the spectral flux computation method.
    #[setter]
    const fn set_flux_method(&mut self, value: PySpectralFluxMethod) {
        self.inner.flux_method = value.inner;
    }

    /// Get the peak picking configuration.
    ///
    /// Returns a **copy**: setting fields on the returned object does not change
    /// this config. Modify the copy and assign it back:
    ///
    /// ::
    ///
    ///     pp = config.peak_picking
    ///     pp.min_peak_separation = 256
    ///     config.peak_picking = pp
    #[getter]
    const fn peak_picking(&self) -> PyPeakPickingConfig {
        PyPeakPickingConfig {
            inner: self.inner.peak_picking,
        }
    }

    /// Set the peak picking configuration.
    #[setter]
    const fn set_peak_picking(&mut self, value: PyPeakPickingConfig) {
        self.inner.peak_picking = value.inner;
    }

    /// Get whether to rectify the onset strength function.
    #[getter]
    const fn rectify(&self) -> bool {
        self.inner.rectify
    }

    /// Set whether to rectify the onset strength function.
    #[setter]
    const fn set_rectify(&mut self, value: bool) {
        self.inner.rectify = value;
    }

    /// Get the logarithmic compression factor.
    #[getter]
    const fn log_compression(&self) -> f64 {
        self.inner.log_compression
    }

    /// Set the logarithmic compression factor.
    #[setter]
    const fn set_log_compression(&mut self, value: f64) {
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
    /// Create a new ``ComplexOnsetConfig``.
    ///
    /// Parameters
    /// ----------
    /// ``cqt_config`` : ``CqtConfig``
    ///     CQT configuration object.
    /// ``hop_size`` : int
    ///     Hop size in samples.
    /// ``window_size`` : Optional[int]
    ///     Window size in samples. If ``None``, defaults to CQT's minimum window size.
    /// ``peak_picking`` : ``PeakPickingConfig``
    ///     Peak picking configuration.
    /// ``magnitude_weight`` : float
    ///     Weighting factor for magnitude component.
    /// ``phase_weight`` : float    
    ///     Weighting factor for phase component.
    /// ``magnitude_rectify`` : bool
    ///     Whether to rectify the magnitude component.
    /// ``phase_rectify`` : bool
    ///     Whether to rectify the phase component.
    /// ``log_compression`` : float
    ///     Logarithmic compression factor.
    ///
    /// Returns
    /// -------
    /// ComplexOnsetConfig
    ///     A new complex onset detection configuration.
    #[new]
    #[pyo3(signature = (*, cqt_params: "CqtParams", hop_size: "int", window_size: "Optional[int]", peak_picking: "PeakPickingConfig", magnitude_weight: "float", phase_weight: "float", magnitude_rectify: "bool", phase_rectify: "bool", log_compression: "float"))]
    #[allow(
        clippy::too_many_arguments,
        reason = "the Python signature takes these arguments"
    )] // PyO3 constructor mirrors the Python signature
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
        let window_size_nz = window_size.map(nzu_or_err).transpose()?;

        Ok(Self {
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
    ///
    /// Args:
    ///
    ///     sample_rate (int): Sample rate of the audio the configuration will be
    ///         applied to; the peak picker's durations are converted to frames
    ///         at ``sample_rate / hop_size``.
    #[staticmethod]
    #[pyo3(signature = (sample_rate: "int"))]
    fn percussive(sample_rate: std::num::NonZeroU32) -> Self {
        Self {
            inner: ComplexOnsetConfig::percussive(sample_rate),
        }
    }

    /// Musical preset configuration.
    ///
    /// Args:
    ///
    ///     sample_rate (int): Sample rate of the audio the configuration will be
    ///         applied to; the peak picker's durations are converted to frames
    ///         at ``sample_rate / hop_size``.
    #[staticmethod]
    #[pyo3(signature = (sample_rate: "int"))]
    fn musical(sample_rate: std::num::NonZeroU32) -> Self {
        Self {
            inner: ComplexOnsetConfig::musical(sample_rate),
        }
    }

    /// Speech preset configuration.
    ///
    /// Args:
    ///
    ///     sample_rate (int): Sample rate of the audio the configuration will be
    ///         applied to; the peak picker's durations are converted to frames
    ///         at ``sample_rate / hop_size``.
    #[staticmethod]
    #[pyo3(signature = (sample_rate: "int"))]
    fn speech(sample_rate: std::num::NonZeroU32) -> Self {
        Self {
            inner: ComplexOnsetConfig::speech(sample_rate),
        }
    }

    /// Set the magnitude and phase weighting factors.
    #[pyo3(signature = (magnitude_weight: "float", phase_weight: "float"))]
    const fn set_weights(&mut self, magnitude_weight: f64, phase_weight: f64) {
        self.inner.magnitude_weight = magnitude_weight;
        self.inner.phase_weight = phase_weight;
    }

    /// Validate the configuration parameters.
    #[pyo3(signature = ())]
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
    const fn hop_size(&self) -> usize {
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
        self.inner.window_size.map(std::num::NonZero::get)
    }

    /// Set the window size in samples.
    #[setter]
    fn set_window_size(&mut self, value: Option<usize>) -> PyResult<()> {
        self.inner.window_size = value.map(nzu_or_err).transpose()?;
        Ok(())
    }

    /// Get the peak picking configuration.
    ///
    /// Returns a **copy**: setting fields on the returned object does not change
    /// this config. Modify the copy and assign it back:
    ///
    /// ::
    ///
    ///     pp = config.peak_picking
    ///     pp.min_peak_separation = 256
    ///     config.peak_picking = pp
    #[getter]
    const fn peak_picking(&self) -> PyPeakPickingConfig {
        PyPeakPickingConfig {
            inner: self.inner.peak_picking,
        }
    }

    /// Set the peak picking configuration.
    #[setter]
    const fn set_peak_picking(&mut self, value: PyPeakPickingConfig) {
        self.inner.peak_picking = value.inner;
    }

    /// Get the magnitude weighting factor.
    #[getter]
    const fn magnitude_weight(&self) -> f64 {
        self.inner.magnitude_weight
    }

    /// Set the magnitude weighting factor.
    #[setter]
    const fn set_magnitude_weight(&mut self, value: f64) {
        self.inner.magnitude_weight = value;
    }

    /// Get the phase weighting factor.
    #[getter]
    const fn phase_weight(&self) -> f64 {
        self.inner.phase_weight
    }

    /// Set the phase weighting factor.
    #[setter]
    const fn set_phase_weight(&mut self, value: f64) {
        self.inner.phase_weight = value;
    }

    /// Get whether to rectify the magnitude component.
    #[getter]
    const fn magnitude_rectify(&self) -> bool {
        self.inner.magnitude_rectify
    }

    /// Set whether to rectify the magnitude component.
    #[setter]
    const fn set_magnitude_rectify(&mut self, value: bool) {
        self.inner.magnitude_rectify = value;
    }

    /// Get whether to rectify the phase component.
    #[getter]
    const fn phase_rectify(&self) -> bool {
        self.inner.phase_rectify
    }

    /// Set whether to rectify the phase component.
    #[setter]
    const fn set_phase_rectify(&mut self, value: bool) {
        self.inner.phase_rectify = value;
    }

    /// Get the logarithmic compression factor.
    #[getter]
    const fn log_compression(&self) -> f64 {
        self.inner.log_compression
    }

    /// Set the logarithmic compression factor.
    #[setter]
    const fn set_log_compression(&mut self, value: f64) {
        self.inner.log_compression = value;
    }
}

impl_py_wrapper_core!(PyComplexOnsetConfig, ComplexOnsetConfig);
impl_py_repr!(PyComplexOnsetConfig);

/// Python wrapper for ``OnsetDetectionConfig``.
///
/// Configuration for onset detection using spectral flux method.
///
/// Notes:
///
///     The keyword defaults of the constructor
///     ``OnsetDetectionConfig(cqt_params, hop_size, ...)`` are not those of
///     ``OnsetDetectionConfig.default()``: the constructor uses
///     ``min_onset_interval_secs=0.07``, ``pre_emphasis=0.0``,
///     ``median_filter_length=3``, ``adaptive_threshold_multiplier=3.0``,
///     while ``default()`` uses ``0.1``, ``0.1``, ``5``, ``1.5`` (other fields
///     agree). Start from ``default()`` for the library defaults.
///
///     ``peak_picking`` and ``cqt_params`` return copies; assign a modified
///     copy back (``config.peak_picking = pp``) to change them.
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
    /// Create a new ``OnsetDetectionConfig`` with custom parameters.
    ///
    /// Parameters
    /// ----------
    /// ``cqt_params`` : ``PyCqtParams``
    ///     CQT configuration for spectral analysis
    /// ``hop_size`` : int
    ///     Hop size for frame-based analysis in samples
    /// ``window_size`` : Optional[int]
    ///     Window size for CQT analysis in samples (None = auto-calculate)
    /// threshold : float
    ///     Threshold for onset detection
    /// ``min_onset_interval_secs`` : float
    ///     Minimum time between consecutive onsets in seconds
    /// ``pre_emphasis`` : float
    ///     Pre-emphasis factor for signal processing
    /// ``adaptive_threshold`` : bool
    ///     Whether to use adaptive thresholding
    /// ``median_filter_length`` : int
    ///     Length of median filter for smoothing
    /// ``adaptive_threshold_multiplier`` : float
    ///     Multiplier for adaptive threshold
    /// ``peak_picking`` : ``PyPeakPickingConfig``
    ///     Peak picking configuration
    ///
    /// Notes
    /// -----
    /// The keyword defaults of this constructor are not those of
    /// ``OnsetDetectionConfig.default()``: the constructor uses
    /// ``min_onset_interval_secs=0.07``, ``pre_emphasis=0.0``,
    /// ``median_filter_length=3`` and ``adaptive_threshold_multiplier=3.0``,
    /// while ``default()`` uses ``0.1``, ``0.1``, ``5`` and ``1.5`` (other
    /// fields agree). Use ``default()`` and set fields for the library default.
    #[new]
    #[pyo3(signature = (cqt_params, hop_size, window_size=None, threshold=0.3, min_onset_interval_secs=0.07, pre_emphasis=0.0, adaptive_threshold=true, median_filter_length=3, adaptive_threshold_multiplier=3.0, peak_picking=None))]
    #[allow(
        clippy::too_many_arguments,
        reason = "the Python signature takes these arguments"
    )] // PyO3 constructor mirrors the Python signature
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
        let peak_picking_inner = peak_picking.map_or_else(PeakPickingConfig::default, |p| p.inner);

        Ok(Self {
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

    /// Create a default ``OnsetDetectionConfig``.
    #[classmethod]
    fn default(_cls: &Bound<'_, PyType>) -> Self {
        Self {
            inner: OnsetDetectionConfig::default(),
        }
    }

    /// Create an ``OnsetDetectionConfig`` optimized for general music.
    ///
    /// Args:
    ///
    ///     sample_rate (int): Sample rate of the audio the configuration will be
    ///         applied to; the peak picker's durations are converted to frames
    ///         at ``sample_rate / hop_size``.
    #[classmethod]
    #[pyo3(signature = (sample_rate: "int"))]
    fn musical(_cls: &Bound<'_, PyType>, sample_rate: std::num::NonZeroU32) -> Self {
        Self {
            inner: OnsetDetectionConfig::musical(sample_rate),
        }
    }

    /// Create an ``OnsetDetectionConfig`` optimized for percussive sounds.
    ///
    /// Args:
    ///
    ///     sample_rate (int): Sample rate of the audio the configuration will be
    ///         applied to; the peak picker's durations are converted to frames
    ///         at ``sample_rate / hop_size``.
    #[classmethod]
    #[pyo3(signature = (sample_rate: "int"))]
    fn percussive(_cls: &Bound<'_, PyType>, sample_rate: std::num::NonZeroU32) -> Self {
        Self {
            inner: OnsetDetectionConfig::percussive(sample_rate),
        }
    }

    /// Create an ``OnsetDetectionConfig`` optimized for speech.
    ///
    /// Args:
    ///
    ///     sample_rate (int): Sample rate of the audio the configuration will be
    ///         applied to; the peak picker's durations are converted to frames
    ///         at ``sample_rate / hop_size``.
    #[classmethod]
    #[pyo3(signature = (sample_rate: "int"))]
    fn speech(_cls: &Bound<'_, PyType>, sample_rate: std::num::NonZeroU32) -> Self {
        Self {
            inner: OnsetDetectionConfig::speech(sample_rate),
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
    const fn hop_size(&self) -> usize {
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
        self.inner.window_size.map(std::num::NonZero::get)
    }

    /// Set the window size in samples.
    #[setter]
    fn set_window_size(&mut self, value: Option<usize>) -> PyResult<()> {
        self.inner.window_size = value.map(nzu_or_err).transpose()?;
        Ok(())
    }

    /// Get the onset detection threshold.
    #[getter]
    const fn threshold(&self) -> f64 {
        self.inner.threshold
    }

    /// Set the onset detection threshold.
    #[setter]
    const fn set_threshold(&mut self, value: f64) {
        self.inner.threshold = value;
    }

    /// Get the minimum onset interval in seconds.
    #[getter]
    const fn min_onset_interval_secs(&self) -> f64 {
        self.inner.min_onset_interval_secs
    }

    /// Set the minimum onset interval in seconds.
    #[setter]
    const fn set_min_onset_interval_secs(&mut self, value: f64) {
        self.inner.min_onset_interval_secs = value;
    }

    /// Get the pre-emphasis factor.
    #[getter]
    const fn pre_emphasis(&self) -> f64 {
        self.inner.pre_emphasis
    }

    /// Set the pre-emphasis factor.
    #[setter]
    const fn set_pre_emphasis(&mut self, value: f64) {
        self.inner.pre_emphasis = value;
    }

    /// Get whether adaptive thresholding is enabled.
    #[getter]
    const fn adaptive_threshold(&self) -> bool {
        self.inner.adaptive_threshold
    }

    /// Set whether adaptive thresholding is enabled.
    #[setter]
    const fn set_adaptive_threshold(&mut self, value: bool) {
        self.inner.adaptive_threshold = value;
    }

    /// Get the median filter length.
    #[getter]
    const fn median_filter_length(&self) -> usize {
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
    const fn adaptive_threshold_multiplier(&self) -> f64 {
        self.inner.adaptive_threshold_multiplier
    }

    /// Set the adaptive threshold multiplier.
    #[setter]
    const fn set_adaptive_threshold_multiplier(&mut self, value: f64) {
        self.inner.adaptive_threshold_multiplier = value;
    }

    /// Get the peak picking configuration.
    ///
    /// Returns a **copy**: setting fields on the returned object does not change
    /// this config. Modify the copy and assign it back:
    ///
    /// ::
    ///
    ///     pp = config.peak_picking
    ///     pp.min_peak_separation = 256
    ///     config.peak_picking = pp
    #[getter]
    const fn peak_picking(&self) -> PyPeakPickingConfig {
        PyPeakPickingConfig {
            inner: self.inner.peak_picking,
        }
    }

    /// Set the peak picking configuration.
    #[setter]
    const fn set_peak_picking(&mut self, value: PyPeakPickingConfig) {
        self.inner.peak_picking = value.inner;
    }
}

impl_py_wrapper_core!(PyOnsetDetectionConfig, OnsetDetectionConfig);
impl_py_repr!(PyOnsetDetectionConfig);

/// Python wrapper for ``EnvelopeFollower``.
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
    /// Create a new ``EnvelopeFollower``.
    ///
    /// Parameters
    /// ----------
    /// ``attack_ms`` : float
    ///     Attack time in milliseconds
    /// ``release_ms`` : float
    ///     Release time in milliseconds
    /// ``sample_rate`` : float
    ///     Sample rate in Hz
    /// ``detection_method`` : ``DynamicRangeMethod``
    ///     Detection method (Peak, Rms, or Hybrid)
    #[new]
    #[pyo3(signature = (attack_ms, release_ms, sample_rate, detection_method))]
    fn new(
        attack_ms: f64,
        release_ms: f64,
        sample_rate: f64,
        detection_method: PyDynamicRangeMethod,
    ) -> Self {
        Self {
            inner: EnvelopeFollower::new(
                attack_ms,
                release_ms,
                sample_rate,
                detection_method.inner,
            ),
        }
    }

    /// Create a default ``EnvelopeFollower`` with 10ms attack and 100ms release at 44100 Hz.
    #[classmethod]
    fn default(_cls: &Bound<'_, PyType>) -> Self {
        Self {
            inner: EnvelopeFollower::new(10.0, 100.0, 44100.0, DynamicRangeMethod::Peak),
        }
    }

    /// Process one sample and return the current envelope level (>= 0).
    ///
    /// The follower is stateful: successive calls continue from the previous
    /// envelope value. Call :meth:`reset` to start again from 0.
    ///
    /// Parameters
    /// ----------
    /// input : float
    ///     Input sample value (any numeric scale).
    ///
    /// Returns
    /// -------
    /// float
    ///     Smoothed envelope level.
    #[pyo3(signature = (input))]
    fn process(&mut self, input: f64) -> f64 {
        self.inner.process(input)
    }

    /// Process a 1-D array sample by sample and return the envelope per sample.
    ///
    /// Equivalent to ``np.array([follower.process(v) for v in samples])``; the
    /// state carries over between calls.
    ///
    /// Parameters
    /// ----------
    /// samples : numpy.ndarray
    ///     1-D float64 array of input samples.
    ///
    /// Returns
    /// -------
    /// numpy.ndarray
    ///     Envelope level for each input sample.
    #[pyo3(signature = (samples: "NDArray[np.float64]") -> "NDArray[np.float64]")]
    fn process_array<'py>(
        &mut self,
        py: Python<'py>,
        samples: &Bound<'py, PyArray1<f64>>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let out: Vec<f64> = numpy::PyArrayMethods::try_readonly(samples)?
            .as_array()
            .iter()
            .map(|&v| self.inner.process(v))
            .collect();
        Ok(out.into_pyarray(py))
    }

    /// Reset the envelope to 0 (and clear the RMS window).
    #[pyo3(signature = ())]
    fn reset(&mut self) {
        self.inner.reset();
    }
}

impl_py_wrapper_core!(PyEnvelopeFollower, EnvelopeFollower);
impl_py_repr!(PyEnvelopeFollower);

/// Python wrapper for ``BeatTrackingConfig``.
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
    /// Create a new ``BeatTrackingConfig``.
    ///
    /// Parameters
    /// ----------
    /// ``tempo_bpm`` : float
    ///     Target tempo in beats per minute
    /// tolerance : float, optional
    ///     Beat timing tolerance in seconds. If None, defaults to 10% of the inter-beat interval
    /// ``onset_config`` : ``OnsetDetectionConfig``
    ///     Configuration for onset detection
    #[new]
    #[pyo3(signature = (tempo_bpm, onset_config, tolerance=None))]
    fn new(tempo_bpm: f64, onset_config: PyOnsetDetectionConfig, tolerance: Option<f64>) -> Self {
        Self {
            inner: BeatTrackingConfig::new(tempo_bpm, tolerance, onset_config.inner),
        }
    }

    /// Get the tempo in BPM.
    #[getter]
    const fn tempo_bpm(&self) -> f64 {
        self.inner.tempo_bpm
    }

    /// Set the tempo in BPM.
    #[setter]
    const fn set_tempo_bpm(&mut self, tempo_bpm: f64) {
        self.inner.tempo_bpm = tempo_bpm;
    }

    /// Get the tolerance in seconds.
    #[getter]
    const fn tolerance(&self) -> Option<f64> {
        self.inner.tolerance
    }

    /// Set the tolerance in seconds.
    #[setter]
    const fn set_tolerance(&mut self, tolerance: Option<f64>) {
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

/// Python wrapper for ``BeatTrackingData``.
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
    /// Create a new ``BeatTrackingData``.
    ///
    /// Parameters
    /// ----------
    /// ``tempo_bpm`` : float
    ///     Estimated tempo in beats per minute
    /// ``beat_times`` : list[float]
    ///     Beat timestamps in seconds
    /// config : ``BeatTrackingConfig``
    ///     Configuration used for beat detection
    #[new]
    #[pyo3(signature = (tempo_bpm, beat_times, config))]
    fn new(tempo_bpm: f64, beat_times: Vec<f64>, config: PyBeatTrackingConfig) -> Self {
        Self {
            inner: BeatTrackingData::new(tempo_bpm, beat_times, config.inner),
        }
    }

    /// Get the estimated tempo in BPM.
    #[getter]
    const fn tempo_bpm(&self) -> f64 {
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

/// Strategy for reducing a multi-channel signal to a single channel.
///
/// `ChannelReduction` controls how operations that fundamentally require a single
/// channel (such as spectral centroid or roll-off) behave when presented with a
/// multi-channel signal. It lets the caller choose between failing loudly,
/// selecting a single channel, or averaging across channels.
///
/// Instances of `ChannelReduction` are immutable and should be treated as
/// enum-like values. Zero-parameter strategies are accessed via class attributes,
/// while the channel-selection strategy is constructed via a static method.
///
/// Available strategies:
///
/// - ``ChannelReduction.error`` (default):
///   Return an error when the signal has more than one channel.
///
/// - ``ChannelReduction.first``:
///   Use the first channel (index 0) and ignore the rest.
///
/// - ``ChannelReduction.average``:
///   Average the corresponding samples across all channels.
///
/// - ``ChannelReduction.channel(index)``:
///   Use the channel at the given (bounds-checked) index.
#[pyclass(
    name = "ChannelReduction",
    from_py_object,
    module = "audio_samples.types"
)]
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
pub struct PyChannelReduction {
    pub(crate) inner: ChannelReduction,
}

#[pymethods]
impl PyChannelReduction {
    /// Return an error when the signal has more than one channel.
    ///
    /// This is the default strategy and preserves the strictest behaviour by
    /// refusing to silently collapse channels.
    #[classattr]
    const fn error() -> Self {
        Self {
            inner: ChannelReduction::Error,
        }
    }

    /// Use the first channel (index 0) and ignore the rest.
    #[classattr]
    const fn first() -> Self {
        Self {
            inner: ChannelReduction::First,
        }
    }

    /// Average the corresponding samples across all channels.
    #[classattr]
    const fn average() -> Self {
        Self {
            inner: ChannelReduction::Average,
        }
    }

    /// Use the channel at the given index.
    ///
    /// The index is bounds-checked when the reduction is applied.
    ///
    /// Parameters
    /// ----------
    /// index : int
    ///     Zero-based channel index to select.
    #[staticmethod]
    #[pyo3(signature = (index: "int"))]
    const fn channel(index: usize) -> Self {
        Self {
            inner: ChannelReduction::Channel(index),
        }
    }
}

impl_py_wrapper_core!(PyChannelReduction, ChannelReduction);
impl_py_default_static!(PyChannelReduction);
impl_py_repr!(PyChannelReduction);

/// Noise gate configuration parameters.
///
/// `GateConfig` defines how a downward noise gate attenuates a signal that falls
/// below a threshold, including the attenuation ratio and the attack and release
/// envelope time constants.
///
/// Instances of `GateConfig` are immutable value objects in the Python API.
/// Parameters are provided at construction time and exposed via read-only
/// properties. Validation can be performed explicitly using ``validate()``.
///
/// A general-purpose noise-gating preset is provided via ``GateConfig.noise_gate``.
#[pyclass(name = "GateConfig", from_py_object, module = "audio_samples.types")]
#[derive(Default, Debug, Clone, Copy, PartialEq)]
pub struct PyGateConfig {
    pub(crate) inner: GateConfig,
}

#[pymethods]
impl PyGateConfig {
    /// Create a new gate configuration.
    ///
    /// Parameters
    /// ----------
    /// ``threshold_db`` : float
    ///     Gate threshold in dBFS. Signals below this level are attenuated.
    ///     Typical range: [-80.0, 0.0].
    /// ratio : float
    ///     Attenuation ratio applied below the threshold. Higher values produce
    ///     more aggressive gating; values near 1.0 approach unity gain. Must be
    ///     greater than 0.0.
    /// ``attack_ms`` : float
    ///     Attack time in milliseconds. Controls how quickly the gate opens once
    ///     the signal rises above the threshold. Valid range: [0.01, 1000.0] ms.
    /// ``release_ms`` : float
    ///     Release time in milliseconds. Controls how quickly the gate closes once
    ///     the signal falls below the threshold. Valid range: [1.0, 10000.0] ms.
    ///
    /// Returns
    /// -------
    /// ``GateConfig``
    ///     A new gate configuration.
    #[new]
    #[pyo3(signature = (*, threshold_db: "float", ratio: "float", attack_ms: "float", release_ms: "float"))]
    const fn new(threshold_db: f64, ratio: f64, attack_ms: f64, release_ms: f64) -> Self {
        Self {
            inner: GateConfig::with_params(threshold_db, ratio, attack_ms, release_ms),
        }
    }

    /// Gate threshold in dBFS.
    #[getter]
    const fn threshold_db(&self) -> f64 {
        self.inner.threshold_db
    }

    /// Attenuation ratio applied below the threshold.
    #[getter]
    const fn ratio(&self) -> f64 {
        self.inner.ratio
    }

    /// Attack time in milliseconds.
    #[getter]
    const fn attack_ms(&self) -> f64 {
        self.inner.attack_ms
    }

    /// Release time in milliseconds.
    #[getter]
    const fn release_ms(&self) -> f64 {
        self.inner.release_ms
    }

    /// General-purpose noise-gating preset.
    ///
    /// Moderate threshold and a high ratio for firmly attenuating background
    /// noise and room tone between phrases.
    #[classattr]
    const fn noise_gate() -> Self {
        Self {
            inner: GateConfig::noise_gate(),
        }
    }

    /// Validate gate configuration parameters.
    ///
    /// Returns
    /// -------
    /// ``GateConfig``
    ///     The validated gate configuration.
    ///
    /// Raises
    /// ------
    /// ``ValueError``
    ///     If any configuration parameter is invalid.
    #[pyo3(signature = ())]
    fn validate(&mut self) -> PyResult<Self> {
        self.inner.validate().map_err(audio_err_to_py)?;
        Ok(Self { inner: self.inner })
    }
}

impl_py_wrapper_core!(PyGateConfig, GateConfig);
impl_py_default_static!(PyGateConfig);
impl_py_repr!(PyGateConfig);

/// Downward expander configuration parameters.
///
/// `ExpanderConfig` defines how a downward expander attenuates a signal that
/// falls below a threshold, increasing the dynamic range of low-level material.
/// RMS detection is always used for expansion.
///
/// Instances of `ExpanderConfig` are immutable value objects in the Python API.
/// Parameters are provided at construction time and exposed via read-only
/// properties. Validation can be performed explicitly using ``validate()``.
///
/// A gentle expansion preset is provided via ``ExpanderConfig.gentle``.
#[pyclass(
    name = "ExpanderConfig",
    from_py_object,
    module = "audio_samples.types"
)]
#[derive(Default, Debug, Clone, Copy, PartialEq)]
pub struct PyExpanderConfig {
    pub(crate) inner: ExpanderConfig,
}

#[pymethods]
impl PyExpanderConfig {
    /// Create a new expander configuration.
    ///
    /// Parameters
    /// ----------
    /// ``threshold_db`` : float
    ///     Expansion threshold in dBFS. Signals below this level are attenuated.
    ///     Typical range: [-80.0, 0.0].
    /// ratio : float
    ///     Expansion ratio applied below the threshold. Values greater than 1.0
    ///     produce increasing attenuation the further the signal falls below the
    ///     threshold. Must be greater than 0.0.
    /// ``attack_ms`` : float
    ///     Attack time in milliseconds. Valid range: [0.01, 1000.0] ms.
    /// ``release_ms`` : float
    ///     Release time in milliseconds. Valid range: [1.0, 10000.0] ms.
    ///
    /// Returns
    /// -------
    /// ``ExpanderConfig``
    ///     A new expander configuration.
    #[new]
    #[pyo3(signature = (*, threshold_db: "float", ratio: "float", attack_ms: "float", release_ms: "float"))]
    const fn new(threshold_db: f64, ratio: f64, attack_ms: f64, release_ms: f64) -> Self {
        Self {
            inner: ExpanderConfig::with_params(threshold_db, ratio, attack_ms, release_ms),
        }
    }

    /// Expansion threshold in dBFS.
    #[getter]
    const fn threshold_db(&self) -> f64 {
        self.inner.threshold_db
    }

    /// Expansion ratio applied below the threshold.
    #[getter]
    const fn ratio(&self) -> f64 {
        self.inner.ratio
    }

    /// Attack time in milliseconds.
    #[getter]
    const fn attack_ms(&self) -> f64 {
        self.inner.attack_ms
    }

    /// Release time in milliseconds.
    #[getter]
    const fn release_ms(&self) -> f64 {
        self.inner.release_ms
    }

    /// Gentle downward expansion preset.
    ///
    /// A low ratio and relaxed envelope times for subtle dynamic-range
    /// enhancement without obvious pumping.
    #[classattr]
    const fn gentle() -> Self {
        Self {
            inner: ExpanderConfig::gentle(),
        }
    }

    /// Validate expander configuration parameters.
    ///
    /// Returns
    /// -------
    /// ``ExpanderConfig``
    ///     The validated expander configuration.
    ///
    /// Raises
    /// ------
    /// ``ValueError``
    ///     If any configuration parameter is invalid.
    #[pyo3(signature = ())]
    fn validate(&mut self) -> PyResult<Self> {
        self.inner.validate().map_err(audio_err_to_py)?;
        Ok(Self { inner: self.inner })
    }
}

impl_py_wrapper_core!(PyExpanderConfig, ExpanderConfig);
impl_py_default_static!(PyExpanderConfig);
impl_py_repr!(PyExpanderConfig);

/// Three-band parametric equaliser configuration.
///
/// `ThreeBandEqConfig` describes a simple three-band equaliser composed of a low
/// shelf, a mid peaking band, and a high shelf. It is a convenience configuration
/// for common tone-shaping tasks without constructing individual EQ bands.
///
/// Instances of `ThreeBandEqConfig` are immutable value objects in the Python
/// API. Parameters are provided at construction time and exposed via read-only
/// properties. Validation can be performed explicitly using ``validate()``.
///
/// A flat (unity-gain) preset is provided via ``ThreeBandEqConfig.flat``.
#[pyclass(
    name = "ThreeBandEqConfig",
    from_py_object,
    module = "audio_samples.types"
)]
#[derive(Default, Debug, Clone, Copy, PartialEq)]
pub struct PyThreeBandEqConfig {
    pub(crate) inner: ThreeBandEqConfig,
}

#[pymethods]
impl PyThreeBandEqConfig {
    /// Create a new three-band EQ configuration.
    ///
    /// Parameters
    /// ----------
    /// ``low_freq`` : float
    ///     Low shelf corner frequency in Hz. Must be greater than 0.0 and less
    ///     than ``mid_freq``.
    /// ``low_gain`` : float
    ///     Low shelf gain in dB.
    /// ``mid_freq`` : float
    ///     Mid peak centre frequency in Hz. Must be greater than ``low_freq`` and
    ///     less than ``high_freq``.
    /// ``mid_gain`` : float
    ///     Mid peak gain in dB.
    /// ``mid_q`` : float
    ///     Mid peak Q factor. Must be greater than 0.0.
    /// ``high_freq`` : float
    ///     High shelf corner frequency in Hz. Must be greater than ``mid_freq``.
    /// ``high_gain`` : float
    ///     High shelf gain in dB.
    ///
    /// Returns
    /// -------
    /// ``ThreeBandEqConfig``
    ///     A new three-band EQ configuration.
    #[new]
    #[pyo3(signature = (*, low_freq: "float", low_gain: "float", mid_freq: "float", mid_gain: "float", mid_q: "float", high_freq: "float", high_gain: "float"))]
    const fn new(
        low_freq: f64,
        low_gain: f64,
        mid_freq: f64,
        mid_gain: f64,
        mid_q: f64,
        high_freq: f64,
        high_gain: f64,
    ) -> Self {
        Self {
            inner: ThreeBandEqConfig::new(
                low_freq, low_gain, mid_freq, mid_gain, mid_q, high_freq, high_gain,
            ),
        }
    }

    /// Low shelf corner frequency in Hz.
    #[getter]
    const fn low_freq(&self) -> f64 {
        self.inner.low_freq
    }

    /// Low shelf gain in dB.
    #[getter]
    const fn low_gain(&self) -> f64 {
        self.inner.low_gain
    }

    /// Mid peak centre frequency in Hz.
    #[getter]
    const fn mid_freq(&self) -> f64 {
        self.inner.mid_freq
    }

    /// Mid peak gain in dB.
    #[getter]
    const fn mid_gain(&self) -> f64 {
        self.inner.mid_gain
    }

    /// Mid peak Q factor.
    #[getter]
    const fn mid_q(&self) -> f64 {
        self.inner.mid_q
    }

    /// High shelf corner frequency in Hz.
    #[getter]
    const fn high_freq(&self) -> f64 {
        self.inner.high_freq
    }

    /// High shelf gain in dB.
    #[getter]
    const fn high_gain(&self) -> f64 {
        self.inner.high_gain
    }

    /// Flat (unity-gain) three-band EQ preset.
    ///
    /// Low shelf at 200 Hz, mid peak at 1 kHz (Q = 1.0), high shelf at 4 kHz, all
    /// gains at 0 dB. A neutral starting point for further adjustment.
    #[classattr]
    const fn flat() -> Self {
        Self {
            inner: ThreeBandEqConfig::flat(),
        }
    }

    /// Validate the three-band EQ parameters.
    ///
    /// Returns
    /// -------
    /// ``ThreeBandEqConfig``
    ///     The validated configuration.
    ///
    /// Raises
    /// ------
    /// ``ValueError``
    ///     If any configuration parameter is invalid (non-positive frequency,
    ///     mis-ordered frequencies, or non-positive mid Q).
    #[pyo3(signature = ())]
    fn validate(&mut self) -> PyResult<Self> {
        self.inner.validate().map_err(audio_err_to_py)?;
        Ok(Self { inner: self.inner })
    }
}

impl_py_wrapper_core!(PyThreeBandEqConfig, ThreeBandEqConfig);
impl_py_default_static!(PyThreeBandEqConfig);
impl_py_repr!(PyThreeBandEqConfig);

/// Power spectral density (PSD) estimate.
///
/// `Psd` pairs a frequency axis with the estimated power-per-Hz at each bin. The
/// two arrays are always the same length: ``frequencies[i]`` is the centre
/// frequency of bin ``i`` in Hz, and ``density[i]`` is the estimated power
/// spectral density at that frequency.
///
/// Instances of `Psd` are produced by the power spectral density transform and
/// are not constructed directly from Python. The frequency and density data are
/// exposed as NumPy arrays.
#[pyclass(name = "Psd", from_py_object, module = "audio_samples.types")]
#[derive(Clone, Debug, PartialEq)]
pub struct PyPsd {
    pub(crate) inner: Psd,
}

#[pymethods]
impl PyPsd {
    /// The frequency axis, in Hz.
    ///
    /// Returns
    /// -------
    /// numpy.ndarray
    ///     A 1-D array of bin centre frequencies in Hz. Same length as
    ///     ``density``.
    #[getter]
    fn frequencies<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.frequencies().to_vec().into_pyarray(py)
    }

    /// The estimated power spectral density (power per Hz) at each frequency bin.
    ///
    /// Returns
    /// -------
    /// numpy.ndarray
    ///     A 1-D array of power-per-Hz values. Same length as ``frequencies``.
    #[getter]
    fn density<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.density().to_vec().into_pyarray(py)
    }

    /// Return the ``(frequencies, density)`` arrays as a tuple.
    ///
    /// Returns
    /// -------
    /// tuple[numpy.ndarray, numpy.ndarray]
    ///     A pair of 1-D NumPy arrays containing the frequency axis and the
    ///     density values respectively.
    #[pyo3(name = "into_parts", signature = ())]
    fn parts<'py>(
        &self,
        py: Python<'py>,
    ) -> (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>) {
        let (freqs, density) = self.inner.clone().into_parts();
        (freqs.into_pyarray(py), density.into_pyarray(py))
    }

    /// The number of frequency bins.
    fn __len__(&self) -> usize {
        self.inner.frequencies().len()
    }

    /// String representation showing the number of bins.
    fn __repr__(&self) -> String {
        format!("Psd(bins={})", self.inner.frequencies().len())
    }
}

impl_py_wrapper_core!(PyPsd, Psd);

/// One of the twelve pitch classes of the chromatic scale.
///
/// `PitchClass` represents a pitch class independent of octave. Pitch class ``C``
/// has index 0, ascending chromatically to ``B`` at index 11.
///
/// Instances of `PitchClass` are immutable and should be treated as enum-like
/// values. They are accessed via class attributes, or constructed from a
/// chromatic index via ``PitchClass.from_index(index)``.
///
/// Available pitch classes:
///
/// - ``PitchClass.c``, ``PitchClass.c_sharp``, ``PitchClass.d``,
///   ``PitchClass.d_sharp``, ``PitchClass.e``, ``PitchClass.f``,
///   ``PitchClass.f_sharp``, ``PitchClass.g``, ``PitchClass.g_sharp``,
///   ``PitchClass.a``, ``PitchClass.a_sharp``, ``PitchClass.b``.
#[pyclass(name = "PitchClass", from_py_object, module = "audio_samples.types")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PyPitchClass {
    pub(crate) inner: PitchClass,
}

#[pymethods]
impl PyPitchClass {
    /// Pitch class C (index 0).
    #[classattr]
    const fn c() -> Self {
        Self {
            inner: PitchClass::C,
        }
    }
    /// Pitch class C# / Db (index 1).
    #[classattr]
    const fn c_sharp() -> Self {
        Self {
            inner: PitchClass::CSharp,
        }
    }
    /// Pitch class D (index 2).
    #[classattr]
    const fn d() -> Self {
        Self {
            inner: PitchClass::D,
        }
    }
    /// Pitch class D# / Eb (index 3).
    #[classattr]
    const fn d_sharp() -> Self {
        Self {
            inner: PitchClass::DSharp,
        }
    }
    /// Pitch class E (index 4).
    #[classattr]
    const fn e() -> Self {
        Self {
            inner: PitchClass::E,
        }
    }
    /// Pitch class F (index 5).
    #[classattr]
    const fn f() -> Self {
        Self {
            inner: PitchClass::F,
        }
    }
    /// Pitch class F# / Gb (index 6).
    #[classattr]
    const fn f_sharp() -> Self {
        Self {
            inner: PitchClass::FSharp,
        }
    }
    /// Pitch class G (index 7).
    #[classattr]
    const fn g() -> Self {
        Self {
            inner: PitchClass::G,
        }
    }
    /// Pitch class G# / Ab (index 8).
    #[classattr]
    const fn g_sharp() -> Self {
        Self {
            inner: PitchClass::GSharp,
        }
    }
    /// Pitch class A (index 9).
    #[classattr]
    const fn a() -> Self {
        Self {
            inner: PitchClass::A,
        }
    }
    /// Pitch class A# / Bb (index 10).
    #[classattr]
    const fn a_sharp() -> Self {
        Self {
            inner: PitchClass::ASharp,
        }
    }
    /// Pitch class B (index 11).
    #[classattr]
    const fn b() -> Self {
        Self {
            inner: PitchClass::B,
        }
    }

    /// Construct a pitch class from a chromatic index in ``0..=11``.
    ///
    /// Parameters
    /// ----------
    /// index : int
    ///     Chromatic index where 0 maps to C and 11 maps to B.
    ///
    /// Returns
    /// -------
    /// ``PitchClass``
    ///     The pitch class for the given index.
    ///
    /// Raises
    /// ------
    /// ``ValueError``
    ///     If ``index`` is greater than 11.
    #[staticmethod]
    #[pyo3(signature = (index: "int"))]
    fn from_index(index: u8) -> PyResult<Self> {
        let inner = PitchClass::from_index(index).map_err(audio_err_to_py)?;
        Ok(Self { inner })
    }

    /// The chromatic index of this pitch class (C = 0, B = 11).
    #[pyo3(name = "to_index", signature = ())]
    #[allow(clippy::trivially_copy_pass_by_ref)] // PyO3 methods receive `&self`
    fn index(&self) -> u8 {
        self.inner.to_index()
    }

    /// Human-readable name of the pitch class (e.g. ``"C#"``).
    #[allow(clippy::trivially_copy_pass_by_ref)] // PyO3 methods receive `&self`
    fn __str__(&self) -> String {
        self.inner.to_string()
    }

    /// Human-readable representation of the pitch class.
    #[allow(clippy::trivially_copy_pass_by_ref)] // PyO3 methods receive `&self`
    fn __repr__(&self) -> String {
        format!("PitchClass({})", self.inner)
    }
}

impl_py_wrapper_core!(PyPitchClass, PitchClass);
impl_py_repr!(PyPitchClass);

/// The mode (tonality) of an estimated musical key.
///
/// `Mode` distinguishes between the major and minor tonalities of a key estimate.
///
/// Instances of `Mode` are immutable and should be treated as enum-like values.
/// They are accessed via class attributes rather than being constructed directly.
///
/// Available modes:
///
/// - ``Mode.major``
/// - ``Mode.minor``
#[pyclass(name = "Mode", from_py_object, module = "audio_samples.types")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PyMode {
    pub(crate) inner: Mode,
}

#[pymethods]
impl PyMode {
    /// Major mode.
    #[classattr]
    const fn major() -> Self {
        Self { inner: Mode::Major }
    }

    /// Minor mode.
    #[classattr]
    const fn minor() -> Self {
        Self { inner: Mode::Minor }
    }

    /// Human-readable name of the mode.
    #[allow(clippy::trivially_copy_pass_by_ref)] // PyO3 methods receive `&self`
    fn __str__(&self) -> String {
        match self.inner {
            Mode::Major => "major".to_string(),
            Mode::Minor => "minor".to_string(),
        }
    }

    /// Human-readable representation of the mode.
    #[allow(clippy::trivially_copy_pass_by_ref)] // PyO3 methods receive `&self`
    fn __repr__(&self) -> String {
        format!("Mode({})", self.__str__())
    }
}

impl_py_wrapper_core!(PyMode, Mode);
impl_py_repr!(PyMode);

/// A musical key estimate.
///
/// `Key` pairs a tonic pitch class with a mode (major or minor) and a confidence
/// score describing how strongly the estimate matched the analysed signal.
///
/// Instances are typically produced by key estimation, but can also be
/// constructed directly. Fields are exposed via read-only properties.
#[pyclass(name = "Key", from_py_object, module = "audio_samples.types")]
#[derive(Debug, Clone, Copy)]
pub struct PyKey {
    pub(crate) inner: Key,
}

#[pymethods]
impl PyKey {
    /// Create a new key estimate.
    ///
    /// Parameters
    /// ----------
    /// tonic : ``PitchClass``
    ///     The tonic pitch class of the key.
    /// mode : Mode
    ///     The mode (major or minor) of the key.
    /// confidence : float
    ///     Match confidence in [0.0, 1.0]; higher values indicate a stronger
    ///     match.
    ///
    /// Returns
    /// -------
    /// Key
    ///     A new key estimate.
    #[new]
    #[pyo3(signature = (tonic: "PitchClass", mode: "Mode", confidence: "float"))]
    const fn new(tonic: PyPitchClass, mode: PyMode, confidence: f64) -> Self {
        Self {
            inner: Key {
                tonic: tonic.inner,
                mode: mode.inner,
                confidence,
            },
        }
    }

    /// The tonic pitch class of the key.
    #[getter]
    const fn tonic(&self) -> PyPitchClass {
        PyPitchClass {
            inner: self.inner.tonic,
        }
    }

    /// The mode (major or minor) of the key.
    #[getter]
    const fn mode(&self) -> PyMode {
        PyMode {
            inner: self.inner.mode,
        }
    }

    /// Match confidence in [0.0, 1.0].
    #[getter]
    const fn confidence(&self) -> f64 {
        self.inner.confidence
    }

    /// Human-readable key name (e.g. ``"C# minor"``).
    fn __str__(&self) -> String {
        let mode = match self.inner.mode {
            Mode::Major => "major",
            Mode::Minor => "minor",
        };
        format!("{} {}", self.inner.tonic, mode)
    }

    /// Human-readable representation including the confidence score.
    fn __repr__(&self) -> String {
        let mode = match self.inner.mode {
            Mode::Major => "major",
            Mode::Minor => "minor",
        };
        format!(
            "Key(tonic={}, mode={}, confidence={:.3})",
            self.inner.tonic, mode, self.inner.confidence
        )
    }
}

impl_py_wrapper_core!(PyKey, Key);
impl_py_repr!(PyKey);

/// A single frame of a pitch contour.
///
/// `PitchFrame` pairs a frame onset time with the fundamental frequency detected
/// in that frame. The frequency is ``None`` for unvoiced frames where no pitch
/// was detected.
///
/// Fields are exposed via read-only properties.
#[pyclass(name = "PitchFrame", from_py_object, module = "audio_samples.types")]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PyPitchFrame {
    pub(crate) inner: PitchFrame,
}

#[pymethods]
impl PyPitchFrame {
    /// Create a new pitch frame.
    ///
    /// Parameters
    /// ----------
    /// time : float
    ///     Frame onset time, in seconds from the start of the signal.
    /// frequency : float, optional
    ///     Detected fundamental frequency in Hz, or ``None`` if the frame is
    ///     unvoiced.
    ///
    /// Returns
    /// -------
    /// ``PitchFrame``
    ///     A new pitch frame.
    #[new]
    #[pyo3(signature = (time: "float", frequency = None))]
    const fn new(time: f64, frequency: Option<f64>) -> Self {
        Self {
            inner: PitchFrame { time, frequency },
        }
    }

    /// Frame onset time, in seconds from the start of the signal.
    #[getter]
    const fn time(&self) -> f64 {
        self.inner.time
    }

    /// Detected fundamental frequency in Hz, or ``None`` if the frame is unvoiced.
    #[getter]
    const fn frequency(&self) -> Option<f64> {
        self.inner.frequency
    }

    /// Whether this frame is voiced (a pitch was detected).
    #[getter]
    const fn voiced(&self) -> bool {
        self.inner.frequency.is_some()
    }

    /// Human-readable representation of the frame.
    fn __repr__(&self) -> String {
        self.inner.frequency.map_or_else(
            || format!("PitchFrame(time={:.4}, frequency=None)", self.inner.time),
            |hz| format!("PitchFrame(time={:.4}, frequency={hz:.2})", self.inner.time),
        )
    }
}

impl_py_wrapper_core!(PyPitchFrame, PitchFrame);
impl_py_repr!(PyPitchFrame);

/// A time-ordered pitch track.
///
/// `PitchContour` is a sequence of [`PitchFrame`] values produced by pitch
/// tracking. Each frame pairs a frame onset time with the pitch detected in that
/// frame (or ``None`` when no pitch was found).
///
/// Instances are produced by pitch tracking and are not constructed directly from
/// Python.
#[pyclass(name = "PitchContour", from_py_object, module = "audio_samples.types")]
#[derive(Debug, Clone, PartialEq)]
pub struct PyPitchContour {
    pub(crate) inner: PitchContour,
}

#[pymethods]
impl PyPitchContour {
    /// All frames in time order, voiced and unvoiced alike.
    ///
    /// Returns
    /// -------
    /// list[``PitchFrame``]
    ///     The complete list of pitch frames.
    #[pyo3(signature = ())]
    fn frames(&self) -> Vec<PyPitchFrame> {
        self.inner
            .frames()
            .iter()
            .map(|&inner| PyPitchFrame { inner })
            .collect()
    }

    /// Voiced frames as ``(time_seconds, frequency_hz)`` pairs.
    ///
    /// Unvoiced frames are skipped.
    ///
    /// Returns
    /// -------
    /// list[tuple[float, float]]
    ///     The voiced frames as (time, frequency) tuples.
    #[pyo3(signature = ())]
    fn voiced_frames(&self) -> Vec<(f64, f64)> {
        self.inner.voiced_frames().collect()
    }

    /// The mean of all voiced frequencies, or ``None`` when no frame is voiced.
    #[pyo3(signature = ())]
    fn mean_pitch(&self) -> Option<f64> {
        self.inner.mean_pitch()
    }

    /// The total number of frames (voiced and unvoiced).
    fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// Human-readable representation showing the frame count.
    fn __repr__(&self) -> String {
        format!("PitchContour(frames={})", self.inner.len())
    }
}

impl_py_wrapper_core!(PyPitchContour, PitchContour);
impl_py_repr!(PyPitchContour);

// Field-listing `__repr__` for configuration classes (shows the Rust Debug
// form of the wrapped value, e.g. `VadConfig { frame_size: 512, ... }`).
crate::impl_py_debug_repr!(PyVadConfig);
crate::impl_py_debug_repr!(PyCompressorConfig);
crate::impl_py_debug_repr!(PyLimiterConfig);
crate::impl_py_debug_repr!(PyGateConfig);
crate::impl_py_debug_repr!(PyExpanderConfig);
crate::impl_py_debug_repr!(PyIirFilterDesign);
crate::impl_py_debug_repr!(PyEqBand);
crate::impl_py_debug_repr!(PyParametricEq);
crate::impl_py_debug_repr!(PyHpssConfig);
crate::impl_py_debug_repr!(PyOnsetDetectionConfig);
crate::impl_py_debug_repr!(PyPeakPickingConfig);
crate::impl_py_debug_repr!(PyBeatTrackingConfig);
crate::impl_py_debug_repr!(PySpectralFluxConfig);
crate::impl_py_debug_repr!(PyComplexOnsetConfig);
crate::impl_py_debug_repr!(PyAdaptiveThresholdConfig);
crate::impl_py_debug_repr!(PySideChainConfig);
crate::impl_py_debug_repr!(PyThreeBandEqConfig);
crate::impl_py_debug_repr!(PyPerturbationConfig);
crate::impl_py_debug_repr!(PyEnvelopeFollower);
crate::impl_py_debug_repr!(PyVadMethod);
crate::impl_py_debug_repr!(PyVadChannelPolicy);
