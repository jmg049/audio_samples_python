use crate::args::Int;
use audio_samples::{AudioProcessing, CastInto, ConvertTo, NormalizationConfig};
use non_empty_slice::NonEmptyVec;
use numpy::{PyArray1, PyArrayMethods};
use pyo3::{exceptions::PyValueError, prelude::*, types::IntoPyDict};

use pyo3::types::PyAnyMethods;

use crate::{
    PyAudioSamples, audio_err_to_py, dispatch_with_view, dispatch_with_view_detached,
    types::{PyNormalizationMethod, PyResamplingQuality},
};

/// A resampling quality given as a ``ResamplingQuality`` or by name
/// (``"fast"``, ``"medium"``, ``"high"``, ``"very_high"``).
#[cfg(feature = "stubs")]
use pyo3::type_hint_union;

#[derive(Debug, Clone, Copy)]
pub struct QualityArg(pub PyResamplingQuality);

impl<'a, 'py> FromPyObject<'a, 'py> for QualityArg {
    type Error = PyErr;
    #[cfg(feature = "stubs")]
    const INPUT_TYPE: pyo3::inspect::PyStaticExpr = type_hint_union!(
        <PyResamplingQuality as pyo3::PyTypeInfo>::TYPE_HINT,
        crate::args::hints::QUALITY_NAME
    );

    fn extract(obj: pyo3::Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        if let Ok(q) = obj.extract::<PyResamplingQuality>() {
            return Ok(Self(q));
        }
        if let Ok(s) = obj.extract::<String>() {
            return match s.to_ascii_lowercase().as_str() {
                "fast" => Ok(Self(PyResamplingQuality::resample_fast())),
                "medium" => Ok(Self(PyResamplingQuality::resample_mediuem())),
                "high" => Ok(Self(PyResamplingQuality::resample_high())),
                "very_high" => Ok(Self(PyResamplingQuality::resample_very_high())),
                other => Err(PyValueError::new_err(format!(
                    "quality: unknown resampling quality {other:?}; expected 'fast', 'medium', \
                     'high' or 'very_high'"
                ))),
            };
        }
        Err(pyo3::exceptions::PyTypeError::new_err(
            "quality must be a ResamplingQuality or one of 'fast', 'medium', 'high', 'very_high'",
        ))
    }
}

#[pymethods]
impl PyAudioSamples {
    /// Scale samples by a constant factor in-place.
    ///
    /// Modifies the signal in place and returns ``None`` (the NumPy array it
    /// was created from, if any, is left unchanged).
    ///
    /// Args:
    ///
    ///     factor (float): Multiplier applied to every sample; values above 1.0 amplify and below 1.0 attenuate.
    #[pyo3(signature = (factor: "float"))]
    pub fn scale(&mut self, py: Python<'_>, factor: f64) -> PyResult<()> {
        let result = dispatch_with_view!(self, py, |audio| {
            Self::from_audio_samples(audio.scale(factor).into_owned())
        });
        *self = result;
        Ok(())
    }

    /// Normalize samples in-place using one of the supported strategies.
    ///
    /// Modifies the signal in place and returns ``None`` (the NumPy array it
    /// was created from, if any, is left unchanged).
    ///
    /// Args:
    ///
    ///     min (float): Lower bound for the normalized output.
    ///     max (float): Upper bound for the normalized output.
    ///     method (NormalizationMethod | str): Strategy controlling the normalization.
    ///
    /// Raises:
    ///
    ///     AudioError: If normalization fails.
    #[pyo3(signature = (min: "float", max: "float", method: "NormalizationMethod | str"))]
    pub fn normalize(
        &mut self,
        py: Python<'_>,
        min: f64,
        max: f64,
        method: &pyo3::Bound<'_, pyo3::PyAny>,
    ) -> PyResult<()> {
        use std::str::FromStr;
        let method: PyNormalizationMethod = if let Ok(m) = method.extract::<PyNormalizationMethod>()
        {
            m
        } else if let Ok(s) = method.extract::<String>() {
            PyNormalizationMethod::from_str(&s).map_err(|_| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "Unknown normalization method: '{s}'"
                ))
            })?
        } else {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "method must be a NormalizationMethod or str",
            ));
        };
        let result = dispatch_with_view!(self, py, |audio| {
            let mut normalization_config =
                NormalizationConfig::min_max(min.convert_to(), max.convert_to());
            normalization_config.method = method.into();
            audio
                .normalize(normalization_config)
                .map_err(audio_err_to_py)
                .map(|a| Self::from_audio_samples(a.into_owned()))
        })?;
        *self = result;
        Ok(())
    }

    /// Clamp samples in-place to the provided minimum and maximum.
    ///
    /// Modifies the signal in place and returns ``None`` (the NumPy array it
    /// was created from, if any, is left unchanged).
    ///
    /// Args:
    ///
    ///     min_val (float): Lower bound for clamping.
    ///     max_val (float): Upper bound for clamping.
    ///
    /// Raises:
    ///
    ///     AudioError: If clamping fails or the bounds are invalid.
    #[pyo3(signature = (min_val: "float", max_val: "float"))]
    pub fn clip(&mut self, py: Python<'_>, min_val: f64, max_val: f64) -> PyResult<()> {
        let result = dispatch_with_view!(self, py, |audio| {
            audio
                .clip(min_val.cast_into(), max_val.cast_into())
                .map_err(audio_err_to_py)
                .map(|a| Self::from_audio_samples(a.into_owned()))
        })?;
        *self = result;
        Ok(())
    }

    /// Remove the DC offset from the signal in-place.
    ///
    /// Modifies the signal in place and returns ``None`` (the NumPy array it
    /// was created from, if any, is left unchanged).
    ///
    /// Raises:
    ///
    ///     AudioError: If offset removal fails.
    #[pyo3(signature = ())]
    pub fn remove_dc_offset(&mut self, py: Python<'_>) -> PyResult<()> {
        let result = dispatch_with_view!(self, py, |audio| audio
            .remove_dc_offset()
            .map_err(audio_err_to_py)
            .map(|a| Self::from_audio_samples(a.into_owned())))?;
        *self = result;
        Ok(())
    }

    /// Apply mu-law compression to the signal.
    ///
    /// Args:
    ///
    ///     mu (float): Mu parameter controlling the compression curve.
    ///
    /// Returns:
    ///
    ///     AudioSamples: New buffer containing the compressed samples.
    ///
    /// Raises:
    ///
    ///     AudioError: If compression fails.
    #[pyo3(signature = (mu: "float"))]
    pub fn mu_compress(&self, py: Python<'_>, mu: f64) -> PyResult<Self> {
        dispatch_with_view!(self, py, |audio| {
            let a = audio.mu_compress(mu.cast_into()).map_err(audio_err_to_py)?;
            Ok(Self::from_audio_samples(a.into_owned()))
        })
    }

    /// Apply mu-law expansion to the signal.
    ///
    /// Args:
    ///
    ///     mu (float): Mu parameter used during compression.
    ///
    /// Returns:
    ///
    ///     AudioSamples: New buffer containing the expanded samples.
    ///
    /// Raises:
    ///
    ///     AudioError: If expansion fails.
    #[pyo3(signature = (mu: "float"))]
    pub fn mu_expand(&self, py: Python<'_>, mu: f64) -> PyResult<Self> {
        dispatch_with_view!(self, py, |audio| {
            let a = audio.mu_expand(mu.cast_into()).map_err(audio_err_to_py)?;
            Ok(Self::from_audio_samples(a.into_owned()))
        })
    }

    /// Apply a first-order Butterworth low-pass filter, returning a new buffer.
    ///
    /// -3 dB at ``cutoff_hz``, 6 dB/octave: the bilinear-transform design
    /// prewarped at ``cutoff_hz``, run causally from a zero initial state.
    /// ``self`` is not modified.
    ///
    /// Args:
    ///
    ///     cutoff_hz (float): Cutoff frequency in hertz, in (0, Nyquist).
    ///
    /// Returns:
    ///
    ///     AudioSamples: New buffer containing the filtered signal.
    ///
    /// Raises:
    ///
    ///     AudioError: If filtering fails.
    #[pyo3(signature = (cutoff_hz: "float"))]
    pub fn low_pass_filter(&self, py: Python<'_>, cutoff_hz: f64) -> PyResult<Self> {
        dispatch_with_view!(self, py, |audio| {
            audio
                .low_pass_filter(cutoff_hz)
                .map_err(audio_err_to_py)
                .map(|a| Self::from_audio_samples(a.into_owned()))
        })
    }

    /// Apply a first-order Butterworth high-pass filter, returning a new buffer.
    ///
    /// -3 dB at ``cutoff_hz``, 6 dB/octave: the bilinear-transform design
    /// prewarped at ``cutoff_hz``, run causally from a zero initial state.
    /// ``self`` is not modified.
    ///
    /// Args:
    ///
    ///     cutoff_hz (float): Cutoff frequency in hertz, in (0, Nyquist).
    ///
    /// Returns:
    ///
    ///     AudioSamples: New buffer containing the filtered signal.
    ///
    /// Raises:
    ///
    ///     AudioError: If filtering fails.
    #[pyo3(signature = (cutoff_hz: "float"))]
    pub fn high_pass_filter(&self, py: Python<'_>, cutoff_hz: f64) -> PyResult<Self> {
        dispatch_with_view!(self, py, |audio| {
            audio
                .high_pass_filter(cutoff_hz)
                .map_err(audio_err_to_py)
                .map(|a| Self::from_audio_samples(a.into_owned()))
        })
    }

    /// Apply a band-pass filter (first-order Butterworth prototype), returning a new buffer.
    ///
    /// A second-order band-pass from the first-order Butterworth prototype by
    /// the bilinear transform, prewarped at both edges, run causally from a
    /// zero initial state. ``self`` is not modified.
    ///
    /// Args:
    ///
    ///     low_cutoff_hz (float): Lower -3 dB edge in hertz.
    ///     high_cutoff_hz (float): Upper -3 dB edge in hertz.
    ///
    /// Returns:
    ///
    ///     AudioSamples: New buffer containing the filtered signal.
    ///
    /// Raises:
    ///
    ///     AudioError: If filtering fails.
    #[pyo3(signature = (low_cutoff_hz: "float", high_cutoff_hz: "float"))]
    pub fn band_pass_filter(
        &self,
        py: Python<'_>,
        low_cutoff_hz: f64,
        high_cutoff_hz: f64,
    ) -> PyResult<Self> {
        dispatch_with_view!(self, py, |audio| {
            audio
                .band_pass_filter(low_cutoff_hz, high_cutoff_hz)
                .map_err(audio_err_to_py)
                .map(|a| Self::from_audio_samples(a.into_owned()))
        })
    }
    /// Resample the audio to a new sample rate.
    ///
    /// A Kaiser-windowed sinc interpolator evaluated at the exact rational
    /// position of every output sample: the output is time-aligned with the
    /// input (no filter delay, nothing to trim), the signal is treated as zero
    /// outside its ends (so the tail is flushed), and the result has exactly
    /// ``ceil(n * target_sample_rate / sample_rate)`` samples per channel. The
    /// cut-off sits just below the Nyquist frequency of the lower rate, so
    /// downsampling is anti-aliased and upsampling anti-imaged. Processing is
    /// in the sample type's float (float32 for uint8, int16, int24 and float32;
    /// float64 for int32 and float64); the result keeps the input's sample type.
    ///
    /// Quality tiers (Nyquist of the lower rate):
    ///
    /// * ``"fast"`` (default): downsampling with a short windowed sinc
    ///   (passband 0.80 x Nyquist, 70 dB stopband); upsampling by linear
    ///   interpolation (images are not suppressed).
    /// * ``"medium"``: passband 0.90 x Nyquist, 100 dB stopband.
    /// * ``"high"``: passband 0.913 x Nyquist, 125 dB stopband.
    /// * ``"very_high"``: 185 dB stopband; passband 0.95 x Nyquist for
    ///   decimation by a power of two, 0.98 for other ratios with no prime
    ///   factor above 7, 0.996 for the rest (such as pitch shifts).
    ///
    /// Args:
    ///
    ///     target_sample_rate (int): Output sample rate in Hz (> 0).
    ///     quality (ResamplingQuality | str): ``ResamplingQuality.fast`` (default),
    ///         ``.medium``, ``.high``, ``.very_high``, or the names ``"fast"``,
    ///         ``"medium"``, ``"high"``, ``"very_high"``.
    ///
    /// Returns:
    ///
    ///     AudioSamples: the resampled audio (a copy when the rates are equal).
    ///
    /// Raises:
    ///
    ///     ValueError: If ``target_sample_rate`` is not a positive integer, or an
    ///         unknown quality name is given.
    #[pyo3(signature = (target_sample_rate: "int", quality: "ResamplingQuality | str" = QualityArg(PyResamplingQuality::resample_fast())), text_signature = "($self, target_sample_rate, quality='fast')")]
    pub fn resample(
        &self,
        py: Python<'_>,
        target_sample_rate: Int,
        quality: QualityArg,
    ) -> PyResult<Self> {
        let target_sample_rate = u32::try_from(target_sample_rate.0)
            .ok()
            .and_then(std::num::NonZeroU32::new)
            .ok_or_else(|| {
                PyValueError::new_err(format!(
                    "target_sample_rate: must be a positive integer, got {target_sample_rate}"
                ))
            })?;
        let quality = *quality.0;
        dispatch_with_view_detached!(self, py, |audio| audio
            .resample(target_sample_rate, quality)
            .map(Self::from_audio_samples))
        .map_err(audio_err_to_py)
    }

    /// ``resample`` returning the samples as an array of their own type. Float samples are
    /// moved into the array rather than copied out of an ``AudioSamples``.
    ///
    /// Args:
    ///
    ///     ``target_sample_rate`` (int): As ``resample``.
    ///     quality (ResamplingQuality | str): As ``resample``.
    ///     gain (float | None): Float samples only: each output ``y`` becomes ``y * gain``
    ///         computed in float64 and rounded to the sample type, as it is written.
    ///
    /// Returns:
    ///
    ///     numpy.ndarray: Time on the last axis.
    #[pyo3(signature = (target_sample_rate, quality, gain = None))]
    fn _resample_array<'py>(
        &self,
        py: Python<'py>,
        target_sample_rate: Int,
        quality: QualityArg,
        gain: Option<f64>,
    ) -> PyResult<Bound<'py, PyAny>> {
        macro_rules! moved {
            ($a:expr) => {{
                let rate = u32::try_from(target_sample_rate.0)
                    .ok()
                    .and_then(std::num::NonZeroU32::new)
                    .ok_or_else(|| {
                        PyValueError::new_err(format!(
                            "target_sample_rate: must be a positive integer, got {target_sample_rate}"
                        ))
                    })?;
                let q = *quality.0;
                let out = $a
                    .with_view_detached(py, |audio| match gain {
                        Some(g) => audio_samples::resample_scaled(&audio, rate, q, g),
                        None => audio.resample(rate, q),
                    })
                    .map_err(audio_err_to_py)?;
                Ok(if out.is_mono() {
                    let data = out.into_array1().expect("mono audio has one dimension");
                    numpy::IntoPyArray::into_pyarray(data, py).into_any()
                } else {
                    let data = out.into_array2().expect("multi-channel audio has two dimensions");
                    numpy::IntoPyArray::into_pyarray(data, py).into_any()
                })
            }};
        }
        match self.inner() {
            crate::PyAudioDataInner::F32(a) => moved!(a),
            crate::PyAudioDataInner::F64(a) => moved!(a),
            _ if gain.is_some() => Err(PyValueError::new_err(
                "gain: only float32 and float64 samples are scaled while resampling",
            )),
            _ => self.resample(py, target_sample_rate, quality)?.to_numpy(py),
        }
    }

    /// Resample the audio by a ratio of sample rates.
    ///
    /// The target rate is ``round(sample_rate * ratio)``; the conversion is then
    /// exactly ``resample(target_rate, quality)`` (delay-free, exactly
    /// ``ceil(n * target_rate / sample_rate)`` samples; see ``resample`` for the
    /// quality tiers).
    ///
    /// Args:
    ///
    ///     ratio (float): Output rate / input rate (> 0).
    ///     quality (ResamplingQuality | str): As ``resample``; default ``"fast"``.
    ///
    /// Returns:
    ///
    ///     AudioSamples: the resampled audio at ``round(sample_rate * ratio)`` Hz.
    ///
    /// Raises:
    ///
    ///     ValueError: If ``ratio`` is not positive or the target rate rounds to 0.
    #[pyo3(signature = (ratio: "float", quality: "ResamplingQuality | str" = QualityArg(PyResamplingQuality::resample_fast())), text_signature = "($self, ratio, quality='fast')")]
    pub fn resample_by_ratio(
        &self,
        py: Python<'_>,
        ratio: f64,
        quality: QualityArg,
    ) -> PyResult<Self> {
        let quality = quality.0;
        dispatch_with_view!(self, py, |audio| {
            let resampled = audio
                .resample_by_ratio(ratio, *quality)
                .map_err(audio_err_to_py)?;
            return Ok(Self::from_audio_samples(resampled));
        })
    }
    /// Apply a sample-wise window to the signal.
    ///
    /// Each sample is multiplied by its float64 coefficient. Integer audio is
    /// multiplied on its raw scale and the products are rounded to nearest and
    /// saturated to the sample type, so fractional windows (Hann, Hamming, ...)
    /// work for every sample type.
    ///
    /// Args:
    ///
    ///     window (numpy.ndarray): 1-D sequence of coefficients matching the
    ///         audio length. Any real array-like is accepted (e.g. float32);
    ///         it is converted to float64.
    ///
    /// Returns:
    ///
    ///     AudioSamples: New buffer of the same sample type with the window
    ///     applied.
    ///
    /// Raises:
    ///
    ///     ValueError: If the window length does not equal the audio length.
    ///     AudioError: If windowing fails.
    #[pyo3(signature = (window: "np.typing.NDArray[np.floating]"))]
    pub fn apply_window(&self, py: Python<'_>, window: &Bound<'_, PyAny>) -> PyResult<Self> {
        let window = as_f64_vector(py, window, "window")?;
        let window_data = window.try_readonly()?.as_array().to_owned();
        if window_data.len() != self.len(py) {
            return Err(PyValueError::new_err(format!(
                "Window length {} does not match audio length {}",
                window_data.len(),
                self.len(py)
            )));
        }
        let window = window_data.to_vec();
        // SAFETY: the window length equals the audio length, which is non-empty
        let window: NonEmptyVec<f64> = unsafe { NonEmptyVec::new_unchecked(window) };
        dispatch_with_view!(self, py, |audio| {
            audio
                .apply_window_f64(&window)
                .map_err(audio_err_to_py)
                .map(|a| Self::from_audio_samples(a.into_owned()))
        })
    }

    /// Apply an FIR filter by "valid" linear convolution, returning a new buffer.
    ///
    /// For taps ``h[0..M]`` and a channel ``x`` of ``N`` samples this computes
    ///
    /// ``y[i] = sum_{k=0}^{M-1} h[k] * x[i + M - 1 - k]``,  ``i = 0 .. N - M``
    ///
    /// i.e. only the ``N - M + 1`` outputs where every tap overlaps the input.
    /// The result is ``numpy.convolve(x, h, mode='valid')`` applied to each
    /// channel independently. ``self`` is not modified.
    ///
    /// ``convolve`` vs ``apply_filter``: :meth:`convolve` returns the *full*
    /// convolution of length ``N + M - 1`` (``numpy.convolve(..., 'full')``) and
    /// takes another ``AudioSamples``; ``apply_filter`` takes a coefficient
    /// array and returns only the *valid* part of length ``N - M + 1``, so the
    /// output is shorter than the input by ``M - 1`` samples. Output index
    /// ``i`` corresponds to input time ``i + M - 1``; the filter's group delay
    /// (``(M - 1) / 2`` samples for a symmetric, linear-phase filter) is not
    /// compensated. Filters with 160 or more taps use FFT convolution; the
    /// result is the same up to rounding.
    ///
    /// Computation is carried out in float64 with float64 coefficients. Integer
    /// audio is filtered on its raw scale and the result is rounded to nearest
    /// and saturated to the sample type, so taps outside ``[-1, 1]`` work for
    /// every sample type.
    ///
    /// Args:
    ///
    ///     coefficients (numpy.ndarray): 1-D array-like of FIR taps ``h``
    ///         (non-empty, converted to float64). ``[1.0]`` leaves the signal
    ///         unchanged.
    ///
    /// Returns:
    ///
    ///     AudioSamples: Filtered buffer with ``N - M + 1`` samples per channel.
    ///
    /// Raises:
    ///
    ///     ValueError: If ``coefficients`` is empty.
    ///     AudioParameterError: If the audio is shorter than the filter.
    ///
    /// Examples:
    ///
    ///     >>> import numpy as np, audio_samples as aus
    ///     >>> x = aus.AudioSamples.new_mono(np.array([1.0, 2.0, 3.0, 4.0, 5.0]), sample_rate=8000)
    ///     >>> x.apply_filter(np.array([1.0, 0.5])).to_numpy()
    ///     array([2.5, 4. , 5.5, 7. ])
    #[pyo3(signature = (coefficients: "np.typing.NDArray[np.floating]"))]
    pub fn apply_filter(&self, py: Python<'_>, coefficients: &Bound<'_, PyAny>) -> PyResult<Self> {
        let coefficients = as_f64_vector(py, coefficients, "coefficients")?;
        let coeffs = coefficients.try_readonly()?.as_array().to_vec();
        if coeffs.is_empty() {
            return Err(PyValueError::new_err(
                "apply_filter requires at least one filter coefficient",
            ));
        }
        // SAFETY: checked non-empty above
        let taps: NonEmptyVec<f64> = unsafe { NonEmptyVec::new_unchecked(coeffs) };
        dispatch_with_view!(self, py, |audio| {
            audio
                .apply_filter_f64(&taps)
                .map_err(audio_err_to_py)
                .map(|a| Self::from_audio_samples(a.into_owned()))
        })
    }

    /// Apply an FIR filter by "valid" convolution in place.
    ///
    /// Modifies the signal in place and returns ``None`` (the NumPy array it
    /// was created from, if any, is left unchanged).
    ///
    /// In-place twin of :meth:`apply_filter` (same formula and conventions):
    /// the buffer is replaced by the ``N - M + 1`` valid outputs, so it becomes
    /// ``M - 1`` samples shorter.
    ///
    /// Args:
    ///
    ///     coefficients (numpy.ndarray): 1-D array-like of FIR taps (non-empty).
    ///
    /// Returns:
    ///
    ///     None: Operation mutates the current buffer.
    ///
    /// Raises:
    ///
    ///     ValueError: If ``coefficients`` is empty.
    ///     AudioParameterError: If the audio is shorter than the filter.
    #[pyo3(signature = (coefficients: "np.typing.NDArray[np.floating]"))]
    pub fn apply_filter_in_place(
        &mut self,
        py: Python<'_>,
        coefficients: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let result = self.apply_filter(py, coefficients)?;
        *self = result;
        Ok(())
    }
}

/// Convert any real 1-D array-like to a float64 NumPy array
/// (`numpy.asarray(obj, dtype=float64)`), with a clear `TypeError` otherwise.
fn as_f64_vector<'py>(
    py: Python<'py>,
    obj: &Bound<'py, PyAny>,
    name: &str,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    if let Ok(arr) = obj.cast::<PyArray1<f64>>() {
        return Ok(arr.clone());
    }
    let np = py.import("numpy")?;
    let converted = np
        .getattr("asarray")?
        .call(
            (obj,),
            Some(&[("dtype", np.getattr("float64")?)].into_py_dict(py)?),
        )
        .map_err(|e| {
            pyo3::exceptions::PyTypeError::new_err(format!(
                "{name} must be a 1-D array of real numbers convertible to float64: {e}"
            ))
        })?;
    converted.cast_into::<PyArray1<f64>>().map_err(|_| {
        pyo3::exceptions::PyTypeError::new_err(format!(
            "{name} must be 1-D, got an array of shape {}",
            converted_shape(obj)
        ))
    })
}

fn converted_shape(obj: &Bound<'_, PyAny>) -> String {
    obj.getattr("shape")
        .and_then(|s| s.str().map(|s| s.to_string()))
        .unwrap_or_else(|_| "unknown".to_string())
}
