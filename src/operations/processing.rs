use audio_samples::{AudioProcessing, CastInto, ConvertTo, NormalizationConfig};
use non_empty_slice::NonEmptyVec;
use numpy::{PyArray1, PyArrayMethods};
use pyo3::{exceptions::PyValueError, prelude::*, types::IntoPyDict};

use pyo3::types::PyAnyMethods;

use crate::{
    PyAudioSamples, audio_err_to_py, dispatch_with_view, nzu32_or_err,
    types::{PyNormalizationMethod, PyResamplingQuality},
};

#[pymethods]
impl PyAudioSamples {
    /// Scale samples by a constant factor in-place.
    ///
    /// Modifies the signal in place and returns ``None`` (the NumPy array it
    /// was created from, if any, is left unchanged).
    ///
    /// Args:
    ///     factor (float): Multiplier applied to every sample; values above 1.0 amplify and below 1.0 attenuate.
    #[pyo3(signature = (factor: "float"))]
    pub fn scale(&mut self, py: Python<'_>, factor: f64) {
        let result = dispatch_with_view!(self, py, |audio| {
            PyAudioSamples::from_audio_samples(audio.scale(factor).into_owned())
        });
        *self = result;
    }

    /// Normalize samples in-place using one of the supported strategies.
    ///
    /// Modifies the signal in place and returns ``None`` (the NumPy array it
    /// was created from, if any, is left unchanged).
    ///
    /// Args:
    ///     min (float): Lower bound for the normalized output.
    ///     max (float): Upper bound for the normalized output.
    ///     method (NormalizationMethod | str): Strategy controlling the normalization.
    ///
    /// Raises:
    ///     AudioError: If normalization fails.
    #[pyo3(signature = (min: "float", max: "float", method: "NormalizationMethod | str"))]
    pub fn normalize(
        &mut self,
        py: Python<'_>,
        min: f64,
        max: f64,
        method: pyo3::Bound<'_, pyo3::PyAny>,
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
    ///     min_val (float): Lower bound for clamping.
    ///     max_val (float): Upper bound for clamping.
    ///
    /// Raises:
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
    ///     mu (float): Mu parameter controlling the compression curve.
    ///
    /// Returns:
    ///     AudioSamples: New buffer containing the compressed samples.
    ///
    /// Raises:
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
    ///     mu (float): Mu parameter used during compression.
    ///
    /// Returns:
    ///     AudioSamples: New buffer containing the expanded samples.
    ///
    /// Raises:
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
    /// -3 dB at ``cutoff_hz``, 6 dB/octave; identical to
    /// ``scipy.signal.sosfilt(scipy.signal.butter(1, cutoff_hz, fs=fs, output='sos'), x)``.
    /// ``self`` is not modified.
    ///
    /// Args:
    ///     cutoff_hz (float): Cutoff frequency in hertz, in (0, Nyquist).
    ///
    /// Returns:
    ///     AudioSamples: New buffer containing the filtered signal.
    ///
    /// Raises:
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
    /// -3 dB at ``cutoff_hz``, 6 dB/octave; identical to
    /// ``scipy.signal.sosfilt(scipy.signal.butter(1, cutoff_hz, 'highpass', fs=fs, output='sos'), x)``.
    /// ``self`` is not modified.
    ///
    /// Args:
    ///     cutoff_hz (float): Cutoff frequency in hertz, in (0, Nyquist).
    ///
    /// Returns:
    ///     AudioSamples: New buffer containing the filtered signal.
    ///
    /// Raises:
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
    /// Identical to ``scipy.signal.sosfilt(scipy.signal.butter(1, [low, high],
    /// 'bandpass', fs=fs, output='sos'), x)``. ``self`` is not modified.
    ///
    /// Args:
    ///     low_cutoff_hz (float): Lower -3 dB edge in hertz.
    ///     high_cutoff_hz (float): Upper -3 dB edge in hertz.
    ///
    /// Returns:
    ///     AudioSamples: New buffer containing the filtered signal.
    ///
    /// Raises:
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
    /// Args:
    ///     target_sample_rate (int): Desired output sample rate in hertz; must be greater than zero.
    ///     quality (ResamplingQuality): Quality preset controlling the resampler.
    ///         Defaults to ``ResamplingQuality.fast``; pass
    ///         ``ResamplingQuality.high`` for the highest-fidelity resampler.
    ///
    /// Returns:
    ///     AudioSamples: New buffer rendered at the requested sample rate.
    ///
    /// Raises:
    ///     ValueError: If target_sample_rate is zero.
    ///     AudioError: If resampling fails.
    #[pyo3(signature = (target_sample_rate: "int", quality:"ResamplingQuality" = PyResamplingQuality::resample_fast()))]
    pub fn resample(
        &self,
        py: Python<'_>,
        target_sample_rate: usize,
        quality: PyResamplingQuality,
    ) -> PyResult<Self> {
        let target_sample_rate = nzu32_or_err(target_sample_rate as u32)?;
        dispatch_with_view!(self, py, |audio| {
            let resampled = audio
                .resample(target_sample_rate, *quality)
                .map_err(audio_err_to_py)?;
            return Ok(PyAudioSamples::from_audio_samples(resampled));
        })
    }

    /// Resample the audio by a multiplicative ratio.
    ///
    /// Args:
    ///     ratio (float): Factor applied to the original sample rate.
    ///     quality (ResamplingQuality): Quality preset controlling the resampler.
    ///         Defaults to ``ResamplingQuality.fast``.
    ///
    /// Returns:
    ///     AudioSamples: New buffer rendered at the scaled sample rate.
    ///
    /// Raises:
    ///     AudioError: If resampling fails.
    #[pyo3(signature = (ratio: "float", quality: "ResamplingQuality" = PyResamplingQuality::resample_fast()))]
    pub fn resample_by_ratio<'py>(
        &'py self,
        py: Python<'_>,
        ratio: f64,
        quality: PyResamplingQuality,
    ) -> PyResult<Self> {
        dispatch_with_view!(self, py, |audio| {
            let resampled = audio
                .resample_by_ratio(ratio, *quality)
                .map_err(audio_err_to_py)?;
            return Ok(PyAudioSamples::from_audio_samples(resampled));
        })
    }

    /// Apply a sample-wise window to the signal.
    ///
    /// Args:
    ///     window (numpy.ndarray): 1-D sequence of coefficients matching the
    ///         audio length. Any real array-like is accepted (e.g. float32);
    ///         it is converted to float64.
    ///
    /// Returns:
    ///     AudioSamples: New buffer with the window applied.
    ///
    /// Raises:
    ///     ValueError: If the window length does not equal the audio length.
    ///     AudioError: If windowing fails.
    #[pyo3(signature = (window: "np.typing.NDArray[np.floating]"))]
    pub fn apply_window(&self, py: Python<'_>, window: &Bound<'_, PyAny>) -> PyResult<Self> {
        let window = as_f64_vector(py, window, "window")?;
        let window_data = window.readonly().as_array().to_owned();
        if window_data.len() != self.len(py) {
            return Err(PyValueError::new_err(format!(
                "Window length {} does not match audio length {}",
                window_data.len(),
                self.len(py)
            )));
        }
        dispatch_with_view!(self, py, |audio| {
            let window_t: Vec<_> = window_data.into_iter().map(ConvertTo::convert_to).collect();
            // safety: have checked for non-empty already
            let window_t: NonEmptyVec<_> = unsafe { NonEmptyVec::new_unchecked(window_t) };
            audio
                .apply_window(&window_t)
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
    /// The result is identical to ``numpy.convolve(x, h, mode='valid')`` (and to
    /// ``scipy.signal.lfilter(h, 1, x)[M-1:]``), applied to each channel
    /// independently. ``self`` is not modified.
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
    /// Computation is carried out in float64. For integer audio both the
    /// samples and the coefficients are interpreted on the normalised
    /// ``[-1, 1]`` scale, so coefficients are quantised to the sample type and
    /// must lie in ``[-1, 1]``; convert to float first (``as_f64()``) for
    /// general filters.
    ///
    /// Args:
    ///     coefficients (numpy.ndarray): 1-D array-like of FIR taps ``h``
    ///         (non-empty, converted to float64). ``[1.0]`` leaves the signal
    ///         unchanged.
    ///
    /// Returns:
    ///     AudioSamples: Filtered buffer with ``N - M + 1`` samples per channel.
    ///
    /// Raises:
    ///     ValueError: If ``coefficients`` is empty.
    ///     AudioParameterError: If the audio is shorter than the filter.
    ///
    /// Examples:
    ///     >>> import numpy as np, audio_samples as aus
    ///     >>> x = aus.AudioSamples.new_mono(np.array([1.0, 2.0, 3.0, 4.0, 5.0]), sample_rate=8000)
    ///     >>> x.apply_filter(np.array([1.0, 0.5])).to_numpy()
    ///     array([2.5, 4. , 5.5, 7. ])
    #[pyo3(signature = (coefficients: "np.typing.NDArray[np.floating]"))]
    pub fn apply_filter(&self, py: Python<'_>, coefficients: &Bound<'_, PyAny>) -> PyResult<Self> {
        let coefficients = as_f64_vector(py, coefficients, "coefficients")?;
        let coeffs = coefficients.readonly().as_array().to_vec();
        if coeffs.is_empty() {
            return Err(PyValueError::new_err(
                "apply_filter requires at least one filter coefficient",
            ));
        }
        dispatch_with_view!(self, py, |audio| {
            let taps: Vec<_> = coeffs.iter().copied().map(ConvertTo::convert_to).collect();
            // safety: checked non-empty above
            let taps: NonEmptyVec<_> = unsafe { NonEmptyVec::new_unchecked(taps) };
            audio
                .apply_filter(&taps)
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
    ///     coefficients (numpy.ndarray): 1-D array-like of FIR taps (non-empty).
    ///
    /// Returns:
    ///     None: Operation mutates the current buffer.
    ///
    /// Raises:
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
