use audio_samples::{AudioProcessing, CastInto, ConvertTo, NormalizationConfig};
use non_empty_slice::NonEmptyVec;
use numpy::{PyArray1, PyArrayMethods};
use pyo3::{exceptions::PyValueError, prelude::*};

use pyo3::types::PyAnyMethods;

use crate::{
    PyAudioSamples, audio_err_to_py, dispatch_with_view, nzu32_or_err,
    types::{PyNormalizationMethod, PyResamplingQuality},
};

#[pymethods]
impl PyAudioSamples {
    /// Scale samples by a constant factor in-place.
    ///
    /// Args:
    ///     factor (float): Multiplier applied to every sample; values above 1.0 amplify and below 1.0 attenuate.
    #[pyo3(signature = (factor: "float"), text_signature = "($self, factor: float)")]
    pub fn scale(&mut self, py: Python<'_>, factor: f64) {
        let result = dispatch_with_view!(self, py, |audio| {
            PyAudioSamples::from_audio_samples(audio.scale(factor).into_owned())
        });
        *self = result;
    }

    /// Normalize samples in-place using one of the supported strategies.
    ///
    /// Args:
    ///     min (float): Lower bound for the normalized output.
    ///     max (float): Upper bound for the normalized output.
    ///     method (NormalizationMethod | str): Strategy controlling the normalization.
    ///
    /// Raises:
    ///     AudioError: If normalization fails.
    #[pyo3(signature = (min: "float", max: "float", method: "NormalizationMethod | str"), text_signature = "($self, min: float, max: float, method: NormalizationMethod | str)")]
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
    /// Args:
    ///     min_val (float): Lower bound for clamping.
    ///     max_val (float): Upper bound for clamping.
    ///
    /// Raises:
    ///     AudioError: If clamping fails or the bounds are invalid.
    #[pyo3(signature = (min_val: "float", max_val: "float"), text_signature = "($self, min_val: float, max_val: float)")]
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
    /// Raises:
    ///     AudioError: If offset removal fails.
    #[pyo3(signature = (), text_signature = "($self)")]
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
    #[pyo3(signature = (mu: "float"), text_signature = "($self, mu: float) -> AudioSamples")]
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
    #[pyo3(signature = (mu: "float"), text_signature = "($self, mu: float) -> AudioSamples")]
    pub fn mu_expand(&self, py: Python<'_>, mu: f64) -> PyResult<Self> {
        dispatch_with_view!(self, py, |audio| {
            let a = audio.mu_expand(mu.cast_into()).map_err(audio_err_to_py)?;
            Ok(Self::from_audio_samples(a.into_owned()))
        })
    }

    /// Apply a low-pass filter to the signal.
    ///
    /// Args:
    ///     cutoff_hz (float): Frequency threshold above which content is attenuated.
    ///
    /// Returns:
    ///     AudioSamples: New buffer containing the filtered signal.
    ///
    /// Raises:
    ///     AudioError: If filtering fails.
    #[pyo3(signature = (cutoff_hz: "float"), text_signature = "($self, cutoff_hz: float) -> None")]
    pub fn low_pass_filter(&self, py: Python<'_>, cutoff_hz: f64) -> PyResult<Self> {
        dispatch_with_view!(self, py, |audio| {
            audio
                .low_pass_filter(cutoff_hz)
                .map_err(audio_err_to_py)
                .map(|a| Self::from_audio_samples(a.into_owned()))
        })
    }

    /// Apply a high-pass filter to the signal.
    ///
    /// Args:
    ///     cutoff_hz (float): Frequency threshold below which content is attenuated.
    ///
    /// Returns:
    ///     AudioSamples: New buffer containing the filtered signal.
    ///
    /// Raises:
    ///     AudioError: If filtering fails.
    #[pyo3(signature = (cutoff_hz: "float"), text_signature = "($self, cutoff_hz: float) -> None")]
    pub fn high_pass_filter(&self, py: Python<'_>, cutoff_hz: f64) -> PyResult<Self> {
        dispatch_with_view!(self, py, |audio| {
            audio
                .high_pass_filter(cutoff_hz)
                .map_err(audio_err_to_py)
                .map(|a| Self::from_audio_samples(a.into_owned()))
        })
    }

    /// Apply a band-pass filter to the signal.
    ///
    /// Args:
    ///     low_cutoff_hz (float): Lower frequency bound to retain.
    ///     high_cutoff_hz (float): Upper frequency bound to retain.
    ///
    /// Returns:
    ///     AudioSamples: New buffer containing the filtered signal.
    ///
    /// Raises:
    ///     AudioError: If filtering fails.
    #[pyo3(signature = (low_cutoff_hz: "float", high_cutoff_hz: "float"), text_signature = "($self, low_cutoff_hz: float, high_cutoff_hz: float) -> None")]
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
    ///
    /// Returns:
    ///     AudioSamples: New buffer rendered at the requested sample rate.
    ///
    /// Raises:
    ///     ValueError: If target_sample_rate is zero.
    ///     AudioError: If resampling fails.
    #[pyo3(signature = (target_sample_rate: "int", quality:"ResamplingQuality" = PyResamplingQuality::resample_fast()), text_signature = "($self, target_sample_rate: int, quality: ResamplingQuality) -> AudioSamples")]
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
    ///
    /// Returns:
    ///     AudioSamples: New buffer rendered at the scaled sample rate.
    ///
    /// Raises:
    ///     AudioError: If resampling fails.
    #[pyo3(signature = (ratio: "float", quality: "ResamplingQuality" = PyResamplingQuality::resample_fast()), text_signature = "($self, ratio: float, quality: Literal['fast', 'medium', 'high']) -> AudioSamples")]
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
    ///     window (numpy.ndarray): Sequence of coefficients matching the audio length.
    ///
    /// Returns:
    ///     AudioSamples: New buffer with the window applied.
    ///
    /// Raises:
    ///     ValueError: If the window length does not equal the audio length.
    ///     AudioError: If windowing fails.
    #[pyo3(signature = (window: "np.typing.NDArray[np.float64]"), text_signature = "($self, window: np.typing.NDArray[np.float64]) -> None")]
    pub fn apply_window(
        &self,
        py: Python<'_>,
        window: &Bound<'_, PyArray1<f64>>,
    ) -> PyResult<Self> {
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
}
