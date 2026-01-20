use audio_samples::{AudioData, AudioSamples, AudioStatistics, StandardSample};
use numpy::{Element, PyArrayMethods};
use pyo3::{IntoPyObjectExt, prelude::*};

use crate::{
    PyAudioBacking, PyAudioDataInner, PyAudioSamples, TypedAudioSamples, audio_err_to_py,
    dispatch_with_view, nzu_or_err,
};

#[pymethods]
impl PyAudioSamples {
    #[pyo3(signature = (), text_signature = "($self) -> int | float")]
    /// Compute the peak sample value across all channels.
    ///
    /// Returns:
    ///     int | float: Maximum absolute amplitude present in the buffer.
    fn peak<'py>(&self, py: Python<'py>) -> Bound<'py, PyAny> {
        dispatch_with_view!(self, py, |audio| {
            audio
                .peak()
                .into_bound_py_any(py)
                .expect("Numbers should always be able to be converted to a pyobject.")
        })
    }

    #[pyo3(name = "min", signature = (), text_signature = "($self) -> int | float")]
    /// Compute the minimum sample value across all channels.
    ///
    /// Returns:
    ///     int | float: Lowest signed amplitude value found in the buffer.
    fn min_sample<'py>(&self, py: Python<'py>) -> Bound<'py, PyAny> {
        dispatch_with_view!(self, py, |audio| {
            audio
                .min_sample()
                .into_bound_py_any(py)
                .expect("Numbers should always be able to be converted to a pyobject.")
        })
    }

    #[pyo3(name = "max", signature = (), text_signature = "($self) -> int | float")]
    /// Compute the maximum sample value across all channels.
    ///
    /// Returns:
    ///     int | float: Highest signed amplitude value found in the buffer.
    fn max_sample<'py>(&self, py: Python<'py>) -> Bound<'py, PyAny> {
        dispatch_with_view!(self, py, |audio| {
            audio
                .max_sample()
                .into_bound_py_any(py)
                .expect("Numbers should always be able to be converted to a pyobject.")
        })
    }

    #[pyo3(signature = (), text_signature = "($self) -> float")]
    /// Calculate the arithmetic mean of the sample values.
    ///
    /// Returns:
    ///     float: Mean amplitude across all samples and channels.
    fn mean(&self, py: Python<'_>) -> f64 {
        dispatch_with_view!(self, py, |audio| audio.mean())
    }

    #[pyo3(signature = (), text_signature = "($self) -> Optional[float]")]
    /// Calculate the median sample value.
    ///
    /// Returns:
    ///     Optional[float]: Median amplitude if the view is non-empty, None otherwise.
    fn median(&self, py: Python<'_>) -> Option<f64> {
        dispatch_with_view!(self, py, |audio| audio.median())
    }

    #[pyo3(signature = (), text_signature = "($self) -> float")]
    /// Compute the root mean square (RMS) energy of the samples.
    ///
    /// Returns:
    ///     float: RMS amplitude across all samples and channels.
    fn rms(&self, py: Python<'_>) -> f64 {
        dispatch_with_view!(self, py, |audio| audio.rms())
    }

    #[pyo3(signature = ())]
    /// Compute the variance of the sample values.
    ///
    /// Returns:
    ///     float: Variance of amplitudes across all samples and channels.
    fn variance(&self, py: Python<'_>) -> f64 {
        dispatch_with_view!(self, py, |audio| audio.variance())
    }

    #[pyo3(signature = (), text_signature = "($self) -> float")]
    /// Compute the standard deviation of the sample values.
    ///
    /// Returns:
    ///     float: Standard deviation derived from the variance.
    fn std_dev(&self, py: Python<'_>) -> f64 {
        self.variance(py).sqrt()
    }

    #[pyo3(signature = (), text_signature = "($self) -> int")]
    /// Count the number of zero crossings in the signal.
    ///
    /// Returns:
    ///     int: Number of sign changes across consecutive samples.
    fn zero_crossings(&self, py: Python<'_>) -> usize {
        dispatch_with_view!(self, py, |audio| audio.zero_crossings())
    }

    #[pyo3(signature = (), text_signature = "($self) -> float")]
    /// Calculate the zero crossing rate of the signal.
    ///
    /// Returns:
    ///     float: Zero crossings normalized by the number of samples.
    fn zero_crossing_rate(&self, py: Python<'_>) -> f64 {
        dispatch_with_view!(self, py, |audio| audio.zero_crossing_rate())
    }

    #[pyo3(signature = (max_lag), text_signature = "($self, max_lag: int) -> Optional[list[float]]")]
    /// Compute the autocorrelation sequence up to a lag limit.
    ///
    /// Args:
    ///     max_lag (int): Maximum lag to include; must be greater than zero.
    ///
    /// Returns:
    ///     Optional[list[float]]: Autocorrelation coefficients if the buffer has samples, None otherwise.
    ///
    /// Raises:
    ///     ValueError: If max_lag is zero.
    fn autocorrelation(&self, py: Python<'_>, max_lag: usize) -> PyResult<Option<Vec<f64>>> {
        let max_lag = nzu_or_err(max_lag)?;
        dispatch_with_view!(self, py, |audio| Ok(audio
            .autocorrelation(max_lag)
            .map(|nev| nev.to_vec())))
    }

    #[pyo3(signature = (other, max_lag), text_signature = "($self, other: AudioSamples, max_lag: int) -> Optional[list[float]]")]
    /// Compute the normalized cross-correlation with another audio buffer.
    ///
    /// Args:
    ///     other (AudioSamples): Audio buffer to correlate against.
    ///     max_lag (int): Maximum lag to include; must be greater than zero.
    ///
    /// Returns:
    ///     list[float]: Cross-correlation coefficients up to the requested lag.
    ///
    /// Raises:
    ///     ValueError: If max_lag is zero or the sample rates or layouts differ.
    ///     TypeError: If the buffers have incompatible backings or data types.
    ///     NotImplementedError: If either buffer uses an interleaved NumPy backing.
    fn cross_correlation(
        &self,
        py: Python<'_>,
        other: &PyAudioSamples,
        max_lag: usize,
    ) -> PyResult<Vec<f64>> {
        let max_lag = nzu_or_err(max_lag)?;
        fn cross_correlation_dual<T>(
            a: &TypedAudioSamples<T>,
            b: &TypedAudioSamples<T>,
            py: Python<'_>,
            max_lag: usize,
        ) -> PyResult<Vec<f64>>
        where
            T: StandardSample + Element,
        {
            // Validate compatibility
            if a.sample_rate != b.sample_rate {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "AudioSamples must have the same sample rate and layout for cross-correlation",
                ));
            }
            let max_lag = nzu_or_err(max_lag)?;
            let sr = a.sample_rate;

            // Create both AudioSamples in the same scope by manually implementing
            // the with_view logic without closures
            use PyAudioBacking::*;
            let result = match (&a.backing, &b.backing) {
                (OwnedMono(arr_a), OwnedMono(arr_b)) => {
                    let view_a = arr_a.view();
                    let view_b = arr_b.view();
                    // safety: we know view_a is not empty
                    let audio_a = AudioSamples {
                        data: unsafe { AudioData::from_borrowed_array1_unchecked(view_a) },
                        sample_rate: sr,
                    };
                    // safety: we know view_b is not empty
                    let audio_b = AudioSamples {
                        data: unsafe { AudioData::from_borrowed_array1_unchecked(view_b) },
                        sample_rate: sr,
                    };
                    audio_a.cross_correlation(&audio_b, max_lag)
                }
                (OwnedMulti(arr_a), OwnedMulti(arr_b)) => {
                    let view_a = arr_a.view();
                    let view_b = arr_b.view();
                    // safety: we know view_a is not empty
                    let audio_a = AudioSamples {
                        data: unsafe { AudioData::from_borrowed_array2_unchecked(view_a) },
                        sample_rate: sr,
                    };
                    // safety: we know view_b is not empty
                    let audio_b = AudioSamples {
                        data: unsafe { AudioData::from_borrowed_array2_unchecked(view_b) },
                        sample_rate: sr,
                    };
                    audio_a.cross_correlation(&audio_b, max_lag)
                }
                (NumpyMono(handle_a), NumpyMono(handle_b)) => {
                    let bound_a = handle_a.bind(py);
                    let bound_b = handle_b.bind(py);
                    let view_a = unsafe { bound_a.as_array() };
                    let view_b = unsafe { bound_b.as_array() };
                    // safety: We know view_a is not empty
                    let audio_a = AudioSamples {
                        data: unsafe { AudioData::from_borrowed_array1_unchecked(view_a) },
                        sample_rate: sr,
                    };
                    // safety: We know view_b is not empty
                    let audio_b = AudioSamples {
                        data: unsafe { AudioData::from_borrowed_array1_unchecked(view_b) },
                        sample_rate: sr,
                    };
                    audio_a.cross_correlation(&audio_b, max_lag)
                }
                (NumpyMulti(handle_a), NumpyMulti(handle_b)) => {
                    let bound_a = handle_a.bind(py);
                    let bound_b = handle_b.bind(py);
                    let view_a = unsafe { bound_a.as_array() };
                    let view_b = unsafe { bound_b.as_array() };
                    // safety: We know view_a is not empty
                    let audio_a = AudioSamples {
                        data: unsafe { AudioData::from_borrowed_array2_unchecked(view_a) },
                        sample_rate: sr,
                    };
                    // safety: We know view_b is not empty
                    let audio_b = AudioSamples {
                        data: unsafe { AudioData::from_borrowed_array2_unchecked(view_b) },
                        sample_rate: sr,
                    };
                    audio_a.cross_correlation(&audio_b, max_lag)
                }
                (NumpyInterleaved(_), NumpyInterleaved(_)) => {
                    // Interleaved cross-correlation requires complex deinterleaving
                    // For now, return an error for this case
                    return Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
                        "Cross-correlation on interleaved NumPy arrays not yet supported",
                    ));
                }
                _ => {
                    return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                        "Incompatible backing types for cross-correlation",
                    ));
                }
            };

            result.map(|nev| nev.to_vec()).map_err(audio_err_to_py)
        }

        match (&self.inner, &other.inner) {
            (PyAudioDataInner::I16(a), PyAudioDataInner::I16(b)) => {
                cross_correlation_dual(a, b, py, max_lag.get())
            }
            (PyAudioDataInner::I24(a), PyAudioDataInner::I24(b)) => {
                cross_correlation_dual(a, b, py, max_lag.get())
            }
            (PyAudioDataInner::I32(a), PyAudioDataInner::I32(b)) => {
                cross_correlation_dual(a, b, py, max_lag.get())
            }
            (PyAudioDataInner::F32(a), PyAudioDataInner::F32(b)) => {
                cross_correlation_dual(a, b, py, max_lag.get())
            }
            (PyAudioDataInner::F64(a), PyAudioDataInner::F64(b)) => {
                cross_correlation_dual(a, b, py, max_lag.get())
            }
            _ => Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "Cross-correlation requires both AudioSamples to have the same data type",
            )),
        }
    }

    #[pyo3(signature = (), text_signature = "($self) -> float")]
    /// Compute the spectral centroid using the current FFT backend.
    ///
    /// Returns:
    ///     float: Frequency-weighted centroid of the magnitude spectrum.
    ///
    /// Raises:
    ///     AudioError: If the FFT computation fails.
    fn spectral_centroid(&self, py: Python<'_>) -> PyResult<f64> {
        dispatch_with_view!(self, py, |audio| audio
            .spectral_centroid()
            .map_err(audio_err_to_py))
    }

    #[pyo3(signature = (rolloff_percent: "float" = 0.85), text_signature = "($self, rolloff_percent: float = 0.85) -> float")]
    /// Compute the spectral rolloff point for the provided energy fraction.
    ///
    /// Args:
    ///     rolloff_percent (float): Target cumulative spectral energy fraction between 0 and 1.
    ///
    /// Returns:
    ///     float: Frequency where the cumulative spectrum reaches the rolloff threshold.
    ///
    /// Raises:
    ///     AudioError: If the FFT computation fails or rolloff_percent is invalid.
    fn spectral_rolloff(&self, py: Python<'_>, rolloff_percent: f64) -> PyResult<f64> {
        dispatch_with_view!(self, py, |audio| audio
            .spectral_rolloff(rolloff_percent)
            .map_err(audio_err_to_py))
    }
}
