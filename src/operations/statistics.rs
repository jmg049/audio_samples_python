use audio_samples::operations::types::ChannelReduction;
use audio_samples::{AudioData, AudioSamples, AudioStatistics, StandardSample};
use numpy::{Element, PyArrayMethods};
use pyo3::{IntoPyObjectExt, prelude::*};

use crate::{
    PyAudioBacking, PyAudioDataInner, PyAudioSamples, TypedAudioSamples, audio_err_to_py,
    dispatch_with_view, nzu_or_err,
    types::PyChannelReduction,
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
        dispatch_with_view!(self, py, |audio| audio.midpoint_sample())
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
                    let audio_a = AudioSamples::new(unsafe { AudioData::from_borrowed_array1_unchecked(view_a) }, sr);
                    // safety: we know view_b is not empty
                    let audio_b = AudioSamples::new(unsafe { AudioData::from_borrowed_array1_unchecked(view_b) }, sr);
                    audio_a.cross_correlation(&audio_b, max_lag)
                }
                (OwnedMulti(arr_a), OwnedMulti(arr_b)) => {
                    let view_a = arr_a.view();
                    let view_b = arr_b.view();
                    // safety: we know view_a is not empty
                    let audio_a = AudioSamples::new(unsafe { AudioData::from_borrowed_array2_unchecked(view_a) }, sr);
                    // safety: we know view_b is not empty
                    let audio_b = AudioSamples::new(unsafe { AudioData::from_borrowed_array2_unchecked(view_b) }, sr);
                    audio_a.cross_correlation(&audio_b, max_lag)
                }
                (NumpyMono(handle_a), NumpyMono(handle_b)) => {
                    let bound_a = handle_a.bind(py);
                    let bound_b = handle_b.bind(py);
                    let view_a = unsafe { bound_a.as_array() };
                    let view_b = unsafe { bound_b.as_array() };
                    // safety: We know view_a is not empty
                    let audio_a = AudioSamples::new(unsafe { AudioData::from_borrowed_array1_unchecked(view_a) }, sr);
                    // safety: We know view_b is not empty
                    let audio_b = AudioSamples::new(unsafe { AudioData::from_borrowed_array1_unchecked(view_b) }, sr);
                    audio_a.cross_correlation(&audio_b, max_lag)
                }
                (NumpyMulti(handle_a), NumpyMulti(handle_b)) => {
                    let bound_a = handle_a.bind(py);
                    let bound_b = handle_b.bind(py);
                    let view_a = unsafe { bound_a.as_array() };
                    let view_b = unsafe { bound_b.as_array() };
                    // safety: We know view_a is not empty
                    let audio_a = AudioSamples::new(unsafe { AudioData::from_borrowed_array2_unchecked(view_a) }, sr);
                    // safety: We know view_b is not empty
                    let audio_b = AudioSamples::new(unsafe { AudioData::from_borrowed_array2_unchecked(view_b) }, sr);
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

    #[pyo3(signature = (reduction: "ChannelReduction" = ChannelReduction::Average.into()), text_signature = "($self, reduction: ChannelReduction = ChannelReduction.average) -> float")]
    /// Compute the spectral centroid using the current FFT backend.
    ///
    /// Args:
    ///     reduction (ChannelReduction): Channel-reduction policy applied to
    ///         multi-channel input. Defaults to ``ChannelReduction.average``.
    ///
    /// Returns:
    ///     float: Frequency-weighted centroid of the magnitude spectrum.
    ///
    /// Raises:
    ///     AudioError: If the FFT computation fails or the reduction is invalid
    ///         for the channel layout.
    fn spectral_centroid(
        &self,
        py: Python<'_>,
        reduction: PyChannelReduction,
    ) -> PyResult<f64> {
        let reduction = ChannelReduction::from(reduction);
        dispatch_with_view!(self, py, |audio| audio
            .spectral_centroid(reduction)
            .map_err(audio_err_to_py))
    }

    #[pyo3(signature = (rolloff_percent: "float" = 0.85, reduction: "ChannelReduction" = ChannelReduction::Average.into()), text_signature = "($self, rolloff_percent: float = 0.85, reduction: ChannelReduction = ChannelReduction.average) -> float")]
    /// Compute the spectral rolloff point for the provided energy fraction.
    ///
    /// Args:
    ///     rolloff_percent (float): Target cumulative spectral energy fraction between 0 and 1.
    ///     reduction (ChannelReduction): Channel-reduction policy applied to
    ///         multi-channel input. Defaults to ``ChannelReduction.average``.
    ///
    /// Returns:
    ///     float: Frequency where the cumulative spectrum reaches the rolloff threshold.
    ///
    /// Raises:
    ///     AudioError: If the FFT computation fails or rolloff_percent is invalid.
    fn spectral_rolloff(
        &self,
        py: Python<'_>,
        rolloff_percent: f64,
        reduction: PyChannelReduction,
    ) -> PyResult<f64> {
        let reduction = ChannelReduction::from(reduction);
        dispatch_with_view!(self, py, |audio| audio
            .spectral_rolloff(rolloff_percent, reduction)
            .map_err(audio_err_to_py))
    }

    #[pyo3(signature = (reduction: "ChannelReduction" = ChannelReduction::Average.into()), text_signature = "($self, reduction: ChannelReduction = ChannelReduction.average) -> float")]
    /// Compute the spectral bandwidth (spectral spread) of the signal.
    ///
    /// The magnitude-weighted standard deviation of the spectrum about its
    /// spectral centroid. Larger values indicate energy spread across a wider
    /// frequency range; a pure tone yields a value near zero.
    ///
    /// Args:
    ///     reduction (ChannelReduction): Channel-reduction policy applied to
    ///         multi-channel input. Defaults to ``ChannelReduction.average``.
    ///
    /// Returns:
    ///     float: Spectral bandwidth in Hz. Returns 0.0 for silence.
    ///
    /// Raises:
    ///     AudioError: If the FFT computation fails or the reduction is invalid
    ///         for the channel layout.
    fn spectral_bandwidth(
        &self,
        py: Python<'_>,
        reduction: PyChannelReduction,
    ) -> PyResult<f64> {
        let reduction = ChannelReduction::from(reduction);
        dispatch_with_view!(self, py, |audio| audio
            .spectral_bandwidth(reduction)
            .map_err(audio_err_to_py))
    }

    #[pyo3(signature = (reduction: "ChannelReduction" = ChannelReduction::Average.into()), text_signature = "($self, reduction: ChannelReduction = ChannelReduction.average) -> float")]
    /// Compute the spectral flatness (Wiener entropy) of the signal.
    ///
    /// The ratio of the geometric mean to the arithmetic mean of the power
    /// spectrum. Values near 1 indicate noise-like (flat) spectra, while values
    /// near 0 indicate tonal spectra dominated by a few peaks.
    ///
    /// Args:
    ///     reduction (ChannelReduction): Channel-reduction policy applied to
    ///         multi-channel input. Defaults to ``ChannelReduction.average``.
    ///
    /// Returns:
    ///     float: Spectral flatness in [0, 1]. Returns 0.0 for silence.
    ///
    /// Raises:
    ///     AudioError: If the FFT computation fails or the reduction is invalid
    ///         for the channel layout.
    fn spectral_flatness(
        &self,
        py: Python<'_>,
        reduction: PyChannelReduction,
    ) -> PyResult<f64> {
        let reduction = ChannelReduction::from(reduction);
        dispatch_with_view!(self, py, |audio| audio
            .spectral_flatness(reduction)
            .map_err(audio_err_to_py))
    }

    #[pyo3(signature = (reduction: "ChannelReduction" = ChannelReduction::Average.into()), text_signature = "($self, reduction: ChannelReduction = ChannelReduction.average) -> float")]
    /// Compute the spectral crest factor of the signal.
    ///
    /// The ratio of the peak magnitude to the mean magnitude of the spectrum.
    /// High values indicate a strongly peaked (tonal) spectrum; a flat spectrum
    /// approaches 1.
    ///
    /// Args:
    ///     reduction (ChannelReduction): Channel-reduction policy applied to
    ///         multi-channel input. Defaults to ``ChannelReduction.average``.
    ///
    /// Returns:
    ///     float: Spectral crest factor (>= 1 for non-silent signals). Returns
    ///         0.0 for silence.
    ///
    /// Raises:
    ///     AudioError: If the FFT computation fails or the reduction is invalid
    ///         for the channel layout.
    fn spectral_crest(
        &self,
        py: Python<'_>,
        reduction: PyChannelReduction,
    ) -> PyResult<f64> {
        let reduction = ChannelReduction::from(reduction);
        dispatch_with_view!(self, py, |audio| audio
            .spectral_crest(reduction)
            .map_err(audio_err_to_py))
    }

    #[pyo3(signature = (reduction: "ChannelReduction" = ChannelReduction::Average.into()), text_signature = "($self, reduction: ChannelReduction = ChannelReduction.average) -> float")]
    /// Compute the spectral slope of the signal.
    ///
    /// The slope of an ordinary least-squares linear fit of the linear magnitude
    /// spectrum against frequency (Hz), in units of magnitude per Hz. A negative
    /// slope indicates energy concentrated at low frequencies.
    ///
    /// Args:
    ///     reduction (ChannelReduction): Channel-reduction policy applied to
    ///         multi-channel input. Defaults to ``ChannelReduction.average``.
    ///
    /// Returns:
    ///     float: Least-squares slope (magnitude per Hz). Returns 0.0 for
    ///         silence or a degenerate single-bin spectrum.
    ///
    /// Raises:
    ///     AudioError: If the FFT computation fails or the reduction is invalid
    ///         for the channel layout.
    fn spectral_slope(
        &self,
        py: Python<'_>,
        reduction: PyChannelReduction,
    ) -> PyResult<f64> {
        let reduction = ChannelReduction::from(reduction);
        dispatch_with_view!(self, py, |audio| audio
            .spectral_slope(reduction)
            .map_err(audio_err_to_py))
    }

    #[pyo3(signature = (n_bands: "int", reduction: "ChannelReduction" = ChannelReduction::Average.into()), text_signature = "($self, n_bands: int, reduction: ChannelReduction = ChannelReduction.average) -> list[float]")]
    /// Compute spectral contrast across octave-spaced sub-bands.
    ///
    /// The spectrum is partitioned into ``n_bands`` octave-spaced sub-bands.
    /// Within each band the contrast is the dB difference between the mean of the
    /// top quantile (peaks) and the mean of the bottom quantile (valleys). High
    /// contrast indicates clear tonal/harmonic structure.
    ///
    /// Args:
    ///     n_bands (int): Number of octave-spaced sub-bands; must be greater than zero.
    ///     reduction (ChannelReduction): Channel-reduction policy applied to
    ///         multi-channel input. Defaults to ``ChannelReduction.average``.
    ///
    /// Returns:
    ///     list[float]: ``n_bands`` contrast values in dB, low band to high band.
    ///
    /// Raises:
    ///     ValueError: If n_bands is zero.
    ///     AudioError: If the FFT computation fails or the reduction is invalid
    ///         for the channel layout.
    fn spectral_contrast(
        &self,
        py: Python<'_>,
        n_bands: usize,
        reduction: PyChannelReduction,
    ) -> PyResult<Vec<f64>> {
        let n_bands = nzu_or_err(n_bands)?;
        let reduction = ChannelReduction::from(reduction);
        dispatch_with_view!(self, py, |audio| audio
            .spectral_contrast(n_bands, reduction)
            .map_err(audio_err_to_py))
    }

    #[pyo3(signature = (), text_signature = "($self) -> Optional[float]")]
    /// Return the value at the temporal midpoint of a mono signal.
    ///
    /// This is the 2.0 name for what was historically exposed as ``median``.
    /// For even-length signals the result is the average of the two central
    /// samples; for odd-length signals the single central sample is returned.
    /// Samples are selected by index position; the buffer is not sorted.
    ///
    /// Returns:
    ///     Optional[float]: Midpoint value for mono audio, or None if the signal
    ///         is multi-channel.
    fn midpoint_sample(&self, py: Python<'_>) -> Option<f64> {
        dispatch_with_view!(self, py, |audio| audio.midpoint_sample())
    }

    #[pyo3(signature = (), text_signature = "($self) -> tuple[float, int | float]")]
    /// Compute the RMS and peak absolute value in a single pass.
    ///
    /// Equivalent to calling ``rms`` and ``peak`` separately, but reads the
    /// sample buffer only once.
    ///
    /// Returns:
    ///     tuple[float, int | float]: A ``(rms, peak)`` pair where ``rms`` is the
    ///         root-mean-square as a float and ``peak`` is the maximum absolute
    ///         sample value in the native sample type.
    fn rms_and_peak<'py>(&self, py: Python<'py>) -> (f64, Bound<'py, PyAny>) {
        dispatch_with_view!(self, py, |audio| {
            let (rms, peak) = audio.rms_and_peak();
            (
                rms,
                peak.into_bound_py_any(py)
                    .expect("Numbers should always be able to be converted to a pyobject."),
            )
        })
    }

    #[pyo3(signature = (), text_signature = "($self) -> int | float")]
    /// Return the peak (maximum absolute value) across all samples and channels.
    ///
    /// Alias for ``peak``, provided to match the conventional term "amplitude"
    /// used in some audio contexts.
    ///
    /// Returns:
    ///     int | float: Maximum absolute amplitude present in the buffer.
    fn amplitude<'py>(&self, py: Python<'py>) -> Bound<'py, PyAny> {
        dispatch_with_view!(self, py, |audio| {
            audio
                .amplitude()
                .into_bound_py_any(py)
                .expect("Numbers should always be able to be converted to a pyobject.")
        })
    }
}
