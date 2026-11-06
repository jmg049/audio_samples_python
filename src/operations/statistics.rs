use audio_samples::operations::types::ChannelReduction;
use audio_samples::{AudioData, AudioSamples, AudioStatistics, StandardSample};
use numpy::{Element, PyArrayMethods};
use pyo3::{IntoPyObjectExt, prelude::*};

use crate::{
    PyAudioBacking, PyAudioDataInner, PyAudioSamples, TypedAudioSamples, audio_err_to_py,
    dispatch_with_view, nzu_or_err, types::PyChannelReduction,
};

#[pymethods]
impl PyAudioSamples {
    #[pyo3(signature = ())]
    /// Compute the peak sample value across all channels.
    ///
    /// Returns:
    ///
    ///     int | float: Maximum absolute amplitude present in the buffer.
    fn peak<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        Ok({
            dispatch_with_view!(self, py, |audio| {
                audio
                    .peak()
                    .into_bound_py_any(py)
                    .expect("Numbers should always be able to be converted to a pyobject.")
            })
        })
    }

    #[pyo3(name = "min", signature = ())]
    /// Compute the minimum sample value across all channels.
    ///
    /// Returns:
    ///
    ///     int | float: Lowest signed amplitude value found in the buffer.
    fn min_sample<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        Ok({
            dispatch_with_view!(self, py, |audio| {
                audio
                    .min_sample()
                    .into_bound_py_any(py)
                    .expect("Numbers should always be able to be converted to a pyobject.")
            })
        })
    }

    #[pyo3(name = "max", signature = ())]
    /// Compute the maximum sample value across all channels.
    ///
    /// Returns:
    ///
    ///     int | float: Highest signed amplitude value found in the buffer.
    fn max_sample<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        Ok({
            dispatch_with_view!(self, py, |audio| {
                audio
                    .max_sample()
                    .into_bound_py_any(py)
                    .expect("Numbers should always be able to be converted to a pyobject.")
            })
        })
    }

    #[pyo3(signature = ())]
    /// Calculate the arithmetic mean of the sample values.
    ///
    /// Returns:
    ///
    ///     float: Mean amplitude across all samples and channels.
    fn mean(&self, py: Python<'_>) -> PyResult<f64> {
        Ok({ dispatch_with_view!(self, py, |audio| audio.mean()) })
    }

    #[pyo3(signature = ())]
    /// Calculate the median sample value.
    ///
    /// Returns:
    ///
    ///     Optional[float]: Median amplitude if the view is non-empty, None otherwise.
    fn median(&self, py: Python<'_>) -> PyResult<Option<f64>> {
        Ok({ dispatch_with_view!(self, py, |audio| audio.midpoint_sample()) })
    }

    #[pyo3(signature = ())]
    /// Compute the root mean square (RMS) energy of the samples.
    ///
    /// Returns:
    ///
    ///     float: RMS amplitude across all samples and channels.
    fn rms(&self, py: Python<'_>) -> PyResult<f64> {
        Ok({ dispatch_with_view!(self, py, |audio| audio.rms()) })
    }

    #[pyo3(signature = ())]
    /// Compute the variance of the sample values.
    ///
    /// Returns:
    ///
    ///     float: Variance of amplitudes across all samples and channels.
    fn variance(&self, py: Python<'_>) -> PyResult<f64> {
        Ok({ dispatch_with_view!(self, py, |audio| audio.variance()) })
    }

    #[pyo3(signature = ())]
    /// Compute the standard deviation of the sample values.
    ///
    /// Returns:
    ///
    ///     float: Standard deviation derived from the variance.
    fn std_dev(&self, py: Python<'_>) -> PyResult<f64> {
        Ok(self.variance(py)?.sqrt())
    }

    #[pyo3(signature = ())]
    /// Count the number of zero crossings in the signal.
    ///
    /// A crossing is counted between consecutive samples ``x[n-1]`` and ``x[n]``
    /// when one is ``> 0`` and the other is ``<= 0`` (zero counts as
    /// non-positive). For multi-channel audio the per-channel counts are summed.
    ///
    /// Returns:
    ///
    ///     int: Number of crossings (summed over channels).
    fn zero_crossings(&self, py: Python<'_>) -> PyResult<usize> {
        Ok({ dispatch_with_view!(self, py, |audio| audio.zero_crossings()) })
    }

    #[pyo3(signature = (), text_signature = "($self)")]
    /// Calculate the zero-crossing rate of the signal, in crossings per second.
    ///
    /// ``zero_crossing_rate = zero_crossings() / (channels * duration_seconds)``,
    /// where ``duration_seconds = samples_per_channel / sample_rate``: for
    /// multi-channel audio it is the mean of the per-channel rates. A pure sine
    /// of frequency ``f`` gives about ``2 * f``.
    ///
    /// Note: unlike ``librosa.feature.zero_crossing_rate``, which returns a
    /// per-frame fraction of crossings per sample, this is one global value in
    /// crossings per second. Divide by ``sample_rate`` for crossings per sample.
    ///
    /// Returns:
    ///
    ///     float: Zero crossings **per second** (crossings / duration),
    ///         averaged over channels. ``librosa.feature.zero_crossing_rate`` is
    ///         instead a per-frame fraction of samples.
    fn zero_crossing_rate(&self, py: Python<'_>) -> PyResult<f64> {
        Ok({ dispatch_with_view!(self, py, |audio| audio.zero_crossing_rate()) })
    }

    #[pyo3(signature = (max_lag))]
    /// Compute the autocorrelation sequence up to a lag limit.
    ///
    /// Uses the **unbiased** estimate
    /// ``r[k] = (1 / (N - k)) * sum_{n=0}^{N-1-k} x[n] * x[n + k]`` for
    /// lags ``0..=min(max_lag, N - 1)``, first channel only. It is not
    /// normalised: ``r[0]`` is the mean square, not 1. Divide by ``r[0]`` for a
    /// normalised sequence; multiply ``r[k]`` by ``(N - k) / N`` for the biased
    /// estimate. ``r[k] * (N - k)`` equals
    /// ``librosa.autocorrelate(x, max_size=max_lag + 1)[k]``.
    ///
    /// Args:
    ///
    ///     max_lag (int): Maximum lag to include; must be greater than zero.
    ///
    /// Returns:
    ///
    ///     Optional[list[float]]: ``max_lag + 1`` values (fewer for short
    ///         signals), or None if the FFT fails.
    ///
    /// Raises:
    ///
    ///     ValueError: If max_lag is zero.
    fn autocorrelation(&self, py: Python<'_>, max_lag: usize) -> PyResult<Option<Vec<f64>>> {
        let max_lag = nzu_or_err(max_lag)?;
        dispatch_with_view!(self, py, |audio| Ok(audio
            .autocorrelation(max_lag)
            .map(|nev| nev.to_vec())))
    }

    #[pyo3(signature = (other, max_lag))]
    /// Compute the cross-correlation with another audio buffer.
    ///
    /// Uses the same unbiased, unnormalised estimate as :meth:`autocorrelation`:
    /// ``r[k] = (1 / (N - k)) * sum_n self[n] * other[n + k]`` for
    /// ``k = 0 .. max_lag``.
    ///
    /// Args:
    ///
    ///     other (AudioSamples): Audio buffer to correlate against.
    ///     max_lag (int): Maximum lag to include; must be greater than zero.
    ///
    /// Returns:
    ///
    ///     list[float]: Cross-correlation coefficients up to the requested lag.
    ///
    /// Raises:
    ///
    ///     ValueError: If max_lag is zero.
    ///     AudioParameterError: If the sample rates or channel counts differ
    ///         (the message names both values).
    ///     TypeError: If the buffers have different data types.
    fn cross_correlation(
        &self,
        py: Python<'_>,
        other: &Self,
        max_lag: usize,
    ) -> PyResult<Vec<f64>> {
        let max_lag = nzu_or_err(max_lag)?;
        crate::check_rates_match(
            "cross_correlation",
            self.sample_rate(py).get(),
            other.sample_rate(py).get(),
        )?;
        let (self_ch, other_ch) = (self.num_channels(py), other.num_channels(py));
        if self_ch != other_ch {
            return Err(crate::make_audio_err(
                "AudioParameterError",
                format!(
                    "cross_correlation requires both signals to have the same number of \
channels, got {self_ch} and {other_ch}; use to_mono() or extract_channel() first"
                ),
                None,
                None,
                pyo3::exceptions::PyValueError::new_err,
            ));
        }
        fn cross_correlation_dual<T>(
            a: &TypedAudioSamples<T>,
            b: &TypedAudioSamples<T>,
            py: Python<'_>,
            max_lag: usize,
        ) -> PyResult<Vec<f64>>
        where
            T: StandardSample + Element,
        {
            let max_lag = nzu_or_err(max_lag)?;

            // Same-kind backings are viewed in place; others are copied below.
            use PyAudioBacking::{NumpyMono, NumpyMulti, OwnedMono, OwnedMulti};
            let result = match (&a.backing, &b.backing) {
                (OwnedMono(_), OwnedMono(_))
                | (OwnedMulti(_), OwnedMulti(_))
                | (NumpyMono(_), NumpyMono(_))
                | (NumpyMulti(_), NumpyMulti(_)) => {
                    a.borrowed(py).cross_correlation(&b.borrowed(py), max_lag)
                }
                _ => {
                    // Mixed backings (e.g. one owned, one NumPy-backed) or
                    // interleaved NumPy arrays: materialise both as owned
                    // buffers and correlate those.
                    let _guards = (a.read_guard(py)?, b.read_guard(py)?);
                    let owned_a = a.with_view(py, |audio| audio.into_owned());
                    let owned_b = b.with_view(py, |audio| audio.into_owned());
                    owned_a.cross_correlation(&owned_b, max_lag)
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

    #[pyo3(signature = (reduction: "ChannelReduction" = ChannelReduction::Average.into()))]
    /// Compute the spectral centroid using the current FFT backend.
    ///
    /// Args:
    ///
    ///     reduction (ChannelReduction): Channel-reduction policy applied to
    ///         multi-channel input. Defaults to ``ChannelReduction.average``.
    ///
    /// Returns:
    ///
    ///     float: Centroid in Hz of the **power** spectrum ``|X|**2`` of the whole
    ///         signal (a single FFT, not frame-wise). Returns 0.0 for silence.
    ///
    /// Raises:
    ///
    ///     AudioError: If the FFT computation fails or the reduction is invalid
    ///         for the channel layout.
    fn spectral_centroid(&self, py: Python<'_>, reduction: PyChannelReduction) -> PyResult<f64> {
        let reduction = ChannelReduction::from(reduction);
        dispatch_with_view!(self, py, |audio| audio
            .spectral_centroid(reduction)
            .map_err(audio_err_to_py))
    }

    #[pyo3(signature = (rolloff_percent: "float" = 0.85, reduction: "ChannelReduction" = ChannelReduction::Average.into()))]
    /// Compute the spectral rolloff point for the provided energy fraction.
    ///
    /// Args:
    ///
    ///     rolloff_percent (float): Target cumulative spectral energy fraction between 0 and 1.
    ///     reduction (ChannelReduction): Channel-reduction policy applied to
    ///         multi-channel input. Defaults to ``ChannelReduction.average``.
    ///
    /// Returns:
    ///
    ///     float: Frequency where the cumulative spectrum reaches the rolloff threshold.
    ///
    /// Raises:
    ///
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

    #[pyo3(signature = (reduction: "ChannelReduction" = ChannelReduction::Average.into()))]
    /// Compute the spectral bandwidth (spectral spread) of the signal.
    ///
    /// The magnitude-weighted standard deviation of the spectrum about its
    /// spectral centroid. Larger values indicate energy spread across a wider
    /// frequency range; a pure tone yields a value near zero.
    ///
    /// Args:
    ///
    ///     reduction (ChannelReduction): Channel-reduction policy applied to
    ///         multi-channel input. Defaults to ``ChannelReduction.average``.
    ///
    /// Returns:
    ///
    ///     float: Spectral bandwidth in Hz. Returns 0.0 for silence.
    ///
    /// Raises:
    ///
    ///     AudioError: If the FFT computation fails or the reduction is invalid
    ///         for the channel layout.
    fn spectral_bandwidth(&self, py: Python<'_>, reduction: PyChannelReduction) -> PyResult<f64> {
        let reduction = ChannelReduction::from(reduction);
        dispatch_with_view!(self, py, |audio| audio
            .spectral_bandwidth(reduction)
            .map_err(audio_err_to_py))
    }

    #[pyo3(signature = (reduction: "ChannelReduction" = ChannelReduction::Average.into()))]
    /// Compute the spectral flatness (Wiener entropy) of the signal.
    ///
    /// The ratio of the geometric mean to the arithmetic mean of the power
    /// spectrum. Values near 1 indicate noise-like (flat) spectra, while values
    /// near 0 indicate tonal spectra dominated by a few peaks.
    ///
    /// Args:
    ///
    ///     reduction (ChannelReduction): Channel-reduction policy applied to
    ///         multi-channel input. Defaults to ``ChannelReduction.average``.
    ///
    /// Returns:
    ///
    ///     float: Spectral flatness in [0, 1]. Returns 0.0 for silence.
    ///
    /// Raises:
    ///
    ///     AudioError: If the FFT computation fails or the reduction is invalid
    ///         for the channel layout.
    fn spectral_flatness(&self, py: Python<'_>, reduction: PyChannelReduction) -> PyResult<f64> {
        let reduction = ChannelReduction::from(reduction);
        dispatch_with_view!(self, py, |audio| audio
            .spectral_flatness(reduction)
            .map_err(audio_err_to_py))
    }

    #[pyo3(signature = (reduction: "ChannelReduction" = ChannelReduction::Average.into()))]
    /// Compute the spectral crest factor of the signal.
    ///
    /// The ratio of the peak magnitude to the mean magnitude of the spectrum.
    /// High values indicate a strongly peaked (tonal) spectrum; a flat spectrum
    /// approaches 1.
    ///
    /// Args:
    ///
    ///     reduction (ChannelReduction): Channel-reduction policy applied to
    ///         multi-channel input. Defaults to ``ChannelReduction.average``.
    ///
    /// Returns:
    ///
    ///     float: Spectral crest factor (>= 1 for non-silent signals). Returns
    ///         0.0 for silence.
    ///
    /// Raises:
    ///
    ///     AudioError: If the FFT computation fails or the reduction is invalid
    ///         for the channel layout.
    fn spectral_crest(&self, py: Python<'_>, reduction: PyChannelReduction) -> PyResult<f64> {
        let reduction = ChannelReduction::from(reduction);
        dispatch_with_view!(self, py, |audio| audio
            .spectral_crest(reduction)
            .map_err(audio_err_to_py))
    }

    #[pyo3(signature = (reduction: "ChannelReduction" = ChannelReduction::Average.into()))]
    /// Compute the spectral slope of the signal.
    ///
    /// The slope of an ordinary least-squares linear fit of the linear magnitude
    /// spectrum against frequency (Hz), in units of magnitude per Hz. A negative
    /// slope indicates energy concentrated at low frequencies.
    ///
    /// Args:
    ///
    ///     reduction (ChannelReduction): Channel-reduction policy applied to
    ///         multi-channel input. Defaults to ``ChannelReduction.average``.
    ///
    /// Returns:
    ///
    ///     float: Least-squares slope (magnitude per Hz). Returns 0.0 for
    ///         silence or a degenerate single-bin spectrum.
    ///
    /// Raises:
    ///
    ///     AudioError: If the FFT computation fails or the reduction is invalid
    ///         for the channel layout.
    fn spectral_slope(&self, py: Python<'_>, reduction: PyChannelReduction) -> PyResult<f64> {
        let reduction = ChannelReduction::from(reduction);
        dispatch_with_view!(self, py, |audio| audio
            .spectral_slope(reduction)
            .map_err(audio_err_to_py))
    }

    #[pyo3(signature = (n_bands: "int", reduction: "ChannelReduction" = ChannelReduction::Average.into()))]
    /// Compute spectral contrast across octave-spaced sub-bands.
    ///
    /// The spectrum is partitioned into ``n_bands`` octave-spaced sub-bands.
    /// Within each band the contrast is the dB difference between the mean of the
    /// top quantile (peaks) and the mean of the bottom quantile (valleys). High
    /// contrast indicates clear tonal/harmonic structure.
    ///
    /// Args:
    ///
    ///     n_bands (int): Number of octave-spaced sub-bands; must be greater than zero.
    ///     reduction (ChannelReduction): Channel-reduction policy applied to
    ///         multi-channel input. Defaults to ``ChannelReduction.average``.
    ///
    /// Returns:
    ///
    ///     list[float]: ``n_bands`` contrast values in dB, low band to high band.
    ///
    /// Raises:
    ///
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

    #[pyo3(signature = ())]
    /// Return the value at the temporal midpoint of a mono signal.
    ///
    /// This is the 2.0 name for what was historically exposed as ``median``.
    /// For even-length signals the result is the average of the two central
    /// samples; for odd-length signals the single central sample is returned.
    /// Samples are selected by index position; the buffer is not sorted.
    ///
    /// Returns:
    ///
    ///     Optional[float]: Midpoint value for mono audio, or None if the signal
    ///         is multi-channel.
    fn midpoint_sample(&self, py: Python<'_>) -> PyResult<Option<f64>> {
        Ok({ dispatch_with_view!(self, py, |audio| audio.midpoint_sample()) })
    }

    #[pyo3(signature = ())]
    /// Compute the RMS and peak absolute value in a single pass.
    ///
    /// Equivalent to calling ``rms`` and ``peak`` separately, but reads the
    /// sample buffer only once.
    ///
    /// Returns:
    ///
    ///     tuple[float, int | float]: A ``(rms, peak)`` pair where ``rms`` is the
    ///         root-mean-square as a float and ``peak`` is the maximum absolute
    ///         sample value in the native sample type.
    fn rms_and_peak<'py>(&self, py: Python<'py>) -> PyResult<(f64, Bound<'py, PyAny>)> {
        Ok({
            dispatch_with_view!(self, py, |audio| {
                let (rms, peak) = audio.rms_and_peak();
                (
                    rms,
                    peak.into_bound_py_any(py)
                        .expect("Numbers should always be able to be converted to a pyobject."),
                )
            })
        })
    }

    #[pyo3(signature = ())]
    /// Return the peak (maximum absolute value) across all samples and channels.
    ///
    /// Alias for ``peak``, provided to match the conventional term "amplitude"
    /// used in some audio contexts.
    ///
    /// Returns:
    ///
    ///     int | float: Maximum absolute amplitude present in the buffer.
    fn amplitude<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        Ok({
            dispatch_with_view!(self, py, |audio| {
                audio
                    .amplitude()
                    .into_bound_py_any(py)
                    .expect("Numbers should always be able to be converted to a pyobject.")
            })
        })
    }
}
