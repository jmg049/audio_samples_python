use audio_samples::operations::AudioEnvelopes;
use pyo3::prelude::*;

use crate::{
    PyAudioSamples, dispatch_with_view, ndresult_to_numpy, nzu_or_err,
    types::{PyDynamicRangeMethod, PyEnvelopeFollower},
};

#[pymethods]
impl PyAudioSamples {
    /// Compute a rectified amplitude envelope.
    ///
    /// Returns:
    ///
    ///     numpy.ndarray: Envelope array shaped like the input (channels by samples).
    ///
    /// Raises:
    ///
    ///     AudioError: If envelope computation fails.
    #[pyo3(signature = ())]
    fn amplitude_envelope<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        dispatch_with_view!(self, py, |audio| {
            let result = audio.amplitude_envelope();
            Ok(ndresult_to_numpy(py, result))
        })
    }

    /// Compute a root-mean-square envelope over sliding windows.
    ///
    /// Args:
    ///
    ///     window_size (int): Analysis window size in samples; must be greater than zero.
    ///     hop_size (int): Step between windows in samples; must be greater than zero.
    ///
    /// Returns:
    ///
    ///     numpy.ndarray: RMS envelope shaped (channels, frames).
    ///
    /// Raises:
    ///
    ///     ValueError: If window_size or hop_size is zero.
    ///     AudioError: If envelope computation fails.
    #[pyo3(signature = (window_size: "int", hop_size: "int"))]
    fn rms_envelope<'py>(
        &self,
        py: Python<'py>,
        window_size: usize,
        hop_size: usize,
    ) -> PyResult<Bound<'py, PyAny>> {
        dispatch_with_view!(self, py, |audio| {
            let window_size_nz = nzu_or_err(window_size)?;
            let hop_size_nz = nzu_or_err(hop_size)?;
            let result = audio.rms_envelope(window_size_nz, hop_size_nz);
            Ok(ndresult_to_numpy(py, result))
        })
    }

    /// Compute attack and decay envelopes using an envelope follower.
    ///
    /// Args:
    ///
    ///     follower (EnvelopeFollower): Configuration describing attack and release behavior.
    ///     method (DynamicRangeMethod): Detection strategy applied to the follower.
    ///
    /// Returns:
    ///
    ///     tuple[numpy.ndarray, numpy.ndarray]: Attack and decay envelopes shaped like the input.
    ///
    /// Raises:
    ///
    ///     AudioError: If envelope computation fails.
    #[pyo3(signature = (follower: "EnvelopeFollower", method: "DynamicRangeMethod"))]
    fn attack_decay_envelope<'py>(
        &self,
        py: Python<'py>,
        follower: &Bound<'_, PyEnvelopeFollower>,
        method: PyDynamicRangeMethod,
    ) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)> {
        let follower = follower.borrow().clone();
        dispatch_with_view!(self, py, |audio| {
            let (attack, decay) = audio.attack_decay_envelope(&follower.inner, method.inner);
            Ok((ndresult_to_numpy(py, attack), ndresult_to_numpy(py, decay)))
        })
    }

    /// Compute the analytic envelope via the Hilbert transform.
    ///
    /// Returns:
    ///
    ///     numpy.ndarray: Instantaneous amplitude shaped like the input.
    ///
    /// Raises:
    ///
    ///     AudioError: If envelope computation fails.
    #[pyo3(signature = ())]
    fn analytic_envelope<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        dispatch_with_view!(self, py, |audio| {
            let result = audio.analytic_envelope();
            Ok(ndresult_to_numpy(py, result))
        })
    }

    /// Compute a moving-average envelope over the rectified signal.
    ///
    /// Args:
    ///
    ///     window_size (int): Number of samples per averaging window; must be greater than zero.
    ///     hop_size (int): Step between windows in samples; must be greater than zero.
    ///
    /// Returns:
    ///
    ///     numpy.ndarray: Moving-average envelope shaped (channels, frames).
    ///
    /// Raises:
    ///
    ///     ValueError: If window_size or hop_size is zero.
    ///     AudioError: If envelope computation fails.
    #[pyo3(signature = (window_size: "int", hop_size: "int"))]
    fn moving_average_envelope<'py>(
        &self,
        py: Python<'py>,
        window_size: usize,
        hop_size: usize,
    ) -> PyResult<Bound<'py, PyAny>> {
        dispatch_with_view!(self, py, |audio| {
            let window_size_nz = nzu_or_err(window_size)?;
            let hop_size_nz = nzu_or_err(hop_size)?;
            let result = audio.moving_average_envelope(window_size_nz, hop_size_nz);
            Ok(ndresult_to_numpy(py, result))
        })
    }
}
