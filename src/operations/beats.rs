use audio_samples::operations::AudioBeatTracking;
use pyo3::{PyResult, Python, pymethods};

use crate::{
    PyAudioSamples, audio_err_to_py, dispatch_with_view,
    types::{PyBeatTrackingConfig, PyBeatTrackingData},
};

#[pymethods]
impl PyAudioSamples {
    /// Detect beat locations using onset-driven dynamic programming.
    ///
    /// Args:
    ///     config (BeatTrackingConfig): Parameters controlling tempo targets and onset detection.
    ///
    /// Returns:
    ///     BeatTrackingData: Result object containing estimated tempo, beat times, and metadata.
    ///
    /// Raises:
    ///     AudioError: If beat tracking fails.
    #[pyo3(signature = (config: "BeatTrackingConfig"), text_signature = "($self, config: BeatTrackingConfig) -> BeatTrackingData")]
    fn detect_beats(
        &self,
        py: Python<'_>,
        config: PyBeatTrackingConfig,
    ) -> PyResult<PyBeatTrackingData> {
        dispatch_with_view!(self, py, |audio| {
            let result = audio.detect_beats(&config.inner).map_err(audio_err_to_py)?;
            Ok(PyBeatTrackingData { inner: result })
        })
    }

    /// Estimate the global tempo of the signal in beats per minute.
    ///
    /// This runs the same onset and tempo analysis as :meth:`detect_beats` but
    /// returns only the estimated tempo, skipping the per-beat tracking step.
    ///
    /// Args:
    ///     config (BeatTrackingConfig): Parameters controlling tempo targets and
    ///         onset detection.
    ///
    /// Returns:
    ///     float: Estimated tempo in beats per minute.
    ///
    /// Raises:
    ///     AudioError: If tempo estimation fails.
    #[pyo3(signature = (config: "BeatTrackingConfig"), text_signature = "($self, config: BeatTrackingConfig) -> float")]
    fn estimate_tempo(&self, py: Python<'_>, config: PyBeatTrackingConfig) -> PyResult<f64> {
        dispatch_with_view!(self, py, |audio| {
            audio
                .estimate_tempo(&config.inner)
                .map_err(audio_err_to_py)
        })
    }
}
