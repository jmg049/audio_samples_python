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
}
