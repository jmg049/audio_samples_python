use audio_samples::operations::AudioBeatTracking;
use pyo3::{Bound, PyResult, Python, pymethods};

use crate::{
    PyAudioSamples, audio_err_to_py, dispatch_with_view,
    types::{PyBeatTrackingConfig, PyBeatTrackingData},
};

#[pymethods]
impl PyAudioSamples {
    /// Place beats on a fixed-tempo grid anchored at the strongest onset.
    ///
    /// Computes an onset strength envelope, then walks forward and backward
    /// from the global onset peak in steps of one inter-beat interval at
    /// ``config.tempo_bpm`` (snapping to nearby onsets within the configured
    /// tolerance). The tempo is *not* estimated: the returned ``tempo_bpm`` echoes
    /// the configuration. This is not librosa's dynamic-programming
    /// ``beat_track``.
    ///
    /// Args:
    ///
    ///     config (BeatTrackingConfig): Parameters controlling tempo targets and onset detection.
    ///
    /// Returns:
    ///
    ///     BeatTrackingData: Result object containing estimated tempo, beat times, and metadata.
    ///
    /// Raises:
    ///
    ///     AudioError: If beat tracking fails.
    #[pyo3(signature = (config: "BeatTrackingConfig"))]
    fn detect_beats(
        &self,
        py: Python<'_>,
        config: &Bound<'_, PyBeatTrackingConfig>,
    ) -> PyResult<PyBeatTrackingData> {
        let config = config.borrow().clone();
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
    ///
    ///     config (BeatTrackingConfig): Parameters controlling tempo targets and
    ///         onset detection.
    ///
    /// Returns:
    ///
    ///     float: Estimated tempo in beats per minute.
    ///
    /// Raises:
    ///
    ///     AudioError: If tempo estimation fails.
    #[pyo3(signature = (config: "BeatTrackingConfig"))]
    fn estimate_tempo(
        &self,
        py: Python<'_>,
        config: &Bound<'_, PyBeatTrackingConfig>,
    ) -> PyResult<f64> {
        let config = config.borrow().clone();
        dispatch_with_view!(self, py, |audio| {
            audio.estimate_tempo(&config.inner).map_err(audio_err_to_py)
        })
    }
}
