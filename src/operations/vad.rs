use audio_samples::AudioVoiceActivityDetection;
use pyo3::{PyResult, Python, pymethods};

use crate::{PyAudioSamples, audio_err_to_py, dispatch_with_view, types::PyVadConfig};

#[pymethods]
impl PyAudioSamples {
    /// Compute a per-frame voice activity mask.
    ///
    /// Args:
    ///     config (VadConfig): Parameters controlling the voice activity detector.
    ///
    /// Returns:
    ///     list[bool]: Boolean mask where True marks frames with speech activity.
    ///
    /// Raises:
    ///     AudioError: If mask computation fails.
    #[pyo3(signature = (config: "VadConfig"), text_signature = "($self, config: VadConfig) -> list[bool]")]
    fn voice_activity_mask(&self, py: Python<'_>, config: PyVadConfig) -> PyResult<Vec<bool>> {
        dispatch_with_view!(self, py, |audio| {
            let result = audio
                .voice_activity_mask(&config.inner)
                .map_err(audio_err_to_py)?;
            Ok(result.into_vec())
        })
    }

    /// Compute contiguous speech regions as (start_sample, end_sample) pairs.
    ///
    /// Args:
    ///     config (VadConfig): Parameters controlling the voice activity detector.
    ///
    /// Returns:
    ///     list[tuple[int, int]]: Regions marked as speech with sample-based start and exclusive end.
    ///
    /// Raises:
    ///     AudioError: If region extraction fails.
    #[pyo3(signature = (config: "VadConfig"), text_signature = "($self, config: VadConfig) -> list[tuple[int, int]]")]
    fn speech_regions(&self, py: Python<'_>, config: PyVadConfig) -> PyResult<Vec<(usize, usize)>> {
        dispatch_with_view!(self, py, |audio| {
            let result = audio
                .speech_regions(&config.inner)
                .map_err(audio_err_to_py)?;
            Ok(result)
        })
    }
}
