use audio_samples::AudioDynamicRange;
use pyo3::{PyResult, Python, pymethods};

use crate::{
    PyAudioSamples, audio_err_to_py, dispatch_with_view_mut,
    types::{PyCompressorConfig, PyLimiterConfig},
};

#[pymethods]
impl PyAudioSamples {
    /// Apply a compressor configuration in place.
    ///
    /// Args:
    ///     compressor_config (CompressorConfig): Parameters controlling threshold, ratio, and timing.
    ///
    /// Returns:
    ///     None: Operation mutates the current buffer.
    ///
    /// Raises:
    ///     AudioError: If compression fails.
    #[pyo3(signature = (compressor_config: "CompressorConfig"), text_signature = "($self, compressor_config: CompressorConfig) -> None")]
    fn apply_compressor(
        &mut self,
        py: Python<'_>,
        compressor_config: PyCompressorConfig,
    ) -> PyResult<()> {
        dispatch_with_view_mut!(self, py, |mut audio| {
            audio
                .apply_compressor(&compressor_config.inner)
                .map_err(audio_err_to_py)
        })
    }

    /// Apply a limiter configuration in place.
    ///
    /// Args:
    ///     config (LimiterConfig): Parameters controlling ceiling, attack, and release.
    ///
    /// Returns:
    ///     None: Operation mutates the current buffer.
    ///
    /// Raises:
    ///     AudioError: If limiting fails.
    #[pyo3(signature = (config: "LimiterConfig"), text_signature = "($self, config: LimiterConfig) -> None")]
    fn apply_limiter(&mut self, py: Python<'_>, config: PyLimiterConfig) -> PyResult<()> {
        dispatch_with_view_mut!(self, py, |mut audio| {
            audio.apply_limiter(&config.into()).map_err(audio_err_to_py)
        })
    }

    /// Apply a downward noise gate in place.
    ///
    /// Args:
    ///     threshold_db (float): Level in decibels below which gating engages.
    ///     ratio (float): Gain reduction ratio applied to gated signals.
    ///     attack_ms (float): Attack time in milliseconds.
    ///     release_ms (float): Release time in milliseconds.
    ///
    /// Returns:
    ///     None: Operation mutates the current buffer.
    ///
    /// Raises:
    ///     AudioError: If gating fails.
    #[pyo3(signature = (threshold_db: "float", ratio: "float", attack_ms: "float", release_ms: "float"), text_signature = "($self, threshold_db: float, ratio: float, attack_ms: float, release_ms: float) -> None")]
    fn apply_gate(
        &mut self,
        py: Python<'_>,
        threshold_db: f64,
        ratio: f64,
        attack_ms: f64,
        release_ms: f64,
    ) -> PyResult<()> {
        dispatch_with_view_mut!(self, py, |mut audio| {
            audio
                .apply_gate(threshold_db, ratio, attack_ms, release_ms)
                .map_err(audio_err_to_py)
        })
    }

    /// Apply a downward expander in place.
    ///
    /// Args:
    ///     threshold_db (float): Level in decibels below which expansion engages.
    ///     ratio (float): Expansion ratio applied to signals below the threshold.
    ///     attack_ms (float): Attack time in milliseconds.
    ///     release_ms (float): Release time in milliseconds.
    ///
    /// Returns:
    ///     None: Operation mutates the current buffer.
    ///
    /// Raises:
    ///     AudioError: If expansion fails.
    #[pyo3(signature = (threshold_db: "float", ratio: "float", attack_ms: "float", release_ms: "float"), text_signature = "($self, threshold_db: float, ratio: float, attack_ms: float, release_ms: float) -> None")]
    fn apply_expander(
        &mut self,
        py: Python<'_>,
        threshold_db: f64,
        ratio: f64,
        attack_ms: f64,
        release_ms: f64,
    ) -> PyResult<()> {
        dispatch_with_view_mut!(self, py, |mut audio| {
            audio
                .apply_expander(threshold_db, ratio, attack_ms, release_ms)
                .map_err(audio_err_to_py)
        })
    }
}
