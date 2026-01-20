use audio_samples::AudioOnsetDetection;
use numpy::{IntoPyArray, PyArray2};
use pyo3::prelude::*;
use spectrograms::python::PyCqtParams;

use crate::{
    PyAudioSamples, audio_err_to_py, dispatch_with_view, nzu_or_err,
    types::{
        PyComplexOnsetConfig, PyOnsetDetectionConfig, PySpectralFluxConfig, PySpectralFluxMethod,
    },
};

#[pymethods]
impl PyAudioSamples {
    /// Detect onset times using the configured algorithm.
    ///
    /// Args:
    ///     config (OnsetDetectionConfig): Parameters controlling peak picking and preprocessing.
    ///
    /// Returns:
    ///     list[float]: Onset times in seconds.
    ///
    /// Raises:
    ///     AudioError: If onset detection fails.
    #[pyo3(signature = (config: "OnsetDetectionConfig"), text_signature = "($self, config: OnsetDetectionConfig) -> list[float]")]
    fn detect_onsets(&self, py: Python<'_>, config: PyOnsetDetectionConfig) -> PyResult<Vec<f64>> {
        dispatch_with_view!(self, py, |audio| {
            audio.detect_onsets(&config.inner).map_err(audio_err_to_py)
        })
    }

    /// Compute the energy-based onset detection function.
    ///
    /// Args:
    ///     config (OnsetDetectionConfig): Parameters controlling the detection function.
    ///
    /// Returns:
    ///     tuple[list[float], list[float]]: Detection function values and matching timestamps.
    ///
    /// Raises:
    ///     AudioError: If the detection function computation fails.
    #[pyo3(signature = (config: "OnsetDetectionConfig"), text_signature = "($self, config: OnsetDetectionConfig) -> tuple[list[float], list[float]]")]
    fn onset_detection_function(
        &self,
        py: Python<'_>,
        config: PyOnsetDetectionConfig,
    ) -> PyResult<(Vec<f64>, Vec<f64>)> {
        dispatch_with_view!(self, py, |audio| {
            let (odf, times) = audio
                .onset_detection_function(&config.inner)
                .map_err(audio_err_to_py)?;
            Ok((odf.into_vec(), times.into_vec()))
        })
    }

    /// Detect onset times using spectral flux.
    ///
    /// Args:
    ///     config (SpectralFluxConfig): Parameters configuring the flux computation and picker.
    ///
    /// Returns:
    ///     list[float]: Onset times in seconds.
    ///
    /// Raises:
    ///     AudioError: If onset detection fails.
    #[pyo3(signature = (config: "SpectralFluxConfig"), text_signature = "($self, config: SpectralFluxConfig) -> list[float]")]
    fn detect_onsets_spectral_flux(
        &self,
        py: Python<'_>,
        config: PySpectralFluxConfig,
    ) -> PyResult<Vec<f64>> {
        dispatch_with_view!(self, py, |audio| {
            audio
                .detect_onsets_spectral_flux(&config.inner)
                .map_err(audio_err_to_py)
        })
    }

    /// Compute spectral flux values without peak picking.
    ///
    /// Args:
    ///     cqt_params (CqtParams): Parameters controlling the constant-Q transform.
    ///     window_size (int): Frame size in samples; must be greater than zero.
    ///     hop_size (int): Hop size in samples; must be greater than zero.
    ///     method (SpectralFluxMethod): Flux computation strategy.
    ///
    /// Returns:
    ///     tuple[list[float], list[float]]: Flux values and matching timestamps.
    ///
    /// Raises:
    ///     ValueError: If window_size or hop_size is zero.
    ///     AudioError: If flux computation fails.
    #[pyo3(signature = (cqt_params: "CqtParams", window_size: "int", hop_size: "int", method: "SpectralFluxMethod"), text_signature = "($self, cqt_params: CqtParams, window_size: int, hop_size: int, method: SpectralFluxMethod) -> tuple[list[float], list[float]]")]
    fn spectral_flux(
        &self,
        py: Python<'_>,
        cqt_params: PyCqtParams,
        window_size: usize,
        hop_size: usize,
        method: PySpectralFluxMethod,
    ) -> PyResult<(Vec<f64>, Vec<f64>)> {
        dispatch_with_view!(self, py, |audio| {
            let window_size_nz = nzu_or_err(window_size)?;
            let hop_size_nz = nzu_or_err(hop_size)?;
            let (flux, times) = audio
                .spectral_flux(
                    &cqt_params.into(),
                    window_size_nz,
                    hop_size_nz,
                    method.inner,
                )
                .map_err(audio_err_to_py)?;
            Ok((flux.into_vec(), times.into_vec()))
        })
    }

    /// Detect onset times using the complex-domain method.
    ///
    /// Args:
    ///     config (ComplexOnsetConfig): Parameters controlling magnitude and phase processing.
    ///
    /// Returns:
    ///     list[float]: Onset times in seconds.
    ///
    /// Raises:
    ///     AudioError: If onset detection fails.
    #[pyo3(signature = (config: "ComplexOnsetConfig"), text_signature = "($self, config: ComplexOnsetConfig) -> list[float]")]
    fn complex_onset_detection(
        &self,
        py: Python<'_>,
        config: PyComplexOnsetConfig,
    ) -> PyResult<Vec<f64>> {
        dispatch_with_view!(self, py, |audio| {
            audio
                .complex_onset_detection(&config.inner)
                .map_err(audio_err_to_py)
        })
    }

    /// Compute the complex-domain onset detection function without peak picking.
    ///
    /// Args:
    ///     config (ComplexOnsetConfig): Parameters controlling magnitude and phase processing.
    ///
    /// Returns:
    ///     list[float]: Complex onset detection function values.
    ///
    /// Raises:
    ///     AudioError: If detection function computation fails.
    #[pyo3(signature = (config: "ComplexOnsetConfig"), text_signature = "($self, config: ComplexOnsetConfig) -> list[float]")]
    fn onset_detection_function_complex(
        &self,
        py: Python<'_>,
        config: PyComplexOnsetConfig,
    ) -> PyResult<Vec<f64>> {
        dispatch_with_view!(self, py, |audio| {
            let odf = audio
                .onset_detection_function_complex(&config.inner)
                .map_err(audio_err_to_py)?;
            Ok(odf.into_vec())
        })
    }
    /// Compute the complex-domain magnitude difference matrix.
    ///
    /// Args:
    ///     config (ComplexOnsetConfig): Parameters controlling magnitude and phase processing.
    ///
    /// Returns:
    ///     numpy.ndarray: Magnitude difference matrix with axes (frequency_bins, frames).
    ///
    /// Raises:
    ///     AudioError: If matrix computation fails.
    #[pyo3(signature = (config: "ComplexOnsetConfig"), text_signature = "($self, config: ComplexOnsetConfig) -> numpy.ndarray")]
    fn magnitude_difference_matrix<'py>(
        &self,
        py: Python<'py>,
        config: PyComplexOnsetConfig,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        dispatch_with_view!(self, py, |audio| {
            let mdm = audio
                .magnitude_difference_matrix(&config.inner)
                .map_err(audio_err_to_py)?;
            Ok(mdm.into_pyarray(py))
        })
    }

    /// Compute the complex-domain phase deviation matrix.
    ///
    /// Args:
    ///     config (ComplexOnsetConfig): Parameters controlling magnitude and phase processing.
    ///
    /// Returns:
    ///     numpy.ndarray: Phase deviation matrix with axes (frequency_bins, frames).
    ///
    /// Raises:
    ///     AudioError: If matrix computation fails.
    #[pyo3(signature = (config: "ComplexOnsetConfig"), text_signature = "($self, config: ComplexOnsetConfig) -> numpy.ndarray")]
    fn phase_deviation_matrix<'py>(
        &self,
        py: Python<'py>,
        config: PyComplexOnsetConfig,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        dispatch_with_view!(self, py, |audio| {
            let pdm = audio
                .phase_deviation_matrix(&config.inner)
                .map_err(audio_err_to_py)?;
            Ok(pdm.into_pyarray(py))
        })
    }
}
