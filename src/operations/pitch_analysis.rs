use crate::{
    PyAudioSamples, audio_err_to_py, dispatch_with_view, nzu_or_err, types::PyPitchDetectionMethod,
};
use audio_samples::AudioPitchAnalysis;
use pyo3::prelude::*;
use spectrograms::python::{PyStftParams, PyWindowType};

#[pymethods]
impl PyAudioSamples {
    /// Detect the fundamental frequency using the YIN algorithm.
    ///
    /// Args:
    ///     threshold (float): Detection threshold; lower values make detection more permissive.
    ///     min_frequency (float): Minimum expected frequency in hertz.
    ///     max_frequency (float): Maximum expected frequency in hertz.
    ///
    /// Returns:
    ///     float | None: Detected pitch in hertz, or None when no stable pitch is found.
    ///
    /// Raises:
    ///     AudioError: If pitch detection fails.
    #[pyo3(signature = (threshold: "float" = 0.1, min_frequency: "float" = 50.0, max_frequency:"float" = 2000.0), text_signature = "($self, threshold: float = 0.1, min_frequency: float = 50.0, max_frequency: float = 2000.0) -> Optional[float]")]
    fn detect_pitch_yin(
        &self,
        py: Python<'_>,
        threshold: f64,
        min_frequency: f64,
        max_frequency: f64,
    ) -> PyResult<Option<f64>> {
        dispatch_with_view!(self, py, |audio| {
            audio
                .detect_pitch_yin(threshold, min_frequency, max_frequency)
                .map_err(audio_err_to_py)
        })
    }

    /// Detect the fundamental frequency using autocorrelation.
    ///
    /// Args:
    ///     min_frequency (float): Minimum expected frequency in hertz.
    ///     max_frequency (float): Maximum expected frequency in hertz.
    ///
    /// Returns:
    ///     float | None: Detected pitch in hertz, or None when no stable pitch is found.
    ///
    /// Raises:
    ///     AudioError: If pitch detection fails.
    #[inline]
    pub fn detect_pitch_autocorr(
        &self,
        py: Python<'_>,
        min_frequency: f64,
        max_frequency: f64,
    ) -> PyResult<Option<f64>> {
        dispatch_with_view!(self, py, |audio| {
            audio
                .detect_pitch_autocorr(min_frequency, max_frequency)
                .map_err(audio_err_to_py)
        })
    }

    /// Track the detected pitch over successive analysis windows.
    ///
    /// Args:
    ///     window_size (int): Size of each analysis window in samples; must be greater than zero.
    ///     hop_size (int): Step between windows in samples; must be greater than zero.
    ///     pitch_method (PitchDetectionMethod): Detection strategy to apply per window.
    ///     threshold (float): Detection threshold interpreted by the selected method.
    ///     min_frequency (float): Minimum expected frequency in hertz.
    ///     max_frequency (float): Maximum expected frequency in hertz.
    ///
    /// Returns:
    ///     list[tuple[float, Optional[float]]]: Sequence of (time_seconds, pitch_hz) pairs.
    ///
    /// Raises:
    ///     ValueError: If window_size or hop_size is zero.
    ///     AudioError: If pitch tracking fails.
    #[pyo3(signature = (window_size: "int", hop_size: "int", pitch_method: "PitchDetectionMethod" = PyPitchDetectionMethod::detect_yin(), threshold: "float" = 0.1, min_frequency: "float" = 50.0, max_frequency: "float" = 2000.0), text_signature = "($self, window_size: int, hop_size: int, pitch_method=PitchDetection.yin, threshold: float = 0.1, min_frequency: float = 50.0, max_frequency: float = 2000.0) -> list[tuple[float, Optional[float]]]")]
    fn track_pitch(
        &self,
        py: Python<'_>,
        window_size: usize,
        hop_size: usize,
        pitch_method: PyPitchDetectionMethod,
        threshold: f64,
        min_frequency: f64,
        max_frequency: f64,
    ) -> PyResult<Vec<(f64, Option<f64>)>> {
        let window_size = nzu_or_err(window_size)?;
        let hop_size = nzu_or_err(hop_size)?;
        dispatch_with_view!(self, py, |audio| {
            audio
                .track_pitch(
                    window_size,
                    hop_size,
                    *pitch_method,
                    threshold,
                    min_frequency,
                    max_frequency,
                )
                .map_err(audio_err_to_py)
        })
    }

    /// Estimate the harmonic-to-noise ratio around a fundamental frequency.
    ///
    /// Args:
    ///     fundamental_freq (float): Fundamental frequency in hertz.
    ///     num_harmonics (int): Number of harmonics to evaluate; must be greater than zero.
    ///     n_fft (Optional[int]): Optional FFT size in samples; must be greater than zero when provided.
    ///     window_type (Optional[WindowType]): Optional window type applied before the FFT.
    ///
    /// Returns:
    ///     float: Harmonic-to-noise ratio value.
    ///
    /// Raises:
    ///     ValueError: If num_harmonics or n_fft is zero.
    ///     AudioError: If the analysis fails.
    pub fn harmonic_to_noise_ratio(
        &self,
        py: Python<'_>,
        fundamental_freq: f64,
        num_harmonics: usize,
        n_fft: Option<usize>,
        window_type: Option<PyWindowType>,
    ) -> PyResult<f64> {
        let num_harmonics = nzu_or_err(num_harmonics)?;
        let n_fft = n_fft.map(nzu_or_err).transpose()?;
        dispatch_with_view!(self, py, |audio| {
            audio
                .harmonic_to_noise_ratio(
                    fundamental_freq,
                    num_harmonics,
                    n_fft,
                    window_type.map(|w| w.into_inner()),
                )
                .map_err(audio_err_to_py)
        })
    }

    /// Measure harmonic amplitudes relative to a fundamental frequency.
    ///
    /// Args:
    ///     fundamental_freq (float): Fundamental frequency in hertz.
    ///     num_harmonics (int): Number of harmonics to report; must be greater than zero.
    ///     tolerance (float): Frequency tolerance expressed as a fraction of the harmonic.
    ///     n_fft (Optional[int]): Optional FFT size in samples; must be greater than zero when provided.
    ///     window_type (Optional[WindowType]): Optional window type applied before the FFT.
    ///
    /// Returns:
    ///     list[float]: Harmonic magnitudes ordered by harmonic index.
    ///
    /// Raises:
    ///     ValueError: If num_harmonics or n_fft is zero.
    ///     AudioError: If the analysis fails.
    pub fn harmonic_analysis(
        &self,
        py: Python<'_>,
        fundamental_freq: f64,
        num_harmonics: usize,
        tolerance: f64,
        n_fft: Option<usize>,
        window_type: Option<PyWindowType>,
    ) -> PyResult<Vec<f64>> {
        let num_harmonics = nzu_or_err(num_harmonics)?;
        let n_fft = n_fft.map(nzu_or_err).transpose()?;
        dispatch_with_view!(self, py, |audio| {
            audio
                .harmonic_analysis(
                    fundamental_freq,
                    num_harmonics,
                    tolerance,
                    n_fft,
                    window_type.map(|w| w.into_inner()),
                )
                .map_err(audio_err_to_py)
        })
    }

    /// Estimate the musical key of the audio signal.
    ///
    /// Args:
    ///     stft_params (StftParams): Parameters controlling the underlying STFT analysis.
    ///
    /// Returns:
    ///     tuple[int, float]: Detected key index and confidence score.
    ///
    /// Raises:
    ///     AudioError: If key estimation fails.
    pub fn estimate_key(
        &self,
        py: Python<'_>,
        stft_params: &PyStftParams,
    ) -> PyResult<(usize, f64)> {
        let stft_params = &stft_params.inner;

        dispatch_with_view!(self, py, |audio| {
            audio.estimate_key(stft_params).map_err(audio_err_to_py)
        })
    }
}
