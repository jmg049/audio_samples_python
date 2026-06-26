//! Python bindings for audio analysis and detection utilities.
//!
//! Exposes the `audio_samples::utils::detection` functions to Python: sample-rate and
//! fundamental-frequency detection, silence/clipping region detection, dynamic-range and
//! noise-floor analysis, and lower-level spectral/autocorrelation helpers.

use crate::{PyAudioSamples, audio_err_to_py, dispatch_with_view, reexport};
use audio_samples::CastInto;
use audio_samples::utils::detection::{
    analyze_spectrum_for_cutoff, detect_clipping, detect_dynamic_range,
    detect_fundamental_autocorrelation, detect_fundamental_frequency, detect_sample_rate,
    detect_silence_regions, estimate_frequency_range, estimate_noise_floor,
};
use non_empty_slice::NonEmptyVec;
use numpy::{PyArray1, PyArrayMethods};
use pyo3::prelude::*;

#[pyfunction]
#[pyo3(name = "detect_sample_rate", signature=(audio), text_signature="(audio: AudioSamples) -> Optional[int]")]
/// Heuristically detect the original sample rate of a signal from its spectral content.
///
/// Analyses the power spectrum for sharp high-frequency cutoffs characteristic of
/// anti-aliasing filters and matches them against common sample rates. Only the first
/// channel is used. Returns ``None`` when no candidate rate can be identified.
///
/// Args:
///     audio: The audio signal to analyse.
///
/// Returns:
///     The detected original sample rate in Hz, or ``None``.
pub fn py_detect_sample_rate(py: Python<'_>, audio: &PyAudioSamples) -> PyResult<Option<u32>> {
    dispatch_with_view!(audio, py, |audio| {
        detect_sample_rate::<_>(&audio)
            .map_err(audio_err_to_py)
            .map(|rate| rate.map(|r| r.get()))
    })
}

#[pyfunction]
#[pyo3(name = "detect_fundamental_frequency", signature=(audio), text_signature="(audio: AudioSamples) -> Optional[float]")]
/// Estimate the fundamental frequency of a signal using autocorrelation.
///
/// Args:
///     audio: The audio signal to analyse.
///
/// Returns:
///     The estimated fundamental frequency in Hz, or ``None`` if no periodic component
///     is found.
pub fn py_detect_fundamental_frequency(
    py: Python<'_>,
    audio: &PyAudioSamples,
) -> PyResult<Option<f64>> {
    dispatch_with_view!(audio, py, |audio| {
        detect_fundamental_frequency::<_>(&audio).map_err(audio_err_to_py)
    })
}

#[pyfunction]
#[pyo3(name = "detect_silence_regions", signature=(audio, threshold), text_signature="(audio: AudioSamples, threshold: float) -> list[tuple[float, float]]")]
/// Detect time intervals where the signal amplitude falls below a threshold.
///
/// For mono signals each sample is checked directly. For multi-channel signals a position
/// is considered silent only when **all** channels are below the threshold.
///
/// Args:
///     audio: The audio signal to analyse.
///     threshold: Amplitude threshold in the signal's native sample scale (e.g. 0..1 for
///         float audio). Samples with absolute value below this are considered silent.
///
/// Returns:
///     A list of ``(start_time, end_time)`` tuples in seconds, one per silent region.
pub fn py_detect_silence_regions(
    py: Python<'_>,
    audio: &PyAudioSamples,
    threshold: f64,
) -> PyResult<Vec<(f64, f64)>> {
    dispatch_with_view!(audio, py, |audio| {
        detect_silence_regions(&audio, threshold.cast_into()).map_err(audio_err_to_py)
    })
}

#[pyfunction]
#[pyo3(name = "detect_clipping", signature=(audio, threshold_ratio=0.99), text_signature="(audio: AudioSamples, threshold_ratio: float = 0.99) -> list[tuple[float, float]]")]
/// Detect time intervals where the signal reaches or exceeds the full-scale value.
///
/// A sample is considered clipped when it reaches or exceeds ``threshold_ratio`` of the
/// sample type's positive full scale, or falls at or below ``threshold_ratio`` of its
/// negative full scale. For multi-channel signals a position is clipped when **any**
/// channel is clipped.
///
/// Args:
///     audio: The audio signal to analyse.
///     threshold_ratio: Fraction of full scale that constitutes clipping, in (0, 1]
///         (default: 0.99).
///
/// Returns:
///     A list of ``(start_time, end_time)`` tuples in seconds, one per clipped region.
pub fn py_detect_clipping(
    py: Python<'_>,
    audio: &PyAudioSamples,
    threshold_ratio: f64,
) -> PyResult<Vec<(f64, f64)>> {
    dispatch_with_view!(audio, py, |audio| {
        detect_clipping(&audio, threshold_ratio).map_err(audio_err_to_py)
    })
}

#[pyfunction]
#[pyo3(name = "detect_dynamic_range", signature=(audio), text_signature="(audio: AudioSamples) -> tuple[float, float, float]")]
/// Compute the dynamic-range characteristics of a signal.
///
/// All samples across all channels are considered together.
///
/// Args:
///     audio: The audio signal to analyse.
///
/// Returns:
///     A tuple ``(peak_amplitude, rms_amplitude, dynamic_range_db)`` where the dynamic
///     range is the crest factor ``20 * log10(peak / rms)`` in decibels (0.0 when rms is 0).
pub fn py_detect_dynamic_range(
    py: Python<'_>,
    audio: &PyAudioSamples,
) -> PyResult<(f64, f64, f64)> {
    dispatch_with_view!(audio, py, |audio| {
        detect_dynamic_range(&audio).map_err(audio_err_to_py)
    })
}

#[pyfunction]
#[pyo3(name = "estimate_noise_floor", signature=(audio), text_signature="(audio: AudioSamples) -> Optional[float]")]
/// Estimate the noise floor of a signal in dBFS.
///
/// Computes the noise floor from the quietest 10th percentile of sample magnitudes. Only
/// the first channel is used for multi-channel signals.
///
/// Args:
///     audio: The audio signal to analyse.
///
/// Returns:
///     The estimated noise floor in dBFS (always below 0), or ``None`` if it cannot be
///     estimated (e.g. the signal is too short or entirely silent).
pub fn py_estimate_noise_floor(
    py: Python<'_>,
    audio: &PyAudioSamples,
) -> PyResult<Option<f64>> {
    dispatch_with_view!(audio, py, |audio| {
        estimate_noise_floor::<_, ()>(&audio).map_err(audio_err_to_py)
    })
}

#[pyfunction]
#[pyo3(name = "estimate_frequency_range", signature=(audio), text_signature="(audio: AudioSamples) -> Optional[tuple[float, float]]")]
/// Estimate the active frequency range of a signal.
///
/// Computes the power spectrum of the first channel and returns the lowest and highest
/// frequencies carrying more than 1% of the peak spectral energy.
///
/// Args:
///     audio: The audio signal to analyse.
///
/// Returns:
///     A ``(low_hz, high_hz)`` tuple, or ``None`` when the signal is shorter than 1024
///     samples or no bin exceeds the threshold.
pub fn py_estimate_frequency_range(
    py: Python<'_>,
    audio: &PyAudioSamples,
) -> PyResult<Option<(f64, f64)>> {
    dispatch_with_view!(audio, py, |audio| {
        estimate_frequency_range::<_>(&audio).map_err(audio_err_to_py)
    })
}

#[pyfunction]
#[pyo3(name = "analyze_spectrum_for_cutoff", signature=(spectrum, nyquist_freq), text_signature="(spectrum: numpy.ndarray, nyquist_freq: float) -> Optional[int]")]
/// Analyse a power spectrum for a spectral cutoff indicating prior resampling.
///
/// Checks candidate Nyquist frequencies (derived from common sample rates) for a 2x or
/// greater energy drop, and returns the first (lowest-frequency) matching sample rate.
///
/// Args:
///     spectrum: A non-empty 1-D power spectrum (FFT magnitude-squared). Only the lower
///         half of the bins is examined.
///     nyquist_freq: The Nyquist frequency of the audio that produced ``spectrum`` in Hz.
///
/// Returns:
///     The first candidate sample rate in Hz with a significant energy drop, or ``None``.
///
/// Raises:
///     ValueError: If ``spectrum`` is empty or ``nyquist_freq`` is non-finite.
pub fn py_analyze_spectrum_for_cutoff(
    spectrum: &Bound<'_, PyArray1<f64>>,
    nyquist_freq: f64,
) -> PyResult<Option<u32>> {
    if !nyquist_freq.is_finite() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "nyquist_freq must be finite",
        ));
    }
    let data = spectrum.readonly().as_array().to_vec();
    let spectrum = NonEmptyVec::new(data)
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("spectrum must be non-empty"))?;
    Ok(analyze_spectrum_for_cutoff(&spectrum, nyquist_freq))
}

#[pyfunction]
#[pyo3(name = "detect_fundamental_autocorrelation", signature=(data, sample_rate), text_signature="(data: numpy.ndarray, sample_rate: float) -> Optional[float]")]
/// Estimate the fundamental frequency of a raw sample buffer using autocorrelation.
///
/// Searches candidate periods corresponding to fundamentals in the range 50..2000 Hz.
///
/// Args:
///     data: A non-empty 1-D array of mono f64 samples.
///     sample_rate: The sample rate in Hz (must be finite and positive).
///
/// Returns:
///     The estimated fundamental frequency in Hz, or ``None`` if no periodic component is
///     found or the signal is too short.
///
/// Raises:
///     ValueError: If ``data`` is empty or ``sample_rate`` is not finite and positive.
pub fn py_detect_fundamental_autocorrelation(
    data: &Bound<'_, PyArray1<f64>>,
    sample_rate: f64,
) -> PyResult<Option<f64>> {
    if !sample_rate.is_finite() || sample_rate <= 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "sample_rate must be finite and positive",
        ));
    }
    let buf = data.readonly().as_array().to_vec();
    let buf = NonEmptyVec::new(buf)
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("data must be non-empty"))?;
    detect_fundamental_autocorrelation(&buf, sample_rate).map_err(audio_err_to_py)
}

#[pymodule]
pub fn detection(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    let detection_mod = PyModule::new(py, "detection")?;

    detection_mod.add_function(wrap_pyfunction!(py_detect_sample_rate, &detection_mod)?)?;
    detection_mod.add_function(wrap_pyfunction!(
        py_detect_fundamental_frequency,
        &detection_mod
    )?)?;
    detection_mod.add_function(wrap_pyfunction!(py_detect_silence_regions, &detection_mod)?)?;
    detection_mod.add_function(wrap_pyfunction!(py_detect_clipping, &detection_mod)?)?;
    detection_mod.add_function(wrap_pyfunction!(py_detect_dynamic_range, &detection_mod)?)?;
    detection_mod.add_function(wrap_pyfunction!(py_estimate_noise_floor, &detection_mod)?)?;
    detection_mod.add_function(wrap_pyfunction!(
        py_estimate_frequency_range,
        &detection_mod
    )?)?;
    detection_mod.add_function(wrap_pyfunction!(
        py_analyze_spectrum_for_cutoff,
        &detection_mod
    )?)?;
    detection_mod.add_function(wrap_pyfunction!(
        py_detect_fundamental_autocorrelation,
        &detection_mod
    )?)?;

    reexport!(
        m,
        detection_mod,
        "detect_sample_rate",
        "detect_fundamental_frequency",
        "detect_silence_regions",
        "detect_clipping",
        "detect_dynamic_range",
        "estimate_noise_floor",
        "estimate_frequency_range",
        "analyze_spectrum_for_cutoff",
        "detect_fundamental_autocorrelation"
    );

    m.add_submodule(&detection_mod)?;
    Ok(())
}
