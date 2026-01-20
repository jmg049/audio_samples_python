use crate::{PyAudioSamples, audio_err_to_py, dispatch_with_view, reexport};
use audio_samples::utils::detection::{detect_fundamental_frequency, detect_sample_rate};
use pyo3::prelude::*;

#[pyfunction]
#[pyo3(name = "detect_sample_rate", signature=(audio: "AudioSamples"), text_signature="(audio: AudioSamples) -> Optional[int]")]
pub fn py_detect_sample_rate(py: Python<'_>, audio: &PyAudioSamples) -> PyResult<Option<u32>> {
    dispatch_with_view!(audio, py, |audio| {
        detect_sample_rate::<_>(&audio)
            .map_err(audio_err_to_py)
            .map(|rate| rate.map(|r| r.get()))
    })
}

#[pyfunction]
#[pyo3(name = "detect_fundamental_frequency", signature=(audio: "AudioSamples"), text_signature="(audio: AudioSamples) -> Optional[int]")]
pub fn py_detect_fundamental_frequency(
    py: Python<'_>,
    audio: &PyAudioSamples,
) -> PyResult<Option<f64>> {
    dispatch_with_view!(audio, py, |audio| {
        detect_fundamental_frequency::<_>(&audio).map_err(audio_err_to_py)
    })
}

#[pymodule]
pub fn detection(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    let detection_mod = PyModule::new(py, "detection")?;

    detection_mod.add_function(wrap_pyfunction!(py_detect_sample_rate, &detection_mod)?)?;
    detection_mod.add_function(wrap_pyfunction!(
        py_detect_fundamental_frequency,
        &detection_mod
    )?)?;
    reexport!(
        m,
        detection_mod,
        "detect_sample_rate",
        "detect_fundamental_frequency"
    );

    // m.add_submodule(&generation_mod)?; // FIXME: generation_mod is not defined
    Ok(())
}
