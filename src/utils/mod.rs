pub mod audio_math;
pub mod comparison;
pub mod detection;
pub mod generation;

use pyo3::prelude::*;

#[pymodule]
pub fn utils(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    let utils_mod = PyModule::new(py, "utils")?;

    // TODO: Implement concatenate and stack as class methods on PyAudioSamples first
    // utils_mod.add_function(wrap_pyfunction!(concatenate, &utils_mod)?)?;
    // utils_mod.add_function(wrap_pyfunction!(stack, &utils_mod)?)?;

    // Audio math functions - frequency conversions
    utils_mod.add_function(wrap_pyfunction!(audio_math::hz_to_mel, &utils_mod)?)?;
    utils_mod.add_function(wrap_pyfunction!(audio_math::mel_to_hz, &utils_mod)?)?;
    utils_mod.add_function(wrap_pyfunction!(audio_math::mel_scale, &utils_mod)?)?;
    utils_mod.add_function(wrap_pyfunction!(audio_math::hz_to_midi, &utils_mod)?)?;
    utils_mod.add_function(wrap_pyfunction!(audio_math::midi_to_hz, &utils_mod)?)?;

    // Audio math functions - amplitude/power conversions
    utils_mod.add_function(wrap_pyfunction!(audio_math::amplitude_to_db, &utils_mod)?)?;
    utils_mod.add_function(wrap_pyfunction!(audio_math::db_to_amplitude, &utils_mod)?)?;
    utils_mod.add_function(wrap_pyfunction!(audio_math::power_to_db, &utils_mod)?)?;
    utils_mod.add_function(wrap_pyfunction!(audio_math::db_to_power, &utils_mod)?)?;

    // Audio math functions - time/frame conversions
    utils_mod.add_function(wrap_pyfunction!(audio_math::frames_to_time, &utils_mod)?)?;
    utils_mod.add_function(wrap_pyfunction!(audio_math::time_to_frames, &utils_mod)?)?;
    utils_mod.add_function(wrap_pyfunction!(audio_math::samples_to_time, &utils_mod)?)?;
    utils_mod.add_function(wrap_pyfunction!(
        audio_math::seconds_to_samples,
        &utils_mod
    )?)?;
    utils_mod.add_function(wrap_pyfunction!(audio_math::ms_to_samples, &utils_mod)?)?;

    // Audio math functions - musical theory
    utils_mod.add_function(wrap_pyfunction!(audio_math::note_to_midi, &utils_mod)?)?;
    utils_mod.add_function(wrap_pyfunction!(audio_math::midi_to_note, &utils_mod)?)?;
    utils_mod.add_function(wrap_pyfunction!(audio_math::note_to_frequency, &utils_mod)?)?;
    utils_mod.add_function(wrap_pyfunction!(audio_math::frequency_to_note, &utils_mod)?)?;
    utils_mod.add_function(wrap_pyfunction!(audio_math::cents_to_ratio, &utils_mod)?)?;
    utils_mod.add_function(wrap_pyfunction!(audio_math::ratio_to_cents, &utils_mod)?)?;

    // Audio math functions - spectral utilities
    utils_mod.add_function(wrap_pyfunction!(audio_math::fft_frequencies, &utils_mod)?)?;
    utils_mod.add_function(wrap_pyfunction!(audio_math::mel_frequencies, &utils_mod)?)?;
    utils_mod.add_function(wrap_pyfunction!(audio_math::linspace, &utils_mod)?)?;

    // Comparison functions
    utils_mod.add_function(wrap_pyfunction!(comparison::correlation, &utils_mod)?)?;
    utils_mod.add_function(wrap_pyfunction!(comparison::mse, &utils_mod)?)?;
    utils_mod.add_function(wrap_pyfunction!(comparison::snr, &utils_mod)?)?;
    utils_mod.add_function(wrap_pyfunction!(comparison::align_signals, &utils_mod)?)?;
    utils_mod.add_function(wrap_pyfunction!(comparison::psnr, &utils_mod)?)?;
    utils_mod.add_function(wrap_pyfunction!(comparison::segmental_snr, &utils_mod)?)?;
    utils_mod.add_function(wrap_pyfunction!(
        comparison::log_spectral_distance,
        &utils_mod
    )?)?;
    utils_mod.add_function(wrap_pyfunction!(
        comparison::correlation_per_channel,
        &utils_mod
    )?)?;
    utils_mod.add_function(wrap_pyfunction!(comparison::mse_per_channel, &utils_mod)?)?;
    utils_mod.add_function(wrap_pyfunction!(comparison::snr_per_channel, &utils_mod)?)?;

    generation::generation(py, &utils_mod)?;
    detection::detection(py, &utils_mod)?;
    m.add_submodule(&utils_mod)?;
    Ok(())
}

// TODO: Implement concatenate and stack as class methods on PyAudioSamples first
// #[pyfunction]
// #[pyo3(signature = (signals), text_signature = "(signals: list[AudioSamples] -> AudioSamples)")]
// /// Convenience function for concatenating AudioSamples.
// ///
// /// Uses the class method AudioSamples::concatentate to create the output.
// pub fn concatenate(py: Python, signals: &Bound<'_, PyList>) -> PyResult<PyAudioSamples> {
//     PyAudioSamples::concatenate(&PyType::new::<PyAudioSamples>(py), py, signals)
// }

// #[pyfunction]
// #[pyo3(signature = (signals), text_signature = "(signals: list[AudioSamples] -> AudioSamples)")]
// /// Convenience function for stacking AudioSamples.
// ///
// /// Uses the class method AudioSamples::stack to create the output.
// pub fn stack(py: Python, signals: &Bound<'_, PyList>) -> PyResult<PyAudioSamples> {
//     PyAudioSamples::stack(&PyType::new::<PyAudioSamples>(py), py, signals)
// }
