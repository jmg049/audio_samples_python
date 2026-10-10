//! `audio_samples.features`: librosa-compatible feature, tonal, constant-Q and
//! pitch helpers over NumPy arrays.
//!
//! Binds the free functions of `audio_samples::operations::{features, tonal,
//! constant_q, pitch_tracking}`. The same analyses on audio are methods of
//! `AudioSamples` (`spectral_centroid_frames`, `chroma_stft`, `cqt`, `pyin`, ...).
//! Array shapes follow librosa: a 2-D spectrogram `(F, T)` gives 2-D output, a
//! 3-D `(channels, F, T)` spectrogram gives 3-D output.

pub(crate) mod args;
pub mod constant_q;
pub mod pitch;
pub mod spectral;
pub mod tonal;

use pyo3::prelude::*;

/// Registers `audio_samples.features`.
pub fn register(py: Python<'_>, parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(py, "features")?;
    m.add(
        "__doc__",
        "librosa-compatible feature helpers over NumPy arrays: spectral descriptors \
         from spectrograms (librosa.feature.*(S=...)), pcen, frequency weightings, \
         chroma filterbanks and tuning (librosa.filters.chroma, pitch_tuning, \
         estimate_tuning, util.normalize), constant-Q filters (cq_to_chroma, wavelet), \
         chroma_cens / tonnetz from a chromagram, yin / pyin on arrays and the harmonic \
         helpers (salience, interp_harmonics, f0_harmonics). float32 / complex64 \
         arrays are processed in float32 and give float32 output; any other input is \
         converted to float64.",
    )?;
    spectral::add(&m)?;
    tonal::add(&m)?;
    constant_q::add(&m)?;
    pitch::add(&m)?;
    parent.add_submodule(&m)?;
    Ok(())
}
