//! `audio_samples.features`: feature, tonal, constant-Q and pitch helpers over
//! NumPy arrays.
//!
//! Binds the free functions of `audio_samples::operations::{features, tonal,
//! constant_q, pitch_tracking}`. The same analyses on audio are methods of
//! `AudioSamples` (`spectral_centroid_frames`, `chroma_stft`, `cqt`, `pyin`, ...).
//! A 2-D spectrogram `(F, T)` gives 2-D output, a 3-D `(channels, F, T)`
//! spectrogram gives 3-D output.

pub mod args;
pub mod constant_q;
pub mod pitch;
pub mod spectral;
pub mod tonal;

use pyo3::prelude::*;

#[doc = "Feature, tonal, constant-Q and pitch functions over NumPy arrays: spectral \
         descriptors of a spectrogram, pcen, frequency weightings, chroma filterbanks \
         and tuning (chroma_filterbank, pitch_tuning, estimate_tuning_from_spectrogram, \
         normalize), constant-Q filters (cq_to_chroma, wavelet), \
         chroma_cens / tonnetz from a chromagram, yin / pyin on arrays and the harmonic \
         helpers (salience, interp_harmonics, f0_harmonics). float32 / complex64 \
         arrays are processed in float32 and give float32 output; any other input is \
         converted to float64."]
#[pymodule(submodule, name = "features")]
pub mod py_module {
    #[pymodule_export]
    use super::constant_q::{
        chroma_cens_from_chroma, cq_to_chroma, tonnetz_from_chroma, wavelet, wavelet_lengths,
    };
    #[pymodule_export]
    use super::pitch::{f0_harmonics, interp_harmonics, pyin, salience, yin};
    #[pymodule_export]
    use super::spectral::{
        a_weighting, b_weighting, c_weighting, d_weighting, frequency_weighting,
        multi_frequency_weighting, pcen, perceptual_weighting, poly_features_from_spectrogram,
        rms_from_spectrogram, spectral_bandwidth_from_spectrogram,
        spectral_centroid_from_spectrogram, spectral_contrast_from_spectrogram,
        spectral_flatness_from_spectrogram, spectral_rolloff_from_spectrogram, z_weighting,
    };
    #[pymodule_export]
    use super::tonal::{
        chroma_filterbank, chroma_stft_from_spectrogram, estimate_tuning_from_signal,
        estimate_tuning_from_spectrogram, normalize, piptrack_from_spectrogram, pitch_tuning,
    };
}
