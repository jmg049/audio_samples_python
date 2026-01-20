use crate::{PyAudioSamples, audio_err_to_py, dispatch_with_view};
use audio_samples::{AudioDecomposition, operations::HpssConfig};
use pyo3::prelude::*;
use spectrograms::python::PyStftParams;

#[pymethods]
impl PyAudioSamples {
    /// Separate harmonic and percussive components using HPSS.
    ///
    /// Uses Harmonic-Percussive Source Separation to split audio into tonal
    /// (harmonic) and transient (percussive) components.
    ///
    /// Args:
    ///     n_fft (int): STFT window size in samples (default: 2048)
    ///     hop_size (int): STFT hop size in samples (default: 512)
    ///     median_filter_harmonic (int): Harmonic median filter size (default: 17)
    ///     median_filter_percussive (int): Percussive median filter size (default: 17)
    ///     mask_softness (float): Soft masking parameter 0.0-1.0 (default: 0.5)
    ///
    /// Returns:
    ///     tuple: (harmonic_audio, percussive_audio) as separate AudioSamples
    ///
    /// Examples:
    ///     >>> harmonic, percussive = audio.hpss(win_size=2048, hop_size=512)
    #[pyo3(signature = (stft_params: "StftParams", median_filter_harmonic: "int" = 17, median_filter_percussive: "int" = 17, mask_softness: "float" = 0.5), text_signature = "($self, win_size: int = 2048, hop_size: int = 512, median_filter_harmonic: int = 17, median_filter_percussive: int = 17, mask_softness: float = 0.5) -> tuple[AudioSamples, AudioSamples]")]
    fn hpss(
        &self,
        py: Python<'_>,
        stft_params: PyStftParams,
        median_filter_harmonic: usize,
        median_filter_percussive: usize,
        mask_softness: f64,
    ) -> PyResult<(Self, Self)> {
        dispatch_with_view!(self, py, |audio| {
            let config = HpssConfig::new(
                stft_params.inner,
                median_filter_harmonic,
                median_filter_percussive,
                mask_softness,
            );
            let (harmonic, percussive) = audio.hpss(&config).map_err(audio_err_to_py)?;
            Ok((
                Self::from_audio_samples(harmonic),
                Self::from_audio_samples(percussive),
            ))
        })
    }
}
