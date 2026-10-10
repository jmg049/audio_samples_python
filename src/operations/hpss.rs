use crate::args::{Int, opt_positive};
use crate::{PyAudioSamples, audio_err_to_py, dispatch_with_view, types::PyHpssConfig};
use audio_samples::{AudioDecomposition, operations::HpssConfig};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use spectrograms::python::PyStftParams;
use std::num::NonZeroUsize;

#[pymethods]
impl PyAudioSamples {
    /// Separate harmonic and percussive components using median-filter HPSS.
    ///
    /// Args:
    ///     config (StftParams | HpssConfig): Either a complete ``HpssConfig``
    ///         (the remaining arguments must then be omitted), or the
    ///         ``StftParams`` of the analysis STFT.
    ///     median_filter_harmonic (Optional[int]): Harmonic (time-direction)
    ///         median filter length in frames. Default 17. ``StftParams`` only.
    ///     median_filter_percussive (Optional[int]): Percussive
    ///         (frequency-direction) median filter length in bins. Default 17.
    ///         ``StftParams`` only.
    ///     mask_softness (Optional[float]): Mask softness in [0, 1] (0 = hard
    ///         mask). Default 0.5. ``StftParams`` only.
    ///
    /// Returns:
    ///     tuple[AudioSamples, AudioSamples]: ``(harmonic, percussive)``.
    ///
    /// Raises:
    ///     TypeError: If ``config`` is neither ``StftParams`` nor ``HpssConfig``,
    ///         or filter arguments are given together with an ``HpssConfig``.
    ///     ValueError: If the configuration is invalid.
    ///
    /// Examples:
    ///     >>> harmonic, percussive = audio.hpss(HpssConfig.musical)
    ///     >>> harmonic, percussive = audio.hpss(StftParams(2048, 512, WindowType.hanning, True))
    #[pyo3(
        signature = (config, median_filter_harmonic = None, median_filter_percussive = None, mask_softness = None),
        text_signature = "($self, config, median_filter_harmonic=None, median_filter_percussive=None, mask_softness=None)"
    )]
    fn hpss(
        &self,
        py: Python<'_>,
        config: &Bound<'_, PyAny>,
        median_filter_harmonic: Option<Int>,
        median_filter_percussive: Option<Int>,
        mask_softness: Option<f64>,
    ) -> PyResult<(Self, Self)> {
        let median_filter_harmonic =
            opt_positive("median_filter_harmonic", median_filter_harmonic)?;
        let median_filter_percussive =
            opt_positive("median_filter_percussive", median_filter_percussive)?;
        let config = if let Ok(cfg) = config.extract::<PyHpssConfig>() {
            if median_filter_harmonic.is_some()
                || median_filter_percussive.is_some()
                || mask_softness.is_some()
            {
                return Err(PyTypeError::new_err(
                    "hpss: pass either an HpssConfig alone, or StftParams with the filter arguments",
                ));
            }
            cfg.inner
        } else if let Ok(stft) = config.extract::<PyStftParams>() {
            HpssConfig::new(
                stft.inner,
                median_filter_harmonic.map_or(17, NonZeroUsize::get),
                median_filter_percussive.map_or(17, NonZeroUsize::get),
                mask_softness.unwrap_or(0.5),
            )
        } else {
            return Err(PyTypeError::new_err(
                "hpss: config must be an HpssConfig or a StftParams",
            ));
        };
        dispatch_with_view!(self, py, |audio| {
            let (harmonic, percussive) = audio.hpss(&config).map_err(audio_err_to_py)?;
            Ok((
                Self::from_audio_samples(harmonic),
                Self::from_audio_samples(percussive),
            ))
        })
    }
}
