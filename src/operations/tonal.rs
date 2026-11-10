//! Tonal analysis on `AudioSamples` (`chroma_stft`,
//! `piptrack`, `estimate_tuning`), binding the Rust `AudioTonal` trait.

use crate::args::Int;
use audio_samples::operations::filterbanks::Window;
use audio_samples::operations::tonal::{AudioTonal, ChromaStftConfig, TuningConfig};
use numpy::IntoPyArray;
use pyo3::IntoPyObjectExt;
use pyo3::prelude::*;

use crate::args::{WindowArg, positive, stft_config};
use crate::{
    PyAudioSamples, audio_err_to_py, dispatch_array_detached, dispatch_typed_detached,
    dispatch_with_view_detached,
};

#[pymethods]
impl PyAudioSamples {
    /// Chromagram from the power STFT.
    ///
    /// ``chroma()`` (spectrograms crate) is a separate implementation; this one
    /// uses the ``chroma_filterbank`` with optional tuning estimation and
    /// per-frame ``norm`` normalisation.
    ///
    /// Args:
    ///
    ///     norm (Optional[float]): Per-frame normalisation (``numpy.inf`` default,
    ///         any p-norm, or None for raw energy).
    ///     n_fft (int): FFT length. Default 2048.
    ///     hop_length (int): Samples between frames. Default 512.
    ///     win_length, window, center, pad_mode: STFT parameters (see ``stft_matrix``).
    ///     tuning (Optional[float]): Tuning offset in fractions of a chroma bin.
    ///         Default 0.0; None estimates it from the signal.
    ///     n_chroma (int): Number of chroma bins. Default 12.
    ///     ctroct (float): Centre of the octave weighting (octaves above A0). Default 5.
    ///     octwidth (Optional[float]): Octave-weighting half-width; None = flat. Default 2.
    ///     base_c (bool): Row 0 is C (True) or A. Default True.
    ///
    /// Returns:
    ///
    ///     numpy.ndarray[float32 | float64]: shape ``(channels, n_chroma, n_frames)``,
    ///     also for mono audio (index ``[0]`` for ``(n_chroma, n_frames)``).
    ///
    /// Raises:
    ///
    ///     ValueError: For invalid parameters.
    #[pyo3(
        signature = (*, norm = Some(f64::INFINITY), n_fft = Int(2048), hop_length = Int(512), win_length = None, window = WindowArg(Window::Hann), center = true, pad_mode = "constant", tuning = Some(0.0), n_chroma = Int(12), ctroct = 5.0, octwidth = Some(2.0), base_c = true) -> "NDArray[Any]",
        text_signature = "($self, *, norm=math.inf, n_fft=2048, hop_length=512, win_length=None, window='hann', center=True, pad_mode='constant', tuning=0.0, n_chroma=12, ctroct=5.0, octwidth=2.0, base_c=True)"
    )]
    #[allow(
        clippy::too_many_arguments,
        reason = "the Python signature has 12 parameters"
    )]
    fn chroma_stft<'py>(
        &self,
        py: Python<'py>,
        norm: Option<f64>,
        n_fft: Int,
        hop_length: Int,
        win_length: Option<Int>,
        window: WindowArg,
        center: bool,
        pad_mode: &str,
        tuning: Option<f64>,
        n_chroma: Int,
        ctroct: f64,
        octwidth: Option<f64>,
        base_c: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        let cfg = ChromaStftConfig {
            stft: stft_config(
                n_fft,
                Some(hop_length),
                win_length,
                window,
                center,
                pad_mode,
            )?,
            norm,
            tuning,
            n_chroma: positive("n_chroma", n_chroma)?,
            ctroct,
            octwidth,
            base_c,
        };
        dispatch_array_detached!(self, py, |audio| audio.chroma_stft(&cfg))
    }

    /// Spectral peak tracking with parabolic interpolation.
    ///
    /// Args:
    ///
    ///     n_fft (int): FFT length. Default 2048.
    ///     hop_length (Optional[int]): None = ``win_length // 4``.
    ///     fmin (float): Lowest reported frequency (Hz). Default 150.
    ///     fmax (float): Peaks must lie below this frequency (Hz). Default 4000.
    ///     threshold (float): Relative threshold. Default 0.1.
    ///     win_length, window, center, pad_mode: STFT parameters.
    ///     ref (None | float | numpy.max | numpy.mean): Threshold reference; None
    ///         (default) = per-frame maximum, a number = absolute threshold.
    ///
    /// Returns:
    ///
    ///     tuple: ``(pitches, magnitudes)``, each numpy.ndarray[float32 | float64] of shape
    ///     ``(channels, 1 + n_fft // 2, n_frames)``; zero except at detected peaks.
    ///
    /// Raises:
    ///
    ///     ValueError: For invalid parameters.
    #[pyo3(
        signature = (*, n_fft = Int(2048), hop_length = None, fmin = 150.0, fmax = 4000.0, threshold = 0.1, win_length = None, window = WindowArg(Window::Hann), center = true, pad_mode = "constant", r#ref = None) -> "tuple[NDArray[Any], NDArray[Any]] | tuple[NDArray[np.float32], NDArray[np.float32]] | tuple[NDArray[np.float64], NDArray[np.float64]]",
        text_signature = "($self, *, n_fft=2048, hop_length=None, fmin=150.0, fmax=4000.0, threshold=0.1, win_length=None, window='hann', center=True, pad_mode='constant', ref=None)"
    )]
    #[allow(
        clippy::too_many_arguments,
        reason = "the Python signature has 10 parameters"
    )]
    fn piptrack<'py>(
        &self,
        py: Python<'py>,
        n_fft: Int,
        hop_length: Option<Int>,
        fmin: f64,
        fmax: f64,
        threshold: f64,
        win_length: Option<Int>,
        window: WindowArg,
        center: bool,
        pad_mode: &str,
        r#ref: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let stft = stft_config(n_fft, hop_length, win_length, window, center, pad_mode)?;
        let cfg = piptrack_config(stft, fmin, fmax, threshold, r#ref)?;
        dispatch_typed_detached!(self, py, |audio| audio.piptrack(&cfg), |r| {
            let (p, m) = r.map_err(audio_err_to_py)?;
            (p.into_pyarray(py), m.into_pyarray(py)).into_bound_py_any(py)
        })
    }

    /// Tuning offset relative to A440 in fractions of a bin, estimated from
    /// spectral peaks.
    ///
    /// All channels are pooled.
    ///
    /// Args:
    ///
    ///     n_fft (int): FFT length. Default 2048.
    ///     resolution (float): Resolution in fractions of a bin. Default 0.01.
    ///     bins_per_octave (int): Bins per octave of the grid. Default 12.
    ///     hop_length, win_length, window, center, pad_mode, fmin, fmax, threshold, ref:
    ///         ``piptrack`` parameters.
    ///
    /// Returns:
    ///
    ///     float: tuning in ``[-0.5, 0.5)`` fractions of a bin.
    ///
    /// Raises:
    ///
    ///     ValueError: For invalid parameters.
    #[pyo3(
        signature = (*, n_fft = Int(2048), resolution = 0.01, bins_per_octave = Int(12), hop_length = None, win_length = None, window = WindowArg(Window::Hann), center = true, pad_mode = "constant", fmin = 150.0, fmax = 4000.0, threshold = 0.1, r#ref = None),
        text_signature = "($self, *, n_fft=2048, resolution=0.01, bins_per_octave=12, hop_length=None, win_length=None, window='hann', center=True, pad_mode='constant', fmin=150.0, fmax=4000.0, threshold=0.1, ref=None)"
    )]
    #[allow(
        clippy::too_many_arguments,
        reason = "the Python signature has 12 parameters"
    )]
    fn estimate_tuning(
        &self,
        py: Python<'_>,
        n_fft: Int,
        resolution: f64,
        bins_per_octave: Int,
        hop_length: Option<Int>,
        win_length: Option<Int>,
        window: WindowArg,
        center: bool,
        pad_mode: &str,
        fmin: f64,
        fmax: f64,
        threshold: f64,
        r#ref: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<f64> {
        let stft = stft_config(n_fft, hop_length, win_length, window, center, pad_mode)?;
        let cfg = TuningConfig {
            piptrack: piptrack_config(stft, fmin, fmax, threshold, r#ref)?,
            resolution,
            bins_per_octave: positive("bins_per_octave", bins_per_octave)?,
        };
        dispatch_with_view_detached!(self, py, |audio| audio.estimate_tuning(&cfg))
            .map_err(audio_err_to_py)
    }
}

fn piptrack_config(
    stft: audio_samples::operations::spectrum::StftConfig,
    fmin: f64,
    fmax: f64,
    threshold: f64,
    reference: Option<&Bound<'_, PyAny>>,
) -> PyResult<audio_samples::operations::tonal::PiptrackConfig> {
    let reference = match reference {
        Some(r) => crate::features::args::parse_piptrack_ref(r)?,
        None => audio_samples::operations::tonal::PiptrackReference::FrameMax,
    };
    Ok(audio_samples::operations::tonal::PiptrackConfig {
        stft,
        fmin,
        fmax,
        threshold,
        reference,
    })
}
