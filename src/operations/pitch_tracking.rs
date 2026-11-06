//! Frame-wise pitch tracking on `AudioSamples` (`yin`, `pyin`), binding the
//! Rust `AudioPitchTracking` trait.
//!
//! Outputs are `(channels, n_frames)` arrays, also for mono audio (index `[0]`
//! for `(n_frames,)`).

use crate::args::Int;
use audio_samples::operations::pitch_tracking::{AudioPitchTracking, UnvoicedFill};
use numpy::IntoPyArray;
use pyo3::IntoPyObjectExt;
use pyo3::prelude::*;

use crate::features::args::FillNaArg;
use crate::features::pitch::{PyinArgs, yin_config};
use crate::{PyAudioSamples, audio_err_to_py, dispatch_array_detached, dispatch_typed_detached};

#[pymethods]
impl PyAudioSamples {
    /// Frame-wise YIN fundamental frequency.
    ///
    /// Unlike ``detect_pitch_yin()`` / ``track_pitch()``, this uses centred
    /// frames, the cumulative mean normalised difference (CMND), parabolic
    /// refinement of the chosen trough and a fallback to the global minimum
    /// when no trough is below ``trough_threshold``.
    ///
    /// Args:
    ///
    ///     fmin (float): Lowest frequency searched (Hz). Required (e.g. C2 = 65.41).
    ///     fmax (float): Highest frequency searched (Hz). Required (e.g. C7 = 2093.0).
    ///     frame_length (int): Samples per frame. Default 2048.
    ///     hop_length (Optional[int]): None = ``frame_length // 4``.
    ///     trough_threshold (float): Absolute CMND threshold. Default 0.1.
    ///     center (bool): Centre frames on ``t * hop_length``. Default True.
    ///     pad_mode (str): Padding for ``center``. Default ``"constant"``.
    ///
    /// Returns:
    ///
    ///     numpy.ndarray[float32 | float64]: f0 in Hz, shape ``(channels, n_frames)``.
    ///
    /// Raises:
    ///
    ///     ValueError: For invalid parameters (e.g. ``fmax > sr / 2``) or a
    ///         too-short uncentred signal.
    #[pyo3(
        signature = (*, fmin, fmax, frame_length = Int(2048), hop_length = None, trough_threshold = 0.1, center = true, pad_mode = "constant"),
        text_signature = "($self, *, fmin, fmax, frame_length=2048, hop_length=None, trough_threshold=0.1, center=True, pad_mode='constant')"
    )]
    #[allow(
        clippy::too_many_arguments,
        reason = "the Python signature has 7 parameters"
    )]
    fn yin<'py>(
        &self,
        py: Python<'py>,
        fmin: f64,
        fmax: f64,
        frame_length: Int,
        hop_length: Option<Int>,
        trough_threshold: f64,
        center: bool,
        pad_mode: &str,
    ) -> PyResult<Bound<'py, PyAny>> {
        let cfg = yin_config(
            fmin,
            fmax,
            frame_length,
            hop_length,
            trough_threshold,
            center,
            pad_mode,
        )?;
        dispatch_array_detached!(self, py, |audio| audio.yin(&cfg))
    }

    /// Probabilistic YIN with HMM decoding: f0 tracking with voicing decisions.
    ///
    /// Args:
    ///
    ///     fmin, fmax (float): Search range in Hz. Required.
    ///     frame_length (int): Default 2048.
    ///     hop_length (Optional[int]): None = ``frame_length // 4``.
    ///     n_thresholds (int): Default 100.
    ///     beta_parameters (tuple[float, float]): Beta prior. Default (2, 18).
    ///     boltzmann_parameter (float): Default 2.
    ///     resolution (float): Pitch-bin width in semitones. Default 0.1.
    ///     max_transition_rate (float): Octaves per second. Default 35.92.
    ///     switch_prob (float): Voiced/unvoiced switch probability. Default 0.01.
    ///     no_trough_prob (float): Default 0.01.
    ///     fill_na (Optional[float]): f0 of unvoiced frames: NaN (default), None
    ///         (the best-guess pitch) or a number.
    ///     center (bool): Default True.
    ///     pad_mode (str): Default ``"constant"``.
    ///     transition_min_prob (Optional[float]): Viterbi pruning threshold
    ///         (None or 0 = exact decoding). Default 1e-4.
    ///
    /// Returns:
    ///
    ///     tuple: ``(f0, voiced_flag, voiced_prob)``, each of shape
    ///     ``(channels, n_frames)``: Hz, ``bool``, values in ``[0, 1]``; the float
    ///     arrays are float32 for uint8, int16, int24 and float32 audio and
    ///     float64 otherwise.
    ///
    /// Raises:
    ///
    ///     ValueError: For invalid parameters or a too-short uncentred signal.
    #[pyo3(
        signature = (*, fmin, fmax, frame_length = Int(2048), hop_length = None, n_thresholds = Int(100), beta_parameters = (2.0, 18.0), boltzmann_parameter = 2.0, resolution = 0.1, max_transition_rate = 35.92, switch_prob = 0.01, no_trough_prob = 0.01, fill_na = FillNaArg(UnvoicedFill::Nan), center = true, pad_mode = "constant", transition_min_prob = Some(1e-4)),
        text_signature = "($self, *, fmin, fmax, frame_length=2048, hop_length=None, n_thresholds=100, beta_parameters=(2, 18), boltzmann_parameter=2, resolution=0.1, max_transition_rate=35.92, switch_prob=0.01, no_trough_prob=0.01, fill_na=math.nan, center=True, pad_mode='constant', transition_min_prob=0.0001)"
    )]
    #[allow(
        clippy::too_many_arguments,
        reason = "the Python signature has 15 parameters"
    )]
    fn pyin<'py>(
        &self,
        py: Python<'py>,
        fmin: f64,
        fmax: f64,
        frame_length: Int,
        hop_length: Option<Int>,
        n_thresholds: Int,
        beta_parameters: (f64, f64),
        boltzmann_parameter: f64,
        resolution: f64,
        max_transition_rate: f64,
        switch_prob: f64,
        no_trough_prob: f64,
        fill_na: FillNaArg,
        center: bool,
        pad_mode: &str,
        transition_min_prob: Option<f64>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let cfg = PyinArgs {
            fmin,
            fmax,
            frame_length,
            hop_length,
            n_thresholds,
            beta_parameters,
            boltzmann_parameter,
            resolution,
            max_transition_rate,
            switch_prob,
            no_trough_prob,
            fill_na,
            center,
            pad_mode,
            transition_min_prob,
        }
        .config()?;
        dispatch_typed_detached!(self, py, |audio| audio.pyin(&cfg), |r| {
            let r = r.map_err(audio_err_to_py)?;
            (
                r.f0.into_pyarray(py),
                r.voiced_flag.into_pyarray(py),
                r.voiced_prob.into_pyarray(py),
            )
                .into_bound_py_any(py)
        })
    }
}
