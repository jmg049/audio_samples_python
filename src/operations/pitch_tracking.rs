//! librosa-compatible frame-wise pitch tracking on `AudioSamples` (`yin`,
//! `pyin`), binding the Rust `AudioPitchTracking` trait.
//!
//! Outputs are `(channels, n_frames)` arrays (librosa returns `(n_frames,)` for
//! 1-D input; index `[0]`).

use audio_samples::operations::pitch_tracking::{AudioPitchTracking, UnvoicedFill};
use numpy::{IntoPyArray, PyArray2};
use pyo3::IntoPyObjectExt;
use pyo3::prelude::*;

use crate::features::args::FillNaArg;
use crate::features::pitch::{pyin_config, yin_config};
use crate::{PyAudioSamples, audio_err_to_py, dispatch_with_view};

#[pymethods]
impl PyAudioSamples {
    /// Frame-wise YIN fundamental frequency, equivalent to
    /// ``librosa.yin(y, fmin=fmin, fmax=fmax, sr=self.sample_rate, ...)``.
    ///
    /// Unlike ``detect_pitch_yin()`` / ``track_pitch()``, this follows librosa's
    /// framing (centred frames), CMND, parabolic refinement and fallback.
    ///
    /// Args:
    ///     fmin (float): Lowest frequency searched (Hz). Required, as in librosa
    ///         (librosa recommends ``librosa.note_to_hz("C2")`` = 65.41).
    ///     fmax (float): Highest frequency searched (Hz). Required (e.g. C7 = 2093.0).
    ///     frame_length (int): Samples per frame. Default 2048.
    ///     hop_length (Optional[int]): None = ``frame_length // 4``.
    ///     trough_threshold (float): Absolute CMND threshold. Default 0.1.
    ///     center (bool): Centre frames on ``t * hop_length``. Default True.
    ///     pad_mode (str): Padding for ``center``. Default ``"constant"``.
    ///
    /// Returns:
    ///     numpy.ndarray[float64]: f0 in Hz, shape ``(channels, n_frames)``.
    ///
    /// Raises:
    ///     ValueError: For invalid parameters (e.g. ``fmax > sr / 2``) or a
    ///         too-short uncentred signal.
    #[pyo3(
        signature = (*, fmin, fmax, frame_length = 2048, hop_length = None, trough_threshold = 0.1, center = true, pad_mode = "constant"),
        text_signature = "($self, *, fmin, fmax, frame_length=2048, hop_length=None, trough_threshold=0.1, center=True, pad_mode='constant')"
    )]
    #[allow(clippy::too_many_arguments)]
    fn yin<'py>(
        &self,
        py: Python<'py>,
        fmin: f64,
        fmax: f64,
        frame_length: i64,
        hop_length: Option<i64>,
        trough_threshold: f64,
        center: bool,
        pad_mode: &str,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let cfg = yin_config(
            fmin,
            fmax,
            frame_length,
            hop_length,
            trough_threshold,
            center,
            pad_mode,
        )?;
        let f0 = dispatch_with_view!(self, py, |audio| audio.yin(&cfg)).map_err(audio_err_to_py)?;
        Ok(f0.into_pyarray(py))
    }

    /// Probabilistic YIN with HMM decoding, equivalent to
    /// ``librosa.pyin(y, fmin=fmin, fmax=fmax, sr=self.sample_rate, ...)``.
    ///
    /// Args:
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
    ///     tuple: ``(f0, voiced_flag, voiced_prob)``, each of shape
    ///     ``(channels, n_frames)``: float64 Hz, ``bool``, float64 in ``[0, 1]``.
    ///
    /// Raises:
    ///     ValueError: For invalid parameters or a too-short uncentred signal.
    #[pyo3(
        signature = (*, fmin, fmax, frame_length = 2048, hop_length = None, n_thresholds = 100, beta_parameters = (2.0, 18.0), boltzmann_parameter = 2.0, resolution = 0.1, max_transition_rate = 35.92, switch_prob = 0.01, no_trough_prob = 0.01, fill_na = FillNaArg(UnvoicedFill::Nan), center = true, pad_mode = "constant", transition_min_prob = Some(1e-4)),
        text_signature = "($self, *, fmin, fmax, frame_length=2048, hop_length=None, n_thresholds=100, beta_parameters=(2, 18), boltzmann_parameter=2, resolution=0.1, max_transition_rate=35.92, switch_prob=0.01, no_trough_prob=0.01, fill_na=math.nan, center=True, pad_mode='constant', transition_min_prob=0.0001)"
    )]
    #[allow(clippy::too_many_arguments)]
    fn pyin<'py>(
        &self,
        py: Python<'py>,
        fmin: f64,
        fmax: f64,
        frame_length: i64,
        hop_length: Option<i64>,
        n_thresholds: i64,
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
        let cfg = pyin_config(
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
        )?;
        let r = dispatch_with_view!(self, py, |audio| audio.pyin(&cfg)).map_err(audio_err_to_py)?;
        (
            r.f0.into_pyarray(py),
            r.voiced_flag.into_pyarray(py),
            r.voiced_prob.into_pyarray(py),
        )
            .into_bound_py_any(py)
    }
}
