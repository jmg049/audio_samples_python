//! Pitch-tracking free functions of `audio_samples.features` (binding
//! `audio_samples::operations::pitch_tracking`): `librosa.yin` / `pyin` on
//! arrays and the harmonic helpers `salience`, `interp_harmonics`,
//! `f0_harmonics`.

use crate::args::Int;
use audio_samples::operations::pitch_tracking::{
    self as pt, FrequencyGrid, PyinConfig, SalienceConfig, YinConfig,
};
use ndarray::{Array1, Array2, Array3, Axis, Ix1, Ix2};
use pyo3::IntoPyObjectExt;
use pyo3::prelude::*;

use super::args::{FillNaArg, parse_aggregate, signal2, spec3, squeeze_out};
use crate::args::{f64_array, opt_positive, parse_pad_mode, positive, value_err};
use crate::audio_err_to_py;

/// Builds a [`YinConfig`] from librosa's `yin` keyword arguments.
#[allow(clippy::too_many_arguments)]
pub(crate) fn yin_config(
    fmin: f64,
    fmax: f64,
    frame_length: Int,
    hop_length: Option<Int>,
    trough_threshold: f64,
    center: bool,
    pad_mode: &str,
) -> PyResult<YinConfig> {
    Ok(YinConfig {
        fmin,
        fmax,
        frame_length: positive("frame_length", frame_length)?,
        hop_length: opt_positive("hop_length", hop_length)?,
        trough_threshold,
        center,
        pad_mode: parse_pad_mode(pad_mode)?,
    })
}

/// Builds a [`PyinConfig`] from librosa's `pyin` keyword arguments.
#[allow(clippy::too_many_arguments)]
pub(crate) fn pyin_config(
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
) -> PyResult<PyinConfig> {
    Ok(PyinConfig {
        fmin,
        fmax,
        frame_length: positive("frame_length", frame_length)?,
        hop_length: opt_positive("hop_length", hop_length)?,
        n_thresholds: positive("n_thresholds", n_thresholds)?,
        beta_parameters,
        boltzmann_parameter,
        resolution,
        max_transition_rate,
        switch_prob,
        no_trough_prob,
        fill: fill_na.0,
        center,
        pad_mode: parse_pad_mode(pad_mode)?,
        transition_min_prob,
    })
}

fn rows_out<'py, T: numpy::Element>(
    py: Python<'py>,
    a: Array2<T>,
    ndim_in: usize,
) -> Bound<'py, PyAny> {
    squeeze_out(py, a.into_dyn(), ndim_in == 1)
}

/// Frame-wise YIN fundamental frequency of an array, equivalent to ``librosa.yin``.
///
/// Args:
///     y (array_like): Signal, ``(n,)`` or ``(channels, n)``.
///     fmin (float): Lowest frequency searched (Hz). Required.
///     fmax (float): Highest frequency searched (Hz). Required.
///     sr (float): Sample rate. Default 22050.
///     frame_length (int): Samples per frame. Default 2048.
///     hop_length (Optional[int]): None = ``frame_length // 4``.
///     trough_threshold (float): Absolute CMND threshold. Default 0.1.
///     center (bool): Centre frames on ``t * hop_length``. Default True.
///     pad_mode (str): Padding for ``center``. Default ``"constant"``.
///
/// Returns:
///     numpy.ndarray[float64]: f0 in Hz, shape ``(n_frames,)`` or ``(channels, n_frames)``.
///
/// Raises:
///     ValueError: For invalid parameters or a too-short signal.
#[pyfunction]
#[pyo3(
    signature = (y, *, fmin, fmax, sr = 22050.0, frame_length = Int(2048), hop_length = None, trough_threshold = 0.1, center = true, pad_mode = "constant"),
    text_signature = "(y, *, fmin, fmax, sr=22050, frame_length=2048, hop_length=None, trough_threshold=0.1, center=True, pad_mode='constant')"
)]
#[allow(clippy::too_many_arguments)]
pub fn yin<'py>(
    py: Python<'py>,
    y: &Bound<'py, PyAny>,
    fmin: f64,
    fmax: f64,
    sr: f64,
    frame_length: Int,
    hop_length: Option<Int>,
    trough_threshold: f64,
    center: bool,
    pad_mode: &str,
) -> PyResult<Bound<'py, PyAny>> {
    let (x, nd) = signal2(y, "y")?;
    let cfg = yin_config(
        fmin,
        fmax,
        frame_length,
        hop_length,
        trough_threshold,
        center,
        pad_mode,
    )?;
    let x = x.view();
    let f0 = py
        .detach(|| pt::yin(x, sr, &cfg))
        .map_err(audio_err_to_py)?;
    Ok(rows_out(py, f0, nd))
}

/// Probabilistic YIN of an array, equivalent to ``librosa.pyin``.
///
/// Args:
///     y (array_like): Signal, ``(n,)`` or ``(channels, n)``.
///     fmin, fmax (float): Search range in Hz. Required.
///     sr (float): Sample rate. Default 22050.
///     frame_length (int): Default 2048.
///     hop_length (Optional[int]): None = ``frame_length // 4``.
///     n_thresholds (int): Default 100.
///     beta_parameters (tuple[float, float]): Beta prior. Default (2, 18).
///     boltzmann_parameter (float): Default 2.
///     resolution (float): Pitch-bin width in semitones. Default 0.1.
///     max_transition_rate (float): Octaves per second. Default 35.92.
///     switch_prob (float): Voicing switch probability. Default 0.01.
///     no_trough_prob (float): Default 0.01.
///     fill_na (Optional[float]): f0 of unvoiced frames: NaN (default), None
///         (best-guess pitch) or a number.
///     center (bool): Default True.
///     pad_mode (str): Default ``"constant"``.
///     transition_min_prob (Optional[float]): Viterbi pruning threshold
///         (None or 0 = exact). Default 1e-4.
///
/// Returns:
///     tuple: ``(f0, voiced_flag, voiced_prob)``, each ``(n_frames,)`` or
///     ``(channels, n_frames)``; ``voiced_flag`` is ``bool``, the others float64.
///
/// Raises:
///     ValueError: For invalid parameters or a too-short signal.
#[pyfunction]
#[pyo3(
    signature = (y, *, fmin, fmax, sr = 22050.0, frame_length = Int(2048), hop_length = None, n_thresholds = Int(100), beta_parameters = (2.0, 18.0), boltzmann_parameter = 2.0, resolution = 0.1, max_transition_rate = 35.92, switch_prob = 0.01, no_trough_prob = 0.01, fill_na = FillNaArg(audio_samples::operations::pitch_tracking::UnvoicedFill::Nan), center = true, pad_mode = "constant", transition_min_prob = Some(1e-4)),
    text_signature = "(y, *, fmin, fmax, sr=22050, frame_length=2048, hop_length=None, n_thresholds=100, beta_parameters=(2, 18), boltzmann_parameter=2, resolution=0.1, max_transition_rate=35.92, switch_prob=0.01, no_trough_prob=0.01, fill_na=math.nan, center=True, pad_mode='constant', transition_min_prob=0.0001)"
)]
#[allow(clippy::too_many_arguments)]
pub fn pyin<'py>(
    py: Python<'py>,
    y: &Bound<'py, PyAny>,
    fmin: f64,
    fmax: f64,
    sr: f64,
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
    let (x, nd) = signal2(y, "y")?;
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
    let x = x.view();
    let r = py
        .detach(|| pt::pyin(x, sr, &cfg))
        .map_err(audio_err_to_py)?;
    (
        rows_out(py, r.f0, nd),
        rows_out(py, r.voiced_flag, nd),
        rows_out(py, r.voiced_prob, nd),
    )
        .into_bound_py_any(py)
}

/// A `freqs` argument: 1-D (fixed grid) or shaped like the spectrogram.
enum Grid {
    Fixed(Array1<f64>),
    Varying(Array3<f64>),
}

impl Grid {
    fn parse(obj: &Bound<'_, PyAny>, s_ndim: usize) -> PyResult<Self> {
        let a = f64_array(obj, "freqs")?;
        let v = a.as_array();
        match v.ndim() {
            1 => Ok(Self::Fixed(
                v.into_dimensionality::<Ix1>().expect("1-D").to_owned(),
            )),
            n if n == s_ndim => {
                let a3 = if n == 2 {
                    v.into_dimensionality::<Ix2>()
                        .expect("2-D")
                        .insert_axis(Axis(0))
                        .to_owned()
                } else {
                    v.into_dimensionality::<ndarray::Ix3>()
                        .expect("3-D")
                        .to_owned()
                };
                Ok(Self::Varying(a3))
            }
            n => Err(value_err(
                "freqs",
                format!(
                    "expected a 1-D array of bin frequencies or an array shaped like the \
                     input ({s_ndim} dimensions), got {n} dimensions"
                ),
            )),
        }
    }

    fn view(&self) -> FrequencyGrid<'_> {
        match self {
            Self::Fixed(a) => FrequencyGrid::Fixed(a.view()),
            Self::Varying(a) => FrequencyGrid::TimeVarying(a.view()),
        }
    }
}

fn check_kind_axis(kind: &str, axis: i64, ndim: usize) -> PyResult<()> {
    if kind != "linear" {
        return Err(value_err(
            "kind",
            format!("only kind='linear' is supported, got {kind:?}"),
        ));
    }
    if axis != -2 && axis != ndim as i64 - 2 {
        return Err(value_err(
            "axis",
            format!("only the frequency axis -2 is supported, got {axis}"),
        ));
    }
    Ok(())
}

/// Harmonic salience of a spectrogram, equivalent to ``librosa.salience``.
///
/// Args:
///     S (array_like): Non-negative magnitudes, ``(F, T)`` or ``(channels, F, T)``.
///     freqs (array_like): Bin frequencies: 1-D ``(F,)`` or shaped like ``S``
///         (time-varying).
///     harmonics (Sequence[float]): Harmonic multipliers (1 is ``S`` itself).
///     weights (Optional[array_like]): One weight per harmonic (weighted average).
///     aggregate: None / ``numpy.average`` (default), ``numpy.mean``, ``numpy.sum``,
///         ``numpy.max``, ``numpy.min``, ``numpy.median``.
///     filter_peaks (bool): Keep only local maxima along frequency. Default True.
///     fill_value (float): Value of non-peaks. Default NaN.
///     kind (str): Must be ``"linear"``.
///     axis (int): Must be -2.
///
/// Returns:
///     numpy.ndarray[float64]: same shape as ``S``.
///
/// Raises:
///     ValueError: For shape mismatches or unsupported options.
#[pyfunction]
#[pyo3(
    signature = (S, *, freqs, harmonics, weights = None, aggregate = None, filter_peaks = true, fill_value = f64::NAN, kind = "linear", axis = Int(-2)),
    text_signature = "(S, *, freqs, harmonics, weights=None, aggregate=None, filter_peaks=True, fill_value=math.nan, kind='linear', axis=-2)"
)]
#[allow(non_snake_case, clippy::too_many_arguments)]
pub fn salience<'py>(
    py: Python<'py>,
    S: &Bound<'py, PyAny>,
    freqs: &Bound<'py, PyAny>,
    harmonics: &Bound<'py, PyAny>,
    weights: Option<&Bound<'py, PyAny>>,
    aggregate: Option<&Bound<'py, PyAny>>,
    filter_peaks: bool,
    fill_value: f64,
    kind: &str,
    axis: Int,
) -> PyResult<Bound<'py, PyAny>> {
    let (s, nd) = spec3(S, "S")?;
    check_kind_axis(kind, axis.0, nd)?;
    let grid = Grid::parse(freqs, nd)?;
    let h = super::args::f64_vec(harmonics, "harmonics")?;
    let (agg, is_mean) = match aggregate {
        Some(a) => parse_aggregate(a)?,
        None => parse_aggregate(&py.None().into_bound(py))?,
    };
    let weights = match weights {
        Some(w) if !w.is_none() => Some(super::args::f64_vec(w, "weights")?),
        _ => None,
    };
    let cfg = SalienceConfig {
        // numpy.mean ignores weights (librosa passes them only to numpy.average).
        weights: if is_mean { None } else { weights },
        aggregate: agg,
        filter_peaks,
        fill_value,
    };
    let r = pt::salience(s.view(), grid.view(), &h, &cfg).map_err(audio_err_to_py)?;
    Ok(squeeze_out(py, r.into_dyn(), nd == 2))
}

/// Interpolated harmonics of a spectrogram, equivalent to ``librosa.interp_harmonics``.
///
/// Args:
///     x (array_like): Energy, ``(F, T)`` or ``(channels, F, T)``.
///     freqs (array_like): 1-D bin frequencies or an array shaped like ``x``.
///     harmonics (array_like): Harmonic multipliers.
///     kind (str): Must be ``"linear"``.
///     fill_value (float): Value outside the frequency range. Default 0.
///     axis (int): Must be -2.
///
/// Returns:
///     numpy.ndarray[float64]: shape ``(len(harmonics), F, T)`` or
///     ``(channels, len(harmonics), F, T)``.
///
/// Raises:
///     ValueError: For shape mismatches or unsupported options.
#[pyfunction]
#[pyo3(
    signature = (x, *, freqs, harmonics, kind = "linear", fill_value = 0.0, axis = Int(-2)),
    text_signature = "(x, *, freqs, harmonics, kind='linear', fill_value=0, axis=-2)"
)]
pub fn interp_harmonics<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyAny>,
    freqs: &Bound<'py, PyAny>,
    harmonics: &Bound<'py, PyAny>,
    kind: &str,
    fill_value: f64,
    axis: Int,
) -> PyResult<Bound<'py, PyAny>> {
    let (s, nd) = spec3(x, "x")?;
    check_kind_axis(kind, axis.0, nd)?;
    let grid = Grid::parse(freqs, nd)?;
    let h = super::args::f64_vec(harmonics, "harmonics")?;
    let r = pt::interp_harmonics(s.view(), grid.view(), &h, fill_value).map_err(audio_err_to_py)?;
    Ok(squeeze_out(py, r.into_dyn(), nd == 2))
}

/// Energy at the harmonics of a time-varying f0, equivalent to ``librosa.f0_harmonics``.
///
/// Args:
///     x (array_like): Energy, ``(F, T)`` or ``(channels, F, T)``.
///     f0 (array_like): Fundamental per frame, ``(T,)`` or ``(channels, T)``;
///         NaN frames get ``fill_value``.
///     freqs (array_like): 1-D bin frequencies or an array shaped like ``x``.
///     harmonics (array_like): Harmonic multipliers.
///     kind (str): Must be ``"linear"``.
///     fill_value (float): Value for out-of-range targets. Default 0.
///     axis (int): Must be -2.
///
/// Returns:
///     numpy.ndarray[float64]: shape ``(len(harmonics), T)`` or
///     ``(channels, len(harmonics), T)``.
///
/// Raises:
///     ValueError: For shape mismatches or unsupported options.
#[pyfunction]
#[pyo3(
    signature = (x, *, f0, freqs, harmonics, kind = "linear", fill_value = 0.0, axis = Int(-2)),
    text_signature = "(x, *, f0, freqs, harmonics, kind='linear', fill_value=0, axis=-2)"
)]
#[allow(clippy::too_many_arguments)]
pub fn f0_harmonics<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyAny>,
    f0: &Bound<'py, PyAny>,
    freqs: &Bound<'py, PyAny>,
    harmonics: &Bound<'py, PyAny>,
    kind: &str,
    fill_value: f64,
    axis: Int,
) -> PyResult<Bound<'py, PyAny>> {
    let (s, nd) = spec3(x, "x")?;
    check_kind_axis(kind, axis.0, nd)?;
    let grid = Grid::parse(freqs, nd)?;
    let h = super::args::f64_vec(harmonics, "harmonics")?;
    let (f0, f0_nd) = signal2(f0, "f0")?;
    if f0_nd + 1 != nd {
        return Err(value_err(
            "f0",
            format!(
                "expected {} dimensions for an input with {nd} dimensions, got {f0_nd}",
                nd - 1
            ),
        ));
    }
    let r = pt::f0_harmonics(s.view(), f0.view(), grid.view(), &h, fill_value)
        .map_err(audio_err_to_py)?;
    Ok(squeeze_out(py, r.into_dyn(), nd == 2))
}

/// Registers the pitch-tracking functions.
pub(crate) fn add(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(yin, m)?)?;
    m.add_function(wrap_pyfunction!(pyin, m)?)?;
    m.add_function(wrap_pyfunction!(salience, m)?)?;
    m.add_function(wrap_pyfunction!(interp_harmonics, m)?)?;
    m.add_function(wrap_pyfunction!(f0_harmonics, m)?)?;
    Ok(())
}
