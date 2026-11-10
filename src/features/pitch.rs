//! Pitch-tracking free functions of `audio_samples.features` (binding
//! `audio_samples::operations::pitch_tracking`): `yin` / `pyin` on arrays
//! and the harmonic helpers `salience`, `interp_harmonics`,
//! `f0_harmonics`.

use crate::args::Int;
use audio_samples::operations::pitch_tracking::{
    self as pt, FrequencyGrid, PyinConfig, SalienceConfig, YinConfig,
};
use ndarray::{Array1, Array2, Array3, Axis, Ix1, Ix2};
use pyo3::IntoPyObjectExt;
use pyo3::prelude::*;

use super::args::{FillNaArg, parse_aggregate, signal2, spec3, squeeze_out};
use crate::args::{
    FloatElement, float_array_as, opt_positive, parse_pad_mode, positive, precision_of, value_err,
    with_precision,
};
use crate::audio_err_to_py;

/// Builds a [`YinConfig`] from the `yin` keyword arguments.
pub fn yin_config(
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

/// The `pyin` keyword arguments, before validation.
pub struct PyinArgs<'a> {
    pub fmin: f64,
    pub fmax: f64,
    pub frame_length: Int,
    pub hop_length: Option<Int>,
    pub n_thresholds: Int,
    pub beta_parameters: (f64, f64),
    pub boltzmann_parameter: f64,
    pub resolution: f64,
    pub max_transition_rate: f64,
    pub switch_prob: f64,
    pub no_trough_prob: f64,
    pub fill_na: FillNaArg,
    pub center: bool,
    pub pad_mode: &'a str,
    pub transition_min_prob: Option<f64>,
}

impl PyinArgs<'_> {
    /// Validates the arguments into a [`PyinConfig`].
    pub fn config(self) -> PyResult<PyinConfig> {
        Ok(PyinConfig {
            fmin: self.fmin,
            fmax: self.fmax,
            frame_length: positive("frame_length", self.frame_length)?,
            hop_length: opt_positive("hop_length", self.hop_length)?,
            n_thresholds: positive("n_thresholds", self.n_thresholds)?,
            beta_parameters: self.beta_parameters,
            boltzmann_parameter: self.boltzmann_parameter,
            resolution: self.resolution,
            max_transition_rate: self.max_transition_rate,
            switch_prob: self.switch_prob,
            no_trough_prob: self.no_trough_prob,
            fill: self.fill_na.0,
            center: self.center,
            pad_mode: parse_pad_mode(self.pad_mode)?,
            transition_min_prob: self.transition_min_prob,
        })
    }
}

fn rows_out<T: numpy::Element>(py: Python<'_>, a: Array2<T>, ndim_in: usize) -> Bound<'_, PyAny> {
    squeeze_out(py, a.into_dyn(), ndim_in == 1)
}

/// Frame-wise YIN fundamental frequency of an array.
///
/// Args:
///
///     y (array_like): Signal, ``(n,)`` or ``(channels, n)``.
///     fmin (float): Lowest frequency searched (Hz). Required.
///     fmax (float): Highest frequency searched (Hz). Required.
///     sr (float): Sample rate. Required.
///     frame_length (int): Samples per frame. Default 2048.
///     hop_length (Optional[int]): None = ``frame_length // 4``.
///     trough_threshold (float): Absolute CMND threshold. Default 0.1.
///     center (bool): Centre frames on ``t * hop_length``. Default True.
///     pad_mode (str): Padding for ``center``. Default ``"constant"``.
///
/// Returns:
///
///     numpy.ndarray[float32 | float64]: f0 in Hz, shape ``(n_frames,)`` or ``(channels, n_frames)``.
///
/// Raises:
///
///     ValueError: For invalid parameters or a too-short signal.
#[pyfunction]
#[pyo3(
    signature = (y: "ArrayLike", *, fmin, fmax, sr, frame_length = Int(2048), hop_length = None, trough_threshold = 0.1, center = true, pad_mode = "constant") -> "NDArray[Any]",
    text_signature = "(y, *, fmin, fmax, sr, frame_length=2048, hop_length=None, trough_threshold=0.1, center=True, pad_mode='constant')"
)]
#[allow(
    clippy::too_many_arguments,
    reason = "the Python signature has 9 parameters"
)]
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
    let cfg = yin_config(
        fmin,
        fmax,
        frame_length,
        hop_length,
        trough_threshold,
        center,
        pad_mode,
    )?;
    with_precision!(precision_of(y), |F| {
        let (x, nd) = signal2::<F>(y, "y")?;
        let x = x.view();
        let f0 = py
            .detach(|| pt::yin(x, sr, &cfg))
            .map_err(audio_err_to_py)?;
        Ok(rows_out(py, f0, nd))
    })
}

/// Probabilistic YIN of an array: f0 tracking with voicing decisions.
///
/// Args:
///
///     y (array_like): Signal, ``(n,)`` or ``(channels, n)``.
///     fmin, fmax (float): Search range in Hz. Required.
///     sr (float): Sample rate. Required.
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
///
///     tuple: ``(f0, voiced_flag, voiced_prob)``, each ``(n_frames,)`` or
///     ``(channels, n_frames)``; ``voiced_flag`` is ``bool``, the others in ``y``'s precision.
///
/// Raises:
///
///     ValueError: For invalid parameters or a too-short signal.
#[pyfunction]
#[pyo3(
    signature = (y: "ArrayLike", *, fmin, fmax, sr, frame_length = Int(2048), hop_length = None, n_thresholds = Int(100), beta_parameters = (2.0, 18.0), boltzmann_parameter = 2.0, resolution = 0.1, max_transition_rate = 35.92, switch_prob = 0.01, no_trough_prob = 0.01, fill_na = FillNaArg(audio_samples::operations::pitch_tracking::UnvoicedFill::Nan), center = true, pad_mode = "constant", transition_min_prob = Some(1e-4)) -> "tuple[NDArray[Any], NDArray[Any], NDArray[Any]] | tuple[NDArray[np.float32], NDArray[np.bool_], NDArray[np.float32]] | tuple[NDArray[np.float64], NDArray[np.bool_], NDArray[np.float64]]",
    text_signature = "(y, *, fmin, fmax, sr, frame_length=2048, hop_length=None, n_thresholds=100, beta_parameters=(2, 18), boltzmann_parameter=2, resolution=0.1, max_transition_rate=35.92, switch_prob=0.01, no_trough_prob=0.01, fill_na=math.nan, center=True, pad_mode='constant', transition_min_prob=0.0001)"
)]
#[allow(
    clippy::too_many_arguments,
    reason = "the Python signature has 17 parameters"
)]
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
    with_precision!(precision_of(y), |F| {
        let (x, nd) = signal2::<F>(y, "y")?;
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
    })
}

/// A `freqs` argument: 1-D (fixed grid) or shaped like the spectrogram.
enum Grid<F> {
    Fixed(Array1<F>),
    Varying(Array3<F>),
}

impl<F: FloatElement> Grid<F> {
    fn parse(obj: &Bound<'_, PyAny>, s_ndim: usize) -> PyResult<Self> {
        let a = float_array_as::<F>(obj, "freqs")?;
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

    fn view(&self) -> FrequencyGrid<'_, F> {
        match self {
            Self::Fixed(a) => FrequencyGrid::Fixed(a.view()),
            Self::Varying(a) => FrequencyGrid::TimeVarying(a.view()),
        }
    }
}

/// Rejects what the salience kernel does not provide: another interpolant or axis.
fn check_kind_axis(kind: &str, axis: i64, ndim: usize) -> PyResult<()> {
    if !native_kind_axis(kind, axis, ndim) {
        return Err(value_err(
            "kind",
            "salience uses linear interpolation on the frequency axis; use interp_harmonics for others",
        ));
    }
    Ok(())
}

/// Whether the native harmonic kernels apply: linear interpolation on the frequency axis.
fn native_kind_axis(kind: &str, axis: i64, ndim: usize) -> bool {
    kind == "linear" && (axis == -2 || usize::try_from(axis).is_ok_and(|a| a + 2 == ndim))
}

/// `harmonics * freqs` by any interpolant along any axis of `x`.
fn interp_harmonics_any<'py, F: crate::args::PyFloat>(
    x: &Bound<'py, PyAny>,
    freqs: &Bound<'py, PyAny>,
    harmonics: &[f64],
    kind: &str,
    fill_value: f64,
    axis: i64,
) -> PyResult<Bound<'py, PyAny>> {
    use audio_samples::utils::array_ops as ops;
    use numpy::IntoPyArray;
    let py = x.py();
    let parsed = crate::arrays::parse_kind(pyo3::types::PyString::new(py, kind).as_any())?;
    crate::args::with_float_view::<F, _>(x, "x", |v| {
        let ax = crate::arrays::norm_axis("axis", axis, v.ndim())?;
        let grid = super::args::f64_vec1(freqs, "freqs").map_err(|_| {
            value_err(
                "freqs",
                "other interpolants and axes need a 1-D frequency grid",
            )
        })?;
        let n = grid.len();
        let mut xi = ndarray::Array2::<f64>::zeros((harmonics.len(), n));
        for (mut row, &h) in xi.rows_mut().into_iter().zip(harmonics) {
            for (o, &f) in row.iter_mut().zip(&grid) {
                *o = h * f;
            }
        }
        let out = ops::interp1d(
            &grid,
            v,
            xi.view().into_dyn(),
            parsed,
            <F as crate::args::FloatElement>::from_f64(fill_value),
            ax,
        )
        .map_err(audio_err_to_py)?;
        Ok(out.into_pyarray(py).into_any())
    })
}

/// The energy at the harmonics of an f0 per frame, by any interpolant along any axis of `x`.
fn f0_harmonics_any<'py, F: crate::args::PyFloat>(
    x: &Bound<'py, PyAny>,
    f0: &Bound<'py, PyAny>,
    freqs: &Bound<'py, PyAny>,
    harmonics: &[f64],
    kind: &str,
    fill_value: f64,
    axis: i64,
) -> PyResult<Bound<'py, PyAny>> {
    use audio_samples::utils::array_ops as ops;
    use ndarray::Axis;
    use numpy::IntoPyArray;
    let py = x.py();
    let parsed = crate::arrays::parse_kind(pyo3::types::PyString::new(py, kind).as_any())?;
    let grid = super::args::f64_vec1(freqs, "freqs").map_err(|_| {
        value_err(
            "freqs",
            "other interpolants and axes need a 1-D frequency grid",
        )
    })?;
    let positions: Vec<f64> = grid.iter().copied().filter(|f| f.is_finite()).collect();
    let f0_values = super::args::f64_vec(f0, "f0")?;
    crate::args::with_float_view::<F, _>(x, "x", |v| {
        let ax = crate::arrays::norm_axis("axis", axis, v.ndim())?;
        let mut v_lane_first = v.view();
        v_lane_first.swap_axes(ax.index(), v.ndim() - 1);
        // Lanes along the last axis are (..., F); one f0 per lane.
        let lanes = v_lane_first.lanes(Axis(v.ndim() - 1));
        let count = lanes.clone().into_iter().count();
        if count != f0_values.len() {
            return Err(value_err(
                "f0",
                format!("{} values for {count} frames", f0_values.len()),
            ));
        }
        let mut shape = v_lane_first.shape().to_vec();
        shape.pop();
        shape.push(harmonics.len());
        let mut flat = Vec::with_capacity(count * harmonics.len());
        let fill = <F as crate::args::FloatElement>::from_f64(fill_value);
        for (lane, &f) in lanes.into_iter().zip(&f0_values) {
            let data: ndarray::Array1<F> = lane
                .iter()
                .zip(&grid)
                .filter(|(_, f)| f.is_finite())
                .map(|(&e, _)| e)
                .collect();
            let xi: ndarray::Array1<f64> = harmonics.iter().map(|&h| h * f).collect();
            let y = ops::interp1d(
                &positions,
                data.into_dyn().view(),
                xi.into_dyn().view(),
                parsed,
                fill,
                Axis(0),
            )
            .map_err(audio_err_to_py)?;
            flat.extend(y.iter().map(|&e| if e.is_finite() { e } else { fill }));
        }
        let mut out = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&shape), flat)
            .map_err(|e| value_err("x", e))?;
        out.swap_axes(ax.index(), v.ndim() - 1);
        Ok(out
            .as_standard_layout()
            .into_owned()
            .into_pyarray(py)
            .into_any())
    })
}

/// Harmonic salience of a spectrogram: the weighted sum of the energy at the
/// harmonics of every bin.
///
/// Args:
///
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
///
///     numpy.ndarray[float32 | float64]: same shape as ``S``.
///
/// Raises:
///
///     ValueError: For shape mismatches or unsupported options.
#[pyfunction]
#[pyo3(
    signature = (S: "ArrayLike", *, freqs: "ArrayLike", harmonics: "Sequence[float]", weights = None, aggregate = None, filter_peaks = true, fill_value = f64::NAN, kind = "linear", axis = Int(-2)) -> "NDArray[Any]",
    text_signature = "(S, *, freqs, harmonics, weights=None, aggregate=None, filter_peaks=True, fill_value=math.nan, kind='linear', axis=-2)"
)]
#[allow(
    non_snake_case,
    clippy::too_many_arguments,
    reason = "the Python signature has 9 parameters"
)]
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
        // Only the weighted average uses the weights; numpy.mean ignores them.
        weights: if is_mean { None } else { weights },
        aggregate: agg,
        filter_peaks,
        fill_value,
    };
    with_precision!(precision_of(S), |F| {
        let (s, nd) = spec3::<F>(S, "S")?;
        check_kind_axis(kind, axis.0, nd)?;
        let grid = Grid::<F>::parse(freqs, nd)?;
        let r = pt::salience(s.view(), grid.view(), &h, &cfg).map_err(audio_err_to_py)?;
        Ok(squeeze_out(py, r.into_dyn(), nd == 2))
    })
}

/// Energy of a spectrogram interpolated at the harmonics of every bin.
///
/// Args:
///
///     x (array_like): Energy, ``(F, T)`` or ``(channels, F, T)``.
///     freqs (array_like): 1-D bin frequencies or an array shaped like ``x``.
///     harmonics (array_like): Harmonic multipliers.
///     kind (str | int): Interpolation kind, as ``arrays.interp1d``. Default ``"linear"``.
///     fill_value (float): Value outside the frequency range. Default 0.
///     axis (int): The frequency axis. Default -2.
///
/// Returns:
///
///     numpy.ndarray[float32 | float64]: shape ``(len(harmonics), F, T)`` or
///     ``(channels, len(harmonics), F, T)``.
///
/// Raises:
///
///     ValueError: For shape mismatches or unsupported options.
#[pyfunction]
#[pyo3(
    signature = (x: "ArrayLike", *, freqs: "ArrayLike", harmonics: "ArrayLike", kind = "linear", fill_value = 0.0, axis = Int(-2)) -> "NDArray[Any]",
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
    let h = super::args::f64_vec(harmonics, "harmonics")?;
    let nd = x
        .cast::<numpy::PyUntypedArray>()
        .map_or(2, numpy::PyUntypedArrayMethods::ndim);
    if !native_kind_axis(kind, axis.0, nd) {
        return with_precision!(precision_of(x), |F| {
            interp_harmonics_any::<F>(x, freqs, &h, kind, fill_value, axis.0)
        });
    }
    with_precision!(precision_of(x), |F| {
        let (s, nd) = spec3::<F>(x, "x")?;
        let grid = Grid::<F>::parse(freqs, nd)?;
        let r =
            pt::interp_harmonics(s.view(), grid.view(), &h, fill_value).map_err(audio_err_to_py)?;
        Ok(squeeze_out(py, r.into_dyn(), nd == 2))
    })
}

/// Energy at the harmonics of a time-varying f0.
///
/// Args:
///
///     x (array_like): Energy, ``(F, T)`` or ``(channels, F, T)``.
///     f0 (array_like): Fundamental per frame, ``(T,)`` or ``(channels, T)``;
///         NaN frames get ``fill_value``.
///     freqs (array_like): 1-D bin frequencies or an array shaped like ``x``.
///     harmonics (array_like): Harmonic multipliers.
///     kind (str): The interpolant, as for ``arrays.interp1d``. Default ``"linear"``.
///     fill_value (float): Value for out-of-range targets. Default 0.
///     axis (int): The frequency axis. Default -2.
///
/// Returns:
///
///     numpy.ndarray[float32 | float64]: shape ``(len(harmonics), T)`` or
///     ``(channels, len(harmonics), T)``.
///
/// Raises:
///
///     ValueError: For shape mismatches or unsupported options.
#[pyfunction]
#[pyo3(
    signature = (x: "ArrayLike", *, f0: "ArrayLike", freqs: "ArrayLike", harmonics: "ArrayLike", kind = "linear", fill_value = 0.0, axis = Int(-2)) -> "NDArray[Any]",
    text_signature = "(x, *, f0, freqs, harmonics, kind='linear', fill_value=0, axis=-2)"
)]
#[allow(
    clippy::too_many_arguments,
    reason = "the Python signature has 7 parameters"
)]
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
    let h = super::args::f64_vec(harmonics, "harmonics")?;
    let nd = x
        .cast::<numpy::PyUntypedArray>()
        .map_or(2, numpy::PyUntypedArrayMethods::ndim);
    if !native_kind_axis(kind, axis.0, nd) {
        return with_precision!(precision_of(x), |F| {
            f0_harmonics_any::<F>(x, f0, freqs, &h, kind, fill_value, axis.0)
        });
    }
    with_precision!(precision_of(x), |F| {
        let (s, nd) = spec3::<F>(x, "x")?;
        let grid = Grid::<F>::parse(freqs, nd)?;
        let (f0, f0_nd) = signal2::<F>(f0, "f0")?;
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
    })
}
