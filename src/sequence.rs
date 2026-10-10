//! `audio_samples.sequence`: DTW, Viterbi decoding, transition matrices,
//! recurrence/similarity matrices and `sync`, binding
//! `audio_samples::operations::sequence` (librosa's `sequence`, `segment` and
//! `util.sync`).
//!
//! Feature matrices use librosa's layout `(n_features, n_frames)`; 1-D input is
//! treated as one feature (`(1, n_frames)`), as librosa's `numpy.atleast_2d`.
//! Return values follow librosa: tuples only when the corresponding flag asks
//! for them (`backtrack`, `return_steps`, `return_logp`).

use crate::args::Int;
use audio_samples::operations::sequence::{
    self as seq, Bandwidth, CrossSimilarityConfig, DistanceMetric, DtwConfig, RecurrenceConfig,
    RecurrenceMode, RqaConfig, SyncAggregate, TransitionWindow,
};
use ndarray::{Array1, Array2, Axis, Ix1, Ix2, Ix3};
use numpy::{IntoPyArray, PyArray2, PyUntypedArrayMethods};
use pyo3::IntoPyObjectExt;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyString, PyTuple};

use crate::args::{f64_array, i64_vec, positive, value_err};
use crate::audio_err_to_py;

/// A 1-D or 2-D float array as an owned `(rows, cols)` matrix (1-D -> `(1, n)`).
fn matrix(obj: &Bound<'_, PyAny>, name: &str) -> PyResult<Array2<f64>> {
    let a = f64_array(obj, name)?;
    let a = a.as_array();
    match a.ndim() {
        1 => Ok(a
            .into_dimensionality::<Ix1>()
            .expect("1-D")
            .insert_axis(Axis(0))
            .to_owned()),
        2 => Ok(a.into_dimensionality::<Ix2>().expect("2-D").to_owned()),
        n => Err(value_err(
            name,
            format!("expected a 1-D or 2-D array (n_features, n_frames), got {n} dimensions"),
        )),
    }
}

/// A strictly 2-D float array.
fn matrix2(obj: &Bound<'_, PyAny>, name: &str) -> PyResult<Array2<f64>> {
    let a = f64_array(obj, name)?;
    if a.ndim() != 2 {
        return Err(value_err(
            name,
            format!("expected a 2-D array, got {} dimensions", a.ndim()),
        ));
    }
    Ok(a.as_array()
        .into_dimensionality::<Ix2>()
        .expect("2-D")
        .to_owned())
}

/// An optional 1-D float vector.
fn opt_vec(obj: Option<&Bound<'_, PyAny>>, name: &str) -> PyResult<Option<Array1<f64>>> {
    obj.filter(|o| !o.is_none())
        .map(|o| {
            let a = f64_array(o, name)?;
            Ok(a.as_array().iter().copied().collect::<Array1<f64>>())
        })
        .transpose()
}

/// Parses a `scipy.spatial.distance` metric name.
pub(crate) fn parse_metric(metric: &str) -> PyResult<DistanceMetric> {
    match metric {
        "euclidean" => Ok(DistanceMetric::Euclidean),
        "sqeuclidean" => Ok(DistanceMetric::SqEuclidean),
        "cosine" => Ok(DistanceMetric::Cosine),
        "cityblock" | "manhattan" => Ok(DistanceMetric::Cityblock),
        "chebyshev" => Ok(DistanceMetric::Chebyshev),
        other => Err(value_err(
            "metric",
            format!(
                "unsupported metric {other:?}; expected 'euclidean', 'sqeuclidean', 'cosine', 'cityblock' or 'chebyshev'"
            ),
        )),
    }
}

/// `(n, 2)` step list from an array-like of `[row_step, col_step]` pairs.
fn parse_steps(obj: Option<&Bound<'_, PyAny>>) -> PyResult<Option<Vec<[usize; 2]>>> {
    let Some(obj) = obj.filter(|o| !o.is_none()) else {
        return Ok(None);
    };
    let v = i64_vec(obj, "step_sizes_sigma")?;
    let a = f64_array(obj, "step_sizes_sigma")?;
    if a.ndim() != 2 || a.shape()[1] != 2 {
        return Err(value_err(
            "step_sizes_sigma",
            "expected an array of shape (n_steps, 2)",
        ));
    }
    v.chunks(2)
        .map(|c| {
            let r = usize::try_from(c[0]);
            let s = usize::try_from(c[1]);
            match (r, s) {
                (Ok(r), Ok(s)) => Ok([r, s]),
                _ => Err(value_err("step_sizes_sigma", "steps must be non-negative")),
            }
        })
        .collect::<PyResult<Vec<_>>>()
        .map(Some)
}

/// A warping path as an `(n, 2)` int64 array.
fn path_array<'py>(py: Python<'py>, path: &[[usize; 2]]) -> Bound<'py, PyArray2<i64>> {
    let mut a = Array2::<i64>::zeros((path.len(), 2));
    for (i, p) in path.iter().enumerate() {
        a[[i, 0]] = p[0] as i64;
        a[[i, 1]] = p[1] as i64;
    }
    a.into_pyarray(py)
}

/// Dynamic time warping, equivalent to ``librosa.sequence.dtw``.
///
/// Pass either the feature sequences ``X`` and ``Y`` (``(n_features, N)`` and
/// ``(n_features, M)``; 1-D input is one feature), or a precomputed cost
/// matrix ``C`` of shape ``(N, M)``.
///
/// Args:
///     X, Y (Optional[array_like]): Feature sequences.
///     C (Optional[array_like]): Precomputed local cost matrix.
///     metric (str): ``"euclidean"`` (default), ``"sqeuclidean"``, ``"cosine"``,
///         ``"cityblock"`` or ``"chebyshev"`` (used with X/Y only).
///     step_sizes_sigma (Optional[array_like]): Extra steps, shape ``(n, 2)``;
///         appended to librosa's defaults ``[[1, 1], [0, 1], [1, 0]]``.
///     weights_add, weights_mul (Optional[array_like]): Per-step weights.
///     subseq (bool): Subsequence alignment. Default False.
///     backtrack (bool): Compute the warping path. Default True.
///     global_constraints (bool): Sakoe-Chiba band. Default False.
///     band_rad (float): Band radius (fraction of the shorter length). Default 0.25.
///     return_steps (bool): Also return the step matrix. Default False.
///
/// Returns:
///     ``D`` (numpy.ndarray[float64], shape ``(N, M)``); with ``backtrack=True``
///     the tuple ``(D, wp)`` where ``wp`` is an int64 array of shape
///     ``(path_length, 2)`` ordered from the end of the alignment to its start;
///     with ``return_steps=True`` the step matrix (int64, same shape as ``D``)
///     is appended. Exactly librosa's return convention.
///
/// Raises:
///     ValueError: Incompatible inputs, NaN costs, or no valid path.
#[pyfunction]
#[pyo3(
    signature = (X = None, Y = None, *, C = None, metric = "euclidean", step_sizes_sigma = None, weights_add = None, weights_mul = None, subseq = false, backtrack = true, global_constraints = false, band_rad = 0.25, return_steps = false),
    text_signature = "(X=None, Y=None, *, C=None, metric='euclidean', step_sizes_sigma=None, weights_add=None, weights_mul=None, subseq=False, backtrack=True, global_constraints=False, band_rad=0.25, return_steps=False)"
)]
#[allow(non_snake_case, clippy::too_many_arguments)]
pub fn dtw<'py>(
    py: Python<'py>,
    X: Option<&Bound<'py, PyAny>>,
    Y: Option<&Bound<'py, PyAny>>,
    C: Option<&Bound<'py, PyAny>>,
    metric: &str,
    step_sizes_sigma: Option<&Bound<'py, PyAny>>,
    weights_add: Option<&Bound<'py, PyAny>>,
    weights_mul: Option<&Bound<'py, PyAny>>,
    subseq: bool,
    backtrack: bool,
    global_constraints: bool,
    band_rad: f64,
    return_steps: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let cfg = DtwConfig {
        metric: parse_metric(metric)?,
        step_sizes_sigma: parse_steps(step_sizes_sigma)?,
        weights_add: opt_vec(weights_add, "weights_add")?.map(|a| a.to_vec()),
        weights_mul: opt_vec(weights_mul, "weights_mul")?.map(|a| a.to_vec()),
        subseq,
        backtrack,
        global_constraints,
        band_rad,
    };
    let res = match (X, Y, C) {
        (Some(x), Some(y), None) => {
            let x = matrix(x, "X")?;
            let y = matrix(y, "Y")?;
            seq::dtw(x.view(), y.view(), &cfg)
        }
        (None, None, Some(c)) => {
            let c = matrix2(c, "C")?;
            seq::dtw_with_cost(c.view(), &cfg)
        }
        _ => {
            return Err(PyValueError::new_err(
                "dtw: pass either both X and Y, or C (and not both)",
            ));
        }
    }
    .map_err(audio_err_to_py)?;

    let d = res.cost.into_pyarray(py).into_any();
    let mut out = vec![d];
    if backtrack {
        let path = res.path.unwrap_or_default();
        out.push(path_array(py, &path).into_any());
    }
    if return_steps {
        out.push(res.steps.mapv(|s| s as i64).into_pyarray(py).into_any());
    }
    if out.len() == 1 {
        Ok(out.pop().expect("one element"))
    } else {
        Ok(PyTuple::new(py, out)?.into_any())
    }
}

/// Backtrack a warping path from a DTW step matrix, equivalent to
/// ``librosa.sequence.dtw_backtracking``.
///
/// Args:
///     steps (array_like[int]): Step matrix from ``dtw(..., return_steps=True)``.
///     step_sizes_sigma (Optional[array_like]): The custom steps passed to ``dtw``.
///     subseq (bool): Whether ``dtw`` ran with ``subseq=True``.
///     start (Optional[int]): Column of the last row to start from (subseq only).
///
/// Returns:
///     numpy.ndarray[int64]: shape ``(path_length, 2)``, end first.
#[pyfunction]
#[pyo3(
    signature = (steps, *, step_sizes_sigma = None, subseq = false, start = None),
    text_signature = "(steps, *, step_sizes_sigma=None, subseq=False, start=None)"
)]
pub fn dtw_backtracking<'py>(
    py: Python<'py>,
    steps: &Bound<'py, PyAny>,
    step_sizes_sigma: Option<&Bound<'py, PyAny>>,
    subseq: bool,
    start: Option<Int>,
) -> PyResult<Bound<'py, PyArray2<i64>>> {
    let a = f64_array(steps, "steps")?;
    if a.ndim() != 2 {
        return Err(value_err("steps", "expected a 2-D step matrix"));
    }
    let shape = (a.shape()[0], a.shape()[1]);
    let v = i64_vec(steps, "steps")?;
    let v: Vec<usize> = v
        .into_iter()
        .map(|s| usize::try_from(s).map_err(|_| value_err("steps", "step indices must be >= 0")))
        .collect::<PyResult<_>>()?;
    let steps = Array2::from_shape_vec(shape, v).map_err(|e| value_err("steps", e))?;
    let custom = parse_steps(step_sizes_sigma)?;
    let start = start
        .map(|s| usize::try_from(s.0).map_err(|_| value_err("start", "must be >= 0")))
        .transpose()?;
    let path = seq::dtw_backtracking(steps.view(), custom.as_deref(), subseq, start)
        .map_err(audio_err_to_py)?;
    Ok(path_array(py, &path))
}

/// Return `states` (int64) or `(states, logp)`.
fn states_out<'py, D: ndarray::Dimension>(
    py: Python<'py>,
    states: ndarray::Array<i64, D>,
    logp: Bound<'py, PyAny>,
    return_logp: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let s = states.into_pyarray(py).into_any();
    if return_logp {
        Ok(PyTuple::new(py, [s, logp])?.into_any())
    } else {
        Ok(s)
    }
}

/// Viterbi decoding of a generative HMM, equivalent to ``librosa.sequence.viterbi``.
///
/// Args:
///     prob (array_like): ``(n_states, n_steps)`` observation likelihoods in [0, 1].
///     transition (array_like): ``(n_states, n_states)`` row-stochastic matrix.
///     p_init (Optional[array_like]): Initial distribution; None = uniform.
///     return_logp (bool): Also return the log-likelihood. Default False.
///     transition_min_prob (Optional[float]): Prune transitions below this
///         probability (approximate, faster). None = exact.
///
/// Returns:
///     numpy.ndarray[int64] of shape ``(n_steps,)`` (librosa returns uint16), or
///     ``(states, logp)`` with ``logp`` a float when ``return_logp=True``.
///
/// Raises:
///     ValueError: Shape mismatch or invalid probabilities.
#[pyfunction]
#[pyo3(
    signature = (prob, transition, *, p_init = None, return_logp = false, transition_min_prob = None),
    text_signature = "(prob, transition, *, p_init=None, return_logp=False, transition_min_prob=None)"
)]
pub fn viterbi<'py>(
    py: Python<'py>,
    prob: &Bound<'py, PyAny>,
    transition: &Bound<'py, PyAny>,
    p_init: Option<&Bound<'py, PyAny>>,
    return_logp: bool,
    transition_min_prob: Option<f64>,
) -> PyResult<Bound<'py, PyAny>> {
    let prob = matrix2(prob, "prob")?;
    let trans = matrix2(transition, "transition")?;
    let p_init = opt_vec(p_init, "p_init")?;
    let r = seq::viterbi(
        prob.view(),
        trans.view(),
        p_init.as_ref().map(|a| a.view()),
        transition_min_prob,
    )
    .map_err(audio_err_to_py)?;
    let states: Array1<i64> = r.states.iter().map(|&s| s as i64).collect();
    states_out(py, states, r.log_prob.into_bound_py_any(py)?, return_logp)
}

/// Viterbi decoding from discriminative (posterior) state probabilities,
/// equivalent to ``librosa.sequence.viterbi_discriminative``.
///
/// Args:
///     prob (array_like): ``(n_states, n_steps)``; each column sums to 1.
///     transition (array_like): ``(n_states, n_states)`` row-stochastic matrix.
///     p_state (Optional[array_like]): Marginal state distribution; None = uniform.
///     p_init (Optional[array_like]): Initial distribution; None = uniform.
///     return_logp (bool): Also return the (unnormalised) log-likelihood.
///     transition_min_prob (Optional[float]): As ``viterbi``.
///
/// Returns:
///     numpy.ndarray[int64] ``(n_steps,)`` or ``(states, logp)``.
#[pyfunction]
#[pyo3(
    signature = (prob, transition, *, p_state = None, p_init = None, return_logp = false, transition_min_prob = None),
    text_signature = "(prob, transition, *, p_state=None, p_init=None, return_logp=False, transition_min_prob=None)"
)]
pub fn viterbi_discriminative<'py>(
    py: Python<'py>,
    prob: &Bound<'py, PyAny>,
    transition: &Bound<'py, PyAny>,
    p_state: Option<&Bound<'py, PyAny>>,
    p_init: Option<&Bound<'py, PyAny>>,
    return_logp: bool,
    transition_min_prob: Option<f64>,
) -> PyResult<Bound<'py, PyAny>> {
    let prob = matrix2(prob, "prob")?;
    let trans = matrix2(transition, "transition")?;
    let p_state = opt_vec(p_state, "p_state")?;
    let p_init = opt_vec(p_init, "p_init")?;
    let r = seq::viterbi_discriminative(
        prob.view(),
        trans.view(),
        p_state.as_ref().map(|a| a.view()),
        p_init.as_ref().map(|a| a.view()),
        transition_min_prob,
    )
    .map_err(audio_err_to_py)?;
    let states: Array1<i64> = r.states.iter().map(|&s| s as i64).collect();
    states_out(py, states, r.log_prob.into_bound_py_any(py)?, return_logp)
}

/// Independent binary (off/on) Viterbi decoding per state, equivalent to
/// ``librosa.sequence.viterbi_binary``.
///
/// Args:
///     prob (array_like): ``(n_states, n_steps)`` activation probabilities
///         (a 1-D array is one state).
///     transition (array_like): ``(2, 2)`` shared or ``(n_states, 2, 2)`` per state.
///     p_state (Optional[float | array_like]): Marginal P[on] per state; None = 0.5.
///     p_init (Optional[float | array_like]): Initial P[on] per state; None = 0.5.
///     return_logp (bool): Also return per-state log-likelihoods.
///     transition_min_prob (Optional[float]): As ``viterbi``.
///
/// Returns:
///     numpy.ndarray[int64] ``(n_states, n_steps)`` of 0/1, or
///     ``(states, logp)`` with ``logp`` of shape ``(n_states,)``.
#[pyfunction]
#[pyo3(
    signature = (prob, transition, *, p_state = None, p_init = None, return_logp = false, transition_min_prob = None),
    text_signature = "(prob, transition, *, p_state=None, p_init=None, return_logp=False, transition_min_prob=None)"
)]
pub fn viterbi_binary<'py>(
    py: Python<'py>,
    prob: &Bound<'py, PyAny>,
    transition: &Bound<'py, PyAny>,
    p_state: Option<&Bound<'py, PyAny>>,
    p_init: Option<&Bound<'py, PyAny>>,
    return_logp: bool,
    transition_min_prob: Option<f64>,
) -> PyResult<Bound<'py, PyAny>> {
    let one_d = f64_array(prob, "prob")?.ndim() == 1;
    let prob = matrix(prob, "prob")?;
    let n_states = prob.nrows();
    let t = f64_array(transition, "transition")?;
    let t = t.as_array();
    let t3 = match t.ndim() {
        2 => t
            .into_dimensionality::<Ix2>()
            .expect("2-D")
            .insert_axis(Axis(0))
            .to_owned(),
        3 => t.into_dimensionality::<Ix3>().expect("3-D").to_owned(),
        n => {
            return Err(value_err(
                "transition",
                format!("expected shape (2, 2) or (n_states, 2, 2), got {n} dimensions"),
            ));
        }
    };
    // librosa broadcasts a scalar p_state / p_init to every state.
    let broadcast = |v: Option<Array1<f64>>| -> Option<Array1<f64>> {
        v.map(|a| {
            if a.len() == 1 && n_states > 1 {
                Array1::from_elem(n_states, a[0])
            } else {
                a
            }
        })
    };
    let p_state = broadcast(opt_vec(p_state, "p_state")?);
    let p_init = broadcast(opt_vec(p_init, "p_init")?);
    let r = seq::viterbi_binary(
        prob.view(),
        t3.view(),
        p_state.as_ref().map(|a| a.view()),
        p_init.as_ref().map(|a| a.view()),
        transition_min_prob,
    )
    .map_err(audio_err_to_py)?;
    let states = r.states.mapv(|s| s as i64);
    if one_d {
        let s1 = states.index_axis_move(Axis(0), 0);
        let lp = r.log_prob[0].into_bound_py_any(py)?;
        return states_out(py, s1, lp, return_logp);
    }
    let lp = r.log_prob.into_pyarray(py).into_any();
    states_out(py, states, lp, return_logp)
}

/// Uniform transition matrix, equivalent to ``librosa.sequence.transition_uniform``.
///
/// Returns:
///     numpy.ndarray[float64]: ``(n_states, n_states)`` filled with ``1 / n_states``.
#[pyfunction]
#[pyo3(signature = (n_states), text_signature = "(n_states)")]
pub fn transition_uniform(py: Python<'_>, n_states: Int) -> PyResult<Bound<'_, PyArray2<f64>>> {
    let t =
        seq::transition_uniform(positive("n_states", n_states)?.get()).map_err(audio_err_to_py)?;
    Ok(t.into_pyarray(py))
}

/// A float or a sequence of floats as a vector.
fn float_or_vec(obj: &Bound<'_, PyAny>, name: &str) -> PyResult<Vec<f64>> {
    Ok(f64_array(obj, name)?.as_array().iter().copied().collect())
}

/// Self-loop transition matrix, equivalent to ``librosa.sequence.transition_loop``.
///
/// Args:
///     n_states (int): Number of states (> 1).
///     prob (float | array_like): Self-loop probability, one value or one per state.
///
/// Returns:
///     numpy.ndarray[float64]: ``(n_states, n_states)``.
#[pyfunction]
#[pyo3(signature = (n_states, prob), text_signature = "(n_states, prob)")]
pub fn transition_loop<'py>(
    py: Python<'py>,
    n_states: Int,
    prob: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let n = positive("n_states", n_states)?.get();
    let t = seq::transition_loop(n, &float_or_vec(prob, "prob")?).map_err(audio_err_to_py)?;
    Ok(t.into_pyarray(py))
}

/// Cyclic transition matrix, equivalent to ``librosa.sequence.transition_cycle``.
///
/// Args:
///     n_states (int): Number of states (> 1).
///     prob (float | array_like): Self-loop probability, one value or one per state.
///
/// Returns:
///     numpy.ndarray[float64]: ``(n_states, n_states)``.
#[pyfunction]
#[pyo3(signature = (n_states, prob), text_signature = "(n_states, prob)")]
pub fn transition_cycle<'py>(
    py: Python<'py>,
    n_states: Int,
    prob: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let n = positive("n_states", n_states)?.get();
    let t = seq::transition_cycle(n, &float_or_vec(prob, "prob")?).map_err(audio_err_to_py)?;
    Ok(t.into_pyarray(py))
}

/// Localised transition matrix, equivalent to ``librosa.sequence.transition_local``.
///
/// Args:
///     n_states (int): Number of states (> 1).
///     width (int | array_like[int]): Neighbourhood width, one value or one per state.
///     window (str): ``"triangle"`` (default, also ``"triang"``), ``"hann"``,
///         ``"hamming"``, ``"blackman"`` or ``"boxcar"``.
///     wrap (bool): Circular state space. Default False.
///
/// Returns:
///     numpy.ndarray[float64]: ``(n_states, n_states)``, row-stochastic.
#[pyfunction]
#[pyo3(
    signature = (n_states, width, *, window = "triangle", wrap = false),
    text_signature = "(n_states, width, *, window='triangle', wrap=False)"
)]
pub fn transition_local<'py>(
    py: Python<'py>,
    n_states: Int,
    width: &Bound<'py, PyAny>,
    window: &str,
    wrap: bool,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let window = match window {
        "triangle" | "triang" => TransitionWindow::Triangle,
        "hann" | "hanning" => TransitionWindow::Hann,
        "hamming" => TransitionWindow::Hamming,
        "blackman" => TransitionWindow::Blackman,
        "boxcar" | "rectangular" | "ones" => TransitionWindow::Boxcar,
        other => {
            return Err(value_err(
                "window",
                format!(
                    "unsupported window {other:?}; expected 'triangle', 'hann', 'hamming', 'blackman' or 'boxcar'"
                ),
            ));
        }
    };
    let widths: Vec<usize> = i64_vec(width, "width")?
        .into_iter()
        .map(|w| positive("width", w).map(std::num::NonZeroUsize::get))
        .collect::<PyResult<_>>()?;
    let n = positive("n_states", n_states)?.get();
    let t = seq::transition_local(n, &widths, window, wrap).map_err(audio_err_to_py)?;
    Ok(t.into_pyarray(py))
}

/// Recurrence quantification analysis, equivalent to ``librosa.sequence.rqa``.
///
/// Args:
///     sim (array_like): ``(N, M)`` similarity / recurrence matrix (bool or float).
///     gap_onset (float): Gap opening penalty. Default 1.
///     gap_extend (float): Gap extension penalty. Default 1.
///     knight_moves (bool): Allow (1, 2) / (2, 1) moves. Default True.
///     backtrack (bool): Return the best path too. Default True.
///
/// Returns:
///     ``score`` (numpy.ndarray[float64], shape ``(N, M)``), or
///     ``(score, path)`` with ``path`` an int64 ``(length, 2)`` array ordered
///     from start to end (empty when there is no recurrence).
#[pyfunction]
#[pyo3(
    signature = (sim, *, gap_onset = 1.0, gap_extend = 1.0, knight_moves = true, backtrack = true),
    text_signature = "(sim, *, gap_onset=1.0, gap_extend=1.0, knight_moves=True, backtrack=True)"
)]
pub fn rqa<'py>(
    py: Python<'py>,
    sim: &Bound<'py, PyAny>,
    gap_onset: f64,
    gap_extend: f64,
    knight_moves: bool,
    backtrack: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let sim = matrix2(sim, "sim")?;
    let cfg = RqaConfig {
        gap_onset,
        gap_extend,
        knight_moves,
        backtrack,
    };
    let r = seq::rqa(sim.view(), &cfg).map_err(audio_err_to_py)?;
    let score = r.score.into_pyarray(py).into_any();
    if backtrack {
        let path = path_array(py, &r.path.unwrap_or_default()).into_any();
        Ok(PyTuple::new(py, [score, path])?.into_any())
    } else {
        Ok(score)
    }
}

pub(crate) fn parse_mode(mode: &str) -> PyResult<RecurrenceMode> {
    match mode {
        "connectivity" => Ok(RecurrenceMode::Connectivity),
        "distance" => Ok(RecurrenceMode::Distance),
        "affinity" => Ok(RecurrenceMode::Affinity),
        other => Err(value_err(
            "mode",
            format!(
                "unsupported mode {other:?}; expected 'connectivity', 'distance' or 'affinity'"
            ),
        )),
    }
}

pub(crate) fn parse_bandwidth(obj: Option<&Bound<'_, PyAny>>) -> PyResult<Bandwidth> {
    let Some(obj) = obj.filter(|o| !o.is_none()) else {
        return Ok(Bandwidth::MedKScalar);
    };
    if let Ok(s) = obj.cast::<PyString>() {
        return match &*s.to_cow()? {
            "med_k_scalar" => Ok(Bandwidth::MedKScalar),
            "mean_k" => Ok(Bandwidth::MeanK),
            "gmean_k" => Ok(Bandwidth::GmeanK),
            "mean_k_avg" => Ok(Bandwidth::MeanKAvg),
            "gmean_k_avg" => Ok(Bandwidth::GmeanKAvg),
            "mean_k_avg_and_pair" => Ok(Bandwidth::MeanKAvgAndPair),
            other => Err(value_err(
                "bandwidth",
                format!("unknown bandwidth heuristic {other:?}"),
            )),
        };
    }
    let a = f64_array(obj, "bandwidth")?;
    match a.ndim() {
        0 => Ok(Bandwidth::Scalar(*a.as_array().iter().next().expect("0-d"))),
        2 => Ok(Bandwidth::Matrix(
            a.as_array()
                .into_dimensionality::<Ix2>()
                .expect("2-D")
                .to_owned(),
        )),
        n => Err(value_err(
            "bandwidth",
            format!("expected None, a heuristic name, a number or a 2-D array, got {n} dimensions"),
        )),
    }
}

/// Connectivity output as a bool array (as librosa), other modes as float64.
fn similarity_out<'py>(py: Python<'py>, m: Array2<f64>, mode: RecurrenceMode) -> Bound<'py, PyAny> {
    if mode == RecurrenceMode::Connectivity {
        m.mapv(|v| v != 0.0).into_pyarray(py).into_any()
    } else {
        m.into_pyarray(py).into_any()
    }
}

fn reject_sparse(sparse: bool) -> PyResult<()> {
    if sparse {
        Err(value_err(
            "sparse",
            "sparse output is not supported; convert the dense result with scipy.sparse.csc_array",
        ))
    } else {
        Ok(())
    }
}

/// Recurrence (self-similarity) matrix, equivalent to
/// ``librosa.segment.recurrence_matrix(data, ..., sparse=False)``.
///
/// Args:
///     data (array_like): ``(n_features, t)`` features (1-D = one feature).
///     k (Optional[int]): Neighbours per frame; None = ``2 * ceil(sqrt(t - 2 * width + 1))``.
///     width (int): Minimum time separation of linked frames. Default 1.
///     metric (str): Distance metric (see ``dtw``). Default "euclidean".
///     sym (bool): Keep only mutual links. Default False.
///     sparse (bool): Must be False (dense output only).
///     mode (str): ``"connectivity"`` (default, bool output), ``"distance"`` or ``"affinity"``.
///     bandwidth: Affinity bandwidth: None (``"med_k_scalar"``), a heuristic
///         name, a positive number, or a ``(t, t)`` array.
///     self_links (bool): librosa's ``self`` (link each frame to itself).
///         ``self=...`` is also accepted as a keyword.
///     axis (int): Must be -1 (librosa's default layout).
///     full (bool): Link every pair in distance/affinity mode. Default False.
///
/// Returns:
///     numpy.ndarray: ``(t, t)``; bool for connectivity, float64 otherwise.
#[pyfunction]
#[pyo3(
    signature = (data, *, k = None, width = Int(1), metric = "euclidean", sym = false, sparse = false, mode = "connectivity", bandwidth = None, self_links = false, axis = Int(-1), full = false, **kwargs),
    text_signature = "(data, *, k=None, width=1, metric='euclidean', sym=False, sparse=False, mode='connectivity', bandwidth=None, self=False, axis=-1, full=False, self_links=False)"
)]
#[allow(clippy::too_many_arguments)]
pub fn recurrence_matrix<'py>(
    py: Python<'py>,
    data: &Bound<'py, PyAny>,
    k: Option<Int>,
    width: Int,
    metric: &str,
    sym: bool,
    sparse: bool,
    mode: &str,
    bandwidth: Option<&Bound<'py, PyAny>>,
    self_links: bool,
    axis: Int,
    full: bool,
    kwargs: Option<&Bound<'py, PyDict>>,
) -> PyResult<Bound<'py, PyAny>> {
    reject_sparse(sparse)?;
    if axis != Int(-1) && axis != Int(1) {
        return Err(value_err("axis", "only axis=-1 (frames last) is supported"));
    }
    let mut self_links = self_links;
    if let Some(kw) = kwargs {
        for (key, value) in kw.iter() {
            let key: String = key.extract()?;
            if key == "self" {
                self_links = value.extract()?;
            } else {
                return Err(PyTypeError::new_err(format!(
                    "recurrence_matrix() got an unexpected keyword argument '{key}'"
                )));
            }
        }
    }
    let mode = parse_mode(mode)?;
    let cfg = RecurrenceConfig {
        k: k.map(|k| positive("k", k).map(std::num::NonZeroUsize::get))
            .transpose()?,
        width: positive("width", width)?.get(),
        metric: parse_metric(metric)?,
        sym,
        mode,
        bandwidth: parse_bandwidth(bandwidth)?,
        self_links,
        full,
    };
    let data = matrix(data, "data")?;
    let r = seq::recurrence_matrix(data.view(), &cfg).map_err(audio_err_to_py)?;
    Ok(similarity_out(py, r, mode))
}

/// Cross-similarity matrix, equivalent to
/// ``librosa.segment.cross_similarity(data, data_ref, ..., sparse=False)``.
///
/// Args:
///     data (array_like): Query features ``(n_features, n)``.
///     data_ref (array_like): Reference features ``(n_features, n_ref)``.
///     k (Optional[int]): Reference neighbours per query frame; None =
///         ``min(n_ref, 2 * ceil(sqrt(n_ref)))``.
///     metric (str): Distance metric. Default "euclidean".
///     sparse (bool): Must be False.
///     mode (str): ``"connectivity"`` (bool output), ``"distance"`` or ``"affinity"``.
///     bandwidth: As ``recurrence_matrix`` (array shape ``(n, n_ref)``).
///     full (bool): Link every query frame to its ``min(n_ref, n)`` nearest. Default False.
///
/// Returns:
///     numpy.ndarray: ``(n_ref, n)``; bool for connectivity, float64 otherwise.
#[pyfunction]
#[pyo3(
    signature = (data, data_ref, *, k = None, metric = "euclidean", sparse = false, mode = "connectivity", bandwidth = None, full = false),
    text_signature = "(data, data_ref, *, k=None, metric='euclidean', sparse=False, mode='connectivity', bandwidth=None, full=False)"
)]
#[allow(clippy::too_many_arguments)]
pub fn cross_similarity<'py>(
    py: Python<'py>,
    data: &Bound<'py, PyAny>,
    data_ref: &Bound<'py, PyAny>,
    k: Option<Int>,
    metric: &str,
    sparse: bool,
    mode: &str,
    bandwidth: Option<&Bound<'py, PyAny>>,
    full: bool,
) -> PyResult<Bound<'py, PyAny>> {
    reject_sparse(sparse)?;
    let mode = parse_mode(mode)?;
    let cfg = CrossSimilarityConfig {
        k: k.map(|k| positive("k", k).map(std::num::NonZeroUsize::get))
            .transpose()?,
        metric: parse_metric(metric)?,
        mode,
        bandwidth: parse_bandwidth(bandwidth)?,
        full,
    };
    let data = matrix(data, "data")?;
    let data_ref = matrix(data_ref, "data_ref")?;
    let r = seq::cross_similarity(data.view(), data_ref.view(), &cfg).map_err(audio_err_to_py)?;
    Ok(similarity_out(py, r, mode))
}

/// Convert a recurrence matrix to a time-lag matrix, equivalent to
/// ``librosa.segment.recurrence_to_lag(rec, pad=pad, axis=axis)`` (dense input).
///
/// Args:
///     rec (array_like): ``(t, t)`` recurrence matrix (bool or float).
///     pad (bool): Zero-pad to ``(2t, t)`` first. Default True.
///     axis (int): The time axis: ``-1``/``1`` (columns, default) or
///         ``0``/``-2`` (rows; the result is then ``(t, 2t)``).
///
/// Returns:
///     numpy.ndarray: ``(2t, t)`` or ``(t, t)``; bool if ``rec`` is bool, else float64.
#[pyfunction]
#[pyo3(
    signature = (rec, *, pad = true, axis = Int(-1)),
    text_signature = "(rec, *, pad=True, axis=-1)"
)]
pub fn recurrence_to_lag<'py>(
    py: Python<'py>,
    rec: &Bound<'py, PyAny>,
    pad: bool,
    axis: Int,
) -> PyResult<Bound<'py, PyAny>> {
    let rows_are_time = lag_axis_is_rows(axis)?;
    let is_bool = is_bool_array(rec)?;
    let m = matrix2(rec, "rec")?;
    // axis=0 is the axis=-1 transform of the transpose, transposed back.
    let lag = if rows_are_time {
        seq::recurrence_to_lag(m.t(), pad)
            .map_err(audio_err_to_py)?
            .reversed_axes()
    } else {
        seq::recurrence_to_lag(m.view(), pad).map_err(audio_err_to_py)?
    };
    Ok(bool_or_float(
        py,
        lag.as_standard_layout().into_owned(),
        is_bool,
    ))
}

/// Convert a time-lag matrix back to a recurrence matrix, equivalent to
/// ``librosa.segment.lag_to_recurrence(lag, axis=axis)`` (dense input).
///
/// Args:
///     lag (array_like): ``(t, t)`` or ``(2t, t)`` lag matrix.
///     axis (int): The time axis: ``-1``/``1`` (columns, default) or
///         ``0``/``-2`` (rows; a padded lag matrix is then ``(t, 2t)``).
///
/// Returns:
///     numpy.ndarray: ``(t, t)``; bool if ``lag`` is bool, else float64.
#[pyfunction]
#[pyo3(signature = (lag, *, axis = Int(-1)), text_signature = "(lag, *, axis=-1)")]
pub fn lag_to_recurrence<'py>(
    py: Python<'py>,
    lag: &Bound<'py, PyAny>,
    axis: Int,
) -> PyResult<Bound<'py, PyAny>> {
    let rows_are_time = lag_axis_is_rows(axis)?;
    let is_bool = is_bool_array(lag)?;
    let m = matrix2(lag, "lag")?;
    let rec = if rows_are_time {
        seq::lag_to_recurrence(m.t())
            .map_err(audio_err_to_py)?
            .reversed_axes()
    } else {
        seq::lag_to_recurrence(m.view()).map_err(audio_err_to_py)?
    };
    Ok(bool_or_float(
        py,
        rec.as_standard_layout().into_owned(),
        is_bool,
    ))
}

/// librosa's `axis` for the (2-D) lag transforms: `-1`/`1` (columns are time,
/// the default) or `0`/`-2` (rows are time). Returns whether rows are time.
fn lag_axis_is_rows(axis: Int) -> PyResult<bool> {
    match axis.0 {
        -1 | 1 => Ok(false),
        0 | -2 => Ok(true),
        _ => Err(value_err(
            "axis",
            format!("must be -1, 1, 0 or -2 for a 2-D matrix, got {axis}"),
        )),
    }
}

fn is_bool_array(obj: &Bound<'_, PyAny>) -> PyResult<bool> {
    let np = obj.py().import("numpy")?;
    let dt = np.call_method1("asarray", (obj,))?.getattr("dtype")?;
    dt.eq(np.getattr("bool_")?)
}

fn bool_or_float<'py>(py: Python<'py>, m: Array2<f64>, is_bool: bool) -> Bound<'py, PyAny> {
    if is_bool {
        m.mapv(|v| v != 0.0).into_pyarray(py).into_any()
    } else {
        m.into_pyarray(py).into_any()
    }
}

/// Aggregate frames between boundaries (e.g. beats), equivalent to
/// ``librosa.util.sync(data, idx, aggregate=..., pad=...)`` (axis=-1).
///
/// Args:
///     data (array_like): Any shape with frames on the last axis.
///     idx (array_like[int]): Boundary frame indices (sorted and de-duplicated
///         internally). Slices are not supported.
///     aggregate: None / ``numpy.mean`` (default), ``numpy.median``,
///         ``numpy.max``, ``numpy.min``, ``numpy.sum``, the same names as strings,
///         or any callable ``f(x, axis=-1) -> float`` applied to each 1-D segment.
///     pad (bool): Add 0 and ``n_frames`` to the boundaries. Default True.
///
/// Returns:
///     numpy.ndarray[float64]: ``data`` with the last axis replaced by one entry
///     per segment.
///
/// Raises:
///     ValueError: Negative indices or 0-d data.
#[pyfunction]
#[pyo3(
    signature = (data, idx, *, aggregate = None, pad = true, axis = Int(-1)),
    text_signature = "(data, idx, *, aggregate=None, pad=True, axis=-1)"
)]
pub fn sync<'py>(
    py: Python<'py>,
    data: &Bound<'py, PyAny>,
    idx: &Bound<'py, PyAny>,
    aggregate: Option<&Bound<'py, PyAny>>,
    pad: bool,
    axis: Int,
) -> PyResult<Bound<'py, PyAny>> {
    let x = f64_array(data, "data")?;
    if axis != Int(-1) && axis != Int(x.ndim() as i64 - 1) {
        return Err(value_err(
            "axis",
            "only the last axis (axis=-1) is supported",
        ));
    }
    let idx: Vec<usize> = i64_vec(idx, "idx")?
        .into_iter()
        .map(|i| usize::try_from(i).map_err(|_| value_err("idx", "indices must be >= 0")))
        .collect::<PyResult<_>>()?;
    let builtin = match aggregate.filter(|a| !a.is_none()) {
        None => Some(SyncAggregate::Mean),
        Some(a) => builtin_aggregate(a)?,
    };
    let out = if let Some(agg) = builtin {
        seq::sync(x.as_array(), &idx, agg, pad).map_err(audio_err_to_py)?
    } else {
        let f = aggregate.expect("custom aggregate present");
        let kwargs = PyDict::new(py);
        kwargs.set_item("axis", -1)?;
        let mut first_err: Option<PyErr> = None;
        let out = seq::sync_with(x.as_array(), &idx, pad, |lane| {
            if first_err.is_some() {
                return f64::NAN;
            }
            let res = (|| -> PyResult<f64> {
                let arr = lane.to_owned().into_pyarray(py);
                f.call((arr,), Some(&kwargs))?.extract::<f64>()
            })();
            res.unwrap_or_else(|e| {
                first_err = Some(e);
                f64::NAN
            })
        })
        .map_err(audio_err_to_py)?;
        if let Some(e) = first_err {
            return Err(e);
        }
        out
    };
    Ok(out.into_pyarray(py).into_any())
}

/// Maps numpy reductions / names to the built-in aggregates (`None` = custom callable).
fn builtin_aggregate(a: &Bound<'_, PyAny>) -> PyResult<Option<SyncAggregate>> {
    if let Ok(s) = a.cast::<PyString>() {
        return match &*s.to_cow()? {
            "mean" => Ok(Some(SyncAggregate::Mean)),
            "median" => Ok(Some(SyncAggregate::Median)),
            "max" => Ok(Some(SyncAggregate::Max)),
            "min" => Ok(Some(SyncAggregate::Min)),
            "sum" => Ok(Some(SyncAggregate::Sum)),
            other => Err(value_err(
                "aggregate",
                format!(
                    "unknown aggregate {other:?}; expected 'mean', 'median', 'max', 'min', 'sum' or a callable"
                ),
            )),
        };
    }
    if !a.is_callable() {
        return Err(PyTypeError::new_err(
            "aggregate must be None, a name or a callable",
        ));
    }
    let np = a.py().import("numpy")?;
    for (name, agg) in [
        ("mean", SyncAggregate::Mean),
        ("median", SyncAggregate::Median),
        ("max", SyncAggregate::Max),
        ("amax", SyncAggregate::Max),
        ("min", SyncAggregate::Min),
        ("amin", SyncAggregate::Min),
        ("sum", SyncAggregate::Sum),
    ] {
        if a.is(&np.getattr(name)?) {
            return Ok(Some(agg));
        }
    }
    Ok(None)
}

/// Registers `audio_samples.sequence`.
pub fn register(py: Python<'_>, parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(py, "sequence")?;
    m.add(
        "__doc__",
        "Sequence alignment and decoding (librosa.sequence / librosa.segment / \
         librosa.util.sync): DTW, Viterbi, transition matrices, RQA, recurrence \
         and cross-similarity matrices, lag conversions and sync.",
    )?;
    m.add_function(wrap_pyfunction!(dtw, &m)?)?;
    m.add_function(wrap_pyfunction!(dtw_backtracking, &m)?)?;
    m.add_function(wrap_pyfunction!(viterbi, &m)?)?;
    m.add_function(wrap_pyfunction!(viterbi_discriminative, &m)?)?;
    m.add_function(wrap_pyfunction!(viterbi_binary, &m)?)?;
    m.add_function(wrap_pyfunction!(transition_uniform, &m)?)?;
    m.add_function(wrap_pyfunction!(transition_loop, &m)?)?;
    m.add_function(wrap_pyfunction!(transition_cycle, &m)?)?;
    m.add_function(wrap_pyfunction!(transition_local, &m)?)?;
    m.add_function(wrap_pyfunction!(rqa, &m)?)?;
    m.add_function(wrap_pyfunction!(recurrence_matrix, &m)?)?;
    m.add_function(wrap_pyfunction!(cross_similarity, &m)?)?;
    m.add_function(wrap_pyfunction!(recurrence_to_lag, &m)?)?;
    m.add_function(wrap_pyfunction!(lag_to_recurrence, &m)?)?;
    m.add_function(wrap_pyfunction!(sync, &m)?)?;
    parent.add_submodule(&m)?;
    Ok(())
}
