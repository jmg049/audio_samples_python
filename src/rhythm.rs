//! `audio_samples.rhythm`: librosa-compatible rhythm analysis over NumPy arrays.
//!
//! Free functions mirroring `librosa.onset` (`onset_strength`,
//! `onset_strength_multi`, `onset_detect`, `onset_backtrack`),
//! `librosa.util.peak_pick`, `librosa.feature` (`tempogram`,
//! `fourier_tempogram`, `tempo`, `tempogram_ratio`, `hybrid_tempogram`,
//! `metrogram`), `librosa.beat` (`beat_track`, `plp`) and
//! `librosa.tempo_frequencies` / `fourier_tempo_frequencies`, binding
//! `audio_samples::operations::rhythm`. They take precomputed inputs
//! (`S=`, `onset_envelope=`, `tg=`); the `AudioSamples` methods of the same
//! names compute them from audio.
//!
//! Shapes follow librosa: a 1-D envelope gives librosa's 1-D/2-D result, a
//! `(channels, n)` envelope adds a leading channel axis.
//!
//! librosa's callable arguments are supported: `aggregate` (any NumPy-style
//! reduction called as `f(x, axis=-1)`; `numpy.mean`/`median`/`max`/`min`/`sum`
//! and their names run natively) and `prior` (an object with a `logpdf` method,
//! e.g. a frozen `scipy.stats` distribution, or a callable of the BPM).

use crate::args::Int;
use std::cell::RefCell;
use std::num::NonZeroUsize;

use audio_samples::operations::filterbanks::Window;
use audio_samples::operations::rhythm::{
    self as rh, Aggregate, BeatTrackConfig, FourierTempogramConfig, FrameEvents,
    HybridTempogramConfig, OnsetDetectConfig, OnsetStrengthConfig, PeakPickConfig, PeakPickMethod,
    PlpConfig, TempoConfig, TempoLogPrior, TempogramConfig, TempogramRatioConfig,
};
use ndarray::{Array2, Array3, Axis, CowArray, Ix2, Ix3};
use num_complex::Complex;
use numpy::{IntoPyArray, PyArray1, PyUntypedArrayMethods};
use pyo3::IntoPyObjectExt;
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyString};

use crate::args::{
    FloatElement, PyFloat, WindowArg, complex_array_as, f64_array, float_array_as, is_scalar,
    non_negative, positive, precision_of, value_err, with_precision,
};
use crate::audio_err_to_py;

// =============================================================================
// Python callables (aggregate, prior) behind plain Rust fn pointers
// =============================================================================
//
// The Rust configs take `fn` pointers (`Aggregate::Custom(fn(&[f64]) -> f64)`,
// `TempoLogPrior(fn(f64) -> f64)`), which cannot capture a Python object. The
// binding stores the callable in a thread-local slot for the duration of one
// call (the GIL is held throughout and the Rust code does not spawn threads)
// and passes trampolines that call it. The first Python exception raised by a
// callback is kept and re-raised when the call returns.

thread_local! {
    static AGGREGATE_CB: RefCell<Option<Py<PyAny>>> = const { RefCell::new(None) };
    static PRIOR_CB: RefCell<Option<Py<PyAny>>> = const { RefCell::new(None) };
    static CALLBACK_ERR: RefCell<Option<PyErr>> = const { RefCell::new(None) };
}

fn record_callback_err(e: PyErr) {
    CALLBACK_ERR.with(|c| {
        let mut c = c.borrow_mut();
        if c.is_none() {
            *c = Some(e);
        }
    });
}

/// `Aggregate::Custom` trampoline: `aggregate(numpy.array(values), axis=-1)`.
fn py_aggregate(values: &[f64]) -> f64 {
    Python::attach(|py| {
        let Some(f) = AGGREGATE_CB.with(|c| c.borrow().as_ref().map(|f| f.clone_ref(py))) else {
            record_callback_err(missing_callback("aggregate"));
            return f64::NAN;
        };
        let result = (|| -> PyResult<f64> {
            let kwargs = PyDict::new(py);
            kwargs.set_item("axis", -1)?;
            f.bind(py)
                .call((PyArray1::from_slice(py, values),), Some(&kwargs))?
                .extract::<f64>()
                .map_err(|e| {
                    PyTypeError::new_err(format!(
                        "aggregate: the callable must reduce a 1-D array to a number ({e})"
                    ))
                })
        })();
        result.unwrap_or_else(|e| {
            record_callback_err(e);
            f64::NAN
        })
    })
}

/// `TempoLogPrior` trampoline: `prior.logpdf(bpm)` (or `prior(bpm)`).
fn py_prior(bpm: f64) -> f64 {
    Python::attach(|py| {
        let Some(f) = PRIOR_CB.with(|c| c.borrow().as_ref().map(|f| f.clone_ref(py))) else {
            record_callback_err(missing_callback("prior"));
            return f64::NAN;
        };
        let result = f.bind(py).call1((bpm,)).and_then(|r| {
            r.extract::<f64>().map_err(|e| {
                PyTypeError::new_err(format!("prior: logpdf must return a number ({e})"))
            })
        });
        result.unwrap_or_else(|e| {
            record_callback_err(e);
            f64::NAN
        })
    })
}

/// Installs the Python callbacks for one call and restores the previous ones
/// on drop.
///
/// The slots behave as a stack: a callback may itself call another
/// callback-taking binding (directly or through library code), and the inner
/// call must not clobber the outer call's callbacks or its pending error.
pub(crate) struct CallbackScope {
    prev_aggregate: Option<Py<PyAny>>,
    prev_prior: Option<Py<PyAny>>,
    prev_err: Option<PyErr>,
}

impl CallbackScope {
    pub(crate) fn install(
        aggregate: Option<&Py<PyAny>>,
        prior: Option<&Py<PyAny>>,
        py: Python<'_>,
    ) -> Self {
        let prev_aggregate = AGGREGATE_CB.with(|c| c.replace(aggregate.map(|f| f.clone_ref(py))));
        let prev_prior = PRIOR_CB.with(|c| c.replace(prior.map(|f| f.clone_ref(py))));
        let prev_err = CALLBACK_ERR.with(|c| c.replace(None));
        Self {
            prev_aggregate,
            prev_prior,
            prev_err,
        }
    }

    /// Converts the Rust result, giving precedence to an exception raised by a
    /// callback.
    pub(crate) fn finish<T>(self, r: audio_samples::AudioSampleResult<T>) -> PyResult<T> {
        if let Some(e) = CALLBACK_ERR.with(|c| c.borrow_mut().take()) {
            return Err(e);
        }
        r.map_err(audio_err_to_py)
    }
}

impl Drop for CallbackScope {
    fn drop(&mut self) {
        AGGREGATE_CB.with(|c| *c.borrow_mut() = self.prev_aggregate.take());
        PRIOR_CB.with(|c| *c.borrow_mut() = self.prev_prior.take());
        CALLBACK_ERR.with(|c| *c.borrow_mut() = self.prev_err.take());
    }
}

/// The error recorded when a trampoline runs with no callback installed. This
/// is a binding bug (the Rust call outlived its [`CallbackScope`]); it is
/// raised rather than silently producing NaN.
fn missing_callback(name: &str) -> PyErr {
    pyo3::exceptions::PyRuntimeError::new_err(format!(
        "{name}: internal error: the Python callback is no longer installed"
    ))
}

/// A parsed `aggregate` argument.
#[derive(Debug)]
pub(crate) enum AggregateSpec {
    /// `None`.
    Unset,
    /// `False` (no aggregation, `onset_strength_multi` only).
    Disabled,
    /// A built-in reduction.
    Builtin(Aggregate),
    /// Any other callable.
    Callable(Py<PyAny>),
}

impl AggregateSpec {
    /// The Rust aggregate, with `Unset` mapped to `unset` and `Disabled` to
    /// `None`. Callables become the trampoline (install them with
    /// [`CallbackScope`]).
    pub(crate) fn resolve(&self, unset: Option<Aggregate>) -> Option<Aggregate> {
        match self {
            Self::Unset => unset,
            Self::Disabled => None,
            Self::Builtin(a) => Some(*a),
            Self::Callable(_) => Some(Aggregate::Custom(py_aggregate)),
        }
    }

    pub(crate) fn callable(&self) -> Option<&Py<PyAny>> {
        match self {
            Self::Callable(f) => Some(f),
            _ => None,
        }
    }
}

/// Parses librosa's `aggregate`: `None`, `False`, a name (`"mean"`, `"median"`,
/// `"max"`, `"min"`, `"sum"`), a NumPy reduction or any callable.
pub(crate) fn parse_aggregate(obj: Option<&Bound<'_, PyAny>>) -> PyResult<AggregateSpec> {
    let Some(obj) = obj.filter(|o| !o.is_none()) else {
        return Ok(AggregateSpec::Unset);
    };
    if let Ok(b) = obj.cast::<PyBool>() {
        return if b.is_true() {
            Err(value_err(
                "aggregate",
                "True is not a valid aggregate; pass False, None or a function",
            ))
        } else {
            Ok(AggregateSpec::Disabled)
        };
    }
    if let Ok(s) = obj.cast::<PyString>() {
        return match &*s.to_cow()? {
            "mean" => Ok(AggregateSpec::Builtin(Aggregate::Mean)),
            "median" => Ok(AggregateSpec::Builtin(Aggregate::Median)),
            "max" | "amax" => Ok(AggregateSpec::Builtin(Aggregate::Max)),
            "min" | "amin" => Ok(AggregateSpec::Builtin(Aggregate::Min)),
            "sum" => Ok(AggregateSpec::Builtin(Aggregate::Sum)),
            other => Err(value_err(
                "aggregate",
                format!(
                    "unknown aggregate {other:?}; expected 'mean', 'median', 'max', 'min', 'sum' or a callable"
                ),
            )),
        };
    }
    if !obj.is_callable() {
        return Err(PyTypeError::new_err(
            "aggregate must be None, False, a name or a callable",
        ));
    }
    let np = obj.py().import("numpy")?;
    for (name, agg) in [
        ("mean", Aggregate::Mean),
        ("median", Aggregate::Median),
        ("max", Aggregate::Max),
        ("amax", Aggregate::Max),
        ("min", Aggregate::Min),
        ("amin", Aggregate::Min),
        ("sum", Aggregate::Sum),
    ] {
        if obj.is(&np.getattr(name)?) {
            return Ok(AggregateSpec::Builtin(agg));
        }
    }
    Ok(AggregateSpec::Callable(obj.clone().unbind()))
}

/// An `aggregate` keyword whose default is not `None` (pyo3 cannot tell an
/// omitted `Option` from an explicit `None`).
#[derive(Debug)]
pub struct AggregateArg(pub(crate) AggregateSpec);

impl AggregateArg {
    pub(crate) const fn mean() -> Self {
        Self(AggregateSpec::Builtin(Aggregate::Mean))
    }

    pub(crate) const fn sum() -> Self {
        Self(AggregateSpec::Builtin(Aggregate::Sum))
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for AggregateArg {
    type Error = PyErr;

    fn extract(obj: pyo3::Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        parse_aggregate(Some(&obj.to_owned())).map(Self)
    }
}

/// Parses librosa's `prior`: `None`, an object with `logpdf` (a frozen
/// `scipy.stats` distribution), or a callable of the BPM returning a log-density.
pub(crate) fn parse_prior(obj: Option<&Bound<'_, PyAny>>) -> PyResult<Option<Py<PyAny>>> {
    let Some(obj) = obj.filter(|o| !o.is_none()) else {
        return Ok(None);
    };
    if obj.hasattr("logpdf")? {
        return Ok(Some(obj.getattr("logpdf")?.unbind()));
    }
    if obj.is_callable() {
        return Ok(Some(obj.clone().unbind()));
    }
    Err(PyTypeError::new_err(
        "prior must be None, a scipy.stats distribution (an object with .logpdf) or a callable",
    ))
}

/// The Rust prior for a parsed `prior`.
pub(crate) fn prior_fn(prior: Option<&Py<PyAny>>) -> Option<TempoLogPrior> {
    prior.map(|_| TempoLogPrior(py_prior))
}

// =============================================================================
// Array helpers
// =============================================================================

/// A `(n,)` or `(channels, n)` array as `(channels, n)` (a view of the
/// NumPy data when no conversion is needed); the flag is `true` for 1-D input.
pub(crate) fn rows<'a, F: FloatElement>(
    obj: &'a Bound<'_, PyAny>,
    name: &str,
) -> PyResult<(CowArray<'a, F, Ix2>, bool)> {
    let a = crate::args::array_cow(obj, |o| float_array_as::<F>(o, name))?;
    match a.ndim() {
        1 => Ok((
            a.insert_axis(Axis(0))
                .into_dimensionality::<Ix2>()
                .expect("2-D"),
            true,
        )),
        2 => Ok((a.into_dimensionality::<Ix2>().expect("2-D"), false)),
        n => Err(value_err(
            name,
            format!("expected shape (n,) or (channels, n), got {n} dimensions"),
        )),
    }
}

/// A 1-D float vector, borrowed from a contiguous NumPy array of element type
/// `F` (no copy) or converted otherwise.
pub(crate) fn vec1<'a, F: FloatElement>(
    obj: &'a Bound<'_, PyAny>,
    name: &str,
) -> PyResult<std::borrow::Cow<'a, [F]>> {
    use numpy::PyArrayMethods;
    if let Ok(arr) = obj.cast::<numpy::PyArray1<F>>() {
        // Fails if Rust code holds a mutable borrow of the array.
        drop(arr.try_readonly()?);
        // SAFETY: the slice borrows `obj`, which keeps the array alive; no
        // Rust code holds a mutable borrow of it (checked above).
        if let Ok(slice) = unsafe { arr.as_slice() } {
            return Ok(std::borrow::Cow::Borrowed(slice));
        }
    }
    let a = float_array_as::<F>(obj, name)?;
    if a.ndim() != 1 {
        return Err(value_err(
            name,
            format!("expected a 1-D array, got {} dimensions", a.ndim()),
        ));
    }
    Ok(std::borrow::Cow::Owned(
        a.as_array().iter().copied().collect(),
    ))
}

/// A `(F, T)` or `(channels, F, T)` array as 3-D (a view of the NumPy data
/// when no conversion is needed); the flag is `true` for 2-D input.
pub(crate) fn planes<'a, F: FloatElement>(
    obj: &'a Bound<'_, PyAny>,
    name: &str,
) -> PyResult<(CowArray<'a, F, Ix3>, bool)> {
    let a = crate::args::array_cow(obj, |o| float_array_as::<F>(o, name))?;
    lift3(a, name)
}

/// Complex variant of [`planes`].
pub(crate) fn cplanes<'a, F: FloatElement>(
    obj: &'a Bound<'_, PyAny>,
    name: &str,
) -> PyResult<(CowArray<'a, Complex<F>, Ix3>, bool)>
where
    Complex<F>: numpy::Element,
{
    let a = crate::args::array_cow(obj, |o| complex_array_as::<F>(o, name))?;
    lift3(a, name)
}

/// `(F, T)` -> `(1, F, T)`; `(channels, F, T)` as is.
fn lift3<'a, T>(
    a: CowArray<'a, T, ndarray::IxDyn>,
    name: &str,
) -> PyResult<(CowArray<'a, T, Ix3>, bool)> {
    match a.ndim() {
        2 => Ok((
            a.insert_axis(Axis(0))
                .into_dimensionality::<Ix3>()
                .expect("3-D"),
            true,
        )),
        3 => Ok((a.into_dimensionality::<Ix3>().expect("3-D"), false)),
        n => Err(value_err(
            name,
            format!(
                "expected shape (n_bins, n_frames) or (channels, n_bins, n_frames), got {n} dimensions"
            ),
        )),
    }
}

/// `(channels, n)` output, as `(n,)` when `squeeze`.
pub(crate) fn out2<'py, T: numpy::Element>(
    py: Python<'py>,
    a: Array2<T>,
    squeeze: bool,
) -> Bound<'py, PyAny> {
    if squeeze && a.nrows() == 1 {
        a.index_axis_move(Axis(0), 0).into_pyarray(py).into_any()
    } else {
        a.into_pyarray(py).into_any()
    }
}

/// `(channels, F, T)` output, as `(F, T)` when `squeeze`.
pub(crate) fn out3<'py, T: numpy::Element>(
    py: Python<'py>,
    a: Array3<T>,
    squeeze: bool,
) -> Bound<'py, PyAny> {
    if squeeze && a.dim().0 == 1 {
        a.index_axis_move(Axis(0), 0).into_pyarray(py).into_any()
    } else {
        a.into_pyarray(py).into_any()
    }
}

/// Converts event frames to librosa's `units`.
pub(crate) fn events_out<'py>(
    py: Python<'py>,
    frames: Vec<usize>,
    units: &str,
    sample_rate: f64,
    hop_length: NonZeroUsize,
) -> PyResult<Bound<'py, PyAny>> {
    let ev = FrameEvents {
        frames,
        sample_rate,
        hop_length,
    };
    match units {
        "frames" => Ok(ev
            .frames
            .iter()
            .map(|&f| f as i64)
            .collect::<Vec<_>>()
            .into_pyarray(py)
            .into_any()),
        "samples" => Ok(ev
            .samples()
            .into_iter()
            .map(|s| s as i64)
            .collect::<Vec<_>>()
            .into_pyarray(py)
            .into_any()),
        "time" => Ok(ev.times().into_pyarray(py).into_any()),
        other => Err(value_err(
            "units",
            format!("invalid unit type {other:?}; expected 'frames', 'samples' or 'time'"),
        )),
    }
}

pub(crate) fn require_sparse(sparse: bool) -> PyResult<()> {
    if sparse {
        Ok(())
    } else {
        Err(value_err(
            "sparse",
            "only sparse=True (event indices) is supported",
        ))
    }
}

/// Non-negative integer events as `usize`.
fn usize_vec(obj: &Bound<'_, PyAny>, name: &str) -> PyResult<Vec<usize>> {
    crate::args::with_i64_view(obj, name, |v| {
        let mut out = Vec::with_capacity(v.len());
        for &x in &v {
            out.push(non_negative(name, x)?);
        }
        Ok(out)
    })
}

/// A tempogram `norm`: `None` or a number (`numpy.inf` for max-normalisation).
pub(crate) fn tempogram_config(
    win_length: Int,
    center: bool,
    window: WindowArg,
    norm: Option<f64>,
) -> PyResult<TempogramConfig> {
    Ok(TempogramConfig::new()
        .with_win_length(positive("win_length", win_length)?)
        .with_center(center)
        .with_window(window.0)
        .with_norm(norm))
}

pub(crate) fn fourier_tempogram_config(
    win_length: Int,
    center: bool,
    window: WindowArg,
) -> PyResult<FourierTempogramConfig> {
    Ok(FourierTempogramConfig::new()
        .with_win_length(positive("win_length", win_length)?)
        .with_center(center)
        .with_window(window.0))
}

pub(crate) fn hybrid_tempogram_config(
    win_length: Int,
    center: bool,
    window: WindowArg,
) -> PyResult<HybridTempogramConfig> {
    let mut cfg = HybridTempogramConfig::new().with_win_length(positive("win_length", win_length)?);
    cfg.center = center;
    cfg.window = window.0;
    Ok(cfg)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn tempo_config(
    start_bpm: f64,
    std_bpm: f64,
    ac_size: f64,
    max_tempo: Option<f64>,
    aggregate: Option<Aggregate>,
    prior: Option<&Py<PyAny>>,
) -> TempoConfig {
    TempoConfig::new()
        .with_start_bpm(start_bpm)
        .with_std_bpm(std_bpm)
        .with_ac_size(ac_size)
        .with_max_tempo(max_tempo)
        .with_aggregate(aggregate)
        .with_prior(prior_fn(prior))
}

pub(crate) fn plp_config(
    win_length: Int,
    tempo_min: Option<f64>,
    tempo_max: Option<f64>,
    prior: Option<&Py<PyAny>>,
) -> PyResult<PlpConfig> {
    Ok(PlpConfig::new()
        .with_win_length(positive("win_length", win_length)?)
        .with_tempo_range(tempo_min, tempo_max)
        .with_prior(prior_fn(prior)))
}

/// Parses librosa's peak-picking `method`.
pub(crate) fn parse_peak_method(method: &str) -> PyResult<PeakPickMethod> {
    match method {
        "greedy" => Ok(PeakPickMethod::Greedy),
        "dp_count" => Ok(PeakPickMethod::DpCount),
        "dp_value" => Ok(PeakPickMethod::DpValue),
        other => Err(value_err(
            "method",
            format!("unknown method {other:?}; expected 'greedy', 'dp_count' or 'dp_value'"),
        )),
    }
}

/// librosa `onset_detect` peak-picking parameters: the given ones, the rest
/// derived from the frame rate.
#[allow(clippy::too_many_arguments)]
pub(crate) fn onset_peak_pick(
    sr: f64,
    hop: NonZeroUsize,
    pre_max: Option<Int>,
    post_max: Option<Int>,
    pre_avg: Option<Int>,
    post_avg: Option<Int>,
    delta: Option<f64>,
    wait: Option<Int>,
    method: &str,
) -> PyResult<PeakPickConfig> {
    if !(sr.is_finite() && sr > 0.0) {
        return Err(value_err(
            "sr",
            format!("must be finite and positive, got {sr}"),
        ));
    }
    let mut p = PeakPickConfig::for_onsets(sr, hop);
    if let Some(v) = pre_max {
        p.pre_max = non_negative("pre_max", v)?;
    }
    if let Some(v) = post_max {
        p.post_max = non_negative("post_max", v)?;
    }
    if let Some(v) = pre_avg {
        p.pre_avg = non_negative("pre_avg", v)?;
    }
    if let Some(v) = post_avg {
        p.post_avg = non_negative("post_avg", v)?;
    }
    if let Some(v) = delta {
        p.delta = v;
    }
    if let Some(v) = wait {
        p.wait = non_negative("wait", v)?;
    }
    p.method = parse_peak_method(method)?;
    Ok(p)
}

/// Onset detection on one envelope, with librosa's `units`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn detect_onsets<'py, F: PyFloat>(
    py: Python<'py>,
    envelope: &[F],
    sr: f64,
    hop: NonZeroUsize,
    peak: PeakPickConfig,
    backtrack: bool,
    energy: Option<&Bound<'py, PyAny>>,
    units: &str,
    normalize: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let energy = energy
        .filter(|e| !e.is_none())
        .map(|e| vec1::<F>(e, "energy"))
        .transpose()?;
    let cfg = OnsetDetectConfig::new()
        .with_peak_pick(Some(peak))
        .with_normalize(normalize)
        .with_backtrack(backtrack);
    let frames = rh::onset_detect_from_envelope(envelope, sr, hop, &cfg, energy.as_deref())
        .map_err(audio_err_to_py)?;
    events_out(py, frames, units, sr, hop)
}

/// Beat tracking on one envelope with librosa's return conventions.
#[allow(clippy::too_many_arguments)]
pub(crate) fn track_beats<'py, F: PyFloat>(
    py: Python<'py>,
    envelope: &[F],
    sr: f64,
    hop: NonZeroUsize,
    start_bpm: f64,
    tightness: f64,
    trim: bool,
    bpm: Option<&Bound<'py, PyAny>>,
    prior: Option<&Bound<'py, PyAny>>,
    units: &str,
) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)> {
    let bpm = bpm.filter(|b| !b.is_none());
    let prior = parse_prior(prior)?;
    // Validate units before doing any work.
    if !matches!(units, "frames" | "samples" | "time") {
        return Err(value_err(
            "units",
            format!("invalid unit type {units:?}; expected 'frames', 'samples' or 'time'"),
        ));
    }
    if !envelope.iter().any(|&v| v != F::of(0.0)) {
        // librosa: no onsets -> tempo 0.0 and no beats, whatever `bpm` is.
        return Ok((
            0.0f64.into_bound_py_any(py)?,
            events_out(py, Vec::new(), units, sr, hop)?,
        ));
    }
    let (tempo, frames) = match bpm {
        Some(b) if !is_scalar(b)? => {
            let values: Vec<f64> = f64_array(b, "bpm")?.as_array().iter().copied().collect();
            if !(tightness.is_finite() && tightness > 0.0) {
                return Err(value_err("tightness", "must be finite and positive"));
            }
            let frames =
                rh::beat_track_with_bpm(envelope, &values, sr / hop.get() as f64, tightness, trim)
                    .map_err(audio_err_to_py)?;
            (b.clone(), frames)
        }
        _ => {
            let fixed = bpm.map(|b| b.extract::<f64>()).transpose()?;
            let cfg = BeatTrackConfig::new()
                .with_start_bpm(start_bpm)
                .with_tightness(tightness)
                .with_trim(trim)
                .with_bpm(fixed)
                .with_prior(prior_fn(prior.as_ref()));
            let bt = if prior.is_some() {
                // A Python prior is called back from the Rust code: keep the GIL.
                let scope = CallbackScope::install(None, prior.as_ref(), py);
                scope.finish(rh::beat_track_from_envelope(envelope, sr, hop, &cfg))?
            } else {
                py.detach(|| rh::beat_track_from_envelope(envelope, sr, hop, &cfg))
                    .map_err(audio_err_to_py)?
            };
            // librosa returns the given `bpm` unchanged, or the estimate as a
            // shape-(1,) array.
            let tempo = match bpm {
                Some(b) => b.clone(),
                None => vec![bt.tempo].into_pyarray(py).into_any(),
            };
            (tempo, bt.beats.frames)
        }
    };
    Ok((tempo, events_out(py, frames, units, sr, hop)?))
}

fn hop_nz(hop_length: i64) -> PyResult<NonZeroUsize> {
    positive("hop_length", hop_length)
}

fn check_kind(kind: &str) -> PyResult<()> {
    if kind == "linear" {
        Ok(())
    } else {
        Err(value_err(
            "kind",
            format!("only kind='linear' interpolation is supported, got {kind:?}"),
        ))
    }
}

/// Builds an [`OnsetStrengthConfig`] for spectrogram input (`n_fft`/`hop_length`
/// only set the centring shift).
#[allow(clippy::too_many_arguments)]
fn onset_cfg_for_s(
    lag: Int,
    max_size: Int,
    detrend: bool,
    center: bool,
    aggregate: Option<Aggregate>,
    n_fft: Int,
    hop_length: Int,
) -> PyResult<OnsetStrengthConfig> {
    let mut cfg = OnsetStrengthConfig::new()
        .with_lag(positive("lag", lag)?)
        .with_max_size(positive("max_size", max_size)?)
        .with_detrend(detrend)
        .with_center(center)
        .with_aggregate(aggregate);
    cfg.mel.stft.n_fft = positive("n_fft", n_fft)?;
    cfg.mel.stft.hop_length = Some(hop_nz(hop_length.0)?);
    Ok(cfg)
}

/// Sub-band boundaries (librosa `channels` as integers).
pub(crate) fn parse_channels(obj: Option<&Bound<'_, PyAny>>) -> PyResult<Option<Vec<usize>>> {
    obj.filter(|o| !o.is_none())
        .map(|o| {
            if o.try_iter()?
                .any(|v| v.is_ok_and(|v| v.is_instance_of::<pyo3::types::PySlice>()))
            {
                return Err(value_err(
                    "channels",
                    "slices are not supported; pass integer sub-band boundaries",
                ));
            }
            usize_vec(o, "channels")
        })
        .transpose()
}

// =============================================================================
// Onsets
// =============================================================================

/// Onset strength envelope of a precomputed spectrogram, equivalent to
/// ``librosa.onset.onset_strength(S=S, ...)``.
///
/// ``S`` is used as given (librosa computes ``power_to_db`` of a mel
/// spectrogram when it is not passed; use ``AudioSamples.onset_strength`` for
/// that).
///
/// Args:
///     S (array_like): Log-power spectrogram, shape ``(n_bins, n_frames)`` or
///         ``(channels, n_bins, n_frames)``.
///     sr (float): Required (librosa defaults to 22050). Not used: librosa
///         only uses it to compute ``S`` from ``y``.
///     lag (int): Time lag of the difference in frames. Default 1.
///     max_size (int): Size of the maximum filter across frequency applied to
///         the reference (SuperFlux); 1 disables it. Default 1.
///     ref (Optional[array_like]): Reference spectrogram of the same shape as
///         ``S``; None uses ``S`` (maximum-filtered when ``max_size > 1``).
///     detrend (bool): High-pass filter the envelope. Default False.
///     center (bool): Shift by ``n_fft // (2 * hop_length)`` frames to undo the
///         centred-STFT delay. Default True.
///     aggregate: Reduction across bins: None (mean), ``numpy.mean``,
///         ``numpy.median``, ``numpy.max``, ``numpy.min``, ``numpy.sum``, a
///         name, or a callable ``f(x, axis=-1)``. ``False`` returns the first
///         bin's flux (as librosa).
///     n_fft (int): FFT size behind ``S`` (centring shift). Default 2048.
///     hop_length (int): Hop behind ``S`` (centring shift). Default 512.
///
/// Returns:
///     numpy.ndarray[float32 | float64]: shape ``(n_frames,)`` for 2-D ``S``,
///     ``(channels, n_frames)`` for 3-D.
///
/// Raises:
///     ValueError: For invalid parameters or an empty spectrogram.
#[pyfunction]
#[pyo3(
    signature = (*, S, sr, lag = Int(1), max_size = Int(1), r#ref = None, detrend = false, center = true, aggregate = None, n_fft = Int(2048), hop_length = Int(512)),
    text_signature = "(*, S, sr, lag=1, max_size=1, ref=None, detrend=False, center=True, aggregate=None, n_fft=2048, hop_length=512)"
)]
#[allow(non_snake_case, clippy::too_many_arguments, unused_variables)]
pub fn onset_strength<'py>(
    py: Python<'py>,
    S: &Bound<'py, PyAny>,
    sr: f64,
    lag: Int,
    max_size: Int,
    r#ref: Option<&Bound<'py, PyAny>>,
    detrend: bool,
    center: bool,
    aggregate: Option<&Bound<'py, PyAny>>,
    n_fft: Int,
    hop_length: Int,
) -> PyResult<Bound<'py, PyAny>> {
    let agg = parse_aggregate(aggregate)?;
    let cfg = onset_cfg_for_s(
        lag,
        max_size,
        detrend,
        center,
        agg.resolve(Some(Aggregate::Mean)),
        n_fft,
        hop_length,
    )?;
    with_precision!(precision_of(S), |F| {
        let (s, squeeze) = planes::<F>(S, "S")?;
        let reference = r#ref
            .filter(|r| !r.is_none())
            .map(|r| planes::<F>(r, "ref").map(|p| p.0))
            .transpose()?;
        let scope = CallbackScope::install(agg.callable(), None, py);
        let multi = scope.finish(rh::onset_strength_multi_from_db(
            s.view(),
            reference.as_ref().map(|r| r.view()),
            &cfg,
            None,
        ))?;
        // One band (or, with aggregate=False, the first bin, as librosa's `[..., 0, :]`).
        let env = multi.index_axis_move(Axis(1), 0);
        Ok(out2(py, env, squeeze))
    })
}

/// Multi-band onset strength of a precomputed spectrogram, equivalent to
/// ``librosa.onset.onset_strength_multi(S=S, ...)``.
///
/// Args:
///     S (array_like): Log-power spectrogram ``(n_bins, n_frames)`` or
///         ``(channels, n_bins, n_frames)``, used as given.
///     sr (float): Required (librosa defaults to 22050). Not used (see
///         ``onset_strength``).
///     n_fft (int): FFT size behind ``S`` (centring shift). Default 2048.
///     hop_length (int): Hop behind ``S`` (centring shift). Default 512.
///     lag, max_size, ref, detrend, center: As ``onset_strength``.
///     aggregate: As ``onset_strength``; ``False`` keeps every bin.
///     channels (Optional[Sequence[int]]): Sub-band boundaries in bins; values
///         above ``n_bins`` are dropped, the rest sorted and de-duplicated, and
///         each consecutive pair is one band. None gives one band. Slices are
///         not supported.
///
/// Returns:
///     numpy.ndarray[float32 | float64]: shape ``(n_bands, n_frames)`` for 2-D ``S``,
///     ``(channels, n_bands, n_frames)`` for 3-D.
///
/// Raises:
///     ValueError: For invalid parameters or boundaries.
#[pyfunction]
#[pyo3(
    signature = (*, S, sr, n_fft = Int(2048), hop_length = Int(512), lag = Int(1), max_size = Int(1), r#ref = None, detrend = false, center = true, aggregate = None, channels = None),
    text_signature = "(*, S, sr, n_fft=2048, hop_length=512, lag=1, max_size=1, ref=None, detrend=False, center=True, aggregate=None, channels=None)"
)]
#[allow(non_snake_case, clippy::too_many_arguments, unused_variables)]
pub fn onset_strength_multi<'py>(
    py: Python<'py>,
    S: &Bound<'py, PyAny>,
    sr: f64,
    n_fft: Int,
    hop_length: Int,
    lag: Int,
    max_size: Int,
    r#ref: Option<&Bound<'py, PyAny>>,
    detrend: bool,
    center: bool,
    aggregate: Option<&Bound<'py, PyAny>>,
    channels: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let agg = parse_aggregate(aggregate)?;
    let cfg = onset_cfg_for_s(
        lag,
        max_size,
        detrend,
        center,
        agg.resolve(Some(Aggregate::Mean)),
        n_fft,
        hop_length,
    )?;
    let bounds = parse_channels(channels)?;
    with_precision!(precision_of(S), |F| {
        let (s, squeeze) = planes::<F>(S, "S")?;
        let reference = r#ref
            .filter(|r| !r.is_none())
            .map(|r| planes::<F>(r, "ref").map(|p| p.0))
            .transpose()?;
        let scope = CallbackScope::install(agg.callable(), None, py);
        let multi = scope.finish(rh::onset_strength_multi_from_db(
            s.view(),
            reference.as_ref().map(|r| r.view()),
            &cfg,
            bounds.as_deref(),
        ))?;
        Ok(out3(py, multi, squeeze))
    })
}

/// Picks peaks in a signal, equivalent to ``librosa.util.peak_pick``.
///
/// A sample ``n`` is a peak if ``x[n] == max(x[n - pre_max : n + post_max])``,
/// ``x[n] >= mean(x[n - pre_avg : n + post_avg]) + delta`` and (greedy method)
/// it is more than ``wait`` samples after the previous peak.
///
/// Args:
///     x (array_like): 1-D finite signal.
///     pre_max, post_max, pre_avg, post_avg (int): Window extents in samples
///         (``post_max`` and ``post_avg`` at least 1).
///     delta (float): Threshold offset above the local mean (>= 0).
///     wait (int): Minimum number of samples between peaks.
///     sparse (bool): Must be True (indices are returned).
///     method (str): ``"greedy"`` (default), ``"dp_count"`` or ``"dp_value"``.
///     axis (int): Must be -1 (1-D input only).
///
/// Returns:
///     numpy.ndarray[int64]: ascending peak indices.
///
/// Raises:
///     ValueError: For invalid parameters, non-finite input or ``sparse=False``.
#[pyfunction]
#[pyo3(
    signature = (x, *, pre_max, post_max, pre_avg, post_avg, delta, wait, sparse = true, method = "greedy", axis = Int(-1)),
    text_signature = "(x, *, pre_max, post_max, pre_avg, post_avg, delta, wait, sparse=True, method='greedy', axis=-1)"
)]
#[allow(clippy::too_many_arguments)]
pub fn peak_pick<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyAny>,
    pre_max: Int,
    post_max: Int,
    pre_avg: Int,
    post_avg: Int,
    delta: f64,
    wait: Int,
    sparse: bool,
    method: &str,
    axis: Int,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    require_sparse(sparse)?;
    if axis != Int(-1) && axis != Int(0) {
        return Err(value_err("axis", "only 1-D input (axis=-1) is supported"));
    }
    let prec = precision_of(x);
    let cfg = PeakPickConfig {
        pre_max: non_negative("pre_max", pre_max)?,
        post_max: non_negative("post_max", post_max)?,
        pre_avg: non_negative("pre_avg", pre_avg)?,
        post_avg: non_negative("post_avg", post_avg)?,
        delta,
        wait: non_negative("wait", wait)?,
        method: parse_peak_method(method)?,
    };
    let peaks = with_precision!(prec, |F| rh::peak_pick(&vec1::<F>(x, "x")?, &cfg))
        .map_err(audio_err_to_py)?;
    Ok(peaks
        .into_iter()
        .map(|p| p as i64)
        .collect::<Vec<_>>()
        .into_pyarray(py))
}

/// Moves events back to the preceding local minimum of ``energy``, equivalent
/// to ``librosa.onset.onset_backtrack``.
///
/// Args:
///     events (array_like): Event frame indices (non-negative integers).
///     energy (array_like): 1-D function to search for minima (onset envelope
///         or RMS energy).
///
/// Returns:
///     numpy.ndarray[int64]: one backtracked frame per event. An empty
///     ``events`` gives an empty array (librosa raises).
#[pyfunction]
#[pyo3(signature = (events, energy), text_signature = "(events, energy)")]
pub fn onset_backtrack<'py>(
    py: Python<'py>,
    events: &Bound<'py, PyAny>,
    energy: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let events = usize_vec(events, "events")?;
    let backtracked = with_precision!(precision_of(energy), |F| {
        rh::onset_backtrack(&events, &vec1::<F>(energy, "energy")?)
    });
    Ok(backtracked
        .into_iter()
        .map(|p| p as i64)
        .collect::<Vec<_>>()
        .into_pyarray(py))
}

/// Detects onsets in a precomputed onset envelope, equivalent to
/// ``librosa.onset.onset_detect(onset_envelope=..., ...)``.
///
/// The envelope is normalised to ``[0, 1]`` (``normalize=True``), peak-picked
/// with the parameters below (defaults derived from the frame rate as librosa:
/// 30 ms ``pre_max`` and ``wait``, 100 ms averaging windows, ``delta=0.07``)
/// and optionally backtracked.
///
/// Args:
///     onset_envelope (array_like): 1-D onset strength envelope.
///     sr (float): Sampling rate of the analysed audio. Required (librosa defaults to 22050).
///     hop_length (int): Hop of the envelope in samples. Default 512.
///     backtrack (bool): Move onsets back to the preceding local minimum of
///         ``energy``. Default False.
///     energy (Optional[array_like]): 1-D function to backtrack on; None uses
///         the (normalised) envelope.
///     units (str): ``"frames"`` (default), ``"samples"`` or ``"time"``.
///     normalize (bool): Scale the envelope to ``[0, 1]`` first. Default True.
///     sparse (bool): Must be True.
///     pre_max, post_max, pre_avg, post_avg, wait (Optional[int]),
///     delta (Optional[float]), method (str): Peak-picking parameters
///         (``librosa.util.peak_pick``); None derives librosa's default.
///
/// Returns:
///     numpy.ndarray: onset frames or samples (int64), or times (float64).
///
/// Raises:
///     ValueError: For invalid parameters, a 2-D envelope or ``sparse=False``.
#[pyfunction]
#[pyo3(
    signature = (*, onset_envelope, sr, hop_length = Int(512), backtrack = false, energy = None, units = "frames", normalize = true, sparse = true, pre_max = None, post_max = None, pre_avg = None, post_avg = None, delta = None, wait = None, method = "greedy"),
    text_signature = "(*, onset_envelope, sr, hop_length=512, backtrack=False, energy=None, units='frames', normalize=True, sparse=True, pre_max=None, post_max=None, pre_avg=None, post_avg=None, delta=None, wait=None, method='greedy')"
)]
#[allow(clippy::too_many_arguments)]
pub fn onset_detect<'py>(
    py: Python<'py>,
    onset_envelope: &Bound<'py, PyAny>,
    sr: f64,
    hop_length: Int,
    backtrack: bool,
    energy: Option<&Bound<'py, PyAny>>,
    units: &str,
    normalize: bool,
    sparse: bool,
    pre_max: Option<Int>,
    post_max: Option<Int>,
    pre_avg: Option<Int>,
    post_avg: Option<Int>,
    delta: Option<f64>,
    wait: Option<Int>,
    method: &str,
) -> PyResult<Bound<'py, PyAny>> {
    require_sparse(sparse)?;
    let hop = hop_nz(hop_length.0)?;
    let peak = onset_peak_pick(
        sr, hop, pre_max, post_max, pre_avg, post_avg, delta, wait, method,
    )?;
    with_precision!(precision_of(onset_envelope), |F| {
        let env = vec1::<F>(onset_envelope, "onset_envelope")?;
        detect_onsets(py, &env, sr, hop, peak, backtrack, energy, units, normalize)
    })
}

// =============================================================================
// Tempograms and tempo
// =============================================================================

/// Tempo (BPM) of each lag bin of an autocorrelation tempogram, equivalent to
/// ``librosa.tempo_frequencies``.
///
/// Args:
///     n_bins (int): Number of lag bins.
///     hop_length (int): Hop in samples. Default 512.
///     sr (float): Sampling rate. Required (librosa defaults to 22050).
///
/// Returns:
///     numpy.ndarray[float64]: shape ``(n_bins,)``; bin 0 (zero lag) is ``inf``.
#[pyfunction]
#[pyo3(
    signature = (n_bins, *, hop_length = Int(512), sr),
    text_signature = "(n_bins, *, hop_length=512, sr)"
)]
pub fn tempo_frequencies(
    py: Python<'_>,
    n_bins: Int,
    hop_length: Int,
    sr: f64,
) -> PyResult<Bound<'_, PyArray1<f64>>> {
    let n = non_negative("n_bins", n_bins)?;
    Ok(rh::tempo_frequencies(n, hop_nz(hop_length.0)?, sr).into_pyarray(py))
}

/// Tempo (BPM) of each bin of a Fourier tempogram, equivalent to
/// ``librosa.fourier_tempo_frequencies``.
///
/// Args:
///     sr (float): Sampling rate. Required (librosa defaults to 22050).
///     win_length (int): Fourier tempogram window in frames. Default 384.
///     hop_length (int): Hop in samples. Default 512.
///
/// Returns:
///     numpy.ndarray[float64]: shape ``(win_length // 2 + 1,)``.
#[pyfunction]
#[pyo3(
    signature = (*, sr, win_length = Int(384), hop_length = Int(512)),
    text_signature = "(*, sr, win_length=384, hop_length=512)"
)]
pub fn fourier_tempo_frequencies(
    py: Python<'_>,
    sr: f64,
    win_length: Int,
    hop_length: Int,
) -> PyResult<Bound<'_, PyArray1<f64>>> {
    Ok(rh::fourier_tempo_frequencies(
        sr,
        hop_nz(hop_length.0)?,
        positive("win_length", win_length)?,
    )
    .into_pyarray(py))
}

/// Autocorrelation tempogram of an onset envelope, equivalent to
/// ``librosa.feature.tempogram(onset_envelope=..., ...)``.
///
/// Args:
///     onset_envelope (array_like): ``(n_frames,)`` or ``(channels, n_frames)``.
///     sr (float): Required (librosa defaults to 22050). Not used: librosa
///         only uses it to compute the envelope.
///     hop_length (int): Unused (as above).
///     win_length (int): Autocorrelation window in frames (number of lag
///         bins). Default 384.
///     center (bool): Centre the windows (linear-ramp padding). Default True.
///     window: Window applied before the autocorrelation. Default ``"hann"``.
///     norm (Optional[float]): Column normalisation: ``numpy.inf`` (default,
///         divide by the maximum), a p-norm, or None.
///
/// Returns:
///     numpy.ndarray[float32 | float64]: shape ``(win_length, n_frames)`` for 1-D input,
///     ``(channels, win_length, n_frames)`` for 2-D (``n_frames - win_length + 1``
///     frames when not centred).
///
/// Raises:
///     ValueError: For invalid parameters or (uncentred) a too-short envelope.
#[pyfunction]
#[pyo3(
    signature = (*, onset_envelope, sr, hop_length = Int(512), win_length = Int(384), center = true, window = WindowArg(Window::Hann), norm = Some(f64::INFINITY)),
    text_signature = "(*, onset_envelope, sr, hop_length=512, win_length=384, center=True, window='hann', norm=math.inf)"
)]
#[allow(clippy::too_many_arguments, unused_variables)]
pub fn tempogram<'py>(
    py: Python<'py>,
    onset_envelope: &Bound<'py, PyAny>,
    sr: f64,
    hop_length: Int,
    win_length: Int,
    center: bool,
    window: WindowArg,
    norm: Option<f64>,
) -> PyResult<Bound<'py, PyAny>> {
    let cfg = tempogram_config(win_length, center, window, norm)?;
    with_precision!(precision_of(onset_envelope), |F| {
        let (env, squeeze) = rows::<F>(onset_envelope, "onset_envelope")?;
        let tg = rh::tempogram_from_envelope(env.view(), &cfg).map_err(audio_err_to_py)?;
        Ok(out3(py, tg, squeeze))
    })
}

/// Fourier tempogram of an onset envelope, equivalent to
/// ``librosa.feature.fourier_tempogram(onset_envelope=..., ...)``.
///
/// Args:
///     onset_envelope (array_like): ``(n_frames,)`` or ``(channels, n_frames)``.
///     sr, hop_length: Unused (librosa only uses them to compute the envelope).
///     win_length (int): STFT length in frames. Default 384.
///     center (bool): Centre the frames (zero padding). Default True.
///     window: Analysis window. Default ``"hann"``.
///
/// Returns:
///     numpy.ndarray[complex64 | complex128]: shape ``(win_length // 2 + 1, n_frames + 1)``
///     (centred) for 1-D input, with a leading channel axis for 2-D.
#[pyfunction]
#[pyo3(
    signature = (*, onset_envelope, sr, hop_length = Int(512), win_length = Int(384), center = true, window = WindowArg(Window::Hann)),
    text_signature = "(*, onset_envelope, sr, hop_length=512, win_length=384, center=True, window='hann')"
)]
#[allow(unused_variables)]
pub fn fourier_tempogram<'py>(
    py: Python<'py>,
    onset_envelope: &Bound<'py, PyAny>,
    sr: f64,
    hop_length: Int,
    win_length: Int,
    center: bool,
    window: WindowArg,
) -> PyResult<Bound<'py, PyAny>> {
    let cfg = fourier_tempogram_config(win_length, center, window)?;
    with_precision!(precision_of(onset_envelope), |F| {
        let (env, squeeze) = rows::<F>(onset_envelope, "onset_envelope")?;
        let tg = rh::fourier_tempogram_from_envelope(env.view(), &cfg).map_err(audio_err_to_py)?;
        Ok(out3(py, tg, squeeze))
    })
}

/// Estimates the tempo from an onset envelope or tempogram, equivalent to
/// ``librosa.feature.tempo(onset_envelope=..., ...)`` / ``(tg=..., ...)``.
///
/// The tempo is the lag bin maximising ``log1p(1e6 * tempogram) + log prior``
/// (default prior: log-normal around ``start_bpm`` with ``std_bpm`` octaves).
///
/// Args:
///     onset_envelope (Optional[array_like]): ``(n_frames,)`` or
///         ``(channels, n_frames)``. Exactly one of ``onset_envelope`` / ``tg``.
///     tg (Optional[array_like]): Autocorrelation tempogram
///         ``(win_length, n_frames)`` or ``(channels, win_length, n_frames)``.
///     sr (float): Sampling rate. Required (librosa defaults to 22050).
///     hop_length (int): Hop in samples. Default 512.
///     start_bpm (float): Prior centre. Default 120.
///     std_bpm (float): Prior spread in octaves. Default 1.
///     ac_size (float): Autocorrelation window in seconds (envelope input).
///         Default 8.
///     max_tempo (Optional[float]): Tempi at or above this are excluded.
///         Default 320; None allows any.
///     aggregate: Reduction over time (default ``numpy.mean``); None estimates
///         one tempo per frame.
///     prior: None (log-normal default), a ``scipy.stats`` distribution (its
///         ``logpdf`` is used) or a callable of the BPM.
///
/// Returns:
///     numpy.ndarray[float64]: shape ``(1,)`` (or ``(n_frames,)`` with
///     ``aggregate=None``) for 1-D input, with a leading channel axis for 2-D.
///
/// Raises:
///     ValueError: For invalid parameters, or neither/both inputs given.
#[pyfunction]
#[pyo3(
    signature = (*, onset_envelope = None, tg = None, sr, hop_length = Int(512), start_bpm = 120.0, std_bpm = 1.0, ac_size = 8.0, max_tempo = Some(320.0), aggregate = AggregateArg::mean(), prior = None),
    text_signature = "(*, onset_envelope=None, tg=None, sr, hop_length=512, start_bpm=120, std_bpm=1.0, ac_size=8.0, max_tempo=320.0, aggregate='mean', prior=None)"
)]
#[allow(clippy::too_many_arguments)]
pub fn tempo<'py>(
    py: Python<'py>,
    onset_envelope: Option<&Bound<'py, PyAny>>,
    tg: Option<&Bound<'py, PyAny>>,
    sr: f64,
    hop_length: Int,
    start_bpm: f64,
    std_bpm: f64,
    ac_size: f64,
    max_tempo: Option<f64>,
    aggregate: AggregateArg,
    prior: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let hop = hop_nz(hop_length.0)?;
    let prior = parse_prior(prior)?;
    let agg = aggregate.0;
    let cfg = tempo_config(
        start_bpm,
        std_bpm,
        ac_size,
        max_tempo,
        agg.resolve(None),
        prior.as_ref(),
    );
    let onset_envelope = onset_envelope.filter(|o| !o.is_none());
    let tg = tg.filter(|o| !o.is_none());
    let scope = CallbackScope::install(agg.callable(), prior.as_ref(), py);
    let (out, squeeze) = match (onset_envelope, tg) {
        (Some(env), None) => with_precision!(precision_of(env), |F| {
            let (env, squeeze) = rows::<F>(env, "onset_envelope")?;
            (rh::tempo_from_envelope(env.view(), sr, hop, &cfg), squeeze)
        }),
        (None, Some(tg)) => with_precision!(precision_of(tg), |F| {
            let (tg, squeeze) = planes::<F>(tg, "tg")?;
            (rh::tempo_from_tempogram(tg.view(), sr, hop, &cfg), squeeze)
        }),
        _ => {
            return Err(value_err(
                "onset_envelope",
                "pass exactly one of onset_envelope or tg (use AudioSamples.tempo for audio)",
            ));
        }
    };
    let out = scope.finish(out)?;
    Ok(out2(py, out, squeeze))
}

/// Tracks beats in an onset envelope, equivalent to
/// ``librosa.beat.beat_track(onset_envelope=..., ...)``.
///
/// Ellis's dynamic-programming tracker: beats maximise onset strength while
/// keeping inter-beat intervals close to the tempo's period.
///
/// Args:
///     onset_envelope (array_like): 1-D onset envelope (librosa's beat tracker
///         uses ``aggregate=numpy.median`` for it).
///     sr (float): Sampling rate. Required (librosa defaults to 22050).
///     hop_length (int): Hop in samples. Default 512.
///     start_bpm (float): Prior centre of the tempo estimate. Default 120.
///     tightness (float): Tempo-consistency weight. Default 100.
///     trim (bool): Remove weak leading/trailing beats. Default False
///         (librosa: True).
///     bpm (Optional[float | array_like]): Tempo to track at (a number, or one
///         value per frame for a time-varying tempo); None estimates it.
///     prior: Tempo prior (see ``tempo``).
///     units (str): ``"frames"`` (default), ``"samples"`` or ``"time"``.
///     sparse (bool): Must be True.
///
/// Returns:
///     tuple: ``(tempo, beats)``, as librosa: ``tempo`` is the estimate as a
///     float64 array of shape ``(1,)``, or ``bpm`` unchanged when given, or
///     ``0.0`` when the envelope is all zeros; ``beats`` are int64 frames or
///     samples, or float64 times.
///
/// Raises:
///     ValueError: For invalid parameters, a 2-D envelope or ``sparse=False``.
#[pyfunction]
#[pyo3(
    signature = (*, onset_envelope, sr, hop_length = Int(512), start_bpm = 120.0, tightness = 100.0, trim = false, bpm = None, prior = None, units = "frames", sparse = true),
    text_signature = "(*, onset_envelope, sr, hop_length=512, start_bpm=120.0, tightness=100, trim=False, bpm=None, prior=None, units='frames', sparse=True)"
)]
#[allow(clippy::too_many_arguments)]
pub fn beat_track<'py>(
    py: Python<'py>,
    onset_envelope: &Bound<'py, PyAny>,
    sr: f64,
    hop_length: Int,
    start_bpm: f64,
    tightness: f64,
    trim: bool,
    bpm: Option<&Bound<'py, PyAny>>,
    prior: Option<&Bound<'py, PyAny>>,
    units: &str,
    sparse: bool,
) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)> {
    require_sparse(sparse)?;
    let hop = hop_nz(hop_length.0)?;
    with_precision!(precision_of(onset_envelope), |F| {
        let env = vec1::<F>(onset_envelope, "onset_envelope")?;
        track_beats(
            py, &env, sr, hop, start_bpm, tightness, trim, bpm, prior, units,
        )
    })
}

/// Runs the dynamic-programming beat tracker at a given tempo (librosa's
/// internal ``__beat_tracker``, what ``beat_track(onset_envelope=..., bpm=...)``
/// runs).
///
/// Args:
///     onset_envelope (array_like): 1-D onset envelope.
///     bpm (float | array_like): Tempo in BPM: one value, or one per frame.
///     sr (float): Sampling rate. Required (librosa defaults to 22050).
///     hop_length (int): Hop in samples (frame rate ``sr / hop_length``).
///         Default 512.
///     tightness (float): Tempo-consistency weight. Default 100.
///     trim (bool): Remove weak leading/trailing beats. Default False
///         (librosa: True).
///
/// Returns:
///     numpy.ndarray[int64]: ascending beat frames.
///
/// Raises:
///     ValueError: For a wrong-length or non-positive ``bpm``, or a tempo so
///         fast that its period is under one frame.
#[pyfunction]
#[pyo3(
    signature = (onset_envelope, bpm, *, sr, hop_length = Int(512), tightness = 100.0, trim = false),
    text_signature = "(onset_envelope, bpm, *, sr, hop_length=512, tightness=100.0, trim=False)"
)]
pub fn beat_track_with_bpm<'py>(
    py: Python<'py>,
    onset_envelope: &Bound<'py, PyAny>,
    bpm: &Bound<'py, PyAny>,
    sr: f64,
    hop_length: Int,
    tightness: f64,
    trim: bool,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let bpm: Vec<f64> = f64_array(bpm, "bpm")?.as_array().iter().copied().collect();
    let hop = hop_nz(hop_length.0)?;
    let frames = with_precision!(precision_of(onset_envelope), |F| {
        let env = vec1::<F>(onset_envelope, "onset_envelope")?;
        rh::beat_track_with_bpm(&env, &bpm, sr / hop.get() as f64, tightness, trim)
    })
    .map_err(audio_err_to_py)?;
    Ok(frames
        .into_iter()
        .map(|f| f as i64)
        .collect::<Vec<_>>()
        .into_pyarray(py))
}

/// Predominant local pulse of an onset envelope, equivalent to
/// ``librosa.beat.plp(onset_envelope=..., ...)``.
///
/// Args:
///     onset_envelope (array_like): ``(n_frames,)`` or ``(channels, n_frames)``
///         (librosa's PLP uses ``aggregate=numpy.median`` for it).
///     sr (float): Sampling rate. Required (librosa defaults to 22050).
///     hop_length (int): Hop in samples. Default 512.
///     win_length (int): Fourier tempogram window in frames. Default 384.
///     tempo_min (Optional[float]): Lowest tempo (BPM). Default 30.
///     tempo_max (Optional[float]): Highest tempo (BPM). Default 300.
///     prior: Tempo prior (see ``tempo``), added in the log domain.
///
/// Returns:
///     numpy.ndarray[float32 | float64]: pulse curve, same shape as the envelope, in
///     ``[0, 1]``.
///
/// Raises:
///     ValueError: If ``tempo_max <= tempo_min`` or the envelope is empty.
#[pyfunction]
#[pyo3(
    signature = (*, onset_envelope, sr, hop_length = Int(512), win_length = Int(384), tempo_min = Some(30.0), tempo_max = Some(300.0), prior = None),
    text_signature = "(*, onset_envelope, sr, hop_length=512, win_length=384, tempo_min=30, tempo_max=300, prior=None)"
)]
#[allow(clippy::too_many_arguments)]
pub fn plp<'py>(
    py: Python<'py>,
    onset_envelope: &Bound<'py, PyAny>,
    sr: f64,
    hop_length: Int,
    win_length: Int,
    tempo_min: Option<f64>,
    tempo_max: Option<f64>,
    prior: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let prior = parse_prior(prior)?;
    let cfg = plp_config(win_length, tempo_min, tempo_max, prior.as_ref())?;
    let hop = hop_nz(hop_length.0)?;
    with_precision!(precision_of(onset_envelope), |F| {
        let (env, squeeze) = rows::<F>(onset_envelope, "onset_envelope")?;
        let scope = CallbackScope::install(None, prior.as_ref(), py);
        let p = scope.finish(rh::plp_from_envelope(env.view(), sr, hop, &cfg))?;
        Ok(out2(py, p, squeeze))
    })
}

/// Tempogram ratio features, equivalent to
/// ``librosa.feature.tempogram_ratio(onset_envelope=... | tg=..., ...)``.
///
/// The tempogram is sampled at ``factors`` times the per-frame tempo estimate
/// (Prockup et al. 2015).
///
/// Args:
///     onset_envelope (Optional[array_like]): ``(n_frames,)`` or
///         ``(channels, n_frames)``; used to compute the tempogram when ``tg``
///         is None.
///     tg (Optional[array_like]): Autocorrelation tempogram
///         ``(win_length, n_frames)`` or ``(channels, win_length, n_frames)``.
///     bpm (Optional[array_like]): Per-frame tempo ``(n_frames,)`` /
///         ``(channels, n_frames)``; None estimates it from the tempogram.
///     sr (float): Sampling rate. Required (librosa defaults to 22050).
///     hop_length (int): Hop in samples. Default 512.
///     win_length, center, window, norm: Tempogram parameters (envelope input).
///     start_bpm, std_bpm, max_tempo, prior: Per-frame tempo estimation.
///     freqs: Must be None (the ``tempo_frequencies`` grid is used).
///     factors (Optional[array_like]): Tempo multiples; None uses librosa's 13
///         default factors.
///     aggregate: Optional reduction over time (default None).
///     kind (str): Must be ``"linear"``.
///     fill_value (float): Value outside the tempogram's range. Default 0.
///
/// Returns:
///     numpy.ndarray[float32 | float64]: shape ``(n_factors, n_frames)`` for 1-D input
///     (``(n_factors,)`` when aggregated), with a leading channel axis for 2-D.
///
/// Raises:
///     ValueError: For invalid parameters, unsupported ``freqs`` / ``kind``,
///         or neither input given.
#[pyfunction]
#[pyo3(
    signature = (*, onset_envelope = None, tg = None, bpm = None, sr, hop_length = Int(512), win_length = Int(384), start_bpm = 120.0, std_bpm = 1.0, max_tempo = Some(320.0), freqs = None, factors = None, aggregate = None, prior = None, center = true, window = WindowArg(Window::Hann), kind = "linear", fill_value = 0.0, norm = Some(f64::INFINITY)),
    text_signature = "(*, onset_envelope=None, tg=None, bpm=None, sr, hop_length=512, win_length=384, start_bpm=120, std_bpm=1.0, max_tempo=320.0, freqs=None, factors=None, aggregate=None, prior=None, center=True, window='hann', kind='linear', fill_value=0, norm=math.inf)"
)]
#[allow(clippy::too_many_arguments)]
pub fn tempogram_ratio<'py>(
    py: Python<'py>,
    onset_envelope: Option<&Bound<'py, PyAny>>,
    tg: Option<&Bound<'py, PyAny>>,
    bpm: Option<&Bound<'py, PyAny>>,
    sr: f64,
    hop_length: Int,
    win_length: Int,
    start_bpm: f64,
    std_bpm: f64,
    max_tempo: Option<f64>,
    freqs: Option<&Bound<'py, PyAny>>,
    factors: Option<&Bound<'py, PyAny>>,
    aggregate: Option<&Bound<'py, PyAny>>,
    prior: Option<&Bound<'py, PyAny>>,
    center: bool,
    window: WindowArg,
    kind: &str,
    fill_value: f64,
    norm: Option<f64>,
) -> PyResult<Bound<'py, PyAny>> {
    check_kind(kind)?;
    if freqs.is_some_and(|f| !f.is_none()) {
        return Err(value_err(
            "freqs",
            "custom tempogram frequencies are not supported; the tempo_frequencies grid is used",
        ));
    }
    let hop = hop_nz(hop_length.0)?;
    let prior = parse_prior(prior)?;
    let agg = parse_aggregate(aggregate)?;
    let cfg = ratio_config(
        win_length,
        center,
        window,
        norm,
        start_bpm,
        std_bpm,
        max_tempo,
        prior.as_ref(),
        factors,
        agg.resolve(None),
        fill_value,
    )?;
    let onset_envelope = onset_envelope.filter(|o| !o.is_none());
    let tg = tg.filter(|o| !o.is_none());
    let primary = match (onset_envelope, tg) {
        (_, Some(tg)) => tg,
        (Some(env), None) => env,
        (None, None) => {
            return Err(value_err(
                "onset_envelope",
                "pass onset_envelope or tg (use AudioSamples.tempogram_ratio for audio)",
            ));
        }
    };
    let bpm = bpm
        .filter(|b| !b.is_none())
        .map(|b| rows::<f64>(b, "bpm").map(|r| r.0))
        .transpose()?;
    with_precision!(precision_of(primary), |F| {
        let (tg, squeeze) = if let Some(tg) = tg {
            planes::<F>(tg, "tg")?
        } else {
            let (env, squeeze) = rows::<F>(primary, "onset_envelope")?;
            (
                CowArray::from(
                    rh::tempogram_from_envelope(env.view(), &cfg.tempogram)
                        .map_err(audio_err_to_py)?,
                ),
                squeeze,
            )
        };
        let scope = CallbackScope::install(agg.callable(), prior.as_ref(), py);
        let r = scope.finish(rh::tempogram_ratio_from_tempogram(
            tg.view(),
            sr,
            hop,
            &cfg,
            bpm.as_ref().map(|b| b.view()),
        ))?;
        Ok(ratio_out(py, r, squeeze, cfg.aggregate.is_some()))
    })
}

/// librosa drops the time axis of an aggregated tempogram ratio.
fn ratio_out<'py, T: numpy::Element>(
    py: Python<'py>,
    r: Array3<T>,
    squeeze: bool,
    aggregated: bool,
) -> Bound<'py, PyAny> {
    if aggregated {
        out2(py, r.index_axis_move(Axis(2), 0), squeeze)
    } else {
        out3(py, r, squeeze)
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn ratio_config(
    win_length: Int,
    center: bool,
    window: WindowArg,
    norm: Option<f64>,
    start_bpm: f64,
    std_bpm: f64,
    max_tempo: Option<f64>,
    prior: Option<&Py<PyAny>>,
    factors: Option<&Bound<'_, PyAny>>,
    aggregate: Option<Aggregate>,
    fill_value: f64,
) -> PyResult<TempogramRatioConfig> {
    let mut cfg = TempogramRatioConfig::new().with_aggregate(aggregate);
    cfg.tempogram = tempogram_config(win_length, center, window, norm)?;
    cfg.tempo = tempo_config(start_bpm, std_bpm, 8.0, max_tempo, None, prior);
    if let Some(f) = factors.filter(|f| !f.is_none()) {
        cfg.factors = f64_array(f, "factors")?
            .as_array()
            .iter()
            .copied()
            .collect();
    }
    cfg.fill_value = fill_value;
    Ok(cfg)
}

/// Hybrid (Fourier x autocorrelation) tempogram, equivalent to
/// ``librosa.feature.hybrid_tempogram(onset_envelope=..., ...)``.
///
/// Args:
///     onset_envelope (array_like): ``(n_frames,)`` or ``(channels, n_frames)``.
///     sr (float): Sampling rate. Required (librosa defaults to 22050).
///     hop_length (int): Hop in samples. Default 512.
///     win_length (int): Window of both tempograms in frames (>= 3). Default 384.
///     center (bool): Centred frames. Default True.
///     window: Window of both tempograms. Default ``"hann"``.
///
/// Returns:
///     numpy.ndarray[float32 | float64]: shape ``(win_length // 2 + 1, n_frames)`` on the
///     ``fourier_tempo_frequencies`` grid, with a leading channel axis for 2-D input.
#[pyfunction]
#[pyo3(
    signature = (*, onset_envelope, sr, hop_length = Int(512), win_length = Int(384), center = true, window = WindowArg(Window::Hann)),
    text_signature = "(*, onset_envelope, sr, hop_length=512, win_length=384, center=True, window='hann')"
)]
pub fn hybrid_tempogram<'py>(
    py: Python<'py>,
    onset_envelope: &Bound<'py, PyAny>,
    sr: f64,
    hop_length: Int,
    win_length: Int,
    center: bool,
    window: WindowArg,
) -> PyResult<Bound<'py, PyAny>> {
    let cfg = hybrid_tempogram_config(win_length, center, window)?;
    let hop = hop_nz(hop_length.0)?;
    with_precision!(precision_of(onset_envelope), |F| {
        let (env, squeeze) = rows::<F>(onset_envelope, "onset_envelope")?;
        let h = rh::hybrid_tempogram_from_envelope(env.view(), sr, hop, &cfg)
            .map_err(audio_err_to_py)?;
        Ok(out3(py, h, squeeze))
    })
}

/// Metrogram (metrical subharmonic salience) of a tempogram, equivalent to
/// ``librosa.feature.metrogram``.
///
/// Args:
///     tg (array_like): Tempogram magnitude ``(n_freqs, n_frames)`` or
///         ``(channels, n_freqs, n_frames)``.
///     freqs (array_like): Tempo (BPM) of each bin; at least two finite values
///         (drop the infinite zero-lag bin of ``tempo_frequencies`` first).
///     factors (Optional[array_like]): Subharmonic factors; None uses
///         ``[1/3, 1/4, 1/5, 1/7]``.
///     aggregate: Reduction over the tempo axis (default ``numpy.sum``). None
///         (librosa's 4-D per-bin output) is not supported.
///     kind (str): Must be ``"linear"``.
///     fill_value (float): Value outside the frequency grid. Default 0.
///
/// Returns:
///     numpy.ndarray[float32 | float64]: shape ``(n_factors, n_frames)``, with a leading
///     channel axis for 3-D input.
#[pyfunction]
#[pyo3(
    signature = (*, tg, freqs, factors = None, aggregate = AggregateArg::sum(), kind = "linear", fill_value = 0.0),
    text_signature = "(*, tg, freqs, factors=None, aggregate='sum', kind='linear', fill_value=0)"
)]
pub fn metrogram<'py>(
    py: Python<'py>,
    tg: &Bound<'py, PyAny>,
    freqs: &Bound<'py, PyAny>,
    factors: Option<&Bound<'py, PyAny>>,
    aggregate: AggregateArg,
    kind: &str,
    fill_value: f64,
) -> PyResult<Bound<'py, PyAny>> {
    check_kind(kind)?;
    let agg = aggregate.0;
    let Some(aggregate) = agg.resolve(None) else {
        return Err(value_err(
            "aggregate",
            "aggregate=None (librosa's per-bin output) is not supported; pass a reduction",
        ));
    };
    let freqs = vec1::<f64>(freqs, "freqs")?;
    let factors: Option<Vec<f64>> = factors
        .filter(|f| !f.is_none())
        .map(|f| f64_array(f, "factors").map(|a| a.as_array().iter().copied().collect()))
        .transpose()?;
    with_precision!(precision_of(tg), |F| {
        let (tg, squeeze) = planes::<F>(tg, "tg")?;
        let scope = CallbackScope::install(agg.callable(), None, py);
        let m = scope.finish(rh::metrogram(
            tg.view(),
            &freqs,
            factors.as_deref(),
            aggregate,
            fill_value,
        ))?;
        Ok(out3(py, m, squeeze))
    })
}

/// Registers `audio_samples.rhythm`.
pub fn register(py: Python<'_>, parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(py, "rhythm")?;
    m.add(
        "__doc__",
        "librosa-compatible rhythm analysis over NumPy arrays (onset strength and \
         detection, peak picking, tempograms, tempo, beat tracking, PLP). \
         float32 / complex64 arrays are processed in float32 and give float32 \
         output; any other input is converted to float64.",
    )?;
    m.add_function(wrap_pyfunction!(onset_strength, &m)?)?;
    m.add_function(wrap_pyfunction!(onset_strength_multi, &m)?)?;
    m.add_function(wrap_pyfunction!(onset_detect, &m)?)?;
    m.add_function(wrap_pyfunction!(onset_backtrack, &m)?)?;
    m.add_function(wrap_pyfunction!(peak_pick, &m)?)?;
    m.add_function(wrap_pyfunction!(tempo_frequencies, &m)?)?;
    m.add_function(wrap_pyfunction!(fourier_tempo_frequencies, &m)?)?;
    m.add_function(wrap_pyfunction!(tempogram, &m)?)?;
    m.add_function(wrap_pyfunction!(fourier_tempogram, &m)?)?;
    m.add_function(wrap_pyfunction!(tempo, &m)?)?;
    m.add_function(wrap_pyfunction!(beat_track, &m)?)?;
    m.add_function(wrap_pyfunction!(beat_track_with_bpm, &m)?)?;
    m.add_function(wrap_pyfunction!(plp, &m)?)?;
    m.add_function(wrap_pyfunction!(tempogram_ratio, &m)?)?;
    m.add_function(wrap_pyfunction!(hybrid_tempogram, &m)?)?;
    m.add_function(wrap_pyfunction!(metrogram, &m)?)?;
    parent.add_submodule(&m)?;
    Ok(())
}
