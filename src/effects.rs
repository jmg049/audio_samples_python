//! `audio_samples.effects`: librosa-compatible effect and decomposition helpers
//! over NumPy arrays.
//!
//! Free functions mirroring `librosa.phase_vocoder`, `librosa.effects.preemphasis`
//! / `deemphasis` (with `zi` / `return_zf`), `librosa.util.softmask`,
//! `librosa.decompose.hpss` / `nn_filter`, `librosa.griffinlim`, `librosa.lpc`
//! and `librosa.clicks`, binding `audio_samples::operations::effects`. The
//! audio-to-audio effects (`time_stretch`, `pitch_shift`, `trim`, `split`,
//! `remix`, `harmonic`, `percussive`, ...) are `AudioSamples` methods.

use crate::args::Int;
use std::num::NonZeroUsize;

use audio_samples::operations::effects::{
    self as fx, ClickPositions, ClicksConfig, GriffinLimConfig, GriffinLimInit, HpssMaskConfig,
    NnAggregate,
};
use audio_samples::AudioFloat;
use audio_samples::operations::filterbanks::Window;
use audio_samples::operations::sequence::RecurrenceConfig;
use ndarray::{Array2, Array3, Ix2};
use num_complex::Complex;
use numpy::{IntoPyArray, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::IntoPyObjectExt;
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyString};

use crate::args::{
    FloatElement, PyFloat, WindowArg, f64_array, float_array_as, i64_vec, is_scalar, non_negative, positive,
    precision_of, stft_config, value_err, with_precision,
};
use crate::audio_err_to_py;
use crate::rhythm::{cplanes, out3, planes, rows};

/// Maps the Rust error, turning "empty result" errors into `ValueError`.
pub(crate) fn effects_err(e: audio_samples::AudioSampleError) -> PyErr {
    if matches!(e, audio_samples::AudioSampleError::EmptyData { .. }) {
        value_err("y", e)
    } else {
        audio_err_to_py(e)
    }
}

/// librosa's scalar-or-pair `kernel_size`.
pub(crate) fn parse_kernel_size(obj: &Bound<'_, PyAny>) -> PyResult<(NonZeroUsize, NonZeroUsize)> {
    let v = i64_vec(obj, "kernel_size")?;
    match (is_scalar(obj)?, v.as_slice()) {
        (true, [k]) => {
            let k = positive("kernel_size", *k)?;
            Ok((k, k))
        }
        (false, [h, p]) => Ok((positive("kernel_size", *h)?, positive("kernel_size", *p)?)),
        _ => Err(value_err(
            "kernel_size",
            "expected an integer or a pair (harmonic, percussive)",
        )),
    }
}

/// librosa's scalar-or-pair `margin`.
pub(crate) fn parse_margin(obj: &Bound<'_, PyAny>) -> PyResult<(f64, f64)> {
    let a = f64_array(obj, "margin")?;
    let v: Vec<f64> = a.as_array().iter().copied().collect();
    match (is_scalar(obj)?, v.as_slice()) {
        (true, [m]) => Ok((*m, *m)),
        (false, [h, p]) => Ok((*h, *p)),
        _ => Err(value_err(
            "margin",
            "expected a number or a pair (harmonic, percussive)",
        )),
    }
}

/// [`HpssMaskConfig`] from librosa's `kernel_size`, `power`, `margin`.
pub(crate) fn hpss_mask_config(
    kernel_size: &Bound<'_, PyAny>,
    power: f64,
    margin: &Bound<'_, PyAny>,
) -> PyResult<HpssMaskConfig> {
    let (kh, kp) = parse_kernel_size(kernel_size)?;
    let (mh, mp) = parse_margin(margin)?;
    Ok(HpssMaskConfig::new()
        .with_kernel_sizes(kh, kp)
        .with_power(power)
        .with_margins(mh, mp))
}

/// A `(n,)` or `(channels, n)` output, matching the input's rank.
fn rows_out<'py, T: numpy::Element>(
    py: Python<'py>,
    a: Array2<T>,
    squeeze: bool,
) -> Bound<'py, PyAny> {
    crate::rhythm::out2(py, a, squeeze)
}

// =============================================================================
// Phase vocoder
// =============================================================================

/// Time-stretches an STFT matrix, equivalent to ``librosa.phase_vocoder``.
///
/// Give exactly one of ``rate`` (output frames at ``0, rate, 2 rate, ...``) or
/// ``t_out`` (explicit fractional frame positions in ``[0, n_frames)``).
/// Magnitudes are linearly interpolated; the phase accumulates the raw
/// frame-to-frame phase differences (librosa 1.0's algorithm).
///
/// Args:
///     D (array_like): complex STFT of shape ``(n_bins, n_frames)`` or
///         ``(channels, n_bins, n_frames)``.
///     rate (Optional[float]): Speed-up factor (> 1 is faster / fewer frames).
///     t_out (Optional[array_like]): Output frame positions.
///     kind (str): Must be ``"linear"``.
///     hop_length (Optional[int]): Deprecated in librosa 1.0 and unused (the
///         phase advance cancels); accepted for compatibility with a
///         ``FutureWarning`` when given, as librosa does.
///     n_fft (Optional[int]): Deprecated and unused, as ``hop_length``.
///
/// Returns:
///     numpy.ndarray[complex64 | complex128]: shape ``(n_bins, ceil(n_frames / rate))`` (or
///     ``len(t_out)`` frames), with a leading channel axis for 3-D input.
///
/// Raises:
///     ValueError: If neither or both of ``rate`` / ``t_out`` are given, or a
///         value is out of range.
#[pyfunction]
#[pyo3(
    signature = (D, *, rate = None, t_out = None, kind = "linear", hop_length = None, n_fft = None),
    text_signature = "(D, *, rate=None, t_out=None, kind='linear', hop_length=None, n_fft=None)"
)]
#[allow(non_snake_case)]
pub fn phase_vocoder<'py>(
    py: Python<'py>,
    D: &Bound<'py, PyAny>,
    rate: Option<f64>,
    t_out: Option<&Bound<'py, PyAny>>,
    kind: &str,
    hop_length: Option<&Bound<'py, PyAny>>,
    n_fft: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    // librosa 1.0 deprecated both (they no longer affect the result) and warns
    // with FutureWarning whenever they are passed.
    if hop_length.is_some() {
        PyErr::warn(
            py,
            &py.get_type::<pyo3::exceptions::PyFutureWarning>(),
            c"The `hop_length` parameter is deprecated as of librosa 1.0 and is unused.",
            1,
        )?;
    }
    if n_fft.is_some() {
        PyErr::warn(
            py,
            &py.get_type::<pyo3::exceptions::PyFutureWarning>(),
            c"The `n_fft` parameter is deprecated as of librosa 1.0 and is unused.",
            1,
        )?;
    }
    if kind != "linear" {
        return Err(value_err("kind", "only kind='linear' is supported"));
    }
    let t_out: Option<Vec<f64>> = t_out
        .filter(|t| !t.is_none())
        .map(|t| f64_array(t, "t_out").map(|a| a.as_array().iter().copied().collect()))
        .transpose()?;
    with_precision!(precision_of(D), |F| {
        let (d, squeeze) = cplanes::<F>(D, "D")?;
        let out = match (rate, t_out.as_deref()) {
            (Some(rate), None) => fx::phase_vocoder(d.view(), rate),
            (None, Some(t)) => fx::phase_vocoder_at(d.view(), t),
            _ => {
                return Err(value_err(
                    "rate",
                    "exactly one of rate or t_out must be given",
                ));
            }
        }
        .map_err(audio_err_to_py)?;
        Ok(out3(py, out, squeeze))
    })
}

// =============================================================================
// Pre-/de-emphasis with filter state
// =============================================================================

/// Per-row emphasis with librosa's `zi` / `return_zf` handling.
fn emphasis<'py, F: PyFloat>(
    py: Python<'py>,
    y: &Bound<'py, PyAny>,
    coef: f64,
    zi: Option<&Bound<'py, PyAny>>,
    return_zf: bool,
    f: fn(&[F], f64, Option<F>) -> audio_samples::AudioSampleResult<(Vec<F>, F)>,
) -> PyResult<Bound<'py, PyAny>> {
    let (x, squeeze) = rows::<F>(y, "y")?;
    let n_ch = x.nrows();
    let zi: Option<Vec<F>> = zi
        .filter(|z| !z.is_none())
        .map(|z| -> PyResult<Vec<F>> {
            let v: Vec<F> = float_array_as::<F>(z, "zi")?
                .as_array()
                .iter()
                .copied()
                .collect();
            match v.len() {
                1 => Ok(vec![v[0]; n_ch]),
                n if n == n_ch => Ok(v),
                n => Err(value_err(
                    "zi",
                    format!("expected 1 or {n_ch} initial states (one per channel), got {n}"),
                )),
            }
        })
        .transpose()?;
    let mut out = Array2::<F>::zeros(x.dim());
    let mut zf = Vec::with_capacity(n_ch);
    for (c, row) in x.outer_iter().enumerate() {
        let row = row.to_vec();
        let (yc, z) = f(&row, coef, zi.as_ref().map(|z| z[c])).map_err(effects_err)?;
        out.row_mut(c).assign(&ndarray::Array1::from(yc));
        zf.push(z);
    }
    let y_out = rows_out(py, out, squeeze);
    if !return_zf {
        return Ok(y_out);
    }
    // librosa's zf has shape (..., 1): (1,) for 1-D input, (channels, 1) for 2-D.
    let zf_out = if squeeze {
        zf.into_pyarray(py).into_any()
    } else {
        Array2::from_shape_vec((n_ch, 1), zf)
            .expect("one state per channel")
            .into_pyarray(py)
            .into_any()
    };
    (y_out, zf_out).into_bound_py_any(py)
}

/// Pre-emphasis filter ``y[n] = x[n] - coef * x[n - 1]``, equivalent to
/// ``librosa.effects.preemphasis``.
///
/// Args:
///     y (array_like): Signal ``(n,)`` or ``(channels, n)``.
///     coef (float): Coefficient. Default 0.97.
///     zi (Optional[float | array_like]): Initial filter state (one value, or
///         one per channel). None uses librosa's ``2 * y[0] - y[1]``.
///     return_zf (bool): Also return the final state. Default False.
///
/// Returns:
///     numpy.ndarray[float32 | float64] of the input's shape, or ``(y, zf)`` with
///     ``return_zf=True`` (``zf`` of shape ``(1,)`` or ``(channels, 1)``).
///
/// Raises:
///     ValueError: For a non-finite ``coef`` / ``zi``, or fewer than two samples
///         with the default ``zi``.
#[pyfunction]
#[pyo3(
    signature = (y, *, coef = 0.97, zi = None, return_zf = false),
    text_signature = "(y, *, coef=0.97, zi=None, return_zf=False)"
)]
pub fn preemphasis<'py>(
    py: Python<'py>,
    y: &Bound<'py, PyAny>,
    coef: f64,
    zi: Option<&Bound<'py, PyAny>>,
    return_zf: bool,
) -> PyResult<Bound<'py, PyAny>> {
    with_precision!(precision_of(y), |F| emphasis::<F>(
        py,
        y,
        coef,
        zi,
        return_zf,
        fx::preemphasis_with_state::<F>
    ))
}

/// De-emphasis filter ``y[n] = x[n] + coef * y[n - 1]`` (inverse of
/// ``preemphasis``), equivalent to ``librosa.effects.deemphasis``.
///
/// Args:
///     y (array_like): Signal ``(n,)`` or ``(channels, n)``.
///     coef (float): Coefficient. Default 0.97 (3 only with an explicit ``zi``).
///     zi (Optional[float | array_like]): Initial filter state. None uses
///         librosa's default, which exactly inverts ``preemphasis`` with its
///         default state.
///     return_zf (bool): Also return the final state. Default False.
///
/// Returns:
///     numpy.ndarray[float32 | float64] of the input's shape, or ``(y, zf)``.
///
/// Raises:
///     ValueError: As ``preemphasis``.
#[pyfunction]
#[pyo3(
    signature = (y, *, coef = 0.97, zi = None, return_zf = false),
    text_signature = "(y, *, coef=0.97, zi=None, return_zf=False)"
)]
pub fn deemphasis<'py>(
    py: Python<'py>,
    y: &Bound<'py, PyAny>,
    coef: f64,
    zi: Option<&Bound<'py, PyAny>>,
    return_zf: bool,
) -> PyResult<Bound<'py, PyAny>> {
    with_precision!(precision_of(y), |F| emphasis::<F>(
        py,
        y,
        coef,
        zi,
        return_zf,
        fx::deemphasis_with_state::<F>
    ))
}

// =============================================================================
// Masks and HPSS
// =============================================================================

/// Soft mask ``X**p / (X**p + X_ref**p)``, equivalent to ``librosa.util.softmask``.
///
/// Args:
///     X (array_like): Non-negative values selected by the mask.
///     X_ref (array_like): Non-negative reference, same shape.
///     power (float): Exponent (> 0). ``numpy.inf`` gives a hard (bool) mask
///         ``X > X_ref``. Default 1.
///     split_zeros (bool): Mask value where both inputs are zero: 0.5 if True,
///         else 0. Default False.
///
/// Returns:
///     numpy.ndarray: values in ``[0, 1]`` in ``X``'s precision (bool for
///     ``power=inf``), same shape.
///
/// Raises:
///     ValueError: For mismatched shapes, negative input or ``power <= 0``.
#[pyfunction]
#[pyo3(
    signature = (X, X_ref, *, power = 1.0, split_zeros = false),
    text_signature = "(X, X_ref, *, power=1, split_zeros=False)"
)]
#[allow(non_snake_case)]
pub fn softmask<'py>(
    py: Python<'py>,
    X: &Bound<'py, PyAny>,
    X_ref: &Bound<'py, PyAny>,
    power: f64,
    split_zeros: bool,
) -> PyResult<Bound<'py, PyAny>> {
    with_precision!(precision_of(X), |F| {
        let x = float_array_as::<F>(X, "X")?;
        let r = float_array_as::<F>(X_ref, "X_ref")?;
        let m = fx::softmask(&x.as_array(), &r.as_array(), power, split_zeros)
            .map_err(audio_err_to_py)?;
        if power.is_infinite() {
            Ok(m.mapv(|v| v != 0.0).into_pyarray(py).into_any())
        } else {
            Ok(m.into_pyarray(py).into_any())
        }
    })
}

/// Median-filtering harmonic/percussive separation of a spectrogram,
/// equivalent to ``librosa.decompose.hpss``.
///
/// ``H`` is ``|S|`` median-filtered along time, ``P`` along frequency (SciPy
/// ``reflect`` boundary); the masks are ``softmask(H, margin_h * P)`` and
/// ``softmask(P, margin_p * H)``.
///
/// Args:
///     S (array_like): Magnitude/power spectrogram (real, non-negative) or
///         complex STFT, shape ``(n_bins, n_frames)`` or
///         ``(channels, n_bins, n_frames)``.
///     kernel_size (int | tuple[int, int]): Median filter lengths (harmonic
///         along time, percussive along frequency). Default 31.
///     power (float): Mask exponent (``numpy.inf`` for hard masks). Default 2.
///     mask (bool): Return the masks instead of the components. Default False.
///     margin (float | tuple[float, float]): Separation margins (>= 1).
///         Default 1.
///
/// Returns:
///     tuple: ``(harmonic, percussive)`` of the input's shape and dtype
///     (complex input keeps its phase), or with ``mask=True`` the masks in ``S``'s precision
///     (bool for ``power=inf``, as librosa).
///
/// Raises:
///     ValueError: For invalid parameters or negative magnitudes.
#[pyfunction]
#[pyo3(
    signature = (S, *, kernel_size = None, power = 2.0, mask = false, margin = None),
    text_signature = "(S, *, kernel_size=31, power=2.0, mask=False, margin=1.0)"
)]
#[allow(non_snake_case)]
pub fn hpss<'py>(
    py: Python<'py>,
    S: &Bound<'py, PyAny>,
    kernel_size: Option<&Bound<'py, PyAny>>,
    power: f64,
    mask: bool,
    margin: Option<&Bound<'py, PyAny>>,
) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)> {
    let ks = kernel_size
        .filter(|k| !k.is_none())
        .map_or_else(|| 31i64.into_bound_py_any(py), |k| Ok(k.clone()))?;
    let mg = margin
        .filter(|m| !m.is_none())
        .map_or_else(|| 1.0f64.into_bound_py_any(py), |m| Ok(m.clone()))?;
    let cfg = hpss_mask_config(&ks, power, &mg)?;
    let np = py.import("numpy")?;
    let complex = np.call_method1("iscomplexobj", (S,))?.extract::<bool>()?;
    with_precision!(precision_of(S), |F| hpss_typed::<F>(
        py, S, &cfg, complex, mask, power
    ))
}

/// [`hpss`] computed in `F`.
#[allow(non_snake_case)]
fn hpss_typed<'py, F: PyFloat>(
    py: Python<'py>,
    S: &Bound<'py, PyAny>,
    cfg: &HpssMaskConfig,
    complex: bool,
    mask: bool,
    power: f64,
) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)>
where
    Complex<F>: numpy::Element,
{
    let as_mask = |m: Array3<F>, squeeze: bool| -> Bound<'py, PyAny> {
        if power.is_infinite() {
            out3(py, m.mapv(|v| v != F::of(0.0)), squeeze)
        } else {
            out3(py, m, squeeze)
        }
    };
    if complex {
        let (d, squeeze) = cplanes::<F>(S, "S")?;
        if mask {
            let mag = d.mapv(Complex::norm);
            let mag = mag.view();
            let (mh, mp) = py
                .detach(|| fx::hpss_masks(mag, cfg))
                .map_err(audio_err_to_py)?;
            return Ok((as_mask(mh, squeeze), as_mask(mp, squeeze)));
        }
        let dv = d.view();
        let (h, p) = py
            .detach(|| fx::hpss_spectrogram(dv, cfg))
            .map_err(audio_err_to_py)?;
        return Ok((out3(py, h, squeeze), out3(py, p, squeeze)));
    }
    let (s, squeeze) = planes::<F>(S, "S")?;
    if mask {
        let sv = s.view();
        let (mh, mp) = py
            .detach(|| fx::hpss_masks(sv, cfg))
            .map_err(audio_err_to_py)?;
        return Ok((as_mask(mh, squeeze), as_mask(mp, squeeze)));
    }
    let sv = s.view();
    let (h, p) = py
        .detach(|| fx::hpss_magnitude(sv, cfg))
        .map_err(audio_err_to_py)?;
    Ok((out3(py, h, squeeze), out3(py, p, squeeze)))
}

/// Parses `nn_filter`'s `aggregate`: None/`numpy.mean`, `numpy.median`,
/// `numpy.average` (or their names).
enum NnArg<'py> {
    /// Computed by [`fx::nn_filter`] and its neighbours.
    Builtin(NnAggregate),
    /// A reduction of the neighbours over axis 0, without the GIL's cost per frame.
    Max,
    Min,
    Sum,
    /// Any other callable, `f(neighbours, axis=0)`.
    Callable(Bound<'py, PyAny>),
}

fn parse_nn_aggregate<'py>(obj: Option<&Bound<'py, PyAny>>) -> PyResult<NnArg<'py>> {
    let Some(obj) = obj.filter(|o| !o.is_none()) else {
        return Ok(NnArg::Builtin(NnAggregate::Mean));
    };
    if let Ok(s) = obj.cast::<PyString>() {
        return match &*s.to_cow()? {
            "mean" => Ok(NnArg::Builtin(NnAggregate::Mean)),
            "median" => Ok(NnArg::Builtin(NnAggregate::Median)),
            "average" => Ok(NnArg::Builtin(NnAggregate::WeightedAverage)),
            "max" => Ok(NnArg::Max),
            "min" => Ok(NnArg::Min),
            "sum" => Ok(NnArg::Sum),
            other => Err(value_err(
                "aggregate",
                format!(
                    "unknown aggregate {other:?}; expected 'mean', 'median', 'average', 'max', \
                     'min', 'sum' or a callable"
                ),
            )),
        };
    }
    if !obj.is_callable() {
        return Err(PyTypeError::new_err(
            "aggregate must be None, a name or a callable",
        ));
    }
    let np = obj.py().import("numpy")?;
    for (name, agg) in [
        ("mean", NnAggregate::Mean),
        ("median", NnAggregate::Median),
        ("average", NnAggregate::WeightedAverage),
    ] {
        if obj.is(&np.getattr(name)?) {
            return Ok(NnArg::Builtin(agg));
        }
    }
    if obj.is(&np.getattr("max")?) || obj.is(&np.getattr("amax")?) {
        return Ok(NnArg::Max);
    }
    if obj.is(&np.getattr("min")?) || obj.is(&np.getattr("amin")?) {
        return Ok(NnArg::Min);
    }
    if obj.is(&np.getattr("sum")?) {
        return Ok(NnArg::Sum);
    }
    Ok(NnArg::Callable(obj.clone()))
}

/// Reduces the `(n_neighbours, n_features)` neighbours over the neighbours with
/// `pick(best, value)`, in order; NaN propagates, as in NumPy's reductions.
fn nn_extreme<F: FloatElement + AudioFloat>(
    neighbours: ndarray::ArrayView2<'_, F>,
    frame: &mut [F],
    pick: impl Fn(F, F) -> bool,
) {
    for (o, column) in frame.iter_mut().zip(neighbours.columns()) {
        let mut best = column[0];
        for &v in column.iter() {
            if pick(v, best) || v.is_nan() {
                best = v;
            }
        }
        *o = best;
    }
}

/// One frame of a custom aggregate: `f(neighbours, axis=0)` must give one value per
/// feature, or a scalar for all of them, as assigning to a frame of a NumPy array does.
fn nn_call<'py, F: FloatElement>(
    f: &Bound<'py, PyAny>,
    kwargs: &Bound<'py, PyDict>,
    neighbours: ndarray::ArrayView2<'_, F>,
    frame: &mut [F],
) -> PyResult<()> {
    let py = f.py();
    let arr = neighbours.to_owned().into_pyarray(py);
    let res = f.call((arr,), Some(kwargs))?;
    let res = float_array_as::<F>(&res, "aggregate")?;
    let res = res.as_array();
    match res.ndim() {
        0 => frame.fill(res.iter().copied().next().expect("a 0-d array holds a value")),
        1 if res.len() == frame.len() => {
            for (o, &v) in frame.iter_mut().zip(res.iter()) {
                *o = v;
            }
        }
        _ => {
            return Err(value_err(
                "aggregate",
                format!(
                    "the callable returned shape {:?} for {} features",
                    res.shape(),
                    frame.len()
                ),
            ));
        }
    }
    Ok(())
}

/// A `scipy.sparse` recurrence matrix in compressed sparse column form.
struct SparseRec<'py> {
    csc: Bound<'py, PyAny>,
    indptr: Bound<'py, PyAny>,
    indices: Bound<'py, PyAny>,
}

/// The sparse form of `rec`, if it is one: `tocsc` is the sign of a `scipy.sparse`
/// array or matrix. A NumPy array is told apart without looking the attribute up.
fn sparse_rec<'py>(rec: &Bound<'py, PyAny>) -> PyResult<Option<SparseRec<'py>>> {
    if rec.cast::<numpy::PyUntypedArray>().is_ok() || !rec.hasattr("tocsc")? {
        return Ok(None);
    }
    let csc = rec.call_method0("tocsc")?;
    let indptr = csc.getattr("indptr")?;
    let indices = csc.getattr("indices")?;
    Ok(Some(SparseRec {
        csc,
        indptr,
        indices,
    }))
}

/// Nearest-neighbour filtering, equivalent to ``librosa.decompose.nn_filter``.
///
/// Each frame is replaced by an aggregate of its neighbours in a recurrence
/// matrix (``rec[j, i] != 0`` links frame ``j`` to output frame ``i``); frames
/// without neighbours are copied.
///
/// Args:
///     S (array_like): Features ``(n_features, n_frames)``.
///     rec (Optional[array_like]): ``(n_frames, n_frames)`` recurrence matrix, dense
///         (bool or float) or ``scipy.sparse`` in any format, where every stored entry
///         is a neighbour. None computes
///         ``audio_samples.sequence.recurrence_matrix(S, **kwargs)``.
///     aggregate: None / ``numpy.mean`` (default), ``numpy.median``, ``numpy.average``
///         (weighted by ``rec``; use an affinity matrix), ``numpy.max``, ``numpy.min``,
///         ``numpy.sum``, or any callable ``f(neighbours, axis=0)`` of the
///         ``(n_neighbours, n_features)`` neighbours of a frame, which runs with the
///         GIL held.
///     axis (int): Must be -1.
///     **kwargs: ``recurrence_matrix`` arguments when ``rec`` is None: ``k``,
///         ``width``, ``metric``, ``sym``, ``mode``, ``bandwidth``, ``self``,
///         ``full``.
///
/// Returns:
///     numpy.ndarray[float32 | float64]: filtered ``(n_features, n_frames)``.
///
/// Raises:
///     ValueError: For a wrong-shaped ``rec`` or invalid arguments.
///     TypeError: For an aggregate that is not callable or unknown keyword arguments.
#[pyfunction]
#[pyo3(
    signature = (S, *, rec = None, aggregate = None, axis = Int(-1), **kwargs),
    text_signature = "(S, *, rec=None, aggregate=None, axis=-1, **kwargs)"
)]
#[allow(non_snake_case)]
pub fn nn_filter<'py>(
    py: Python<'py>,
    S: &Bound<'py, PyAny>,
    rec: Option<&Bound<'py, PyAny>>,
    aggregate: Option<&Bound<'py, PyAny>>,
    axis: Int,
    kwargs: Option<&Bound<'py, PyDict>>,
) -> PyResult<Bound<'py, PyAny>> {
    if axis != Int(-1) && axis != Int(1) {
        return Err(value_err("axis", "only axis=-1 (frames last) is supported"));
    }
    let agg = parse_nn_aggregate(aggregate)?;
    let rec = rec.filter(|r| !r.is_none());
    if rec.is_some() && kwargs.is_some_and(|k| !k.is_empty()) {
        return Err(PyTypeError::new_err(
            "nn_filter: recurrence_matrix keyword arguments are only used when rec is None",
        ));
    }
    let sparse = rec.map(sparse_rec).transpose()?.flatten();
    let axis_kw = PyDict::new(py);
    axis_kw.set_item("axis", 0)?;
    with_precision!(precision_of(S), |F| {
        let (data, squeeze) = rows::<F>(S, "S")?;
        let mut first_err: Option<PyErr> = None;
        let mut custom = |n: ndarray::ArrayView2<'_, F>, frame: &mut [F]| match &agg {
            NnArg::Max => nn_extreme(n, frame, |v, best| v > best),
            NnArg::Min => nn_extreme(n, frame, |v, best| v < best),
            NnArg::Sum => {
                frame.fill(F::of(0.0));
                for row in n.rows() {
                    for (o, &v) in frame.iter_mut().zip(row.iter()) {
                        *o += v;
                    }
                }
            }
            NnArg::Callable(f) => {
                if first_err.is_none() {
                    first_err = nn_call(f, &axis_kw, n, frame).err();
                }
            }
            NnArg::Builtin(_) => unreachable!("built-in aggregates do not reach this closure"),
        };
        let out = if let Some(sp) = &sparse {
            let n_frames = data.ncols();
            let shape: (usize, usize) = sp.csc.getattr("shape")?.extract()?;
            if shape != (n_frames, n_frames) {
                return Err(value_err(
                    "rec",
                    format!(
                        "invalid self-similarity matrix shape {shape:?} for S with {n_frames} frames"
                    ),
                ));
            }
            // The weights are read in their own float type, without a converted copy;
            // other types (bool connectivity) are converted.
            let stored = match &agg {
                NnArg::Builtin(NnAggregate::WeightedAverage) => Some(sp.csc.getattr("data")?),
                _ => None,
            };
            let w32 = stored
                .as_ref()
                .and_then(|d| d.extract::<numpy::PyReadonlyArray1<'_, f32>>().ok());
            let w64 = match (&stored, &w32) {
                (Some(d), None) => Some(match d.extract::<numpy::PyReadonlyArray1<'_, f64>>() {
                    Ok(w) => w,
                    Err(_) => float_array_as::<f64>(d, "rec")?
                        .as_array()
                        .to_owned()
                        .into_dimensionality::<ndarray::Ix1>()
                        .map_err(|e| value_err("rec", e))?
                        .into_pyarray(py)
                        .try_readonly()?,
                }),
                _ => None,
            };
            let slice_of = |e| value_err("rec", e);
            let v32 = w32.as_ref().map(|w| w.as_slice()).transpose().map_err(slice_of)?;
            let v64 = w64.as_ref().map(|w| w.as_slice()).transpose().map_err(slice_of)?;
            macro_rules! with_values {
                ($int:ty, $values:expr) => {{
                    let indptr = sp.indptr.extract::<numpy::PyReadonlyArray1<'_, $int>>()?;
                    let indices = sp.indices.extract::<numpy::PyReadonlyArray1<'_, $int>>()?;
                    let rec = fx::SparseRecurrence {
                        indptr: indptr.as_slice().map_err(|e| value_err("rec", e))?,
                        indices: indices.as_slice().map_err(|e| value_err("rec", e))?,
                        values: $values,
                    };
                    match &agg {
                        NnArg::Builtin(a) => fx::nn_filter_sparse(data.view(), &rec, *a),
                        _ => fx::nn_filter_sparse_with(data.view(), &rec, &mut custom),
                    }
                }};
            }
            macro_rules! filtered {
                ($int:ty) => {{
                    match (v32, v64) {
                        (Some(v), _) => with_values!($int, Some(v)),
                        (None, v) => with_values!($int, v),
                    }
                }};
            }
            if sp.indptr.extract::<numpy::PyReadonlyArray1<'_, i32>>().is_ok() {
                filtered!(i32)
            } else {
                filtered!(i64)
            }
        } else if let Some(r) = rec {
            let r = float_array_as::<F>(r, "rec")?;
            if r.ndim() != 2 {
                return Err(value_err("rec", "expected a 2-D recurrence matrix"));
            }
            let r = r
                .as_array()
                .into_dimensionality::<Ix2>()
                .expect("2-D")
                .to_owned();
            match &agg {
                NnArg::Builtin(a) => fx::nn_filter(data.view(), r.view(), *a),
                _ => fx::nn_filter_with(data.view(), r.view(), &mut custom),
            }
        } else {
            let cfg = recurrence_config(kwargs)?;
            match &agg {
                NnArg::Builtin(a) => fx::nn_filter_recurrence(data.view(), &cfg, *a),
                _ => fx::nn_filter_recurrence_with(data.view(), &cfg, &mut custom),
            }
        };
        if let Some(e) = first_err {
            return Err(e);
        }
        let out = out.map_err(audio_err_to_py)?;
        Ok(rows_out(py, out, squeeze))
    })
}

/// [`RecurrenceConfig`] from `recurrence_matrix` keyword arguments.
fn recurrence_config(kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<RecurrenceConfig> {
    let mut cfg = RecurrenceConfig::default();
    let Some(kw) = kwargs else {
        return Ok(cfg);
    };
    let mut bandwidth = None;
    for (key, value) in kw.iter() {
        let key: String = key.extract()?;
        match key.as_str() {
            "k" => {
                cfg.k = if value.is_none() {
                    None
                } else {
                    Some(positive("k", value.extract::<Int>()?)?.get())
                };
            }
            "width" => cfg.width = positive("width", value.extract::<Int>()?)?.get(),
            "metric" => cfg.metric = crate::sequence::parse_metric(&value.extract::<String>()?)?,
            "sym" => cfg.sym = value.extract()?,
            "mode" => cfg.mode = crate::sequence::parse_mode(&value.extract::<String>()?)?,
            "bandwidth" => bandwidth = Some(value),
            "self" | "self_links" => cfg.self_links = value.extract()?,
            "full" => cfg.full = value.extract()?,
            // The neighbour lists are sparse either way, as librosa forces sparse=True.
            "sparse" => {}
            "axis" => {
                let a: i64 = value.extract()?;
                if a != -1 && a != 1 {
                    return Err(value_err("axis", "only axis=-1 is supported"));
                }
            }
            other => {
                return Err(PyTypeError::new_err(format!(
                    "nn_filter() got an unexpected keyword argument '{other}'"
                )));
            }
        }
    }
    if let Some(b) = bandwidth {
        cfg.bandwidth = crate::sequence::parse_bandwidth(Some(&b))?;
    }
    Ok(cfg)
}

// =============================================================================
// Griffin-Lim, LPC, clicks
// =============================================================================

/// Parses `init` / `rng` into a [`GriffinLimInit`].
pub(crate) fn griffinlim_init(init: Option<&str>, rng: RngArg) -> PyResult<GriffinLimInit> {
    match init {
        None => Ok(GriffinLimInit::ZeroPhase),
        Some("random") => Ok(GriffinLimInit::Random(rng.0)),
        Some(other) => Err(value_err(
            "init",
            format!("unsupported init {other:?}; expected 'random' or None"),
        )),
    }
}

/// The `rng` argument of the Griffin-Lim bindings: an integer seed, or `None`
/// for a fresh seed on every call. A NumPy generator, or anything else
/// `numpy.random.default_rng` takes, gives one seed drawn from its stream, which
/// advances the generator; the phases then come from this library's RNG.
#[derive(Debug, Clone, Copy)]
pub struct RngArg(pub(crate) Option<u64>);

impl RngArg {
    /// The default seed, 0 (reproducible output).
    pub(crate) const SEEDED: Self = Self(Some(0));
}

impl<'a, 'py> FromPyObject<'a, 'py> for RngArg {
    type Error = PyErr;

    #[inline]
    fn extract(obj: pyo3::Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        if obj.is_none() {
            return Ok(Self(None));
        }
        if let Ok(s) = obj.extract::<i64>() {
            return Ok(Self(Some(non_negative("rng", s)? as u64)));
        }
        Ok(Self(Some(seed_from_numpy(&obj)?)))
    }
}

/// One seed in `[0, 2**63)` drawn from a `numpy.random.Generator` or `RandomState`, or
/// from `numpy.random.default_rng(obj)` for anything else that function takes (a
/// sequence of integers, a `SeedSequence`, a `BitGenerator`).
fn seed_from_numpy(obj: &Bound<'_, PyAny>) -> PyResult<u64> {
    let py = obj.py();
    let high = (1u64 << 63) - 1;
    let invalid = |e: PyErr| {
        PyTypeError::new_err(format!(
            "rng: expected an integer seed, None, a NumPy generator or a value \
             numpy.random.default_rng accepts ({e})"
        ))
    };
    let seed: i64 = if obj.hasattr("integers")? {
        obj.call_method1("integers", (0, high))?.extract()?
    } else if obj.hasattr("randint")? {
        let kwargs = PyDict::new(py);
        kwargs.set_item("dtype", "int64")?;
        obj.call_method("randint", (0, high), Some(&kwargs))?
            .extract()?
    } else {
        py.import("numpy")?
            .getattr("random")?
            .call_method1("default_rng", (obj,))
            .map_err(invalid)?
            .call_method1("integers", (0, high))?
            .extract()?
    };
    Ok(seed as u64)
}

/// [`GriffinLimConfig`] from librosa's `griffinlim` keyword arguments.
#[allow(clippy::too_many_arguments)]
pub(crate) fn griffinlim_config(
    n_bins: usize,
    n_iter: Int,
    hop_length: Option<Int>,
    win_length: Option<Int>,
    n_fft: Option<Int>,
    window: WindowArg,
    center: bool,
    length: Option<Int>,
    pad_mode: &str,
    momentum: f64,
    init: Option<&str>,
    rng: RngArg,
) -> PyResult<GriffinLimConfig> {
    let n_fft = n_fft.unwrap_or(Int(2 * (n_bins as i64 - 1)));
    Ok(GriffinLimConfig::new()
        .with_n_iter(non_negative("n_iter", n_iter)?)
        .with_momentum(momentum)
        .with_init(griffinlim_init(init, rng)?)
        .with_stft(stft_config(
            n_fft, hop_length, win_length, window, center, pad_mode,
        )?)
        .with_length(crate::args::opt_positive("length", length)?.map(NonZeroUsize::get)))
}

/// Reconstructs a signal from a magnitude spectrogram with fast Griffin-Lim,
/// equivalent to ``librosa.griffinlim``.
///
/// Args:
///     S (array_like): Non-negative magnitude ``(1 + n_fft // 2, n_frames)`` or
///         ``(channels, 1 + n_fft // 2, n_frames)``.
///     n_iter (int): Iterations. Default 32.
///     hop_length, win_length, window, center, pad_mode: STFT parameters of ``S``.
///     n_fft (Optional[int]): None infers ``2 * (n_bins - 1)``.
///     dtype: Ignored; the output follows ``S``'s precision (float32 for a
///         float32 ``S``, float64 otherwise).
///     length (Optional[int]): Exact output length.
///     momentum (float): Fast Griffin-Lim momentum (0 = original). Default 0.99.
///     init (Optional[str]): ``"random"`` (default) or None (zero phase,
///         deterministic).
///     rng: Seed for ``init="random"``: an integer, None for a fresh seed on every
///         call, or a NumPy generator or anything else ``numpy.random.default_rng``
///         takes, from which one seed is drawn (advancing the generator). The random
///         phases come from this library's generator, so they differ from librosa's
///         for the same seed.
///     random_state: Deprecated alias of ``rng``.
///
/// Returns:
///     numpy.ndarray[float32 | float64]: shape ``(n,)`` for 2-D ``S``, ``(channels, n)`` for 3-D.
///
/// Raises:
///     ValueError: For invalid parameters or a bin count that does not match ``n_fft``.
#[pyfunction]
#[pyo3(
    signature = (S, *, n_iter = Int(32), hop_length = None, win_length = None, n_fft = None, window = WindowArg(Window::Hann), center = true, dtype = None, length = None, pad_mode = "constant", momentum = 0.99, init = Some("random"), rng = RngArg::SEEDED, random_state = None),
    text_signature = "(S, *, n_iter=32, hop_length=None, win_length=None, n_fft=None, window='hann', center=True, dtype=None, length=None, pad_mode='constant', momentum=0.99, init='random', rng=0, random_state=None)"
)]
#[allow(non_snake_case, clippy::too_many_arguments, unused_variables)]
pub fn griffinlim<'py>(
    py: Python<'py>,
    S: &Bound<'py, PyAny>,
    n_iter: Int,
    hop_length: Option<Int>,
    win_length: Option<Int>,
    n_fft: Option<Int>,
    window: WindowArg,
    center: bool,
    dtype: Option<&Bound<'py, PyAny>>,
    length: Option<Int>,
    pad_mode: &str,
    momentum: f64,
    init: Option<&str>,
    rng: RngArg,
    random_state: Option<RngArg>,
) -> PyResult<Bound<'py, PyAny>> {
    let rng = random_state.unwrap_or(rng);
    with_precision!(precision_of(S), |F| {
        let (s, squeeze) = planes::<F>(S, "S")?;
        let cfg = griffinlim_config(
            s.dim().1,
            n_iter,
            hop_length,
            win_length,
            n_fft,
            window,
            center,
            length,
            pad_mode,
            momentum,
            init,
            rng,
        )?;
        let view = s.view();
        let y = py
            .detach(|| fx::griffinlim(view, &cfg))
            .map_err(audio_err_to_py)?;
        Ok(rows_out(py, y, squeeze))
    })
}

/// Linear-prediction coefficients by Burg's method, equivalent to ``librosa.lpc``.
///
/// Args:
///     y (array_like): Finite signal ``(n,)`` or ``(channels, n)`` with more
///         than ``order`` samples.
///     order (int): Prediction order.
///     axis (int): Must be -1.
///
/// Returns:
///     numpy.ndarray[float32 | float64]: ``(order + 1,)`` coefficients ``[1, a_1, ...]``
///     (``(channels, order + 1)`` for 2-D input).
///
/// Raises:
///     ValueError: For a too-short or non-finite signal.
#[pyfunction]
#[pyo3(signature = (y, *, order, axis = Int(-1)), text_signature = "(y, *, order, axis=-1)")]
pub fn lpc<'py>(
    py: Python<'py>,
    y: &Bound<'py, PyAny>,
    order: Int,
    axis: Int,
) -> PyResult<Bound<'py, PyAny>> {
    if axis != Int(-1) {
        let a = f64_array(y, "y")?;
        if !(a.ndim() == 1 && axis == Int(0)) {
            return Err(value_err("axis", "only axis=-1 is supported"));
        }
    }
    let order = positive("order", order)?;
    with_precision!(precision_of(y), |F| {
        let (x, squeeze) = rows::<F>(y, "y")?;
        if order.get() >= x.ncols() {
            return Err(value_err(
                "order",
                format!(
                    "order={order} needs more than {order} samples (got {})",
                    x.ncols()
                ),
            ));
        }
        let mut out = Array2::<F>::zeros((x.nrows(), order.get() + 1));
        for (c, row) in x.outer_iter().enumerate() {
            // Contiguous rows are passed in place.
            let a = match row.as_slice() {
                Some(r) => fx::lpc(r, order),
                None => fx::lpc(&row.to_vec(), order),
            }
            .map_err(audio_err_to_py)?;
            out.row_mut(c).assign(&ndarray::Array1::from(a));
        }
        Ok(rows_out(py, out, squeeze))
    })
}

/// Synthesises a click track, equivalent to ``librosa.clicks``.
///
/// Args:
///     times (Optional[array_like]): Click times in seconds.
///     frames (Optional[array_like]): Click frame indices (used when ``times``
///         is None).
///     sr (int): Sampling rate (a positive integer). Required (librosa defaults to 22050).
///     hop_length (Optional[int]): Samples per frame for ``frames``. Required
///         when ``frames`` is given (librosa defaults to 512); unused with
///         ``times``.
///     click_freq (float): Frequency of the default click. Default 1000.
///     click_duration (float): Duration of the default click (s). Default 0.1.
///     click (Optional[array_like]): Custom 1-D click waveform.
///     length (Optional[int]): Output length; clicks starting at or after it
///         are dropped. None: up to the end of the last click.
///     dtype (Optional[numpy.dtype]): Output dtype, ``numpy.float32`` (librosa's
///         dtype) or ``numpy.float64``. Default None: float64.
///
/// Returns:
///     numpy.ndarray[float64 | float32]: 1-D click signal (float64 unless
///     ``dtype`` says otherwise; librosa returns float32, and the values agree
///     to float32 rounding).
///
/// Raises:
///     ValueError: If neither ``times`` nor ``frames`` is given, a time is
///         negative, or parameters are invalid.
#[pyfunction]
#[pyo3(
    signature = (*, times = None, frames = None, sr, hop_length = None, click_freq = 1000.0, click_duration = 0.1, click = None, length = None, dtype = None),
    text_signature = "(*, times=None, frames=None, sr, hop_length=None, click_freq=1000.0, click_duration=0.1, click=None, length=None, dtype=None)"
)]
#[allow(clippy::too_many_arguments)]
pub fn clicks<'py>(
    py: Python<'py>,
    times: Option<&Bound<'py, PyAny>>,
    frames: Option<&Bound<'py, PyAny>>,
    sr: f64,
    hop_length: Option<Int>,
    click_freq: f64,
    click_duration: f64,
    click: Option<&Bound<'py, PyAny>>,
    length: Option<Int>,
    dtype: Option<crate::args::DTypeArg<'py>>,
) -> PyResult<Bound<'py, PyAny>> {
    use numpy::PyArrayDescrMethods;
    let single = match &dtype {
        None => false,
        Some(d) if d.0.is_equiv_to(&numpy::dtype::<f32>(py)) => true,
        Some(d) if d.0.is_equiv_to(&numpy::dtype::<f64>(py)) => false,
        Some(d) => {
            return Err(value_err(
                "dtype",
                format!("expected numpy.float32 or numpy.float64, got {}", d.0),
            ));
        }
    };
    if !(sr.is_finite() && sr >= 1.0 && sr.fract() == 0.0 && sr <= f64::from(u32::MAX)) {
        return Err(value_err(
            "sr",
            format!("must be a positive integer, got {sr}"),
        ));
    }
    let rate = crate::nzu32_or_err(sr as u32)?;
    let hop = crate::args::opt_positive("hop_length", hop_length)?;
    let click = click
        .filter(|c| !c.is_none())
        .map(|c| crate::rhythm::vec1::<f64>(c, "click").map(std::borrow::Cow::into_owned))
        .transpose()?;
    let cfg = ClicksConfig::new()
        .with_click_freq(click_freq)
        .with_click_duration(click_duration)
        .with_click(click)
        .with_length(crate::args::opt_positive("length", length)?.map(NonZeroUsize::get));
    let times = times.filter(|t| !t.is_none());
    let frames = frames.filter(|f| !f.is_none());
    let (t, f, hop) = match (times, frames) {
        (Some(t), _) => {
            let t: Vec<f64> = f64_array(t, "times")?.as_array().iter().copied().collect();
            (Some(t), None, None)
        }
        (None, Some(f)) => {
            let Some(hop) = hop else {
                return Err(value_err(
                    "hop_length",
                    "required with frames= (the hop of the analysis that produced the frames; \
                     librosa defaults to 512)",
                ));
            };
            let f: Vec<usize> = i64_vec(f, "frames")?
                .into_iter()
                .map(|v| non_negative("frames", v))
                .collect::<PyResult<_>>()?;
            if f.iter().any(|&v| v.checked_mul(hop.get()).is_none()) {
                return Err(value_err("frames", "frame index too large"));
            }
            (None, Some(f), Some(hop))
        }
        (None, None) => {
            return Err(value_err(
                "times",
                "either times or frames must be provided",
            ));
        }
    };
    let positions = match (&t, &f, hop) {
        (Some(t), _, _) => ClickPositions::Times(t),
        (None, Some(f), Some(hop)) => ClickPositions::Frames {
            frames: f,
            hop_length: hop,
        },
        _ => unreachable!("positions checked above"),
    };
    // The samples are moved out of the result (no copy of the click track).
    if single {
        let y = fx::clicks::<f32>(positions, rate, &cfg).map_err(effects_err)?;
        let data = y
            .into_array1()
            .ok_or_else(|| value_err("clicks", "unexpected multi-channel output"))?;
        Ok(data.into_pyarray(py).into_any())
    } else {
        let y = fx::clicks::<f64>(positions, rate, &cfg).map_err(effects_err)?;
        let data = y
            .into_array1()
            .ok_or_else(|| value_err("clicks", "unexpected multi-channel output"))?;
        Ok(data.into_pyarray(py).into_any())
    }
}

/// Registers `audio_samples.effects`.
pub fn register(py: Python<'_>, parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(py, "effects")?;
    m.add(
        "__doc__",
        "librosa-compatible effect and decomposition helpers over NumPy arrays \
         (phase vocoder, pre-/de-emphasis with filter state, soft masks, HPSS, \
         nearest-neighbour filtering, Griffin-Lim, LPC, click tracks). \
         float32 / complex64 arrays are processed in float32 and give float32 \
         output; any other input is converted to float64.",
    )?;
    m.add_function(wrap_pyfunction!(phase_vocoder, &m)?)?;
    m.add_function(wrap_pyfunction!(preemphasis, &m)?)?;
    m.add_function(wrap_pyfunction!(deemphasis, &m)?)?;
    m.add_function(wrap_pyfunction!(softmask, &m)?)?;
    m.add_function(wrap_pyfunction!(hpss, &m)?)?;
    m.add_function(wrap_pyfunction!(nn_filter, &m)?)?;
    m.add_function(wrap_pyfunction!(griffinlim, &m)?)?;
    m.add_function(wrap_pyfunction!(lpc, &m)?)?;
    m.add_function(wrap_pyfunction!(clicks, &m)?)?;
    parent.add_submodule(&m)?;
    Ok(())
}
