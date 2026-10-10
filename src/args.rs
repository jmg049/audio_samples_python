//! Shared argument parsing for the librosa-compatible bindings.
//!
//! The `audio_samples.spectral`, `audio_samples.sequence` and the
//! librosa-style `AudioSamples` methods (`stft_matrix`, `melspectrogram`, ...)
//! take librosa's keyword arguments: windows as SciPy-style strings, tuples or
//! arrays, `ref=numpy.max`, `norm="slaney"`, array-likes of any dtype, and so on.
//! Every binding parses them with the helpers in this module so that the
//! accepted spellings and the error messages are identical everywhere.
//!
//! Conventions:
//!
//! * Integer parameters are taken as [`Int`] (an `i64` that saturates instead of
//!   raising `OverflowError` for Python ints outside the 64-bit range) and checked
//!   here, so a negative or huge value raises `ValueError` naming the parameter.
//!   Size-like integers ([`positive`], [`opt_positive`], [`non_negative`]) are
//!   capped at [`MAX_SIZE_PARAM`] (2**31): larger values would only ever request
//!   impossible allocations, which abort the interpreter instead of raising.
//! * Array inputs are taken as `&Bound<PyAny>` and converted with
//!   `numpy.asarray(x, dtype=float64)` (or `complex128`), so lists, integer
//!   arrays and float32 arrays are all accepted, as in librosa.
//! * Outputs are always new `float64` / `complex128` / `int64` NumPy arrays.

use std::num::NonZeroUsize;

use audio_samples::operations::filterbanks::{MelFilterNorm, MelFilterbankConfig, Window};
use audio_samples::operations::spectrum::{DbAxes, DbReference, StftConfig, StftPadMode};
use audio_samples::utils::audio_math::MelScale;
use num_complex::Complex;
use numpy::{PyArrayDyn, PyArrayMethods, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::IntoPyObjectExt;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyString, PyTuple};

/// `ValueError(f"{name} ...")`.
pub(crate) fn value_err(name: &str, msg: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(format!("{name}: {msg}"))
}

/// Largest value accepted by the size-like integer helpers ([`positive`],
/// [`opt_positive`], [`non_negative`]): 2**31, the Rust crate's limit for
/// signal-sized buffers. An FFT size, a window length, a number of states or a
/// sample count beyond it can only request an allocation that fails, and Rust
/// aborts the whole process on allocation failure.
pub(crate) const MAX_SIZE_PARAM: i64 = 1 << 31;

/// A Python `int` parameter.
///
/// Extracted like `i64`, except that ints outside the 64-bit range saturate to
/// `i64::MIN` / `i64::MAX` instead of raising pyo3's `OverflowError`; the
/// validation that follows then raises `ValueError` naming the parameter.
/// Non-integers still raise `TypeError`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Int(pub i64);

impl<'a, 'py> FromPyObject<'a, 'py> for Int {
    type Error = PyErr;

    fn extract(obj: pyo3::Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        match obj.extract::<i64>() {
            Ok(v) => Ok(Self(v)),
            Err(e) if e.is_instance_of::<pyo3::exceptions::PyOverflowError>(obj.py()) => {
                Ok(Self(if obj.lt(0)? { i64::MIN } else { i64::MAX }))
            }
            Err(e) => Err(e),
        }
    }
}

impl From<Int> for i64 {
    fn from(v: Int) -> Self {
        v.0
    }
}

impl std::fmt::Display for Int {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.0 {
            i64::MAX => f.write_str(">= 2**63"),
            i64::MIN => f.write_str("<= -2**63"),
            v => write!(f, "{v}"),
        }
    }
}

/// Rejects a size-like integer above [`MAX_SIZE_PARAM`].
fn check_size(name: &str, v: i64) -> PyResult<()> {
    if v > MAX_SIZE_PARAM {
        Err(value_err(
            name,
            format!("must be at most {MAX_SIZE_PARAM} (2**31), got {}", Int(v)),
        ))
    } else {
        Ok(())
    }
}

/// A `sample_rate` keyword given as a Python int, as `u32` (out of range ->
/// `ValueError`; zero is rejected by the caller's `NonZeroU32` conversion).
pub(crate) fn sample_rate_u32(v: Int) -> PyResult<u32> {
    u32::try_from(v.0).map_err(|_| {
        value_err(
            "sample_rate",
            format!("must be a positive integer below 2**32, got {v}"),
        )
    })
}

/// An axis or signed offset parameter as `isize`.
pub(crate) fn signed(name: &str, v: Int) -> PyResult<isize> {
    isize::try_from(v.0).map_err(|_| value_err(name, format!("out of range, got {v}")))
}

/// A strictly positive, size-like integer parameter (at most [`MAX_SIZE_PARAM`]).
pub(crate) fn positive(name: &str, v: impl Into<i64>) -> PyResult<NonZeroUsize> {
    let v = v.into();
    check_size(name, v)?;
    usize::try_from(v)
        .ok()
        .and_then(NonZeroUsize::new)
        .ok_or_else(|| value_err(name, format!("must be a positive integer, got {}", Int(v))))
}

/// An optional strictly positive integer parameter (`None` stays `None`).
pub(crate) fn opt_positive<T: Into<i64>>(
    name: &str,
    v: Option<T>,
) -> PyResult<Option<NonZeroUsize>> {
    v.map(|v| positive(name, v)).transpose()
}

/// A non-negative, size-like integer parameter (at most [`MAX_SIZE_PARAM`]).
pub(crate) fn non_negative(name: &str, v: impl Into<i64>) -> PyResult<usize> {
    let v = v.into();
    check_size(name, v)?;
    usize::try_from(v).map_err(|_| value_err(name, format!("must be >= 0, got {}", Int(v))))
}

/// `numpy` module handle.
fn numpy(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    py.import("numpy")
}

/// Converts an array-like to a read-only `float64` array of any dimension
/// (`numpy.asarray(obj, dtype=numpy.float64)`).
///
/// Complex input raises `TypeError` rather than silently dropping the
/// imaginary part.
pub(crate) fn f64_array<'py>(
    obj: &Bound<'py, PyAny>,
    name: &str,
) -> PyResult<PyReadonlyArrayDyn<'py, f64>> {
    let py = obj.py();
    let np = numpy(py)?;
    if np.call_method1("iscomplexobj", (obj,))?.extract::<bool>()? {
        return Err(PyTypeError::new_err(format!(
            "{name}: expected a real-valued array, got complex values (take numpy.abs first)"
        )));
    }
    let kwargs = PyDict::new(py);
    kwargs.set_item("dtype", np.getattr("float64")?)?;
    let arr = np
        .call_method("asarray", (obj,), Some(&kwargs))
        .map_err(|e| {
            PyTypeError::new_err(format!("{name}: cannot convert to a float64 array ({e})"))
        })?;
    let arr = arr.cast_into::<PyArrayDyn<f64>>().map_err(|e| {
        PyTypeError::new_err(format!("{name}: cannot convert to a float64 array ({e})"))
    })?;
    Ok(arr.readonly())
}

/// Converts an array-like to a read-only `complex128` array.
pub(crate) fn c128_array<'py>(
    obj: &Bound<'py, PyAny>,
    name: &str,
) -> PyResult<PyReadonlyArrayDyn<'py, Complex<f64>>> {
    let py = obj.py();
    let np = numpy(py)?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("dtype", np.getattr("complex128")?)?;
    let arr = np
        .call_method("asarray", (obj,), Some(&kwargs))
        .map_err(|e| {
            PyTypeError::new_err(format!(
                "{name}: cannot convert to a complex128 array ({e})"
            ))
        })?;
    let arr = arr.cast_into::<PyArrayDyn<Complex<f64>>>().map_err(|e| {
        PyTypeError::new_err(format!(
            "{name}: cannot convert to a complex128 array ({e})"
        ))
    })?;
    Ok(arr.readonly())
}

/// Converts an array-like of integers to a `Vec<i64>` (flattened), rejecting
/// non-integral values.
pub(crate) fn i64_vec(obj: &Bound<'_, PyAny>, name: &str) -> PyResult<Vec<i64>> {
    let arr = f64_array(obj, name)?;
    arr.as_array()
        .iter()
        .map(|&v| {
            if v.is_finite() && v.fract() == 0.0 && v.abs() < 9.007_199_254_740_992e15 {
                Ok(v as i64)
            } else {
                Err(value_err(name, format!("expected integer values, got {v}")))
            }
        })
        .collect()
}

/// `True` if `obj` is a scalar for NumPy (`numpy.ndim(obj) == 0`).
pub(crate) fn is_scalar(obj: &Bound<'_, PyAny>) -> PyResult<bool> {
    Ok(numpy(obj.py())?
        .call_method1("ndim", (obj,))?
        .extract::<usize>()?
        == 0)
}

/// Applies `f` element-wise, librosa style: a scalar input gives a Python
/// `float`, an array-like gives a `float64` array of the same shape.
pub(crate) fn map_f64<'py>(
    obj: &Bound<'py, PyAny>,
    name: &str,
    f: impl Fn(f64) -> f64,
) -> PyResult<Bound<'py, PyAny>> {
    use numpy::IntoPyArray;
    let py = obj.py();
    let arr = f64_array(obj, name)?;
    if is_scalar(obj)? {
        let v = *arr
            .as_array()
            .iter()
            .next()
            .expect("0-d array has one element");
        return f(v).into_bound_py_any(py);
    }
    Ok(arr.as_array().mapv(f).into_pyarray(py).into_any())
}

/// Like [`map_f64`] but producing integers (`int` / `int64` array).
pub(crate) fn map_f64_to_i64<'py>(
    obj: &Bound<'py, PyAny>,
    name: &str,
    f: impl Fn(f64) -> i64,
) -> PyResult<Bound<'py, PyAny>> {
    use numpy::IntoPyArray;
    let py = obj.py();
    let arr = f64_array(obj, name)?;
    if is_scalar(obj)? {
        let v = *arr
            .as_array()
            .iter()
            .next()
            .expect("0-d array has one element");
        return f(v).into_bound_py_any(py);
    }
    Ok(arr.as_array().mapv(f).into_pyarray(py).into_any())
}

/// Integer-in variant of [`map_f64`]: the input must hold integers.
pub(crate) fn map_i64<'py, R>(
    obj: &Bound<'py, PyAny>,
    name: &str,
    f: impl Fn(i64) -> R,
) -> PyResult<Bound<'py, PyAny>>
where
    R: numpy::Element + for<'a> IntoPyObject<'a> + Copy,
{
    use numpy::IntoPyArray;
    let py = obj.py();
    let values = i64_vec(obj, name)?;
    if is_scalar(obj)? {
        return f(values[0]).into_bound_py_any(py);
    }
    let arr = f64_array(obj, name)?;
    let shape = arr.shape().to_vec();
    let out: Vec<R> = values.into_iter().map(f).collect();
    let out = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&shape), out)
        .map_err(|e| value_err(name, e))?;
    Ok(out.into_pyarray(py).into_any())
}

// =============================================================================
// Windows, padding, filterbank norms, dB references
// =============================================================================

/// Accepted window names, for error messages.
const WINDOW_NAMES: &str = "'hann', 'hamming', 'blackman', 'blackmanharris', 'boxcar' \
    (or 'rectangular'/'ones'), 'bartlett', 'triang', ('kaiser', beta), ('gaussian', std), \
    a number (Kaiser beta, as SciPy), or an array of coefficients";

fn window_from_name(name: &str) -> PyResult<Window> {
    match name.to_ascii_lowercase().as_str() {
        "hann" | "hanning" | "han" => Ok(Window::Hann),
        "hamming" | "hamm" | "ham" => Ok(Window::Hamming),
        "blackman" | "black" | "blk" => Ok(Window::Blackman),
        "blackmanharris" | "blackman_harris" | "blackman-harris" | "bh" => {
            Ok(Window::BlackmanHarris)
        }
        "boxcar" | "box" | "ones" | "rect" | "rectangular" => Ok(Window::Rectangular),
        "bartlett" | "bart" | "brt" => Ok(Window::Bartlett),
        "triang" | "triangle" | "tri" => Ok(Window::Triangular),
        "kaiser" | "gaussian" => Err(value_err(
            "window",
            format!("the {name} window needs a parameter; pass a tuple such as ('{name}', 8.6)"),
        )),
        other => Err(value_err(
            "window",
            format!("unknown window {other:?}; expected {WINDOW_NAMES}"),
        )),
    }
}

/// Parses a librosa/SciPy window specification.
///
/// Accepts a name (`"hann"`, `"hamming"`, `"blackman"`, `"blackmanharris"`,
/// `"boxcar"`/`"rectangular"`, `"bartlett"`, `"triang"`), a tuple
/// `("kaiser", beta)` / `("gaussian", std)` (or `(name,)`), a bare number
/// (a Kaiser `beta`, as `scipy.signal.get_window` does), or an array of
/// coefficients (used as given, like librosa).
pub(crate) fn parse_window(obj: &Bound<'_, PyAny>) -> PyResult<Window> {
    if let Ok(s) = obj.cast::<PyString>() {
        return window_from_name(&s.to_cow()?);
    }
    if let Ok(t) = obj.cast::<PyTuple>() {
        let name: String = t
            .get_item(0)
            .map_err(|_| value_err("window", "empty window tuple"))?
            .extract()
            .map_err(|_| value_err("window", "the first tuple element must be a window name"))?;
        if t.len() == 1 {
            return window_from_name(&name);
        }
        if t.len() != 2 {
            return Err(value_err(
                "window",
                format!(
                    "expected (name, parameter), got a tuple of length {}",
                    t.len()
                ),
            ));
        }
        let param: f64 = t
            .get_item(1)?
            .extract()
            .map_err(|_| value_err("window", "the window parameter must be a number"))?;
        return match name.to_ascii_lowercase().as_str() {
            "kaiser" | "ksr" => Ok(Window::Kaiser { beta: param }),
            "gaussian" | "gauss" | "gss" => Ok(Window::Gaussian { std: param }),
            other => Err(value_err(
                "window",
                format!("window {other:?} takes no parameter; expected {WINDOW_NAMES}"),
            )),
        };
    }
    if obj.is_callable() {
        return Err(PyTypeError::new_err(
            "window: callables are not supported; pass a window name, tuple or array",
        ));
    }
    if is_scalar(obj)? {
        let beta: f64 = obj
            .extract()
            .map_err(|_| value_err("window", format!("expected {WINDOW_NAMES}")))?;
        return Ok(Window::Kaiser { beta });
    }
    let arr = f64_array(obj, "window")?;
    if arr.ndim() != 1 {
        return Err(value_err(
            "window",
            "a window array must be one-dimensional",
        ));
    }
    Ok(Window::Custom(arr.as_array().iter().copied().collect()))
}

/// Parses librosa's `pad_mode` (the subset supported by the Rust STFT).
pub(crate) fn parse_pad_mode(mode: &str) -> PyResult<StftPadMode> {
    match mode {
        "constant" => Ok(StftPadMode::Constant),
        "reflect" => Ok(StftPadMode::Reflect),
        "edge" => Ok(StftPadMode::Edge),
        "symmetric" => Ok(StftPadMode::Symmetric),
        other => Err(value_err(
            "pad_mode",
            format!(
                "unsupported pad_mode {other:?}; expected 'constant', 'reflect', 'edge' or 'symmetric'"
            ),
        )),
    }
}

/// Builds an [`StftConfig`] from librosa's STFT keyword arguments.
pub(crate) fn stft_config(
    n_fft: Int,
    hop_length: Option<Int>,
    win_length: Option<Int>,
    window: WindowArg,
    center: bool,
    pad_mode: &str,
) -> PyResult<StftConfig> {
    Ok(StftConfig {
        n_fft: positive("n_fft", n_fft)?,
        hop_length: opt_positive("hop_length", hop_length)?,
        win_length: opt_positive("win_length", win_length)?,
        window: window.0,
        center,
        pad_mode: parse_pad_mode(pad_mode)?,
    })
}

/// Parses a mel filter `norm`: `None`, `"slaney"` or a number `p` (Lp norm,
/// `numpy.inf` allowed).
pub(crate) fn parse_mel_norm(obj: &Bound<'_, PyAny>) -> PyResult<MelFilterNorm> {
    if obj.is_none() {
        return Ok(MelFilterNorm::None);
    }
    if let Ok(s) = obj.cast::<PyString>() {
        return match &*s.to_cow()? {
            "slaney" => Ok(MelFilterNorm::Slaney),
            other => Err(value_err(
                "norm",
                format!("unsupported norm {other:?}; expected 'slaney', None or a number"),
            )),
        };
    }
    let p: f64 = obj
        .extract()
        .map_err(|_| value_err("norm", "expected 'slaney', None or a number"))?;
    Ok(MelFilterNorm::Lp(p))
}

/// Builds a [`MelFilterbankConfig`] from librosa's mel keyword arguments.
pub(crate) fn mel_config(
    n_mels: Int,
    fmin: f64,
    fmax: Option<f64>,
    htk: bool,
    norm: MelNormArg,
) -> PyResult<MelFilterbankConfig> {
    Ok(MelFilterbankConfig {
        n_mels: positive("n_mels", n_mels)?,
        fmin,
        fmax,
        scale: mel_scale(htk),
        norm: norm.0,
    })
}

/// `htk` flag to [`MelScale`].
pub(crate) const fn mel_scale(htk: bool) -> MelScale {
    if htk { MelScale::Htk } else { MelScale::Slaney }
}

/// Parses a dB reference: a number, the string `"max"`, or `numpy.max` /
/// `numpy.amax` (librosa's `ref=np.max`).
pub(crate) fn parse_db_ref(obj: &Bound<'_, PyAny>) -> PyResult<DbReference> {
    let py = obj.py();
    if let Ok(s) = obj.cast::<PyString>() {
        return match &*s.to_cow()? {
            "max" => Ok(DbReference::Max),
            other => Err(value_err(
                "ref",
                format!("unsupported reference {other:?}; expected a number, 'max' or numpy.max"),
            )),
        };
    }
    if obj.is_callable() {
        let np = numpy(py)?;
        if obj.is(&np.getattr("max")?) || obj.is(&np.getattr("amax")?) {
            return Ok(DbReference::Max);
        }
        return Err(PyTypeError::new_err(
            "ref: only numpy.max (or the string 'max') is supported as a callable reference; \
             compute other references yourself and pass the number",
        ));
    }
    let v: f64 = obj
        .extract()
        .map_err(|_| value_err("ref", "expected a number, 'max' or numpy.max"))?;
    Ok(DbReference::Value(v))
}

/// Parses librosa's `axes` argument of the dB conversions: `"auto"` or `None`.
pub(crate) fn parse_db_axes(obj: &Bound<'_, PyAny>) -> PyResult<DbAxes> {
    if obj.is_none() {
        return Ok(DbAxes::All);
    }
    match obj.extract::<String>().as_deref() {
        Ok("auto") => Ok(DbAxes::Auto),
        _ => Err(value_err(
            "axes",
            "only 'auto' (per channel for arrays with 3+ dimensions) and None (whole array) are supported",
        )),
    }
}

// =============================================================================
// Keyword-argument newtypes
// =============================================================================
//
// pyo3 signature defaults must be Rust values, and an `Option<Bound<PyAny>>`
// cannot tell "argument omitted" from an explicit `None` (which matters for
// `norm=None`). These newtypes implement `FromPyObject` with the parsers above,
// so a binding writes e.g. `window = WindowArg(Window::Hann)` as its default.

macro_rules! arg_newtype {
    ($(#[$m:meta])* $name:ident($inner:ty) via $parse:ident) => {
        $(#[$m])*
        #[derive(Debug, Clone, PartialEq)]
        pub struct $name(pub $inner);

        impl<'a, 'py> FromPyObject<'a, 'py> for $name {
            type Error = PyErr;

            fn extract(obj: pyo3::Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
                $parse(&obj.to_owned()).map($name)
            }
        }
    };
}

arg_newtype!(
    /// A window specification, parsed by [`parse_window`].
    WindowArg(Window) via parse_window
);
arg_newtype!(
    /// A mel filter `norm` (`None`, `"slaney"` or a number), parsed by [`parse_mel_norm`].
    MelNormArg(MelFilterNorm) via parse_mel_norm
);
arg_newtype!(
    /// A dB reference (number, `"max"` or `numpy.max`), parsed by [`parse_db_ref`].
    DbRefArg(DbReference) via parse_db_ref
);
arg_newtype!(
    /// The dB `axes` argument (`"auto"` or `None`), parsed by [`parse_db_axes`].
    DbAxesArg(DbAxes) via parse_db_axes
);

/// A NumPy dtype given in any form `numpy.dtype()` accepts: a `numpy.dtype`
/// instance, a scalar type such as `numpy.float64`, or a string like `"int16"`.
#[derive(Debug, Clone)]
pub struct DTypeArg<'py>(pub Bound<'py, numpy::PyArrayDescr>);

impl<'a, 'py> FromPyObject<'a, 'py> for DTypeArg<'py> {
    type Error = PyErr;

    fn extract(obj: pyo3::Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let np = numpy(obj.py())?;
        let d = np
            .getattr("dtype")?
            .call1((obj,))
            .map_err(|e| PyTypeError::new_err(format!("dtype: not a NumPy dtype ({e})")))?;
        let d = d
            .cast_into::<numpy::PyArrayDescr>()
            .map_err(|e| PyTypeError::new_err(format!("dtype: {e}")))?;
        Ok(Self(d))
    }
}
