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
//! * Array inputs are taken as `&Bound<PyAny>`. Functions that compute on an
//!   array keep its precision: a `float32` (or `complex64`) numpy array is
//!   processed in `f32` and gives `float32` / `complex64` output; any other
//!   array-like (lists, integer arrays, `float64`, `float16`, ...) is converted
//!   with `numpy.asarray(x, dtype=float64)` (or `complex128`) and gives
//!   `float64` / `complex128` output. See [`precision_of`] and [`with_precision`].
//! * Analysis methods of `AudioSamples` return `float32` / `complex64` arrays
//!   for `uint8`, `int16`, `int24` and `float32` audio, and `float64` /
//!   `complex128` arrays for `int32` and `float64` audio (the Rust sample
//!   type's `Float`).

use std::num::NonZeroUsize;

use audio_samples::operations::filterbanks::{MelFilterNorm, MelFilterbankConfig, Window};
use audio_samples::operations::spectrum::{DbAxes, DbReference, StftConfig, StftPadMode};
use audio_samples::utils::audio_math::MelScale;
use num_complex::Complex;
use numpy::{
    PyArrayDescrMethods, PyArrayDyn, PyArrayMethods, PyReadonlyArrayDyn, PyUntypedArrayMethods,
};
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

/// Floating-point precision of an array argument, chosen by [`precision_of`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Precision {
    /// `float32` / `complex64`.
    F32,
    /// `float64` / `complex128`.
    F64,
}

/// Precision a free function computes in for the array-like `obj`: [`Precision::F32`]
/// for a numpy array of dtype `float32` or `complex64` and for `float32` audio,
/// [`Precision::F64`] for everything else (other dtypes, lists and scalars are converted to `float64`).
pub(crate) fn precision_of(obj: &Bound<'_, PyAny>) -> Precision {
    let py = obj.py();
    if let Ok(arr) = obj.cast::<numpy::PyUntypedArray>() {
        let dt = arr.dtype();
        if dt.is_equiv_to(&numpy::dtype::<f32>(py))
            || dt.is_equiv_to(&numpy::dtype::<Complex<f32>>(py))
        {
            return Precision::F32;
        }
    } else if let Ok(audio) = obj.cast::<crate::PyAudioSamples>() {
        // Passed as an array: float32 audio keeps its precision.
        if matches!(audio.borrow().inner(), crate::PyAudioDataInner::F32(_)) {
            return Precision::F32;
        }
    }
    Precision::F64
}

/// A real floating-point element type of a numpy array (`f32` or `f64`).
pub(crate) trait FloatElement: numpy::Element + Copy {
    /// The numpy name of the real dtype.
    const REAL_NAME: &'static str;
    /// The numpy name of the matching complex dtype.
    const COMPLEX_NAME: &'static str;
    /// `v` rounded to this type (as `numpy.asarray(v, dtype=...)` does).
    fn from_f64(v: f64) -> Self;
}

impl FloatElement for f32 {
    const REAL_NAME: &'static str = "float32";
    const COMPLEX_NAME: &'static str = "complex64";
    #[allow(clippy::cast_possible_truncation)]
    fn from_f64(v: f64) -> Self {
        v as f32
    }
}

impl FloatElement for f64 {
    const REAL_NAME: &'static str = "float64";
    const COMPLEX_NAME: &'static str = "complex128";
    fn from_f64(v: f64) -> Self {
        v
    }
}

/// The float types the bindings compute in (`f32`, `f64`): the Rust crate's
/// [`AudioFloat`](audio_samples::AudioFloat) that is also a numpy element.
pub(crate) trait PyFloat: audio_samples::AudioFloat + FloatElement {}

impl PyFloat for f32 {}
impl PyFloat for f64 {}

/// Runs `$body` with the type alias `$F` set to `f32` or `f64` according to a
/// [`Precision`]. Both arms must evaluate to the same type (convert arrays to a
/// Python object inside the body).
macro_rules! with_precision {
    ($prec:expr, |$F:ident| $body:expr) => {
        match $prec {
            $crate::args::Precision::F32 => {
                #[allow(dead_code)]
                type $F = f32;
                $body
            }
            $crate::args::Precision::F64 => {
                #[allow(dead_code)]
                type $F = f64;
                $body
            }
        }
    };
}
pub(crate) use with_precision;

/// `numpy.asarray(arr, dtype=descr)` for an ndarray `arr` through the NumPy
/// C API (`PyArray_CheckFromAny` with `NPY_ARRAY_FORCECAST`, as `numpy.asarray`
/// does): no copy when the dtype already matches, otherwise a cast that keeps
/// the memory order.
fn asarray_cast<'py>(
    arr: &Bound<'py, numpy::PyUntypedArray>,
    descr: Bound<'py, numpy::PyArrayDescr>,
) -> PyResult<Bound<'py, PyAny>> {
    use numpy::PyArrayDescrMethods as _;
    use numpy::npyffi::{NPY_ARRAY_FORCECAST, PY_ARRAY_API};
    let py = arr.py();
    // SAFETY: `PyArray_CheckFromAny` steals the descriptor reference handed
    // over by `into_dtype_ptr` and returns a new reference (or NULL with an
    // exception set).
    unsafe {
        let ptr = PY_ARRAY_API.PyArray_CheckFromAny(
            py,
            arr.as_ptr(),
            descr.into_dtype_ptr(),
            0,
            0,
            NPY_ARRAY_FORCECAST,
            std::ptr::null_mut(),
        );
        Bound::from_owned_ptr_or_err(py, ptr)
    }
}

/// The value of a Python `float` or `int` (including `bool`) as `f64`, or
/// `None` for anything else (and for ints too large for `f64`, which take the
/// `numpy.asarray` path and its error).
fn python_real(obj: &Bound<'_, PyAny>) -> Option<f64> {
    use pyo3::types::{PyFloat, PyInt};
    if obj.is_instance_of::<PyFloat>() || obj.is_instance_of::<PyInt>() {
        obj.extract::<f64>().ok()
    } else {
        None
    }
}

/// Converts an array-like to a read-only real array of element type `F`
/// (`numpy.asarray(obj, dtype=F)`), of any dimension.
///
/// A NumPy array that already has dtype `F` is used as is (no Python-level
/// calls, no copy); other NumPy arrays are cast through the C API; anything
/// else goes through `numpy.asarray`.
///
/// Complex input raises `TypeError` rather than silently dropping the
/// imaginary part.
pub(crate) fn float_array_as<'py, F: FloatElement>(
    obj: &Bound<'py, PyAny>,
    name: &str,
) -> PyResult<PyReadonlyArrayDyn<'py, F>> {
    if let Ok(arr) = obj.cast::<PyArrayDyn<F>>() {
        return Ok(arr.try_readonly()?);
    }
    let py = obj.py();
    let complex_err = || {
        PyTypeError::new_err(format!(
            "{name}: expected a real-valued array, got complex values (take numpy.abs first)"
        ))
    };
    let conv_err = |e: &dyn std::fmt::Display| {
        PyTypeError::new_err(format!(
            "{name}: cannot convert to a {} array ({e})",
            F::REAL_NAME
        ))
    };
    let arr = if let Ok(a) = obj.cast::<numpy::PyUntypedArray>() {
        if a.dtype().kind() == b'c' {
            return Err(complex_err());
        }
        asarray_cast(a, numpy::dtype::<F>(py)).map_err(|e| conv_err(&e))?
    } else if let Some(v) = python_real(obj) {
        // A Python float or int: a 0-d array built directly.
        numpy::PyArray::from_owned_array(py, ndarray::arr0(F::from_f64(v)).into_dyn()).into_any()
    } else {
        let np = numpy(py)?;
        if np.call_method1("iscomplexobj", (obj,))?.extract::<bool>()? {
            return Err(complex_err());
        }
        let kwargs = PyDict::new(py);
        kwargs.set_item("dtype", np.getattr(F::REAL_NAME)?)?;
        np.call_method("asarray", (obj,), Some(&kwargs))
            .map_err(|e| conv_err(&e))?
    };
    let arr = arr.cast_into::<PyArrayDyn<F>>().map_err(|e| conv_err(&e))?;
    Ok(arr.try_readonly()?)
}

/// Converts an array-like to a read-only complex array of element type
/// `Complex<F>` (`numpy.asarray(obj, dtype=complex64/complex128)`).
///
/// A NumPy array that already has that dtype is used as is; other NumPy arrays
/// are cast through the C API; anything else goes through `numpy.asarray`.
pub(crate) fn complex_array_as<'py, F: FloatElement>(
    obj: &Bound<'py, PyAny>,
    name: &str,
) -> PyResult<PyReadonlyArrayDyn<'py, Complex<F>>>
where
    Complex<F>: numpy::Element,
{
    if let Ok(arr) = obj.cast::<PyArrayDyn<Complex<F>>>() {
        return Ok(arr.try_readonly()?);
    }
    let py = obj.py();
    let conv_err = |e: &dyn std::fmt::Display| {
        PyTypeError::new_err(format!(
            "{name}: cannot convert to a {} array ({e})",
            F::COMPLEX_NAME
        ))
    };
    let arr = if let Ok(a) = obj.cast::<numpy::PyUntypedArray>() {
        asarray_cast(a, numpy::dtype::<Complex<F>>(py)).map_err(|e| conv_err(&e))?
    } else {
        let np = numpy(py)?;
        let kwargs = PyDict::new(py);
        kwargs.set_item("dtype", np.getattr(F::COMPLEX_NAME)?)?;
        np.call_method("asarray", (obj,), Some(&kwargs))
            .map_err(|e| conv_err(&e))?
    };
    let arr = arr
        .cast_into::<PyArrayDyn<Complex<F>>>()
        .map_err(|e| conv_err(&e))?;
    Ok(arr.try_readonly()?)
}

/// An array argument as a `CowArray` borrowing `obj`'s data when it is a
/// NumPy array of element type `E` (no copy), or an owned copy of the
/// converted array (`convert`) otherwise. The borrowed view lives as long as
/// `obj`.
///
/// On a free-threaded build the result is always a copy, made under a
/// rust-numpy borrow: another thread can start writing the array as soon as a
/// borrow is released, and this function's result outlives any borrow it takes.
pub(crate) fn array_cow<'a, 'py, E: numpy::Element + Clone>(
    obj: &'a Bound<'py, PyAny>,
    convert: impl FnOnce(&Bound<'py, PyAny>) -> PyResult<PyReadonlyArrayDyn<'py, E>>,
) -> PyResult<ndarray::CowArray<'a, E, ndarray::IxDyn>> {
    #[cfg(not(Py_GIL_DISABLED))]
    if let Ok(arr) = obj.cast::<PyArrayDyn<E>>() {
        // Fails if Rust code holds a mutable borrow of the array; the read-only
        // borrow is released at once and the data viewed in place.
        drop(arr.try_readonly()?);
        // SAFETY: the array lives as long as `obj` (which the view borrows),
        // and no Rust code holds a mutable borrow of it (checked above, and the
        // only such borrow is taken with the GIL held). As with the read-only
        // borrows used elsewhere, Python code is not expected to modify the
        // array while the call runs.
        return Ok(ndarray::CowArray::from(unsafe { arr.as_array() }));
    }
    Ok(ndarray::CowArray::from(convert(obj)?.as_array().to_owned()))
}

/// Runs `f` on a view of [`float_array_as`]`(obj)`.
///
/// With the GIL, a NumPy array that already has element type `F` is viewed in
/// place without taking a rust-numpy borrow: each borrow allocates an entry in
/// rust-numpy's borrow table (about 180 bytes), which for a small conversion
/// is a large part of the call's memory. No binding holds a mutable borrow of
/// an argument while it calls back into Python, so none can be active when a
/// binding starts.
///
/// On a free-threaded build a mutable borrow can be active on another thread,
/// so `f` runs under a real borrow, and the call raises `TypeError` if the array
/// is being written. Other inputs are converted as by [`float_array_as`].
pub(crate) fn with_float_view<F: FloatElement, R>(
    obj: &Bound<'_, PyAny>,
    name: &str,
    f: impl FnOnce(ndarray::ArrayViewD<'_, F>) -> PyResult<R>,
) -> PyResult<R> {
    #[cfg(not(Py_GIL_DISABLED))]
    if let Ok(arr) = obj.cast::<PyArrayDyn<F>>() {
        // SAFETY: the array lives as long as `obj`, which outlives `f`; no Rust
        // code holds a mutable borrow of it (see above), and `f` does not run
        // Python code that could create one.
        return f(unsafe { arr.as_array() });
    }
    let arr = float_array_as::<F>(obj, name)?;
    f(arr.as_array())
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
    float_array_as::<f64>(obj, name)
}

/// A 2-D `(n_bins, T)` or 3-D `(channels, n_bins, T)` array as 3-D (a 2-D
/// array becomes one channel).
pub(crate) fn to_3d<T: Clone>(
    a: ndarray::ArrayViewD<'_, T>,
    name: &str,
) -> PyResult<ndarray::Array3<T>> {
    use ndarray::{Axis, Ix2, Ix3};
    match a.ndim() {
        2 => Ok(a
            .into_dimensionality::<Ix2>()
            .expect("ndim checked")
            .insert_axis(Axis(0))
            .to_owned()),
        3 => Ok(a
            .into_dimensionality::<Ix3>()
            .expect("ndim checked")
            .to_owned()),
        n => Err(value_err(
            name,
            format!(
                "expected a 2-D (n_bins, T) or 3-D (channels, n_bins, T) array, got {n} dimensions"
            ),
        )),
    }
}

/// Runs `$body` with the type alias `$T` set to the Rust sample type of a
/// `SampleType` (`u8`, `i16`, `I24`, `i32`, `f32` or `f64`).
macro_rules! with_sample_type {
    ($st:expr, |$T:ident| $body:expr) => {
        match $st {
            $crate::types::PySampleType::U8 => {
                type $T = u8;
                $body
            }
            $crate::types::PySampleType::I16 => {
                type $T = i16;
                $body
            }
            $crate::types::PySampleType::I24 => {
                type $T = audio_samples::I24;
                $body
            }
            $crate::types::PySampleType::I32 => {
                type $T = i32;
                $body
            }
            $crate::types::PySampleType::F32 => {
                type $T = f32;
                $body
            }
            $crate::types::PySampleType::F64 => {
                type $T = f64;
                $body
            }
        }
    };
}
pub(crate) use with_sample_type;

/// Sample type of the audio returned by the static reconstruction methods
/// (`istft_matrix`, `icqt`, `griffinlim_cqt`): the `dtype` argument if given,
/// otherwise `float32` audio for a `complex64`/`float32` input matrix and
/// `float64` audio for anything else.
pub(crate) fn output_sample_type(
    dtype: Option<crate::types::PySampleType>,
    matrix: &Bound<'_, PyAny>,
) -> crate::types::PySampleType {
    dtype.unwrap_or_else(|| match precision_of(matrix) {
        Precision::F32 => crate::types::PySampleType::F32,
        Precision::F64 => crate::types::PySampleType::F64,
    })
}

/// Largest magnitude accepted from a non-integer array-like of integers: 2**53,
/// the range in which `float64` holds every integer exactly.
const MAX_EXACT_INT: f64 = 9.007_199_254_740_992e15;

/// Runs `f` on an array-like of integers as an `int64` view of any shape.
///
/// Integer (and bool) NumPy arrays are read directly (`int64`) or cast through
/// the C API, without a range check (the Rust conversions saturate). Anything
/// else is converted to `float64` and must hold integers below 2**53 in
/// magnitude (the range in which `float64` holds every integer exactly).
pub(crate) fn with_i64_view<R>(
    obj: &Bound<'_, PyAny>,
    name: &str,
    f: impl FnOnce(ndarray::ArrayViewD<'_, i64>) -> PyResult<R>,
) -> PyResult<R> {
    if let Ok(arr) = obj.cast::<PyArrayDyn<i64>>() {
        #[cfg(Py_GIL_DISABLED)]
        return f(arr.try_readonly()?.as_array());
        // SAFETY: as in `with_float_view` (viewed without a rust-numpy
        // borrow); the array lives as long as `obj`, which outlives `f`.
        #[cfg(not(Py_GIL_DISABLED))]
        return f(unsafe { arr.as_array() });
    }
    if let Ok(a) = obj.cast::<numpy::PyUntypedArray>() {
        let dt = a.dtype();
        // Signed ints of any size, unsigned below 64 bits and bool convert to
        // int64 exactly; uint64 takes the float64 path (range check).
        if matches!(dt.kind(), b'i' | b'b') || (dt.kind() == b'u' && dt.itemsize() < 8) {
            let c = asarray_cast(a, numpy::dtype::<i64>(obj.py()))?;
            let c = c
                .cast_into::<PyArrayDyn<i64>>()
                .map_err(|e| value_err(name, e))?;
            let r = c.readonly();
            return f(r.as_array());
        }
    }
    let mut bad = None;
    let ints = with_float_view::<f64, _>(obj, name, |arr| {
        Ok(arr.mapv(|v| {
            if v.is_finite() && v.fract() == 0.0 && v.abs() < MAX_EXACT_INT {
                v as i64
            } else {
                bad.get_or_insert(v);
                0
            }
        }))
    })?;
    if let Some(v) = bad {
        return Err(value_err(name, format!("expected integer values, got {v}")));
    }
    f(ints.view())
}

/// Converts an array-like of integers to a `Vec<i64>` (flattened, logical
/// order), rejecting non-integral values.
pub(crate) fn i64_vec(obj: &Bound<'_, PyAny>, name: &str) -> PyResult<Vec<i64>> {
    with_i64_view(obj, name, |v| Ok(v.iter().copied().collect()))
}

/// `True` if `obj` is a scalar for NumPy (`numpy.ndim(obj) == 0`).
///
/// NumPy arrays, Python numbers, lists and tuples are answered without
/// calling into NumPy.
pub(crate) fn is_scalar(obj: &Bound<'_, PyAny>) -> PyResult<bool> {
    use pyo3::types::{PyFloat, PyInt, PyList};
    if let Ok(a) = obj.cast::<numpy::PyUntypedArray>() {
        return Ok(a.ndim() == 0);
    }
    if obj.is_instance_of::<PyFloat>() || obj.is_instance_of::<PyInt>() {
        return Ok(true);
    }
    if obj.is_instance_of::<PyList>() || obj.is_instance_of::<PyTuple>() {
        return Ok(false);
    }
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
    if let Some(v) = python_real(obj) {
        return f(v).into_bound_py_any(py);
    }
    with_float_view::<f64, _>(obj, name, |arr| {
        if is_scalar(obj)? {
            let v = *arr.iter().next().expect("0-d array has one element");
            return f(v).into_bound_py_any(py);
        }
        Ok(arr.mapv(f).into_pyarray(py).into_any())
    })
}

/// Like [`map_f64`] but producing integers (`int` / `int64` array).
pub(crate) fn map_f64_to_i64<'py>(
    obj: &Bound<'py, PyAny>,
    name: &str,
    f: impl Fn(f64) -> i64,
) -> PyResult<Bound<'py, PyAny>> {
    use numpy::IntoPyArray;
    let py = obj.py();
    with_float_view::<f64, _>(obj, name, |arr| {
        if is_scalar(obj)? {
            let v = *arr.iter().next().expect("0-d array has one element");
            return f(v).into_bound_py_any(py);
        }
        Ok(arr.mapv(f).into_pyarray(py).into_any())
    })
}

/// Runs a slice kernel over the elements of `v` and returns the result as a
/// NumPy array of `v`'s shape. C- or Fortran-contiguous input is passed in
/// place (the output keeps that memory order); anything else is gathered in
/// logical order first.
fn slice_map_out<'py, T: Copy, R: numpy::Element>(
    py: Python<'py>,
    v: ndarray::ArrayViewD<'_, T>,
    name: &str,
    slice_f: impl FnOnce(&[T]) -> Vec<R>,
) -> PyResult<Bound<'py, PyAny>> {
    use ndarray::ShapeBuilder;
    use numpy::IntoPyArray;
    let fortran = !v.is_standard_layout() && v.t().is_standard_layout();
    let (out, fortran) = match v.as_slice_memory_order() {
        Some(flat) if v.is_standard_layout() || fortran => (slice_f(flat), fortran),
        _ => (slice_f(&v.iter().copied().collect::<Vec<T>>()), false),
    };
    let shape = ndarray::IxDyn(v.shape());
    let arr = if fortran {
        ndarray::ArrayD::from_shape_vec(shape.f(), out)
    } else {
        ndarray::ArrayD::from_shape_vec(shape, out)
    }
    .map_err(|e| value_err(name, e))?;
    Ok(arr.into_pyarray(py).into_any())
}

/// [`map_f64`] for conversions with a slice form: a scalar input gives
/// `scalar_f(x)` as a Python number, an array-like gives `slice_f` over its
/// elements as an array of the same shape. `slice_f` must agree with
/// `scalar_f` element by element.
pub(crate) fn map_f64_slice<'py, R>(
    obj: &Bound<'py, PyAny>,
    name: &str,
    scalar_f: impl Fn(f64) -> R,
    slice_f: impl FnOnce(&[f64]) -> Vec<R>,
) -> PyResult<Bound<'py, PyAny>>
where
    R: numpy::Element + for<'a> IntoPyObject<'a> + Copy,
{
    let py = obj.py();
    if let Some(v) = python_real(obj) {
        return scalar_f(v).into_bound_py_any(py);
    }
    with_float_view::<f64, _>(obj, name, |arr| {
        if is_scalar(obj)? {
            let v = *arr.iter().next().expect("0-d array has one element");
            return scalar_f(v).into_bound_py_any(py);
        }
        slice_map_out(py, arr, name, slice_f)
    })
}

/// [`map_i64`] for conversions with a slice form (see [`map_f64_slice`]).
pub(crate) fn map_i64_slice<'py, R>(
    obj: &Bound<'py, PyAny>,
    name: &str,
    scalar_f: impl Fn(i64) -> R,
    slice_f: impl FnOnce(&[i64]) -> Vec<R>,
) -> PyResult<Bound<'py, PyAny>>
where
    R: numpy::Element + for<'a> IntoPyObject<'a> + Copy,
{
    let py = obj.py();
    let scalar = is_scalar(obj)?;
    with_i64_view(obj, name, |v| {
        if scalar {
            let x = *v.iter().next().expect("0-d array has one element");
            return scalar_f(x).into_bound_py_any(py);
        }
        slice_map_out(py, v, name, slice_f)
    })
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
    let scalar = is_scalar(obj)?;
    with_i64_view(obj, name, |v| {
        if scalar {
            let x = *v.iter().next().expect("0-d array has one element");
            return f(x).into_bound_py_any(py);
        }
        Ok(v.mapv(&f).into_pyarray(py).into_any())
    })
}

// =============================================================================
// Windows, padding, filterbank norms, dB references
// =============================================================================

/// Accepted window names, for error messages.
const WINDOW_NAMES: &str = "'hann', 'hamming', 'blackman', 'blackmanharris', 'boxcar' \
    (or 'rectangular'/'ones'), 'bartlett', 'triang', ('kaiser', beta), ('gaussian', std), \
    a number (Kaiser beta, as SciPy), or an array of coefficients";

fn window_from_name(name: &str) -> PyResult<Window> {
    // Lower-case short names on the stack (no allocation per call).
    let mut buf = [0u8; 32];
    let owned;
    let lower: &str = if name.len() <= buf.len() {
        let b = &mut buf[..name.len()];
        b.copy_from_slice(name.as_bytes());
        b.make_ascii_lowercase();
        std::str::from_utf8(b).expect("ASCII lower-casing keeps UTF-8 valid")
    } else {
        owned = name.to_ascii_lowercase();
        &owned
    };
    match lower {
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
