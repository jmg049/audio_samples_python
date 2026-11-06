//! `audio_samples.arrays`: array helpers, binding `audio_samples::utils::array_ops`.
//!
//! Functions accept any ``NumPy`` numeric dtype and keep it where the result has the
//! input's element type. Results computed in floating point follow the precision
//! rule of the other array modules: `float32` and `complex64` stay single precision,
//! everything else is computed in `float64`.

use audio_samples::utils::array_ops::{
    self as ops, AxisSortKey, FindFilesConfig, InterpKind, PadMode,
};
use ndarray::{ArrayD, ArrayViewD, Axis, Ix2, Zip};
use num_complex::Complex;
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyArrayDescr, PyArrayDescrMethods, PyArrayDyn, PyArrayMethods,
    PyUntypedArray, PyUntypedArrayMethods,
};
use pyo3::IntoPyObjectExt;
use pyo3::buffer::PyBuffer;
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyBytes, PyComplex, PyDict, PyFloat, PyInt, PyList, PyString, PyTuple, PyType};

use crate::args::{
    FloatElement, Int, Precision, PyFloat as PyFloatElement, i64_vec, precision_of, value_err,
    with_float_view, with_precision,
};
use crate::audio_err_to_py;

/// Inputs below this many elements are computed without releasing the GIL, which
/// costs more than the work it would share.
const DETACH_ABOVE: usize = 1 << 14;

fn detached<R: Send>(py: Python<'_>, n: usize, f: impl FnOnce() -> R + Send) -> R {
    if n >= DETACH_ABOVE { py.detach(f) } else { f() }
}

fn numpy_attr<'py>(py: Python<'py>, name: &str) -> PyResult<Bound<'py, PyAny>> {
    py.import("numpy")?.getattr(name)
}

fn as_ndarray<'py>(obj: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyUntypedArray>> {
    if let Ok(a) = obj.cast::<PyUntypedArray>() {
        return Ok(a.clone());
    }
    Ok(numpy_attr(obj.py(), "asarray")?
        .call1((obj,))?
        .cast_into::<PyUntypedArray>()?)
}

pub fn norm_axis(name: &str, axis: i64, ndim: usize) -> PyResult<Axis> {
    let n = ndim as i64;
    if axis < -n || axis >= n {
        return Err(value_err(
            name,
            format!("{axis} is out of range for {ndim} dimensions"),
        ));
    }
    Ok(Axis(if axis < 0 { axis + n } else { axis } as usize))
}

/// Runs `$body` with `$a` the array cast to its element type `$T`, for each ordered dtype.
macro_rules! each_ordered {
    ($arr:expr, |$a:ident, $T:ident| $body:expr, else $other:expr) => {{
        let __o = $arr;
        each_ordered!(@try __o, $a, $T, $body, $other, f64, f32, i64, i32, i16, i8, u64, u32, u16, u8, bool)
    }};
    (@try $o:ident, $a:ident, $T:ident, $body:expr, $other:expr, $t0:ty $(, $t:ty)*) => {
        if let Ok($a) = $o.cast::<PyArrayDyn<$t0>>() {
            #[allow(dead_code)]
            type $T = $t0;
            $body
        } else {
            each_ordered!(@try $o, $a, $T, $body, $other $(, $t)*)
        }
    };
    (@try $o:ident, $a:ident, $T:ident, $body:expr, $other:expr) => { $other };
}

/// As [`each_ordered`], and the complex dtypes.
macro_rules! each_numeric {
    ($arr:expr, |$a:ident, $T:ident| $body:expr, else $other:expr) => {{
        let __o = $arr;
        if let Ok($a) = __o.cast::<PyArrayDyn<Complex<f64>>>() {
            #[allow(dead_code)]
            type $T = Complex<f64>;
            $body
        } else if let Ok($a) = __o.cast::<PyArrayDyn<Complex<f32>>>() {
            #[allow(dead_code)]
            type $T = Complex<f32>;
            $body
        } else {
            each_ordered!(__o, |$a, $T| $body, else $other)
        }
    }};
}

fn unsupported_dtype<R>(name: &str, arr: &Bound<'_, PyUntypedArray>) -> PyResult<R> {
    Err(PyTypeError::new_err(format!(
        "{name}: unsupported dtype {}",
        arr.dtype().str()?
    )))
}

/// Local maxima along an axis.
///
/// ``x[i] > x[i - 1]`` and ``x[i] >= x[i + 1]``. The first element is never a
/// maximum; the last is when it exceeds its neighbour.
///
/// Args:
///
///     x (``array_like``): Values of any real dtype.
///     axis (int): Axis to compare along. Default 0.
///
/// Returns:
///
///     numpy.ndarray[bool]: Same shape as ``x``.
#[pyfunction]
#[pyo3(signature = (x, *, axis = Int(0)), text_signature = "(x, *, axis=0)")]
#[inline]
pub fn localmax<'py>(x: &Bound<'py, PyAny>, axis: Int) -> PyResult<Bound<'py, PyAny>> {
    local_extrema(x, axis, true)
}

/// Local minima along an axis.
///
/// ``x[i] < x[i - 1]`` and ``x[i] <= x[i + 1]``. The first element is never a
/// minimum; the last is when it is below its neighbour.
///
/// Args:
///
///     x (``array_like``): Values of any real dtype.
///     axis (int): Axis to compare along. Default 0.
///
/// Returns:
///
///     numpy.ndarray[bool]: Same shape as ``x``.
#[pyfunction]
#[pyo3(signature = (x, *, axis = Int(0)), text_signature = "(x, *, axis=0)")]
#[inline]
pub fn localmin<'py>(x: &Bound<'py, PyAny>, axis: Int) -> PyResult<Bound<'py, PyAny>> {
    local_extrema(x, axis, false)
}

fn local_extrema<'py>(x: &Bound<'py, PyAny>, axis: Int, max: bool) -> PyResult<Bound<'py, PyAny>> {
    let py = x.py();
    let arr = as_ndarray(x)?;
    let ax = norm_axis("axis", axis.0, arr.ndim())?;
    each_ordered!(&arr, |a, T| {
        // SAFETY: the array outlives the view and no Rust code writes to it.
        let v = unsafe { a.as_array() };
        let out = detached(py, v.len(), || {
            if max { ops::localmax(&v, ax) } else { ops::localmin(&v, ax) }
        })
        .map_err(audio_err_to_py)?;
        Ok(out.into_pyarray(py).into_any())
    }, else unsupported_dtype("x", &arr))
}

/// Squared magnitude.
///
/// Args:
///
///     x (``complex`` | ``array_like``): Values. Complex input gives its real type; real input
///         keeps its dtype.
///     dtype (Optional[numpy.dtype]): Output dtype.
///
/// Returns:
///
///     float | int | numpy.ndarray: ``x.real ** 2 + x.imag ** 2``.
#[pyfunction]
#[pyo3(signature = (x, dtype = None))]
#[inline]
pub fn abs2<'py>(
    x: &Bound<'py, PyAny>,
    dtype: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let py = x.py();
    let dtype = dtype.filter(|d| !d.is_none());
    if dtype.is_none() {
        if let Ok(z) = x.cast::<PyComplex>() {
            let (re, im) = (z.real(), z.imag());
            return (re * re + im * im).into_bound_py_any(py);
        }
        if x.is_instance_of::<PyFloat>() || x.is_instance_of::<PyInt>() {
            return x.mul(x);
        }
    }
    let arr = as_ndarray(x)?;
    macro_rules! complex_abs2 {
        ($a:ident, $t:ty) => {{
            // SAFETY: the array outlives the view and no Rust code writes to it.
            let v = unsafe { $a.as_array() };
            // SAFETY: every element of the new array is written below before it is read.
            let out = unsafe { PyArrayDyn::<$t>::new(py, v.raw_dim(), false) };
            // SAFETY: the array was just created, so nothing else refers to it. Viewing it
            // without a borrow avoids an entry in rust-numpy's borrow table.
            let m = unsafe { out.as_array_mut() };
            detached(py, v.len(), || ops::abs2_into(v, m)).map_err(audio_err_to_py)?;
            out.into_any()
        }};
    }
    let out: Bound<'py, PyAny> = if let Ok(a) = arr.cast::<PyArrayDyn<Complex<f64>>>() {
        complex_abs2!(a, f64)
    } else if let Ok(a) = arr.cast::<PyArrayDyn<Complex<f32>>>() {
        complex_abs2!(a, f32)
    } else if let Ok(a) = arr.cast::<PyArrayDyn<f64>>() {
        // SAFETY: as in `local_extrema`.
        let v = unsafe { a.as_array() };
        detached(py, v.len(), || v.mapv(|e| e * e))
            .into_pyarray(py)
            .into_any()
    } else if let Ok(a) = arr.cast::<PyArrayDyn<f32>>() {
        // SAFETY: as in `local_extrema`.
        let v = unsafe { a.as_array() };
        detached(py, v.len(), || v.mapv(|e| e * e))
            .into_pyarray(py)
            .into_any()
    } else {
        numpy_attr(py, "square")?.call1((arr.as_any(),))?
    };
    let out = match dtype {
        Some(d) => out.call_method1("astype", (d,))?,
        None => out,
    };
    if arr.ndim() == 0 {
        return out.get_item(PyTuple::empty(py));
    }
    Ok(out)
}

fn phasor_in<'py, F: PyFloatElement + 'static>(
    angles: &Bound<'py, PyAny>,
    mag: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>>
where
    Complex<F>: numpy::Element,
{
    let py = angles.py();
    let mut z = with_float_view::<F, _>(angles, "angles", |a| {
        Ok(detached(py, a.len(), || ops::phasor(&a)))
    })?;
    if let Some(m) = mag.filter(|m| !m.is_none()) {
        let complex_mag = m.is_instance_of::<PyComplex>()
            || m.cast::<PyUntypedArray>()
                .is_ok_and(|a| a.dtype().kind() == b'c');
        if complex_mag {
            let as_complex =
                numpy_attr(py, "asarray")?.call1((m, numpy::dtype::<Complex<F>>(py)))?;
            let a = as_complex.cast_into::<PyArrayDyn<Complex<F>>>()?;
            // SAFETY: `a` is a new reference held for the view's lifetime.
            let v = unsafe { a.as_array() };
            let v = v
                .broadcast(z.raw_dim())
                .ok_or_else(|| value_err("mag", "does not broadcast to the shape of angles"))?;
            Zip::from(&mut z).and(&v).for_each(|o, &g| *o *= g);
        } else {
            with_float_view::<F, _>(m, "mag", |v| {
                let v = v
                    .broadcast(z.raw_dim())
                    .ok_or_else(|| value_err("mag", "does not broadcast to the shape of angles"))?;
                Zip::from(&mut z).and(&v).for_each(|o, &g| *o = o.scale(g));
                Ok(())
            })?;
        }
    }
    let out = z.into_pyarray(py).into_any();
    if out.cast::<PyUntypedArray>()?.ndim() == 0 {
        return out.get_item(PyTuple::empty(py));
    }
    Ok(out)
}

/// Complex phasors ``mag * (cos(angles) + 1j * sin(angles))``.
///
/// Args:
///
///     angles (float | ``array_like``): Angles in radians.
///     mag (Optional[``complex`` | ``array_like``]): Multiplier, broadcast to ``angles``.
///
/// Returns:
///
///     ``complex`` | numpy.ndarray: ``complex64`` for ``float32`` angles, otherwise
///     ``complex128``.
#[pyfunction]
#[pyo3(signature = (angles, *, mag = None))]
#[inline]
pub fn phasor<'py>(
    angles: &Bound<'py, PyAny>,
    mag: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    with_precision!(precision_of(angles), |F| phasor_in::<F>(angles, mag))
}

fn zero_crossings_in<'py, F: PyFloatElement>(
    y: &Bound<'py, PyAny>,
    threshold: f64,
    pad: bool,
    zero_pos: bool,
    axis: i64,
) -> PyResult<Bound<'py, PyAny>> {
    let py = y.py();
    let z = with_float_view::<F, _>(y, "y", |v| {
        let axis = norm_axis("axis", axis, v.ndim())?;
        let threshold = F::cast_from(threshold);
        ops::zero_crossings(&v, threshold, pad, zero_pos, axis).map_err(audio_err_to_py)
    })?;
    Ok(z.into_pyarray(py).into_any())
}

/// Marks the sign changes of a signal along one axis.
///
/// Args:
///
///     y (``array_like``): The signal, of any dimension.
///     threshold (Optional[float]): Magnitudes up to this count as zero. None means 0.
///         Default 1e-10.
///     ``ref_magnitude`` (Optional[float | Callable]): Scales ``threshold``. A callable receives
///         ``numpy.abs(y)`` and returns the scale.
///     pad (bool): The mark of the first element. Default True.
///     ``zero_pos`` (bool): Zero counts as positive; otherwise it is a sign of its own.
///         Default True.
///     axis (int): The time axis. Default -1.
///
/// Returns:
///
///     numpy.ndarray[bool]: The marks, of the shape of ``y``.
#[pyfunction]
#[pyo3(
    signature = (y, *, threshold = Some(1e-10), ref_magnitude = None, pad = true, zero_pos = true, axis = -1),
    text_signature = "(y, *, threshold=1e-10, ref_magnitude=None, pad=True, zero_pos=True, axis=-1)"
)]
#[inline]
pub fn zero_crossings<'py>(
    y: &Bound<'py, PyAny>,
    threshold: Option<f64>,
    ref_magnitude: Option<&Bound<'py, PyAny>>,
    pad: bool,
    zero_pos: bool,
    axis: i64,
) -> PyResult<Bound<'py, PyAny>> {
    let mut threshold = threshold.unwrap_or(0.0);
    if let Some(r) = ref_magnitude.filter(|r| !r.is_none()) {
        let scale = if r.is_callable() {
            let magnitudes = numpy_attr(y.py(), "abs")?.call1((y,))?;
            r.call1((magnitudes,))?.extract::<f64>()?
        } else {
            r.extract::<f64>()?
        };
        threshold *= scale;
    }
    with_precision!(precision_of(y), |F| zero_crossings_in::<F>(
        y, threshold, pad, zero_pos, axis
    ))
}

/// The number of samples of a generator, from its length or its duration.
fn generated_length(sr: f64, length: Option<Int>, duration: Option<f64>) -> PyResult<usize> {
    match (length, duration) {
        (Some(n), _) => {
            usize::try_from(n.0).map_err(|_| value_err("length", "must not be negative"))
        }
        (None, Some(d)) if d >= 0.0 => Ok((d * sr) as usize),
        (None, Some(_)) => Err(value_err("duration", "must not be negative")),
        (None, None) => Err(value_err("length", "give length or duration")),
    }
}

/// A cosine of a given frequency and phase, as float64.
///
/// Args:
///
///     frequency (float): In Hz.
///     sr (float): Sample rate in Hz. Default 22050.
///     length (Optional[int]): Samples.
///     duration (Optional[float]): Seconds, if ``length`` is None.
///     phi (Optional[float]): Phase offset in radians. None means ``-pi / 2``, which starts
///         like a sine.
///
/// Returns:
///
///     numpy.ndarray[float64]: The samples.
#[pyfunction]
#[pyo3(
    signature = (frequency, *, sr = 22050.0, length = None, duration = None, phi = None),
    text_signature = "(frequency, *, sr=22050, length=None, duration=None, phi=None)"
)]
#[inline]
pub fn tone(
    py: Python<'_>,
    frequency: f64,
    sr: f64,
    length: Option<Int>,
    duration: Option<f64>,
    phi: Option<f64>,
) -> PyResult<Bound<'_, PyAny>> {
    let n = generated_length(sr, length, duration)?;
    let phi = phi.unwrap_or(-std::f64::consts::FRAC_PI_2);
    Ok(numpy::PyArray1::from_vec(py, ops::tone::<f64>(frequency, sr, n, phi)).into_any())
}

/// A swept cosine, as float64.
///
/// Args:
///
///     fmin (float): Starting frequency in Hz.
///     fmax (float): Frequency at the end in Hz.
///     sr (float): Sample rate in Hz. Default 22050.
///     length (Optional[int]): Samples.
///     duration (Optional[float]): Seconds, if ``length`` is None.
///     linear (bool): Sweep linearly; otherwise exponentially. Default False.
///     phi (Optional[float]): Phase offset in radians. None means ``-pi / 2``.
///
/// Returns:
///
///     numpy.ndarray[float64]: The samples.
#[pyfunction]
#[pyo3(
    signature = (*, fmin, fmax, sr = 22050.0, length = None, duration = None, linear = false, phi = None),
    text_signature = "(*, fmin, fmax, sr=22050, length=None, duration=None, linear=False, phi=None)"
)]
#[allow(clippy::too_many_arguments)]
#[inline]
pub fn chirp(
    py: Python<'_>,
    fmin: f64,
    fmax: f64,
    sr: f64,
    length: Option<Int>,
    duration: Option<f64>,
    linear: bool,
    phi: Option<f64>,
) -> PyResult<Bound<'_, PyAny>> {
    let n = generated_length(sr, length, duration)?;
    let phi = phi.unwrap_or(-std::f64::consts::FRAC_PI_2);
    let v = ops::chirp::<f64>(fmin, fmax, sr, n, linear, phi).map_err(audio_err_to_py)?;
    Ok(numpy::PyArray1::from_vec(py, v).into_any())
}

/// The smallest positive normal value of the input's floating-point type.
///
/// Args:
///
///     x (``complex`` | ``array_like`` | numpy.dtype-bearing value): Determines the type. Complex
///         types use their real type; non-float input uses ``float32``.
///
/// Returns:
///
///     numpy.floating: The value, as a ``NumPy`` scalar of that type.
#[pyfunction]
#[inline]
pub fn tiny<'py>(x: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    static F32: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
    static F64: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
    let py = x.py();
    let single = |py: Python<'py>| -> PyResult<Bound<'py, PyAny>> {
        Ok(F32
            .get_or_try_init(py, || {
                Ok::<_, PyErr>(
                    numpy_attr(py, "float32")?
                        .call1((f32::MIN_POSITIVE,))?
                        .unbind(),
                )
            })?
            .bind(py)
            .clone())
    };
    let double = |py: Python<'py>| -> PyResult<Bound<'py, PyAny>> {
        Ok(F64
            .get_or_try_init(py, || {
                Ok::<_, PyErr>(
                    numpy_attr(py, "float64")?
                        .call1((f64::MIN_POSITIVE,))?
                        .unbind(),
                )
            })?
            .bind(py)
            .clone())
    };
    if x.is_exact_instance_of::<PyFloat>() || x.is_exact_instance_of::<PyComplex>() {
        return double(py);
    }
    if x.is_instance_of::<PyInt>() {
        return single(py);
    }
    let arr = as_ndarray(x)?;
    let dt = arr.dtype();
    match (dt.kind(), dt.itemsize()) {
        (b'f', 4) | (b'c', 8) => single(py),
        (b'f', 8) | (b'c', 16) => double(py),
        (b'f' | b'c', _) => numpy_attr(py, "finfo")?.call1((dt,))?.getattr("tiny"),
        _ => single(py),
    }
}

/// Places the axes of an array at given positions of a higher-dimensional view.
///
/// Args:
///
///     x (numpy.ndarray): The array.
///     ndim (int): Dimensions of the result.
///     axes (int | sequence of int): Position of each axis of ``x``; negative counts
///         from the end. Positions out of order reorder the axes, so axis ``i`` of ``x``
///         is always axis ``axes[i]`` of the result.
///
/// Returns:
///
///     numpy.ndarray: A view of the same data.
#[pyfunction]
#[pyo3(signature = (x, *, ndim, axes))]
#[inline]
pub fn expand_to<'py>(
    x: &Bound<'py, PyAny>,
    ndim: Int,
    axes: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let py = x.py();
    let arr = as_ndarray(x)?;
    let mut list = [0_isize; ops::MAX_DIMS];
    let count = if axes.is_instance_of::<PyInt>() {
        list[0] = axes.extract()?;
        1
    } else if let Ok(items) = axes.cast::<PyList>() {
        if items.len() > list.len() {
            return Err(value_err("axes", "too many axes"));
        }
        for (slot, a) in list.iter_mut().zip(items.iter()) {
            *slot = a.extract()?;
        }
        items.len()
    } else {
        let mut n = 0;
        for a in axes.try_iter()? {
            if n == list.len() {
                return Err(value_err("axes", "too many axes"));
            }
            list[n] = a?.extract()?;
            n += 1;
        }
        n
    };
    let ndim = usize::try_from(ndim.0).map_err(|_| value_err("ndim", "must not be negative"))?;
    if ndim > ops::MAX_DIMS {
        return Err(value_err("ndim", "too many dimensions"));
    }
    let mut positions = [0_usize; ops::MAX_DIMS];
    ops::expand_positions(arr.ndim(), ndim, &list[..count], &mut positions)
        .map_err(audio_err_to_py)?;
    let positions = &positions[..count];
    let mut shape = [1_usize; ops::MAX_DIMS];
    for (i, &p) in positions.iter().enumerate() {
        shape[p] = arr.shape()[i];
    }
    if positions.windows(2).all(|w| w[0] < w[1]) {
        let mut dims = [0_isize; ops::MAX_DIMS];
        for (d, s) in dims.iter_mut().zip(&shape[..ndim]) {
            *d = *s as isize;
        }
        let mut newdims = numpy::npyffi::PyArray_Dims {
            ptr: dims.as_mut_ptr(),
            len: ndim as i32,
        };
        // SAFETY: `arr` is a valid array and `newdims` points at `ndim` dimensions for the
        // duration of the call; the result is a new reference, or NULL with an exception set.
        return unsafe {
            let view = numpy::npyffi::PY_ARRAY_API.PyArray_Newshape(
                py,
                arr.as_array_ptr(),
                &raw mut newdims,
                numpy::npyffi::NPY_ORDER::NPY_CORDER,
            );
            Bound::from_owned_ptr_or_err(py, view)
        };
    }
    let shape = PyTuple::new(py, &shape[..ndim])?;
    let mut order: Vec<usize> = (0..count).collect();
    order.sort_by_key(|&i| positions[i]);
    arr.call_method1("transpose", (PyTuple::new(py, order)?,))?
        .call_method1("reshape", (shape,))
}

/// Stacks arrays of one shape along a new axis.
///
/// Args:
///
///     arrays (sequence of ``array_like``): At least one array.
///     axis (int): Position of the new axis. Default 0.
///
/// Returns:
///
///     numpy.ndarray: The stacked array, of the arrays' common dtype.
#[pyfunction]
#[pyo3(signature = (arrays, *, axis = Int(0)), text_signature = "(arrays, *, axis=0)")]
#[inline]
pub fn stack<'py>(arrays: &Bound<'py, PyAny>, axis: Int) -> PyResult<Bound<'py, PyAny>> {
    let py = arrays.py();
    if arrays.len()? == 0 {
        return Err(value_err("arrays", "at least one array must be provided"));
    }
    let kwargs = PyDict::new(py);
    kwargs.set_item("axis", axis.0)?;
    numpy_attr(py, "stack")?.call((arrays,), Some(&kwargs))
}

fn parse_pad_mode(kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<Option<(&'static str, f64)>> {
    let Some(kw) = kwargs else {
        return Ok(Some(("constant", 0.0)));
    };
    let mut mode = "constant";
    let mut value = 0.0;
    for (k, v) in kw.iter() {
        let k = k.cast::<PyString>()?.to_cow()?;
        match k.as_ref() {
            "mode" => {
                let Ok(s) = v.cast::<PyString>() else {
                    return Ok(None);
                };
                mode = match s.to_cow()?.as_ref() {
                    "constant" => "constant",
                    "edge" => "edge",
                    "reflect" => "reflect",
                    "symmetric" => "symmetric",
                    "wrap" => "wrap",
                    _ => return Ok(None),
                };
            }
            "constant_values" => {
                let Ok(c) = v.extract::<f64>() else {
                    return Ok(None);
                };
                value = c;
            }
            _ => return Ok(None),
        }
    }
    Ok(Some((mode, value)))
}

/// Pads an array along one axis to a given length, keeping the data centred.
///
/// Args:
///
///     data (numpy.ndarray): The array.
///     size (int): Length of ``axis`` in the result.
///     axis (int): The axis to pad. Default -1.
///     **kwargs: ``mode`` and its arguments, as for ``numpy.pad``. Default: zeros.
///
/// Returns:
///
///     numpy.ndarray: The padded array, of the input's dtype.
#[pyfunction]
#[pyo3(signature = (data, *, size, axis = Int(-1), **kwargs), text_signature = "(data, *, size, axis=-1, **kwargs)")]
#[inline]
pub fn pad_center<'py>(
    data: &Bound<'py, PyAny>,
    size: Int,
    axis: Int,
    kwargs: Option<&Bound<'py, PyDict>>,
) -> PyResult<Bound<'py, PyAny>> {
    let py = data.py();
    let arr = as_ndarray(data)?;
    let ax = norm_axis("axis", axis.0, arr.ndim())?;
    let n = arr.shape()[ax.index()];
    let size_u = usize::try_from(size.0).map_err(|_| value_err("size", "must not be negative"))?;
    if size_u < n {
        return Err(value_err(
            "size",
            format!("target size {size_u} is below the input size {n}"),
        ));
    }
    let through_numpy = || -> PyResult<Bound<'py, PyAny>> {
        let lpad = (size_u - n) / 2;
        let widths: Vec<(usize, usize)> = (0..arr.ndim())
            .map(|i| {
                if i == ax.index() {
                    (lpad, size_u - n - lpad)
                } else {
                    (0, 0)
                }
            })
            .collect();
        let kw = PyDict::new(py);
        if let Some(k) = kwargs {
            kw.update(k.as_mapping())?;
        }
        if !kw.contains("mode")? {
            kw.set_item("mode", "constant")?;
        }
        numpy_attr(py, "pad")?.call((arr.as_any(), widths), Some(&kw))
    };
    let Some((mode, value)) = parse_pad_mode(kwargs)? else {
        return through_numpy();
    };
    macro_rules! pad {
        ($a:ident, $T:ident, $constant:expr) => {{
            let constant: $T = $constant;
            let m = match mode {
                "constant" => PadMode::Constant(constant),
                "edge" => PadMode::Edge,
                "reflect" => PadMode::Reflect,
                "symmetric" => PadMode::Symmetric,
                _ => PadMode::Wrap,
            };
            // SAFETY: as in `local_extrema`.
            let v = unsafe { $a.as_array() };
            let out = detached(py, v.len(), || ops::pad_center(&v, size_u, ax, m))
                .map_err(audio_err_to_py)?;
            Ok(out.into_pyarray(py).into_any())
        }};
    }
    if let Ok(a) = arr.cast::<PyArrayDyn<f64>>() {
        pad!(a, f64, value)
    } else if let Ok(a) = arr.cast::<PyArrayDyn<f32>>() {
        pad!(a, f32, value as f32)
    } else if let Ok(a) = arr.cast::<PyArrayDyn<Complex<f64>>>() {
        pad!(a, Complex64T, Complex::new(value, 0.0))
    } else if let Ok(a) = arr.cast::<PyArrayDyn<Complex<f32>>>() {
        pad!(a, Complex32T, Complex::new(value as f32, 0.0))
    } else {
        through_numpy()
    }
}

#[allow(non_camel_case_types)]
type Complex64T = Complex<f64>;
#[allow(non_camel_case_types)]
type Complex32T = Complex<f32>;

fn opt_i64(obj: Option<Int>) -> Option<i64> {
    obj.map(|v| v.0)
}

/// Normalises frame indices: clips, optionally adds the bounds, removes duplicates and
/// sorts.
///
/// Args:
///
///     frames (``array_like[int]``): Non-negative frame indices.
///     ``x_min`` (Optional[int]): Lower bound. Default 0.
///     ``x_max`` (Optional[int]): Upper bound.
///     pad (bool): Clip to the bounds and include them, rather than only filter.
///
/// Returns:
///
///     numpy.ndarray[int64]: Sorted, unique indices.
#[pyfunction]
#[pyo3(signature = (frames, *, x_min = Some(Int(0)), x_max = None, pad = true), text_signature = "(frames, *, x_min=0, x_max=None, pad=True)")]
#[inline]
pub fn fix_frames<'py>(
    frames: &Bound<'py, PyAny>,
    x_min: Option<Int>,
    x_max: Option<Int>,
    pad: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let py = frames.py();
    let list = i64_vec(frames, "frames")?;
    let out =
        ops::fix_frames(&list, opt_i64(x_min), opt_i64(x_max), pad).map_err(audio_err_to_py)?;
    Ok(PyArray1::from_vec(py, out).into_any())
}

/// Converts boundary indices to the slices between consecutive boundaries.
///
/// Args:
///
///     idx (``array_like[int]``): Boundary indices.
///     ``idx_min`` (Optional[int]): Lower bound.
///     ``idx_max`` (Optional[int]): Upper bound.
///     step (Optional[int]): Step of each slice.
///     pad (bool): Clip to the bounds and include them, rather than only filter.
///
/// Returns:
///
///     list[slice]: One slice per pair of consecutive boundaries.
#[pyfunction]
#[pyo3(signature = (idx, *, idx_min = None, idx_max = None, step = None, pad = true), text_signature = "(idx, *, idx_min=None, idx_max=None, step=None, pad=True)")]
#[inline]
pub fn index_to_slice<'py>(
    idx: &Bound<'py, PyAny>,
    idx_min: Option<Int>,
    idx_max: Option<Int>,
    step: Option<Int>,
    pad: bool,
) -> PyResult<Bound<'py, PyList>> {
    let py = idx.py();
    let fixed = crate::args::with_i64_view(idx, "idx", |v| {
        let flat = v.to_shape(v.len()).map_err(|e| value_err("idx", e))?;
        let flat = flat.as_standard_layout();
        let slice = flat
            .as_slice()
            .ok_or_else(|| value_err("idx", "expected a contiguous array"))?;
        ops::fix_frames(slice, opt_i64(idx_min), opt_i64(idx_max), pad).map_err(audio_err_to_py)
    })?;
    let out = PyList::empty(py);
    let step = step.map_or_else(
        || py.None().into_bound(py),
        |s| {
            s.0.into_bound_py_any(py)
                .unwrap_or_else(|_| py.None().into_bound(py))
        },
    );
    for w in fixed.windows(2) {
        let (start, stop) = (w[0].into_bound_py_any(py)?, w[1].into_bound_py_any(py)?);
        // SAFETY: the three arguments are valid objects for the duration of the call, and
        // the result is a new reference or NULL with an exception set.
        let slice = unsafe {
            Bound::from_owned_ptr_or_err(
                py,
                pyo3::ffi::PySlice_New(start.as_ptr(), stop.as_ptr(), step.as_ptr()),
            )?
        };
        out.append(slice)?;
    }
    Ok(out)
}

/// Sorts the rows or columns of a matrix by a key computed from them.
///
/// Args:
///
///     S (numpy.ndarray): The matrix.
///     axis (int): The axis to reorder: -1 reorders columns, 0 rows. Default -1.
///     index (bool): Also return the permutation.
///     value (Optional[Callable]): ``value(S, axis=other_axis)`` gives the sort key.
///         Default: the position of the peak (``numpy.argmax``). Equal keys keep their
///         order.
///
/// Returns:
///
///     numpy.ndarray | tuple[numpy.ndarray, numpy.ndarray[int64]]: The sorted matrix,
///     and the permutation when ``index`` is true.
#[pyfunction]
#[pyo3(name = "axis_sort", signature = (S, *, axis = Int(-1), index = false, value = None), text_signature = "(S, *, axis=-1, index=False, value=None)")]
#[allow(non_snake_case)]
#[inline]
pub fn axis_sort<'py>(
    S: &Bound<'py, PyAny>,
    axis: Int,
    index: bool,
    value: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let py = S.py();
    let arr = as_ndarray(S)?;
    if arr.ndim() != 2 {
        return Err(value_err("S", "axis_sort is only defined for 2-D arrays"));
    }
    let ax = norm_axis("axis", axis.0, 2)?;
    let finish = |sorted: Bound<'py, PyAny>, idx: Vec<usize>| -> PyResult<Bound<'py, PyAny>> {
        if index {
            let idx = PyArray1::from_vec(py, idx.into_iter().map(|i| i as i64).collect());
            (sorted, idx).into_bound_py_any(py)
        } else {
            Ok(sorted)
        }
    };
    let by_key = |key: Option<&Bound<'py, PyAny>>| -> PyResult<Bound<'py, PyAny>> {
        let default = numpy_attr(py, "argmax")?;
        let f = key.unwrap_or(&default);
        let kw = PyDict::new(py);
        kw.set_item("axis", 1 - ax.index())?;
        let keys = f.call((arr.as_any(),), Some(&kw))?;
        let kw = PyDict::new(py);
        kw.set_item("kind", "stable")?;
        let idx = numpy_attr(py, "argsort")?.call((keys,), Some(&kw))?;
        let sorted = arr.call_method1("take", (&idx, ax.index()))?;
        if index {
            (sorted, idx).into_bound_py_any(py)
        } else {
            Ok(sorted)
        }
    };
    if let Some(f) = value.filter(|v| !v.is_none()) {
        return by_key(Some(f));
    }
    each_ordered!(&arr, |a, T| {
        // SAFETY: as in `local_extrema`.
        let v = unsafe { a.as_array() };
        let v = v.into_dimensionality::<Ix2>().map_err(|e| value_err("S", e))?;
        let (sorted, idx) = detached(py, v.len(), || ops::axis_sort(v, ax, AxisSortKey::Argmax))
            .map_err(audio_err_to_py)?;
        finish(sorted.into_pyarray(py).into_any(), idx)
    }, else by_key(None))
}

/// Circularly shifts each column (or row) of a matrix by ``factor`` times its index.
///
/// Args:
///
///     X (numpy.ndarray | scipy.sparse array or matrix | SparseRows): The matrix.
///     factor (int): Shift per index. Default 1.
///     axis (int): -1 shifts columns by their column index, 0 shifts rows by their row
///         index. Default -1.
///
/// Returns:
///
///     The sheared matrix, in the input's form and dtype.
#[pyfunction]
#[pyo3(name = "shear", signature = (X, *, factor = Int(1), axis = Int(-1)), text_signature = "(X, *, factor=1, axis=-1)")]
#[allow(non_snake_case)]
#[inline]
pub fn shear<'py>(X: &Bound<'py, PyAny>, factor: Int, axis: Int) -> PyResult<Bound<'py, PyAny>> {
    let py = X.py();
    if let Ok(rows) = X.cast::<SparseRows>() {
        return shear(&rows.call_method0("tocsr")?, factor, axis);
    }
    if X.hasattr("tocsc")? {
        let ax = norm_axis("axis", axis.0, 2)?;
        let source = if ax.index() == 0 {
            X.getattr("T")?
        } else {
            X.clone()
        };
        let csc = source.call_method0("tocsc")?;
        let (rows, cols): (usize, usize) = csc.getattr("shape")?.extract()?;
        let indptr = i64_vec(&csc.getattr("indptr")?, "X.indptr")?;
        let old = i64_vec(&csc.getattr("indices")?, "X.indices")?;
        let mut indices = vec![0_i32; old.len()];
        for c in 0..cols {
            let roll = factor.0.wrapping_mul(c as i64);
            for k in indptr[c] as usize..indptr[c + 1] as usize {
                indices[k] = old[k].wrapping_add(roll).rem_euclid(rows as i64) as i32;
            }
        }
        let parts = (
            csc.getattr("data")?,
            PyArray1::from_vec(py, indices),
            csc.getattr("indptr")?,
        );
        let kw = PyDict::new(py);
        kw.set_item("shape", (rows, cols))?;
        let out = csc.get_type().call((parts,), Some(&kw))?;
        let out = if ax.index() == 0 {
            out.getattr("T")?
        } else {
            out
        };
        return out.call_method1("asformat", (X.getattr("format")?,));
    }
    let arr = as_ndarray(X)?;
    if arr.ndim() != 2 {
        return Err(value_err("X", "shear is only defined for 2-D arrays"));
    }
    let ax = norm_axis("axis", axis.0, 2)?;
    each_numeric!(&arr, |a, T| {
        // SAFETY: as in `local_extrema`.
        let v = unsafe { a.as_array() };
        let v = v.into_dimensionality::<Ix2>().map_err(|e| value_err("X", e))?;
        let out = detached(py, v.len(), || ops::shear(v, factor.0, ax)).map_err(audio_err_to_py)?;
        Ok(out.into_pyarray(py).into_any())
    }, else unsupported_dtype("X", &arr))
}

/// A matrix in compressed sparse row form.
///
/// ``data``, ``indices`` and ``indptr`` are ``NumPy`` arrays with the usual meaning.
/// ``toarray()`` gives the dense form without needing ``SciPy``; ``tocsr()`` gives a
/// ``scipy.sparse.csr_array`` and needs ``SciPy``.
#[pyclass(module = "audio_samples.arrays", frozen)]
pub struct SparseRows {
    /// Stored values.
    #[pyo3(get)]
    data: Py<PyAny>,
    /// Column of each stored value.
    #[pyo3(get)]
    indices: Py<PyAny>,
    /// Where each row starts in ``data``, with one extra entry at the end.
    #[pyo3(get)]
    indptr: Py<PyAny>,
    /// ``(rows, columns)``.
    #[pyo3(get)]
    shape: (usize, usize),
}

#[pymethods]
impl SparseRows {
    /// Number of stored values.
    #[getter]
    fn nnz(&self, py: Python<'_>) -> PyResult<usize> {
        self.data.bind(py).len()
    }

    /// The dense matrix.
    fn toarray<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let data = self.data.bind(py);
        let out = numpy_attr(py, "zeros")?.call1((self.shape, data.getattr("dtype")?))?;
        let indptr = i64_vec(self.indptr.bind(py), "indptr")?;
        let rows: Vec<i64> = (0..self.shape.0)
            .flat_map(|r| std::iter::repeat_n(r as i64, (indptr[r + 1] - indptr[r]) as usize))
            .collect();
        let key = (PyArray1::from_vec(py, rows), self.indices.bind(py));
        out.set_item(key, data)?;
        Ok(out)
    }

    /// The matrix as a ``scipy.sparse.csr_array``.
    fn tocsr<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let kw = PyDict::new(py);
        kw.set_item("shape", self.shape)?;
        py.import("scipy.sparse")?.getattr("csr_array")?.call(
            ((
                self.data.bind(py),
                self.indices.bind(py),
                self.indptr.bind(py),
            ),),
            Some(&kw),
        )
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        Ok(format!(
            "SparseRows(shape={:?}, nnz={}, dtype={})",
            self.shape,
            self.nnz(py)?,
            self.data.bind(py).getattr("dtype")?
        ))
    }
}

/// The non-zero entries of a dense matrix as [`SparseRows`].
pub fn dense_to_rows<'py>(dense: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    fn compress<'py, T: numpy::Element + Copy + PartialEq + Default>(
        py: Python<'py>,
        dense: &Bound<'py, PyArray2<T>>,
    ) -> PyResult<(
        Bound<'py, PyAny>,
        Bound<'py, PyAny>,
        Bound<'py, PyAny>,
        (usize, usize),
    )> {
        let view = dense.try_readonly()?;
        let m = view.as_array();
        let zero = T::default();
        let stored = m.iter().filter(|&&v| v != zero).count();
        let mut data = Vec::with_capacity(stored);
        let mut indices = Vec::with_capacity(stored);
        let mut indptr = Vec::with_capacity(m.nrows() + 1);
        indptr.push(0_i32);
        for row in m.rows() {
            for (column, &v) in row.iter().enumerate() {
                if v != zero {
                    data.push(v);
                    indices.push(column as i32);
                }
            }
            indptr.push(indices.len() as i32);
        }
        Ok((
            PyArray1::from_vec(py, data).into_any(),
            PyArray1::from_vec(py, indices).into_any(),
            PyArray1::from_vec(py, indptr).into_any(),
            m.dim(),
        ))
    }
    let py = dense.py();
    let (data, indices, indptr, shape) = if let Ok(a) = dense.cast::<PyArray2<bool>>() {
        compress(py, a)?
    } else if let Ok(a) = dense.cast::<PyArray2<f32>>() {
        compress(py, a)?
    } else {
        compress(py, dense.cast::<PyArray2<f64>>()?)?
    };
    Ok(Py::new(
        py,
        SparseRows {
            data: data.unbind(),
            indices: indices.unbind(),
            indptr: indptr.unbind(),
            shape,
        },
    )?
    .into_bound(py)
    .into_any())
}

/// Removes the smallest values of each row, keeping a sparse matrix.
///
/// In each row the sorted magnitudes are divided by the row's total and summed
/// cumulatively; entries below the first magnitude whose running total reaches
/// ``quantile`` are dropped.
///
/// Args:
///
///     x (``array_like``): 1-D or 2-D values; 1-D is one row.
///     quantile (float): Fraction of each row's total magnitude that may be discarded,
///         in ``[0, 1)``. Default 0.01.
///     dtype (Optional[numpy.dtype]): dtype of the stored values. Default: the input's.
///
/// Returns:
///
///     ``SparseRows``: The kept entries.
#[pyfunction]
#[pyo3(signature = (x, *, quantile = 0.01, dtype = None))]
#[inline]
pub fn sparsify_rows<'py>(
    x: &Bound<'py, PyAny>,
    quantile: f64,
    dtype: Option<&Bound<'py, PyAny>>,
) -> PyResult<Py<SparseRows>> {
    let py = x.py();
    let arr = as_ndarray(x)?;
    let arr = match arr.ndim() {
        1 => arr
            .call_method1("reshape", ((1_i64, -1_i64),))?
            .cast_into::<PyUntypedArray>()?,
        2 => arr,
        n => {
            return Err(value_err(
                "x",
                format!("expected at most 2 dimensions, got {n}"),
            ));
        }
    };
    macro_rules! sparse {
        ($a:expr) => {{
            let a = $a;
            // SAFETY: as in `local_extrema`.
            let v = unsafe { a.as_array() };
            let v = v
                .into_dimensionality::<Ix2>()
                .map_err(|e| value_err("x", e))?;
            let m = detached(py, v.len(), || ops::sparsify_rows(v, quantile))
                .map_err(audio_err_to_py)?;
            (
                m.shape,
                PyArray1::from_vec(py, m.data).into_any(),
                PyArray1::from_vec(
                    py,
                    m.indices
                        .into_iter()
                        .map(|i| i as i32)
                        .collect::<Vec<i32>>(),
                ),
                PyArray1::from_vec(
                    py,
                    m.indptr.into_iter().map(|i| i as i32).collect::<Vec<i32>>(),
                ),
            )
        }};
    }
    let (shape, data, indices, indptr) = if let Ok(a) = arr.cast::<PyArrayDyn<f64>>() {
        sparse!(a)
    } else if let Ok(a) = arr.cast::<PyArrayDyn<f32>>() {
        sparse!(a)
    } else if let Ok(a) = arr.cast::<PyArrayDyn<Complex<f64>>>() {
        sparse!(a)
    } else if let Ok(a) = arr.cast::<PyArrayDyn<Complex<f32>>>() {
        sparse!(a)
    } else {
        let as64 = arr.call_method1("astype", ("float64",))?;
        let as64 = as64.cast_into::<PyArrayDyn<f64>>()?;
        let (shape, data, indices, indptr) = sparse!(&as64);
        (
            shape,
            data.call_method1("astype", (arr.dtype(),))?,
            indices,
            indptr,
        )
    };
    let data = match dtype.filter(|d| !d.is_none()) {
        Some(d) => data.call_method1("astype", (d,))?,
        None => data,
    };
    Py::new(
        py,
        SparseRows {
            data: data.unbind(),
            indices: indices.into_any().unbind(),
            indptr: indptr.into_any().unbind(),
            shape,
        },
    )
}

/// Central-difference gradient of periodic data along one axis.
///
/// Args:
///
///     data (``array_like``): Values. ``float32`` and ``complex64`` keep their precision;
///         anything else is computed in ``float64``.
///     ``edge_order`` (int): 1 or 2. Accepted for both orders: wrapped central differences
///         have no edges, so the result does not depend on it.
///     axis (int): The periodic axis. Default -1.
///
/// Returns:
///
///     numpy.ndarray: Same shape as ``data``.
#[pyfunction]
#[pyo3(signature = (data, *, edge_order = Int(1), axis = Int(-1)), text_signature = "(data, *, edge_order=1, axis=-1)")]
#[inline]
pub fn cyclic_gradient<'py>(
    data: &Bound<'py, PyAny>,
    edge_order: Int,
    axis: Int,
) -> PyResult<Bound<'py, PyAny>> {
    let py = data.py();
    if !matches!(edge_order.0, 1 | 2) {
        return Err(value_err("edge_order", "must be 1 or 2"));
    }
    macro_rules! complex {
        ($t:ty) => {
            if let Ok(a) = data.cast::<PyArrayDyn<Complex<$t>>>() {
                // SAFETY: as in `local_extrema`.
                let v = unsafe { a.as_array() };
                let ax = norm_axis("axis", axis.0, v.ndim())?;
                let re = ops::cyclic_gradient(&v.mapv(|z| z.re), ax).map_err(audio_err_to_py)?;
                let im = ops::cyclic_gradient(&v.mapv(|z| z.im), ax).map_err(audio_err_to_py)?;
                let mut out = v.to_owned();
                Zip::from(&mut out)
                    .and(&re)
                    .and(&im)
                    .for_each(|o, &r, &i| *o = Complex::new(r, i));
                return Ok(out.into_pyarray(py).into_any());
            }
        };
    }
    complex!(f64);
    complex!(f32);
    fn run<'py, F: PyFloatElement>(
        data: &Bound<'py, PyAny>,
        axis: Int,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = data.py();
        with_float_view::<F, _>(data, "data", |v| {
            let ax = norm_axis("axis", axis.0, v.ndim())?;
            let out =
                detached(py, v.len(), || ops::cyclic_gradient(&v, ax)).map_err(audio_err_to_py)?;
            Ok(out.into_pyarray(py).into_any())
        })
    }
    with_precision!(precision_of(data), |F| run::<F>(data, axis))
}

/// Sets the entries of a matrix outside a diagonal band to a value, in place.
///
/// Args:
///
///     x (numpy.ndarray): The 2-D matrix, changed in place.
///     radius (float): Half-width of the band as a fraction of the shorter side.
///     value (number): The value written outside the band. Default 0.
#[pyfunction]
#[pyo3(signature = (x, *, radius, value = None))]
#[inline]
pub fn fill_off_diagonal<'py>(
    x: &Bound<'py, PyAny>,
    radius: f64,
    value: Option<&Bound<'py, PyAny>>,
) -> PyResult<()> {
    let py = x.py();
    let arr = x
        .cast::<PyUntypedArray>()
        .map_err(|_| PyTypeError::new_err("x: expected a numpy.ndarray"))?;
    if arr.ndim() != 2 {
        return Err(value_err("x", "expected a 2-D array"));
    }
    let zero = 0_i64.into_bound_py_any(py)?;
    let value = value.filter(|v| !v.is_none()).unwrap_or(&zero);
    let scalar = numpy_attr(py, "asarray")?.call1((value, arr.dtype()))?;
    each_numeric!(arr, |a, T| {
        let s = scalar.cast::<PyArrayDyn<T>>()?;
        // SAFETY: `s` is a 0-d array held for the read.
        let fill = unsafe { s.as_array() }.first().copied().ok_or_else(|| value_err("value", "expected a scalar"))?;
        let mut w = a.try_readwrite()?;
        let m = w.as_array_mut().into_dimensionality::<Ix2>().map_err(|e| value_err("x", e))?;
        ops::fill_off_diagonal(m, radius, fill).map_err(audio_err_to_py)
    }, else unsupported_dtype("x", arr))
}

/// Counts the distinct values in each lane along an axis.
///
/// Args:
///
///     data (``array_like``): Values of any numeric dtype. All NaNs count as one value.
///     axis (int): The axis of the lanes. Default -1.
///
/// Returns:
///
///     numpy.ndarray[int64]: The input's shape without ``axis``.
#[pyfunction]
#[pyo3(signature = (data, *, axis = Int(-1)), text_signature = "(data, *, axis=-1)")]
#[inline]
pub fn count_unique<'py>(data: &Bound<'py, PyAny>, axis: Int) -> PyResult<Bound<'py, PyAny>> {
    let py = data.py();
    let arr = as_ndarray(data)?;
    let ax = norm_axis("axis", axis.0, arr.ndim())?;
    each_numeric!(&arr, |a, T| {
        // SAFETY: as in `local_extrema`.
        let v = unsafe { a.as_array() };
        let out = detached(py, v.len(), || ops::count_unique(&v, ax)).map_err(audio_err_to_py)?;
        Ok(out.into_pyarray(py).into_any())
    }, else unsupported_dtype("data", &arr))
}

/// Tells for each lane along an axis whether all its values are distinct.
///
/// Args:
///
///     data (``array_like``): Values of any numeric dtype. All NaNs count as one value.
///     axis (int): The axis of the lanes. Default -1.
///
/// Returns:
///
///     numpy.ndarray[bool]: The input's shape without ``axis``.
#[pyfunction]
#[pyo3(signature = (data, *, axis = Int(-1)), text_signature = "(data, *, axis=-1)")]
#[inline]
pub fn is_unique<'py>(data: &Bound<'py, PyAny>, axis: Int) -> PyResult<Bound<'py, PyAny>> {
    let py = data.py();
    let arr = as_ndarray(data)?;
    let ax = norm_axis("axis", axis.0, arr.ndim())?;
    each_numeric!(&arr, |a, T| {
        // SAFETY: as in `local_extrema`.
        let v = unsafe { a.as_array() };
        let out = detached(py, v.len(), || ops::is_unique(&v, ax)).map_err(audio_err_to_py)?;
        Ok(out.into_pyarray(py).into_any())
    }, else unsupported_dtype("data", &arr))
}

pub fn parse_kind(kind: &Bound<'_, PyAny>) -> PyResult<InterpKind> {
    if let Ok(k) = kind.extract::<i64>() {
        return match k {
            0 => Ok(InterpKind::Zero),
            1 => Ok(InterpKind::SLinear),
            2 => Ok(InterpKind::Quadratic),
            3 => Ok(InterpKind::Cubic),
            _ => Err(value_err(
                "kind",
                format!("spline order {k} is not 0, 1, 2 or 3"),
            )),
        };
    }
    let s: String = kind.extract()?;
    Ok(match s.as_str() {
        "linear" => InterpKind::Linear,
        "nearest" => InterpKind::Nearest,
        "nearest-up" => InterpKind::NearestUp,
        "zero" => InterpKind::Zero,
        "slinear" => InterpKind::SLinear,
        "quadratic" => InterpKind::Quadratic,
        "cubic" => InterpKind::Cubic,
        "previous" => InterpKind::Previous,
        "next" => InterpKind::Next,
        other => {
            return Err(value_err(
                "kind",
                format!("unknown interpolation kind {other:?}"),
            ));
        }
    })
}

/// Interpolates sampled data along one axis.
///
/// Args:
///
///     x (``array_like``): Sample positions, 1-D, in any order.
///     y (``array_like``): Samples; ``axis`` has one element per position.
///     xi (``array_like``): Query positions, of any shape.
///     kind (str | int): ``"linear"``, ``"nearest"``, ``"nearest-up"``, ``"zero"``,
///         ``"slinear"``, ``"quadratic"``, ``"cubic"``, ``"previous"``, ``"next"``, or a
///         spline order 0 to 3. Default ``"linear"``.
///     ``fill_value`` (float): Result outside the range of ``x``. Default 0.
///     axis (int): The axis of ``y`` that follows ``x``. Default -1.
///
/// Returns:
///
///     numpy.ndarray: ``y``'s shape with ``axis`` replaced by the axes of ``xi``, of
///     ``float32`` for ``float32`` ``y``, otherwise ``float64``.
#[pyfunction]
#[pyo3(signature = (x, y, xi, *, kind = None, fill_value = 0.0, axis = Int(-1)), text_signature = "(x, y, xi, *, kind=None, fill_value=0.0, axis=-1)")]
#[inline]
pub fn interp1d<'py>(
    x: &Bound<'py, PyAny>,
    y: &Bound<'py, PyAny>,
    xi: &Bound<'py, PyAny>,
    kind: Option<&Bound<'py, PyAny>>,
    fill_value: f64,
    axis: Int,
) -> PyResult<Bound<'py, PyAny>> {
    let py = x.py();
    let kind = kind
        .filter(|k| !k.is_none())
        .map_or(Ok(InterpKind::Linear), parse_kind)?;
    fn run<'py, F: PyFloatElement>(
        x: &Bound<'py, PyAny>,
        y: &Bound<'py, PyAny>,
        xi: &Bound<'py, PyAny>,
        kind: InterpKind,
        fill: F,
        axis: Int,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = x.py();
        with_float_view::<F, _>(y, "y", |y| {
            let ax = norm_axis("axis", axis.0, y.ndim())?;
            with_float_view::<f64, _>(x, "x", |pos| {
                with_float_view::<f64, _>(xi, "xi", |q| {
                    let pos_cow = pos.as_standard_layout();
                    let slice = pos_cow
                        .as_slice()
                        .ok_or_else(|| value_err("x", "expected 1-D"))?;
                    let out = detached(py, y.len(), || {
                        ops::interp1d(slice, y.view(), q.view(), kind, fill, ax)
                    })
                    .map_err(audio_err_to_py)?;
                    Ok(out.into_pyarray(py).into_any())
                })
            })
        })
    }
    let _ = py;
    with_precision!(precision_of(y), |F| run::<F>(
        x,
        y,
        xi,
        kind,
        F::from_f64(fill_value),
        axis
    ))
}

/// Interpolates two arrays sampled on different grids onto one grid and combines them.
///
/// Args:
///
///     x1 (``array_like``): The first array.
///     ``x1_pos`` (``array_like``): Its sample positions along ``axis``.
///     x2 (``array_like``): The second array.
///     ``x2_pos`` (``array_like``): Its sample positions along ``axis``.
///     ``interp_pos`` (Optional[array_like]): The common positions. Default ``x1_pos``.
///     op (Callable | str | None): Combines the two interpolated arrays. ``"multiply"``
///         (the default) multiplies with broadcasting; ``None`` returns both arrays.
///     kind (str | int): As for :func:`interp1d`. Default ``"linear"``.
///     ``fill_value`` (float): Result outside the range of each grid. Default 0.
///     axis (int): The interpolated axis. Default -2.
///
/// Returns:
///
///     numpy.ndarray | tuple[numpy.ndarray, numpy.ndarray]: The combined array, or both
///     arrays when ``op`` is ``None``.
#[pyfunction]
#[pyo3(signature = (*, x1, x1_pos, x2, x2_pos, interp_pos = None, op = "multiply".to_string().into(), kind = None, fill_value = 0.0, axis = Int(-2)), text_signature = "(*, x1, x1_pos, x2, x2_pos, interp_pos=None, op='multiply', kind=None, fill_value=0.0, axis=-2)")]
#[inline]
pub fn interp_broadcast<'py>(
    x1: &Bound<'py, PyAny>,
    x1_pos: &Bound<'py, PyAny>,
    x2: &Bound<'py, PyAny>,
    x2_pos: &Bound<'py, PyAny>,
    interp_pos: Option<&Bound<'py, PyAny>>,
    op: OpArg<'py>,
    kind: Option<&Bound<'py, PyAny>>,
    fill_value: f64,
    axis: Int,
) -> PyResult<Bound<'py, PyAny>> {
    let py = x1.py();
    let kind = kind
        .filter(|k| !k.is_none())
        .map_or(Ok(InterpKind::Linear), parse_kind)?;
    let single = precision_of(x1) == Precision::F32 && precision_of(x2) == Precision::F32;
    fn pos_vec(obj: &Bound<'_, PyAny>, name: &str) -> PyResult<Vec<f64>> {
        with_float_view::<f64, _>(obj, name, |v| Ok(v.iter().copied().collect()))
    }
    let p1 = pos_vec(x1_pos, "x1_pos")?;
    let p2 = pos_vec(x2_pos, "x2_pos")?;
    let q = interp_pos.filter(|p| !p.is_none());
    fn run<'py, F: PyFloatElement>(
        x1: &Bound<'py, PyAny>,
        p1: &[f64],
        x2: &Bound<'py, PyAny>,
        p2: &[f64],
        q: Option<&Bound<'py, PyAny>>,
        kind: InterpKind,
        fill: F,
        axis: isize,
        op: OpArg<'py>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = x1.py();
        let (mut y1, y2): (ArrayD<F>, ArrayD<F>) = with_float_view::<F, _>(x1, "x1", |a| {
            with_float_view::<F, _>(x2, "x2", |b| {
                let interp = |xi: Option<ArrayViewD<'_, f64>>| {
                    ops::interp_broadcast(a.view(), p1, b.view(), p2, xi, kind, fill, axis)
                        .map_err(audio_err_to_py)
                };
                match q {
                    Some(q) => {
                        with_float_view::<f64, _>(q, "interp_pos", |v| interp(Some(v.view())))
                    }
                    None => interp(None),
                }
            })
        })?;
        match op.kind(py)? {
            OpKind::Multiply if y1.shape() == y2.shape() => {
                Zip::from(&mut y1).and(&y2).for_each(|a, &b| *a *= b);
                Ok(y1.into_pyarray(py).into_any())
            }
            OpKind::Multiply => {
                numpy_attr(py, "multiply")?.call1((y1.into_pyarray(py), y2.into_pyarray(py)))
            }
            OpKind::Tuple => (y1.into_pyarray(py), y2.into_pyarray(py)).into_bound_py_any(py),
            OpKind::Call(f) => f.call1((y1.into_pyarray(py), y2.into_pyarray(py))),
        }
    }
    let _ = py;
    if single {
        run::<f32>(
            x1,
            &p1,
            x2,
            &p2,
            q,
            kind,
            fill_value as f32,
            axis.0 as isize,
            op,
        )
    } else {
        run::<f64>(x1, &p1, x2, &p2, q, kind, fill_value, axis.0 as isize, op)
    }
}

/// The ``op`` argument of [`interp_broadcast`].
pub struct OpArg<'py>(Option<Bound<'py, PyAny>>);

impl From<String> for OpArg<'_> {
    #[inline]
    fn from(_: String) -> Self {
        Self(None)
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for OpArg<'py> {
    type Error = PyErr;
    #[inline]
    fn extract(obj: pyo3::Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        Ok(Self(Some(obj.to_owned())))
    }
}

enum OpKind<'py> {
    Multiply,
    Tuple,
    Call(Bound<'py, PyAny>),
}

impl<'py> OpArg<'py> {
    fn kind(&self, py: Python<'py>) -> PyResult<OpKind<'py>> {
        let Some(op) = &self.0 else {
            return Ok(OpKind::Multiply);
        };
        if op.is_none() {
            return Ok(OpKind::Tuple);
        }
        if let Ok(s) = op.cast::<PyString>() {
            if s.to_cow()? == "multiply" {
                return Ok(OpKind::Multiply);
            }
            return Err(value_err("op", "expected a callable, 'multiply' or None"));
        }
        if op.is(&numpy_attr(py, "multiply")?) {
            return Ok(OpKind::Multiply);
        }
        Ok(OpKind::Call(op.clone()))
    }
}

fn numpy_dtype<'py>(d: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyArrayDescr>> {
    static DTYPE: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
    let py = d.py();
    Ok(DTYPE
        .get_or_try_init(py, || Ok::<_, PyErr>(numpy_attr(py, "dtype")?.unbind()))?
        .bind(py)
        .call1((d,))?
        .cast_into::<PyArrayDescr>()?)
}

fn default_dtype<'py>(
    py: Python<'py>,
    default: Option<&Bound<'py, PyAny>>,
    fallback: &str,
) -> PyResult<Bound<'py, PyArrayDescr>> {
    match default {
        Some(d) => numpy_dtype(d),
        None => numpy_dtype(PyString::new(py, fallback).as_any()),
    }
}

/// The real dtype of a complex dtype.
///
/// Args:
///
///     d (numpy.dtype): A dtype, or anything ``numpy.dtype`` accepts. A real floating
///         dtype is returned as it is.
///     default (numpy.dtype): Returned for any other dtype. Default ``float32``.
///
/// Returns:
///
///     numpy.dtype: The real dtype.
#[pyfunction]
#[pyo3(signature = (d, *, default = None))]
#[inline]
pub fn dtype_c2r<'py>(
    d: &Bound<'py, PyAny>,
    default: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyArrayDescr>> {
    let py = d.py();
    let dt = numpy_dtype(d)?;
    match (dt.kind(), dt.itemsize()) {
        (b'f', _) => Ok(dt),
        (b'c', 8) => Ok(numpy::dtype::<f32>(py)),
        (b'c', 16) => Ok(numpy::dtype::<f64>(py)),
        _ => default_dtype(py, default, "float32"),
    }
}

/// The complex dtype of a real dtype.
///
/// Args:
///
///     d (numpy.dtype): A dtype, or anything ``numpy.dtype`` accepts. A complex dtype is
///         returned as it is.
///     default (numpy.dtype): Returned for any other dtype. Default ``complex64``.
///
/// Returns:
///
///     numpy.dtype: The complex dtype.
#[pyfunction]
#[pyo3(signature = (d, *, default = None))]
#[inline]
pub fn dtype_r2c<'py>(
    d: &Bound<'py, PyAny>,
    default: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyArrayDescr>> {
    let py = d.py();
    let dt = numpy_dtype(d)?;
    match (dt.kind(), dt.itemsize()) {
        (b'c', _) => Ok(dt),
        (b'f', 4) => Ok(numpy::dtype::<Complex<f32>>(py)),
        (b'f', 8) => Ok(numpy::dtype::<Complex<f64>>(py)),
        _ => default_dtype(py, default, "complex64"),
    }
}

/// Decodes little-endian signed integer PCM to floats in ``[-1, 1)``.
///
/// Args:
///
///     x (bytes-like | numpy.ndarray): The buffer.
///     ``n_bytes`` (int): Bytes per sample: 1, 2, 3, 4 or 8. Default 2.
///     dtype (Optional[numpy.dtype]): Output dtype. Default ``float32`` up to 3 bytes per
///         sample, otherwise ``float64``, so that the conversion loses no bits it need
///         not.
///
/// Returns:
///
///     numpy.ndarray: One float per sample.
#[pyfunction]
#[pyo3(signature = (x, *, n_bytes = Int(2), dtype = None), text_signature = "(x, *, n_bytes=2, dtype=None)")]
#[inline]
pub fn buf_to_float<'py>(
    x: &Bound<'py, PyAny>,
    n_bytes: Int,
    dtype: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let py = x.py();
    let n = usize::try_from(n_bytes.0).map_err(|_| value_err("n_bytes", "must not be negative"))?;
    let dtype = dtype
        .filter(|d| !d.is_none())
        .map(numpy_dtype)
        .transpose()?;
    let double = match &dtype {
        Some(d) => d.is_equiv_to(&numpy::dtype::<f64>(py)),
        None => n > 3,
    };
    let single = dtype
        .as_ref()
        .is_none_or(|d| d.is_equiv_to(&numpy::dtype::<f32>(py)));
    if !double && !single {
        return Err(value_err("dtype", "expected float32, float64 or None"));
    }
    fn decode<'py, F: PyFloatElement>(
        py: Python<'py>,
        bytes: &[u8],
        n: usize,
    ) -> PyResult<Bound<'py, PyAny>> {
        if !matches!(n, 1 | 2 | 3 | 4 | 8) {
            return Err(value_err(
                "n_bytes",
                format!("must be 1, 2, 3, 4 or 8, got {n}"),
            ));
        }
        // SAFETY: `buf_to_float_into` writes every element, or the error drops the array.
        let out = unsafe { PyArray1::<F>::new(py, bytes.len() / n, false) };
        // SAFETY: the array was just created and is contiguous, so nothing else refers to it.
        let slice = unsafe { out.as_slice_mut() }.map_err(|e| value_err("x", e))?;
        detached(py, bytes.len(), || ops::buf_to_float_into(bytes, n, slice))
            .map_err(audio_err_to_py)?;
        Ok(out.into_any())
    }
    let run = |bytes: &[u8]| {
        if double {
            decode::<f64>(py, bytes, n)
        } else {
            decode::<f32>(py, bytes, n)
        }
    };
    if let Ok(b) = x.cast::<PyBytes>() {
        return run(b.as_bytes());
    }
    let view = py
        .get_type::<pyo3::types::PyMemoryView>()
        .call1((x,))?
        .call_method1("cast", ("B",))?;
    let buffer = PyBuffer::<u8>::get(&view)?;
    if !buffer.is_c_contiguous() {
        return Err(value_err("x", "expected a contiguous buffer"));
    }
    // SAFETY: the buffer is C-contiguous bytes, held for the call, and the GIL or the
    // buffer export keeps it alive while it is read.
    let bytes = unsafe {
        std::slice::from_raw_parts(
            buffer.buf_ptr().cast::<u8>().cast_const(),
            buffer.len_bytes(),
        )
    };
    run(bytes)
}

/// Lists audio files under a directory.
///
/// Args:
///
///     directory (str | os.PathLike): Where to look; ``~`` expands to the home directory.
///     ext (Optional[str | list[str]]): Extensions without the dot. Default: ``aac``,
///         ``au``, ``flac``, ``m4a``, ``mp3``, ``ogg``, ``wav``.
///     recurse (bool): Search subdirectories. Default true.
///     ``case_sensitive`` (bool): Compare extensions exactly; otherwise any letter case
///         matches. Default false.
///     limit (Optional[int]): Keep at most this many paths after ``offset``; a negative
///         value drops that many from the end.
///     offset (int): Skip this many paths; a negative value counts from the end.
///
/// Returns:
///
///     list[str]: Sorted absolute paths.
#[pyfunction]
#[pyo3(signature = (directory, *, ext = None, recurse = true, case_sensitive = false, limit = None, offset = Int(0)), text_signature = "(directory, *, ext=None, recurse=True, case_sensitive=False, limit=None, offset=0)")]
#[inline]
pub fn find_files<'py>(
    directory: std::path::PathBuf,
    ext: Option<&Bound<'py, PyAny>>,
    recurse: bool,
    case_sensitive: bool,
    limit: Option<Int>,
    offset: Int,
    py: Python<'py>,
) -> PyResult<Bound<'py, PyList>> {
    let mut cfg = FindFilesConfig::new();
    if let Some(e) = ext.filter(|e| !e.is_none()) {
        cfg.extensions = if let Ok(s) = e.extract::<String>() {
            vec![s]
        } else {
            e.extract()?
        };
    }
    cfg.recurse = recurse;
    cfg.case_sensitive = case_sensitive;
    // Negative bounds need the total, so they read the listing twice as little as they can.
    let negative = limit.is_some_and(|l| l.0 < 0) || offset.0 < 0;
    let out = PyList::empty(py);
    let mut failure = None;
    if negative {
        let files = py
            .detach(|| ops::find_files(&directory, &cfg))
            .map_err(audio_err_to_py)?;
        let n = files.len() as i64;
        let start = if offset.0 < 0 {
            (n + offset.0).max(0)
        } else {
            offset.0.min(n)
        };
        let remaining = n - start;
        let keep = match limit {
            None => remaining,
            Some(l) if l.0 < 0 => (remaining + l.0).max(0),
            Some(l) => l.0.min(remaining),
        };
        for f in files.into_iter().skip(start as usize).take(keep as usize) {
            out.append(f.into_os_string())?;
        }
        return Ok(out);
    }
    cfg.offset = offset.0 as usize;
    cfg.limit = limit.map(|l| l.0 as usize);
    ops::find_files_each(&directory, &cfg, |p| {
        if failure.is_none() {
            if let Err(e) = out.append(p.as_os_str()) {
                failure = Some(e);
            }
        }
    })
    .map_err(audio_err_to_py)?;
    match failure {
        Some(e) => Err(e),
        None => Ok(out),
    }
}

/// Checks that audio samples are a floating-point array, at least 1-D, finite everywhere.
///
/// Args:
///
///     y (numpy.ndarray): The samples.
///
/// Returns:
///
///     bool: ``True``.
///
/// Raises:
///
///     ``TypeError``: ``y`` is not a floating-point ``numpy.ndarray``.
///     ``ValueError``: ``y`` is 0-D or holds a NaN or infinity.
#[pyfunction]
#[inline]
pub fn valid_audio(y: &Bound<'_, PyAny>) -> PyResult<bool> {
    let py = y.py();
    let arr = y
        .cast::<PyUntypedArray>()
        .map_err(|_| PyTypeError::new_err("y: audio data must be a numpy.ndarray"))?;
    if arr.dtype().kind() != b'f' {
        return Err(PyTypeError::new_err("y: audio data must be floating-point"));
    }
    if let Ok(a) = arr.cast::<PyArrayDyn<f64>>() {
        // SAFETY: as in `local_extrema`.
        let v = unsafe { a.as_array() };
        detached(py, v.len(), || ops::valid_audio(&v)).map_err(audio_err_to_py)?;
    } else if let Ok(a) = arr.cast::<PyArrayDyn<f32>>() {
        // SAFETY: as in `local_extrema`.
        let v = unsafe { a.as_array() };
        detached(py, v.len(), || ops::valid_audio(&v)).map_err(audio_err_to_py)?;
    } else {
        if arr.ndim() == 0 {
            return Err(value_err(
                "y",
                "audio data must be at least one-dimensional",
            ));
        }
        let finite = numpy_attr(py, "isfinite")?
            .call1((arr,))?
            .call_method0("all")?;
        if !finite.is_truthy()? {
            return Err(value_err("y", "audio buffer is not finite everywhere"));
        }
    }
    Ok(true)
}

/// Checks that intervals have shape ``(n, 2)`` and non-negative durations.
///
/// Args:
///
///     intervals (``array_like``): One ``[start, end]`` row per interval.
///
/// Returns:
///
///     bool: ``True``.
///
/// Raises:
///
///     ``ValueError``: The shape is wrong or a duration is negative.
#[pyfunction]
#[inline]
pub fn valid_intervals(intervals: &Bound<'_, PyAny>) -> PyResult<bool> {
    with_float_view::<f64, _>(intervals, "intervals", |v| {
        let v = v
            .into_dimensionality::<Ix2>()
            .map_err(|_| value_err("intervals", "must have shape (n, 2)"))?;
        ops::valid_intervals(v).map_err(audio_err_to_py)
    })?;
    Ok(true)
}

/// Whether a value is an integer above zero.
///
/// Args:
///
///     x (Any): The value. ``None``, floats and everything else that is not an integer
///         give ``False``.
///
/// Returns:
///
///     bool: The answer.
#[pyfunction]
#[inline]
pub fn is_positive_int(x: &Bound<'_, PyAny>) -> PyResult<bool> {
    static INTEGER: PyOnceLock<Py<PyType>> = PyOnceLock::new();
    let py = x.py();
    if x.is_instance_of::<PyInt>() {
        return x.gt(0);
    }
    if x.is_instance_of::<PyFloat>() || x.is_none() {
        return Ok(false);
    }
    let integer = INTEGER.get_or_try_init(py, || {
        Ok::<_, PyErr>(numpy_attr(py, "integer")?.cast_into::<PyType>()?.unbind())
    })?;
    if x.is_instance(integer.bind(py))? {
        return x.gt(0);
    }
    Ok(false)
}

/// Converts a number to an ``int`` with a chosen rounding.
///
/// Args:
///
///     x (float): The value.
///     cast (Optional[Callable]): Rounds ``x`` before conversion. Default: floor.
///
/// Returns:
///
///     int: The integer.
///
/// Raises:
///
///     ``TypeError``: ``cast`` is not callable.
#[pyfunction]
#[pyo3(signature = (x, *, cast = None))]
#[inline]
pub fn valid_int<'py>(
    x: &Bound<'py, PyAny>,
    cast: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let py = x.py();
    if let Some(f) = cast.filter(|c| !c.is_none()) {
        if !f.is_callable() {
            return Err(PyTypeError::new_err("cast: must be callable"));
        }
        let rounded = f.call1((x,))?;
        return py.get_type::<PyInt>().call1((rounded,));
    }
    if x.is_instance_of::<PyInt>() {
        return py.get_type::<PyInt>().call1((x,));
    }
    let v: f64 = x.extract()?;
    let out = ops::valid_int(v, ops::IntCast::Floor).map_err(audio_err_to_py)?;
    out.into_bound_py_any(py)
}

fn f64_vec(obj: &Bound<'_, PyAny>, name: &str) -> PyResult<Vec<f64>> {
    with_float_view::<f64, _>(obj, name, |v| Ok(v.iter().copied().collect()))
}

/// Matches each source event to the nearest target event.
///
/// Args:
///
///     ``events_from`` (``array_like``): Source events, in any order.
///     ``events_to`` (``array_like``): Target events, in any order.
///     left (bool): A source event may match a target before it. Default true.
///     right (bool): A source event may match a target after it. Default true.
///
/// Returns:
///
///     numpy.ndarray[int64]: For each source event, an index into ``events_to``. A source
///     event half-way between two targets goes to the later one.
///
/// Raises:
///
///     ``ValueError``: A list is empty, or the restrictions leave an event without a match.
#[pyfunction]
#[pyo3(signature = (events_from, events_to, left = true, right = true))]
#[inline]
pub fn match_events<'py>(
    events_from: &Bound<'py, PyAny>,
    events_to: &Bound<'py, PyAny>,
    left: bool,
    right: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let py = events_from.py();
    let a = f64_vec(events_from, "events_from")?;
    let b = f64_vec(events_to, "events_to")?;
    let out = detached(py, a.len(), || ops::match_events(&a, &b, left, right))
        .map_err(audio_err_to_py)?;
    Ok(PyArray1::from_vec(py, out.into_iter().map(|i| i as i64).collect()).into_any())
}

/// Matches each source interval to the target interval it overlaps most.
///
/// Overlap is the Jaccard index; on a tie the lowest target index wins.
///
/// Args:
///
///     ``intervals_from`` (``array_like``): Source intervals, ``(n, 2)``.
///     ``intervals_to`` (``array_like``): Target intervals, ``(m, 2)``.
///     strict (bool): A source interval that overlaps nothing is an error; when false it
///         matches the nearest target (the earlier at equal distance). Default true.
///
/// Returns:
///
///     numpy.ndarray[int64]: For each source interval, an index into ``intervals_to``.
///
/// Raises:
///
///     ``ValueError``: A list is empty or invalid, or with ``strict`` an interval overlaps
///     no target.
#[pyfunction]
#[pyo3(signature = (intervals_from, intervals_to, strict = true))]
#[inline]
pub fn match_intervals<'py>(
    intervals_from: &Bound<'py, PyAny>,
    intervals_to: &Bound<'py, PyAny>,
    strict: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let py = intervals_from.py();
    with_float_view::<f64, _>(intervals_from, "intervals_from", |a| {
        with_float_view::<f64, _>(intervals_to, "intervals_to", |b| {
            let a = a
                .into_dimensionality::<Ix2>()
                .map_err(|_| value_err("intervals_from", "must have shape (n, 2)"))?;
            let b = b
                .into_dimensionality::<Ix2>()
                .map_err(|_| value_err("intervals_to", "must have shape (m, 2)"))?;
            let out = detached(py, a.len(), || ops::match_intervals(a, b, strict))
                .map_err(audio_err_to_py)?;
            Ok(PyArray1::from_vec(py, out.into_iter().map(|i| i as i64).collect()).into_any())
        })
    })
}

#[inline]
pub fn register(py: Python<'_>, parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(py, "arrays")?;
    m.add(
        "__doc__",
        "Array helpers: local extrema, event and interval matching, one-dimensional \
         interpolation, padding, reordering, compressed sparse rows and argument checks.",
    )?;
    m.add("MAX_MEM_BLOCK", ops::MAX_MEM_BLOCK)?;
    m.add_class::<SparseRows>()?;
    m.add_function(wrap_pyfunction!(localmax, &m)?)?;
    m.add_function(wrap_pyfunction!(localmin, &m)?)?;
    m.add_function(wrap_pyfunction!(abs2, &m)?)?;
    m.add_function(wrap_pyfunction!(phasor, &m)?)?;
    m.add_function(wrap_pyfunction!(tiny, &m)?)?;
    m.add_function(wrap_pyfunction!(expand_to, &m)?)?;
    m.add_function(wrap_pyfunction!(stack, &m)?)?;
    m.add_function(wrap_pyfunction!(pad_center, &m)?)?;
    m.add_function(wrap_pyfunction!(fix_frames, &m)?)?;
    m.add_function(wrap_pyfunction!(index_to_slice, &m)?)?;
    m.add_function(wrap_pyfunction!(axis_sort, &m)?)?;
    m.add_function(wrap_pyfunction!(shear, &m)?)?;
    m.add_function(wrap_pyfunction!(sparsify_rows, &m)?)?;
    m.add_function(wrap_pyfunction!(cyclic_gradient, &m)?)?;
    m.add_function(wrap_pyfunction!(fill_off_diagonal, &m)?)?;
    m.add_function(wrap_pyfunction!(count_unique, &m)?)?;
    m.add_function(wrap_pyfunction!(is_unique, &m)?)?;
    m.add_function(wrap_pyfunction!(interp1d, &m)?)?;
    m.add_function(wrap_pyfunction!(interp_broadcast, &m)?)?;
    m.add_function(wrap_pyfunction!(dtype_c2r, &m)?)?;
    m.add_function(wrap_pyfunction!(dtype_r2c, &m)?)?;
    m.add_function(wrap_pyfunction!(buf_to_float, &m)?)?;
    m.add_function(wrap_pyfunction!(find_files, &m)?)?;
    m.add_function(wrap_pyfunction!(valid_audio, &m)?)?;
    m.add_function(wrap_pyfunction!(valid_intervals, &m)?)?;
    m.add_function(wrap_pyfunction!(is_positive_int, &m)?)?;
    m.add_function(wrap_pyfunction!(valid_int, &m)?)?;
    m.add_function(wrap_pyfunction!(zero_crossings, &m)?)?;
    m.add_function(wrap_pyfunction!(tone, &m)?)?;
    m.add_function(wrap_pyfunction!(chirp, &m)?)?;
    m.add_function(wrap_pyfunction!(match_events, &m)?)?;
    m.add_function(wrap_pyfunction!(match_intervals, &m)?)?;
    parent.add_submodule(&m)?;
    Ok(())
}
