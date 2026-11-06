//! `audio_samples.segment`: path enhancement, temporally constrained clustering,
//! subsegmentation and lag-domain filtering; and `sequence.path_to_steps`.

#![allow(
    // Sizes and axes from Python are checked before these casts.
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    // Docstrings are the Python documentation; failures are listed under `Raises`.
    clippy::missing_errors_doc,
    // Indices come from lengths checked at the top of each function.
    clippy::indexing_slicing,
    // PyO3 passes extracted arguments by value.
    clippy::needless_pass_by_value,
)]

use audio_samples::operations::filterbanks::Window;
use audio_samples::operations::segment::{self as sg, BoundaryMode, PathEnhanceConfig};
use audio_samples::operations::sequence as seq;
use ndarray::{Ix2, IxDyn};
use numpy::{IntoPyArray, PyArray1, PyUntypedArray, PyUntypedArrayMethods};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};

use crate::args::{
    Int, PyFloat, WindowArg, i64_vec, positive, precision_of, value_err, with_float_view,
    with_precision,
};
use crate::audio_err_to_py;

/// A normalised smoothing kernel along a line through the origin.
///
/// Args:
///     window (str | tuple | ``array_like``): The window along the line.
///     n (int): Its length, before rotation.
///     slope (float): The line's slope; 1 is the main diagonal. Default 1.
///     angle (Optional[float]): The line's angle in radians, in place of ``slope``.
///     ``zero_mean`` (bool): Subtract the mean. Default false.
///
/// Returns:
///     numpy.ndarray[float64]: The kernel.
#[pyfunction]
#[pyo3(
    signature = (window, n, *, slope = 1.0, angle = None, zero_mean = false),
    text_signature = "(window, n, *, slope=1.0, angle=None, zero_mean=False)"
)]
#[inline]
pub fn diagonal_filter(
    py: Python<'_>,
    window: WindowArg,
    n: Int,
    slope: f64,
    angle: Option<f64>,
    zero_mean: bool,
) -> PyResult<Bound<'_, PyAny>> {
    let n = positive("n", n)?.get();
    let k = sg::diagonal_filter(&window.0, n, slope, angle, zero_mean).map_err(audio_err_to_py)?;
    Ok(k.into_pyarray(py).into_any())
}

fn parse_boundary(kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<(BoundaryMode, (isize, isize))> {
    let mut mode = "reflect".to_string();
    let mut cval = 0.0;
    let mut origin = (0, 0);
    if let Some(kw) = kwargs {
        for (k, v) in kw.iter() {
            let key: String = k.extract()?;
            match key.as_str() {
                "mode" => mode = v.extract()?,
                "cval" => cval = v.extract()?,
                "origin" => {
                    origin = if let Ok(o) = v.extract::<isize>() {
                        (o, o)
                    } else {
                        let pair: Vec<isize> = v.extract()?;
                        if pair.len() != 2 {
                            return Err(value_err("origin", "expected one number or two"));
                        }
                        (pair[0], pair[1])
                    };
                }
                other => return Err(value_err(other, "is not a parameter of the convolution")),
            }
        }
    }
    let boundary = match mode.as_str() {
        "reflect" | "grid-mirror" => BoundaryMode::Reflect,
        "constant" | "grid-constant" => BoundaryMode::Constant(cval),
        "nearest" => BoundaryMode::Nearest,
        "mirror" => BoundaryMode::Mirror,
        "wrap" | "grid-wrap" => BoundaryMode::Wrap,
        other => return Err(value_err("mode", format!("unknown {other:?}"))),
    };
    Ok((boundary, origin))
}

fn enhance<'py, F: PyFloat>(
    r: &Bound<'py, PyAny>,
    n: usize,
    cfg: &PathEnhanceConfig,
) -> PyResult<Bound<'py, PyAny>> {
    let py = r.py();
    with_float_view::<F, _>(r, "R", |v| {
        if v.ndim() < 2 {
            return Err(value_err("R", "expected at least 2 dimensions"));
        }
        if v.ndim() == 2 {
            let m = v
                .into_dimensionality::<Ix2>()
                .map_err(|e| value_err("R", e))?;
            let out = py
                .detach(|| sg::path_enhance(m, n, cfg))
                .map_err(audio_err_to_py)?;
            return Ok(out.into_pyarray(py).into_any());
        }
        let shape = v.shape().to_vec();
        let (rows, cols) = (shape[shape.len() - 2], shape[shape.len() - 1]);
        let flat = v.as_standard_layout();
        let flat = flat
            .as_slice()
            .ok_or_else(|| value_err("R", "expected contiguous data"))?;
        let mut out = Vec::with_capacity(flat.len());
        let mut failure = None;
        py.detach(|| {
            for plane in flat.chunks(rows * cols) {
                let m = match ndarray::ArrayView2::from_shape((rows, cols), plane) {
                    Ok(m) => m,
                    Err(e) => {
                        failure = Some(e.to_string());
                        return Ok(());
                    }
                };
                let e = sg::path_enhance(m, n, cfg)?;
                out.extend(e.iter().copied());
            }
            Ok(())
        })
        .map_err(audio_err_to_py)?;
        if let Some(e) = failure {
            return Err(value_err("R", e));
        }
        let a =
            ndarray::ArrayD::from_shape_vec(IxDyn(&shape), out).map_err(|e| value_err("R", e))?;
        Ok(a.into_pyarray(py).into_any())
    })
}

/// The largest response of a recurrence matrix to diagonal kernels of several slopes.
///
/// Args:
///     R (``array_like``): ``(..., n, n)`` or lag matrix.
///     n (int): Length of each kernel, before rotation.
///     window (str | tuple | ``array_like``): The window along each diagonal. Default ``"hann"``.
///     ``max_ratio`` (float): Largest slope. Default 2.
///     ``min_ratio`` (Optional[float]): Smallest slope. Default ``1 / max_ratio``.
///     ``n_filters`` (int): Number of slopes. Default 7.
///     ``zero_mean`` (bool): Subtract each kernel's mean. Default false.
///     clip (bool): Set negative results to zero. Default true.
///     **kwargs: ``mode`` (``"reflect"``, ``"constant"``, ``"nearest"``, ``"mirror"``,
///         ``"wrap"``), ``cval`` and ``origin``.
///
/// Returns:
///     numpy.ndarray: The shape of ``R``, of its precision.
#[pyfunction]
#[pyo3(
    signature = (R, n, *, window = WindowArg(Window::Hann), max_ratio = 2.0, min_ratio = None, n_filters = Int(7), zero_mean = false, clip = true, **kwargs),
    text_signature = "(R, n, *, window='hann', max_ratio=2.0, min_ratio=None, n_filters=7, zero_mean=False, clip=True, **kwargs)"
)]
#[allow(non_snake_case, clippy::too_many_arguments)]
#[inline]
pub fn path_enhance<'py>(
    R: &Bound<'py, PyAny>,
    n: Int,
    window: WindowArg,
    max_ratio: f64,
    min_ratio: Option<f64>,
    n_filters: Int,
    zero_mean: bool,
    clip: bool,
    kwargs: Option<&Bound<'py, PyDict>>,
) -> PyResult<Bound<'py, PyAny>> {
    let n = positive("n", n)?.get();
    let (mode, origin) = parse_boundary(kwargs)?;
    let cfg = PathEnhanceConfig::new()
        .with_window(window.0)
        .with_ratios(
            max_ratio,
            min_ratio,
            positive("n_filters", n_filters)?.get(),
        )
        .with_zero_mean(zero_mean)
        .with_clip(clip)
        .with_mode(mode)
        .with_origin(origin);
    with_precision!(precision_of(R), |F| enhance::<F>(R, n, &cfg))
}

/// The filter that [`timelag_filter`] returns.
#[pyclass(module = "audio_samples.segment", frozen)]
pub struct TimelagFiltered {
    function: Py<PyAny>,
    pad: bool,
    index: usize,
}

#[pymethods]
impl TimelagFiltered {
    /// Applies the wrapped filter in lag coordinates.
    #[pyo3(signature = (*args, **kwargs))]
    fn __call__<'py>(
        &self,
        py: Python<'py>,
        args: &Bound<'py, PyTuple>,
        kwargs: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let input = args.get_item(self.index).map_err(|_| {
            value_err(
                "index",
                format!("the call has no positional argument {}", self.index),
            )
        })?;
        let lag = crate::sequence::recurrence_to_lag(py, &input, self.pad, Int(-1))?;
        let call_args = PyList::empty(py);
        for (i, a) in args.iter().enumerate() {
            if i == self.index {
                call_args.append(&lag)?;
            } else {
                call_args.append(a)?;
            }
        }
        let result = self.function.bind(py).call(call_args.to_tuple(), kwargs)?;
        crate::sequence::lag_to_recurrence(py, &result, Int(-1))
    }

    /// The wrapped function.
    #[getter]
    fn __wrapped__(&self, py: Python<'_>) -> Py<PyAny> {
        self.function.clone_ref(py)
    }
}

/// Wraps a filter so that it runs on a recurrence matrix in lag coordinates.
///
/// Args:
///     function (Callable): The filter. It receives the time-lag matrix at position ``index``.
///     pad (bool): Pad the lag matrix to twice the length. Default true.
///     index (int): Position of the matrix in the call's arguments. Default 0.
///
/// Returns:
///     Callable: Takes the same arguments as ``function`` and returns a recurrence matrix.
#[pyfunction]
#[pyo3(signature = (function, pad = true, index = Int(0)), text_signature = "(function, pad=True, index=0)")]
#[inline]
pub fn timelag_filter(function: Py<PyAny>, pad: bool, index: Int) -> PyResult<TimelagFiltered> {
    Ok(TimelagFiltered {
        function,
        pad,
        index: crate::args::non_negative("index", index)?,
    })
}

/// `(features, frames)` as an array, from `data` of any dimension with time along `axis`.
fn frames_last<'py>(data: &Bound<'py, PyAny>, axis: Int) -> PyResult<Bound<'py, PyAny>> {
    let py = data.py();
    let numpy = py.import("numpy")?;
    let a = numpy.getattr("atleast_2d")?.call1((data,))?;
    let arr = a.cast::<PyUntypedArray>()?;
    let n = arr.ndim() as i64;
    if axis.0 < -n || axis.0 >= n {
        return Err(value_err("axis", format!("{} is out of range", axis.0)));
    }
    if arr.ndim() == 2 && (axis.0 == -1 || axis.0 == 1) {
        return Ok(a);
    }
    let swapped = numpy.getattr("swapaxes")?.call1((&a, axis.0, 0))?;
    let frames: usize = swapped.getattr("shape")?.get_item(0)?.extract()?;
    let kw = PyDict::new(py);
    kw.set_item("order", "F")?;
    swapped
        .call_method("reshape", ((frames, -1_i64),), Some(&kw))?
        .getattr("T")
}

fn boundaries_out(py: Python<'_>, v: Vec<usize>) -> Bound<'_, PyAny> {
    PyArray1::from_vec(py, v.into_iter().map(|i| i as i64).collect::<Vec<i64>>()).into_any()
}

/// Segment boundaries from Ward clustering in which only neighbouring segments merge.
///
/// Args:
///     data (``array_like``): Features with time along ``axis``.
///     k (int): Number of segments.
///     clusterer (Optional[object]): Any object with ``fit`` and ``labels_``, in place of
///         the built-in clustering.
///     axis (int): The time axis. Default -1.
///
/// Returns:
///     numpy.ndarray[int64]: The first frame of each segment.
#[pyfunction]
#[pyo3(
    signature = (data, k, *, clusterer = None, axis = Int(-1)),
    text_signature = "(data, k, *, clusterer=None, axis=-1)"
)]
#[inline]
pub fn agglomerative<'py>(
    data: &Bound<'py, PyAny>,
    k: Int,
    clusterer: Option<&Bound<'py, PyAny>>,
    axis: Int,
) -> PyResult<Bound<'py, PyAny>> {
    let py = data.py();
    let features = frames_last(data, axis)?;
    if let Some(model) = clusterer.filter(|c| !c.is_none()) {
        model.call_method1("fit", (features.getattr("T")?,))?;
        let labels = i64_vec(&model.getattr("labels_")?, "labels_")?;
        let mut out = vec![0_usize];
        out.extend(
            labels
                .windows(2)
                .enumerate()
                .filter(|(_, w)| w[0] != w[1])
                .map(|(i, _)| i + 1),
        );
        return Ok(boundaries_out(py, out));
    }
    let k = positive("k", k)?.get();
    with_precision!(precision_of(&features), |F| {
        with_float_view::<F, _>(&features, "data", |v| {
            let v = v
                .into_dimensionality::<Ix2>()
                .map_err(|e| value_err("data", e))?;
            let b = py
                .detach(|| sg::agglomerative(v, k))
                .map_err(audio_err_to_py)?;
            Ok(boundaries_out(py, b))
        })
    })
}

/// Subdivides each interval of a segmentation with :func:`agglomerative`.
///
/// Args:
///     data (``array_like``): Features with time along ``axis``.
///     frames (``array_like[int]``): Segment boundaries; the ends are added.
///     ``n_segments`` (int): The most subsegments per interval. Default 4.
///     axis (int): The time axis. Default -1.
///
/// Returns:
///     numpy.ndarray[int64]: The first frame of each subsegment.
#[pyfunction]
#[pyo3(
    signature = (data, frames, *, n_segments = Int(4), axis = Int(-1)),
    text_signature = "(data, frames, *, n_segments=4, axis=-1)"
)]
#[inline]
pub fn subsegment<'py>(
    data: &Bound<'py, PyAny>,
    frames: &Bound<'py, PyAny>,
    n_segments: Int,
    axis: Int,
) -> PyResult<Bound<'py, PyAny>> {
    let py = data.py();
    let features = frames_last(data, axis)?;
    let bounds = i64_vec(frames, "frames")?;
    let n = positive("n_segments", n_segments)?.get();
    with_precision!(precision_of(&features), |F| {
        with_float_view::<F, _>(&features, "data", |v| {
            let v = v
                .into_dimensionality::<Ix2>()
                .map_err(|e| value_err("data", e))?;
            let b = py
                .detach(|| sg::subsegment(v, &bounds, n))
                .map_err(audio_err_to_py)?;
            Ok(boundaries_out(py, b))
        })
    })
}

/// The fractional index on one side of an alignment path for each frame on the other.
///
/// Args:
///     path (``array_like``): ``(n, 2)`` index pairs, as ``sequence.dtw`` gives.
///     inverse (bool): False gives column 0 for each frame of column 1. Default false.
///
/// Returns:
///     numpy.ndarray[float64]: One value per frame on the path.
#[pyfunction]
#[pyo3(signature = (path, *, inverse = false), text_signature = "(path, *, inverse=False)")]
#[inline]
pub fn path_to_steps<'py>(path: &Bound<'py, PyAny>, inverse: bool) -> PyResult<Bound<'py, PyAny>> {
    let py = path.py();
    let values = i64_vec(path, "path")?;
    if values.len() % 2 != 0 || values.iter().any(|&v| v < 0) {
        return Err(value_err("path", "expected (n, 2) non-negative indices"));
    }
    let pairs: Vec<[usize; 2]> = values
        .chunks_exact(2)
        .map(|c| [c[0] as usize, c[1] as usize])
        .collect();
    let out = seq::path_to_steps(&pairs, inverse).map_err(audio_err_to_py)?;
    Ok(out.into_pyarray(py).into_any())
}

#[inline]
pub fn register(py: Python<'_>, parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(py, "segment")?;
    m.add(
        "__doc__",
        "Structure analysis over feature matrices: path enhancement, temporally constrained clustering, subsegmentation.",
    )?;
    m.add_class::<TimelagFiltered>()?;
    m.add_function(wrap_pyfunction!(diagonal_filter, &m)?)?;
    m.add_function(wrap_pyfunction!(path_enhance, &m)?)?;
    m.add_function(wrap_pyfunction!(timelag_filter, &m)?)?;
    m.add_function(wrap_pyfunction!(agglomerative, &m)?)?;
    m.add_function(wrap_pyfunction!(subsegment, &m)?)?;
    parent.add_submodule(&m)?;
    let sequence = parent.getattr("sequence")?.cast_into::<PyModule>()?;
    sequence.add_function(wrap_pyfunction!(path_to_steps, &sequence)?)?;
    Ok(())
}
