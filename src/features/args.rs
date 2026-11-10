//! Argument parsers specific to the analysis bindings (features, tonal,
//! constant-Q, pitch tracking): callable references (`ref=np.max`,
//! `aggregate=np.median`), `fill_na`, `res_type`, weighting kinds and the
//! array-shape helpers shared by `audio_samples.features` and the
//! corresponding `AudioSamples` methods.

use audio_samples::operations::ResamplingQuality;
use audio_samples::operations::constant_q::CqtMode;
use audio_samples::operations::features::{
    FrequencyWeighting, ReassignRefPower, ZeroCrossingReference,
};
use audio_samples::operations::pitch_tracking::{SalienceAggregate, UnvoicedFill};
use audio_samples::operations::tonal::PiptrackReference;
use ndarray::{ArrayD, Axis, Ix2, Ix3};
use numpy::{IntoPyArray, PyUntypedArrayMethods};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::PyString;

use crate::args::{f64_array, size_f64, value_err};

/// If `obj` is one of the named NumPy functions (`numpy.<name>`), returns the
/// matching name.
fn numpy_callable<'a>(obj: &Bound<'_, PyAny>, names: &[&'a str]) -> PyResult<Option<&'a str>> {
    if !obj.is_callable() {
        return Ok(None);
    }
    let np = obj.py().import("numpy")?;
    for &n in names {
        if let Ok(f) = np.getattr(n)
            && obj.is(&f)
        {
            return Ok(Some(n));
        }
    }
    Ok(None)
}

fn string_arg(obj: &Bound<'_, PyAny>) -> PyResult<Option<String>> {
    match obj.cast::<PyString>() {
        Ok(s) => Ok(Some(s.to_cow()?.into_owned())),
        Err(_) => Ok(None),
    }
}

/// piptrack `ref`: `None` / `numpy.max` / `"max"` (per-frame maximum),
/// `numpy.mean` / `"mean"` (per-frame mean) or a number (absolute threshold).
pub fn parse_piptrack_ref(obj: &Bound<'_, PyAny>) -> PyResult<PiptrackReference> {
    if obj.is_none() {
        return Ok(PiptrackReference::FrameMax);
    }
    let name = match numpy_callable(obj, &["max", "amax", "mean"])? {
        Some(n) => Some(n.to_string()),
        None => string_arg(obj)?,
    };
    if let Some(name) = name {
        return match name.as_str() {
            "max" | "amax" => Ok(PiptrackReference::FrameMax),
            "mean" => Ok(PiptrackReference::FrameMean),
            other => Err(value_err(
                "ref",
                format!(
                    "unsupported reference {other:?}; expected None, numpy.max, numpy.mean or a number"
                ),
            )),
        };
    }
    if obj.is_callable() {
        return Err(PyTypeError::new_err(
            "ref: only numpy.max and numpy.mean are supported as callables",
        ));
    }
    let v: f64 = obj
        .extract()
        .map_err(|_| value_err("ref", "expected None, numpy.max, numpy.mean or a number"))?;
    Ok(PiptrackReference::Value(v))
}

/// `reassigned_spectrogram` `ref_power`: a number or `numpy.max` / `"max"`.
pub fn parse_reassign_ref(obj: &Bound<'_, PyAny>) -> PyResult<ReassignRefPower> {
    if numpy_callable(obj, &["max", "amax"])?.is_some()
        || string_arg(obj)?.as_deref() == Some("max")
    {
        return Ok(ReassignRefPower::Max);
    }
    if obj.is_callable() {
        return Err(PyTypeError::new_err(
            "ref_power: only numpy.max is supported as a callable",
        ));
    }
    let v: f64 = obj
        .extract()
        .map_err(|_| value_err("ref_power", "expected a number or numpy.max"))?;
    Ok(ReassignRefPower::Value(v))
}

/// `zero_crossing_rate` `ref_magnitude`: `None`, a number or `numpy.max` / `"max"`.
pub fn parse_zcr_ref(obj: &Bound<'_, PyAny>) -> PyResult<ZeroCrossingReference> {
    if obj.is_none() {
        return Ok(ZeroCrossingReference::None);
    }
    if numpy_callable(obj, &["max", "amax"])?.is_some()
        || string_arg(obj)?.as_deref() == Some("max")
    {
        return Ok(ZeroCrossingReference::Max);
    }
    if obj.is_callable() {
        return Err(PyTypeError::new_err(
            "ref_magnitude: only numpy.max is supported as a callable",
        ));
    }
    let v: f64 = obj
        .extract()
        .map_err(|_| value_err("ref_magnitude", "expected None, a number or numpy.max"))?;
    Ok(ZeroCrossingReference::Value(v))
}

/// pyin `fill_na`: `None` (best-guess pitch), NaN (default) or a number.
pub fn parse_fill_na(obj: &Bound<'_, PyAny>) -> PyResult<UnvoicedFill> {
    if obj.is_none() {
        return Ok(UnvoicedFill::BestGuess);
    }
    let v: f64 = obj.extract().map_err(|_| {
        value_err(
            "fill_na",
            "expected None or a number (numpy.nan by default)",
        )
    })?;
    Ok(if v.is_nan() {
        UnvoicedFill::Nan
    } else {
        UnvoicedFill::Value(v)
    })
}

/// salience `aggregate`: `None` / `numpy.average` (weighted average, the
/// default), `numpy.mean` (unweighted average), `numpy.sum`, `numpy.max`,
/// `numpy.min`, `numpy.median`, or those names as strings.
///
/// Returns the aggregate and whether it was `numpy.mean` (which ignores
/// `weights`).
pub fn parse_aggregate(obj: &Bound<'_, PyAny>) -> PyResult<(SalienceAggregate, bool)> {
    if obj.is_none() {
        return Ok((SalienceAggregate::Average, false));
    }
    let names = [
        "average", "mean", "sum", "max", "amax", "min", "amin", "median",
    ];
    let name = match numpy_callable(obj, &names)? {
        Some(n) => n.to_string(),
        None => match string_arg(obj)? {
            Some(s) => s,
            None => {
                return Err(PyTypeError::new_err(
                    "aggregate: expected None, numpy.average, numpy.mean, numpy.sum, numpy.max, \
                     numpy.min or numpy.median",
                ));
            }
        },
    };
    Ok(match name.as_str() {
        "average" => (SalienceAggregate::Average, false),
        "mean" => (SalienceAggregate::Average, true),
        "sum" => (SalienceAggregate::Sum, false),
        "max" | "amax" => (SalienceAggregate::Max, false),
        "min" | "amin" => (SalienceAggregate::Min, false),
        "median" => (SalienceAggregate::Median, false),
        other => {
            return Err(value_err(
                "aggregate",
                format!("unsupported aggregate {other:?}"),
            ));
        }
    })
}

/// `res_type` -> resampler quality. `"soxr_hq"` (default), `"soxr_vhq"`
/// and `"high"` select [`ResamplingQuality::High`]; `"soxr_mq"`/`"medium"`
/// [`ResamplingQuality::Medium`]; `"soxr_lq"`/`"soxr_qq"`/`"fast"`
/// [`ResamplingQuality::Fast`].
pub fn parse_res_type(res_type: &str) -> PyResult<ResamplingQuality> {
    match res_type {
        "soxr_hq" | "soxr_vhq" | "high" => Ok(ResamplingQuality::High),
        "soxr_mq" | "medium" => Ok(ResamplingQuality::Medium),
        "soxr_lq" | "soxr_qq" | "fast" => Ok(ResamplingQuality::Fast),
        other => Err(value_err(
            "res_type",
            format!(
                "unsupported res_type {other:?}; expected 'soxr_hq' (default), 'soxr_vhq', \
                 'soxr_mq', 'soxr_lq', 'high', 'medium' or 'fast' (the crate's own resampler \
                 is used for all of them)"
            ),
        )),
    }
}

/// chroma `cqt_mode`: `"full"` or `"hybrid"`.
pub fn parse_cqt_mode(mode: &str) -> PyResult<CqtMode> {
    match mode {
        "full" => Ok(CqtMode::Full),
        "hybrid" => Ok(CqtMode::Hybrid),
        other => Err(value_err(
            "cqt_mode",
            format!("expected 'full' or 'hybrid', got {other:?}"),
        )),
    }
}

/// One weighting curve name (`"A"`, `"B"`, `"C"`, `"D"`, `"Z"`; case-insensitive).
pub fn parse_weighting(name: &str, kind: &str) -> PyResult<FrequencyWeighting> {
    match kind.to_ascii_uppercase().as_str() {
        "A" => Ok(FrequencyWeighting::A),
        "B" => Ok(FrequencyWeighting::B),
        "C" => Ok(FrequencyWeighting::C),
        "D" => Ok(FrequencyWeighting::D),
        "Z" => Ok(FrequencyWeighting::Z),
        other => Err(value_err(
            name,
            format!("unknown weighting {other:?}; expected 'A', 'B', 'C', 'D' or 'Z'"),
        )),
    }
}

/// The `intervals` of the VQT: only `"equal"` (equal temperament) is supported.
pub fn check_intervals(obj: &Bound<'_, PyAny>) -> PyResult<()> {
    match string_arg(obj)?.as_deref() {
        Some("equal") => Ok(()),
        _ => Err(value_err(
            "intervals",
            "only intervals='equal' is supported (just-intonation interval sets are not)",
        )),
    }
}

/// pyin `fill_na` as a signature parameter: the default is NaN and an explicit
/// `None` means "best guess", which `Option<f64>` cannot express.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FillNaArg(pub UnvoicedFill);

impl<'a, 'py> FromPyObject<'a, 'py> for FillNaArg {
    type Error = PyErr;
    #[cfg(feature = "stubs")]
    const INPUT_TYPE: pyo3::inspect::PyStaticExpr = crate::args::hints::OPT_FLOAT;

    fn extract(obj: pyo3::Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        parse_fill_na(&obj.to_owned()).map(FillNaArg)
    }
}

/// `reassigned_spectrogram` `ref_power` as a signature parameter.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ReassignRefArg(pub ReassignRefPower);

impl<'a, 'py> FromPyObject<'a, 'py> for ReassignRefArg {
    type Error = PyErr;
    #[cfg(feature = "stubs")]
    const INPUT_TYPE: pyo3::inspect::PyStaticExpr = crate::args::hints::FLOAT_OR_CALLABLE;

    fn extract(obj: pyo3::Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        parse_reassign_ref(&obj.to_owned()).map(ReassignRefArg)
    }
}

// =============================================================================
// Array shape helpers
// =============================================================================

/// A NumPy array argument viewed with a leading channel axis added when the
/// input lacks one (`(F, T)` as `(1, F, T)`, `(n,)` as `(1, n)`), without
/// copying the data. `D` is the dimension of the view.
pub struct ChannelView<'py, F: numpy::Element, D> {
    arr: numpy::PyReadonlyArrayDyn<'py, F>,
    _dim: std::marker::PhantomData<D>,
}

impl<F: numpy::Element, D: ndarray::Dimension> ChannelView<'_, F, D> {
    /// The data as a `D`-dimensional view (channel axis first).
    pub(crate) fn view(&self) -> ndarray::ArrayView<'_, F, D> {
        let v = self.arr.as_array();
        let v = if v.ndim() + 1 == D::NDIM.expect("fixed dimension") {
            v.insert_axis(Axis(0))
        } else {
            v
        };
        v.into_dimensionality::<D>().expect("ndim checked")
    }

    /// Shape of [`Self::view`].
    pub(crate) fn dim(&self) -> D::Pattern {
        self.view().dim()
    }
}

/// Wraps `obj` as a [`ChannelView`] of dimension `D` if it has `D::NDIM`
/// dimensions, or `D::NDIM - 1` when `lift` is set; returns the input's number
/// of dimensions too. Otherwise raises `ValueError("{name}: expected {what},
/// got {n} dimensions")`.
pub fn channel_view<'py, F: crate::args::FloatElement, D: ndarray::Dimension>(
    obj: &Bound<'py, PyAny>,
    name: &str,
    what: &str,
    lift: bool,
) -> PyResult<(ChannelView<'py, F, D>, usize)> {
    let arr = crate::args::float_array_as::<F>(obj, name)?;
    let nd = arr.ndim();
    let full = D::NDIM.expect("fixed dimension");
    if !(nd == full || (lift && nd + 1 == full)) {
        return Err(value_err(
            name,
            format!("expected {what}, got {nd} dimensions"),
        ));
    }
    Ok((
        ChannelView {
            arr,
            _dim: std::marker::PhantomData,
        },
        nd,
    ))
}

/// A spectrogram-like input of shape `(F, T)` or `(channels, F, T)` as a
/// `(channels, F, T)` view (no copy) plus the input's number of dimensions.
pub fn spec3<'py, F: crate::args::FloatElement>(
    obj: &Bound<'py, PyAny>,
    name: &str,
) -> PyResult<(ChannelView<'py, F, Ix3>, usize)> {
    channel_view(
        obj,
        name,
        "a 2-D (F, T) or 3-D (channels, F, T) array",
        true,
    )
}

/// A signal of shape `(n,)` or `(channels, n)` as a `(channels, n)` view (no
/// copy) plus the input's number of dimensions.
pub fn signal2<'py, F: crate::args::FloatElement>(
    obj: &Bound<'py, PyAny>,
    name: &str,
) -> PyResult<(ChannelView<'py, F, Ix2>, usize)> {
    channel_view(obj, name, "a 1-D (n,) or 2-D (channels, n) signal", true)
}

/// Drops the leading (length-1) channel axis of `a` when `drop_channel` is set,
/// i.e. when the input had no channel axis, so the output has none either.
pub fn squeeze_out<T: numpy::Element>(
    py: Python<'_>,
    a: ArrayD<T>,
    drop_channel: bool,
) -> Bound<'_, PyAny> {
    if drop_channel && a.shape().first() == Some(&1) {
        a.index_axis_move(Axis(0), 0).into_pyarray(py).into_any()
    } else {
        a.into_pyarray(py).into_any()
    }
}

/// A 1-D float vector from an array-like (any shape is flattened).
pub fn f64_vec(obj: &Bound<'_, PyAny>, name: &str) -> PyResult<Vec<f64>> {
    let a = f64_array(obj, name)?;
    Ok(a.as_array().iter().copied().collect())
}

/// A strictly one-dimensional float vector.
pub fn f64_vec1(obj: &Bound<'_, PyAny>, name: &str) -> PyResult<Vec<f64>> {
    let a = f64_array(obj, name)?;
    if a.ndim() != 1 {
        return Err(value_err(
            name,
            format!("expected a 1-D array, got {} dimensions", a.ndim()),
        ));
    }
    Ok(a.as_array().iter().copied().collect())
}

/// The frequency of each rFFT bin, `k * sr / n_fft` for `k` in `0..=n_fft / 2`,
/// rounded as `numpy.fft.rfftfreq(n_fft, 1 / sr)`.
pub fn rfft_frequencies(n_fft: usize, sr: f64) -> Vec<f64> {
    let val = 1.0 / (size_f64(n_fft) * (1.0 / sr));
    (0..=n_fft / 2).map(|k| size_f64(k) * val).collect()
}

/// The `freq` argument of the spectral descriptors: `None` means
/// `fft_frequencies(sr, n_fft = 2 * (n_bins - 1))`; otherwise a 1-D array of
/// one centre frequency per bin.
pub fn bin_frequencies(
    freq: Option<&Bound<'_, PyAny>>,
    sr: f64,
    n_bins: usize,
) -> PyResult<Vec<f64>> {
    match freq {
        None => Ok(rfft_frequencies(2 * n_bins.saturating_sub(1), sr)),
        Some(f) if f.is_none() => Ok(rfft_frequencies(2 * n_bins.saturating_sub(1), sr)),
        Some(f) => f64_vec1(f, "freq"),
    }
}

/// `freq` as an array when it has a frequency for every frame (two or more dimensions).
pub fn freq_matrix<'py, F: crate::args::FloatElement>(
    freq: Option<&Bound<'py, PyAny>>,
) -> PyResult<Option<numpy::PyReadonlyArrayDyn<'py, F>>> {
    let Some(f) = freq.filter(|f| freq_is_matrix(f)) else {
        return Ok(None);
    };
    Ok(Some(crate::args::float_array_as::<F>(f, "freq")?))
}

/// Whether `freq` has two or more dimensions (a frequency for every frame). Anything
/// that is not an array of them is left to the 1-D conversion and its errors.
pub fn freq_is_matrix(f: &Bound<'_, PyAny>) -> bool {
    if f.is_none() {
        return false;
    }
    f.cast::<numpy::PyUntypedArray>().map_or_else(
        |_| f64_array(f, "freq").is_ok_and(|a| a.ndim() >= 2),
        |a| a.ndim() >= 2,
    )
}

/// A `(bins, frames)` or `(channels, bins, frames)` `freq` as a 3-D view.
pub fn freq3<'a, F: crate::args::FloatElement>(
    freq: &'a numpy::PyReadonlyArrayDyn<'_, F>,
) -> PyResult<ndarray::ArrayView3<'a, F>> {
    let view = freq.as_array();
    match view.ndim() {
        2 => Ok(view
            .insert_axis(ndarray::Axis(0))
            .into_dimensionality()
            .expect("3-D")),
        3 => Ok(view.into_dimensionality().expect("3-D")),
        n => Err(value_err(
            "freq",
            format!("expected 1, 2 or 3 dimensions, got {n}"),
        )),
    }
}
