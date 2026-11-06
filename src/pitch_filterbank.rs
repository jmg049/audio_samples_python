//! `audio_samples.pitch_filterbank`: multirate semitone filterbank and the IIR
//! time-frequency representation, and `spectral.mellin_transform`.

#![allow(
    // Sizes and axes from Python are checked before these casts.
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    // Docstrings are the Python documentation; failures are listed under `Raises`.
    clippy::missing_errors_doc,
    // Indices come from lengths checked at the top of each function.
    clippy::indexing_slicing,
    // A pair of NumPy arrays per filter, spelled out for PyO3.
    clippy::type_complexity,
)]

use audio_samples::operations::mellin::{self, MellinConfig};
use audio_samples::operations::pitch_filterbank::{
    self as pf, AudioPitchFilterbank, BandFilter, FilterLayout, IirtConfig, MultirateFilterConfig,
};
use audio_samples::operations::types::IirFilterType;
use audio_samples::utils::array_ops::{InterpKind, PadMode};
use ndarray::Axis;
use numpy::{IntoPyArray, PyArray1, PyArray2};
use pyo3::IntoPyObjectExt;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyString};

use crate::args::{
    Int, PyFloat, opt_positive, positive, precision_of, value_err, with_float_view, with_precision,
};
use crate::features::args::{f64_vec, parse_res_type, signal2, squeeze_out};
use crate::{PyAudioSamples, audio_err_to_py, dispatch_typed};

/// Centre frequencies and sample rates of the semitones from MIDI 24 to 108.
///
/// Args:
///     tuning (float): Deviation from A440 in fractions of a semitone.
///
/// Returns:
///     tuple[numpy.ndarray[float64], numpy.ndarray[float64]]: Centre frequencies and sample
///     rates in Hz, 85 of each.
#[pyfunction]
#[inline]
pub fn mr_frequencies(py: Python<'_>, tuning: f64) -> PyResult<Bound<'_, PyAny>> {
    let (f, r) = pf::mr_frequencies(tuning);
    (PyArray1::from_slice(py, &f), PyArray1::from_slice(py, &r)).into_bound_py_any(py)
}

fn filter_config(
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<(MultirateFilterConfig, Option<(Vec<f64>, Vec<f64>)>)> {
    let mut cfg = MultirateFilterConfig::new();
    let mut freqs = None;
    let mut rates = None;
    if let Some(kw) = kwargs {
        for (k, v) in kw.iter() {
            let key = k.cast::<PyString>()?.to_cow()?;
            match key.as_ref() {
                "Q" => cfg.q = v.extract()?,
                "passband_ripple" => cfg.passband_ripple = v.extract()?,
                "stopband_attenuation" => cfg.stopband_attenuation = v.extract()?,
                "ftype" => {
                    cfg.filter_type = match v.extract::<String>()?.as_str() {
                        "ellip" => IirFilterType::Elliptic,
                        "butter" => IirFilterType::Butterworth,
                        "cheby1" => IirFilterType::ChebyshevI,
                        "cheby2" => IirFilterType::ChebyshevII,
                        other => {
                            return Err(value_err(
                                "ftype",
                                format!(
                                    "unknown {other:?}; expected 'ellip', 'butter', 'cheby1' or 'cheby2'"
                                ),
                            ));
                        }
                    };
                }
                "center_freqs" => {
                    if !v.is_none() {
                        freqs = Some(f64_vec(&v, "center_freqs")?);
                    }
                }
                "sample_rates" => {
                    if !v.is_none() {
                        rates = Some(f64_vec(&v, "sample_rates")?);
                    }
                }
                other => return Err(value_err(other, "is not a parameter of the filterbank")),
            }
        }
    }
    let bands = match (freqs, rates) {
        (Some(f), Some(r)) => Some((f, r)),
        (None, None) => None,
        _ => {
            return Err(value_err(
                "center_freqs",
                "give center_freqs and sample_rates together",
            ));
        }
    };
    Ok((cfg, bands))
}

fn parse_layout(flayout: &str) -> PyResult<FilterLayout> {
    match flayout {
        "sos" => Ok(FilterLayout::Sos),
        "ba" => Ok(FilterLayout::Ba),
        other => Err(value_err(
            "flayout",
            format!("unknown {other:?}; expected 'sos' or 'ba'"),
        )),
    }
}

/// One band-pass filter per semitone, of minimum order for the specified response.
///
/// Args:
///     ``center_freqs`` (Optional[array_like]): Centre frequencies in Hz; give together with
///         ``sample_rates``. Default: :func:`mr_frequencies`.
///     tuning (float): Used when ``center_freqs`` is not given.
///     ``sample_rates`` (Optional[array_like]): The sample rate of each band in Hz.
///     flayout (str): ``"sos"`` (default) or ``"ba"``. Polynomials lose precision at the
///         orders these narrow filters reach.
///     **kwargs: ``Q`` (25), ``passband_ripple`` (1 dB), ``stopband_attenuation`` (50 dB),
///         ``ftype`` (``"ellip"``, ``"butter"``, ``"cheby1"``, ``"cheby2"``).
///
/// Returns:
///     tuple[list, numpy.ndarray[float64]]: One filter per band: an ``(n, 6)`` array of
///     sections, or a ``(b, a)`` tuple; and the sample rates.
#[pyfunction]
#[pyo3(
    signature = (*, center_freqs = None, tuning = 0.0, sample_rates = None, flayout = "sos", **kwargs),
    text_signature = "(*, center_freqs=None, tuning=0.0, sample_rates=None, flayout='sos', **kwargs)"
)]
#[inline]
pub fn semitone_filterbank<'py>(
    py: Python<'py>,
    center_freqs: Option<&Bound<'py, PyAny>>,
    tuning: f64,
    sample_rates: Option<&Bound<'py, PyAny>>,
    flayout: &str,
    kwargs: Option<&Bound<'py, PyDict>>,
) -> PyResult<Bound<'py, PyAny>> {
    let (cfg, _) = filter_config(kwargs)?;
    let layout = parse_layout(flayout)?;
    let (freqs, rates) = match (
        center_freqs.filter(|v| !v.is_none()),
        sample_rates.filter(|v| !v.is_none()),
    ) {
        (Some(f), Some(r)) => (f64_vec(f, "center_freqs")?, f64_vec(r, "sample_rates")?),
        (None, None) => {
            let (f, r) = pf::mr_frequencies(tuning);
            (f.to_vec(), r.to_vec())
        }
        _ => {
            return Err(value_err(
                "center_freqs",
                "give center_freqs and sample_rates together",
            ));
        }
    };
    let bank = pf::semitone_filterbank(&freqs, &rates, &cfg, layout).map_err(audio_err_to_py)?;
    let out = PyList::empty(py);
    for f in bank {
        match f {
            BandFilter::Sos(sos) => {
                let rows: Vec<Vec<f64>> = sos.iter().map(|s| s.to_vec()).collect();
                out.append(PyArray2::from_vec2(py, &rows).map_err(|e| value_err("sos", e))?)?;
            }
            BandFilter::Ba { b, a } => {
                out.append((PyArray1::from_vec(py, b), PyArray1::from_vec(py, a)))?;
            }
        }
    }
    (out, PyArray1::from_slice(py, &rates)).into_bound_py_any(py)
}

fn pad_mode(mode: &str) -> PyResult<PadMode<f64>> {
    Ok(match mode {
        "constant" => PadMode::Constant(0.0),
        "edge" => PadMode::Edge,
        "reflect" => PadMode::Reflect,
        "symmetric" => PadMode::Symmetric,
        "wrap" => PadMode::Wrap,
        other => return Err(value_err("pad_mode", format!("unsupported {other:?}"))),
    })
}

#[allow(clippy::too_many_arguments)]
fn iirt_config(
    win_length: Int,
    hop_length: Option<Int>,
    center: bool,
    tuning: f64,
    pad: &str,
    flayout: &str,
    res_type: &str,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<IirtConfig> {
    let (filter, bands) = filter_config(kwargs)?;
    Ok(IirtConfig::new()
        .with_win_length(positive("win_length", win_length)?)
        .with_hop_length(opt_positive("hop_length", hop_length)?)
        .with_center(center)
        .with_tuning(tuning)
        .with_pad_mode(pad_mode(pad)?)
        .with_layout(parse_layout(flayout)?)
        .with_quality(parse_res_type(res_type)?)
        .with_filter(filter)
        .with_bands(bands))
}

/// The energy of each semitone band per frame.
///
/// Args:
///     y (``array_like``): Signal ``(n,)`` or ``(channels, n)``, finite.
///     sr (int): Sample rate in Hz.
///     ``win_length`` (int): Frame length in samples. Default 2048.
///     ``hop_length`` (Optional[int]): Samples between frames. Default ``win_length // 4``.
///     center (bool): Pad so that frames are centred. Default true.
///     tuning (float): Deviation from A440 in fractions of a semitone.
///     ``pad_mode`` (str): ``"constant"``, ``"edge"``, ``"reflect"``, ``"symmetric"``, ``"wrap"``.
///     flayout (str): ``"sos"`` (default) or ``"ba"``.
///     ``res_type`` (str): Resampler quality, as for ``AudioSamples.resample``.
///     **kwargs: The filterbank's parameters; see :func:`semitone_filterbank`.
///
/// Returns:
///     numpy.ndarray: ``(85, n_frames)``, with a leading channel axis for 2-D input, of
///     ``y``'s precision.
#[pyfunction]
#[pyo3(
    signature = (y, *, sr, win_length = Int(2048), hop_length = None, center = true, tuning = 0.0, pad_mode = "constant", flayout = "sos", res_type = "soxr_hq", **kwargs),
    text_signature = "(y, *, sr, win_length=2048, hop_length=None, center=True, tuning=0.0, pad_mode='constant', flayout='sos', res_type='soxr_hq', **kwargs)"
)]
#[allow(clippy::too_many_arguments)]
#[inline]
pub fn iirt<'py>(
    py: Python<'py>,
    y: &Bound<'py, PyAny>,
    sr: Int,
    win_length: Int,
    hop_length: Option<Int>,
    center: bool,
    tuning: f64,
    pad_mode: &str,
    flayout: &str,
    res_type: &str,
    kwargs: Option<&Bound<'py, PyDict>>,
) -> PyResult<Bound<'py, PyAny>> {
    let cfg = iirt_config(
        win_length, hop_length, center, tuning, pad_mode, flayout, res_type, kwargs,
    )?;
    let rate = crate::args::sample_rate_u32(sr)?;
    with_precision!(precision_of(y), |F| {
        let (s, nd) = signal2::<F>(y, "y")?;
        let view = s.view();
        let out = py
            .detach(|| pf::iirt(view, rate, &cfg))
            .map_err(audio_err_to_py)?;
        Ok(squeeze_out(py, out.into_dyn(), nd == 1))
    })
}

#[pymethods]
impl PyAudioSamples {
    /// The energy of each semitone band per frame.
    ///
    /// Args:
    ///     ``win_length`` (int): Frame length in samples. Default 2048.
    ///     ``hop_length`` (Optional[int]): Samples between frames. Default ``win_length // 4``.
    ///     center (bool): Pad so that frames are centred. Default true.
    ///     tuning (float): Deviation from A440 in fractions of a semitone.
    ///     ``pad_mode`` (str): ``"constant"``, ``"edge"``, ``"reflect"``, ``"symmetric"``, ``"wrap"``.
    ///     flayout (str): ``"sos"`` (default) or ``"ba"``.
    ///     ``res_type`` (str): Resampler quality.
    ///     **kwargs: The filterbank's parameters.
    ///
    /// Returns:
    ///     numpy.ndarray: ``(channels, 85, n_frames)``, ``float32`` for ``u8``, ``i16``,
    ///     ``I24`` and ``f32`` audio, otherwise ``float64``.
    #[pyo3(
        name = "iirt",
        signature = (*, win_length = Int(2048), hop_length = None, center = true, tuning = 0.0, pad_mode = "constant", flayout = "sos", res_type = "soxr_hq", **kwargs),
        text_signature = "(self, *, win_length=2048, hop_length=None, center=True, tuning=0.0, pad_mode='constant', flayout='sos', res_type='soxr_hq', **kwargs)"
    )]
    #[allow(clippy::too_many_arguments)]
    fn iirt_method<'py>(
        &self,
        py: Python<'py>,
        win_length: Int,
        hop_length: Option<Int>,
        center: bool,
        tuning: f64,
        pad_mode: &str,
        flayout: &str,
        res_type: &str,
        kwargs: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let cfg = iirt_config(
            win_length, hop_length, center, tuning, pad_mode, flayout, res_type, kwargs,
        )?;
        dispatch_typed!(self, py, |audio| audio.iirt(&cfg), |r| r
            .map_err(audio_err_to_py)
            .map(|a| a.into_pyarray(py).into_any()))
    }
}

fn parse_kind(kind: &str) -> PyResult<InterpKind> {
    Ok(match kind {
        "linear" => InterpKind::Linear,
        "nearest" => InterpKind::Nearest,
        "nearest-up" => InterpKind::NearestUp,
        "zero" => InterpKind::Zero,
        "slinear" => InterpKind::SLinear,
        "quadratic" => InterpKind::Quadratic,
        "cubic" => InterpKind::Cubic,
        "previous" => InterpKind::Previous,
        "next" => InterpKind::Next,
        other => return Err(value_err("kind", format!("unknown {other:?}"))),
    })
}

fn mellin_in<'py, F: PyFloat>(
    y: &Bound<'py, PyAny>,
    cfg: &MellinConfig,
    axis: Int,
) -> PyResult<Bound<'py, PyAny>>
where
    num_complex::Complex<F>: numpy::Element,
{
    let py = y.py();
    with_float_view::<F, _>(y, "y", |v| {
        let n = v.ndim() as i64;
        if axis.0 < -n || axis.0 >= n {
            return Err(value_err("axis", "is out of range"));
        }
        let ax = Axis(if axis.0 < 0 { axis.0 + n } else { axis.0 } as usize);
        let out = py
            .detach(|| mellin::mellin_transform(v, cfg, ax))
            .map_err(audio_err_to_py)?;
        Ok(out.into_pyarray(py).into_any())
    })
}

/// The fast Mellin (scale) transform along one axis.
///
/// Args:
///     y (``array_like``): Finite samples, at least 3 along ``axis``.
///     ``t_min`` (float): Where the exponential grid starts, in samples. Default 0.5.
///     ``n_fmt`` (Optional[int]): Grid points. Default: enough for the signal's resolution.
///     kind (str): The interpolant, as for ``arrays.interp1d``. Default ``"cubic"``.
///     beta (float): Exponent of the weighting; 0.5 is the scale transform.
///     ``over_sample`` (float): Oversampling of the grid. Default 1.
///     axis (int): The time axis. Default -1.
///
/// Returns:
///     numpy.ndarray: ``n_fmt // 2 + 1`` complex coefficients along ``axis``, ``complex64``
///     for ``float32`` input, otherwise ``complex128``.
#[pyfunction]
#[pyo3(
    signature = (y, *, t_min = 0.5, n_fmt = None, kind = "cubic", beta = 0.5, over_sample = 1.0, axis = Int(-1)),
    text_signature = "(y, *, t_min=0.5, n_fmt=None, kind='cubic', beta=0.5, over_sample=1.0, axis=-1)"
)]
#[inline]
pub fn mellin_transform<'py>(
    y: &Bound<'py, PyAny>,
    t_min: f64,
    n_fmt: Option<Int>,
    kind: &str,
    beta: f64,
    over_sample: f64,
    axis: Int,
) -> PyResult<Bound<'py, PyAny>> {
    let n_fmt = match n_fmt {
        None => None,
        Some(v) if v.0 < 0 => return Err(value_err("n_fmt", "must not be negative")),
        Some(v) => Some(v.0 as usize),
    };
    let cfg = MellinConfig::new()
        .with_t_min(t_min)
        .with_n_fmt(n_fmt)
        .with_kind(parse_kind(kind)?)
        .with_beta(beta)
        .with_over_sample(over_sample);
    with_precision!(precision_of(y), |F| mellin_in::<F>(y, &cfg, axis))
}

#[inline]
pub fn register(py: Python<'_>, parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(py, "pitch_filterbank")?;
    m.add(
        "__doc__",
        "Multirate semitone filterbank and the IIR time-frequency representation.",
    )?;
    m.add_function(wrap_pyfunction!(mr_frequencies, &m)?)?;
    m.add_function(wrap_pyfunction!(semitone_filterbank, &m)?)?;
    m.add_function(wrap_pyfunction!(iirt, &m)?)?;
    parent.add_submodule(&m)?;
    let spectral = parent.getattr("spectral")?.cast_into::<PyModule>()?;
    spectral.add_function(wrap_pyfunction!(mellin_transform, &spectral)?)?;
    Ok(())
}
