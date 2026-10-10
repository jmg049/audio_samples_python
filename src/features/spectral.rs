//! Spectral-descriptor free functions of `audio_samples.features` (binding
//! `audio_samples::operations::features`): the `S=` forms of librosa's
//! `feature.spectral_*`, `poly_features` and `rms`, `pcen`, the frequency
//! weighting curves and `perceptual_weighting`.

use crate::args::Int;
use audio_samples::operations::features::{
    self as feat, FrequencyWeighting, PcenConfig, PolyFeaturesConfig, SpectralBandwidthConfig,
    SpectralContrastConfig, SpectralFlatnessConfig, SpectralRolloffConfig,
};
use audio_samples::operations::spectrum::DbConfig;
use ndarray::{Array2, Array3, ArrayD, Axis, IxDyn};
use numpy::{IntoPyArray, PyUntypedArrayMethods};
use pyo3::IntoPyObjectExt;
use pyo3::prelude::*;

use super::args::{bin_frequencies, f64_vec, parse_weighting, spec3, squeeze_out};
use crate::args::{
    DbRefArg, FloatElement, f64_array, float_array_as, is_scalar, positive, precision_of,
    value_err, with_precision,
};
use crate::audio_err_to_py;

/// Wraps a `(channels, rows, T)` result in librosa's shape: `(rows, T)` for 2-D input.
fn out<'py, T: numpy::Element>(py: Python<'py>, a: Array3<T>, ndim_in: usize) -> Bound<'py, PyAny> {
    squeeze_out(py, a.into_dyn(), ndim_in == 2)
}

/// Spectral centroid of every frame of a spectrogram, equivalent to
/// ``librosa.feature.spectral_centroid(S=S, sr=sr, freq=freq)``.
///
/// Args:
///     S (array_like): Non-negative magnitude spectrogram, shape ``(F, T)`` or
///         ``(channels, F, T)``.
///     sr (float): Sample rate (Hz), used for the default ``freq``. Required (librosa defaults to 22050).
///     freq (Optional[array_like]): 1-D centre frequency of each bin (Hz).
///         ``None`` uses ``fft_frequencies(sr, n_fft=2 * (F - 1))``.
///
/// Returns:
///     numpy.ndarray[float32 | float64]: shape ``(1, T)`` or ``(channels, 1, T)``, in Hz.
///
/// Raises:
///     ValueError: If ``S`` has negative or non-finite values, or ``freq`` has
///         the wrong length. librosa also accepts time-varying 2-D ``freq``;
///         only 1-D is supported here.
#[pyfunction]
#[pyo3(signature = (S, *, sr, freq = None), text_signature = "(S, *, sr, freq=None)")]
#[allow(non_snake_case)]
pub fn spectral_centroid_from_spectrogram<'py>(
    py: Python<'py>,
    S: &Bound<'py, PyAny>,
    sr: f64,
    freq: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    with_precision!(precision_of(S), |F| {
        let (s, nd) = spec3::<F>(S, "S")?;
        let f = bin_frequencies(freq, sr, s.dim().1)?;
        let r = feat::spectral_centroid_from_spectrogram(s.view(), &f).map_err(audio_err_to_py)?;
        Ok(out(py, r, nd))
    })
}

/// Order-``p`` spectral bandwidth of every frame, equivalent to
/// ``librosa.feature.spectral_bandwidth(S=S, sr=sr, freq=freq, centroid=centroid, norm=norm, p=p)``.
///
/// Args:
///     S (array_like): Non-negative spectrogram, ``(F, T)`` or ``(channels, F, T)``.
///     sr (float): Sample rate for the default ``freq``. Required (librosa defaults to 22050).
///     freq (Optional[array_like]): 1-D bin frequencies; ``None`` = FFT bin centres.
///     centroid (Optional[array_like]): Precomputed centroid, shape ``(1, T)`` or
///         ``(channels, 1, T)``; ``None`` computes it.
///     norm (bool): Normalise each frame to unit sum first. Default True.
///     p (float): Order of the deviation. Default 2.
///
/// Returns:
///     numpy.ndarray[float32 | float64]: shape ``(1, T)`` or ``(channels, 1, T)``, in Hz.
///
/// Raises:
///     ValueError: For invalid spectrogram values or shapes, or ``p`` zero / not finite.
#[pyfunction]
#[pyo3(
    signature = (S, *, sr, freq = None, centroid = None, norm = true, p = 2.0),
    text_signature = "(S, *, sr, freq=None, centroid=None, norm=True, p=2.0)"
)]
#[allow(non_snake_case)]
pub fn spectral_bandwidth_from_spectrogram<'py>(
    py: Python<'py>,
    S: &Bound<'py, PyAny>,
    sr: f64,
    freq: Option<&Bound<'py, PyAny>>,
    centroid: Option<&Bound<'py, PyAny>>,
    norm: bool,
    p: f64,
) -> PyResult<Bound<'py, PyAny>> {
    with_precision!(precision_of(S), |F| {
        let (s, nd) = spec3::<F>(S, "S")?;
        let f = bin_frequencies(freq, sr, s.dim().1)?;
        let c = match centroid {
            Some(c) if !c.is_none() => Some(spec3(c, "centroid")?.0),
            _ => None,
        };
        let cfg = SpectralBandwidthConfig {
            p,
            norm,
            ..SpectralBandwidthConfig::default()
        };
        let r = feat::spectral_bandwidth_from_spectrogram(
            s.view(),
            &f,
            c.as_ref().map(|a| a.view()),
            &cfg,
        )
        .map_err(audio_err_to_py)?;
        Ok(out(py, r, nd))
    })
}

/// Roll-off frequency of every frame, equivalent to
/// ``librosa.feature.spectral_rolloff(S=S, sr=sr, freq=freq, roll_percent=roll_percent)``.
///
/// Args:
///     S (array_like): Non-negative spectrogram, ``(F, T)`` or ``(channels, F, T)``.
///     sr (float): Sample rate for the default ``freq``. Required (librosa defaults to 22050).
///     freq (Optional[array_like]): 1-D bin frequencies; ``None`` = FFT bin centres.
///     roll_percent (float): Energy fraction in ``(0, 1)``. Default 0.85.
///
/// Returns:
///     numpy.ndarray[float32 | float64]: shape ``(1, T)`` or ``(channels, 1, T)``, in Hz.
///
/// Raises:
///     ValueError: For invalid spectrogram values or ``roll_percent`` outside ``(0, 1)``.
#[pyfunction]
#[pyo3(
    signature = (S, *, sr, freq = None, roll_percent = 0.85),
    text_signature = "(S, *, sr, freq=None, roll_percent=0.85)"
)]
#[allow(non_snake_case)]
pub fn spectral_rolloff_from_spectrogram<'py>(
    py: Python<'py>,
    S: &Bound<'py, PyAny>,
    sr: f64,
    freq: Option<&Bound<'py, PyAny>>,
    roll_percent: f64,
) -> PyResult<Bound<'py, PyAny>> {
    with_precision!(precision_of(S), |F| {
        let (s, nd) = spec3::<F>(S, "S")?;
        let f = bin_frequencies(freq, sr, s.dim().1)?;
        let cfg = SpectralRolloffConfig {
            roll_percent,
            ..SpectralRolloffConfig::default()
        };
        let r =
            feat::spectral_rolloff_from_spectrogram(s.view(), &f, &cfg).map_err(audio_err_to_py)?;
        Ok(out(py, r, nd))
    })
}

/// Spectral flatness (Wiener entropy) of every frame, equivalent to
/// ``librosa.feature.spectral_flatness(S=S, amin=amin, power=power)``.
///
/// Args:
///     S (array_like): Non-negative magnitude spectrogram, ``(F, T)`` or
///         ``(channels, F, T)``.
///     amin (float): Floor applied to ``S ** power``. Default 1e-10.
///     power (float): Exponent applied to ``S``. Default 2.0.
///
/// Returns:
///     numpy.ndarray[float32 | float64]: shape ``(1, T)`` or ``(channels, 1, T)``, in ``[0, 1]``.
///
/// Raises:
///     ValueError: For invalid spectrogram values, ``amin <= 0`` or ``power <= 0``.
#[pyfunction]
#[pyo3(
    signature = (S, *, amin = 1e-10, power = 2.0),
    text_signature = "(S, *, amin=1e-10, power=2.0)"
)]
#[allow(non_snake_case)]
pub fn spectral_flatness_from_spectrogram<'py>(
    py: Python<'py>,
    S: &Bound<'py, PyAny>,
    amin: f64,
    power: f64,
) -> PyResult<Bound<'py, PyAny>> {
    with_precision!(precision_of(S), |F| {
        let (s, nd) = spec3::<F>(S, "S")?;
        let cfg = SpectralFlatnessConfig {
            amin,
            power,
            ..SpectralFlatnessConfig::default()
        };
        let r =
            feat::spectral_flatness_from_spectrogram(s.view(), &cfg).map_err(audio_err_to_py)?;
        Ok(out(py, r, nd))
    })
}

/// Octave-band spectral contrast of every frame, equivalent to
/// ``librosa.feature.spectral_contrast(S=S, sr=sr, freq=freq, fmin=..., n_bands=..., quantile=..., linear=...)``.
///
/// Args:
///     S (array_like): Non-negative magnitude spectrogram, ``(F, T)`` or
///         ``(channels, F, T)``.
///     sr (float): Sample rate (Hz). Required (librosa defaults to 22050).
///     freq (Optional[array_like]): 1-D bin frequencies; ``None`` = FFT bin centres.
///     fmin (float): Upper edge of the lowest band (Hz). Default 200.
///     n_bands (int): Number of octave bands above ``fmin``. Default 6.
///     quantile (float): Fraction of each band used for peak / valley. Default 0.02.
///     linear (bool): Linear difference instead of a dB ratio. Default False.
///
/// Returns:
///     numpy.ndarray[float32 | float64]: shape ``(n_bands + 1, T)`` or
///     ``(channels, n_bands + 1, T)``.
///
/// Raises:
///     ValueError: For invalid parameters (e.g. the top band above Nyquist).
#[pyfunction]
#[pyo3(
    signature = (S, *, sr, freq = None, fmin = 200.0, n_bands = Int(6), quantile = 0.02, linear = false),
    text_signature = "(S, *, sr, freq=None, fmin=200.0, n_bands=6, quantile=0.02, linear=False)"
)]
#[allow(non_snake_case, clippy::too_many_arguments)]
pub fn spectral_contrast_from_spectrogram<'py>(
    py: Python<'py>,
    S: &Bound<'py, PyAny>,
    sr: f64,
    freq: Option<&Bound<'py, PyAny>>,
    fmin: f64,
    n_bands: Int,
    quantile: f64,
    linear: bool,
) -> PyResult<Bound<'py, PyAny>> {
    with_precision!(precision_of(S), |F| {
        let (s, nd) = spec3::<F>(S, "S")?;
        let f = bin_frequencies(freq, sr, s.dim().1)?;
        let cfg = SpectralContrastConfig {
            fmin,
            n_bands: positive("n_bands", n_bands)?,
            quantile,
            linear,
            ..SpectralContrastConfig::default()
        };
        let r = feat::spectral_contrast_from_spectrogram(s.view(), &f, sr, &cfg)
            .map_err(audio_err_to_py)?;
        Ok(out(py, r, nd))
    })
}

/// Coefficients of a polynomial fit to every frame, equivalent to
/// ``librosa.feature.poly_features(S=S, sr=sr, freq=freq, order=order)``.
///
/// Args:
///     S (array_like): Spectrogram, ``(F, T)`` or ``(channels, F, T)``.
///     sr (float): Sample rate for the default ``freq``. Required (librosa defaults to 22050).
///     freq (Optional[array_like]): 1-D bin frequencies; ``None`` = FFT bin centres.
///     order (int): Polynomial order. Default 1.
///
/// Returns:
///     numpy.ndarray[float32 | float64]: shape ``(order + 1, T)`` or ``(channels, order + 1, T)``,
///     highest power first (as ``numpy.polyfit``).
///
/// Raises:
///     ValueError: If ``order`` is negative or not below the number of bins.
#[pyfunction]
#[pyo3(
    signature = (S, *, sr, freq = None, order = Int(1)),
    text_signature = "(S, *, sr, freq=None, order=1)"
)]
#[allow(non_snake_case)]
pub fn poly_features_from_spectrogram<'py>(
    py: Python<'py>,
    S: &Bound<'py, PyAny>,
    sr: f64,
    freq: Option<&Bound<'py, PyAny>>,
    order: Int,
) -> PyResult<Bound<'py, PyAny>> {
    with_precision!(precision_of(S), |F| {
        let (s, nd) = spec3::<F>(S, "S")?;
        let f = bin_frequencies(freq, sr, s.dim().1)?;
        let cfg = PolyFeaturesConfig {
            order: crate::args::non_negative("order", order)?,
            ..PolyFeaturesConfig::default()
        };
        let r =
            feat::poly_features_from_spectrogram(s.view(), &f, &cfg).map_err(audio_err_to_py)?;
        Ok(out(py, r, nd))
    })
}

/// Frame-wise RMS from STFT magnitudes (Parseval), equivalent to
/// ``librosa.feature.rms(S=S, frame_length=frame_length)``.
///
/// Args:
///     S (array_like): STFT magnitudes, ``(1 + frame_length // 2, T)`` or
///         ``(channels, 1 + frame_length // 2, T)``.
///     frame_length (int): The FFT length used to compute ``S``. Default 2048.
///
/// Returns:
///     numpy.ndarray[float32 | float64]: shape ``(1, T)`` or ``(channels, 1, T)``.
///     (librosa defaults to ``dtype=numpy.float32``.)
///
/// Raises:
///     ValueError: If the bin count is not ``1 + frame_length // 2``.
#[pyfunction]
#[pyo3(signature = (S, *, frame_length = Int(2048)), text_signature = "(S, *, frame_length=2048)")]
#[allow(non_snake_case)]
pub fn rms_from_spectrogram<'py>(
    py: Python<'py>,
    S: &Bound<'py, PyAny>,
    frame_length: Int,
) -> PyResult<Bound<'py, PyAny>> {
    with_precision!(precision_of(S), |F| {
        let (s, nd) = spec3::<F>(S, "S")?;
        let r = feat::rms_from_spectrogram(s.view(), positive("frame_length", frame_length)?)
            .map_err(audio_err_to_py)?;
        Ok(out(py, r, nd))
    })
}

/// Reshapes librosa's `zi` (shape `(..., F, 1)`, `(..., F)` or a scalar) to `(channels, F)`.
fn pcen_zi<F: FloatElement>(
    zi: &Bound<'_, PyAny>,
    channels: usize,
    bins: usize,
) -> PyResult<Array2<F>> {
    let a = float_array_as::<F>(zi, "zi")?;
    let v: Vec<F> = a.as_array().iter().copied().collect();
    if let [single] = v[..] {
        return Ok(Array2::from_elem((channels, bins), single));
    }
    if v.len() != channels * bins {
        return Err(value_err(
            "zi",
            format!(
                "expected {channels} x {bins} initial states (shape (..., F, 1)), got {} values",
                v.len()
            ),
        ));
    }
    Array2::from_shape_vec((channels, bins), v).map_err(|e| value_err("zi", e))
}

/// Per-channel energy normalisation, equivalent to ``librosa.pcen``.
///
/// Time is the last axis and frequency the one before it (librosa's defaults
/// ``axis=-1``; ``max_axis`` = the frequency axis).
///
/// Args:
///     S (array_like): Non-negative spectrogram, ``(F, T)`` or ``(channels, F, T)``
///         (typically a mel power spectrogram scaled to integer range).
///     sr (float): Sample rate (Hz). Required (librosa defaults to 22050).
///     hop_length (int): Hop of ``S`` in samples. Required (librosa defaults
///         to 512).
///     gain (float): Gain-normalisation exponent. Default 0.98.
///     bias (float): Bias added before compression. Default 2.
///     power (float): Compression exponent (0 selects ``log1p``). Default 0.5.
///     time_constant (float): Smoothing time constant (s). Default 0.4.
///     eps (float): Floor of the smoother. Default 1e-6.
///     b (Optional[float]): Explicit smoothing coefficient in ``[0, 1]``.
///     max_size (int): Width of a maximum filter along frequency applied to ``S``
///         before smoothing. Default 1 (off).
///     ref (Optional[array_like]): Array (shape of ``S``) to smooth instead of ``S``.
///     axis (int): Must be -1 (or ``S.ndim - 1``).
///     max_axis (Optional[int]): Must be None or the frequency axis (-2).
///     zi (Optional[array_like]): Initial filter state, shape ``(..., F, 1)``
///         (e.g. a previous ``zf``); None = steady state for a unit input.
///     return_zf (bool): Also return the final filter state. Default False.
///
/// Returns:
///     numpy.ndarray[float32 | float64] | tuple: the PCEN spectrogram (shape of ``S``), and
///     with ``return_zf=True`` also ``zf`` of shape ``(..., F, 1)``.
///
/// Raises:
///     ValueError: For invalid parameters, unsupported ``axis`` / ``max_axis`` or
///         shape mismatches. Complex ``S`` raises TypeError (librosa takes the
///         magnitude with a warning; pass ``numpy.abs(S)``).
#[pyfunction]
#[pyo3(
    signature = (S, *, sr, hop_length, gain = 0.98, bias = 2.0, power = 0.5, time_constant = 0.4, eps = 1e-6, b = None, max_size = Int(1), r#ref = None, axis = Int(-1), max_axis = None, zi = None, return_zf = false),
    text_signature = "(S, *, sr, hop_length, gain=0.98, bias=2.0, power=0.5, time_constant=0.4, eps=1e-06, b=None, max_size=1, ref=None, axis=-1, max_axis=None, zi=None, return_zf=False)"
)]
#[allow(non_snake_case, clippy::too_many_arguments)]
pub fn pcen<'py>(
    py: Python<'py>,
    S: &Bound<'py, PyAny>,
    sr: f64,
    hop_length: Int,
    gain: f64,
    bias: f64,
    power: f64,
    time_constant: f64,
    eps: f64,
    b: Option<f64>,
    max_size: Int,
    r#ref: Option<&Bound<'py, PyAny>>,
    axis: Int,
    max_axis: Option<Int>,
    zi: Option<&Bound<'py, PyAny>>,
    return_zf: bool,
) -> PyResult<Bound<'py, PyAny>> {
    with_precision!(precision_of(S), |F| {
        let (s, nd) = spec3::<F>(S, "S")?;
        let nd_i = nd as i64;
        if axis != Int(-1) && axis != Int(nd_i - 1) {
            return Err(value_err(
                "axis",
                format!("only the last axis (-1) is supported as the time axis, got {axis}"),
            ));
        }
        if let Some(m) = max_axis
            && m != Int(-2)
            && m != Int(nd_i - 2)
        {
            return Err(value_err(
                "max_axis",
                format!("only the frequency axis (-2) is supported, got {m}"),
            ));
        }
        let cfg = PcenConfig {
            hop_length: positive("hop_length", hop_length)?,
            gain,
            bias,
            power,
            time_constant,
            eps,
            b,
            max_size: positive("max_size", max_size)?,
        };
        let reference = match r#ref {
            Some(r) if !r.is_none() => {
                let (r3, rnd) = spec3(r, "ref")?;
                if rnd != nd || r3.dim() != s.dim() {
                    return Err(value_err("ref", "must have the same shape as S"));
                }
                Some(r3)
            }
            _ => None,
        };
        let (c, f, _) = s.dim();
        let zi = match zi {
            Some(z) if !z.is_none() => Some(pcen_zi(z, c, f)?),
            _ => None,
        };
        let (p, zf) = feat::pcen_with_state(
            s.view(),
            sr,
            &cfg,
            reference.as_ref().map(|a| a.view()),
            zi.as_ref().map(|a| a.view()),
        )
        .map_err(audio_err_to_py)?;
        let p_out = out(py, p, nd);
        if !return_zf {
            return Ok(p_out);
        }
        let zf3 = zf.insert_axis(Axis(2));
        let zf_out = squeeze_out(py, zf3.into_dyn(), nd == 2);
        (p_out, zf_out).into_bound_py_any(py)
    })
}

/// Applies a weighting function to an array-like of frequencies, librosa style:
/// a scalar gives a float, an array an array of the same shape.
fn weighting_map<'py>(
    frequencies: &Bound<'py, PyAny>,
    f: impl Fn(&[f64]) -> Vec<f64>,
) -> PyResult<Bound<'py, PyAny>> {
    let py = frequencies.py();
    let a = f64_array(frequencies, "frequencies")?;
    let shape = a.shape().to_vec();
    let vals: Vec<f64> = a.as_array().iter().copied().collect();
    let w = f(&vals);
    if is_scalar(frequencies)?
        && let Some(&v) = w.first()
    {
        return v.into_bound_py_any(py);
    }
    let arr = ArrayD::from_shape_vec(IxDyn(&shape), w).map_err(|e| value_err("frequencies", e))?;
    Ok(arr.into_pyarray(py).into_any())
}

macro_rules! weighting_fn {
    ($name:ident, $rust:ident, $letter:literal) => {
        #[doc = concat!(
            "The ", $letter, "-weighting of a set of frequencies in dB, equivalent to ``librosa.",
            $letter, "_weighting(frequencies, min_db=min_db)``.\n\n",
            "Args:\n",
            "    frequencies (float | array_like): Frequencies in Hz.\n",
            "    min_db (Optional[float]): Lower clip in dB (default -80); None disables it.\n\n",
            "Returns:\n",
            "    float | numpy.ndarray[float64]: dB weights, same shape as ``frequencies``\n",
            "    (a float for scalar input)."
        )]
        #[pyfunction]
        #[pyo3(signature = (frequencies, *, min_db = Some(-80.0)), text_signature = "(frequencies, *, min_db=-80.0)")]
        pub fn $name<'py>(
            frequencies: &Bound<'py, PyAny>,
            min_db: Option<f64>,
        ) -> PyResult<Bound<'py, PyAny>> {
            weighting_map(frequencies, |f| feat::$rust(f, min_db))
        }
    };
}

weighting_fn!(a_weighting, a_weighting, "A");
weighting_fn!(b_weighting, b_weighting, "B");
weighting_fn!(c_weighting, c_weighting, "C");
weighting_fn!(d_weighting, d_weighting, "D");
weighting_fn!(z_weighting, z_weighting, "Z");

/// A frequency-weighting curve chosen by name, equivalent to
/// ``librosa.frequency_weighting(frequencies, kind=kind, min_db=min_db)``.
///
/// Args:
///     frequencies (float | array_like): Frequencies in Hz.
///     kind (str): ``"A"`` (default), ``"B"``, ``"C"``, ``"D"`` or ``"Z"``.
///     min_db (Optional[float]): Lower clip in dB. Default None (no clip);
///         librosa clips at -80 dB.
///
/// Returns:
///     float | numpy.ndarray[float64]: dB weights, same shape as ``frequencies``.
///
/// Raises:
///     ValueError: For an unknown ``kind``.
#[pyfunction]
#[pyo3(
    signature = (frequencies, *, kind = "A", min_db = None),
    text_signature = "(frequencies, *, kind='A', min_db=None)"
)]
pub fn frequency_weighting<'py>(
    frequencies: &Bound<'py, PyAny>,
    kind: &str,
    min_db: Option<f64>,
) -> PyResult<Bound<'py, PyAny>> {
    let k = parse_weighting("kind", kind)?;
    weighting_map(frequencies, |f| feat::frequency_weighting(f, k, min_db))
}

fn parse_kinds(kinds: &Bound<'_, PyAny>) -> PyResult<Vec<FrequencyWeighting>> {
    if let Ok(s) = kinds.extract::<String>() {
        return s
            .chars()
            .map(|c| parse_weighting("kinds", &c.to_string()))
            .collect();
    }
    let items: Vec<String> = kinds.extract().map_err(|_| {
        value_err(
            "kinds",
            "expected a string such as 'ZAC' or a sequence of names",
        )
    })?;
    items.iter().map(|s| parse_weighting("kinds", s)).collect()
}

/// Several weighting curves stacked, equivalent to
/// ``librosa.multi_frequency_weighting(frequencies, kinds=kinds, min_db=min_db)``.
///
/// Args:
///     frequencies (array_like): Frequencies in Hz.
///     kinds (str | Sequence[str]): Curve names, e.g. ``"ZAC"`` (default).
///     min_db (Optional[float]): Lower clip in dB. Default None (no clip);
///         librosa clips at -80 dB.
///
/// Returns:
///     numpy.ndarray[float64]: shape ``(len(kinds), *frequencies.shape)``.
///
/// Raises:
///     ValueError: For an unknown curve name.
#[pyfunction]
#[pyo3(
    signature = (frequencies, *, kinds = None, min_db = None),
    text_signature = "(frequencies, *, kinds='ZAC', min_db=None)"
)]
pub fn multi_frequency_weighting<'py>(
    py: Python<'py>,
    frequencies: &Bound<'py, PyAny>,
    kinds: Option<&Bound<'py, PyAny>>,
    min_db: Option<f64>,
) -> PyResult<Bound<'py, PyAny>> {
    let ks = match kinds {
        Some(k) if !k.is_none() => parse_kinds(k)?,
        _ => vec![
            FrequencyWeighting::Z,
            FrequencyWeighting::A,
            FrequencyWeighting::C,
        ],
    };
    let a = f64_array(frequencies, "frequencies")?;
    let mut shape = vec![ks.len()];
    shape.extend_from_slice(a.shape());
    let vals = f64_vec(frequencies, "frequencies")?;
    let m = feat::multi_frequency_weighting(&vals, &ks, min_db);
    let flat: Vec<f64> = m.iter().copied().collect();
    let arr =
        ArrayD::from_shape_vec(IxDyn(&shape), flat).map_err(|e| value_err("frequencies", e))?;
    Ok(arr.into_pyarray(py).into_any())
}

/// Perceptually weighted dB spectrogram, equivalent to
/// ``librosa.perceptual_weighting(S, frequencies, kind=kind, ref=ref, amin=amin, top_db=top_db)``.
///
/// Computes ``power_to_db(S, ref, amin, top_db) + weighting(frequencies)``.
///
/// Args:
///     S (array_like): Power spectrogram, ``(F, T)`` or ``(channels, F, T)``.
///     frequencies (array_like): 1-D centre frequency of each bin (Hz).
///     kind (str): Weighting curve, ``"A"`` (default) ... ``"Z"``.
///     ref (float | "max" | numpy.max): dB reference. Default 1.0.
///     amin (float): Floor before the logarithm. Default 1e-10.
///     top_db (Optional[float]): Dynamic range below the peak in dB (per
///         channel). Default None (no floor); librosa's default is 80.
///
/// Returns:
///     numpy.ndarray[float32 | float64]: same shape as ``S``, in dB.
///
/// Raises:
///     ValueError: For shape mismatches or invalid dB parameters.
#[pyfunction]
#[pyo3(
    signature = (S, frequencies, *, kind = "A", r#ref = DbRefArg(audio_samples::DbReference::Value(1.0)), amin = 1e-10, top_db = None),
    text_signature = "(S, frequencies, *, kind='A', ref=1.0, amin=1e-10, top_db=None)"
)]
#[allow(non_snake_case)]
pub fn perceptual_weighting<'py>(
    py: Python<'py>,
    S: &Bound<'py, PyAny>,
    frequencies: &Bound<'py, PyAny>,
    kind: &str,
    r#ref: DbRefArg,
    amin: f64,
    top_db: Option<f64>,
) -> PyResult<Bound<'py, PyAny>> {
    with_precision!(precision_of(S), |F| {
        let (s, nd) = spec3::<F>(S, "S")?;
        let f = super::args::f64_vec1(frequencies, "frequencies")?;
        let k = parse_weighting("kind", kind)?;
        let db = DbConfig::power()
            .with_reference(r#ref.0)
            .with_amin(amin)
            .with_top_db(top_db);
        let r = feat::perceptual_weighting(s.view(), &f, k, &db).map_err(audio_err_to_py)?;
        Ok(out(py, r, nd))
    })
}

/// Registers the spectral-descriptor functions.
pub(crate) fn add(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(spectral_centroid_from_spectrogram, m)?)?;
    m.add_function(wrap_pyfunction!(spectral_bandwidth_from_spectrogram, m)?)?;
    m.add_function(wrap_pyfunction!(spectral_rolloff_from_spectrogram, m)?)?;
    m.add_function(wrap_pyfunction!(spectral_flatness_from_spectrogram, m)?)?;
    m.add_function(wrap_pyfunction!(spectral_contrast_from_spectrogram, m)?)?;
    m.add_function(wrap_pyfunction!(poly_features_from_spectrogram, m)?)?;
    m.add_function(wrap_pyfunction!(rms_from_spectrogram, m)?)?;
    m.add_function(wrap_pyfunction!(pcen, m)?)?;
    m.add_function(wrap_pyfunction!(a_weighting, m)?)?;
    m.add_function(wrap_pyfunction!(b_weighting, m)?)?;
    m.add_function(wrap_pyfunction!(c_weighting, m)?)?;
    m.add_function(wrap_pyfunction!(d_weighting, m)?)?;
    m.add_function(wrap_pyfunction!(z_weighting, m)?)?;
    m.add_function(wrap_pyfunction!(frequency_weighting, m)?)?;
    m.add_function(wrap_pyfunction!(multi_frequency_weighting, m)?)?;
    m.add_function(wrap_pyfunction!(perceptual_weighting, m)?)?;
    Ok(())
}
