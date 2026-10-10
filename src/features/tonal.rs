//! Tonal free functions of `audio_samples.features` (binding
//! `audio_samples::operations::tonal`): `librosa.filters.chroma`,
//! `librosa.util.normalize`, `librosa.pitch_tuning` and the `S=` / `y=` array
//! forms of `chroma_stft`, `piptrack` and `estimate_tuning`.

use crate::args::Int;
use audio_samples::operations::filterbanks::Window;
use audio_samples::operations::spectrum::StftConfig;
use audio_samples::operations::tonal::{
    self as tonal, ChromaFilterbankConfig, ChromaStftConfig, NormalizeConfig, PiptrackConfig,
    PiptrackReference, TuningConfig,
};
use numpy::{IntoPyArray, PyArray2};
use pyo3::IntoPyObjectExt;
use pyo3::prelude::*;

use super::args::{parse_piptrack_ref, signal2, spec3, squeeze_out};
use crate::args::{
    WindowArg, float_array_as, positive, precision_of, stft_config, value_err, with_precision,
};
use crate::audio_err_to_py;

/// Infers `n_fft = 2 * (bins - 1)` when not given (librosa's convention).
fn infer_n_fft(n_fft: Option<Int>, n_bins: usize) -> PyResult<std::num::NonZeroUsize> {
    positive("n_fft", n_fft.unwrap_or(Int(2 * (n_bins as i64 - 1))))
}

/// The chroma filterbank, equivalent to
/// ``librosa.filters.chroma(sr=sr, n_fft=n_fft, ..., dtype=numpy.float64)``.
///
/// Args:
///     sr (float): Sample rate (Hz).
///     n_fft (int): FFT length.
///     n_chroma (int): Number of chroma bins. Default 12.
///     tuning (float): Tuning deviation from A440 in fractions of a chroma bin.
///     ctroct (float): Centre of the octave-dominance window (octaves above A0).
///         Default 5.0.
///     octwidth (Optional[float]): Half-width of the octave weighting; None = flat.
///         Default 2.0.
///     norm (Optional[float]): Column normalisation (p-norm, ``numpy.inf``), None
///         to disable. Default 2.
///     base_c (bool): Row 0 is C (True, default) or A.
///
/// Returns:
///     numpy.ndarray[float64]: shape ``(n_chroma, 1 + n_fft // 2)`` (librosa's
///     default dtype is float32; this is always float64).
///
/// Raises:
///     ValueError: For invalid parameters.
#[pyfunction]
#[pyo3(
    signature = (*, sr, n_fft, n_chroma = Int(12), tuning = 0.0, ctroct = 5.0, octwidth = Some(2.0), norm = Some(2.0), base_c = true),
    text_signature = "(*, sr, n_fft, n_chroma=12, tuning=0.0, ctroct=5.0, octwidth=2.0, norm=2.0, base_c=True)"
)]
#[allow(clippy::too_many_arguments)]
pub fn chroma_filterbank<'py>(
    py: Python<'py>,
    sr: f64,
    n_fft: Int,
    n_chroma: Int,
    tuning: f64,
    ctroct: f64,
    octwidth: Option<f64>,
    norm: Option<f64>,
    base_c: bool,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let cfg = ChromaFilterbankConfig {
        n_chroma: positive("n_chroma", n_chroma)?,
        tuning,
        ctroct,
        octwidth,
        norm,
        base_c,
    };
    let m =
        tonal::chroma_filterbank(sr, positive("n_fft", n_fft)?, &cfg).map_err(audio_err_to_py)?;
    Ok(m.into_pyarray(py))
}

/// Scales every slice of an array to unit norm, equivalent to
/// ``librosa.util.normalize``.
///
/// Args:
///     S (array_like): Array of any shape (finite values).
///     norm (Optional[float]): ``numpy.inf`` (max-abs, default), ``-numpy.inf``
///         (min-abs), 0 (count of non-zeros), ``p > 0`` (Lp), or None (no-op).
///     axis (Optional[int]): Axis along which norms are computed (default 0);
///         None normalises the whole array.
///     threshold (Optional[float]): Slices with a norm below this are "small"
///         (default: the smallest positive normal float64).
///     fill (Optional[bool]): Small slices: None leaves them, False zeroes them,
///         True fills them with the unit-norm constant.
///
/// Returns:
///     numpy.ndarray[float32 | float64]: same shape as ``S``.
///
/// Raises:
///     ValueError: For an invalid norm/threshold/axis, ``fill=True`` with
///         ``norm=0``, or non-finite input.
#[pyfunction]
#[pyo3(
    signature = (S, *, norm = Some(f64::INFINITY), axis = Some(Int(0)), threshold = None, fill = None),
    text_signature = "(S, *, norm=math.inf, axis=0, threshold=None, fill=None)"
)]
#[allow(non_snake_case)]
pub fn normalize<'py>(
    py: Python<'py>,
    S: &Bound<'py, PyAny>,
    norm: Option<f64>,
    axis: Option<Int>,
    threshold: Option<f64>,
    fill: Option<bool>,
) -> PyResult<Bound<'py, PyAny>> {
    let cfg = NormalizeConfig {
        norm,
        axis: axis.map(|a| crate::args::signed("axis", a)).transpose()?,
        threshold,
        fill,
    };
    with_precision!(precision_of(S), |F| {
        let s = float_array_as::<F>(S, "S")?;
        let r = tonal::normalize_array(&s.as_array(), &cfg).map_err(audio_err_to_py)?;
        Ok(r.into_pyarray(py).into_any())
    })
}

/// Tuning offset of a set of frequencies relative to A440, equivalent to
/// ``librosa.pitch_tuning``.
///
/// Args:
///     frequencies (array_like): Frequencies in Hz; non-positive values are ignored.
///     resolution (float): Histogram resolution in fractions of a bin, in ``(0, 1)``.
///         Default 0.01.
///     bins_per_octave (int): Bins per octave of the reference grid. Default 12.
///
/// Returns:
///     float: tuning in ``[-0.5, 0.5)`` fractions of a bin (0.0 for no input).
///
/// Raises:
///     ValueError: If ``resolution`` is not in ``(0, 1)``.
#[pyfunction]
#[pyo3(
    signature = (frequencies, *, resolution = 0.01, bins_per_octave = Int(12)),
    text_signature = "(frequencies, *, resolution=0.01, bins_per_octave=12)"
)]
pub fn pitch_tuning(
    frequencies: &Bound<'_, PyAny>,
    resolution: f64,
    bins_per_octave: Int,
) -> PyResult<f64> {
    let f = super::args::f64_vec(frequencies, "frequencies")?;
    tonal::pitch_tuning(
        &f,
        resolution,
        positive("bins_per_octave", bins_per_octave)?,
    )
    .map_err(audio_err_to_py)
}

/// Chromagram of a precomputed spectrogram, equivalent to
/// ``librosa.feature.chroma_stft(S=S, sr=sr, ...)``.
///
/// Args:
///     S (array_like): Power spectrogram (librosa's expectation; a magnitude
///         spectrogram gives "energy" chroma), shape ``(1 + n_fft // 2, T)`` or
///         ``(channels, 1 + n_fft // 2, T)``.
///     sr (float): Sample rate (Hz). Required (librosa defaults to 22050).
///     norm (Optional[float]): Per-frame normalisation (default ``numpy.inf``).
///     n_fft (Optional[int]): FFT length; None infers ``2 * (F - 1)``.
///     tuning (Optional[float]): Tuning in fractions of a chroma bin; None
///         estimates it from ``S`` (as librosa).
///     n_chroma (int): Number of chroma bins. Default 12.
///     ctroct, octwidth, base_c: Filterbank parameters (see ``chroma_filterbank``).
///
/// Returns:
///     numpy.ndarray[float32 | float64]: shape ``(n_chroma, T)`` or ``(channels, n_chroma, T)``.
///
/// Raises:
///     ValueError: For invalid parameters or a bin count that does not match ``n_fft``.
#[pyfunction]
#[pyo3(
    signature = (S, *, sr, norm = Some(f64::INFINITY), n_fft = None, tuning = Some(0.0), n_chroma = Int(12), ctroct = 5.0, octwidth = Some(2.0), base_c = true),
    text_signature = "(S, *, sr, norm=math.inf, n_fft=None, tuning=0.0, n_chroma=12, ctroct=5.0, octwidth=2.0, base_c=True)"
)]
#[allow(non_snake_case, clippy::too_many_arguments)]
pub fn chroma_stft_from_spectrogram<'py>(
    py: Python<'py>,
    S: &Bound<'py, PyAny>,
    sr: f64,
    norm: Option<f64>,
    n_fft: Option<Int>,
    tuning: Option<f64>,
    n_chroma: Int,
    ctroct: f64,
    octwidth: Option<f64>,
    base_c: bool,
) -> PyResult<Bound<'py, PyAny>> {
    with_precision!(precision_of(S), |F| {
        let (s, nd) = spec3::<F>(S, "S")?;
        let cfg = ChromaStftConfig {
            stft: StftConfig::default().with_n_fft(infer_n_fft(n_fft, s.dim().1)?),
            norm,
            tuning,
            n_chroma: positive("n_chroma", n_chroma)?,
            ctroct,
            octwidth,
            base_c,
        };
        let r = tonal::chroma_stft_from_spectrogram(s.view(), sr, &cfg).map_err(audio_err_to_py)?;
        Ok(squeeze_out(py, r.into_dyn(), nd == 2))
    })
}

fn piptrack_cfg(
    stft: StftConfig,
    fmin: f64,
    fmax: f64,
    threshold: f64,
    reference: Option<&Bound<'_, PyAny>>,
) -> PyResult<PiptrackConfig> {
    let reference = match reference {
        Some(r) => parse_piptrack_ref(r)?,
        None => PiptrackReference::FrameMax,
    };
    Ok(PiptrackConfig {
        stft,
        fmin,
        fmax,
        threshold,
        reference,
    })
}

/// Spectral peak tracking on a precomputed magnitude spectrogram, equivalent to
/// ``librosa.piptrack(S=S, sr=sr, ...)``.
///
/// Args:
///     S (array_like): Magnitude spectrogram, ``(1 + n_fft // 2, T)`` or
///         ``(channels, 1 + n_fft // 2, T)``.
///     sr (float): Sample rate (Hz). Required (librosa defaults to 22050).
///     n_fft (Optional[int]): FFT length; None infers ``2 * (F - 1)``.
///     fmin (float): Lowest reported frequency (Hz). Default 150.
///     fmax (float): Peaks must lie below this frequency (Hz). Default 4000.
///     threshold (float): Relative threshold. Default 0.1.
///     ref (None | float | numpy.max | numpy.mean): Threshold reference; None
///         (default) = per-frame maximum, a number = absolute threshold.
///
/// Returns:
///     tuple[numpy.ndarray[float32 | float64], numpy.ndarray[float32 | float64]]: ``(pitches,
///     magnitudes)``, both shaped like ``S``; zero except at detected peaks.
///
/// Raises:
///     ValueError: For invalid parameters or a bin count that does not match ``n_fft``.
#[pyfunction]
#[pyo3(
    signature = (S, *, sr, n_fft = None, fmin = 150.0, fmax = 4000.0, threshold = 0.1, r#ref = None),
    text_signature = "(S, *, sr, n_fft=None, fmin=150.0, fmax=4000.0, threshold=0.1, ref=None)"
)]
#[allow(non_snake_case, clippy::too_many_arguments)]
pub fn piptrack_from_spectrogram<'py>(
    py: Python<'py>,
    S: &Bound<'py, PyAny>,
    sr: f64,
    n_fft: Option<Int>,
    fmin: f64,
    fmax: f64,
    threshold: f64,
    r#ref: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    with_precision!(precision_of(S), |F| {
        let (s, nd) = spec3::<F>(S, "S")?;
        let stft = StftConfig::default().with_n_fft(infer_n_fft(n_fft, s.dim().1)?);
        let cfg = piptrack_cfg(stft, fmin, fmax, threshold, r#ref)?;
        let (p, m) =
            tonal::piptrack_from_spectrogram(s.view(), sr, &cfg).map_err(audio_err_to_py)?;
        (
            squeeze_out(py, p.into_dyn(), nd == 2),
            squeeze_out(py, m.into_dyn(), nd == 2),
        )
            .into_bound_py_any(py)
    })
}

/// Tuning estimate from a precomputed spectrogram, equivalent to
/// ``librosa.estimate_tuning(S=S, sr=sr, ...)``.
///
/// Args:
///     S (array_like): Magnitude or power spectrogram, ``(F, T)`` or
///         ``(channels, F, T)`` (all channels are pooled).
///     sr (float): Sample rate (Hz). Required (librosa defaults to 22050).
///     n_fft (Optional[int]): FFT length; None infers ``2 * (F - 1)``.
///     resolution (float): Resolution in fractions of a bin. Default 0.01.
///     bins_per_octave (int): Bins per octave of the grid. Default 12.
///     fmin, fmax, threshold, ref: ``piptrack`` parameters (librosa's ``**kwargs``).
///
/// Returns:
///     float: tuning in ``[-0.5, 0.5)`` fractions of a bin.
///
/// Raises:
///     ValueError: For invalid parameters.
#[pyfunction]
#[pyo3(
    signature = (S, *, sr, n_fft = None, resolution = 0.01, bins_per_octave = Int(12), fmin = 150.0, fmax = 4000.0, threshold = 0.1, r#ref = None),
    text_signature = "(S, *, sr, n_fft=None, resolution=0.01, bins_per_octave=12, fmin=150.0, fmax=4000.0, threshold=0.1, ref=None)"
)]
#[allow(non_snake_case, clippy::too_many_arguments)]
pub fn estimate_tuning_from_spectrogram(
    S: &Bound<'_, PyAny>,
    sr: f64,
    n_fft: Option<Int>,
    resolution: f64,
    bins_per_octave: Int,
    fmin: f64,
    fmax: f64,
    threshold: f64,
    r#ref: Option<&Bound<'_, PyAny>>,
) -> PyResult<f64> {
    with_precision!(precision_of(S), |F| {
        let (s, _) = spec3::<F>(S, "S")?;
        let stft = StftConfig::default().with_n_fft(infer_n_fft(n_fft, s.dim().1)?);
        let cfg = TuningConfig {
            piptrack: piptrack_cfg(stft, fmin, fmax, threshold, r#ref)?,
            resolution,
            bins_per_octave: positive("bins_per_octave", bins_per_octave)?,
        };
        tonal::estimate_tuning_from_spectrogram(s.view(), sr, &cfg).map_err(audio_err_to_py)
    })
}

/// Tuning estimate from a signal, equivalent to ``librosa.estimate_tuning(y=y, sr=sr, ...)``.
///
/// Args:
///     y (array_like): Signal, ``(n,)`` or ``(channels, n)`` (channels are pooled).
///     sr (float): Sample rate (Hz). Required (librosa defaults to 22050).
///     n_fft (int): FFT length. Default 2048.
///     resolution (float): Resolution in fractions of a bin. Default 0.01.
///     bins_per_octave (int): Bins per octave of the grid. Default 12.
///     hop_length, win_length, window, center, pad_mode: STFT parameters
///         (librosa's ``**kwargs`` to ``piptrack``).
///     fmin, fmax, threshold, ref: ``piptrack`` parameters.
///
/// Returns:
///     float: tuning in ``[-0.5, 0.5)`` fractions of a bin.
///
/// Raises:
///     ValueError: For invalid parameters or a signal too short for an uncentred STFT.
#[pyfunction]
#[pyo3(
    signature = (y, *, sr, n_fft = Int(2048), resolution = 0.01, bins_per_octave = Int(12), hop_length = None, win_length = None, window = WindowArg(Window::Hann), center = true, pad_mode = "constant", fmin = 150.0, fmax = 4000.0, threshold = 0.1, r#ref = None),
    text_signature = "(y, *, sr, n_fft=2048, resolution=0.01, bins_per_octave=12, hop_length=None, win_length=None, window='hann', center=True, pad_mode='constant', fmin=150.0, fmax=4000.0, threshold=0.1, ref=None)"
)]
#[allow(clippy::too_many_arguments)]
pub fn estimate_tuning_from_signal(
    y: &Bound<'_, PyAny>,
    sr: f64,
    n_fft: Int,
    resolution: f64,
    bins_per_octave: Int,
    hop_length: Option<Int>,
    win_length: Option<Int>,
    window: WindowArg,
    center: bool,
    pad_mode: &str,
    fmin: f64,
    fmax: f64,
    threshold: f64,
    r#ref: Option<&Bound<'_, PyAny>>,
) -> PyResult<f64> {
    with_precision!(precision_of(y), |F| {
        let (x, _) = signal2::<F>(y, "y")?;
        let stft = stft_config(n_fft, hop_length, win_length, window, center, pad_mode)?;
        let cfg = TuningConfig {
            piptrack: piptrack_cfg(stft, fmin, fmax, threshold, r#ref)?,
            resolution,
            bins_per_octave: positive("bins_per_octave", bins_per_octave)?,
        };
        if !(sr.is_finite() && sr > 0.0) {
            return Err(value_err("sr", format!("must be positive, got {sr}")));
        }
        tonal::estimate_tuning_from_signal(x.view(), sr, &cfg).map_err(audio_err_to_py)
    })
}

/// Registers the tonal functions.
pub(crate) fn add(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(chroma_filterbank, m)?)?;
    m.add_function(wrap_pyfunction!(normalize, m)?)?;
    m.add_function(wrap_pyfunction!(pitch_tuning, m)?)?;
    m.add_function(wrap_pyfunction!(chroma_stft_from_spectrogram, m)?)?;
    m.add_function(wrap_pyfunction!(piptrack_from_spectrogram, m)?)?;
    m.add_function(wrap_pyfunction!(estimate_tuning_from_spectrogram, m)?)?;
    m.add_function(wrap_pyfunction!(estimate_tuning_from_signal, m)?)?;
    Ok(())
}
