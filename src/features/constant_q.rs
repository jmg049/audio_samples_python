//! Constant-Q free functions of `audio_samples.features` (binding
//! `audio_samples::operations::constant_q`): `librosa.filters.cq_to_chroma`,
//! `filters.wavelet` / `wavelet_lengths`, and the chroma-input forms of
//! `feature.chroma_cens` / `feature.tonnetz`.

use crate::args::Int;
use audio_samples::operations::constant_q::{self as cq, WaveletConfig};
use audio_samples::operations::filterbanks::Window;
use num_complex::Complex;
use numpy::{IntoPyArray, PyArray1, PyArray2};
use pyo3::prelude::*;

use super::args::{spec3, squeeze_out};
use crate::args::{WindowArg, non_negative, opt_positive, positive, precision_of, with_precision};
use crate::audio_err_to_py;

/// The CQT-to-chroma folding matrix, equivalent to
/// ``librosa.filters.cq_to_chroma(n_input, ..., dtype=numpy.float64)``.
///
/// Args:
///
///     n_input (int): Number of CQT bins.
///     bins_per_octave (int): CQT bins per octave (a multiple of ``n_chroma``). Default 12.
///     n_chroma (int): Number of chroma bins. Default 12.
///     fmin (Optional[float]): Frequency of CQT bin 0 (Hz); None = C1.
///     window (Optional[``array_like``]): Smooths each row by convolution with these weights.
///     base_c (bool): Chroma row 0 is C (default) or A.
///
/// Returns:
///
///     numpy.ndarray[float64]: shape ``(n_chroma, n_input)`` of zeros and ones.
///
/// Raises:
///
///     ``ValueError``: For invalid parameters.
#[pyfunction]
#[pyo3(
    signature = (n_input, *, bins_per_octave = Int(12), n_chroma = Int(12), fmin = None, window = None, base_c = true),
    text_signature = "(n_input, *, bins_per_octave=12, n_chroma=12, fmin=None, window=None, base_c=True)"
)]
pub fn cq_to_chroma<'py>(
    py: Python<'py>,
    n_input: Int,
    bins_per_octave: Int,
    n_chroma: Int,
    fmin: Option<f64>,
    window: Option<&Bound<'py, PyAny>>,
    base_c: bool,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let smoothing = match window.filter(|w| !w.is_none()) {
        Some(w) => Some(super::args::f64_vec(w, "window")?),
        None => None,
    };
    let mut m = cq::cq_to_chroma(
        non_negative("n_input", n_input)?,
        positive("bins_per_octave", bins_per_octave)?,
        positive("n_chroma", n_chroma)?,
        fmin,
        base_c,
    )
    .map_err(audio_err_to_py)?;
    if let Some(w) = smoothing.filter(|w| !w.is_empty()) {
        cq::smooth_chroma_map(&mut m, &w);
    }
    Ok(m.into_pyarray(py))
}

/// One relative bandwidth per frequency, from one value or a sequence.
fn alpha_per_frequency(alpha: &Bound<'_, PyAny>, n: usize) -> PyResult<Vec<f64>> {
    if let Ok(a) = alpha.extract::<f64>() {
        return Ok(vec![a; n]);
    }
    super::args::f64_vec1(alpha, "alpha")
}

fn wavelet_config(
    window: WindowArg,
    filter_scale: f64,
    gamma: Option<f64>,
    norm: Option<f64>,
    pad_fft: bool,
) -> WaveletConfig {
    WaveletConfig {
        window: window.0,
        filter_scale,
        gamma,
        norm,
        pad_fft,
    }
}

/// Fractional lengths of constant-Q / variable-Q wavelet filters, equivalent to
/// ``librosa.filters.wavelet_lengths(freqs=freqs, sr=sr, ...)``.
///
/// Args:
///
///     freqs (array_like): Centre frequencies in Hz (at least two, ascending).
///     sr (float): Sample rate. Required (librosa defaults to 22050).
///     window: Window (named; see ``spectral.get_window``). Default ``"hann"``.
///     filter_scale (float): Filter-length scale. Default 1.
///     gamma (Optional[float]): Bandwidth offset (Hz); 0 (default) = constant Q,
///         None = ERB-derived.
///     alpha (Optional[float | ``array_like``]): Relative bandwidth, one value or one per
///         frequency. None derives it from the spacing of the frequencies.
///
/// Returns:
///
///     tuple[numpy.ndarray[float64], float]: ``(lengths, cutoff)``: the length of
///     each filter in samples and the highest cut-off frequency in Hz.
///
/// Raises:
///
///     ``ValueError``: For invalid frequencies or parameters.
#[pyfunction]
#[pyo3(
    signature = (*, freqs, sr, window = WindowArg(Window::Hann), filter_scale = 1.0, gamma = Some(0.0), alpha = None),
    text_signature = "(*, freqs, sr, window='hann', filter_scale=1.0, gamma=0.0, alpha=None)"
)]
pub fn wavelet_lengths<'py>(
    py: Python<'py>,
    freqs: &Bound<'py, PyAny>,
    sr: f64,
    window: WindowArg,
    filter_scale: f64,
    gamma: Option<f64>,
    alpha: Option<&Bound<'py, PyAny>>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, f64)> {
    use numpy::PyArrayMethods;
    let cfg = wavelet_config(window, filter_scale, gamma, Some(1.0), true);
    // A contiguous 1-D float64 array is read in place; the lengths are
    // written straight into a NumPy-allocated array.
    crate::args::with_float_view::<f64, _>(freqs, "freqs", |v| {
        if v.ndim() != 1 {
            return Err(crate::args::value_err(
                "freqs",
                format!("expected a 1-D array, got {} dimensions", v.ndim()),
            ));
        }
        let copied;
        let f = if let Some(s) = v.as_slice() {
            s
        } else {
            copied = v.iter().copied().collect::<Vec<f64>>();
            &copied
        };
        let out = PyArray1::<f64>::zeros(py, f.len(), false);
        // SAFETY: the array was just created and is not shared yet.
        let lengths = unsafe { out.as_slice_mut() }.expect("a new array is contiguous");
        let cutoff = match alpha.filter(|a| !a.is_none()) {
            None => cq::wavelet_lengths_into(f, sr, &cfg, lengths).map_err(audio_err_to_py)?,
            Some(a) if a.extract::<f64>().is_ok() => {
                let bandwidths = alpha_per_frequency(a, f.len())?;
                cq::wavelet_lengths_with_alpha_into(f, sr, &cfg, &bandwidths, lengths)
                    .map_err(audio_err_to_py)?
            }
            Some(a) => crate::args::with_float_view::<f64, _>(a, "alpha", |b| {
                if b.ndim() != 1 {
                    return Err(crate::args::value_err(
                        "alpha",
                        format!("expected a 1-D array, got {} dimensions", b.ndim()),
                    ));
                }
                let copied;
                let bandwidths = if let Some(s) = b.as_slice() {
                    s
                } else {
                    copied = b.iter().copied().collect::<Vec<f64>>();
                    &copied
                };
                cq::wavelet_lengths_with_alpha_into(f, sr, &cfg, bandwidths, lengths)
                    .map_err(audio_err_to_py)
            })?,
        };
        Ok((out, cutoff))
    })
}

/// Constant-Q / variable-Q wavelet filters, equivalent to
/// ``librosa.filters.wavelet(freqs=freqs, sr=sr, ..., dtype=numpy.complex128)``.
///
/// Args:
///
///     freqs (array_like): Centre frequencies in Hz (at least two, ascending).
///     sr (float): Sample rate. Required (librosa defaults to 22050).
///     window: Named window. Default ``"hann"``.
///     filter_scale (float): Filter-length scale. Default 1.
///     pad_fft (bool): Centre the filters in a power-of-two buffer. Default True.
///     norm (Optional[float]): Per-filter normalisation (p-norm). Default 1.
///     gamma (Optional[float]): Bandwidth offset (Hz). Default 0.
///     alpha (Optional[float | ``array_like``]): As ``wavelet_lengths``.
///
/// Returns:
///
///     tuple[numpy.ndarray[complex128], numpy.ndarray[float64]]: ``(filters,
///     lengths)``, ``filters`` of shape ``(len(freqs), width)`` (librosa's default
///     dtype is complex64; this is complex128).
///
/// Raises:
///
///     ValueError: For invalid frequencies or parameters.
#[pyfunction]
#[pyo3(
    signature = (*, freqs, sr, window = WindowArg(Window::Hann), filter_scale = 1.0, pad_fft = true, norm = Some(1.0), gamma = Some(0.0), alpha = None),
    text_signature = "(*, freqs, sr, window='hann', filter_scale=1.0, pad_fft=True, norm=1.0, gamma=0.0, alpha=None)"
)]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn wavelet<'py>(
    py: Python<'py>,
    freqs: &Bound<'py, PyAny>,
    sr: f64,
    window: WindowArg,
    filter_scale: f64,
    pad_fft: bool,
    norm: Option<f64>,
    gamma: Option<f64>,
    alpha: Option<&Bound<'py, PyAny>>,
) -> PyResult<(
    Bound<'py, PyArray2<Complex<f64>>>,
    Bound<'py, PyArray1<f64>>,
)> {
    let f = super::args::f64_vec1(freqs, "freqs")?;
    let cfg = wavelet_config(window, filter_scale, gamma, norm, pad_fft);
    let (filters, lengths) = match alpha.filter(|a| !a.is_none()) {
        Some(a) => {
            let bandwidths = alpha_per_frequency(a, f.len())?;
            cq::wavelet_with_alpha(&f, sr, &cfg, &bandwidths)
        }
        None => cq::wavelet(&f, sr, &cfg),
    }
    .map_err(audio_err_to_py)?;
    Ok((filters.into_pyarray(py), lengths.into_pyarray(py)))
}

/// Chroma Energy Normalized Statistics (CENS) from a chromagram, equivalent to
/// the post-processing of ``librosa.feature.chroma_cens``.
///
/// Args:
///
///     chroma (array_like): Chromagram, ``(n_chroma, T)`` or ``(channels, n_chroma, T)``.
///     norm (Optional[float]): Final per-frame normalisation. Default 2.
///     win_len_smooth (Optional[int]): Smoothing length in frames; None disables.
///         Default 41.
///     smoothing_window: Smoothing window. Default ``"hann"``.
///
/// Returns:
///
///     numpy.ndarray[float32 | float64]: same shape as ``chroma``.
///
/// Raises:
///
///     ValueError: For invalid parameters; RuntimeError for non-finite input.
#[pyfunction]
#[pyo3(
    signature = (chroma, *, norm = Some(2.0), win_len_smooth = Some(Int(41)), smoothing_window = WindowArg(Window::Hann)),
    text_signature = "(chroma, *, norm=2.0, win_len_smooth=41, smoothing_window='hann')"
)]
#[allow(
    clippy::needless_pass_by_value,
    reason = "PyO3 passes extracted argument types by value"
)]
pub fn chroma_cens_from_chroma<'py>(
    py: Python<'py>,
    chroma: &Bound<'py, PyAny>,
    norm: Option<f64>,
    win_len_smooth: Option<Int>,
    smoothing_window: WindowArg,
) -> PyResult<Bound<'py, PyAny>> {
    with_precision!(precision_of(chroma), |F| {
        let (c, nd) = spec3::<F>(chroma, "chroma")?;
        let r = cq::chroma_cens_from_chroma(
            &c.view(),
            norm,
            opt_positive("win_len_smooth", win_len_smooth)?,
            &smoothing_window.0,
        )
        .map_err(audio_err_to_py)?;
        Ok(squeeze_out(py, r.into_dyn(), nd == 2))
    })
}

/// Tonal centroid features from a chromagram, equivalent to
/// ``librosa.feature.tonnetz(chroma=chroma)``.
///
/// Args:
///
///     chroma (array_like): Chromagram, ``(n_chroma, T)`` or ``(channels, n_chroma, T)``.
///
/// Returns:
///
///     numpy.ndarray[float32 | float64]: shape ``(6, T)`` or ``(channels, 6, T)``.
///
/// Raises:
///
///     RuntimeError: For non-finite input.
#[pyfunction]
#[pyo3(signature = (chroma), text_signature = "(chroma)")]
pub fn tonnetz_from_chroma<'py>(
    py: Python<'py>,
    chroma: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    with_precision!(precision_of(chroma), |F| {
        let (c, nd) = spec3::<F>(chroma, "chroma")?;
        let r = cq::tonnetz_from_chroma(&c.view()).map_err(audio_err_to_py)?;
        Ok(squeeze_out(py, r.into_dyn(), nd == 2))
    })
}

/// Registers the constant-Q functions.
pub fn add(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(cq_to_chroma, m)?)?;
    m.add_function(wrap_pyfunction!(wavelet_lengths, m)?)?;
    m.add_function(wrap_pyfunction!(wavelet, m)?)?;
    m.add_function(wrap_pyfunction!(chroma_cens_from_chroma, m)?)?;
    m.add_function(wrap_pyfunction!(tonnetz_from_chroma, m)?)?;
    Ok(())
}
