//! `audio_samples.spectral`: spectral functions over NumPy arrays.
//!
//! Free functions for the STFT and its inverse (`stft`, `istft`), dB
//! conversions, windows and filterbanks (`mel_filterbank`, `get_window`,
//! `window_sumsquare`, `window_bandwidth`), `delta`, `stack_memory`, mel
//! projection and MFCCs of a spectrogram and signal framing (`frame`), binding
//! `audio_samples::operations::{spectrum, filterbanks}`. A 1-D signal gives a
//! 2-D STFT, a `(channels, n)` signal gives `(channels, F, T)`.

use crate::args::Int;
use audio_samples::operations::filterbanks::{self, Window};
use audio_samples::operations::spectrum::{self, DbConfig, DeltaConfig, DeltaMode};
use ndarray::{ArrayD, Axis, Ix1, Ix2, Ix3};
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyArrayDescrMethods, PyArrayMethods, PyUntypedArrayMethods,
};
use pyo3::IntoPyObjectExt;
use pyo3::prelude::*;

use crate::args::{
    DbAxesArg, DbRefArg, MelNormArg, WindowArg, complex_array_as, float_array_as, is_scalar,
    mel_config, non_negative, opt_positive, positive, precision_of, stft_config, value_err,
    with_precision,
};
use crate::audio_err_to_py;
use crate::operations::spectrum::mfcc_cepstrum_config;

/// Returns a 0-d result as a Python float and anything else as an array.
fn dyn_out<'py, F>(py: Python<'py>, a: ArrayD<F>, scalar: bool) -> PyResult<Bound<'py, PyAny>>
where
    F: numpy::Element + Copy + for<'a> IntoPyObject<'a>,
{
    if scalar {
        if let Some(&v) = a.iter().next() {
            return v.into_bound_py_any(py);
        }
    }
    Ok(a.into_pyarray(py).into_any())
}

/// Short-time Fourier transform.
///
/// Args:
///
///     y (numpy.ndarray): Real signal, shape ``(n,)`` or ``(channels, n)``.
///     n_fft (int): FFT length. Default 2048.
///     hop_length (Optional[int]): ``None`` means ``win_length // 4``.
///     win_length (Optional[int]): ``None`` means ``n_fft``.
///     window: Window name, ``(name, param)`` tuple or coefficient array
///         (see ``get_window``). Default ``"hann"``.
///     center (bool): Centre frames (pad ``n_fft // 2`` each side). Default True.
///     pad_mode (str): ``"constant"`` (default), ``"reflect"``, ``"edge"``, ``"symmetric"``.
///
/// Returns:
///
///     numpy.ndarray[complex64 | complex128]: shape ``(1 + n_fft // 2, n_frames)`` for 1-D
///     input, ``(channels, 1 + n_fft // 2, n_frames)`` for 2-D input.
///
/// Raises:
///
///     ValueError: For invalid parameters or input with more than 2 dimensions.
#[pyfunction]
#[pyo3(
    signature = (y, *, n_fft = Int(2048), hop_length = None, win_length = None, window = WindowArg(Window::Hann), center = true, pad_mode = "constant"),
    text_signature = "(y, *, n_fft=2048, hop_length=None, win_length=None, window='hann', center=True, pad_mode='constant')"
)]
#[allow(
    clippy::too_many_arguments,
    reason = "the Python signature takes these arguments"
)] // PyO3 signature mirrors the Python keyword arguments
pub fn stft<'py>(
    py: Python<'py>,
    y: &Bound<'py, PyAny>,
    n_fft: Int,
    hop_length: Option<Int>,
    win_length: Option<Int>,
    window: WindowArg,
    center: bool,
    pad_mode: &str,
) -> PyResult<Bound<'py, PyAny>> {
    let cfg = stft_config(n_fft, hop_length, win_length, window, center, pad_mode)?;
    with_precision!(precision_of(y), |F| {
        let y = float_array_as::<F>(y, "y")?;
        let y = y.as_array();
        match y.ndim() {
            1 => {
                let y2 = y
                    .into_dimensionality::<Ix1>()
                    .expect("1-D")
                    .insert_axis(Axis(0));
                let s = py
                    .detach(|| spectrum::stft(y2, &cfg))
                    .map_err(audio_err_to_py)?;
                Ok(s.index_axis_move(Axis(0), 0).into_pyarray(py).into_any())
            }
            2 => {
                let y2 = y.into_dimensionality::<Ix2>().expect("2-D");
                let s = py
                    .detach(|| spectrum::stft(y2, &cfg))
                    .map_err(audio_err_to_py)?;
                Ok(s.into_pyarray(py).into_any())
            }
            n => Err(value_err(
                "y",
                format!("expected shape (n,) or (channels, n), got {n} dimensions"),
            )),
        }
    })
}

/// Inverse short-time Fourier transform by weighted overlap-add.
///
/// Args:
///
///     stft_matrix (array_like): complex array of shape ``(1 + n_fft // 2, n_frames)``
///         or ``(channels, 1 + n_fft // 2, n_frames)``.
///     hop_length (Optional[int]): ``None`` means ``win_length // 4``.
///     win_length (Optional[int]): ``None`` means ``n_fft``.
///     n_fft (Optional[int]): ``None`` infers ``2 * (n_bins - 1)``.
///     window: As in ``stft``.
///     center (bool): Whether the frames were centred. Default True.
///     length (Optional[int]): Exact output length.
///
/// Returns:
///
///     numpy.ndarray[float32 | float64]: shape ``(n,)`` for 2-D input, ``(channels, n)`` for 3-D.
///
/// Raises:
///
///     ValueError: For invalid parameters or a bin count that does not match ``n_fft``.
#[pyfunction]
#[pyo3(
    signature = (stft_matrix, *, hop_length = None, win_length = None, n_fft = None, window = WindowArg(Window::Hann), center = true, length = None),
    text_signature = "(stft_matrix, *, hop_length=None, win_length=None, n_fft=None, window='hann', center=True, length=None)"
)]
#[allow(
    clippy::too_many_arguments,
    reason = "the Python signature takes these arguments"
)] // PyO3 signature mirrors the Python keyword arguments
pub fn istft<'py>(
    py: Python<'py>,
    stft_matrix: &Bound<'py, PyAny>,
    hop_length: Option<Int>,
    win_length: Option<Int>,
    n_fft: Option<Int>,
    window: WindowArg,
    center: bool,
    length: Option<Int>,
) -> PyResult<Bound<'py, PyAny>> {
    with_precision!(precision_of(stft_matrix), |F| {
        let d = complex_array_as::<F>(stft_matrix, "stft_matrix")?;
        let d = d.as_array();
        let ndim = d.ndim();
        if !(ndim == 2 || ndim == 3) {
            return Err(value_err(
                "stft_matrix",
                format!("expected 2 or 3 dimensions, got {ndim}"),
            ));
        }
        let n_fft = match (n_fft, d.shape()) {
            (Some(n_fft), _) => n_fft,
            (None, [.., n_bins, _]) => {
                Int(2 * (i64::try_from(*n_bins).map_err(|e| value_err("stft_matrix", e))? - 1))
            }
            (None, _) => {
                return Err(value_err(
                    "stft_matrix",
                    format!("expected 2 or 3 dimensions, got {ndim}"),
                ));
            }
        };
        let cfg = stft_config(n_fft, hop_length, win_length, window, center, "constant")?;
        let length = opt_positive("length", length)?.map(std::num::NonZeroUsize::get);
        if ndim == 2 {
            let d3 = d
                .into_dimensionality::<Ix2>()
                .expect("2-D")
                .insert_axis(Axis(0));
            let y = py
                .detach(|| spectrum::istft(d3, &cfg, length))
                .map_err(audio_err_to_py)?;
            Ok(y.index_axis_move(Axis(0), 0).into_pyarray(py).into_any())
        } else {
            let d3 = d.into_dimensionality::<Ix3>().expect("3-D");
            let y = py
                .detach(|| spectrum::istft(d3, &cfg, length))
                .map_err(audio_err_to_py)?;
            Ok(y.into_pyarray(py).into_any())
        }
    })
}

const fn db_config(
    DbRefArg(reference): DbRefArg,
    amin: f64,
    top_db: Option<f64>,
    DbAxesArg(axes): DbAxesArg,
) -> DbConfig {
    DbConfig {
        reference,
        amin,
        top_db,
        axes,
    }
}

/// Convert a power spectrogram to decibels.
///
/// Computes ``10 * log10(max(amin, S)) - 10 * log10(max(amin, |ref|))``, floored
/// at ``max - top_db``.
///
/// Args:
///
///     S (array_like): Power values, any shape (or a scalar).
///     ref (float | "max" | numpy.max): 0 dB reference. Default 1.0.
///     amin (float): Lower clamp before the log. Default 1e-10.
///     top_db (Optional[float]): Dynamic range below the maximum. Default None
///         (no floor).
///     axes ("auto" | None): Reduction axes for ``ref=numpy.max`` and ``top_db``.
///         ``"auto"`` (default) uses the last two axes, i.e. per channel for
///         ``(channels, F, T)`` input; ``None`` uses the whole array.
///
/// Returns:
///
///     float | numpy.ndarray[float32 | float64]: Same shape as ``S``.
///
/// Raises:
///
///     ValueError: If ``amin <= 0`` or ``top_db < 0``.
///     TypeError: For complex input (pass power, e.g. ``numpy.abs(D) ** 2``).
#[pyfunction]
#[pyo3(
    signature = (S, *, r#ref = DbRefArg(audio_samples::DbReference::Value(1.0)), amin = 1e-10, top_db = None, axes = DbAxesArg(audio_samples::DbAxes::Auto)),
    text_signature = "(S, *, ref=1.0, amin=1e-10, top_db=None, axes='auto')"
)]
#[allow(non_snake_case, reason = "argument names follow the Python API")]
pub fn power_to_db<'py>(
    py: Python<'py>,
    S: &Bound<'py, PyAny>,
    r#ref: DbRefArg,
    amin: f64,
    top_db: Option<f64>,
    axes: DbAxesArg,
) -> PyResult<Bound<'py, PyAny>> {
    let scalar = is_scalar(S)?;
    let cfg = db_config(r#ref, amin, top_db, axes);
    with_precision!(precision_of(S), |F| {
        let s = float_array_as::<F>(S, "S")?;
        let view = s.as_array();
        let out = py
            .detach(|| spectrum::power_to_db(&view, &cfg))
            .map_err(audio_err_to_py)?;
        dyn_out(py, out, scalar)
    })
}

/// Convert an amplitude spectrogram to decibels.
///
/// ``20 * log10(max(amin, |S|) / max(amin, |ref|))``, floored at ``max - top_db``.
/// Complex input is replaced by its magnitude.
///
/// Args:
///
///     S (array_like): Amplitudes (real or complex), any shape.
///     ref (float | "max" | numpy.max): 0 dB reference. Default 1.0.
///     amin (float): Lower clamp. Default 1e-5.
///     top_db (Optional[float]): Dynamic range below the maximum. Default None
///         (no floor).
///     axes ("auto" | None): As ``power_to_db``.
///
/// Returns:
///
///     float | numpy.ndarray[float32 | float64]: Same shape as ``S``.
///
/// Raises:
///
///     ValueError: If ``amin <= 0`` or ``top_db < 0``.
#[pyfunction]
#[pyo3(
    signature = (S, *, r#ref = DbRefArg(audio_samples::DbReference::Value(1.0)), amin = 1e-5, top_db = None, axes = DbAxesArg(audio_samples::DbAxes::Auto)),
    text_signature = "(S, *, ref=1.0, amin=1e-05, top_db=None, axes='auto')"
)]
#[allow(non_snake_case, reason = "argument names follow the Python API")]
pub fn amplitude_to_db<'py>(
    py: Python<'py>,
    S: &Bound<'py, PyAny>,
    r#ref: DbRefArg,
    amin: f64,
    top_db: Option<f64>,
    axes: DbAxesArg,
) -> PyResult<Bound<'py, PyAny>> {
    let scalar = is_scalar(S)?;
    // The Rust conversion takes |S| itself, so only complex input needs
    // numpy.abs first (it gives the real magnitude array).
    let complex = match S.cast::<numpy::PyUntypedArray>() {
        Ok(a) => a.dtype().kind() == b'c',
        Err(_) => py
            .import("numpy")?
            .call_method1("iscomplexobj", (S,))?
            .extract::<bool>()?,
    };
    let mag = if complex {
        py.import("numpy")?.call_method1("abs", (S,))?
    } else {
        S.clone()
    };
    let cfg = db_config(r#ref, amin, top_db, axes);
    with_precision!(precision_of(&mag), |F| {
        let s = float_array_as::<F>(&mag, "S")?;
        let out = spectrum::amplitude_to_db(&s.as_array(), &cfg).map_err(audio_err_to_py)?;
        dyn_out(py, out, scalar)
    })
}

/// Convert decibels to power:
/// ``ref * 10 ** (S_db / 10)``.
///
/// Args:
///
///     S_db (array_like): dB values, any shape (or a scalar).
///     ref (float): Reference power. Default 1.0.
///
/// Returns:
///
///     float | numpy.ndarray[float32 | float64]: Same shape as ``S_db``.
#[pyfunction]
#[pyo3(signature = (S_db, *, r#ref = 1.0), text_signature = "(S_db, *, ref=1.0)")]
#[allow(non_snake_case, reason = "argument names follow the Python API")]
pub fn db_to_power<'py>(
    py: Python<'py>,
    S_db: &Bound<'py, PyAny>,
    r#ref: f64,
) -> PyResult<Bound<'py, PyAny>> {
    let scalar = is_scalar(S_db)?;
    with_precision!(precision_of(S_db), |F| {
        let s = float_array_as::<F>(S_db, "S_db")?;
        dyn_out(py, spectrum::db_to_power(&s.as_array(), r#ref), scalar)
    })
}

/// Convert decibels to amplitude:
/// ``sqrt(ref ** 2 * 10 ** (S_db / 10))``.
///
/// Args:
///
///     S_db (array_like): dB values, any shape (or a scalar).
///     ref (float): Reference amplitude. Default 1.0.
///
/// Returns:
///
///     float | numpy.ndarray[float32 | float64]: Same shape as ``S_db``.
#[pyfunction]
#[pyo3(signature = (S_db, *, r#ref = 1.0), text_signature = "(S_db, *, ref=1.0)")]
#[allow(non_snake_case, reason = "argument names follow the Python API")]
pub fn db_to_amplitude<'py>(
    py: Python<'py>,
    S_db: &Bound<'py, PyAny>,
    r#ref: f64,
) -> PyResult<Bound<'py, PyAny>> {
    let scalar = is_scalar(S_db)?;
    with_precision!(precision_of(S_db), |F| {
        let s = float_array_as::<F>(S_db, "S_db")?;
        dyn_out(py, spectrum::db_to_amplitude(&s.as_array(), r#ref), scalar)
    })
}

/// Mel filterbank matrix mapping STFT bins to mel bands (float64).
///
/// Args:
///
///     sr (float): Sample rate (Hz).
///     n_fft (int): FFT length.
///     n_mels (int): Number of bands. Default 128.
///     fmin (float): Lowest edge (Hz). Default 0.
///     fmax (Optional[float]): Highest edge (Hz); ``None`` = sr / 2.
///     htk (bool): HTK mel scale instead of Slaney. Default False.
///     norm (None | "slaney" | float): ``"slaney"`` (default, area
///         normalisation), None, or a p-norm (``numpy.inf`` allowed).
///
/// Returns:
///
///     numpy.ndarray[float64]: shape ``(n_mels, 1 + n_fft // 2)``.
///
/// Raises:
///
///     ValueError: For invalid parameters (e.g. ``fmax <= fmin``).
#[pyfunction]
#[pyo3(
    signature = (*, sr, n_fft, n_mels = Int(128), fmin = 0.0, fmax = None, htk = false, norm = MelNormArg(audio_samples::MelFilterNorm::Slaney)),
    text_signature = "(*, sr, n_fft, n_mels=128, fmin=0.0, fmax=None, htk=False, norm='slaney')"
)]
#[allow(
    clippy::too_many_arguments,
    reason = "the Python signature takes these arguments"
)] // PyO3 signature mirrors the Python keyword arguments
pub fn mel_filterbank(
    py: Python<'_>,
    sr: f64,
    n_fft: Int,
    n_mels: Int,
    fmin: f64,
    fmax: Option<f64>,
    htk: bool,
    norm: MelNormArg,
) -> PyResult<Bound<'_, PyArray2<f64>>> {
    let cfg = mel_config(n_mels, fmin, fmax, htk, norm)?;
    let m = filterbanks::mel_filterbank(sr, positive("n_fft", n_fft)?, &cfg)
        .map_err(audio_err_to_py)?;
    Ok(m.into_pyarray(py))
}

/// Window coefficients for a window specification.
///
/// Args:
///
///     window: ``"hann"``, ``"hamming"``, ``"blackman"``, ``"blackmanharris"``,
///         ``"boxcar"`` (``"rectangular"``), ``"bartlett"``, ``"triang"``,
///         ``("kaiser", beta)``, ``("gaussian", std)``, a number (Kaiser beta),
///         or an array of length ``Nx`` (returned as given).
///     Nx (int): Window length.
///     fftbins (bool): Periodic window (True, default) or symmetric (False).
///
/// Returns:
///
///     numpy.ndarray[float64]: shape ``(Nx,)``.
///
/// Raises:
///
///     ValueError: Unknown window, invalid parameter, or an array of the wrong length.
#[pyfunction]
#[pyo3(signature = (window, Nx, *, fftbins = true), text_signature = "(window, Nx, *, fftbins=True)")]
#[allow(non_snake_case, clippy::needless_pass_by_value)] // PyO3 argument names; `WindowArg` is extracted by value
pub fn get_window(
    py: Python<'_>,
    window: WindowArg,
    Nx: Int,
    fftbins: bool,
) -> PyResult<Bound<'_, PyArray1<f64>>> {
    let w = filterbanks::get_window(&window.0, non_negative("Nx", Nx)?, fftbins)
        .map_err(audio_err_to_py)?;
    Ok(w.into_pyarray(py))
}

/// Sum-squared window envelope of an overlap-add of ``n_frames`` windows.
///
/// Args:
///
///     window: Window specification (see ``get_window``).
///     n_frames (int): Number of frames.
///     hop_length (int): Default 512.
///     win_length (Optional[int]): ``None`` means ``n_fft``.
///     n_fft (int): Default 2048.
///     norm (Optional[float]): Normalise the window before squaring, as
///         ``features.normalize(w, norm=norm)``: ``None`` (default), a
///         positive number (L-p norm), ``math.inf`` / ``-math.inf`` (max / min
///         absolute value) or ``0`` (count of non-zeros).
///     dtype (Optional[numpy.dtype]): ``float32`` or ``float64``. ``None``
///         (default) means ``float64``.
///
/// Returns:
///
///     numpy.ndarray[float64 | float32]: shape ``(n_fft + hop_length * (n_frames - 1),)``.
///
/// Raises:
///
///     ValueError: For invalid parameters.
#[pyfunction]
#[pyo3(
    signature = (*, window, n_frames, hop_length = Int(512), win_length = None, n_fft = Int(2048), norm = None, dtype = None),
    text_signature = "(*, window, n_frames, hop_length=512, win_length=None, n_fft=2048, norm=None, dtype=None)"
)]
#[allow(
    clippy::too_many_arguments,
    reason = "the Python signature takes these arguments"
)]
pub fn window_sumsquare<'py>(
    py: Python<'py>,
    window: WindowArg,
    n_frames: Int,
    hop_length: Int,
    win_length: Option<Int>,
    n_fft: Int,
    norm: Option<f64>,
    dtype: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let single = match dtype {
        None => false,
        Some(d) => {
            let d = numpy::PyArrayDescr::new(py, d)?;
            if d.is_equiv_to(&numpy::dtype::<f32>(py)) {
                true
            } else if d.is_equiv_to(&numpy::dtype::<f64>(py)) {
                false
            } else {
                return Err(value_err("dtype", "expected float32 or float64"));
            }
        }
    };
    let n_fft = positive("n_fft", n_fft)?;
    let win_length = opt_positive("win_length", win_length)?;
    // Normalise the (periodic) window before squaring and hand the result over
    // as a custom window of length win_length.
    let window = match norm {
        None => window.0,
        Some(p) => {
            let wl = win_length.unwrap_or(n_fft).get();
            if wl > n_fft.get() {
                return Err(value_err("win_length", "cannot be longer than n_fft"));
            }
            let w = filterbanks::get_window(&window.0, wl, true).map_err(audio_err_to_py)?;
            let cfg = audio_samples::operations::tonal::NormalizeConfig::new()
                .with_norm(Some(p))
                .with_axis(Some(0));
            let w = audio_samples::operations::tonal::normalize_array(&w, &cfg)
                .map_err(audio_err_to_py)?;
            Window::Custom(w.to_vec())
        }
    };
    let n_frames = positive("n_frames", n_frames)?.get();
    let hop_length = positive("hop_length", hop_length)?;
    Ok(if single {
        filterbanks::window_sumsquare_as::<f32>(&window, n_frames, hop_length, win_length, n_fft)
            .map_err(audio_err_to_py)?
            .into_pyarray(py)
            .into_any()
    } else {
        filterbanks::window_sumsquare(&window, n_frames, hop_length, win_length, n_fft)
            .map_err(audio_err_to_py)?
            .into_pyarray(py)
            .into_any()
    })
}

/// Equivalent noise bandwidth of a window in FFT bins:
/// ``n * sum(w ** 2) / sum(w) ** 2`` for the periodic window of length ``n``.
/// The definition is evaluated for every window, named or not.
///
/// Args:
///
///     window: Window specification (see ``get_window``).
///     n (int): Window length. Default 1000.
///
/// Returns:
///
///     float: Bandwidth in bins (1.0 for boxcar, 1.5 for Hann).
#[pyfunction]
#[pyo3(signature = (window, n = Int(1000)), text_signature = "(window, n=1000)")]
pub fn window_bandwidth<'py>(
    py: Python<'py>,
    window: &Bound<'py, PyAny>,
    n: Int,
) -> PyResult<Bound<'py, pyo3::types::PyFloat>> {
    use pyo3::types::{PyFloat, PyString};
    // The last result for a window given as a string, keyed by the string
    // object (immutable, so the same object gives the same value): a repeated
    // call with the same name returns the same float without parsing the
    // name or allocating.
    /// The window object, its length and its bandwidth.
    type LastWindow = Option<(Py<PyAny>, i64, Py<PyFloat>)>;
    thread_local! {
        static LAST: std::cell::RefCell<LastWindow> = const { std::cell::RefCell::new(None) };
    }
    let hit = LAST.with(|c| {
        c.borrow()
            .as_ref()
            .filter(|(w, len, _)| *len == n.0 && w.as_ptr() == window.as_ptr())
            .map(|(_, _, v)| v.clone_ref(py))
    });
    if let Some(v) = hit {
        return Ok(v.into_bound(py));
    }
    let w = crate::args::parse_window(window)?;
    let n_key = n.0;
    let n = positive("n", n)?;
    let v = PyFloat::new(
        py,
        filterbanks::window_bandwidth(&w, n).map_err(audio_err_to_py)?,
    );
    if window.is_instance_of::<PyString>() {
        LAST.with(|c| {
            *c.borrow_mut() = Some((window.clone().unbind(), n_key, v.clone().unbind()));
        });
    }
    Ok(v)
}

/// Savitzky-Golay delta features along the last axis.
///
/// Args:
///
///     data (array_like): Feature array, any shape (e.g. ``(n_mfcc, T)`` or
///         ``(channels, n_mfcc, T)``).
///     width (int): Odd window width >= 3. Default 9.
///     order (int): Derivative order (1 = delta, 2 = delta-delta). Default 1.
///     axis (int): Axis to differentiate along. Default -1.
///     mode (str): ``"interp"`` (default), ``"nearest"``, ``"mirror"``,
///         ``"constant"`` or ``"wrap"``.
///
/// Returns:
///
///     numpy.ndarray[float32 | float64]: Same shape as ``data``.
///
/// Raises:
///
///     ValueError: Invalid width/order/axis, or ``mode="interp"`` with
///         ``width`` larger than the axis.
#[pyfunction]
#[pyo3(
    signature = (data, *, width = Int(9), order = Int(1), axis = Int(-1), mode = "interp"),
    text_signature = "(data, *, width=9, order=1, axis=-1, mode='interp')"
)]
pub fn delta<'py>(
    py: Python<'py>,
    data: &Bound<'py, PyAny>,
    width: Int,
    order: Int,
    axis: Int,
    mode: &str,
) -> PyResult<Bound<'py, PyAny>> {
    let mode = match mode {
        "interp" => DeltaMode::Interp,
        "nearest" => DeltaMode::Nearest,
        "mirror" => DeltaMode::Mirror,
        "constant" => DeltaMode::Constant,
        "wrap" => DeltaMode::Wrap,
        other => {
            return Err(value_err(
                "mode",
                format!(
                    "unsupported mode {other:?}; expected 'interp', 'nearest', 'mirror', 'constant' or 'wrap'"
                ),
            ));
        }
    };
    let cfg = DeltaConfig {
        width: non_negative("width", width)?,
        order: non_negative("order", order)?,
        axis: crate::args::signed("axis", axis)?,
        mode,
    };
    with_precision!(precision_of(data), |F| {
        let x = float_array_as::<F>(data, "data")?;
        let out = spectrum::delta(&x.as_array(), &cfg).map_err(audio_err_to_py)?;
        Ok(out.into_pyarray(py).into_any())
    })
}

/// Stack time-delayed copies of a feature array (zero padding).
///
/// Args:
///
///     data (array_like): shape ``(..., d, t)``; 1-D input is treated as ``(1, t)``.
///     n_steps (int): Number of copies. Default 2.
///     delay (int): Frame delay between copies (non-zero; negative looks
///         ahead). Default 1.
///
/// Returns:
///
///     numpy.ndarray[float32 | float64]: shape ``(..., d * n_steps, t)``.
///
/// Raises:
///
///     ValueError: If ``n_steps < 1`` or ``delay == 0``.
#[pyfunction]
#[pyo3(signature = (data, *, n_steps = Int(2), delay = Int(1)), text_signature = "(data, *, n_steps=2, delay=1)")]
pub fn stack_memory<'py>(
    py: Python<'py>,
    data: &Bound<'py, PyAny>,
    n_steps: Int,
    delay: Int,
) -> PyResult<Bound<'py, PyAny>> {
    let n_steps = positive("n_steps", n_steps)?;
    let delay = crate::args::signed("delay", delay)?;
    with_precision!(precision_of(data), |F| {
        let x = float_array_as::<F>(data, "data")?;
        let x = x.as_array();
        let x = if x.ndim() == 1 {
            x.insert_axis(Axis(0))
        } else {
            x
        };
        let out = spectrum::stack_memory(&x, n_steps, delay).map_err(audio_err_to_py)?;
        Ok(out.into_pyarray(py).into_any())
    })
}

/// Slice a 1-D signal into overlapping frames along the last axis.
///
/// By default the frames are a new, writable array (a copy). With
/// ``copy=False`` the result is instead a read-only strided view into ``x``:
/// no data is copied, the view keeps ``x`` alive and reflects later changes to
/// it. Input that is not a contiguous float32 / float64 array is converted
/// first, and the view is into the converted array.
///
/// Args:
///
///     x (array_like): 1-D signal.
///     frame_length (int): Samples per frame.
///     hop_length (int): Samples between frame starts.
///     copy (bool): Return a writable copy (default) or, if False, a
///         read-only view into ``x``.
///
/// Returns:
///
///     numpy.ndarray[float32 | float64]: shape
///     ``(frame_length, 1 + (len(x) - frame_length) // hop_length)``.
///
/// Raises:
///
///     ValueError: If ``x`` is not 1-D or is shorter than ``frame_length``,
///         or (copy only) the frame matrix would exceed 2^31 elements.
#[pyfunction]
#[pyo3(
    signature = (x, *, frame_length, hop_length, copy = true),
    text_signature = "(x, *, frame_length, hop_length, copy=True)"
)]
pub fn frame<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyAny>,
    frame_length: Int,
    hop_length: Int,
    copy: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let frame_length = positive("frame_length", frame_length)?;
    let hop_length = positive("hop_length", hop_length)?;
    with_precision!(precision_of(x), |F| {
        // Fast path: a contiguous 1-D array of the right dtype is read in
        // place (no conversion, no borrow bookkeeping). Without a GIL, reads go
        // through the borrow below instead.
        #[cfg(not(Py_GIL_DISABLED))]
        if let Ok(arr) = x.cast::<numpy::PyArray1<F>>() {
            if arr.is_contiguous() {
                // SAFETY: the GIL is held for the lifetime of `data`, so no
                // Python code can resize or free the array meanwhile; the
                // slice is only read.
                let data = unsafe { arr.as_slice() }.expect("contiguous");
                return if copy {
                    let out =
                        spectrum::frame(data, frame_length, hop_length).map_err(audio_err_to_py)?;
                    Ok(out.into_pyarray(py).into_any())
                } else {
                    let view = spectrum::frame_view(data, frame_length, hop_length)
                        .map_err(audio_err_to_py)?;
                    Ok(readonly_frames(&view, arr.clone().into_any()))
                };
            }
        }
        let a = float_array_as::<F>(x, "x")?;
        if a.ndim() != 1 {
            return Err(value_err("x", "only 1-D input is supported"));
        }
        // Both paths read contiguous data: a strided input is copied once.
        let contiguous = if a.as_slice().is_ok() {
            a
        } else {
            let c = a.as_array().to_owned().into_pyarray(py);
            c.readonly()
        };
        let data = contiguous.as_slice().expect("contiguous");
        if copy {
            let out = spectrum::frame(data, frame_length, hop_length).map_err(audio_err_to_py)?;
            return Ok(out.into_pyarray(py).into_any());
        }
        let view = spectrum::frame_view(data, frame_length, hop_length).map_err(audio_err_to_py)?;
        let base: Bound<'py, PyAny> = (*contiguous).clone().into_any();
        Ok(readonly_frames(&view, base))
    })
}

/// A read-only numpy array over `view`, whose data belongs to the numpy array
/// `base` (kept alive as the result's base object).
fn readonly_frames<'py, F: numpy::Element>(
    view: &ndarray::ArrayView2<'_, F>,
    base: Bound<'py, PyAny>,
) -> Bound<'py, PyAny> {
    // SAFETY: `base` holds the data of `view` and becomes the base object of
    // the result, so the data outlives it; numpy arrays are never
    // reallocated.
    let out = unsafe { PyArray2::borrow_from_array(view, base) };
    // Overlapping frames share memory: the view is read-only.
    // SAFETY: `out` is a valid array object created above.
    unsafe {
        (*out.as_array_ptr()).flags &= !numpy::npyffi::NPY_ARRAY_WRITEABLE;
    }
    out.into_any()
}

/// Project a spectrogram onto mel bands: the mel filterbank applied to ``S``.
///
/// Args:
///
///     S (array_like): Non-negative spectrogram, shape ``(1 + n_fft // 2, T)`` or
///         ``(channels, 1 + n_fft // 2, T)`` (power or magnitude, as you choose).
///     sr (float): Sample rate (Hz).
///     n_fft (Optional[int]): FFT length; ``None`` infers ``2 * (F - 1)``.
///     n_mels, fmin, fmax, htk, norm: As ``mel_filterbank``.
///
/// Returns:
///
///     numpy.ndarray[float32 | float64]: shape ``(n_mels, T)`` or ``(channels, n_mels, T)``.
#[pyfunction]
#[pyo3(
    signature = (S, *, sr, n_fft = None, n_mels = Int(128), fmin = 0.0, fmax = None, htk = false, norm = MelNormArg(audio_samples::MelFilterNorm::Slaney)),
    text_signature = "(S, *, sr, n_fft=None, n_mels=128, fmin=0.0, fmax=None, htk=False, norm='slaney')"
)]
#[allow(
    non_snake_case,
    clippy::too_many_arguments,
    reason = "argument names and count follow the Python signature"
)]
pub fn melspectrogram_from_spectrogram<'py>(
    py: Python<'py>,
    S: &Bound<'py, PyAny>,
    sr: f64,
    n_fft: Option<Int>,
    n_mels: Int,
    fmin: f64,
    fmax: Option<f64>,
    htk: bool,
    norm: MelNormArg,
) -> PyResult<Bound<'py, PyAny>> {
    with_precision!(precision_of(S), |F| {
        let s = float_array_as::<F>(S, "S")?;
        let s = s.as_array();
        let ndim = s.ndim();
        let s3 = match ndim {
            2 => s
                .into_dimensionality::<Ix2>()
                .expect("2-D")
                .insert_axis(Axis(0)),
            3 => s.into_dimensionality::<Ix3>().expect("3-D"),
            n => {
                return Err(value_err(
                    "S",
                    format!("expected 2 or 3 dimensions, got {n}"),
                ));
            }
        };
        let n_fft = match n_fft {
            Some(n_fft) => n_fft,
            None => Int(2 * (i64::try_from(s3.dim().1).map_err(|e| value_err("S", e))? - 1)),
        };
        let cfg = mel_config(n_mels, fmin, fmax, htk, norm)?;
        let m = spectrum::melspectrogram_from_spectrogram(s3, sr, positive("n_fft", n_fft)?, &cfg)
            .map_err(audio_err_to_py)?;
        if ndim == 2 {
            Ok(m.index_axis_move(Axis(0), 0).into_pyarray(py).into_any())
        } else {
            Ok(m.into_pyarray(py).into_any())
        }
    })
}

/// MFCCs from a log-power (dB) mel spectrogram: DCT along the mel axis, first
/// ``n_mfcc`` rows, liftering.
///
/// Args:
///
///     S (array_like): dB mel spectrogram, shape ``(..., n_mels, T)`` (at least 2-D).
///     n_mfcc (int): Default 20 (must not exceed ``n_mels``).
///     dct_type (int): 1, 2 (default) or 3.
///     norm (Optional[str]): ``"ortho"`` (default) or None.
///     lifter (float): Liftering coefficient; 0 (default) disables.
///
/// Returns:
///
///     numpy.ndarray[float32 | float64]: shape ``(..., n_mfcc, T)``.
#[pyfunction]
#[pyo3(
    signature = (S, *, n_mfcc = Int(20), dct_type = Int(2), norm = Some("ortho"), lifter = 0.0),
    text_signature = "(S, *, n_mfcc=20, dct_type=2, norm='ortho', lifter=0.0)"
)]
#[allow(non_snake_case, reason = "argument names follow the Python API")]
pub fn mfcc_from_mel_db<'py>(
    py: Python<'py>,
    S: &Bound<'py, PyAny>,
    n_mfcc: Int,
    dct_type: Int,
    norm: Option<&str>,
    lifter: f64,
) -> PyResult<Bound<'py, PyAny>> {
    let cfg = mfcc_cepstrum_config(n_mfcc, dct_type, norm, lifter)?;
    with_precision!(precision_of(S), |F| {
        let s = float_array_as::<F>(S, "S")?;
        let out = spectrum::mfcc_from_mel_db(&s.as_array(), &cfg).map_err(audio_err_to_py)?;
        Ok(out.into_pyarray(py).into_any())
    })
}

/// Registers `audio_samples.spectral`.
pub fn register(py: Python<'_>, parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(py, "spectral")?;
    m.add(
        "__doc__",
        "Spectral functions over NumPy arrays (STFT/ISTFT, dB \
         conversions, mel filterbank, windows, delta, stack_memory, frame). \
         float32 / complex64 arrays are processed in float32 and give float32 \
         output; any other input is converted to float64.",
    )?;
    m.add_function(wrap_pyfunction!(stft, &m)?)?;
    m.add_function(wrap_pyfunction!(istft, &m)?)?;
    m.add_function(wrap_pyfunction!(power_to_db, &m)?)?;
    m.add_function(wrap_pyfunction!(amplitude_to_db, &m)?)?;
    m.add_function(wrap_pyfunction!(db_to_power, &m)?)?;
    m.add_function(wrap_pyfunction!(db_to_amplitude, &m)?)?;
    m.add_function(wrap_pyfunction!(mel_filterbank, &m)?)?;
    m.add_function(wrap_pyfunction!(get_window, &m)?)?;
    m.add_function(wrap_pyfunction!(window_sumsquare, &m)?)?;
    m.add_function(wrap_pyfunction!(window_bandwidth, &m)?)?;
    m.add_function(wrap_pyfunction!(delta, &m)?)?;
    m.add_function(wrap_pyfunction!(stack_memory, &m)?)?;
    m.add_function(wrap_pyfunction!(frame, &m)?)?;
    m.add_function(wrap_pyfunction!(melspectrogram_from_spectrogram, &m)?)?;
    m.add_function(wrap_pyfunction!(mfcc_from_mel_db, &m)?)?;
    parent.add_submodule(&m)?;
    Ok(())
}
