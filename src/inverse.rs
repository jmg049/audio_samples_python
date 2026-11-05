//! `audio_samples.inverse`: inversion of mel spectrograms and MFCCs to spectrograms and audio.

use audio_samples::operations::effects::GriffinLimConfig;
use audio_samples::operations::filterbanks::Window;
use audio_samples::operations::inverse::{self as inv, MelInversionConfig, MfccInversionConfig};
use audio_samples::operations::linalg::NnlsConfig;
use pyo3::prelude::*;

use crate::args::{Int, MelNormArg, PyFloat, WindowArg, mel_config, positive, precision_of, stft_config, with_precision};
use crate::audio_err_to_py;
use crate::effects::{RngArg, griffinlim_init};
use crate::features::args::{squeeze_out, spec3};
use crate::operations::spectrum::{parse_dct_norm, parse_dct_type};

#[allow(clippy::too_many_arguments)]
fn mel_inversion(
    n_mels: usize,
    sr: f64,
    n_fft: Int,
    power: f64,
    fmin: f64,
    fmax: Option<f64>,
    htk: bool,
    norm: MelNormArg,
) -> PyResult<MelInversionConfig> {
    let mel = mel_config(Int(n_mels.max(1) as i64), fmin, fmax, htk, norm)?;
    Ok(MelInversionConfig::new(sr)
        .with_n_fft(positive("n_fft", n_fft)?)
        .with_power(power)
        .with_mel(mel)
        .with_nnls(NnlsConfig::new()))
}

/// Estimates an STFT magnitude from a mel spectrogram by non-negative least squares.
///
/// Args:
///     M (array_like): Mel spectrogram ``(n_mels, n_frames)`` or
///         ``(channels, n_mels, n_frames)``.
///     sr (float): Sample rate in Hz.
///     n_fft (int): FFT size of the spectrogram to estimate. Default 2048.
///     power (float): Exponent that was applied to the magnitude. Default 2.
///     fmin, fmax, htk, norm: The filterbank, as ``spectral.mel_filterbank``.
///
/// Returns:
///     numpy.ndarray: ``(1 + n_fft // 2, n_frames)``, with a leading channel axis for 3-D
///     input, of ``M``'s precision.
#[pyfunction]
#[pyo3(
    signature = (M, *, sr, n_fft = Int(2048), power = 2.0, fmin = 0.0, fmax = None, htk = false, norm = MelNormArg(audio_samples::MelFilterNorm::Slaney)),
    text_signature = "(M, *, sr, n_fft=2048, power=2.0, fmin=0.0, fmax=None, htk=False, norm='slaney')"
)]
#[allow(non_snake_case, clippy::too_many_arguments)]
pub fn mel_to_stft<'py>(
    M: &Bound<'py, PyAny>,
    sr: f64,
    n_fft: Int,
    power: f64,
    fmin: f64,
    fmax: Option<f64>,
    htk: bool,
    norm: MelNormArg,
) -> PyResult<Bound<'py, PyAny>> {
    let py = M.py();
    with_precision!(precision_of(M), |F| {
        let (m, nd) = spec3::<F>(M, "M")?;
        let view = m.view();
        let cfg = mel_inversion(view.dim().1, sr, n_fft, power, fmin, fmax, htk, norm)?;
        let out = py.detach(|| inv::mel_to_stft(view, &cfg)).map_err(audio_err_to_py)?;
        Ok(squeeze_out(py, out.into_dyn(), nd == 2))
    })
}

#[allow(clippy::too_many_arguments)]
fn griffin_config(
    n_fft: Int,
    hop_length: Option<Int>,
    win_length: Option<Int>,
    window: WindowArg,
    center: bool,
    pad_mode: &str,
    n_iter: Int,
    length: Option<Int>,
    momentum: f64,
    init: Option<&str>,
    rng: RngArg,
) -> PyResult<GriffinLimConfig> {
    Ok(GriffinLimConfig::new()
        .with_n_iter(crate::args::non_negative("n_iter", n_iter)?)
        .with_momentum(momentum)
        .with_init(griffinlim_init(init, rng)?)
        .with_stft(stft_config(n_fft, hop_length, win_length, window, center, pad_mode)?)
        .with_length(crate::args::opt_positive("length", length)?.map(std::num::NonZeroUsize::get)))
}

fn rows_out<'py, F: PyFloat>(py: Python<'py>, y: ndarray::Array2<F>, squeeze: bool) -> Bound<'py, PyAny> {
    squeeze_out(py, y.into_dyn(), squeeze)
}

/// Estimates audio from a mel spectrogram: least squares, then fast Griffin-Lim.
///
/// Args:
///     M (array_like): Mel spectrogram ``(n_mels, n_frames)`` or
///         ``(channels, n_mels, n_frames)``.
///     sr (float): Sample rate in Hz.
///     n_fft, hop_length, win_length, window, center, pad_mode: STFT parameters.
///     power (float): Exponent that was applied to the magnitude. Default 2.
///     n_iter (int): Griffin-Lim iterations. Default 32.
///     length (Optional[int]): Exact output length.
///     momentum (float): Fast Griffin-Lim momentum. Default 0.99.
///     init (Optional[str]): ``"random"`` (default) or None (zero phase).
///     rng (Optional[int]): Seed for ``init="random"``. Default 0.
///     fmin, fmax, htk, norm: The filterbank.
///
/// Returns:
///     numpy.ndarray: ``(n,)``, or ``(channels, n)`` for 3-D input, of ``M``'s precision.
#[pyfunction]
#[pyo3(
    signature = (M, *, sr, n_fft = Int(2048), hop_length = None, win_length = None, window = WindowArg(Window::Hann), center = true, pad_mode = "constant", power = 2.0, n_iter = Int(32), length = None, momentum = 0.99, init = Some("random"), rng = RngArg::SEEDED, fmin = 0.0, fmax = None, htk = false, norm = MelNormArg(audio_samples::MelFilterNorm::Slaney)),
    text_signature = "(M, *, sr, n_fft=2048, hop_length=None, win_length=None, window='hann', center=True, pad_mode='constant', power=2.0, n_iter=32, length=None, momentum=0.99, init='random', rng=0, fmin=0.0, fmax=None, htk=False, norm='slaney')"
)]
#[allow(non_snake_case, clippy::too_many_arguments)]
pub fn mel_to_audio<'py>(
    M: &Bound<'py, PyAny>,
    sr: f64,
    n_fft: Int,
    hop_length: Option<Int>,
    win_length: Option<Int>,
    window: WindowArg,
    center: bool,
    pad_mode: &str,
    power: f64,
    n_iter: Int,
    length: Option<Int>,
    momentum: f64,
    init: Option<&str>,
    rng: RngArg,
    fmin: f64,
    fmax: Option<f64>,
    htk: bool,
    norm: MelNormArg,
) -> PyResult<Bound<'py, PyAny>> {
    let py = M.py();
    let gl = griffin_config(n_fft, hop_length, win_length, window, center, pad_mode, n_iter, length, momentum, init, rng)?;
    with_precision!(precision_of(M), |F| {
        let (m, nd) = spec3::<F>(M, "M")?;
        let view = m.view();
        let cfg = mel_inversion(view.dim().1, sr, n_fft, power, fmin, fmax, htk, norm)?;
        let y = py.detach(|| inv::mel_to_audio(view, &cfg, &gl)).map_err(audio_err_to_py)?;
        Ok(rows_out(py, y, nd == 2))
    })
}

fn mfcc_inversion(
    n_mels: Int,
    dct_type: Int,
    norm: Option<&str>,
    reference: f64,
    lifter: f64,
) -> PyResult<MfccInversionConfig> {
    Ok(MfccInversionConfig::new()
        .with_n_mels(positive("n_mels", n_mels)?)
        .with_dct_type(parse_dct_type(dct_type.0)?)
        .with_dct_norm(parse_dct_norm(norm)?)
        .with_reference(reference)
        .with_lifter(lifter))
}

/// Recovers a mel power spectrogram from MFCCs.
///
/// Args:
///     mfcc (array_like): ``(n_mfcc, n_frames)`` or ``(channels, n_mfcc, n_frames)``.
///     n_mels (int): Mel bands to recover. Default 128.
///     dct_type (int): 1, 2 or 3. Default 2.
///     norm (Optional[str]): ``"ortho"`` or None. Default ``"ortho"``.
///     ref (float): The reference of the decibel scale. Default 1.
///     lifter (float): The liftering coefficient to undo. Default 0.
///
/// Returns:
///     numpy.ndarray: ``(n_mels, n_frames)``, with a leading channel axis for 3-D input,
///     of ``mfcc``'s precision.
#[pyfunction]
#[pyo3(
    signature = (mfcc, *, n_mels = Int(128), dct_type = Int(2), norm = Some("ortho"), r#ref = 1.0, lifter = 0.0),
    text_signature = "(mfcc, *, n_mels=128, dct_type=2, norm='ortho', ref=1.0, lifter=0.0)"
)]
pub fn mfcc_to_mel<'py>(
    mfcc: &Bound<'py, PyAny>,
    n_mels: Int,
    dct_type: Int,
    norm: Option<&str>,
    r#ref: f64,
    lifter: f64,
) -> PyResult<Bound<'py, PyAny>> {
    let py = mfcc.py();
    let cfg = mfcc_inversion(n_mels, dct_type, norm, r#ref, lifter)?;
    with_precision!(precision_of(mfcc), |F| {
        let (m, nd) = spec3::<F>(mfcc, "mfcc")?;
        let view = m.view();
        let out = py.detach(|| inv::mfcc_to_mel(view, &cfg)).map_err(audio_err_to_py)?;
        Ok(squeeze_out(py, out.into_dyn(), nd == 2))
    })
}

/// Estimates audio from MFCCs: :func:`mfcc_to_mel`, then :func:`mel_to_audio`.
///
/// Args:
///     mfcc (array_like): ``(n_mfcc, n_frames)`` or ``(channels, n_mfcc, n_frames)``.
///     sr (float): Sample rate in Hz.
///     n_mels, dct_type, norm, ref, lifter: As for :func:`mfcc_to_mel`.
///     n_fft, hop_length, win_length, window, center, pad_mode, power, n_iter, length,
///         momentum, init, rng, fmin, fmax, htk, mel_norm: As for :func:`mel_to_audio`,
///         where ``mel_norm`` is its ``norm``.
///
/// Returns:
///     numpy.ndarray: ``(n,)``, or ``(channels, n)`` for 3-D input, of ``mfcc``'s precision.
#[pyfunction]
#[pyo3(
    signature = (mfcc, *, sr, n_mels = Int(128), dct_type = Int(2), norm = Some("ortho"), r#ref = 1.0, lifter = 0.0, n_fft = Int(2048), hop_length = None, win_length = None, window = WindowArg(Window::Hann), center = true, pad_mode = "constant", power = 2.0, n_iter = Int(32), length = None, momentum = 0.99, init = Some("random"), rng = RngArg::SEEDED, fmin = 0.0, fmax = None, htk = false, mel_norm = MelNormArg(audio_samples::MelFilterNorm::Slaney)),
    text_signature = "(mfcc, *, sr, n_mels=128, dct_type=2, norm='ortho', ref=1.0, lifter=0.0, n_fft=2048, hop_length=None, win_length=None, window='hann', center=True, pad_mode='constant', power=2.0, n_iter=32, length=None, momentum=0.99, init='random', rng=0, fmin=0.0, fmax=None, htk=False, mel_norm='slaney')"
)]
#[allow(clippy::too_many_arguments)]
pub fn mfcc_to_audio<'py>(
    mfcc: &Bound<'py, PyAny>,
    sr: f64,
    n_mels: Int,
    dct_type: Int,
    norm: Option<&str>,
    r#ref: f64,
    lifter: f64,
    n_fft: Int,
    hop_length: Option<Int>,
    win_length: Option<Int>,
    window: WindowArg,
    center: bool,
    pad_mode: &str,
    power: f64,
    n_iter: Int,
    length: Option<Int>,
    momentum: f64,
    init: Option<&str>,
    rng: RngArg,
    fmin: f64,
    fmax: Option<f64>,
    htk: bool,
    mel_norm: MelNormArg,
) -> PyResult<Bound<'py, PyAny>> {
    let py = mfcc.py();
    let m_cfg = mfcc_inversion(n_mels, dct_type, norm, r#ref, lifter)?;
    let gl = griffin_config(n_fft, hop_length, win_length, window, center, pad_mode, n_iter, length, momentum, init, rng)?;
    let cfg = mel_inversion(m_cfg.n_mels.get(), sr, n_fft, power, fmin, fmax, htk, mel_norm)?;
    with_precision!(precision_of(mfcc), |F| {
        let (m, nd) = spec3::<F>(mfcc, "mfcc")?;
        let view = m.view();
        let y = py
            .detach(|| inv::mfcc_to_audio(view, &m_cfg, &cfg, &gl))
            .map_err(audio_err_to_py)?;
        Ok(rows_out(py, y, nd == 2))
    })
}

pub fn register(py: Python<'_>, parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(py, "inverse")?;
    m.add("__doc__", "Inversion of mel spectrograms and MFCCs to spectrograms and audio.")?;
    m.add_function(wrap_pyfunction!(mel_to_stft, &m)?)?;
    m.add_function(wrap_pyfunction!(mel_to_audio, &m)?)?;
    m.add_function(wrap_pyfunction!(mfcc_to_mel, &m)?)?;
    m.add_function(wrap_pyfunction!(mfcc_to_audio, &m)?)?;
    parent.add_submodule(&m)?;
    Ok(())
}
