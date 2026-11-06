//! librosa-compatible spectral analysis methods on `AudioSamples`
//! (`stft_matrix`, `istft_matrix`, `spectrogram_matrix`, `melspectrogram`,
//! `mfcc_frames`), binding the Rust `AudioSpectrum` trait.
//!
//! Every method returns arrays shaped `(channels, n_features, n_frames)` (mono
//! audio gives a leading axis of length 1) and takes librosa's keyword
//! arguments. Output arrays are `float32` / `complex64` for `uint8`, `int16`,
//! `int24` and `float32` audio, and `float64` / `complex128` for `int32` and
//! `float64` audio. The defaults equal librosa 1.0's except `mfcc_frames`'
//! `top_db`, which defaults to `None` (librosa: 80).

use crate::args::Int;
use audio_samples::operations::filterbanks::Window;
use audio_samples::operations::spectrum::{
    DbConfig, DctNorm, DctType, MelSpectrogramConfig, MfccConfig, stft_n_frames,
};
use audio_samples::{AudioSample, AudioSamples, AudioSpectrum};
use ndarray::{Axis, Ix3};
use num_complex::Complex;
use numpy::{PyArrayDyn, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::prelude::*;

use crate::args::{
    DbRefArg, MelNormArg, WindowArg, complex_array_as, mel_config, output_sample_type, positive,
    stft_config, to_3d, value_err, with_sample_type,
};
use crate::types::PySampleType;
use crate::{
    PyAudioDataInner, PyAudioSamples, audio_err_to_py, dispatch_array_detached, nzu32_or_err,
};

/// Parses `dct_type` (1, 2 or 3).
pub(crate) fn parse_dct_type(dct_type: i64) -> PyResult<DctType> {
    match dct_type {
        1 => Ok(DctType::One),
        2 => Ok(DctType::Two),
        3 => Ok(DctType::Three),
        other => Err(value_err(
            "dct_type",
            format!("must be 1, 2 or 3, got {other}"),
        )),
    }
}

/// Parses the DCT `norm` (`"ortho"`, or `None`/`"backward"` for no normalisation).
pub(crate) fn parse_dct_norm(norm: Option<&str>) -> PyResult<DctNorm> {
    match norm {
        Some("ortho") => Ok(DctNorm::Ortho),
        None | Some("backward") => Ok(DctNorm::Unnormalized),
        Some(other) => Err(value_err(
            "norm",
            format!("must be 'ortho' or None, got {other:?}"),
        )),
    }
}

/// Builds an [`MfccConfig`] from librosa `mfcc` keyword arguments (shared with
/// `audio_samples.spectral.mfcc_from_mel_db`).
pub(crate) fn mfcc_cepstrum_config(
    n_mfcc: Int,
    dct_type: Int,
    norm: Option<&str>,
    lifter: f64,
) -> PyResult<MfccConfig> {
    Ok(MfccConfig::new()
        .with_n_mfcc(positive("n_mfcc", n_mfcc)?)
        .with_dct_type(parse_dct_type(dct_type.0)?)
        .with_dct_norm(parse_dct_norm(norm)?)
        .with_lifter(lifter))
}

#[pymethods]
impl PyAudioSamples {
    /// Complex short-time Fourier transform of every channel, equivalent to
    /// ``librosa.stft``.
    ///
    /// Args:
    ///     n_fft (int): FFT length in samples. Default 2048.
    ///     hop_length (Optional[int]): Samples between frames. ``None`` means
    ///         ``win_length // 4`` (librosa's default).
    ///     win_length (Optional[int]): Window length (<= n_fft); ``None`` means
    ///         ``n_fft``. Shorter windows are zero-padded to ``n_fft``.
    ///     window (str | tuple | numpy.ndarray): Window specification, as
    ///         ``scipy.signal.get_window``: ``"hann"`` (default), ``"hamming"``,
    ///         ``"blackman"``, ``"blackmanharris"``, ``"boxcar"``/``"rectangular"``,
    ///         ``"bartlett"``, ``"triang"``, ``("kaiser", beta)``,
    ///         ``("gaussian", std)``, or an array of ``win_length`` coefficients.
    ///         Named windows are periodic (``fftbins=True``), as in librosa.
    ///     center (bool): Pad by ``n_fft // 2`` on both sides so frame ``t`` is
    ///         centred on sample ``t * hop_length``. Default True.
    ///     pad_mode (str): ``"constant"`` (zeros, default), ``"reflect"``,
    ///         ``"edge"`` or ``"symmetric"``.
    ///
    /// Returns:
    ///     numpy.ndarray[complex64 | complex128]: shape ``(channels, 1 + n_fft // 2, n_frames)``;
    ///     complex64 for uint8, int16, int24 and float32 audio, complex128 for
    ///     int32 and float64 audio.
    ///     Mono audio gives a leading axis of length 1 (librosa returns
    ///     ``(1 + n_fft // 2, n_frames)`` for 1-D input; index ``[0]`` to get it).
    ///
    /// Raises:
    ///     ValueError: For invalid parameters, or ``center=False`` with a signal
    ///         shorter than ``n_fft``.
    #[pyo3(
        signature = (*, n_fft = Int(2048), hop_length = None, win_length = None, window = WindowArg(Window::Hann), center = true, pad_mode = "constant"),
        text_signature = "($self, *, n_fft=2048, hop_length=None, win_length=None, window='hann', center=True, pad_mode='constant')"
    )]
    fn stft_matrix<'py>(
        &self,
        py: Python<'py>,
        n_fft: Int,
        hop_length: Option<Int>,
        win_length: Option<Int>,
        window: WindowArg,
        center: bool,
        pad_mode: &str,
    ) -> PyResult<Bound<'py, PyAny>> {
        let cfg = stft_config(n_fft, hop_length, win_length, window, center, pad_mode)?;
        dispatch_array_detached!(self, py, |audio| audio.stft_matrix(&cfg))
    }

    /// ``stft_matrix`` of float audio written into ``out``, which needs no memory for the
    /// matrix.
    ///
    /// Args:
    ///     out (numpy.ndarray): Of ``complex64`` for float32 audio and ``complex128`` for
    ///         float64 audio, shape ``(1 + n_fft // 2, frames)`` for mono audio or
    ///         ``(channels, 1 + n_fft // 2, frames)``, with at least as many frames as the
    ///         result. Only the first frames that the result needs are written.
    ///     ``n_fft``, ``hop_length``, ``win_length``, window, center, ``pad_mode``: As ``stft_matrix``.
    ///
    /// Returns:
    ///     int: The number of frames written.
    #[pyo3(
        signature = (out, *, n_fft = Int(2048), hop_length = None, win_length = None, window = WindowArg(Window::Hann), center = true, pad_mode = "constant"),
        text_signature = "($self, out, *, n_fft=2048, hop_length=None, win_length=None, window='hann', center=True, pad_mode='constant')"
    )]
    #[allow(clippy::too_many_arguments)]
    fn _stft_matrix_into<'py>(
        &self,
        py: Python<'py>,
        out: &Bound<'py, PyAny>,
        n_fft: Int,
        hop_length: Option<Int>,
        win_length: Option<Int>,
        window: WindowArg,
        center: bool,
        pad_mode: &str,
    ) -> PyResult<usize> {
        let cfg = stft_config(n_fft, hop_length, win_length, window, center, pad_mode)?;
        macro_rules! write_into {
            ($a:expr, $F:ty) => {
                $a.with_view(py, |audio| {
                    let array = out.cast::<PyArrayDyn<Complex<$F>>>()?;
                    let mut guard = array.try_readwrite()?;
                    let mut view = guard.as_array_mut();
                    if view.ndim() == 2 {
                        view.insert_axis_inplace(Axis(0));
                    }
                    let channels = audio.num_channels().get() as usize;
                    let frames = stft_n_frames(audio.samples_per_channel().get(), &cfg)
                        .map_err(audio_err_to_py)?;
                    let expected = [channels, cfg.n_bins(), frames];
                    let shape = view.shape();
                    if shape.len() != 3 || shape[..2] != expected[..2] || shape[2] < frames {
                        return Err(pyo3::exceptions::PyValueError::new_err(format!(
                            "out: has shape {:?}, expected at least {:?}",
                            array.shape(),
                            if array.ndim() == 2 {
                                &expected[1..]
                            } else {
                                &expected[..]
                            }
                        )));
                    }
                    let view = view
                        .into_dimensionality::<Ix3>()
                        .expect("three dimensions")
                        .slice_move(ndarray::s![.., .., ..frames]);
                    py.detach(|| audio.stft_matrix_into(&cfg, view))
                        .map_err(audio_err_to_py)?;
                    Ok(frames)
                })
            };
        }
        match self.inner() {
            PyAudioDataInner::F32(a) => write_into!(a, f32),
            PyAudioDataInner::F64(a) => write_into!(a, f64),
            _ => Err(pyo3::exceptions::PyTypeError::new_err(
                "out: needs float32 or float64 audio",
            )),
        }
    }

    /// Reconstruct audio from an STFT matrix, equivalent to ``librosa.istft``.
    ///
    /// This is a static method: ``AudioSamples.istft_matrix(D, sample_rate=sr)``.
    ///
    /// Args:
    ///     stft_matrix (array_like): complex array of shape ``(channels, 1 + n_fft // 2, n_frames)``
    ///         or ``(1 + n_fft // 2, n_frames)`` (treated as one channel).
    ///     sample_rate (int): Sample rate of the returned audio (keyword-only).
    ///     hop_length (Optional[int]): ``None`` means ``win_length // 4``.
    ///     win_length (Optional[int]): ``None`` means ``n_fft``.
    ///     n_fft (Optional[int]): ``None`` infers ``2 * (n_bins - 1)``, as librosa.
    ///     window: Window used for analysis (see ``stft_matrix``).
    ///     center (bool): Whether the analysis frames were centred.
    ///     length (Optional[int]): Exact output length per channel; pass the
    ///         original length to undo the centre padding exactly.
    ///     dtype (Optional[SampleType]): Sample type of the returned audio.
    ///         ``None`` (default) gives float32 audio for a complex64 matrix and
    ///         float64 audio for any other input. The inverse transform runs in
    ///         the float type of ``dtype`` (float32 for U8, I16, I24 and F32;
    ///         float64 for I32 and F64); integer output is rounded and saturated.
    ///
    /// Returns:
    ///     AudioSamples: audio of sample type ``dtype``, mono for a single channel.
    ///
    /// Raises:
    ///     ValueError: If the matrix is not 2-D/3-D, the bin count does not match
    ///         ``n_fft``, or parameters are invalid.
    #[staticmethod]
    #[pyo3(
        signature = (stft_matrix, *, sample_rate, hop_length = None, win_length = None, n_fft = None, window = WindowArg(Window::Hann), center = true, length = None, dtype = None),
        text_signature = "(stft_matrix, *, sample_rate, hop_length=None, win_length=None, n_fft=None, window='hann', center=True, length=None, dtype=None)"
    )]
    #[allow(clippy::too_many_arguments)]
    fn istft_matrix(
        py: Python<'_>,
        stft_matrix: &Bound<'_, PyAny>,
        sample_rate: Int,
        hop_length: Option<Int>,
        win_length: Option<Int>,
        n_fft: Option<Int>,
        window: WindowArg,
        center: bool,
        length: Option<Int>,
        dtype: Option<PySampleType>,
    ) -> PyResult<Self> {
        let sr = nzu32_or_err(crate::args::sample_rate_u32(sample_rate)?)?;
        let length = crate::args::opt_positive("length", length)?;
        let target = output_sample_type(dtype, stft_matrix);
        with_sample_type!(target, |T| {
            let d = complex_array_as::<<T as AudioSample>::Float>(stft_matrix, "stft_matrix")?;
            let d3 = to_3d(d.as_array(), "stft_matrix")?;
            let n_bins = d3.dim().1;
            let n_fft = match n_fft {
                Some(n) => n,
                None => Int(2 * (n_bins as i64 - 1)),
            };
            let cfg = stft_config(n_fft, hop_length, win_length, window, center, "constant")?;
            let y = py
                .detach(|| {
                    <AudioSamples<'static, T> as AudioSpectrum>::istft_matrix(&d3, &cfg, sr, length)
                })
                .map_err(audio_err_to_py)?;
            Ok(Self::from_audio_samples(y))
        })
    }

    /// Magnitude spectrogram ``|STFT| ** power`` of every channel.
    ///
    /// Equivalent to ``numpy.abs(librosa.stft(y, ...)) ** power``.
    ///
    /// Args:
    ///     n_fft, hop_length, win_length, window, center, pad_mode: As ``stft_matrix``.
    ///     power (float): Exponent; 1.0 (default) gives magnitude, 2.0 power.
    ///
    /// Returns:
    ///     numpy.ndarray[float32 | float64]: shape ``(channels, 1 + n_fft // 2, n_frames)``;
    ///     float32 for uint8, int16, int24 and float32 audio, float64 for int32
    ///     and float64 audio.
    ///
    /// Raises:
    ///     ValueError: For invalid parameters or a non-positive power.
    #[pyo3(
        signature = (*, n_fft = Int(2048), hop_length = None, win_length = None, window = WindowArg(Window::Hann), center = true, pad_mode = "constant", power = 1.0),
        text_signature = "($self, *, n_fft=2048, hop_length=None, win_length=None, window='hann', center=True, pad_mode='constant', power=1.0)"
    )]
    #[allow(clippy::too_many_arguments)]
    fn spectrogram_matrix<'py>(
        &self,
        py: Python<'py>,
        n_fft: Int,
        hop_length: Option<Int>,
        win_length: Option<Int>,
        window: WindowArg,
        center: bool,
        pad_mode: &str,
        power: f64,
    ) -> PyResult<Bound<'py, PyAny>> {
        let cfg = stft_config(n_fft, hop_length, win_length, window, center, pad_mode)?;
        dispatch_array_detached!(self, py, |audio| audio.spectrogram_matrix(&cfg, power))
    }

    /// Mel spectrogram of every channel, equivalent to
    /// ``librosa.feature.melspectrogram(y=..., sr=self.sample_rate, ...)``.
    ///
    /// Args:
    ///     n_fft (int): Default 2048.
    ///     hop_length (int): Default 512 (librosa's melspectrogram default).
    ///     win_length, window, center, pad_mode: As ``stft_matrix``.
    ///     power (float): Exponent of the STFT magnitude; 2.0 (default) for power,
    ///         1.0 for magnitude.
    ///     n_mels (int): Number of mel bands. Default 128.
    ///     fmin (float): Lowest frequency (Hz). Default 0.
    ///     fmax (Optional[float]): Highest frequency (Hz); ``None`` = sr / 2.
    ///     htk (bool): HTK mel scale instead of Slaney (default False).
    ///     norm (None | "slaney" | float): Filter normalisation. Default "slaney".
    ///
    /// Returns:
    ///     numpy.ndarray[float32 | float64]: shape ``(channels, n_mels, n_frames)``;
    ///     float32 for uint8, int16, int24 and float32 audio, float64 for int32
    ///     and float64 audio.
    ///
    /// Raises:
    ///     ValueError: For invalid parameters.
    #[pyo3(
        signature = (*, n_fft = Int(2048), hop_length = Int(512), win_length = None, window = WindowArg(Window::Hann), center = true, pad_mode = "constant", power = 2.0, n_mels = Int(128), fmin = 0.0, fmax = None, htk = false, norm = MelNormArg(audio_samples::MelFilterNorm::Slaney)),
        text_signature = "($self, *, n_fft=2048, hop_length=512, win_length=None, window='hann', center=True, pad_mode='constant', power=2.0, n_mels=128, fmin=0.0, fmax=None, htk=False, norm='slaney')"
    )]
    #[allow(clippy::too_many_arguments)]
    fn melspectrogram<'py>(
        &self,
        py: Python<'py>,
        n_fft: Int,
        hop_length: Int,
        win_length: Option<Int>,
        window: WindowArg,
        center: bool,
        pad_mode: &str,
        power: f64,
        n_mels: Int,
        fmin: f64,
        fmax: Option<f64>,
        htk: bool,
        norm: MelNormArg,
    ) -> PyResult<Bound<'py, PyAny>> {
        let cfg = MelSpectrogramConfig {
            stft: stft_config(
                n_fft,
                Some(hop_length),
                win_length,
                window,
                center,
                pad_mode,
            )?,
            mel: mel_config(n_mels, fmin, fmax, htk, norm)?,
            power,
        };
        dispatch_array_detached!(self, py, |audio| audio.melspectrogram(&cfg))
    }

    /// Mel-frequency cepstral coefficients of every channel, equivalent to
    /// ``librosa.feature.mfcc(y=..., sr=self.sample_rate, ...)``.
    ///
    /// The pipeline is ``melspectrogram`` -> ``power_to_db(ref, amin, top_db)`` ->
    /// DCT along the mel axis -> first ``n_mfcc`` rows -> optional liftering.
    ///
    /// Args:
    ///     n_mfcc (int): Number of coefficients. Default 20.
    ///     dct_type (int): 1, 2 (default) or 3.
    ///     norm (Optional[str]): DCT normalisation, ``"ortho"`` (default) or None.
    ///     lifter (float): Liftering coefficient (0 disables). Default 0.
    ///     mel_norm (None | "slaney" | float): Mel filter normalisation (librosa's
    ///         ``mel_norm``). Default "slaney".
    ///     n_fft, hop_length (default 512), win_length, window, center, pad_mode,
    ///     power, n_mels, fmin, fmax, htk: Passed to the mel spectrogram, as in
    ///         ``melspectrogram``.
    ///     ref (float | "max" | numpy.max): dB reference. Default 1.0.
    ///     amin (float): dB floor before the log. Default 1e-10.
    ///     top_db (Optional[float]): Dynamic range floor in dB below the peak
    ///         (per channel). Default None (no floor); librosa's default is 80.
    ///
    /// Returns:
    ///     numpy.ndarray[float32 | float64]: shape ``(channels, n_mfcc, n_frames)``;
    ///     float32 for uint8, int16, int24 and float32 audio, float64 for int32
    ///     and float64 audio.
    ///
    /// Raises:
    ///     ValueError: For invalid parameters (including ``n_mfcc > n_mels``,
    ///         which librosa silently truncates).
    #[pyo3(
        signature = (*, n_mfcc = Int(20), dct_type = Int(2), norm = Some("ortho"), lifter = 0.0, mel_norm = MelNormArg(audio_samples::MelFilterNorm::Slaney), n_fft = Int(2048), hop_length = Int(512), win_length = None, window = WindowArg(Window::Hann), center = true, pad_mode = "constant", power = 2.0, n_mels = Int(128), fmin = 0.0, fmax = None, htk = false, r#ref = DbRefArg(audio_samples::DbReference::Value(1.0)), amin = 1e-10, top_db = None),
        text_signature = "($self, *, n_mfcc=20, dct_type=2, norm='ortho', lifter=0.0, mel_norm='slaney', n_fft=2048, hop_length=512, win_length=None, window='hann', center=True, pad_mode='constant', power=2.0, n_mels=128, fmin=0.0, fmax=None, htk=False, ref=1.0, amin=1e-10, top_db=None)"
    )]
    #[allow(clippy::too_many_arguments)]
    fn mfcc_frames<'py>(
        &self,
        py: Python<'py>,
        n_mfcc: Int,
        dct_type: Int,
        norm: Option<&str>,
        lifter: f64,
        mel_norm: MelNormArg,
        n_fft: Int,
        hop_length: Int,
        win_length: Option<Int>,
        window: WindowArg,
        center: bool,
        pad_mode: &str,
        power: f64,
        n_mels: Int,
        fmin: f64,
        fmax: Option<f64>,
        htk: bool,
        r#ref: DbRefArg,
        amin: f64,
        top_db: Option<f64>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let mut cfg = mfcc_cepstrum_config(n_mfcc, dct_type, norm, lifter)?;
        cfg.mel_spectrogram = MelSpectrogramConfig {
            stft: stft_config(
                n_fft,
                Some(hop_length),
                win_length,
                window,
                center,
                pad_mode,
            )?,
            mel: mel_config(n_mels, fmin, fmax, htk, mel_norm)?,
            power,
        };
        cfg.db = DbConfig::power()
            .with_reference(r#ref.0)
            .with_amin(amin)
            .with_top_db(top_db);
        dispatch_array_detached!(self, py, |audio| audio.mfcc_frames(&cfg))
    }
}
