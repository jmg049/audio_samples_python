use audio_samples::{AudioSamples, AudioTransforms};
use num_complex::Complex;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::{prelude::*, types::PyType};
use spectrograms::{
    ChromaParams, CqtParams, ErbParams, MelParams, MfccParams, SpectrogramParams, StftParams,
    python::{
        PyChromaParams, PyCqtParams, PyErbParams, PyLogParams, PyMelParams, PyMfccParams,
        PySpectrogramParams, PyStftParams, PyStftResult,
    },
};

use crate::{
    PyAudioDataInner, PyAudioSamples, TypedAudioSamples, audio_err_to_py, dispatch_with_view,
    nzu_or_err, types::PyPsd,
};
use audio_samples::StandardSample;
use numpy::Element;

#[pymethods]
impl PyAudioSamples {
    /// Compute the Fast Fourier Transform (FFT) of the signal.
    ///
    /// The spectrum is **one-sided**: only the ``n_fft // 2 + 1`` non-negative
    /// frequency bins are returned, as ``numpy.fft.rfft(x, n_fft)`` per channel
    /// (the signal is zero-padded to ``n_fft``, which must be at least the
    /// signal length; longer input raises an error). Bin ``k`` is at
    /// ``k * sample_rate / n_fft`` Hz.
    ///
    /// Args:
    ///
    ///     n_fft (int): FFT size in samples; must be greater than zero.
    ///
    /// Returns:
    ///
    ///     numpy.ndarray: Complex spectrum shaped (channels, n_fft // 2 + 1).
    ///
    /// Raises:
    ///
    ///     ValueError: If n_fft is zero.
    ///     AudioError: If the FFT computation fails.
    #[pyo3(signature = (n_fft: "int"))]
    fn fft<'py>(
        &self,
        py: Python<'py>,
        n_fft: usize,
    ) -> PyResult<Bound<'py, PyArray2<Complex<f64>>>> {
        dispatch_with_view!(self, py, |audio| {
            let n_fft_nz = nzu_or_err(n_fft)?;
            let result = audio.fft(n_fft_nz).map_err(audio_err_to_py)?;
            Ok(result.into_pyarray(py))
        })
    }

    /// Compute the magnitude of the one-sided FFT.
    ///
    /// Equal to ``numpy.abs(numpy.fft.rfft(x, n_fft))`` per channel: real
    /// magnitudes of the ``n_fft // 2 + 1`` non-negative frequency bins (use
    /// :meth:`fft` for the complex values).
    ///
    /// Args:
    ///
    ///     n_fft (int): FFT size in samples; must be greater than zero.
    ///
    /// Returns:
    ///
    ///     numpy.ndarray: Magnitude spectrum shaped (channels, n_fft // 2 + 1).
    ///
    /// Raises:
    ///
    ///     ValueError: If n_fft is zero.
    ///     AudioError: If the RFFT computation fails.
    #[pyo3(signature = (n_fft: "int"))]
    fn rfft<'py>(&self, py: Python<'py>, n_fft: usize) -> PyResult<Bound<'py, PyArray2<f64>>> {
        dispatch_with_view!(self, py, |audio| {
            let n_fft_nz = nzu_or_err(n_fft)?;
            let result = audio.rfft(n_fft_nz).map_err(audio_err_to_py)?;
            Ok(result.into_pyarray(py))
        })
    }
    /// Compute the Short-Time Fourier Transform (STFT).
    ///
    /// Args:
    ///
    ///     params (StftParams): Configuration controlling window length, hop size, and window type.
    ///
    /// Returns:
    ///
    ///     StftResult: Object containing complex spectrum, frequency bins, and time stamps.
    ///
    /// Raises:
    ///
    ///     AudioError: If the STFT computation fails.
    #[pyo3(signature = (params: "StftParams"))]
    fn stft(&self, py: Python<'_>, params: &Bound<'_, PyStftParams>) -> PyResult<PyStftResult> {
        let params = params.borrow().clone();
        dispatch_with_view!(self, py, |audio| {
            let result = audio.stft(&params.inner).map_err(audio_err_to_py)?;
            Ok(PyStftResult::from_stft_result(py, result))
        })
    }

    /// Invert a Short-Time Fourier Transform result.
    ///
    /// Args:
    ///
    ///     stft_result (StftResult): Output from `stft` to be reconstructed.
    ///
    /// Returns:
    ///
    ///     AudioSamples: Reconstructed time-domain signal.
    ///
    /// Raises:
    ///
    ///     AudioError: If reconstruction fails.
    #[classmethod]
    #[pyo3(signature = (stft_result: "StftResult"))]
    fn istft(
        _cls: &Bound<'_, PyType>,
        py: Python<'_>,
        stft_result: PyStftResult,
    ) -> PyResult<Self> {
        // Reconstruction only supports f64; reject StftResults computed at f32 precision.
        let stft_result = stft_result.try_into_stft_result::<f64>(py).map_err(|_| {
            pyo3::exceptions::PyValueError::new_err(
                "istft only supports a float64-precision StftResult (dtype='float64')",
            )
        })?;

        let reconstructed =
            <AudioSamples<f64> as AudioTransforms>::istft(stft_result).map_err(audio_err_to_py)?;

        Ok(Self::from_audio_samples(reconstructed))
    }

    /// Estimate the power spectral density across overlapping windows.
    ///
    /// Args:
    ///
    ///     window_size (int): STFT window length in samples; must be greater than zero.
    ///     overlap (float): Fractional overlap between successive windows.
    ///
    /// Returns:
    ///
    ///     Psd: Power spectral density carrying both the frequency axis and the
    ///         power-per-Hz density, averaged over time.
    ///
    /// Raises:
    ///
    ///     ValueError: If window_size is zero.
    ///     AudioError: If the PSD computation fails.
    #[pyo3(signature = (window_size: "int"=2048, overlap: "float"=0.5))]
    fn power_spectral_density(
        &self,
        py: Python<'_>,
        window_size: usize,
        overlap: f64,
    ) -> PyResult<PyPsd> {
        let window_size = nzu_or_err(window_size)?;

        dispatch_with_view!(self, py, |audio| {
            let psd = audio
                .power_spectral_density(window_size, overlap)
                .map_err(audio_err_to_py)?;
            Ok(PyPsd::from(psd))
        })
    }

    /// Compute a mel-frequency spectrogram.
    ///
    /// Args:
    ///
    ///     spec_params (SpectrogramParams): Parameters controlling the underlying STFT.
    ///     mel_params (MelParams): Configuration for the mel filterbank.
    ///     log_params (Optional[LogParams]): Optional parameters for decibel conversion.
    ///
    /// Returns:
    ///
    ///     numpy.ndarray: Mel spectrogram shaped (mel_bins, frames). Without
    ///         ``log_params`` the values are ``sqrt(M @ |X|**2)`` (the square root
    ///         of the mel power spectrogram, following the ``spectrograms`` crate),
    ///         which is neither librosa's default (``power=2``, ``M @ |X|**2``) nor
    ///         ``power=1`` (``M @ |X|``). For librosa-compatible output use
    ///         ``melspectrogram()``.
    ///
    /// Raises:
    ///
    ///     AudioError: If spectrogram computation fails.
    ///
    /// Notes:
    ///
    ///     With ``M`` the mel filterbank and ``|X|**2`` the STFT power:
    ///
    ///     - without ``log_params`` the result is ``sqrt(M @ |X|**2)``, the
    ///       square root of the mel *power* (not ``M @ |X|``);
    ///     - with ``log_params`` it is ``max(10*log10(M @ |X|**2), floor_db)``.
    ///       ``floor_db`` is an **absolute** floor in dB re 1 (e.g. ``-80``
    ///       clips values below -80 dB), not a range below the maximum like
    ///       librosa's ``top_db``.
    ///
    ///     The mel scale is Slaney's (linear below 1 kHz, logarithmic above;
    ///     librosa ``htk=False``). The filters are **not** normalised by default
    ///     (``MelParams(norm=None)``, triangles of peak 1); pass
    ///     ``norm="slaney"`` for librosa's default area normalisation. So
    ///     ``librosa.filters.mel(sr=..., n_fft=..., n_mels=..., norm=None)``
    ///     reproduces ``M``.
    #[pyo3(signature = (spec_params: "SpectrogramParams", mel_params: "MelParams", log_params: "Optional[LogParams]"=None))]
    fn mel_spectrogram<'py>(
        &self,
        py: Python<'py>,
        spec_params: PySpectrogramParams,
        mel_params: PyMelParams,
        log_params: Option<PyLogParams>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let spec_params: SpectrogramParams = spec_params.into();
        check_spec_sample_rate(self, py, &spec_params)?;
        let mel_params: MelParams = mel_params.into();
        let log_params_ref = log_params.as_ref().map(PyLogParams::as_inner);

        dispatch_with_view!(self, py, |audio| {
            use spectrograms::Decibels;

            let result = if log_params_ref.is_some() {
                audio
                    .mel_spectrogram::<Decibels>(&spec_params, &mel_params, log_params_ref)
                    .map_err(audio_err_to_py)?
                    .into_data()
            } else {
                audio
                    .mel_mag_spectrogram(&spec_params, &mel_params)
                    .map_err(audio_err_to_py)?
                    .into_data()
            };

            Ok(result.into_pyarray(py))
        })
    }

    /// Compute a gammatone spectrogram using ERB-scaled filters.
    ///
    /// Args:
    ///
    ///     spec_params (SpectrogramParams): Parameters controlling the underlying STFT.
    ///     erb_params (ErbParams): Configuration for the ERB filterbank.
    ///     log_params (Optional[LogParams]): Optional parameters for decibel conversion.
    ///
    /// Returns:
    ///
    ///     numpy.ndarray: Gammatone spectrogram shaped (bands, frames).
    ///
    /// Raises:
    ///
    ///     AudioError: If spectrogram computation fails.
    ///
    /// Notes:
    ///
    ///     ``log_params.floor_db`` is an absolute floor in dB re 1, not a
    ///     range below the maximum (librosa's ``top_db``).
    #[pyo3(signature = (spec_params: "SpectrogramParams", erb_params: "ErbParams", log_params: "Optional[LogParams]"=None))]
    fn gammatone_spectrogram<'py>(
        &self,
        py: Python<'py>,
        spec_params: PySpectrogramParams,
        erb_params: PyErbParams,
        log_params: Option<PyLogParams>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let spec_params: SpectrogramParams = spec_params.into();
        check_spec_sample_rate(self, py, &spec_params)?;
        let erb_params: ErbParams = erb_params.into_inner();
        let log_params_ref = log_params.as_ref().map(PyLogParams::as_inner);

        dispatch_with_view!(self, py, |audio| {
            let result = if let Some(log_params_ref) = log_params_ref {
                audio
                    .gammatone_db_spectrogram(&spec_params, &erb_params, log_params_ref)
                    .map_err(audio_err_to_py)?
                    .into_data()
            } else {
                audio
                    .gammatone_magnitude_spectrogram(&spec_params, &erb_params)
                    .map_err(audio_err_to_py)?
                    .into_data()
            };

            Ok(result.into_pyarray(py))
        })
    }

    /// Compute mel-frequency cepstral coefficients (MFCCs).
    ///
    /// Args:
    ///
    ///     stft_params (StftParams): Parameters used to generate the STFT.
    ///     n_mels (int): Number of mel bands; must be greater than zero.
    ///     mfcc_params (MfccParams): Configuration for the cepstral projection.
    ///
    /// Returns:
    ///
    ///     numpy.ndarray: MFCC matrix shaped (coefficients, frames). This follows
    ///         the ``spectrograms`` crate's MFCC (not librosa's); for
    ///         ``librosa.feature.mfcc``-compatible output use ``mfcc_frames()``.
    ///
    /// Raises:
    ///
    ///     ValueError: If n_mels is zero.
    ///     AudioError: If the MFCC computation fails.
    ///
    /// Notes:
    ///
    ///     Pipeline: mel power spectrogram (Slaney scale, ``n_mels`` bands from
    ///     0 Hz to ``sample_rate / 2``, no filter normalisation), then
    ///     ``L = 10*log10(max(., 1e-8))``, then an **unnormalised** DCT-II
    ///     ``c[q] = sum_b L[b] * cos(pi*q*(b + 0.5)/n_mels)`` keeping the first
    ///     ``n_mfcc`` coefficients, then the HTK sinusoidal lifter
    ///     ``1 + (22/2)*sin(pi*q/22)`` (lifter 22).
    ///
    ///     Compared with ``librosa.feature.mfcc`` (orthonormal DCT, no lifter,
    ///     ``top_db=80``): ``ours = librosa * s_q * lifter_q`` with
    ///     ``s_0 = sqrt(n_mels)`` and ``s_q = sqrt(n_mels / 2)`` for ``q >= 1``.
    #[pyo3(signature = (stft_params: "StftParams", n_mels: "int", mfcc_params: "MfccParams"))]
    fn mfcc<'py>(
        &self,
        py: Python<'py>,
        stft_params: PyStftParams,
        n_mels: usize,
        mfcc_params: PyMfccParams,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let stft_params: StftParams = stft_params.into();
        let mfcc_params: MfccParams = mfcc_params.into();
        let n_mels = nzu_or_err(n_mels)?;

        dispatch_with_view!(self, py, |audio| {
            let mfcc = audio
                .mfcc(&stft_params, n_mels, &mfcc_params)
                .map_err(audio_err_to_py)?;
            let mfcc = PyArray2::from_owned_array(py, mfcc.data);
            Ok(mfcc)
        })
    }

    /// Compute chromagram (pitch-class energy) features.
    ///
    /// Args:
    ///
    ///     stft_params (StftParams): Parameters used to generate the STFT.
    ///     chroma_params (ChromaParams): Configuration for chroma projection.
    ///
    /// Returns:
    ///
    ///     numpy.ndarray: Chromagram shaped (pitch_classes, frames).
    ///
    /// Raises:
    ///
    ///     AudioError: If chroma computation fails.
    #[pyo3(signature = (stft_params: "StftParams", chroma_params: "ChromaParams"))]
    fn chroma<'py>(
        &self,
        py: Python<'py>,
        stft_params: PyStftParams,
        chroma_params: PyChromaParams,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let stft_params: StftParams = stft_params.into();
        let chroma_params: ChromaParams = chroma_params.into();

        dispatch_with_view!(self, py, |audio| {
            let chromagram = audio
                .chromagram(&stft_params, &chroma_params)
                .map_err(audio_err_to_py)?;
            Ok(chromagram.data.into_pyarray(py))
        })
    }

    /// Compute the linear convolution of this signal with another.
    ///
    /// Both signals are converted to f64 internally. The result is mono and
    /// carries this signal's sample rate, with length
    /// ``len(self) + len(other) - 1``.
    ///
    /// This is the *full* convolution (``numpy.convolve(x, h, 'full')``). For
    /// FIR filtering with a coefficient array, :meth:`apply_filter` returns
    /// only the *valid* part, ``N - M + 1`` samples
    /// (``numpy.convolve(x, h, 'valid')``).
    ///
    /// Args:
    ///
    ///     other (AudioSamples): Signal to convolve with. Must be mono and share
    ///         this signal's data type and sample rate.
    ///
    /// Returns:
    ///
    ///     AudioSamples: The convolved mono signal.
    ///
    /// Raises:
    ///
    ///     AudioParameterError: If the two signals have different sample rates.
    ///     TypeError: If the two signals have different data types.
    ///     AudioError: If either signal is multi-channel or the FFT fails.
    #[pyo3(signature = (other: "AudioSamples"))]
    fn convolve(&self, py: Python<'_>, other: &Self) -> PyResult<Self> {
        convolve_or_deconvolve(self, other, py, ConvOp::Convolve)
    }

    /// Recover a system response by regularised spectral-division deconvolution.
    ///
    /// Solves ``self = denominator (*) h`` for ``h``. The ``regularization``
    /// term (>= 0) is scaled by the denominator's peak power to stabilise the
    /// division near spectral nulls. The result is mono at this signal's sample
    /// rate.
    ///
    /// Args:
    ///
    ///     other (AudioSamples): The denominator signal. Must be mono and share
    ///         this signal's data type.
    ///     regularization (float): Non-negative regularisation factor.
    ///
    /// Returns:
    ///
    ///     AudioSamples: The deconvolved mono signal.
    ///
    /// Raises:
    ///
    ///     AudioParameterError: If the two signals have different sample rates.
    ///     TypeError: If the two signals have different data types.
    ///     AudioError: If either signal is multi-channel or the FFT fails.
    #[pyo3(signature = (other: "AudioSamples", regularization: "float" = 0.0))]
    fn deconvolve(&self, py: Python<'_>, other: &Self, regularization: f64) -> PyResult<Self> {
        convolve_or_deconvolve(self, other, py, ConvOp::Deconvolve(regularization))
    }

    /// Compute the Constant-Q Transform (CQT) of a mono signal.
    ///
    /// Args:
    ///
    ///     params (CqtParams): CQT configuration (bins per octave, number of
    ///         octaves, minimum frequency).
    ///     hop_size (int): Hop size in samples between successive frames; must be
    ///         greater than zero.
    ///
    /// Returns:
    ///
    ///     numpy.ndarray: Complex CQT coefficients shaped (frequency_bins, time_frames).
    ///
    /// Raises:
    ///
    ///     ValueError: If hop_size is zero.
    ///     AudioError: If the signal is multi-channel or the computation fails.
    #[pyo3(signature = (params: "CqtParams", hop_size: "int"))]
    fn constant_q_transform<'py>(
        &self,
        py: Python<'py>,
        params: PyCqtParams,
        hop_size: usize,
    ) -> PyResult<Bound<'py, PyArray2<Complex<f64>>>> {
        let cqt_params: CqtParams = params.into();
        let hop_size = nzu_or_err(hop_size)?;
        dispatch_with_view!(self, py, |audio| {
            let result = audio
                .constant_q_transform(&cqt_params, hop_size)
                .map_err(audio_err_to_py)?;
            Ok(result.data.into_pyarray(py))
        })
    }

    /// Decompose a complex spectrogram into magnitude and phase.
    ///
    /// Given a complex matrix ``D``, returns ``(magnitude, phase)`` such that
    /// ``D = magnitude * phase`` elementwise, where ``phase`` contains
    /// unit-magnitude complex factors. Bins where the magnitude is zero are
    /// assigned a phase of ``1 + 0j``.
    ///
    /// This is a static method that operates on a complex spectrogram array
    /// (such as the output of ``fft`` or ``constant_q_transform``) rather than on
    /// an ``AudioSamples`` instance.
    ///
    /// Args:
    ///
    ///     complex_spect (numpy.ndarray): Complex 2-D STFT or FFT matrix.
    ///     power (Optional[float]): Exponent applied to the magnitude values, as
    ///         ``numpy.abs(D) ** power``; any number, fractional or negative.
    ///         Defaults to 1 (raw magnitude).
    ///
    /// Returns:
    ///
    ///     tuple[numpy.ndarray, numpy.ndarray]: The real-valued magnitude matrix
    ///         and the complex unit-magnitude phase matrix.
    #[staticmethod]
    #[pyo3(signature = (complex_spect: "numpy.ndarray", power: "Optional[float]" = None))]
    fn magphase<'py>(
        py: Python<'py>,
        complex_spect: &Bound<'py, PyAny>,
        power: Option<f64>,
    ) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)> {
        use audio_samples::operations::transforms::{magphase, magphase_powf};
        // On the borrowed numpy data: no copy of the input.
        if let Ok(d) = complex_spect.extract::<PyReadonlyArray2<'py, Complex<f64>>>() {
            let (mag, phase) = power.map_or_else(
                || magphase(d.as_array(), None),
                |p| magphase_powf(d.as_array(), p),
            );
            return Ok((
                mag.into_pyarray(py).into_any(),
                phase.into_pyarray(py).into_any(),
            ));
        }
        if let Ok(d) = complex_spect.extract::<PyReadonlyArray2<'py, Complex<f32>>>() {
            #[allow(
                clippy::cast_possible_truncation,
                reason = "a float32 spectrogram takes the exponent rounded to f32"
            )]
            let (mag, phase) = power.map_or_else(
                || magphase(d.as_array(), None),
                |p| magphase_powf(d.as_array(), p as f32),
            );
            return Ok((
                mag.into_pyarray(py).into_any(),
                phase.into_pyarray(py).into_any(),
            ));
        }
        Err(pyo3::exceptions::PyTypeError::new_err(
            "complex_spect: expected a 2-D complex64 or complex128 array \
(index [0] for the first channel of stft_matrix)",
        ))
    }
}

/// Raise `AudioParameterError` (a `ValueError`) when `SpectrogramParams` was
/// built for a different sample rate than the audio: its frequency and mel
/// axes would silently be wrong.
fn check_spec_sample_rate(
    audio: &PyAudioSamples,
    py: Python<'_>,
    params: &SpectrogramParams,
) -> PyResult<()> {
    let audio_sr = f64::from(audio.sample_rate(py).get());
    let params_sr = params.sample_rate_hz();
    if (audio_sr - params_sr).abs() > f64::EPSILON * audio_sr.max(1.0) {
        return Err(crate::make_audio_err(
            "AudioParameterError",
            format!(
                "SpectrogramParams.sample_rate ({params_sr} Hz) does not match the audio's \
sample rate ({audio_sr} Hz); build it with SpectrogramParams(stft, sample_rate=audio.sample_rate)"
            ),
            None,
            None,
            pyo3::exceptions::PyValueError::new_err,
        ));
    }
    Ok(())
}

/// Which spectral-division operation to perform in [`convolve_or_deconvolve`].
#[derive(Clone, Copy)]
enum ConvOp {
    Convolve,
    Deconvolve(f64),
}

/// Apply a binary convolution-style operation that requires both operands to be
/// the same concrete `AudioSamples<T>`.
///
/// Mirrors the dual-dispatch used for cross-correlation: both signals are
/// materialised as `AudioSamples` views in the same scope so the borrowing
/// closures do not escape.
fn convolve_or_deconvolve(
    this: &PyAudioSamples,
    other: &PyAudioSamples,
    py: Python<'_>,
    op: ConvOp,
) -> PyResult<PyAudioSamples> {
    fn run<T>(
        a: &TypedAudioSamples<T>,
        b: &TypedAudioSamples<T>,
        py: Python<'_>,
        op: ConvOp,
    ) -> PyResult<PyAudioSamples>
    where
        T: StandardSample + Element,
    {
        // Materialise both views as owned 'static buffers so the borrowed view
        // lifetimes do not entangle the 'static result of convolve/deconvolve.
        let _guards = (a.read_guard(py)?, b.read_guard(py)?);
        let owned_a: AudioSamples<'static, T> = a.borrowed(py).into_owned();
        let owned_b: AudioSamples<'static, T> = b.borrowed(py).into_owned();
        let result = match op {
            ConvOp::Convolve => owned_a.convolve(&owned_b),
            ConvOp::Deconvolve(reg) => owned_a.deconvolve(&owned_b, reg),
        }
        .map_err(audio_err_to_py)?;
        Ok(PyAudioSamples::from_audio_samples(result))
    }

    let op_name = match op {
        ConvOp::Convolve => "convolve",
        ConvOp::Deconvolve(_) => "deconvolve",
    };
    crate::check_rates_match(
        op_name,
        this.sample_rate(py).get(),
        other.sample_rate(py).get(),
    )?;
    match (&this.inner, &other.inner) {
        (PyAudioDataInner::U8(a), PyAudioDataInner::U8(b)) => run(a, b, py, op),
        (PyAudioDataInner::I16(a), PyAudioDataInner::I16(b)) => run(a, b, py, op),
        (PyAudioDataInner::I24(a), PyAudioDataInner::I24(b)) => run(a, b, py, op),
        (PyAudioDataInner::I32(a), PyAudioDataInner::I32(b)) => run(a, b, py, op),
        (PyAudioDataInner::F32(a), PyAudioDataInner::F32(b)) => run(a, b, py, op),
        (PyAudioDataInner::F64(a), PyAudioDataInner::F64(b)) => run(a, b, py, op),
        _ => Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
            "convolution requires both AudioSamples to have the same data type",
        )),
    }
}
