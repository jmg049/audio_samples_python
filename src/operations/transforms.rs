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
    /// Args:
    ///     n_fft (int): FFT size in samples; must be greater than zero.
    ///
    /// Returns:
    ///     numpy.ndarray: Complex spectrum shaped (channels, frequency_bins).
    ///
    /// Raises:
    ///     ValueError: If n_fft is zero.
    ///     AudioError: If the FFT computation fails.
    #[pyo3(signature = (n_fft: "int"), text_signature = "($self, n_fft: int) -> numpy.ndarray")]
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

    /// Compute the real-valued Fast Fourier Transform (RFFT).
    ///
    /// Args:
    ///     n_fft (int): FFT size in samples; must be greater than zero.
    ///
    /// Returns:
    ///     numpy.ndarray: Real-valued spectrum shaped (channels, frequency_bins).
    ///
    /// Raises:
    ///     ValueError: If n_fft is zero.
    ///     AudioError: If the RFFT computation fails.
    #[pyo3(signature = (n_fft: "int"), text_signature = "($self, n_fft: int) -> numpy.ndarray")]
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
    ///     params (StftParams): Configuration controlling window length, hop size, and window type.
    ///
    /// Returns:
    ///     StftResult: Object containing complex spectrum, frequency bins, and time stamps.
    ///
    /// Raises:
    ///     AudioError: If the STFT computation fails.
    #[pyo3(signature = (params: "StftParams"), text_signature = "($self, params: StftParams) -> StftResult")]
    fn stft(&self, py: Python<'_>, params: PyStftParams) -> PyResult<PyStftResult> {
        dispatch_with_view!(self, py, |audio| {
            let result = audio.stft(&params.inner).map_err(audio_err_to_py)?;
            Ok(PyStftResult::from(result))
        })
    }

    /// Invert a Short-Time Fourier Transform result.
    ///
    /// Args:
    ///     stft_result (StftResult): Output from `stft` to be reconstructed.
    ///
    /// Returns:
    ///     AudioSamples: Reconstructed time-domain signal.
    ///
    /// Raises:
    ///     AudioError: If reconstruction fails.
    #[classmethod]
    #[pyo3(signature = (stft_result: "StftResult"), text_signature = "($cls, stft_result: StftResult) -> AudioSamples")]
    fn istft(
        _cls: &Bound<'_, PyType>,
        _py: Python<'_>,
        stft_result: PyStftResult,
    ) -> PyResult<Self> {
        // Use f64 for reconstruction
        let reconstructed = <AudioSamples<f64> as AudioTransforms>::istft(stft_result.into_inner())
            .map_err(Into::into)
            .map_err(audio_err_to_py)?;

        Ok(Self::from_audio_samples(reconstructed))
    }

    /// Estimate the power spectral density across overlapping windows.
    ///
    /// Args:
    ///     window_size (int): STFT window length in samples; must be greater than zero.
    ///     overlap (float): Fractional overlap between successive windows.
    ///
    /// Returns:
    ///     Psd: Power spectral density carrying both the frequency axis and the
    ///         power-per-Hz density, averaged over time.
    ///
    /// Raises:
    ///     ValueError: If window_size is zero.
    ///     AudioError: If the PSD computation fails.
    #[pyo3(signature = (window_size: "int"=2048, overlap: "float"=0.5), text_signature = "($self, window_size: int = 2048, overlap: float = 0.5) -> Psd")]
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
    ///     spec_params (SpectrogramParams): Parameters controlling the underlying STFT.
    ///     mel_params (MelParams): Configuration for the mel filterbank.
    ///     log_params (Optional[LogParams]): Optional parameters for decibel conversion.
    ///
    /// Returns:
    ///     numpy.ndarray: Mel spectrogram shaped (mel_bins, frames).
    ///
    /// Raises:
    ///     AudioError: If spectrogram computation fails.
    #[pyo3(signature = (spec_params: "SpectrogramParams", mel_params: "MelParams", log_params: "Optional[LogParams]"=None), text_signature = "($self, spec_params: SpectrogramParams, mel_params: MelParams, log_params: Optional[LogParams] = None) -> numpy.ndarray")]
    fn mel_spectrogram<'py>(
        &self,
        py: Python<'py>,
        spec_params: PySpectrogramParams,
        mel_params: PyMelParams,
        log_params: Option<PyLogParams>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let spec_params: SpectrogramParams = spec_params.into();
        let mel_params: MelParams = mel_params.into();
        let log_params_ref = log_params.as_ref().map(|lp| lp.as_inner());

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
    ///     spec_params (SpectrogramParams): Parameters controlling the underlying STFT.
    ///     erb_params (ErbParams): Configuration for the ERB filterbank.
    ///     log_params (Optional[LogParams]): Optional parameters for decibel conversion.
    ///
    /// Returns:
    ///     numpy.ndarray: Gammatone spectrogram shaped (bands, frames).
    ///
    /// Raises:
    ///     AudioError: If spectrogram computation fails.
    #[pyo3(signature = (spec_params: "SpectrogramParams", erb_params: "ErbParams", log_params: "Optional[LogParams]"=None), text_signature = "($self, spec_params: SpectrogramParams, erb_params: ErbParams, log_params: Optional[LogParams] = None) -> numpy.ndarray")]
    fn gammatone_spectrogram<'py>(
        &self,
        py: Python<'py>,
        spec_params: PySpectrogramParams,
        erb_params: PyErbParams,
        log_params: Option<PyLogParams>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let spec_params: SpectrogramParams = spec_params.into();
        let erb_params: ErbParams = erb_params.into_inner();
        let log_params_ref = log_params.as_ref().map(|lp| lp.as_inner());

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
    ///     stft_params (StftParams): Parameters used to generate the STFT.
    ///     n_mels (int): Number of mel bands; must be greater than zero.
    ///     mfcc_params (MfccParams): Configuration for the cepstral projection.
    ///
    /// Returns:
    ///     numpy.ndarray: MFCC matrix shaped (coefficients, frames).
    ///
    /// Raises:
    ///     ValueError: If n_mels is zero.
    ///     AudioError: If the MFCC computation fails.
    #[pyo3(signature = (stft_params: "StftParams", n_mels: "int", mfcc_params: "MfccParams"), text_signature = "($self, stft_params: StftParams, n_mels: int, mfcc_params: MfccParams) -> numpy.ndarray")]
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
    ///     stft_params (StftParams): Parameters used to generate the STFT.
    ///     chroma_params (ChromaParams): Configuration for chroma projection.
    ///
    /// Returns:
    ///     numpy.ndarray: Chromagram shaped (pitch_classes, frames).
    ///
    /// Raises:
    ///     AudioError: If chroma computation fails.
    #[pyo3(signature = (stft_params: "StftParams", chroma_params: "ChromaParams"), text_signature = "($self, stft_params: StftParams, chroma_params: ChromaParams) -> numpy.ndarray")]
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
    /// Args:
    ///     other (AudioSamples): Signal to convolve with. Must be mono and share
    ///         this signal's data type.
    ///
    /// Returns:
    ///     AudioSamples: The convolved mono signal.
    ///
    /// Raises:
    ///     TypeError: If the two signals have different data types.
    ///     AudioError: If either signal is multi-channel or the FFT fails.
    #[pyo3(signature = (other: "AudioSamples"), text_signature = "($self, other: AudioSamples) -> AudioSamples")]
    fn convolve(&self, py: Python<'_>, other: &PyAudioSamples) -> PyResult<Self> {
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
    ///     other (AudioSamples): The denominator signal. Must be mono and share
    ///         this signal's data type.
    ///     regularization (float): Non-negative regularisation factor.
    ///
    /// Returns:
    ///     AudioSamples: The deconvolved mono signal.
    ///
    /// Raises:
    ///     TypeError: If the two signals have different data types.
    ///     AudioError: If either signal is multi-channel or the FFT fails.
    #[pyo3(signature = (other: "AudioSamples", regularization: "float" = 0.0), text_signature = "($self, other: AudioSamples, regularization: float = 0.0) -> AudioSamples")]
    fn deconvolve(
        &self,
        py: Python<'_>,
        other: &PyAudioSamples,
        regularization: f64,
    ) -> PyResult<Self> {
        convolve_or_deconvolve(self, other, py, ConvOp::Deconvolve(regularization))
    }

    /// Compute the Constant-Q Transform (CQT) of a mono signal.
    ///
    /// Args:
    ///     params (CqtParams): CQT configuration (bins per octave, number of
    ///         octaves, minimum frequency).
    ///     hop_size (int): Hop size in samples between successive frames; must be
    ///         greater than zero.
    ///
    /// Returns:
    ///     numpy.ndarray: Complex CQT coefficients shaped (frequency_bins, time_frames).
    ///
    /// Raises:
    ///     ValueError: If hop_size is zero.
    ///     AudioError: If the signal is multi-channel or the computation fails.
    #[pyo3(signature = (params: "CqtParams", hop_size: "int"), text_signature = "($self, params: CqtParams, hop_size: int) -> numpy.ndarray")]
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
    ///     complex_spect (numpy.ndarray): Complex 2-D STFT or FFT matrix.
    ///     power (Optional[int]): Exponent applied to the magnitude values.
    ///         Defaults to 1 (raw magnitude); must be greater than zero if given.
    ///
    /// Returns:
    ///     tuple[numpy.ndarray, numpy.ndarray]: The real-valued magnitude matrix
    ///         and the complex unit-magnitude phase matrix.
    ///
    /// Raises:
    ///     ValueError: If power is provided and is zero.
    #[staticmethod]
    #[pyo3(signature = (complex_spect: "numpy.ndarray", power: "Optional[int]" = None), text_signature = "(complex_spect: numpy.ndarray, power: Optional[int] = None) -> tuple[numpy.ndarray, numpy.ndarray]")]
    fn magphase<'py>(
        py: Python<'py>,
        complex_spect: PyReadonlyArray2<'py, Complex<f64>>,
        power: Option<usize>,
    ) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<Complex<f64>>>)> {
        let power = power.map(nzu_or_err).transpose()?;
        let spect = complex_spect.as_array().to_owned();
        let (mag, phase) = <AudioSamples<f64> as AudioTransforms>::magphase(&spect, power);
        Ok((mag.into_pyarray(py), phase.into_pyarray(py)))
    }
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
        let owned_a: AudioSamples<'static, T> = a.with_view(py, |audio| audio.into_owned());
        let owned_b: AudioSamples<'static, T> = b.with_view(py, |audio| audio.into_owned());
        let result = match op {
            ConvOp::Convolve => owned_a.convolve(&owned_b),
            ConvOp::Deconvolve(reg) => owned_a.deconvolve(&owned_b, reg),
        }
        .map_err(audio_err_to_py)?;
        Ok(PyAudioSamples::from_audio_samples(result))
    }

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
