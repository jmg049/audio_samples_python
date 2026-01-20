use audio_samples::{AudioSamples, AudioTransforms};
use num_complex::Complex;
use numpy::{IntoPyArray, PyArray, PyArray1, PyArray2};
use pyo3::{prelude::*, types::PyType};
use spectrograms::{
    ChromaParams, ErbParams, MelParams, MfccParams, SpectrogramParams, StftParams,
    python::{
        PyChromaParams, PyErbParams, PyLogParams, PyMelParams, PyMfccParams, PySpectrogramParams,
        PyStftParams, PyStftResult,
    },
};

use crate::{PyAudioSamples, audio_err_to_py, dispatch_with_view, nzu_or_err};

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
    ///     numpy.ndarray: Power spectral density averaged over time.
    ///
    /// Raises:
    ///     ValueError: If window_size is zero.
    ///     AudioError: If the PSD computation fails.
    #[pyo3(signature = (window_size: "int"=2048, overlap: "float"=0.5), text_signature = "($self, window_size: int = 2048, overlap: float = 0.5) -> numpy.ndarray")]
    fn power_spectral_density<'py>(
        &self,
        py: Python<'py>,
        window_size: usize,
        overlap: f64,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let window_size = nzu_or_err(window_size)?;

        dispatch_with_view!(self, py, |audio| {
            let (_, psd) = audio
                .power_spectral_density(window_size, overlap)
                .map_err(audio_err_to_py)?;
            Ok(PyArray::from_vec(py, psd))
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
}
