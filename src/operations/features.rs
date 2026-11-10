//! Frame-wise features on `AudioSamples`
//! (`spectral_centroid_frames`, `spectral_bandwidth_frames`,
//! `spectral_rolloff_frames`, `spectral_flatness_frames`,
//! `spectral_contrast_frames`, `poly_features`, `rms_frames`,
//! `zero_crossing_rate_frames`, `reassigned_spectrogram`), binding the Rust
//! `AudioFeatures` trait.
//!
//! Every method returns arrays shaped `(channels, n_features, n_frames)` (mono
//! audio gives a leading axis of length 1): float32 for `uint8`, `int16`,
//! `int24` and `float32` audio, float64 for `int32` and `float64` audio. The
//! spectral descriptors default to `hop_length=512` whatever the `n_fft`.

use crate::args::Int;
use audio_samples::operations::features::{
    AudioFeatures, PolyFeaturesConfig, ReassignedSpectrogramConfig, RmsConfig,
    SpectralBandwidthConfig, SpectralCentroidConfig, SpectralContrastConfig,
    SpectralFlatnessConfig, SpectralRolloffConfig, ZeroCrossingRateConfig,
};
use audio_samples::operations::filterbanks::Window;
use numpy::IntoPyArray;
use pyo3::IntoPyObjectExt;
use pyo3::prelude::*;

use crate::args::{WindowArg, non_negative, parse_pad_mode, positive, stft_config};
use crate::features::args::{ReassignRefArg, parse_zcr_ref};
use crate::{PyAudioSamples, audio_err_to_py, dispatch_array_detached, dispatch_typed_detached};

#[pymethods]
impl PyAudioSamples {
    /// Spectral centroid of every frame: the magnitude-weighted mean frequency.
    ///
    /// Computed from the magnitude STFT. For several descriptors of one
    /// spectrogram use ``features.spectral_centroid_from_spectrogram``.
    ///
    /// Args:
    ///
    ///     n_fft (int): FFT length. Default 2048.
    ///     hop_length (int): Samples between frames. Default 512.
    ///     win_length (Optional[int]): Window length; None = ``n_fft``.
    ///     window: Window specification (see ``stft_matrix``). Default ``"hann"``.
    ///     center (bool): Centre frames. Default True.
    ///     pad_mode (str): Padding for ``center``. Default ``"constant"``.
    ///
    /// Returns:
    ///
    ///     numpy.ndarray[float32 | float64]: shape ``(channels, 1, n_frames)`` in Hz, also
    ///     for mono audio (index ``[0]`` for ``(1, n_frames)``).
    ///
    /// Raises:
    ///
    ///     ValueError: For invalid parameters or a too-short uncentred signal.
    #[pyo3(
        signature = (*, n_fft = Int(2048), hop_length = Int(512), win_length = None, window = WindowArg(Window::Hann), center = true, pad_mode = "constant") -> "NDArray[Any]",
        text_signature = "($self, *, n_fft=2048, hop_length=512, win_length=None, window='hann', center=True, pad_mode='constant')"
    )]
    #[allow(
        clippy::too_many_arguments,
        reason = "the Python signature has 6 parameters, plus self and py"
    )]
    fn spectral_centroid_frames<'py>(
        &self,
        py: Python<'py>,
        n_fft: Int,
        hop_length: Int,
        win_length: Option<Int>,
        window: WindowArg,
        center: bool,
        pad_mode: &str,
    ) -> PyResult<Bound<'py, PyAny>> {
        let cfg = SpectralCentroidConfig {
            stft: stft_config(
                n_fft,
                Some(hop_length),
                win_length,
                window,
                center,
                pad_mode,
            )?,
        };
        dispatch_array_detached!(self, py, |audio| audio.spectral_centroid_frames(&cfg))
    }

    /// Order-``p`` spectral bandwidth of every frame around the spectral centroid.
    ///
    /// Args:
    ///
    ///     n_fft, hop_length (default 512), win_length, window, center, pad_mode:
    ///         As ``spectral_centroid_frames``.
    ///     norm (bool): Normalise each frame to unit sum. Default True.
    ///     p (float): Order of the deviation. Default 2.
    ///
    /// Returns:
    ///
    ///     numpy.ndarray[float32 | float64]: shape ``(channels, 1, n_frames)`` in Hz.
    ///
    /// Raises:
    ///
    ///     ValueError: For invalid parameters.
    #[pyo3(
        signature = (*, n_fft = Int(2048), hop_length = Int(512), win_length = None, window = WindowArg(Window::Hann), center = true, pad_mode = "constant", norm = true, p = 2.0) -> "NDArray[Any]",
        text_signature = "($self, *, n_fft=2048, hop_length=512, win_length=None, window='hann', center=True, pad_mode='constant', norm=True, p=2.0)"
    )]
    #[allow(
        clippy::too_many_arguments,
        reason = "the Python signature has 8 parameters"
    )]
    fn spectral_bandwidth_frames<'py>(
        &self,
        py: Python<'py>,
        n_fft: Int,
        hop_length: Int,
        win_length: Option<Int>,
        window: WindowArg,
        center: bool,
        pad_mode: &str,
        norm: bool,
        p: f64,
    ) -> PyResult<Bound<'py, PyAny>> {
        let cfg = SpectralBandwidthConfig {
            stft: stft_config(
                n_fft,
                Some(hop_length),
                win_length,
                window,
                center,
                pad_mode,
            )?,
            p,
            norm,
        };
        dispatch_array_detached!(self, py, |audio| audio.spectral_bandwidth_frames(&cfg))
    }

    /// Roll-off frequency of every frame: the frequency below which
    /// ``roll_percent`` of the energy lies.
    ///
    /// Args:
    ///
    ///     n_fft, hop_length (default 512), win_length, window, center, pad_mode:
    ///         As ``spectral_centroid_frames``.
    ///     roll_percent (float): Energy fraction in ``(0, 1)``. Default 0.85.
    ///
    /// Returns:
    ///
    ///     numpy.ndarray[float32 | float64]: shape ``(channels, 1, n_frames)`` in Hz.
    ///
    /// Raises:
    ///
    ///     ValueError: For invalid parameters.
    #[pyo3(
        signature = (*, n_fft = Int(2048), hop_length = Int(512), win_length = None, window = WindowArg(Window::Hann), center = true, pad_mode = "constant", roll_percent = 0.85) -> "NDArray[Any]",
        text_signature = "($self, *, n_fft=2048, hop_length=512, win_length=None, window='hann', center=True, pad_mode='constant', roll_percent=0.85)"
    )]
    #[allow(
        clippy::too_many_arguments,
        reason = "the Python signature has 7 parameters"
    )]
    fn spectral_rolloff_frames<'py>(
        &self,
        py: Python<'py>,
        n_fft: Int,
        hop_length: Int,
        win_length: Option<Int>,
        window: WindowArg,
        center: bool,
        pad_mode: &str,
        roll_percent: f64,
    ) -> PyResult<Bound<'py, PyAny>> {
        let cfg = SpectralRolloffConfig {
            stft: stft_config(
                n_fft,
                Some(hop_length),
                win_length,
                window,
                center,
                pad_mode,
            )?,
            roll_percent,
        };
        dispatch_array_detached!(self, py, |audio| audio.spectral_rolloff_frames(&cfg))
    }

    /// Spectral flatness of every frame: the geometric over the arithmetic mean
    /// of the power spectrum.
    ///
    /// Args:
    ///
    ///     n_fft, hop_length (default 512), win_length, window, center, pad_mode:
    ///         As ``spectral_centroid_frames``.
    ///     amin (float): Floor of ``|S| ** power``. Default 1e-10.
    ///     power (float): Exponent of the magnitude. Default 2.0.
    ///
    /// Returns:
    ///
    ///     numpy.ndarray[float32 | float64]: shape ``(channels, 1, n_frames)``, in ``[0, 1]``.
    ///
    /// Raises:
    ///
    ///     ValueError: For invalid parameters.
    #[pyo3(
        signature = (*, n_fft = Int(2048), hop_length = Int(512), win_length = None, window = WindowArg(Window::Hann), center = true, pad_mode = "constant", amin = 1e-10, power = 2.0) -> "NDArray[Any]",
        text_signature = "($self, *, n_fft=2048, hop_length=512, win_length=None, window='hann', center=True, pad_mode='constant', amin=1e-10, power=2.0)"
    )]
    #[allow(
        clippy::too_many_arguments,
        reason = "the Python signature has 8 parameters"
    )]
    fn spectral_flatness_frames<'py>(
        &self,
        py: Python<'py>,
        n_fft: Int,
        hop_length: Int,
        win_length: Option<Int>,
        window: WindowArg,
        center: bool,
        pad_mode: &str,
        amin: f64,
        power: f64,
    ) -> PyResult<Bound<'py, PyAny>> {
        let cfg = SpectralFlatnessConfig {
            stft: stft_config(
                n_fft,
                Some(hop_length),
                win_length,
                window,
                center,
                pad_mode,
            )?,
            amin,
            power,
        };
        dispatch_array_detached!(self, py, |audio| audio.spectral_flatness_frames(&cfg))
    }

    /// Octave-band spectral contrast of every frame: the peak-to-valley
    /// difference in each band, in dB.
    ///
    /// Args:
    ///
    ///     n_fft, hop_length (default 512), win_length, window, center, pad_mode:
    ///         As ``spectral_centroid_frames``.
    ///     fmin (float): Upper edge of the lowest band (Hz). Default 200.
    ///     n_bands (int): Number of octave bands. Default 6.
    ///     quantile (float): Fraction used for peak / valley. Default 0.02.
    ///     linear (bool): Linear difference instead of dB. Default False.
    ///
    /// Returns:
    ///
    ///     numpy.ndarray[float32 | float64]: shape ``(channels, n_bands + 1, n_frames)``.
    ///
    /// Raises:
    ///
    ///     ValueError: For invalid parameters (e.g. the top band above Nyquist).
    #[pyo3(
        signature = (*, n_fft = Int(2048), hop_length = Int(512), win_length = None, window = WindowArg(Window::Hann), center = true, pad_mode = "constant", fmin = 200.0, n_bands = Int(6), quantile = 0.02, linear = false) -> "NDArray[Any]",
        text_signature = "($self, *, n_fft=2048, hop_length=512, win_length=None, window='hann', center=True, pad_mode='constant', fmin=200.0, n_bands=6, quantile=0.02, linear=False)"
    )]
    #[allow(
        clippy::too_many_arguments,
        reason = "the Python signature has 10 parameters"
    )]
    fn spectral_contrast_frames<'py>(
        &self,
        py: Python<'py>,
        n_fft: Int,
        hop_length: Int,
        win_length: Option<Int>,
        window: WindowArg,
        center: bool,
        pad_mode: &str,
        fmin: f64,
        n_bands: Int,
        quantile: f64,
        linear: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        let cfg = SpectralContrastConfig {
            stft: stft_config(
                n_fft,
                Some(hop_length),
                win_length,
                window,
                center,
                pad_mode,
            )?,
            fmin,
            n_bands: positive("n_bands", n_bands)?,
            quantile,
            linear,
        };
        dispatch_array_detached!(self, py, |audio| audio.spectral_contrast_frames(&cfg))
    }

    /// Coefficients of a least-squares polynomial fit to every
    /// magnitude-spectrum frame against the bin frequencies.
    ///
    /// Args:
    ///
    ///     n_fft, hop_length (default 512), win_length, window, center, pad_mode:
    ///         As ``spectral_centroid_frames``.
    ///     order (int): Polynomial order. Default 1.
    ///
    /// Returns:
    ///
    ///     numpy.ndarray[float32 | float64]: shape ``(channels, order + 1, n_frames)``,
    ///     highest power first.
    ///
    /// Raises:
    ///
    ///     ValueError: For invalid parameters.
    #[pyo3(
        signature = (*, n_fft = Int(2048), hop_length = Int(512), win_length = None, window = WindowArg(Window::Hann), center = true, pad_mode = "constant", order = Int(1)) -> "NDArray[Any]",
        text_signature = "($self, *, n_fft=2048, hop_length=512, win_length=None, window='hann', center=True, pad_mode='constant', order=1)"
    )]
    #[allow(
        clippy::too_many_arguments,
        reason = "the Python signature has 7 parameters"
    )]
    fn poly_features<'py>(
        &self,
        py: Python<'py>,
        n_fft: Int,
        hop_length: Int,
        win_length: Option<Int>,
        window: WindowArg,
        center: bool,
        pad_mode: &str,
        order: Int,
    ) -> PyResult<Bound<'py, PyAny>> {
        let cfg = PolyFeaturesConfig {
            stft: stft_config(
                n_fft,
                Some(hop_length),
                win_length,
                window,
                center,
                pad_mode,
            )?,
            order: non_negative("order", order)?,
        };
        dispatch_array_detached!(self, py, |audio| audio.poly_features(&cfg))
    }

    /// Frame-wise root-mean-square energy.
    ///
    /// Unlike the scalar ``rms()``, this returns one value per frame.
    ///
    /// Args:
    ///
    ///     frame_length (int): Samples per frame. Default 2048.
    ///     hop_length (int): Samples between frames. Default 512.
    ///     center (bool): Centre frames (pad ``frame_length // 2``). Default True.
    ///     pad_mode (str): Padding for ``center``. Default ``"constant"``.
    ///
    /// Returns:
    ///
    ///     numpy.ndarray[float32 | float64]: shape ``(channels, 1, n_frames)``.
    ///
    /// Raises:
    ///
    ///     ValueError: For invalid parameters or a too-short uncentred signal.
    #[pyo3(
        signature = (*, frame_length = Int(2048), hop_length = Int(512), center = true, pad_mode = "constant") -> "NDArray[Any]",
        text_signature = "($self, *, frame_length=2048, hop_length=512, center=True, pad_mode='constant')"
    )]
    fn rms_frames<'py>(
        &self,
        py: Python<'py>,
        frame_length: Int,
        hop_length: Int,
        center: bool,
        pad_mode: &str,
    ) -> PyResult<Bound<'py, PyAny>> {
        let cfg = RmsConfig {
            frame_length: positive("frame_length", frame_length)?,
            hop_length: positive("hop_length", hop_length)?,
            center,
            pad_mode: parse_pad_mode(pad_mode)?,
        };
        dispatch_array_detached!(self, py, |audio| audio.rms_frames(&cfg))
    }

    /// Frame-wise zero-crossing rate (fraction of sign changes per frame).
    ///
    /// Unlike the scalar ``zero_crossing_rate()`` (crossings per second), this
    /// returns one fraction per frame. Centring always uses edge padding.
    ///
    /// Args:
    ///
    ///     frame_length (int): Samples per frame. Default 2048.
    ///     hop_length (int): Samples between frames. Default 512.
    ///     center (bool): Centre frames. Default True.
    ///     threshold (float): ``|x| <= threshold * ref`` counts as zero. Default 1e-10.
    ///     ref_magnitude (None | float | numpy.max): Threshold scale. Default None.
    ///     pad (bool): Count the first sample of every frame as a crossing. Default False.
    ///     zero_pos (bool): Zero counts as positive. Default True.
    ///
    /// Returns:
    ///
    ///     numpy.ndarray[float32 | float64]: shape ``(channels, 1, n_frames)``.
    ///
    /// Raises:
    ///
    ///     ValueError: For invalid parameters.
    #[pyo3(
        signature = (*, frame_length = Int(2048), hop_length = Int(512), center = true, threshold = 1e-10, ref_magnitude = None, pad = false, zero_pos = true) -> "NDArray[Any]",
        text_signature = "($self, *, frame_length=2048, hop_length=512, center=True, threshold=1e-10, ref_magnitude=None, pad=False, zero_pos=True)"
    )]
    #[allow(
        clippy::too_many_arguments,
        reason = "the Python signature has 7 parameters"
    )]
    fn zero_crossing_rate_frames<'py>(
        &self,
        py: Python<'py>,
        frame_length: Int,
        hop_length: Int,
        center: bool,
        threshold: f64,
        ref_magnitude: Option<&Bound<'py, PyAny>>,
        pad: bool,
        zero_pos: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        let ref_magnitude = match ref_magnitude {
            Some(r) => parse_zcr_ref(r)?,
            None => audio_samples::operations::features::ZeroCrossingReference::None,
        };
        let cfg = ZeroCrossingRateConfig {
            frame_length: positive("frame_length", frame_length)?,
            hop_length: positive("hop_length", hop_length)?,
            center,
            threshold,
            ref_magnitude,
            pad,
            zero_pos,
        };
        dispatch_array_detached!(self, py, |audio| audio.zero_crossing_rate_frames(&cfg))
    }

    /// Time-frequency reassigned spectrogram: the instantaneous frequency and
    /// group-delay time of every bin.
    ///
    /// Args:
    ///
    ///     n_fft (int): FFT length. Default 2048.
    ///     hop_length (Optional[int]): None = ``win_length // 4``.
    ///     win_length (Optional[int]): None = ``n_fft``.
    ///     window: Window specification. Default ``"hann"``.
    ///     center (bool): Centre frames. Default True.
    ///     reassign_frequencies (bool): Reassign frequencies. Default True.
    ///     reassign_times (bool): Reassign times. Default True.
    ///     ref_power (float | numpy.max): Power threshold below which estimates are
    ///         NaN. Default 1e-6.
    ///     fill_nan (bool): Replace NaN estimates by bin / frame centres. Default False.
    ///     clip (bool): Clip to ``[0, sr / 2]`` and ``[0, duration]``. Default True.
    ///     pad_mode (str): Padding for ``center``. Default ``"constant"``.
    ///
    /// Returns:
    ///
    ///     tuple: ``(freqs, times, mags)``, each numpy.ndarray[float32 | float64] of shape
    ///     ``(channels, 1 + n_fft // 2, n_frames)``.
    ///
    /// Raises:
    ///
    ///     ValueError: For invalid parameters.
    #[pyo3(
        signature = (*, n_fft = Int(2048), hop_length = None, win_length = None, window = WindowArg(Window::Hann), center = true, reassign_frequencies = true, reassign_times = true, ref_power = ReassignRefArg(audio_samples::operations::features::ReassignRefPower::Value(1e-6)), fill_nan = false, clip = true, pad_mode = "constant") -> "tuple[NDArray[Any], NDArray[Any], NDArray[Any]] | tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]] | tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]",
        text_signature = "($self, *, n_fft=2048, hop_length=None, win_length=None, window='hann', center=True, reassign_frequencies=True, reassign_times=True, ref_power=1e-06, fill_nan=False, clip=True, pad_mode='constant')"
    )]
    #[allow(
        clippy::too_many_arguments,
        reason = "the Python signature has 11 parameters"
    )]
    #[allow(
        clippy::fn_params_excessive_bools,
        reason = "the Python signature has these boolean keywords"
    )]
    fn reassigned_spectrogram<'py>(
        &self,
        py: Python<'py>,
        n_fft: Int,
        hop_length: Option<Int>,
        win_length: Option<Int>,
        window: WindowArg,
        center: bool,
        reassign_frequencies: bool,
        reassign_times: bool,
        ref_power: ReassignRefArg,
        fill_nan: bool,
        clip: bool,
        pad_mode: &str,
    ) -> PyResult<Bound<'py, PyAny>> {
        let cfg = ReassignedSpectrogramConfig {
            stft: stft_config(n_fft, hop_length, win_length, window, center, pad_mode)?,
            reassign_frequencies,
            reassign_times,
            ref_power: ref_power.0,
            fill_nan,
            clip,
        };
        dispatch_typed_detached!(self, py, |audio| audio.reassigned_spectrogram(&cfg), |r| {
            let r = r.map_err(audio_err_to_py)?;
            (
                r.freqs.into_pyarray(py),
                r.times.into_pyarray(py),
                r.mags.into_pyarray(py),
            )
                .into_bound_py_any(py)
        })
    }
}
