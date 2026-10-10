//! librosa-compatible constant-Q analysis on `AudioSamples` (`cqt`, `vqt`,
//! `pseudo_cqt`, `hybrid_cqt`, the static `icqt` / `griffinlim_cqt`,
//! `chroma_cqt`, `chroma_cens`, `chroma_vqt`, `tonnetz`), binding the Rust
//! `AudioConstantQ` trait.
//!
//! Frame `t` is centred on sample `t * hop_length`. librosa decimates between
//! octaves with soxr; the Rust crate uses its own windowed-sinc resampler
//! (`res_type` selects its quality), so octaves below the top one differ from
//! librosa by about 1e-4 of the peak.

use crate::args::Int;
use audio_samples::AudioSamples;
use audio_samples::operations::constant_q::{
    AudioConstantQ, ChromaCensConfig, ChromaCqtConfig, ChromaVqtConfig, CqtConfig, TonnetzConfig,
    VqtConfig,
};
use audio_samples::operations::filterbanks::Window;
use ndarray::{Array3, Axis, Ix2, Ix3};
use num_complex::Complex;
use numpy::{IntoPyArray, PyArray3};
use pyo3::prelude::*;

use crate::args::{
    WindowArg, c128_array, f64_array, non_negative, opt_positive, parse_pad_mode, positive,
    value_err,
};
use crate::features::args::{check_intervals, parse_cqt_mode, parse_res_type};
use crate::{PyAudioSamples, audio_err_to_py, dispatch_with_view, nzu32_or_err};

/// Builds a [`CqtConfig`] from librosa's `cqt` keyword arguments.
#[allow(clippy::too_many_arguments)]
fn cqt_config(
    hop_length: Int,
    fmin: Option<f64>,
    n_bins: Option<Int>,
    bins_per_octave: Int,
    tuning: Option<f64>,
    filter_scale: f64,
    norm: Option<f64>,
    sparsity: f64,
    window: WindowArg,
    scale: bool,
    pad_mode: &str,
    res_type: &str,
) -> PyResult<CqtConfig> {
    Ok(CqtConfig {
        hop_length: positive("hop_length", hop_length)?,
        fmin,
        n_bins: opt_positive("n_bins", n_bins)?,
        bins_per_octave: positive("bins_per_octave", bins_per_octave)?,
        tuning,
        filter_scale,
        norm,
        sparsity,
        window: window.0,
        scale,
        pad_mode: parse_pad_mode(pad_mode)?,
        resampling_quality: parse_res_type(res_type)?,
    })
}

/// A 2-D `(n_bins, T)` or 3-D `(channels, n_bins, T)` array as 3-D.
fn to3<T: Clone>(a: ndarray::ArrayViewD<'_, T>, name: &str) -> PyResult<Array3<T>> {
    match a.ndim() {
        2 => Ok(a
            .into_dimensionality::<Ix2>()
            .expect("ndim checked")
            .insert_axis(Axis(0))
            .to_owned()),
        3 => Ok(a
            .into_dimensionality::<Ix3>()
            .expect("ndim checked")
            .to_owned()),
        n => Err(value_err(
            name,
            format!(
                "expected a 2-D (n_bins, T) or 3-D (channels, n_bins, T) array, got {n} dimensions"
            ),
        )),
    }
}

fn check_window_none(window: Option<&Bound<'_, PyAny>>) -> PyResult<()> {
    if window.is_some_and(|w| !w.is_none()) {
        return Err(value_err(
            "window",
            "a chroma smoothing window is not supported; pass None",
        ));
    }
    Ok(())
}

/// Builds a [`ChromaCqtConfig`] from librosa's `chroma_cqt` keyword arguments.
#[allow(clippy::too_many_arguments)]
fn chroma_cqt_config(
    hop_length: Int,
    fmin: Option<f64>,
    norm: Option<f64>,
    threshold: Option<f64>,
    tuning: Option<f64>,
    n_chroma: Int,
    n_octaves: Int,
    bins_per_octave: Option<Int>,
    cqt_mode: &str,
) -> PyResult<ChromaCqtConfig> {
    Ok(ChromaCqtConfig {
        hop_length: positive("hop_length", hop_length)?,
        fmin,
        norm,
        threshold,
        tuning,
        n_chroma: positive("n_chroma", n_chroma)?,
        n_octaves: positive("n_octaves", n_octaves)?,
        bins_per_octave: opt_positive("bins_per_octave", bins_per_octave)?,
        cqt_mode: parse_cqt_mode(cqt_mode)?,
    })
}

#[pymethods]
impl PyAudioSamples {
    /// Constant-Q transform of every channel, equivalent to
    /// ``librosa.cqt(y, sr=self.sample_rate, ...)``.
    ///
    /// Unlike ``constant_q_transform()`` (spectrograms crate), this reproduces
    /// librosa's filters, scaling and framing.
    ///
    /// Args:
    ///     hop_length (int): Samples between frames. Default 512.
    ///     fmin (Optional[float]): Lowest bin frequency (Hz); None = C1 (32.70 Hz).
    ///     n_bins (Optional[int]): Number of bins (default 84); None = as many as
    ///         fit below Nyquist.
    ///     bins_per_octave (int): Default 12.
    ///     tuning (Optional[float]): Tuning offset in fractions of a bin (default
    ///         0.0); None estimates it from the signal.
    ///     filter_scale (float): Filter-length scale. Default 1.
    ///     norm (Optional[float]): Filter normalisation (p-norm). Default 1.
    ///     sparsity (float): Fraction of each filter's mass that may be dropped.
    ///         Default 0.01.
    ///     window: Named window of the filters. Default ``"hann"``.
    ///     scale (bool): Divide by ``sqrt(filter length)``. Default True.
    ///     pad_mode (str): Padding at the signal ends. Default ``"constant"``.
    ///     res_type (str): Resampler for the octave decimation. ``"soxr_hq"``
    ///         (default) and ``"soxr_vhq"`` use the crate's high-quality
    ///         resampler, ``"soxr_mq"`` medium, ``"soxr_lq"`` fast (soxr itself
    ///         is not used).
    ///
    /// Returns:
    ///     numpy.ndarray[complex128]: shape ``(channels, n_bins, n_frames)``
    ///     (librosa returns ``(n_bins, n_frames)`` for 1-D input; index ``[0]``).
    ///
    /// Raises:
    ///     ValueError: For invalid parameters (e.g. bins above Nyquist).
    #[pyo3(
        signature = (*, hop_length = Int(512), fmin = None, n_bins = Some(Int(84)), bins_per_octave = Int(12), tuning = Some(0.0), filter_scale = 1.0, norm = Some(1.0), sparsity = 0.01, window = WindowArg(Window::Hann), scale = true, pad_mode = "constant", res_type = "soxr_hq"),
        text_signature = "($self, *, hop_length=512, fmin=None, n_bins=84, bins_per_octave=12, tuning=0.0, filter_scale=1, norm=1, sparsity=0.01, window='hann', scale=True, pad_mode='constant', res_type='soxr_hq')"
    )]
    #[allow(clippy::too_many_arguments)]
    fn cqt<'py>(
        &self,
        py: Python<'py>,
        hop_length: Int,
        fmin: Option<f64>,
        n_bins: Option<Int>,
        bins_per_octave: Int,
        tuning: Option<f64>,
        filter_scale: f64,
        norm: Option<f64>,
        sparsity: f64,
        window: WindowArg,
        scale: bool,
        pad_mode: &str,
        res_type: &str,
    ) -> PyResult<Bound<'py, PyArray3<Complex<f64>>>> {
        let cfg = cqt_config(
            hop_length,
            fmin,
            n_bins,
            bins_per_octave,
            tuning,
            filter_scale,
            norm,
            sparsity,
            window,
            scale,
            pad_mode,
            res_type,
        )?;
        let c = dispatch_with_view!(self, py, |audio| audio.cqt(&cfg)).map_err(audio_err_to_py)?;
        Ok(c.into_pyarray(py))
    }

    /// Variable-Q transform of every channel, equivalent to
    /// ``librosa.vqt(y, sr=self.sample_rate, ...)``.
    ///
    /// Args:
    ///     hop_length, fmin, n_bins, bins_per_octave, tuning, filter_scale, norm, sparsity, window, scale, pad_mode, res_type: As ``cqt``.
    ///     intervals (str): Only ``"equal"`` (equal temperament) is supported.
    ///     gamma (Optional[float]): Bandwidth offset (Hz); None (default) =
    ///         ERB-derived, 0 = constant-Q.
    ///
    /// Returns:
    ///     numpy.ndarray[complex128]: shape ``(channels, n_bins, n_frames)``.
    ///
    /// Raises:
    ///     ValueError: For invalid parameters or ``intervals`` other than ``"equal"``.
    #[pyo3(
        signature = (*, hop_length = Int(512), fmin = None, n_bins = Some(Int(84)), intervals = None, gamma = None, bins_per_octave = Int(12), tuning = Some(0.0), filter_scale = 1.0, norm = Some(1.0), sparsity = 0.01, window = WindowArg(Window::Hann), scale = true, pad_mode = "constant", res_type = "soxr_hq"),
        text_signature = "($self, *, hop_length=512, fmin=None, n_bins=84, intervals='equal', gamma=None, bins_per_octave=12, tuning=0.0, filter_scale=1, norm=1, sparsity=0.01, window='hann', scale=True, pad_mode='constant', res_type='soxr_hq')"
    )]
    #[allow(clippy::too_many_arguments)]
    fn vqt<'py>(
        &self,
        py: Python<'py>,
        hop_length: Int,
        fmin: Option<f64>,
        n_bins: Option<Int>,
        intervals: Option<&Bound<'py, PyAny>>,
        gamma: Option<f64>,
        bins_per_octave: Int,
        tuning: Option<f64>,
        filter_scale: f64,
        norm: Option<f64>,
        sparsity: f64,
        window: WindowArg,
        scale: bool,
        pad_mode: &str,
        res_type: &str,
    ) -> PyResult<Bound<'py, PyArray3<Complex<f64>>>> {
        if let Some(i) = intervals {
            check_intervals(i)?;
        }
        let cfg = VqtConfig {
            cqt: cqt_config(
                hop_length,
                fmin,
                n_bins,
                bins_per_octave,
                tuning,
                filter_scale,
                norm,
                sparsity,
                window,
                scale,
                pad_mode,
                res_type,
            )?,
            gamma,
        };
        let c = dispatch_with_view!(self, py, |audio| audio.vqt(&cfg)).map_err(audio_err_to_py)?;
        Ok(c.into_pyarray(py))
    }

    /// Pseudo constant-Q magnitude (a single STFT with CQT filters), equivalent
    /// to ``librosa.pseudo_cqt(y, sr=self.sample_rate, ...)``.
    ///
    /// Args:
    ///     hop_length, fmin, n_bins, bins_per_octave, tuning, filter_scale, norm, sparsity, window, scale, pad_mode: As ``cqt``.
    ///
    /// Returns:
    ///     numpy.ndarray[float64]: magnitudes, shape ``(channels, n_bins, n_frames)``.
    ///
    /// Raises:
    ///     ValueError: For invalid parameters.
    #[pyo3(
        signature = (*, hop_length = Int(512), fmin = None, n_bins = Some(Int(84)), bins_per_octave = Int(12), tuning = Some(0.0), filter_scale = 1.0, norm = Some(1.0), sparsity = 0.01, window = WindowArg(Window::Hann), scale = true, pad_mode = "constant"),
        text_signature = "($self, *, hop_length=512, fmin=None, n_bins=84, bins_per_octave=12, tuning=0.0, filter_scale=1, norm=1, sparsity=0.01, window='hann', scale=True, pad_mode='constant')"
    )]
    #[allow(clippy::too_many_arguments)]
    fn pseudo_cqt<'py>(
        &self,
        py: Python<'py>,
        hop_length: Int,
        fmin: Option<f64>,
        n_bins: Option<Int>,
        bins_per_octave: Int,
        tuning: Option<f64>,
        filter_scale: f64,
        norm: Option<f64>,
        sparsity: f64,
        window: WindowArg,
        scale: bool,
        pad_mode: &str,
    ) -> PyResult<Bound<'py, PyArray3<f64>>> {
        let cfg = cqt_config(
            hop_length,
            fmin,
            n_bins,
            bins_per_octave,
            tuning,
            filter_scale,
            norm,
            sparsity,
            window,
            scale,
            pad_mode,
            "soxr_hq",
        )?;
        let c = dispatch_with_view!(self, py, |audio| audio.pseudo_cqt(&cfg))
            .map_err(audio_err_to_py)?;
        Ok(c.into_pyarray(py))
    }

    /// Hybrid constant-Q magnitude (pseudo-CQT for the high, full CQT for the
    /// low bins), equivalent to ``librosa.hybrid_cqt(y, sr=self.sample_rate, ...)``.
    ///
    /// Args:
    ///     As ``cqt``.
    ///
    /// Returns:
    ///     numpy.ndarray[float64]: magnitudes, shape ``(channels, n_bins, n_frames)``.
    ///
    /// Raises:
    ///     ValueError: For invalid parameters.
    #[pyo3(
        signature = (*, hop_length = Int(512), fmin = None, n_bins = Some(Int(84)), bins_per_octave = Int(12), tuning = Some(0.0), filter_scale = 1.0, norm = Some(1.0), sparsity = 0.01, window = WindowArg(Window::Hann), scale = true, pad_mode = "constant", res_type = "soxr_hq"),
        text_signature = "($self, *, hop_length=512, fmin=None, n_bins=84, bins_per_octave=12, tuning=0.0, filter_scale=1, norm=1, sparsity=0.01, window='hann', scale=True, pad_mode='constant', res_type='soxr_hq')"
    )]
    #[allow(clippy::too_many_arguments)]
    fn hybrid_cqt<'py>(
        &self,
        py: Python<'py>,
        hop_length: Int,
        fmin: Option<f64>,
        n_bins: Option<Int>,
        bins_per_octave: Int,
        tuning: Option<f64>,
        filter_scale: f64,
        norm: Option<f64>,
        sparsity: f64,
        window: WindowArg,
        scale: bool,
        pad_mode: &str,
        res_type: &str,
    ) -> PyResult<Bound<'py, PyArray3<f64>>> {
        let cfg = cqt_config(
            hop_length,
            fmin,
            n_bins,
            bins_per_octave,
            tuning,
            filter_scale,
            norm,
            sparsity,
            window,
            scale,
            pad_mode,
            res_type,
        )?;
        let c = dispatch_with_view!(self, py, |audio| audio.hybrid_cqt(&cfg))
            .map_err(audio_err_to_py)?;
        Ok(c.into_pyarray(py))
    }

    /// Inverse constant-Q transform, equivalent to ``librosa.icqt``.
    ///
    /// This is a static method: ``AudioSamples.icqt(C, sample_rate=sr)``.
    ///
    /// Args:
    ///     C (array_like): complex CQT of shape ``(n_bins, T)`` or ``(channels, n_bins, T)``.
    ///     sample_rate (int): Sample rate of the analysed signal (keyword-only).
    ///     hop_length, fmin, bins_per_octave, filter_scale, norm, sparsity, window, scale, res_type: The forward parameters (see ``cqt``).
    ///     tuning (float): Tuning used by the forward transform. Default 0.0.
    ///     length (Optional[int]): Exact output length per channel.
    ///
    /// Returns:
    ///     AudioSamples: float64 audio (mono for a single channel).
    ///
    /// Raises:
    ///     ValueError: For invalid parameters or an empty ``C``.
    #[staticmethod]
    #[pyo3(
        signature = (C, *, sample_rate, hop_length = Int(512), fmin = None, bins_per_octave = Int(12), tuning = 0.0, filter_scale = 1.0, norm = Some(1.0), sparsity = 0.01, window = WindowArg(Window::Hann), scale = true, length = None, res_type = "soxr_hq"),
        text_signature = "(C, *, sample_rate, hop_length=512, fmin=None, bins_per_octave=12, tuning=0.0, filter_scale=1, norm=1, sparsity=0.01, window='hann', scale=True, length=None, res_type='soxr_hq')"
    )]
    #[allow(non_snake_case, clippy::too_many_arguments)]
    fn icqt(
        C: &Bound<'_, PyAny>,
        sample_rate: Int,
        hop_length: Int,
        fmin: Option<f64>,
        bins_per_octave: Int,
        tuning: f64,
        filter_scale: f64,
        norm: Option<f64>,
        sparsity: f64,
        window: WindowArg,
        scale: bool,
        length: Option<Int>,
        res_type: &str,
    ) -> PyResult<Self> {
        let c = c128_array(C, "C")?;
        let c3 = to3(c.as_array(), "C")?;
        let cfg = cqt_config(
            hop_length,
            fmin,
            None,
            bins_per_octave,
            Some(tuning),
            filter_scale,
            norm,
            sparsity,
            window,
            scale,
            "constant",
            res_type,
        )?;
        let sr = nzu32_or_err(crate::args::sample_rate_u32(sample_rate)?)?;
        let length = opt_positive("length", length)?;
        let y = <AudioSamples<'static, f64> as AudioConstantQ>::icqt(&c3, &cfg, sr, length)
            .map_err(audio_err_to_py)?;
        Ok(Self::from_audio_samples(y))
    }

    /// Reconstructs audio from a constant-Q magnitude by Griffin-Lim, equivalent
    /// to ``librosa.griffinlim_cqt(C, sr=sample_rate, ..., init=None)``.
    ///
    /// This is a static method. Only the deterministic zero-phase start
    /// (librosa's ``init=None``) is available; librosa's default
    /// ``init="random"`` depends on NumPy's random generator and is rejected.
    ///
    /// Args:
    ///     C (array_like): Non-negative CQT magnitude, ``(n_bins, T)`` or
    ///         ``(channels, n_bins, T)``.
    ///     sample_rate (int): Sample rate (keyword-only).
    ///     n_iter (int): Iterations. Default 32.
    ///     hop_length, fmin, bins_per_octave, tuning, filter_scale, norm, sparsity, window, scale, pad_mode, res_type: The forward parameters (see ``cqt``).
    ///     length (Optional[int]): Output length per channel.
    ///     momentum (float): Fast Griffin-Lim momentum. Default 0.99.
    ///     init (None): Must be None.
    ///
    /// Returns:
    ///     AudioSamples: float64 audio.
    ///
    /// Raises:
    ///     ValueError: For invalid parameters or ``init`` other than None.
    #[staticmethod]
    #[pyo3(
        signature = (C, *, sample_rate, n_iter = Int(32), hop_length = Int(512), fmin = None, bins_per_octave = Int(12), tuning = 0.0, filter_scale = 1.0, norm = Some(1.0), sparsity = 0.01, window = WindowArg(Window::Hann), scale = true, pad_mode = "constant", res_type = "soxr_hq", length = None, momentum = 0.99, init = None),
        text_signature = "(C, *, sample_rate, n_iter=32, hop_length=512, fmin=None, bins_per_octave=12, tuning=0.0, filter_scale=1, norm=1, sparsity=0.01, window='hann', scale=True, pad_mode='constant', res_type='soxr_hq', length=None, momentum=0.99, init=None)"
    )]
    #[allow(non_snake_case, clippy::too_many_arguments)]
    fn griffinlim_cqt(
        C: &Bound<'_, PyAny>,
        sample_rate: Int,
        n_iter: Int,
        hop_length: Int,
        fmin: Option<f64>,
        bins_per_octave: Int,
        tuning: f64,
        filter_scale: f64,
        norm: Option<f64>,
        sparsity: f64,
        window: WindowArg,
        scale: bool,
        pad_mode: &str,
        res_type: &str,
        length: Option<Int>,
        momentum: f64,
        init: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        if init.is_some_and(|i| !i.is_none()) {
            return Err(value_err(
                "init",
                "only init=None (zero initial phase) is supported; librosa's 'random' \
                 initialisation cannot be reproduced",
            ));
        }
        let m = f64_array(C, "C")?;
        let m3 = to3(m.as_array(), "C")?;
        let cfg = cqt_config(
            hop_length,
            fmin,
            None,
            bins_per_octave,
            Some(tuning),
            filter_scale,
            norm,
            sparsity,
            window,
            scale,
            pad_mode,
            res_type,
        )?;
        let sr = nzu32_or_err(crate::args::sample_rate_u32(sample_rate)?)?;
        let length = opt_positive("length", length)?;
        let y = <AudioSamples<'static, f64> as AudioConstantQ>::griffinlim_cqt(
            &m3,
            &cfg,
            sr,
            non_negative("n_iter", n_iter)?,
            momentum,
            length,
        )
        .map_err(audio_err_to_py)?;
        Ok(Self::from_audio_samples(y))
    }

    /// Constant-Q chromagram, equivalent to
    /// ``librosa.feature.chroma_cqt(y=..., sr=self.sample_rate, ...)``.
    ///
    /// Args:
    ///     hop_length (int): Default 512.
    ///     fmin (Optional[float]): Lowest CQT frequency; None = C1.
    ///     norm (Optional[float]): Per-frame normalisation. Default ``numpy.inf``.
    ///     threshold (Optional[float]): Values below are zeroed. Default 0.0.
    ///     tuning (Optional[float]): Tuning in fractions of a CQT bin; None
    ///         (default) estimates it.
    ///     n_chroma (int): Default 12.
    ///     n_octaves (int): Default 7.
    ///     window: Must be None (librosa's chroma smoothing kernel is unsupported).
    ///     bins_per_octave (Optional[int]): CQT resolution (multiple of
    ///         ``n_chroma``). Default 36; None = ``n_chroma``.
    ///     cqt_mode (str): ``"full"`` (default) or ``"hybrid"``.
    ///
    /// Returns:
    ///     numpy.ndarray[float64]: shape ``(channels, n_chroma, n_frames)``.
    ///
    /// Raises:
    ///     ValueError: For invalid parameters.
    #[pyo3(
        signature = (*, hop_length = Int(512), fmin = None, norm = Some(f64::INFINITY), threshold = Some(0.0), tuning = None, n_chroma = Int(12), n_octaves = Int(7), window = None, bins_per_octave = Some(Int(36)), cqt_mode = "full"),
        text_signature = "($self, *, hop_length=512, fmin=None, norm=math.inf, threshold=0.0, tuning=None, n_chroma=12, n_octaves=7, window=None, bins_per_octave=36, cqt_mode='full')"
    )]
    #[allow(clippy::too_many_arguments)]
    fn chroma_cqt<'py>(
        &self,
        py: Python<'py>,
        hop_length: Int,
        fmin: Option<f64>,
        norm: Option<f64>,
        threshold: Option<f64>,
        tuning: Option<f64>,
        n_chroma: Int,
        n_octaves: Int,
        window: Option<&Bound<'py, PyAny>>,
        bins_per_octave: Option<Int>,
        cqt_mode: &str,
    ) -> PyResult<Bound<'py, PyArray3<f64>>> {
        check_window_none(window)?;
        let cfg = chroma_cqt_config(
            hop_length,
            fmin,
            norm,
            threshold,
            tuning,
            n_chroma,
            n_octaves,
            bins_per_octave,
            cqt_mode,
        )?;
        let c = dispatch_with_view!(self, py, |audio| audio.chroma_cqt(&cfg))
            .map_err(audio_err_to_py)?;
        Ok(c.into_pyarray(py))
    }

    /// Chroma Energy Normalized Statistics, equivalent to
    /// ``librosa.feature.chroma_cens(y=..., sr=self.sample_rate, ...)``.
    ///
    /// Args:
    ///     hop_length, fmin, tuning, n_chroma, n_octaves, bins_per_octave, cqt_mode, window: As ``chroma_cqt``.
    ///     norm (Optional[float]): Final per-frame normalisation. Default 2.
    ///     win_len_smooth (Optional[int]): Smoothing length in frames (None
    ///         disables). Default 41.
    ///     smoothing_window: Smoothing window shape. Default ``"hann"``.
    ///
    /// Returns:
    ///     numpy.ndarray[float64]: shape ``(channels, n_chroma, n_frames)``.
    ///
    /// Raises:
    ///     ValueError: For invalid parameters.
    #[pyo3(
        signature = (*, hop_length = Int(512), fmin = None, tuning = None, n_chroma = Int(12), n_octaves = Int(7), bins_per_octave = Int(36), cqt_mode = "full", window = None, norm = Some(2.0), win_len_smooth = Some(Int(41)), smoothing_window = WindowArg(Window::Hann)),
        text_signature = "($self, *, hop_length=512, fmin=None, tuning=None, n_chroma=12, n_octaves=7, bins_per_octave=36, cqt_mode='full', window=None, norm=2, win_len_smooth=41, smoothing_window='hann')"
    )]
    #[allow(clippy::too_many_arguments)]
    fn chroma_cens<'py>(
        &self,
        py: Python<'py>,
        hop_length: Int,
        fmin: Option<f64>,
        tuning: Option<f64>,
        n_chroma: Int,
        n_octaves: Int,
        bins_per_octave: Int,
        cqt_mode: &str,
        window: Option<&Bound<'py, PyAny>>,
        norm: Option<f64>,
        win_len_smooth: Option<Int>,
        smoothing_window: WindowArg,
    ) -> PyResult<Bound<'py, PyArray3<f64>>> {
        check_window_none(window)?;
        let cfg = ChromaCensConfig {
            chroma: chroma_cqt_config(
                hop_length,
                fmin,
                None,
                Some(0.0),
                tuning,
                n_chroma,
                n_octaves,
                Some(bins_per_octave),
                cqt_mode,
            )?,
            norm,
            win_len_smooth: opt_positive("win_len_smooth", win_len_smooth)?,
            smoothing_window: smoothing_window.0,
        };
        let c = dispatch_with_view!(self, py, |audio| audio.chroma_cens(&cfg))
            .map_err(audio_err_to_py)?;
        Ok(c.into_pyarray(py))
    }

    /// Variable-Q chromagram, equivalent to
    /// ``librosa.feature.chroma_vqt(y=..., sr=self.sample_rate, intervals="equal", ...)``.
    ///
    /// Args:
    ///     hop_length (int): Default 512.
    ///     fmin (Optional[float]): Lowest VQT frequency; None = C1.
    ///     intervals (str): Only ``"equal"`` (the default here; required in librosa).
    ///     norm (Optional[float]): Per-frame normalisation. Default ``numpy.inf``.
    ///     threshold (Optional[float]): Values below are zeroed. Default 0.0.
    ///     n_octaves (int): Default 7.
    ///     bins_per_octave (int): VQT bins per octave (= chroma bins). Default 12.
    ///     gamma (Optional[float]): Bandwidth offset (Hz); 0 (default) = CQT, None =
    ///         ERB-derived.
    ///
    /// Returns:
    ///     numpy.ndarray[float64]: shape ``(channels, bins_per_octave, n_frames)``.
    ///
    /// Raises:
    ///     ValueError: For invalid parameters or ``intervals`` other than ``"equal"``.
    #[pyo3(
        signature = (*, hop_length = Int(512), fmin = None, intervals = None, norm = Some(f64::INFINITY), threshold = Some(0.0), n_octaves = Int(7), bins_per_octave = Int(12), gamma = Some(0.0)),
        text_signature = "($self, *, hop_length=512, fmin=None, intervals='equal', norm=math.inf, threshold=0.0, n_octaves=7, bins_per_octave=12, gamma=0)"
    )]
    #[allow(clippy::too_many_arguments)]
    fn chroma_vqt<'py>(
        &self,
        py: Python<'py>,
        hop_length: Int,
        fmin: Option<f64>,
        intervals: Option<&Bound<'py, PyAny>>,
        norm: Option<f64>,
        threshold: Option<f64>,
        n_octaves: Int,
        bins_per_octave: Int,
        gamma: Option<f64>,
    ) -> PyResult<Bound<'py, PyArray3<f64>>> {
        if let Some(i) = intervals {
            check_intervals(i)?;
        }
        let cfg = ChromaVqtConfig {
            hop_length: positive("hop_length", hop_length)?,
            fmin,
            norm,
            threshold,
            n_octaves: positive("n_octaves", n_octaves)?,
            bins_per_octave: positive("bins_per_octave", bins_per_octave)?,
            gamma,
        };
        let c = dispatch_with_view!(self, py, |audio| audio.chroma_vqt(&cfg))
            .map_err(audio_err_to_py)?;
        Ok(c.into_pyarray(py))
    }

    /// Tonal centroid features (tonnetz) from the constant-Q chroma, equivalent
    /// to ``librosa.feature.tonnetz(y=..., sr=self.sample_rate, ...)``.
    ///
    /// Args:
    ///     hop_length, fmin, norm, threshold, tuning, n_chroma, n_octaves, bins_per_octave, cqt_mode: ``chroma_cqt`` parameters (librosa's
    ///     ``**kwargs``).
    ///
    /// Returns:
    ///     numpy.ndarray[float64]: shape ``(channels, 6, n_frames)``.
    ///
    /// Raises:
    ///     ValueError: For invalid parameters.
    #[pyo3(
        signature = (*, hop_length = Int(512), fmin = None, norm = Some(f64::INFINITY), threshold = Some(0.0), tuning = None, n_chroma = Int(12), n_octaves = Int(7), bins_per_octave = Some(Int(36)), cqt_mode = "full"),
        text_signature = "($self, *, hop_length=512, fmin=None, norm=math.inf, threshold=0.0, tuning=None, n_chroma=12, n_octaves=7, bins_per_octave=36, cqt_mode='full')"
    )]
    #[allow(clippy::too_many_arguments)]
    fn tonnetz<'py>(
        &self,
        py: Python<'py>,
        hop_length: Int,
        fmin: Option<f64>,
        norm: Option<f64>,
        threshold: Option<f64>,
        tuning: Option<f64>,
        n_chroma: Int,
        n_octaves: Int,
        bins_per_octave: Option<Int>,
        cqt_mode: &str,
    ) -> PyResult<Bound<'py, PyArray3<f64>>> {
        let cfg = TonnetzConfig {
            chroma: chroma_cqt_config(
                hop_length,
                fmin,
                norm,
                threshold,
                tuning,
                n_chroma,
                n_octaves,
                bins_per_octave,
                cqt_mode,
            )?,
        };
        let c =
            dispatch_with_view!(self, py, |audio| audio.tonnetz(&cfg)).map_err(audio_err_to_py)?;
        Ok(c.into_pyarray(py))
    }
}
