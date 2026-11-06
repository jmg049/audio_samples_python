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
use audio_samples::operations::constant_q::griffinlim_cqt_random;
use audio_samples::operations::constant_q::{
    AudioConstantQ, ChromaCensConfig, ChromaCqtConfig, ChromaVqtConfig, CqtConfig, TonnetzConfig,
    VqtConfig, vqt_with_intervals,
};
use audio_samples::operations::effects::GriffinLimInit;
use audio_samples::operations::filterbanks::Window;
use audio_samples::{AudioSample, AudioSamples};
use pyo3::prelude::*;

use crate::args::{
    WindowArg, complex_array_as, float_array_as, non_negative, opt_positive, output_sample_type,
    parse_pad_mode, positive, to_3d, value_err, with_sample_type,
};
use crate::effects::{RngArg, griffinlim_init};
use crate::features::args::{check_intervals, parse_cqt_mode, parse_res_type};
use crate::types::PySampleType;
use crate::{PyAudioSamples, audio_err_to_py, dispatch_array_detached, nzu32_or_err};

/// Builds a [`CqtConfig`] from librosa's `cqt` keyword arguments.
#[allow(clippy::too_many_arguments)]
/// The ratios of one octave for a tuning system; None for equal temperament.
fn interval_ratios(
    intervals: &Bound<'_, PyAny>,
    bins_per_octave: Int,
) -> PyResult<Option<Vec<f64>>> {
    let system = crate::notation::parse_tuning(intervals)?;
    if matches!(system, audio_samples::utils::notation::TuningSystem::Equal) {
        return Ok(None);
    }
    let per_octave = match &system {
        audio_samples::utils::notation::TuningSystem::Custom(r) => r.len(),
        _ => crate::args::non_negative("bins_per_octave", bins_per_octave)?,
    };
    audio_samples::utils::notation::interval_frequencies(
        per_octave, 1.0, &system, per_octave, 0.0, true,
    )
    .map(Some)
    .map_err(crate::audio_err_to_py)
}

/// librosa's constant-Q keyword arguments, before validation.
struct CqtArgs<'a> {
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
    pad_mode: &'a str,
    res_type: &'a str,
}

impl CqtArgs<'_> {
    /// Validates the arguments into a [`CqtConfig`].
    fn config(self) -> PyResult<CqtConfig> {
        Ok(CqtConfig {
            hop_length: positive("hop_length", self.hop_length)?,
            fmin: self.fmin,
            n_bins: opt_positive("n_bins", self.n_bins)?,
            bins_per_octave: positive("bins_per_octave", self.bins_per_octave)?,
            tuning: self.tuning,
            filter_scale: self.filter_scale,
            norm: self.norm,
            sparsity: self.sparsity,
            window: self.window.0,
            scale: self.scale,
            pad_mode: parse_pad_mode(self.pad_mode)?,
            resampling_quality: parse_res_type(self.res_type)?,
        })
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
    ///
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
    ///
    ///     numpy.ndarray[complex64 | complex128]: shape ``(channels, n_bins, n_frames)``
    ///     (librosa returns ``(n_bins, n_frames)`` for 1-D input; index ``[0]``).
    ///
    /// Raises:
    ///
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
    ) -> PyResult<Bound<'py, PyAny>> {
        let cfg = CqtArgs {
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
        }
        .config()?;
        dispatch_array_detached!(self, py, |audio| audio.cqt(&cfg))
    }

    /// Variable-Q transform of every channel, equivalent to
    /// ``librosa.vqt(y, sr=self.sample_rate, ...)``.
    ///
    /// Args:
    ///
    ///     hop_length, fmin, n_bins, bins_per_octave, tuning, filter_scale, norm, sparsity, window, scale, pad_mode, res_type: As ``cqt``.
    ///     intervals (str): Only ``"equal"`` (equal temperament) is supported.
    ///     gamma (Optional[float]): Bandwidth offset (Hz); None (default) =
    ///         ERB-derived, 0 = constant-Q.
    ///
    /// Returns:
    ///
    ///     numpy.ndarray[complex64 | complex128]: shape ``(channels, n_bins, n_frames)``.
    ///
    /// Raises:
    ///
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
    ) -> PyResult<Bound<'py, PyAny>> {
        let ratios = match intervals.filter(|i| !i.is_none()) {
            Some(i) => interval_ratios(i, bins_per_octave)?,
            None => None,
        };
        let cfg = VqtConfig {
            cqt: CqtArgs {
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
            }
            .config()?,
            gamma,
        };
        if let Some(ratios) = ratios {
            return dispatch_array_detached!(self, py, |audio| vqt_with_intervals(
                &audio, &cfg, &ratios
            ));
        }
        dispatch_array_detached!(self, py, |audio| audio.vqt(&cfg))
    }

    /// Pseudo constant-Q magnitude (a single STFT with CQT filters), equivalent
    /// to ``librosa.pseudo_cqt(y, sr=self.sample_rate, ...)``.
    ///
    /// Args:
    ///
    ///     hop_length, fmin, n_bins, bins_per_octave, tuning, filter_scale, norm, sparsity, window, scale, pad_mode: As ``cqt``.
    ///
    /// Returns:
    ///
    ///     numpy.ndarray[float32 | float64]: magnitudes, shape ``(channels, n_bins, n_frames)``.
    ///
    /// Raises:
    ///
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
    ) -> PyResult<Bound<'py, PyAny>> {
        let cfg = CqtArgs {
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
            res_type: "soxr_hq",
        }
        .config()?;
        dispatch_array_detached!(self, py, |audio| audio.pseudo_cqt(&cfg))
    }

    /// Hybrid constant-Q magnitude (pseudo-CQT for the high, full CQT for the
    /// low bins), equivalent to ``librosa.hybrid_cqt(y, sr=self.sample_rate, ...)``.
    ///
    /// Args:
    ///
    ///     As ``cqt``.
    ///
    /// Returns:
    ///
    ///     numpy.ndarray[float32 | float64]: magnitudes, shape ``(channels, n_bins, n_frames)``.
    ///
    /// Raises:
    ///
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
    ) -> PyResult<Bound<'py, PyAny>> {
        let cfg = CqtArgs {
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
        }
        .config()?;
        dispatch_array_detached!(self, py, |audio| audio.hybrid_cqt(&cfg))
    }

    /// Inverse constant-Q transform, equivalent to ``librosa.icqt``.
    ///
    /// This is a static method: ``AudioSamples.icqt(C, sample_rate=sr)``.
    ///
    /// Args:
    ///
    ///     C (array_like): complex CQT of shape ``(n_bins, T)`` or ``(channels, n_bins, T)``.
    ///     sample_rate (int): Sample rate of the analysed signal (keyword-only).
    ///     hop_length, fmin, bins_per_octave, filter_scale, norm, sparsity, window, scale, res_type: The forward parameters (see ``cqt``).
    ///     tuning (float): Tuning used by the forward transform. Default 0.0.
    ///     length (Optional[int]): Exact output length per channel.
    ///     dtype (Optional[SampleType]): Sample type of the returned audio.
    ///         ``None`` (default) gives float32 audio for a complex64 ``C`` and
    ///         float64 audio for any other input. The computation runs in the
    ///         float type of ``dtype`` (float32 for U8, I16, I24 and F32; float64
    ///         for I32 and F64); integer output is rounded and saturated.
    ///
    /// Returns:
    ///
    ///     AudioSamples: audio of sample type ``dtype`` (mono for a single channel).
    ///
    /// Raises:
    ///
    ///     ValueError: For invalid parameters or an empty ``C``.
    #[staticmethod]
    #[pyo3(
        signature = (C, *, sample_rate, hop_length = Int(512), fmin = None, bins_per_octave = Int(12), tuning = 0.0, filter_scale = 1.0, norm = Some(1.0), sparsity = 0.01, window = WindowArg(Window::Hann), scale = true, length = None, res_type = "soxr_hq", dtype = None),
        text_signature = "(C, *, sample_rate, hop_length=512, fmin=None, bins_per_octave=12, tuning=0.0, filter_scale=1, norm=1, sparsity=0.01, window='hann', scale=True, length=None, res_type='soxr_hq', dtype=None)"
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
        dtype: Option<PySampleType>,
    ) -> PyResult<Self> {
        let cfg = CqtArgs {
            hop_length,
            fmin,
            n_bins: None,
            bins_per_octave,
            tuning: Some(tuning),
            filter_scale,
            norm,
            sparsity,
            window,
            scale,
            pad_mode: "constant",
            res_type,
        }
        .config()?;
        let sr = nzu32_or_err(crate::args::sample_rate_u32(sample_rate)?)?;
        let length = opt_positive("length", length)?;
        with_sample_type!(output_sample_type(dtype, C), |T| {
            let c = complex_array_as::<<T as AudioSample>::Float>(C, "C")?;
            let c3 = to_3d(c.as_array(), "C")?;
            let y = <AudioSamples<'static, T> as AudioConstantQ>::icqt(&c3, &cfg, sr, length)
                .map_err(audio_err_to_py)?;
            Ok(Self::from_audio_samples(y))
        })
    }

    /// Reconstructs audio from a constant-Q magnitude by Griffin-Lim, equivalent
    /// to ``librosa.griffinlim_cqt(C, sr=sample_rate, ...)``.
    ///
    /// This is a static method. The default is the deterministic zero-phase start
    /// (librosa's ``init=None``); librosa's default is ``init="random"``.
    ///
    /// Args:
    ///
    ///     C (array_like): Non-negative CQT magnitude, ``(n_bins, T)`` or
    ///         ``(channels, n_bins, T)``.
    ///     sample_rate (int): Sample rate (keyword-only).
    ///     n_iter (int): Iterations. Default 32.
    ///     hop_length, fmin, bins_per_octave, tuning, filter_scale, norm, sparsity, window, scale, pad_mode, res_type: The forward parameters (see ``cqt``).
    ///     length (Optional[int]): Output length per channel.
    ///     momentum (float): Fast Griffin-Lim momentum. Default 0.99.
    ///     init (Optional[str]): None (default: zero phase, deterministic) or
    ///         ``"random"``.
    ///     rng: Seed for ``init="random"``, as ``griffinlim``: an integer (default 0),
    ///         None for a fresh seed on every call, or a NumPy generator, from which
    ///         one seed is drawn. The random phases come from this library's
    ///         generator, so they differ from librosa's for the same seed.
    ///     random_state: Deprecated alias of ``rng``.
    ///     dtype (Optional[SampleType]): Sample type of the returned audio.
    ///         ``None`` (default) gives float32 audio for a float32 ``C`` and
    ///         float64 audio for any other input. The computation runs in the
    ///         float type of ``dtype`` (float32 for U8, I16, I24 and F32; float64
    ///         for I32 and F64); integer output is rounded and saturated.
    ///
    /// Returns:
    ///
    ///     AudioSamples: audio of sample type ``dtype``.
    ///
    /// Raises:
    ///
    ///     ValueError: For invalid parameters or ``init`` other than None or
    ///         ``"random"``.
    #[staticmethod]
    #[pyo3(
        signature = (C, *, sample_rate, n_iter = Int(32), hop_length = Int(512), fmin = None, bins_per_octave = Int(12), tuning = 0.0, filter_scale = 1.0, norm = Some(1.0), sparsity = 0.01, window = WindowArg(Window::Hann), scale = true, pad_mode = "constant", res_type = "soxr_hq", length = None, momentum = 0.99, init = None, rng = RngArg::SEEDED, random_state = None, dtype = None),
        text_signature = "(C, *, sample_rate, n_iter=32, hop_length=512, fmin=None, bins_per_octave=12, tuning=0.0, filter_scale=1, norm=1, sparsity=0.01, window='hann', scale=True, pad_mode='constant', res_type='soxr_hq', length=None, momentum=0.99, init=None, rng=0, random_state=None, dtype=None)"
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
        init: Option<&str>,
        rng: RngArg,
        random_state: Option<RngArg>,
        dtype: Option<PySampleType>,
    ) -> PyResult<Self> {
        let init = griffinlim_init(init, random_state.unwrap_or(rng))?;
        let cfg = CqtArgs {
            hop_length,
            fmin,
            n_bins: None,
            bins_per_octave,
            tuning: Some(tuning),
            filter_scale,
            norm,
            sparsity,
            window,
            scale,
            pad_mode,
            res_type,
        }
        .config()?;
        let sr = nzu32_or_err(crate::args::sample_rate_u32(sample_rate)?)?;
        let length = opt_positive("length", length)?;
        let n_iter = non_negative("n_iter", n_iter)?;
        with_sample_type!(output_sample_type(dtype, C), |T| {
            let m = float_array_as::<<T as AudioSample>::Float>(C, "C")?;
            let m3 = to_3d(m.as_array(), "C")?;
            let y = match init {
                GriffinLimInit::ZeroPhase => {
                    <AudioSamples<'static, T> as AudioConstantQ>::griffinlim_cqt(
                        &m3, &cfg, sr, n_iter, momentum, length,
                    )
                }
                GriffinLimInit::Random(seed) => {
                    griffinlim_cqt_random::<T>(&m3, &cfg, sr, n_iter, momentum, length, seed)
                }
            }
            .map_err(audio_err_to_py)?;
            Ok(Self::from_audio_samples(y))
        })
    }

    /// Constant-Q chromagram, equivalent to
    /// ``librosa.feature.chroma_cqt(y=..., sr=self.sample_rate, ...)``.
    ///
    /// Args:
    ///
    ///     hop_length (int): Default 512.
    ///     fmin (Optional[float]): Lowest CQT frequency; None = C1.
    ///     norm (Optional[float]): Per-frame normalisation. Default ``numpy.inf``.
    ///     threshold (Optional[float]): Values below are zeroed. Default 0.0.
    ///     tuning (Optional[float]): Tuning in fractions of a CQT bin. Default
    ///         0.0; None estimates it from the signal (librosa's default).
    ///     n_chroma (int): Default 12.
    ///     n_octaves (int): Default 7.
    ///     window: Must be None (librosa's chroma smoothing kernel is unsupported).
    ///     bins_per_octave (Optional[int]): CQT resolution (multiple of
    ///         ``n_chroma``). Default 36; None = ``n_chroma``.
    ///     cqt_mode (str): ``"full"`` (default) or ``"hybrid"``.
    ///
    /// Returns:
    ///
    ///     numpy.ndarray[float32 | float64]: shape ``(channels, n_chroma, n_frames)``.
    ///
    /// Raises:
    ///
    ///     ValueError: For invalid parameters.
    #[pyo3(
        signature = (*, hop_length = Int(512), fmin = None, norm = Some(f64::INFINITY), threshold = Some(0.0), tuning = Some(0.0), n_chroma = Int(12), n_octaves = Int(7), window = None, bins_per_octave = Some(Int(36)), cqt_mode = "full"),
        text_signature = "($self, *, hop_length=512, fmin=None, norm=math.inf, threshold=0.0, tuning=0.0, n_chroma=12, n_octaves=7, window=None, bins_per_octave=36, cqt_mode='full')"
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
    ) -> PyResult<Bound<'py, PyAny>> {
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
        dispatch_array_detached!(self, py, |audio| audio.chroma_cqt(&cfg))
    }

    /// Chroma Energy Normalized Statistics, equivalent to
    /// ``librosa.feature.chroma_cens(y=..., sr=self.sample_rate, ...)``.
    ///
    /// Args:
    ///
    ///     hop_length, fmin, tuning, n_chroma, n_octaves, bins_per_octave, cqt_mode, window: As ``chroma_cqt``.
    ///     norm (Optional[float]): Final per-frame normalisation. Default 2.
    ///     win_len_smooth (Optional[int]): Smoothing length in frames (None
    ///         disables). Default 41.
    ///     smoothing_window: Smoothing window shape. Default ``"hann"``.
    ///
    /// Returns:
    ///
    ///     numpy.ndarray[float32 | float64]: shape ``(channels, n_chroma, n_frames)``.
    ///
    /// Raises:
    ///
    ///     ValueError: For invalid parameters.
    #[pyo3(
        signature = (*, hop_length = Int(512), fmin = None, tuning = Some(0.0), n_chroma = Int(12), n_octaves = Int(7), bins_per_octave = Int(36), cqt_mode = "full", window = None, norm = Some(2.0), win_len_smooth = Some(Int(41)), smoothing_window = WindowArg(Window::Hann)),
        text_signature = "($self, *, hop_length=512, fmin=None, tuning=0.0, n_chroma=12, n_octaves=7, bins_per_octave=36, cqt_mode='full', window=None, norm=2, win_len_smooth=41, smoothing_window='hann')"
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
    ) -> PyResult<Bound<'py, PyAny>> {
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
        dispatch_array_detached!(self, py, |audio| audio.chroma_cens(&cfg))
    }

    /// Variable-Q chromagram, equivalent to
    /// ``librosa.feature.chroma_vqt(y=..., sr=self.sample_rate, intervals="equal", ...)``.
    ///
    /// Args:
    ///
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
    ///
    ///     numpy.ndarray[float32 | float64]: shape ``(channels, bins_per_octave, n_frames)``.
    ///
    /// Raises:
    ///
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
    ) -> PyResult<Bound<'py, PyAny>> {
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
        dispatch_array_detached!(self, py, |audio| audio.chroma_vqt(&cfg))
    }

    /// Tonal centroid features (tonnetz) from the constant-Q chroma, equivalent
    /// to ``librosa.feature.tonnetz(y=..., sr=self.sample_rate, ...)``.
    ///
    /// Args:
    ///
    ///     hop_length, fmin, norm, threshold, tuning, n_chroma, n_octaves, bins_per_octave, cqt_mode: ``chroma_cqt`` parameters (librosa's
    ///     ``**kwargs``).
    ///
    /// Returns:
    ///
    ///     numpy.ndarray[float32 | float64]: shape ``(channels, 6, n_frames)``.
    ///
    /// Raises:
    ///
    ///     ValueError: For invalid parameters.
    #[pyo3(
        signature = (*, hop_length = Int(512), fmin = None, norm = Some(f64::INFINITY), threshold = Some(0.0), tuning = Some(0.0), n_chroma = Int(12), n_octaves = Int(7), bins_per_octave = Some(Int(36)), cqt_mode = "full"),
        text_signature = "($self, *, hop_length=512, fmin=None, norm=math.inf, threshold=0.0, tuning=0.0, n_chroma=12, n_octaves=7, bins_per_octave=36, cqt_mode='full')"
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
    ) -> PyResult<Bound<'py, PyAny>> {
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
        dispatch_array_detached!(self, py, |audio| audio.tonnetz(&cfg))
    }
}
