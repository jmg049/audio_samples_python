//! Rhythm methods on `AudioSamples` (`onset_strength`,
//! `onset_strength_multi`, `onset_detect`, `tempogram`, `fourier_tempogram`,
//! `tempo`, `beat_track`, `plp`, `tempogram_ratio`, `hybrid_tempogram`),
//! binding the Rust `AudioRhythm` trait.
//!
//! Frame-wise outputs carry a leading channel axis (mono audio gives length 1):
//! the onset envelope is `(channels, n_frames)`, tempograms
//! `(channels, n_bins, n_frames)`. Event detectors (`onset_detect`,
//! `beat_track`) need mono audio and return 1-D event arrays. The envelope
//! behind the tempo/beat methods is computed from the audio (default mel onset
//! strength with the given `hop_length`; median aggregation for `beat_track`
//! and `plp`).

use crate::args::Int;
use audio_samples::operations::filterbanks::Window;
use audio_samples::operations::rhythm::{self as rh, Aggregate, OnsetStrengthConfig};
use audio_samples::operations::spectrum::MelSpectrogramConfig;
use audio_samples::{AudioRhythm, MelFilterNorm};
use ndarray::Array2;
use numpy::IntoPyArray;
use pyo3::prelude::*;

use crate::args::{MelNormArg, WindowArg, mel_config, positive, stft_config, value_err};
use crate::rhythm::{
    AggregateArg, CallbackScope, ChannelsArg, detect_onsets, fourier_tempogram_config,
    hybrid_tempogram_config, onset_peak_pick, parse_aggregate, parse_channels, parse_prior,
    plp_config, ratio_config, require_sparse, tempo_config, tempogram_config, track_beats,
};
use crate::{PyAudioSamples, audio_err_to_py, dispatch_typed, dispatch_typed_detached};

/// An onset envelope `(channels, n_frames)` in the float type of the audio's
/// sample type.
enum Envelope {
    F32(Array2<f32>),
    F64(Array2<f64>),
}

impl From<Array2<f32>> for Envelope {
    fn from(a: Array2<f32>) -> Self {
        Self::F32(a)
    }
}

impl From<Array2<f64>> for Envelope {
    fn from(a: Array2<f64>) -> Self {
        Self::F64(a)
    }
}

impl Envelope {
    fn nrows(&self) -> usize {
        match self {
            Self::F32(a) => a.nrows(),
            Self::F64(a) => a.nrows(),
        }
    }
}

/// Runs `$body` on the array inside an [`Envelope`], whatever its float type.
macro_rules! with_envelope {
    ($env:expr, |$e:ident| $body:expr) => {
        match $env {
            Envelope::F32($e) => $body,
            Envelope::F64($e) => $body,
        }
    };
}

/// The default onset envelope for a hop length (what `tempo`,
/// `tempogram`, `beat_track`, ... compute from `y`).
fn envelope_config(hop_length: i64, aggregate: Aggregate) -> PyResult<OnsetStrengthConfig> {
    let mut cfg = OnsetStrengthConfig::new().with_aggregate(Some(aggregate));
    cfg.mel.stft.hop_length = Some(positive("hop_length", hop_length)?);
    Ok(cfg)
}

impl PyAudioSamples {
    fn sample_rate_f64(&self, py: Python<'_>) -> f64 {
        crate::dispatch_without_borrow!(self, py, |audio| audio.sample_rate_hz())
    }

    /// Onset envelope `(channels, n_frames)` for `cfg`.
    fn envelope(&self, py: Python<'_>, cfg: &OnsetStrengthConfig) -> PyResult<Envelope> {
        // Built-in aggregates are pure Rust: release the GIL. A Python callable
        // (Aggregate::Custom trampoline) needs it.
        if matches!(cfg.aggregate, Some(Aggregate::Custom(_))) {
            dispatch_typed!(self, py, |audio| audio.onset_strength(cfg), |r| r
                .map(Envelope::from))
        } else {
            dispatch_typed_detached!(self, py, |audio| audio.onset_strength(cfg), |r| r
                .map(Envelope::from))
        }
        .map_err(audio_err_to_py)
    }

    /// The single envelope row of mono audio, for the event detectors.
    fn mono_envelope(
        &self,
        py: Python<'_>,
        cfg: &OnsetStrengthConfig,
        operation: &str,
    ) -> PyResult<Envelope> {
        let env = self.envelope(py, cfg)?;
        if env.nrows() != 1 {
            return Err(value_err(
                operation,
                format!(
                    "requires mono audio, got {} channels (convert with to_mono first)",
                    env.nrows()
                ),
            ));
        }
        Ok(env)
    }
}

/// Builds the onset-strength config of the `onset_strength*` methods.
#[allow(
    clippy::too_many_arguments,
    reason = "the Python signature takes these arguments"
)]
fn onset_strength_config(
    lag: Int,
    max_size: Int,
    detrend: bool,
    center: bool,
    aggregate: Option<Aggregate>,
    n_fft: Int,
    hop_length: Int,
    win_length: Option<Int>,
    window: WindowArg,
    pad_mode: &str,
    power: f64,
    n_mels: Int,
    fmin: f64,
    fmax: Option<f64>,
    htk: bool,
    norm: MelNormArg,
) -> PyResult<OnsetStrengthConfig> {
    Ok(OnsetStrengthConfig::new()
        .with_lag(positive("lag", lag)?)
        .with_max_size(positive("max_size", max_size)?)
        .with_detrend(detrend)
        .with_center(center)
        .with_aggregate(aggregate)
        .with_mel(MelSpectrogramConfig {
            stft: stft_config(n_fft, Some(hop_length), win_length, window, true, pad_mode)?,
            mel: mel_config(n_mels, fmin, fmax, htk, norm)?,
            power,
        }))
}

#[pymethods]
impl PyAudioSamples {
    /// Onset strength envelope of every channel.
    ///
    /// The envelope is the aggregated positive increase of the log-power mel
    /// spectrogram (``power_to_db`` with ``ref=1``, ``top_db=80``) from frame
    /// ``t - lag`` to ``t``.
    ///
    /// Args:
    ///
    ///     lag (int): Time lag of the difference in frames. Default 1.
    ///     max_size (int): Maximum filter across frequency on the reference
    ///         (SuperFlux); 1 disables it. Default 1.
    ///     detrend (bool): High-pass filter the envelope. Default False.
    ///     center (bool): Compensate the centred-STFT delay. Default True.
    ///     aggregate: Reduction across mel bins: None (mean), ``numpy.mean``,
    ///         ``numpy.median``, ``numpy.max``, ``numpy.min``, ``numpy.sum``, a
    ///         name, or a callable ``f(x, axis=-1)``.
    ///     n_fft (int): Default 2048.
    ///     hop_length (int): Default 512.
    ///     win_length, window, pad_mode, power, n_mels, fmin, fmax, htk, norm:
    ///         Mel spectrogram parameters, as ``melspectrogram``.
    ///
    /// Returns:
    ///
    ///     numpy.ndarray[float32 | float64]: shape ``(channels, n_frames)`` with
    ///     ``n_frames = 1 + len // hop_length``; index ``[0]`` for the
    ///     ``(n_frames,)`` envelope of mono audio.
    ///
    /// Raises:
    ///
    ///     ValueError: For invalid parameters.
    #[pyo3(
        signature = (*, lag = Int(1), max_size = Int(1), detrend = false, center = true, aggregate = None, n_fft = Int(2048), hop_length = Int(512), win_length = None, window = WindowArg(Window::Hann), pad_mode = "constant", power = 2.0, n_mels = Int(128), fmin = 0.0, fmax = None, htk = false, norm = MelNormArg(MelFilterNorm::Slaney)) -> "NDArray[Any]",
        text_signature = "($self, *, lag=1, max_size=1, detrend=False, center=True, aggregate=None, n_fft=2048, hop_length=512, win_length=None, window='hann', pad_mode='constant', power=2.0, n_mels=128, fmin=0.0, fmax=None, htk=False, norm='slaney')"
    )]
    #[allow(
        clippy::too_many_arguments,
        reason = "the Python signature takes these arguments"
    )]
    fn onset_strength<'py>(
        &self,
        py: Python<'py>,
        lag: Int,
        max_size: Int,
        detrend: bool,
        center: bool,
        aggregate: Option<&Bound<'py, PyAny>>,
        n_fft: Int,
        hop_length: Int,
        win_length: Option<Int>,
        window: WindowArg,
        pad_mode: &str,
        power: f64,
        n_mels: Int,
        fmin: f64,
        fmax: Option<f64>,
        htk: bool,
        norm: MelNormArg,
    ) -> PyResult<Bound<'py, PyAny>> {
        let agg = parse_aggregate(aggregate)?;
        let cfg = onset_strength_config(
            lag,
            max_size,
            detrend,
            center,
            agg.resolve(Some(Aggregate::Mean)),
            n_fft,
            hop_length,
            win_length,
            window,
            pad_mode,
            power,
            n_mels,
            fmin,
            fmax,
            htk,
            norm,
        )?;
        if agg.callable().is_none() {
            // Pure Rust: the GIL is released. A Python callable needs it.
            return dispatch_typed_detached!(self, py, |audio| audio.onset_strength(&cfg), |r| r
                .map_err(audio_err_to_py)
                .map(|env| env.into_pyarray(py).into_any()));
        }
        let scope = CallbackScope::install(agg.callable(), None, py);
        dispatch_typed!(self, py, |audio| audio.onset_strength(&cfg), |r| scope
            .finish(r)
            .map(|env| env.into_pyarray(py).into_any()))
    }

    /// Multi-band onset strength of every channel: one envelope per group of
    /// mel bands.
    ///
    /// Args:
    ///
    ///     channels (Optional[Sequence[int] | Sequence[slice]]): Sub-band boundaries in
    ///         mel bins: values above ``n_mels`` are dropped, the rest sorted and
    ///         de-duplicated, each consecutive pair is one band. Or one slice of the
    ///         mel bins per band, in any order. None gives one band.
    ///     aggregate: As ``onset_strength``; ``False`` keeps every mel bin.
    ///     **kwargs: ``lag``, ``max_size``, ``detrend``, ``center``, ``n_fft``,
    ///         ``hop_length``, ``win_length``, ``window``, ``pad_mode``, ``power``,
    ///         ``n_mels``, ``fmin``, ``fmax``, ``htk``, ``norm`` as ``onset_strength``.
    ///
    /// Returns:
    ///
    ///     numpy.ndarray[float32 | float64]: shape ``(channels, n_bands, n_frames)``
    ///     (``n_mels`` rows with ``aggregate=False``); index ``[0]`` for the
    ///     ``(n_bands, n_frames)`` array of mono audio.
    ///
    /// Raises:
    ///
    ///     ValueError: For invalid parameters or boundaries.
    #[pyo3(
        signature = (*, channels = None, lag = Int(1), max_size = Int(1), detrend = false, center = true, aggregate = None, n_fft = Int(2048), hop_length = Int(512), win_length = None, window = WindowArg(Window::Hann), pad_mode = "constant", power = 2.0, n_mels = Int(128), fmin = 0.0, fmax = None, htk = false, norm = MelNormArg(MelFilterNorm::Slaney)) -> "NDArray[Any]",
        text_signature = "($self, *, channels=None, lag=1, max_size=1, detrend=False, center=True, aggregate=None, n_fft=2048, hop_length=512, win_length=None, window='hann', pad_mode='constant', power=2.0, n_mels=128, fmin=0.0, fmax=None, htk=False, norm='slaney')"
    )]
    #[allow(
        clippy::too_many_arguments,
        reason = "the Python signature takes these arguments"
    )]
    fn onset_strength_multi<'py>(
        &self,
        py: Python<'py>,
        channels: Option<&Bound<'py, PyAny>>,
        lag: Int,
        max_size: Int,
        detrend: bool,
        center: bool,
        aggregate: Option<&Bound<'py, PyAny>>,
        n_fft: Int,
        hop_length: Int,
        win_length: Option<Int>,
        window: WindowArg,
        pad_mode: &str,
        power: f64,
        n_mels: Int,
        fmin: f64,
        fmax: Option<f64>,
        htk: bool,
        norm: MelNormArg,
    ) -> PyResult<Bound<'py, PyAny>> {
        let agg = parse_aggregate(aggregate)?;
        let cfg = onset_strength_config(
            lag,
            max_size,
            detrend,
            center,
            agg.resolve(Some(Aggregate::Mean)),
            n_fft,
            hop_length,
            win_length,
            window,
            pad_mode,
            power,
            n_mels,
            fmin,
            fmax,
            htk,
            norm,
        )?;
        let bands = parse_channels(channels, cfg.mel.mel.n_mels.get())?;
        let scope = CallbackScope::install(agg.callable(), None, py);
        dispatch_typed!(
            self,
            py,
            |audio| match &bands {
                ChannelsArg::Bounds(bounds) => audio.onset_strength_multi(&cfg, bounds.as_deref()),
                ChannelsArg::Slices(slices) =>
                    rh::onset_strength_multi_slices(&audio, &cfg, slices),
            },
            |r| scope.finish(r).map(|env| env.into_pyarray(py).into_any())
        )
    }

    /// Detects onsets by peak picking on the onset strength envelope.
    ///
    /// The default onset envelope (``onset_strength(hop_length=hop_length)``) is
    /// normalised to ``[0, 1]``, peak-picked and optionally backtracked.
    ///
    /// Args:
    ///
    ///     hop_length (int): Hop of the onset envelope. Default 512.
    ///     backtrack (bool): Move onsets back to the preceding local minimum of
    ///         ``energy``. Default False.
    ///     energy (Optional[array_like]): 1-D function to backtrack on (one value
    ///         per frame); None uses the normalised envelope.
    ///     units (str): ``"frames"`` (default), ``"samples"`` or ``"time"``.
    ///     normalize (bool): Scale the envelope to ``[0, 1]``. Default True.
    ///     sparse (bool): Must be True.
    ///     pre_max, post_max, pre_avg, post_avg, wait (Optional[int]),
    ///     delta (Optional[float]), method (str): Peak-picking parameters
    ///         (see ``rhythm.peak_pick``); None derives the value from the frame
    ///         rate (30 ms / 100 ms windows, ``delta=0.07``).
    ///
    /// Returns:
    ///
    ///     numpy.ndarray: 1-D onset frames or samples (int64), or times (float64).
    ///
    /// Raises:
    ///
    ///     ValueError: For multi-channel audio, invalid parameters or
    ///         ``sparse=False``.
    #[pyo3(
        signature = (*, hop_length = Int(512), backtrack = false, energy = None, units = "frames", normalize = true, sparse = true, pre_max = None, post_max = None, pre_avg = None, post_avg = None, delta = None, wait = None, method = "greedy") -> "NDArray[Any]",
        text_signature = "($self, *, hop_length=512, backtrack=False, energy=None, units='frames', normalize=True, sparse=True, pre_max=None, post_max=None, pre_avg=None, post_avg=None, delta=None, wait=None, method='greedy')"
    )]
    #[allow(
        clippy::too_many_arguments,
        reason = "the Python signature takes these arguments"
    )]
    fn onset_detect<'py>(
        &self,
        py: Python<'py>,
        hop_length: Int,
        backtrack: bool,
        energy: Option<&Bound<'py, PyAny>>,
        units: &str,
        normalize: bool,
        sparse: bool,
        pre_max: Option<Int>,
        post_max: Option<Int>,
        pre_avg: Option<Int>,
        post_avg: Option<Int>,
        delta: Option<f64>,
        wait: Option<Int>,
        method: &str,
    ) -> PyResult<Bound<'py, PyAny>> {
        require_sparse(sparse)?;
        let cfg = envelope_config(hop_length.0, Aggregate::Mean)?;
        let hop = positive("hop_length", hop_length)?;
        let sr = self.sample_rate_f64(py);
        let peak = onset_peak_pick(
            sr, hop, pre_max, post_max, pre_avg, post_avg, delta, wait, method,
        )?;
        let env = self.mono_envelope(py, &cfg, "onset_detect")?;
        with_envelope!(env, |e| detect_onsets(
            py,
            &e.row(0).to_vec(),
            sr,
            hop,
            peak,
            backtrack,
            energy,
            units,
            normalize
        ))
    }

    /// Autocorrelation tempogram of every channel.
    ///
    /// Args:
    ///
    ///     hop_length (int): Hop of the onset envelope. Default 512.
    ///     win_length (int): Autocorrelation window (lag bins). Default 384.
    ///     center (bool): Centre the windows. Default True.
    ///     window: Window before the autocorrelation. Default ``"hann"``.
    ///     norm (Optional[float]): Column normalisation (``numpy.inf`` default,
    ///         a p-norm, or None).
    ///
    /// Returns:
    ///
    ///     numpy.ndarray[float32 | float64]: shape ``(channels, win_length, n_frames)``;
    ///     index ``[0]`` for the 2-D array of mono audio.
    ///
    /// Raises:
    ///
    ///     ValueError: For invalid parameters.
    #[pyo3(
        signature = (*, hop_length = Int(512), win_length = Int(384), center = true, window = WindowArg(Window::Hann), norm = Some(f64::INFINITY)) -> "NDArray[Any]",
        text_signature = "($self, *, hop_length=512, win_length=384, center=True, window='hann', norm=math.inf)"
    )]
    fn tempogram<'py>(
        &self,
        py: Python<'py>,
        hop_length: Int,
        win_length: Int,
        center: bool,
        window: WindowArg,
        norm: Option<f64>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let tcfg = tempogram_config(win_length, center, window, norm)?;
        let env = self.envelope(py, &envelope_config(hop_length.0, Aggregate::Mean)?)?;
        with_envelope!(env, |e| rh::tempogram_from_envelope(e.view(), &tcfg)
            .map_err(audio_err_to_py)
            .map(|tg| tg.into_pyarray(py).into_any()))
    }

    /// Fourier tempogram of every channel.
    ///
    /// Args:
    ///
    ///     hop_length (int): Hop of the onset envelope. Default 512.
    ///     win_length (int): STFT length in frames. Default 384.
    ///     center (bool): Centred frames. Default True.
    ///     window: Analysis window. Default ``"hann"``.
    ///
    /// Returns:
    ///
    ///     numpy.ndarray[complex64 | complex128]: shape
    ///     ``(channels, win_length // 2 + 1, n_frames + 1)`` (centred); index
    ///     ``[0]`` for the 2-D array of mono audio.
    #[pyo3(
        signature = (*, hop_length = Int(512), win_length = Int(384), center = true, window = WindowArg(Window::Hann)) -> "NDArray[Any]",
        text_signature = "($self, *, hop_length=512, win_length=384, center=True, window='hann')"
    )]
    fn fourier_tempogram<'py>(
        &self,
        py: Python<'py>,
        hop_length: Int,
        win_length: Int,
        center: bool,
        window: WindowArg,
    ) -> PyResult<Bound<'py, PyAny>> {
        let fcfg = fourier_tempogram_config(win_length, center, window)?;
        let env = self.envelope(py, &envelope_config(hop_length.0, Aggregate::Mean)?)?;
        with_envelope!(env, |e| rh::fourier_tempogram_from_envelope(
            e.view(),
            &fcfg
        )
        .map_err(audio_err_to_py)
        .map(|tg| tg.into_pyarray(py).into_any()))
    }

    /// Estimates the tempo of every channel in BPM.
    ///
    /// Args:
    ///
    ///     hop_length (int): Hop of the onset envelope. Default 512.
    ///     start_bpm (float): Prior centre. Default 120.
    ///     std_bpm (float): Prior spread in octaves. Default 1.
    ///     ac_size (float): Autocorrelation window in seconds. Default 8.
    ///     max_tempo (Optional[float]): Exclude tempi at or above this. Default 320.
    ///     aggregate: Reduction over time (default ``numpy.mean``); None gives
    ///         one tempo per frame.
    ///     prior: None (log-normal), a ``scipy.stats`` distribution or a callable
    ///         of the BPM returning a log-density.
    ///
    /// Returns:
    ///
    ///     numpy.ndarray[float64]: shape ``(channels, 1)`` (``(channels, n_frames)``
    ///     with ``aggregate=None``); index ``[0]`` for the ``(1,)`` array of
    ///     mono audio.
    ///
    /// Raises:
    ///
    ///     ValueError: For invalid parameters.
    #[pyo3(
        signature = (*, hop_length = Int(512), start_bpm = 120.0, std_bpm = 1.0, ac_size = 8.0, max_tempo = Some(320.0), aggregate: "Callable[..., Any] | Literal['mean', 'median', 'max', 'min', 'sum']" = AggregateArg::mean(), prior = None) -> "NDArray[np.float64]",
        text_signature = "($self, *, hop_length=512, start_bpm=120, std_bpm=1.0, ac_size=8.0, max_tempo=320.0, aggregate='mean', prior=None)"
    )]
    #[allow(
        clippy::too_many_arguments,
        reason = "the Python signature takes these arguments"
    )]
    fn tempo<'py>(
        &self,
        py: Python<'py>,
        hop_length: Int,
        start_bpm: f64,
        std_bpm: f64,
        ac_size: f64,
        max_tempo: Option<f64>,
        aggregate: AggregateArg,
        prior: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let prior = parse_prior(prior)?;
        let agg = aggregate.0;
        let cfg = tempo_config(
            start_bpm,
            std_bpm,
            ac_size,
            max_tempo,
            agg.resolve(None),
            prior.as_ref(),
        );
        let hop = positive("hop_length", hop_length)?;
        let env = self.envelope(py, &envelope_config(hop_length.0, Aggregate::Mean)?)?;
        let sr = self.sample_rate_f64(py);
        let scope = CallbackScope::install(agg.callable(), prior.as_ref(), py);
        let t = scope.finish(with_envelope!(env, |e| rh::tempo_from_envelope(
            e.view(),
            sr,
            hop,
            &cfg
        )))?;
        Ok(t.into_pyarray(py).into_any())
    }

    /// Tracks beats with the dynamic-programming beat tracker.
    ///
    /// The onset envelope is ``onset_strength(hop_length=hop_length,
    /// aggregate=numpy.median)``.
    ///
    /// Args:
    ///
    ///     hop_length (int): Hop of the onset envelope. Default 512.
    ///     start_bpm (float): Prior centre of the tempo estimate. Default 120.
    ///     tightness (float): Tempo-consistency weight. Default 100.
    ///     trim (bool): Remove weak leading/trailing beats. Default False.
    ///     bpm (Optional[float | array_like]): Tempo to track at (one value or
    ///         one per frame); None estimates it.
    ///     prior: Tempo prior (see ``tempo``).
    ///     units (str): ``"frames"`` (default), ``"samples"`` or ``"time"``.
    ///     sparse (bool): Must be True.
    ///
    /// Returns:
    ///
    ///     tuple: ``(tempo, beats)``: ``tempo`` is the estimate as a
    ///     float64 array of shape ``(1,)`` (``bpm`` unchanged when given, ``0.0``
    ///     for silence); ``beats`` is a 1-D int64 (frames/samples) or float64
    ///     (time) array.
    ///
    /// Raises:
    ///
    ///     ValueError: For multi-channel audio, invalid parameters or
    ///         ``sparse=False``.
    #[pyo3(
        signature = (*, hop_length = Int(512), start_bpm = 120.0, tightness = 100.0, trim = false, bpm = None, prior = None, units = "frames", sparse = true),
        text_signature = "($self, *, hop_length=512, start_bpm=120.0, tightness=100, trim=False, bpm=None, prior=None, units='frames', sparse=True)"
    )]
    #[allow(
        clippy::too_many_arguments,
        reason = "the Python signature takes these arguments"
    )]
    fn beat_track<'py>(
        &self,
        py: Python<'py>,
        hop_length: Int,
        start_bpm: f64,
        tightness: f64,
        trim: bool,
        bpm: Option<&Bound<'py, PyAny>>,
        prior: Option<&Bound<'py, PyAny>>,
        units: &str,
        sparse: bool,
    ) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)> {
        require_sparse(sparse)?;
        let hop = positive("hop_length", hop_length)?;
        let env = self.mono_envelope(
            py,
            &envelope_config(hop_length.0, Aggregate::Median)?,
            "beat_track",
        )?;
        let sr = self.sample_rate_f64(py);
        with_envelope!(env, |e| track_beats(
            py,
            &e.row(0).to_vec(),
            sr,
            hop,
            start_bpm,
            tightness,
            trim,
            bpm,
            prior,
            units
        ))
    }

    /// Predominant local pulse of every channel.
    ///
    /// Args:
    ///
    ///     hop_length (int): Hop of the (median-aggregated) onset envelope.
    ///         Default 512.
    ///     win_length (int): Fourier tempogram window in frames. Default 384.
    ///     tempo_min (Optional[float]): Lowest tempo (BPM). Default 30.
    ///     tempo_max (Optional[float]): Highest tempo (BPM). Default 300.
    ///     prior: Tempo prior (see ``tempo``).
    ///
    /// Returns:
    ///
    ///     numpy.ndarray[float32 | float64]: shape ``(channels, n_frames)``, values in
    ///     ``[0, 1]``; index ``[0]`` for the ``(n_frames,)`` array of mono audio.
    ///
    /// Raises:
    ///
    ///     ValueError: If ``tempo_max <= tempo_min`` or parameters are invalid.
    #[pyo3(
        signature = (*, hop_length = Int(512), win_length = Int(384), tempo_min = Some(30.0), tempo_max = Some(300.0), prior = None) -> "NDArray[Any]",
        text_signature = "($self, *, hop_length=512, win_length=384, tempo_min=30, tempo_max=300, prior=None)"
    )]
    fn plp<'py>(
        &self,
        py: Python<'py>,
        hop_length: Int,
        win_length: Int,
        tempo_min: Option<f64>,
        tempo_max: Option<f64>,
        prior: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let prior = parse_prior(prior)?;
        let cfg = plp_config(win_length, tempo_min, tempo_max, prior.as_ref())?;
        let hop = positive("hop_length", hop_length)?;
        let env = self.envelope(py, &envelope_config(hop_length.0, Aggregate::Median)?)?;
        let sr = self.sample_rate_f64(py);
        let scope = CallbackScope::install(None, prior.as_ref(), py);
        with_envelope!(env, |e| scope
            .finish(rh::plp_from_envelope(e.view(), sr, hop, &cfg))
            .map(|p| p.into_pyarray(py).into_any()))
    }

    /// Tempogram ratio features of every channel: the tempogram sampled at
    /// fixed multiples of the per-frame tempo estimate.
    ///
    /// Args:
    ///
    ///     hop_length (int): Hop of the onset envelope. Default 512.
    ///     win_length, center, window, norm: Tempogram parameters (as ``tempogram``).
    ///     start_bpm, std_bpm, max_tempo, prior: Per-frame tempo estimation (as
    ///         ``tempo``).
    ///     factors (Optional[array_like]): Tempo multiples; None uses the 13
    ///         default factors.
    ///     aggregate: Optional reduction over time (default None).
    ///     fill_value (float): Value outside the tempogram range. Default 0.
    ///
    /// Returns:
    ///
    ///     numpy.ndarray[float32 | float64]: shape ``(channels, n_factors, n_frames)``, or
    ///     ``(channels, n_factors, 1)`` when aggregated (the time axis is kept);
    ///     index ``[0]`` for the array of mono audio.
    #[pyo3(
        signature = (*, hop_length = Int(512), win_length = Int(384), start_bpm = 120.0, std_bpm = 1.0, max_tempo = Some(320.0), factors = None, aggregate = None, prior = None, center = true, window = WindowArg(Window::Hann), fill_value = 0.0, norm = Some(f64::INFINITY)) -> "NDArray[Any]",
        text_signature = "($self, *, hop_length=512, win_length=384, start_bpm=120, std_bpm=1.0, max_tempo=320.0, factors=None, aggregate=None, prior=None, center=True, window='hann', fill_value=0, norm=math.inf)"
    )]
    #[allow(
        clippy::too_many_arguments,
        reason = "the Python signature takes these arguments"
    )]
    fn tempogram_ratio<'py>(
        &self,
        py: Python<'py>,
        hop_length: Int,
        win_length: Int,
        start_bpm: f64,
        std_bpm: f64,
        max_tempo: Option<f64>,
        factors: Option<&Bound<'py, PyAny>>,
        aggregate: Option<&Bound<'py, PyAny>>,
        prior: Option<&Bound<'py, PyAny>>,
        center: bool,
        window: WindowArg,
        fill_value: f64,
        norm: Option<f64>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let prior = parse_prior(prior)?;
        let agg = parse_aggregate(aggregate)?;
        let cfg = ratio_config(
            win_length,
            center,
            window,
            norm,
            start_bpm,
            std_bpm,
            max_tempo,
            prior.as_ref(),
            factors,
            agg.resolve(None),
            fill_value,
        )?;
        let hop = positive("hop_length", hop_length)?;
        let env = self.envelope(py, &envelope_config(hop_length.0, Aggregate::Mean)?)?;
        let sr = self.sample_rate_f64(py);
        with_envelope!(env, |e| {
            let tg =
                rh::tempogram_from_envelope(e.view(), &cfg.tempogram).map_err(audio_err_to_py)?;
            let scope = CallbackScope::install(agg.callable(), prior.as_ref(), py);
            scope
                .finish(rh::tempogram_ratio_from_tempogram(
                    tg.view(),
                    sr,
                    hop,
                    &cfg,
                    None,
                ))
                .map(|r| r.into_pyarray(py).into_any())
        })
    }

    /// Hybrid tempogram of every channel: the Fourier and autocorrelation
    /// tempograms combined on the Fourier tempo grid.
    ///
    /// Args:
    ///
    ///     hop_length (int): Hop of the onset envelope. Default 512.
    ///     win_length (int): Window of both tempograms (>= 3). Default 384.
    ///     center (bool): Centred frames. Default True.
    ///     window: Window of both tempograms. Default ``"hann"``.
    ///
    /// Returns:
    ///
    ///     numpy.ndarray[float32 | float64]: shape ``(channels, win_length // 2 + 1, n_frames)``
    ///     on the ``fourier_tempo_frequencies`` grid; index ``[0]`` for the 2-D
    ///     array of mono audio.
    #[pyo3(
        signature = (*, hop_length = Int(512), win_length = Int(384), center = true, window = WindowArg(Window::Hann)) -> "NDArray[Any]",
        text_signature = "($self, *, hop_length=512, win_length=384, center=True, window='hann')"
    )]
    fn hybrid_tempogram<'py>(
        &self,
        py: Python<'py>,
        hop_length: Int,
        win_length: Int,
        center: bool,
        window: WindowArg,
    ) -> PyResult<Bound<'py, PyAny>> {
        let hcfg = hybrid_tempogram_config(win_length, center, window)?;
        let hop = positive("hop_length", hop_length)?;
        let env = self.envelope(py, &envelope_config(hop_length.0, Aggregate::Mean)?)?;
        let sr = self.sample_rate_f64(py);
        with_envelope!(env, |e| rh::hybrid_tempogram_from_envelope(
            e.view(),
            sr,
            hop,
            &hcfg
        )
        .map_err(audio_err_to_py)
        .map(|h| h.into_pyarray(py).into_any()))
    }
}
