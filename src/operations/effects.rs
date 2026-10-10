//! librosa-compatible audio effects on `AudioSamples` (`time_stretch`,
//! `pitch_shift`, `preemphasis`, `deemphasis`, `trim_db`, `split_db`, `remix`,
//! `harmonic`, `percussive`, `harmonic_percussive`, `lpc`, `griffinlim_audio`),
//! binding the Rust `AudioEffects` trait.
//!
//! Audio-to-audio effects return a new `AudioSamples` of the input's sample
//! type and rate; every one has an `_in_place` variant that replaces the
//! audio (its length may change) and returns `None` (`trim_db_in_place`
//! returns the kept `[start, end]`).

use crate::args::Int;
use audio_samples::operations::effects::{
    HarmonicPercussiveConfig, PitchShiftConfig, SilenceConfig,
};
use audio_samples::operations::filterbanks::Window;
use audio_samples::operations::spectrum::DbReference;
use audio_samples::operations::types::ResamplingQuality;
use audio_samples::{AudioEffects, AudioSamples, ChannelAggregate};
use ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::PyString;

use crate::args::{DbRefArg, WindowArg, i64_vec, non_negative, positive, stft_config, value_err};
use crate::effects::{effects_err, griffinlim_config, hpss_mask_config};
use crate::rhythm::planes;
use crate::types::PyResamplingQuality;
use crate::{
    PyAudioSamples, audio_err_to_py, dispatch_with_view, dispatch_with_view_detached, nzu32_or_err,
};

/// Parses `res_type`: a `ResamplingQuality` or a name. librosa's soxr names map
/// to the closest tier of this library's resampler.
fn parse_res_type(obj: &Bound<'_, PyAny>) -> PyResult<ResamplingQuality> {
    if let Ok(s) = obj.cast::<PyString>() {
        return match &*s.to_cow()? {
            "soxr_hq" | "soxr_vhq" | "high" | "kaiser_best" => Ok(ResamplingQuality::High),
            "soxr_mq" | "medium" | "kaiser_fast" => Ok(ResamplingQuality::Medium),
            "soxr_lq" | "soxr_qq" | "fast" | "linear" => Ok(ResamplingQuality::Fast),
            other => Err(value_err(
                "res_type",
                format!(
                    "unsupported res_type {other:?}; expected 'soxr_hq' (default, -> High), \
                     'soxr_mq' (-> Medium), 'soxr_lq' (-> Fast), 'high', 'medium', 'fast' \
                     or a ResamplingQuality"
                ),
            )),
        };
    }
    obj.extract::<PyResamplingQuality>()
        .map(|q| q.inner)
        .map_err(|_| PyTypeError::new_err("res_type must be a string or a ResamplingQuality"))
}

/// librosa's `ref` / `aggregate` of `trim` / `split`.
fn silence_config(
    py: Python<'_>,
    top_db: f64,
    r#ref: DbRefArg,
    frame_length: Int,
    hop_length: Int,
    aggregate: Option<&Bound<'_, PyAny>>,
) -> PyResult<SilenceConfig> {
    let reference = r#ref.0;
    let aggregate = match aggregate.filter(|a| !a.is_none()) {
        None => ChannelAggregate::Max,
        Some(a) => {
            let np = py.import("numpy")?;
            let name = a.cast::<PyString>().ok().map(|s| s.to_string());
            let is = |n: &str| -> PyResult<bool> {
                Ok(name.as_deref() == Some(n) || a.is(&np.getattr(n)?))
            };
            if is("max")? || is("amax")? {
                ChannelAggregate::Max
            } else if is("min")? || is("amin")? {
                ChannelAggregate::Min
            } else if is("mean")? {
                ChannelAggregate::Mean
            } else {
                return Err(PyTypeError::new_err(
                    "aggregate: only numpy.max, numpy.min and numpy.mean are supported",
                ));
            }
        }
    };
    Ok(SilenceConfig::new()
        .with_top_db(top_db)
        .with_reference(reference)
        .with_frame_length(positive("frame_length", frame_length)?)
        .with_hop_length(positive("hop_length", hop_length)?)
        .with_aggregate(aggregate))
}

/// Parses `intervals` (`(n, 2)` non-negative integers).
fn parse_intervals(obj: &Bound<'_, PyAny>) -> PyResult<Vec<(usize, usize)>> {
    let v = i64_vec(obj, "intervals")?;
    let n = crate::args::f64_array(obj, "intervals")?;
    let shape = n.shape().to_vec();
    if shape.len() != 2 || shape[1] != 2 {
        return Err(value_err(
            "intervals",
            format!("expected shape (n, 2), got {shape:?}"),
        ));
    }
    v.chunks_exact(2)
        .map(|p| {
            Ok((
                non_negative("intervals", p[0])?,
                non_negative("intervals", p[1])?,
            ))
        })
        .collect()
}

/// [`HarmonicPercussiveConfig`] from librosa's `effects.hpss` keywords.
#[allow(clippy::too_many_arguments)]
fn hp_config(
    py: Python<'_>,
    kernel_size: Option<&Bound<'_, PyAny>>,
    power: f64,
    mask: bool,
    margin: Option<&Bound<'_, PyAny>>,
    n_fft: Int,
    hop_length: Option<Int>,
    win_length: Option<Int>,
    window: WindowArg,
    center: bool,
    pad_mode: &str,
) -> PyResult<HarmonicPercussiveConfig> {
    use pyo3::IntoPyObjectExt;
    if mask {
        return Err(value_err(
            "mask",
            "mask=True is not supported for audio output (librosa would invert the mask itself); \
             use audio_samples.effects.hpss(S, mask=True) on a spectrogram",
        ));
    }
    let ks = match kernel_size.filter(|k| !k.is_none()) {
        Some(k) => k.clone(),
        None => 31i64.into_bound_py_any(py)?,
    };
    let mg = match margin.filter(|m| !m.is_none()) {
        Some(m) => m.clone(),
        None => 1.0f64.into_bound_py_any(py)?,
    };
    Ok(HarmonicPercussiveConfig::new()
        .with_stft(stft_config(
            n_fft, hop_length, win_length, window, center, pad_mode,
        )?)
        .with_hpss(hpss_mask_config(&ks, power, &mg)?))
}

fn bounds_array(py: Python<'_>, b: (usize, usize)) -> Bound<'_, PyArray1<i64>> {
    vec![b.0 as i64, b.1 as i64].into_pyarray(py)
}

#[pymethods]
impl PyAudioSamples {
    /// Changes the speed without changing the pitch, equivalent to
    /// ``librosa.effects.time_stretch(y, rate=rate)``.
    ///
    /// STFT, ``phase_vocoder`` by ``rate``, inverse STFT to exactly
    /// ``round(n / rate)`` samples.
    ///
    /// Args:
    ///     rate (float): Speed-up factor (2.0 = twice as fast, half the length).
    ///     n_fft, hop_length, win_length, window, center, pad_mode: STFT
    ///         parameters (librosa's ``**kwargs``), as ``stft_matrix``.
    ///
    /// Returns:
    ///     AudioSamples: the stretched audio (input sample type and rate).
    ///
    /// Raises:
    ///     ValueError: If ``rate`` is not positive, the result would be empty,
    ///         or STFT parameters are invalid.
    #[pyo3(
        signature = (rate, *, n_fft = Int(2048), hop_length = None, win_length = None, window = WindowArg(Window::Hann), center = true, pad_mode = "constant"),
        text_signature = "($self, rate, *, n_fft=2048, hop_length=None, win_length=None, window='hann', center=True, pad_mode='constant')"
    )]
    #[allow(clippy::too_many_arguments)]
    fn time_stretch(
        &self,
        py: Python<'_>,
        rate: f64,
        n_fft: Int,
        hop_length: Option<Int>,
        win_length: Option<Int>,
        window: WindowArg,
        center: bool,
        pad_mode: &str,
    ) -> PyResult<Self> {
        let cfg = stft_config(n_fft, hop_length, win_length, window, center, pad_mode)?;
        dispatch_with_view_detached!(self, py, |audio| audio
            .time_stretch(rate, &cfg)
            .map(Self::from_audio_samples))
        .map_err(effects_err)
    }

    /// In-place ``time_stretch`` (the length changes); arguments as ``time_stretch``.
    #[pyo3(
        signature = (rate, *, n_fft = Int(2048), hop_length = None, win_length = None, window = WindowArg(Window::Hann), center = true, pad_mode = "constant"),
        text_signature = "($self, rate, *, n_fft=2048, hop_length=None, win_length=None, window='hann', center=True, pad_mode='constant')"
    )]
    #[allow(clippy::too_many_arguments)]
    fn time_stretch_in_place(
        &mut self,
        py: Python<'_>,
        rate: f64,
        n_fft: Int,
        hop_length: Option<Int>,
        win_length: Option<Int>,
        window: WindowArg,
        center: bool,
        pad_mode: &str,
    ) -> PyResult<()> {
        *self = self.time_stretch(
            py, rate, n_fft, hop_length, win_length, window, center, pad_mode,
        )?;
        Ok(())
    }

    /// Shifts the pitch by ``n_steps`` without changing the duration, equivalent
    /// to ``librosa.effects.pitch_shift(y, sr=self.sample_rate, n_steps=...)``.
    ///
    /// Time-stretch by ``rate = 2 ** (-n_steps / bins_per_octave)``, resample by
    /// ``rate``, then cut/pad to the original length.
    ///
    /// Args:
    ///     n_steps (float): Shift in steps (semitones by default); fractional
    ///         and negative values allowed.
    ///     bins_per_octave (int): Steps per octave. Default 12.
    ///     res_type (str | ResamplingQuality): Resampler. librosa uses soxr; this
    ///         library's windowed-sinc resampler is used instead, with
    ///         ``"soxr_hq"`` (default) mapped to ``ResamplingQuality.high``,
    ///         ``"soxr_mq"`` to ``medium`` and ``"soxr_lq"`` to ``fast``. The
    ///         result agrees with librosa to about 68 dB SNR on band-limited
    ///         audio (the time-stretch stage is exact).
    ///     scale (bool): Must be False.
    ///     n_fft, hop_length, win_length, window, center, pad_mode: STFT
    ///         parameters of the time stretch.
    ///
    /// Returns:
    ///     AudioSamples: the shifted audio, same length, type and rate.
    ///
    /// Raises:
    ///     ValueError: For a non-finite ``n_steps``, ``scale=True`` or invalid
    ///         parameters.
    #[pyo3(
        signature = (n_steps, *, bins_per_octave = Int(12), res_type = None, scale = false, n_fft = Int(2048), hop_length = None, win_length = None, window = WindowArg(Window::Hann), center = true, pad_mode = "constant"),
        text_signature = "($self, n_steps, *, bins_per_octave=12, res_type='soxr_hq', scale=False, n_fft=2048, hop_length=None, win_length=None, window='hann', center=True, pad_mode='constant')"
    )]
    #[allow(clippy::too_many_arguments)]
    fn pitch_shift(
        &self,
        py: Python<'_>,
        n_steps: f64,
        bins_per_octave: Int,
        res_type: Option<&Bound<'_, PyAny>>,
        scale: bool,
        n_fft: Int,
        hop_length: Option<Int>,
        win_length: Option<Int>,
        window: WindowArg,
        center: bool,
        pad_mode: &str,
    ) -> PyResult<Self> {
        if scale {
            return Err(value_err("scale", "scale=True is not supported"));
        }
        let quality = match res_type.filter(|r| !r.is_none()) {
            Some(r) => parse_res_type(r)?,
            None => ResamplingQuality::High,
        };
        let cfg = PitchShiftConfig::new()
            .with_bins_per_octave(positive("bins_per_octave", bins_per_octave)?)
            .with_quality(quality)
            .with_stft(stft_config(
                n_fft, hop_length, win_length, window, center, pad_mode,
            )?);
        dispatch_with_view_detached!(self, py, |audio| audio
            .pitch_shift(n_steps, &cfg)
            .map(Self::from_audio_samples))
        .map_err(effects_err)
    }

    /// In-place ``pitch_shift``; arguments as ``pitch_shift``.
    #[pyo3(
        signature = (n_steps, *, bins_per_octave = Int(12), res_type = None, scale = false, n_fft = Int(2048), hop_length = None, win_length = None, window = WindowArg(Window::Hann), center = true, pad_mode = "constant"),
        text_signature = "($self, n_steps, *, bins_per_octave=12, res_type='soxr_hq', scale=False, n_fft=2048, hop_length=None, win_length=None, window='hann', center=True, pad_mode='constant')"
    )]
    #[allow(clippy::too_many_arguments)]
    fn pitch_shift_in_place(
        &mut self,
        py: Python<'_>,
        n_steps: f64,
        bins_per_octave: Int,
        res_type: Option<&Bound<'_, PyAny>>,
        scale: bool,
        n_fft: Int,
        hop_length: Option<Int>,
        win_length: Option<Int>,
        window: WindowArg,
        center: bool,
        pad_mode: &str,
    ) -> PyResult<()> {
        *self = self.pitch_shift(
            py,
            n_steps,
            bins_per_octave,
            res_type,
            scale,
            n_fft,
            hop_length,
            win_length,
            window,
            center,
            pad_mode,
        )?;
        Ok(())
    }

    /// Pre-emphasis ``y[n] = x[n] - coef * x[n - 1]`` of every channel,
    /// equivalent to ``librosa.effects.preemphasis(y, coef=coef)`` (initial
    /// state ``2 x[0] - x[1]``). Use ``audio_samples.effects.preemphasis`` for
    /// ``zi`` / ``return_zf``.
    ///
    /// Args:
    ///     coef (float): Coefficient. Default 0.97.
    ///
    /// Returns:
    ///     AudioSamples: the filtered audio.
    ///
    /// Raises:
    ///     ValueError: For a non-finite ``coef`` or fewer than two samples.
    #[pyo3(signature = (*, coef = 0.97), text_signature = "($self, *, coef=0.97)")]
    fn preemphasis(&self, py: Python<'_>, coef: f64) -> PyResult<Self> {
        dispatch_with_view!(self, py, |audio| audio
            .preemphasis(coef)
            .map(Self::from_audio_samples))
        .map_err(effects_err)
    }

    /// In-place ``preemphasis``.
    #[pyo3(signature = (*, coef = 0.97), text_signature = "($self, *, coef=0.97)")]
    fn preemphasis_in_place(&mut self, py: Python<'_>, coef: f64) -> PyResult<()> {
        *self = self.preemphasis(py, coef)?;
        Ok(())
    }

    /// De-emphasis ``y[n] = x[n] + coef * y[n - 1]`` of every channel (inverse
    /// of ``preemphasis``), equivalent to ``librosa.effects.deemphasis(y, coef=coef)``.
    ///
    /// Args:
    ///     coef (float): Coefficient (not 3). Default 0.97.
    ///
    /// Returns:
    ///     AudioSamples: the filtered audio.
    ///
    /// Raises:
    ///     ValueError: For a non-finite ``coef``, ``coef == 3`` or fewer than two
    ///         samples.
    #[pyo3(signature = (*, coef = 0.97), text_signature = "($self, *, coef=0.97)")]
    fn deemphasis(&self, py: Python<'_>, coef: f64) -> PyResult<Self> {
        dispatch_with_view!(self, py, |audio| audio
            .deemphasis(coef)
            .map(Self::from_audio_samples))
        .map_err(effects_err)
    }

    /// In-place ``deemphasis``.
    #[pyo3(signature = (*, coef = 0.97), text_signature = "($self, *, coef=0.97)")]
    fn deemphasis_in_place(&mut self, py: Python<'_>, coef: f64) -> PyResult<()> {
        *self = self.deemphasis(py, coef)?;
        Ok(())
    }

    /// Trims leading and trailing silence, equivalent to
    /// ``librosa.effects.trim(y, ...)``.
    ///
    /// A frame is non-silent when its RMS level (centred frames, dB relative to
    /// ``ref``) is above ``-top_db``; the kept region runs from the first
    /// non-silent frame to the end of the last. Not to be confused with
    /// ``trim_silence`` (sample-wise absolute dBFS threshold).
    ///
    /// Args:
    ///     top_db (float): Threshold below the reference in dB. Default 60.
    ///     ref (float | "max" | numpy.max): Reference RMS amplitude; the default
    ///         ``numpy.max`` uses the loudest frame.
    ///     frame_length (int): RMS frame length. Default 2048.
    ///     hop_length (int): Hop between frames. Default 512.
    ///     aggregate: Channel combination: ``numpy.max`` (default), ``numpy.min``
    ///         or ``numpy.mean``.
    ///
    /// Returns:
    ///     tuple: ``(trimmed, index)``: the trimmed ``AudioSamples`` and an int64
    ///     array ``[start, end]`` of the kept sample range.
    ///
    /// Raises:
    ///     ValueError: For invalid parameters, or if everything is silent
    ///         (librosa returns an empty array and ``[0, 0]``; an
    ///         ``AudioSamples`` cannot be empty, use ``split_db`` to test).
    #[pyo3(
        signature = (*, top_db = 60.0, r#ref = DbRefArg(DbReference::Max), frame_length = Int(2048), hop_length = Int(512), aggregate = None),
        text_signature = "($self, *, top_db=60, ref='max', frame_length=2048, hop_length=512, aggregate=None)"
    )]
    fn trim_db<'py>(
        &self,
        py: Python<'py>,
        top_db: f64,
        r#ref: DbRefArg,
        frame_length: Int,
        hop_length: Int,
        aggregate: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<(Self, Bound<'py, PyArray1<i64>>)> {
        let cfg = silence_config(py, top_db, r#ref, frame_length, hop_length, aggregate)?;
        let (audio, bounds) = dispatch_with_view!(self, py, |audio| audio
            .trim_db(&cfg)
            .map(|(a, b)| (Self::from_audio_samples(a), b)))
        .map_err(|e| {
            if matches!(e, audio_samples::AudioSampleError::EmptyData { .. }) {
                value_err("trim_db", "every frame is silent; nothing to keep")
            } else {
                audio_err_to_py(e)
            }
        })?;
        Ok((audio, bounds_array(py, bounds)))
    }

    /// In-place ``trim_db``; returns the kept ``[start, end]`` (int64 array).
    #[pyo3(
        signature = (*, top_db = 60.0, r#ref = DbRefArg(DbReference::Max), frame_length = Int(2048), hop_length = Int(512), aggregate = None),
        text_signature = "($self, *, top_db=60, ref='max', frame_length=2048, hop_length=512, aggregate=None)"
    )]
    fn trim_db_in_place<'py>(
        &mut self,
        py: Python<'py>,
        top_db: f64,
        r#ref: DbRefArg,
        frame_length: Int,
        hop_length: Int,
        aggregate: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyArray1<i64>>> {
        let (audio, bounds) =
            self.trim_db(py, top_db, r#ref, frame_length, hop_length, aggregate)?;
        *self = audio;
        Ok(bounds)
    }

    /// Non-silent intervals, equivalent to ``librosa.effects.split(y, ...)``.
    ///
    /// Args:
    ///     top_db, ref, frame_length, hop_length, aggregate: As ``trim_db``.
    ///
    /// Returns:
    ///     numpy.ndarray[int64]: shape ``(n_intervals, 2)`` of ``[start, end)``
    ///     sample ranges (``(0, 2)`` if everything is silent).
    #[pyo3(
        signature = (*, top_db = 60.0, r#ref = DbRefArg(DbReference::Max), frame_length = Int(2048), hop_length = Int(512), aggregate = None),
        text_signature = "($self, *, top_db=60, ref='max', frame_length=2048, hop_length=512, aggregate=None)"
    )]
    fn split_db<'py>(
        &self,
        py: Python<'py>,
        top_db: f64,
        r#ref: DbRefArg,
        frame_length: Int,
        hop_length: Int,
        aggregate: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyArray2<i64>>> {
        let cfg = silence_config(py, top_db, r#ref, frame_length, hop_length, aggregate)?;
        let iv =
            dispatch_with_view!(self, py, |audio| audio.split_db(&cfg)).map_err(audio_err_to_py)?;
        let flat: Vec<i64> = iv.iter().flat_map(|&(a, b)| [a as i64, b as i64]).collect();
        Ok(Array2::from_shape_vec((iv.len(), 2), flat)
            .expect("two values per interval")
            .into_pyarray(py))
    }

    /// Concatenates sample intervals, equivalent to
    /// ``librosa.effects.remix(y, intervals, align_zeros=...)``.
    ///
    /// Args:
    ///     intervals (array_like): ``(n, 2)`` ``[start, end)`` sample ranges, in
    ///         output order; clipped to the signal, empty ranges skipped.
    ///     align_zeros (bool): Snap boundaries to the nearest zero crossing of
    ///         the (mono downmix of the) signal. Default True.
    ///
    /// Returns:
    ///     AudioSamples: the remixed audio.
    ///
    /// Raises:
    ///     ValueError: For malformed or all-empty intervals.
    #[pyo3(
        signature = (intervals, *, align_zeros = true),
        text_signature = "($self, intervals, *, align_zeros=True)"
    )]
    fn remix(
        &self,
        py: Python<'_>,
        intervals: &Bound<'_, PyAny>,
        align_zeros: bool,
    ) -> PyResult<Self> {
        let iv = parse_intervals(intervals)?;
        dispatch_with_view!(self, py, |audio| audio
            .remix(&iv, align_zeros)
            .map(Self::from_audio_samples))
        .map_err(|e| {
            if matches!(e, audio_samples::AudioSampleError::EmptyData { .. }) {
                value_err("intervals", "every interval is empty")
            } else {
                audio_err_to_py(e)
            }
        })
    }

    /// In-place ``remix``.
    #[pyo3(
        signature = (intervals, *, align_zeros = true),
        text_signature = "($self, intervals, *, align_zeros=True)"
    )]
    fn remix_in_place(
        &mut self,
        py: Python<'_>,
        intervals: &Bound<'_, PyAny>,
        align_zeros: bool,
    ) -> PyResult<()> {
        *self = self.remix(py, intervals, align_zeros)?;
        Ok(())
    }

    /// Harmonic component, equivalent to ``librosa.effects.harmonic(y, ...)``.
    ///
    /// STFT, ``audio_samples.effects.hpss`` (median filtering + soft masks),
    /// inverse STFT to the input length. Unlike the older ``hpss`` method this
    /// follows librosa exactly.
    ///
    /// Args:
    ///     kernel_size (int | tuple[int, int]): Median filter lengths. Default 31.
    ///     power (float): Mask exponent. Default 2.
    ///     mask (bool): Must be False.
    ///     margin (float | tuple[float, float]): Margins (>= 1). Default 1.
    ///     n_fft, hop_length, win_length, window, center, pad_mode: STFT
    ///         parameters. librosa ignores ``window`` here (always Hann); this
    ///         method uses it.
    ///
    /// Returns:
    ///     AudioSamples: the harmonic component, same length.
    #[pyo3(
        signature = (*, kernel_size = None, power = 2.0, mask = false, margin = None, n_fft = Int(2048), hop_length = None, win_length = None, window = WindowArg(Window::Hann), center = true, pad_mode = "constant"),
        text_signature = "($self, *, kernel_size=31, power=2.0, mask=False, margin=1.0, n_fft=2048, hop_length=None, win_length=None, window='hann', center=True, pad_mode='constant')"
    )]
    #[allow(clippy::too_many_arguments)]
    fn harmonic(
        &self,
        py: Python<'_>,
        kernel_size: Option<&Bound<'_, PyAny>>,
        power: f64,
        mask: bool,
        margin: Option<&Bound<'_, PyAny>>,
        n_fft: Int,
        hop_length: Option<Int>,
        win_length: Option<Int>,
        window: WindowArg,
        center: bool,
        pad_mode: &str,
    ) -> PyResult<Self> {
        let cfg = hp_config(
            py,
            kernel_size,
            power,
            mask,
            margin,
            n_fft,
            hop_length,
            win_length,
            window,
            center,
            pad_mode,
        )?;
        dispatch_with_view_detached!(self, py, |audio| audio
            .harmonic(&cfg)
            .map(Self::from_audio_samples))
        .map_err(effects_err)
    }

    /// In-place ``harmonic``; arguments as ``harmonic``.
    #[pyo3(
        signature = (*, kernel_size = None, power = 2.0, mask = false, margin = None, n_fft = Int(2048), hop_length = None, win_length = None, window = WindowArg(Window::Hann), center = true, pad_mode = "constant"),
        text_signature = "($self, *, kernel_size=31, power=2.0, mask=False, margin=1.0, n_fft=2048, hop_length=None, win_length=None, window='hann', center=True, pad_mode='constant')"
    )]
    #[allow(clippy::too_many_arguments)]
    fn harmonic_in_place(
        &mut self,
        py: Python<'_>,
        kernel_size: Option<&Bound<'_, PyAny>>,
        power: f64,
        mask: bool,
        margin: Option<&Bound<'_, PyAny>>,
        n_fft: Int,
        hop_length: Option<Int>,
        win_length: Option<Int>,
        window: WindowArg,
        center: bool,
        pad_mode: &str,
    ) -> PyResult<()> {
        *self = self.harmonic(
            py,
            kernel_size,
            power,
            mask,
            margin,
            n_fft,
            hop_length,
            win_length,
            window,
            center,
            pad_mode,
        )?;
        Ok(())
    }

    /// Percussive component, equivalent to ``librosa.effects.percussive(y, ...)``.
    ///
    /// Args:
    ///     As ``harmonic``.
    ///
    /// Returns:
    ///     AudioSamples: the percussive component, same length.
    #[pyo3(
        signature = (*, kernel_size = None, power = 2.0, mask = false, margin = None, n_fft = Int(2048), hop_length = None, win_length = None, window = WindowArg(Window::Hann), center = true, pad_mode = "constant"),
        text_signature = "($self, *, kernel_size=31, power=2.0, mask=False, margin=1.0, n_fft=2048, hop_length=None, win_length=None, window='hann', center=True, pad_mode='constant')"
    )]
    #[allow(clippy::too_many_arguments)]
    fn percussive(
        &self,
        py: Python<'_>,
        kernel_size: Option<&Bound<'_, PyAny>>,
        power: f64,
        mask: bool,
        margin: Option<&Bound<'_, PyAny>>,
        n_fft: Int,
        hop_length: Option<Int>,
        win_length: Option<Int>,
        window: WindowArg,
        center: bool,
        pad_mode: &str,
    ) -> PyResult<Self> {
        let cfg = hp_config(
            py,
            kernel_size,
            power,
            mask,
            margin,
            n_fft,
            hop_length,
            win_length,
            window,
            center,
            pad_mode,
        )?;
        dispatch_with_view_detached!(self, py, |audio| audio
            .percussive(&cfg)
            .map(Self::from_audio_samples))
        .map_err(effects_err)
    }

    /// In-place ``percussive``; arguments as ``harmonic``.
    #[pyo3(
        signature = (*, kernel_size = None, power = 2.0, mask = false, margin = None, n_fft = Int(2048), hop_length = None, win_length = None, window = WindowArg(Window::Hann), center = true, pad_mode = "constant"),
        text_signature = "($self, *, kernel_size=31, power=2.0, mask=False, margin=1.0, n_fft=2048, hop_length=None, win_length=None, window='hann', center=True, pad_mode='constant')"
    )]
    #[allow(clippy::too_many_arguments)]
    fn percussive_in_place(
        &mut self,
        py: Python<'_>,
        kernel_size: Option<&Bound<'_, PyAny>>,
        power: f64,
        mask: bool,
        margin: Option<&Bound<'_, PyAny>>,
        n_fft: Int,
        hop_length: Option<Int>,
        win_length: Option<Int>,
        window: WindowArg,
        center: bool,
        pad_mode: &str,
    ) -> PyResult<()> {
        *self = self.percussive(
            py,
            kernel_size,
            power,
            mask,
            margin,
            n_fft,
            hop_length,
            win_length,
            window,
            center,
            pad_mode,
        )?;
        Ok(())
    }

    /// Harmonic and percussive components, equivalent to
    /// ``librosa.effects.hpss(y, ...)`` (named ``harmonic_percussive`` because
    /// ``hpss`` is the older, non-librosa method).
    ///
    /// Args:
    ///     As ``harmonic``.
    ///
    /// Returns:
    ///     tuple[AudioSamples, AudioSamples]: ``(harmonic, percussive)``; with
    ///     margins of 1 they sum to the input up to round-off.
    #[pyo3(
        signature = (*, kernel_size = None, power = 2.0, mask = false, margin = None, n_fft = Int(2048), hop_length = None, win_length = None, window = WindowArg(Window::Hann), center = true, pad_mode = "constant"),
        text_signature = "($self, *, kernel_size=31, power=2.0, mask=False, margin=1.0, n_fft=2048, hop_length=None, win_length=None, window='hann', center=True, pad_mode='constant')"
    )]
    #[allow(clippy::too_many_arguments)]
    fn harmonic_percussive(
        &self,
        py: Python<'_>,
        kernel_size: Option<&Bound<'_, PyAny>>,
        power: f64,
        mask: bool,
        margin: Option<&Bound<'_, PyAny>>,
        n_fft: Int,
        hop_length: Option<Int>,
        win_length: Option<Int>,
        window: WindowArg,
        center: bool,
        pad_mode: &str,
    ) -> PyResult<(Self, Self)> {
        let cfg = hp_config(
            py,
            kernel_size,
            power,
            mask,
            margin,
            n_fft,
            hop_length,
            win_length,
            window,
            center,
            pad_mode,
        )?;
        dispatch_with_view_detached!(self, py, |audio| audio
            .harmonic_percussive(&cfg)
            .map(|(h, p)| (Self::from_audio_samples(h), Self::from_audio_samples(p))))
        .map_err(effects_err)
    }

    /// LPC coefficients of every channel (Burg's method), equivalent to
    /// ``librosa.lpc(y, order=order)``.
    ///
    /// Args:
    ///     order (int): Prediction order; each channel needs more than
    ///         ``order`` samples.
    ///
    /// Returns:
    ///     numpy.ndarray[float64]: shape ``(channels, order + 1)``, rows
    ///     ``[1, a_1, ..., a_order]``. librosa returns ``(order + 1,)`` for 1-D
    ///     input; index ``[0]``.
    ///
    /// Raises:
    ///     ValueError: For a too-short or non-finite channel.
    #[pyo3(signature = (*, order), text_signature = "($self, *, order)")]
    fn lpc<'py>(&self, py: Python<'py>, order: Int) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let order = positive("order", order)?;
        let a = dispatch_with_view!(self, py, |audio| AudioEffects::lpc(&audio, order))
            .map_err(audio_err_to_py)?;
        Ok(a.into_pyarray(py))
    }

    /// Reconstructs audio from a magnitude spectrogram with fast Griffin-Lim,
    /// equivalent to ``librosa.griffinlim(S, ...)`` returning ``AudioSamples``.
    ///
    /// This is a static method: ``AudioSamples.griffinlim_audio(S, sample_rate=sr)``.
    /// See ``audio_samples.effects.griffinlim`` for the arguments.
    ///
    /// Args:
    ///     S (array_like): ``(1 + n_fft // 2, n_frames)`` or
    ///         ``(channels, 1 + n_fft // 2, n_frames)`` non-negative magnitude.
    ///     sample_rate (int): Sample rate of the result (keyword-only).
    ///     **kwargs: ``n_iter``, ``hop_length``, ``win_length``, ``n_fft``,
    ///         ``window``, ``center``, ``length``, ``pad_mode``, ``momentum``,
    ///         ``init``, ``rng`` as ``audio_samples.effects.griffinlim``.
    ///
    /// Returns:
    ///     AudioSamples: float64 audio (mono for one channel).
    #[staticmethod]
    #[pyo3(
        signature = (S, *, sample_rate, n_iter = Int(32), hop_length = None, win_length = None, n_fft = None, window = WindowArg(Window::Hann), center = true, length = None, pad_mode = "constant", momentum = 0.99, init = Some("random"), rng = None),
        text_signature = "(S, *, sample_rate, n_iter=32, hop_length=None, win_length=None, n_fft=None, window='hann', center=True, length=None, pad_mode='constant', momentum=0.99, init='random', rng=None)"
    )]
    #[allow(non_snake_case, clippy::too_many_arguments)]
    fn griffinlim_audio(
        S: &Bound<'_, PyAny>,
        sample_rate: Int,
        n_iter: Int,
        hop_length: Option<Int>,
        win_length: Option<Int>,
        n_fft: Option<Int>,
        window: WindowArg,
        center: bool,
        length: Option<Int>,
        pad_mode: &str,
        momentum: f64,
        init: Option<&str>,
        rng: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let (s, _) = planes(S, "S")?;
        let cfg = griffinlim_config(
            s.dim().1,
            n_iter,
            hop_length,
            win_length,
            n_fft,
            window,
            center,
            length,
            pad_mode,
            momentum,
            init,
            rng,
        )?;
        let sr = nzu32_or_err(crate::args::sample_rate_u32(sample_rate)?)?;
        let y = <AudioSamples<'static, f64> as AudioEffects>::griffinlim_audio(&s, &cfg, sr)
            .map_err(audio_err_to_py)?;
        Ok(Self::from_audio_samples(y))
    }
}
