//! Python bindings for audio mathematical utilities and canonical unit conversions.
//!
//! This module exposes the audio_samples::utils::audio_math functions to Python,
//! providing frequency conversions, amplitude/power conversions, time/frame conversions,
//! musical theory functions, and spectral utilities.

use crate::args::Int;
use audio_samples::utils::audio_math;
use numpy::{IntoPyArray, PyArray1, PyUntypedArrayMethods};
use pyo3::prelude::*;

use crate::args::{is_scalar, map_f64, map_f64_to_i64, map_i64, non_negative, positive, value_err};
use crate::audio_err_to_py;

/// `htk` flag to the Rust mel scale.
const fn mel_scale_of(htk: bool) -> audio_math::MelScale {
    crate::args::mel_scale(htk)
}

/// Optional non-negative `n_fft` offset.
fn opt_non_negative(name: &str, v: Option<Int>) -> PyResult<Option<usize>> {
    v.map(|v| non_negative(name, v)).transpose()
}

/// Converts frequencies in Hertz to mels (scalar or array, librosa-style).
///
/// **Default scale is HTK** (``2595 * log10(1 + f / 700)``), unchanged from
/// earlier releases. librosa's ``hz_to_mel`` defaults to the *Slaney* scale
/// (``htk=False``); pass ``htk=False`` to match it.
///
/// Args:
///     freq_hz (float | array_like): Frequency or frequencies in Hertz.
///     htk (bool): Use the HTK scale (default True) or the Slaney scale (False,
///         librosa's default).
///
/// Returns:
///     float | numpy.ndarray[float64]: Mels, same shape as the input.
///
/// Example:
///     >>> from audio_samples.utils import hz_to_mel
///     >>> hz_to_mel(1000.0)              # HTK: ~999.99
///     >>> hz_to_mel(1000.0, htk=False)   # Slaney (librosa default): 15.0
#[pyfunction]
#[pyo3(signature = (freq_hz, *, htk = true), text_signature = "(freq_hz, *, htk=True)")]
pub fn hz_to_mel<'py>(freq_hz: &Bound<'py, PyAny>, htk: bool) -> PyResult<Bound<'py, PyAny>> {
    let scale = mel_scale_of(htk);
    map_f64(freq_hz, "freq_hz", |f| {
        audio_math::hz_to_mel_with_scale(f, scale)
    })
}

/// Converts mels to frequencies in Hertz (scalar or array, librosa-style).
///
/// **Default scale is HTK**, unchanged from earlier releases; librosa's
/// ``mel_to_hz`` defaults to Slaney (``htk=False``).
///
/// Args:
///     mel (float | array_like): Mel value(s).
///     htk (bool): HTK scale (default True) or Slaney (False, librosa's default).
///
/// Returns:
///     float | numpy.ndarray[float64]: Frequencies in Hertz, same shape as the input.
///
/// Example:
///     >>> from audio_samples.utils import mel_to_hz
///     >>> mel_to_hz(15.0, htk=False)  # 1000.0
#[pyfunction]
#[pyo3(signature = (mel, *, htk = true), text_signature = "(mel, *, htk=True)")]
pub fn mel_to_hz<'py>(mel: &Bound<'py, PyAny>, htk: bool) -> PyResult<Bound<'py, PyAny>> {
    let scale = mel_scale_of(htk);
    map_f64(mel, "mel", |m| audio_math::mel_to_hz_with_scale(m, scale))
}

#[pyfunction]
#[pyo3(signature = (n_mels, fmin, fmax))]
/// Generates a sequence of frequencies whose spacing is uniform in the mel domain.
///
/// Args:
///     n_mels: The number of frequency points to generate.
///     fmin: The lower bound of the frequency range in Hertz.
///     fmax: The upper bound of the frequency range in Hertz.
///
/// Returns:
///     A numpy array of frequencies in Hertz, spaced uniformly in mel scale.
///
/// Example:
///     >>> from audio_samples.utils import mel_scale
///     >>> freqs = mel_scale(10, 100.0, 8000.0)
#[inline]
pub fn mel_scale<'py>(
    py: Python<'py>,
    n_mels: usize,
    fmin: f64,
    fmax: f64,
) -> Bound<'py, PyArray1<f64>> {
    let result = audio_math::mel_scale(n_mels, fmin, fmax);
    result.into_pyarray(py)
}

/// Converts frequencies in Hertz to (fractional) MIDI note numbers, like
/// ``librosa.hz_to_midi``: ``12 * log2(f / 440) + 69``.
///
/// Args:
///     freq_hz (float | array_like): Frequency or frequencies in Hertz.
///
/// Returns:
///     float | numpy.ndarray[float64]: MIDI numbers, same shape as the input.
///
/// Example:
///     >>> from audio_samples.utils import hz_to_midi
///     >>> hz_to_midi(440.0)  # 69.0
#[pyfunction]
#[pyo3(signature = (freq_hz), text_signature = "(freq_hz)")]
pub fn hz_to_midi<'py>(freq_hz: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    map_f64(freq_hz, "freq_hz", audio_math::hz_to_midi)
}

/// Converts MIDI note numbers (fractional allowed) to Hertz, like
/// ``librosa.midi_to_hz``: ``440 * 2 ** ((m - 69) / 12)``.
///
/// Args:
///     midi_note (float | array_like): MIDI number(s).
///
/// Returns:
///     float | numpy.ndarray[float64]: Frequencies in Hertz, same shape as the input.
///
/// Example:
///     >>> from audio_samples.utils import midi_to_hz
///     >>> midi_to_hz(69.0)  # 440.0
#[pyfunction]
#[pyo3(signature = (midi_note), text_signature = "(midi_note)")]
pub fn midi_to_hz<'py>(midi_note: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    map_f64(midi_note, "midi_note", audio_math::midi_to_hz)
}

// =============================================================================
// AMPLITUDE CONVERSIONS
// =============================================================================

#[pyfunction]
#[pyo3(signature = (amplitude))]
/// Converts a linear amplitude ratio into a logarithmic decibel representation.
///
/// Non-positive amplitudes are mapped to -80 dB.
///
/// Args:
///     amplitude: The input amplitude ratio.
///
/// Returns:
///     The amplitude expressed in decibels.
///
/// Example:
///     >>> from audio_samples.utils import amplitude_to_db
///     >>> db = amplitude_to_db(0.5)  # Returns ~-6.02 dB
#[inline]
pub fn amplitude_to_db(amplitude: f64) -> f64 {
    audio_math::amplitude_to_db(amplitude)
}

#[pyfunction]
#[pyo3(signature = (db))]
/// Converts a decibel value into a linear amplitude ratio.
///
/// Args:
///     db: A value expressed in decibels.
///
/// Returns:
///     The corresponding linear amplitude ratio.
///
/// Example:
///     >>> from audio_samples.utils import db_to_amplitude
///     >>> amp = db_to_amplitude(-6.0)  # Returns ~0.501
#[inline]
pub fn db_to_amplitude(db: f64) -> f64 {
    audio_math::db_to_amplitude(db)
}

#[pyfunction]
#[pyo3(signature = (power))]
/// Converts a linear power ratio into a logarithmic decibel representation.
///
/// Non-positive power values are mapped to -80 dB.
///
/// Args:
///     power: The input power ratio.
///
/// Returns:
///     The power expressed in decibels.
///
/// Example:
///     >>> from audio_samples.utils import power_to_db
///     >>> db = power_to_db(0.5)  # Returns ~-3.01 dB
#[inline]
pub fn power_to_db(power: f64) -> f64 {
    audio_math::power_to_db(power)
}

#[pyfunction]
#[pyo3(signature = (db))]
/// Converts a decibel value into a linear power ratio.
///
/// Args:
///     db: A value expressed in decibels.
///
/// Returns:
///     The corresponding linear power ratio.
///
/// Example:
///     >>> from audio_samples.utils import db_to_power
///     >>> power = db_to_power(-3.0)  # Returns ~0.501
#[inline]
pub fn db_to_power(db: f64) -> f64 {
    audio_math::db_to_power(db)
}

// =============================================================================
// TIME/FRAME CONVERSIONS
// =============================================================================

/// Converts frame indices to time in seconds, like ``librosa.frames_to_time``.
///
/// ``frames * hop_size / sample_rate``, plus ``(n_fft // 2) / sample_rate`` when
/// ``n_fft`` is given (the frame centre of an uncentred STFT). Negative frames
/// are allowed.
///
/// Args:
///     frames (int | array_like[int]): Frame index or indices.
///     sample_rate (float): Sampling rate in Hertz.
///     hop_size (int): Samples between frames (librosa's ``hop_length``).
///     n_fft (Optional[int]): Optional frame-length offset. Default None.
///
/// Returns:
///     float | numpy.ndarray[float64]: Times in seconds, same shape as the input.
///
/// Example:
///     >>> from audio_samples.utils import frames_to_time
///     >>> frames_to_time(100, 44100.0, 512)
///     >>> frames_to_time([0, 1, 2], 22050.0, 512, n_fft=2048)
#[pyfunction]
#[pyo3(signature = (frames, sample_rate, hop_size, *, n_fft = None), text_signature = "(frames, sample_rate, hop_size, *, n_fft=None)")]
pub fn frames_to_time<'py>(
    frames: &Bound<'py, PyAny>,
    sample_rate: f64,
    hop_size: Int,
    n_fft: Option<Int>,
) -> PyResult<Bound<'py, PyAny>> {
    let hop = positive("hop_size", hop_size)?;
    let n_fft = opt_non_negative("n_fft", n_fft)?;
    map_i64(frames, "frames", |f| {
        audio_math::frames_to_time_with_offset(f, sample_rate, hop, n_fft)
    })
}

/// Converts times in seconds to the **nearest** frame index (rounding).
///
/// Kept for backward compatibility: ``round(t * sample_rate / hop_size)``, with
/// negative results clamped to 0. librosa's ``time_to_frames`` instead returns the
/// frame *containing* the time (floor division, negatives kept, optional
/// ``n_fft`` offset); use ``time_to_frames_floor`` for that.
///
/// Args:
///     time_seconds (float | array_like): Time(s) in seconds.
///     sample_rate (float): Sampling rate in Hertz.
///     hop_size (int): Samples between frames.
///
/// Returns:
///     int | numpy.ndarray[int64]: Nearest frame indices, same shape as the input.
///
/// Example:
///     >>> from audio_samples.utils import time_to_frames
///     >>> time_to_frames(1.0, 44100.0, 512)  # 86
#[pyfunction]
#[pyo3(signature = (time_seconds, sample_rate, hop_size), text_signature = "(time_seconds, sample_rate, hop_size)")]
pub fn time_to_frames<'py>(
    time_seconds: &Bound<'py, PyAny>,
    sample_rate: f64,
    hop_size: Int,
) -> PyResult<Bound<'py, PyAny>> {
    let hop = positive("hop_size", hop_size)?.get();
    map_f64_to_i64(time_seconds, "time_seconds", |t| {
        audio_math::time_to_frames(t, sample_rate, hop) as i64
    })
}

#[pyfunction]
#[pyo3(signature = (samples, sample_rate))]
/// Converts a sample index into a time offset expressed in seconds.
///
/// Args:
///     samples: The sample index to convert.
///     sample_rate: The sampling rate in Hertz.
///
/// Returns:
///     The corresponding time offset in seconds.
///
/// Example:
///     >>> from audio_samples.utils import samples_to_time
///     >>> t = samples_to_time(44100, 44100.0)  # Returns 1.0
#[inline]
pub fn samples_to_time(samples: usize, sample_rate: f64) -> f64 {
    audio_math::samples_to_time(samples, sample_rate)
}

#[pyfunction]
#[pyo3(signature = (time_seconds, sample_rate))]
/// Converts a time offset expressed in seconds into the nearest sample index.
///
/// Rounds to the nearest sample and clamps negatives to 0. librosa's
/// ``time_to_samples`` truncates instead and keeps negative values; use
/// ``time_to_samples`` for that.
///
/// Args:
///     time_seconds: The time offset in seconds.
///     sample_rate: The sampling rate in Hertz.
///
/// Returns:
///     The nearest sample index.
///
/// Example:
///     >>> from audio_samples.utils import seconds_to_samples
///     >>> samples = seconds_to_samples(1.0, 44100.0)  # Returns 44100
#[inline]
pub fn seconds_to_samples(time_seconds: f64, sample_rate: f64) -> usize {
    audio_math::seconds_to_samples(time_seconds, sample_rate)
}

#[pyfunction]
#[pyo3(signature = (ms, sample_rate))]
/// Converts time in milliseconds to sample count.
///
/// Args:
///     ms: Time in milliseconds.
///     sample_rate: The sampling rate in Hertz.
///
/// Returns:
///     The corresponding number of samples.
///
/// Example:
///     >>> from audio_samples.utils import ms_to_samples
///     >>> samples = ms_to_samples(1000.0, 44100.0)  # Returns 44100
#[inline]
pub fn ms_to_samples(ms: f64, sample_rate: f64) -> usize {
    audio_math::ms_to_samples(ms, sample_rate)
}

// =============================================================================
// MUSICAL THEORY FUNCTIONS
// =============================================================================

#[pyfunction]
#[pyo3(signature = (note_name))]
/// Parses a scientific pitch notation string and converts it into a MIDI note number.
///
/// Follows librosa's grammar: upper- or lowercase note letters, any run of the accidentals
/// `#` `♯` `b` `!` `♭` `𝄪` `𝄫` `♮`, a signed octave (`C-1` = 0) and an optional cents suffix
/// (`A4+50`, rounded half to even). The octave is required.
///
/// Args:
///     note_name: A note name in scientific pitch notation (e.g., "A4", "C#3", "Bb2").
///
/// Returns:
///     The corresponding MIDI note number (0-127).
///
/// Raises:
///     ValueError: If the note name is invalid or out of MIDI range.
///
/// Example:
///     >>> from audio_samples.utils import note_to_midi
///     >>> midi = note_to_midi("A4")  # Returns 69
///     >>> midi = note_to_midi("C#4")  # Returns 61
#[inline]
pub fn note_to_midi(note_name: &str) -> PyResult<u8> {
    audio_math::note_to_midi(note_name).map_err(audio_err_to_py)
}

/// Converts a MIDI note number into a scientific pitch notation string.
///
/// Always uses sharp notation for accidentals.
///
/// Args:
///     midi_note (int): A MIDI note number (0-127).
///
/// Returns:
///     str: A note name in scientific pitch notation (e.g., "A4", "C#3").
///
/// Raises:
///     ValueError: If the MIDI note is outside the valid range 0-127.
///
/// Example:
///     >>> from audio_samples.utils import midi_to_note
///     >>> midi_to_note(69)  # "A4"
///     >>> midi_to_note(61)  # "C#4"
#[pyfunction]
#[pyo3(signature = (midi_note), text_signature = "(midi_note)")]
pub fn midi_to_note(midi_note: Int) -> PyResult<String> {
    let m = u8::try_from(midi_note.0)
        .ok()
        .filter(|&m| m <= 127)
        .ok_or_else(|| value_err("midi_note", format!("must be in 0..=127, got {midi_note}")))?;
    audio_math::midi_to_note(m).map_err(audio_err_to_py)
}

#[pyfunction]
#[pyo3(signature = (note_name))]
/// Converts a scientific pitch notation string into a frequency expressed in Hertz.
///
/// Args:
///     note_name: A note name in scientific pitch notation (e.g., "A4", "C#3").
///
/// Returns:
///     The corresponding frequency in Hertz.
///
/// Raises:
///     ValueError: If the note name is invalid or out of MIDI range.
///
/// Example:
///     >>> from audio_samples.utils import note_to_frequency
///     >>> freq = note_to_frequency("A4")  # Returns 440.0
#[inline]
pub fn note_to_frequency(note_name: &str) -> PyResult<f64> {
    audio_math::note_to_frequency(note_name).map_err(audio_err_to_py)
}

#[pyfunction]
#[pyo3(signature = (freq_hz))]
/// Converts a frequency expressed in Hertz into the nearest musical note and cents offset.
///
/// Args:
///     freq_hz: The input frequency in Hertz. Must be positive.
///
/// Returns:
///     A tuple (note_name, cents) where note_name is the nearest pitch and cents is
///     the signed deviation from that pitch.
///
/// Raises:
///     ValueError: If the frequency is non-positive.
///
/// Example:
///     >>> from audio_samples.utils import frequency_to_note
///     >>> note, cents = frequency_to_note(440.0)  # Returns ("A4", 0.0)
///     >>> note, cents = frequency_to_note(445.0)  # Returns ("A4", ~19.56)
#[inline]
pub fn frequency_to_note(freq_hz: f64) -> PyResult<(String, f64)> {
    audio_math::frequency_to_note(freq_hz).map_err(audio_err_to_py)
}

#[pyfunction]
#[pyo3(signature = (cents))]
/// Converts a pitch deviation expressed in cents into a frequency ratio.
///
/// Args:
///     cents: A pitch deviation in cents.
///
/// Returns:
///     A multiplicative frequency ratio.
///
/// Example:
///     >>> from audio_samples.utils import cents_to_ratio
///     >>> ratio = cents_to_ratio(1200.0)  # Returns 2.0 (one octave)
///     >>> ratio = cents_to_ratio(100.0)  # Returns ~1.059 (one semitone)
#[inline]
pub fn cents_to_ratio(cents: f64) -> f64 {
    audio_math::cents_to_ratio(cents)
}

#[pyfunction]
#[pyo3(signature = (ratio))]
/// Converts a frequency ratio into a pitch deviation expressed in cents.
///
/// Args:
///     ratio: A frequency ratio. Must be positive.
///
/// Returns:
///     The corresponding pitch deviation in cents.
///
/// Example:
///     >>> from audio_samples.utils import ratio_to_cents
///     >>> cents = ratio_to_cents(2.0)  # Returns 1200.0 (one octave)
///     >>> cents = ratio_to_cents(1.5)  # Returns ~702.0 (perfect fifth)
#[inline]
pub fn ratio_to_cents(ratio: f64) -> f64 {
    audio_math::ratio_to_cents(ratio)
}

// =============================================================================
// SPECTRAL HELPER FUNCTIONS
// =============================================================================

#[pyfunction]
#[pyo3(signature = (n_fft, sample_rate))]
/// Generates the positive-frequency bin centres for a real-valued FFT.
///
/// Args:
///     n_fft: The FFT size in samples.
///     sample_rate: The sampling rate in Hertz.
///
/// Returns:
///     A numpy array of frequency bin centres in Hertz, from DC to Nyquist.
///     Length is n_fft/2 + 1.
///
/// Example:
///     >>> from audio_samples.utils import fft_frequencies
///     >>> freqs = fft_frequencies(1024, 44100.0)
///     >>> freqs.shape  # (513,)
#[inline]
pub fn fft_frequencies<'py>(
    py: Python<'py>,
    n_fft: usize,
    sample_rate: f64,
) -> Bound<'py, PyArray1<f64>> {
    let result = audio_math::fft_frequencies(n_fft, sample_rate);
    result.into_pyarray(py)
}

/// Frequencies uniformly spaced on the mel scale, like ``librosa.mel_frequencies``.
///
/// **Default scale is HTK** (unchanged from earlier releases); librosa defaults
/// to Slaney (``htk=False``).
///
/// Args:
///     n_mels (int): Number of frequencies.
///     fmin (float): Lowest frequency in Hertz.
///     fmax (float): Highest frequency in Hertz.
///     htk (bool): HTK scale (default True) or Slaney (False, librosa's default).
///
/// Returns:
///     numpy.ndarray[float64]: shape ``(n_mels,)``, in Hertz.
///
/// Example:
///     >>> from audio_samples.utils import mel_frequencies
///     >>> mel_frequencies(128, 0.0, 11025.0, htk=False)  # == librosa.mel_frequencies()
#[pyfunction]
#[pyo3(signature = (n_mels, fmin, fmax, *, htk = true), text_signature = "(n_mels, fmin, fmax, *, htk=True)")]
pub fn mel_frequencies(
    py: Python<'_>,
    n_mels: Int,
    fmin: f64,
    fmax: f64,
    htk: bool,
) -> PyResult<Bound<'_, PyArray1<f64>>> {
    let n = non_negative("n_mels", n_mels)?;
    let result = audio_math::mel_frequencies_with_scale(n, fmin, fmax, mel_scale_of(htk));
    Ok(result.into_pyarray(py))
}

#[pyfunction]
#[pyo3(signature = (start, end, num))]
/// Generates a linearly spaced sequence of values between two endpoints.
///
/// Args:
///     start: The starting value of the sequence.
///     end: The final value of the sequence.
///     num: The number of points to generate.
///
/// Returns:
///     A numpy array of linearly interpolated values.
///
/// Example:
///     >>> from audio_samples.utils import linspace
///     >>> vals = linspace(0.0, 1.0, 11)  # [0.0, 0.1, 0.2, ..., 1.0]
#[inline]
pub fn linspace<'py>(
    py: Python<'py>,
    start: f64,
    end: f64,
    num: usize,
) -> Bound<'py, PyArray1<f64>> {
    let result = audio_math::linspace(start, end, num);
    result.into_pyarray(py)
}

// =============================================================================
// LIBROSA-COMPATIBLE FRAME / SAMPLE / TIME CONVERSIONS (vectorised)
// =============================================================================

/// Converts times in seconds to sample indices, like ``librosa.time_to_samples``.
///
/// ``int(t * sample_rate)`` truncated toward zero (negative times stay negative).
/// Differs from ``seconds_to_samples``, which rounds.
///
/// Args:
///     times (float | array_like): Time(s) in seconds.
///     sample_rate (Optional[float]): Sampling rate in Hertz. Default 22050.
///     sr (Optional[float]): librosa's name for ``sample_rate`` (give at most one).
///
/// Returns:
///     int | numpy.ndarray[int64]: Sample indices, same shape as the input.
///
/// Example:
///     >>> from audio_samples.utils import time_to_samples
///     >>> time_to_samples([0.0, 0.5, -0.5], sample_rate=100.0)  # [0, 50, -50]
#[pyfunction]
#[pyo3(signature = (times, *, sample_rate = None, sr = None), text_signature = "(times, *, sample_rate=None, sr=None)")]
pub fn time_to_samples<'py>(
    times: &Bound<'py, PyAny>,
    sample_rate: Option<f64>,
    sr: Option<f64>,
) -> PyResult<Bound<'py, PyAny>> {
    let sample_rate = sample_rate_or_sr(sample_rate, sr)?;
    map_f64_to_i64(times, "times", |t| {
        audio_math::time_to_samples(t, sample_rate)
    })
}

/// Converts sample indices to frame indices, like ``librosa.samples_to_frames``.
///
/// ``floor((samples - n_fft // 2) / hop_length)`` (the offset only when ``n_fft``
/// is given).
///
/// Args:
///     samples (int | array_like[int]): Sample index or indices (may be negative).
///     hop_length (int): Samples between frames. Default 512.
///     n_fft (Optional[int]): Optional frame-length offset.
///
/// Returns:
///     int | numpy.ndarray[int64]: Frame indices, same shape as the input.
#[pyfunction]
#[pyo3(signature = (samples, *, hop_length = Int(512), n_fft = None), text_signature = "(samples, *, hop_length=512, n_fft=None)")]
pub fn samples_to_frames<'py>(
    samples: &Bound<'py, PyAny>,
    hop_length: Int,
    n_fft: Option<Int>,
) -> PyResult<Bound<'py, PyAny>> {
    let hop = positive("hop_length", hop_length)?;
    let n_fft = opt_non_negative("n_fft", n_fft)?;
    map_i64(samples, "samples", |s| {
        audio_math::samples_to_frames(s, hop, n_fft)
    })
}

/// Converts frame indices to sample indices, like ``librosa.frames_to_samples``.
///
/// ``frames * hop_length + n_fft // 2`` (the offset only when ``n_fft`` is given).
///
/// Args:
///     frames (int | array_like[int]): Frame index or indices.
///     hop_length (int): Samples between frames. Default 512.
///     n_fft (Optional[int]): Optional frame-length offset.
///
/// Returns:
///     int | numpy.ndarray[int64]: Sample indices, same shape as the input.
#[pyfunction]
#[pyo3(signature = (frames, *, hop_length = Int(512), n_fft = None), text_signature = "(frames, *, hop_length=512, n_fft=None)")]
pub fn frames_to_samples<'py>(
    frames: &Bound<'py, PyAny>,
    hop_length: Int,
    n_fft: Option<Int>,
) -> PyResult<Bound<'py, PyAny>> {
    let hop = positive("hop_length", hop_length)?;
    let n_fft = opt_non_negative("n_fft", n_fft)?;
    map_i64(frames, "frames", |f| {
        audio_math::frames_to_samples(f, hop, n_fft)
    })
}

/// Converts times in seconds to the frame **containing** each time, exactly as
/// ``librosa.time_to_frames``.
///
/// ``samples_to_frames(time_to_samples(t, sample_rate), hop_length, n_fft)``:
/// truncation to samples, then floor division (negatives kept). Compare
/// ``time_to_frames``, which rounds to the nearest frame.
///
/// Args:
///     times (float | array_like): Time(s) in seconds.
///     sample_rate (Optional[float]): Sampling rate in Hertz. Default 22050.
///     sr (Optional[float]): librosa's name for ``sample_rate`` (give at most one).
///     hop_length (int): Samples between frames. Default 512.
///     n_fft (Optional[int]): Optional frame-length offset.
///
/// Returns:
///     int | numpy.ndarray[int64]: Frame indices, same shape as the input.
///
/// Example:
///     >>> from audio_samples.utils import time_to_frames_floor
///     >>> time_to_frames_floor([0.0, 0.1, 0.2])  # [0, 4, 8]
#[pyfunction]
#[pyo3(signature = (times, *, sample_rate = None, hop_length = Int(512), n_fft = None, sr = None), text_signature = "(times, *, sample_rate=None, hop_length=512, n_fft=None, sr=None)")]
pub fn time_to_frames_floor<'py>(
    times: &Bound<'py, PyAny>,
    sample_rate: Option<f64>,
    hop_length: Int,
    n_fft: Option<Int>,
    sr: Option<f64>,
) -> PyResult<Bound<'py, PyAny>> {
    let sample_rate = sample_rate_or_sr(sample_rate, sr)?;
    let hop = positive("hop_length", hop_length)?;
    let n_fft = opt_non_negative("n_fft", n_fft)?;
    map_f64_to_i64(times, "times", |t| {
        audio_math::time_to_frames_floor(t, sample_rate, hop, n_fft)
    })
}

/// `sample_rate` or its librosa spelling `sr` (default 22050).
fn sample_rate_or_sr(sample_rate: Option<f64>, sr: Option<f64>) -> PyResult<f64> {
    match (sample_rate, sr) {
        (Some(_), Some(_)) => Err(pyo3::exceptions::PyTypeError::new_err(
            "give sample_rate or its alias sr, not both",
        )),
        (Some(v), None) | (None, Some(v)) => Ok(v),
        (None, None) => Ok(22050.0),
    }
}

/// Number of frames from an int, or from axis `axis` of an array (librosa's
/// ``X.shape[axis]``).
fn frame_count(x: &Bound<'_, PyAny>, axis: Int) -> PyResult<usize> {
    if is_scalar(x)? {
        let n: Int = x
            .extract()
            .map_err(|_| value_err("X", "expected an integer frame count or an array"))?;
        return non_negative("X", n);
    }
    let shape: Vec<usize> = x
        .py()
        .import("numpy")?
        .call_method1("shape", (x,))?
        .extract()?;
    if shape.is_empty() {
        return Err(value_err("X", "cannot take the frame count of a 0-d array"));
    }
    let nd = shape.len() as i64;
    let ax = if axis.0 < 0 {
        axis.0.saturating_add(nd)
    } else {
        axis.0
    };
    if !(0..nd).contains(&ax) {
        return Err(value_err(
            "axis",
            format!("{axis} is out of range for an array with {nd} dimensions"),
        ));
    }
    Ok(shape[ax as usize])
}

/// Time (seconds) of each frame of a feature matrix, like ``librosa.times_like``.
///
/// Args:
///     X (int | array_like): Number of frames, or an array whose ``axis`` is time.
///     sample_rate (Optional[float]): Sampling rate in Hertz. Default 22050.
///     sr (Optional[float]): librosa's name for ``sample_rate`` (give at most one).
///     hop_length (int): Samples between frames. Default 512.
///     n_fft (Optional[int]): Optional frame-length offset.
///     axis (int): The time axis of ``X``. Default -1.
///
/// Returns:
///     numpy.ndarray[float64]: shape ``(n_frames,)``.
#[pyfunction]
#[pyo3(signature = (X, *, sample_rate = None, hop_length = Int(512), n_fft = None, axis = Int(-1), sr = None), text_signature = "(X, *, sample_rate=None, hop_length=512, n_fft=None, axis=-1, sr=None)")]
#[allow(non_snake_case)]
pub fn times_like<'py>(
    py: Python<'py>,
    X: &Bound<'py, PyAny>,
    sample_rate: Option<f64>,
    hop_length: Int,
    n_fft: Option<Int>,
    axis: Int,
    sr: Option<f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let sample_rate = sample_rate_or_sr(sample_rate, sr)?;
    let hop = positive("hop_length", hop_length)?;
    let n_fft = opt_non_negative("n_fft", n_fft)?;
    Ok(audio_math::times_like(frame_count(X, axis)?, sample_rate, hop, n_fft).into_pyarray(py))
}

/// Sample index of each frame of a feature matrix, like ``librosa.samples_like``.
///
/// Args:
///     X (int | array_like): Number of frames, or an array whose ``axis`` is time.
///     hop_length (int): Samples between frames. Default 512.
///     n_fft (Optional[int]): Optional frame-length offset.
///     axis (int): The time axis of ``X``. Default -1.
///
/// Returns:
///     numpy.ndarray[int64]: shape ``(n_frames,)``.
#[pyfunction]
#[pyo3(signature = (X, *, hop_length = Int(512), n_fft = None, axis = Int(-1)), text_signature = "(X, *, hop_length=512, n_fft=None, axis=-1)")]
#[allow(non_snake_case)]
pub fn samples_like<'py>(
    py: Python<'py>,
    X: &Bound<'py, PyAny>,
    hop_length: Int,
    n_fft: Option<Int>,
    axis: Int,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let hop = positive("hop_length", hop_length)?;
    let n_fft = opt_non_negative("n_fft", n_fft)?;
    Ok(audio_math::samples_like(frame_count(X, axis)?, hop, n_fft).into_pyarray(py))
}

// =============================================================================
// MU-LAW CODES
// =============================================================================

/// Quantise samples to integer μ-law codes, like
/// ``librosa.mu_compress(x, mu=mu, quantize=True)``.
///
/// Companding ``sign(x) * log1p(mu * |x|) / log1p(mu)`` followed by binning
/// into ``floor(1 + mu)`` levels; for ``mu=255`` the codes are the integers
/// ``-128..=127``. For float-in/float-out companding of a whole signal use
/// ``AudioSamples.mu_compress``.
///
/// Args:
///     x (float | array_like): Samples in ``[-1, 1]``, any shape.
///     mu (float): Compression parameter (> 0). Default 255.
///
/// Returns:
///     int | numpy.ndarray[int64]: Codes, same shape as ``x``.
///
/// Raises:
///     ValueError: If ``mu <= 0`` or any sample is outside ``[-1, 1]``.
///
/// Example:
///     >>> from audio_samples import mu_law_encode
///     >>> mu_law_encode([-1.0, 0.0, 1.0])  # array([-128, 0, 127])
#[pyfunction]
#[pyo3(signature = (x, *, mu = 255.0), text_signature = "(x, *, mu=255.0)")]
pub fn mu_law_encode<'py>(x: &Bound<'py, PyAny>, mu: f64) -> PyResult<Bound<'py, PyAny>> {
    use pyo3::IntoPyObjectExt;
    let py = x.py();
    let a = crate::args::f64_array(x, "x")?;
    let flat: Vec<f64> = a.as_array().iter().copied().collect();
    let codes = audio_samples::mu_law_encode(&flat, mu).map_err(audio_err_to_py)?;
    if is_scalar(x)? {
        return codes[0].into_bound_py_any(py);
    }
    let out = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(a.shape()), codes)
        .map_err(|e| value_err("x", e))?;
    Ok(out.into_pyarray(py).into_any())
}

/// Map integer μ-law codes back to amplitudes, like
/// ``librosa.mu_expand(codes, mu=mu, quantize=True)``.
///
/// Not an exact inverse of ``mu_law_encode`` (quantisation error up to ~0.043
/// for ``mu=255``).
///
/// Args:
///     codes (int | array_like[int]): μ-law codes, any shape.
///     mu (float): The ``mu`` used for encoding. Default 255.
///
/// Returns:
///     float | numpy.ndarray[float64]: Amplitudes in ``[-1, 1]``, same shape.
///
/// Raises:
///     ValueError: If ``mu <= 0`` or a code is out of range.
#[pyfunction]
#[pyo3(signature = (codes, *, mu = 255.0), text_signature = "(codes, *, mu=255.0)")]
pub fn mu_law_decode<'py>(codes: &Bound<'py, PyAny>, mu: f64) -> PyResult<Bound<'py, PyAny>> {
    use pyo3::IntoPyObjectExt;
    let py = codes.py();
    let a = crate::args::f64_array(codes, "codes")?;
    let ints = crate::args::i64_vec(codes, "codes")?;
    let vals = audio_samples::mu_law_decode(&ints, mu).map_err(audio_err_to_py)?;
    if is_scalar(codes)? {
        return vals[0].into_bound_py_any(py);
    }
    let out = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(a.shape()), vals)
        .map_err(|e| value_err("codes", e))?;
    Ok(out.into_pyarray(py).into_any())
}
