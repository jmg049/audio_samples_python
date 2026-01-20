//! Python bindings for audio mathematical utilities and canonical unit conversions.
//!
//! This module exposes the audio_samples::utils::audio_math functions to Python,
//! providing frequency conversions, amplitude/power conversions, time/frame conversions,
//! musical theory functions, and spectral utilities.

use audio_samples::utils::audio_math;
use numpy::{IntoPyArray, PyArray1};
use pyo3::prelude::*;

use crate::audio_err_to_py;

// =============================================================================
// FREQUENCY CONVERSIONS
// =============================================================================
/// Converts a frequency expressed in Hertz into the mel perceptual scale.
///
/// Args:
///     freq_hz: The input frequency in Hertz.
///
/// Returns:
///     The corresponding value on the mel scale.
///
/// Example:
///     >>> from audio_samples.audio_math import hz_to_mel
///     >>> mel = hz_to_mel(1000.0)
#[pyfunction]
#[pyo3(signature = (freq_hz: "float"), text_signature = "(freq_hz: float) -> float")]
#[inline]
pub fn hz_to_mel(freq_hz: f64) -> f64 {
    audio_math::hz_to_mel(freq_hz)
}

#[pyfunction]
#[pyo3(signature = (mel), text_signature = "(mel: float) -> float")]
/// Converts a value expressed on the mel perceptual scale back into Hertz.
///
/// Args:
///     mel: A value on the mel scale.
///
/// Returns:
///     The corresponding frequency in Hertz.
///
/// Example:
///     >>> from audio_samples.audio_math import mel_to_hz
///     >>> hz = mel_to_hz(1000.0)
#[inline]
pub fn mel_to_hz(mel: f64) -> f64 {
    audio_math::mel_to_hz(mel)
}

#[pyfunction]
#[pyo3(signature = (n_mels, fmin, fmax), text_signature = "(n_mels: int, fmin: float, fmax: float) -> ndarray")]
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
///     >>> from audio_samples.audio_math import mel_scale
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

#[pyfunction]
#[pyo3(signature = (freq_hz), text_signature = "(freq_hz: float) -> float")]
/// Converts a frequency expressed in Hertz into a MIDI note value.
///
/// Args:
///     freq_hz: The input frequency in Hertz. Must be positive.
///
/// Returns:
///     A MIDI note value (continuous, not quantized to integer semitones).
///
/// Example:
///     >>> from audio_samples.audio_math import hz_to_midi
///     >>> midi = hz_to_midi(440.0)  # Returns ~69.0 (A4)
#[inline]
pub fn hz_to_midi(freq_hz: f64) -> f64 {
    audio_math::hz_to_midi(freq_hz)
}

#[pyfunction]
#[pyo3(signature = (midi_note), text_signature = "(midi_note: float) -> float")]
/// Converts a MIDI note value into a frequency expressed in Hertz.
///
/// Args:
///     midi_note: A MIDI note value (fractional values are permitted).
///
/// Returns:
///     The corresponding frequency in Hertz.
///
/// Example:
///     >>> from audio_samples.audio_math import midi_to_hz
///     >>> hz = midi_to_hz(69.0)  # Returns 440.0 (A4)
#[inline]
pub fn midi_to_hz(midi_note: f64) -> f64 {
    audio_math::midi_to_hz(midi_note)
}

// =============================================================================
// AMPLITUDE CONVERSIONS
// =============================================================================

#[pyfunction]
#[pyo3(signature = (amplitude), text_signature = "(amplitude: float) -> float")]
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
///     >>> from audio_samples.audio_math import amplitude_to_db
///     >>> db = amplitude_to_db(0.5)  # Returns ~-6.02 dB
#[inline]
pub fn amplitude_to_db(amplitude: f64) -> f64 {
    audio_math::amplitude_to_db(amplitude)
}

#[pyfunction]
#[pyo3(signature = (db), text_signature = "(db: float) -> float")]
/// Converts a decibel value into a linear amplitude ratio.
///
/// Args:
///     db: A value expressed in decibels.
///
/// Returns:
///     The corresponding linear amplitude ratio.
///
/// Example:
///     >>> from audio_samples.audio_math import db_to_amplitude
///     >>> amp = db_to_amplitude(-6.0)  # Returns ~0.501
#[inline]
pub fn db_to_amplitude(db: f64) -> f64 {
    audio_math::db_to_amplitude(db)
}

#[pyfunction]
#[pyo3(signature = (power), text_signature = "(power: float) -> float")]
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
///     >>> from audio_samples.audio_math import power_to_db
///     >>> db = power_to_db(0.5)  # Returns ~-3.01 dB
#[inline]
pub fn power_to_db(power: f64) -> f64 {
    audio_math::power_to_db(power)
}

#[pyfunction]
#[pyo3(signature = (db), text_signature = "(db: float) -> float")]
/// Converts a decibel value into a linear power ratio.
///
/// Args:
///     db: A value expressed in decibels.
///
/// Returns:
///     The corresponding linear power ratio.
///
/// Example:
///     >>> from audio_samples.audio_math import db_to_power
///     >>> power = db_to_power(-3.0)  # Returns ~0.501
#[inline]
pub fn db_to_power(db: f64) -> f64 {
    audio_math::db_to_power(db)
}

// =============================================================================
// TIME/FRAME CONVERSIONS
// =============================================================================

#[pyfunction]
#[pyo3(signature = (frames, sample_rate, hop_size), text_signature = "(frames: int, sample_rate: float, hop_size: int) -> float")]
/// Converts a frame index into a time offset expressed in seconds.
///
/// Args:
///     frames: The frame index to convert.
///     sample_rate: The sampling rate in Hertz.
///     hop_size: The number of samples between successive frames.
///
/// Returns:
///     The corresponding time offset in seconds.
///
/// Example:
///     >>> from audio_samples.audio_math import frames_to_time
///     >>> t = frames_to_time(100, 44100.0, 512)
#[inline]
pub fn frames_to_time(frames: usize, sample_rate: f64, hop_size: usize) -> f64 {
    audio_math::frames_to_time(frames, sample_rate, hop_size)
}

#[pyfunction]
#[pyo3(signature = (time_seconds, sample_rate, hop_size), text_signature = "(time_seconds: float, sample_rate: float, hop_size: int) -> int")]
/// Converts a time offset expressed in seconds into the nearest frame index.
///
/// Args:
///     time_seconds: The time offset in seconds.
///     sample_rate: The sampling rate in Hertz.
///     hop_size: The number of samples between successive frames.
///
/// Returns:
///     The nearest frame index.
///
/// Example:
///     >>> from audio_samples.audio_math import time_to_frames
///     >>> frame = time_to_frames(1.0, 44100.0, 512)
#[inline]
pub fn time_to_frames(time_seconds: f64, sample_rate: f64, hop_size: usize) -> usize {
    audio_math::time_to_frames(time_seconds, sample_rate, hop_size)
}

#[pyfunction]
#[pyo3(signature = (samples, sample_rate), text_signature = "(samples: int, sample_rate: float) -> float")]
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
///     >>> from audio_samples.audio_math import samples_to_time
///     >>> t = samples_to_time(44100, 44100.0)  # Returns 1.0
#[inline]
pub fn samples_to_time(samples: usize, sample_rate: f64) -> f64 {
    audio_math::samples_to_time(samples, sample_rate)
}

#[pyfunction]
#[pyo3(signature = (time_seconds, sample_rate), text_signature = "(time_seconds: float, sample_rate: float) -> int")]
/// Converts a time offset expressed in seconds into the nearest sample index.
///
/// Args:
///     time_seconds: The time offset in seconds.
///     sample_rate: The sampling rate in Hertz.
///
/// Returns:
///     The nearest sample index.
///
/// Example:
///     >>> from audio_samples.audio_math import seconds_to_samples
///     >>> samples = seconds_to_samples(1.0, 44100.0)  # Returns 44100
#[inline]
pub fn seconds_to_samples(time_seconds: f64, sample_rate: f64) -> usize {
    audio_math::seconds_to_samples(time_seconds, sample_rate)
}

#[pyfunction]
#[pyo3(signature = (ms, sample_rate), text_signature = "(ms: float, sample_rate: float) -> int")]
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
///     >>> from audio_samples.audio_math import ms_to_samples
///     >>> samples = ms_to_samples(1000.0, 44100.0)  # Returns 44100
#[inline]
pub fn ms_to_samples(ms: f64, sample_rate: f64) -> usize {
    audio_math::ms_to_samples(ms, sample_rate)
}

// =============================================================================
// MUSICAL THEORY FUNCTIONS
// =============================================================================

#[pyfunction]
#[pyo3(signature = (note_name), text_signature = "(note_name: str) -> int")]
/// Parses a scientific pitch notation string and converts it into a MIDI note number.
///
/// Supports sharps (#) and flats (b). Case-sensitive, requires uppercase note letters.
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
///     >>> from audio_samples.audio_math import note_to_midi
///     >>> midi = note_to_midi("A4")  # Returns 69
///     >>> midi = note_to_midi("C#4")  # Returns 61
#[inline]
pub fn note_to_midi(note_name: &str) -> PyResult<u8> {
    audio_math::note_to_midi(note_name).map_err(audio_err_to_py)
}

#[pyfunction]
#[pyo3(signature = (midi_note), text_signature = "(midi_note: int) -> str")]
/// Converts a MIDI note number into a scientific pitch notation string.
///
/// Always uses sharp notation for accidentals.
///
/// Args:
///     midi_note: A MIDI note number (0-127).
///
/// Returns:
///     A note name in scientific pitch notation (e.g., "A4", "C#3").
///
/// Raises:
///     ValueError: If the MIDI note is outside the valid range 0-127.
///
/// Example:
///     >>> from audio_samples.audio_math import midi_to_note
///     >>> note = midi_to_note(69)  # Returns "A4"
///     >>> note = midi_to_note(61)  # Returns "C#4"
#[inline]
pub fn midi_to_note(midi_note: u8) -> PyResult<String> {
    audio_math::midi_to_note(midi_note).map_err(audio_err_to_py)
}

#[pyfunction]
#[pyo3(signature = (note_name), text_signature = "(note_name: str) -> float")]
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
///     >>> from audio_samples.audio_math import note_to_frequency
///     >>> freq = note_to_frequency("A4")  # Returns 440.0
#[inline]
pub fn note_to_frequency(note_name: &str) -> PyResult<f64> {
    audio_math::note_to_frequency(note_name).map_err(audio_err_to_py)
}

#[pyfunction]
#[pyo3(signature = (freq_hz), text_signature = "(freq_hz: float) -> tuple[str, float]")]
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
///     >>> from audio_samples.audio_math import frequency_to_note
///     >>> note, cents = frequency_to_note(440.0)  # Returns ("A4", 0.0)
///     >>> note, cents = frequency_to_note(445.0)  # Returns ("A4", ~19.56)
#[inline]
pub fn frequency_to_note(freq_hz: f64) -> PyResult<(String, f64)> {
    audio_math::frequency_to_note(freq_hz).map_err(audio_err_to_py)
}

#[pyfunction]
#[pyo3(signature = (cents), text_signature = "(cents: float) -> float")]
/// Converts a pitch deviation expressed in cents into a frequency ratio.
///
/// Args:
///     cents: A pitch deviation in cents.
///
/// Returns:
///     A multiplicative frequency ratio.
///
/// Example:
///     >>> from audio_samples.audio_math import cents_to_ratio
///     >>> ratio = cents_to_ratio(1200.0)  # Returns 2.0 (one octave)
///     >>> ratio = cents_to_ratio(100.0)  # Returns ~1.059 (one semitone)
#[inline]
pub fn cents_to_ratio(cents: f64) -> f64 {
    audio_math::cents_to_ratio(cents)
}

#[pyfunction]
#[pyo3(signature = (ratio), text_signature = "(ratio: float) -> float")]
/// Converts a frequency ratio into a pitch deviation expressed in cents.
///
/// Args:
///     ratio: A frequency ratio. Must be positive.
///
/// Returns:
///     The corresponding pitch deviation in cents.
///
/// Example:
///     >>> from audio_samples.audio_math import ratio_to_cents
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
#[pyo3(signature = (n_fft, sample_rate), text_signature = "(n_fft: int, sample_rate: float) -> ndarray")]
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
///     >>> from audio_samples.audio_math import fft_frequencies
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

#[pyfunction]
#[pyo3(signature = (n_mels, fmin, fmax), text_signature = "(n_mels: int, fmin: float, fmax: float) -> ndarray")]
/// Generates frequencies whose spacing is uniform in the mel perceptual domain.
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
///     >>> from audio_samples.audio_math import mel_frequencies
///     >>> freqs = mel_frequencies(10, 0.0, 8000.0)
#[inline]
pub fn mel_frequencies<'py>(
    py: Python<'py>,
    n_mels: usize,
    fmin: f64,
    fmax: f64,
) -> Bound<'py, PyArray1<f64>> {
    let result = audio_math::mel_frequencies(n_mels, fmin, fmax);
    result.into_pyarray(py)
}

#[pyfunction]
#[pyo3(signature = (start, end, num), text_signature = "(start: float, end: float, num: int) -> ndarray")]
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
///     >>> from audio_samples.audio_math import linspace
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
