"""Audio mathematical utilities and canonical unit conversions.

This module provides frequency conversions, amplitude/power conversions,
time/frame conversions, musical theory functions, and spectral utilities.
"""

from typing import Any, Optional, Union

import numpy as np
import numpy.typing as npt
from numpy.typing import NDArray

# =============================================================================
# FREQUENCY CONVERSIONS
# =============================================================================

def hz_to_mel(freq_hz: npt.ArrayLike, *, htk: bool = True) -> Any:
    """Convert frequencies in Hertz to mels (scalar or array).

    **The default is the HTK scale** (``2595 * log10(1 + f / 700)``), unchanged
    from earlier releases. librosa's ``hz_to_mel`` defaults to the Slaney scale:
    pass ``htk=False`` to match it.

    Returns:
        float for scalar input, otherwise numpy.ndarray[float64] of the same shape.
    """
    ...

def mel_to_hz(mel: npt.ArrayLike, *, htk: bool = True) -> Any:
    """Convert mels to Hertz (scalar or array). Default HTK (librosa default:
    Slaney, ``htk=False``).

    Returns:
        float for scalar input, otherwise numpy.ndarray[float64] of the same shape.
    """
    ...

def mel_scale(n_mels: int, fmin: float, fmax: float) -> NDArray[np.float64]:
    """Generate frequencies spaced uniformly in the mel domain.

    Args:
        n_mels: The number of frequency points to generate.
        fmin: The lower bound of the frequency range in Hertz.
        fmax: The upper bound of the frequency range in Hertz.

    Returns:
        A numpy array of frequencies in Hertz, spaced uniformly in mel scale.
    """
    ...

def hz_to_midi(freq_hz: npt.ArrayLike) -> Any:
    """``12 * log2(f / 440) + 69`` (``librosa.hz_to_midi``); float for scalar
    input, otherwise numpy.ndarray[float64] of the same shape."""
    ...

def midi_to_hz(midi_note: npt.ArrayLike) -> Any:
    """``440 * 2 ** ((m - 69) / 12)`` (``librosa.midi_to_hz``); float for scalar
    input, otherwise numpy.ndarray[float64] of the same shape."""
    ...

# =============================================================================
# AMPLITUDE CONVERSIONS
# =============================================================================

def amplitude_to_db(amplitude: float) -> float:
    """Convert linear amplitude ratio to logarithmic decibels.

    Non-positive amplitudes are mapped to -80 dB.

    Args:
        amplitude: The input amplitude ratio.

    Returns:
        The amplitude expressed in decibels.
    """
    ...

def db_to_amplitude(db: float) -> float:
    """Convert decibel value to linear amplitude ratio.

    Args:
        db: A value expressed in decibels.

    Returns:
        The corresponding linear amplitude ratio.
    """
    ...

def power_to_db(power: float) -> float:
    """Convert linear power ratio to logarithmic decibels.

    Non-positive power values are mapped to -80 dB.

    Args:
        power: The input power ratio.

    Returns:
        The power expressed in decibels.
    """
    ...

def db_to_power(db: float) -> float:
    """Convert decibel value to linear power ratio.

    Args:
        db: A value expressed in decibels.

    Returns:
        The corresponding linear power ratio.
    """
    ...

# =============================================================================
# TIME/FRAME CONVERSIONS
# =============================================================================

def frames_to_time(
    frames: npt.ArrayLike, sample_rate: float, hop_size: int, *, n_fft: Optional[int] = None
) -> Any:
    """Frame indices to seconds, like ``librosa.frames_to_time``:
    ``(frames * hop_size + n_fft // 2) / sample_rate`` (offset only with ``n_fft``).

    Args:
        frames: Frame index or array of indices (negative allowed).
        sample_rate: Sampling rate in Hz.
        hop_size: Samples between frames.
        n_fft: Optional frame-length offset.

    Returns:
        float for scalar input, otherwise numpy.ndarray[float64] of the same shape.
    """
    ...

def time_to_frames(time_seconds: npt.ArrayLike, sample_rate: float, hop_size: int) -> Any:
    """Times to the **nearest** frame (``round(t * sr / hop)``, negatives clamp to 0).

    Kept for backward compatibility; librosa's ``time_to_frames`` (floor, negatives
    kept, ``n_fft`` offset) is ``time_to_frames_floor``.

    Returns:
        int for scalar input, otherwise numpy.ndarray[int64] of the same shape.
    """
    ...

def samples_to_time(samples: int, sample_rate: float) -> float:
    """Convert sample index to time offset in seconds.

    Args:
        samples: The sample index to convert.
        sample_rate: The sampling rate in Hertz.

    Returns:
        The corresponding time offset in seconds.
    """
    ...

def seconds_to_samples(time_seconds: float, sample_rate: float) -> int:
    """Seconds to the **nearest** sample index (rounds; negatives clamp to 0).

    librosa's ``time_to_samples`` truncates instead: use ``time_to_samples``.
    """
    ...

def ms_to_samples(ms: float, sample_rate: float) -> int:
    """Convert time in milliseconds to sample count.

    Args:
        ms: Time in milliseconds.
        sample_rate: The sampling rate in Hertz.

    Returns:
        The corresponding number of samples.
    """
    ...

# =============================================================================
# MUSICAL THEORY FUNCTIONS
# =============================================================================

def note_to_midi(note_name: str) -> int:
    """Parse scientific pitch notation string to MIDI note number.

    Supports sharps (#) and flats (b). Case-sensitive, requires uppercase
    note letters.

    Args:
        note_name: A note name in scientific pitch notation (e.g., "A4", "C#3", "Bb2").

    Returns:
        The corresponding MIDI note number (0-127).

    Raises:
        ValueError: If the note name is invalid or out of MIDI range.
    """
    ...

def midi_to_note(midi_note: int) -> str:
    """MIDI number (0-127) to a note name with sharps, e.g. ``"A4"``, ``"C#3"``.

    Raises:
        ValueError: If ``midi_note`` is outside 0-127 (including negative values).
    """
    ...

def note_to_frequency(note_name: str) -> float:
    """Convert scientific pitch notation string to frequency in Hertz.

    Args:
        note_name: A note name in scientific pitch notation (e.g., "A4", "C#3").

    Returns:
        The corresponding frequency in Hertz.

    Raises:
        ValueError: If the note name is invalid or out of MIDI range.
    """
    ...

def frequency_to_note(freq_hz: float) -> tuple[str, float]:
    """Convert frequency in Hertz to nearest musical note and cents offset.

    Args:
        freq_hz: The input frequency in Hertz. Must be positive.

    Returns:
        A tuple (note_name, cents) where note_name is the nearest pitch and
        cents is the signed deviation from that pitch.

    Raises:
        ValueError: If the frequency is non-positive.
    """
    ...

def cents_to_ratio(cents: float) -> float:
    """Convert pitch deviation in cents to frequency ratio.

    Args:
        cents: A pitch deviation in cents.

    Returns:
        A multiplicative frequency ratio.
    """
    ...

def ratio_to_cents(ratio: float) -> float:
    """Convert frequency ratio to pitch deviation in cents.

    Args:
        ratio: A frequency ratio. Must be positive.

    Returns:
        The corresponding pitch deviation in cents.
    """
    ...

# =============================================================================
# SPECTRAL HELPER FUNCTIONS
# =============================================================================

def fft_frequencies(n_fft: int, sample_rate: float) -> NDArray[np.float64]:
    """Generate positive-frequency bin centres for a real-valued FFT.

    Args:
        n_fft: The FFT size in samples.
        sample_rate: The sampling rate in Hertz.

    Returns:
        A numpy array of frequency bin centres in Hertz, from DC to Nyquist.
        Length is n_fft/2 + 1.
    """
    ...

def mel_frequencies(n_mels: int, fmin: float, fmax: float, *, htk: bool = True) -> NDArray[np.float64]:
    """``n_mels`` frequencies uniformly spaced in mels, like ``librosa.mel_frequencies``.

    **Default HTK** (unchanged); librosa defaults to Slaney: pass ``htk=False``.

    Returns:
        numpy.ndarray[float64], shape ``(n_mels,)``.
    """
    ...

def linspace(start: float, end: float, num: int) -> NDArray[np.float64]:
    """Generate linearly spaced sequence of values between two endpoints.

    Args:
        start: The starting value of the sequence.
        end: The final value of the sequence.
        num: The number of points to generate.

    Returns:
        A numpy array of linearly interpolated values.
    """
    ...

# =============================================================================
# LIBROSA-COMPATIBLE FRAME / SAMPLE / TIME CONVERSIONS (vectorised)
# =============================================================================

def time_to_samples(
    times: npt.ArrayLike, *, sample_rate: Optional[float] = None, sr: Optional[float] = None
) -> Any:
    """``int(t * sample_rate)`` truncated toward zero, like ``librosa.time_to_samples``.

    ``sample_rate`` defaults to 22050; ``sr`` is librosa's name for it (give at
    most one, both is a ``TypeError``).

    Returns:
        int for scalar input, otherwise numpy.ndarray[int64] of the same shape.
    """
    ...

def samples_to_frames(samples: npt.ArrayLike, *, hop_length: int = 512, n_fft: Optional[int] = None) -> Any:
    """``floor((samples - n_fft // 2) / hop_length)``, like ``librosa.samples_to_frames``.

    Returns:
        int for scalar input, otherwise numpy.ndarray[int64] of the same shape.
    """
    ...

def frames_to_samples(frames: npt.ArrayLike, *, hop_length: int = 512, n_fft: Optional[int] = None) -> Any:
    """``frames * hop_length + n_fft // 2``, like ``librosa.frames_to_samples``.

    Returns:
        int for scalar input, otherwise numpy.ndarray[int64] of the same shape.
    """
    ...

def time_to_frames_floor(
    times: npt.ArrayLike,
    *,
    sample_rate: Optional[float] = None,
    hop_length: int = 512,
    n_fft: Optional[int] = None,
    sr: Optional[float] = None,
) -> Any:
    """The frame containing each time, exactly ``librosa.time_to_frames``
    (truncate to samples, then floor-divide; negatives kept).

    ``sample_rate`` defaults to 22050; ``sr`` is librosa's name for it.

    Returns:
        int for scalar input, otherwise numpy.ndarray[int64] of the same shape.
    """
    ...

def times_like(
    X: Union[int, npt.ArrayLike],
    *,
    sample_rate: Optional[float] = None,
    hop_length: int = 512,
    n_fft: Optional[int] = None,
    axis: int = -1,
    sr: Optional[float] = None,
) -> NDArray[np.float64]:
    """Frame times in seconds for ``X`` frames (an int, or an array whose
    ``axis`` is time), like ``librosa.times_like``. Shape ``(n_frames,)``.

    ``sample_rate`` defaults to 22050; ``sr`` is librosa's name for it."""
    ...

def samples_like(
    X: Union[int, npt.ArrayLike], *, hop_length: int = 512, n_fft: Optional[int] = None, axis: int = -1
) -> NDArray[np.int64]:
    """Frame sample indices, like ``librosa.samples_like`` (``axis`` is the time
    axis of an array ``X``). Shape ``(n_frames,)``."""
    ...

# =============================================================================
# MU-LAW CODES
# =============================================================================

def mu_law_encode(x: npt.ArrayLike, *, mu: float = 255.0) -> Any:
    """Integer mu-law codes, like ``librosa.mu_compress(x, mu=mu, quantize=True)``
    (codes ``-128..=127`` for ``mu=255``).

    Args:
        x: Samples in ``[-1, 1]``, any shape.

    Returns:
        int for scalar input, otherwise numpy.ndarray[int64] of the same shape.

    Raises:
        ValueError: ``mu <= 0`` or a sample outside ``[-1, 1]``.
    """
    ...

def mu_law_decode(codes: npt.ArrayLike, *, mu: float = 255.0) -> Any:
    """Amplitudes from mu-law codes, like ``librosa.mu_expand(codes, mu=mu, quantize=True)``.

    Returns:
        float for scalar input, otherwise numpy.ndarray[float64] of the same shape.

    Raises:
        ValueError: ``mu <= 0`` or a code out of range.
    """
    ...

# =============================================================================
# TUNING, OCTAVES AND KEYS (librosa-compatible)
# =============================================================================

def hz_to_octs(frequencies: npt.ArrayLike, *, tuning: float = 0.0, bins_per_octave: int = 12) -> Any:
    """Frequencies (Hz) to fractional octaves, like ``librosa.hz_to_octs`` (A0 = 27.5 Hz is 0, A4 = 4.0).

    Returns:
        float for scalar input, otherwise numpy.ndarray[float64] of the same shape.
    """
    ...

def octs_to_hz(octs: npt.ArrayLike, *, tuning: float = 0.0, bins_per_octave: int = 12) -> Any:
    """Fractional octaves to Hz, like ``librosa.octs_to_hz``.

    Returns:
        float for scalar input, otherwise numpy.ndarray[float64] of the same shape.
    """
    ...

def a4_to_tuning(A4: npt.ArrayLike, *, bins_per_octave: int = 12) -> Any:
    """A4 reference frequency to a tuning deviation in fractions of a bin, like
    ``librosa.A4_to_tuning``.

    Returns:
        float for scalar input, otherwise numpy.ndarray[float64] of the same shape.
    """
    ...

def tuning_to_a4(tuning: npt.ArrayLike, *, bins_per_octave: int = 12) -> Any:
    """Tuning deviation to the A4 reference frequency, like ``librosa.tuning_to_A4``.

    Returns:
        float for scalar input, otherwise numpy.ndarray[float64] of the same shape.
    """
    ...

def cqt_frequencies(n_bins: int, *, fmin: float, bins_per_octave: int = 12, tuning: float = 0.0) -> NDArray[np.float64]:
    """Centre frequencies of a constant-Q filterbank, like ``librosa.cqt_frequencies``.

    Returns:
        numpy.ndarray[float64], shape ``(n_bins,)``.
    """
    ...

def key_to_notes(key: str, *, unicode: bool = True, natural: bool = False) -> list[str]:
    """The 12 pitch-class names (from C) spelled for a key, like ``librosa.key_to_notes``.

    Raises:
        ValueError: Unparsable key.
    """
    ...

def key_to_degrees(key: str) -> NDArray[np.int64]:
    """Pitch classes (0 = C) of a key's 7 scale degrees, like ``librosa.key_to_degrees``.

    Returns:
        numpy.ndarray[int64], shape ``(7,)``.
    """
    ...
