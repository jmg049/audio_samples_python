"""Audio mathematical utilities and canonical unit conversions.

This module provides frequency conversions, amplitude/power conversions,
time/frame conversions, musical theory functions, and spectral utilities.
"""

import numpy as np
from numpy.typing import NDArray

# =============================================================================
# FREQUENCY CONVERSIONS
# =============================================================================

def hz_to_mel(freq_hz: float) -> float:
    """Convert frequency in Hertz to mel perceptual scale.

    Args:
        freq_hz: The input frequency in Hertz.

    Returns:
        The corresponding value on the mel scale.
    """
    ...

def mel_to_hz(mel: float) -> float:
    """Convert mel perceptual scale value to Hertz.

    Args:
        mel: A value on the mel scale.

    Returns:
        The corresponding frequency in Hertz.
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

def hz_to_midi(freq_hz: float) -> float:
    """Convert frequency in Hertz to MIDI note value.

    Args:
        freq_hz: The input frequency in Hertz. Must be positive.

    Returns:
        A MIDI note value (continuous, not quantized to integer semitones).
    """
    ...

def midi_to_hz(midi_note: float) -> float:
    """Convert MIDI note value to frequency in Hertz.

    Args:
        midi_note: A MIDI note value (fractional values are permitted).

    Returns:
        The corresponding frequency in Hertz.
    """
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

def frames_to_time(frames: int, sample_rate: float, hop_size: int) -> float:
    """Convert frame index to time offset in seconds.

    Args:
        frames: The frame index to convert.
        sample_rate: The sampling rate in Hertz.
        hop_size: The number of samples between successive frames.

    Returns:
        The corresponding time offset in seconds.
    """
    ...

def time_to_frames(time_seconds: float, sample_rate: float, hop_size: int) -> int:
    """Convert time offset in seconds to nearest frame index.

    Args:
        time_seconds: The time offset in seconds.
        sample_rate: The sampling rate in Hertz.
        hop_size: The number of samples between successive frames.

    Returns:
        The nearest frame index.
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
    """Convert time offset in seconds to nearest sample index.

    Args:
        time_seconds: The time offset in seconds.
        sample_rate: The sampling rate in Hertz.

    Returns:
        The nearest sample index.
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
    """Convert MIDI note number to scientific pitch notation string.

    Always uses sharp notation for accidentals.

    Args:
        midi_note: A MIDI note number (0-127).

    Returns:
        A note name in scientific pitch notation (e.g., "A4", "C#3").

    Raises:
        ValueError: If the MIDI note is outside the valid range 0-127.
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

def mel_frequencies(n_mels: int, fmin: float, fmax: float) -> NDArray[np.float64]:
    """Generate frequencies spaced uniformly in the mel perceptual domain.

    Args:
        n_mels: The number of frequency points to generate.
        fmin: The lower bound of the frequency range in Hertz.
        fmax: The upper bound of the frequency range in Hertz.

    Returns:
        A numpy array of frequencies in Hertz, spaced uniformly in mel scale.
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
