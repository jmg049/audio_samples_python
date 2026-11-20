"""Utility functions for audio signal processing.

This module provides comparison, audio math, and other utility functions.
"""

from typing import Any, Optional, Sequence, Tuple, Union, overload

import numpy as np
import numpy.typing as npt
from numpy.typing import NDArray

from audio_samples import AudioSamples
from . import detection as detection, generation as generation
from audio_samples.generation import (
    am_signal as am_signal,
    brown_noise as brown_noise,
    chirp as chirp,
    compound_tone as compound_tone,
    cosine_wave as cosine_wave,
    exponential_bursts as exponential_bursts,
    exponential_chirp as exponential_chirp,
    fm_signal as fm_signal,
    impulse as impulse,
    pink_noise as pink_noise,
    sawtooth_wave as sawtooth_wave,
    sawtooth_wave_bandlimited as sawtooth_wave_bandlimited,
    silence as silence,
    sine_wave as sine_wave,
    square_wave as square_wave,
    square_wave_bandlimited as square_wave_bandlimited,
    stereo_chirp as stereo_chirp,
    stereo_silence as stereo_silence,
    stereo_sine_wave as stereo_sine_wave,
    triangle_wave as triangle_wave,
    triangle_wave_bandlimited as triangle_wave_bandlimited,
    white_noise as white_noise,
)
from audio_samples.utils.detection import (
    analyze_spectrum_for_cutoff as analyze_spectrum_for_cutoff,
    detect_clipping as detect_clipping,
    detect_dynamic_range as detect_dynamic_range,
    detect_fundamental_autocorrelation as detect_fundamental_autocorrelation,
    detect_fundamental_frequency as detect_fundamental_frequency,
    detect_sample_rate as detect_sample_rate,
    detect_silence_regions as detect_silence_regions,
    estimate_frequency_range as estimate_frequency_range,
    estimate_noise_floor as estimate_noise_floor,
)

# =============================================================================
# AUDIO COMPARISON FUNCTIONS
# =============================================================================

def correlation(a: AudioSamples, b: AudioSamples) -> float:
    """Compute Pearson correlation coefficient between two audio signals.

    This function measures linear similarity between two signals on a per-sample
    basis. Returns a scalar correlation coefficient in the range [-1, 1].

    For mono signals, the correlation is computed directly over the single channel.
    For multi-channel signals, the correlation is computed independently for each
    channel and the results are averaged.

    Args:
        a: The first audio signal.
        b: The second audio signal.

    Returns:
        A scalar correlation coefficient in the range [-1, 1].

    Raises:
        ValueError: If the signals have different dimensions or channel configurations.
        TypeError: If the signals have different data types.

    Example:
        >>> import numpy as np
        >>> from audio_samples import AudioSamples
        >>> from audio_samples.utils import correlation
        >>> audio1 = AudioSamples.new_mono(np.sin(np.arange(100) / 5.0), sample_rate=44100)
        >>> correlation(audio1, audio1)  # identical signals
        1.0
    """
    ...

def mse(a: AudioSamples, b: AudioSamples) -> float:
    """Compute mean squared error (MSE) between two audio signals.

    This function measures the average squared per-sample difference between two
    signals. Lower values indicate higher similarity, with 0.0 indicating identical
    signals.

    For mono signals, the MSE is computed directly over the single channel.
    For multi-channel signals, the MSE is computed independently for each channel
    and the results are averaged.

    Args:
        a: The first audio signal.
        b: The second audio signal.

    Returns:
        The mean squared error as a non-negative scalar value.

    Raises:
        ValueError: If the signals have different dimensions or channel configurations.
        TypeError: If the signals have different data types.

    Example:
        >>> import numpy as np
        >>> from audio_samples import AudioSamples
        >>> from audio_samples.utils import mse
        >>> audio1 = AudioSamples.new_mono(np.sin(np.arange(100) / 5.0), sample_rate=44100)
        >>> mse(audio1, audio1)  # identical signals
        0.0
    """
    ...

def snr(signal: AudioSamples, noise: AudioSamples) -> float:
    """Compute signal-to-noise ratio (SNR) in decibels.

    This function measures the ratio between the average power of a signal and the
    average power of a noise signal, expressed in decibels. Higher values indicate
    greater dominance of the signal relative to noise.

    For mono inputs, power is computed over the single channel.
    For multi-channel inputs, power is computed over all samples across all channels.

    Args:
        signal: The signal component.
        noise: The noise component.

    Returns:
        The signal-to-noise ratio in decibels. Returns positive infinity if noise
        power is zero.

    Raises:
        ValueError: If the signals have different dimensions or channel configurations.
        TypeError: If the signals have different data types.

    Example:
        >>> import numpy as np
        >>> from audio_samples import AudioSamples
        >>> from audio_samples.utils import snr
        >>> signal = AudioSamples.new_mono(np.array([1.0, 2.0, 3.0]), sample_rate=44100)
        >>> noise = AudioSamples.new_mono(np.array([0.1, 0.2, 0.1]), sample_rate=44100)
        >>> round(snr(signal, noise), 2)  # 10*log10(mean(s**2) / mean(n**2))
        23.68
    """
    ...

def align_signals(
    reference: AudioSamples, signal: AudioSamples
) -> Tuple[AudioSamples, int]:
    """Align two audio signals by maximizing correlation.

    This function temporally aligns `signal` to `reference` by searching for a
    non-negative sample offset that maximizes cross-correlation. Returns a shifted
    version of the signal padded at the start, together with the estimated offset.

    For mono signals, alignment is computed directly on the single channel.
    For multi-channel signals, all channels are averaged to a single signal before
    alignment. The estimated offset is then applied uniformly to all channels.

    Only non-negative offsets are considered (signal is shifted forward in time only).
    The maximum offset searched is half of the shorter signal length.

    Args:
        reference: The reference signal that defines the target alignment.
        signal: The signal to be aligned to the reference.

    Returns:
        A tuple (aligned_signal, offset_samples) where aligned_signal is the shifted
        signal and offset_samples is the number of samples by which it was shifted.

    Raises:
        ValueError: If the signals have different channel counts or configurations.
        TypeError: If the signals have different data types.

    Example:
        >>> import numpy as np
        >>> from audio_samples import AudioSamples
        >>> from audio_samples.utils import align_signals
        >>> reference = AudioSamples.new_mono(np.array([0, 0, 0, 0, 1.0, 0, 0, 0]), sample_rate=44100)
        >>> signal = AudioSamples.new_mono(np.array([0, 0, 1.0, 0, 0, 0, 0, 0]), sample_rate=44100)  # 2 samples early
        >>> aligned, offset = align_signals(reference, signal)
        >>> print(f"Offset: {offset} samples")
        Offset: 2 samples
    """
    ...

def psnr(reference: AudioSamples, test: AudioSamples) -> float:
    """Compute the peak signal-to-noise ratio (PSNR) in decibels between two signals.

    PSNR relates the peak amplitude of the reference signal to the mean squared error
    between the two signals. Higher values indicate greater similarity; identical
    signals yield positive infinity.

    Args:
        reference: The reference (clean) signal.
        test: The signal to compare against the reference.

    Returns:
        The PSNR in decibels. Returns positive infinity for identical signals.

    Raises:
        ValueError: If the signals have different dimensions or channel configurations.
        TypeError: If the signals have different data types.
    """
    ...

def segmental_snr(
    signal: AudioSamples, noise: AudioSamples, segment_len: int = 256
) -> float:
    """Compute the segmental signal-to-noise ratio (segmental SNR) in decibels.

    The signals are divided into fixed-length segments; the SNR of each segment is
    computed, clamped to a perceptually motivated range, and then averaged. This often
    correlates better with perceived quality than a single global SNR.

    Args:
        signal: The signal component.
        noise: The noise component.
        segment_len: Number of samples per segment (must be positive, default: 256).

    Returns:
        The mean of the per-segment SNR values in decibels.

    Raises:
        ValueError: If the signals have different dimensions/channels, or `segment_len` is 0.
        TypeError: If the signals have different data types.
    """
    ...

def log_spectral_distance(a: AudioSamples, b: AudioSamples) -> float:
    """Compute the log-spectral distance (LSD) between two audio signals.

    LSD measures the average difference between the log-power spectra of the two
    signals. Lower values indicate greater spectral similarity; identical signals
    yield 0.0.

    Args:
        a: The first audio signal.
        b: The second audio signal.

    Returns:
        The log-spectral distance as a non-negative scalar value.

    Raises:
        ValueError: If the signals have different dimensions or channel configurations.
        TypeError: If the signals have different data types.
    """
    ...

def correlation_per_channel(a: AudioSamples, b: AudioSamples) -> NDArray[np.float64]:
    """Compute the Pearson correlation coefficient for each channel independently.

    Mirrors :func:`correlation` but returns one value per channel instead of an average.
    For mono input the returned array has a single element.

    Args:
        a: The first audio signal.
        b: The second audio signal.

    Returns:
        A 1-D NumPy array of per-channel correlation coefficients, in channel order.

    Raises:
        ValueError: If the signals have different dimensions or channel configurations.
        TypeError: If the signals have different data types.
    """
    ...

def mse_per_channel(a: AudioSamples, b: AudioSamples) -> NDArray[np.float64]:
    """Compute the mean squared error (MSE) for each channel independently.

    Mirrors :func:`mse` but returns one value per channel instead of an average.
    For mono input the returned array has a single element.

    Args:
        a: The first audio signal.
        b: The second audio signal.

    Returns:
        A 1-D NumPy array of per-channel MSE values, in channel order.

    Raises:
        ValueError: If the signals have different dimensions or channel configurations.
        TypeError: If the signals have different data types.
    """
    ...

def snr_per_channel(signal: AudioSamples, noise: AudioSamples) -> NDArray[np.float64]:
    """Compute the signal-to-noise ratio (SNR) in decibels for each channel independently.

    Mirrors :func:`snr` but returns one value per channel instead of aggregating across
    all channels. A channel with zero noise power yields positive infinity. For mono
    input the returned array has a single element.

    Args:
        signal: The signal component.
        noise: The noise component.

    Returns:
        A 1-D NumPy array of per-channel SNR values in decibels, in channel order.

    Raises:
        ValueError: If the signals have different dimensions or channel configurations.
        TypeError: If the signals have different data types.
    """
    ...

# =============================================================================
# FREQUENCY CONVERSIONS
# =============================================================================

@overload
def hz_to_mel(freq_hz: float, *, htk: bool=True) -> float: ...
@overload
def hz_to_mel(freq_hz: npt.ArrayLike, *, htk: bool=True) -> NDArray[np.float64]:
    """Convert frequencies in Hertz to mels (scalar or array).

    **The default is the HTK scale** (``2595 * log10(1 + f / 700)``). Pass
    ``htk=False`` for the Slaney scale (linear below 1 kHz, logarithmic
    above), as used by ``spectral.mel_filterbank``.

    Returns:
        float for scalar input, otherwise numpy.ndarray[float64] of the same shape.
    """
    ...

@overload
def mel_to_hz(mel: float, *, htk: bool=True) -> float: ...
@overload
def mel_to_hz(mel: npt.ArrayLike, *, htk: bool=True) -> NDArray[np.float64]:
    """Convert mels to Hertz (scalar or array). Default HTK; ``htk=False``
    gives the Slaney scale.

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

@overload
def hz_to_midi(freq_hz: float) -> float: ...
@overload
def hz_to_midi(freq_hz: npt.ArrayLike) -> NDArray[np.float64]:
    """Hertz to MIDI note numbers, ``12 * log2(f / 440) + 69``; float for scalar
    input, otherwise numpy.ndarray[float64] of the same shape."""
    ...

@overload
def midi_to_hz(midi_note: float) -> float: ...
@overload
def midi_to_hz(midi_note: npt.ArrayLike) -> NDArray[np.float64]:
    """MIDI note numbers to Hertz, ``440 * 2 ** ((m - 69) / 12)``; float for scalar
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

@overload
def frames_to_time(
    frames: float,
    sample_rate: float,
    hop_size: int,
    *,
    n_fft: Optional[int] = None,
) -> float: ...
@overload
def frames_to_time(
    frames: npt.ArrayLike,
    sample_rate: float,
    hop_size: int,
    *,
    n_fft: Optional[int] = None,
) -> NDArray[np.float64]:
    """Frame indices to seconds:
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

@overload
def time_to_frames(time_seconds: float, sample_rate: float, hop_size: int) -> int: ...
@overload
def time_to_frames(
    time_seconds: npt.ArrayLike,
    sample_rate: float,
    hop_size: int,
) -> NDArray[np.int64]:
    """Times to the **nearest** frame (``round(t * sr / hop)``, negatives clamp to 0).

    ``time_to_frames_floor`` gives the frame containing each time instead
    (floor, negatives kept, ``n_fft`` offset).

    Returns:
        int for scalar input, otherwise numpy.ndarray[int64] of the same shape.
    """
    ...

@overload
def samples_to_time(samples: int, sample_rate: float) -> float: ...
@overload
def samples_to_time(samples: npt.ArrayLike, sample_rate: float) -> NDArray[np.float64]:
    """Convert sample indices to time offsets in seconds.

    Args:
        samples: A sample index, or an array-like of non-negative integer indices.
        sample_rate: The sampling rate in Hertz.

    Returns:
        float for a scalar index, otherwise a float64 array of the input's shape.

    Raises:
        ValueError: For negative or non-integral indices in an array.
    """
    ...

def seconds_to_samples(time_seconds: float, sample_rate: float) -> int:
    """Seconds to the **nearest** sample index (rounds; negatives clamp to 0).

    ``time_to_samples`` truncates toward zero instead.
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

@overload
def note_to_midi(note_name: str) -> int: ...  # type: ignore[overload-overlap]
@overload
def note_to_midi(note_name: Sequence[str]) -> NDArray[np.int64]:
    """Parse scientific pitch notation to MIDI note numbers.

    Accepts upper- or lowercase note letters, any run of
    the accidentals ``#`` ``♯`` ``b`` ``!`` ``♭`` ``𝄪`` ``𝄫`` ``♮``, a signed octave
    (``C-1`` = 0) and an optional cents suffix (``A4+50``, rounded half to
    even). The octave is required.

    Args:
        note_name: A note name in scientific pitch notation (e.g., "A4", "C#3",
            "Bb2"), or a sequence of names.

    Returns:
        The corresponding MIDI note number (0-127); an int64 array for a sequence.

    Raises:
        ValueError: If a note name is invalid or out of MIDI range.
    """
    ...

@overload
def midi_to_note(midi_note: int) -> str: ...
@overload
def midi_to_note(midi_note: npt.ArrayLike) -> list[str]:
    """MIDI numbers (0-127) to note names with sharps, e.g. ``"A4"``, ``"C#3"``.

    A scalar gives a str; an array-like gives a list of names (flattened).

    Raises:
        ValueError: If a ``midi_note`` is outside 0-127 (including negative values).
    """
    ...

@overload
def note_to_frequency(note_name: str) -> float: ...  # type: ignore[overload-overlap]
@overload
def note_to_frequency(note_name: Sequence[str]) -> NDArray[np.float64]:
    """Convert scientific pitch notation to frequencies in Hertz.

    Args:
        note_name: A note name in scientific pitch notation (e.g., "A4", "C#3"),
            or a sequence of names.

    Returns:
        The corresponding frequency in Hertz; a float64 array for a sequence.

    Raises:
        ValueError: If a note name is invalid or out of MIDI range.
    """
    ...

@overload
def frequency_to_note(freq_hz: float) -> tuple[str, float]: ...
@overload
def frequency_to_note(freq_hz: npt.ArrayLike) -> tuple[list[str], NDArray[np.float64]]:
    """Convert frequencies in Hertz to the nearest notes and cents offsets.

    Args:
        freq_hz: The input frequency in Hertz (positive), or an array-like of
            frequencies.

    Returns:
        A tuple (note_name, cents) where note_name is the nearest pitch and
        cents is the signed deviation from that pitch. For an array-like: a
        list of names (flattened) and a float64 array of cents of the input's
        shape.

    Raises:
        ValueError: If a frequency is non-positive.
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

def mel_frequencies(
    n_mels: int,
    fmin: float,
    fmax: float,
    *,
    htk: bool  =  True,
) -> NDArray[np.float64]:
    """``n_mels`` frequencies uniformly spaced in mels.

    **Default HTK**; pass ``htk=False`` for the Slaney scale.

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
# FRAME / SAMPLE / TIME CONVERSIONS (vectorised)
# =============================================================================

@overload
def time_to_samples(
    times: float,
    *,
    sample_rate: Optional[float] = None,
    sr: Optional[float] = None,
) -> int: ...
@overload
def time_to_samples(
    times: npt.ArrayLike,
    *,
    sample_rate: Optional[float] = None,
    sr: Optional[float] = None,
) -> NDArray[np.int64]:
    """``int(t * sample_rate)`` truncated toward zero.

    Give ``sample_rate`` or its short name ``sr`` (exactly one). Neither or
    both is a ``TypeError``.

    Returns:
        int for scalar input, otherwise numpy.ndarray[int64] of the same shape.
    """
    ...

@overload
def samples_to_frames(samples: float, *, hop_length: int, n_fft: Optional[int]=None) -> int: ...
@overload
def samples_to_frames(
    samples: npt.ArrayLike,
    *,
    hop_length: int,
    n_fft: Optional[int] = None,
) -> NDArray[np.int64]:
    """``floor((samples - n_fft // 2) / hop_length)``.

    ``hop_length`` is required.

    Returns:
        int for scalar input, otherwise numpy.ndarray[int64] of the same shape.
    """
    ...

@overload
def blocks_to_frames(blocks: float, *, block_length: int) -> int: ...
@overload
def blocks_to_frames(blocks: npt.ArrayLike, *, block_length: int) -> NDArray[np.int64]:
    """``blocks * block_length``."""
@overload
def blocks_to_samples(blocks: float, *, block_length: int, hop_length: int) -> int: ...
@overload
def blocks_to_samples(blocks: npt.ArrayLike, *, block_length: int, hop_length: int) -> NDArray[np.int64]:
    """``blocks * block_length * hop_length``."""
@overload
def blocks_to_time(blocks: float, *, block_length: int, hop_length: int, sample_rate: float) -> float: ...
@overload
def blocks_to_time(blocks: npt.ArrayLike, *, block_length: int, hop_length: int, sample_rate: float) -> NDArray[np.float64]:
    """``blocks * block_length * hop_length / sample_rate``."""
@overload
def frames_to_samples(frames: float, *, hop_length: int, n_fft: Optional[int]=None) -> int: ...
@overload
def frames_to_samples(
    frames: npt.ArrayLike,
    *,
    hop_length: int,
    n_fft: Optional[int] = None,
) -> NDArray[np.int64]:
    """``frames * hop_length + n_fft // 2``.

    ``hop_length`` is required.

    Returns:
        int for scalar input, otherwise numpy.ndarray[int64] of the same shape.
    """
    ...

@overload
def time_to_frames_floor(
    times: float,
    *,
    sample_rate: Optional[float] = None,
    hop_length: int,
    n_fft: Optional[int] = None,
    sr: Optional[float] = None,
) -> int: ...
@overload
def time_to_frames_floor(
    times: npt.ArrayLike,
    *,
    sample_rate: Optional[float] = None,
    hop_length: int,
    n_fft: Optional[int] = None,
    sr: Optional[float] = None,
) -> NDArray[np.int64]:
    """The frame containing each time (truncate to samples, then
    floor-divide; negatives kept).

    Give ``sample_rate`` or its short name ``sr`` (exactly one).
    ``hop_length`` is required.

    Returns:
        int for scalar input, otherwise numpy.ndarray[int64] of the same shape.
    """
    ...

def times_like(
    X: Union[int, npt.ArrayLike],
    *,
    sample_rate: Optional[float] = None,
    hop_length: int,
    n_fft: Optional[int] = None,
    axis: int = -1,
    sr: Optional[float] = None,
) -> NDArray[np.float64]:
    """Frame times in seconds for ``X`` frames (an int, or an array whose
    ``axis`` is time). Shape ``(n_frames,)``.

    Give ``sample_rate`` or its short name ``sr`` (exactly one).
    ``hop_length`` is required."""
    ...

def samples_like(
    X: Union[int, npt.ArrayLike], *, hop_length: int, n_fft: Optional[int] = None, axis: int = -1
) -> NDArray[np.int64]:
    """Frame sample indices for ``X`` frames (``axis`` is the time axis of an
    array ``X``). Shape ``(n_frames,)``. ``hop_length`` is required."""
    ...

# =============================================================================
# MU-LAW CODES
# =============================================================================

@overload
def mu_law_encode(x: float, *, mu: float = 255.0) -> int: ...
@overload
def mu_law_encode(x: npt.ArrayLike, *, mu: float = 255.0) -> NDArray[np.int64]:
    """Mu-law compression quantised to integer codes (``-128..=127`` for
    ``mu=255``).

    Args:
        x: Samples in ``[-1, 1]``, any shape.

    Returns:
        int for scalar input, otherwise numpy.ndarray[int64] of the same shape.

    Raises:
        ValueError: ``mu <= 0`` or a sample outside ``[-1, 1]``.
    """
    ...

@overload
def mu_law_decode(codes: int, *, mu: float = 255.0) -> float: ...
@overload
def mu_law_decode(codes: npt.ArrayLike, *, mu: float = 255.0) -> NDArray[np.float64]:
    """Amplitudes from integer mu-law codes (the inverse of ``mu_law_encode``).

    Returns:
        float for scalar input, otherwise numpy.ndarray[float64] of the same shape.

    Raises:
        ValueError: ``mu <= 0`` or a code out of range.
    """
    ...

# =============================================================================
# TUNING, OCTAVES AND KEYS
# =============================================================================

@overload
def hz_to_octs(frequencies: float, *, tuning: float=0.0, bins_per_octave: int=12) -> float: ...
@overload
def hz_to_octs(
    frequencies: npt.ArrayLike,
    *,
    tuning: float = 0.0,
    bins_per_octave: int = 12,
) -> NDArray[np.float64]:
    """Frequencies (Hz) to fractional octaves (A0 = 27.5 Hz is 0, A4 = 4.0).

    Returns:
        float for scalar input, otherwise numpy.ndarray[float64] of the same shape.
    """
    ...

@overload
def octs_to_hz(octs: float, *, tuning: float=0.0, bins_per_octave: int=12) -> float: ...
@overload
def octs_to_hz(
    octs: npt.ArrayLike,
    *,
    tuning: float = 0.0,
    bins_per_octave: int = 12,
) -> NDArray[np.float64]:
    """Fractional octaves to Hz (the inverse of ``hz_to_octs``).

    Returns:
        float for scalar input, otherwise numpy.ndarray[float64] of the same shape.
    """
    ...

@overload
def a4_to_tuning(A4: float, *, bins_per_octave: int=12) -> float: ...
@overload
def a4_to_tuning(A4: npt.ArrayLike, *, bins_per_octave: int=12) -> NDArray[np.float64]:
    """A4 reference frequency to a tuning deviation in fractions of a bin.

    Returns:
        float for scalar input, otherwise numpy.ndarray[float64] of the same shape.
    """
    ...

@overload
def tuning_to_a4(tuning: float, *, bins_per_octave: int=12) -> float: ...
@overload
def tuning_to_a4(tuning: npt.ArrayLike, *, bins_per_octave: int=12) -> NDArray[np.float64]:
    """Tuning deviation in fractions of a bin to the A4 reference frequency.

    Returns:
        float for scalar input, otherwise numpy.ndarray[float64] of the same shape.
    """
    ...

def cqt_frequencies(
    n_bins: int,
    *,
    fmin: float,
    bins_per_octave: int  =  12,
    tuning: float  =  0.0,
) -> NDArray[np.float64]:
    """Centre frequencies of a constant-Q filterbank: ``fmin * 2 ** (k / bins_per_octave)``, tuned.

    Returns:
        numpy.ndarray[float64], shape ``(n_bins,)``.
    """
    ...

def key_to_notes(key: str, *, unicode: bool = True, natural: bool = False) -> list[str]:
    """The 12 pitch-class names (from C) spelled for a key.

    Raises:
        ValueError: Unparsable key.
    """
    ...

def key_to_degrees(key: str) -> NDArray[np.int64]:
    """Pitch classes (0 = C) of a key's 7 scale degrees.

    Returns:
        numpy.ndarray[int64], shape ``(7,)``.
    """
    ...

__all__ = [
    "a4_to_tuning",
    "align_signals",
    "am_signal",
    "amplitude_to_db",
    "analyze_spectrum_for_cutoff",
    "brown_noise",
    "cents_to_ratio",
    "chirp",
    "compound_tone",
    "correlation",
    "correlation_per_channel",
    "cosine_wave",
    "cqt_frequencies",
    "db_to_amplitude",
    "db_to_power",
    "detect_clipping",
    "detect_dynamic_range",
    "detect_fundamental_autocorrelation",
    "detect_fundamental_frequency",
    "detect_sample_rate",
    "detect_silence_regions",
    "detection",
    "estimate_frequency_range",
    "estimate_noise_floor",
    "exponential_bursts",
    "exponential_chirp",
    "fft_frequencies",
    "blocks_to_frames",
    "blocks_to_samples",
    "blocks_to_time",
    "fm_signal",
    "frames_to_samples",
    "frames_to_time",
    "frequency_to_note",
    "generation",
    "hz_to_mel",
    "hz_to_midi",
    "hz_to_octs",
    "impulse",
    "key_to_degrees",
    "key_to_notes",
    "linspace",
    "log_spectral_distance",
    "mel_frequencies",
    "mel_scale",
    "mel_to_hz",
    "midi_to_hz",
    "midi_to_note",
    "ms_to_samples",
    "mse",
    "mse_per_channel",
    "mu_law_decode",
    "mu_law_encode",
    "note_to_frequency",
    "note_to_midi",
    "octs_to_hz",
    "pink_noise",
    "power_to_db",
    "psnr",
    "ratio_to_cents",
    "samples_like",
    "samples_to_frames",
    "samples_to_time",
    "sawtooth_wave",
    "sawtooth_wave_bandlimited",
    "seconds_to_samples",
    "segmental_snr",
    "silence",
    "sine_wave",
    "snr",
    "snr_per_channel",
    "square_wave",
    "square_wave_bandlimited",
    "stereo_chirp",
    "stereo_silence",
    "stereo_sine_wave",
    "time_to_frames",
    "time_to_frames_floor",
    "time_to_samples",
    "times_like",
    "triangle_wave",
    "triangle_wave_bandlimited",
    "tuning_to_a4",
    "white_noise",
]
