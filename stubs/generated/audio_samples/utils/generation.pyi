"""
Signal generators (tones, noise, chirps, band-limited, modulated and compound
waveforms) and the example clips.
"""

from .. import AudioSamples
from _typeshed import Incomplete
from collections.abc import Sequence
from typing import Any, Final

__all__ = ["EXAMPLE_NOISY_SNR_DB", "EXAMPLE_TEMPO_BPM", "am_signal", "brown_noise", "chirp", "compound_tone", "cosine_wave", "example_clip", "example_clip_info", "example_clip_names", "example_speech_f0", "exponential_bursts", "exponential_chirp", "fm_signal", "impulse", "pink_noise", "sawtooth_wave", "sawtooth_wave_bandlimited", "silence", "sine_wave", "square_wave", "square_wave_bandlimited", "stereo_chirp", "stereo_silence", "stereo_sine_wave", "triangle_wave", "triangle_wave_bandlimited", "white_noise"]

EXAMPLE_NOISY_SNR_DB: Final[float]
EXAMPLE_TEMPO_BPM: Final[float]

def am_signal(carrier_freq: float, modulator_freq: float, modulation_depth: float, duration_secs: float, sample_rate: int = 44100, amplitude: float = 1.0, dtype: Any | None = None) -> AudioSamples:
    """
    Generate an amplitude-modulated (AM) signal.

    A carrier signal modulated by a low-frequency envelope.

    Args:

        carrier_freq: Frequency of the carrier signal in Hz
        modulator_freq: Frequency of the modulating envelope in Hz
        modulation_depth: Depth of modulation (0.0 = none, 1.0 = full)
        duration_secs: Duration of the signal in seconds
        sample_rate: Sample rate in samples per second (default: 44100)
        amplitude: Overall amplitude (default: 1.0)
        dtype: Output sample type as a NumPy dtype, scalar type (``np.float64``)
            or string (``"float64"``). Default: float32

    Returns:

        AudioSamples: Generated AM signal audio data
    """

def brown_noise(duration_secs: float, sample_rate: int = 44100, step: float = 0.01, amplitude: float = 1.0, dtype: Any | None = None, seed: int | None = None) -> AudioSamples:
    """
    Generate brown noise (Brownian/red noise) audio signal.

    Args:

        duration_secs: Duration of the signal in seconds
        sample_rate: Sample rate in samples per second (default: 44100)
        step: Step size for the random walk (default: 0.01)
        amplitude: Peak amplitude of the noise (default: 1.0)
        dtype: Output sample type as a NumPy dtype, scalar type (``np.float64``)
            or string (``"float64"``). Default: float32

    Returns:

        AudioSamples: Generated brown noise audio data
    """

def chirp(f0: float, f1: float, duration_secs: float, sample_rate: int = 44100, amplitude: float = 1.0, dtype: Any | None = None) -> AudioSamples:
    """
    Generate a frequency chirp (sweep) audio signal.

    Args:

        f0: Starting frequency in Hz
        f1: Ending frequency in Hz
        duration_secs: Duration of the signal in seconds
        sample_rate: Sample rate in samples per second (default: 44100)
        amplitude: Peak amplitude of the wave (default: 1.0)
        dtype: Output sample type as a NumPy dtype, scalar type (``np.float64``)
            or string (``"float64"``). Default: float32

    Returns:

        AudioSamples: Generated chirp audio data
    """

def compound_tone(components: Sequence[tuple[float, float]], duration_secs: float, sample_rate: int = 44100, dtype: Any | None = None) -> AudioSamples:
    """
    Generate a compound tone from multiple frequency components.

    Useful for creating signals with harmonics or multiple simultaneous tones. Each
    component is given as a ``(frequency_hz, amplitude)`` tuple.

    Args:

        components: List of ``(frequency_hz, amplitude)`` tuples. Must be non-empty.
        duration_secs: Duration of the signal in seconds
        sample_rate: Sample rate in samples per second (default: 44100)
        dtype: Output sample type as a NumPy dtype, scalar type (``np.float64``)
            or string (``"float64"``). Default: float32

    Returns:

        AudioSamples: Generated compound tone audio data

    Raises:

        ValueError: If `components` is empty or `duration_secs` is not positive.
    """

def cosine_wave(frequency: float, duration_secs: float, sample_rate: int = 44100, amplitude: float = 1.0, dtype: Any | None = None) -> AudioSamples:
    """
    Generate a cosine wave audio signal.

    Args:

        frequency: Frequency of the cosine wave in Hz
        duration_secs: Duration of the signal in seconds
        sample_rate: Sample rate in samples per second (default: 44100)
        amplitude: Peak amplitude of the wave (default: 1.0)
        dtype: Output sample type as a NumPy dtype, scalar type (``np.float64``)
            or string (``"float64"``). Default: float32

    Returns:

        AudioSamples: Generated cosine wave audio data
    """

def example_clip(name: str, duration_secs: float = 8.0, sample_rate: int = 44100, dtype: Any | None = None) -> AudioSamples:
    """
    Render a named example clip of realistic audio.

    The clip is synthesised on each call: nothing is stored or downloaded. The same arguments
    give the same samples on any platform, and a shorter clip is a prefix of a longer one.

    ======================  ==================================================================
    Name                    Contents
    ======================  ==================================================================
    ``"chords"``            C, G, Am, F in equal temperament (A4 = 440 Hz), one chord per four
                            beats at ``EXAMPLE_TEMPO_BPM``, soft attacks
    ``"drum_loop"``         Kick on beats 1 and 3, snare on 2 and 4, hi-hat on every eighth
                            note at ``EXAMPLE_TEMPO_BPM``, first kick at time zero
    ``"chords_and_drums"``  ``0.5 * chords + 0.5 * drum_loop``
    ``"speech"``            Voiced speech-like signal with formants, following
                            ``example_speech_f0``; a 2.4 s phrase every 3 s, pauses between
    ``"noisy_recording"``   ``speech`` with low-passed noise and 50 Hz mains hum, at
                            ``EXAMPLE_NOISY_SNR_DB`` over the voiced parts
    ======================  ==================================================================

    Args:

        name: The clip, one of the names above (see ``example_clip_names``)
        duration_secs: Duration in seconds (default: 8.0); the pattern repeats for as long as
            requested
        sample_rate: Sample rate in samples per second (default: 44100)
        dtype: Output sample type as a NumPy dtype, scalar type (``np.float64``)
            or string (``"float64"``). Default: float32

    Returns:

        AudioSamples: Mono audio of the given sample type at the given sample rate

    Raises:

        ValueError: Unknown name, or a duration that gives no samples
    """

def example_clip_info(name: str) -> str:
    """
    What an example clip contains.

    Args:

        name: The clip, one of ``example_clip_names()``

    Returns:

        str: One sentence on the clip's known contents

    Raises:

        ``ValueError``: Unknown name
    """

def example_clip_names() -> list[str]:
    """
    The names ``example_clip`` accepts.

    Returns:

        list[str]: The clip names
    """

def example_speech_f0(times: Sequence[float]) -> Incomplete:
    """
    The F0 of the ``"speech"`` and ``"noisy_recording"`` clips at the given times.

    Args:

        times: Times in seconds from the start of the clip

    Returns:

        numpy.ndarray: F0 in Hz as 1-D float64, NaN during pauses
    """

def exponential_bursts(burst_rate: float, decay_rate: float, duration_secs: float, sample_rate: int = 44100, amplitude: float = 1.0, dtype: Any | None = None) -> AudioSamples:
    """
    Generate a signal with periodic exponential-decay bursts.

    Creates percussive-like transients useful for testing onset detection.

    Args:

        burst_rate: Number of bursts per second
        decay_rate: Exponential decay rate (higher = faster decay)
        duration_secs: Duration of the signal in seconds
        sample_rate: Sample rate in samples per second (default: 44100)
        amplitude: Peak amplitude of bursts (default: 1.0)
        dtype: Output sample type as a NumPy dtype, scalar type (``np.float64``)
            or string (``"float64"``). Default: float32

    Returns:

        AudioSamples: Generated burst signal audio data
    """

def exponential_chirp(start_freq: float, end_freq: float, duration_secs: float, sample_rate: int = 44100, amplitude: float = 1.0, dtype: Any | None = None) -> AudioSamples:
    """
    Generate an exponential (logarithmic) frequency chirp (sweep) audio signal.

    The instantaneous frequency sweeps geometrically from `start_freq` to `end_freq`.
    Both frequencies must be strictly positive.

    Args:

        start_freq: Starting frequency in Hz (must be > 0)
        end_freq: Ending frequency in Hz (must be > 0)
        duration_secs: Duration of the signal in seconds
        sample_rate: Sample rate in samples per second (default: 44100)
        amplitude: Peak amplitude of the wave (default: 1.0)
        dtype: Output sample type as a NumPy dtype, scalar type (``np.float64``)
            or string (``"float64"``). Default: float32

    Returns:

        AudioSamples: Generated exponential chirp audio data
    """

def fm_signal(carrier_freq: float, modulator_freq: float, modulation_index: float, duration_secs: float, sample_rate: int = 44100, amplitude: float = 1.0, dtype: Any | None = None) -> AudioSamples:
    """
    Generate a frequency-modulated (FM) signal.

    A sinusoidal carrier whose instantaneous phase is modulated by a sinusoidal
    modulator. With `modulation_index == 0` this reduces to a pure sine at the
    carrier frequency.

    Args:

        carrier_freq: Carrier frequency in Hz
        modulator_freq: Modulator frequency in Hz
        modulation_index: Modulation index (peak phase deviation in radians)
        duration_secs: Duration of the signal in seconds
        sample_rate: Sample rate in samples per second (default: 44100)
        amplitude: Peak amplitude (default: 1.0)
        dtype: Output sample type as a NumPy dtype, scalar type (``np.float64``)
            or string (``"float64"``). Default: float32

    Returns:

        AudioSamples: Generated FM signal audio data
    """

def impulse(duration_secs: float, sample_rate: int = 44100, amplitude: float = 1.0, position: float = 0.5, dtype: Any | None = None) -> AudioSamples:
    """
    Generate an impulse (delta function) audio signal.

    Args:

        duration_secs: Duration of the signal in seconds
        sample_rate: Sample rate in samples per second (default: 44100)
        amplitude: Peak amplitude of the impulse (default: 1.0)
        position: Position of the impulse as fraction of duration (default: 0.5)
        dtype: Output sample type as a NumPy dtype, scalar type (``np.float64``)
            or string (``"float64"``). Default: float32

    Returns:

        AudioSamples: Generated impulse audio data
    """

def pink_noise(duration_secs: float, sample_rate: int = 44100, amplitude: float = 1.0, dtype: Any | None = None, seed: int | None = None) -> AudioSamples:
    """
    Generate pink noise audio signal.

    Args:

        duration_secs: Duration of the signal in seconds
        sample_rate: Sample rate in samples per second (default: 44100)
        amplitude: Peak amplitude of the noise (default: 1.0)
        dtype: Output sample type as a NumPy dtype, scalar type (``np.float64``)
            or string (``"float64"``). Default: float32

    Returns:

        AudioSamples: Generated pink noise audio data
    """

def sawtooth_wave(frequency: float, duration_secs: float, sample_rate: int = 44100, amplitude: float = 1.0, dtype: Any | None = None) -> AudioSamples:
    """
    Generate a sawtooth wave audio signal.

    Args:

        frequency: Frequency of the sawtooth wave in Hz
        duration_secs: Duration of the signal in seconds
        sample_rate: Sample rate in samples per second (default: 44100)
        amplitude: Peak amplitude of the wave (default: 1.0)
        dtype: Output sample type as a NumPy dtype, scalar type (``np.float64``)
            or string (``"float64"``). Default: float32

    Returns:

        AudioSamples: Generated sawtooth wave audio data
    """

def sawtooth_wave_bandlimited(frequency: float, duration_secs: float, sample_rate: int = 44100, amplitude: float = 1.0, dtype: Any | None = None) -> AudioSamples:
    """
    Generate a band-limited (alias-free) sawtooth wave via additive synthesis.

    Sums only the harmonics that lie strictly below the Nyquist frequency, producing
    an exactly band-limited signal with no aliasing.

    Args:

        frequency: Fundamental frequency of the sawtooth wave in Hz
        duration_secs: Duration of the signal in seconds
        sample_rate: Sample rate in samples per second (default: 44100)
        amplitude: Peak amplitude of the underlying ideal sawtooth wave (default: 1.0)
        dtype: Output sample type as a NumPy dtype, scalar type (``np.float64``)
            or string (``"float64"``). Default: float32

    Returns:

        AudioSamples: Generated band-limited sawtooth wave audio data
    """

def silence(duration_secs: float, sample_rate: int = 44100, dtype: Any | None = None) -> AudioSamples:
    """
    Generate silence (zero amplitude) audio signal.

    Args:

        duration_secs: Duration of the signal in seconds
        sample_rate: Sample rate in samples per second (default: 44100)
        dtype: Output sample type as a NumPy dtype, scalar type (``np.float64``)
            or string (``"float64"``). Default: float32

    Returns:

        AudioSamples: Generated silence audio data
    """

def sine_wave(frequency: float, duration_secs: float, sample_rate: int = 44100, amplitude: float = 1.0, dtype: Any | None = None) -> AudioSamples:
    """
    Generate a sine wave audio signal.

    Args:

        frequency: Frequency of the sine wave in Hz
        duration_secs: Duration of the signal in seconds
        sample_rate: Sample rate in samples per second (default: 44100)
        amplitude: Peak amplitude of the wave (default: 1.0)
        dtype: Output sample type as a NumPy dtype, scalar type (``np.float64``)
            or string (``"float64"``). Default: float32

    Returns:

        AudioSamples: Generated sine wave audio data
    """

def square_wave(frequency: float, duration_secs: float, sample_rate: int = 44100, amplitude: float = 1.0, dtype: Any | None = None) -> AudioSamples:
    """
    Generate a square wave audio signal.

    Args:

        frequency: Frequency of the square wave in Hz
        duration_secs: Duration of the signal in seconds
        sample_rate: Sample rate in samples per second (default: 44100)
        amplitude: Peak amplitude of the wave (default: 1.0)
        dtype: Output sample type as a NumPy dtype, scalar type (``np.float64``)
            or string (``"float64"``). Default: float32

    Returns:

        AudioSamples: Generated square wave audio data
    """

def square_wave_bandlimited(frequency: float, duration_secs: float, sample_rate: int = 44100, amplitude: float = 1.0, dtype: Any | None = None) -> AudioSamples:
    """
    Generate a band-limited (alias-free) square wave via additive synthesis.

    Unlike a hard-clipped square wave, this sums only the odd harmonics that lie
    strictly below the Nyquist frequency, producing an exactly band-limited signal
    with no aliasing.

    Args:

        frequency: Fundamental frequency of the square wave in Hz
        duration_secs: Duration of the signal in seconds
        sample_rate: Sample rate in samples per second (default: 44100)
        amplitude: Peak amplitude of the underlying ideal square wave (default: 1.0)
        dtype: Output sample type as a NumPy dtype, scalar type (``np.float64``)
            or string (``"float64"``). Default: float32

    Returns:

        AudioSamples: Generated band-limited square wave audio data
    """

def stereo_chirp(start_freq: float, end_freq: float, duration_secs: float, sample_rate: int = 44100, amplitude: float = 1.0, dtype: Any | None = None) -> AudioSamples:
    """
    Generate a stereo chirp signal (frequency sweep on both channels).

    Args:

        start_freq: Starting frequency in Hz
        end_freq: Ending frequency in Hz
        duration_secs: Duration of the signal in seconds
        sample_rate: Sample rate in samples per second (default: 44100)
        amplitude: Peak amplitude of the chirp (default: 1.0)
        dtype: Output sample type as a NumPy dtype, scalar type (``np.float64``)
            or string (``"float64"``). Default: float32

    Returns:

        AudioSamples: Generated stereo chirp audio data
    """

def stereo_silence(duration_secs: float, sample_rate: int = 44100, dtype: Any | None = None) -> AudioSamples:
    """
    Generate stereo silence.

    Args:

        duration_secs: Duration of the signal in seconds
        sample_rate: Sample rate in samples per second (default: 44100)
        dtype: Output sample type as a NumPy dtype, scalar type (``np.float64``)
            or string (``"float64"``). Default: float32

    Returns:

        AudioSamples: Stereo silence audio data
    """

def stereo_sine_wave(frequency: float, duration_secs: float, sample_rate: int = 44100, amplitude: float = 1.0, dtype: Any | None = None) -> AudioSamples:
    """
    Generate a stereo sine wave audio signal (same frequency on both channels).

    Args:

        frequency: Frequency of the sine wave in Hz
        duration_secs: Duration of the signal in seconds
        sample_rate: Sample rate in samples per second (default: 44100)
        amplitude: Peak amplitude of the wave (default: 1.0)
        dtype: Output sample type as a NumPy dtype, scalar type (``np.float64``)
            or string (``"float64"``). Default: float32

    Returns:

        AudioSamples: Generated stereo sine wave audio data
    """

def triangle_wave(frequency: float, duration_secs: float, sample_rate: int = 44100, amplitude: float = 1.0, dtype: Any | None = None) -> AudioSamples:
    """
    Generate a triangle wave audio signal.

    Args:

        frequency: Frequency of the triangle wave in Hz
        duration_secs: Duration of the signal in seconds
        sample_rate: Sample rate in samples per second (default: 44100)
        amplitude: Peak amplitude of the wave (default: 1.0)
        dtype: Output sample type as a NumPy dtype, scalar type (``np.float64``)
            or string (``"float64"``). Default: float32

    Returns:

        AudioSamples: Generated triangle wave audio data
    """

def triangle_wave_bandlimited(frequency: float, duration_secs: float, sample_rate: int = 44100, amplitude: float = 1.0, dtype: Any | None = None) -> AudioSamples:
    """
    Generate a band-limited (alias-free) triangle wave via additive synthesis.

    Sums only the odd harmonics that lie strictly below the Nyquist frequency,
    producing an exactly band-limited signal with no aliasing.

    Args:

        frequency: Fundamental frequency of the triangle wave in Hz
        duration_secs: Duration of the signal in seconds
        sample_rate: Sample rate in samples per second (default: 44100)
        amplitude: Peak amplitude of the underlying ideal triangle wave (default: 1.0)
        dtype: Output sample type as a NumPy dtype, scalar type (``np.float64``)
            or string (``"float64"``). Default: float32

    Returns:

        AudioSamples: Generated band-limited triangle wave audio data
    """

def white_noise(duration_secs: float, sample_rate: int = 44100, amplitude: float = 1.0, dtype: Any | None = None, seed: int | None = None) -> AudioSamples:
    """
    Generate white noise audio signal.

    Args:

        duration_secs: Duration of the signal in seconds
        sample_rate: Sample rate in samples per second (default: 44100)
        amplitude: Peak amplitude of the noise (default: 1.0)
        dtype: Output sample type as a NumPy dtype, scalar type (``np.float64``)
            or string (``"float64"``). Default: float32
        seed: Optional seed for reproducible noise generation

    Returns:

        AudioSamples: Generated white noise audio data
    """
