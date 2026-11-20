"""
Unit conversions (Hz, mel, MIDI, notes, octaves, frames, samples, time, dB), signal
comparison metrics and tuning helpers, with the ``generation`` and ``detection``
submodules.
"""

from . import detection as detection, generation as generation
from .. import AudioSamples
from _typeshed import Incomplete

def a4_to_tuning(A4: "float | ArrayLike", *, bins_per_octave: int = ...) -> "float | NDArray[np.float64]":
    """
    Converts an A4 reference frequency to a tuning deviation in fractions of a
    bin.

    Args:

        A4 (float | array_like): Reference frequency of A4 in Hz.
        bins_per_octave (int): Default 12.

    Returns:

        float | numpy.ndarray[float64]: tuning, same shape as the input.
    """

def align_signals(reference: AudioSamples, signal: AudioSamples) -> tuple[AudioSamples, int]:
    """
    Aligns two audio signals by estimating a time offset that maximizes correlation.

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

def amplitude_to_db(amplitude: float) -> float:
    """
    Converts a linear amplitude ratio into a logarithmic decibel representation.

    Non-positive amplitudes are mapped to -80 dB.

    Args:

        amplitude: The input amplitude ratio.

    Returns:

        The amplitude expressed in decibels.

    Example:

        >>> from audio_samples.utils import amplitude_to_db
        >>> db = amplitude_to_db(0.5)  # Returns ~-6.02 dB
    """

def blocks_to_frames(blocks: "float | ArrayLike", *, block_length: int) -> "int | NDArray[np.int64]":
    """
    Converts block indices to frame indices: ``blocks * block_length``.

    Args:

        blocks (int | ``array_like[int]``): Block index or indices.
        ``block_length`` (int): Frames per block.

    Returns:

        int | numpy.ndarray[int64]: Frame indices, same shape as the input.
    """

def blocks_to_samples(blocks: "float | ArrayLike", *, block_length: int, hop_length: int) -> "int | NDArray[np.int64]":
    """
    Converts block indices to sample indices: ``blocks * block_length * hop_length``.

    Args:

        blocks (int | ``array_like[int]``): Block index or indices.
        ``block_length`` (int): Frames per block.
        ``hop_length`` (int): Samples between frames.

    Returns:

        int | numpy.ndarray[int64]: Sample indices, same shape as the input.
    """

def blocks_to_time(blocks: "float | ArrayLike", *, block_length: int, hop_length: int, sample_rate: float) -> "float | NDArray[np.float64]":
    """
    Converts block indices to the time in seconds at which each block starts.

    Args:

        blocks (int | ``array_like[int]``): Block index or indices.
        ``block_length`` (int): Frames per block.
        ``hop_length`` (int): Samples between frames.
        ``sample_rate`` (float): Sampling rate in Hertz.

    Returns:

        float | numpy.ndarray[float64]: Times in seconds, same shape as the input.
    """

def cents_to_ratio(cents: float) -> float:
    """
    Converts a pitch deviation expressed in cents into a frequency ratio.

    Args:

        cents: A pitch deviation in cents.

    Returns:

        A multiplicative frequency ratio.

    Example:

        >>> from audio_samples.utils import cents_to_ratio
        >>> ratio = cents_to_ratio(1200.0)  # Returns 2.0 (one octave)
        >>> ratio = cents_to_ratio(100.0)  # Returns ~1.059 (one semitone)
    """

def correlation(a: AudioSamples, b: AudioSamples) -> float:
    """
    Computes the Pearson correlation coefficient between two audio signals.

    This function measures linear similarity between two signals on a per-sample basis.
    Returns a scalar correlation coefficient in the range [-1, 1].

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

    Example:

        >>> import numpy as np
        >>> from audio_samples import AudioSamples
        >>> from audio_samples.utils import correlation
        >>> audio1 = AudioSamples.new_mono(np.sin(np.arange(100) / 5.0), sample_rate=44100)
        >>> correlation(audio1, audio1)  # identical signals
        1.0
    """

def correlation_per_channel(a: AudioSamples, b: AudioSamples) -> "NDArray[np.float64]":
    """
    Computes the Pearson correlation coefficient for each channel independently.

    Mirrors :func:`correlation` but returns one value per channel instead of an average.
    For mono input the returned array has a single element.

    Args:

        a: The first audio signal.
        b: The second audio signal.

    Returns:

        A 1-D NumPy array of per-channel correlation coefficients, in channel order.

    Raises:

        ValueError: If the signals have different dimensions or channel configurations.
    """

def cqt_frequencies(n_bins: int, *, fmin: float, bins_per_octave: int = ..., tuning: float = 0.0) -> "NDArray[np.float64]":
    """
    Centre frequencies of a constant-Q filterbank.

    Args:

        n_bins (int): Number of bins.
        fmin (float): Frequency of the first bin (Hz, before tuning). Required.
        bins_per_octave (int): Default 12.
        tuning (float): Tuning deviation in fractions of a bin. Default 0.

    Returns:

        numpy.ndarray[float64]: shape ``(n_bins,)``.
    """

def db_to_amplitude(db: float) -> float:
    """
    Converts a decibel value into a linear amplitude ratio.

    Args:

        db: A value expressed in decibels.

    Returns:

        The corresponding linear amplitude ratio.

    Example:

        >>> from audio_samples.utils import db_to_amplitude
        >>> amp = db_to_amplitude(-6.0)  # Returns ~0.501
    """

def db_to_power(db: float) -> float:
    """
    Converts a decibel value into a linear power ratio.

    Args:

        db: A value expressed in decibels.

    Returns:

        The corresponding linear power ratio.

    Example:

        >>> from audio_samples.utils import db_to_power
        >>> power = db_to_power(-3.0)  # Returns ~0.501
    """

def fft_frequencies(n_fft: int, sample_rate: float) -> "NDArray[np.float64]":
    """
    Generates the positive-frequency bin centres for a real-valued FFT.

    Args:

        n_fft: The FFT size in samples.
        sample_rate: The sampling rate in Hertz.

    Returns:

        A numpy array of frequency bin centres in Hertz, from DC to Nyquist.
        Length is n_fft/2 + 1.

    Example:

        >>> from audio_samples.utils import fft_frequencies
        >>> freqs = fft_frequencies(1024, 44100.0)
        >>> freqs.shape  # (513,)
    """

def frames_to_samples(frames: "float | ArrayLike", *, hop_length: int, n_fft: int | None = None) -> "int | NDArray[np.int64]":
    """
    Converts frame indices to sample indices.

    ``frames * hop_length + n_fft // 2`` (the offset only when ``n_fft`` is given).

    Args:

        frames (int | array_like[int]): Frame index or indices.
        hop_length (int): Samples between frames. Required.
        n_fft (Optional[int]): Optional frame-length offset.

    Returns:

        int | numpy.ndarray[int64]: Sample indices, same shape as the input.
    """

def frames_to_time(frames: "float | ArrayLike", sample_rate: float, hop_size: int, *, n_fft: int | None = None) -> "float | NDArray[np.float64]":
    """
    Converts frame indices to time in seconds.

    ``frames * hop_size / sample_rate``, plus ``(n_fft // 2) / sample_rate`` when
    ``n_fft`` is given (the frame centre of an uncentred STFT). Negative frames
    are allowed.

    Args:

        frames (int | array_like[int]): Frame index or indices.
        sample_rate (float): Sampling rate in Hertz.
        hop_size (int): Samples between frames.
        n_fft (Optional[int]): Optional frame-length offset. Default None.

    Returns:

        float | numpy.ndarray[float64]: Times in seconds, same shape as the input.

    Example:

        >>> from audio_samples.utils import frames_to_time
        >>> frames_to_time(100, 44100.0, 512)
        >>> frames_to_time([0, 1, 2], 22050.0, 512, n_fft=2048)
    """

def frequency_to_note(freq_hz: "float | ArrayLike") -> "tuple[str, float] | tuple[list[str], NDArray[np.float64]]":
    """
    Converts a frequency expressed in Hertz into the nearest musical note and cents offset.

    Args:

        freq_hz: The input frequency in Hertz (positive), or an array-like of
            frequencies.

    Returns:

        A tuple (note_name, cents) where note_name is the nearest pitch and cents is
        the signed deviation from that pitch. For an array-like: a list of names
        (flattened) and a float64 array of cents of the input's shape.

    Raises:

        ValueError: If the frequency is non-positive.

    Example:

        >>> from audio_samples.utils import frequency_to_note
        >>> note, cents = frequency_to_note(440.0)  # Returns ("A4", 0.0)
        >>> note, cents = frequency_to_note(445.0)  # Returns ("A4", ~19.56)
    """

def hz_to_mel(freq_hz: "float | ArrayLike", *, htk: bool = True) -> "float | NDArray[np.float64]":
    """
    Converts frequencies in Hertz to mels (scalar or array).

    **The default scale is HTK** (``2595 * log10(1 + f / 700)``). Pass
    ``htk=False`` for the Slaney scale (linear below 1 kHz, logarithmic above),
    as used by ``spectral.mel_filterbank``.

    Args:

        freq_hz (float | array_like): Frequency or frequencies in Hertz.
        htk (bool): Use the HTK scale (default True) or the Slaney scale (False).

    Returns:

        float | numpy.ndarray[float64]: Mels, same shape as the input.

    Example:

        >>> from audio_samples.utils import hz_to_mel
        >>> hz_to_mel(1000.0)              # HTK: ~999.99
        >>> hz_to_mel(1000.0, htk=False)   # Slaney: 15.0
    """

def hz_to_midi(freq_hz: "float | ArrayLike") -> "float | NDArray[np.float64]":
    """
    Converts frequencies in Hertz to (fractional) MIDI note numbers:
    ``12 * log2(f / 440) + 69``.

    Args:

        freq_hz (float | array_like): Frequency or frequencies in Hertz.

    Returns:

        float | numpy.ndarray[float64]: MIDI numbers, same shape as the input.

    Example:

        >>> from audio_samples.utils import hz_to_midi
        >>> hz_to_midi(440.0)  # 69.0
    """

def hz_to_octs(frequencies: "float | ArrayLike", *, tuning: float = 0.0, bins_per_octave: int = ...) -> "float | NDArray[np.float64]":
    """
    Converts frequencies (Hz) to fractional octave numbers. Octave 0 is A0
    (27.5 Hz for A440 tuning), so A4 is octave 4.0.

    Args:

        frequencies (float | array_like): Frequencies in Hz.
        tuning (float): Tuning deviation from A440 in fractions of a bin. Default 0.
        bins_per_octave (int): Bins per octave the tuning refers to. Default 12.

    Returns:

        float | numpy.ndarray[float64]: octave numbers, same shape as the input.
    """

def key_to_degrees(key: str) -> "NDArray[np.int64]":
    """
    Pitch classes (0 = C) of the 7 scale degrees of a key.

    Args:

        key (str): A key such as ``"C:maj"`` or ``"A:min"``.

    Returns:

        numpy.ndarray[int64]: shape ``(7,)``.

    Raises:

        ValueError: For an unparsable key.
    """

def key_to_notes(key: str, *, unicode: bool = True, natural: bool = False) -> list:
    """
    Spells the 12 pitch classes for a key.

    Args:

        key (str): A key such as ``"C:maj"``, ``"F#:min"``, ``"Bb:dorian"``.
        unicode (bool): Use ``♯`` / ``♭`` (default True) instead of ``#`` / ``b``.
        natural (bool): Mark naturals with ``♮`` (or ``n``). Default False.

    Returns:

        list[str]: 12 note names, starting from C.

    Raises:

        ValueError: For an unparsable key.
    """

def linspace(start: float, end: float, num: int) -> "NDArray[np.float64]":
    """
    Generates a linearly spaced sequence of values between two endpoints.

    Args:

        start: The starting value of the sequence.
        end: The final value of the sequence.
        num: The number of points to generate.

    Returns:

        A numpy array of linearly interpolated values.

    Example:

        >>> from audio_samples.utils import linspace
        >>> vals = linspace(0.0, 1.0, 11)  # [0.0, 0.1, 0.2, ..., 1.0]
    """

def log_spectral_distance(a: AudioSamples, b: AudioSamples) -> float:
    """
    Computes the log-spectral distance (LSD) between two audio signals.

    LSD measures the average difference between the log-power spectra of the two signals.
    Lower values indicate greater spectral similarity; identical signals yield 0.0.

    Args:

        a: The first audio signal.
        b: The second audio signal.

    Returns:

        The log-spectral distance as a non-negative scalar value.

    Raises:

        ValueError: If the signals have different dimensions or channel configurations.
    """

def mel_frequencies(n_mels: int, fmin: float, fmax: float, *, htk: bool = True) -> "NDArray[np.float64]":
    """
    Frequencies uniformly spaced on the mel scale.

    **The default scale is HTK**; pass ``htk=False`` for the Slaney scale.

    Args:

        n_mels (int): Number of frequencies.
        fmin (float): Lowest frequency in Hertz.
        fmax (float): Highest frequency in Hertz.
        htk (bool): HTK scale (default True) or Slaney (False).

    Returns:

        numpy.ndarray[float64]: shape ``(n_mels,)``, in Hertz.

    Example:

        >>> from audio_samples.utils import mel_frequencies
        >>> mel_frequencies(128, 0.0, 11025.0, htk=False)  # 128 Slaney-spaced frequencies
    """

def mel_scale(n_mels: int, fmin: float, fmax: float) -> "NDArray[np.float64]":
    """
    Generates a sequence of frequencies whose spacing is uniform in the mel domain.

    Args:

        n_mels: The number of frequency points to generate.
        fmin: The lower bound of the frequency range in Hertz.
        fmax: The upper bound of the frequency range in Hertz.

    Returns:

        A numpy array of frequencies in Hertz, spaced uniformly in mel scale.

    Example:

        >>> from audio_samples.utils import mel_scale
        >>> freqs = mel_scale(10, 100.0, 8000.0)
    """

def mel_to_hz(mel: "float | ArrayLike", *, htk: bool = True) -> "float | NDArray[np.float64]":
    """
    Converts mels to frequencies in Hertz (scalar or array).

    **The default scale is HTK**; ``htk=False`` gives the Slaney scale.

    Args:

        mel (float | array_like): Mel value(s).
        htk (bool): HTK scale (default True) or Slaney (False).

    Returns:

        float | numpy.ndarray[float64]: Frequencies in Hertz, same shape as the input.

    Example:

        >>> from audio_samples.utils import mel_to_hz
        >>> mel_to_hz(15.0, htk=False)  # 1000.0
    """

def midi_to_hz(midi_note: "float | ArrayLike") -> "float | NDArray[np.float64]":
    """
    Converts MIDI note numbers (fractional allowed) to Hertz:
    ``440 * 2 ** ((m - 69) / 12)``.

    Args:

        midi_note (float | array_like): MIDI number(s).

    Returns:

        float | numpy.ndarray[float64]: Frequencies in Hertz, same shape as the input.

    Example:

        >>> from audio_samples.utils import midi_to_hz
        >>> midi_to_hz(69.0)  # 440.0
    """

def midi_to_note(midi_note: "int | ArrayLike") -> "str | list[str]":
    """
    Converts a MIDI note number into a scientific pitch notation string.

    Always uses sharp notation for accidentals.

    Args:

        midi_note (int | array_like[int]): A MIDI note number (0-127), or an
            array-like of them.

    Returns:

        str: A note name in scientific pitch notation (e.g., "A4", "C#3"); a
        list of names (flattened) for an array-like.

    Raises:

        ValueError: If the MIDI note is outside the valid range 0-127.

    Example:

        >>> from audio_samples.utils import midi_to_note
        >>> midi_to_note(69)  # "A4"
        >>> midi_to_note(61)  # "C#4"
    """

def ms_to_samples(ms: float, sample_rate: float) -> int:
    """
    Converts time in milliseconds to sample count.

    Args:

        ms: Time in milliseconds.
        sample_rate: The sampling rate in Hertz.

    Returns:

        The corresponding number of samples.

    Example:

        >>> from audio_samples.utils import ms_to_samples
        >>> samples = ms_to_samples(1000.0, 44100.0)  # Returns 44100
    """

def mse(a: AudioSamples, b: AudioSamples) -> float:
    """
    Computes the mean squared error (MSE) between two audio signals.

    This function measures the average squared per-sample difference between two signals.
    Lower values indicate higher similarity, with 0.0 indicating identical signals.

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

    Example:

        >>> import numpy as np
        >>> from audio_samples import AudioSamples
        >>> from audio_samples.utils import mse
        >>> audio1 = AudioSamples.new_mono(np.sin(np.arange(100) / 5.0), sample_rate=44100)
        >>> mse(audio1, audio1)  # identical signals
        0.0
    """

def mse_per_channel(a: AudioSamples, b: AudioSamples) -> "NDArray[np.float64]":
    """
    Computes the mean squared error (MSE) for each channel independently.

    Mirrors :func:`mse` but returns one value per channel instead of an average.
    For mono input the returned array has a single element.

    Args:

        a: The first audio signal.
        b: The second audio signal.

    Returns:

        A 1-D NumPy array of per-channel MSE values, in channel order.

    Raises:

        ValueError: If the signals have different dimensions or channel configurations.
    """

def mu_law_decode(codes: "int | ArrayLike", *, mu: float = 255.0) -> "float | NDArray[np.float64]":
    """
    Map integer μ-law codes back to amplitudes.

    Not an exact inverse of ``mu_law_encode`` (quantisation error up to ~0.043
    for ``mu=255``).

    Args:

        codes (int | array_like[int]): μ-law codes, any shape.
        mu (float): The ``mu`` used for encoding. Default 255.

    Returns:

        float | numpy.ndarray[float64]: Amplitudes in ``[-1, 1]``, same shape.

    Raises:

        ValueError: If ``mu <= 0`` or a code is out of range.
    """

def mu_law_encode(x: "float | ArrayLike", *, mu: float = 255.0) -> "int | NDArray[np.int64]":
    """
    Quantise samples to integer μ-law codes.

    Companding ``sign(x) * log1p(mu * |x|) / log1p(mu)`` followed by binning
    into ``floor(1 + mu)`` levels; for ``mu=255`` the codes are the integers
    ``-128..=127``. For float-in/float-out companding of a whole signal use
    ``AudioSamples.mu_compress``.

    Args:

        x (float | array_like): Samples in ``[-1, 1]``, any shape.
        mu (float): Compression parameter (> 0). Default 255.

    Returns:

        int | numpy.ndarray[int64]: Codes, same shape as ``x``.

    Raises:

        ValueError: If ``mu <= 0`` or any sample is outside ``[-1, 1]``.

    Example:

        >>> from audio_samples import mu_law_encode
        >>> mu_law_encode([-1.0, 0.0, 1.0])  # array([-128, 0, 127])
    """

def note_to_frequency(note_name: "str | Sequence[str]") -> "float | NDArray[np.float64]":
    """
    Converts scientific pitch notation into frequencies expressed in Hertz.

    Args:

        note_name: A note name in scientific pitch notation (e.g., "A4", "C#3"),
            or a sequence of names.

    Returns:

        The corresponding frequency in Hertz; a float64 array for a sequence.

    Raises:

        ValueError: If the note name is invalid or out of MIDI range.

    Example:

        >>> from audio_samples.utils import note_to_frequency
        >>> freq = note_to_frequency("A4")  # Returns 440.0
    """

def note_to_midi(note_name: "str | Sequence[str]") -> "int | NDArray[np.int64]":
    """
    Parses scientific pitch notation and converts it into MIDI note numbers.

    Accepts upper- or lowercase note letters, any run of the accidentals
    `#` `♯` `b` `!` `♭` `𝄪` `𝄫` `♮`, a signed octave (`C-1` = 0) and an optional cents suffix
    (`A4+50`, rounded half to even). The octave is required.

    Args:

        note_name: A note name in scientific pitch notation (e.g., "A4", "C#3",
            "Bb2"), or a sequence of names.

    Returns:

        The corresponding MIDI note number (0-127); an int64 array for a sequence.

    Raises:

        ValueError: If the note name is invalid or out of MIDI range.

    Example:

        >>> from audio_samples.utils import note_to_midi
        >>> midi = note_to_midi("A4")  # Returns 69
        >>> midi = note_to_midi("C#4")  # Returns 61
    """

def octs_to_hz(octs: "float | ArrayLike", *, tuning: float = 0.0, bins_per_octave: int = ...) -> "float | NDArray[np.float64]":
    """
    Converts fractional octave numbers to frequencies (Hz), the inverse of
    ``hz_to_octs``.

    Args:

        octs (float | array_like): Octave numbers.
        tuning (float): Tuning deviation in fractions of a bin. Default 0.
        bins_per_octave (int): Default 12.

    Returns:

        float | numpy.ndarray[float64]: frequencies in Hz, same shape as the input.
    """

def power_to_db(power: float) -> float:
    """
    Converts a linear power ratio into a logarithmic decibel representation.

    Non-positive power values are mapped to -80 dB.

    Args:

        power: The input power ratio.

    Returns:

        The power expressed in decibels.

    Example:

        >>> from audio_samples.utils import power_to_db
        >>> db = power_to_db(0.5)  # Returns ~-3.01 dB
    """

def psnr(reference: AudioSamples, test: AudioSamples) -> float:
    """
    Computes the peak signal-to-noise ratio (PSNR) in decibels between two signals.

    PSNR relates the peak amplitude of the reference signal to the mean squared error
    between the two signals. Higher values indicate greater similarity; identical signals
    yield positive infinity.

    Args:

        reference: The reference (clean) signal.
        test: The signal to compare against the reference.

    Returns:

        The PSNR in decibels. Returns positive infinity for identical signals.

    Raises:

        ValueError: If the signals have different dimensions or channel configurations.
    """

def ratio_to_cents(ratio: float) -> float:
    """
    Converts a frequency ratio into a pitch deviation expressed in cents.

    Args:

        ratio: A frequency ratio. Must be positive.

    Returns:

        The corresponding pitch deviation in cents.

    Example:

        >>> from audio_samples.utils import ratio_to_cents
        >>> cents = ratio_to_cents(2.0)  # Returns 1200.0 (one octave)
        >>> cents = ratio_to_cents(1.5)  # Returns ~702.0 (perfect fifth)
    """

def samples_like(X: "int | ArrayLike", *, hop_length: int, n_fft: int | None = None, axis: int = ...) -> "NDArray[np.int64]":
    """
    Sample index of each frame of a feature matrix.

    Args:

        X (int | array_like): Number of frames, or an array whose ``axis`` is time.
        hop_length (int): Samples between frames. Required.
        n_fft (Optional[int]): Optional frame-length offset.
        axis (int): The time axis of ``X``. Default -1.

    Returns:

        numpy.ndarray[int64]: shape ``(n_frames,)``.
    """

def samples_to_frames(samples: "float | ArrayLike", *, hop_length: int, n_fft: int | None = None) -> "int | NDArray[np.int64]":
    """
    Converts sample indices to frame indices.

    ``floor((samples - n_fft // 2) / hop_length)`` (the offset only when ``n_fft``
    is given).

    Args:

        samples (int | array_like[int]): Sample index or indices (may be negative).
        hop_length (int): Samples between frames. Required.
        n_fft (Optional[int]): Optional frame-length offset.

    Returns:

        int | numpy.ndarray[int64]: Frame indices, same shape as the input.
    """

def samples_to_time(samples: "int | ArrayLike", sample_rate: float) -> "float | NDArray[np.float64]":
    """
    Converts sample indices into time offsets expressed in seconds.

    Args:

        samples: A sample index, or an array-like of non-negative integer indices.
        sample_rate: The sampling rate in Hertz.

    Returns:

        float for a scalar index, otherwise a float64 array of the input's shape.

    Raises:

        ValueError: For negative or non-integral indices in an array.

    Example:

        >>> from audio_samples.utils import samples_to_time
        >>> t = samples_to_time(44100, 44100.0)  # Returns 1.0
        >>> ts = samples_to_time([0, 22050], 44100.0)  # array([0. , 0.5])
    """

def seconds_to_samples(time_seconds: float, sample_rate: float) -> int:
    """
    Converts a time offset expressed in seconds into the nearest sample index.

    Rounds to the nearest sample and clamps negatives to 0.
    ``time_to_samples`` truncates toward zero instead and keeps negative values.

    Args:

        time_seconds: The time offset in seconds.
        sample_rate: The sampling rate in Hertz.

    Returns:

        The nearest sample index.

    Example:

        >>> from audio_samples.utils import seconds_to_samples
        >>> samples = seconds_to_samples(1.0, 44100.0)  # Returns 44100
    """

def segmental_snr(signal: AudioSamples, noise: AudioSamples, segment_len: int = 256) -> float:
    """
    Computes the segmental signal-to-noise ratio (segmental SNR) in decibels.

    The signals are divided into fixed-length segments; the SNR of each segment is computed,
    clamped to a perceptually motivated range, and then averaged. This often correlates better
    with perceived quality than a single global SNR.

    Args:

        signal: The signal component.
        noise: The noise component.
        segment_len: Number of samples per segment (must be positive, default: 256).

    Returns:

        The mean of the per-segment SNR values in decibels.

    Raises:

        ValueError: If the signals have different dimensions/channels, or `segment_len` is 0.
    """

def snr(signal: AudioSamples, noise: AudioSamples) -> float:
    """
    Computes the signal-to-noise ratio (SNR) in decibels between a signal and noise.

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

    Example:

        >>> import numpy as np
        >>> from audio_samples import AudioSamples
        >>> from audio_samples.utils import snr
        >>> signal = AudioSamples.new_mono(np.array([1.0, 2.0, 3.0]), sample_rate=44100)
        >>> noise = AudioSamples.new_mono(np.array([0.1, 0.2, 0.1]), sample_rate=44100)
        >>> round(snr(signal, noise), 2)  # 10*log10(mean(s**2) / mean(n**2))
        23.68
    """

def snr_per_channel(signal: AudioSamples, noise: AudioSamples) -> "NDArray[np.float64]":
    """
    Computes the signal-to-noise ratio (SNR) in decibels for each channel independently.

    Mirrors :func:`snr` but returns one value per channel instead of aggregating across all
    channels. A channel with zero noise power yields positive infinity. For mono input the
    returned array has a single element.

    Args:

        signal: The signal component.
        noise: The noise component.

    Returns:

        A 1-D NumPy array of per-channel SNR values in decibels, in channel order.

    Raises:

        ValueError: If the signals have different dimensions or channel configurations.
    """

def time_to_frames(time_seconds: "float | ArrayLike", sample_rate: float, hop_size: int) -> "int | NDArray[np.int64]":
    """
    Converts times in seconds to the **nearest** frame index (rounding).

    ``round(t * sample_rate / hop_size)``, with negative results clamped to 0.
    ``time_to_frames_floor`` returns the frame *containing* the time instead
    (floor division, negatives kept, optional ``n_fft`` offset).

    Args:

        time_seconds (float | array_like): Time(s) in seconds.
        sample_rate (float): Sampling rate in Hertz.
        hop_size (int): Samples between frames.

    Returns:

        int | numpy.ndarray[int64]: Nearest frame indices, same shape as the input.

    Example:

        >>> from audio_samples.utils import time_to_frames
        >>> time_to_frames(1.0, 44100.0, 512)  # 86
    """

def time_to_frames_floor(times: "float | ArrayLike", *, sample_rate: float | None = None, hop_length: int, n_fft: int | None = None, sr: float | None = None) -> "int | NDArray[np.int64]":
    """
    Converts times in seconds to the frame **containing** each time.

    ``samples_to_frames(time_to_samples(t, sample_rate), hop_length, n_fft)``:
    truncation to samples, then floor division (negatives kept). Compare
    ``time_to_frames``, which rounds to the nearest frame.

    Args:

        times (float | array_like): Time(s) in seconds.
        sample_rate (Optional[float]): Sampling rate in Hertz. Give this or
            ``sr`` (exactly one).
        sr (Optional[float]): Short name for ``sample_rate``.
        hop_length (int): Samples between frames. Required.
        n_fft (Optional[int]): Optional frame-length offset.

    Returns:

        int | numpy.ndarray[int64]: Frame indices, same shape as the input.

    Example:

        >>> from audio_samples.utils import time_to_frames_floor
        >>> time_to_frames_floor([0.0, 0.1, 0.2], sample_rate=22050, hop_length=512)  # [0, 4, 8]
    """

def time_to_samples(times: "float | ArrayLike", *, sample_rate: float | None = None, sr: float | None = None) -> "int | NDArray[np.int64]":
    """
    Converts times in seconds to sample indices.

    ``int(t * sample_rate)`` truncated toward zero (negative times stay negative).
    Differs from ``seconds_to_samples``, which rounds.

    Args:

        times (float | array_like): Time(s) in seconds.
        sample_rate (Optional[float]): Sampling rate in Hertz. Give this or
            ``sr`` (exactly one).
        sr (Optional[float]): Short name for ``sample_rate``.

    Returns:

        int | numpy.ndarray[int64]: Sample indices, same shape as the input.

    Example:

        >>> from audio_samples.utils import time_to_samples
        >>> time_to_samples([0.0, 0.5, -0.5], sample_rate=100.0)  # [0, 50, -50]
    """

def times_like(X: "int | ArrayLike", *, sample_rate: float | None = None, hop_length: int, n_fft: int | None = None, axis: int = ..., sr: float | None = None) -> "NDArray[np.float64]":
    """
    Time (seconds) of each frame of a feature matrix.

    Args:

        X (int | array_like): Number of frames, or an array whose ``axis`` is time.
        sample_rate (Optional[float]): Sampling rate in Hertz. Give this or
            ``sr`` (exactly one).
        sr (Optional[float]): Short name for ``sample_rate``.
        hop_length (int): Samples between frames. Required.
        n_fft (Optional[int]): Optional frame-length offset.
        axis (int): The time axis of ``X``. Default -1.

    Returns:

        numpy.ndarray[float64]: shape ``(n_frames,)``.
    """

def tuning_to_a4(tuning: "float | ArrayLike", *, bins_per_octave: int = ...) -> "float | NDArray[np.float64]":
    """
    Converts a tuning deviation (fractions of a bin) to the A4 reference
    frequency.

    Args:

        tuning (float | array_like): Tuning deviation.
        bins_per_octave (int): Default 12.

    Returns:

        float | numpy.ndarray[float64]: A4 in Hz, same shape as the input.
    """

def __getattr__(name: str) -> Incomplete: ...
