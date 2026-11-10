"""
Feature, tonal, constant-Q and pitch functions over NumPy arrays: spectral descriptors of a spectrogram, pcen, frequency weightings, chroma filterbanks and tuning (chroma_filterbank, pitch_tuning, estimate_tuning_from_spectrogram, normalize), constant-Q filters (cq_to_chroma, wavelet), chroma_cens / tonnetz from a chromagram, yin / pyin on arrays and the harmonic helpers (salience, interp_harmonics, f0_harmonics). float32 / complex64 arrays are processed in float32 and give float32 output; any other input is converted to float64.
"""

from _typeshed import Incomplete
from collections.abc import Callable
from numpy.typing import ArrayLike
from typing import Any, Literal

__all__ = ["a_weighting", "b_weighting", "c_weighting", "chroma_cens_from_chroma", "chroma_filterbank", "chroma_stft_from_spectrogram", "cq_to_chroma", "d_weighting", "estimate_tuning_from_signal", "estimate_tuning_from_spectrogram", "f0_harmonics", "frequency_weighting", "interp_harmonics", "multi_frequency_weighting", "normalize", "pcen", "perceptual_weighting", "piptrack_from_spectrogram", "pitch_tuning", "poly_features_from_spectrogram", "pyin", "rms_from_spectrogram", "salience", "spectral_bandwidth_from_spectrogram", "spectral_centroid_from_spectrogram", "spectral_contrast_from_spectrogram", "spectral_flatness_from_spectrogram", "spectral_rolloff_from_spectrogram", "tonnetz_from_chroma", "wavelet", "wavelet_lengths", "yin", "z_weighting"]

def a_weighting(frequencies: Any, *, min_db: float | None = ...) -> Any:
    """
    The A-weighting of a set of frequencies in dB.

    Args:

        frequencies (float | array_like): Frequencies in Hz.
        min_db (Optional[float]): Lower clip in dB (default -80); None disables it.

    Returns:

        float | numpy.ndarray[float64]: dB weights, same shape as ``frequencies``
        (a float for scalar input).
    """

def b_weighting(frequencies: Any, *, min_db: float | None = ...) -> Any:
    """
    The B-weighting of a set of frequencies in dB.

    Args:

        frequencies (float | array_like): Frequencies in Hz.
        min_db (Optional[float]): Lower clip in dB (default -80); None disables it.

    Returns:

        float | numpy.ndarray[float64]: dB weights, same shape as ``frequencies``
        (a float for scalar input).
    """

def c_weighting(frequencies: Any, *, min_db: float | None = ...) -> Any:
    """
    The C-weighting of a set of frequencies in dB.

    Args:

        frequencies (float | array_like): Frequencies in Hz.
        min_db (Optional[float]): Lower clip in dB (default -80); None disables it.

    Returns:

        float | numpy.ndarray[float64]: dB weights, same shape as ``frequencies``
        (a float for scalar input).
    """

def chroma_cens_from_chroma(chroma: "ArrayLike", *, norm: float | None = ..., win_len_smooth: int | None = ..., smoothing_window: str | tuple[str, float] | float | ArrayLike = ...) -> "NDArray[Any]":
    """
    Chroma Energy Normalized Statistics (CENS) from a chromagram: the
    quantisation, smoothing and normalisation steps of ``AudioSamples.chroma_cens``.

    Args:

        chroma (array_like): Chromagram, ``(n_chroma, T)`` or ``(channels, n_chroma, T)``.
        norm (Optional[float]): Final per-frame normalisation. Default 2.
        win_len_smooth (Optional[int]): Smoothing length in frames; None disables.
            Default 41.
        smoothing_window: Smoothing window. Default ``"hann"``.

    Returns:

        numpy.ndarray[float32 | float64]: same shape as ``chroma``.

    Raises:

        ValueError: For invalid parameters; RuntimeError for non-finite input.
    """

def chroma_filterbank(*, sr: float, n_fft: int, n_chroma: int = ..., tuning: float = 0.0, ctroct: float = 5.0, octwidth: float | None = ..., norm: float | None = ..., base_c: bool = True) -> "NDArray[np.float64]":
    """
    The chroma filterbank (float64).

    Args:

        sr (float): Sample rate (Hz).
        n_fft (int): FFT length.
        n_chroma (int): Number of chroma bins. Default 12.
        tuning (float): Tuning deviation from A440 in fractions of a chroma bin.
        ctroct (float): Centre of the octave-dominance window (octaves above A0).
            Default 5.0.
        octwidth (Optional[float]): Half-width of the octave weighting; None = flat.
            Default 2.0.
        norm (Optional[float]): Column normalisation (p-norm, ``numpy.inf``), None
            to disable. Default 2.
        base_c (bool): Row 0 is C (True, default) or A.

    Returns:

        numpy.ndarray[float64]: shape ``(n_chroma, 1 + n_fft // 2)``.

    Raises:

        ValueError: For invalid parameters.
    """

def chroma_stft_from_spectrogram(S: "ArrayLike", *, sr: float, norm: float | None = ..., n_fft: int | None = None, tuning: float | None = ..., n_chroma: int = ..., ctroct: float = 5.0, octwidth: float | None = ..., base_c: bool = True) -> "NDArray[Any]":
    """
    Chromagram of a precomputed spectrogram.

    Args:

        S (array_like): Power spectrogram (a magnitude spectrogram gives
            "energy" chroma), shape ``(1 + n_fft // 2, T)`` or
            ``(channels, 1 + n_fft // 2, T)``.
        sr (float): Sample rate (Hz). Required.
        norm (Optional[float]): Per-frame normalisation (default ``numpy.inf``).
        n_fft (Optional[int]): FFT length; None infers ``2 * (F - 1)``.
        tuning (Optional[float]): Tuning in fractions of a chroma bin; None
            estimates it from ``S``.
        n_chroma (int): Number of chroma bins. Default 12.
        ctroct, octwidth, base_c: Filterbank parameters (see ``chroma_filterbank``).

    Returns:

        numpy.ndarray[float32 | float64]: shape ``(n_chroma, T)`` or ``(channels, n_chroma, T)``.

    Raises:

        ValueError: For invalid parameters or a bin count that does not match ``n_fft``.
    """

def cq_to_chroma(n_input: int, *, bins_per_octave: int = ..., n_chroma: int = ..., fmin: float | None = None, window: Any | None = None, base_c: bool = True) -> "NDArray[np.float64]":
    """
    The CQT-to-chroma folding matrix (float64).

    Args:

        n_input (int): Number of CQT bins.
        bins_per_octave (int): CQT bins per octave (a multiple of ``n_chroma``). Default 12.
        n_chroma (int): Number of chroma bins. Default 12.
        fmin (Optional[float]): Frequency of CQT bin 0 (Hz); None = C1.
        window (Optional[``array_like``]): Smooths each row by convolution with these weights.
        base_c (bool): Chroma row 0 is C (default) or A.

    Returns:

        numpy.ndarray[float64]: shape ``(n_chroma, n_input)`` of zeros and ones.

    Raises:

        ``ValueError``: For invalid parameters.
    """

def d_weighting(frequencies: Any, *, min_db: float | None = ...) -> Any:
    """
    The D-weighting of a set of frequencies in dB.

    Args:

        frequencies (float | array_like): Frequencies in Hz.
        min_db (Optional[float]): Lower clip in dB (default -80); None disables it.

    Returns:

        float | numpy.ndarray[float64]: dB weights, same shape as ``frequencies``
        (a float for scalar input).
    """

def estimate_tuning_from_signal(y: "ArrayLike", *, sr: float, n_fft: int = ..., resolution: float = 0.01, bins_per_octave: int = ..., hop_length: int | None = None, win_length: int | None = None, window: str | tuple[str, float] | float | ArrayLike = ..., center: bool = True, pad_mode: str = "constant", fmin: float = 150.0, fmax: float = 4000.0, threshold: float = 0.1, ref: Any | None = None) -> float:
    """
    Tuning estimate from a signal.

    Args:

        y (array_like): Signal, ``(n,)`` or ``(channels, n)`` (channels are pooled).
        sr (float): Sample rate (Hz). Required.
        n_fft (int): FFT length. Default 2048.
        resolution (float): Resolution in fractions of a bin. Default 0.01.
        bins_per_octave (int): Bins per octave of the grid. Default 12.
        hop_length, win_length, window, center, pad_mode: STFT parameters of
            the spectrogram passed to ``piptrack``.
        fmin, fmax, threshold, ref: ``piptrack`` parameters.

    Returns:

        float: tuning in ``[-0.5, 0.5)`` fractions of a bin.

    Raises:

        ValueError: For invalid parameters or a signal too short for an uncentred STFT.
    """

def estimate_tuning_from_spectrogram(S: "ArrayLike", *, sr: float, n_fft: int | None = None, resolution: float = 0.01, bins_per_octave: int = ..., fmin: float = 150.0, fmax: float = 4000.0, threshold: float = 0.1, ref: Any | None = None) -> float:
    """
    Tuning estimate from a precomputed spectrogram.

    Args:

        S (array_like): Magnitude or power spectrogram, ``(F, T)`` or
            ``(channels, F, T)`` (all channels are pooled).
        sr (float): Sample rate (Hz). Required.
        n_fft (Optional[int]): FFT length; None infers ``2 * (F - 1)``.
        resolution (float): Resolution in fractions of a bin. Default 0.01.
        bins_per_octave (int): Bins per octave of the grid. Default 12.
        fmin, fmax, threshold, ref: ``piptrack`` parameters.

    Returns:

        float: tuning in ``[-0.5, 0.5)`` fractions of a bin.

    Raises:

        ValueError: For invalid parameters.
    """

def f0_harmonics(x: "ArrayLike", *, f0: "ArrayLike", freqs: "ArrayLike", harmonics: "ArrayLike", kind: str = "linear", fill_value: float = 0.0, axis: int = ...) -> "NDArray[Any]":
    """
    Energy at the harmonics of a time-varying f0.

    Args:

        x (array_like): Energy, ``(F, T)`` or ``(channels, F, T)``.
        f0 (array_like): Fundamental per frame, ``(T,)`` or ``(channels, T)``;
            NaN frames get ``fill_value``.
        freqs (array_like): 1-D bin frequencies or an array shaped like ``x``.
        harmonics (array_like): Harmonic multipliers.
        kind (str): The interpolant, as for ``arrays.interp1d``. Default ``"linear"``.
        fill_value (float): Value for out-of-range targets. Default 0.
        axis (int): The frequency axis. Default -2.

    Returns:

        numpy.ndarray[float32 | float64]: shape ``(len(harmonics), T)`` or
        ``(channels, len(harmonics), T)``.

    Raises:

        ValueError: For shape mismatches or unsupported options.
    """

def frequency_weighting(frequencies: "float | ArrayLike", *, kind: str = "A", min_db: float | None = None) -> "float | NDArray[np.float64]":
    """
    A frequency-weighting curve chosen by name.

    Args:

        frequencies (float | array_like): Frequencies in Hz.
        kind (str): ``"A"`` (default), ``"B"``, ``"C"``, ``"D"`` or ``"Z"``.
        min_db (Optional[float]): Lower clip in dB. Default None (no clip).

    Returns:

        float | numpy.ndarray[float64]: dB weights, same shape as ``frequencies``.

    Raises:

        ValueError: For an unknown ``kind``.
    """

def interp_harmonics(x: "ArrayLike", *, freqs: "ArrayLike", harmonics: "ArrayLike", kind: str = "linear", fill_value: float = 0.0, axis: int = ...) -> "NDArray[Any]":
    """
    Energy of a spectrogram interpolated at the harmonics of every bin.

    Args:

        x (array_like): Energy, ``(F, T)`` or ``(channels, F, T)``.
        freqs (array_like): 1-D bin frequencies or an array shaped like ``x``.
        harmonics (array_like): Harmonic multipliers.
        kind (str | int): Interpolation kind, as ``arrays.interp1d``. Default ``"linear"``.
        fill_value (float): Value outside the frequency range. Default 0.
        axis (int): The frequency axis. Default -2.

    Returns:

        numpy.ndarray[float32 | float64]: shape ``(len(harmonics), F, T)`` or
        ``(channels, len(harmonics), F, T)``.

    Raises:

        ValueError: For shape mismatches or unsupported options.
    """

def multi_frequency_weighting(frequencies: "ArrayLike", *, kinds: Any | None = None, min_db: float | None = None) -> "NDArray[np.float64]":
    """
    Several weighting curves stacked, one row per name in ``kinds``.

    Args:

        frequencies (array_like): Frequencies in Hz.
        kinds (str | Sequence[str]): Curve names, e.g. ``"ZAC"`` (default).
        min_db (Optional[float]): Lower clip in dB. Default None (no clip).

    Returns:

        numpy.ndarray[float64]: shape ``(len(kinds), *frequencies.shape)``.

    Raises:

        ValueError: For an unknown curve name.
    """

def normalize(S: "ArrayLike", *, norm: float | None = ..., axis: int | None = ..., threshold: float | None = None, fill: bool | None = None) -> "NDArray[Any]":
    """
    Scales every slice of an array to unit norm.

    Args:

        S (array_like): Array of any shape (finite values).
        norm (Optional[float]): ``numpy.inf`` (max-abs, default), ``-numpy.inf``
            (min-abs), 0 (count of non-zeros), ``p > 0`` (Lp), or None (no-op).
        axis (Optional[int]): Axis along which norms are computed (default 0);
            None normalises the whole array.
        threshold (Optional[float]): Slices with a norm below this are "small"
            (default: the smallest positive normal float64).
        fill (Optional[bool]): Small slices: None leaves them, False zeroes them,
            True fills them with the unit-norm constant.

    Returns:

        numpy.ndarray[float32 | float64]: same shape as ``S``.

    Raises:

        ValueError: For an invalid norm/threshold/axis, ``fill=True`` with
            ``norm=0``, or non-finite input.
    """

def pcen(S: "ArrayLike", *, sr: float, hop_length: int, gain: float = 0.98, bias: float = 2.0, power: float = 0.5, time_constant: float = 0.4, eps: float = 1e-6, b: float | None = None, max_size: int = ..., ref: Any | None = None, axis: int = ..., max_axis: int | None = None, zi: Any | None = None, return_zf: bool = False) -> "NDArray[Any] | tuple[NDArray[Any], NDArray[Any]] | tuple[NDArray[np.float32], NDArray[np.float32]] | tuple[NDArray[np.float64], NDArray[np.float64]]":
    """
    Per-channel energy normalisation (PCEN) of a spectrogram.

    Time is the last axis (``axis=-1``) and the max filter runs along frequency,
    the axis before it (``max_axis=-2``).

    Args:

        S (array_like): Non-negative spectrogram, ``(F, T)`` or ``(channels, F, T)``
            (typically a mel power spectrogram scaled to integer range).
        sr (float): Sample rate (Hz). Required.
        hop_length (int): Hop of ``S`` in samples. Required.
        gain (float): Gain-normalisation exponent. Default 0.98.
        bias (float): Bias added before compression. Default 2.
        power (float): Compression exponent (0 selects ``log1p``). Default 0.5.
        time_constant (float): Smoothing time constant (s). Default 0.4.
        eps (float): Floor of the smoother. Default 1e-6.
        b (Optional[float]): Explicit smoothing coefficient in ``[0, 1]``.
        max_size (int): Width of a maximum filter along frequency applied to ``S``
            before smoothing. Default 1 (off).
        ref (Optional[array_like]): Array (shape of ``S``) to smooth instead of ``S``.
        axis (int): Must be -1 (or ``S.ndim - 1``).
        max_axis (Optional[int]): Must be None or the frequency axis (-2).
        zi (Optional[array_like]): Initial filter state, shape ``(..., F, 1)``
            (e.g. a previous ``zf``); None = steady state for a unit input.
        return_zf (bool): Also return the final filter state. Default False.

    Returns:

        numpy.ndarray[float32 | float64] | tuple: the PCEN spectrogram (shape of ``S``), and
        with ``return_zf=True`` also ``zf`` of shape ``(..., F, 1)``.

    Raises:

        ValueError: For invalid parameters, unsupported ``axis`` / ``max_axis`` or
            shape mismatches. Complex ``S`` raises TypeError (pass
            ``numpy.abs(S)``).
    """

def perceptual_weighting(S: "ArrayLike", frequencies: "ArrayLike", *, kind: str = "A", ref: float | Literal["max"] | Callable[..., Any] = ..., amin: float = 1e-10, top_db: float | None = None) -> "NDArray[Any]":
    """
    Perceptually weighted dB spectrogram.

    Computes ``power_to_db(S, ref, amin, top_db) + weighting(frequencies)``.

    Args:

        S (array_like): Power spectrogram, ``(F, T)`` or ``(channels, F, T)``.
        frequencies (array_like): 1-D centre frequency of each bin (Hz).
        kind (str): Weighting curve, ``"A"`` (default) ... ``"Z"``.
        ref (float | "max" | numpy.max): dB reference. Default 1.0.
        amin (float): Floor before the logarithm. Default 1e-10.
        top_db (Optional[float]): Dynamic range below the peak in dB (per
            channel). Default None (no floor).

    Returns:

        numpy.ndarray[float32 | float64]: same shape as ``S``, in dB.

    Raises:

        ValueError: For shape mismatches or invalid dB parameters.
    """

def piptrack_from_spectrogram(S: "ArrayLike", *, sr: float, n_fft: int | None = None, fmin: float = 150.0, fmax: float = 4000.0, threshold: float = 0.1, ref: Any | None = None) -> "tuple[NDArray[Any], NDArray[Any]] | tuple[NDArray[np.float32], NDArray[np.float32]] | tuple[NDArray[np.float64], NDArray[np.float64]]":
    """
    Spectral peak tracking on a precomputed magnitude spectrogram.

    Args:

        S (array_like): Magnitude spectrogram, ``(1 + n_fft // 2, T)`` or
            ``(channels, 1 + n_fft // 2, T)``.
        sr (float): Sample rate (Hz). Required.
        n_fft (Optional[int]): FFT length; None infers ``2 * (F - 1)``.
        fmin (float): Lowest reported frequency (Hz). Default 150.
        fmax (float): Peaks must lie below this frequency (Hz). Default 4000.
        threshold (float): Relative threshold. Default 0.1.
        ref (None | float | numpy.max | numpy.mean): Threshold reference; None
            (default) = per-frame maximum, a number = absolute threshold.

    Returns:

        tuple[numpy.ndarray[float32 | float64], numpy.ndarray[float32 | float64]]: ``(pitches,
        magnitudes)``, both shaped like ``S``; zero except at detected peaks.

    Raises:

        ValueError: For invalid parameters or a bin count that does not match ``n_fft``.
    """

def pitch_tuning(frequencies: "ArrayLike", *, resolution: float = 0.01, bins_per_octave: int = ...) -> float:
    """
    Tuning offset of a set of frequencies relative to A440.

    Args:

        frequencies (array_like): Frequencies in Hz; non-positive values are ignored.
        resolution (float): Histogram resolution in fractions of a bin, in ``(0, 1)``.
            Default 0.01.
        bins_per_octave (int): Bins per octave of the reference grid. Default 12.

    Returns:

        float: tuning in ``[-0.5, 0.5)`` fractions of a bin (0.0 for no input).

    Raises:

        ValueError: If ``resolution`` is not in ``(0, 1)``.
    """

def poly_features_from_spectrogram(S: "ArrayLike", *, sr: float, freq: Any | None = None, order: int = ...) -> "NDArray[Any]":
    """
    Coefficients of a least-squares polynomial fit of every frame against the
    bin frequencies.

    Args:

        S (array_like): Spectrogram, ``(F, T)`` or ``(channels, F, T)``.
        sr (float): Sample rate for the default ``freq``. Required.
        freq (Optional[array_like]): 1-D bin frequencies; ``None`` = FFT bin centres.
        order (int): Polynomial order. Default 1.

    Returns:

        numpy.ndarray[float32 | float64]: shape ``(order + 1, T)`` or ``(channels, order + 1, T)``,
        highest power first (as ``numpy.polyfit``).

    Raises:

        ValueError: If ``order`` is negative or not below the number of bins.
    """

def pyin(y: "ArrayLike", *, fmin: float, fmax: float, sr: float, frame_length: int = ..., hop_length: int | None = None, n_thresholds: int = ..., beta_parameters: tuple[float, float] = ..., boltzmann_parameter: float = 2.0, resolution: float = 0.1, max_transition_rate: float = 35.92, switch_prob: float = 0.01, no_trough_prob: float = 0.01, fill_na: float | None = ..., center: bool = True, pad_mode: str = "constant", transition_min_prob: float | None = ...) -> "tuple[NDArray[Any], NDArray[Any], NDArray[Any]] | tuple[NDArray[np.float32], NDArray[np.bool_], NDArray[np.float32]] | tuple[NDArray[np.float64], NDArray[np.bool_], NDArray[np.float64]]":
    """
    Probabilistic YIN of an array: f0 tracking with voicing decisions.

    Args:

        y (array_like): Signal, ``(n,)`` or ``(channels, n)``.
        fmin, fmax (float): Search range in Hz. Required.
        sr (float): Sample rate. Required.
        frame_length (int): Default 2048.
        hop_length (Optional[int]): None = ``frame_length // 4``.
        n_thresholds (int): Default 100.
        beta_parameters (tuple[float, float]): Beta prior. Default (2, 18).
        boltzmann_parameter (float): Default 2.
        resolution (float): Pitch-bin width in semitones. Default 0.1.
        max_transition_rate (float): Octaves per second. Default 35.92.
        switch_prob (float): Voicing switch probability. Default 0.01.
        no_trough_prob (float): Default 0.01.
        fill_na (Optional[float]): f0 of unvoiced frames: NaN (default), None
            (best-guess pitch) or a number.
        center (bool): Default True.
        pad_mode (str): Default ``"constant"``.
        transition_min_prob (Optional[float]): Viterbi pruning threshold
            (None or 0 = exact). Default 1e-4.

    Returns:

        tuple: ``(f0, voiced_flag, voiced_prob)``, each ``(n_frames,)`` or
        ``(channels, n_frames)``; ``voiced_flag`` is ``bool``, the others in ``y``'s precision.

    Raises:

        ValueError: For invalid parameters or a too-short signal.
    """

def rms_from_spectrogram(S: "ArrayLike", *, frame_length: int = ...) -> "NDArray[Any]":
    """
    Frame-wise RMS from STFT magnitudes (Parseval).

    Args:

        S (array_like): STFT magnitudes, ``(1 + frame_length // 2, T)`` or
            ``(channels, 1 + frame_length // 2, T)``.
        frame_length (int): The FFT length used to compute ``S``. Default 2048.

    Returns:

        numpy.ndarray[float32 | float64]: shape ``(1, T)`` or ``(channels, 1, T)``.

    Raises:

        ValueError: If the bin count is not ``1 + frame_length // 2``.
    """

def salience(S: "ArrayLike", *, freqs: "ArrayLike", harmonics: "Sequence[float]", weights: Any | None = None, aggregate: Any | None = None, filter_peaks: bool = True, fill_value: float = ..., kind: str = "linear", axis: int = ...) -> "NDArray[Any]":
    """
    Harmonic salience of a spectrogram: the weighted sum of the energy at the
    harmonics of every bin.

    Args:

        S (array_like): Non-negative magnitudes, ``(F, T)`` or ``(channels, F, T)``.
        freqs (array_like): Bin frequencies: 1-D ``(F,)`` or shaped like ``S``
            (time-varying).
        harmonics (Sequence[float]): Harmonic multipliers (1 is ``S`` itself).
        weights (Optional[array_like]): One weight per harmonic (weighted average).
        aggregate: None / ``numpy.average`` (default), ``numpy.mean``, ``numpy.sum``,
            ``numpy.max``, ``numpy.min``, ``numpy.median``.
        filter_peaks (bool): Keep only local maxima along frequency. Default True.
        fill_value (float): Value of non-peaks. Default NaN.
        kind (str): Must be ``"linear"``.
        axis (int): Must be -2.

    Returns:

        numpy.ndarray[float32 | float64]: same shape as ``S``.

    Raises:

        ValueError: For shape mismatches or unsupported options.
    """

def spectral_bandwidth_from_spectrogram(S: "ArrayLike", *, sr: float, freq: Any | None = None, centroid: Any | None = None, norm: bool = True, p: float = 2.0) -> "NDArray[Any]":
    """
    Order-``p`` spectral bandwidth of every frame around the spectral centroid.

    Args:

        S (array_like): Non-negative spectrogram, ``(F, T)`` or ``(channels, F, T)``.
        sr (float): Sample rate for the default ``freq``. Required.
        freq (Optional[array_like]): Bin frequencies, 1-D or time-varying with the shape
            of ``S``; ``None`` = FFT bin centres.
        centroid (Optional[array_like]): Precomputed centroid, shape ``(1, T)`` or
            ``(channels, 1, T)``; ``None`` computes it.
        norm (bool): Normalise each frame to unit sum first. Default True.
        p (float): Order of the deviation. Default 2.

    Returns:

        numpy.ndarray[float32 | float64]: shape ``(1, T)`` or ``(channels, 1, T)``, in Hz.

    Raises:

        ValueError: For invalid spectrogram values or shapes, or ``p`` zero / not finite.
    """

def spectral_centroid_from_spectrogram(S: "ArrayLike", *, sr: float, freq: Any | None = None) -> "NDArray[Any]":
    """
    Spectral centroid of every frame of a spectrogram: the magnitude-weighted
    mean frequency.

    Args:

        S (array_like): Non-negative magnitude spectrogram, shape ``(F, T)`` or
            ``(channels, F, T)``.
        sr (float): Sample rate (Hz), used for the default ``freq``. Required.
        freq (Optional[array_like]): Centre frequency of each bin (Hz): 1-D, or
            time-varying with the shape of ``S`` (``(F, T)`` is shared by all channels).
            ``None`` uses ``fft_frequencies(sr, n_fft=2 * (F - 1))``.

    Returns:

        numpy.ndarray[float32 | float64]: shape ``(1, T)`` or ``(channels, 1, T)``, in Hz.

    Raises:

        ValueError: If ``S`` has negative or non-finite values, or ``freq`` has
            the wrong shape.
    """

def spectral_contrast_from_spectrogram(S: "ArrayLike", *, sr: float, freq: Any | None = None, fmin: float = 200.0, n_bands: int = ..., quantile: float = 0.02, linear: bool = False) -> "NDArray[Any]":
    """
    Octave-band spectral contrast of every frame: the peak-to-valley difference
    in each band, in dB.

    Args:

        S (array_like): Non-negative magnitude spectrogram, ``(F, T)`` or
            ``(channels, F, T)``.
        sr (float): Sample rate (Hz). Required.
        freq (Optional[array_like]): 1-D bin frequencies; ``None`` = FFT bin centres.
        fmin (float): Upper edge of the lowest band (Hz). Default 200.
        n_bands (int): Number of octave bands above ``fmin``. Default 6.
        quantile (float): Fraction of each band used for peak / valley. Default 0.02.
        linear (bool): Linear difference instead of a dB ratio. Default False.

    Returns:

        numpy.ndarray[float32 | float64]: shape ``(n_bands + 1, T)`` or
        ``(channels, n_bands + 1, T)``.

    Raises:

        ValueError: For invalid parameters (e.g. the top band above Nyquist).
    """

def spectral_flatness_from_spectrogram(S: "ArrayLike", *, amin: float = 1e-10, power: float = 2.0) -> "NDArray[Any]":
    """
    Spectral flatness (Wiener entropy) of every frame: the geometric over the
    arithmetic mean of ``S ** power``.

    Args:

        S (array_like): Non-negative magnitude spectrogram, ``(F, T)`` or
            ``(channels, F, T)``.
        amin (float): Floor applied to ``S ** power``. Default 1e-10.
        power (float): Exponent applied to ``S``. Default 2.0.

    Returns:

        numpy.ndarray[float32 | float64]: shape ``(1, T)`` or ``(channels, 1, T)``, in ``[0, 1]``.

    Raises:

        ValueError: For invalid spectrogram values, ``amin <= 0`` or ``power <= 0``.
    """

def spectral_rolloff_from_spectrogram(S: "ArrayLike", *, sr: float, freq: Any | None = None, roll_percent: float = 0.85) -> "NDArray[Any]":
    """
    Roll-off frequency of every frame: the frequency below which
    ``roll_percent`` of the energy lies.

    Args:

        S (array_like): Non-negative spectrogram, ``(F, T)`` or ``(channels, F, T)``.
        sr (float): Sample rate for the default ``freq``. Required.
        freq (Optional[array_like]): Bin frequencies, 1-D or time-varying with the shape
            of ``S``; ``None`` = FFT bin centres.
        roll_percent (float): Energy fraction in ``(0, 1)``. Default 0.85.

    Returns:

        numpy.ndarray[float32 | float64]: shape ``(1, T)`` or ``(channels, 1, T)``, in Hz.

    Raises:

        ValueError: For invalid spectrogram values or ``roll_percent`` outside ``(0, 1)``.
    """

def tonnetz_from_chroma(chroma: "ArrayLike") -> "NDArray[Any]":
    """
    Tonal centroid features (tonnetz) from a chromagram.

    Args:

        chroma (array_like): Chromagram, ``(n_chroma, T)`` or ``(channels, n_chroma, T)``.

    Returns:

        numpy.ndarray[float32 | float64]: shape ``(6, T)`` or ``(channels, 6, T)``.

    Raises:

        RuntimeError: For non-finite input.
    """

def wavelet(*, freqs: "ArrayLike", sr: float, window: str | tuple[str, float] | float | ArrayLike = ..., filter_scale: float = 1.0, pad_fft: bool = True, norm: float | None = ..., gamma: float | None = ..., alpha: Any | None = None) -> tuple[Incomplete, Incomplete]:
    """
    Constant-Q / variable-Q wavelet filters.

    Args:

        freqs (array_like): Centre frequencies in Hz, as for ``wavelet_lengths``.
        sr (float): Sample rate. Required.
        window: Named window. Default ``"hann"``.
        filter_scale (float): Filter-length scale. Default 1.
        pad_fft (bool): Centre the filters in a power-of-two buffer. Default True.
        norm (Optional[float]): Per-filter normalisation (p-norm). Default 1.
        gamma (Optional[float]): Bandwidth offset (Hz). Default 0.
        alpha (Optional[float | ``array_like``]): As ``wavelet_lengths``.

    Returns:

        tuple[numpy.ndarray[complex128], numpy.ndarray[float64]]: ``(filters,
        lengths)``, complex128 ``filters`` of shape ``(len(freqs), width)`` and
        float64 ``lengths``.

    Raises:

        ValueError: For invalid frequencies or parameters.
    """

def wavelet_lengths(*, freqs: "ArrayLike", sr: float, window: str | tuple[str, float] | float | ArrayLike = ..., filter_scale: float = 1.0, gamma: float | None = ..., alpha: Any | None = None) -> tuple[Incomplete, float]:
    """
    Fractional lengths of constant-Q / variable-Q wavelet filters.

    Args:

        freqs (array_like): Centre frequencies in Hz, ascending: at least two, or one
            when ``alpha`` is given.
        sr (float): Sample rate. Required.
        window: Window (named; see ``spectral.get_window``). Default ``"hann"``.
        filter_scale (float): Filter-length scale. Default 1.
        gamma (Optional[float]): Bandwidth offset (Hz); 0 (default) = constant Q,
            None = ERB-derived.
        alpha (Optional[float | ``array_like``]): Relative bandwidth, one value or one per
            frequency. None derives it from the spacing of the frequencies.

    Returns:

        tuple[numpy.ndarray[float64], float]: ``(lengths, cutoff)``: the length of
        each filter in samples and the highest cut-off frequency in Hz.

    Raises:

        ``ValueError``: For invalid frequencies or parameters.
    """

def yin(y: "ArrayLike", *, fmin: float, fmax: float, sr: float, frame_length: int = ..., hop_length: int | None = None, trough_threshold: float = 0.1, center: bool = True, pad_mode: str = "constant") -> "NDArray[Any]":
    """
    Frame-wise YIN fundamental frequency of an array.

    Args:

        y (array_like): Signal, ``(n,)`` or ``(channels, n)``.
        fmin (float): Lowest frequency searched (Hz). Required.
        fmax (float): Highest frequency searched (Hz). Required.
        sr (float): Sample rate. Required.
        frame_length (int): Samples per frame. Default 2048.
        hop_length (Optional[int]): None = ``frame_length // 4``.
        trough_threshold (float): Absolute CMND threshold. Default 0.1.
        center (bool): Centre frames on ``t * hop_length``. Default True.
        pad_mode (str): Padding for ``center``. Default ``"constant"``.

    Returns:

        numpy.ndarray[float32 | float64]: f0 in Hz, shape ``(n_frames,)`` or ``(channels, n_frames)``.

    Raises:

        ValueError: For invalid parameters or a too-short signal.
    """

def z_weighting(frequencies: "float | ArrayLike", *, min_db: float | None = ...) -> "float | NDArray[np.float64]":
    """
    The Z-weighting of a set of frequencies in dB (flat, 0 dB).

    Args:

        frequencies (float | array_like): Frequencies in Hz.
        min_db (Optional[float]): Lower clip in dB (default -80); None disables it.

    Returns:

        float | numpy.ndarray[float64]: dB weights, same shape as ``frequencies``
        (a float for scalar input).
    """
