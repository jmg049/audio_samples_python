"""
Spectral functions over NumPy arrays (STFT/ISTFT, dB conversions, mel filterbank, windows, delta, stack_memory, frame). float32 / complex64 arrays are processed in float32 and give float32 output; any other input is converted to float64.
"""

from collections.abc import Callable
from numpy.typing import ArrayLike
from typing import Any, Literal

__all__ = ["amplitude_to_db", "db_to_amplitude", "db_to_power", "delta", "frame", "get_window", "istft", "mel_filterbank", "mellin_transform", "melspectrogram_from_spectrogram", "mfcc_from_mel_db", "power_to_db", "stack_memory", "stft", "window_bandwidth", "window_sumsquare"]

def amplitude_to_db(S: "float | ArrayLike", *, ref: float | Literal["max"] | Callable[..., Any] = ..., amin: float = 1e-5, top_db: float | None = None, axes: Literal["auto"] | None = ...) -> "NDArray[Any] | float":
    """
    Convert an amplitude spectrogram to decibels.

    ``20 * log10(max(amin, |S|) / max(amin, |ref|))``, floored at ``max - top_db``.
    Complex input is replaced by its magnitude.

    Args:

        S (array_like): Amplitudes (real or complex), any shape.
        ref (float | "max" | numpy.max): 0 dB reference. Default 1.0.
        amin (float): Lower clamp. Default 1e-5.
        top_db (Optional[float]): Dynamic range below the maximum. Default None
            (no floor).
        axes ("auto" | None): As ``power_to_db``.

    Returns:

        float | numpy.ndarray[float32 | float64]: Same shape as ``S``.

    Raises:

        ValueError: If ``amin <= 0`` or ``top_db < 0``.
    """

def db_to_amplitude(S_db: "float | ArrayLike", *, ref: float = 1.0) -> "NDArray[Any] | float":
    """
    Convert decibels to amplitude:
    ``sqrt(ref ** 2 * 10 ** (S_db / 10))``.

    Args:

        S_db (array_like): dB values, any shape (or a scalar).
        ref (float): Reference amplitude. Default 1.0.

    Returns:

        float | numpy.ndarray[float32 | float64]: Same shape as ``S_db``.
    """

def db_to_power(S_db: "float | ArrayLike", *, ref: float = 1.0) -> "NDArray[Any] | float":
    """
    Convert decibels to power:
    ``ref * 10 ** (S_db / 10)``.

    Args:

        S_db (array_like): dB values, any shape (or a scalar).
        ref (float): Reference power. Default 1.0.

    Returns:

        float | numpy.ndarray[float32 | float64]: Same shape as ``S_db``.
    """

def delta(data: "ArrayLike", *, width: int = ..., order: int = ..., axis: int = ..., mode: str = "interp") -> "NDArray[Any]":
    """
    Savitzky-Golay delta features along the last axis.

    Args:

        data (array_like): Feature array, any shape (e.g. ``(n_mfcc, T)`` or
            ``(channels, n_mfcc, T)``).
        width (int): Odd window width >= 3. Default 9.
        order (int): Derivative order (1 = delta, 2 = delta-delta). Default 1.
        axis (int): Axis to differentiate along. Default -1.
        mode (str): ``"interp"`` (default), ``"nearest"``, ``"mirror"``,
            ``"constant"`` or ``"wrap"``.

    Returns:

        numpy.ndarray[float32 | float64]: Same shape as ``data``.

    Raises:

        ValueError: Invalid width/order/axis, or ``mode="interp"`` with
            ``width`` larger than the axis.
    """

def frame(x: "ArrayLike", *, frame_length: int, hop_length: int, copy: bool = True) -> "NDArray[Any]":
    """
    Slice a 1-D signal into overlapping frames along the last axis.

    By default the frames are a new, writable array (a copy). With
    ``copy=False`` the result is instead a read-only strided view into ``x``:
    no data is copied, the view keeps ``x`` alive and reflects later changes to
    it. Input that is not a contiguous float32 / float64 array is converted
    first, and the view is into the converted array.

    Args:

        x (array_like): 1-D signal.
        frame_length (int): Samples per frame.
        hop_length (int): Samples between frame starts.
        copy (bool): Return a writable copy (default) or, if False, a
            read-only view into ``x``.

    Returns:

        numpy.ndarray[float32 | float64]: shape
        ``(frame_length, 1 + (len(x) - frame_length) // hop_length)``.

    Raises:

        ValueError: If ``x`` is not 1-D or is shorter than ``frame_length``,
            or (copy only) the frame matrix would exceed 2^31 elements.
    """

def get_window(window: str | tuple[str, float] | float | ArrayLike, Nx: int, *, fftbins: bool = True) -> "NDArray[np.float64]":
    """
    Window coefficients for a window specification.

    Args:

        window: ``"hann"``, ``"hamming"``, ``"blackman"``, ``"blackmanharris"``,
            ``"boxcar"`` (``"rectangular"``), ``"bartlett"``, ``"triang"``,
            ``("kaiser", beta)``, ``("gaussian", std)``, a number (Kaiser beta),
            or an array of length ``Nx`` (returned as given).
        Nx (int): Window length.
        fftbins (bool): Periodic window (True, default) or symmetric (False).

    Returns:

        numpy.ndarray[float64]: shape ``(Nx,)``.

    Raises:

        ValueError: Unknown window, invalid parameter, or an array of the wrong length.
    """

def istft(stft_matrix: "ArrayLike", *, hop_length: int | None = None, win_length: int | None = None, n_fft: int | None = None, window: str | tuple[str, float] | float | ArrayLike = ..., center: bool = True, length: int | None = None) -> "NDArray[Any]":
    """
    Inverse short-time Fourier transform by weighted overlap-add.

    Args:

        stft_matrix (array_like): complex array of shape ``(1 + n_fft // 2, n_frames)``
            or ``(channels, 1 + n_fft // 2, n_frames)``.
        hop_length (Optional[int]): ``None`` means ``win_length // 4``.
        win_length (Optional[int]): ``None`` means ``n_fft``.
        n_fft (Optional[int]): ``None`` infers ``2 * (n_bins - 1)``.
        window: As in ``stft``.
        center (bool): Whether the frames were centred. Default True.
        length (Optional[int]): Exact output length.

    Returns:

        numpy.ndarray[float32 | float64]: shape ``(n,)`` for 2-D input, ``(channels, n)`` for 3-D.

    Raises:

        ValueError: For invalid parameters or a bin count that does not match ``n_fft``.
    """

def mel_filterbank(*, sr: float, n_fft: int, n_mels: int = ..., fmin: float = 0.0, fmax: float | None = None, htk: bool = False, norm: Literal["slaney"] | float | None = ...) -> "NDArray[np.float64]":
    """
    Mel filterbank matrix mapping STFT bins to mel bands (float64).

    Args:

        sr (float): Sample rate (Hz).
        n_fft (int): FFT length.
        n_mels (int): Number of bands. Default 128.
        fmin (float): Lowest edge (Hz). Default 0.
        fmax (Optional[float]): Highest edge (Hz); ``None`` = sr / 2.
        htk (bool): HTK mel scale instead of Slaney. Default False.
        norm (None | "slaney" | float): ``"slaney"`` (default, area
            normalisation), None, or a p-norm (``numpy.inf`` allowed).

    Returns:

        numpy.ndarray[float64]: shape ``(n_mels, 1 + n_fft // 2)``.

    Raises:

        ValueError: For invalid parameters (e.g. ``fmax <= fmin``).
    """

def mellin_transform(y: "ArrayLike", *, t_min: float = 0.5, n_fmt: int | None = None, kind: str = "cubic", beta: float = 0.5, over_sample: float = 1.0, axis: int = ...) -> "NDArray[Any]":
    """
    The fast Mellin (scale) transform along one axis.

    Args:

        y (``array_like``): Finite samples, at least 3 along ``axis``.
        ``t_min`` (float): Where the exponential grid starts, in samples. Default 0.5.
        ``n_fmt`` (Optional[int]): Grid points. Default: enough for the signal's resolution.
        kind (str): The interpolant, as for ``arrays.interp1d``. Default ``"cubic"``.
        beta (float): Exponent of the weighting; 0.5 is the scale transform.
        ``over_sample`` (float): Oversampling of the grid. Default 1.
        axis (int): The time axis. Default -1.

    Returns:

        numpy.ndarray: ``n_fmt // 2 + 1`` complex coefficients along ``axis``, ``complex64``
        for ``float32`` input, otherwise ``complex128``.
    """

def melspectrogram_from_spectrogram(S: "ArrayLike", *, sr: float, n_fft: int | None = None, n_mels: int = ..., fmin: float = 0.0, fmax: float | None = None, htk: bool = False, norm: Literal["slaney"] | float | None = ...) -> "NDArray[Any]":
    """
    Project a spectrogram onto mel bands: the mel filterbank applied to ``S``.

    Args:

        S (array_like): Non-negative spectrogram, shape ``(1 + n_fft // 2, T)`` or
            ``(channels, 1 + n_fft // 2, T)`` (power or magnitude, as you choose).
        sr (float): Sample rate (Hz).
        n_fft (Optional[int]): FFT length; ``None`` infers ``2 * (F - 1)``.
        n_mels, fmin, fmax, htk, norm: As ``mel_filterbank``.

    Returns:

        numpy.ndarray[float32 | float64]: shape ``(n_mels, T)`` or ``(channels, n_mels, T)``.
    """

def mfcc_from_mel_db(S: "ArrayLike", *, n_mfcc: int = ..., dct_type: int = ..., norm: str | None = ..., lifter: float = 0.0) -> "NDArray[Any]":
    """
    MFCCs from a log-power (dB) mel spectrogram: DCT along the mel axis, first
    ``n_mfcc`` rows, liftering.

    Args:

        S (array_like): dB mel spectrogram, shape ``(..., n_mels, T)`` (at least 2-D).
        n_mfcc (int): Default 20 (must not exceed ``n_mels``).
        dct_type (int): 1, 2 (default) or 3.
        norm (Optional[str]): ``"ortho"`` (default) or None.
        lifter (float): Liftering coefficient; 0 (default) disables.

    Returns:

        numpy.ndarray[float32 | float64]: shape ``(..., n_mfcc, T)``.
    """

def power_to_db(S: "float | ArrayLike", *, ref: float | Literal["max"] | Callable[..., Any] = ..., amin: float = 1e-10, top_db: float | None = None, axes: Literal["auto"] | None = ...) -> "NDArray[Any] | float":
    """
    Convert a power spectrogram to decibels.

    Computes ``10 * log10(max(amin, S)) - 10 * log10(max(amin, |ref|))``, floored
    at ``max - top_db``.

    Args:

        S (array_like): Power values, any shape (or a scalar).
        ref (float | "max" | numpy.max): 0 dB reference. Default 1.0.
        amin (float): Lower clamp before the log. Default 1e-10.
        top_db (Optional[float]): Dynamic range below the maximum. Default None
            (no floor).
        axes ("auto" | None): Reduction axes for ``ref=numpy.max`` and ``top_db``.
            ``"auto"`` (default) uses the last two axes, i.e. per channel for
            ``(channels, F, T)`` input; ``None`` uses the whole array.

    Returns:

        float | numpy.ndarray[float32 | float64]: Same shape as ``S``.

    Raises:

        ValueError: If ``amin <= 0`` or ``top_db < 0``.
        TypeError: For complex input (pass power, e.g. ``numpy.abs(D) ** 2``).
    """

def stack_memory(data: "ArrayLike", *, n_steps: int = ..., delay: int = ...) -> "NDArray[Any]":
    """
    Stack time-delayed copies of a feature array (zero padding).

    Args:

        data (array_like): shape ``(..., d, t)``; 1-D input is treated as ``(1, t)``.
        n_steps (int): Number of copies. Default 2.
        delay (int): Frame delay between copies (non-zero; negative looks
            ahead). Default 1.

    Returns:

        numpy.ndarray[float32 | float64]: shape ``(..., d * n_steps, t)``.

    Raises:

        ValueError: If ``n_steps < 1`` or ``delay == 0``.
    """

def stft(y: "ArrayLike", *, n_fft: int = ..., hop_length: int | None = None, win_length: int | None = None, window: str | tuple[str, float] | float | ArrayLike = ..., center: bool = True, pad_mode: str = "constant") -> "NDArray[Any]":
    """
    Short-time Fourier transform.

    Args:

        y (numpy.ndarray): Real signal, shape ``(n,)`` or ``(channels, n)``.
        n_fft (int): FFT length. Default 2048.
        hop_length (Optional[int]): ``None`` means ``win_length // 4``.
        win_length (Optional[int]): ``None`` means ``n_fft``.
        window: Window name, ``(name, param)`` tuple or coefficient array
            (see ``get_window``). Default ``"hann"``.
        center (bool): Centre frames (pad ``n_fft // 2`` each side). Default True.
        pad_mode (str): ``"constant"`` (default), ``"reflect"``, ``"edge"``, ``"symmetric"``.

    Returns:

        numpy.ndarray[complex64 | complex128]: shape ``(1 + n_fft // 2, n_frames)`` for 1-D
        input, ``(channels, 1 + n_fft // 2, n_frames)`` for 2-D input.

    Raises:

        ValueError: For invalid parameters or input with more than 2 dimensions.
    """

def window_bandwidth(window: "str | tuple[str, float] | float | ArrayLike", n: int = ...) -> float:
    """
    Equivalent noise bandwidth of a window in FFT bins:
    ``n * sum(w ** 2) / sum(w) ** 2`` for the periodic window of length ``n``.
    The definition is evaluated for every window, named or not.

    Args:

        window: Window specification (see ``get_window``).
        n (int): Window length. Default 1000.

    Returns:

        float: Bandwidth in bins (1.0 for boxcar, 1.5 for Hann).
    """

def window_sumsquare(*, window: str | tuple[str, float] | float | ArrayLike, n_frames: int, hop_length: int = ..., win_length: int | None = None, n_fft: int = ..., norm: float | None = None, dtype: Any | None = None) -> "NDArray[np.float32] | NDArray[np.float64]":
    """
    Sum-squared window envelope of an overlap-add of ``n_frames`` windows.

    Args:

        window: Window specification (see ``get_window``).
        n_frames (int): Number of frames.
        hop_length (int): Default 512.
        win_length (Optional[int]): ``None`` means ``n_fft``.
        n_fft (int): Default 2048.
        norm (Optional[float]): Normalise the window before squaring, as
            ``features.normalize(w, norm=norm)``: ``None`` (default), a
            positive number (L-p norm), ``math.inf`` / ``-math.inf`` (max / min
            absolute value) or ``0`` (count of non-zeros).
        dtype (Optional[numpy.dtype]): ``float32`` or ``float64``. ``None``
            (default) means ``float64``.

    Returns:

        numpy.ndarray[float64 | float32]: shape ``(n_fft + hop_length * (n_frames - 1),)``.

    Raises:

        ValueError: For invalid parameters.
    """
