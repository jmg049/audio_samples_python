"""librosa-compatible spectral helpers over NumPy arrays.

Free functions mirroring ``librosa.stft`` / ``istft``, the dB conversions,
``librosa.filters.mel`` / ``get_window`` / ``window_sumsquare`` /
``window_bandwidth``, ``librosa.feature.delta`` / ``stack_memory`` /
``melspectrogram(S=...)`` / ``mfcc(S=...)`` and ``librosa.util.frame``.
Defaults are librosa 1.0.0's, except that ``power_to_db`` and
``amplitude_to_db`` default to ``top_db=None`` (librosa: 80). float32 / complex64 NumPy arrays are processed in float32 and give float32 /
complex64 output; any other array-like (lists, integer, float16 or float64
arrays) is converted to float64 and gives float64 / complex128 output.
The mel filterbank and windows are float64 tables. Array shapes follow
librosa: a 1-D signal gives a 2-D STFT and a ``(channels, n)`` signal a
``(channels, F, T)`` STFT.

For the same analysis directly on audio see ``AudioSamples.stft_matrix``,
``melspectrogram`` and ``mfcc_frames``, which always return
``(channels, F, T)``.
"""

from __future__ import annotations

from typing import Any, Callable, Literal, Optional, Union

import numpy as np
import numpy.typing as npt
from numpy.typing import NDArray

#: Window specification: a SciPy window name (``"hann"``, ``"hamming"``,
#: ``"blackman"``, ``"blackmanharris"``, ``"boxcar"``/``"rectangular"``,
#: ``"bartlett"``, ``"triang"``), ``("kaiser", beta)`` / ``("gaussian", std)``,
#: a number (Kaiser beta, as SciPy) or an array of coefficients.
WindowSpec = Union[str, tuple[str, float], float, npt.ArrayLike]
#: STFT padding mode used when ``center=True``.
PadMode = Literal["constant", "reflect", "edge", "symmetric"]
#: Mel filter normalisation: ``"slaney"``, None, or a p-norm exponent (``numpy.inf`` allowed).
MelNorm = Union[Literal["slaney"], float, None]
#: dB reference: a number, ``"max"`` or ``numpy.max``.
DbRef = Union[float, Literal["max"], Callable[..., Any]]

def stft(
    y: npt.ArrayLike,
    *,
    n_fft: int = 2048,
    hop_length: Optional[int] = None,
    win_length: Optional[int] = None,
    window: WindowSpec = "hann",
    center: bool = True,
    pad_mode: PadMode = "constant",
) -> NDArray[np.complexfloating]:
    """Short-time Fourier transform, equivalent to ``librosa.stft``.

    Args:
        y: Real signal, shape ``(n,)`` or ``(channels, n)``.
        n_fft: FFT length.
        hop_length: Samples between frames; None means ``win_length // 4``.
        win_length: Window length (<= n_fft); None means ``n_fft``.
        window: Window specification (periodic for named windows).
        center: Pad ``n_fft // 2`` samples on both sides.
        pad_mode: Padding mode when centred.

    Returns:
        numpy.ndarray[complex64 | complex128]: shape ``(1 + n_fft // 2, n_frames)`` for 1-D
        input or ``(channels, 1 + n_fft // 2, n_frames)`` for 2-D input.

    Raises:
        ValueError: Invalid parameters or input with more than 2 dimensions.
    """
    ...

def istft(
    stft_matrix: npt.ArrayLike,
    *,
    hop_length: Optional[int] = None,
    win_length: Optional[int] = None,
    n_fft: Optional[int] = None,
    window: WindowSpec = "hann",
    center: bool = True,
    length: Optional[int] = None,
) -> NDArray[np.floating]:
    """Inverse STFT, equivalent to ``librosa.istft``.

    Args:
        stft_matrix: Complex array, ``(1 + n_fft // 2, n_frames)`` or
            ``(channels, 1 + n_fft // 2, n_frames)``.
        hop_length: None means ``win_length // 4``.
        win_length: None means ``n_fft``.
        n_fft: None infers ``2 * (n_bins - 1)``.
        window: Analysis window.
        center: Whether the frames were centred.
        length: Exact output length (pass the original length).

    Returns:
        numpy.ndarray[float32 | float64]: shape ``(n,)`` for 2-D input, ``(channels, n)`` for 3-D.

    Raises:
        ValueError: Invalid parameters or a bin count that does not match ``n_fft``.
    """
    ...

def power_to_db(
    S: npt.ArrayLike, *, ref: DbRef = 1.0, amin: float = 1e-10, top_db: Optional[float] = None,
    axes: Optional[Literal["auto"]] = "auto",
) -> Any:
    """Power to decibels, equivalent to ``librosa.power_to_db``.

    ``10 * log10(max(amin, S)) - 10 * log10(max(amin, |ref|))``, floored at
    ``max - top_db``.

    Args:
        S: Non-negative power values, any shape (or a scalar).
        ref: 0 dB reference: a number, ``"max"`` or ``numpy.max``.
        amin: Lower clamp before the logarithm (> 0).
        top_db: Dynamic range below the maximum; None disables the floor.
        axes: ``"auto"`` (default): ``ref=numpy.max`` and ``top_db`` use the last
            two axes, i.e. each channel of a ``(channels, F, T)`` array
            separately; None: the whole array.

    Returns:
        float for scalar input, otherwise numpy.ndarray[float32 | float64] of ``S``'s shape.

    Raises:
        ValueError: ``amin <= 0`` or ``top_db < 0``.
        TypeError: Complex input (pass ``numpy.abs(D) ** 2``).
    """
    ...

def amplitude_to_db(
    S: npt.ArrayLike, *, ref: DbRef = 1.0, amin: float = 1e-5, top_db: Optional[float] = None,
    axes: Optional[Literal["auto"]] = "auto",
) -> Any:
    """Amplitude to decibels, equivalent to ``librosa.amplitude_to_db``.

    ``20 * log10(max(amin, |S|) / max(amin, |ref|))``, floored at ``max - top_db``.
    Complex input is replaced by its magnitude.

    Returns:
        float for scalar input, otherwise numpy.ndarray[float32 | float64] of ``S``'s shape.

    Raises:
        ValueError: ``amin <= 0`` or ``top_db < 0``.
    """
    ...

def db_to_power(S_db: npt.ArrayLike, *, ref: float = 1.0) -> Any:
    """``ref * 10 ** (S_db / 10)`` (``librosa.db_to_power``); float for scalar
    input, otherwise numpy.ndarray[float32 | float64] of the same shape."""
    ...

def db_to_amplitude(S_db: npt.ArrayLike, *, ref: float = 1.0) -> Any:
    """``sqrt(ref ** 2 * 10 ** (S_db / 10))`` (``librosa.db_to_amplitude``);
    float for scalar input, otherwise numpy.ndarray[float32 | float64] of the same shape."""
    ...

def mel_filterbank(
    *,
    sr: float,
    n_fft: int,
    n_mels: int = 128,
    fmin: float = 0.0,
    fmax: Optional[float] = None,
    htk: bool = False,
    norm: MelNorm = "slaney",
) -> NDArray[np.float64]:
    """Mel filterbank, equivalent to ``librosa.filters.mel(..., dtype=numpy.float64)``.

    Returns:
        numpy.ndarray[float64], shape ``(n_mels, 1 + n_fft // 2)``.

    Raises:
        ValueError: Invalid parameters (e.g. ``fmax <= fmin``, ``n_mels < 1``).
    """
    ...

def get_window(window: WindowSpec, Nx: int, *, fftbins: bool = True) -> NDArray[np.float64]:
    """Window coefficients, equivalent to ``scipy.signal.get_window`` /
    ``librosa.filters.get_window``.

    Args:
        window: Window specification. An array is returned as given (its length
            must be ``Nx``).
        Nx: Window length.
        fftbins: Periodic (True, for spectral analysis) or symmetric (False).

    Returns:
        numpy.ndarray[float64], shape ``(Nx,)``.

    Raises:
        ValueError: Unknown window, invalid parameter or wrong array length.
    """
    ...

def window_sumsquare(
    *,
    window: WindowSpec,
    n_frames: int,
    hop_length: int = 512,
    win_length: Optional[int] = None,
    n_fft: int = 2048,
    norm: Optional[float] = None,
) -> NDArray[np.float64]:
    """Sum-squared window envelope (``librosa.filters.window_sumsquare`` with
    ``dtype=float64``).

    ``norm`` normalises the window before squaring as
    ``librosa.util.normalize(w, norm=norm)``: ``None`` (default), a positive
    number, ``math.inf`` / ``-math.inf`` or ``0``.

    Returns:
        numpy.ndarray[float64], shape ``(n_fft + hop_length * (n_frames - 1),)``.
    """
    ...

def window_bandwidth(window: WindowSpec, n: int = 1000) -> float:
    """Equivalent noise bandwidth ``n * sum(w**2) / sum(w)**2`` in FFT bins of the
    periodic window of length ``n`` (librosa returns a pre-computed table value for
    named windows; they agree to ~1e-4)."""
    ...

def delta(
    data: npt.ArrayLike,
    *,
    width: int = 9,
    order: int = 1,
    axis: int = -1,
    mode: Literal["interp", "nearest", "mirror", "constant", "wrap"] = "interp",
) -> NDArray[np.floating]:
    """Savitzky-Golay delta features, equivalent to ``librosa.feature.delta``.

    Returns:
        numpy.ndarray[float32 | float64] with the shape of ``data``.

    Raises:
        ValueError: Even/too small ``width``, ``order`` outside ``1..width-1``,
            bad ``axis``, or ``mode="interp"`` with ``width`` > axis length.
    """
    ...

def stack_memory(data: npt.ArrayLike, *, n_steps: int = 2, delay: int = 1) -> NDArray[np.floating]:
    """Stack delayed copies (zero padded), equivalent to ``librosa.feature.stack_memory``.

    Args:
        data: shape ``(..., d, t)``; 1-D input is treated as ``(1, t)``.
        n_steps: Number of copies (>= 1).
        delay: Frame delay between copies (non-zero; negative looks ahead).

    Returns:
        numpy.ndarray[float32 | float64], shape ``(..., d * n_steps, t)``.
    """
    ...

def frame(x: npt.ArrayLike, *, frame_length: int, hop_length: int) -> NDArray[np.floating]:
    """Overlapping frames of a 1-D signal (a copy), equivalent to ``librosa.util.frame``.

    Returns:
        numpy.ndarray[float32 | float64], shape
        ``(frame_length, 1 + (len(x) - frame_length) // hop_length)``.

    Raises:
        ValueError: ``x`` not 1-D or shorter than ``frame_length``.
    """
    ...

def melspectrogram_from_spectrogram(
    S: npt.ArrayLike,
    *,
    sr: float,
    n_fft: Optional[int] = None,
    n_mels: int = 128,
    fmin: float = 0.0,
    fmax: Optional[float] = None,
    htk: bool = False,
    norm: MelNorm = "slaney",
) -> NDArray[np.floating]:
    """Mel projection of a spectrogram, equivalent to
    ``librosa.feature.melspectrogram(S=S, sr=sr, n_fft=n_fft, ...)``.

    Args:
        S: shape ``(1 + n_fft // 2, T)`` or ``(channels, 1 + n_fft // 2, T)``.
        n_fft: None infers ``2 * (F - 1)``.

    Returns:
        numpy.ndarray[float32 | float64], shape ``(n_mels, T)`` or ``(channels, n_mels, T)``.
    """
    ...

def mfcc_from_mel_db(
    S: npt.ArrayLike,
    *,
    n_mfcc: int = 20,
    dct_type: Literal[1, 2, 3] = 2,
    norm: Optional[Literal["ortho"]] = "ortho",
    lifter: float = 0.0,
) -> NDArray[np.floating]:
    """MFCCs of a dB mel spectrogram, equivalent to ``librosa.feature.mfcc(S=S, ...)``.

    Args:
        S: shape ``(..., n_mels, T)`` (at least 2-D).

    Returns:
        numpy.ndarray[float32 | float64], shape ``(..., n_mfcc, T)``.

    Raises:
        ValueError: ``n_mfcc > n_mels`` (librosa silently truncates) or invalid DCT
            parameters.
    """
    ...
