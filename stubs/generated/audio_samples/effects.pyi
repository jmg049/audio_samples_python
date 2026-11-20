"""
Effect and decomposition functions over NumPy arrays (phase vocoder, pre-/de-emphasis with filter state, soft masks, HPSS, nearest-neighbour filtering, Griffin-Lim, LPC, click tracks). float32 / complex64 arrays are processed in float32 and give float32 output; any other input is converted to float64.
"""

from collections.abc import Sequence
from numpy.random import BitGenerator, Generator, RandomState, SeedSequence
from numpy.typing import ArrayLike, DTypeLike
from typing import Any

__all__ = ["clicks", "deemphasis", "griffinlim", "hpss", "lpc", "nn_filter", "phase_vocoder", "preemphasis", "softmask"]

def clicks(*, times: Any | None = None, frames: Any | None = None, sr: float, hop_length: int | None = None, click_freq: float = 1000.0, click_duration: float = 0.1, click: Any | None = None, length: int | None = None, dtype: DTypeLike | None = None) -> "NDArray[np.float32] | NDArray[np.float64]":
    """
    Synthesises a click track.

    Args:

        times (Optional[array_like]): Click times in seconds.
        frames (Optional[array_like]): Click frame indices (used when ``times``
            is None).
        sr (int): Sampling rate (a positive integer). Required.
        hop_length (Optional[int]): Samples per frame for ``frames``. Required
            when ``frames`` is given; unused with ``times``.
        click_freq (float): Frequency of the default click. Default 1000.
        click_duration (float): Duration of the default click (s). Default 0.1.
        click (Optional[array_like]): Custom 1-D click waveform.
        length (Optional[int]): Output length; clicks starting at or after it
            are dropped. None: up to the end of the last click.
        dtype (Optional[numpy.dtype]): Output dtype, ``numpy.float32`` or
            ``numpy.float64``. Default None: float64.

    Returns:

        numpy.ndarray[float64 | float32]: 1-D click signal (float64 unless
        ``dtype`` says otherwise).

    Raises:

        ValueError: If neither ``times`` nor ``frames`` is given, a time is
            negative, or parameters are invalid.
    """

def deemphasis(y: "ArrayLike", *, coef: float = 0.97, zi: Any | None = None, return_zf: bool = False) -> "NDArray[Any] | tuple[NDArray[Any], NDArray[Any]] | tuple[NDArray[np.float32], NDArray[np.float32]] | tuple[NDArray[np.float64], NDArray[np.float64]]":
    """
    De-emphasis filter ``y[n] = x[n] + coef * y[n - 1]`` (inverse of
    ``preemphasis``).

    Args:

        y (array_like): Signal ``(n,)`` or ``(channels, n)``.
        coef (float): Coefficient. Default 0.97 (3 only with an explicit ``zi``).
        zi (Optional[float | array_like]): Initial filter state. None uses the
            state that exactly inverts ``preemphasis`` with its default state.
        return_zf (bool): Also return the final state. Default False.

    Returns:

        numpy.ndarray[float32 | float64] of the input's shape, or ``(y, zf)``.

    Raises:

        ValueError: As ``preemphasis``.
    """

def griffinlim(S: "ArrayLike", *, n_iter: int = ..., hop_length: int | None = None, win_length: int | None = None, n_fft: int | None = None, window: str | tuple[str, float] | float | ArrayLike = ..., center: bool = True, dtype: Any | None = None, length: int | None = None, pad_mode: str = "constant", momentum: float = 0.99, init: str | None = ..., rng: int | None | Generator | RandomState | SeedSequence | BitGenerator | Sequence[int] = ..., random_state: int | None | Generator | RandomState | SeedSequence | BitGenerator | Sequence[int] = None) -> "NDArray[Any]":
    """
    Reconstructs a signal from a magnitude spectrogram with fast Griffin-Lim.

    Args:

        S (array_like): Non-negative magnitude ``(1 + n_fft // 2, n_frames)`` or
            ``(channels, 1 + n_fft // 2, n_frames)``.
        n_iter (int): Iterations. Default 32.
        hop_length, win_length, window, center, pad_mode: STFT parameters of ``S``.
        n_fft (Optional[int]): None infers ``2 * (n_bins - 1)``.
        dtype: Ignored; the output follows ``S``'s precision (float32 for a
            float32 ``S``, float64 otherwise).
        length (Optional[int]): Exact output length.
        momentum (float): Fast Griffin-Lim momentum (0 = plain Griffin-Lim). Default 0.99.
        init (Optional[str]): ``"random"`` (default) or None (zero phase,
            deterministic).
        rng: Seed for ``init="random"``: an integer, None for a fresh seed on every
            call, or a NumPy generator or anything else ``numpy.random.default_rng``
            takes, from which one seed is drawn (advancing the generator). The random
            phases come from this library's generator.
        random_state: Deprecated alias of ``rng``.

    Returns:

        numpy.ndarray[float32 | float64]: shape ``(n,)`` for 2-D ``S``, ``(channels, n)`` for 3-D.

    Raises:

        ValueError: For invalid parameters or a bin count that does not match ``n_fft``.
    """

def hpss(S: "ArrayLike", *, kernel_size: Any | None = None, power: float = 2.0, mask: bool = False, margin: Any | None = None) -> tuple[Any, Any]:
    """
    Median-filtering harmonic/percussive separation of a spectrogram.

    ``H`` is ``|S|`` median-filtered along time, ``P`` along frequency (edges
    mirrored, edge sample repeated); the masks are ``softmask(H, margin_h * P)`` and
    ``softmask(P, margin_p * H)``.

    Args:

        S (array_like): Magnitude/power spectrogram (real, non-negative) or
            complex STFT, shape ``(n_bins, n_frames)`` or
            ``(channels, n_bins, n_frames)``.
        kernel_size (int | tuple[int, int]): Median filter lengths (harmonic
            along time, percussive along frequency). Default 31.
        power (float): Mask exponent (``numpy.inf`` for hard masks). Default 2.
        mask (bool): Return the masks instead of the components. Default False.
        margin (float | tuple[float, float]): Separation margins (>= 1).
            Default 1.

    Returns:

        tuple: ``(harmonic, percussive)`` of the input's shape and dtype
        (complex input keeps its phase), or with ``mask=True`` the masks in ``S``'s precision
        (bool for ``power=inf``).

    Raises:

        ValueError: For invalid parameters or negative magnitudes.
    """

def lpc(y: "ArrayLike", *, order: int, axis: int = ...) -> "NDArray[Any]":
    """
    Linear-prediction coefficients by Burg's method.

    Args:

        y (array_like): Finite signal ``(n,)`` or ``(channels, n)`` with more
            than ``order`` samples.
        order (int): Prediction order.
        axis (int): Must be -1.

    Returns:

        numpy.ndarray[float32 | float64]: ``(order + 1,)`` coefficients ``[1, a_1, ...]``
        (``(channels, order + 1)`` for 2-D input).

    Raises:

        ValueError: For a too-short or non-finite signal.
    """

def nn_filter(S: "ArrayLike", *, rec: Any | None = None, aggregate: Any | None = None, axis: int = ..., **kwargs) -> "NDArray[Any]":
    """
    Nearest-neighbour filtering of a feature matrix.

    Each frame is replaced by an aggregate of its neighbours in a recurrence
    matrix (``rec[j, i] != 0`` links frame ``j`` to output frame ``i``); frames
    without neighbours are copied.

    Args:

        S (array_like): Features ``(n_features, n_frames)``.
        rec (Optional[array_like]): ``(n_frames, n_frames)`` recurrence matrix, dense
            (bool or float) or ``scipy.sparse`` in any format, where every stored entry
            is a neighbour. None computes
            ``audio_samples.sequence.recurrence_matrix(S, **kwargs)``.
        aggregate: None / ``numpy.mean`` (default), ``numpy.median``, ``numpy.average``
            (weighted by ``rec``; use an affinity matrix), ``numpy.max``, ``numpy.min``,
            ``numpy.sum``, or any callable ``f(neighbours, axis=0)`` of the
            ``(n_neighbours, n_features)`` neighbours of a frame, which runs with the
            GIL held.
        axis (int): Must be -1.
        **kwargs: ``recurrence_matrix`` arguments when ``rec`` is None: ``k``,
            ``width``, ``metric``, ``sym``, ``mode``, ``bandwidth``, ``self``,
            ``full``.

    Returns:

        numpy.ndarray[float32 | float64]: filtered ``(n_features, n_frames)``.

    Raises:

        ValueError: For a wrong-shaped ``rec`` or invalid arguments.
        TypeError: For an aggregate that is not callable or unknown keyword arguments.
    """

def phase_vocoder(D: "ArrayLike", *, rate: float | None = None, t_out: Any | None = None, kind: str = "linear", hop_length: Any | None = None, n_fft: Any | None = None) -> "NDArray[Any]":
    """
    Time-stretches an STFT matrix with a phase vocoder.

    Give exactly one of ``rate`` (output frames at ``0, rate, 2 rate, ...``) or
    ``t_out`` (explicit fractional frame positions in ``[0, n_frames)``).
    Magnitudes are linearly interpolated; the phase accumulates the raw
    frame-to-frame phase differences.

    Args:

        D (array_like): complex STFT of shape ``(n_bins, n_frames)`` or
            ``(channels, n_bins, n_frames)``.
        rate (Optional[float]): Speed-up factor (> 1 is faster / fewer frames).
        t_out (Optional[array_like]): Output frame positions.
        kind (str): Must be ``"linear"``.
        hop_length (Optional[int]): Deprecated and unused (the phase advance
            cancels); accepted with a ``FutureWarning`` when given.
        n_fft (Optional[int]): Deprecated and unused, as ``hop_length``.

    Returns:

        numpy.ndarray[complex64 | complex128]: shape ``(n_bins, ceil(n_frames / rate))`` (or
        ``len(t_out)`` frames), with a leading channel axis for 3-D input.

    Raises:

        ValueError: If neither or both of ``rate`` / ``t_out`` are given, or a
            value is out of range.
    """

def preemphasis(y: "ArrayLike", *, coef: float = 0.97, zi: Any | None = None, return_zf: bool = False) -> "NDArray[Any] | tuple[NDArray[Any], NDArray[Any]] | tuple[NDArray[np.float32], NDArray[np.float32]] | tuple[NDArray[np.float64], NDArray[np.float64]]":
    """
    Pre-emphasis filter ``y[n] = x[n] - coef * x[n - 1]``.

    Args:

        y (array_like): Signal ``(n,)`` or ``(channels, n)``.
        coef (float): Coefficient. Default 0.97.
        zi (Optional[float | array_like]): Initial filter state (one value, or
            one per channel). None uses ``2 * y[0] - y[1]``.
        return_zf (bool): Also return the final state. Default False.

    Returns:

        numpy.ndarray[float32 | float64] of the input's shape, or ``(y, zf)`` with
        ``return_zf=True`` (``zf`` of shape ``(1,)`` or ``(channels, 1)``).

    Raises:

        ValueError: For a non-finite ``coef`` / ``zi``, or fewer than two samples
            with the default ``zi``.
    """

def softmask(X: "ArrayLike", X_ref: "ArrayLike", *, power: float = 1.0, split_zeros: bool = False) -> "NDArray[Any]":
    """
    Soft mask ``X**p / (X**p + X_ref**p)``.

    Args:

        X (array_like): Non-negative values selected by the mask.
        X_ref (array_like): Non-negative reference, same shape.
        power (float): Exponent (> 0). ``numpy.inf`` gives a hard (bool) mask
            ``X > X_ref``. Default 1.
        split_zeros (bool): Mask value where both inputs are zero: 0.5 if True,
            else 0. Default False.

    Returns:

        numpy.ndarray: values in ``[0, 1]`` in ``X``'s precision (bool for
        ``power=inf``), same shape.

    Raises:

        ValueError: For mismatched shapes, negative input or ``power <= 0``.
    """
