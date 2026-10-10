"""librosa-compatible effect and decomposition helpers over NumPy arrays.

Free functions mirroring ``librosa.phase_vocoder``,
``librosa.effects.preemphasis`` / ``deemphasis`` (with ``zi`` / ``return_zf``),
``librosa.util.softmask``, ``librosa.decompose.hpss`` / ``nn_filter``,
``librosa.griffinlim``, ``librosa.lpc`` and ``librosa.clicks``. Defaults are
librosa 1.0.0's. The audio-to-audio effects (``time_stretch``, ``pitch_shift``,
``trim``/``split`` as ``trim_db``/``split_db``, ``remix``, ``harmonic``,
``percussive``, ``effects.hpss`` as ``harmonic_percussive``) are
``AudioSamples`` methods.
"""

from __future__ import annotations

from typing import Any, Callable, Literal, Optional, Union

import numpy as np
import numpy.typing as npt
from numpy.typing import NDArray

#: Window specification: a SciPy window name, ``(name, parameter)``, a number
#: (Kaiser beta) or an array of coefficients.
WindowSpec = Union[str, tuple[str, float], float, npt.ArrayLike]
#: STFT padding mode used when ``center=True``.
PadMode = Literal["constant", "reflect", "edge", "symmetric"]

def phase_vocoder(
    D: npt.ArrayLike,
    *,
    rate: Optional[float] = None,
    t_out: Optional[npt.ArrayLike] = None,
    kind: Literal["linear"] = "linear",
    hop_length: Optional[int] = None,
    n_fft: Optional[int] = None,
) -> NDArray[np.complex128]:
    """Phase-vocoder time stretch of an STFT, ``librosa.phase_vocoder``.

    Args:
        D: complex ``(n_bins, n_frames)`` or ``(channels, n_bins, n_frames)``.
        rate: Speed-up factor (exactly one of ``rate`` / ``t_out``).
        t_out: Fractional output frame positions in ``[0, n_frames)``.
        hop_length: Deprecated in librosa 1.0 and unused; accepted with a
            ``FutureWarning`` when given, as librosa does.
        n_fft: Deprecated and unused, as ``hop_length``.
        kind: Must be ``"linear"``.

    Returns:
        numpy.ndarray[complex128]: ``(n_bins, ceil(n_frames / rate))`` (or
        ``len(t_out)`` frames), leading channel axis for 3-D input.
    """
    ...

def preemphasis(
    y: npt.ArrayLike,
    *,
    coef: float = 0.97,
    zi: Optional[npt.ArrayLike] = None,
    return_zf: bool = False,
) -> Any:
    """Pre-emphasis ``y[n] = x[n] - coef * x[n-1]``, ``librosa.effects.preemphasis``.

    Args:
        y: ``(n,)`` or ``(channels, n)``.
        coef: Coefficient.
        zi: Initial state (one value or one per channel); None = ``2 y[0] - y[1]``.
        return_zf: Also return the final state.

    Returns:
        numpy.ndarray[float64] of ``y``'s shape, or ``(y, zf)`` with ``zf`` of
        shape ``(1,)`` / ``(channels, 1)`` when ``return_zf``.
    """
    ...

def deemphasis(
    y: npt.ArrayLike,
    *,
    coef: float = 0.97,
    zi: Optional[npt.ArrayLike] = None,
    return_zf: bool = False,
) -> Any:
    """De-emphasis ``y[n] = x[n] + coef * y[n-1]``, ``librosa.effects.deemphasis``.

    Returns:
        As ``preemphasis``. With ``zi=None`` it exactly inverts ``preemphasis``'s
        default state. ``coef=3`` needs an explicit ``zi``.
    """
    ...

def softmask(
    X: npt.ArrayLike,
    X_ref: npt.ArrayLike,
    *,
    power: float = 1,
    split_zeros: bool = False,
) -> NDArray[Any]:
    """Soft mask ``X**p / (X**p + X_ref**p)``, ``librosa.util.softmask``.

    Returns:
        numpy.ndarray: float64 in ``[0, 1]``; bool (``X > X_ref``) for ``power=inf``.
    """
    ...

def hpss(
    S: npt.ArrayLike,
    *,
    kernel_size: Union[int, tuple[int, int]] = 31,
    power: float = 2.0,
    mask: bool = False,
    margin: Union[float, tuple[float, float]] = 1.0,
) -> tuple[NDArray[Any], NDArray[Any]]:
    """Median-filtering harmonic/percussive separation, ``librosa.decompose.hpss``.

    Args:
        S: Non-negative magnitude/power spectrogram or complex STFT,
            ``(n_bins, n_frames)`` or ``(channels, n_bins, n_frames)``.
        kernel_size: Median filter lengths (harmonic along time, percussive
            along frequency).
        power: Mask exponent (``numpy.inf`` = hard masks).
        mask: Return the masks instead of the components.
        margin: Separation margins (>= 1).

    Returns:
        tuple: ``(harmonic, percussive)`` of ``S``'s shape and dtype (complex
        input keeps its phase), or float64 masks (bool for ``power=inf``).
    """
    ...

def nn_filter(
    S: npt.ArrayLike,
    *,
    rec: Optional[npt.ArrayLike] = None,
    aggregate: Optional[Union[Callable[..., Any], Literal["mean", "median", "average"]]] = None,
    axis: int = -1,
    **kwargs: Any,
) -> NDArray[np.float64]:
    """Nearest-neighbour filtering, ``librosa.decompose.nn_filter``.

    Args:
        S: ``(n_features, n_frames)``.
        rec: Dense ``(n_frames, n_frames)`` recurrence matrix; None computes
            ``sequence.recurrence_matrix(S, **kwargs)``.
        aggregate: None/``numpy.mean``, ``numpy.median`` or ``numpy.average``
            (weighted by ``rec``).
        axis: Must be -1.
        **kwargs: ``recurrence_matrix`` arguments (``k``, ``width``, ``metric``,
            ``sym``, ``mode``, ``bandwidth``, ``self``, ``full``).

    Returns:
        numpy.ndarray[float64]: filtered ``(n_features, n_frames)``.
    """
    ...

def griffinlim(
    S: npt.ArrayLike,
    *,
    n_iter: int = 32,
    hop_length: Optional[int] = None,
    win_length: Optional[int] = None,
    n_fft: Optional[int] = None,
    window: WindowSpec = "hann",
    center: bool = True,
    dtype: Any = None,
    length: Optional[int] = None,
    pad_mode: PadMode = "constant",
    momentum: float = 0.99,
    init: Optional[Literal["random"]] = "random",
    rng: Optional[int] = None,
    random_state: Optional[int] = None,
) -> NDArray[np.float64]:
    """Fast Griffin-Lim phase reconstruction, ``librosa.griffinlim``.

    ``init=None`` (zero phase) matches librosa to round-off. Random phases come
    from this library's RNG (``rng`` is an integer seed; NumPy generators are
    rejected), so they differ from librosa's. ``dtype`` is ignored (float64 out).

    Returns:
        numpy.ndarray[float64]: ``(n,)`` for 2-D ``S``, ``(channels, n)`` for 3-D.
    """
    ...

def lpc(y: npt.ArrayLike, *, order: int, axis: int = -1) -> NDArray[np.float64]:
    """Burg LPC coefficients ``[1, a_1, ..., a_order]``, ``librosa.lpc``.

    Returns:
        numpy.ndarray[float64]: ``(order + 1,)`` (``(channels, order + 1)`` for 2-D ``y``).
    """
    ...

def clicks(
    *,
    times: Optional[npt.ArrayLike] = None,
    frames: Optional[npt.ArrayLike] = None,
    sr: float = 22050,
    hop_length: int = 512,
    click_freq: float = 1000.0,
    click_duration: float = 0.1,
    click: Optional[npt.ArrayLike] = None,
    length: Optional[int] = None,
) -> NDArray[np.float64]:
    """Click track, ``librosa.clicks`` (float64 output; librosa returns float32).

    ``sr`` must be a positive integer.

    Returns:
        numpy.ndarray[float64]: 1-D signal.
    """
    ...
