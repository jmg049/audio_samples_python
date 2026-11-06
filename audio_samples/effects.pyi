"""Effect and decomposition functions over NumPy arrays.

Free functions for the phase vocoder (``phase_vocoder``), pre- and
de-emphasis with filter state (``preemphasis`` / ``deemphasis`` with ``zi`` /
``return_zf``), soft masks (``softmask``), harmonic/percussive separation and
nearest-neighbour filtering of spectrograms (``hpss``, ``nn_filter``),
Griffin-Lim reconstruction (``griffinlim``), linear prediction (``lpc``) and
click tracks (``clicks``). ``griffinlim`` defaults to ``rng=0`` (repeatable
output); ``clicks`` requires ``sr`` and, with ``frames``, ``hop_length``.
float32 / complex64 NumPy arrays are processed in float32 and give float32 /
complex64 output; any other array-like (lists, integer, float16 or float64
arrays) is converted to float64 and gives float64 / complex128 output.
``clicks`` returns float64. The audio-to-audio effects (``time_stretch``, ``pitch_shift``,
``trim``/``split`` as ``trim_db``/``split_db``, ``remix``, ``harmonic``,
``percussive``, ``effects.hpss`` as ``harmonic_percussive``) are
``AudioSamples`` methods.
"""

from __future__ import annotations

from typing import Sequence

from typing import Any, Callable, Literal, Optional, Union

import numpy as np
import numpy.typing as npt
from numpy.typing import NDArray
from typing import Any, overload, type_check_only
from typing_extensions import TypeAlias

_RngLike = Union[int, None, np.random.Generator, np.random.RandomState, np.random.SeedSequence, np.random.BitGenerator, Sequence[int]]

#: Matches only an array whose dtype is unknown to the type checker (Any). The overloads use it so that
#: such an array gives a result of unknown precision instead of the first overload's.
@type_check_only
class _UnknownDtype(np.generic):
    def __new__(cls) -> _UnknownDtype: ...

#: float32 and complex64 arrays are processed in float32. Anything else is converted to float64.
_SingleArray: TypeAlias = NDArray[np.float32] | NDArray[np.complex64]


#: Window specification: a SciPy window name, ``(name, parameter)``, a number
#: (Kaiser beta) or an array of coefficients.
_WindowSpec = Union[str, tuple[str, float], float, npt.ArrayLike]
#: STFT padding mode used when ``center=True``.
_PadMode = Literal["constant", "reflect", "edge", "symmetric"]

@overload
def phase_vocoder(
    D: NDArray[_UnknownDtype],
    *,
    rate: Optional[float] = None,
    t_out: Optional[npt.ArrayLike] = None,
    kind: Literal['linear'] = 'linear',
    hop_length: Optional[int] = None,
    n_fft: Optional[int] = None,
) -> NDArray[Any]: ...
@overload
def phase_vocoder(  # type: ignore[overload-overlap]
    D: _SingleArray,
    *,
    rate: Optional[float] = None,
    t_out: Optional[npt.ArrayLike] = None,
    kind: Literal['linear'] = 'linear',
    hop_length: Optional[int] = None,
    n_fft: Optional[int] = None,
) -> NDArray[np.complex64]: ...
@overload
def phase_vocoder(
    D: npt.ArrayLike,
    *,
    rate: Optional[float] = None,
    t_out: Optional[npt.ArrayLike] = None,
    kind: Literal['linear'] = 'linear',
    hop_length: Optional[int] = None,
    n_fft: Optional[int] = None,
) -> NDArray[np.complex128]:
    """Phase-vocoder time stretch of an STFT.

    Args:
        D: complex ``(n_bins, n_frames)`` or ``(channels, n_bins, n_frames)``.
        rate: Speed-up factor (exactly one of ``rate`` / ``t_out``).
        t_out: Fractional output frame positions in ``[0, n_frames)``.
        hop_length: Deprecated and unused; accepted with a ``FutureWarning``
            when given.
        n_fft: Deprecated and unused, as ``hop_length``.
        kind: Must be ``"linear"``.

    Returns:
        numpy.ndarray[complex64 | complex128]: ``(n_bins, ceil(n_frames / rate))`` (or
        ``len(t_out)`` frames), leading channel axis for 3-D input.
    """
    ...

@overload
def preemphasis(
    y: NDArray[_UnknownDtype],
    *,
    coef: float = 0.97,
    zi: Optional[npt.ArrayLike] = None,
    return_zf: Literal[False] = False,
) -> NDArray[Any]: ...
@overload
def preemphasis(
    y: NDArray[_UnknownDtype],
    *,
    coef: float = 0.97,
    zi: Optional[npt.ArrayLike] = None,
    return_zf: Literal[True],
) -> tuple[NDArray[Any], NDArray[Any]]: ...
@overload
def preemphasis(  # type: ignore[overload-overlap]
    y: _SingleArray,
    *,
    coef: float = 0.97,
    zi: Optional[npt.ArrayLike] = None,
    return_zf: Literal[False] = False,
) -> NDArray[np.float32]: ...
@overload
def preemphasis(  # type: ignore[overload-overlap]
    y: _SingleArray,
    *,
    coef: float = 0.97,
    zi: Optional[npt.ArrayLike] = None,
    return_zf: Literal[True],
) -> tuple[NDArray[np.float32], NDArray[np.float32]]: ...
@overload
def preemphasis(
    y: npt.ArrayLike,
    *,
    coef: float = 0.97,
    zi: Optional[npt.ArrayLike] = None,
    return_zf: Literal[False] = False,
) -> NDArray[np.float64]: ...
@overload
def preemphasis(
    y: npt.ArrayLike,
    *,
    coef: float = 0.97,
    zi: Optional[npt.ArrayLike] = None,
    return_zf: Literal[True],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Pre-emphasis ``y[n] = x[n] - coef * x[n-1]``.

    Args:
        y: ``(n,)`` or ``(channels, n)``.
        coef: Coefficient.
        zi: Initial state (one value or one per channel); None = ``2 y[0] - y[1]``.
        return_zf: Also return the final state.

    Returns:
        numpy.ndarray[float32 | float64] of ``y``'s shape, or ``(y, zf)`` with ``zf`` of
        shape ``(1,)`` / ``(channels, 1)`` when ``return_zf``.
    """
    ...

@overload
def deemphasis(
    y: NDArray[_UnknownDtype],
    *,
    coef: float = 0.97,
    zi: Optional[npt.ArrayLike] = None,
    return_zf: Literal[False] = False,
) -> NDArray[Any]: ...
@overload
def deemphasis(
    y: NDArray[_UnknownDtype],
    *,
    coef: float = 0.97,
    zi: Optional[npt.ArrayLike] = None,
    return_zf: Literal[True],
) -> tuple[NDArray[Any], NDArray[Any]]: ...
@overload
def deemphasis(  # type: ignore[overload-overlap]
    y: _SingleArray,
    *,
    coef: float = 0.97,
    zi: Optional[npt.ArrayLike] = None,
    return_zf: Literal[False] = False,
) -> NDArray[np.float32]: ...
@overload
def deemphasis(  # type: ignore[overload-overlap]
    y: _SingleArray,
    *,
    coef: float = 0.97,
    zi: Optional[npt.ArrayLike] = None,
    return_zf: Literal[True],
) -> tuple[NDArray[np.float32], NDArray[np.float32]]: ...
@overload
def deemphasis(
    y: npt.ArrayLike,
    *,
    coef: float = 0.97,
    zi: Optional[npt.ArrayLike] = None,
    return_zf: Literal[False] = False,
) -> NDArray[np.float64]: ...
@overload
def deemphasis(
    y: npt.ArrayLike,
    *,
    coef: float = 0.97,
    zi: Optional[npt.ArrayLike] = None,
    return_zf: Literal[True],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """De-emphasis ``y[n] = x[n] + coef * y[n-1]``.

    Returns:
        As ``preemphasis``. With ``zi=None`` it exactly inverts ``preemphasis``'s
        default state. ``coef=3`` needs an explicit ``zi``.
    """
    ...

@overload
def softmask(
    X: NDArray[_UnknownDtype],
    X_ref: npt.ArrayLike,
    *,
    power: float = 1,
    split_zeros: bool = False,
) -> NDArray[Any]: ...
@overload
def softmask(  # type: ignore[overload-overlap]
    X: _SingleArray,
    X_ref: npt.ArrayLike,
    *,
    power: float = 1,
    split_zeros: bool = False,
) -> NDArray[np.float32]: ...
@overload
def softmask(
    X: npt.ArrayLike,
    X_ref: npt.ArrayLike,
    *,
    power: float = 1,
    split_zeros: bool = False,
) -> NDArray[np.float64]:
    """Soft mask ``X**p / (X**p + X_ref**p)``.

    Returns:
        numpy.ndarray: values in ``[0, 1]`` in ``X``'s precision; bool (``X > X_ref``)
        for ``power=inf``.
    """
    ...

@overload
def hpss(
    S: NDArray[_UnknownDtype],
    *,
    kernel_size: Union[int, tuple[int, int]] = 31,
    power: float = 2.0,
    mask: bool = False,
    margin: Union[float, tuple[float, float]] = 1.0,
) -> tuple[NDArray[Any], NDArray[Any]]: ...
@overload
def hpss(  # type: ignore[overload-overlap]
    S: NDArray[np.float32],
    *,
    kernel_size: Union[int, tuple[int, int]] = 31,
    power: float = 2.0,
    mask: bool = False,
    margin: Union[float, tuple[float, float]] = 1.0,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]: ...
@overload
def hpss(  # type: ignore[overload-overlap]
    S: NDArray[np.complex64],
    *,
    kernel_size: Union[int, tuple[int, int]] = 31,
    power: float = 2.0,
    mask: Literal[False] = False,
    margin: Union[float, tuple[float, float]] = 1.0,
) -> tuple[NDArray[np.complex64], NDArray[np.complex64]]: ...
@overload
def hpss(  # type: ignore[overload-overlap]
    S: NDArray[np.complex64],
    *,
    kernel_size: Union[int, tuple[int, int]] = 31,
    power: float = 2.0,
    mask: Literal[True],
    margin: Union[float, tuple[float, float]] = 1.0,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]: ...
@overload
def hpss(  # type: ignore[overload-overlap]
    S: NDArray[np.complex128],
    *,
    kernel_size: Union[int, tuple[int, int]] = 31,
    power: float = 2.0,
    mask: Literal[False] = False,
    margin: Union[float, tuple[float, float]] = 1.0,
) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]: ...
@overload
def hpss(
    S: NDArray[np.complex128],
    *,
    kernel_size: Union[int, tuple[int, int]] = 31,
    power: float = 2.0,
    mask: Literal[True],
    margin: Union[float, tuple[float, float]] = 1.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...
@overload
def hpss(
    S: npt.ArrayLike,
    *,
    kernel_size: Union[int, tuple[int, int]] = 31,
    power: float = 2.0,
    mask: bool = False,
    margin: Union[float, tuple[float, float]] = 1.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Median-filtering harmonic/percussive separation of a spectrogram.

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
        input keeps its phase), or masks in ``S``'s precision (bool for
        ``power=inf``).
    """
    ...

@overload
def nn_filter(
    S: NDArray[_UnknownDtype],
    *,
    rec: Optional[npt.ArrayLike] = None,
    aggregate: Optional[Union[Callable[..., Any], Literal['mean', 'median', 'average', 'max', 'min', 'sum']]] = None,
    axis: int = -1,
    **kwargs: Any,
) -> NDArray[Any]: ...
@overload
def nn_filter(
    S: _SingleArray,
    *,
    rec: Optional[npt.ArrayLike] = None,
    aggregate: Optional[Union[Callable[..., Any], Literal['mean', 'median', 'average', 'max', 'min', 'sum']]] = None,
    axis: int = -1,
    **kwargs: Any,
) -> NDArray[np.float32]: ...
@overload
def nn_filter(
    S: npt.ArrayLike,
    *,
    rec: Optional[npt.ArrayLike] = None,
    aggregate: Optional[Union[Callable[..., Any], Literal['mean', 'median', 'average', 'max', 'min', 'sum']]] = None,
    axis: int = -1,
    **kwargs: Any,
) -> NDArray[np.float64]:
    """Nearest-neighbour filtering: each frame replaced by an aggregate of its
    most similar frames.

    Args:
        S: ``(n_features, n_frames)``.
        rec: ``(n_frames, n_frames)`` recurrence matrix, dense or ``scipy.sparse``
            (every stored entry is a neighbour); None computes
            ``sequence.recurrence_matrix(S, **kwargs)``.
        aggregate: None/``numpy.mean``, ``numpy.median``, ``numpy.average``
            (weighted by ``rec``), ``numpy.max``, ``numpy.min``, ``numpy.sum``, or any
            callable ``f(neighbours, axis=0)`` of the ``(n_neighbours, n_features)``
            neighbours, which runs with the GIL held.
        axis: Must be -1.
        **kwargs: ``recurrence_matrix`` arguments (``k``, ``width``, ``metric``,
            ``sym``, ``mode``, ``bandwidth``, ``self``, ``full``).

    Returns:
        numpy.ndarray[float32 | float64]: filtered ``(n_features, n_frames)``.
    """
    ...

@overload
def griffinlim(
    S: NDArray[_UnknownDtype],
    *,
    n_iter: int = 32,
    hop_length: Optional[int] = None,
    win_length: Optional[int] = None,
    n_fft: Optional[int] = None,
    window: _WindowSpec = 'hann',
    center: bool = True,
    dtype: Any = None,
    length: Optional[int] = None,
    pad_mode: _PadMode = 'constant',
    momentum: float = 0.99,
    init: Optional[Literal['random']] = 'random',
    rng: _RngLike = 0,
    random_state: _RngLike = None,
) -> NDArray[Any]: ...
@overload
def griffinlim(  # type: ignore[overload-overlap]
    S: _SingleArray,
    *,
    n_iter: int = 32,
    hop_length: Optional[int] = None,
    win_length: Optional[int] = None,
    n_fft: Optional[int] = None,
    window: _WindowSpec = 'hann',
    center: bool = True,
    dtype: Any = None,
    length: Optional[int] = None,
    pad_mode: _PadMode = 'constant',
    momentum: float = 0.99,
    init: Optional[Literal['random']] = 'random',
    rng: _RngLike = 0,
    random_state: _RngLike = None,
) -> NDArray[np.float32]: ...
@overload
def griffinlim(
    S: npt.ArrayLike,
    *,
    n_iter: int = 32,
    hop_length: Optional[int] = None,
    win_length: Optional[int] = None,
    n_fft: Optional[int] = None,
    window: _WindowSpec = 'hann',
    center: bool = True,
    dtype: Any = None,
    length: Optional[int] = None,
    pad_mode: _PadMode = 'constant',
    momentum: float = 0.99,
    init: Optional[Literal['random']] = 'random',
    rng: _RngLike = 0,
    random_state: _RngLike = None,
) -> NDArray[np.float64]:
    """Fast Griffin-Lim phase reconstruction from an STFT magnitude.

    ``init=None`` starts from zero phase; ``init="random"`` (default) draws the
    initial phases from this library's RNG. ``rng`` is an integer seed, None, or a
    NumPy generator (or anything ``numpy.random.default_rng`` takes), from
    which one seed is drawn. ``rng`` defaults to the seed 0, so repeated calls
    give the same output; ``rng=None`` draws a new seed on every call. The
    output is float32 for a float32 ``S`` and float64 otherwise; ``dtype`` is
    ignored.

    Returns:
        numpy.ndarray[float32 | float64]: ``(n,)`` for 2-D ``S``, ``(channels, n)`` for 3-D.
    """
    ...

@overload
def lpc(y: NDArray[_UnknownDtype], *, order: int, axis: int=-1) -> NDArray[Any]: ...
@overload
def lpc(  # type: ignore[overload-overlap]
    y: _SingleArray,
    *,
    order: int,
    axis: int = -1,
) -> NDArray[np.float32]: ...
@overload
def lpc(y: npt.ArrayLike, *, order: int, axis: int=-1) -> NDArray[np.float64]:
    """Linear prediction coefficients ``[1, a_1, ..., a_order]`` by Burg's method.

    Returns:
        numpy.ndarray[float32 | float64]: ``(order + 1,)`` (``(channels, order + 1)`` for 2-D ``y``).
    """
    ...

@overload
def clicks(  # type: ignore[overload-overlap]
    *,
    times: Optional[npt.ArrayLike] = None,
    frames: Optional[npt.ArrayLike] = None,
    sr: float,
    hop_length: Optional[int] = None,
    click_freq: float = 1000.0,
    click_duration: float = 0.1,
    click: Optional[npt.ArrayLike] = None,
    length: Optional[int] = None,
    dtype: type[np.float32],
) -> NDArray[np.float32]: ...
@overload
def clicks(
    *,
    times: Optional[npt.ArrayLike] = None,
    frames: Optional[npt.ArrayLike] = None,
    sr: float,
    hop_length: Optional[int] = None,
    click_freq: float = 1000.0,
    click_duration: float = 0.1,
    click: Optional[npt.ArrayLike] = None,
    length: Optional[int] = None,
    dtype: None | type[np.float64] = None,
) -> NDArray[np.float64]:
    """Click track: a click placed at each of ``times`` or ``frames`` (float64
    output by default).

    ``sr`` must be a positive integer.
    ``hop_length`` is the hop of the analysis that produced ``frames``; it is
    required with ``frames`` and unused with ``times``. ``dtype`` selects
    ``numpy.float32`` or ``numpy.float64`` output; None gives float64.

    Returns:
        numpy.ndarray[float64 | float32]: 1-D signal.
    """
    ...

__all__ = [
    "__doc__",
    "clicks",
    "deemphasis",
    "griffinlim",
    "hpss",
    "lpc",
    "nn_filter",
    "phase_vocoder",
    "preemphasis",
    "softmask",
]
