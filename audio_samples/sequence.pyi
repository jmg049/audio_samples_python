"""Sequence alignment and decoding (``librosa.sequence``, ``librosa.segment``,
``librosa.util.sync``).

Feature matrices use librosa's layout ``(n_features, n_frames)``; 1-D input is
one feature. Probability matrices are ``(n_states, n_steps)``. Return values
follow librosa: tuples only when ``backtrack`` / ``return_steps`` /
``return_logp`` ask for them. Index outputs are int64 (librosa: uint16 states,
intp paths). Defaults are librosa 1.0.0's. Float outputs keep the input's
precision: a float32 array gives float32 costs, scores and matrices (librosa
accumulates in float64), any other input float64. Transition matrices are
float64.

Differences from librosa: dense output only (``sparse=True`` raises), only
``axis=-1``; ``sync`` with an empty index list returns one segment; ``rqa``
follows the move actually taken in the second column (librosa stores a wrong
back-pointer there).
"""

from __future__ import annotations

from typing import Any, Callable, Literal, Optional, Sequence, Union

import numpy as np
import numpy.typing as npt
from numpy.typing import NDArray
from typing import Any, overload, type_check_only
from typing_extensions import TypeAlias

#: Matches only an array whose dtype is unknown to the type checker (Any). The overloads use it so that
#: such an array gives a result of unknown precision instead of the first overload's.
@type_check_only
class _UnknownDtype(np.generic):
    def __new__(cls) -> _UnknownDtype: ...

#: float32 and complex64 arrays are processed in float32. Anything else is converted to float64.
_SingleArray: TypeAlias = NDArray[np.float32] | NDArray[np.complex64]


_Metric = Literal["euclidean", "sqeuclidean", "cosine", "cityblock", "chebyshev"]
_Mode = Literal["connectivity", "distance", "affinity"]
#: Affinity bandwidth: None (``"med_k_scalar"``), a heuristic name, a positive
#: number, or a per-link array.
_BandwidthSpec = Union[
    None,
    float,
    Literal["med_k_scalar", "mean_k", "gmean_k", "mean_k_avg", "gmean_k_avg", "mean_k_avg_and_pair"],
    npt.ArrayLike,
]

@overload
def dtw(
    X: NDArray[_UnknownDtype],
    Y: npt.ArrayLike,
    *,
    C: None = None,
    metric: _Metric = 'euclidean',
    step_sizes_sigma: Optional[npt.ArrayLike] = None,
    weights_add: Optional[npt.ArrayLike] = None,
    weights_mul: Optional[npt.ArrayLike] = None,
    subseq: bool = False,
    backtrack: Literal[True] = True,
    global_constraints: bool = False,
    band_rad: float = 0.25,
    return_steps: Literal[False] = False,
) -> tuple[NDArray[Any], NDArray[Any]]: ...
@overload
def dtw(
    X: NDArray[_UnknownDtype],
    Y: npt.ArrayLike,
    *,
    C: None = None,
    metric: _Metric = 'euclidean',
    step_sizes_sigma: Optional[npt.ArrayLike] = None,
    weights_add: Optional[npt.ArrayLike] = None,
    weights_mul: Optional[npt.ArrayLike] = None,
    subseq: bool = False,
    backtrack: Literal[True] = True,
    global_constraints: bool = False,
    band_rad: float = 0.25,
    return_steps: Literal[True],
) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any]]: ...
@overload
def dtw(
    X: NDArray[_UnknownDtype],
    Y: npt.ArrayLike,
    *,
    C: None = None,
    metric: _Metric = 'euclidean',
    step_sizes_sigma: Optional[npt.ArrayLike] = None,
    weights_add: Optional[npt.ArrayLike] = None,
    weights_mul: Optional[npt.ArrayLike] = None,
    subseq: bool = False,
    backtrack: Literal[False],
    global_constraints: bool = False,
    band_rad: float = 0.25,
    return_steps: bool = False,
) -> NDArray[Any]: ...
@overload
def dtw(
    X: None = None,
    Y: None = None,
    *,
    C: NDArray[_UnknownDtype],
    metric: _Metric = 'euclidean',
    step_sizes_sigma: Optional[npt.ArrayLike] = None,
    weights_add: Optional[npt.ArrayLike] = None,
    weights_mul: Optional[npt.ArrayLike] = None,
    subseq: bool = False,
    backtrack: Literal[True] = True,
    global_constraints: bool = False,
    band_rad: float = 0.25,
    return_steps: Literal[False] = False,
) -> tuple[NDArray[Any], NDArray[Any]]: ...
@overload
def dtw(
    X: None = None,
    Y: None = None,
    *,
    C: NDArray[_UnknownDtype],
    metric: _Metric = 'euclidean',
    step_sizes_sigma: Optional[npt.ArrayLike] = None,
    weights_add: Optional[npt.ArrayLike] = None,
    weights_mul: Optional[npt.ArrayLike] = None,
    subseq: bool = False,
    backtrack: Literal[True] = True,
    global_constraints: bool = False,
    band_rad: float = 0.25,
    return_steps: Literal[True],
) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any]]: ...
@overload
def dtw(
    X: None = None,
    Y: None = None,
    *,
    C: NDArray[_UnknownDtype],
    metric: _Metric = 'euclidean',
    step_sizes_sigma: Optional[npt.ArrayLike] = None,
    weights_add: Optional[npt.ArrayLike] = None,
    weights_mul: Optional[npt.ArrayLike] = None,
    subseq: bool = False,
    backtrack: Literal[False],
    global_constraints: bool = False,
    band_rad: float = 0.25,
    return_steps: bool = False,
) -> NDArray[Any]: ...
@overload
def dtw(  # type: ignore[overload-overlap]
    X: _SingleArray,
    Y: npt.ArrayLike,
    *,
    C: None = None,
    metric: _Metric = 'euclidean',
    step_sizes_sigma: Optional[npt.ArrayLike] = None,
    weights_add: Optional[npt.ArrayLike] = None,
    weights_mul: Optional[npt.ArrayLike] = None,
    subseq: bool = False,
    backtrack: Literal[True] = True,
    global_constraints: bool = False,
    band_rad: float = 0.25,
    return_steps: Literal[False] = False,
) -> tuple[NDArray[np.float32], NDArray[np.int64]]: ...
@overload
def dtw(  # type: ignore[overload-overlap]
    X: _SingleArray,
    Y: npt.ArrayLike,
    *,
    C: None = None,
    metric: _Metric = 'euclidean',
    step_sizes_sigma: Optional[npt.ArrayLike] = None,
    weights_add: Optional[npt.ArrayLike] = None,
    weights_mul: Optional[npt.ArrayLike] = None,
    subseq: bool = False,
    backtrack: Literal[True] = True,
    global_constraints: bool = False,
    band_rad: float = 0.25,
    return_steps: Literal[True],
) -> tuple[NDArray[np.float32], NDArray[np.int64], NDArray[np.int64]]: ...
@overload
def dtw(  # type: ignore[overload-overlap]
    X: _SingleArray,
    Y: npt.ArrayLike,
    *,
    C: None = None,
    metric: _Metric = 'euclidean',
    step_sizes_sigma: Optional[npt.ArrayLike] = None,
    weights_add: Optional[npt.ArrayLike] = None,
    weights_mul: Optional[npt.ArrayLike] = None,
    subseq: bool = False,
    backtrack: Literal[False],
    global_constraints: bool = False,
    band_rad: float = 0.25,
    return_steps: bool = False,
) -> NDArray[np.float32]: ...
@overload
def dtw(
    X: npt.ArrayLike,
    Y: npt.ArrayLike,
    *,
    C: None = None,
    metric: _Metric = 'euclidean',
    step_sizes_sigma: Optional[npt.ArrayLike] = None,
    weights_add: Optional[npt.ArrayLike] = None,
    weights_mul: Optional[npt.ArrayLike] = None,
    subseq: bool = False,
    backtrack: Literal[True] = True,
    global_constraints: bool = False,
    band_rad: float = 0.25,
    return_steps: Literal[False] = False,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]: ...
@overload
def dtw(
    X: npt.ArrayLike,
    Y: npt.ArrayLike,
    *,
    C: None = None,
    metric: _Metric = 'euclidean',
    step_sizes_sigma: Optional[npt.ArrayLike] = None,
    weights_add: Optional[npt.ArrayLike] = None,
    weights_mul: Optional[npt.ArrayLike] = None,
    subseq: bool = False,
    backtrack: Literal[True] = True,
    global_constraints: bool = False,
    band_rad: float = 0.25,
    return_steps: Literal[True],
) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.int64]]: ...
@overload
def dtw(
    X: npt.ArrayLike,
    Y: npt.ArrayLike,
    *,
    C: None = None,
    metric: _Metric = 'euclidean',
    step_sizes_sigma: Optional[npt.ArrayLike] = None,
    weights_add: Optional[npt.ArrayLike] = None,
    weights_mul: Optional[npt.ArrayLike] = None,
    subseq: bool = False,
    backtrack: Literal[False],
    global_constraints: bool = False,
    band_rad: float = 0.25,
    return_steps: bool = False,
) -> NDArray[np.float64]: ...
@overload
def dtw(  # type: ignore[overload-overlap]
    X: None = None,
    Y: None = None,
    *,
    C: _SingleArray,
    metric: _Metric = 'euclidean',
    step_sizes_sigma: Optional[npt.ArrayLike] = None,
    weights_add: Optional[npt.ArrayLike] = None,
    weights_mul: Optional[npt.ArrayLike] = None,
    subseq: bool = False,
    backtrack: Literal[True] = True,
    global_constraints: bool = False,
    band_rad: float = 0.25,
    return_steps: Literal[False] = False,
) -> tuple[NDArray[np.float32], NDArray[np.int64]]: ...
@overload
def dtw(  # type: ignore[overload-overlap]
    X: None = None,
    Y: None = None,
    *,
    C: _SingleArray,
    metric: _Metric = 'euclidean',
    step_sizes_sigma: Optional[npt.ArrayLike] = None,
    weights_add: Optional[npt.ArrayLike] = None,
    weights_mul: Optional[npt.ArrayLike] = None,
    subseq: bool = False,
    backtrack: Literal[True] = True,
    global_constraints: bool = False,
    band_rad: float = 0.25,
    return_steps: Literal[True],
) -> tuple[NDArray[np.float32], NDArray[np.int64], NDArray[np.int64]]: ...
@overload
def dtw(  # type: ignore[overload-overlap]
    X: None = None,
    Y: None = None,
    *,
    C: _SingleArray,
    metric: _Metric = 'euclidean',
    step_sizes_sigma: Optional[npt.ArrayLike] = None,
    weights_add: Optional[npt.ArrayLike] = None,
    weights_mul: Optional[npt.ArrayLike] = None,
    subseq: bool = False,
    backtrack: Literal[False],
    global_constraints: bool = False,
    band_rad: float = 0.25,
    return_steps: bool = False,
) -> NDArray[np.float32]: ...
@overload
def dtw(
    X: None = None,
    Y: None = None,
    *,
    C: npt.ArrayLike,
    metric: _Metric = 'euclidean',
    step_sizes_sigma: Optional[npt.ArrayLike] = None,
    weights_add: Optional[npt.ArrayLike] = None,
    weights_mul: Optional[npt.ArrayLike] = None,
    subseq: bool = False,
    backtrack: Literal[True] = True,
    global_constraints: bool = False,
    band_rad: float = 0.25,
    return_steps: Literal[False] = False,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]: ...
@overload
def dtw(
    X: None = None,
    Y: None = None,
    *,
    C: npt.ArrayLike,
    metric: _Metric = 'euclidean',
    step_sizes_sigma: Optional[npt.ArrayLike] = None,
    weights_add: Optional[npt.ArrayLike] = None,
    weights_mul: Optional[npt.ArrayLike] = None,
    subseq: bool = False,
    backtrack: Literal[True] = True,
    global_constraints: bool = False,
    band_rad: float = 0.25,
    return_steps: Literal[True],
) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.int64]]: ...
@overload
def dtw(
    X: None = None,
    Y: None = None,
    *,
    C: npt.ArrayLike,
    metric: _Metric = 'euclidean',
    step_sizes_sigma: Optional[npt.ArrayLike] = None,
    weights_add: Optional[npt.ArrayLike] = None,
    weights_mul: Optional[npt.ArrayLike] = None,
    subseq: bool = False,
    backtrack: Literal[False],
    global_constraints: bool = False,
    band_rad: float = 0.25,
    return_steps: bool = False,
) -> NDArray[np.float64]:
    """Dynamic time warping, equivalent to ``librosa.sequence.dtw``.

    Pass ``X`` ``(n_features, N)`` and ``Y`` ``(n_features, M)``, or a cost matrix
    ``C`` ``(N, M)``.

    Returns:
        ``D`` (numpy.ndarray[float32 | float64] ``(N, M)``, unreachable cells ``inf``) when
        ``backtrack=False``; otherwise ``(D, wp)`` with ``wp`` an int64
        ``(path_length, 2)`` array ordered from the end of the alignment to the
        start; ``return_steps=True`` appends the int64 step matrix.

    Raises:
        ValueError: Incompatible inputs, NaN costs, or no valid path.
    """
    ...

def dtw_backtracking(
    steps: npt.ArrayLike,
    *,
    step_sizes_sigma: Optional[npt.ArrayLike] = None,
    subseq: bool = False,
    start: Optional[int] = None,
) -> NDArray[np.int64]:
    """Warping path from a DTW step matrix (``librosa.sequence.dtw_backtracking``).

    Returns:
        numpy.ndarray[int64], shape ``(path_length, 2)``, end first.
    """
    ...

@overload
def viterbi(
    prob: npt.ArrayLike,
    transition: npt.ArrayLike,
    *,
    p_init: Optional[npt.ArrayLike] = None,
    return_logp: Literal[False] = False,
    transition_min_prob: Optional[float] = None,
) -> NDArray[np.int64]: ...
@overload
def viterbi(
    prob: npt.ArrayLike,
    transition: npt.ArrayLike,
    *,
    p_init: Optional[npt.ArrayLike] = None,
    return_logp: Literal[True],
    transition_min_prob: Optional[float] = None,
) -> tuple[NDArray[np.int64], float]:
    """Viterbi decoding of an HMM (``librosa.sequence.viterbi``).

    Args:
        prob: ``(n_states, n_steps)`` observation likelihoods in [0, 1].
        transition: ``(n_states, n_states)`` row-stochastic matrix.
        p_init: Initial distribution; None = uniform.
        return_logp: Also return the log-likelihood (a float).
        transition_min_prob: Prune transitions below this probability.

    Returns:
        numpy.ndarray[int64] ``(n_steps,)``, or ``(states, logp)``.

    Raises:
        ValueError: Shape mismatch or invalid probabilities.
    """
    ...

@overload
def viterbi_discriminative(
    prob: npt.ArrayLike,
    transition: npt.ArrayLike,
    *,
    p_state: Optional[npt.ArrayLike] = None,
    p_init: Optional[npt.ArrayLike] = None,
    return_logp: Literal[False] = False,
    transition_min_prob: Optional[float] = None,
) -> NDArray[np.int64]: ...
@overload
def viterbi_discriminative(
    prob: npt.ArrayLike,
    transition: npt.ArrayLike,
    *,
    p_state: Optional[npt.ArrayLike] = None,
    p_init: Optional[npt.ArrayLike] = None,
    return_logp: Literal[True],
    transition_min_prob: Optional[float] = None,
) -> tuple[NDArray[np.int64], float]:
    """Viterbi decoding from posterior state probabilities
    (``librosa.sequence.viterbi_discriminative``); each column of ``prob`` sums to 1.

    Returns:
        numpy.ndarray[int64] ``(n_steps,)``, or ``(states, logp)`` with an
        unnormalised float log-likelihood.
    """
    ...

@overload
def viterbi_binary(
    prob: npt.ArrayLike,
    transition: npt.ArrayLike,
    *,
    p_state: Optional[Union[float, npt.ArrayLike]] = None,
    p_init: Optional[Union[float, npt.ArrayLike]] = None,
    return_logp: Literal[False] = False,
    transition_min_prob: Optional[float] = None,
    dtype: None = None,
) -> NDArray[np.bool_]: ...
@overload
def viterbi_binary(
    prob: npt.ArrayLike,
    transition: npt.ArrayLike,
    *,
    p_state: Optional[Union[float, npt.ArrayLike]] = None,
    p_init: Optional[Union[float, npt.ArrayLike]] = None,
    return_logp: Literal[True],
    transition_min_prob: Optional[float] = None,
    dtype: None = None,
) -> tuple[NDArray[np.bool_], NDArray[np.float64]]: ...
@overload
def viterbi_binary(
    prob: npt.ArrayLike,
    transition: npt.ArrayLike,
    *,
    p_state: Optional[Union[float, npt.ArrayLike]] = None,
    p_init: Optional[Union[float, npt.ArrayLike]] = None,
    return_logp: Literal[False] = False,
    transition_min_prob: Optional[float] = None,
    dtype: type[np.uint16],
) -> NDArray[np.uint16]: ...
@overload
def viterbi_binary(
    prob: npt.ArrayLike,
    transition: npt.ArrayLike,
    *,
    p_state: Optional[Union[float, npt.ArrayLike]] = None,
    p_init: Optional[Union[float, npt.ArrayLike]] = None,
    return_logp: Literal[True],
    transition_min_prob: Optional[float] = None,
    dtype: type[np.uint16],
) -> tuple[NDArray[np.uint16], NDArray[np.float64]]: ...
@overload
def viterbi_binary(
    prob: npt.ArrayLike,
    transition: npt.ArrayLike,
    *,
    p_state: Optional[Union[float, npt.ArrayLike]] = None,
    p_init: Optional[Union[float, npt.ArrayLike]] = None,
    return_logp: Literal[False] = False,
    transition_min_prob: Optional[float] = None,
    dtype: type[np.int64],
) -> NDArray[np.int64]: ...
@overload
def viterbi_binary(
    prob: npt.ArrayLike,
    transition: npt.ArrayLike,
    *,
    p_state: Optional[Union[float, npt.ArrayLike]] = None,
    p_init: Optional[Union[float, npt.ArrayLike]] = None,
    return_logp: Literal[True],
    transition_min_prob: Optional[float] = None,
    dtype: type[np.int64],
) -> tuple[NDArray[np.int64], NDArray[np.float64]]: ...
@overload
def viterbi_binary(
    prob: npt.ArrayLike,
    transition: npt.ArrayLike,
    *,
    p_state: Optional[Union[float, npt.ArrayLike]] = None,
    p_init: Optional[Union[float, npt.ArrayLike]] = None,
    return_logp: bool = False,
    transition_min_prob: Optional[float] = None,
    dtype: Optional[npt.DTypeLike] = None,
) -> NDArray[Any] | tuple[NDArray[Any], NDArray[np.float64]]:
    """Independent off/on Viterbi per state (``librosa.sequence.viterbi_binary``).

    Args:
        prob: ``(n_states, n_steps)`` activation probabilities (1-D = one state).
        transition: ``(2, 2)`` shared or ``(n_states, 2, 2)``.
        dtype: dtype of the states: ``bool`` (default None, one byte per
            entry), ``numpy.uint16`` (librosa's dtype) or ``numpy.int64``.

    Returns:
        numpy.ndarray[bool | uint16 | int64] ``(n_states, n_steps)``
        (``(n_steps,)`` for 1-D input), true or 1 where the state is on, or
        ``(states, logp)`` with ``logp`` ``(n_states,)``.
    """
    ...

def transition_uniform(n_states: int) -> NDArray[np.float64]:
    """``(n_states, n_states)`` matrix of ``1 / n_states``."""
    ...

def transition_loop(n_states: int, prob: Union[float, npt.ArrayLike]) -> NDArray[np.float64]:
    """Self-loop transition matrix (``librosa.sequence.transition_loop``), ``(n_states, n_states)``."""
    ...

def transition_cycle(n_states: int, prob: Union[float, npt.ArrayLike]) -> NDArray[np.float64]:
    """Cyclic transition matrix (``librosa.sequence.transition_cycle``), ``(n_states, n_states)``."""
    ...

def transition_local(
    n_states: int,
    width: Union[int, npt.ArrayLike],
    *,
    window: Literal["triangle", "triang", "hann", "hamming", "blackman", "boxcar"] = "triangle",
    wrap: bool = False,
) -> NDArray[np.float64]:
    """Localised transition matrix (``librosa.sequence.transition_local``),
    ``(n_states, n_states)``, row-stochastic."""
    ...

@overload
def rqa(
    sim: NDArray[_UnknownDtype],
    *,
    gap_onset: float = 1.0,
    gap_extend: float = 1.0,
    knight_moves: bool = True,
    backtrack: Literal[True] = True,
) -> tuple[NDArray[Any], NDArray[Any]]: ...
@overload
def rqa(
    sim: NDArray[_UnknownDtype],
    *,
    gap_onset: float = 1.0,
    gap_extend: float = 1.0,
    knight_moves: bool = True,
    backtrack: Literal[False],
) -> NDArray[Any]: ...
@overload
def rqa(  # type: ignore[overload-overlap]
    sim: _SingleArray,
    *,
    gap_onset: float = 1.0,
    gap_extend: float = 1.0,
    knight_moves: bool = True,
    backtrack: Literal[True] = True,
) -> tuple[NDArray[np.float32], NDArray[np.int64]]: ...
@overload
def rqa(  # type: ignore[overload-overlap]
    sim: _SingleArray,
    *,
    gap_onset: float = 1.0,
    gap_extend: float = 1.0,
    knight_moves: bool = True,
    backtrack: Literal[False],
) -> NDArray[np.float32]: ...
@overload
def rqa(
    sim: npt.ArrayLike,
    *,
    gap_onset: float = 1.0,
    gap_extend: float = 1.0,
    knight_moves: bool = True,
    backtrack: Literal[True] = True,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]: ...
@overload
def rqa(
    sim: npt.ArrayLike,
    *,
    gap_onset: float = 1.0,
    gap_extend: float = 1.0,
    knight_moves: bool = True,
    backtrack: Literal[False],
) -> NDArray[np.float64]:
    """Recurrence quantification analysis (``librosa.sequence.rqa``).

    Returns:
        ``score`` (numpy.ndarray[float32 | float64], shape of ``sim``) or ``(score, path)``
        with ``path`` an int64 ``(length, 2)`` array from start to end.
    """
    ...

@overload
def recurrence_matrix(
    data: NDArray[_UnknownDtype],
    *,
    k: Optional[int] = None,
    width: int = 1,
    metric: _Metric = 'euclidean',
    sym: bool = False,
    sparse: bool = False,
    mode: Literal['distance', 'affinity'],
    bandwidth: _BandwidthSpec = None,
    self: bool = False,
    axis: int = -1,
    full: bool = False,
    self_links: bool = False,
) -> NDArray[Any]: ...
@overload
def recurrence_matrix(
    data: npt.ArrayLike,
    *,
    k: Optional[int] = None,
    width: int = 1,
    metric: _Metric = 'euclidean',
    sym: bool = False,
    sparse: bool = False,
    mode: Literal['connectivity'] = 'connectivity',
    bandwidth: _BandwidthSpec = None,
    self: bool = False,
    axis: int = -1,
    full: bool = False,
    self_links: bool = False,
) -> NDArray[np.bool_]: ...
@overload
def recurrence_matrix(  # type: ignore[overload-overlap]
    data: _SingleArray,
    *,
    k: Optional[int] = None,
    width: int = 1,
    metric: _Metric = 'euclidean',
    sym: bool = False,
    sparse: bool = False,
    mode: Literal['distance', 'affinity'],
    bandwidth: _BandwidthSpec = None,
    self: bool = False,
    axis: int = -1,
    full: bool = False,
    self_links: bool = False,
) -> NDArray[np.float32]: ...
@overload
def recurrence_matrix(
    data: npt.ArrayLike,
    *,
    k: Optional[int] = None,
    width: int = 1,
    metric: _Metric = 'euclidean',
    sym: bool = False,
    sparse: bool = False,
    mode: Literal['distance', 'affinity'],
    bandwidth: _BandwidthSpec = None,
    self: bool = False,
    axis: int = -1,
    full: bool = False,
    self_links: bool = False,
) -> NDArray[np.float64]:
    """Recurrence matrix (``librosa.segment.recurrence_matrix``, dense).

    ``self`` is librosa's keyword (whether the diagonal links each frame to
    itself); ``self_links`` is an equivalent spelling that avoids the Python
    keyword-like name. Either one set to ``True`` sets the diagonal.

    Returns:
        ``(t, t)``: bool for ``mode="connectivity"``, float64 otherwise.
        ``R[j, i]`` links frame ``j`` to frame ``i``.
    """
    ...

@overload
def cross_similarity(
    data: NDArray[_UnknownDtype],
    data_ref: npt.ArrayLike,
    *,
    k: Optional[int] = None,
    metric: _Metric = 'euclidean',
    sparse: bool = False,
    mode: Literal['distance', 'affinity'],
    bandwidth: _BandwidthSpec = None,
    full: bool = False,
) -> NDArray[Any]: ...
@overload
def cross_similarity(
    data: npt.ArrayLike,
    data_ref: npt.ArrayLike,
    *,
    k: Optional[int] = None,
    metric: _Metric = 'euclidean',
    sparse: bool = False,
    mode: Literal['connectivity'] = 'connectivity',
    bandwidth: _BandwidthSpec = None,
    full: bool = False,
) -> NDArray[np.bool_]: ...
@overload
def cross_similarity(  # type: ignore[overload-overlap]
    data: _SingleArray,
    data_ref: npt.ArrayLike,
    *,
    k: Optional[int] = None,
    metric: _Metric = 'euclidean',
    sparse: bool = False,
    mode: Literal['distance', 'affinity'],
    bandwidth: _BandwidthSpec = None,
    full: bool = False,
) -> NDArray[np.float32]: ...
@overload
def cross_similarity(
    data: npt.ArrayLike,
    data_ref: npt.ArrayLike,
    *,
    k: Optional[int] = None,
    metric: _Metric = 'euclidean',
    sparse: bool = False,
    mode: Literal['distance', 'affinity'],
    bandwidth: _BandwidthSpec = None,
    full: bool = False,
) -> NDArray[np.float64]:
    """Cross-similarity matrix (``librosa.segment.cross_similarity``, dense).

    Returns:
        ``(n_ref, n)``: bool for connectivity, float64 otherwise.
    """
    ...

@overload
def recurrence_to_lag(
    rec: NDArray[_UnknownDtype],
    *,
    pad: bool = True,
    axis: int = -1,
) -> NDArray[Any]: ...
@overload
def recurrence_to_lag(  # type: ignore[overload-overlap]
    rec: NDArray[np.bool_],
    *,
    pad: bool = True,
    axis: int = -1,
) -> NDArray[np.bool_]: ...
@overload
def recurrence_to_lag(  # type: ignore[overload-overlap]
    rec: _SingleArray,
    *,
    pad: bool = True,
    axis: int = -1,
) -> NDArray[np.float32]: ...
@overload
def recurrence_to_lag(rec: npt.ArrayLike, *, pad: bool=True, axis: int=-1) -> NDArray[np.float64]:
    """Recurrence to time-lag matrix (``librosa.segment.recurrence_to_lag``):
    ``(2t, t)`` with ``pad`` else ``(t, t)``; bool in, bool out.

    ``axis`` is the time axis: ``-1``/``1`` (columns, default) or ``0``/``-2``
    (rows; padded output ``(t, 2t)``). Sparse input is not supported.
    """
    ...

@overload
def lag_to_recurrence(lag: NDArray[_UnknownDtype], *, axis: int=-1) -> NDArray[Any]: ...
@overload
def lag_to_recurrence(  # type: ignore[overload-overlap]
    lag: NDArray[np.bool_],
    *,
    axis: int = -1,
) -> NDArray[np.bool_]: ...
@overload
def lag_to_recurrence(  # type: ignore[overload-overlap]
    lag: _SingleArray,
    *,
    axis: int = -1,
) -> NDArray[np.float32]: ...
@overload
def lag_to_recurrence(lag: npt.ArrayLike, *, axis: int=-1) -> NDArray[np.float64]:
    """Time-lag to recurrence matrix (``librosa.segment.lag_to_recurrence``), ``(t, t)``.

    ``axis`` is the time axis: ``-1``/``1`` (columns, default) or ``0``/``-2`` (rows).
    """
    ...

@overload
def sync(
    data: NDArray[_UnknownDtype],
    idx: Union[npt.ArrayLike, Sequence[slice]],
    *,
    aggregate: Union[None, str, Callable[..., Any]] = None,
    pad: bool = True,
    axis: int = -1,
) -> NDArray[Any]: ...
@overload
def sync(  # type: ignore[overload-overlap]
    data: _SingleArray,
    idx: Union[npt.ArrayLike, Sequence[slice]],
    *,
    aggregate: Union[None, str, Callable[..., Any]] = None,
    pad: bool = True,
    axis: int = -1,
) -> NDArray[np.float32]: ...
@overload
def sync(
    data: npt.ArrayLike,
    idx: Union[npt.ArrayLike, Sequence[slice]],
    *,
    aggregate: Union[None, str, Callable[..., Any]] = None,
    pad: bool = True,
    axis: int = -1,
) -> NDArray[np.float64]:
    """Aggregate frames between boundaries (``librosa.util.sync``, last axis).

    Args:
        idx: Boundary frame indices, or one slice of the frames per output entry
            (``pad`` is then not used).
        aggregate: None/``numpy.mean`` (default), ``numpy.median``, ``numpy.max``,
            ``numpy.min``, ``numpy.sum``, the names ``"mean"`` etc., or any
            callable ``f(x, axis=-1)`` returning a float for a 1-D segment.

    Returns:
        numpy.ndarray[float32 | float64]: ``data`` with the last axis replaced by one entry
        per segment.
    """
    ...

def path_to_steps(path: npt.ArrayLike, *, inverse: bool = False) -> NDArray[np.float64]:
    """The fractional index on one side of an alignment path for each frame on the other."""

__all__ = [
    "__doc__",
    "cross_similarity",
    "dtw",
    "dtw_backtracking",
    "path_to_steps",
    "lag_to_recurrence",
    "recurrence_matrix",
    "recurrence_to_lag",
    "rqa",
    "sync",
    "transition_cycle",
    "transition_local",
    "transition_loop",
    "transition_uniform",
    "viterbi",
    "viterbi_binary",
    "viterbi_discriminative",
]
