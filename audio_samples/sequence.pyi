"""Sequence alignment and decoding (``librosa.sequence``, ``librosa.segment``,
``librosa.util.sync``).

Feature matrices use librosa's layout ``(n_features, n_frames)``; 1-D input is
one feature. Probability matrices are ``(n_states, n_steps)``. Return values
follow librosa: tuples only when ``backtrack`` / ``return_steps`` /
``return_logp`` ask for them. Index outputs are int64 (librosa: uint16 states,
intp paths). Defaults are librosa 1.0.0's.

Differences from librosa: dense output only (``sparse=True`` raises), only
``axis=-1``; ``sync`` with an empty index list returns one segment; ``rqa``
follows the move actually taken in the second column (librosa stores a wrong
back-pointer there).
"""

from __future__ import annotations

from typing import Any, Callable, Literal, Optional, Union

import numpy as np
import numpy.typing as npt
from numpy.typing import NDArray

Metric = Literal["euclidean", "sqeuclidean", "cosine", "cityblock", "chebyshev"]
Mode = Literal["connectivity", "distance", "affinity"]
#: Affinity bandwidth: None (``"med_k_scalar"``), a heuristic name, a positive
#: number, or a per-link array.
BandwidthSpec = Union[
    None,
    float,
    Literal["med_k_scalar", "mean_k", "gmean_k", "mean_k_avg", "gmean_k_avg", "mean_k_avg_and_pair"],
    npt.ArrayLike,
]

def dtw(
    X: Optional[npt.ArrayLike] = None,
    Y: Optional[npt.ArrayLike] = None,
    *,
    C: Optional[npt.ArrayLike] = None,
    metric: Metric = "euclidean",
    step_sizes_sigma: Optional[npt.ArrayLike] = None,
    weights_add: Optional[npt.ArrayLike] = None,
    weights_mul: Optional[npt.ArrayLike] = None,
    subseq: bool = False,
    backtrack: bool = True,
    global_constraints: bool = False,
    band_rad: float = 0.25,
    return_steps: bool = False,
) -> Any:
    """Dynamic time warping, equivalent to ``librosa.sequence.dtw``.

    Pass ``X`` ``(n_features, N)`` and ``Y`` ``(n_features, M)``, or a cost matrix
    ``C`` ``(N, M)``.

    Returns:
        ``D`` (numpy.ndarray[float64] ``(N, M)``, unreachable cells ``inf``) when
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

def viterbi(
    prob: npt.ArrayLike,
    transition: npt.ArrayLike,
    *,
    p_init: Optional[npt.ArrayLike] = None,
    return_logp: bool = False,
    transition_min_prob: Optional[float] = None,
) -> Any:
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

def viterbi_discriminative(
    prob: npt.ArrayLike,
    transition: npt.ArrayLike,
    *,
    p_state: Optional[npt.ArrayLike] = None,
    p_init: Optional[npt.ArrayLike] = None,
    return_logp: bool = False,
    transition_min_prob: Optional[float] = None,
) -> Any:
    """Viterbi decoding from posterior state probabilities
    (``librosa.sequence.viterbi_discriminative``); each column of ``prob`` sums to 1.

    Returns:
        numpy.ndarray[int64] ``(n_steps,)``, or ``(states, logp)`` with an
        unnormalised float log-likelihood.
    """
    ...

def viterbi_binary(
    prob: npt.ArrayLike,
    transition: npt.ArrayLike,
    *,
    p_state: Optional[Union[float, npt.ArrayLike]] = None,
    p_init: Optional[Union[float, npt.ArrayLike]] = None,
    return_logp: bool = False,
    transition_min_prob: Optional[float] = None,
) -> Any:
    """Independent off/on Viterbi per state (``librosa.sequence.viterbi_binary``).

    Args:
        prob: ``(n_states, n_steps)`` activation probabilities (1-D = one state).
        transition: ``(2, 2)`` shared or ``(n_states, 2, 2)``.

    Returns:
        numpy.ndarray[int64] of 0/1, ``(n_states, n_steps)`` (``(n_steps,)`` for
        1-D input), or ``(states, logp)`` with ``logp`` ``(n_states,)``.
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

def rqa(
    sim: npt.ArrayLike,
    *,
    gap_onset: float = 1.0,
    gap_extend: float = 1.0,
    knight_moves: bool = True,
    backtrack: bool = True,
) -> Any:
    """Recurrence quantification analysis (``librosa.sequence.rqa``).

    Returns:
        ``score`` (numpy.ndarray[float64], shape of ``sim``) or ``(score, path)``
        with ``path`` an int64 ``(length, 2)`` array from start to end.
    """
    ...

def recurrence_matrix(
    data: npt.ArrayLike,
    *,
    k: Optional[int] = None,
    width: int = 1,
    metric: Metric = "euclidean",
    sym: bool = False,
    sparse: bool = False,
    mode: Mode = "connectivity",
    bandwidth: BandwidthSpec = None,
    self: bool = False,
    axis: int = -1,
    full: bool = False,
    self_links: bool = False,
) -> NDArray[Any]:
    """Recurrence matrix (``librosa.segment.recurrence_matrix``, dense).

    ``self`` is librosa's keyword (whether the diagonal links each frame to
    itself); ``self_links`` is an equivalent spelling that avoids the Python
    keyword-like name. Either one set to ``True`` sets the diagonal.

    Returns:
        ``(t, t)``: bool for ``mode="connectivity"``, float64 otherwise.
        ``R[j, i]`` links frame ``j`` to frame ``i``.
    """
    ...

def cross_similarity(
    data: npt.ArrayLike,
    data_ref: npt.ArrayLike,
    *,
    k: Optional[int] = None,
    metric: Metric = "euclidean",
    sparse: bool = False,
    mode: Mode = "connectivity",
    bandwidth: BandwidthSpec = None,
    full: bool = False,
) -> NDArray[Any]:
    """Cross-similarity matrix (``librosa.segment.cross_similarity``, dense).

    Returns:
        ``(n_ref, n)``: bool for connectivity, float64 otherwise.
    """
    ...

def recurrence_to_lag(rec: npt.ArrayLike, *, pad: bool = True, axis: int = -1) -> NDArray[Any]:
    """Recurrence to time-lag matrix (``librosa.segment.recurrence_to_lag``):
    ``(2t, t)`` with ``pad`` else ``(t, t)``; bool in, bool out.

    ``axis`` is the time axis: ``-1``/``1`` (columns, default) or ``0``/``-2``
    (rows; padded output ``(t, 2t)``). Sparse input is not supported.
    """
    ...

def lag_to_recurrence(lag: npt.ArrayLike, *, axis: int = -1) -> NDArray[Any]:
    """Time-lag to recurrence matrix (``librosa.segment.lag_to_recurrence``), ``(t, t)``.

    ``axis`` is the time axis: ``-1``/``1`` (columns, default) or ``0``/``-2`` (rows).
    """
    ...

def sync(
    data: npt.ArrayLike,
    idx: npt.ArrayLike,
    *,
    aggregate: Union[None, str, Callable[..., Any]] = None,
    pad: bool = True,
    axis: int = -1,
) -> NDArray[np.float64]:
    """Aggregate frames between boundaries (``librosa.util.sync``, last axis).

    Args:
        aggregate: None/``numpy.mean`` (default), ``numpy.median``, ``numpy.max``,
            ``numpy.min``, ``numpy.sum``, the names ``"mean"`` etc., or any
            callable ``f(x, axis=-1)`` returning a float for a 1-D segment.

    Returns:
        numpy.ndarray[float64]: ``data`` with the last axis replaced by one entry
        per segment.
    """
    ...
