"""
Sequence alignment and decoding: DTW, Viterbi, transition matrices, RQA, recurrence and cross-similarity matrices, lag conversions and sync. float32 / complex64 arrays are processed in float32 and give float32 output; any other input is converted to float64.
"""

from numpy.typing import DTypeLike
from typing import Any

__all__ = ["cross_similarity", "dtw", "dtw_backtracking", "lag_to_recurrence", "path_to_steps", "recurrence_matrix", "recurrence_to_lag", "rqa", "sync", "transition_cycle", "transition_local", "transition_loop", "transition_uniform", "viterbi", "viterbi_binary", "viterbi_discriminative"]

def cross_similarity(data: "ArrayLike", data_ref: "ArrayLike", *, k: int | None = None, metric: str = "euclidean", sparse: bool = False, mode: str = "connectivity", bandwidth: Any | None = None, full: bool = False) -> "NDArray[Any]":
    """
    Cross-similarity matrix between a query and a reference feature sequence.

    Args:

        data (array_like): Query features ``(n_features, n)``.
        data_ref (array_like): Reference features ``(n_features, n_ref)``.
        k (Optional[int]): Reference neighbours per query frame; None =
            ``min(n_ref, 2 * ceil(sqrt(n_ref)))``.
        metric (str): Distance metric. Default "euclidean".
        sparse (bool): Return ``arrays.SparseRows`` rather than a dense array. Default False.
        mode (str): ``"connectivity"`` (bool output), ``"distance"`` or ``"affinity"``.
        bandwidth: As ``recurrence_matrix`` (array shape ``(n, n_ref)``).
        full (bool): Link every query frame to its ``min(n_ref, n)`` nearest. Default False.

    Returns:

        numpy.ndarray: ``(n_ref, n)``; bool for connectivity, otherwise float in
        ``data``'s precision.
    """

def dtw(X: Any | None = None, Y: Any | None = None, *, C: Any | None = None, metric: str = "euclidean", step_sizes_sigma: Any | None = None, weights_add: Any | None = None, weights_mul: Any | None = None, subseq: bool = False, backtrack: bool = True, global_constraints: bool = False, band_rad: float = 0.25, return_steps: bool = False) -> "tuple[NDArray[Any], NDArray[Any]] | tuple[NDArray[Any], NDArray[Any], NDArray[Any]] | NDArray[Any] | tuple[NDArray[np.float32], NDArray[np.int64]] | tuple[NDArray[np.float32], NDArray[np.int64], NDArray[np.int64]] | tuple[NDArray[np.float64], NDArray[np.int64]] | tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.int64]]":
    """
    Dynamic time warping.

    Pass either the feature sequences ``X`` and ``Y`` (``(n_features, N)`` and
    ``(n_features, M)``; 1-D input is one feature), or a precomputed cost
    matrix ``C`` of shape ``(N, M)``.

    Args:

        X, Y (Optional[array_like]): Feature sequences.
        C (Optional[array_like]): Precomputed local cost matrix.
        metric (str): ``"euclidean"`` (default), ``"sqeuclidean"``, ``"cosine"``,
            ``"cityblock"`` or ``"chebyshev"`` (used with X/Y only).
        step_sizes_sigma (Optional[array_like]): Extra steps, shape ``(n, 2)``;
            appended to the default steps ``[[1, 1], [0, 1], [1, 0]]``.
        weights_add, weights_mul (Optional[array_like]): Per-step weights.
        subseq (bool): Subsequence alignment. Default False.
        backtrack (bool): Compute the warping path. Default True.
        global_constraints (bool): Sakoe-Chiba band. Default False.
        band_rad (float): Band radius (fraction of the shorter length). Default 0.25.
        return_steps (bool): Also return the step matrix. Default False.

    Returns:

        ``D`` (numpy.ndarray[float32 | float64], shape ``(N, M)``); with ``backtrack=True``
        the tuple ``(D, wp)`` where ``wp`` is an int64 array of shape
        ``(path_length, 2)`` ordered from the end of the alignment to its start;
        with ``return_steps=True`` the step matrix (int64, same shape as ``D``)
        is appended.

    Raises:

        ValueError: Incompatible inputs, NaN costs, or no valid path.
    """

def dtw_backtracking(steps: "ArrayLike", *, step_sizes_sigma: Any | None = None, subseq: bool = False, start: int | None = None) -> "NDArray[np.int64]":
    """
    Backtrack a warping path from a DTW step matrix.

    Args:

        steps (array_like[int]): Step matrix from ``dtw(..., return_steps=True)``.
        step_sizes_sigma (Optional[array_like]): The custom steps passed to ``dtw``.
        subseq (bool): Whether ``dtw`` ran with ``subseq=True``.
        start (Optional[int]): Column of the last row to start from (subseq only).

    Returns:

        numpy.ndarray[int64]: shape ``(path_length, 2)``, end first.
    """

def lag_to_recurrence(lag: "ArrayLike", *, axis: int = ...) -> "NDArray[Any]":
    """
    Convert a (dense) time-lag matrix back to a recurrence matrix.

    Args:

        lag (array_like): ``(t, t)`` or ``(2t, t)`` lag matrix.
        axis (int): The time axis: ``-1``/``1`` (columns, default) or
            ``0``/``-2`` (rows; a padded lag matrix is then ``(t, 2t)``).

    Returns:

        numpy.ndarray: ``(t, t)``; bool if ``lag`` is bool, else ``lag``'s precision.
    """

def path_to_steps(path: "ArrayLike", *, inverse: bool = False) -> "NDArray[np.float64]":
    """
    The fractional index on one side of an alignment path for each frame on the other.

    Args:

        path (``array_like``): ``(n, 2)`` index pairs, as ``sequence.dtw`` gives.
        inverse (bool): False gives column 0 for each frame of column 1. Default false.

    Returns:

        numpy.ndarray[float64]: One value per frame on the path.
    """

def recurrence_matrix(data: "ArrayLike", *, k: int | None = None, width: int = ..., metric: str = "euclidean", sym: bool = False, sparse: bool = False, mode: str = "connectivity", bandwidth: Any | None = None, self_links: bool = False, axis: int = ..., full: bool = False, **kwargs) -> "NDArray[Any]":
    """
    Recurrence (self-similarity) matrix of a feature sequence.

    Args:

        data (array_like): ``(n_features, t)`` features (1-D = one feature).
        k (Optional[int]): Neighbours per frame; None = ``2 * ceil(sqrt(t - 2 * width + 1))``.
        width (int): Minimum time separation of linked frames. Default 1.
        metric (str): Distance metric (see ``dtw``). Default "euclidean".
        sym (bool): Keep only mutual links. Default False.
        sparse (bool): Return ``arrays.SparseRows`` rather than a dense array. Default False.
        mode (str): ``"connectivity"`` (default, bool output), ``"distance"`` or ``"affinity"``.
        bandwidth: Affinity bandwidth: None (``"med_k_scalar"``), a heuristic
            name, a positive number, or a ``(t, t)`` array.
        self_links (bool): Link each frame to itself. ``self=...`` is an
            equivalent keyword.
        axis (int): Must be -1 (frames on the last axis).
        full (bool): Link every pair in distance/affinity mode. Default False.

    Returns:

        numpy.ndarray: ``(t, t)``; bool for connectivity, otherwise float in
        ``data``'s precision.
    """

def recurrence_to_lag(rec: "ArrayLike", *, pad: bool = True, axis: int = ...) -> "NDArray[Any]":
    """
    Convert a (dense) recurrence matrix to a time-lag matrix.

    Args:

        rec (array_like): ``(t, t)`` recurrence matrix (bool or float).
        pad (bool): Zero-pad to ``(2t, t)`` first. Default True.
        axis (int): The time axis: ``-1``/``1`` (columns, default) or
            ``0``/``-2`` (rows; the result is then ``(t, 2t)``).

    Returns:

        numpy.ndarray: ``(2t, t)`` or ``(t, t)``; bool if ``rec`` is bool, else ``rec``'s precision.
    """

def rqa(sim: "ArrayLike", *, gap_onset: float = 1.0, gap_extend: float = 1.0, knight_moves: bool = True, backtrack: bool = True) -> "tuple[NDArray[Any], NDArray[Any]] | NDArray[Any] | tuple[NDArray[np.float32], NDArray[np.int64]] | tuple[NDArray[np.float64], NDArray[np.int64]]":
    """
    Recurrence quantification analysis.

    Args:

        sim (array_like): ``(N, M)`` similarity / recurrence matrix (bool or float).
        gap_onset (float): Gap opening penalty. Default 1.
        gap_extend (float): Gap extension penalty. Default 1.
        knight_moves (bool): Allow (1, 2) / (2, 1) moves. Default True.
        backtrack (bool): Return the best path too. Default True.

    Returns:

        ``score`` (numpy.ndarray[float32 | float64], shape ``(N, M)``), or
        ``(score, path)`` with ``path`` an int64 ``(length, 2)`` array ordered
        from start to end (empty when there is no recurrence).
    """

def sync(data: "ArrayLike", idx: "ArrayLike | Sequence[slice]", *, aggregate: Any | None = None, pad: bool = True, axis: int = ...) -> "NDArray[Any]":
    """
    Aggregate frames between boundaries (e.g. beats) along the last axis.

    Args:

        data (array_like): Any shape with frames on the last axis.
        idx (array_like[int] | Sequence[slice]): Boundary frame indices (sorted and
            de-duplicated internally), or one slice of the frames per output entry.
        aggregate: None / ``numpy.mean`` (default), ``numpy.median``,
            ``numpy.max``, ``numpy.min``, ``numpy.sum``, the same names as strings,
            or any callable ``f(x, axis=-1) -> float`` applied to each 1-D segment.
        pad (bool): Add 0 and ``n_frames`` to the boundaries. Default True. Not used
            with slices.

    Returns:

        numpy.ndarray[float32 | float64]: ``data`` with the last axis replaced by one entry
        per segment.

    Raises:

        ValueError: Negative indices or 0-d data.
    """

def transition_cycle(n_states: int, prob: "float | ArrayLike") -> "NDArray[np.float64]":
    """
    Cyclic transition matrix.

    Args:

        n_states (int): Number of states (> 1).
        prob (float | array_like): Self-loop probability, one value or one per state.

    Returns:

        numpy.ndarray[float64]: ``(n_states, n_states)``.
    """

def transition_local(n_states: int, width: "int | ArrayLike", *, window: str = "triangle", wrap: bool = False) -> "NDArray[np.float64]":
    """
    Localised transition matrix (row-stochastic).

    Args:

        n_states (int): Number of states (> 1).
        width (int | array_like[int]): Neighbourhood width, one value or one per state.
        window (str): ``"triangle"`` (default, also ``"triang"``), ``"hann"``,
            ``"hamming"``, ``"blackman"`` or ``"boxcar"``.
        wrap (bool): Circular state space. Default False.

    Returns:

        numpy.ndarray[float64]: ``(n_states, n_states)``, row-stochastic.
    """

def transition_loop(n_states: int, prob: "float | ArrayLike") -> "NDArray[np.float64]":
    """
    Self-loop transition matrix.

    Args:

        n_states (int): Number of states (> 1).
        prob (float | array_like): Self-loop probability, one value or one per state.

    Returns:

        numpy.ndarray[float64]: ``(n_states, n_states)``.
    """

def transition_uniform(n_states: int) -> "NDArray[np.float64]":
    """
    Uniform transition matrix: every entry ``1 / n_states``.

    Returns:

        numpy.ndarray[float64]: ``(n_states, n_states)`` filled with ``1 / n_states``.
    """

def viterbi(prob: "ArrayLike", transition: "ArrayLike", *, p_init: Any | None = None, return_logp: bool = False, transition_min_prob: float | None = None) -> "NDArray[np.int64] | tuple[NDArray[np.int64], float]":
    """
    Viterbi decoding of a generative HMM.

    Args:

        prob (array_like): ``(n_states, n_steps)`` observation likelihoods in [0, 1].
        transition (array_like): ``(n_states, n_states)`` row-stochastic matrix.
        p_init (Optional[array_like]): Initial distribution; None = uniform.
        return_logp (bool): Also return the log-likelihood. Default False.
        transition_min_prob (Optional[float]): Prune transitions below this
            probability (approximate, faster). None = exact.

    Returns:

        numpy.ndarray[int64] of shape ``(n_steps,)``, or
        ``(states, logp)`` with ``logp`` a float when ``return_logp=True``.

    Raises:

        ValueError: Shape mismatch or invalid probabilities.
    """

def viterbi_binary(prob: "ArrayLike", transition: "ArrayLike", *, p_state: Any | None = None, p_init: Any | None = None, return_logp: bool = False, transition_min_prob: float | None = None, dtype: DTypeLike | None = None) -> "tuple[NDArray[np.bool_], NDArray[np.float64]] | tuple[NDArray[np.uint16], NDArray[np.float64]] | tuple[NDArray[np.int64], NDArray[np.float64]] | NDArray[Any] | tuple[NDArray[Any], NDArray[np.float64]]":
    """
    Independent binary (off/on) Viterbi decoding per state.

    Args:

        prob (array_like): ``(n_states, n_steps)`` activation probabilities
            (a 1-D array is one state).
        transition (array_like): ``(2, 2)`` shared or ``(n_states, 2, 2)`` per state.
        p_state (Optional[float | array_like]): Marginal P[on] per state; None = 0.5.
        p_init (Optional[float | array_like]): Initial P[on] per state; None = 0.5.
        return_logp (bool): Also return per-state log-likelihoods.
        transition_min_prob (Optional[float]): As ``viterbi``.
        dtype (Optional[numpy.dtype]): dtype of the states: ``bool`` (default
            None, one byte per entry), ``numpy.uint16`` or ``numpy.int64``.

    Returns:

        numpy.ndarray[bool | uint16 | int64] ``(n_states, n_steps)``, true or 1
        where the state is on, or ``(states, logp)`` with ``logp`` of shape
        ``(n_states,)``.
    """

def viterbi_discriminative(prob: "ArrayLike", transition: "ArrayLike", *, p_state: Any | None = None, p_init: Any | None = None, return_logp: bool = False, transition_min_prob: float | None = None) -> "NDArray[np.int64] | tuple[NDArray[np.int64], float]":
    """
    Viterbi decoding from discriminative (posterior) state probabilities; each
    column of ``prob`` sums to 1.

    Args:

        prob (array_like): ``(n_states, n_steps)``; each column sums to 1.
        transition (array_like): ``(n_states, n_states)`` row-stochastic matrix.
        p_state (Optional[array_like]): Marginal state distribution; None = uniform.
        p_init (Optional[array_like]): Initial distribution; None = uniform.
        return_logp (bool): Also return the (unnormalised) log-likelihood.
        transition_min_prob (Optional[float]): As ``viterbi``.

    Returns:

        numpy.ndarray[int64] ``(n_steps,)`` or ``(states, logp)``.
    """
