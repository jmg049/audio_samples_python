"""Array helpers: local extrema, event and interval matching, one-dimensional
interpolation, padding, reordering, compressed sparse rows and argument checks."""

from __future__ import annotations

import os
from typing import Any, Callable, Literal, Optional, Sequence, Union, final, overload

import numpy as np
import numpy.typing as npt
from numpy.typing import NDArray

_InterpKind = Union[
    Literal["linear", "nearest", "nearest-up", "zero", "slinear", "quadratic", "cubic", "previous", "next"],
    int,
]

MAX_MEM_BLOCK: int
"""Suggested upper limit in bytes for one block of a batched computation."""


@final
class SparseRows:
    """A matrix in compressed sparse row form."""

    @property
    def data(self) -> NDArray[Any]: ...
    @property
    def indices(self) -> NDArray[np.int32]: ...
    @property
    def indptr(self) -> NDArray[np.int32]: ...
    @property
    def shape(self) -> tuple[int, int]: ...
    @property
    def nnz(self) -> int: ...
    def toarray(self) -> NDArray[Any]:
        """The dense matrix."""
    def tocsr(self) -> Any:
        """The matrix as a ``scipy.sparse.csr_array``; needs SciPy."""


def localmax(x: npt.ArrayLike, *, axis: int = 0) -> NDArray[np.bool_]:
    """Local maxima along an axis: ``x[i] > x[i - 1]`` and ``x[i] >= x[i + 1]``."""
def localmin(x: npt.ArrayLike, *, axis: int = 0) -> NDArray[np.bool_]:
    """Local minima along an axis: ``x[i] < x[i - 1]`` and ``x[i] <= x[i + 1]``."""
@overload
def abs2(x: complex, dtype: None = None) -> float: ...
@overload
def abs2(x: npt.ArrayLike, dtype: Optional[npt.DTypeLike] = None) -> Any: ...
@overload
def phasor(angles: NDArray[np.float32], *, mag: Optional[npt.ArrayLike] = None) -> NDArray[np.complex64]: ...
@overload
def phasor(angles: NDArray[Any], *, mag: Optional[npt.ArrayLike] = None) -> NDArray[Any]: ...
@overload
def phasor(angles: npt.ArrayLike, *, mag: Optional[npt.ArrayLike] = None) -> Any: ...
def tiny(x: Any) -> np.floating[Any]:
    """The smallest positive normal value of the input's floating-point type."""
def expand_to(x: npt.ArrayLike, *, ndim: int, axes: Union[int, Sequence[int]]) -> NDArray[Any]:
    """Places the axes of an array at given positions of a higher-dimensional view."""
def stack(arrays: Sequence[npt.ArrayLike], *, axis: int = 0) -> NDArray[Any]:
    """Stacks arrays of one shape along a new axis."""
def pad_center(data: npt.ArrayLike, *, size: int, axis: int = -1, **kwargs: Any) -> NDArray[Any]:
    """Pads along one axis to a given length, keeping the data centred."""
def fix_frames(
    frames: npt.ArrayLike, *, x_min: Optional[int] = 0, x_max: Optional[int] = None, pad: bool = True
) -> NDArray[np.int64]:
    """Clips frame indices, optionally adds the bounds, removes duplicates and sorts."""
def index_to_slice(
    idx: npt.ArrayLike,
    *,
    idx_min: Optional[int] = None,
    idx_max: Optional[int] = None,
    step: Optional[int] = None,
    pad: bool = True,
) -> list[slice]:
    """Converts boundary indices to the slices between consecutive boundaries."""
@overload
def axis_sort(
    S: npt.ArrayLike, *, axis: int = -1, index: Literal[False] = False, value: Optional[Callable[..., Any]] = None
) -> NDArray[Any]: ...
@overload
def axis_sort(
    S: npt.ArrayLike, *, axis: int = -1, index: Literal[True], value: Optional[Callable[..., Any]] = None
) -> tuple[NDArray[Any], NDArray[np.int64]]: ...
def shear(X: Any, *, factor: int = 1, axis: int = -1) -> Any:
    """Circularly shifts each column or row of a matrix by ``factor`` times its index."""
def sparsify_rows(x: npt.ArrayLike, *, quantile: float = 0.01, dtype: Optional[npt.DTypeLike] = None) -> SparseRows:
    """Removes the smallest values of each row, keeping a sparse matrix."""
def cyclic_gradient(data: npt.ArrayLike, *, edge_order: int = 1, axis: int = -1) -> NDArray[Any]:
    """Central-difference gradient of periodic data along one axis."""
def fill_off_diagonal(x: NDArray[Any], *, radius: float, value: Optional[complex] = None) -> None:
    """Sets the entries of a matrix outside a diagonal band to a value, in place."""
def count_unique(data: npt.ArrayLike, *, axis: int = -1) -> NDArray[np.int64]:
    """Counts the distinct values in each lane along an axis."""
def is_unique(data: npt.ArrayLike, *, axis: int = -1) -> NDArray[np.bool_]:
    """Tells for each lane along an axis whether all its values are distinct."""
def interp1d(
    x: npt.ArrayLike,
    y: npt.ArrayLike,
    xi: npt.ArrayLike,
    *,
    kind: Optional[_InterpKind] = None,
    fill_value: float = 0.0,
    axis: int = -1,
) -> NDArray[Any]:
    """Interpolates sampled data along one axis."""
def interp_broadcast(
    *,
    x1: npt.ArrayLike,
    x1_pos: npt.ArrayLike,
    x2: npt.ArrayLike,
    x2_pos: npt.ArrayLike,
    interp_pos: Optional[npt.ArrayLike] = None,
    op: Any = "multiply",
    kind: Optional[_InterpKind] = None,
    fill_value: float = 0.0,
    axis: int = -2,
) -> Any:
    """Interpolates two arrays sampled on different grids onto one grid and combines them."""
def dtype_c2r(d: npt.DTypeLike, *, default: Optional[npt.DTypeLike] = None) -> np.dtype[Any]:
    """The real dtype of a complex dtype."""
def dtype_r2c(d: npt.DTypeLike, *, default: Optional[npt.DTypeLike] = None) -> np.dtype[Any]:
    """The complex dtype of a real dtype."""
def buf_to_float(x: Any, *, n_bytes: int = 2, dtype: Optional[npt.DTypeLike] = None) -> NDArray[Any]:
    """Decodes little-endian signed integer PCM to floats in ``[-1, 1)``."""
def find_files(
    directory: Union[str, os.PathLike[str]],
    *,
    ext: Optional[Union[str, list[str]]] = None,
    recurse: bool = True,
    case_sensitive: bool = False,
    limit: Optional[int] = None,
    offset: int = 0,
) -> list[str]:
    """Lists audio files under a directory, sorted, with absolute paths."""
def valid_audio(y: NDArray[Any]) -> bool:
    """Checks that samples are a floating-point array, at least 1-D, finite everywhere."""
def valid_intervals(intervals: npt.ArrayLike) -> bool:
    """Checks that intervals have shape ``(n, 2)`` and non-negative durations."""
def is_positive_int(x: Any) -> bool:
    """Whether a value is an integer above zero."""
def valid_int(x: Any, *, cast: Optional[Callable[[Any], Any]] = None) -> int:
    """Converts a number to an ``int`` with a chosen rounding (default: floor)."""
def match_events(
    events_from: npt.ArrayLike, events_to: npt.ArrayLike, left: bool = True, right: bool = True
) -> NDArray[np.int64]:
    """Matches each source event to the nearest target event."""
def match_intervals(
    intervals_from: npt.ArrayLike, intervals_to: npt.ArrayLike, strict: bool = True
) -> NDArray[np.int64]:
    """Matches each source interval to the target interval it overlaps most."""

def zero_crossings(
    y: npt.ArrayLike,
    *,
    threshold: float | None = 1e-10,
    ref_magnitude: float | Callable[[NDArray[Any]], float] | None = None,
    pad: bool = True,
    zero_pos: bool = True,
    axis: int = -1,
) -> NDArray[np.bool_]:
    """Marks the sign changes of a signal along one axis.

    Args:
        y: The signal, of any dimension.
        threshold: Magnitudes up to this count as zero. None means 0.
        ref_magnitude: Scales ``threshold``. A callable receives ``numpy.abs(y)``.
        pad: The mark of the first element.
        zero_pos: Zero counts as positive; otherwise it is a sign of its own.
        axis: The time axis.

    Returns:
        The marks, of the shape of ``y``.
    """
    ...


def tone(
    frequency: float,
    *,
    sr: float = 22050,
    length: int | None = None,
    duration: float | None = None,
    phi: float | None = None,
) -> NDArray[np.float64]:
    """A cosine of a given frequency and phase, as float64.

    Args:
        frequency: In Hz.
        sr: Sample rate in Hz.
        length: Samples.
        duration: Seconds, if ``length`` is None.
        phi: Phase offset in radians. None means ``-pi / 2``, which starts like a sine.

    Returns:
        The samples.
    """
    ...


def chirp(
    *,
    fmin: float,
    fmax: float,
    sr: float = 22050,
    length: int | None = None,
    duration: float | None = None,
    linear: bool = False,
    phi: float | None = None,
) -> NDArray[np.float64]:
    """A swept cosine, as float64.

    Args:
        fmin: Starting frequency in Hz.
        fmax: Frequency at the end in Hz.
        sr: Sample rate in Hz.
        length: Samples.
        duration: Seconds, if ``length`` is None.
        linear: Sweep linearly; otherwise exponentially.
        phi: Phase offset in radians. None means ``-pi / 2``.

    Returns:
        The samples.
    """
    ...


__all__ = [
    "__doc__",
    "zero_crossings",
    "tone",
    "chirp",
    "MAX_MEM_BLOCK",
    "SparseRows",
    "abs2",
    "axis_sort",
    "buf_to_float",
    "count_unique",
    "cyclic_gradient",
    "dtype_c2r",
    "dtype_r2c",
    "expand_to",
    "fill_off_diagonal",
    "find_files",
    "fix_frames",
    "index_to_slice",
    "interp1d",
    "interp_broadcast",
    "is_positive_int",
    "is_unique",
    "localmax",
    "localmin",
    "match_events",
    "match_intervals",
    "pad_center",
    "phasor",
    "shear",
    "sparsify_rows",
    "stack",
    "tiny",
    "valid_audio",
    "valid_int",
    "valid_intervals",
]
