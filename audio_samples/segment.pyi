"""Structure analysis over feature matrices: path enhancement, temporally constrained clustering, subsegmentation."""

from __future__ import annotations

from typing import Any, Callable, Optional, final

import numpy as np
import numpy.typing as npt
from numpy.typing import NDArray


def diagonal_filter(
    window: Any, n: int, *, slope: float = 1.0, angle: Optional[float] = None, zero_mean: bool = False
) -> NDArray[np.float64]:
    """A normalised smoothing kernel along a line through the origin."""
def path_enhance(
    R: npt.ArrayLike,
    n: int,
    *,
    window: Any = "hann",
    max_ratio: float = 2.0,
    min_ratio: Optional[float] = None,
    n_filters: int = 7,
    zero_mean: bool = False,
    clip: bool = True,
    **kwargs: Any,
) -> NDArray[Any]:
    """The largest response of a recurrence matrix to diagonal kernels of several slopes."""


@final
class TimelagFiltered:
    """The filter that :func:`timelag_filter` returns."""

    def __call__(self, *args: Any, **kwargs: Any) -> NDArray[Any]: ...
    @property
    def __wrapped__(self) -> Callable[..., Any]: ...


def timelag_filter(function: Callable[..., Any], pad: bool = True, index: int = 0) -> TimelagFiltered:
    """Wraps a filter so that it runs on a recurrence matrix in lag coordinates."""
def agglomerative(
    data: npt.ArrayLike, k: int, *, clusterer: Optional[Any] = None, axis: int = -1
) -> NDArray[np.int64]:
    """Segment boundaries from Ward clustering in which only neighbouring segments merge."""
def subsegment(
    data: npt.ArrayLike, frames: npt.ArrayLike, *, n_segments: int = 4, axis: int = -1
) -> NDArray[np.int64]:
    """Subdivides each interval of a segmentation."""

__all__ = ["TimelagFiltered", "agglomerative", "diagonal_filter", "path_enhance", "subsegment", "timelag_filter"]
