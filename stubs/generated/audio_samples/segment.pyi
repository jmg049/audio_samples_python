"""
Structure analysis over feature matrices: path enhancement, temporally constrained clustering, subsegmentation.
"""

from numpy.typing import ArrayLike
from typing import Any, final

__all__ = ["TimelagFiltered", "agglomerative", "diagonal_filter", "path_enhance", "subsegment", "timelag_filter"]

@final
class TimelagFiltered:
    """
    The filter that [`timelag_filter`] returns.
    """
    def __call__(self, /, *args, **kwargs) -> Any:
        """
        Applies the wrapped filter in lag coordinates.
        """
    @property
    def __wrapped__(self, /) -> Any:
        """
        The wrapped function.
        """

def agglomerative(data: "ArrayLike", k: int, *, clusterer: Any | None = None, axis: int = ...) -> "NDArray[np.int64]":
    """
    Segment boundaries from Ward clustering in which only neighbouring segments merge.

    Args:

        data (``array_like``): Features with time along ``axis``.
        k (int): Number of segments.
        clusterer (Optional[object]): Any object with ``fit`` and ``labels_``, in place of
            the built-in clustering.
        axis (int): The time axis. Default -1.

    Returns:

        numpy.ndarray[int64]: The first frame of each segment.
    """

def diagonal_filter(window: str | tuple[str, float] | float | ArrayLike, n: int, *, slope: float = 1.0, angle: float | None = None, zero_mean: bool = False) -> "NDArray[np.float64]":
    """
    A normalised smoothing kernel along a line through the origin.

    Args:

        window (str | tuple | ``array_like``): The window along the line.
        n (int): Its length, before rotation.
        slope (float): The line's slope; 1 is the main diagonal. Default 1.
        angle (Optional[float]): The line's angle in radians, in place of ``slope``.
        ``zero_mean`` (bool): Subtract the mean. Default false.

    Returns:

        numpy.ndarray[float64]: The kernel.
    """

def path_enhance(R: "ArrayLike", n: int, *, window: str | tuple[str, float] | float | ArrayLike = ..., max_ratio: float = 2.0, min_ratio: float | None = None, n_filters: int = ..., zero_mean: bool = False, clip: bool = True, **kwargs) -> "NDArray[Any]":
    """
    The largest response of a recurrence matrix to diagonal kernels of several slopes.

    Args:

        R (``array_like``): ``(..., n, n)`` or lag matrix.
        n (int): Length of each kernel, before rotation.
        window (str | tuple | ``array_like``): The window along each diagonal. Default ``"hann"``.
        ``max_ratio`` (float): Largest slope. Default 2.
        ``min_ratio`` (Optional[float]): Smallest slope. Default ``1 / max_ratio``.
        ``n_filters`` (int): Number of slopes. Default 7.
        ``zero_mean`` (bool): Subtract each kernel's mean. Default false.
        clip (bool): Set negative results to zero. Default true.
        **kwargs: ``mode`` (``"reflect"``, ``"constant"``, ``"nearest"``, ``"mirror"``,
            ``"wrap"``), ``cval`` and ``origin``.

    Returns:

        numpy.ndarray: The shape of ``R``, of its precision.
    """

def subsegment(data: "ArrayLike", frames: "ArrayLike", *, n_segments: int = ..., axis: int = ...) -> "NDArray[np.int64]":
    """
    Subdivides each interval of a segmentation with :func:`agglomerative`.

    Args:

        data (``array_like``): Features with time along ``axis``.
        frames (``array_like[int]``): Segment boundaries; the ends are added.
        ``n_segments`` (int): The most subsegments per interval. Default 4.
        axis (int): The time axis. Default -1.

    Returns:

        numpy.ndarray[int64]: The first frame of each subsegment.
    """

def timelag_filter(function: "Callable[..., Any]", pad: bool = True, index: int = ...) -> TimelagFiltered:
    """
    Wraps a filter so that it runs on a recurrence matrix in lag coordinates.

    Args:

        function (Callable): The filter. It receives the time-lag matrix at position ``index``.
        pad (bool): Pad the lag matrix to twice the length. Default true.
        index (int): Position of the matrix in the call's arguments. Default 0.

    Returns:

        Callable: Takes the same arguments as ``function`` and returns a recurrence matrix.
    """
