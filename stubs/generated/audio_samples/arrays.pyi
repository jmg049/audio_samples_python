"""
Array helpers: local extrema, event and interval matching, one-dimensional interpolation, padding, reordering, compressed sparse rows and argument checks.
"""

from os import PathLike
from typing import Any, Final, final

__all__ = ["MAX_MEM_BLOCK", "SparseRows", "abs2", "axis_sort", "buf_to_float", "chirp", "count_unique", "cyclic_gradient", "dtype_c2r", "dtype_r2c", "expand_to", "fill_off_diagonal", "find_files", "fix_frames", "index_to_slice", "interp1d", "interp_broadcast", "is_positive_int", "is_unique", "localmax", "localmin", "match_events", "match_intervals", "pad_center", "phasor", "shear", "sparsify_rows", "stack", "tiny", "tone", "valid_audio", "valid_int", "valid_intervals", "zero_crossings"]

MAX_MEM_BLOCK: Final[int]

@final
class SparseRows:
    """
    A matrix in compressed sparse row form.

    ``data``, ``indices`` and ``indptr`` are ``NumPy`` arrays with the usual meaning.
    ``toarray()`` gives the dense form without needing ``SciPy``; ``tocsr()`` gives a
    ``scipy.sparse.csr_array`` and needs ``SciPy``.
    """
    def __repr__(self, /) -> str: ...
    @property
    def data(self, /) -> Any:
        """
        Stored values.
        """
    @property
    def indices(self, /) -> Any:
        """
        Column of each stored value.
        """
    @property
    def indptr(self, /) -> Any:
        """
        Where each row starts in ``data``, with one extra entry at the end.
        """
    @property
    def nnz(self, /) -> int:
        """
        Number of stored values.
        """
    @property
    def shape(self, /) -> tuple[int, int]:
        """
        ``(rows, columns)``.
        """
    def toarray(self, /) -> "NDArray[Any]":
        """
        The dense matrix.
        """
    def tocsr(self, /) -> Any:
        """
        The matrix as a ``scipy.sparse.csr_array``.
        """

def abs2(x: "complex | ArrayLike", dtype: Any | None = None) -> Any:
    """
    Squared magnitude.

    Args:

        x (``complex`` | ``array_like``): Values. Complex input gives its real type; real input
            keeps its dtype.
        dtype (Optional[numpy.dtype]): Output dtype.

    Returns:

        float | int | numpy.ndarray: ``x.real ** 2 + x.imag ** 2``.
    """

def axis_sort(S: "ArrayLike", *, axis: int = ..., index: bool = False, value: Any | None = None) -> "NDArray[Any] | tuple[NDArray[Any], NDArray[np.int64]]":
    """
    Sorts the rows or columns of a matrix by a key computed from them.

    Args:

        S (numpy.ndarray): The matrix.
        axis (int): The axis to reorder: -1 reorders columns, 0 rows. Default -1.
        index (bool): Also return the permutation.
        value (Optional[Callable]): ``value(S, axis=other_axis)`` gives the sort key.
            Default: the position of the peak (``numpy.argmax``). Equal keys keep their
            order.

    Returns:

        numpy.ndarray | tuple[numpy.ndarray, numpy.ndarray[int64]]: The sorted matrix,
        and the permutation when ``index`` is true.
    """

def buf_to_float(x: Any, *, n_bytes: int = ..., dtype: Any | None = None) -> "NDArray[Any]":
    """
    Decodes little-endian signed integer PCM to floats in ``[-1, 1)``.

    Args:

        x (bytes-like | numpy.ndarray): The buffer.
        ``n_bytes`` (int): Bytes per sample: 1, 2, 3, 4 or 8. Default 2.
        dtype (Optional[numpy.dtype]): Output dtype. Default ``float32`` up to 3 bytes per
            sample, otherwise ``float64``, so that the conversion loses no bits it need
            not.

    Returns:

        numpy.ndarray: One float per sample.
    """

def chirp(*, fmin: float, fmax: float, sr: float = 22050.0, length: int | None = None, duration: float | None = None, linear: bool = False, phi: float | None = None) -> "NDArray[np.float64]":
    """
    A swept cosine, as float64.

    Args:

        fmin (float): Starting frequency in Hz.
        fmax (float): Frequency at the end in Hz.
        sr (float): Sample rate in Hz. Default 22050.
        length (Optional[int]): Samples.
        duration (Optional[float]): Seconds, if ``length`` is None.
        linear (bool): Sweep linearly; otherwise exponentially. Default False.
        phi (Optional[float]): Phase offset in radians. None means ``-pi / 2``.

    Returns:

        numpy.ndarray[float64]: The samples.
    """

def count_unique(data: "ArrayLike", *, axis: int = ...) -> "NDArray[np.int64]":
    """
    Counts the distinct values in each lane along an axis.

    Args:

        data (``array_like``): Values of any numeric dtype. All NaNs count as one value.
        axis (int): The axis of the lanes. Default -1.

    Returns:

        numpy.ndarray[int64]: The input's shape without ``axis``.
    """

def cyclic_gradient(data: "ArrayLike", *, edge_order: int = ..., axis: int = ...) -> "NDArray[Any]":
    """
    Central-difference gradient of periodic data along one axis.

    Args:

        data (``array_like``): Values. ``float32`` and ``complex64`` keep their precision;
            anything else is computed in ``float64``.
        ``edge_order`` (int): 1 or 2. Accepted for both orders: wrapped central differences
            have no edges, so the result does not depend on it.
        axis (int): The periodic axis. Default -1.

    Returns:

        numpy.ndarray: Same shape as ``data``.
    """

def dtype_c2r(d: "DTypeLike", *, default: Any | None = None) -> "np.dtype[Any]":
    """
    The real dtype of a complex dtype.

    Args:

        d (numpy.dtype): A dtype, or anything ``numpy.dtype`` accepts. A real floating
            dtype is returned as it is.
        default (numpy.dtype): Returned for any other dtype. Default ``float32``.

    Returns:

        numpy.dtype: The real dtype.
    """

def dtype_r2c(d: "DTypeLike", *, default: Any | None = None) -> "np.dtype[Any]":
    """
    The complex dtype of a real dtype.

    Args:

        d (numpy.dtype): A dtype, or anything ``numpy.dtype`` accepts. A complex dtype is
            returned as it is.
        default (numpy.dtype): Returned for any other dtype. Default ``complex64``.

    Returns:

        numpy.dtype: The complex dtype.
    """

def expand_to(x: "ArrayLike", *, ndim: int, axes: "int | Sequence[int]") -> "NDArray[Any]":
    """
    Places the axes of an array at given positions of a higher-dimensional view.

    Args:

        x (numpy.ndarray): The array.
        ndim (int): Dimensions of the result.
        axes (int | sequence of int): Position of each axis of ``x``; negative counts
            from the end. Positions out of order reorder the axes, so axis ``i`` of ``x``
            is always axis ``axes[i]`` of the result.

    Returns:

        numpy.ndarray: A view of the same data.
    """

def fill_off_diagonal(x: "NDArray[Any]", *, radius: float, value: Any | None = None) -> None:
    """
    Sets the entries of a matrix outside a diagonal band to a value, in place.

    Args:

        x (numpy.ndarray): The 2-D matrix, changed in place.
        radius (float): Half-width of the band as a fraction of the shorter side.
        value (number): The value written outside the band. Default 0.
    """

def find_files(directory: str | PathLike[str], *, ext: Any | None = None, recurse: bool = True, case_sensitive: bool = False, limit: int | None = None, offset: int = ...) -> list:
    """
    Lists audio files under a directory.

    Args:

        directory (str | os.PathLike): Where to look; ``~`` expands to the home directory.
        ext (Optional[str | list[str]]): Extensions without the dot. Default: ``aac``,
            ``au``, ``flac``, ``m4a``, ``mp3``, ``ogg``, ``wav``.
        recurse (bool): Search subdirectories. Default true.
        ``case_sensitive`` (bool): Compare extensions exactly; otherwise any letter case
            matches. Default false.
        limit (Optional[int]): Keep at most this many paths after ``offset``; a negative
            value drops that many from the end.
        offset (int): Skip this many paths; a negative value counts from the end.

    Returns:

        list[str]: Sorted absolute paths.
    """

def fix_frames(frames: "ArrayLike", *, x_min: int | None = ..., x_max: int | None = None, pad: bool = True) -> "NDArray[np.int64]":
    """
    Normalises frame indices: clips, optionally adds the bounds, removes duplicates and
    sorts.

    Args:

        frames (``array_like[int]``): Non-negative frame indices.
        ``x_min`` (Optional[int]): Lower bound. Default 0.
        ``x_max`` (Optional[int]): Upper bound.
        pad (bool): Clip to the bounds and include them, rather than only filter.

    Returns:

        numpy.ndarray[int64]: Sorted, unique indices.
    """

def index_to_slice(idx: "ArrayLike", *, idx_min: int | None = None, idx_max: int | None = None, step: int | None = None, pad: bool = True) -> list:
    """
    Converts boundary indices to the slices between consecutive boundaries.

    Args:

        idx (``array_like[int]``): Boundary indices.
        ``idx_min`` (Optional[int]): Lower bound.
        ``idx_max`` (Optional[int]): Upper bound.
        step (Optional[int]): Step of each slice.
        pad (bool): Clip to the bounds and include them, rather than only filter.

    Returns:

        list[slice]: One slice per pair of consecutive boundaries.
    """

def interp1d(x: "ArrayLike", y: "ArrayLike", xi: "ArrayLike", *, kind: Any | None = None, fill_value: float = 0.0, axis: int = ...) -> "NDArray[Any]":
    """
    Interpolates sampled data along one axis.

    Args:

        x (``array_like``): Sample positions, 1-D, in any order.
        y (``array_like``): Samples; ``axis`` has one element per position.
        xi (``array_like``): Query positions, of any shape.
        kind (str | int): ``"linear"``, ``"nearest"``, ``"nearest-up"``, ``"zero"``,
            ``"slinear"``, ``"quadratic"``, ``"cubic"``, ``"previous"``, ``"next"``, or a
            spline order 0 to 3. Default ``"linear"``.
        ``fill_value`` (float): Result outside the range of ``x``. Default 0.
        axis (int): The axis of ``y`` that follows ``x``. Default -1.

    Returns:

        numpy.ndarray: ``y``'s shape with ``axis`` replaced by the axes of ``xi``, of
        ``float32`` for ``float32`` ``y``, otherwise ``float64``.
    """

def interp_broadcast(*, x1: "ArrayLike", x1_pos: "ArrayLike", x2: "ArrayLike", x2_pos: "ArrayLike", interp_pos: Any | None = None, op: Any = ..., kind: Any | None = None, fill_value: float = 0.0, axis: int = ...) -> Any:
    """
    Interpolates two arrays sampled on different grids onto one grid and combines them.

    Args:

        x1 (``array_like``): The first array.
        ``x1_pos`` (``array_like``): Its sample positions along ``axis``.
        x2 (``array_like``): The second array.
        ``x2_pos`` (``array_like``): Its sample positions along ``axis``.
        ``interp_pos`` (Optional[array_like]): The common positions. Default ``x1_pos``.
        op (Callable | str | None): Combines the two interpolated arrays. ``"multiply"``
            (the default) multiplies with broadcasting; ``None`` returns both arrays.
        kind (str | int): As for :func:`interp1d`. Default ``"linear"``.
        ``fill_value`` (float): Result outside the range of each grid. Default 0.
        axis (int): The interpolated axis. Default -2.

    Returns:

        numpy.ndarray | tuple[numpy.ndarray, numpy.ndarray]: The combined array, or both
        arrays when ``op`` is ``None``.
    """

def is_positive_int(x: Any) -> bool:
    """
    Whether a value is an integer above zero.

    Args:

        x (Any): The value. ``None``, floats and everything else that is not an integer
            give ``False``.

    Returns:

        bool: The answer.
    """

def is_unique(data: "ArrayLike", *, axis: int = ...) -> "NDArray[np.bool_]":
    """
    Tells for each lane along an axis whether all its values are distinct.

    Args:

        data (``array_like``): Values of any numeric dtype. All NaNs count as one value.
        axis (int): The axis of the lanes. Default -1.

    Returns:

        numpy.ndarray[bool]: The input's shape without ``axis``.
    """

def localmax(x: "ArrayLike", *, axis: int = ...) -> "NDArray[np.bool_]":
    """
    Local maxima along an axis.

    ``x[i] > x[i - 1]`` and ``x[i] >= x[i + 1]``. The first element is never a
    maximum; the last is when it exceeds its neighbour.

    Args:

        x (``array_like``): Values of any real dtype.
        axis (int): Axis to compare along. Default 0.

    Returns:

        numpy.ndarray[bool]: Same shape as ``x``.
    """

def localmin(x: "ArrayLike", *, axis: int = ...) -> "NDArray[np.bool_]":
    """
    Local minima along an axis.

    ``x[i] < x[i - 1]`` and ``x[i] <= x[i + 1]``. The first element is never a
    minimum; the last is when it is below its neighbour.

    Args:

        x (``array_like``): Values of any real dtype.
        axis (int): Axis to compare along. Default 0.

    Returns:

        numpy.ndarray[bool]: Same shape as ``x``.
    """

def match_events(events_from: "ArrayLike", events_to: "ArrayLike", left: bool = True, right: bool = True) -> "NDArray[np.int64]":
    """
    Matches each source event to the nearest target event.

    Args:

        ``events_from`` (``array_like``): Source events, in any order.
        ``events_to`` (``array_like``): Target events, in any order.
        left (bool): A source event may match a target before it. Default true.
        right (bool): A source event may match a target after it. Default true.

    Returns:

        numpy.ndarray[int64]: For each source event, an index into ``events_to``. A source
        event half-way between two targets goes to the later one.

    Raises:

        ``ValueError``: A list is empty, or the restrictions leave an event without a match.
    """

def match_intervals(intervals_from: "ArrayLike", intervals_to: "ArrayLike", strict: bool = True) -> "NDArray[np.int64]":
    """
    Matches each source interval to the target interval it overlaps most.

    Overlap is the Jaccard index; on a tie the lowest target index wins.

    Args:

        ``intervals_from`` (``array_like``): Source intervals, ``(n, 2)``.
        ``intervals_to`` (``array_like``): Target intervals, ``(m, 2)``.
        strict (bool): A source interval that overlaps nothing is an error; when false it
            matches the nearest target (the earlier at equal distance). Default true.

    Returns:

        numpy.ndarray[int64]: For each source interval, an index into ``intervals_to``.

    Raises:

        ``ValueError``: A list is empty or invalid, or with ``strict`` an interval overlaps
        no target.
    """

def pad_center(data: "ArrayLike", *, size: int, axis: int = ..., **kwargs) -> "NDArray[Any]":
    """
    Pads an array along one axis to a given length, keeping the data centred.

    Args:

        data (numpy.ndarray): The array.
        size (int): Length of ``axis`` in the result.
        axis (int): The axis to pad. Default -1.
        **kwargs: ``mode`` and its arguments, as for ``numpy.pad``. Default: zeros.

    Returns:

        numpy.ndarray: The padded array, of the input's dtype.
    """

def phasor(angles: "ArrayLike", *, mag: Any | None = None) -> Any:
    """
    Complex phasors ``mag * (cos(angles) + 1j * sin(angles))``.

    Args:

        angles (float | ``array_like``): Angles in radians.
        mag (Optional[``complex`` | ``array_like``]): Multiplier, broadcast to ``angles``.

    Returns:

        ``complex`` | numpy.ndarray: ``complex64`` for ``float32`` angles, otherwise
        ``complex128``.
    """

def shear(X: Any, *, factor: int = ..., axis: int = ...) -> Any:
    """
    Circularly shifts each column (or row) of a matrix by ``factor`` times its index.

    Args:

        X (numpy.ndarray | scipy.sparse array or matrix | SparseRows): The matrix.
        factor (int): Shift per index. Default 1.
        axis (int): -1 shifts columns by their column index, 0 shifts rows by their row
            index. Default -1.

    Returns:

        The sheared matrix, in the input's form and dtype.
    """

def sparsify_rows(x: "ArrayLike", *, quantile: float = 0.01, dtype: Any | None = None) -> SparseRows:
    """
    Removes the smallest values of each row, keeping a sparse matrix.

    In each row the sorted magnitudes are divided by the row's total and summed
    cumulatively; entries below the first magnitude whose running total reaches
    ``quantile`` are dropped.

    Args:

        x (``array_like``): 1-D or 2-D values; 1-D is one row.
        quantile (float): Fraction of each row's total magnitude that may be discarded,
            in ``[0, 1)``. Default 0.01.
        dtype (Optional[numpy.dtype]): dtype of the stored values. Default: the input's.

    Returns:

        ``SparseRows``: The kept entries.
    """

def stack(arrays: "Sequence[ArrayLike]", *, axis: int = ...) -> "NDArray[Any]":
    """
    Stacks arrays of one shape along a new axis.

    Args:

        arrays (sequence of ``array_like``): At least one array.
        axis (int): Position of the new axis. Default 0.

    Returns:

        numpy.ndarray: The stacked array, of the arrays' common dtype.
    """

def tiny(x: Any) -> "np.floating[Any]":
    """
    The smallest positive normal value of the input's floating-point type.

    Args:

        x (``complex`` | ``array_like`` | numpy.dtype-bearing value): Determines the type. Complex
            types use their real type; non-float input uses ``float32``.

    Returns:

        numpy.floating: The value, as a ``NumPy`` scalar of that type.
    """

def tone(frequency: float, *, sr: float = 22050.0, length: int | None = None, duration: float | None = None, phi: float | None = None) -> "NDArray[np.float64]":
    """
    A cosine of a given frequency and phase, as float64.

    Args:

        frequency (float): In Hz.
        sr (float): Sample rate in Hz. Default 22050.
        length (Optional[int]): Samples.
        duration (Optional[float]): Seconds, if ``length`` is None.
        phi (Optional[float]): Phase offset in radians. None means ``-pi / 2``, which starts
            like a sine.

    Returns:

        numpy.ndarray[float64]: The samples.
    """

def valid_audio(y: "NDArray[Any]") -> bool:
    """
    Checks that audio samples are a floating-point array, at least 1-D, finite everywhere.

    Args:

        y (numpy.ndarray): The samples.

    Returns:

        bool: ``True``.

    Raises:

        ``TypeError``: ``y`` is not a floating-point ``numpy.ndarray``.
        ``ValueError``: ``y`` is 0-D or holds a NaN or infinity.
    """

def valid_int(x: Any, *, cast: Any | None = None) -> "int":
    """
    Converts a number to an ``int`` with a chosen rounding.

    Args:

        x (float): The value.
        cast (Optional[Callable]): Rounds ``x`` before conversion. Default: floor.

    Returns:

        int: The integer.

    Raises:

        ``TypeError``: ``cast`` is not callable.
    """

def valid_intervals(intervals: "ArrayLike") -> bool:
    """
    Checks that intervals have shape ``(n, 2)`` and non-negative durations.

    Args:

        intervals (``array_like``): One ``[start, end]`` row per interval.

    Returns:

        bool: ``True``.

    Raises:

        ``ValueError``: The shape is wrong or a duration is negative.
    """

def zero_crossings(y: "ArrayLike", *, threshold: float | None = ..., ref_magnitude: Any | None = None, pad: bool = True, zero_pos: bool = True, axis: int = ...) -> "NDArray[np.bool_]":
    """
    Marks the sign changes of a signal along one axis.

    Args:

        y (``array_like``): The signal, of any dimension.
        threshold (Optional[float]): Magnitudes up to this count as zero. None means 0.
            Default 1e-10.
        ``ref_magnitude`` (Optional[float | Callable]): Scales ``threshold``. A callable receives
            ``numpy.abs(y)`` and returns the scale.
        pad (bool): The mark of the first element. Default True.
        ``zero_pos`` (bool): Zero counts as positive; otherwise it is a sign of its own.
            Default True.
        axis (int): The time axis. Default -1.

    Returns:

        numpy.ndarray[bool]: The marks, of the shape of ``y``.
    """
