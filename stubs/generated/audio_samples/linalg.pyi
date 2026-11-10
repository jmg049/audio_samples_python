"""
Dense linear algebra: non-negative least squares, the pseudo-inverse, the symmetric eigenproblem and the SVD.
"""

__all__ = ["eigh", "nnls", "pinv", "svd"]

def eigh(a: "ArrayLike") -> "tuple[NDArray[np.float64], NDArray[np.float64]]":
    """
    Eigenvalues and eigenvectors of a symmetric matrix; only the lower triangle is read.

    Args:

        a (``array_like``): ``(n, n)`` symmetric matrix.

    Returns:

        tuple[numpy.ndarray, numpy.ndarray]: Increasing eigenvalues, and the matching
        eigenvectors in the columns, both ``float64``.
    """

def nnls(A: "ArrayLike", B: "ArrayLike", *, max_iter: int = ..., tol: float = 1e-5) -> "NDArray[Any]":
    """
    Non-negative least squares: ``min ||A x - B||`` with ``x >= 0``.

    A 1-D ``B`` gets the exact active-set solution. Each column of a 2-D or higher ``B``
    starts from the pseudo-inverse solution with negative entries set to zero and is
    refined by coordinate descent, which keeps it near the minimum-norm solution when
    ``A`` has more columns than rows.

    Args:

        A (``array_like``): ``(m, n)`` matrix.
        B (``array_like``): ``(m,)`` or ``(..., m, t)``.
        ``max_iter`` (int): Most sweeps per column of a 2-D ``B``. Default 200.
        tol (float): Stop a column when no sweep changes ``A x`` by more than this
            fraction of ``max|B|``. Default 1e-5.

    Returns:

        numpy.ndarray: ``(n,)`` or ``(..., n, t)``, of ``B``'s precision.

    Raises:

        ``ValueError``: Dimensions disagree or an input is not finite.
    """

def pinv(a: "ArrayLike", *, rcond: float = 1e-15) -> "NDArray[Any]":
    """
    The Moore-Penrose pseudo-inverse.

    Args:

        a (``array_like``): ``(m, n)`` matrix.
        rcond (float): Singular values below this fraction of the largest count as zero.
            Values below about 1.5e-8 act as that limit. Default 1e-15.

    Returns:

        numpy.ndarray: ``(n, m)``, of ``a``'s precision.
    """

def svd(a: "ArrayLike", *, k: int | None = None, n_iter: int = ..., seed: int = 0) -> "tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]":
    """
    Thin singular value decomposition, or the truncated one with ``k``.

    Computed through the Gram matrix: singular values below about 1.5e-8 of the largest
    are reported as zero. The truncated form finds the leading triplets by randomised range
    finding, deterministically for a given ``seed``.

    Args:

        a (``array_like``): ``(m, n)`` matrix.
        k (Optional[int]): Number of leading triplets. Default: all.
        ``n_iter`` (int): Power iterations of the truncated form. Default 4.
        seed (int): Seed of the truncated form. Default 0.

    Returns:

        tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]: ``u``, ``s``, ``vt``, ``float64``.
    """
