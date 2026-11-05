"""Dense linear algebra: non-negative least squares, the pseudo-inverse, the symmetric eigenproblem and the SVD."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import numpy.typing as npt
from numpy.typing import NDArray


def nnls(A: npt.ArrayLike, B: npt.ArrayLike, *, max_iter: int = 200, tol: float = 1e-05) -> NDArray[Any]:
    """Non-negative least squares: ``min ||A x - B||`` with ``x >= 0``."""
def pinv(a: npt.ArrayLike, *, rcond: float = 1e-15) -> NDArray[Any]:
    """The Moore-Penrose pseudo-inverse."""
def eigh(a: npt.ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Eigenvalues and eigenvectors of a symmetric matrix."""
def svd(
    a: npt.ArrayLike, *, k: Optional[int] = None, n_iter: int = 4, seed: int = 0
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Thin singular value decomposition, or the truncated one with ``k``."""

__all__ = ["__doc__", "eigh", "nnls", "pinv", "svd"]
