"""Non-negative matrix factorisation of spectrograms."""

from __future__ import annotations

from typing import Any, Literal, Optional, Union, final

import numpy as np
import numpy.typing as npt
from numpy.typing import NDArray


@final
class Nmf:
    """Non-negative matrix factorisation ``X = W H``."""

    def __new__(
        cls,
        n_components: int,
        *,
        init: Optional[Literal["random", "nndsvd", "nndsvda", "nndsvdar"]] = None,
        solver: Literal["cd", "mu"] = "cd",
        beta_loss: Optional[Union[float, Literal["frobenius", "kullback-leibler", "itakura-saito"]]] = None,
        tol: float = 0.0001,
        max_iter: int = 200,
        random_state: int = 0,
        alpha_W: float = 0.0,
        alpha_H: Optional[Union[float, Literal["same"]]] = None,
        l1_ratio: float = 0.0,
        shuffle: bool = False,
    ) -> Nmf: ...
    @property
    def components_(self) -> NDArray[Any]: ...
    @property
    def n_iter_(self) -> int: ...
    @property
    def reconstruction_err_(self) -> float: ...
    def fit_transform(
        self, X: npt.ArrayLike, W: Optional[npt.ArrayLike] = None, H: Optional[npt.ArrayLike] = None
    ) -> NDArray[Any]:
        """Fits the factorisation and returns ``W``."""
    def transform(self, X: npt.ArrayLike) -> NDArray[Any]:
        """Finds ``W`` for the fitted ``H``."""


def decompose(
    S: npt.ArrayLike,
    *,
    n_components: Optional[int] = None,
    transformer: Optional[Any] = None,
    sort: bool = False,
    fit: bool = True,
    **kwargs: Any,
) -> tuple[NDArray[Any], NDArray[Any]]:
    """Factorises a spectrogram into templates and activations."""

__all__ = ["__doc__", "Nmf", "decompose"]
