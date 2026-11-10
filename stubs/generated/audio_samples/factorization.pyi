"""
Non-negative matrix factorisation of spectrograms.
"""

from typing import Any, final

__all__ = ["Nmf", "decompose"]

@final
class Nmf:
    """
    Non-negative matrix factorisation ``X = W H``.

    Args:

        ``n_components`` (int): Number of components.
        init (Optional[str]): ``"random"``, ``"nndsvd"``, ``"nndsvda"``, ``"nndsvdar"``, or
            None: ``"nndsvda"`` when the components fit the matrix, else ``"random"``.
        solver (str): ``"cd"`` (coordinate descent, default) or ``"mu"`` (multiplicative).
        ``beta_loss`` (float | str): β of the divergence, or ``"frobenius"``,
            ``"kullback-leibler"``, ``"itakura-saito"``. Default ``"frobenius"``.
        tol (float): Relative stopping tolerance. Default 1e-4.
        ``max_iter`` (int): Most iterations. Default 200.
        ``random_state`` (int): Seed. Default 0.
        ``alpha_W`` (float): Regularisation strength of ``W``. Default 0.
        ``alpha_H`` (float | str): Regularisation strength of ``H``; ``"same"`` uses ``alpha_W``.
        ``l1_ratio`` (float): Share of L1 in the regularisation. Default 0.
        shuffle (bool): Visit coordinates in random order. Default false.
    """
    def __new__(cls, /, n_components: int, *, init: str | None = None, solver: str = "cd", beta_loss: Any | None = None, tol: float = 1e-4, max_iter: int = ..., random_state: int = 0, alpha_W: float = 0.0, alpha_H: Any | None = None, l1_ratio: float = 0.0, shuffle: bool = False) -> Nmf: ...
    def __repr__(self, /) -> str: ...
    @property
    def components_(self, /) -> Any:
        """
        ``(n_components, n_features)`` factor ``H``. Raises ``ValueError`` before fitting.
        """
    def fit_transform(self, /, X: "ArrayLike", W: Any | None = None, H: Any | None = None) -> "NDArray[Any]":
        """
        Fits the factorisation and returns ``W``.

        Args:

            X (``array_like``): ``(n_samples, n_features)``, non-negative.
            W (Optional[array_like]): Starting ``W``; give together with ``H``.
            H (Optional[array_like]): Starting ``H``.

        Returns:

            numpy.ndarray: ``(n_samples, n_components)``, of ``X``'s precision.
        """
    @property
    def n_iter_(self, /) -> int:
        """
        Iterations of the last fit.
        """
    @property
    def reconstruction_err_(self, /) -> float:
        """
        The final error of the last fit.
        """
    def transform(self, /, X: "ArrayLike") -> "NDArray[Any]":
        """
        Finds ``W`` for the fitted ``H``.

        Args:

            X (``array_like``): ``(n_samples, n_features)``, non-negative.

        Returns:

            numpy.ndarray: ``(n_samples, n_components)``, of ``X``'s precision.
        """

def decompose(S: "ArrayLike", *, n_components: int | None = None, transformer: Any | None = None, sort: bool = False, fit: bool = True, **kwargs) -> "tuple[NDArray[Any], NDArray[Any]]":
    """
    Factorises a spectrogram into templates and activations.

    Args:

        S (``array_like``): Non-negative ``(..., n_features, n_frames)`` spectrogram. Dimensions
            before the last are unrolled into one feature axis.
        ``n_components`` (Optional[int]): Number of components. Default: ``n_frames`` features
            of the unrolled matrix.
        transformer (Optional[object]): Any object with ``fit_transform``, ``transform``
            and ``components_``. Default: a new :class:`Nmf`.
        sort (bool): Order the components by the bin of their largest value (2-D only).
        fit (bool): Fit ``transformer``; false uses it as already fitted.
        **kwargs: Passed to :class:`Nmf`.

    Returns:

        tuple[numpy.ndarray, numpy.ndarray]: ``components`` ``(..., n_features, k)`` and
        ``activations`` ``(k, n_frames)``.
    """
