//! `audio_samples.linalg`: non-negative least squares, the pseudo-inverse, the symmetric
//! eigenproblem and the singular value decomposition.

use audio_samples::operations::linalg::{self as la, NnlsConfig, NnlsSolver};
use ndarray::{Array2, Ix1, Ix2};
use numpy::IntoPyArray;
use pyo3::IntoPyObjectExt;
use pyo3::prelude::*;

use crate::args::{
    Int, PyFloat, non_negative, positive, precision_of, value_err, with_float_view, with_precision,
};
use crate::audio_err_to_py;

fn nnls_in<'py, F: PyFloat>(
    a: &Bound<'py, PyAny>,
    b: &Bound<'py, PyAny>,
    max_iter: usize,
    tol: f64,
) -> PyResult<Bound<'py, PyAny>> {
    let py = a.py();
    with_float_view::<F, _>(a, "A", |av| {
        let av = av
            .into_dimensionality::<Ix2>()
            .map_err(|_| value_err("A", "expected a 2-D matrix"))?;
        with_float_view::<F, _>(b, "B", |bv| {
            let cfg = NnlsConfig::new().with_max_iter(max_iter).with_tol(tol);
            match bv.ndim() {
                1 => {
                    let bv = bv
                        .into_dimensionality::<Ix1>()
                        .map_err(|e| value_err("B", e))?;
                    let x = py.detach(|| la::nnls(av, bv)).map_err(audio_err_to_py)?;
                    Ok(x.into_pyarray(py).into_any())
                }
                2 => {
                    let bv = bv
                        .into_dimensionality::<Ix2>()
                        .map_err(|e| value_err("B", e))?;
                    let x = py
                        .detach(|| la::nnls_matrix(av, bv, &cfg))
                        .map_err(audio_err_to_py)?;
                    Ok(x.into_pyarray(py).into_any())
                }
                _ => {
                    let solver = py
                        .detach(|| NnlsSolver::new(av, &cfg))
                        .map_err(audio_err_to_py)?;
                    let shape = bv.shape().to_vec();
                    let (lead, rows, cols) = match shape.as_slice() {
                        [lead @ .., rows, cols] => (lead, *rows, *cols),
                        _ => return Err(value_err("B", "expected at least two dimensions")),
                    };
                    if rows == 0 || cols == 0 {
                        return Err(value_err("B", "must not be empty"));
                    }
                    let n_out = av.ncols();
                    let planes: usize = lead.iter().product();
                    let flat = bv.as_standard_layout();
                    let flat = flat
                        .as_slice()
                        .ok_or_else(|| value_err("B", "expected contiguous data"))?;
                    let mut out = Vec::with_capacity(planes * n_out * cols);
                    for chunk in flat.chunks_exact(rows * cols) {
                        let plane = ndarray::ArrayView2::from_shape((rows, cols), chunk)
                            .map_err(|e| value_err("B", e))?;
                        let solution: Array2<F> =
                            py.detach(|| solver.solve(plane)).map_err(audio_err_to_py)?;
                        out.extend(solution.iter().copied());
                    }
                    let mut full = lead.to_vec();
                    full.extend([n_out, cols]);
                    let result = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&full), out)
                        .map_err(|e| value_err("B", e))?;
                    Ok(result.into_pyarray(py).into_any())
                }
            }
        })
    })
}

/// Non-negative least squares: ``min ||A x - B||`` with ``x >= 0``.
///
/// A 1-D ``B`` gets the exact active-set solution. Each column of a 2-D or higher ``B``
/// starts from the pseudo-inverse solution with negative entries set to zero and is
/// refined by coordinate descent, which keeps it near the minimum-norm solution when
/// ``A`` has more columns than rows.
///
/// Args:
///
///     A (``array_like``): ``(m, n)`` matrix.
///     B (``array_like``): ``(m,)`` or ``(..., m, t)``.
///     ``max_iter`` (int): Most sweeps per column of a 2-D ``B``. Default 200.
///     tol (float): Stop a column when no sweep changes ``A x`` by more than this
///         fraction of ``max|B|``. Default 1e-5.
///
/// Returns:
///
///     numpy.ndarray: ``(n,)`` or ``(..., n, t)``, of ``B``'s precision.
///
/// Raises:
///
///     ``ValueError``: Dimensions disagree or an input is not finite.
#[pyfunction]
#[pyo3(name = "nnls", signature = (A: "ArrayLike", B: "ArrayLike", *, max_iter = Int(200), tol = 1e-5) -> "NDArray[Any]", text_signature = "(A, B, *, max_iter=200, tol=1e-05)")]
#[allow(non_snake_case, reason = "argument names follow the Python API")]
#[inline]
pub fn nnls<'py>(
    A: &Bound<'py, PyAny>,
    B: &Bound<'py, PyAny>,
    max_iter: Int,
    tol: f64,
) -> PyResult<Bound<'py, PyAny>> {
    let max_iter = non_negative("max_iter", max_iter)?;
    // A single-precision matrix and right-hand side stay in single precision.
    let single = precision_of(A) == crate::args::Precision::F32
        && precision_of(B) == crate::args::Precision::F32;
    if single {
        nnls_in::<f32>(A, B, max_iter, tol)
    } else {
        nnls_in::<f64>(A, B, max_iter, tol)
    }
}

/// The Moore-Penrose pseudo-inverse.
///
/// Args:
///
///     a (``array_like``): ``(m, n)`` matrix.
///     rcond (float): Singular values below this fraction of the largest count as zero.
///         Values below about 1.5e-8 act as that limit. Default 1e-15.
///
/// Returns:
///
///     numpy.ndarray: ``(n, m)``, of ``a``'s precision.
#[pyfunction]
#[pyo3(signature = (a: "ArrayLike", *, rcond = 1e-15) -> "NDArray[Any]", text_signature = "(a, *, rcond=1e-15)")]
#[inline]
pub fn pinv<'py>(a: &Bound<'py, PyAny>, rcond: f64) -> PyResult<Bound<'py, PyAny>> {
    let py = a.py();
    with_precision!(precision_of(a), |F| {
        with_float_view::<F, _>(a, "a", |v| {
            let v = v
                .into_dimensionality::<Ix2>()
                .map_err(|_| value_err("a", "expected a 2-D matrix"))?;
            let p = py.detach(|| la::pinv(v, rcond)).map_err(audio_err_to_py)?;
            Ok(p.into_pyarray(py).into_any())
        })
    })
}

/// Eigenvalues and eigenvectors of a symmetric matrix; only the lower triangle is read.
///
/// Args:
///
///     a (``array_like``): ``(n, n)`` symmetric matrix.
///
/// Returns:
///
///     tuple[numpy.ndarray, numpy.ndarray]: Increasing eigenvalues, and the matching
///     eigenvectors in the columns, both ``float64``.
#[pyfunction]
#[inline]
#[pyo3(signature = (a: "ArrayLike") -> "tuple[NDArray[np.float64], NDArray[np.float64]]")]
pub fn eigh<'py>(a: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let py = a.py();
    with_float_view::<f64, _>(a, "a", |v| {
        let v = v
            .into_dimensionality::<Ix2>()
            .map_err(|_| value_err("a", "expected a 2-D matrix"))?;
        let (w, z) = py.detach(|| la::eigh(v)).map_err(audio_err_to_py)?;
        (w.into_pyarray(py), z.into_pyarray(py)).into_bound_py_any(py)
    })
}

/// Thin singular value decomposition, or the truncated one with ``k``.
///
/// Computed through the Gram matrix: singular values below about 1.5e-8 of the largest
/// are reported as zero. The truncated form finds the leading triplets by randomised range
/// finding, deterministically for a given ``seed``.
///
/// Args:
///
///     a (``array_like``): ``(m, n)`` matrix.
///     k (Optional[int]): Number of leading triplets. Default: all.
///     ``n_iter`` (int): Power iterations of the truncated form. Default 4.
///     seed (int): Seed of the truncated form. Default 0.
///
/// Returns:
///
///     tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]: ``u``, ``s``, ``vt``, ``float64``.
#[pyfunction]
#[pyo3(signature = (a: "ArrayLike", *, k = None, n_iter = Int(4), seed = 0) -> "tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]", text_signature = "(a, *, k=None, n_iter=4, seed=0)")]
#[inline]
pub fn svd<'py>(
    a: &Bound<'py, PyAny>,
    k: Option<Int>,
    n_iter: Int,
    seed: u64,
) -> PyResult<Bound<'py, PyAny>> {
    let py = a.py();
    let n_iter = non_negative("n_iter", n_iter)?;
    let k = k.map(|v| positive("k", v)).transpose()?;
    with_float_view::<f64, _>(a, "a", |v| {
        let v = v
            .into_dimensionality::<Ix2>()
            .map_err(|_| value_err("a", "expected a 2-D matrix"))?;
        let d = py
            .detach(|| {
                k.map_or_else(
                    || la::svd(v),
                    |k| la::randomized_svd(v, k.get(), 10, n_iter, seed),
                )
            })
            .map_err(audio_err_to_py)?;
        (
            d.u.into_pyarray(py),
            d.s.into_pyarray(py),
            d.vt.into_pyarray(py),
        )
            .into_bound_py_any(py)
    })
}

#[doc = "Dense linear algebra: non-negative least squares, the pseudo-inverse, the symmetric eigenproblem and the SVD."]
#[pymodule(submodule, name = "linalg")]
pub mod py_module {
    #[pymodule_export]
    use super::{eigh, nnls, pinv, svd};
}
