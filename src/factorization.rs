//! `audio_samples.factorization`: non-negative matrix factorisation of spectrograms.

use audio_samples::operations::factorization::{self as fa, NmfConfig, NmfInit, NmfSolver};
use ndarray::{Array2, Ix2};
use numpy::IntoPyArray;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

use crate::args::{
    Int, PyFloat, positive, precision_of, value_err, with_float_view, with_precision,
};
use crate::audio_err_to_py;

fn parse_init(init: Option<&str>) -> PyResult<NmfInit> {
    Ok(match init {
        None => NmfInit::Auto,
        Some("random") => NmfInit::Random,
        Some("nndsvd") => NmfInit::Nndsvd,
        Some("nndsvda") => NmfInit::Nndsvda,
        Some("nndsvdar") => NmfInit::Nndsvdar,
        Some(other) => {
            return Err(value_err(
                "init",
                format!(
                    "unknown {other:?}; expected 'random', 'nndsvd', 'nndsvda', 'nndsvdar' or None"
                ),
            ));
        }
    })
}

fn parse_solver(solver: &str) -> PyResult<NmfSolver> {
    match solver {
        "cd" => Ok(NmfSolver::CoordinateDescent),
        "mu" => Ok(NmfSolver::Multiplicative),
        other => Err(value_err(
            "solver",
            format!("unknown {other:?}; expected 'cd' or 'mu'"),
        )),
    }
}

fn parse_beta(beta: &Bound<'_, PyAny>) -> PyResult<f64> {
    if let Ok(name) = beta.extract::<String>() {
        return match name.as_str() {
            "frobenius" => Ok(2.0),
            "kullback-leibler" => Ok(1.0),
            "itakura-saito" => Ok(0.0),
            other => Err(value_err("beta_loss", format!("unknown {other:?}"))),
        };
    }
    beta.extract()
}

fn parse_alpha_h(alpha_h: &Bound<'_, PyAny>) -> PyResult<Option<f64>> {
    if let Ok(name) = alpha_h.extract::<String>() {
        if name == "same" {
            return Ok(None);
        }
        return Err(value_err("alpha_H", "expected a number or 'same'"));
    }
    Ok(Some(alpha_h.extract()?))
}

/// Non-negative matrix factorisation ``X = W H``.
///
/// Args:
///
///     ``n_components`` (int): Number of components.
///     init (Optional[str]): ``"random"``, ``"nndsvd"``, ``"nndsvda"``, ``"nndsvdar"``, or
///         None: ``"nndsvda"`` when the components fit the matrix, else ``"random"``.
///     solver (str): ``"cd"`` (coordinate descent, default) or ``"mu"`` (multiplicative).
///     ``beta_loss`` (float | str): β of the divergence, or ``"frobenius"``,
///         ``"kullback-leibler"``, ``"itakura-saito"``. Default ``"frobenius"``.
///     tol (float): Relative stopping tolerance. Default 1e-4.
///     ``max_iter`` (int): Most iterations. Default 200.
///     ``random_state`` (int): Seed. Default 0.
///     ``alpha_W`` (float): Regularisation strength of ``W``. Default 0.
///     ``alpha_H`` (float | str): Regularisation strength of ``H``; ``"same"`` uses ``alpha_W``.
///     ``l1_ratio`` (float): Share of L1 in the regularisation. Default 0.
///     shuffle (bool): Visit coordinates in random order. Default false.
#[pyclass(module = "audio_samples.factorization", frozen)]
pub struct Nmf {
    config: NmfConfig,
    /// ``(n_components, n_features)`` after fitting.
    components: std::sync::Mutex<Option<Py<PyAny>>>,
    /// Iterations of the last fit.
    fitted: std::sync::Mutex<(usize, f64)>,
}

#[pymethods]
impl Nmf {
    #[new]
    #[pyo3(
        signature = (n_components, *, init = None, solver = "cd", beta_loss = None, tol = 1e-4, max_iter = Int(200), random_state = 0, alpha_W = 0.0, alpha_H = None, l1_ratio = 0.0, shuffle = false),
        text_signature = "(n_components, *, init=None, solver='cd', beta_loss=None, tol=0.0001, max_iter=200, random_state=0, alpha_W=0.0, alpha_H=None, l1_ratio=0.0, shuffle=False)"
    )]
    #[allow(non_snake_case, clippy::too_many_arguments)]
    fn new(
        n_components: Int,
        init: Option<&str>,
        solver: &str,
        beta_loss: Option<&Bound<'_, PyAny>>,
        tol: f64,
        max_iter: Int,
        random_state: u64,
        alpha_W: f64,
        alpha_H: Option<&Bound<'_, PyAny>>,
        l1_ratio: f64,
        shuffle: bool,
    ) -> PyResult<Self> {
        let beta = beta_loss
            .filter(|b| !b.is_none())
            .map_or(Ok(2.0), parse_beta)?;
        let alpha_h = alpha_H
            .filter(|a| !a.is_none())
            .map_or(Ok(None), parse_alpha_h)?;
        let config = NmfConfig::new(positive("n_components", n_components)?.get())
            .with_init(parse_init(init)?)
            .with_solver(parse_solver(solver)?)
            .with_beta(beta)
            .with_tol(tol)
            .with_max_iter(crate::args::non_negative("max_iter", max_iter)?)
            .with_seed(random_state)
            .with_regularisation(alpha_W, alpha_h, l1_ratio)
            .with_shuffle(shuffle);
        config.validate().map_err(audio_err_to_py)?;
        Ok(Self {
            config,
            components: std::sync::Mutex::new(None),
            fitted: std::sync::Mutex::new((0, f64::NAN)),
        })
    }

    /// ``(n_components, n_features)`` factor ``H``. Raises ``ValueError`` before fitting.
    #[getter]
    #[allow(non_snake_case)]
    fn components_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.components
            .lock()
            .map_err(|_| value_err("components_", "poisoned"))?
            .as_ref()
            .map(|c| c.bind(py).clone())
            .ok_or_else(|| value_err("components_", "fit first"))
    }

    /// Iterations of the last fit.
    #[getter]
    #[allow(non_snake_case)]
    fn n_iter_(&self) -> PyResult<usize> {
        Ok(self
            .fitted
            .lock()
            .map_err(|_| value_err("n_iter_", "poisoned"))?
            .0)
    }

    /// The final error of the last fit.
    #[getter]
    #[allow(non_snake_case)]
    fn reconstruction_err_(&self) -> PyResult<f64> {
        Ok(self
            .fitted
            .lock()
            .map_err(|_| value_err("reconstruction_err_", "poisoned"))?
            .1)
    }

    /// Fits the factorisation and returns ``W``.
    ///
    /// Args:
    ///
    ///     X (``array_like``): ``(n_samples, n_features)``, non-negative.
    ///     W (Optional[array_like]): Starting ``W``; give together with ``H``.
    ///     H (Optional[array_like]): Starting ``H``.
    ///
    /// Returns:
    ///
    ///     numpy.ndarray: ``(n_samples, n_components)``, of ``X``'s precision.
    #[pyo3(signature = (X, W = None, H = None))]
    #[allow(non_snake_case)]
    fn fit_transform<'py>(
        &self,
        X: &Bound<'py, PyAny>,
        W: Option<&Bound<'py, PyAny>>,
        H: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        with_precision!(precision_of(X), |F| self.fit_in::<F>(X, W, H))
    }

    /// Finds ``W`` for the fitted ``H``.
    ///
    /// Args:
    ///
    ///     X (``array_like``): ``(n_samples, n_features)``, non-negative.
    ///
    /// Returns:
    ///
    ///     numpy.ndarray: ``(n_samples, n_components)``, of ``X``'s precision.
    #[allow(non_snake_case)]
    fn transform<'py>(&self, X: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        let py = X.py();
        let h = self.components_(py)?;
        let cfg = self.config;
        with_precision!(precision_of(X), |F| {
            with_float_view::<F, _>(X, "X", |x| {
                with_float_view::<F, _>(&h, "components_", |hv| {
                    let x = x
                        .into_dimensionality::<Ix2>()
                        .map_err(|_| value_err("X", "expected a 2-D matrix"))?;
                    let hv = hv
                        .into_dimensionality::<Ix2>()
                        .map_err(|e| value_err("components_", e))?;
                    let w = py
                        .detach(|| fa::nmf_transform(x, hv, &cfg))
                        .map_err(audio_err_to_py)?;
                    Ok(w.into_pyarray(py).into_any())
                })
            })
        })
    }

    fn __repr__(&self) -> String {
        format!("Nmf(n_components={})", self.config.n_components)
    }
}

impl Nmf {
    fn fit_in<'py, F: PyFloat>(
        &self,
        x: &Bound<'py, PyAny>,
        w: Option<&Bound<'py, PyAny>>,
        h: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = x.py();
        let cfg = self.config;
        let start = match (w.filter(|v| !v.is_none()), h.filter(|v| !v.is_none())) {
            (Some(w), Some(h)) => {
                let w0: Array2<F> = with_float_view::<F, _>(w, "W", |v| {
                    v.into_dimensionality::<Ix2>()
                        .map(|a| a.to_owned())
                        .map_err(|e| value_err("W", e))
                })?;
                let h0: Array2<F> = with_float_view::<F, _>(h, "H", |v| {
                    v.into_dimensionality::<Ix2>()
                        .map(|a| a.to_owned())
                        .map_err(|e| value_err("H", e))
                })?;
                Some((w0, h0))
            }
            (None, None) => None,
            _ => return Err(value_err("W", "give W and H together")),
        };
        let r = with_float_view::<F, _>(x, "X", |v| {
            let v = v
                .into_dimensionality::<Ix2>()
                .map_err(|_| value_err("X", "expected a 2-D matrix"))?;
            py.detach(|| match start {
                Some((w0, h0)) => fa::nmf_from(v, w0, h0, &cfg),
                None => fa::nmf(v, &cfg),
            })
            .map_err(audio_err_to_py)
        })?;
        *self
            .fitted
            .lock()
            .map_err(|_| value_err("fit", "poisoned"))? = (r.n_iter, r.reconstruction_err);
        *self
            .components
            .lock()
            .map_err(|_| value_err("fit", "poisoned"))? =
            Some(r.h.into_pyarray(py).into_any().unbind());
        Ok(r.w.into_pyarray(py).into_any())
    }
}

/// Factorises a spectrogram into templates and activations.
///
/// Args:
///
///     S (``array_like``): Non-negative ``(..., n_features, n_frames)`` spectrogram. Dimensions
///         before the last are unrolled into one feature axis.
///     ``n_components`` (Optional[int]): Number of components. Default: ``n_frames`` features
///         of the unrolled matrix.
///     transformer (Optional[object]): Any object with ``fit_transform``, ``transform``
///         and ``components_``. Default: a new :class:`Nmf`.
///     sort (bool): Order the components by the bin of their largest value (2-D only).
///     fit (bool): Fit ``transformer``; false uses it as already fitted.
///     **kwargs: Passed to :class:`Nmf`.
///
/// Returns:
///
///     tuple[numpy.ndarray, numpy.ndarray]: ``components`` ``(..., n_features, k)`` and
///     ``activations`` ``(k, n_frames)``.
#[pyfunction]
#[pyo3(
    signature = (S, *, n_components = None, transformer = None, sort = false, fit = true, **kwargs),
    text_signature = "(S, *, n_components=None, transformer=None, sort=False, fit=True, **kwargs)"
)]
#[allow(non_snake_case)]
#[inline]
pub fn decompose<'py>(
    S: &Bound<'py, PyAny>,
    n_components: Option<Int>,
    transformer: Option<&Bound<'py, PyAny>>,
    sort: bool,
    fit: bool,
    kwargs: Option<&Bound<'py, PyDict>>,
) -> PyResult<Bound<'py, PyAny>> {
    let py = S.py();
    let numpy = py.import("numpy")?;
    let s = numpy.getattr("asarray")?.call1((S,))?;
    let shape: Vec<usize> = s.getattr("shape")?.extract()?;
    if shape.len() < 2 {
        return Err(value_err("S", "expected at least 2 dimensions"));
    }
    if shape.len() > 2 && sort {
        return Err(value_err("sort", "is only supported for 2-D input"));
    }
    let Some((&frames, lead)) = shape.split_last() else {
        return Err(value_err("S", "expected at least 2 dimensions"));
    };
    let order_f = PyDict::new(py);
    order_f.set_item("order", "F")?;
    let x = s
        .getattr("T")?
        .call_method("reshape", ((frames, -1_i64),), Some(&order_f))?;
    let features: usize = x.getattr("shape")?.get_item(1)?.extract()?;
    let k = match n_components {
        Some(v) => positive("n_components", v)?.get(),
        None => features,
    };
    let default_model;
    let model: &Bound<'py, PyAny> = if let Some(t) = transformer.filter(|t| !t.is_none()) {
        t
    } else {
        if !fit {
            return Err(value_err("fit", "must be true without a transformer"));
        }
        default_model = py.get_type::<Nmf>().call((k,), kwargs)?;
        &default_model
    };
    let w = if fit {
        model.call_method1("fit_transform", (&x,))?
    } else {
        model.call_method1("transform", (&x,))?
    };
    let activations = w.getattr("T")?;
    let raw = model.getattr("components_")?;
    let mut shape_out = lead
        .iter()
        .map(|&d| i64::try_from(d))
        .collect::<Result<Vec<i64>, _>>()
        .map_err(|e| value_err("S", e))?;
    shape_out.push(-1);
    shape_out.reverse();
    let components = raw
        .call_method("reshape", (PyTuple::new(py, shape_out)?,), Some(&order_f))?
        .getattr("T")?;
    if sort {
        let by = numpy.getattr("argmax")?;
        let kw = PyDict::new(py);
        kw.set_item("axis", 0)?;
        let keys = by.call((&components,), Some(&kw))?;
        let kw = PyDict::new(py);
        kw.set_item("kind", "stable")?;
        let idx = numpy.getattr("argsort")?.call((keys,), Some(&kw))?;
        let sorted = components.call_method1("take", (&idx, 1))?;
        let act = activations.call_method1("take", (&idx, 0))?;
        return pyo3::IntoPyObjectExt::into_bound_py_any((sorted, act), py);
    }
    pyo3::IntoPyObjectExt::into_bound_py_any((components, activations), py)
}

#[inline]
pub fn register(py: Python<'_>, parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(py, "factorization")?;
    m.add(
        "__doc__",
        "Non-negative matrix factorisation of spectrograms.",
    )?;
    m.add_class::<Nmf>()?;
    m.add_function(wrap_pyfunction!(decompose, &m)?)?;
    parent.add_submodule(&m)?;
    Ok(())
}
