//! Python bindings for the "explain" API (`audio_samples.explain`).
//!
//! Wraps `audio_samples::educational::explain`: an [`ExplainDoc`] describes
//! what a configuration (an `StftParams`, an `IirFilterDesign`, ...) or a
//! concept (sampling, aliasing, ...) will do, with the formulas the code
//! evaluates, numbers computed from the settings, trade-offs and warnings.
//!
//! Python surface:
//! * `audio_samples.ExplainDoc` (also `audio_samples.explain.ExplainDoc`);
//! * `audio_samples.explain.explain(obj, *, sample_rate=..., ...)`, a
//!   dispatcher over every bound configuration class with an explainer;
//! * topic functions mirroring the Rust `topics` module;
//! * `AudioSamples.explain(obj)` and `.explain(...)` methods on the config
//!   classes owned by this crate.

use audio_samples::educational::{Explain, ExplainContext, ExplainDoc, WarningLevel, topics};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use spectrograms::python::{
    PyMelParams, PyMfccParams, PySpectrogramParams, PyStftParams, PyWindowType as PySpecWindowType,
};
use spectrograms::{MelParams, MfccParams, SpectrogramParams, WindowType};

use crate::types::{
    PyCompressorConfig, PyEqBand, PyIirFilterDesign, PyLimiterConfig, PyOnsetDetectionConfig,
    PyVadConfig, PyWindowType as PyTypesWindowType,
};
use crate::{PyAudioSamples, dispatch_with_view};

/// Names of the classes `explain()` accepts, for the `TypeError` message.
const SUPPORTED: &str = "StftParams, SpectrogramParams, MelParams, MfccParams, WindowType, \
     IirFilterDesign, EqBand, CompressorConfig, LimiterConfig, VadConfig, OnsetDetectionConfig";

/// A structured explanation of a configuration or concept.
///
/// An ``ExplainDoc`` says what a configuration will do (summary), the formulas
/// the code actually evaluates, the symbols in those formulas with their
/// values, quantities derived from your settings (frequency resolution, group
/// delay, gain at the cutoff, ...), tables, trade-offs and warnings about
/// suspicious settings. Every number is computed from the settings.
///
/// Obtain one from :func:`audio_samples.explain.explain`, from a topic
/// function such as :func:`audio_samples.explain.aliasing`, from
/// :meth:`AudioSamples.explain`, or from ``config.explain(...)`` on a filter,
/// dynamics or detection configuration. It is read-only.
///
/// Rendering: ``str(doc)`` / :meth:`text` for a terminal (Unicode maths),
/// :meth:`markdown` for Jupyter, mdBook or MyST (``$$`` LaTeX blocks), and
/// :meth:`html` for a self-contained HTML fragment. In a notebook the last
/// expression ``doc`` renders as Markdown (maths typeset by MathJax).
///
/// Examples:
///
///     >>> from audio_samples.explain import aliasing
///     >>> doc = aliasing(7000.0, 8000.0)
///     >>> doc.quantity("Alias frequency")
///     1000.0
#[pyclass(name = "ExplainDoc", module = "audio_samples.explain", frozen)]
pub struct PyExplainDoc {
    pub(crate) inner: ExplainDoc,
}

impl From<ExplainDoc> for PyExplainDoc {
    fn from(inner: ExplainDoc) -> Self {
        Self { inner }
    }
}

fn parse_level(level: &str) -> PyResult<WarningLevel> {
    match level.to_ascii_lowercase().as_str() {
        "info" => Ok(WarningLevel::Info),
        "caution" => Ok(WarningLevel::Caution),
        "problem" => Ok(WarningLevel::Problem),
        other => Err(PyValueError::new_err(format!(
            "unknown warning level '{other}' (expected 'info', 'caution' or 'problem')"
        ))),
    }
}

#[pymethods]
impl PyExplainDoc {
    /// Title of the explanation, e.g. ``"Short-time Fourier transform (STFT)"``.
    #[getter]
    fn title(&self) -> &str {
        &self.inner.title
    }

    /// Syllabus topic label, e.g. ``"Windowing"`` or ``"IIR filters"``.
    #[getter]
    const fn topic(&self) -> &'static str {
        self.inner.topic.label()
    }

    /// One to three sentences saying what will happen, in plain words.
    #[getter]
    fn summary(&self) -> &str {
        &self.inner.summary
    }

    /// Formulas the explained code evaluates.
    ///
    /// Returns:
    ///
    ///     list[dict]: One dict per formula with keys ``latex`` (math-mode
    ///         LaTeX without ``$`` delimiters) and ``caption``.
    #[getter]
    fn formulas<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let list = PyList::empty(py);
        for f in &self.inner.formulas {
            let d = PyDict::new(py);
            d.set_item("latex", &f.latex)?;
            d.set_item("caption", &f.caption)?;
            list.append(d)?;
        }
        Ok(list)
    }

    /// Symbols used in the formulas.
    ///
    /// Returns:
    ///
    ///     list[dict]: One dict per symbol with keys ``symbol`` (LaTeX),
    ///         ``meaning``, ``value`` (str or None) and ``unit`` (str or None).
    #[getter]
    fn symbols<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let list = PyList::empty(py);
        for s in &self.inner.symbols {
            let d = PyDict::new(py);
            d.set_item("symbol", &s.symbol)?;
            d.set_item("meaning", &s.meaning)?;
            d.set_item("value", s.value.as_deref())?;
            d.set_item("unit", s.unit.as_deref())?;
            list.append(d)?;
        }
        Ok(list)
    }

    /// Quantities computed from the settings.
    ///
    /// Returns:
    ///
    ///     list[dict]: One dict per quantity with keys ``name``, ``symbol``
    ///         (LaTeX), ``value`` (float, always finite), ``unit`` and ``how``
    ///         (the LaTeX formula used to compute it).
    #[getter]
    fn derived<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let list = PyList::empty(py);
        for q in &self.inner.derived {
            let d = PyDict::new(py);
            d.set_item("name", &q.name)?;
            d.set_item("symbol", &q.symbol)?;
            d.set_item("value", q.value)?;
            d.set_item("unit", &q.unit)?;
            d.set_item("how", &q.how)?;
            list.append(d)?;
        }
        Ok(list)
    }

    /// Tables (pole/zero lists, static curves, band edges, ...).
    ///
    /// Returns:
    ///
    ///     list[dict]: One dict per table with keys ``caption`` (str),
    ///         ``headers`` (list[str]) and ``rows`` (list[list[str]]). Cells
    ///         that start and end with ``$`` are LaTeX.
    #[getter]
    fn tables<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let list = PyList::empty(py);
        for t in &self.inner.tables {
            let d = PyDict::new(py);
            d.set_item("caption", &t.caption)?;
            d.set_item("headers", t.headers.clone())?;
            d.set_item("rows", t.rows.clone())?;
            list.append(d)?;
        }
        Ok(list)
    }

    /// Resolution, latency and leakage trade-offs, stated for these settings.
    ///
    /// Returns:
    ///
    ///     list[str]: One sentence or short paragraph per trade-off.
    #[getter]
    fn trade_offs(&self) -> Vec<String> {
        self.inner.trade_offs.clone()
    }

    /// Computed observations about the settings.
    ///
    /// Returns:
    ///
    ///     list[dict]: One dict per warning with keys ``level`` (``"info"``,
    ///         ``"caution"`` or ``"problem"``), ``message`` and ``hint``
    ///         (str or None).
    #[getter]
    fn warnings<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let list = PyList::empty(py);
        for w in &self.inner.warnings {
            let d = PyDict::new(py);
            d.set_item("level", w.level.label())?;
            d.set_item("message", &w.message)?;
            d.set_item("hint", w.hint.as_deref())?;
            list.append(d)?;
        }
        Ok(list)
    }

    /// Related reading: Rust paths, Python names, guide chapters.
    ///
    /// Returns:
    ///
    ///     list[dict]: One dict per reference with keys ``label`` and ``target``.
    #[getter]
    fn see_also<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let list = PyList::empty(py);
        for s in &self.inner.see_also {
            let d = PyDict::new(py);
            d.set_item("label", &s.label)?;
            d.set_item("target", &s.target)?;
            list.append(d)?;
        }
        Ok(list)
    }

    /// Render for a terminal.
    ///
    /// Formulas are drawn as Unicode maths; no raw LaTeX reaches the output.
    ///
    /// Returns:
    ///
    ///     str: A multi-line string (also ``str(doc)``).
    #[pyo3(signature = ())]
    fn text(&self) -> String {
        self.inner.to_text()
    }

    /// Render as Markdown with ``$$...$$`` LaTeX blocks.
    ///
    /// Renders in Jupyter (MathJax), mdBook and MyST. This is what a notebook
    /// shows for a bare ``doc`` expression.
    ///
    /// Returns:
    ///
    ///     str: Markdown source.
    #[pyo3(signature = ())]
    fn markdown(&self) -> String {
        self.inner.to_markdown()
    }

    /// Render as a self-contained HTML fragment (no JavaScript).
    ///
    /// Returns:
    ///
    ///     str: HTML source.
    #[pyo3(signature = ())]
    fn html(&self) -> String {
        self.inner.to_html()
    }

    /// Look up a derived quantity by name.
    ///
    /// Args:
    ///
    ///     name (str): Quantity name exactly as listed in :attr:`derived`,
    ///         e.g. ``"Frequency resolution"``.
    ///
    /// Returns:
    ///
    ///     float | None: The value of the first quantity with that name, or
    ///         None if there is none.
    ///
    /// Examples:
    ///
    ///     >>> from audio_samples.explain import sampling
    ///     >>> sampling(48000.0).quantity("Nyquist frequency")
    ///     24000.0
    #[pyo3(signature = (name: "str"))]
    fn quantity(&self, name: &str) -> Option<f64> {
        self.inner.quantity(name).map(|q| q.value)
    }

    /// Whether the explanation carries a warning of the given level.
    ///
    /// Args:
    ///
    ///     level (str | None): ``"info"``, ``"caution"`` or ``"problem"``
    ///         (case-insensitive), matched exactly. None (the default) matches
    ///         any warning.
    ///
    /// Returns:
    ///
    ///     bool: True if at least one such warning exists.
    ///
    /// Raises:
    ///
    ///     ValueError: If ``level`` is not a known warning level.
    #[pyo3(signature = (level: "str | None" = None))]
    fn has_warning(&self, level: Option<&str>) -> PyResult<bool> {
        match level {
            None => Ok(!self.inner.warnings.is_empty()),
            Some(l) => Ok(self.inner.has_warning(parse_level(l)?)),
        }
    }

    /// Whether any warning message contains ``text``.
    ///
    /// Args:
    ///
    ///     text (str): Substring to search for (case-sensitive).
    ///
    /// Returns:
    ///
    ///     bool: True if a matching warning exists.
    #[pyo3(signature = (text: "str"))]
    fn has_warning_containing(&self, text: &str) -> bool {
        self.inner.has_warning_containing(text)
    }

    fn __str__(&self) -> String {
        self.inner.to_text()
    }

    fn __repr__(&self) -> String {
        let d = &self.inner;
        format!(
            "<ExplainDoc {:?} ({}): {} formulas, {} quantities, {} tables, {} warnings>",
            d.title,
            d.topic.label(),
            d.formulas.len(),
            d.derived.len(),
            d.tables.len(),
            d.warnings.len()
        )
    }

    /// Jupyter rich display: Markdown with ``$$`` maths (typeset by MathJax).
    ///
    /// No ``_repr_html_`` is defined on purpose: Jupyter prefers HTML over
    /// Markdown when both exist, and the HTML fragment carries no MathJax
    /// delimiters.
    fn _repr_markdown_(&self) -> String {
        self.inner.to_markdown()
    }
}

/// Build an `ExplainContext` from optional Python arguments.
pub fn make_context(
    sample_rate: Option<f64>,
    n_samples: Option<usize>,
    n_channels: Option<usize>,
    f0_range: Option<(f64, f64)>,
    n_fft: Option<usize>,
    n_mels: Option<usize>,
) -> ExplainContext {
    let mut ctx = ExplainContext::new();
    if let Some(fs) = sample_rate {
        ctx = ctx.with_sample_rate(fs);
    }
    if let Some(n) = n_samples {
        ctx = ctx.with_n_samples(n);
    }
    if let Some(n) = n_channels {
        ctx = ctx.with_n_channels(n);
    }
    if let Some((lo, hi)) = f0_range {
        ctx = ctx.with_f0_range(lo, hi);
    }
    if let Some(n) = n_fft {
        ctx = ctx.with_n_fft(n);
    }
    if let Some(n) = n_mels {
        ctx = ctx.with_n_mels(n);
    }
    ctx
}

/// Explain any supported configuration object with `ctx`.
pub fn explain_any(obj: &Bound<'_, PyAny>, ctx: &ExplainContext) -> PyResult<ExplainDoc> {
    if let Ok(p) = obj.cast::<PyStftParams>() {
        return Ok(p.borrow().inner.explain_with(ctx));
    }
    if let Ok(p) = obj.cast::<PySpectrogramParams>() {
        let params: SpectrogramParams = p.borrow().clone().into();
        return Ok(params.explain_with(ctx));
    }
    if let Ok(p) = obj.cast::<PyMelParams>() {
        let params: MelParams = (*p.borrow()).into();
        return Ok(params.explain_with(ctx));
    }
    if let Ok(p) = obj.cast::<PyMfccParams>() {
        let params: MfccParams = (*p.borrow()).into();
        return Ok(params.explain_with(ctx));
    }
    if let Ok(p) = obj.cast::<PySpecWindowType>() {
        return Ok(p.borrow().as_inner().explain_with(ctx));
    }
    if let Ok(p) = obj.cast::<PyTypesWindowType>() {
        let w: WindowType = p.borrow().inner.clone();
        return Ok(w.explain_with(ctx));
    }
    if let Ok(p) = obj.cast::<PyIirFilterDesign>() {
        return Ok(p.borrow().inner.explain_with(ctx));
    }
    if let Ok(p) = obj.cast::<PyEqBand>() {
        return Ok(p.borrow().inner.explain_with(ctx));
    }
    if let Ok(p) = obj.cast::<PyCompressorConfig>() {
        return Ok(p.borrow().inner.explain_with(ctx));
    }
    if let Ok(p) = obj.cast::<PyLimiterConfig>() {
        return Ok(p.borrow().inner.explain_with(ctx));
    }
    if let Ok(p) = obj.cast::<PyVadConfig>() {
        return Ok(p.borrow().inner.explain_with(ctx));
    }
    if let Ok(p) = obj.cast::<PyOnsetDetectionConfig>() {
        return Ok(p.borrow().inner.explain_with(ctx));
    }
    let type_name = obj
        .get_type()
        .qualname()
        .map_or_else(|_| "<unknown>".to_string(), |n| n.to_string());
    Err(PyTypeError::new_err(format!(
        "cannot explain an object of type '{type_name}'; supported types: {SUPPORTED}"
    )))
}

/// Explain a configuration object.
///
/// Dispatches on the type of ``obj`` and returns an :class:`ExplainDoc`
/// describing what the configuration will do, with numbers computed from its
/// settings and the optional context below. Missing context is not an error:
/// quantities that need it are left out and an ``info`` warning says so.
///
/// Args:
///
///     obj: The configuration to explain. One of ``StftParams``,
///         ``SpectrogramParams``, ``MelParams``, ``MfccParams``,
///         ``WindowType``, ``IirFilterDesign``, ``EqBand``,
///         ``CompressorConfig``, ``LimiterConfig``, ``VadConfig`` or
///         ``OnsetDetectionConfig``.
///     sample_rate (float | None): Sample rate in Hz of the signal the
///         configuration will be applied to.
///     n_samples (int | None): Signal length in samples per channel.
///     n_channels (int | None): Number of channels.
///     f0_range (tuple[float, float] | None): Expected fundamental-frequency
///         range ``(lowest, highest)`` in Hz; enables checks such as "window
///         shorter than two pitch periods".
///     n_fft (int | None): FFT size a ``MelParams`` will be combined with
///         (enables the empty-mel-band check).
///     n_mels (int | None): Number of mel bands an ``MfccParams`` will be
///         combined with (enables the ``n_mfcc > n_mels`` check).
///
/// Returns:
///
///     ExplainDoc: The explanation.
///
/// Raises:
///
///     TypeError: If ``obj`` is not one of the supported types.
///
/// Examples:
///
///     >>> import audio_samples as aus
///     >>> from audio_samples.explain import explain
///     >>> params = aus.StftParams(n_fft=256, hop_size=128, window=aus.WindowType.hanning)
///     >>> doc = explain(params, sample_rate=16000, f0_range=(80.0, 300.0))
///     >>> doc.quantity("Frequency resolution")
///     62.5
///     >>> doc.has_warning("caution")
///     True
#[pyfunction(name = "explain")]
#[pyo3(signature = (obj, *, sample_rate: "float | None" = None, n_samples: "int | None" = None, n_channels: "int | None" = None, f0_range: "tuple[float, float] | None" = None, n_fft: "int | None" = None, n_mels: "int | None" = None))]
#[allow(clippy::too_many_arguments)]
fn explain_fn(
    obj: &Bound<'_, PyAny>,
    sample_rate: Option<f64>,
    n_samples: Option<usize>,
    n_channels: Option<usize>,
    f0_range: Option<(f64, f64)>,
    n_fft: Option<usize>,
    n_mels: Option<usize>,
) -> PyResult<PyExplainDoc> {
    let ctx = make_context(sample_rate, n_samples, n_channels, f0_range, n_fft, n_mels);
    explain_any(obj, &ctx).map(Into::into)
}

/// Explain sampling at a given rate: sampling period and Nyquist frequency.
///
/// Args:
///
///     sample_rate (float): Sample rate in Hz.
///
/// Returns:
///
///     ExplainDoc: The explanation (a ``problem`` warning if the rate is not
///         finite and positive).
///
/// Examples:
///
///     >>> from audio_samples.explain import sampling
///     >>> sampling(16000.0).quantity("Nyquist frequency")
///     8000.0
#[pyfunction(name = "sampling")]
#[pyo3(signature = (sample_rate: "float"))]
fn topic_sampling(sample_rate: f64) -> PyExplainDoc {
    topics::sampling(sample_rate).into()
}

/// Explain where a tone of a given frequency lands after sampling (aliasing).
///
/// Args:
///
///     frequency (float): Frequency of the continuous-time tone in Hz.
///     sample_rate (float): Sample rate in Hz.
///
/// Returns:
///
///     ExplainDoc: The explanation, with the ``"Alias frequency"`` quantity.
///
/// Examples:
///
///     >>> from audio_samples.explain import aliasing
///     >>> aliasing(7000.0, 8000.0).quantity("Alias frequency")
///     1000.0
#[pyfunction(name = "aliasing")]
#[pyo3(signature = (frequency: "float", sample_rate: "float"))]
fn topic_aliasing(frequency: f64, sample_rate: f64) -> PyExplainDoc {
    topics::aliasing(frequency, sample_rate).into()
}

/// Explain uniform quantisation at a given bit depth (step size, SQNR).
///
/// Args:
///
///     bits (int): Bits per sample.
///
/// Returns:
///
///     ExplainDoc: The explanation.
///
/// Examples:
///
///     >>> from audio_samples.explain import quantisation
///     >>> round(quantisation(16).quantity("SQNR (full-scale sine)"), 2)
///     98.09
#[pyfunction(name = "quantisation")]
#[pyo3(signature = (bits: "int"))]
fn topic_quantisation(bits: u32) -> PyExplainDoc {
    topics::quantisation(bits).into()
}

/// Explain an N-point discrete Fourier transform at a given sample rate.
///
/// Args:
///
///     n_fft (int): Transform length in samples.
///     sample_rate (float): Sample rate in Hz.
///
/// Returns:
///
///     ExplainDoc: The explanation (bin spacing, number of bins, ...).
#[pyfunction(name = "dft")]
#[pyo3(signature = (n_fft: "int", sample_rate: "float"))]
fn topic_dft(n_fft: usize, sample_rate: f64) -> PyExplainDoc {
    topics::dft(n_fft, sample_rate).into()
}

/// Explain a window function of a given length, measured from its coefficients.
///
/// Args:
///
///     window_type (WindowType): The window (``audio_samples.WindowType``).
///     n (int): Window length in samples.
///     sample_rate (float): Sample rate in Hz.
///
/// Returns:
///
///     ExplainDoc: The explanation (ENBW, main-lobe width, side-lobe level, ...).
///
/// Raises:
///
///     TypeError: If ``window_type`` is not a ``WindowType``.
#[pyfunction(name = "window")]
#[pyo3(signature = (window_type: "WindowType", n: "int", sample_rate: "float"))]
fn topic_window(
    window_type: &Bound<'_, PyAny>,
    n: usize,
    sample_rate: f64,
) -> PyResult<PyExplainDoc> {
    let w: WindowType = if let Ok(p) = window_type.cast::<PySpecWindowType>() {
        p.borrow().as_inner().clone()
    } else if let Ok(p) = window_type.cast::<PyTypesWindowType>() {
        p.borrow().inner.clone()
    } else {
        return Err(PyTypeError::new_err("window_type must be a WindowType"));
    };
    Ok(topics::window(&w, n, sample_rate).into())
}

/// Explain YIN pitch estimation for a search range and frame length.
///
/// Args:
///
///     threshold (float): CMND threshold (typically 0.1 to 0.2).
///     min_frequency (float): Lowest frequency searched, in Hz.
///     max_frequency (float): Highest frequency searched, in Hz.
///     sample_rate (float): Sample rate in Hz.
///     frame_length (int | None): Analysis frame length in samples; when given,
///         a ``problem`` warning fires if it is too short for ``min_frequency``.
///
/// Returns:
///
///     ExplainDoc: The explanation (lag range, minimum frame length, ...).
#[pyfunction(name = "yin")]
#[pyo3(signature = (threshold: "float", min_frequency: "float", max_frequency: "float", sample_rate: "float", frame_length: "int | None" = None))]
fn topic_yin(
    threshold: f64,
    min_frequency: f64,
    max_frequency: f64,
    sample_rate: f64,
    frame_length: Option<usize>,
) -> PyExplainDoc {
    topics::yin(
        threshold,
        min_frequency,
        max_frequency,
        sample_rate,
        frame_length,
    )
    .into()
}

/// Explain autocorrelation pitch estimation for a search range.
///
/// Args:
///
///     min_frequency (float): Lowest frequency searched, in Hz.
///     max_frequency (float): Highest frequency searched, in Hz.
///     sample_rate (float): Sample rate in Hz.
///
/// Returns:
///
///     ExplainDoc: The explanation (lag range, lag resolution, ...).
#[pyfunction(name = "autocorrelation_pitch")]
#[pyo3(signature = (min_frequency: "float", max_frequency: "float", sample_rate: "float"))]
fn topic_autocorrelation_pitch(
    min_frequency: f64,
    max_frequency: f64,
    sample_rate: f64,
) -> PyExplainDoc {
    topics::autocorrelation_pitch(min_frequency, max_frequency, sample_rate).into()
}

/// Explain linear predictive coding (LPC) of a given order.
///
/// Args:
///
///     order (int): Prediction order (number of coefficients).
///     sample_rate (float): Sample rate in Hz.
///
/// Returns:
///
///     ExplainDoc: The explanation (formants resolvable, rule-of-thumb order, ...).
#[pyfunction(name = "lpc")]
#[pyo3(signature = (order: "int", sample_rate: "float"))]
fn topic_lpc(order: usize, sample_rate: f64) -> PyExplainDoc {
    topics::lpc(order, sample_rate).into()
}

/// Explain the real cepstrum for an N-point FFT.
///
/// Args:
///
///     n_fft (int): FFT length in samples.
///     sample_rate (float): Sample rate in Hz.
///
/// Returns:
///
///     ExplainDoc: The explanation (quefrency resolution and range, ...).
#[pyfunction(name = "cepstrum")]
#[pyo3(signature = (n_fft: "int", sample_rate: "float"))]
fn topic_cepstrum(n_fft: usize, sample_rate: f64) -> PyExplainDoc {
    topics::cepstrum(n_fft, sample_rate).into()
}

/// Explain sample-rate conversion between two rates.
///
/// Args:
///
///     from_rate (float): Source sample rate in Hz.
///     to_rate (float): Target sample rate in Hz.
///
/// Returns:
///
///     ExplainDoc: The explanation (ratio, new Nyquist, anti-alias warnings, ...).
#[pyfunction(name = "resampling")]
#[pyo3(signature = (from_rate: "float", to_rate: "float"))]
fn topic_resampling(from_rate: f64, to_rate: f64) -> PyExplainDoc {
    topics::resampling(from_rate, to_rate).into()
}

/// Explain an FIR filter given its coefficients.
///
/// Reports the number of taps, whether the filter is linear phase
/// (symmetric or antisymmetric coefficients) and its group delay.
///
/// Args:
///
///     coefficients (Sequence[float] | numpy.ndarray): Filter taps ``h[0..N-1]``.
///     sample_rate (float): Sample rate in Hz.
///
/// Returns:
///
///     ExplainDoc: The explanation.
#[pyfunction(name = "fir")]
#[pyo3(signature = (coefficients: "Sequence[float] | numpy.ndarray", sample_rate: "float"))]
fn topic_fir(coefficients: Vec<f64>, sample_rate: f64) -> PyExplainDoc {
    topics::fir(&coefficients, sample_rate).into()
}

/// Create the `audio_samples.explain` submodule and re-export `ExplainDoc`.
pub fn explain_module(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    let sub = PyModule::new(py, "explain")?;
    sub.setattr(
        "__doc__",
        "Explanations of configurations and concepts, computed from your settings.\n\n\
         ``explain(obj, ...)`` describes what a configuration (StftParams, MelParams,\n\
         IirFilterDesign, CompressorConfig, ...) will do; the topic functions\n\
         (sampling, aliasing, quantisation, dft, window, yin, autocorrelation_pitch,\n\
         lpc, cepstrum, resampling, fir) explain concepts that have no\n\
         configuration class. All return an ExplainDoc.",
    )?;
    sub.add_class::<PyExplainDoc>()?;
    sub.add_function(wrap_pyfunction!(explain_fn, &sub)?)?;
    sub.add_function(wrap_pyfunction!(topic_sampling, &sub)?)?;
    sub.add_function(wrap_pyfunction!(topic_aliasing, &sub)?)?;
    sub.add_function(wrap_pyfunction!(topic_quantisation, &sub)?)?;
    sub.add_function(wrap_pyfunction!(topic_dft, &sub)?)?;
    sub.add_function(wrap_pyfunction!(topic_window, &sub)?)?;
    sub.add_function(wrap_pyfunction!(topic_yin, &sub)?)?;
    sub.add_function(wrap_pyfunction!(topic_autocorrelation_pitch, &sub)?)?;
    sub.add_function(wrap_pyfunction!(topic_lpc, &sub)?)?;
    sub.add_function(wrap_pyfunction!(topic_cepstrum, &sub)?)?;
    sub.add_function(wrap_pyfunction!(topic_resampling, &sub)?)?;
    sub.add_function(wrap_pyfunction!(topic_fir, &sub)?)?;
    sub.setattr(
        "__all__",
        vec![
            "ExplainDoc",
            "explain",
            "sampling",
            "aliasing",
            "quantisation",
            "dft",
            "window",
            "yin",
            "autocorrelation_pitch",
            "lpc",
            "cepstrum",
            "resampling",
            "fir",
        ],
    )?;
    m.add_submodule(&sub)?;
    m.add("ExplainDoc", sub.getattr("ExplainDoc")?)?;
    Ok(())
}

// =============================================================================
// .explain() methods
// =============================================================================

#[pymethods]
impl PyAudioSamples {
    /// Explain a configuration in the context of this signal.
    ///
    /// Uses this signal's sample rate, length (samples per channel) and
    /// channel count as context, so quantities such as frame counts, durations
    /// and "window shorter than a pitch period" warnings are computed for it.
    ///
    /// Args:
    ///
    ///     obj: The configuration to explain (``StftParams``,
    ///         ``SpectrogramParams``, ``MelParams``, ``MfccParams``,
    ///         ``WindowType``, ``IirFilterDesign``, ``EqBand``,
    ///         ``CompressorConfig``, ``LimiterConfig``, ``VadConfig`` or
    ///         ``OnsetDetectionConfig``).
    ///     f0_range (tuple[float, float] | None): Expected fundamental-frequency
    ///         range ``(lowest, highest)`` in Hz.
    ///     n_fft (int | None): FFT size a ``MelParams`` will be combined with.
    ///     n_mels (int | None): Number of mel bands an ``MfccParams`` will be
    ///         combined with.
    ///
    /// Returns:
    ///
    ///     ExplainDoc: The explanation.
    ///
    /// Raises:
    ///
    ///     TypeError: If ``obj`` is not one of the supported types.
    ///
    /// Examples:
    ///
    ///     >>> import audio_samples as aus
    ///     >>> audio = aus.sine_wave(440.0, 1.0, sample_rate=16000)
    ///     >>> params = aus.StftParams(n_fft=512, hop_size=256, window=aus.WindowType.hanning)
    ///     >>> doc = audio.explain(params)
    ///     >>> doc.quantity("Frequency resolution")
    ///     31.25
    #[pyo3(name = "explain", signature = (obj, *, f0_range: "tuple[float, float] | None" = None, n_fft: "int | None" = None, n_mels: "int | None" = None))]
    fn explain_config(
        &self,
        py: Python<'_>,
        obj: &Bound<'_, PyAny>,
        f0_range: Option<(f64, f64)>,
        n_fft: Option<usize>,
        n_mels: Option<usize>,
    ) -> PyResult<PyExplainDoc> {
        let mut ctx = dispatch_with_view!(self, py, |audio| ExplainContext::for_audio(&audio));
        if let Some((lo, hi)) = f0_range {
            ctx = ctx.with_f0_range(lo, hi);
        }
        if let Some(n) = n_fft {
            ctx = ctx.with_n_fft(n);
        }
        if let Some(n) = n_mels {
            ctx = ctx.with_n_mels(n);
        }
        explain_any(obj, &ctx).map(Into::into)
    }
}

/// Adds `.explain(sample_rate=None, *, n_samples=None, n_channels=None,
/// f0_range=None)` to a config pyclass with an `inner: impl Explain`.
macro_rules! impl_config_explain {
    ($pytype:ty, $what:literal) => {
        #[pymethods]
        impl $pytype {
            #[doc = concat!("Explain what this ", $what, " will do.")]
            ///
            /// Args:
            ///
            ///     sample_rate (float | None): Sample rate in Hz of the signal it
            ///         will be applied to. Without it, quantities that need a
            ///         sample rate are omitted and an ``info`` warning says so.
            ///     n_samples (int | None): Signal length in samples per channel.
            ///     n_channels (int | None): Number of channels.
            ///     f0_range (tuple[float, float] | None): Expected
            ///         fundamental-frequency range ``(lowest, highest)`` in Hz.
            ///
            /// Returns:
            ///
            ///     ExplainDoc: The explanation (see ``audio_samples.explain``).
            #[pyo3(signature = (sample_rate: "float | None" = None, *, n_samples: "int | None" = None, n_channels: "int | None" = None, f0_range: "tuple[float, float] | None" = None))]
            fn explain(
                &self,
                sample_rate: Option<f64>,
                n_samples: Option<usize>,
                n_channels: Option<usize>,
                f0_range: Option<(f64, f64)>,
            ) -> PyExplainDoc {
                let ctx = make_context(sample_rate, n_samples, n_channels, f0_range, None, None);
                self.inner.explain_with(&ctx).into()
            }
        }
    };
}

impl_config_explain!(PyIirFilterDesign, "IIR filter design");
impl_config_explain!(PyEqBand, "EQ band");
impl_config_explain!(PyCompressorConfig, "compressor configuration");
impl_config_explain!(PyLimiterConfig, "limiter configuration");
impl_config_explain!(PyVadConfig, "voice-activity-detection configuration");
impl_config_explain!(PyOnsetDetectionConfig, "onset-detection configuration");
