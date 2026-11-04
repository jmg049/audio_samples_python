#![warn(clippy::all)]
#![warn(clippy::pedantic)]
#![warn(clippy::nursery)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::too_many_lines)]
#![allow(clippy::collapsible_if)]
#![allow(clippy::if_same_then_else)]
#![allow(clippy::unnecessary_cast)]
#![allow(clippy::identity_op)]
#![allow(clippy::needless_borrows_for_generic_args)]
#![warn(clippy::exhaustive_enums)]
#![warn(clippy::exhaustive_structs)]
#![warn(clippy::missing_inline_in_public_items)]
#![warn(clippy::missing_errors_doc)]
#![warn(clippy::doc_markdown)]
#![warn(clippy::iter_cloned_collect)]
#![warn(clippy::needless_pass_by_value)]
#![warn(clippy::indexing_slicing)]
#![warn(clippy::panic_in_result_fn)]
#![warn(clippy::undocumented_unsafe_blocks)]
#![warn(clippy::multiple_unsafe_ops_per_block)]

pub mod args;
pub mod effects;
pub mod explain;
pub mod features;
pub mod io;
pub mod io_streaming;
pub mod operations;
pub mod rhythm;
pub mod sequence;
pub mod spectral;
pub mod types;
pub mod utils;
use audio_samples::NdResult;
use audio_samples::traits::StandardSample;
use audio_samples::{AudioData, AudioSamples, I24};
use audio_samples::{AudioEditing, AudioTypeConversion, ConvertTo};
use audio_samples_io::AudioIOError;
use non_empty_slice::NonEmptyVec;
use numpy::{
    Element, IntoPyArray, PyArray1, PyArray2, PyArrayDescr, PyArrayDescrMethods, PyArrayMethods,
    PyUntypedArray, PyUntypedArrayMethods, ToPyArray,
    ndarray::{Array1, Array2},
};
use pyo3::IntoPyObjectExt;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::types::{PyAnyMethods, PyBool, PyDict, PyFloat, PyInt, PyList, PyTuple};
use std::num::{NonZeroU32, NonZeroUsize};
use std::ops::{Add, Mul, Sub};
use std::{any::TypeId, fmt::Display};

use pyo3::IntoPyObject;
use pyo3::prelude::*;

/// Local channel layout enum replacing the removed `audio_samples::ChannelLayout`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ChannelLayout {
    Interleaved,
    NonInterleaved,
}

pub(crate) fn nzu32_or_err(n: u32) -> PyResult<NonZeroU32> {
    NonZeroU32::new(n).ok_or_else(|| {
        PyValueError::new_err(format!(
            "Value must be a positive integer greater than zero, got {n}"
        ))
    })
}

use crate::types::PySampleType;

pub fn nzu_or_err(n: usize) -> PyResult<NonZeroUsize> {
    NonZeroUsize::new(n).ok_or_else(|| {
        PyValueError::new_err(format!(
            "Value must be a positive integer greater than zero, got {n}"
        ))
    })
}

/// Helper function to convert NdResult to PyArray.
///
/// NdResult can be either Mono (1D array) or MultiChannel (2D array).
/// This function converts both to PyArray, using PyAny to handle the dynamic type.
fn ndresult_to_numpy<'py, T>(py: Python<'py>, result: NdResult<T>) -> Bound<'py, PyAny>
where
    T: StandardSample + Element,
{
    match result {
        NdResult::Mono(arr) => arr.into_pyarray(py).into_any(),
        NdResult::MultiChannel(arr) => arr.into_pyarray(py).into_any(),
        _ => unreachable!(),
    }
}

macro_rules! impl_py_wrapper_core {
    ($pytype:ty, $rusttype:ty) => {
        impl From<$rusttype> for $pytype {
            #[inline]
            fn from(value: $rusttype) -> Self {
                Self { inner: value }
            }
        }

        impl From<$pytype> for $rusttype {
            #[inline]
            fn from(value: $pytype) -> Self {
                value.inner
            }
        }

        impl AsRef<$rusttype> for $pytype {
            #[inline]
            fn as_ref(&self) -> &$rusttype {
                &self.inner
            }
        }

        impl std::ops::Deref for $pytype {
            type Target = $rusttype;

            #[inline]
            fn deref(&self) -> &Self::Target {
                &self.inner
            }
        }
    };
}
pub(crate) use impl_py_wrapper_core;

macro_rules! impl_py_wrapper_fromstr {
    ($pytype:ty, $rusttype:ty) => {
        impl std::str::FromStr for $pytype {
            type Err = pyo3::PyErr;

            fn from_str(s: &str) -> Result<Self, Self::Err> {
                let inner = <$rusttype as std::str::FromStr>::from_str(s)
                    .map_err(|e| PyErr::new::<PyTypeError, _>(format!("Invalid string, {}", e)))?;

                Ok(Self { inner })
            }
        }
    };
}
pub(crate) use impl_py_wrapper_fromstr;

macro_rules! impl_py_repr {
    ($pytype:ty) => {
        #[pymethods]
        impl $pytype {}
    };
}
pub(crate) use impl_py_repr;

/// Add a `__repr__` that prints the wrapped value's `Debug` form, which lists
/// every field (for example `VadConfig { method: Energy, frame_size: 512, .. }`).
macro_rules! impl_py_debug_repr {
    ($pytype:ty) => {
        #[pymethods]
        impl $pytype {
            fn __repr__(&self) -> String {
                format!("{:?}", self.inner)
            }
        }
    };
}
pub(crate) use impl_py_debug_repr;

macro_rules! impl_py_default_static {
    ($pytype:ty) => {
        #[pymethods]
        impl $pytype {
            #[staticmethod]
            fn default() -> Self {
                <Self as Default>::default()
            }
        }
    };
}
pub(crate) use impl_py_default_static;

macro_rules! reexport {
    ($parent:expr, $child:expr, $( $name:literal ),+ $(,)?) => {
        $(
            $parent.add($name, $child.getattr($name)?)?;
        )+
    };
}
pub(crate) use reexport;

macro_rules! register_types {
    (
        $py:ident,
        $parent:ident,
        $submod:ident,
        [
            $(
                ($pytype:ty, $pyname:literal)
            ),+ $(,)?
        ]
    ) => {{
        $(
            $submod.add_class::<$pytype>()?;
        )+

        reexport!(
            $parent,
            $submod,
            $(
                $pyname
            ),+
        );
    }};
}
pub(crate) use register_types;

// =============================================================================
// DISPATCH MACROS
// =============================================================================
// These macros reduce boilerplate for the dtype-dispatch pattern where we need
// to match on PyAudioDataInner variants and call the same operation on each.

/// Dispatch to all dtype variants with a read-only view.
/// Usage: `dispatch_with_view!(self, py, |audio| audio.some_method())`
macro_rules! dispatch_with_view {
    ($samples:expr, $py:expr, |$audio:ident| $body:expr) => {{
        // Force a reference without moving.
        let __samples = &$samples;
        use crate::PyAudioDataInner;
        match __samples.inner() {
            PyAudioDataInner::U8(a) => a.with_view($py, |$audio| $body),
            PyAudioDataInner::I16(a) => a.with_view($py, |$audio| $body),
            PyAudioDataInner::I24(a) => a.with_view($py, |$audio| $body),
            PyAudioDataInner::I32(a) => a.with_view($py, |$audio| $body),
            PyAudioDataInner::F32(a) => a.with_view($py, |$audio| $body),
            PyAudioDataInner::F64(a) => a.with_view($py, |$audio| $body),
        }
    }};
}

/// [`dispatch_with_view!`] with the GIL released while the body runs
/// (`TypedAudioSamples::with_view_detached`). Only for pure-Rust bodies: the
/// body is a `Send` closure, so capturing a Python object does not compile.
macro_rules! dispatch_with_view_detached {
    ($samples:expr, $py:expr, |$audio:ident| $body:expr) => {{
        let __samples = &$samples;
        use crate::PyAudioDataInner;
        match __samples.inner() {
            PyAudioDataInner::U8(a) => a.with_view_detached($py, |$audio| $body),
            PyAudioDataInner::I16(a) => a.with_view_detached($py, |$audio| $body),
            PyAudioDataInner::I24(a) => a.with_view_detached($py, |$audio| $body),
            PyAudioDataInner::I32(a) => a.with_view_detached($py, |$audio| $body),
            PyAudioDataInner::F32(a) => a.with_view_detached($py, |$audio| $body),
            PyAudioDataInner::F64(a) => a.with_view_detached($py, |$audio| $body),
        }
    }};
}

/// [`dispatch_with_view_detached!`] for bodies whose result type depends on
/// the sample type (for example `Array3<T::Float>`): `$post` runs in each match
/// arm, with the GIL held, and turns that arm's result `$out` into one common
/// type (usually a Python object).
macro_rules! dispatch_typed_detached {
    ($samples:expr, $py:expr, |$audio:ident| $body:expr, |$out:ident| $post:expr) => {{
        use crate::PyAudioDataInner;
        let __samples = &$samples;
        match __samples.inner() {
            PyAudioDataInner::U8(a) => {
                let $out = a.with_view_detached($py, |$audio| $body);
                $post
            }
            PyAudioDataInner::I16(a) => {
                let $out = a.with_view_detached($py, |$audio| $body);
                $post
            }
            PyAudioDataInner::I24(a) => {
                let $out = a.with_view_detached($py, |$audio| $body);
                $post
            }
            PyAudioDataInner::I32(a) => {
                let $out = a.with_view_detached($py, |$audio| $body);
                $post
            }
            PyAudioDataInner::F32(a) => {
                let $out = a.with_view_detached($py, |$audio| $body);
                $post
            }
            PyAudioDataInner::F64(a) => {
                let $out = a.with_view_detached($py, |$audio| $body);
                $post
            }
        }
    }};
}

/// [`dispatch_typed_detached!`] with the GIL held while the body runs (for
/// bodies that call back into Python).
macro_rules! dispatch_typed {
    ($samples:expr, $py:expr, |$audio:ident| $body:expr, |$out:ident| $post:expr) => {{
        use crate::PyAudioDataInner;
        let __samples = &$samples;
        match __samples.inner() {
            PyAudioDataInner::U8(a) => {
                let $out = a.with_view($py, |$audio| $body);
                $post
            }
            PyAudioDataInner::I16(a) => {
                let $out = a.with_view($py, |$audio| $body);
                $post
            }
            PyAudioDataInner::I24(a) => {
                let $out = a.with_view($py, |$audio| $body);
                $post
            }
            PyAudioDataInner::I32(a) => {
                let $out = a.with_view($py, |$audio| $body);
                $post
            }
            PyAudioDataInner::F32(a) => {
                let $out = a.with_view($py, |$audio| $body);
                $post
            }
            PyAudioDataInner::F64(a) => {
                let $out = a.with_view($py, |$audio| $body);
                $post
            }
        }
    }};
}

/// Runs an analysis body returning `AudioSampleResult<Array<A, D>>` (with `A`
/// the sample type's float or complex float) on every dtype variant with the
/// GIL released, and converts the array to a NumPy array in each arm. Evaluates
/// to `PyResult<Bound<'py, PyAny>>`.
macro_rules! dispatch_array_detached {
    ($samples:expr, $py:expr, |$audio:ident| $body:expr) => {
        crate::dispatch_typed_detached!($samples, $py, |$audio| $body, |out| out
            .map_err(crate::audio_err_to_py)
            .map(|arr| numpy::IntoPyArray::into_pyarray(arr, $py).into_any()))
    };
}

/// Dispatch to all dtype variants with a mutable view.
/// Usage: `dispatch_with_view_mut!(self, py, |mut audio| audio.mutating_method())`
macro_rules! dispatch_with_view_mut {
    ($samples:expr, $py:expr, |mut $audio:ident| $body:expr) => {{
        let __samples = $samples;
        use crate::PyAudioDataInner;
        match __samples.inner_mut() {
            PyAudioDataInner::U8(a) => a.with_view_mut($py, |mut $audio| $body),
            PyAudioDataInner::I16(a) => a.with_view_mut($py, |mut $audio| $body),
            PyAudioDataInner::I24(a) => a.with_view_mut($py, |mut $audio| $body),
            PyAudioDataInner::I32(a) => a.with_view_mut($py, |mut $audio| $body),
            PyAudioDataInner::F32(a) => a.with_view_mut($py, |mut $audio| $body),
            PyAudioDataInner::F64(a) => a.with_view_mut($py, |mut $audio| $body),
        }
    }};
}

pub(crate) use dispatch_array_detached;
pub(crate) use dispatch_typed;
pub(crate) use dispatch_typed_detached;
pub(crate) use dispatch_with_view;
pub(crate) use dispatch_with_view_detached;
pub(crate) use dispatch_with_view_mut;

/// Python source of the `audio_samples.exceptions` module.
///
/// Every class derives from `AudioError` *and* from the built-in exception the
/// bindings raised before the hierarchy existed, so `except ValueError:` (etc.)
/// keeps working.
const EXCEPTIONS_PY: &std::ffi::CStr = cr#"
"""Exception hierarchy for audio_samples.

Every exception raised from the Rust core derives from :class:`AudioError`.
Each subclass also derives from the built-in exception that was raised before
this hierarchy existed (``ValueError``, ``TypeError``, ``RuntimeError`` or
``OSError``), so existing ``except`` clauses keep working.

When the Rust error carries a diagnostic, the exception has:

- ``code``: a stable machine-readable error code such as
  ``"audio_samples::parameter::out_of_range"`` (``None`` when absent);
- ``help``: a human-readable hint on how to fix the call (``None`` when absent).

On Python 3.11+ the help text is also attached with ``add_note`` so it is
printed under the traceback.
"""


class AudioError(Exception):
    """Base class of every error raised by audio_samples."""

    code = None
    help = None


class AudioParameterError(AudioError, ValueError):
    """An argument has an invalid value (out of range, inconsistent, ...)."""


class AudioLayoutError(AudioError, ValueError):
    """Array shape, channel count or memory layout is incompatible."""


class AudioConversionError(AudioError, TypeError):
    """A sample-type conversion or cast failed."""


class AudioRuntimeError(AudioError, RuntimeError):
    """Base class for failures that happen while processing valid input."""


class AudioProcessingError(AudioRuntimeError):
    """A DSP operation failed while processing the audio."""


class AudioFeatureError(AudioRuntimeError):
    """The operation needs a feature that was not compiled into this build."""


class AudioSpectrogramError(AudioRuntimeError):
    """A spectrogram, mel, MFCC or CQT computation failed."""


class AudioIOError(AudioError, OSError):
    """Reading or writing an audio file failed at the operating-system level."""


class AudioFileNotFoundError(AudioIOError, FileNotFoundError):
    """The audio file does not exist."""


class AudioFormatError(AudioError, ValueError):
    """The file format is unsupported or the file contents are corrupt."""


class AudioDecodeError(AudioError, TypeError):
    """An existing file could not be decoded as audio.

    Inherits ``TypeError`` because earlier versions raised ``TypeError`` here.
    """


def _make(cls_name, message, code=None, help=None, errno=None, filename=None):
    cls = globals()[cls_name]
    if errno is not None:
        exc = cls(errno, message, filename)
    else:
        exc = cls(message)
    exc.code = code
    exc.help = help
    if help and hasattr(exc, "add_note"):
        exc.add_note(f"help: {help}")
    return exc


__all__ = [
    "AudioError",
    "AudioParameterError",
    "AudioLayoutError",
    "AudioConversionError",
    "AudioRuntimeError",
    "AudioProcessingError",
    "AudioFeatureError",
    "AudioSpectrogramError",
    "AudioIOError",
    "AudioFileNotFoundError",
    "AudioFormatError",
    "AudioDecodeError",
]
"#;

/// The `audio_samples.exceptions` module, created once at import time.
static EXCEPTIONS_MODULE: pyo3::sync::PyOnceLock<Py<PyModule>> = pyo3::sync::PyOnceLock::new();

/// Create (once) and return the `audio_samples.exceptions` module.
fn exceptions_module(py: Python<'_>) -> PyResult<&Bound<'_, PyModule>> {
    EXCEPTIONS_MODULE
        .get_or_try_init(py, || {
            PyModule::from_code(
                py,
                EXCEPTIONS_PY,
                c"audio_samples/exceptions.py",
                c"audio_samples.exceptions",
            )
            .map(Bound::unbind)
        })
        .map(|m| m.bind(py))
}

/// Build an exception from the `audio_samples.exceptions` hierarchy.
///
/// `fallback` is used only if the hierarchy cannot be created (which would
/// indicate a broken interpreter state); it keeps the old built-in type.
pub(crate) fn make_audio_err(
    class_name: &str,
    message: String,
    code: Option<String>,
    help: Option<String>,
    fallback: fn(String) -> PyErr,
) -> PyErr {
    Python::attach(|py| {
        let built = exceptions_module(py).and_then(|m| {
            m.getattr("_make")?
                .call1((class_name, message.as_str(), code, help))
        });
        match built {
            Ok(exc) => PyErr::from_value(exc),
            Err(_) => fallback(message),
        }
    })
}

/// Raise `AudioParameterError` (a `ValueError`) naming both rates when a
/// two-signal operation receives inputs at different sample rates.
pub(crate) fn check_rates_match(op: &str, first: u32, other: u32) -> PyResult<()> {
    if first == other {
        return Ok(());
    }
    Err(make_audio_err(
        "AudioParameterError",
        format!(
            "{op} requires all inputs to share a sample rate, got {first} Hz and {other} Hz; \
resample one of them first, e.g. other.resample({first})"
        ),
        None,
        None,
        PyValueError::new_err,
    ))
}

/// Extract miette's `code` and `help` from a diagnostic.
fn diagnostic_parts(d: &dyn miette::Diagnostic) -> (Option<String>, Option<String>) {
    (
        d.code().map(|c| c.to_string()),
        d.help().map(|h| h.to_string()),
    )
}

/// Convert an audio_samples error to a PyErr with appropriate exception type.
///
/// The exception class comes from `audio_samples.exceptions` (all subclasses
/// of `AudioError`), and carries the miette diagnostic `code` and `help`.
fn audio_err_to_py(e: audio_samples::AudioSampleError) -> PyErr {
    use audio_samples::AudioSampleError;
    use pyo3::exceptions::{PyRuntimeError, PyValueError};

    let (code, help) = diagnostic_parts(&e);
    let (class_name, message, fallback): (&str, String, fn(String) -> PyErr) = match e {
        AudioSampleError::Parameter(_) => {
            ("AudioParameterError", e.to_string(), PyValueError::new_err)
        }
        AudioSampleError::Layout(_) => ("AudioLayoutError", e.to_string(), PyValueError::new_err),
        AudioSampleError::Conversion(_) => {
            ("AudioConversionError", e.to_string(), PyTypeError::new_err)
        }
        AudioSampleError::Processing(_) => (
            "AudioProcessingError",
            e.to_string(),
            PyRuntimeError::new_err,
        ),
        AudioSampleError::Feature(_) => (
            "AudioFeatureError",
            format!("{e} (enable the required cargo feature)"),
            PyRuntimeError::new_err,
        ),
        AudioSampleError::Spectrogram(_) => (
            "AudioSpectrogramError",
            e.to_string(),
            PyRuntimeError::new_err,
        ),
        // Catch-all for any future variants (serialization, etc.)
        #[allow(unreachable_patterns)]
        _ => ("AudioRuntimeError", e.to_string(), PyRuntimeError::new_err),
    };
    make_audio_err(class_name, message, code, help, fallback)
}

/// Convert an `std::io::Error` raised while accessing `path` to a Python
/// exception: `AudioFileNotFoundError` (a `FileNotFoundError`) when the file
/// does not exist, `AudioIOError` (an `OSError`) otherwise.
pub(crate) fn io_error_to_py(err: &std::io::Error, path: Option<&str>) -> PyErr {
    Python::attach(|py| {
        let (class_name, message) = if err.kind() == std::io::ErrorKind::NotFound {
            (
                "AudioFileNotFoundError",
                match path {
                    Some(p) => format!("No such audio file: '{p}'"),
                    None => format!("No such audio file: {err}"),
                },
            )
        } else {
            ("AudioIOError", err.to_string())
        };
        let errno = err.raw_os_error().unwrap_or(0);
        let built = exceptions_module(py).and_then(|m| {
            let kwargs = PyDict::new(py);
            kwargs.set_item("errno", errno)?;
            kwargs.set_item("filename", path)?;
            m.getattr("_make")?
                .call((class_name, message.as_str()), Some(&kwargs))
        });
        match built {
            Ok(exc) => PyErr::from_value(exc),
            Err(_) => pyo3::exceptions::PyIOError::new_err(message),
        }
    })
}

/// Convert an audio file error. `path` (when known) is attached to
/// file-not-found errors and named in the message.
pub(crate) fn audio_io_err_to_py_at(e: AudioIOError, path: Option<&str>) -> PyErr {
    use pyo3::exceptions::{PyIOError, PyRuntimeError, PyValueError};

    match e {
        AudioIOError::Io(io_err) => io_error_to_py(&io_err, path),
        AudioIOError::AudioSamples(as_err) => audio_err_to_py(as_err),
        AudioIOError::CorruptedData {
            description,
            details,
            position: _,
        } => make_audio_err(
            "AudioFormatError",
            format!("Corrupted data: {description} - {details}"),
            None,
            None,
            PyValueError::new_err,
        ),
        AudioIOError::WavError(wav_err) => make_audio_err(
            "AudioRuntimeError",
            format!("WAV error: {wav_err}"),
            None,
            None,
            PyRuntimeError::new_err,
        ),
        AudioIOError::SeekError(msg) | AudioIOError::EndOfStream(msg) => {
            make_audio_err("AudioIOError", msg, None, None, PyIOError::new_err)
        }
        AudioIOError::MissingFeature(msg) => make_audio_err(
            "AudioFeatureError",
            format!("Missing feature: {msg}"),
            None,
            None,
            PyRuntimeError::new_err,
        ),
        AudioIOError::UnsupportedFormat(msg) => {
            // The io crate already prefixes some messages; avoid doubling it.
            let msg = msg
                .strip_prefix("Unsupported format: ")
                .map_or_else(|| msg.clone(), str::to_string);
            make_audio_err(
                "AudioFormatError",
                format!("Unsupported format: {msg} (supported: WAV, FLAC)"),
                None,
                None,
                PyValueError::new_err,
            )
        }
        #[allow(unreachable_patterns)]
        _ => make_audio_err(
            "AudioRuntimeError",
            e.to_string(),
            None,
            None,
            PyRuntimeError::new_err,
        ),
    }
}

fn audio_io_err_to_py(e: AudioIOError) -> PyErr {
    audio_io_err_to_py_at(e, None)
}

/// Raise `FileNotFoundError` (as `AudioFileNotFoundError`) naming `path` if
/// it does not exist. Other metadata errors are left for the reader to report.
pub(crate) fn ensure_file_exists(path: &str) -> PyResult<()> {
    match std::fs::metadata(path) {
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Err(io_error_to_py(&e, Some(path))),
        _ => Ok(()),
    }
}

/// Raise the read-path error: missing files become `FileNotFoundError`,
/// anything else an `AudioDecodeError` (a `TypeError`, as before) with
/// `context` as the message prefix.
pub(crate) fn audio_read_err_to_py(e: AudioIOError, path: &str, context: &str) -> PyErr {
    match e {
        AudioIOError::Io(ref io_err) if io_err.kind() == std::io::ErrorKind::NotFound => {
            io_error_to_py(io_err, Some(path))
        }
        other => make_audio_err(
            "AudioDecodeError",
            format!("{context}: {other}"),
            None,
            None,
            PyTypeError::new_err,
        ),
    }
}

#[pymodule(name = "audio_samples")]
fn audio_samples_python(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Exception hierarchy (audio_samples.exceptions), re-exported at top level.
    let exceptions_mod = exceptions_module(py)?;
    m.add_submodule(exceptions_mod)?;
    for name in exceptions_mod
        .getattr("__all__")?
        .extract::<Vec<String>>()?
    {
        m.add(name.as_str(), exceptions_mod.getattr(name.as_str())?)?;
    }

    types::types(py, &m)?;
    m.add_class::<PyAudioSamples>()?;

    io::io(py, &m)?;
    utils::utils(py, &m)?;

    // Re-export spectrograms module
    let spectrograms_mod = PyModule::new(py, "spectrograms")?;
    spectrograms::python::register_module(py, &spectrograms_mod)?;
    // `ChromaParams(norm=...)` takes a ChromaNorm, which the spectrograms crate
    // does not register; expose it so the norm can actually be chosen.
    if !spectrograms_mod.hasattr("ChromaNorm")? {
        spectrograms_mod.add_class::<spectrograms::python::PyChromaNorm>()?;
    }
    m.add_submodule(&spectrograms_mod)?;

    // librosa-compatible array helpers
    spectral::register(py, m)?;
    sequence::register(py, m)?;
    features::register(py, m)?;
    rhythm::register(py, m)?;
    effects::register(py, m)?;

    // Explanations (audio_samples.explain); ExplainDoc is re-exported at top level.
    explain::explain_module(py, m)?;

    // Re-exports
    let utils_mod = m.getattr("utils")?;

    reexport!(
        m,
        utils_mod,
        "sine_wave",
        "cosine_wave",
        "sawtooth_wave",
        "triangle_wave",
        "square_wave",
        "white_noise",
        "pink_noise",
        "brown_noise",
        "impulse",
        "silence",
        "chirp",
        "square_wave_bandlimited",
        "sawtooth_wave_bandlimited",
        "triangle_wave_bandlimited",
        "exponential_chirp",
        "fm_signal",
        "am_signal",
        "compound_tone",
        "exponential_bursts",
        "stereo_sine_wave",
        "stereo_chirp",
        "stereo_silence",
        "mu_law_encode",
        "mu_law_decode"
    );

    let io_mod = m.getattr("io")?;
    reexport!(
        m,
        io_mod,
        "AudioInfo",
        "info",
        "read",
        "read_with_info",
        "read_and_resample",
        "peek_native_type",
        "save",
        "write_with_options",
        "write_with_metadata"
    );

    Ok(())
}

/// Represents homogeneous audio samples with associated metadata.
///
/// Primary container for audio data combining raw sample values with essential
/// metadata including sample rate, channel layout, and type information.
/// Supports both mono and multi-channel audio with unified interface.
///
/// The librosa-compatible analysis methods (``stft_matrix``, ``melspectrogram``,
/// ``chroma_stft``, ``cqt``, ``onset_strength``, ``yin``, ...) return float32 /
/// complex64 arrays for uint8, int16, int24 and float32 audio, and float64 /
/// complex128 arrays for int32 and float64 audio: the smallest float type that
/// holds every sample value exactly. Audio-returning methods keep the sample
/// type.
#[pyclass(name = "AudioSamples", module = "audio_samples")]
pub struct PyAudioSamples {
    inner: PyAudioDataInner,
}

impl PyAudioSamples {
    /// Creates a new mono audio samples container
    ///
    /// # Panics
    ///
    /// Panics if the sample type is not supported (i.e., not one of i16, I24, i32, f32, f64)
    pub fn new_mono<T: StandardSample + Element>(arr: Array1<T>, sample_rate: NonZeroU32) -> Self {
        let backing = PyAudioBacking::OwnedMono(arr);

        match TypeId::of::<T>() {
            id if id == TypeId::of::<u8>() => {
                let typed = TypedAudioSamples {
                    backing: unsafe {
                        std::mem::transmute::<PyAudioBacking<T>, PyAudioBacking<_>>(backing)
                    },
                    sample_rate,
                    layout: ChannelLayout::NonInterleaved,
                };
                Self {
                    inner: PyAudioDataInner::U8(typed),
                }
            }
            id if id == TypeId::of::<i16>() => {
                let typed = TypedAudioSamples {
                    backing: unsafe {
                        std::mem::transmute::<PyAudioBacking<T>, PyAudioBacking<_>>(backing)
                    },
                    sample_rate,
                    layout: ChannelLayout::NonInterleaved,
                };
                Self {
                    inner: PyAudioDataInner::I16(typed),
                }
            }
            id if id == TypeId::of::<I24>() => {
                let typed = TypedAudioSamples {
                    backing: unsafe {
                        std::mem::transmute::<PyAudioBacking<T>, PyAudioBacking<_>>(backing)
                    },
                    sample_rate,
                    layout: ChannelLayout::NonInterleaved,
                };
                Self {
                    inner: PyAudioDataInner::I24(typed),
                }
            }
            id if id == TypeId::of::<i32>() => {
                let typed = TypedAudioSamples {
                    backing: unsafe {
                        std::mem::transmute::<PyAudioBacking<T>, PyAudioBacking<_>>(backing)
                    },
                    sample_rate,
                    layout: ChannelLayout::NonInterleaved,
                };
                Self {
                    inner: PyAudioDataInner::I32(typed),
                }
            }
            id if id == TypeId::of::<f32>() => {
                let typed = TypedAudioSamples {
                    backing: unsafe {
                        std::mem::transmute::<PyAudioBacking<T>, PyAudioBacking<_>>(backing)
                    },
                    sample_rate,
                    layout: ChannelLayout::NonInterleaved,
                };
                Self {
                    inner: PyAudioDataInner::F32(typed),
                }
            }
            id if id == TypeId::of::<f64>() => {
                let typed = TypedAudioSamples {
                    backing: unsafe {
                        std::mem::transmute::<PyAudioBacking<T>, PyAudioBacking<_>>(backing)
                    },
                    sample_rate,
                    layout: ChannelLayout::NonInterleaved,
                };
                Self {
                    inner: PyAudioDataInner::F64(typed),
                }
            }
            _ => unreachable!("Unsupported audio sample type"),
        }
    }

    /// Creates a new multi-channel audio samples container
    ///
    /// # Panics
    ///
    /// Panics if the sample type is not supported (i.e., not one of i16, I24, i32, f32, f64)
    pub fn new_multi<T: StandardSample + Element>(arr: Array2<T>, sample_rate: NonZeroU32) -> Self {
        let backing = PyAudioBacking::OwnedMulti(arr);
        match TypeId::of::<T>() {
            id if id == TypeId::of::<u8>() => {
                let typed = TypedAudioSamples {
                    backing: unsafe {
                        std::mem::transmute::<PyAudioBacking<T>, PyAudioBacking<_>>(backing)
                    },
                    sample_rate,
                    layout: ChannelLayout::NonInterleaved,
                };
                Self {
                    inner: PyAudioDataInner::U8(typed),
                }
            }
            id if id == TypeId::of::<i16>() => {
                let typed = TypedAudioSamples {
                    backing: unsafe {
                        std::mem::transmute::<PyAudioBacking<T>, PyAudioBacking<_>>(backing)
                    },
                    sample_rate,
                    layout: ChannelLayout::NonInterleaved,
                };
                Self {
                    inner: PyAudioDataInner::I16(typed),
                }
            }
            id if id == TypeId::of::<I24>() => {
                let typed = TypedAudioSamples {
                    backing: unsafe {
                        std::mem::transmute::<PyAudioBacking<T>, PyAudioBacking<_>>(backing)
                    },
                    sample_rate,
                    layout: ChannelLayout::NonInterleaved,
                };
                Self {
                    inner: PyAudioDataInner::I24(typed),
                }
            }
            id if id == TypeId::of::<i32>() => {
                let typed = TypedAudioSamples {
                    backing: unsafe {
                        std::mem::transmute::<PyAudioBacking<T>, PyAudioBacking<_>>(backing)
                    },
                    sample_rate,
                    layout: ChannelLayout::NonInterleaved,
                };
                Self {
                    inner: PyAudioDataInner::I32(typed),
                }
            }
            id if id == TypeId::of::<f32>() => {
                let typed = TypedAudioSamples {
                    backing: unsafe {
                        std::mem::transmute::<PyAudioBacking<T>, PyAudioBacking<_>>(backing)
                    },
                    sample_rate,
                    layout: ChannelLayout::NonInterleaved,
                };
                Self {
                    inner: PyAudioDataInner::F32(typed),
                }
            }
            id if id == TypeId::of::<f64>() => {
                let typed = TypedAudioSamples {
                    backing: unsafe {
                        std::mem::transmute::<PyAudioBacking<T>, PyAudioBacking<_>>(backing)
                    },
                    sample_rate,
                    layout: ChannelLayout::NonInterleaved,
                };
                Self {
                    inner: PyAudioDataInner::F64(typed),
                }
            }
            _ => unreachable!("Unsupported audio sample type"),
        }
    }

    /// Creates a new mono audio samples container from a Python numpy array
    ///
    /// # Panics
    ///
    /// Panics if the sample type is not supported (i.e., not one of i16, I24, i32, f32, f64)
    pub fn new_mono_from_python<T: StandardSample + Element>(
        arr: Bound<'_, PyArray1<T>>,
        sample_rate: NonZeroU32,
    ) -> Self {
        let backing = PyAudioBacking::NumpyMono(arr.into());
        match TypeId::of::<T>() {
            id if id == TypeId::of::<u8>() => {
                let typed = TypedAudioSamples {
                    backing: unsafe {
                        std::mem::transmute::<PyAudioBacking<T>, PyAudioBacking<_>>(backing)
                    },
                    sample_rate,
                    layout: ChannelLayout::NonInterleaved,
                };
                Self {
                    inner: PyAudioDataInner::U8(typed),
                }
            }
            id if id == TypeId::of::<i16>() => {
                let typed = TypedAudioSamples {
                    backing: unsafe {
                        std::mem::transmute::<PyAudioBacking<T>, PyAudioBacking<_>>(backing)
                    },
                    sample_rate,
                    layout: ChannelLayout::NonInterleaved,
                };
                Self {
                    inner: PyAudioDataInner::I16(typed),
                }
            }
            id if id == TypeId::of::<I24>() => {
                let typed = TypedAudioSamples {
                    backing: unsafe {
                        std::mem::transmute::<PyAudioBacking<T>, PyAudioBacking<_>>(backing)
                    },
                    sample_rate,
                    layout: ChannelLayout::NonInterleaved,
                };
                Self {
                    inner: PyAudioDataInner::I24(typed),
                }
            }
            id if id == TypeId::of::<i32>() => {
                let typed = TypedAudioSamples {
                    backing: unsafe {
                        std::mem::transmute::<PyAudioBacking<T>, PyAudioBacking<_>>(backing)
                    },
                    sample_rate,
                    layout: ChannelLayout::NonInterleaved,
                };
                Self {
                    inner: PyAudioDataInner::I32(typed),
                }
            }
            id if id == TypeId::of::<f32>() => {
                let typed = TypedAudioSamples {
                    backing: unsafe {
                        std::mem::transmute::<PyAudioBacking<T>, PyAudioBacking<_>>(backing)
                    },
                    sample_rate,
                    layout: ChannelLayout::NonInterleaved,
                };
                Self {
                    inner: PyAudioDataInner::F32(typed),
                }
            }
            id if id == TypeId::of::<f64>() => {
                let typed = TypedAudioSamples {
                    backing: unsafe {
                        std::mem::transmute::<PyAudioBacking<T>, PyAudioBacking<_>>(backing)
                    },
                    sample_rate,
                    layout: ChannelLayout::NonInterleaved,
                };
                Self {
                    inner: PyAudioDataInner::F64(typed),
                }
            }
            _ => unreachable!("Unsupported audio sample type"),
        }
    }

    /// Creates a new PyAudioSamples instance from a numpy 2D array (multi-channel audio)
    ///
    /// # Panics
    ///
    /// Panics if the sample type is not supported (i.e., not one of i16, I24, i32, f32, f64)
    pub fn new_multi_from_python<T: StandardSample + Element>(
        arr: Bound<'_, PyArray2<T>>,
        sample_rate: NonZeroU32,
    ) -> Self {
        let backing = PyAudioBacking::NumpyMulti(arr.into());
        match TypeId::of::<T>() {
            id if id == TypeId::of::<u8>() => {
                let typed = TypedAudioSamples {
                    backing: unsafe {
                        std::mem::transmute::<PyAudioBacking<T>, PyAudioBacking<_>>(backing)
                    },
                    sample_rate,
                    layout: ChannelLayout::NonInterleaved,
                };
                Self {
                    inner: PyAudioDataInner::U8(typed),
                }
            }
            id if id == TypeId::of::<i16>() => {
                let typed = TypedAudioSamples {
                    backing: unsafe {
                        std::mem::transmute::<PyAudioBacking<T>, PyAudioBacking<_>>(backing)
                    },
                    sample_rate,
                    layout: ChannelLayout::NonInterleaved,
                };
                Self {
                    inner: PyAudioDataInner::I16(typed),
                }
            }
            id if id == TypeId::of::<I24>() => {
                let typed = TypedAudioSamples {
                    backing: unsafe {
                        std::mem::transmute::<PyAudioBacking<T>, PyAudioBacking<_>>(backing)
                    },
                    sample_rate,
                    layout: ChannelLayout::NonInterleaved,
                };
                Self {
                    inner: PyAudioDataInner::I24(typed),
                }
            }
            id if id == TypeId::of::<i32>() => {
                let typed = TypedAudioSamples {
                    backing: unsafe {
                        std::mem::transmute::<PyAudioBacking<T>, PyAudioBacking<_>>(backing)
                    },
                    sample_rate,
                    layout: ChannelLayout::NonInterleaved,
                };
                Self {
                    inner: PyAudioDataInner::I32(typed),
                }
            }
            id if id == TypeId::of::<f32>() => {
                let typed = TypedAudioSamples {
                    backing: unsafe {
                        std::mem::transmute::<PyAudioBacking<T>, PyAudioBacking<_>>(backing)
                    },
                    sample_rate,
                    layout: ChannelLayout::NonInterleaved,
                };
                Self {
                    inner: PyAudioDataInner::F32(typed),
                }
            }
            id if id == TypeId::of::<f64>() => {
                let typed = TypedAudioSamples {
                    backing: unsafe {
                        std::mem::transmute::<PyAudioBacking<T>, PyAudioBacking<_>>(backing)
                    },
                    sample_rate,
                    layout: ChannelLayout::NonInterleaved,
                };
                Self {
                    inner: PyAudioDataInner::F64(typed),
                }
            }
            _ => unreachable!("Unsupported audio sample type"),
        }
    }

    /// Create from Fortran-layout (interleaved) PyArray for multichannel audio
    ///
    /// This constructor is used for optimized read operations that create PyArrays
    /// with Fortran (column-major) layout, which matches WAV interleaved format.
    ///
    /// # Arguments
    ///
    /// * `arr` - PyArray2 with Fortran layout where data is interleaved
    /// * `sample_rate` - Sample rate in Hz
    ///
    /// # Panics
    ///
    /// Panics if the sample type is not supported (i.e., not one of i16, I24, i32, f32, f64)
    pub fn new_multi_from_python_interleaved<T: StandardSample + Element>(
        arr: Bound<'_, PyArray2<T>>,
        sample_rate: NonZeroU32,
    ) -> Self {
        let backing = PyAudioBacking::NumpyInterleaved(arr.into());
        match TypeId::of::<T>() {
            id if id == TypeId::of::<u8>() => {
                let typed = TypedAudioSamples {
                    backing: unsafe {
                        std::mem::transmute::<PyAudioBacking<T>, PyAudioBacking<_>>(backing)
                    },
                    sample_rate,
                    layout: ChannelLayout::Interleaved,
                };
                Self {
                    inner: PyAudioDataInner::U8(typed),
                }
            }
            id if id == TypeId::of::<i16>() => {
                let typed = TypedAudioSamples {
                    backing: unsafe {
                        std::mem::transmute::<PyAudioBacking<T>, PyAudioBacking<_>>(backing)
                    },
                    sample_rate,
                    layout: ChannelLayout::Interleaved,
                };
                Self {
                    inner: PyAudioDataInner::I16(typed),
                }
            }
            id if id == TypeId::of::<I24>() => {
                let typed = TypedAudioSamples {
                    backing: unsafe {
                        std::mem::transmute::<PyAudioBacking<T>, PyAudioBacking<_>>(backing)
                    },
                    sample_rate,
                    layout: ChannelLayout::Interleaved,
                };
                Self {
                    inner: PyAudioDataInner::I24(typed),
                }
            }
            id if id == TypeId::of::<i32>() => {
                let typed = TypedAudioSamples {
                    backing: unsafe {
                        std::mem::transmute::<PyAudioBacking<T>, PyAudioBacking<_>>(backing)
                    },
                    sample_rate,
                    layout: ChannelLayout::Interleaved,
                };
                Self {
                    inner: PyAudioDataInner::I32(typed),
                }
            }
            id if id == TypeId::of::<f32>() => {
                let typed = TypedAudioSamples {
                    backing: unsafe {
                        std::mem::transmute::<PyAudioBacking<T>, PyAudioBacking<_>>(backing)
                    },
                    sample_rate,
                    layout: ChannelLayout::Interleaved,
                };
                Self {
                    inner: PyAudioDataInner::F32(typed),
                }
            }
            id if id == TypeId::of::<f64>() => {
                let typed = TypedAudioSamples {
                    backing: unsafe {
                        std::mem::transmute::<PyAudioBacking<T>, PyAudioBacking<_>>(backing)
                    },
                    sample_rate,
                    layout: ChannelLayout::Interleaved,
                };
                Self {
                    inner: PyAudioDataInner::F64(typed),
                }
            }
            _ => unreachable!("Unsupported audio sample type"),
        }
    }

    /// Returns the TypeId of the audio sample type contained within PyAudioSamples
    pub fn type_of(&self) -> TypeId {
        match &self.inner {
            PyAudioDataInner::U8(_) => TypeId::of::<u8>(),
            PyAudioDataInner::I16(_) => TypeId::of::<i16>(),
            PyAudioDataInner::I24(_) => TypeId::of::<I24>(),
            PyAudioDataInner::I32(_) => TypeId::of::<i32>(),
            PyAudioDataInner::F32(_) => TypeId::of::<f32>(),
            PyAudioDataInner::F64(_) => TypeId::of::<f64>(),
        }
    }

    /// Checks if two PyAudioSamples instances contain the same audio sample type
    pub fn same_type_as(&self, other: &Self) -> bool {
        self.type_of() == other.type_of()
    }

    /// Helper method for __array_interface__ to set data pointer and strides
    fn set_array_interface_data<T: StandardSample + Element>(
        &self,
        py: Python<'_>,
        dict: &Bound<'_, PyDict>,
        data: &AudioData<'_, T>,
    ) -> PyResult<Py<PyDict>> {
        match data {
            AudioData::Mono(arr) => {
                let view = arr.as_view();
                let ptr = view.as_ptr() as usize;
                dict.set_item("data", (ptr, false))?;
                dict.set_item("strides", py.None())?;
            }
            AudioData::Multi(arr) => {
                let view = arr.as_view();
                let ptr = view.as_ptr() as usize;
                dict.set_item("data", (ptr, false))?;
                let strides: Vec<usize> = vec![
                    (view.strides()[0] * std::mem::size_of::<T>() as isize) as usize,
                    (view.strides()[1] * std::mem::size_of::<T>() as isize) as usize,
                ];
                dict.set_item("strides", PyTuple::new(py, strides)?)?;
            }
        }
        dict.set_item("version", 3)?;
        Ok(dict.clone().unbind())
    }

    /// Helper function to convert AudioSamples to PyAudioSamples efficiently using ownership transfer
    fn from_audio_samples<T: StandardSample + Element>(
        audio_samples: AudioSamples<'static, T>,
    ) -> Self {
        let sample_rate = audio_samples.sample_rate();
        match audio_samples.is_mono() {
            true => {
                let array = audio_samples.into_array1().expect("Safe since the None variant is only returned if data is not mono, which we have checked");
                Self::new_mono(array, sample_rate)
            }
            false => {
                let array = audio_samples.into_array2().expect("Safe since the None variant is only returned if data is not mono, which we have checked");
                Self::new_multi(array, sample_rate)
            }
        }
    }

    /// Access the inner data for crate-internal use (e.g., io module)
    pub(crate) const fn inner(&self) -> &PyAudioDataInner {
        &self.inner
    }

    pub(crate) const fn inner_mut(&mut self) -> &mut PyAudioDataInner {
        &mut self.inner
    }

    /// Safe clone method that requires a Python token.
    /// This ensures the GIL is held when cloning numpy-backed audio.
    pub(crate) fn clone_py(&self, py: Python<'_>) -> Self {
        PyAudioSamples {
            inner: self.inner.clone_py(py),
        }
    }

    /// `self <operation> other` for the operators, in-place ones included. The result has this
    /// audio's sample type and integer results saturate. `other` is one of:
    ///
    /// - audio, converted to this sample type if it differs;
    /// - a float scalar. For `+` and `-` on integer audio it is in the floating-point range and
    ///   must be within -1 to 1; `*` and `/` take it as a gain. On integer audio it must be finite;
    /// - an integer scalar. For `+` and `-` on integer audio it is a number of sample steps
    ///   relative to silence and must fit the sample type (a NumPy `uint8` scalar on `u8` audio
    ///   is a sample instead); `*` and `/` take it as the same gain as the float. On float audio
    ///   it is that number for every operator;
    /// - a NumPy array of raw sample values.
    ///
    /// Dividing integer audio by zero raises, also for a zero in an array or in other audio. A bool
    /// scalar raises `TypeError`: it is almost always a mistake, and neither reading of
    /// `audio + True` is obvious.
    fn arithmetic(
        &self,
        py: Python<'_>,
        other: &Bound<'_, PyAny>,
        operation: Arithmetic,
    ) -> PyResult<Self> {
        if other.is_instance_of::<Self>() {
            return match other.extract::<PyRef<Self>>() {
                Ok(other_audio) => self.arithmetic_with_audio(py, &other_audio, operation),
                // Only an in-place operator's own receiver is borrowed mutably here: `a *= a`.
                Err(_) => self.arithmetic_with_audio(py, self, operation),
            };
        }
        match classify_scalar(other)? {
            Some(Scalar::Integer(value)) => {
                return if self.is_float_samples()
                    || matches!(operation, Arithmetic::Mul | Arithmetic::Div)
                {
                    // Python's own conversion raises OverflowError for an int beyond f64.
                    self.arithmetic_with_scalar(py, value.extract::<f64>()?, operation)
                } else {
                    self.arithmetic_with_integer(py, &value, operation.name())
                };
            }
            Some(Scalar::Float(value)) => {
                return self.arithmetic_with_scalar(py, value, operation);
            }
            None => {}
        }
        if let Ok(numpy_array) = other.cast::<PyUntypedArray>() {
            return self.arithmetic_with_array(py, numpy_array.as_any(), operation);
        }
        Err(PyTypeError::new_err(format!(
            "AudioSamples can only be {} another AudioSamples, a numeric scalar, or numpy array",
            operation.phrase()
        )))
    }

    /// `+`, `-` or reflected `-` of integer audio with a number of sample steps.
    fn arithmetic_with_integer(
        &self,
        py: Python<'_>,
        value: &Bound<'_, PyAny>,
        operation: &str,
    ) -> PyResult<Self> {
        let result = py
            .import("audio_samples._arith")?
            .getattr("with_integer")?
            .call1((self.to_numpy(py)?, value, operation, self.dtype_py()))?;
        self.audio_from_samples(py, &result)
    }

    /// `other <operation> self`, for a scalar or a NumPy array on the left.
    fn arithmetic_reflected(
        &self,
        py: Python<'_>,
        other: &Bound<'_, PyAny>,
        operation: Arithmetic,
    ) -> PyResult<Self> {
        if classify_scalar(other)?.is_some() {
            return match operation {
                Arithmetic::Add => self.__radd__(py, other),
                Arithmetic::Mul => self.__rmul__(py, other),
                Arithmetic::Sub => self.__rsub__(py, other),
                Arithmetic::Div => self.__rtruediv__(py, 0.0),
            };
        }
        let name = match operation {
            Arithmetic::Add | Arithmetic::Mul => operation.name(),
            Arithmetic::Sub => "rsub",
            Arithmetic::Div => "rdiv",
        };
        let result = py
            .import("audio_samples._arith")?
            .getattr("apply")?
            .call1((self.to_numpy(py)?, other, name, self.dtype_py()))?;
        self.audio_from_samples(py, &result)
    }

    /// A new audio from `audio_samples._arith.<function>(samples, sample_type)`.
    fn unary(&self, py: Python<'_>, function: &str) -> PyResult<Self> {
        let result = py
            .import("audio_samples._arith")?
            .getattr(function)?
            .call1((self.to_numpy(py)?, self.dtype_py()))?;
        self.audio_from_samples(py, &result)
    }

    fn arithmetic_with_audio(
        &self,
        py: Python<'_>,
        other: &Self,
        operation: Arithmetic,
    ) -> PyResult<Self> {
        check_same_sample_rate(py, self, other, operation.verb())?;
        if self.shape(py) != other.shape(py) {
            return Err(PyValueError::new_err(format!(
                "AudioSamples shapes are incompatible: {:?} vs {:?}",
                self.shape(py),
                other.shape(py)
            )));
        }
        let converted = (self.dtype_py() != other.dtype_py())
            .then(|| other.to_format(py, self.sample_type_py()));
        let other = converted.as_ref().unwrap_or(other);
        if self.is_float_samples() && !matches!(operation, Arithmetic::Div) {
            return Ok(match operation {
                Arithmetic::Add => self + other,
                Arithmetic::Sub => self - other,
                _ => self * other,
            });
        }
        self.arithmetic_with_array(py, other.to_numpy(py)?.as_any(), operation)
    }

    fn arithmetic_with_scalar(
        &self,
        py: Python<'_>,
        scalar: f64,
        operation: Arithmetic,
    ) -> PyResult<Self> {
        if !self.is_float_samples() {
            if !scalar.is_finite() {
                return Err(PyValueError::new_err(format!(
                    "{scalar} cannot be used with {} audio: a scalar must be finite",
                    self.dtype_py()
                )));
            }
            if matches!(operation, Arithmetic::Add | Arithmetic::Sub) && scalar.abs() > 1.0 {
                return Err(PyValueError::new_err(format!(
                    "{scalar} is outside the sample range of {} audio: a float scalar is in the \
floating-point range, -1.0 to 1.0, where 1.0 is full scale",
                    self.dtype_py()
                )));
            }
        }
        match operation {
            Arithmetic::Mul => {
                let mut scaled = self.clone_py(py);
                scaled.scale(py, scalar);
                Ok(scaled)
            }
            Arithmetic::Div => {
                if scalar == 0.0 {
                    return Err(pyo3::exceptions::PyZeroDivisionError::new_err(
                        "division by zero",
                    ));
                }
                let mut scaled = self.clone_py(py);
                scaled.scale(py, 1.0 / scalar);
                Ok(scaled)
            }
            Arithmetic::Add | Arithmetic::Sub if self.is_float_samples() => Ok(match operation {
                Arithmetic::Add => self.clone_py(py) + scalar,
                _ => self.clone_py(py) - scalar,
            }),
            Arithmetic::Add | Arithmetic::Sub => {
                let samples = self.to_numpy(py)?;
                let in_sample_units = py
                    .import("audio_samples._arith")?
                    .getattr("scalar_to_samples")?
                    .call1((scalar, self.dtype_py(), samples.getattr("dtype")?))?;
                self.arithmetic_with_array(py, &in_sample_units, operation)
            }
        }
    }

    fn arithmetic_with_array(
        &self,
        py: Python<'_>,
        other: &Bound<'_, PyAny>,
        operation: Arithmetic,
    ) -> PyResult<Self> {
        let result = py
            .import("audio_samples._arith")?
            .getattr("apply")?
            .call1((self.to_numpy(py)?, other, operation.name(), self.dtype_py()))?;
        self.audio_from_samples(py, &result)
    }

    /// Audio with this audio's sample rate, layout and sample type, from an array of samples.
    fn audio_from_samples(&self, py: Python<'_>, samples: &Bound<'_, PyAny>) -> PyResult<Self> {
        let samples = samples.cast::<PyUntypedArray>()?.clone();
        let sample_rate = self.sample_rate(py);
        let audio = if self.channels(py) == 1 {
            Self::py_new_mono(py, samples, sample_rate)?
        } else {
            Self::py_new_multi(py, samples, sample_rate)?
        };
        // 24-bit samples travel through NumPy as int32.
        Ok(if self.dtype_py() == audio.dtype_py() {
            audio
        } else {
            audio.cast_as(py, self.sample_type_py())
        })
    }

    const fn is_float_samples(&self) -> bool {
        matches!(
            self.inner,
            PyAudioDataInner::F32(_) | PyAudioDataInner::F64(_)
        )
    }

    const fn sample_type_py(&self) -> PySampleType {
        match self.inner {
            PyAudioDataInner::U8(_) => PySampleType::U8,
            PyAudioDataInner::I16(_) => PySampleType::I16,
            PyAudioDataInner::I24(_) => PySampleType::I24,
            PyAudioDataInner::I32(_) => PySampleType::I32,
            PyAudioDataInner::F32(_) => PySampleType::F32,
            PyAudioDataInner::F64(_) => PySampleType::F64,
        }
    }

    // Internal dtype method for Rust-side use (not exposed to Python)
    pub fn dtype<'py>(&'py self, py: Python<'py>) -> Bound<'py, PyArrayDescr> {
        self.inner.dtype(py)
    }

    pub fn stack(py: Python<'_>, sources: Vec<Bound<'_, PyAudioSamples>>) -> PyResult<Self> {
        if sources.is_empty() {
            return Err(PyValueError::new_err("Cannot stack empty source list"));
        }

        // Extract PyAudioSamples from the list
        let sources_vec: Vec<PyRef<PyAudioSamples>> = sources
            .iter()
            .map(|item| {
                item.extract::<PyRef<PyAudioSamples>>()
                    .map_err(|e| PyTypeError::new_err(format!("Expected AudioSamples, got: {}", e)))
            })
            .collect::<PyResult<Vec<_>>>()?;

        // Validate all sources have the same dtype
        let first_dtype = sources_vec[0].dtype(py);
        for source in sources_vec.iter().skip(1) {
            if !source.dtype(py).is_equiv_to(&first_dtype) {
                return Err(PyTypeError::new_err(
                    "All sources must have the same dtype for stacking",
                ));
            }
        }
        let first_sr = sources_vec[0].sample_rate(py).get();
        for source in sources_vec.iter().skip(1) {
            check_rates_match("stack", first_sr, source.sample_rate(py).get())?;
        }

        // Dispatch based on the dtype
        match &sources_vec[0].inner {
            PyAudioDataInner::U8(_) => {
                let audio_sources: Vec<_> = sources_vec
                    .iter()
                    .map(|src| {
                        if let PyAudioDataInner::U8(typed) = &src.inner {
                            typed.borrowed(py)
                        } else {
                            unreachable!()
                        }
                    })
                    .collect();
                // safety: have checked for non-empty already
                let audio_sources = unsafe { NonEmptyVec::new_unchecked(audio_sources) };
                let stacked = <AudioSamples<u8> as AudioEditing>::stack(&audio_sources)
                    .map_err(audio_err_to_py)?;
                Ok(Self::from_audio_samples(stacked))
            }
            PyAudioDataInner::I16(_) => {
                let audio_sources: Vec<_> = sources_vec
                    .iter()
                    .map(|src| {
                        if let PyAudioDataInner::I16(typed) = &src.inner {
                            typed.borrowed(py)
                        } else {
                            unreachable!()
                        }
                    })
                    .collect();
                // safety: have checked for non-empty already
                let audio_sources = unsafe { NonEmptyVec::new_unchecked(audio_sources) };
                let stacked = <AudioSamples<i16> as AudioEditing>::stack(&audio_sources)
                    .map_err(audio_err_to_py)?;
                Ok(Self::from_audio_samples(stacked))
            }
            PyAudioDataInner::I24(_) => {
                let audio_sources: Vec<_> = sources_vec
                    .iter()
                    .map(|src| {
                        if let PyAudioDataInner::I24(typed) = &src.inner {
                            typed.borrowed(py)
                        } else {
                            unreachable!()
                        }
                    })
                    .collect();
                // safety: have checked for non-empty already
                let audio_sources = unsafe { NonEmptyVec::new_unchecked(audio_sources) };
                let stacked = <AudioSamples<I24> as AudioEditing>::stack(&audio_sources)
                    .map_err(audio_err_to_py)?;
                Ok(Self::from_audio_samples(stacked))
            }
            PyAudioDataInner::I32(_) => {
                let audio_sources: Vec<_> = sources_vec
                    .iter()
                    .map(|src| {
                        if let PyAudioDataInner::I32(typed) = &src.inner {
                            typed.borrowed(py)
                        } else {
                            unreachable!()
                        }
                    })
                    .collect();
                // safety: have checked for non-empty already
                let audio_sources = unsafe { NonEmptyVec::new_unchecked(audio_sources) };
                let stacked = <AudioSamples<i32> as AudioEditing>::stack(&audio_sources)
                    .map_err(audio_err_to_py)?;
                Ok(Self::from_audio_samples(stacked))
            }
            PyAudioDataInner::F32(_) => {
                let audio_sources: Vec<_> = sources_vec
                    .iter()
                    .map(|src| {
                        if let PyAudioDataInner::F32(typed) = &src.inner {
                            typed.borrowed(py)
                        } else {
                            unreachable!()
                        }
                    })
                    .collect();
                // safety: have checked for non-empty already
                let audio_sources = unsafe { NonEmptyVec::new_unchecked(audio_sources) };
                let stacked = <AudioSamples<f32> as AudioEditing>::stack(&audio_sources)
                    .map_err(audio_err_to_py)?;
                Ok(Self::from_audio_samples(stacked))
            }
            PyAudioDataInner::F64(_) => {
                let audio_sources: Vec<_> = sources_vec
                    .iter()
                    .map(|src| {
                        if let PyAudioDataInner::F64(typed) = &src.inner {
                            typed.borrowed(py)
                        } else {
                            unreachable!()
                        }
                    })
                    .collect();
                // safety: have checked for non-empty already
                let audio_sources = unsafe { NonEmptyVec::new_unchecked(audio_sources) };
                let stacked = <AudioSamples<f64> as AudioEditing>::stack(&audio_sources)
                    .map_err(audio_err_to_py)?;
                Ok(Self::from_audio_samples(stacked))
            }
        }
    }
}

/// Information about audio samples.
#[pyclass(name = "AudioSamplesInfo", module = "audio_samples")]
pub struct PyAudioSamplesInfo {
    #[pyo3(get)]
    pub sample_rate: NonZeroU32,
    #[pyo3(get)]
    pub channels: usize,
    #[pyo3(get)]
    pub frames: usize,
    #[pyo3(get)]
    pub duration_seconds: f64,
    #[pyo3(get)]
    pub layout: String,
}

#[pymethods]
impl PyAudioSamplesInfo {
    fn __repr__(&self) -> String {
        format!(
            "AudioSamplesInfo(sample_rate={}, channels={}, frames={}, duration_seconds={}, layout='{}')",
            self.sample_rate, self.channels, self.frames, self.duration_seconds, self.layout
        )
    }
}

#[pymethods]
impl PyAudioSamples {
    /// Returns the dtype as a string for Python property access
    #[getter(dtype)]
    const fn dtype_py(&self) -> &str {
        match &self.inner {
            PyAudioDataInner::U8(_) => "u8",
            PyAudioDataInner::I16(_) => "i16",
            PyAudioDataInner::I24(_) => "I24",
            PyAudioDataInner::I32(_) => "i32",
            PyAudioDataInner::F32(_) => "f32",
            PyAudioDataInner::F64(_) => "f64",
        }
    }

    /// Get information about the audio samples.
    ////
    /// Returns:
    ///     AudioSamplesInfo: An object containing sample rate, channels, frames, duration, and layout.
    ///
    /// Examples:
    ///     >>> import audio_samples as aus
    ///     >>> audio = aus.AudioSamples.from_array(np.random.randn(2, 44100).astype(np.float32), sample_rate=44100)
    ///     >>> info = audio.info()
    ///
    #[pyo3(signature = ())]
    fn info(&self, py: Python<'_>) -> PyResult<PyAudioSamplesInfo> {
        let layout = match self.inner() {
            PyAudioDataInner::U8(a) => a.layout,
            PyAudioDataInner::I16(a) => a.layout,
            PyAudioDataInner::I24(a) => a.layout,
            PyAudioDataInner::I32(a) => a.layout,
            PyAudioDataInner::F32(a) => a.layout,
            PyAudioDataInner::F64(a) => a.layout,
        };
        let layout_str = match layout {
            ChannelLayout::Interleaved => "interleaved",
            ChannelLayout::NonInterleaved => "non-interleaved",
        }
        .to_string();

        dispatch_with_view!(self, py, |audio| {
            let sample_rate = audio.sample_rate();
            let channels = audio.num_channels().get() as usize;
            let frames = audio.total_samples().get() / channels;
            let duration_seconds = audio.duration_seconds();

            Ok(PyAudioSamplesInfo {
                sample_rate,
                channels,
                frames,
                duration_seconds,
                layout: layout_str,
            })
        })
    }

    /// Create an AudioSamples instance from a NumPy array.
    ///
    /// Args:
    ///     arr (numpy.ndarray): Audio data as a 1D (mono) or 2D (multi-channel) array.
    ///         For multi-channel audio, shape should be (channels, samples).
    ///         Supported dtypes: int16, int32, float32, float64.
    ///     sample_rate (int): Sample rate in Hz (e.g., 44100, 48000).
    ///
    /// Returns:
    ///     AudioSamples: A new audio samples object that wraps ``arr`` without
    ///     copying it. In-place methods (``apply_*``, ``butterworth_*``,
    ///     ``filtfilt``, ...) therefore modify ``arr`` itself; pass
    ///     ``arr.copy()`` or call :meth:`copy` to keep the original. Read-only
    ///     arrays are copied before the first in-place modification.
    ///
    /// Raises:
    ///     TypeError: If array dtype is not supported or dimensions are invalid.
    ///
    /// Examples:
    ///     >>> import numpy as np
    ///     >>> import audio_samples as aus
    ///     >>> # Create mono audio
    ///     >>> mono = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 44100)).astype(np.float32)
    ///     >>> audio = aus.AudioSamples.from_array(mono, sample_rate=44100)
    ///     >>> # Create stereo audio
    ///     >>> stereo = np.random.randn(2, 44100).astype(np.float32)
    ///     >>> audio = aus.AudioSamples.from_array(stereo, sample_rate=44100)
    #[staticmethod]
    #[pyo3(name = "from_array", signature = (arr: "numpy.typing.ArrayLike", sample_rate: "int"))]
    fn from_array(arr: Bound<'_, PyAny>, sample_rate: NonZeroU32) -> PyResult<Self> {
        // Get the array as PyUntypedArray to inspect its properties
        let untyped_array: &Bound<'_, PyUntypedArray> = arr.cast()?;
        check_array_non_empty(untyped_array)?;
        warn_if_samples_first(untyped_array)?;

        // Get array info
        let dtype = untyped_array.dtype();
        let ndim = untyped_array.ndim();

        if dtype.is_equiv_to(&numpy::dtype::<i16>(arr.py())) {
            match ndim {
                1 => {
                    let typed_arr: Bound<'_, PyArray1<i16>> = arr.extract()?;
                    Ok(Self::new_mono_from_python::<i16>(typed_arr, sample_rate))
                }
                2 => {
                    let typed_arr: Bound<'_, PyArray2<i16>> = arr.extract()?;
                    Ok(Self::new_multi_from_python::<i16>(typed_arr, sample_rate))
                }
                _ => Err(PyTypeError::new_err(format!(
                    "Unsupported array dimensions: {ndim}. Expected 1D (mono) or 2D (multi-channel)"
                ))),
            }
        } else if dtype.is_equiv_to(&numpy::dtype::<i32>(arr.py())) {
            match ndim {
                1 => {
                    let typed_arr: Bound<'_, PyArray1<i32>> = arr.extract()?;
                    Ok(Self::new_mono_from_python::<i32>(typed_arr, sample_rate))
                }
                2 => {
                    let typed_arr: Bound<'_, PyArray2<i32>> = arr.extract()?;
                    Ok(Self::new_multi_from_python::<i32>(typed_arr, sample_rate))
                }
                _ => Err(PyTypeError::new_err(format!(
                    "Unsupported array dimensions: {ndim}. Expected 1D (mono) or 2D (multi-channel)"
                ))),
            }
        } else if dtype.is_equiv_to(&numpy::dtype::<f32>(arr.py())) {
            match ndim {
                1 => {
                    let typed_arr: Bound<'_, PyArray1<f32>> = arr.extract()?;
                    Ok(Self::new_mono_from_python::<f32>(typed_arr, sample_rate))
                }
                2 => {
                    let typed_arr: Bound<'_, PyArray2<f32>> = arr.extract()?;
                    Ok(Self::new_multi_from_python::<f32>(typed_arr, sample_rate))
                }
                _ => Err(PyTypeError::new_err(format!(
                    "Unsupported array dimensions: {ndim}. Expected 1D (mono) or 2D (multi-channel)"
                ))),
            }
        } else if dtype.is_equiv_to(&numpy::dtype::<f64>(arr.py())) {
            match ndim {
                1 => {
                    let typed_arr: Bound<'_, PyArray1<f64>> = arr.extract()?;
                    Ok(Self::new_mono_from_python::<f64>(typed_arr, sample_rate))
                }
                2 => {
                    let typed_arr: Bound<'_, PyArray2<f64>> = arr.extract()?;
                    Ok(Self::new_multi_from_python::<f64>(typed_arr, sample_rate))
                }
                _ => Err(PyTypeError::new_err(format!(
                    "Unsupported array dimensions: {ndim}. Expected 1D (mono) or 2D (multi-channel)"
                ))),
            }
        } else if dtype.is_equiv_to(&numpy::dtype::<I24>(arr.py())) {
            match ndim {
                1 => {
                    let typed_arr: Bound<'_, PyArray1<I24>> = arr.extract()?;
                    Ok(Self::new_mono_from_python::<I24>(typed_arr, sample_rate))
                }
                2 => {
                    let typed_arr: Bound<'_, PyArray2<I24>> = arr.extract()?;
                    Ok(Self::new_multi_from_python::<I24>(typed_arr, sample_rate))
                }
                _ => Err(PyTypeError::new_err(format!(
                    "Unsupported array dimensions: {ndim}. Expected 1D (mono) or 2D (multi-channel)"
                ))),
            }
        } else {
            Err(PyTypeError::new_err(format!(
                "Unsupported dtype with {ndim} dimensions. Supported types: int16, int32, float32, float64, I24"
            )))
        }
    }

    #[staticmethod]
    #[pyo3(name = "_new_mono_i16_from_np", signature = (arr: "numpy.typing.ArrayLike", sample_rate: "int"))]
    fn new_mono_i16_from_np(
        arr: Bound<'_, PyArray1<i16>>,
        sample_rate: NonZeroU32,
    ) -> PyResult<Self> {
        check_array_non_empty(arr.as_untyped())?;
        Ok(Self::new_mono_from_python::<i16>(arr, sample_rate))
    }

    #[staticmethod]
    #[pyo3(name = "_new_multi_i16_from_np", signature = (arr: "numpy.typing.ArrayLike", sample_rate: "int"))]
    fn new_multi_i16_from_np(
        arr: Bound<'_, PyArray2<i16>>,
        sample_rate: NonZeroU32,
    ) -> PyResult<Self> {
        check_array_non_empty(arr.as_untyped())?;
        Ok(Self::new_multi_from_python::<i16>(arr, sample_rate))
    }

    #[staticmethod]
    #[pyo3(name = "_new_mono_i24_from_np", signature = (arr: "numpy.typing.ArrayLike", sample_rate: "int"))]
    fn new_mono_i24_from_np(
        arr: Bound<'_, PyArray1<I24>>,
        sample_rate: NonZeroU32,
    ) -> PyResult<Self> {
        check_array_non_empty(arr.as_untyped())?;
        Ok(Self::new_mono_from_python::<I24>(arr, sample_rate))
    }

    #[staticmethod]
    #[pyo3(name = "_new_multi_i24_from_np", signature = (arr: "numpy.typing.ArrayLike", sample_rate: "int"))]
    fn new_multi_i24_from_np(
        arr: Bound<'_, PyArray2<I24>>,
        sample_rate: NonZeroU32,
    ) -> PyResult<Self> {
        check_array_non_empty(arr.as_untyped())?;
        Ok(Self::new_multi_from_python::<I24>(arr, sample_rate))
    }

    #[staticmethod]
    #[pyo3(name = "_new_mono_i32_from_np", signature = (arr: "numpy.typing.ArrayLike", sample_rate: "int"))]
    fn new_mono_i32_from_np(
        arr: Bound<'_, PyArray1<i32>>,
        sample_rate: NonZeroU32,
    ) -> PyResult<Self> {
        check_array_non_empty(arr.as_untyped())?;
        Ok(Self::new_mono_from_python::<i32>(arr, sample_rate))
    }

    #[staticmethod]
    #[pyo3(name = "_new_multi_i32_from_np", signature = (arr: "numpy.typing.ArrayLike", sample_rate: "int"))]
    fn new_multi_i32_from_np(
        arr: Bound<'_, PyArray2<i32>>,
        sample_rate: NonZeroU32,
    ) -> PyResult<Self> {
        check_array_non_empty(arr.as_untyped())?;
        Ok(Self::new_multi_from_python::<i32>(arr, sample_rate))
    }

    #[staticmethod]
    #[pyo3(name = "_new_mono_f32_from_np", signature = (arr: "numpy.typing.ArrayLike", sample_rate: "int"))]
    fn new_mono_f32_from_np(
        arr: Bound<'_, PyArray1<f32>>,
        sample_rate: NonZeroU32,
    ) -> PyResult<Self> {
        check_array_non_empty(arr.as_untyped())?;
        Ok(Self::new_mono_from_python::<f32>(arr, sample_rate))
    }

    #[staticmethod]
    #[pyo3(name = "_new_multi_f32_from_np", signature = (arr: "numpy.typing.ArrayLike", sample_rate: "int"))]
    fn new_multi_f32_from_np(
        arr: Bound<'_, PyArray2<f32>>,
        sample_rate: NonZeroU32,
    ) -> PyResult<Self> {
        check_array_non_empty(arr.as_untyped())?;
        Ok(Self::new_multi_from_python::<f32>(arr, sample_rate))
    }
    #[staticmethod]
    #[pyo3(name = "_new_mono_f64_from_np", signature = (arr: "numpy.typing.ArrayLike", sample_rate: "int"))]
    fn new_mono_f64_from_np(
        arr: Bound<'_, PyArray1<f64>>,
        sample_rate: NonZeroU32,
    ) -> PyResult<Self> {
        check_array_non_empty(arr.as_untyped())?;
        Ok(Self::new_mono_from_python::<f64>(arr, sample_rate))
    }
    #[staticmethod]
    #[pyo3(name = "_new_multi_f64_from_np", signature = (arr: "numpy.typing.ArrayLike", sample_rate: "int"))]
    fn new_multi_f64_from_np(
        arr: Bound<'_, PyArray2<f64>>,
        sample_rate: NonZeroU32,
    ) -> PyResult<Self> {
        check_array_non_empty(arr.as_untyped())?;
        Ok(Self::new_multi_from_python::<f64>(arr, sample_rate))
    }

    /// Create mono audio from a 1-D NumPy array.
    ///
    /// The array's dtype becomes the sample type; no value conversion is
    /// performed (float samples are expected in ``[-1, 1]``, integer samples
    /// span the full integer range). Supported dtypes: uint8, int16, int32,
    /// float32, float64.
    ///
    /// The returned object wraps ``arr`` without copying it (zero-copy). Methods
    /// that work in place (``apply_compressor``, ``apply_limiter``,
    /// ``apply_iir_filter``, ``filtfilt``, ``reverse_in_place``, ...) therefore
    /// modify ``arr`` itself; pass ``arr.copy()`` to keep the original.
    /// Methods that return a new buffer, or that replace the buffer wholesale
    /// (e.g. ``scale``), leave ``arr`` unchanged. Read-only arrays are copied
    /// before the first in-place modification.
    ///
    /// Args:
    ///     arr (numpy.ndarray): 1-D array of samples (non-empty).
    ///     sample_rate (int): Sample rate in Hz.
    ///
    /// Returns:
    ///     AudioSamples: Mono audio viewing ``arr``.
    ///
    /// Raises:
    ///     TypeError: If ``arr`` is not 1-D or its dtype is unsupported.
    ///     ValueError: If ``arr`` is empty.
    ///
    /// Examples:
    ///     >>> import numpy as np, audio_samples as aus
    ///     >>> a = np.zeros(4)
    ///     >>> x = aus.AudioSamples.new_mono(a, sample_rate=8000)
    ///     >>> x.num_channels, x.sample_rate
    ///     (1, 8000)
    #[staticmethod]
    #[pyo3(name = "new_mono", signature = (arr: "numpy.typing.ArrayLike", sample_rate: "int"))]
    fn py_new_mono(
        py: Python<'_>,
        arr: Bound<'_, PyUntypedArray>,
        sample_rate: NonZeroU32,
    ) -> PyResult<Self> {
        check_array_non_empty(&arr)?;
        if arr.ndim() != 1 {
            return Err(PyTypeError::new_err(format!(
                "new_mono expects a 1-D array, got shape {:?}; use new_multi for \
(channels, samples) data",
                arr.shape()
            )));
        }
        let dtype = arr.dtype();
        if dtype.is_equiv_to(&numpy::dtype::<u8>(py)) {
            return Ok(Self::new_mono_from_python::<u8>(
                arr.cast::<PyArray1<u8>>()?.clone(),
                sample_rate,
            ));
        } else if dtype.is_equiv_to(&numpy::dtype::<i16>(py)) {
            return Ok(Self::new_mono_from_python::<i16>(
                arr.cast::<PyArray1<i16>>()?.clone(),
                sample_rate,
            ));
        } else if dtype.is_equiv_to(&numpy::dtype::<i32>(py)) {
            return Ok(Self::new_mono_from_python::<i32>(
                arr.cast::<PyArray1<i32>>()?.clone(),
                sample_rate,
            ));
        } else if dtype.is_equiv_to(&numpy::dtype::<f32>(py)) {
            return Ok(Self::new_mono_from_python::<f32>(
                arr.cast::<PyArray1<f32>>()?.clone(),
                sample_rate,
            ));
        } else if dtype.is_equiv_to(&numpy::dtype::<f64>(py)) {
            return Ok(Self::new_mono_from_python::<f64>(
                arr.cast::<PyArray1<f64>>()?.clone(),
                sample_rate,
            ));
        }
        Err(PyErr::new::<PyTypeError, _>(
            "Unsupported audio sample type for mono audio",
        ))
    }

    /// Create multi-channel audio from a 2-D ``(channels, samples)`` NumPy array.
    ///
    /// Same conventions as :meth:`new_mono`: the dtype becomes the sample type
    /// and no value conversion is performed. A ``(samples, channels)``-shaped
    /// array triggers a warning.
    ///
    /// The returned object wraps ``arr`` without copying it (zero-copy). Methods
    /// that work in place (``apply_compressor``, ``apply_limiter``,
    /// ``apply_iir_filter``, ``filtfilt``, ``reverse_in_place``, ...) therefore
    /// modify ``arr`` itself; pass ``arr.copy()`` to keep the original.
    /// Methods that return a new buffer, or that replace the buffer wholesale
    /// (e.g. ``scale``), leave ``arr`` unchanged. Read-only arrays are copied
    /// before the first in-place modification.
    ///
    /// Args:
    ///     arr (numpy.ndarray): 2-D array shaped ``(channels, samples)``.
    ///     sample_rate (int): Sample rate in Hz.
    ///
    /// Returns:
    ///     AudioSamples: Multi-channel audio viewing ``arr``.
    ///
    /// Raises:
    ///     TypeError: If ``arr`` is not 2-D or its dtype is unsupported.
    ///     ValueError: If ``arr`` is empty.
    ///
    /// Examples:
    ///     >>> import numpy as np, audio_samples as aus
    ///     >>> x = aus.AudioSamples.new_multi(np.zeros((2, 4), dtype=np.float32), sample_rate=8000)
    ///     >>> x.num_channels
    ///     2
    #[staticmethod]
    #[pyo3(name = "new_multi", signature = (arr: "numpy.typing.ArrayLike", sample_rate: "int"))]
    fn py_new_multi(
        py: Python<'_>,
        arr: Bound<'_, PyUntypedArray>,
        sample_rate: NonZeroU32,
    ) -> PyResult<Self> {
        check_array_non_empty(&arr)?;
        if arr.ndim() != 2 {
            return Err(PyTypeError::new_err(format!(
                "new_multi expects a 2-D (channels, samples) array, got shape {:?}; \
use new_mono for 1-D data",
                arr.shape()
            )));
        }
        warn_if_samples_first(&arr)?;
        Self::new_multi_untyped(py, &arr, sample_rate)
    }
}

impl PyAudioSamples {
    /// Wrap a 2-D NumPy array without the `(samples, channels)` layout warning.
    fn new_multi_untyped(
        py: Python<'_>,
        arr: &Bound<'_, PyUntypedArray>,
        sample_rate: NonZeroU32,
    ) -> PyResult<Self> {
        let dtype = arr.dtype();
        if dtype.is_equiv_to(&numpy::dtype::<u8>(py)) {
            return Ok(Self::new_multi_from_python::<u8>(
                arr.cast::<PyArray2<u8>>()?.clone(),
                sample_rate,
            ));
        } else if dtype.is_equiv_to(&numpy::dtype::<i16>(py)) {
            return Ok(Self::new_multi_from_python::<i16>(
                arr.cast::<PyArray2<i16>>()?.clone(),
                sample_rate,
            ));
        } else if dtype.is_equiv_to(&numpy::dtype::<i32>(py)) {
            return Ok(Self::new_multi_from_python::<i32>(
                arr.cast::<PyArray2<i32>>()?.clone(),
                sample_rate,
            ));
        } else if dtype.is_equiv_to(&numpy::dtype::<f32>(py)) {
            return Ok(Self::new_multi_from_python::<f32>(
                arr.cast::<PyArray2<f32>>()?.clone(),
                sample_rate,
            ));
        } else if dtype.is_equiv_to(&numpy::dtype::<f64>(py)) {
            return Ok(Self::new_multi_from_python::<f64>(
                arr.cast::<PyArray2<f64>>()?.clone(),
                sample_rate,
            ));
        }
        Err(PyErr::new::<PyTypeError, _>(
            "Unsupported audio sample type for multi-channel audio",
        ))
    }

    /// Wrap a 1-D or 2-D NumPy array (used to re-wrap ufunc results).
    fn from_untyped_array(
        py: Python<'_>,
        arr: &Bound<'_, PyUntypedArray>,
        sample_rate: NonZeroU32,
    ) -> PyResult<Self> {
        match arr.ndim() {
            1 => Self::py_new_mono(py, arr.clone(), sample_rate),
            2 => {
                check_array_non_empty(arr)?;
                Self::new_multi_untyped(py, arr, sample_rate)
            }
            n => Err(PyTypeError::new_err(format!(
                "expected a 1-D or 2-D array, got {n} dimensions"
            ))),
        }
    }
}

#[pymethods]
impl PyAudioSamples {
    /// Sample rate in Hz.
    ///
    /// Returns:
    ///     int: Sample rate (e.g., 44100, 48000, 96000).
    ///
    /// Examples:
    ///     >>> audio.sample_rate
    ///     44100
    #[getter]
    fn sample_rate(&self, py: Python<'_>) -> NonZeroU32 {
        dispatch_with_view!(self, py, |audio| audio.sample_rate())
    }

    /// Number of audio channels.
    ///
    /// Returns:
    ///     int: Number of channels (1 for mono, 2 for stereo, etc.).
    ///
    /// Examples:
    ///     >>> mono_audio.num_channels
    ///     1
    ///     >>> stereo_audio.num_channels
    ///     2
    #[getter]
    fn num_channels(&self, py: Python<'_>) -> usize {
        dispatch_with_view!(self, py, |audio| audio.num_channels().get() as usize)
    }

    /// Alias for num_channels.
    #[getter(channels)]
    fn channels(&self, py: Python<'_>) -> usize {
        self.num_channels(py)
    }

    /// Total number of samples across all channels (same as ``len(audio)``).
    #[getter]
    fn len(&self, py: Python<'_>) -> usize {
        dispatch_with_view!(self, py, |audio| audio.len().get())
    }

    /// Total number of samples across all channels (NumPy-style alias of ``len``).
    #[getter(size)]
    fn size(&self, py: Python<'_>) -> usize {
        self.len(py)
    }

    /// Number of samples per channel.
    ///
    /// For multi-channel audio, this is the length of each channel.
    /// For mono audio, this equals the total number of samples.
    ///
    /// Returns:
    ///     int: Number of samples in each channel.
    ///
    /// Examples:
    ///     >>> stereo = aus.AudioSamples.from_array(np.zeros((2, 44100)), 44100)
    ///     >>> stereo.samples_per_channel()
    ///     44100
    #[pyo3(signature = ())]
    fn samples_per_channel(&self, py: Python<'_>) -> usize {
        dispatch_with_view!(self, py, |audio| audio.samples_per_channel().get())
    }

    /// Total number of samples across all channels.
    ///
    /// Returns:
    ///     int: Total samples = num_channels x samples_per_channel.
    ///
    /// Examples:
    ///     >>> stereo.total_samples  # 2 channels x 44100 samples
    ///     88200
    #[getter]
    fn total_samples(&self, py: Python<'_>) -> usize {
        dispatch_with_view!(self, py, |audio| audio.total_samples().get())
    }

    /// Shape of the audio data.
    ///
    /// Returns:
    ///     list[int]: Shape as [samples] for mono or [channels, samples] for multi-channel.
    ///
    /// Examples:
    ///     >>> mono.shape
    ///     [44100]
    ///     >>> stereo.shape
    ///     [2, 44100]
    #[getter]
    fn shape(&self, py: Python<'_>) -> Vec<usize> {
        dispatch_with_view!(self, py, |audio| audio.shape().to_vec())
    }

    #[pyo3(signature = ())]
    fn is_mono(&self, py: Python<'_>) -> bool {
        dispatch_with_view!(self, py, |audio| audio.is_mono())
    }

    #[pyo3(signature = ())]
    fn is_multi_channel(&self, py: Python<'_>) -> bool {
        dispatch_with_view!(self, py, |audio| audio.is_multi_channel())
    }

    /// Duration of the audio in seconds.
    ///
    /// Calculated as samples_per_channel / sample_rate.
    ///
    /// Returns:
    ///     float: Duration in seconds.
    ///
    /// Examples:
    ///     >>> audio = aus.AudioSamples.from_array(np.zeros(44100), 44100)
    ///     >>> audio.duration_seconds
    ///     1.0
    ///     >>> audio_3s = aus.AudioSamples.from_array(np.zeros(132300), 44100)
    ///     >>> audio_3s.duration_seconds
    ///     3.0
    #[getter]
    fn duration_seconds(&self, py: Python<'_>) -> f64 {
        dispatch_with_view!(self, py, |audio| audio.duration_seconds())
    }

    /// Duration of the audio in milliseconds (``duration_seconds * 1000``).
    #[getter]
    fn duration_milliseconds(&self, py: Python<'_>) -> f64 {
        self.duration_seconds(py) * 1000.0
    }

    /// Number of array dimensions: 1 for mono, 2 for multi-channel.
    #[getter(ndim)]
    fn ndim(&self, py: Python<'_>) -> usize {
        if self.is_mono(py) { 1 } else { 2 }
    }

    #[pyo3(signature = ())]
    fn copy(&self, py: Python<'_>) -> Self {
        self.clone_py(py)
    }

    /// Returns the total number of samples for `len(audio)`.
    ///
    /// For mono audio, returns samples_per_channel.
    /// For multi-channel audio, returns total_samples (channels x samples_per_channel).
    ///
    /// Examples:
    ///     >>> mono = aus.AudioSamples.from_array(np.zeros(44100), 44100)
    ///     >>> len(mono)
    ///     44100
    ///     >>> stereo = aus.AudioSamples.from_array(np.zeros((2, 44100)), 44100)
    ///     >>> len(stereo)
    ///     88200
    /// `AudioSamples[numpy.float32]` names audio of that sample type in annotations; the type
    /// stubs use it to tie the precision of analysis results to the sample type.
    #[classmethod]
    fn __class_getitem__<'py>(
        cls: &Bound<'py, pyo3::types::PyType>,
        item: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        cls.py()
            .import("types")?
            .getattr("GenericAlias")?
            .call1((cls, item))
    }

    fn __len__(&self, py: Python<'_>) -> usize {
        self.len(py)
    }

    fn __getitem__<'py>(
        &self,
        py: Python<'py>,
        index: Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        use pyo3::IntoPyObject;
        use pyo3::exceptions::PyIndexError;
        if index.is_none() {
            let c = self.clone_py(py);
            let s = Bound::new(py, c)?;
            return Ok(s.into_any());
        }

        // Parse the index into (channel_opt, sample_idx) form
        fn parse_index(
            index: &Bound<'_, PyAny>,
            is_mono: bool,
            num_channels: usize,
            num_samples: usize,
        ) -> PyResult<(Option<usize>, usize)> {
            if index.is_instance_of::<PyInt>() {
                let idx: usize = index.extract()?;
                if !is_mono {
                    return Err(PyIndexError::new_err(
                        "Cannot use integer index on multi-channel audio. Use (channel, sample) indexing instead.",
                    ));
                }
                if idx >= num_samples {
                    return Err(PyIndexError::new_err(format!(
                        "Sample index {idx} out of bounds for audio with {num_samples} samples"
                    )));
                }
                Ok((None, idx))
            } else if index.is_instance_of::<PyTuple>() {
                let (chan, samp): (usize, usize) = index.extract()?;
                if chan >= num_channels {
                    return Err(PyIndexError::new_err(format!(
                        "Channel index {chan} out of bounds for audio with {num_channels} channels"
                    )));
                }
                if samp >= num_samples {
                    return Err(PyIndexError::new_err(format!(
                        "Sample index {samp} out of bounds for audio with {num_samples} samples per channel"
                    )));
                }
                Ok((Some(chan), samp))
            } else if index.is_instance_of::<PyList>() {
                let list: Vec<usize> = index.extract()?;
                if list.len() != 2 {
                    return Err(PyTypeError::new_err(
                        "List index must have exactly two elements [channel, sample]",
                    ));
                }
                let (chan, samp) = (list[0], list[1]);
                if chan >= num_channels {
                    return Err(PyIndexError::new_err(format!(
                        "Channel index {chan} out of bounds for audio with {num_channels} channels"
                    )));
                }
                if samp >= num_samples {
                    return Err(PyIndexError::new_err(format!(
                        "Sample index {samp} out of bounds for audio with {num_samples} samples per channel"
                    )));
                }
                Ok((Some(chan), samp))
            } else {
                Err(PyTypeError::new_err(
                    "Index must be an integer (for mono), tuple (channel, sample), or list [channel, sample]",
                ))
            }
        }

        match &self.inner {
            PyAudioDataInner::U8(a) => a.with_view(py, |audio| {
                let (chan_opt, samp) = parse_index(
                    &index,
                    audio.is_mono(),
                    audio.num_channels().get() as usize,
                    audio.samples_per_channel().get(),
                )?;
                let val = match chan_opt {
                    None => audio[samp],
                    Some(c) => audio[(c, samp)],
                };
                Ok(val
                    .into_pyobject(py)
                    .expect("u8 conversion should not fail")
                    .into_any())
            }),
            PyAudioDataInner::I16(a) => a.with_view(py, |audio| {
                let (chan_opt, samp) = parse_index(
                    &index,
                    audio.is_mono(),
                    audio.num_channels().get() as usize,
                    audio.samples_per_channel().get() as usize,
                )?;
                let val = match chan_opt {
                    None => audio[samp],
                    Some(c) => audio[(c, samp)],
                };
                Ok(val
                    .into_pyobject(py)
                    .expect("i16 conversion should not fail")
                    .into_any())
            }),
            PyAudioDataInner::I24(a) => a.with_view(py, |audio| {
                let (chan_opt, samp) = parse_index(
                    &index,
                    audio.is_mono(),
                    audio.num_channels().get() as usize,
                    audio.samples_per_channel().get(),
                )?;
                let val: i32 = match chan_opt {
                    None => audio[samp].into(),
                    Some(c) => audio[(c, samp)].into(),
                };
                Ok(val
                    .into_pyobject(py)
                    .expect("i32 conversion should not fail")
                    .into_any())
            }),
            PyAudioDataInner::I32(a) => a.with_view(py, |audio| {
                let (chan_opt, samp) = parse_index(
                    &index,
                    audio.is_mono(),
                    audio.num_channels().get() as usize,
                    audio.samples_per_channel().get(),
                )?;
                let val = match chan_opt {
                    None => audio[samp],
                    Some(c) => audio[(c, samp)],
                };
                Ok(val
                    .into_pyobject(py)
                    .expect("i32 conversion should not fail")
                    .into_any())
            }),
            PyAudioDataInner::F32(a) => a.with_view(py, |audio| {
                let (chan_opt, samp) = parse_index(
                    &index,
                    audio.is_mono(),
                    audio.num_channels().get() as usize,
                    audio.samples_per_channel().get(),
                )?;
                let val = match chan_opt {
                    None => audio[samp],
                    Some(c) => audio[(c, samp)],
                };
                Ok(val
                    .into_pyobject(py)
                    .expect("f32 conversion should not fail")
                    .into_any())
            }),
            PyAudioDataInner::F64(a) => a.with_view(py, |audio| {
                let (chan_opt, samp) = parse_index(
                    &index,
                    audio.is_mono(),
                    audio.num_channels().get() as usize,
                    audio.samples_per_channel().get(),
                )?;
                let val = match chan_opt {
                    None => audio[samp],
                    Some(c) => audio[(c, samp)],
                };
                Ok(val
                    .into_pyobject(py)
                    .expect("f64 conversion should not fail")
                    .into_any())
            }),
        }
    }

    fn __str__(&self, py: Python<'_>) -> String {
        // The core crate's compact multi-channel Display indexes channels by
        // the sample count and panics; use the detailed view for those.
        if self.is_multi_channel(py) {
            format!("{self:#}")
        } else {
            format!("{self}")
        }
    }

    fn __repr__(&self) -> String {
        format!("{self:#}")
    }

    /// Rich HTML view for Jupyter: a summary table (dtype, channels, sample
    /// rate, duration, peak, RMS) and, for clips up to 30 s with one or two
    /// channels, an inline ``<audio>`` player (16-bit PCM WAV, base64).
    ///
    /// Never raises: on any failure it returns ``None`` and Jupyter falls back
    /// to the plain-text ``repr``.
    fn _repr_html_(slf: &Bound<'_, Self>) -> Option<String> {
        let py = slf.py();
        py.import("audio_samples._display")
            .and_then(|m| m.getattr("repr_html")?.call1((slf,))?.extract::<String>())
            .ok()
    }

    fn __add__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.arithmetic(py, other, Arithmetic::Add)
    }

    fn __mul__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.arithmetic(py, other, Arithmetic::Mul)
    }

    fn __truediv__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.arithmetic(py, other, Arithmetic::Div)
    }

    fn __sub__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.arithmetic(py, other, Arithmetic::Sub)
    }

    fn __pow__(&self, py: Python<'_>, exponent: f64, modulo: Option<f64>) -> Self {
        dispatch_with_view!(self, py, |audio| {
            let powered = audio
                .powf(exponent, modulo.map(|m| m.convert_to()))
                .into_owned();
            PyAudioSamples::from_audio_samples(powered)
        })
    }

    /// `-audio`, in this audio's sample type. Integer results saturate, so `-(-32768)` is 32767 on
    /// int16, and uint8 mirrors about silence (128), so a stored 0 gives 255.
    fn __neg__(&self, py: Python<'_>) -> PyResult<Self> {
        self.unary(py, "negative")
    }

    /// `+audio`: a copy of this audio.
    fn __pos__(&self, py: Python<'_>) -> Self {
        self.clone_py(py)
    }

    /// `abs(audio)`, in this audio's sample type. Integer results saturate, so `abs(-32768)` is
    /// 32767 on int16, and uint8 is measured from silence (128), so a stored 100 gives 156.
    fn __abs__(&self, py: Python<'_>) -> PyResult<Self> {
        self.unary(py, "absolute")
    }

    // Reverse operations for scalar operands on the left
    fn __radd__(&self, py: Python<'_>, scalar: &Bound<'_, PyAny>) -> PyResult<Self> {
        // scalar + audio is the same as audio + scalar
        self.__add__(py, scalar)
    }

    fn __rmul__(&self, py: Python<'_>, scalar: &Bound<'_, PyAny>) -> PyResult<Self> {
        // scalar * audio is the same as audio * scalar
        self.__mul__(py, scalar)
    }

    fn __rsub__(&self, py: Python<'_>, scalar: &Bound<'_, PyAny>) -> PyResult<Self> {
        match classify_scalar(scalar)? {
            Some(Scalar::Integer(value)) if !self.is_float_samples() => {
                self.arithmetic_with_integer(py, &value, "rsub")
            }
            Some(_) => {
                // scalar - audio = -(audio - scalar) = -audio + scalar
                let neg_one = (-1.0f64).into_pyobject(py)?;
                let negated = self.__mul__(py, &neg_one)?;
                negated.__add__(py, scalar)
            }
            None => Err(PyTypeError::new_err(
                "only a numeric scalar can have AudioSamples subtracted from it",
            )),
        }
    }

    fn __rtruediv__(&self, _py: Python<'_>, _scalar: f64) -> PyResult<Self> {
        Err(PyTypeError::new_err(
            "scalar / AudioSamples is not implemented; use AudioSamples / scalar instead",
        ))
    }

    // In-place operations
    fn __iadd__(&mut self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<()> {
        *self = self.arithmetic(py, other, Arithmetic::Add)?;
        Ok(())
    }

    fn __isub__(&mut self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<()> {
        *self = self.arithmetic(py, other, Arithmetic::Sub)?;
        Ok(())
    }

    fn __imul__(&mut self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<()> {
        *self = self.arithmetic(py, other, Arithmetic::Mul)?;
        Ok(())
    }

    fn __itruediv__(&mut self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<()> {
        *self = self.arithmetic(py, other, Arithmetic::Div)?;
        Ok(())
    }

    // Comparison operations
    fn __eq__(&self, py: Python<'_>, other: &PyAudioSamples) -> bool {
        // First check if they have the same structure
        if self.sample_rate(py) != other.sample_rate(py)
            || self.channels(py) != other.channels(py)
            || self.len(py) != other.len(py)
        {
            return false;
        }

        // For now, we'll do a simple comparison by converting both to numpy arrays
        // This is not the most efficient but it works for basic equality checks
        let self_array = self.to_numpy(py);
        let other_array = other.to_numpy(py);

        match (self_array, other_array) {
            (Ok(_self_arr), Ok(_other_arr)) => {
                // For now, return false for simplicity
                // Proper comparison would need element-wise comparison with tolerance
                false
            }
            _ => false,
        }
    }

    fn __ne__(&self, py: Python<'_>, other: &PyAudioSamples) -> bool {
        !self.__eq__(py, other)
    }

    // Helper methods for numpy array operations
    fn add_numpy_array(
        &self,
        py: Python<'_>,
        numpy_array: &Bound<'_, PyUntypedArray>,
    ) -> PyResult<Self> {
        // Element-wise addition; the result keeps this audio's sample type
        self.arithmetic_with_array(py, numpy_array.as_any(), Arithmetic::Add)
    }

    fn sub_numpy_array(
        &self,
        py: Python<'_>,
        numpy_array: &Bound<'_, PyUntypedArray>,
    ) -> PyResult<Self> {
        // Element-wise subtraction; the result keeps this audio's sample type
        self.arithmetic_with_array(py, numpy_array.as_any(), Arithmetic::Sub)
    }

    fn mul_numpy_array(
        &self,
        py: Python<'_>,
        numpy_array: &Bound<'_, PyUntypedArray>,
    ) -> PyResult<Self> {
        // Element-wise multiplication; the result keeps this audio's sample type
        self.arithmetic_with_array(py, numpy_array.as_any(), Arithmetic::Mul)
    }

    fn div_numpy_array(
        &self,
        py: Python<'_>,
        numpy_array: &Bound<'_, PyUntypedArray>,
    ) -> PyResult<Self> {
        // Element-wise division; the result keeps this audio's sample type
        self.arithmetic_with_array(py, numpy_array.as_any(), Arithmetic::Div)
    }

    fn from_numpy_array_with_metadata(
        &self,
        py: Python<'_>,
        numpy_array: &Bound<'_, PyUntypedArray>,
    ) -> PyResult<Self> {
        // Create AudioSamples from numpy array using the same sample rate and channel layout as self
        let sample_rate = self.sample_rate(py);

        // Check compatibility
        let array_shape = numpy_array.shape();
        let self_shape = match self.channels(py) {
            1 => vec![self.len(py)],
            n => vec![n, self.len(py) / n],
        };

        if array_shape != self_shape {
            return Err(PyValueError::new_err(format!(
                "Incompatible shapes: AudioSamples has shape {:?}, numpy array has shape {:?}",
                self_shape, array_shape
            )));
        }

        // Convert based on array dimensions
        if array_shape.len() == 1 {
            // Mono audio
            Self::py_new_mono(py, numpy_array.clone(), sample_rate)
        } else if array_shape.len() == 2 {
            // Multi-channel audio
            Self::py_new_multi(py, numpy_array.clone(), sample_rate)
        } else {
            Err(PyValueError::new_err("Only 1D and 2D arrays are supported"))
        }
    }

    /// NumPy ufunc protocol (``np.sin(audio)``, ``np.abs(audio)``, ``arr + audio``).
    ///
    /// Every ``AudioSamples`` input is converted with :meth:`to_numpy` and the
    /// ufunc is applied to the plain arrays, so ``np.sin(audio)`` returns the
    /// same NumPy array as ``np.sin(audio.to_numpy())``. The arithmetic ufuncs
    /// ``add``, ``subtract``, ``multiply`` and ``divide`` (which NumPy calls for
    /// ``ndarray + audio`` and friends) are the exception: when the result
    /// keeps this buffer's shape and a supported sample dtype it is wrapped
    /// back into an ``AudioSamples`` with this buffer's sample rate, matching
    /// ``audio + ndarray``.
    ///
    /// Raises:
    ///     AudioParameterError: If two ``AudioSamples`` inputs have different
    ///         sample rates (a ``ValueError``).
    #[pyo3(signature = (ufunc, method, *inputs, **kwargs))]
    fn __array_ufunc__<'py>(
        slf: &Bound<'py, Self>,
        ufunc: &Bound<'py, PyAny>,
        method: &str,
        inputs: &Bound<'py, PyTuple>,
        kwargs: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        // Writing into an AudioSamples through `out=` is not supported.
        if let Some(kw) = kwargs {
            if let Some(out) = kw.get_item("out")? {
                let outs: Vec<Bound<'py, PyAny>> = match out.cast::<PyTuple>() {
                    Ok(t) => t.iter().collect(),
                    Err(_) => vec![out.clone()],
                };
                if outs.iter().any(|o| o.is_instance_of::<PyAudioSamples>()) {
                    return Ok(py.NotImplemented());
                }
            }
        }

        let this = slf.borrow();
        // Arithmetic with NumPy on the left means the same as the operators.
        if let (Some(operation), "__call__", None, 2) = (
            Arithmetic::from_ufunc(ufunc_name_raw(ufunc).as_str()),
            method,
            kwargs,
            inputs.len(),
        ) {
            let (left, right) = (inputs.get_item(0)?, inputs.get_item(1)?);
            if left.is_instance_of::<PyAudioSamples>() {
                return Ok(left
                    .cast::<PyAudioSamples>()?
                    .borrow()
                    .arithmetic(py, &right, operation)?
                    .into_bound_py_any(py)?
                    .unbind());
            }
            if left.is_instance_of::<PyUntypedArray>() || classify_scalar(&left)?.is_some() {
                return Ok(this
                    .arithmetic_reflected(py, &left, operation)?
                    .into_bound_py_any(py)?
                    .unbind());
            }
        }

        let sample_rate = this.sample_rate(py);
        let mut converted = Vec::with_capacity(inputs.len());
        for item in inputs.iter() {
            if let Ok(audio) = item.cast::<PyAudioSamples>() {
                let audio = audio.borrow();
                check_same_sample_rate(py, &this, &audio, ufunc_name(ufunc).as_str())?;
                converted.push(audio.to_numpy(py)?);
            } else {
                converted.push(item);
            }
        }

        let result = ufunc
            .getattr(method)?
            .call(PyTuple::new(py, converted)?, kwargs)?;

        let arithmetic = matches!(
            ufunc_name_raw(ufunc).as_str(),
            "add" | "subtract" | "multiply" | "divide" | "true_divide"
        );
        if method == "__call__" && arithmetic {
            if let Ok(arr) = result.cast::<PyUntypedArray>() {
                if arr.shape() == this.shape(py).as_slice() {
                    if let Ok(wrapped) = Self::from_untyped_array(py, arr, sample_rate) {
                        return Ok(wrapped.into_bound_py_any(py)?.unbind());
                    }
                }
            }
        }
        Ok(result.unbind())
    }

    /// NumPy array-function protocol (``np.mean(audio)``, ``np.fft.rfft(audio)``).
    ///
    /// ``AudioSamples`` arguments (also inside lists and tuples) are converted
    /// with :meth:`to_numpy` and the NumPy function is called on the arrays.
    /// The result is whatever NumPy returns (an array or a scalar).
    #[pyo3(signature = (func, types, args, kwargs))]
    fn __array_function__<'py>(
        &self,
        py: Python<'py>,
        func: &Bound<'py, PyAny>,
        types: &Bound<'py, PyAny>,
        args: &Bound<'py, PyTuple>,
        kwargs: &Bound<'py, PyDict>,
    ) -> PyResult<Py<PyAny>> {
        let _ = types;
        let new_args = PyTuple::new(
            py,
            args.iter()
                .map(|a| audio_args_to_numpy(py, a))
                .collect::<PyResult<Vec<_>>>()?,
        )?;
        let new_kwargs = PyDict::new(py);
        for (k, v) in kwargs.iter() {
            new_kwargs.set_item(k, audio_args_to_numpy(py, v)?)?;
        }
        // `_implementation` skips the dispatch step, so a nested AudioSamples
        // that was not converted above cannot re-enter this method.
        let target = func
            .getattr("_implementation")
            .unwrap_or_else(|_| func.clone());
        Ok(target.call(new_args, Some(&new_kwargs))?.unbind())
    }

    #[staticmethod]
    #[pyo3(name = "zeros_mono", signature = (length: "int", sample_rate: "int"))]
    fn py_zeros_mono_f32(length: usize, sample_rate: NonZeroU32) -> PyResult<Self> {
        check_non_empty(1, length)?;
        let data = Array1::<f32>::zeros(length);
        Ok(Self::new_mono(data, sample_rate))
    }

    #[staticmethod]
    #[pyo3(name = "zeros_mono_i16", signature = (length: "int", sample_rate: "int"))]
    fn py_zeros_mono_i16(length: usize, sample_rate: NonZeroU32) -> PyResult<Self> {
        check_non_empty(1, length)?;
        let data = Array1::<i16>::zeros(length);
        Ok(Self::new_mono(data, sample_rate))
    }

    #[staticmethod]
    #[pyo3(name = "zeros_mono_i32", signature = (length: "int", sample_rate: "int"))]
    fn py_zeros_mono_i32(length: usize, sample_rate: NonZeroU32) -> PyResult<Self> {
        check_non_empty(1, length)?;
        let data = Array1::<i32>::zeros(length);
        Ok(Self::new_mono(data, sample_rate))
    }

    #[staticmethod]
    #[pyo3(name = "zeros_mono_f64", signature = (length: "int", sample_rate: "int"))]
    fn py_zeros_mono_f64(length: usize, sample_rate: NonZeroU32) -> PyResult<Self> {
        check_non_empty(1, length)?;
        let data = Array1::<f64>::zeros(length);
        Ok(Self::new_mono(data, sample_rate))
    }

    #[staticmethod]
    #[pyo3(name = "ones_mono", signature = (length: "int", sample_rate: "int"))]
    fn py_ones_mono_f32(length: usize, sample_rate: NonZeroU32) -> PyResult<Self> {
        check_non_empty(1, length)?;
        let data = Array1::<f32>::ones(length);
        Ok(Self::new_mono(data, sample_rate))
    }

    #[staticmethod]
    #[pyo3(name = "ones_mono_i16", signature = (length: "int", sample_rate: "int"))]
    fn py_ones_mono_i16(length: usize, sample_rate: NonZeroU32) -> PyResult<Self> {
        check_non_empty(1, length)?;
        let data = Array1::<i16>::ones(length);
        Ok(Self::new_mono(data, sample_rate))
    }

    #[staticmethod]
    #[pyo3(name = "ones_mono_i32", signature = (length: "int", sample_rate: "int"))]
    fn py_ones_mono_i32(length: usize, sample_rate: NonZeroU32) -> PyResult<Self> {
        check_non_empty(1, length)?;
        let data = Array1::<i32>::ones(length);
        Ok(Self::new_mono(data, sample_rate))
    }

    #[staticmethod]
    #[pyo3(name = "ones_mono_f64", signature = (length: "int", sample_rate: "int"))]
    fn py_ones_mono_f64(length: usize, sample_rate: NonZeroU32) -> PyResult<Self> {
        check_non_empty(1, length)?;
        let data = Array1::<f64>::ones(length);
        Ok(Self::new_mono(data, sample_rate))
    }

    #[staticmethod]
    #[pyo3(name = "uniform_mono", signature = (length: "int", sample_rate: "int", value: "float"))]
    fn py_uniform_mono_f32(length: usize, sample_rate: NonZeroU32, value: f32) -> PyResult<Self> {
        check_non_empty(1, length)?;
        let data = Array1::<f32>::from_elem(length, value);
        Ok(Self::new_mono(data, sample_rate))
    }

    #[staticmethod]
    #[pyo3(name = "uniform_mono_i16", signature = (length: "int", sample_rate: "int", value: "int"))]
    fn py_uniform_mono_i16(length: usize, sample_rate: NonZeroU32, value: i16) -> PyResult<Self> {
        check_non_empty(1, length)?;
        let data = Array1::<i16>::from_elem(length, value);
        Ok(Self::new_mono(data, sample_rate))
    }

    #[staticmethod]
    #[pyo3(name = "uniform_mono_i32", signature = (length: "int", sample_rate: "int", value: "int"))]
    fn py_uniform_mono_i32(length: usize, sample_rate: NonZeroU32, value: i32) -> PyResult<Self> {
        check_non_empty(1, length)?;
        let data = Array1::<i32>::from_elem(length, value);
        Ok(Self::new_mono(data, sample_rate))
    }

    #[staticmethod]
    #[pyo3(name = "uniform_mono_f64", signature = (length: "int", sample_rate: "int", value: "float"))]
    fn py_uniform_mono_f64(length: usize, sample_rate: NonZeroU32, value: f64) -> PyResult<Self> {
        check_non_empty(1, length)?;
        let data = Array1::<f64>::from_elem(length, value);
        Ok(Self::new_mono(data, sample_rate))
    }

    #[staticmethod]
    #[pyo3(name = "zeros_multi", signature = (channels: "int", length: "int", sample_rate: "int"))]
    fn py_zeros_multi_f32(
        channels: usize,
        length: usize,
        sample_rate: NonZeroU32,
    ) -> PyResult<Self> {
        check_non_empty(channels, length)?;
        let data = Array2::<f32>::zeros((channels, length));
        Ok(Self::new_multi(data, sample_rate))
    }

    #[staticmethod]
    #[pyo3(name = "zeros_multi_i16", signature = (channels: "int", length: "int", sample_rate: "int"))]
    fn py_zeros_multi_i16(
        channels: usize,
        length: usize,
        sample_rate: NonZeroU32,
    ) -> PyResult<Self> {
        check_non_empty(channels, length)?;
        let data = Array2::<i16>::zeros((channels, length));
        Ok(Self::new_multi(data, sample_rate))
    }

    #[staticmethod]
    #[pyo3(name = "zeros_multi_i32", signature = (channels: "int", length: "int", sample_rate: "int"))]
    fn py_zeros_multi_i32(
        channels: usize,
        length: usize,
        sample_rate: NonZeroU32,
    ) -> PyResult<Self> {
        check_non_empty(channels, length)?;
        let data = Array2::<i32>::zeros((channels, length));
        Ok(Self::new_multi(data, sample_rate))
    }

    #[staticmethod]
    #[pyo3(name = "zeros_multi_f64", signature = (channels: "int", length: "int", sample_rate: "int"))]
    fn py_zeros_multi_f64(
        channels: usize,
        length: usize,
        sample_rate: NonZeroU32,
    ) -> PyResult<Self> {
        check_non_empty(channels, length)?;
        let data = Array2::<f64>::zeros((channels, length));
        Ok(Self::new_multi(data, sample_rate))
    }

    #[staticmethod]
    #[pyo3(name = "ones_multi", signature = (channels: "int", length: "int", sample_rate: "int"))]
    fn py_ones_multi_f32(
        channels: usize,
        length: usize,
        sample_rate: NonZeroU32,
    ) -> PyResult<Self> {
        check_non_empty(channels, length)?;
        let data = Array2::<f32>::ones((channels, length));
        Ok(Self::new_multi(data, sample_rate))
    }

    #[staticmethod]
    #[pyo3(name = "ones_multi_i16", signature = (channels: "int", length: "int", sample_rate: "int"))]
    fn py_ones_multi_i16(
        channels: usize,
        length: usize,
        sample_rate: NonZeroU32,
    ) -> PyResult<Self> {
        check_non_empty(channels, length)?;
        let data = Array2::<i16>::ones((channels, length));
        Ok(Self::new_multi(data, sample_rate))
    }

    #[staticmethod]
    #[pyo3(name = "ones_multi_i32", signature = (channels: "int", length: "int", sample_rate: "int"))]
    fn py_ones_multi_i32(
        channels: usize,
        length: usize,
        sample_rate: NonZeroU32,
    ) -> PyResult<Self> {
        check_non_empty(channels, length)?;
        let data = Array2::<i32>::ones((channels, length));
        Ok(Self::new_multi(data, sample_rate))
    }

    #[staticmethod]
    #[pyo3(name = "ones_multi_f64", signature = (channels: "int", length: "int", sample_rate: "int"))]
    fn py_ones_multi_f64(
        channels: usize,
        length: usize,
        sample_rate: NonZeroU32,
    ) -> PyResult<Self> {
        check_non_empty(channels, length)?;
        let data = Array2::<f64>::ones((channels, length));
        Ok(Self::new_multi(data, sample_rate))
    }

    #[staticmethod]
    #[pyo3(name = "uniform_multi", signature = (channels: "int", length: "int", sample_rate: "int", value: "float"))]
    fn py_uniform_multi_f32(
        channels: usize,
        length: usize,
        sample_rate: NonZeroU32,
        value: f32,
    ) -> PyResult<Self> {
        check_non_empty(channels, length)?;
        let data = Array2::<f32>::from_elem((channels, length), value);
        Ok(Self::new_multi(data, sample_rate))
    }

    #[staticmethod]
    #[pyo3(name = "uniform_multi_i16", signature = (channels: "int", length: "int", sample_rate: "int", value: "int"))]
    fn py_uniform_multi_i16(
        channels: usize,
        length: usize,
        sample_rate: NonZeroU32,
        value: i16,
    ) -> PyResult<Self> {
        check_non_empty(channels, length)?;
        let data = Array2::<i16>::from_elem((channels, length), value);
        Ok(Self::new_multi(data, sample_rate))
    }

    #[staticmethod]
    #[pyo3(name = "uniform_multi_i32", signature = (channels: "int", length: "int", sample_rate: "int", value: "int"))]
    fn py_uniform_multi_i32(
        channels: usize,
        length: usize,
        sample_rate: NonZeroU32,
        value: i32,
    ) -> PyResult<Self> {
        check_non_empty(channels, length)?;
        let data = Array2::<i32>::from_elem((channels, length), value);
        Ok(Self::new_multi(data, sample_rate))
    }

    #[staticmethod]
    #[pyo3(name = "uniform_multi_f64", signature = (channels: "int", length: "int", sample_rate: "int", value: "float"))]
    fn py_uniform_multi_f64(
        channels: usize,
        length: usize,
        sample_rate: NonZeroU32,
        value: f64,
    ) -> PyResult<Self> {
        check_non_empty(channels, length)?;
        let data = Array2::<f64>::from_elem((channels, length), value);
        Ok(Self::new_multi(data, sample_rate))
    }

    #[pyo3(name = "nyquist", signature=())]
    /// Nyquist frequency in Hz (``sample_rate / 2``).
    fn py_nyquist(&self, py: Python<'_>) -> f64 {
        dispatch_with_view!(self, py, |audio| { audio.nyquist() })
    }

    /// Convert to another sample type with audio-aware scaling.
    ///
    /// Audio-aware conversion: values are rescaled between the formats'
    /// full-scale ranges, so a float ``1.0`` and an int16 ``32767`` both mean
    /// full scale. Integer scaling is asymmetric because two's-complement
    /// ranges are: float ``-1.0`` maps to ``-32768`` and ``+1.0`` to ``32767``
    /// (int16; likewise for int24/int32), i.e. negative values scale by
    /// ``2**(bits-1)`` and positive values by ``2**(bits-1) - 1``, with
    /// rounding and saturation at the integer limits. uint8 is offset binary
    /// (``-1.0 -> 0``, ``0.0 -> 128``, ``1.0 -> 255``). Float-to-float
    /// conversions do not clip. Compare :meth:`cast_as`, which keeps the
    /// numeric value instead.
    ///
    /// Args:
    ///     dtype (SampleType): Target type, e.g. ``SampleType.I16``.
    ///
    /// Returns:
    ///     AudioSamples: New buffer of the target type.
    ///
    /// Examples:
    ///     >>> import numpy as np, audio_samples as aus
    ///     >>> x = aus.AudioSamples.new_mono(np.array([-1.0, -0.5, 0.0, 0.5, 1.0]), sample_rate=8000)
    ///     >>> x.to_format(aus.SampleType.I16).to_numpy()
    ///     array([-32768, -16384,      0,  16384,  32767], dtype=int16)
    #[pyo3(signature = (dtype))]
    fn to_format(&self, py: Python<'_>, dtype: PySampleType) -> PyAudioSamples {
        dispatch_with_view!(self, py, |audio| {
            match dtype {
                PySampleType::U8 => PyAudioSamples::from_audio_samples(audio.as_u8()),
                PySampleType::I16 => PyAudioSamples::from_audio_samples(audio.as_i16()),
                PySampleType::I24 => PyAudioSamples::from_audio_samples(audio.as_i24()),
                PySampleType::I32 => PyAudioSamples::from_audio_samples(audio.as_i32()),
                PySampleType::F32 => PyAudioSamples::from_audio_samples(audio.as_f32()),
                PySampleType::F64 => PyAudioSamples::from_audio_samples(audio.as_f64()),
            }
        })
    }

    /// Audio-aware conversion to uint8 (offset binary); shorthand for ``to_format(SampleType.U8)``.
    ///
    /// Float ``-1.0``/``+1.0`` map to the integer minimum/maximum
    /// (asymmetric: int16 ``-32768``/``32767``); see :meth:`to_format`.
    /// Use ``cast_as_u8`` for a raw numeric cast instead.
    #[pyo3(signature = ())]
    fn as_u8(&self, py: Python<'_>) -> PyAudioSamples {
        self.to_format(py, PySampleType::U8)
    }

    /// Audio-aware conversion to int16; shorthand for ``to_format(SampleType.I16)``.
    ///
    /// Float ``-1.0``/``+1.0`` map to the integer minimum/maximum
    /// (asymmetric: int16 ``-32768``/``32767``); see :meth:`to_format`.
    /// Use ``cast_as_i16`` for a raw numeric cast instead.
    #[pyo3(signature = ())]
    fn as_i16(&self, py: Python<'_>) -> PyAudioSamples {
        self.to_format(py, PySampleType::I16)
    }

    /// Audio-aware conversion to int32; shorthand for ``to_format(SampleType.I32)``.
    ///
    /// Float ``-1.0``/``+1.0`` map to the integer minimum/maximum
    /// (asymmetric: int16 ``-32768``/``32767``); see :meth:`to_format`.
    /// Use ``cast_as_i32`` for a raw numeric cast instead.
    #[pyo3(signature = ())]
    fn as_i32(&self, py: Python<'_>) -> PyAudioSamples {
        self.to_format(py, PySampleType::I32)
    }

    /// Audio-aware conversion to float32; shorthand for ``to_format(SampleType.F32)``.
    ///
    /// Float ``-1.0``/``+1.0`` map to the integer minimum/maximum
    /// (asymmetric: int16 ``-32768``/``32767``); see :meth:`to_format`.
    /// Use ``cast_as_f32`` for a raw numeric cast instead.
    #[pyo3(signature = ())]
    fn as_f32(&self, py: Python<'_>) -> PyAudioSamples {
        self.to_format(py, PySampleType::F32)
    }

    /// Audio-aware conversion to float64; shorthand for ``to_format(SampleType.F64)``.
    ///
    /// Float ``-1.0``/``+1.0`` map to the integer minimum/maximum
    /// (asymmetric: int16 ``-32768``/``32767``); see :meth:`to_format`.
    /// Use ``cast_as_f64`` for a raw numeric cast instead.
    #[pyo3(signature = ())]
    fn as_f64(&self, py: Python<'_>) -> PyAudioSamples {
        self.to_format(py, PySampleType::F64)
    }

    /// Cast to another sample type without audio-aware scaling.
    ///
    /// Raw numeric cast: each value keeps its number (like
    /// ``numpy.ndarray.astype`` with saturation), with no rescaling between
    /// full-scale ranges. Float-to-integer casts truncate toward zero and
    /// saturate at the type's limits, so float audio in ``[-1, 1]`` becomes
    /// mostly ``-1``, ``0`` and ``1``; int16 ``16384`` cast to float64 is
    /// ``16384.0``, not ``0.5``. Use :meth:`to_format` / ``as_*`` for
    /// audio-aware conversion.
    ///
    /// Args:
    ///     dtype (SampleType): Target type, e.g. ``SampleType.F64``.
    ///
    /// Returns:
    ///     AudioSamples: New buffer of the target type.
    ///
    /// Examples:
    ///     >>> import numpy as np, audio_samples as aus
    ///     >>> x = aus.AudioSamples.new_mono(np.array([16384, -32768], dtype=np.int16), sample_rate=8000)
    ///     >>> x.cast_as(aus.SampleType.F64).to_numpy()
    ///     array([ 16384., -32768.])
    ///     >>> x.to_format(aus.SampleType.F64).to_numpy()
    ///     array([ 0.50001526, -1.        ])
    #[pyo3(signature = (dtype))]
    fn cast_as(&self, py: Python<'_>, dtype: PySampleType) -> PyAudioSamples {
        dispatch_with_view!(self, py, |audio| {
            match dtype {
                PySampleType::U8 => PyAudioSamples::from_audio_samples(audio.cast_as::<u8>()),
                PySampleType::I16 => PyAudioSamples::from_audio_samples(audio.cast_as::<i16>()),
                PySampleType::I24 => PyAudioSamples::from_audio_samples(audio.cast_as::<I24>()),
                PySampleType::I32 => PyAudioSamples::from_audio_samples(audio.cast_as::<i32>()),
                PySampleType::F32 => PyAudioSamples::from_audio_samples(audio.cast_as::<f32>()),
                PySampleType::F64 => PyAudioSamples::from_audio_samples(audio.cast_as::<f64>()),
            }
        })
    }

    /// Raw numeric cast to uint8 (no rescaling); shorthand for
    /// ``cast_as(SampleType.U8)``. See :meth:`cast_as`; use ``as_u8``
    /// for audio-aware conversion.
    #[pyo3(signature = ())]
    fn cast_as_u8(&self, py: Python<'_>) -> PyAudioSamples {
        self.cast_as(py, PySampleType::U8)
    }

    /// Raw numeric cast to int16 (no rescaling); shorthand for
    /// ``cast_as(SampleType.I16)``. See :meth:`cast_as`; use ``as_i16``
    /// for audio-aware conversion.
    #[pyo3(signature = ())]
    fn cast_as_i16(&self, py: Python<'_>) -> PyAudioSamples {
        self.cast_as(py, PySampleType::I16)
    }

    /// Raw numeric cast to int32 (no rescaling); shorthand for
    /// ``cast_as(SampleType.I32)``. See :meth:`cast_as`; use ``as_i32``
    /// for audio-aware conversion.
    #[pyo3(signature = ())]
    fn cast_as_i32(&self, py: Python<'_>) -> PyAudioSamples {
        self.cast_as(py, PySampleType::I32)
    }

    /// Raw numeric cast to float32 (no rescaling); shorthand for
    /// ``cast_as(SampleType.F32)``. See :meth:`cast_as`; use ``as_f32``
    /// for audio-aware conversion.
    #[pyo3(signature = ())]
    fn cast_as_f32(&self, py: Python<'_>) -> PyAudioSamples {
        self.cast_as(py, PySampleType::F32)
    }

    /// Raw numeric cast to float64 (no rescaling); shorthand for
    /// ``cast_as(SampleType.F64)``. See :meth:`cast_as`; use ``as_f64``
    /// for audio-aware conversion.
    #[pyo3(signature = ())]
    fn cast_as_f64(&self, py: Python<'_>) -> PyAudioSamples {
        self.cast_as(py, PySampleType::F64)
    }

    /// NumPy array protocol. ``copy=True`` (NumPy 2) returns a fresh copy.
    #[pyo3(signature = (dtype=None, copy=None))]
    fn __array__(
        &self,
        py: Python<'_>,
        dtype: Option<&Bound<'_, PyArrayDescr>>,
        copy: Option<bool>,
    ) -> PyResult<Py<PyAny>> {
        if copy == Some(true) {
            let arr = self.__array__(py, dtype, None)?;
            return Ok(arr.bind(py).call_method0("copy")?.unbind());
        }
        match dtype {
            Some(dt) if !dt.is_equiv_to(&self.dtype(py)) => {
                // Cast to requested dtype first
                let casted = self.cast_as(py, PySampleType::from_numpy(py, dt)?);
                casted.__array__(py, None, None)
            }
            _ => {
                // Return array with current dtype
                Ok(self.to_numpy(py)?.into_any().unbind())
            }
        }
    }

    #[getter]
    fn __array_interface__(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let dict = PyDict::new(py);

        match &self.inner {
            PyAudioDataInner::U8(typed) => typed.with_view(py, |audio| {
                let shape = match audio.data() {
                    AudioData::Mono(arr) => (arr.as_view().len(),).into_pyobject(py)?,
                    AudioData::Multi(arr) => {
                        let view = arr.as_view();
                        (view.nrows(), view.ncols()).into_pyobject(py)?
                    }
                };
                dict.set_item("shape", &shape)?;
                dict.set_item("typestr", "|u1")?;
                self.set_array_interface_data(py, &dict, audio.data())
            }),
            PyAudioDataInner::I16(typed) => typed.with_view(py, |audio| {
                let shape = match audio.data() {
                    AudioData::Mono(arr) => (arr.as_view().len(),).into_pyobject(py)?,
                    AudioData::Multi(arr) => {
                        let view = arr.as_view();
                        (view.nrows(), view.ncols()).into_pyobject(py)?
                    }
                };
                dict.set_item("shape", &shape)?;
                dict.set_item("typestr", "<i2")?;
                self.set_array_interface_data(py, &dict, audio.data())
            }),
            PyAudioDataInner::I24(_) => {
                // I24 has no numpy equivalent: expose an owned int32 copy. Its
                // interface must reference the copy itself (as the `data`
                // buffer object) rather than a raw pointer, otherwise the copy
                // is freed when this function returns and numpy reads freed
                // memory.
                // C-contiguous so it can be shared through the buffer protocol.
                let arr = py
                    .import("numpy")?
                    .call_method1("ascontiguousarray", (self.cast_as_i32(py).to_numpy(py)?,))?;
                let iface = arr
                    .getattr("__array_interface__")?
                    .cast_into::<PyDict>()
                    .map_err(PyErr::from)?;
                iface.set_item("data", &arr)?;
                Ok(iface.unbind())
            }
            PyAudioDataInner::I32(typed) => typed.with_view(py, |audio| {
                let shape = match audio.data() {
                    AudioData::Mono(arr) => (arr.as_view().len(),).into_pyobject(py)?,
                    AudioData::Multi(arr) => {
                        let view = arr.as_view();
                        (view.nrows(), view.ncols()).into_pyobject(py)?
                    }
                };
                dict.set_item("shape", &shape)?;
                dict.set_item("typestr", "<i4")?;
                self.set_array_interface_data(py, &dict, audio.data())
            }),
            PyAudioDataInner::F32(typed) => typed.with_view(py, |audio| {
                let shape = match audio.data() {
                    AudioData::Mono(arr) => (arr.as_view().len(),).into_pyobject(py)?,
                    AudioData::Multi(arr) => {
                        let view = arr.as_view();
                        (view.nrows(), view.ncols()).into_pyobject(py)?
                    }
                };
                dict.set_item("shape", &shape)?;
                dict.set_item("typestr", "<f4")?;
                self.set_array_interface_data(py, &dict, audio.data())
            }),
            PyAudioDataInner::F64(typed) => typed.with_view(py, |audio| {
                let shape = match audio.data() {
                    AudioData::Mono(arr) => (arr.as_view().len(),).into_pyobject(py)?,
                    AudioData::Multi(arr) => {
                        let view = arr.as_view();
                        (view.nrows(), view.ncols()).into_pyobject(py)?
                    }
                };
                dict.set_item("shape", &shape)?;
                dict.set_item("typestr", "<f8")?;
                self.set_array_interface_data(py, &dict, audio.data())
            }),
        }
    }

    /// Convert AudioSamples to a PyTorch tensor
    #[pyo3(signature = (device: "Optional[str]"))]
    fn to_tensor(&self, py: Python<'_>, device: Option<&str>) -> PyResult<Py<PyAny>> {
        let torch_module = match py.import("torch") {
            Ok(tm) => tm,
            Err(e) => {
                eprintln!(
                    "Failed to import torch. Make sure you install it, it is not a requirement of audio_samples, just for the functions to work."
                );
                return Err(e);
            }
        };

        let numpy_array = self.to_numpy(py)?;
        let device = device.unwrap_or("cpu");
        torch_module
            .call_method1("from_numpy", (numpy_array,))?
            .call_method1("to", (device,))
            .map(|result| result.unbind())
    }

    /// Convert AudioSamples to a PyTorch tensor on the GPU if possible
    #[pyo3(signature = (gpu_id: "Optional[str]" = None))]
    fn to_gpu_tensor(&self, py: Python<'_>, gpu_id: Option<&str>) -> PyResult<Py<PyAny>> {
        let gpu_id: &str = gpu_id.unwrap_or("cuda");
        self.to_tensor(py, Some(gpu_id))
    }

    /// Convert audio samples to a NumPy array.
    ///
    /// Returns the audio data as a numpy.ndarray. If the audio is already
    /// backed by a numpy array, returns the existing array (zero-copy).
    /// Otherwise, creates a new numpy array from the owned data.
    ///
    /// Returns:
    ///     numpy.ndarray: Audio data with shape:
    ///         - (samples,) for mono audio
    ///         - (channels, samples) for multi-channel audio
    ///
    /// Examples:
    ///     >>> audio = aus.AudioSamples.from_array(np.random.randn(2, 44100), 44100)
    ///     >>> arr = audio.to_numpy()
    ///     >>> arr.shape
    ///     (2, 44100)
    ///     >>> arr.dtype
    ///     dtype('float64')
    #[pyo3(signature = ())]
    fn to_numpy<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        use PyAudioBacking::*;

        match &self.inner {
            PyAudioDataInner::U8(typed) => match &typed.backing {
                NumpyMono(handle) => Ok(handle.bind(py).clone().into_any()),
                NumpyMulti(handle) | NumpyInterleaved(handle) => {
                    Ok(handle.bind(py).clone().into_any())
                }
                OwnedMono(arr) => Ok(arr.view().to_pyarray(py).into_any()),
                OwnedMulti(arr) => Ok(arr.view().to_pyarray(py).into_any()),
            },
            PyAudioDataInner::I16(typed) => match &typed.backing {
                NumpyMono(handle) => Ok(handle.bind(py).clone().into_any()),
                NumpyMulti(handle) | NumpyInterleaved(handle) => {
                    Ok(handle.bind(py).clone().into_any())
                }
                OwnedMono(arr) => Ok(arr.view().to_pyarray(py).into_any()),
                OwnedMulti(arr) => Ok(arr.view().to_pyarray(py).into_any()),
            },
            PyAudioDataInner::I24(_) => {
                // Convert I24 to i32 for numpy compatibility
                self.cast_as_i32(py).to_numpy(py)
            }
            PyAudioDataInner::I32(typed) => match &typed.backing {
                NumpyMono(handle) => Ok(handle.bind(py).clone().into_any()),
                NumpyMulti(handle) | NumpyInterleaved(handle) => {
                    Ok(handle.bind(py).clone().into_any())
                }
                OwnedMono(arr) => Ok(arr.view().to_pyarray(py).into_any()),
                OwnedMulti(arr) => Ok(arr.view().to_pyarray(py).into_any()),
            },
            PyAudioDataInner::F32(typed) => match &typed.backing {
                NumpyMono(handle) => Ok(handle.bind(py).clone().into_any()),
                NumpyMulti(handle) | NumpyInterleaved(handle) => {
                    Ok(handle.bind(py).clone().into_any())
                }
                OwnedMono(arr) => Ok(arr.view().to_pyarray(py).into_any()),
                OwnedMulti(arr) => Ok(arr.view().to_pyarray(py).into_any()),
            },
            PyAudioDataInner::F64(typed) => match &typed.backing {
                NumpyMono(handle) => Ok(handle.bind(py).clone().into_any()),
                NumpyMulti(handle) | NumpyInterleaved(handle) => {
                    Ok(handle.bind(py).clone().into_any())
                }
                OwnedMono(arr) => Ok(arr.view().to_pyarray(py).into_any()),
                OwnedMulti(arr) => Ok(arr.view().to_pyarray(py).into_any()),
            },
        }
    }
}

impl Add<Self> for PyAudioSamples {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        PyAudioSamples {
            inner: self.inner + rhs.inner,
        }
    }
}

impl<T: StandardSample + Element> Add<T> for PyAudioSamples {
    type Output = Self;

    fn add(self, rhs: T) -> Self::Output {
        PyAudioSamples {
            inner: self.inner + rhs,
        }
    }
}

impl<'a> Add<&'a PyAudioSamples> for &PyAudioSamples {
    type Output = PyAudioSamples;

    fn add(self, rhs: &'a PyAudioSamples) -> Self::Output {
        PyAudioSamples {
            inner: self.inner.clone() + rhs.inner.clone(),
        }
    }
}

impl Sub<Self> for PyAudioSamples {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        PyAudioSamples {
            inner: self.inner - rhs.inner,
        }
    }
}

impl<T: StandardSample + Element> Sub<T> for PyAudioSamples {
    type Output = Self;

    fn sub(self, rhs: T) -> Self::Output {
        PyAudioSamples {
            inner: self.inner - rhs,
        }
    }
}

impl<'a> Sub<&'a PyAudioSamples> for &PyAudioSamples {
    type Output = PyAudioSamples;

    fn sub(self, rhs: &'a PyAudioSamples) -> Self::Output {
        PyAudioSamples {
            inner: self.inner.clone() - rhs.inner.clone(),
        }
    }
}

impl Mul<Self> for PyAudioSamples {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        PyAudioSamples {
            inner: self.inner * rhs.inner,
        }
    }
}

impl<T: StandardSample + Element> Mul<T> for PyAudioSamples {
    type Output = Self;

    fn mul(self, rhs: T) -> Self::Output {
        PyAudioSamples {
            inner: self.inner * rhs,
        }
    }
}

impl<'a> Mul<&'a PyAudioSamples> for &PyAudioSamples {
    type Output = PyAudioSamples;

    fn mul(self, rhs: &'a PyAudioSamples) -> Self::Output {
        PyAudioSamples {
            inner: self.inner.clone() * rhs.inner.clone(),
        }
    }
}

impl Display for PyAudioSamples {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.inner.fmt(f)
    }
}

/// Reject zero-length buffers at construction (an empty `AudioSamples`
/// violates the core crate's non-empty invariant and would panic later).
fn check_non_empty(channels: usize, length: usize) -> PyResult<()> {
    if channels == 0 || length == 0 {
        return Err(PyValueError::new_err(format!(
            "AudioSamples cannot be empty: got {channels} channel(s) x {length} sample(s); \
both must be greater than zero"
        )));
    }
    Ok(())
}

/// Reject an empty NumPy array (see [`check_non_empty`]).
fn check_array_non_empty(arr: &Bound<'_, PyUntypedArray>) -> PyResult<()> {
    if arr.len() == 0 {
        return Err(PyValueError::new_err(format!(
            "AudioSamples cannot be empty: got an array of shape {:?}; \
it needs at least one sample per channel",
            arr.shape()
        )));
    }
    Ok(())
}

/// Raise `AudioParameterError` (a `ValueError`) when two buffers combined in
/// one operation have different sample rates.
/// A scalar operand of an arithmetic operator.
enum Scalar<'py> {
    /// A Python int or NumPy integer scalar.
    Integer(Bound<'py, PyAny>),
    /// A Python float or NumPy float scalar.
    Float(f64),
}

/// Sorts a scalar operand from anything else, which gives `None`. A bool raises `TypeError`.
fn classify_scalar<'py>(value: &Bound<'py, PyAny>) -> PyResult<Option<Scalar<'py>>> {
    if let Ok(float) = value.cast::<PyFloat>() {
        return Ok(Some(Scalar::Float(float.value())));
    }
    if is_bool_scalar(value)? {
        return Err(PyTypeError::new_err(
            "AudioSamples arithmetic does not take a bool; use an int or a float",
        ));
    }
    // Python ints and NumPy integer scalars. An integer array other than a 0-d one raises here.
    if value.call_method0("__index__").is_ok() {
        return Ok(Some(Scalar::Integer(value.clone())));
    }
    Ok(value.extract::<f64>().ok().map(Scalar::Float))
}

/// A Python bool, a NumPy bool scalar or a 0-d bool array.
fn is_bool_scalar(value: &Bound<'_, PyAny>) -> PyResult<bool> {
    if value.is_instance_of::<PyBool>() {
        return Ok(true);
    }
    let Ok(dtype) = value.getattr("dtype") else {
        return Ok(false);
    };
    Ok(dtype
        .getattr("kind")
        .is_ok_and(|kind| kind.eq("b").unwrap_or(false))
        && value.getattr("ndim")?.extract::<usize>()? == 0)
}

/// The element-wise operators.
#[derive(Clone, Copy)]
enum Arithmetic {
    Add,
    Sub,
    Mul,
    Div,
}

impl Arithmetic {
    /// From the name of a NumPy ufunc.
    fn from_ufunc(name: &str) -> Option<Self> {
        match name {
            "add" => Some(Self::Add),
            "subtract" => Some(Self::Sub),
            "multiply" => Some(Self::Mul),
            "divide" | "true_divide" => Some(Self::Div),
            _ => None,
        }
    }

    /// The name `audio_samples._arith` uses.
    const fn name(self) -> &'static str {
        match self {
            Self::Add => "add",
            Self::Sub => "sub",
            Self::Mul => "mul",
            Self::Div => "div",
        }
    }

    /// As used in the sample rate error.
    const fn verb(self) -> &'static str {
        match self {
            Self::Add => "add",
            Self::Sub => "subtract",
            Self::Mul => "multiply",
            Self::Div => "divide",
        }
    }

    const fn phrase(self) -> &'static str {
        match self {
            Self::Add => "added with",
            Self::Sub => "subtracted by",
            Self::Mul => "multiplied with",
            Self::Div => "divided by",
        }
    }
}

fn check_same_sample_rate(
    py: Python<'_>,
    a: &PyAudioSamples,
    b: &PyAudioSamples,
    op: &str,
) -> PyResult<()> {
    let (sa, sb) = (a.sample_rate(py), b.sample_rate(py));
    if sa == sb {
        return Ok(());
    }
    Err(make_audio_err(
        "AudioParameterError",
        format!(
            "cannot {op} AudioSamples with different sample rates: {sa} Hz and {sb} Hz; \
resample one of them first, e.g. other.resample({sa})"
        ),
        None,
        None,
        PyValueError::new_err,
    ))
}

/// `__name__` of a NumPy ufunc (``"add"``, ``"sin"``, ...); empty if absent.
fn ufunc_name_raw(ufunc: &Bound<'_, PyAny>) -> String {
    ufunc
        .getattr("__name__")
        .and_then(|n| n.extract::<String>())
        .unwrap_or_default()
}

/// Verb phrase naming a NumPy ufunc, for error messages.
fn ufunc_name(ufunc: &Bound<'_, PyAny>) -> String {
    match ufunc_name_raw(ufunc).as_str() {
        "" => "combine".to_string(),
        n => format!("apply np.{n} to"),
    }
}

/// Convert an `AudioSamples` (or a list/tuple containing them) to NumPy for
/// `__array_function__`; other objects are returned unchanged.
fn audio_args_to_numpy<'py>(
    py: Python<'py>,
    obj: Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    if let Ok(audio) = obj.cast::<PyAudioSamples>() {
        return audio.borrow().to_numpy(py);
    }
    if let Ok(list) = obj.cast::<PyList>() {
        if list.iter().any(|x| x.is_instance_of::<PyAudioSamples>()) {
            let items = list
                .iter()
                .map(|x| audio_args_to_numpy(py, x))
                .collect::<PyResult<Vec<_>>>()?;
            return Ok(PyList::new(py, items)?.into_any());
        }
    }
    if let Ok(tuple) = obj.cast::<PyTuple>() {
        if tuple.iter().any(|x| x.is_instance_of::<PyAudioSamples>()) {
            let items = tuple
                .iter()
                .map(|x| audio_args_to_numpy(py, x))
                .collect::<PyResult<Vec<_>>>()?;
            return Ok(PyTuple::new(py, items)?.into_any());
        }
    }
    Ok(obj)
}

/// Warn when a 2-D array looks like it is `(samples, channels)` rather than
/// the `(channels, samples)` layout AudioSamples expects (more "channels"
/// than samples is almost always a transposed soundfile/scipy array).
fn warn_if_samples_first(arr: &Bound<'_, PyUntypedArray>) -> PyResult<()> {
    let shape = arr.shape();
    if shape.len() == 2 && shape[0] > shape[1] {
        let py = arr.py();
        let msg = format!(
            "array of shape {shape:?} is interpreted as {} channels x {} samples; \
AudioSamples expects (channels, samples) - pass arr.T for (samples, channels) data",
            shape[0], shape[1]
        );
        let msg = std::ffi::CString::new(msg)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        // Propagate: under `-W error` the warning must raise.
        PyErr::warn(
            py,
            &py.get_type::<pyo3::exceptions::PyUserWarning>(),
            &msg,
            1,
        )?;
    }
    Ok(())
}

/// Whether a numpy array's `WRITEABLE` flag is set.
fn numpy_is_writeable(ptr: *mut numpy::npyffi::PyArrayObject) -> bool {
    // SAFETY: `ptr` comes from a live, bound numpy array.
    unsafe { (*ptr).flags & numpy::npyffi::NPY_ARRAY_WRITEABLE != 0 }
}

#[derive(Debug)]
enum PyAudioBacking<T: StandardSample + Element> {
    OwnedMono(Array1<T>),
    OwnedMulti(Array2<T>),
    NumpyMono(Py<PyArray1<T>>),
    NumpyMulti(Py<PyArray2<T>>),       // C-order (row-major, planar)
    NumpyInterleaved(Py<PyArray2<T>>), // Fortran-order (column-major, interleaved)
}

impl<T: StandardSample + Element> Clone for PyAudioBacking<T> {
    fn clone(&self) -> Self {
        use PyAudioBacking::*;

        match self {
            OwnedMono(a) => OwnedMono(a.clone()),
            OwnedMulti(a) => OwnedMulti(a.clone()),
            NumpyMono(handle) => {
                // Deep copy: a clone must not share the numpy buffer, or
                // mutating the clone would mutate the original. The memory
                // layout (C / Fortran order) is preserved.
                Python::attach(|py| {
                    let owned = handle.bind(py).to_owned_array();
                    NumpyMono(PyArray1::from_owned_array(py, owned).unbind())
                })
            }
            NumpyMulti(handle) => {
                // Deep copy: a clone must not share the numpy buffer, or
                // mutating the clone would mutate the original. The memory
                // layout (C / Fortran order) is preserved.
                Python::attach(|py| {
                    let owned = handle.bind(py).to_owned_array();
                    NumpyMulti(PyArray2::from_owned_array(py, owned).unbind())
                })
            }
            NumpyInterleaved(handle) => {
                // Deep copy: a clone must not share the numpy buffer, or
                // mutating the clone would mutate the original. The memory
                // layout (C / Fortran order) is preserved.
                Python::attach(|py| {
                    let owned = handle.bind(py).to_owned_array();
                    NumpyInterleaved(PyArray2::from_owned_array(py, owned).unbind())
                })
            }
        }
    }
}

impl<T: StandardSample + Element> PyAudioBacking<T> {
    /// Replace a read-only numpy buffer with a writeable private copy.
    ///
    /// Mutating operations write through the numpy buffer; numpy arrays that
    /// are not writeable (``arr.flags.writeable == False``, or views of
    /// immutable buffers such as ``bytes``) must never be written, so they are
    /// copied first (copy-on-write).
    pub(crate) fn ensure_writeable(&mut self, py: Python<'_>) {
        use PyAudioBacking::*;
        match self {
            OwnedMono(_) | OwnedMulti(_) => {}
            NumpyMono(handle) => {
                let bound = handle.bind(py);
                if !numpy_is_writeable(bound.as_array_ptr()) {
                    let owned = bound.to_owned_array();
                    *handle = PyArray1::from_owned_array(py, owned).unbind();
                }
            }
            NumpyMulti(handle) | NumpyInterleaved(handle) => {
                let bound = handle.bind(py);
                if !numpy_is_writeable(bound.as_array_ptr()) {
                    let owned = bound.to_owned_array();
                    *handle = PyArray2::from_owned_array(py, owned).unbind();
                }
            }
        }
    }

    /// Safe clone that requires a Python token
    /// This is necessary for numpy-backed variants which need GIL access
    pub(crate) fn clone_py(&self, py: Python<'_>) -> Self {
        use PyAudioBacking::*;

        match self {
            OwnedMono(a) => OwnedMono(a.clone()),
            OwnedMulti(a) => OwnedMulti(a.clone()),
            NumpyMono(handle) => {
                // Deep copy (see `Clone`): never share the numpy buffer.
                let owned = handle.bind(py).to_owned_array();
                NumpyMono(PyArray1::from_owned_array(py, owned).unbind())
            }
            NumpyMulti(handle) => {
                // Deep copy (see `Clone`): never share the numpy buffer.
                let owned = handle.bind(py).to_owned_array();
                NumpyMulti(PyArray2::from_owned_array(py, owned).unbind())
            }
            NumpyInterleaved(handle) => {
                // Deep copy (see `Clone`): never share the numpy buffer.
                let owned = handle.bind(py).to_owned_array();
                NumpyInterleaved(PyArray2::from_owned_array(py, owned).unbind())
            }
        }
    }
}

impl<T: StandardSample + Element> Add<Self> for PyAudioBacking<T> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        use PyAudioBacking::*;

        match (self, rhs) {
            (OwnedMono(a), OwnedMono(b)) => OwnedMono(a + b),
            (OwnedMulti(a), OwnedMulti(b)) => OwnedMulti(a + b),
            (NumpyMono(a), NumpyMono(b)) => Python::attach(|py| {
                let a = unsafe { a.bind(py).as_array() };
                let b = unsafe { b.bind(py).as_array() };
                let result = &a + &b;
                NumpyMono(result.into_pyarray(py).unbind())
            }),
            (NumpyMulti(a), NumpyMulti(b)) => Python::attach(|py| {
                let a = unsafe { a.bind(py).as_array() };
                let b = unsafe { b.bind(py).as_array() };
                let result = &a + &b;
                NumpyMulti(result.into_pyarray(py).unbind())
            }),
            (NumpyInterleaved(a), NumpyInterleaved(b)) => Python::attach(|py| {
                let a = unsafe { a.bind(py).as_array() };
                let b = unsafe { b.bind(py).as_array() };
                let result = &a + &b;
                NumpyInterleaved(result.into_pyarray(py).unbind())
            }),
            _ => unreachable!("Addition not supported for mixed backings"),
        }
    }
}

impl<T: StandardSample + Element> Add<T> for PyAudioBacking<T> {
    type Output = Self;

    fn add(self, rhs: T) -> Self::Output {
        use PyAudioBacking::*;

        match self {
            OwnedMono(a) => OwnedMono(a + rhs),
            OwnedMulti(a) => OwnedMulti(a + rhs),
            NumpyMono(handle) => Python::attach(|py| {
                let a = unsafe { handle.bind(py).as_array() };
                let result = &a + rhs;
                NumpyMono(result.into_pyarray(py).unbind())
            }),
            NumpyMulti(handle) => Python::attach(|py| {
                let a = unsafe { handle.bind(py).as_array() };
                let result = &a + rhs;
                NumpyMulti(result.into_pyarray(py).unbind())
            }),
            NumpyInterleaved(handle) => Python::attach(|py| {
                // Convert to planar for operation
                let a = unsafe { handle.bind(py).as_array() };
                let result = &a + rhs;
                NumpyMulti(result.into_pyarray(py).unbind())
            }),
        }
    }
}

impl<T: StandardSample + Element> Sub<Self> for PyAudioBacking<T> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        use PyAudioBacking::*;

        match (self, rhs) {
            (OwnedMono(a), OwnedMono(b)) => OwnedMono(a - b),
            (OwnedMulti(a), OwnedMulti(b)) => OwnedMulti(a - b),
            (NumpyMono(a), NumpyMono(b)) => Python::attach(|py| {
                let a = unsafe { a.bind(py).as_array() };
                let b = unsafe { b.bind(py).as_array() };
                let result = &a - &b;
                NumpyMono(result.into_pyarray(py).unbind())
            }),
            (NumpyMulti(a), NumpyMulti(b)) => Python::attach(|py| {
                let a = unsafe { a.bind(py).as_array() };
                let b = unsafe { b.bind(py).as_array() };
                let result = &a - &b;
                NumpyMulti(result.into_pyarray(py).unbind())
            }),
            (NumpyInterleaved(a), NumpyInterleaved(b)) => Python::attach(|py| {
                let a = unsafe { a.bind(py).as_array() };
                let b = unsafe { b.bind(py).as_array() };
                let result = &a - &b;
                NumpyInterleaved(result.into_pyarray(py).unbind())
            }),
            _ => unreachable!("Subtraction not supported for mixed backings"),
        }
    }
}

impl<T: StandardSample + Element> Sub<T> for PyAudioBacking<T> {
    type Output = Self;

    fn sub(self, rhs: T) -> Self::Output {
        use PyAudioBacking::*;

        match self {
            OwnedMono(a) => OwnedMono(a - rhs),
            OwnedMulti(a) => OwnedMulti(a - rhs),
            NumpyMono(handle) => Python::attach(|py| {
                let a = unsafe { handle.bind(py).as_array() };
                let result = &a - rhs;
                NumpyMono(result.into_pyarray(py).unbind())
            }),
            NumpyMulti(handle) => Python::attach(|py| {
                let a = unsafe { handle.bind(py).as_array() };
                let result = &a - rhs;
                NumpyMulti(result.into_pyarray(py).unbind())
            }),
            NumpyInterleaved(handle) => Python::attach(|py| {
                // Convert to planar for operation
                let a = unsafe { handle.bind(py).as_array() };
                let result = &a - rhs;
                NumpyMulti(result.into_pyarray(py).unbind())
            }),
        }
    }
}

impl<T: StandardSample + Element> Mul<Self> for PyAudioBacking<T> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        use PyAudioBacking::*;

        match (self, rhs) {
            (OwnedMono(a), OwnedMono(b)) => OwnedMono(a * b),
            (OwnedMulti(a), OwnedMulti(b)) => OwnedMulti(a * b),
            (NumpyMono(a), NumpyMono(b)) => Python::attach(|py| {
                let a = unsafe { a.bind(py).as_array() };
                let b = unsafe { b.bind(py).as_array() };
                let result = &a * &b;
                NumpyMono(result.into_pyarray(py).unbind())
            }),
            (NumpyMulti(a), NumpyMulti(b)) => Python::attach(|py| {
                let a = unsafe { a.bind(py).as_array() };
                let b = unsafe { b.bind(py).as_array() };
                let result = &a * &b;
                NumpyMulti(result.into_pyarray(py).unbind())
            }),
            (NumpyInterleaved(a), NumpyInterleaved(b)) => Python::attach(|py| {
                let a = unsafe { a.bind(py).as_array() };
                let b = unsafe { b.bind(py).as_array() };
                let result = &a * &b;
                NumpyInterleaved(result.into_pyarray(py).unbind())
            }),
            _ => unreachable!("Multiplication not supported for mixed backings"),
        }
    }
}

impl<T: StandardSample + Element> Mul<T> for PyAudioBacking<T> {
    type Output = Self;

    fn mul(self, rhs: T) -> Self::Output {
        use PyAudioBacking::*;

        match self {
            OwnedMono(a) => OwnedMono(a * rhs),
            OwnedMulti(a) => OwnedMulti(a * rhs),
            NumpyMono(handle) => Python::attach(|py| {
                let a = unsafe { handle.bind(py).as_array() };
                let result = &a * rhs;
                NumpyMono(result.into_pyarray(py).unbind())
            }),
            NumpyMulti(handle) => Python::attach(|py| {
                let a = unsafe { handle.bind(py).as_array() };
                let result = &a * rhs;
                NumpyMulti(result.into_pyarray(py).unbind())
            }),
            NumpyInterleaved(handle) => Python::attach(|py| {
                // Convert to planar for operation
                let a = unsafe { handle.bind(py).as_array() };
                let result = &a * rhs;
                NumpyMulti(result.into_pyarray(py).unbind())
            }),
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct TypedAudioSamples<T: StandardSample + Element> {
    backing: PyAudioBacking<T>,
    sample_rate: NonZeroU32,
    layout: ChannelLayout,
}

impl<T> TypedAudioSamples<T>
where
    T: StandardSample + Element,
{
    pub fn with_view<R>(
        &self,
        py: Python<'_>,
        f: impl for<'a> FnOnce(AudioSamples<'a, T>) -> R,
    ) -> R {
        use PyAudioBacking::*;
        let sr = self.sample_rate;
        match &self.backing {
            OwnedMono(arr) => {
                let view = arr.view();
                // safety: with_view operates on an instantiated TypedAudioSamples which is non-empty
                let core = AudioSamples::new(
                    unsafe { AudioData::from_borrowed_array1_unchecked(view) },
                    sr,
                );
                f(core)
            }
            OwnedMulti(arr) => {
                let view = arr.view();
                // safety: with_view operates on an instantiated TypedAudioSamples which is non-empty
                let core = AudioSamples::new(
                    unsafe { AudioData::from_borrowed_array2_unchecked(view) },
                    sr,
                );
                f(core)
            }
            NumpyMono(handle) => {
                let bound = handle.bind(py);
                let view = unsafe { bound.as_array() };
                // safety: with_view operates on an instantiated TypedAudioSamples which is non-empty
                let core = AudioSamples::new(
                    unsafe { AudioData::from_borrowed_array1_unchecked(view) },
                    sr,
                );
                f(core)
            }
            NumpyMulti(handle) => {
                let bound = handle.bind(py);
                let view = unsafe { bound.as_array() };
                // safety: with_view operates on an instantiated TypedAudioSamples which is non-empty
                let core = AudioSamples::new(
                    unsafe { AudioData::from_borrowed_array2_unchecked(view) },
                    sr,
                );
                f(core)
            }
            NumpyInterleaved(handle) => {
                // Fortran-layout array: data is interleaved in memory
                // Let ndarray handle the memory layout directly - no conversion needed
                let bound = handle.bind(py);
                let view = unsafe { bound.as_array() };
                // safety: with_view operates on an instantiated TypedAudioSamples which is non-empty
                let core = AudioSamples::new(
                    unsafe { AudioData::from_borrowed_array2_unchecked(view) },
                    sr,
                );
                f(core)
            }
        }
    }

    /// The samples as an `AudioSamples` borrowing this object's data (owned
    /// or NumPy-backed) for as long as `self` is borrowed, without copying.
    /// Like [`Self::with_view`], but the result can be held alongside the
    /// views of other objects (e.g. to stack several sources).
    pub(crate) fn borrowed<'a>(&'a self, py: Python<'a>) -> AudioSamples<'a, T> {
        use PyAudioBacking::*;
        let sr = self.sample_rate;
        // safety: an instantiated TypedAudioSamples is non-empty; a NumPy view
        // stays valid while `self` (which holds the array) is borrowed.
        match &self.backing {
            OwnedMono(arr) => AudioSamples::new(
                unsafe { AudioData::from_borrowed_array1_unchecked(arr.view()) },
                sr,
            ),
            OwnedMulti(arr) => AudioSamples::new(
                unsafe { AudioData::from_borrowed_array2_unchecked(arr.view()) },
                sr,
            ),
            NumpyMono(handle) => AudioSamples::new(
                unsafe { AudioData::from_borrowed_array1_unchecked(handle.bind(py).as_array()) },
                sr,
            ),
            NumpyMulti(handle) | NumpyInterleaved(handle) => AudioSamples::new(
                unsafe { AudioData::from_borrowed_array2_unchecked(handle.bind(py).as_array()) },
                sr,
            ),
        }
    }

    /// [`Self::with_view`] with the GIL released while `f` runs, for long
    /// pure-Rust computations (STFT, CQT, pYIN, HPSS, ...), so other Python
    /// threads keep running.
    ///
    /// Safe for the same reasons as [`Self::with_view_mut_detached`]: owned
    /// data is borrowed through `&self` (the pyclass borrow keeps Rust-side
    /// mutation out), and a NumPy-backed view stays valid because the
    /// `Py<PyArray>` handle keeps the array alive. `f` must not call back into
    /// Python (it is `Send`, so it cannot capture Python objects).
    pub fn with_view_detached<R: Send>(
        &self,
        py: Python<'_>,
        f: impl for<'a> FnOnce(AudioSamples<'a, T>) -> R + Send,
    ) -> R {
        use PyAudioBacking::*;
        let sr = self.sample_rate;
        match &self.backing {
            OwnedMono(arr) => {
                let view = arr.view();
                py.detach(move || {
                    // safety: an instantiated TypedAudioSamples is non-empty
                    f(AudioSamples::new(
                        unsafe { AudioData::from_borrowed_array1_unchecked(view) },
                        sr,
                    ))
                })
            }
            OwnedMulti(arr) => {
                let view = arr.view();
                py.detach(move || {
                    // safety: an instantiated TypedAudioSamples is non-empty
                    f(AudioSamples::new(
                        unsafe { AudioData::from_borrowed_array2_unchecked(view) },
                        sr,
                    ))
                })
            }
            NumpyMono(handle) => {
                let view = unsafe { handle.bind(py).as_array() };
                py.detach(move || {
                    // safety: an instantiated TypedAudioSamples is non-empty
                    f(AudioSamples::new(
                        unsafe { AudioData::from_borrowed_array1_unchecked(view) },
                        sr,
                    ))
                })
            }
            NumpyMulti(handle) | NumpyInterleaved(handle) => {
                let view = unsafe { handle.bind(py).as_array() };
                py.detach(move || {
                    // safety: an instantiated TypedAudioSamples is non-empty
                    f(AudioSamples::new(
                        unsafe { AudioData::from_borrowed_array2_unchecked(view) },
                        sr,
                    ))
                })
            }
        }
    }

    pub fn with_view_mut<R>(
        &mut self,
        py: Python<'_>,
        f: impl for<'a> FnOnce(AudioSamples<'a, T>) -> R,
    ) -> R {
        use PyAudioBacking::*;
        let sr = self.sample_rate;
        self.backing.ensure_writeable(py);

        match &mut self.backing {
            OwnedMono(arr) => {
                let view = arr.view_mut();
                // safety: with_view operates on an instantiated TypedAudioSamples which is non-empty
                let core = AudioSamples::new(
                    unsafe { AudioData::from_borrowed_array1_mut_unchecked(view) },
                    sr,
                );
                f(core)
            }
            OwnedMulti(arr) => {
                let view = arr.view_mut();
                // safety: with_view operates on an instantiated TypedAudioSamples which is non-empty
                let core = AudioSamples::new(
                    unsafe { AudioData::from_borrowed_array2_mut_unchecked(view) },
                    sr,
                );
                f(core)
            }
            NumpyMono(handle) => {
                let bound = handle.bind(py);
                let view = unsafe { bound.as_array_mut() };
                // safety: with_view operates on an instantiated TypedAudioSamples which is non-empty
                let core = AudioSamples::new(
                    unsafe { AudioData::from_borrowed_array1_mut_unchecked(view) },
                    sr,
                );
                f(core)
            }
            NumpyMulti(handle) => {
                let bound = handle.bind(py);
                let view = unsafe { bound.as_array_mut() };
                // safety: with_view operates on an instantiated TypedAudioSamples which is non-empty
                let core = AudioSamples::new(
                    unsafe { AudioData::from_borrowed_array2_mut_unchecked(view) },
                    sr,
                );
                f(core)
            }
            NumpyInterleaved(handle) => {
                // Mutable operations on interleaved arrays using direct ndarray view
                let bound = handle.bind(py);
                let view = unsafe { bound.as_array_mut() };
                // safety: with_view operates on an instantiated TypedAudioSamples which is non-empty
                let core = AudioSamples::new(
                    unsafe { AudioData::from_borrowed_array2_mut_unchecked(view) },
                    sr,
                );
                f(core)
            }
        }
    }

    /// Safe clone that requires a Python token for numpy-backed variants
    pub(crate) fn clone_py(&self, py: Python<'_>) -> Self {
        TypedAudioSamples {
            backing: self.backing.clone_py(py),
            sample_rate: self.sample_rate,
            layout: self.layout,
        }
    }

    /// Execute a mutable operation while releasing the GIL for CPU-intensive work.
    /// This is safe because:
    /// - For Owned variants, we have exclusive ownership
    /// - For Numpy variants, the Py<PyArray> handle keeps the array alive
    ///   and the view remains valid even without the GIL
    #[allow(unused)]
    pub fn with_view_mut_detached<R: Send>(
        &mut self,
        py: Python<'_>,
        f: impl FnOnce(AudioSamples<'_, T>) -> R + Send,
    ) -> R {
        use PyAudioBacking::*;
        let sr = self.sample_rate;
        self.backing.ensure_writeable(py);

        match &mut self.backing {
            OwnedMono(arr) => {
                // For owned data, release GIL and work directly
                py.detach(move || {
                    let view = arr.view_mut();
                    // safety: with_view operates on an instantiated TypedAudioSamples which is non-empty
                    let core = AudioSamples::new(
                        unsafe { AudioData::from_borrowed_array1_mut_unchecked(view) },
                        sr,
                    );
                    f(core)
                })
            }
            OwnedMulti(arr) => py.detach(move || {
                let view = arr.view_mut();
                // safety: with_view operates on an instantiated TypedAudioSamples which is non-empty
                let core = AudioSamples::new(
                    unsafe { AudioData::from_borrowed_array2_mut_unchecked(view) },
                    sr,
                );
                f(core)
            }),
            NumpyMono(handle) => {
                // Get mutable view while holding GIL
                let bound = handle.bind(py);
                let view = unsafe { bound.as_array_mut() };

                // Release GIL for computation
                // Safety: The Py<PyArray> handle keeps the array alive,
                // and ArrayViewMut doesn't require GIL
                py.detach(move || {
                    // safety: with_view operates on an instantiated TypedAudioSamples which is non-empty
                    let core = AudioSamples::new(
                        unsafe { AudioData::from_borrowed_array1_mut_unchecked(view) },
                        sr,
                    );
                    f(core)
                })
            }
            NumpyMulti(handle) => {
                let bound = handle.bind(py);
                let view = unsafe { bound.as_array_mut() };

                py.detach(move || {
                    // safety: with_view operates on an instantiated TypedAudioSamples which is non-empty
                    let core = AudioSamples::new(
                        unsafe { AudioData::from_borrowed_array2_mut_unchecked(view) },
                        sr,
                    );
                    f(core)
                })
            }
            NumpyInterleaved(handle) => {
                // For numpy arrays, we can safely detach the GIL since the Py<PyArray> handle
                // keeps the array alive and the view remains valid even without the GIL
                let bound = handle.bind(py);
                let view = unsafe { bound.as_array_mut() };

                py.detach(move || {
                    // safety: with_view operates on an instantiated TypedAudioSamples which is non-empty
                    let core = AudioSamples::new(
                        unsafe { AudioData::from_borrowed_array2_mut_unchecked(view) },
                        sr,
                    );
                    f(core)
                })
            }
        }
    }
}

impl<T> Add<Self> for TypedAudioSamples<T>
where
    T: StandardSample + Element,
{
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        TypedAudioSamples {
            backing: self.backing + rhs.backing,
            sample_rate: self.sample_rate,
            layout: self.layout,
        }
    }
}

impl<T> Add<T> for TypedAudioSamples<T>
where
    T: StandardSample + Element,
{
    type Output = Self;

    fn add(self, rhs: T) -> Self::Output {
        TypedAudioSamples {
            backing: self.backing + rhs,
            sample_rate: self.sample_rate,
            layout: self.layout,
        }
    }
}

impl<T> Sub<Self> for TypedAudioSamples<T>
where
    T: StandardSample + Element,
{
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        TypedAudioSamples {
            backing: self.backing - rhs.backing,
            sample_rate: self.sample_rate,
            layout: self.layout,
        }
    }
}

impl<T> Sub<T> for TypedAudioSamples<T>
where
    T: StandardSample + Element,
{
    type Output = Self;

    fn sub(self, rhs: T) -> Self::Output {
        TypedAudioSamples {
            backing: self.backing - rhs,
            sample_rate: self.sample_rate,
            layout: self.layout,
        }
    }
}

impl<T> Mul<Self> for TypedAudioSamples<T>
where
    T: StandardSample + Element,
{
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        TypedAudioSamples {
            backing: self.backing * rhs.backing,
            sample_rate: self.sample_rate,
            layout: self.layout,
        }
    }
}

impl<T> Mul<T> for TypedAudioSamples<T>
where
    T: StandardSample + Element,
{
    type Output = Self;

    fn mul(self, rhs: T) -> Self::Output {
        TypedAudioSamples {
            backing: self.backing * rhs,
            sample_rate: self.sample_rate,
            layout: self.layout,
        }
    }
}

#[derive(Debug, Clone)]
#[allow(unused)] // for U8
pub(crate) enum PyAudioDataInner {
    U8(TypedAudioSamples<u8>),
    I16(TypedAudioSamples<i16>),
    I24(TypedAudioSamples<I24>),
    I32(TypedAudioSamples<i32>),
    F32(TypedAudioSamples<f32>),
    F64(TypedAudioSamples<f64>),
}

impl PyAudioDataInner {
    pub fn dtype<'py>(&'py self, py: Python<'py>) -> Bound<'py, PyArrayDescr> {
        match &self {
            PyAudioDataInner::U8(_) => numpy::dtype::<u8>(py),
            PyAudioDataInner::I16(_) => numpy::dtype::<i16>(py),
            PyAudioDataInner::I24(_) => numpy::dtype::<I24>(py),
            PyAudioDataInner::I32(_) => numpy::dtype::<i32>(py),
            PyAudioDataInner::F32(_) => numpy::dtype::<f32>(py),
            PyAudioDataInner::F64(_) => numpy::dtype::<f64>(py),
        }
    }

    /// Safe clone that requires a Python token for numpy-backed variants
    pub(crate) fn clone_py(&self, py: Python<'_>) -> Self {
        match self {
            PyAudioDataInner::U8(a) => PyAudioDataInner::U8(a.clone_py(py)),
            PyAudioDataInner::I16(a) => PyAudioDataInner::I16(a.clone_py(py)),
            PyAudioDataInner::I24(a) => PyAudioDataInner::I24(a.clone_py(py)),
            PyAudioDataInner::I32(a) => PyAudioDataInner::I32(a.clone_py(py)),
            PyAudioDataInner::F32(a) => PyAudioDataInner::F32(a.clone_py(py)),
            PyAudioDataInner::F64(a) => PyAudioDataInner::F64(a.clone_py(py)),
        }
    }
}

impl Display for PyAudioDataInner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Python::attach(|py| match self {
            PyAudioDataInner::U8(a) => a.with_view(py, |audio| audio.fmt(f)),
            PyAudioDataInner::I16(a) => a.with_view(py, |audio| audio.fmt(f)),
            PyAudioDataInner::I24(a) => a.with_view(py, |audio| audio.fmt(f)),
            PyAudioDataInner::I32(a) => a.with_view(py, |audio| audio.fmt(f)),
            PyAudioDataInner::F32(a) => a.with_view(py, |audio| audio.fmt(f)),
            PyAudioDataInner::F64(a) => a.with_view(py, |audio| audio.fmt(f)),
        })
    }
}

impl Add<Self> for PyAudioDataInner {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (PyAudioDataInner::U8(a), PyAudioDataInner::U8(b)) => PyAudioDataInner::U8(a + b),
            (PyAudioDataInner::I16(a), PyAudioDataInner::I16(b)) => PyAudioDataInner::I16(a + b),
            (PyAudioDataInner::I24(a), PyAudioDataInner::I24(b)) => PyAudioDataInner::I24(a + b),
            (PyAudioDataInner::I32(a), PyAudioDataInner::I32(b)) => PyAudioDataInner::I32(a + b),
            (PyAudioDataInner::F32(a), PyAudioDataInner::F32(b)) => PyAudioDataInner::F32(a + b),
            (PyAudioDataInner::F64(a), PyAudioDataInner::F64(b)) => PyAudioDataInner::F64(a + b),
            _ => unreachable!("Addition not supported for different audio data types"),
        }
    }
}

impl<T> Add<T> for PyAudioDataInner
where
    T: StandardSample + Element,
{
    type Output = Self;

    fn add(self, rhs: T) -> Self::Output {
        match self {
            PyAudioDataInner::U8(a) => {
                let val: u8 = rhs.convert_to();
                PyAudioDataInner::U8(a + val)
            }
            PyAudioDataInner::I16(a) => {
                let val: i16 = rhs.convert_to();
                PyAudioDataInner::I16(a + val)
            }
            PyAudioDataInner::I24(a) => {
                let val: I24 = rhs.convert_to();
                PyAudioDataInner::I24(a + val)
            }
            PyAudioDataInner::I32(a) => {
                let val: i32 = rhs.convert_to();
                PyAudioDataInner::I32(a + val)
            }
            PyAudioDataInner::F32(a) => {
                let val: f32 = rhs.convert_to();
                PyAudioDataInner::F32(a + val)
            }
            PyAudioDataInner::F64(a) => {
                let val: f64 = rhs.convert_to();
                PyAudioDataInner::F64(a + val)
            }
        }
    }
}

impl Sub<Self> for PyAudioDataInner {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (PyAudioDataInner::U8(a), PyAudioDataInner::U8(b)) => PyAudioDataInner::U8(a - b),
            (PyAudioDataInner::I16(a), PyAudioDataInner::I16(b)) => PyAudioDataInner::I16(a - b),
            (PyAudioDataInner::I24(a), PyAudioDataInner::I24(b)) => PyAudioDataInner::I24(a - b),
            (PyAudioDataInner::I32(a), PyAudioDataInner::I32(b)) => PyAudioDataInner::I32(a - b),
            (PyAudioDataInner::F32(a), PyAudioDataInner::F32(b)) => PyAudioDataInner::F32(a - b),
            (PyAudioDataInner::F64(a), PyAudioDataInner::F64(b)) => PyAudioDataInner::F64(a - b),
            _ => unreachable!("Subtraction not supported for different audio data types"),
        }
    }
}

impl<T> Sub<T> for PyAudioDataInner
where
    T: StandardSample + Element,
{
    type Output = Self;

    fn sub(self, rhs: T) -> Self::Output {
        match self {
            PyAudioDataInner::U8(a) => {
                let val: u8 = rhs.convert_to();
                PyAudioDataInner::U8(a - val)
            }
            PyAudioDataInner::I16(a) => {
                let val: i16 = rhs.convert_to();
                PyAudioDataInner::I16(a - val)
            }
            PyAudioDataInner::I24(a) => {
                let val: I24 = rhs.convert_to();
                PyAudioDataInner::I24(a - val)
            }
            PyAudioDataInner::I32(a) => {
                let val: i32 = rhs.convert_to();
                PyAudioDataInner::I32(a - val)
            }
            PyAudioDataInner::F32(a) => {
                let val: f32 = rhs.convert_to();
                PyAudioDataInner::F32(a - val)
            }
            PyAudioDataInner::F64(a) => {
                let val: f64 = rhs.convert_to();
                PyAudioDataInner::F64(a - val)
            }
        }
    }
}

impl Mul<Self> for PyAudioDataInner {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (PyAudioDataInner::U8(a), PyAudioDataInner::U8(b)) => PyAudioDataInner::U8(a * b),
            (PyAudioDataInner::I16(a), PyAudioDataInner::I16(b)) => PyAudioDataInner::I16(a * b),
            (PyAudioDataInner::I24(a), PyAudioDataInner::I24(b)) => PyAudioDataInner::I24(a * b),
            (PyAudioDataInner::I32(a), PyAudioDataInner::I32(b)) => PyAudioDataInner::I32(a * b),
            (PyAudioDataInner::F32(a), PyAudioDataInner::F32(b)) => PyAudioDataInner::F32(a * b),
            (PyAudioDataInner::F64(a), PyAudioDataInner::F64(b)) => PyAudioDataInner::F64(a * b),
            _ => unreachable!("Multiplication not supported for mixed types"),
        }
    }
}

impl<T> Mul<T> for PyAudioDataInner
where
    T: StandardSample + Element,
{
    type Output = Self;

    fn mul(self, rhs: T) -> Self::Output {
        match self {
            PyAudioDataInner::U8(a) => {
                let val: u8 = rhs.convert_to();
                PyAudioDataInner::U8(a * val)
            }
            PyAudioDataInner::I16(a) => {
                let val: i16 = rhs.convert_to();
                PyAudioDataInner::I16(a * val)
            }
            PyAudioDataInner::I24(a) => {
                let val: I24 = rhs.convert_to();
                PyAudioDataInner::I24(a * val)
            }
            PyAudioDataInner::I32(a) => {
                let val: i32 = rhs.convert_to();
                PyAudioDataInner::I32(a * val)
            }
            PyAudioDataInner::F32(a) => {
                let val: f32 = rhs.convert_to();
                PyAudioDataInner::F32(a * val)
            }
            PyAudioDataInner::F64(a) => {
                let val: f64 = rhs.convert_to();
                PyAudioDataInner::F64(a * val)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    #[test]
    fn test_clone_py_requires_gil() {
        Python::initialize();
        Python::attach(|py| {
            let data = Array1::<f32>::zeros(100);
            let audio = PyAudioSamples::new_mono(data, unsafe { NonZeroU32::new_unchecked(44100) });

            // Should be able to clone with GIL
            let _cloned = audio.clone_py(py);
        });
    }

    #[test]
    fn test_typed_audio_samples_clone_py() {
        Python::initialize();
        Python::attach(|py| {
            let data = Array1::<f64>::ones(50);
            let typed = TypedAudioSamples {
                backing: PyAudioBacking::OwnedMono(data),
                sample_rate: unsafe { NonZeroU32::new_unchecked(48000) },
                layout: ChannelLayout::NonInterleaved,
            };

            let _cloned = typed.clone_py(py);
        });
    }

    #[test]
    fn test_audio_backing_clone_py() {
        Python::initialize();
        Python::attach(|py| {
            let data = Array1::<i16>::from_elem(200, 100);
            let backing = PyAudioBacking::OwnedMono(data);

            let cloned = backing.clone_py(py);

            // Verify the clone is independent
            match (backing, cloned) {
                (PyAudioBacking::OwnedMono(ref a), PyAudioBacking::OwnedMono(ref b)) => {
                    assert_eq!(a.len(), b.len());
                }
                _ => unreachable!("Unexpected backing type"),
            }
        });
    }

    #[test]
    fn test_with_view_mut_detached_releases_gil() {
        Python::initialize();
        Python::attach(|py| {
            let data = Array1::<f32>::zeros(1000);
            let mut typed = TypedAudioSamples {
                backing: PyAudioBacking::OwnedMono(data),
                sample_rate: unsafe { NonZeroU32::new_unchecked(44100) },
                layout: ChannelLayout::NonInterleaved,
            };

            // This should release GIL during the operation
            let result = typed.with_view_mut_detached(py, |audio| {
                // Simulate CPU-intensive work
                let _ = audio_samples::AudioProcessing::scale(&audio, 2.0);
                42
            });
            assert_eq!(result, 42);
        });
    }
}
