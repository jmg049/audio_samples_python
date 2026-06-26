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

pub mod io;
pub mod io_streaming;
pub mod operations;
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
use pyo3::types::{PyAnyMethods, PyDict, PyInt, PyList, PyTuple};
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

pub(crate) use dispatch_with_view;
pub(crate) use dispatch_with_view_mut;

/// Convert an audio_samples error to a PyErr with appropriate exception type.
fn audio_err_to_py(e: audio_samples::AudioSampleError) -> PyErr {
    use audio_samples::AudioSampleError;
    use pyo3::exceptions::{PyRuntimeError, PyValueError};

    match e {
        AudioSampleError::Parameter(ref _pe) => PyValueError::new_err(e.to_string()),
        AudioSampleError::Layout(ref _le) => PyValueError::new_err(e.to_string()),
        AudioSampleError::Conversion(ref _ce) => PyTypeError::new_err(e.to_string()),
        AudioSampleError::Processing(ref _pe) => PyRuntimeError::new_err(e.to_string()),
        AudioSampleError::Feature(ref _fe) => {
            PyRuntimeError::new_err(format!("{e} (enable the required cargo feature)"))
        }
        AudioSampleError::Spectrogram(ref _se) => {
            PyRuntimeError::new_err(PyRuntimeError::new_err(e.to_string()))
        }
        // Catch-all for any future variants (serialization, etc.)
        #[allow(unreachable_patterns)]
        _ => PyRuntimeError::new_err(e.to_string()),
    }
}

fn audio_io_err_to_py(e: AudioIOError) -> PyErr {
    use pyo3::exceptions::{PyIOError, PyRuntimeError, PyValueError};

    match e {
        AudioIOError::Io(io_err) => PyIOError::new_err(io_err.to_string()),
        AudioIOError::AudioSamples(as_err) => audio_err_to_py(as_err),
        AudioIOError::CorruptedData {
            description,
            details,
            position: _,
        } => PyValueError::new_err(format!("Corrupted data: {} - {}", description, details)),
        AudioIOError::WavError(wav_err) => {
            PyRuntimeError::new_err(format!("WAV error: {}", wav_err))
        }
        AudioIOError::SeekError(msg) => PyIOError::new_err(msg),
        AudioIOError::EndOfStream(msg) => PyIOError::new_err(msg),
        AudioIOError::MissingFeature(msg) => {
            PyRuntimeError::new_err(format!("Missing feature: {}", msg))
        }
        AudioIOError::UnsupportedFormat(msg) => {
            PyValueError::new_err(format!("Unsupported format: {}", msg))
        }
        #[allow(unreachable_patterns)]
        _ => PyRuntimeError::new_err(e.to_string()),
    }
}

#[pymodule(name = "audio_samples")]
fn audio_samples_python(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    types::types(py, &m)?;
    m.add_class::<PyAudioSamples>()?;

    io::io(py, &m)?;
    utils::utils(py, &m)?;

    // Re-export spectrograms module
    let spectrograms_mod = PyModule::new(py, "spectrograms")?;
    spectrograms::python::register_module(py, &spectrograms_mod)?;
    m.add_submodule(&spectrograms_mod)?;

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
        "stereo_silence"
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
#[pyclass(name = "AudioSamples", unsendable, module = "audio_samples")]
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
    /// with Fortran (column-major) layout, which naturally matches WAV interleaved format.
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

        // Dispatch based on the dtype
        match &sources_vec[0].inner {
            PyAudioDataInner::U8(_) => {
                let audio_sources: Vec<_> = sources_vec
                    .iter()
                    .map(|src| {
                        if let PyAudioDataInner::U8(typed) = &src.inner {
                            typed.with_view(py, |audio| audio.clone().into_owned())
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
                            typed.with_view(py, |audio| audio.clone().into_owned())
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
                            typed.with_view(py, |audio| audio.clone().into_owned())
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
                            typed.with_view(py, |audio| audio.clone().into_owned())
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
                            typed.with_view(py, |audio| audio.clone().into_owned())
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
                            typed.with_view(py, |audio| audio.clone().into_owned())
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
#[pyclass(name = "AudioSamplesInfo")]
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
    ///     >>> import audio_python as aus
    ///     >>> audio = aus.AudioSamples.from_array(np.random.randn(2, 44100).astype(np.float32), sample_rate=44100)
    ///     >>> info = audio.info()
    ///
    #[pyo3(signature = (), text_signature = "(self) -> AudioSamplesInfo")]
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
    ///     AudioSamples: A new audio samples object with zero-copy numpy integration.
    ///
    /// Raises:
    ///     TypeError: If array dtype is not supported or dimensions are invalid.
    ///
    /// Examples:
    ///     >>> import numpy as np
    ///     >>> import audio_python as aus
    ///     >>> # Create mono audio
    ///     >>> mono = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 44100)).astype(np.float32)
    ///     >>> audio = aus.AudioSamples.from_array(mono, sample_rate=44100)
    ///     >>> # Create stereo audio
    ///     >>> stereo = np.random.randn(2, 44100).astype(np.float32)
    ///     >>> audio = aus.AudioSamples.from_array(stereo, sample_rate=44100)
    #[staticmethod]
    #[pyo3(name = "from_array", signature = (arr: "numpy.typing.ArrayLike", sample_rate: "float"), text_signature = "($cls, arr: numpy.typing.ArrayLike, sample_rate: int) -> AudioSamples")]
    fn from_array(arr: Bound<'_, PyAny>, sample_rate: NonZeroU32) -> PyResult<Self> {
        // Get the array as PyUntypedArray to inspect its properties
        let untyped_array: &Bound<'_, PyUntypedArray> = arr.cast()?;

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
    #[pyo3(name = "_new_mono_i16_from_np", signature = (arr: "numpy.typing.ArrayLike", sample_rate: "float"), text_signature = "($cls, arr: numpy.typing.ArrayLike, sample_rate: int) -> AudioSamples")]
    fn new_mono_i16_from_np(arr: Bound<'_, PyArray1<i16>>, sample_rate: NonZeroU32) -> Self {
        Self::new_mono_from_python::<i16>(arr, sample_rate)
    }

    #[staticmethod]
    #[pyo3(name = "_new_multi_i16_from_np", signature = (arr: "numpy.typing.ArrayLike", sample_rate: "float"), text_signature = "($cls, arr: numpy.typing.ArrayLike, sample_rate: int) -> AudioSamples")]
    fn new_multi_i16_from_np(arr: Bound<'_, PyArray2<i16>>, sample_rate: NonZeroU32) -> Self {
        Self::new_multi_from_python::<i16>(arr, sample_rate)
    }

    #[staticmethod]
    #[pyo3(name = "_new_mono_i24_from_np", signature = (arr: "numpy.typing.ArrayLike", sample_rate: "float"), text_signature = "($cls, arr: numpy.typing.ArrayLike, sample_rate: int) -> AudioSamples")]
    fn new_mono_i24_from_np(arr: Bound<'_, PyArray1<I24>>, sample_rate: NonZeroU32) -> Self {
        Self::new_mono_from_python::<I24>(arr, sample_rate)
    }

    #[staticmethod]
    #[pyo3(name = "_new_multi_i24_from_np", signature = (arr: "numpy.typing.ArrayLike", sample_rate: "float"), text_signature = "($cls, arr: numpy.typing.ArrayLike, sample_rate: int) -> AudioSamples")]
    fn new_multi_i24_from_np(arr: Bound<'_, PyArray2<I24>>, sample_rate: NonZeroU32) -> Self {
        Self::new_multi_from_python::<I24>(arr, sample_rate)
    }

    #[staticmethod]
    #[pyo3(name = "_new_mono_i32_from_np", signature = (arr: "numpy.typing.ArrayLike", sample_rate: "float"), text_signature = "($cls, arr: numpy.typing.ArrayLike, sample_rate: int) -> AudioSamples")]
    fn new_mono_i32_from_np(arr: Bound<'_, PyArray1<i32>>, sample_rate: NonZeroU32) -> Self {
        Self::new_mono_from_python::<i32>(arr, sample_rate)
    }

    #[staticmethod]
    #[pyo3(name = "_new_multi_i32_from_np", signature = (arr: "numpy.typing.ArrayLike", sample_rate: "float"), text_signature = "($cls, arr: numpy.typing.ArrayLike, sample_rate: int) -> AudioSamples")]
    fn new_multi_i32_from_np(arr: Bound<'_, PyArray2<i32>>, sample_rate: NonZeroU32) -> Self {
        Self::new_multi_from_python::<i32>(arr, sample_rate)
    }

    #[staticmethod]
    #[pyo3(name = "_new_mono_f32_from_np", signature = (arr: "numpy.typing.ArrayLike", sample_rate: "float"), text_signature = "($cls, arr: numpy.typing.ArrayLike, sample_rate: int) -> AudioSamples")]
    fn new_mono_f32_from_np(arr: Bound<'_, PyArray1<f32>>, sample_rate: NonZeroU32) -> Self {
        Self::new_mono_from_python::<f32>(arr, sample_rate)
    }

    #[staticmethod]
    #[pyo3(name = "_new_multi_f32_from_np", signature = (arr: "numpy.typing.ArrayLike", sample_rate: "float"), text_signature = "($cls, arr: numpy.typing.ArrayLike, sample_rate: int) -> AudioSamples")]
    fn new_multi_f32_from_np(arr: Bound<'_, PyArray2<f32>>, sample_rate: NonZeroU32) -> Self {
        Self::new_multi_from_python::<f32>(arr, sample_rate)
    }
    #[staticmethod]
    #[pyo3(name = "_new_mono_f64_from_np", signature = (arr: "numpy.typing.ArrayLike", sample_rate: "float"), text_signature = "($cls, arr: numpy.typing.ArrayLike, sample_rate: int) -> AudioSamples")]
    fn new_mono_f64_from_np(arr: Bound<'_, PyArray1<f64>>, sample_rate: NonZeroU32) -> Self {
        Self::new_mono_from_python::<f64>(arr, sample_rate)
    }
    #[staticmethod]
    #[pyo3(name = "_new_multi_f64_from_np", signature = (arr: "numpy.typing.ArrayLike", sample_rate: "float"), text_signature = "($cls, arr: numpy.typing.ArrayLike, sample_rate: int) -> AudioSamples")]
    fn new_multi_f64_from_np(arr: Bound<'_, PyArray2<f64>>, sample_rate: NonZeroU32) -> Self {
        Self::new_multi_from_python::<f64>(arr, sample_rate)
    }

    #[staticmethod]
    #[pyo3(name = "new_mono", signature = (arr: "numpy.typing.ArrayLike", sample_rate: "float"), text_signature = "($cls, arr: numpy.typing.ArrayLike, sample_rate: int) -> AudioSamples")]
    fn py_new_mono(
        py: Python<'_>,
        arr: Bound<'_, PyUntypedArray>,
        sample_rate: NonZeroU32,
    ) -> PyResult<Self> {
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

    #[staticmethod]
    #[pyo3(name = "new_multi", signature = (arr: "numpy.typing.ArrayLike", sample_rate: "float"), text_signature = "($cls, arr: numpy.typing.ArrayLike, sample_rate: int) -> AudioSamples")]
    fn py_new_multi(
        py: Python<'_>,
        arr: Bound<'_, PyUntypedArray>,
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

    #[getter]
    fn len(&self, py: Python<'_>) -> usize {
        dispatch_with_view!(self, py, |audio| audio.len().get())
    }

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
    #[pyo3(signature = (), text_signature = "($self) -> int")]
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

    #[pyo3(signature = (), text_signature = "($self) -> bool")]
    fn is_mono(&self, py: Python<'_>) -> bool {
        dispatch_with_view!(self, py, |audio| audio.is_mono())
    }

    #[pyo3(signature = (), text_signature = "($self) -> bool")]
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

    #[getter]
    fn duration_milliseconds(&self, py: Python<'_>) -> f64 {
        self.duration_seconds(py) * 1000.0
    }

    #[getter(ndim)]
    fn ndim(&self, py: Python<'_>) -> usize {
        if self.is_mono(py) { 1 } else { 2 }
    }

    #[pyo3(signature = (), text_signature = "($self) -> AudioSamples")]
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

    fn __str__(&self) -> String {
        format!("{self}")
    }

    fn __repr__(&self) -> String {
        format!("{self:#}")
    }

    fn __add__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(other_audio) = other.extract::<PyRef<PyAudioSamples>>() {
            // Check shape compatibility before adding to avoid panic
            if self.len(py) != other_audio.len(py) {
                return Err(PyValueError::new_err(format!(
                    "AudioSamples shapes are incompatible: {} vs {}",
                    self.len(py),
                    other_audio.len(py)
                )));
            }
            Ok(self + &*other_audio)
        } else if let Ok(scalar) = other.extract::<f64>() {
            // Adding a scalar
            Ok(self.clone_py(py) + scalar)
        } else if let Ok(numpy_array) = other.cast::<PyUntypedArray>() {
            // Adding a numpy array - create AudioSamples from it and add
            self.add_numpy_array(py, &numpy_array)
        } else {
            Err(PyTypeError::new_err(
                "AudioSamples can only be added with another AudioSamples, scalar, or numpy array",
            ))
        }
    }

    fn __mul__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(other_audio) = other.extract::<PyRef<PyAudioSamples>>() {
            // Multiplying by another AudioSamples (element-wise)
            // Check shape compatibility before multiplying to avoid panic
            if self.len(py) != other_audio.len(py) {
                return Err(PyValueError::new_err(format!(
                    "AudioSamples shapes are incompatible: {} vs {}",
                    self.len(py),
                    other_audio.len(py)
                )));
            }
            Ok(self * &*other_audio)
        } else if let Ok(factor) = other.extract::<f64>() {
            // Multiplying by a scalar
            let mut c = self.clone_py(py);
            c.scale(py, factor);
            Ok(c)
        } else if let Ok(numpy_array) = other.cast::<PyUntypedArray>() {
            // Multiplying by a numpy array (element-wise)
            self.mul_numpy_array(py, &numpy_array)
        } else {
            Err(PyTypeError::new_err(
                "AudioSamples can only be multiplied with another AudioSamples, a scalar, or numpy array",
            ))
        }
    }

    fn __truediv__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(divisor) = other.extract::<f64>() {
            // Dividing by a scalar
            if divisor == 0.0 {
                return Err(pyo3::exceptions::PyZeroDivisionError::new_err(
                    "division by zero",
                ));
            }
            let mut c = self.clone_py(py);
            c.scale(py, 1.0 / divisor);
            Ok(c)
        } else if let Ok(numpy_array) = other.cast::<PyUntypedArray>() {
            // Dividing by a numpy array (element-wise)
            self.div_numpy_array(py, &numpy_array)
        } else {
            Err(PyTypeError::new_err(
                "AudioSamples can only be divided by a scalar or numpy array",
            ))
        }
    }

    fn __sub__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(other_audio) = other.extract::<PyRef<PyAudioSamples>>() {
            // Check shape compatibility before subtracting to avoid panic
            if self.len(py) != other_audio.len(py) {
                return Err(PyValueError::new_err(format!(
                    "AudioSamples shapes are incompatible: {} vs {}",
                    self.len(py),
                    other_audio.len(py)
                )));
            }
            Ok(self - &*other_audio)
        } else if let Ok(scalar) = other.extract::<f64>() {
            // Subtracting a scalar
            Ok(self.clone_py(py) - scalar)
        } else if let Ok(numpy_array) = other.cast::<PyUntypedArray>() {
            // Subtracting a numpy array
            self.sub_numpy_array(py, &numpy_array)
        } else {
            Err(PyTypeError::new_err(
                "AudioSamples can only subtract another AudioSamples, scalar, or numpy array",
            ))
        }
    }

    fn __pow__(&self, py: Python<'_>, exponent: f64, modulo: Option<f64>) -> Self {
        dispatch_with_view!(self, py, |audio| {
            let powered = audio
                .powf(exponent, modulo.map(|m| m.convert_to()))
                .into_owned();
            PyAudioSamples::from_audio_samples(powered)
        })
    }

    // Reverse operations for scalar operands on the left
    fn __radd__(&self, py: Python<'_>, scalar: f64) -> PyResult<Self> {
        // scalar + audio: create a numpy array of the same shape with the scalar value
        let shape = if self.channels(py) == 1 {
            vec![self.len(py)]
        } else {
            vec![self.channels(py), self.len(py) / self.channels(py)]
        };

        // Create a numpy array filled with the scalar value
        let numpy_module = PyModule::import(py, "numpy")?;
        let full_func = numpy_module.getattr("full")?;
        let scalar_array = full_func.call((shape, scalar), None)?;
        let numpy_array = scalar_array.cast::<PyUntypedArray>()?;

        self.add_numpy_array(py, &numpy_array)
    }

    fn __rmul__(&self, py: Python<'_>, scalar: f64) -> PyResult<Self> {
        // scalar * audio is the same as audio * scalar
        let scalar_bound = scalar.into_pyobject(py)?;
        self.__mul__(py, &scalar_bound)
    }

    fn __rsub__(&self, py: Python<'_>, scalar: f64) -> PyResult<Self> {
        // scalar - audio = -(audio - scalar) = -audio + scalar
        let neg_one = (-1.0f64).into_pyobject(py)?;
        let negated = self.__mul__(py, &neg_one)?;
        let scalar_bound = scalar.into_pyobject(py)?;
        negated.__add__(py, &scalar_bound)
    }

    fn __rtruediv__(&self, _py: Python<'_>, _scalar: f64) -> PyResult<Self> {
        Err(PyTypeError::new_err(
            "scalar / AudioSamples is not implemented; use AudioSamples / scalar instead",
        ))
    }

    // In-place operations
    fn __iadd__(&mut self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<()> {
        if let Ok(other_audio) = other.extract::<PyRef<PyAudioSamples>>() {
            // For simplicity, only support same-type operations for now
            let result = self.clone_py(py) + other_audio.clone_py(py);
            *self = result;
        } else if let Ok(scalar) = other.extract::<f64>() {
            // Adding a scalar
            let result = self.clone_py(py) + scalar;
            *self = result;
        } else if let Ok(numpy_array) = other.cast::<PyUntypedArray>() {
            // Adding a numpy array
            let result = self.add_numpy_array(py, &numpy_array)?;
            *self = result;
        } else {
            return Err(PyTypeError::new_err(
                "AudioSamples can only be added with another AudioSamples, numpy array, or a numeric scalar",
            ));
        }
        Ok(())
    }

    fn __isub__(&mut self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<()> {
        if let Ok(other_audio) = other.extract::<PyRef<PyAudioSamples>>() {
            // For simplicity, only support same-type operations for now
            let result = self.clone_py(py) - other_audio.clone_py(py);
            *self = result;
        } else if let Ok(scalar) = other.extract::<f64>() {
            // Subtracting a scalar
            let result = self.clone_py(py) - scalar;
            *self = result;
        } else if let Ok(numpy_array) = other.cast::<PyUntypedArray>() {
            // Subtracting a numpy array
            let result = self.sub_numpy_array(py, &numpy_array)?;
            *self = result;
        } else {
            return Err(PyTypeError::new_err(
                "AudioSamples can only subtract another AudioSamples, numpy array, or a numeric scalar",
            ));
        }
        Ok(())
    }

    fn __imul__(&mut self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<()> {
        if let Ok(other_audio) = other.extract::<PyRef<PyAudioSamples>>() {
            // For simplicity, only support same-type operations for now
            let result = self.clone_py(py) * other_audio.clone_py(py);
            *self = result;
        } else if let Ok(scalar) = other.extract::<f64>() {
            // Multiplying by a scalar
            self.scale(py, scalar);
        } else if let Ok(numpy_array) = other.cast::<PyUntypedArray>() {
            // Multiplying by a numpy array
            let result = self.mul_numpy_array(py, &numpy_array)?;
            *self = result;
        } else {
            return Err(PyTypeError::new_err(
                "AudioSamples can only be multiplied with another AudioSamples, a numeric scalar, or numpy array",
            ));
        }
        Ok(())
    }

    fn __itruediv__(&mut self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<()> {
        if let Ok(scalar) = other.extract::<f64>() {
            // Dividing by a scalar
            if scalar == 0.0 {
                return Err(pyo3::exceptions::PyZeroDivisionError::new_err(
                    "division by zero",
                ));
            }
            self.scale(py, 1.0 / scalar);
        } else if let Ok(numpy_array) = other.cast::<PyUntypedArray>() {
            // Dividing by a numpy array
            let result = self.div_numpy_array(py, &numpy_array)?;
            *self = result;
        } else {
            return Err(PyTypeError::new_err(
                "AudioSamples can only be divided by a numeric scalar or numpy array",
            ));
        }
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
        // Perform element-wise addition with the numpy array
        let self_numpy = self.to_numpy(py)?;
        let result_numpy = self_numpy.add(numpy_array)?;
        let result_array = result_numpy.as_any().cast::<PyUntypedArray>()?;

        // Create new AudioSamples from the result with same metadata
        let sample_rate = self.sample_rate(py);
        if self.channels(py) == 1 {
            Self::py_new_mono(py, result_array.clone(), sample_rate)
        } else {
            Self::py_new_multi(py, result_array.clone(), sample_rate)
        }
    }

    fn sub_numpy_array(
        &self,
        py: Python<'_>,
        numpy_array: &Bound<'_, PyUntypedArray>,
    ) -> PyResult<Self> {
        // Perform element-wise subtraction with the numpy array
        let self_numpy = self.to_numpy(py)?;
        let result_numpy = self_numpy.sub(numpy_array)?;
        let result_array = result_numpy.as_any().cast::<PyUntypedArray>()?;

        // Create new AudioSamples from the result with same metadata
        let sample_rate = self.sample_rate(py);
        if self.channels(py) == 1 {
            Self::py_new_mono(py, result_array.clone(), sample_rate)
        } else {
            Self::py_new_multi(py, result_array.clone(), sample_rate)
        }
    }

    fn mul_numpy_array(
        &self,
        py: Python<'_>,
        numpy_array: &Bound<'_, PyUntypedArray>,
    ) -> PyResult<Self> {
        // Perform element-wise multiplication with the numpy array
        let self_numpy = self.to_numpy(py)?;
        let result_numpy = self_numpy.mul(numpy_array)?;
        let result_array = result_numpy.as_any().cast::<PyUntypedArray>()?;

        // Create new AudioSamples from the result with same metadata
        let sample_rate = self.sample_rate(py);
        if self.channels(py) == 1 {
            Self::py_new_mono(py, result_array.clone(), sample_rate)
        } else {
            Self::py_new_multi(py, result_array.clone(), sample_rate)
        }
    }

    fn div_numpy_array(
        &self,
        py: Python<'_>,
        numpy_array: &Bound<'_, PyUntypedArray>,
    ) -> PyResult<Self> {
        // Perform element-wise division with the numpy array
        let self_numpy = self.to_numpy(py)?;
        let result_numpy = self_numpy.div(numpy_array)?;
        let result_array = result_numpy.as_any().cast::<PyUntypedArray>()?;

        // Create new AudioSamples from the result with same metadata
        let sample_rate = self.sample_rate(py);
        if self.channels(py) == 1 {
            Self::py_new_mono(py, result_array.clone(), sample_rate)
        } else {
            Self::py_new_multi(py, result_array.clone(), sample_rate)
        }
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

    fn __array_ufunc__(
        &self,
        py: Python<'_>,
        ufunc: &Bound<'_, PyAny>,
        method: &str,
        inputs: &Bound<'_, PyAny>,
        _kwargs: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        // Handle numpy ufuncs by delegating to our arithmetic methods

        // Only handle '__call__' method for now
        if method != "__call__" {
            return Ok(py.NotImplemented());
        }

        // Get the ufunc name
        let ufunc_name = ufunc.getattr("__name__")?.extract::<String>()?;

        // Convert inputs to Python list to extract individual items
        let inputs_list: Vec<Bound<'_, PyAny>> = inputs.extract()?;

        if inputs_list.len() == 2 {
            // Find the numpy array and determine position
            let (numpy_array, is_audio_first) =
                if let Ok(arr) = inputs_list[0].cast::<PyUntypedArray>() {
                    // First input is numpy array, second should be AudioSamples
                    if inputs_list[1].is_instance_of::<PyAudioSamples>() {
                        (arr, false)
                    } else {
                        return Ok(py.NotImplemented());
                    }
                } else if let Ok(arr) = inputs_list[1].cast::<PyUntypedArray>() {
                    // Second input is numpy array, first should be AudioSamples
                    if inputs_list[0].is_instance_of::<PyAudioSamples>() {
                        (arr, true)
                    } else {
                        return Ok(py.NotImplemented());
                    }
                } else {
                    return Ok(py.NotImplemented());
                };

            // Handle specific operations
            let result = match ufunc_name.as_str() {
                "add" | "multiply" => {
                    // Commutative operations
                    match ufunc_name.as_str() {
                        "add" => self.add_numpy_array(py, &numpy_array),
                        "multiply" => self.mul_numpy_array(py, &numpy_array),
                        _ => unreachable!(),
                    }
                }
                "subtract" => {
                    // Only support audio - numpy
                    if is_audio_first {
                        self.sub_numpy_array(py, &numpy_array)
                    } else {
                        return Ok(py.NotImplemented());
                    }
                }
                "divide" | "true_divide" => {
                    // Only support audio / numpy
                    if is_audio_first {
                        self.div_numpy_array(py, &numpy_array)
                    } else {
                        return Ok(py.NotImplemented());
                    }
                }
                _ => return Ok(py.NotImplemented()),
            }?;

            Ok(result.into_bound_py_any(py)?.into())
        } else {
            // Only handle binary operations for now
            Ok(py.NotImplemented())
        }
    }

    fn __array_function__(
        &self,
        py: Python<'_>,
        _func: &Bound<'_, PyAny>,
        _types: &Bound<'_, PyAny>,
        _args: &Bound<'_, PyAny>,
        _kwargs: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        // For now, delegate most array functions to numpy
        // This allows functions like np.mean(audio), np.std(audio), etc. to work
        Ok(py.NotImplemented())
    }

    #[staticmethod]
    #[pyo3(name = "zeros_mono", signature = (length: "int", sample_rate: "int"), text_signature = "(length: int, sample_rate: int) -> AudioSamples")]
    fn py_zeros_mono_f32(length: usize, sample_rate: NonZeroU32) -> Self {
        let data = Array1::<f32>::zeros(length);
        Self::new_mono(data, sample_rate)
    }

    #[staticmethod]
    #[pyo3(name = "zeros_mono_i16", signature = (length: "int", sample_rate: "int"), text_signature = "(length: int, sample_rate: int) -> AudioSamples")]
    fn py_zeros_mono_i16(length: usize, sample_rate: NonZeroU32) -> Self {
        let data = Array1::<i16>::zeros(length);
        Self::new_mono(data, sample_rate)
    }

    #[staticmethod]
    #[pyo3(name = "zeros_mono_i32", signature = (length: "int", sample_rate: "int"), text_signature = "(length: int, sample_rate: int) -> AudioSamples")]
    fn py_zeros_mono_i32(length: usize, sample_rate: NonZeroU32) -> Self {
        let data = Array1::<i32>::zeros(length);
        Self::new_mono(data, sample_rate)
    }

    #[staticmethod]
    #[pyo3(name = "zeros_mono_f64", signature = (length: "int", sample_rate: "int"), text_signature = "(length: int, sample_rate: int) -> AudioSamples")]
    fn py_zeros_mono_f64(length: usize, sample_rate: NonZeroU32) -> Self {
        let data = Array1::<f64>::zeros(length);
        Self::new_mono(data, sample_rate)
    }

    #[staticmethod]
    #[pyo3(name = "ones_mono", signature = (length: "int", sample_rate: "int"), text_signature = "(length: int, sample_rate: int) -> AudioSamples")]
    fn py_ones_mono_f32(length: usize, sample_rate: NonZeroU32) -> Self {
        let data = Array1::<f32>::ones(length);
        Self::new_mono(data, sample_rate)
    }

    #[staticmethod]
    #[pyo3(name = "ones_mono_i16", signature = (length: "int", sample_rate: "int"), text_signature = "(length: int, sample_rate: int) -> AudioSamples")]
    fn py_ones_mono_i16(length: usize, sample_rate: NonZeroU32) -> Self {
        let data = Array1::<i16>::ones(length);
        Self::new_mono(data, sample_rate)
    }

    #[staticmethod]
    #[pyo3(name = "ones_mono_i32", signature = (length: "int", sample_rate: "int"), text_signature = "(length: int, sample_rate: int) -> AudioSamples")]
    fn py_ones_mono_i32(length: usize, sample_rate: NonZeroU32) -> Self {
        let data = Array1::<i32>::ones(length);
        Self::new_mono(data, sample_rate)
    }

    #[staticmethod]
    #[pyo3(name = "ones_mono_f64", signature = (length: "int", sample_rate: "int"), text_signature = "(length: int, sample_rate: int) -> AudioSamples")]
    fn py_ones_mono_f64(length: usize, sample_rate: NonZeroU32) -> Self {
        let data = Array1::<f64>::ones(length);
        Self::new_mono(data, sample_rate)
    }

    #[staticmethod]
    #[pyo3(name = "uniform_mono", signature = (length: "int", sample_rate: "int", value: "float"), text_signature = "(length: int, sample_rate: int, value: float) -> AudioSamples")]
    fn py_uniform_mono_f32(length: usize, sample_rate: NonZeroU32, value: f32) -> Self {
        let data = Array1::<f32>::from_elem(length, value);
        Self::new_mono(data, sample_rate)
    }

    #[staticmethod]
    #[pyo3(name = "uniform_mono_i16", signature = (length: "int", sample_rate: "int", value: "int"), text_signature = "(length: int, sample_rate: int, value: int) -> AudioSamples")]
    fn py_uniform_mono_i16(length: usize, sample_rate: NonZeroU32, value: i16) -> Self {
        let data = Array1::<i16>::from_elem(length, value);
        Self::new_mono(data, sample_rate)
    }

    #[staticmethod]
    #[pyo3(name = "uniform_mono_i32", signature = (length: "int", sample_rate: "int", value: "int"), text_signature = "(length: int, sample_rate: int, value: int) -> AudioSamples")]
    fn py_uniform_mono_i32(length: usize, sample_rate: NonZeroU32, value: i32) -> Self {
        let data = Array1::<i32>::from_elem(length, value);
        Self::new_mono(data, sample_rate)
    }

    #[staticmethod]
    #[pyo3(name = "uniform_mono_f64", signature = (length: "int", sample_rate: "int", value: "float"), text_signature = "(length: int, sample_rate: int, value: float) -> AudioSamples")]
    fn py_uniform_mono_f64(length: usize, sample_rate: NonZeroU32, value: f64) -> Self {
        let data = Array1::<f64>::from_elem(length, value);
        Self::new_mono(data, sample_rate)
    }

    #[staticmethod]
    #[pyo3(name = "zeros_multi", signature = (channels: "int", length: "int", sample_rate: "int"), text_signature = "(channels: int, length: int, sample_rate: int) -> AudioSamples")]
    fn py_zeros_multi_f32(channels: usize, length: usize, sample_rate: NonZeroU32) -> Self {
        let data = Array2::<f32>::zeros((channels, length));
        Self::new_multi(data, sample_rate)
    }

    #[staticmethod]
    #[pyo3(name = "zeros_multi_i16", signature = (channels: "int", length: "int", sample_rate: "int"), text_signature = "(channels: int, length: int, sample_rate: int) -> AudioSamples")]
    fn py_zeros_multi_i16(channels: usize, length: usize, sample_rate: NonZeroU32) -> Self {
        let data = Array2::<i16>::zeros((channels, length));
        Self::new_multi(data, sample_rate)
    }

    #[staticmethod]
    #[pyo3(name = "zeros_multi_i32", signature = (channels: "int", length: "int", sample_rate: "int"), text_signature = "(channels: int, length: int, sample_rate: int) -> AudioSamples")]
    fn py_zeros_multi_i32(channels: usize, length: usize, sample_rate: NonZeroU32) -> Self {
        let data = Array2::<i32>::zeros((channels, length));
        Self::new_multi(data, sample_rate)
    }

    #[staticmethod]
    #[pyo3(name = "zeros_multi_f64", signature = (channels: "int", length: "int", sample_rate: "int"), text_signature = "(channels: int, length: int, sample_rate: int) -> AudioSamples")]
    fn py_zeros_multi_f64(channels: usize, length: usize, sample_rate: NonZeroU32) -> Self {
        let data = Array2::<f64>::zeros((channels, length));
        Self::new_multi(data, sample_rate)
    }

    #[staticmethod]
    #[pyo3(name = "ones_multi", signature = (channels: "int", length: "int", sample_rate: "int"), text_signature = "(channels: int, length: int, sample_rate: int) -> AudioSamples")]
    fn py_ones_multi_f32(channels: usize, length: usize, sample_rate: NonZeroU32) -> Self {
        let data = Array2::<f32>::ones((channels, length));
        Self::new_multi(data, sample_rate)
    }

    #[staticmethod]
    #[pyo3(name = "ones_multi_i16", signature = (channels: "int", length: "int", sample_rate: "int"), text_signature = "(channels: int, length: int, sample_rate: int) -> AudioSamples")]
    fn py_ones_multi_i16(channels: usize, length: usize, sample_rate: NonZeroU32) -> Self {
        let data = Array2::<i16>::ones((channels, length));
        Self::new_multi(data, sample_rate)
    }

    #[staticmethod]
    #[pyo3(name = "ones_multi_i32", signature = (channels: "int", length: "int", sample_rate: "int"), text_signature = "(channels: int, length: int, sample_rate: int) -> AudioSamples")]
    fn py_ones_multi_i32(channels: usize, length: usize, sample_rate: NonZeroU32) -> Self {
        let data = Array2::<i32>::ones((channels, length));
        Self::new_multi(data, sample_rate)
    }

    #[staticmethod]
    #[pyo3(name = "ones_multi_f64", signature = (channels: "int", length: "int", sample_rate: "int"), text_signature = "(channels: int, length: int, sample_rate: int) -> AudioSamples")]
    fn py_ones_multi_f64(channels: usize, length: usize, sample_rate: NonZeroU32) -> Self {
        let data = Array2::<f64>::ones((channels, length));
        Self::new_multi(data, sample_rate)
    }

    #[staticmethod]
    #[pyo3(name = "uniform_multi", signature = (channels: "int", length: "int", sample_rate: "int", value: "float"), text_signature = "(channels: int, length: int, sample_rate: int, value: float) -> AudioSamples")]
    fn py_uniform_multi_f32(
        channels: usize,
        length: usize,
        sample_rate: NonZeroU32,
        value: f32,
    ) -> Self {
        let data = Array2::<f32>::from_elem((channels, length), value);
        Self::new_multi(data, sample_rate)
    }

    #[staticmethod]
    #[pyo3(name = "uniform_multi_i16", signature = (channels: "int", length: "int", sample_rate: "int", value: "int"), text_signature = "(channels: int, length: int, sample_rate: int, value: int) -> AudioSamples")]
    fn py_uniform_multi_i16(
        channels: usize,
        length: usize,
        sample_rate: NonZeroU32,
        value: i16,
    ) -> Self {
        let data = Array2::<i16>::from_elem((channels, length), value);
        Self::new_multi(data, sample_rate)
    }

    #[staticmethod]
    #[pyo3(name = "uniform_multi_i32", signature = (channels: "int", length: "int", sample_rate: "int", value: "int"), text_signature = "(channels: int, length: int, sample_rate: int, value: int) -> AudioSamples")]
    fn py_uniform_multi_i32(
        channels: usize,
        length: usize,
        sample_rate: NonZeroU32,
        value: i32,
    ) -> Self {
        let data = Array2::<i32>::from_elem((channels, length), value);
        Self::new_multi(data, sample_rate)
    }

    #[staticmethod]
    #[pyo3(name = "uniform_multi_f64", signature = (channels: "int", length: "int", sample_rate: "int", value: "float"), text_signature = "(channels: int, length: int, sample_rate: int, value: float) -> AudioSamples")]
    fn py_uniform_multi_f64(
        channels: usize,
        length: usize,
        sample_rate: NonZeroU32,
        value: f64,
    ) -> Self {
        let data = Array2::<f64>::from_elem((channels, length), value);
        Self::new_multi(data, sample_rate)
    }

    #[pyo3(name = "nyquist", signature=(), text_signature="($self) -> f64")]
    /// Calculates the nyquist frequency of the signal
    fn py_nyquist(&self, py: Python<'_>) -> f64 {
        dispatch_with_view!(self, py, |audio| { audio.nyquist() })
    }

    #[pyo3(signature = (dtype), text_signature = "($self, dtype: numpy.dtype) -> AudioSamples")]
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

    #[pyo3(signature = (), text_signature = "($self) -> AudioSamples")]
    fn as_u8(&self, py: Python<'_>) -> PyAudioSamples {
        self.to_format(py, PySampleType::U8)
    }

    #[pyo3(signature = (), text_signature = "($self) -> AudioSamples")]
    fn as_i16(&self, py: Python<'_>) -> PyAudioSamples {
        self.to_format(py, PySampleType::I16)
    }

    #[pyo3(signature = (), text_signature = "($self) -> AudioSamples")]
    fn as_i32(&self, py: Python<'_>) -> PyAudioSamples {
        self.to_format(py, PySampleType::I32)
    }

    #[pyo3(signature = (), text_signature = "($self) -> AudioSamples")]
    fn as_f32(&self, py: Python<'_>) -> PyAudioSamples {
        self.to_format(py, PySampleType::F32)
    }

    #[pyo3(signature = (), text_signature = "($self) -> AudioSamples")]
    fn as_f64(&self, py: Python<'_>) -> PyAudioSamples {
        self.to_format(py, PySampleType::F64)
    }

    #[pyo3(signature = (dtype), text_signature = "($self, dtype: numpy.dtype) -> AudioSamples")]
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

    #[pyo3(signature = (), text_signature = "($self) -> AudioSamples")]
    fn cast_as_u8(&self, py: Python<'_>) -> PyAudioSamples {
        self.cast_as(py, PySampleType::U8)
    }

    #[pyo3(signature = (), text_signature = "($self) -> AudioSamples")]
    fn cast_as_i16(&self, py: Python<'_>) -> PyAudioSamples {
        self.cast_as(py, PySampleType::I16)
    }

    #[pyo3(signature = (), text_signature = "($self) -> AudioSamples")]
    fn cast_as_i32(&self, py: Python<'_>) -> PyAudioSamples {
        self.cast_as(py, PySampleType::I32)
    }

    #[pyo3(signature = (), text_signature = "($self) -> AudioSamples")]
    fn cast_as_f32(&self, py: Python<'_>) -> PyAudioSamples {
        self.cast_as(py, PySampleType::F32)
    }

    #[pyo3(signature = (), text_signature = "($self) -> AudioSamples")]
    fn cast_as_f64(&self, py: Python<'_>) -> PyAudioSamples {
        self.cast_as(py, PySampleType::F64)
    }

    fn __array__(
        &self,
        py: Python<'_>,
        dtype: Option<&Bound<'_, PyArrayDescr>>,
    ) -> PyResult<Py<PyAny>> {
        match dtype {
            Some(dt) if !dt.is_equiv_to(&self.dtype(py)) => {
                // Cast to requested dtype first
                let casted = self.cast_as(py, PySampleType::from_numpy(py, dt)?);
                casted.__array__(py, None)
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
                // I24 doesn't have a direct numpy equivalent, so we'll expose as int32
                let shape = self.shape(py);
                dict.set_item("shape", shape)?;
                dict.set_item("typestr", "<i4")?;
                // For I24, we need to convert to get the data pointer
                let as_i32 = self.cast_as_i32(py);
                Ok(as_i32.__array_interface__(py)?)
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
    #[pyo3(signature = (device: "Optional[str]"), text_signature = "($self, device: Optional[str] = None) -> torch.Tensor")]
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
    #[pyo3(signature = (gpu_id: "Optional[str]" = None), text_signature = "($self, gpu_id: Optional[str] = None) -> torch.Tensor")]
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
    #[pyo3(signature = (), text_signature = "($self) -> numpy.ndarray")]
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
                // Safe: Use with_gil to ensure GIL is held when cloning numpy handle
                Python::attach(|py| {
                    let bound = handle.bind(py);
                    let cloned = bound.clone();
                    NumpyMono(cloned.unbind())
                })
            }
            NumpyMulti(handle) => {
                // Safe: Use with_gil to ensure GIL is held when cloning numpy handle
                Python::attach(|py| {
                    let bound = handle.bind(py);
                    let cloned = bound.clone();
                    NumpyMulti(cloned.unbind())
                })
            }
            NumpyInterleaved(handle) => {
                // Safe: Use with_gil to ensure GIL is held when cloning numpy handle
                Python::attach(|py| {
                    let bound = handle.bind(py);
                    let cloned = bound.clone();
                    NumpyInterleaved(cloned.unbind())
                })
            }
        }
    }
}

impl<T: StandardSample + Element> PyAudioBacking<T> {
    /// Safe clone that requires a Python token
    /// This is necessary for numpy-backed variants which need GIL access
    pub(crate) fn clone_py(&self, py: Python<'_>) -> Self {
        use PyAudioBacking::*;

        match self {
            OwnedMono(a) => OwnedMono(a.clone()),
            OwnedMulti(a) => OwnedMulti(a.clone()),
            NumpyMono(handle) => {
                let bound = handle.bind(py);
                let cloned = bound.clone();
                NumpyMono(cloned.unbind())
            }
            NumpyMulti(handle) => {
                let bound = handle.bind(py);
                let cloned = bound.clone();
                NumpyMulti(cloned.unbind())
            }
            NumpyInterleaved(handle) => {
                let bound = handle.bind(py);
                let cloned = bound.clone();
                NumpyInterleaved(cloned.unbind())
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
                let core = AudioSamples::new(unsafe { AudioData::from_borrowed_array1_unchecked(view) }, sr);
                f(core)
            }
            OwnedMulti(arr) => {
                let view = arr.view();
                // safety: with_view operates on an instantiated TypedAudioSamples which is non-empty
                let core = AudioSamples::new(unsafe { AudioData::from_borrowed_array2_unchecked(view) }, sr);
                f(core)
            }
            NumpyMono(handle) => {
                let bound = handle.bind(py);
                let view = unsafe { bound.as_array() };
                // safety: with_view operates on an instantiated TypedAudioSamples which is non-empty
                let core = AudioSamples::new(unsafe { AudioData::from_borrowed_array1_unchecked(view) }, sr);
                f(core)
            }
            NumpyMulti(handle) => {
                let bound = handle.bind(py);
                let view = unsafe { bound.as_array() };
                // safety: with_view operates on an instantiated TypedAudioSamples which is non-empty
                let core = AudioSamples::new(unsafe { AudioData::from_borrowed_array2_unchecked(view) }, sr);
                f(core)
            }
            NumpyInterleaved(handle) => {
                // Fortran-layout array: data is interleaved in memory
                // Let ndarray handle the memory layout directly - no conversion needed
                let bound = handle.bind(py);
                let view = unsafe { bound.as_array() };
                // safety: with_view operates on an instantiated TypedAudioSamples which is non-empty
                let core = AudioSamples::new(unsafe { AudioData::from_borrowed_array2_unchecked(view) }, sr);
                f(core)
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

        match &mut self.backing {
            OwnedMono(arr) => {
                let view = arr.view_mut();
                // safety: with_view operates on an instantiated TypedAudioSamples which is non-empty
                let core = AudioSamples::new(unsafe { AudioData::from_borrowed_array1_mut_unchecked(view) }, sr);
                f(core)
            }
            OwnedMulti(arr) => {
                let view = arr.view_mut();
                // safety: with_view operates on an instantiated TypedAudioSamples which is non-empty
                let core = AudioSamples::new(unsafe { AudioData::from_borrowed_array2_mut_unchecked(view) }, sr);
                f(core)
            }
            NumpyMono(handle) => {
                let bound = handle.bind(py);
                let view = unsafe { bound.as_array_mut() };
                // safety: with_view operates on an instantiated TypedAudioSamples which is non-empty
                let core = AudioSamples::new(unsafe { AudioData::from_borrowed_array1_mut_unchecked(view) }, sr);
                f(core)
            }
            NumpyMulti(handle) => {
                let bound = handle.bind(py);
                let view = unsafe { bound.as_array_mut() };
                // safety: with_view operates on an instantiated TypedAudioSamples which is non-empty
                let core = AudioSamples::new(unsafe { AudioData::from_borrowed_array2_mut_unchecked(view) }, sr);
                f(core)
            }
            NumpyInterleaved(handle) => {
                // Mutable operations on interleaved arrays using direct ndarray view
                let bound = handle.bind(py);
                let view = unsafe { bound.as_array_mut() };
                // safety: with_view operates on an instantiated TypedAudioSamples which is non-empty
                let core = AudioSamples::new(unsafe { AudioData::from_borrowed_array2_mut_unchecked(view) }, sr);
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

        match &mut self.backing {
            OwnedMono(arr) => {
                // For owned data, release GIL and work directly
                py.detach(move || {
                    let view = arr.view_mut();
                    // safety: with_view operates on an instantiated TypedAudioSamples which is non-empty
                    let core = AudioSamples::new(unsafe { AudioData::from_borrowed_array1_mut_unchecked(view) }, sr);
                    f(core)
                })
            }
            OwnedMulti(arr) => py.detach(move || {
                let view = arr.view_mut();
                // safety: with_view operates on an instantiated TypedAudioSamples which is non-empty
                let core = AudioSamples::new(unsafe { AudioData::from_borrowed_array2_mut_unchecked(view) }, sr);
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
                    let core = AudioSamples::new(unsafe { AudioData::from_borrowed_array1_mut_unchecked(view) }, sr);
                    f(core)
                })
            }
            NumpyMulti(handle) => {
                let bound = handle.bind(py);
                let view = unsafe { bound.as_array_mut() };

                py.detach(move || {
                    // safety: with_view operates on an instantiated TypedAudioSamples which is non-empty
                    let core = AudioSamples::new(unsafe { AudioData::from_borrowed_array2_mut_unchecked(view) }, sr);
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
                    let core = AudioSamples::new(unsafe { AudioData::from_borrowed_array2_mut_unchecked(view) }, sr);
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
        Python::attach(|py| {
            let data = Array1::<f32>::zeros(100);
            let audio = PyAudioSamples::new_mono(data, unsafe { NonZeroU32::new_unchecked(44100) });

            // Should be able to clone with GIL
            let _cloned = audio.clone_py(py);
        });
    }

    #[test]
    fn test_typed_audio_samples_clone_py() {
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
                let _ = audio_samples::AudioProcessing::scale(audio, 2.0);
                42
            });
            assert_eq!(result, 42);
        });
    }
}
