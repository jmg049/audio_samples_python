#![warn(clippy::unit_cmp)] // Detects comparing unit types
#![warn(clippy::match_same_arms)]
// Duplicate match arms
// #![warn(clippy::unreachable)] // Detects unreachable code

// Performance-focused
#![warn(clippy::inefficient_to_string)] // `format!("{}", x)` vs `x.to_string()`
#![warn(clippy::map_clone)] // Cloning inside `map()` unnecessarily
#![warn(clippy::unnecessary_to_owned)] // Detects redundant `.to_owned()` or `.clone()`
#![warn(clippy::large_stack_arrays)] // Helps avoid stack overflows
#![warn(clippy::box_collection)] // Warns on boxed `Vec`, `String`, etc.
#![warn(clippy::vec_box)] // Avoids using `Vec<Box<T>>` when unnecessary
#![warn(clippy::needless_collect)] // Avoids `.collect().iter()` chains

// Style and idiomatic Rust
#![warn(clippy::redundant_clone)] // Detects unnecessary `.clone()`
#![warn(clippy::identity_op)] // e.g., `x + 0`, `x * 1`
#![warn(clippy::needless_return)] // Avoids `return` at the end of functions
#![warn(clippy::let_unit_value)] // Avoids binding `()` to variables
#![warn(clippy::manual_map)] // Use `.map()` instead of manual `match`
#![warn(clippy::unwrap_used)] // Avoids using `unwrap()`

// Maintainability
#![warn(clippy::missing_panics_doc)] // Docs for functions that might panic
#![warn(clippy::missing_safety_doc)] // Docs for `unsafe` functions
#![warn(clippy::missing_const_for_fn)] // Suggests making eligible functions `const`
#![allow(clippy::too_many_arguments)]
// Allow functions with many parameters (very few and far between)

pub mod io;

use pyo3::types::{PyAnyMethods, PyDict, PyInt, PyList, PyTuple};
use std::num::NonZeroU32;
use std::ops::{Add, Sub};
use std::time::Duration;
use std::{any::TypeId, fmt::Display};

use audio_samples::operations::types::{
    CompressorConfig, DynamicRangeMethod, EqBand, EqBandType, FilterResponse, IirFilterDesign,
    IirFilterType, KneeType, LimiterConfig, ParametricEq, SideChainConfig,
};
use audio_samples::operations::{
    AudioDynamicRange, AudioIirFiltering, AudioParametricEq, AudioTransforms,
};
use audio_samples::operations::{
    MonoConversionMethod, NormalizationMethod, StereoConversionMethod,
};
use audio_samples::{
    AudioChannelOps, AudioEditing, AudioProcessing, AudioStatistics, AudioTypeConversion, ConvertTo,
};
use audio_samples::{AudioData, AudioSample, AudioSamples, ChannelLayout, I24};
use numpy::ndarray::{Array1, Array2};
use numpy::{Complex64, IntoPyArray, PyArray};
// Complex is re-exported from audio_samples operations::traits
use numpy::{
    Element, PyArray1, PyArray2, PyArrayDescr, PyArrayDescrMethods, PyArrayMethods, PyUntypedArray,
    PyUntypedArrayMethods,
};
use pyo3::IntoPyObject;
use pyo3::{exceptions::PyTypeError, prelude::*};

use crate::io::audio_io_module;

// =============================================================================
// DISPATCH MACROS
// =============================================================================
// These macros reduce boilerplate for the dtype-dispatch pattern where we need
// to match on PyAudioDataInner variants and call the same operation on each.

/// Dispatch to all dtype variants with a read-only view.
/// Usage: `dispatch_with_view!(self.inner, py, |audio| audio.some_method())`
macro_rules! dispatch_with_view {
    ($inner:expr, $py:expr, |$audio:ident| $body:expr) => {
        match &$inner {
            PyAudioDataInner::I16(a) => a.with_view($py, |$audio| $body),
            PyAudioDataInner::I24(a) => a.with_view($py, |$audio| $body),
            PyAudioDataInner::I32(a) => a.with_view($py, |$audio| $body),
            PyAudioDataInner::F32(a) => a.with_view($py, |$audio| $body),
            PyAudioDataInner::F64(a) => a.with_view($py, |$audio| $body),
        }
    };
}

/// Dispatch to all dtype variants with a mutable view.
/// Usage: `dispatch_with_view_mut!(self.inner, py, |mut audio| audio.mutating_method())`
#[allow(unused_macros)]
macro_rules! dispatch_with_view_mut {
    ($inner:expr, $py:expr, |mut $audio:ident| $body:expr) => {
        match &mut $inner {
            PyAudioDataInner::I16(a) => a.with_view_mut($py, |mut $audio| $body),
            PyAudioDataInner::I24(a) => a.with_view_mut($py, |mut $audio| $body),
            PyAudioDataInner::I32(a) => a.with_view_mut($py, |mut $audio| $body),
            PyAudioDataInner::F32(a) => a.with_view_mut($py, |mut $audio| $body),
            PyAudioDataInner::F64(a) => a.with_view_mut($py, |mut $audio| $body),
        }
    };
}

/// Dispatch for operations that need type-specific handling (e.g., casting a float argument).
/// The closure receives (typed_inner, audio_view) where typed_inner gives access to the sample type.
/// Usage: `dispatch_typed!(self.inner, py, |_typed, audio| { ... })`
#[allow(unused_macros)]
macro_rules! dispatch_typed {
    ($inner:expr, $py:expr, |$typed:ident, $audio:ident| $body:expr) => {
        match &$inner {
            PyAudioDataInner::I16($typed) => $typed.with_view($py, |$audio| $body),
            PyAudioDataInner::I24($typed) => $typed.with_view($py, |$audio| $body),
            PyAudioDataInner::I32($typed) => $typed.with_view($py, |$audio| $body),
            PyAudioDataInner::F32($typed) => $typed.with_view($py, |$audio| $body),
            PyAudioDataInner::F64($typed) => $typed.with_view($py, |$audio| $body),
        }
    };
}

/// Dispatch for mutable operations that need type-specific handling.
#[allow(unused_macros)]
macro_rules! dispatch_typed_mut {
    ($inner:expr, $py:expr, |$typed:ident, mut $audio:ident| $body:expr) => {
        match &mut $inner {
            PyAudioDataInner::I16($typed) => $typed.with_view_mut($py, |mut $audio| $body),
            PyAudioDataInner::I24($typed) => $typed.with_view_mut($py, |mut $audio| $body),
            PyAudioDataInner::I32($typed) => $typed.with_view_mut($py, |mut $audio| $body),
            PyAudioDataInner::F32($typed) => $typed.with_view_mut($py, |mut $audio| $body),
            PyAudioDataInner::F64($typed) => $typed.with_view_mut($py, |mut $audio| $body),
        }
    };
}

/// Dispatch to all dtype variants with a read-only view, mapping AudioSampleError to PyErr.
/// Use for fallible operations that return Result<T, AudioSampleError>.
/// Usage: `dispatch_with_view_result!(self.inner, py, |audio| audio.fallible_method())`
macro_rules! dispatch_with_view_result {
    ($inner:expr, $py:expr, |$audio:ident| $body:expr) => {
        match &$inner {
            PyAudioDataInner::I16(a) => a.with_view($py, |$audio| $body.map_err(audio_err_to_py)),
            PyAudioDataInner::I24(a) => a.with_view($py, |$audio| $body.map_err(audio_err_to_py)),
            PyAudioDataInner::I32(a) => a.with_view($py, |$audio| $body.map_err(audio_err_to_py)),
            PyAudioDataInner::F32(a) => a.with_view($py, |$audio| $body.map_err(audio_err_to_py)),
            PyAudioDataInner::F64(a) => a.with_view($py, |$audio| $body.map_err(audio_err_to_py)),
        }
    };
}

/// Dispatch to all dtype variants with a mutable view, mapping AudioSampleError to PyErr.
/// Use for fallible mutable operations that return Result<T, AudioSampleError>.
macro_rules! dispatch_with_view_mut_result {
    ($inner:expr, $py:expr, |mut $audio:ident| $body:expr) => {
        match &mut $inner {
            PyAudioDataInner::I16(a) => {
                a.with_view_mut($py, |mut $audio| $body.map_err(audio_err_to_py))
            }
            PyAudioDataInner::I24(a) => {
                a.with_view_mut($py, |mut $audio| $body.map_err(audio_err_to_py))
            }
            PyAudioDataInner::I32(a) => {
                a.with_view_mut($py, |mut $audio| $body.map_err(audio_err_to_py))
            }
            PyAudioDataInner::F32(a) => {
                a.with_view_mut($py, |mut $audio| $body.map_err(audio_err_to_py))
            }
            PyAudioDataInner::F64(a) => {
                a.with_view_mut($py, |mut $audio| $body.map_err(audio_err_to_py))
            }
        }
    };
}

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
            PyRuntimeError::new_err(format!("{} (enable the required cargo feature)", e))
        }
        // Catch-all for any future variants (plotting, serialization, etc.)
        #[allow(unreachable_patterns)]
        _ => PyRuntimeError::new_err(e.to_string()),
    }
}

// =============================================================================

#[pymodule]
#[pyo3(name = "audio_samples")]
fn audio_python(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyAudioSamples>()?;
    m.add_class::<PyIirFilterDesign>()?;
    m.add_class::<PyParametricEq>()?;
    m.add_class::<PyEqBand>()?;

    let gen_module = PyModule::new(py, "generation")?;

    gen_module.add_function(wrap_pyfunction!(sine_wave, &gen_module)?)?;
    gen_module.add_function(wrap_pyfunction!(cosine_wave, &gen_module)?)?;
    gen_module.add_function(wrap_pyfunction!(square_wave, &gen_module)?)?;
    gen_module.add_function(wrap_pyfunction!(sawtooth_wave, &gen_module)?)?;
    gen_module.add_function(wrap_pyfunction!(triangle_wave, &gen_module)?)?;
    gen_module.add_function(wrap_pyfunction!(chirp, &gen_module)?)?;
    gen_module.add_function(wrap_pyfunction!(silence, &gen_module)?)?;
    gen_module.add_function(wrap_pyfunction!(impulse, &gen_module)?)?;
    gen_module.add_function(wrap_pyfunction!(white_noise, &gen_module)?)?;
    gen_module.add_function(wrap_pyfunction!(pink_noise, &gen_module)?)?;
    gen_module.add_function(wrap_pyfunction!(brown_noise, &gen_module)?)?;

    m.add_submodule(&gen_module)?;
    m.add_submodule(&audio_io_module(py)?)?;
    Ok(())
}

#[pyclass(name = "AudioSamples")]
#[derive(Clone)]
pub struct PyAudioSamples {
    inner: PyAudioDataInner,
}

impl PyAudioSamples {
    /// Creates a new mono audio samples container
    ///
    /// # Panics
    ///
    /// Panics if the sample type is not supported (i.e., not one of i16, I24, i32, f32, f64)
    pub fn new_mono<T: AudioSample + Element>(arr: Array1<T>, sample_rate: u32) -> Self {
        let backing = PyAudioBacking::OwnedMono(arr);

        match TypeId::of::<T>() {
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
            _ => panic!("Unsupported audio sample type"),
        }
    }

    /// Creates a new multi-channel audio samples container
    ///
    /// # Panics
    ///
    /// Panics if the sample type is not supported (i.e., not one of i16, I24, i32, f32, f64)
    pub fn new_multi<T: AudioSample + Element>(arr: Array2<T>, sample_rate: u32) -> Self {
        let backing = PyAudioBacking::OwnedMulti(arr);
        match TypeId::of::<T>() {
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
            _ => panic!("Unsupported audio sample type"),
        }
    }

    /// Creates a new mono audio samples container from a Python numpy array
    ///
    /// # Panics
    ///
    /// Panics if the sample type is not supported (i.e., not one of i16, I24, i32, f32, f64)
    pub fn new_mono_from_python<T: AudioSample + Element>(
        arr: Bound<'_, PyArray1<T>>,
        sample_rate: u32,
    ) -> Self {
        let backing = PyAudioBacking::NumpyMono(arr.into());
        match TypeId::of::<T>() {
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
            _ => panic!("Unsupported audio sample type"),
        }
    }

    /// Creates a new PyAudioSamples instance from a numpy 2D array (multi-channel audio)
    ///
    /// # Panics
    ///
    /// Panics if the sample type is not supported (i.e., not one of i16, I24, i32, f32, f64)
    pub fn new_multi_from_python<T: AudioSample + Element>(
        arr: Bound<'_, PyArray2<T>>,
        sample_rate: u32,
    ) -> Self {
        let backing = PyAudioBacking::NumpyMulti(arr.into());
        match TypeId::of::<T>() {
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
            _ => panic!("Unsupported audio sample type"),
        }
    }

    /// Returns the TypeId of the audio sample type contained within PyAudioSamples
    pub const fn type_of(&self) -> TypeId {
        match &self.inner {
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
}

impl PyAudioSamples {
    /// Helper method for __array_interface__ to set data pointer and strides
    fn set_array_interface_data<T: AudioSample + Element>(
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
                let strides = vec![
                    (view.strides()[0] as isize * std::mem::size_of::<T>() as isize) as usize,
                    (view.strides()[1] as isize * std::mem::size_of::<T>() as isize) as usize,
                ];
                dict.set_item("strides", strides)?;
            }
        }
        dict.set_item("version", 3)?;
        Ok(dict.clone().unbind())
    }

    /// Helper function to convert AudioSamples to PyAudioSamples efficiently using ownership transfer
    fn from_audio_samples<T: AudioSample + Element>(
        audio_samples: AudioSamples<'static, T>,
    ) -> PyResult<Self> {
        let sample_rate = audio_samples.sample_rate().get();
        match audio_samples.is_mono() {
            true => {
                let array = audio_samples.into_array1().expect("Safe since the None variant is only returned if data is not mono, which we have checked");
                Ok(Self::new_mono(array, sample_rate))
            }
            false => {
                let array = audio_samples.into_array2().expect("Safe since the None variant is only returned if data is not mono, which we have checked");
                Ok(Self::new_multi(array, sample_rate))
            }
        }
    }

    /// Access the inner data for crate-internal use (e.g., io module)
    pub(crate) fn inner(&self) -> &PyAudioDataInner {
        &self.inner
    }

    // Internal dtype method for Rust-side use (not exposed to Python)
    pub fn dtype<'py>(&'py self, py: Python<'py>) -> Bound<'py, PyArrayDescr> {
        self.inner.dtype(py)
    }
}

#[pymethods]
impl PyAudioSamples {
    /// Returns the dtype as a string for Python property access
    #[getter(dtype)]
    fn dtype_py(&self) -> &str {
        match &self.inner {
            PyAudioDataInner::I16(_) => "i16",
            PyAudioDataInner::I24(_) => "I24",
            PyAudioDataInner::I32(_) => "i32",
            PyAudioDataInner::F32(_) => "f32",
            PyAudioDataInner::F64(_) => "f64",
        }
    }
    #[staticmethod]
    #[pyo3(name = "from_array", signature = (arr, sample_rate), text_signature = "($cls, arr: numpy.ndarray, sample_rate: int) -> AudioSamples")]
    fn from_array(arr: Bound<'_, PyAny>, sample_rate: u32) -> PyResult<Self> {
        // Get the array as PyUntypedArray to inspect its properties
        let untyped_array: &Bound<'_, PyUntypedArray> = arr.cast()?;

        // Get array info
        let dtype = untyped_array.dtype();
        let ndim = untyped_array.ndim();

        // Handle based on dtype and dimensions
        // Use dtype comparison instead of string matching for better reliability
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
                    "Unsupported array dimensions: {}. Expected 1D (mono) or 2D (multi-channel)",
                    ndim
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
                    "Unsupported array dimensions: {}. Expected 1D (mono) or 2D (multi-channel)",
                    ndim
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
                    "Unsupported array dimensions: {}. Expected 1D (mono) or 2D (multi-channel)",
                    ndim
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
                    "Unsupported array dimensions: {}. Expected 1D (mono) or 2D (multi-channel)",
                    ndim
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
                    "Unsupported array dimensions: {}. Expected 1D (mono) or 2D (multi-channel)",
                    ndim
                ))),
            }
        } else {
            Err(PyTypeError::new_err(format!(
                "Unsupported dtype with {} dimensions. Supported types: int16, int32, float32, float64, I24",
                ndim
            )))
        }
    }

    #[staticmethod]
    #[pyo3(name = "_new_mono_i16_from_np", signature = (arr, sample_rate), text_signature = "($cls, arr: numpy.ndarray, sample_rate: int) -> AudioSamples")]
    fn new_mono_i16_from_np(arr: Bound<'_, PyArray1<i16>>, sample_rate: u32) -> Self {
        Self::new_mono_from_python::<i16>(arr, sample_rate)
    }

    #[staticmethod]
    #[pyo3(name = "_new_multi_i16_from_np", signature = (arr, sample_rate), text_signature = "($cls, arr: numpy.ndarray, sample_rate: int) -> AudioSamples")]
    fn new_multi_i16_from_np(arr: Bound<'_, PyArray2<i16>>, sample_rate: u32) -> Self {
        Self::new_multi_from_python::<i16>(arr, sample_rate)
    }

    #[staticmethod]
    #[pyo3(name = "_new_mono_i24_from_np", signature = (arr, sample_rate), text_signature = "($cls, arr: numpy.ndarray, sample_rate: int) -> AudioSamples")]
    fn new_mono_i24_from_np(arr: Bound<'_, PyArray1<I24>>, sample_rate: u32) -> Self {
        Self::new_mono_from_python::<I24>(arr, sample_rate)
    }

    #[staticmethod]
    #[pyo3(name = "_new_multi_i24_from_np", signature = (arr, sample_rate), text_signature = "($cls, arr: numpy.ndarray, sample_rate: int) -> AudioSamples")]
    fn new_multi_i24_from_np(arr: Bound<'_, PyArray2<I24>>, sample_rate: u32) -> Self {
        Self::new_multi_from_python::<I24>(arr, sample_rate)
    }

    #[staticmethod]
    #[pyo3(name = "_new_mono_i32_from_np", signature = (arr, sample_rate), text_signature = "($cls, arr: numpy.ndarray, sample_rate: int) -> AudioSamples")]
    fn new_mono_i32_from_np(arr: Bound<'_, PyArray1<i32>>, sample_rate: u32) -> Self {
        Self::new_mono_from_python::<i32>(arr, sample_rate)
    }

    #[staticmethod]
    #[pyo3(name = "_new_multi_i32_from_np", signature = (arr, sample_rate), text_signature = "($cls, arr: numpy.ndarray, sample_rate: int) -> AudioSamples")]
    fn new_multi_i32_from_np(arr: Bound<'_, PyArray2<i32>>, sample_rate: u32) -> Self {
        Self::new_multi_from_python::<i32>(arr, sample_rate)
    }

    #[staticmethod]
    #[pyo3(name = "_new_mono_f32_from_np", signature = (arr, sample_rate), text_signature = "($cls, arr: numpy.ndarray, sample_rate: int) -> AudioSamples")]
    fn new_mono_f32_from_np(arr: Bound<'_, PyArray1<f32>>, sample_rate: u32) -> Self {
        Self::new_mono_from_python::<f32>(arr, sample_rate)
    }

    #[staticmethod]
    #[pyo3(name = "_new_multi_f32_from_np", signature = (arr, sample_rate), text_signature = "($cls, arr: numpy.ndarray, sample_rate: int) -> AudioSamples")]
    fn new_multi_f32_from_np(arr: Bound<'_, PyArray2<f32>>, sample_rate: u32) -> Self {
        Self::new_multi_from_python::<f32>(arr, sample_rate)
    }
    #[staticmethod]
    #[pyo3(name = "_new_mono_f64_from_np", signature = (arr, sample_rate), text_signature = "($cls, arr: numpy.ndarray, sample_rate: int) -> AudioSamples")]
    fn new_mono_f64_from_np(arr: Bound<'_, PyArray1<f64>>, sample_rate: u32) -> Self {
        Self::new_mono_from_python::<f64>(arr, sample_rate)
    }
    #[staticmethod]
    #[pyo3(name = "_new_multi_f64_from_np", signature = (arr, sample_rate), text_signature = "($cls, arr: numpy.ndarray, sample_rate: int) -> AudioSamples")]
    fn new_multi_f64_from_np(arr: Bound<'_, PyArray2<f64>>, sample_rate: u32) -> Self {
        Self::new_multi_from_python::<f64>(arr, sample_rate)
    }

    #[staticmethod]
    #[pyo3(name = "new_mono", signature = (arr, sample_rate), text_signature = "($cls, arr: numpy.ndarray, sample_rate: int) -> AudioSamples")]
    fn py_new_mono(
        py: Python<'_>,
        arr: Bound<'_, PyUntypedArray>,
        sample_rate: u32,
    ) -> PyResult<Self> {
        let dtype = arr.dtype();
        if dtype.is_equiv_to(&numpy::dtype::<i16>(py)) {
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
    #[pyo3(name = "new_multi", signature = (arr, sample_rate), text_signature = "($cls, arr: numpy.ndarray, sample_rate: int) -> AudioSamples")]
    fn py_new_multi(
        py: Python<'_>,
        arr: Bound<'_, PyUntypedArray>,
        sample_rate: u32,
    ) -> PyResult<Self> {
        let dtype = arr.dtype();
        if dtype.is_equiv_to(&numpy::dtype::<i16>(py)) {
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

    #[pyo3(signature = (), text_signature = "($self) -> int | float")]
    fn peak<'py>(&self, py: Python<'py>) -> Bound<'py, PyAny> {
        match &self.inner {
            PyAudioDataInner::I16(a) => a.with_view(py, |audio| {
                let peak = audio.peak();
                peak.into_pyobject(py)
                    .expect("Primitive type should not fail to convert")
                    .into_any()
            }),
            PyAudioDataInner::I24(a) => a.with_view(py, |audio| {
                let peak = audio.peak();
                peak.into_pyobject(py)
                    .expect("Primitive type should not fail to convert")
                    .into_any()
            }),
            PyAudioDataInner::I32(a) => a.with_view(py, |audio| {
                let peak = audio.peak();
                peak.into_pyobject(py)
                    .expect("Primitive type should not fail to convert")
                    .into_any()
            }),
            PyAudioDataInner::F32(a) => a.with_view(py, |audio| {
                let peak = audio.peak();
                peak.into_pyobject(py)
                    .expect("Primitive type should not fail to convert")
                    .into_any()
            }),
            PyAudioDataInner::F64(a) => a.with_view(py, |audio| {
                let peak = audio.peak();
                peak.into_pyobject(py)
                    .expect("Primitive type should not fail to convert")
                    .into_any()
            }),
        }
    }

    #[pyo3(name = "min", signature = (), text_signature = "($self) -> int | float")]
    fn min_sample<'py>(&self, py: Python<'py>) -> Bound<'py, PyAny> {
        match &self.inner {
            PyAudioDataInner::I16(a) => a.with_view(py, |audio| {
                audio
                    .min_sample()
                    .into_pyobject(py)
                    .expect("Failed to convert sample to Python object")
                    .into_any()
            }),
            PyAudioDataInner::I24(a) => a.with_view(py, |audio| {
                audio
                    .min_sample()
                    .into_pyobject(py)
                    .expect("Failed to convert sample to Python object")
                    .into_any()
            }),
            PyAudioDataInner::I32(a) => a.with_view(py, |audio| {
                audio
                    .min_sample()
                    .into_pyobject(py)
                    .expect("Failed to convert sample to Python object")
                    .into_any()
            }),
            PyAudioDataInner::F32(a) => a.with_view(py, |audio| {
                audio
                    .min_sample()
                    .into_pyobject(py)
                    .expect("Failed to convert sample to Python object")
                    .into_any()
            }),
            PyAudioDataInner::F64(a) => a.with_view(py, |audio| {
                audio
                    .min_sample()
                    .into_pyobject(py)
                    .expect("Failed to convert sample to Python object")
                    .into_any()
            }),
        }
    }

    #[pyo3(name = "max", signature = (), text_signature = "($self) -> int | float")]
    fn max_sample<'py>(&self, py: Python<'py>) -> Bound<'py, PyAny> {
        match &self.inner {
            PyAudioDataInner::I16(a) => a.with_view(py, |audio| {
                audio
                    .max_sample()
                    .into_pyobject(py)
                    .expect("Failed to convert sample to Python object")
                    .into_any()
            }),
            PyAudioDataInner::I24(a) => a.with_view(py, |audio| {
                audio
                    .max_sample()
                    .into_pyobject(py)
                    .expect("Failed to convert sample to Python object")
                    .into_any()
            }),
            PyAudioDataInner::I32(a) => a.with_view(py, |audio| {
                audio
                    .max_sample()
                    .into_pyobject(py)
                    .expect("Failed to convert sample to Python object")
                    .into_any()
            }),
            PyAudioDataInner::F32(a) => a.with_view(py, |audio| {
                audio
                    .max_sample()
                    .into_pyobject(py)
                    .expect("Failed to convert sample to Python object")
                    .into_any()
            }),
            PyAudioDataInner::F64(a) => a.with_view(py, |audio| {
                audio
                    .max_sample()
                    .into_pyobject(py)
                    .expect("Failed to convert sample to Python object")
                    .into_any()
            }),
        }
    }

    #[pyo3(signature = (), text_signature = "($self) -> Optional[float]")]
    fn mean(&self, py: Python<'_>) -> f64 {
        dispatch_with_view!(self.inner, py, |audio| audio.mean::<f64>())
    }

    #[pyo3(signature = (), text_signature = "($self) -> Optional[float]")]
    fn rms(&self, py: Python<'_>) -> f64 {
        dispatch_with_view!(self.inner, py, |audio| audio.rms::<f64>())
    }

    #[pyo3(signature = ())]
    fn variance(&self, py: Python<'_>) -> f64 {
        dispatch_with_view!(self.inner, py, |audio| audio.variance::<f64>())
    }

    #[pyo3(signature = (), text_signature = "($self) -> Optional[float]")]
    fn std_dev(&self, py: Python<'_>) -> f64 {
        self.variance(py).sqrt()
    }

    #[pyo3(signature = (), text_signature = "($self) -> int")]
    fn zero_crossings(&self, py: Python<'_>) -> usize {
        dispatch_with_view!(self.inner, py, |audio| audio.zero_crossings())
    }

    #[pyo3(signature = (), text_signature = "($self) -> float")]
    fn zero_crossing_rate(&self, py: Python<'_>) -> f64 {
        dispatch_with_view!(self.inner, py, |audio| audio.zero_crossing_rate())
    }

    #[pyo3(signature = (max_lag), text_signature = "($self, max_lag: int) -> Optional[list[float]]")]
    fn autocorrelation(&self, py: Python<'_>, max_lag: usize) -> Option<Vec<f64>> {
        use audio_samples::AudioStatistics;
        dispatch_with_view!(self.inner, py, |audio| audio.autocorrelation(max_lag))
    }

    fn cross_correlation(
        &self,
        _py: Python<'_>,
        _other: &PyAudioSamples,
        _max_lag: usize,
    ) -> PyResult<Vec<f64>> {
        use pyo3::exceptions::PyTypeError;
        // For now, we'll return an error indicating cross-correlation requires more complex implementation
        Err(PyErr::new::<PyTypeError, _>(
            "Cross-correlation is not yet implemented due to lifetime constraints",
        ))
    }

    #[pyo3(signature = (), text_signature = "($self) -> float")]
    fn spectral_centroid(&self, py: Python<'_>) -> PyResult<f64> {
        dispatch_with_view_result!(self.inner, py, |audio| audio.spectral_centroid())
    }

    #[pyo3(signature = (rolloff_percent=0.85), text_signature = "($self, rolloff_percent: float = 0.85) -> float")]
    fn spectral_rolloff(&self, py: Python<'_>, rolloff_percent: f64) -> PyResult<f64> {
        dispatch_with_view_result!(self.inner, py, |audio| audio
            .spectral_rolloff(rolloff_percent))
    }

    // AudioProcessing

    #[pyo3(signature = (factor), text_signature = "($self, factor: float)")]
    fn scale(&mut self, py: Python<'_>, factor: f64) {
        match &mut self.inner {
            PyAudioDataInner::I16(a) => a.with_view_mut(py, |mut audio| {
                audio.scale(factor as i16);
            }),
            PyAudioDataInner::I24(a) => a.with_view_mut(py, |mut audio| {
                audio.scale(I24::wrapping_from_i32(factor as i32));
            }),
            PyAudioDataInner::I32(a) => a.with_view_mut(py, |mut audio| {
                audio.scale(factor as i32);
            }),
            PyAudioDataInner::F32(a) => a.with_view_mut(py, |mut audio| {
                audio.scale(factor as f32);
            }),
            PyAudioDataInner::F64(a) => a.with_view_mut(py, |mut audio| {
                audio.scale(factor);
            }),
        }
    }

    #[pyo3(signature = (min, max, method_str), text_signature = "($self, min: float, max: float, method: Literal['minmax', 'zscore', 'mean', 'median'])")]
    fn normalize(&mut self, py: Python<'_>, min: f64, max: f64, method_str: &str) -> PyResult<()> {
        let method = match method_str.to_lowercase().as_str() {
            "minmax" => NormalizationMethod::MinMax,
            "zscore" => NormalizationMethod::ZScore,
            "mean" => NormalizationMethod::Mean,
            "median" => NormalizationMethod::Median,
            _ => {
                return Err(PyErr::new::<PyTypeError, _>(
                    "Invalid normalization method. Use 'minmax', 'zscore', 'mean', or 'median'",
                ));
            }
        };

        match &mut self.inner {
            PyAudioDataInner::I16(a) => a.with_view_mut(py, |mut audio| {
                audio
                    .normalize(min as i16, max as i16, method)
                    .map_err(|e| {
                        PyErr::new::<PyTypeError, _>(format!("Normalization failed: {}", e))
                    })
            }),
            PyAudioDataInner::I24(a) => a.with_view_mut(py, |mut audio| {
                audio
                    .normalize(
                        I24::wrapping_from_i32(min as i32),
                        I24::wrapping_from_i32(max as i32),
                        method,
                    )
                    .map_err(|e| {
                        PyErr::new::<PyTypeError, _>(format!("Normalization failed: {}", e))
                    })
            }),
            PyAudioDataInner::I32(a) => a.with_view_mut(py, |mut audio| {
                audio
                    .normalize(min as i32, max as i32, method)
                    .map_err(|e| {
                        PyErr::new::<PyTypeError, _>(format!("Normalization failed: {}", e))
                    })
            }),
            PyAudioDataInner::F32(a) => a.with_view_mut(py, |mut audio| {
                audio
                    .normalize(min as f32, max as f32, method)
                    .map_err(|e| {
                        PyErr::new::<PyTypeError, _>(format!("Normalization failed: {}", e))
                    })
            }),
            PyAudioDataInner::F64(a) => a.with_view_mut(py, |mut audio| {
                audio.normalize(min, max, method).map_err(|e| {
                    PyErr::new::<PyTypeError, _>(format!("Normalization failed: {}", e))
                })
            }),
        }
    }

    #[pyo3(signature = (min_val, max_val), text_signature = "($self, min_val: float, max_val: float)")]
    fn clip(&mut self, py: Python<'_>, min_val: f64, max_val: f64) -> PyResult<()> {
        match &mut self.inner {
            PyAudioDataInner::I16(a) => a.with_view_mut(py, |mut audio| {
                audio
                    .clip(min_val as i16, max_val as i16)
                    .map_err(|e| PyErr::new::<PyTypeError, _>(format!("Clip failed: {}", e)))
            }),
            PyAudioDataInner::I24(a) => a.with_view_mut(py, |mut audio| {
                audio
                    .clip(
                        I24::wrapping_from_i32(min_val as i32),
                        I24::wrapping_from_i32(max_val as i32),
                    )
                    .map_err(|e| PyErr::new::<PyTypeError, _>(format!("Clip failed: {}", e)))
            }),
            PyAudioDataInner::I32(a) => a.with_view_mut(py, |mut audio| {
                audio
                    .clip(min_val as i32, max_val as i32)
                    .map_err(|e| PyErr::new::<PyTypeError, _>(format!("Clip failed: {}", e)))
            }),
            PyAudioDataInner::F32(a) => a.with_view_mut(py, |mut audio| {
                audio
                    .clip(min_val as f32, max_val as f32)
                    .map_err(|e| PyErr::new::<PyTypeError, _>(format!("Clip failed: {}", e)))
            }),
            PyAudioDataInner::F64(a) => a.with_view_mut(py, |mut audio| {
                audio
                    .clip(min_val, max_val)
                    .map_err(|e| PyErr::new::<PyTypeError, _>(format!("Clip failed: {}", e)))
            }),
        }
    }

    #[pyo3(signature = (), text_signature = "($self)")]
    fn remove_dc_offset(&mut self, py: Python<'_>) -> PyResult<()> {
        dispatch_with_view_mut_result!(self.inner, py, |mut audio| audio.remove_dc_offset())
    }

    // AudioEditing

    #[pyo3(signature = (), text_signature = "($self)")]
    fn reverse_in_place(&mut self, py: Python<'_>) -> PyResult<()> {
        dispatch_with_view_mut_result!(self.inner, py, |mut audio| audio.reverse_in_place())
    }

    #[pyo3(signature = (), text_signature = "($self) -> AudioSamples")]
    fn reverse(&self, py: Python<'_>) -> PyResult<Self> {
        dispatch_with_view!(self.inner, py, |audio| {
            let reversed = audio.reverse();
            Self::from_audio_samples(reversed)
        })
    }

    #[pyo3(signature = (start_seconds, end_seconds), text_signature = "($self, start_seconds: float, end_seconds: float) -> AudioSamples")]
    fn trim(&self, py: Python<'_>, start_seconds: f64, end_seconds: f64) -> PyResult<Self> {
        dispatch_with_view!(self.inner, py, |audio| {
            let trimmed = audio
                .trim(start_seconds, end_seconds)
                .map_err(audio_err_to_py)?;
            Self::from_audio_samples(trimmed)
        })
    }

    #[getter]
    fn sample_rate(&self, py: Python<'_>) -> u32 {
        dispatch_with_view!(self.inner, py, |audio| audio.sample_rate().get())
    }

    #[getter(sr)]
    fn sr(&self, py: Python<'_>) -> u32 {
        self.sample_rate(py)
    }

    #[getter(fs)]
    fn fs(&self, py: Python<'_>) -> u32 {
        self.sample_rate(py)
    }

    #[getter]
    fn num_channels(&self, py: Python<'_>) -> usize {
        dispatch_with_view!(self.inner, py, |audio| audio.num_channels())
    }

    #[getter(channels)]
    fn channels(&self, py: Python<'_>) -> usize {
        self.num_channels(py)
    }

    #[getter]
    fn len(&self, py: Python<'_>) -> usize {
        dispatch_with_view!(self.inner, py, |audio| audio.len())
    }

    #[getter(size)]
    fn size(&self, py: Python<'_>) -> usize {
        self.len(py)
    }

    #[pyo3(signature = (), text_signature = "($self) -> int")]
    fn samples_per_channel(&self, py: Python<'_>) -> usize {
        dispatch_with_view!(self.inner, py, |audio| audio.samples_per_channel())
    }

    #[getter]
    fn total_samples(&self, py: Python<'_>) -> usize {
        dispatch_with_view!(self.inner, py, |audio| audio.total_samples())
    }

    #[getter]
    fn shape(&self, py: Python<'_>) -> Vec<usize> {
        dispatch_with_view!(self.inner, py, |audio| audio.shape().to_vec())
    }

    #[pyo3(signature = (), text_signature = "($self) -> bool")]
    fn is_mono(&self, py: Python<'_>) -> bool {
        dispatch_with_view!(self.inner, py, |audio| audio.is_mono())
    }

    #[pyo3(signature = (), text_signature = "($self) -> bool")]
    fn is_multi_channel(&self, py: Python<'_>) -> bool {
        dispatch_with_view!(self.inner, py, |audio| audio.is_multi_channel())
    }

    #[pyo3(signature = (), text_signature = "($self) -> bool")]
    fn is_empty(&self, py: Python<'_>) -> bool {
        dispatch_with_view!(self.inner, py, |audio| audio.is_empty())
    }

    #[getter]
    fn duration_seconds(&self, py: Python<'_>) -> f64 {
        dispatch_with_view!(self.inner, py, |audio| audio.duration_seconds::<f64>())
    }

    #[getter(duration)]
    fn duration(&self, py: Python<'_>) -> f64 {
        self.duration_seconds(py)
    }

    #[getter(ndim)]
    fn ndim(&self, py: Python<'_>) -> usize {
        if self.is_mono(py) { 1 } else { 2 }
    }

    // TODO: integration with numpy array protocols
    // integrate with tensor protocols if possible (to_gpu, etc)

    fn __getitem__<'py>(
        &self,
        py: Python<'py>,
        index: Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        use pyo3::IntoPyObject;
        use pyo3::exceptions::PyIndexError;

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
                        "Sample index {} out of bounds for audio with {} samples",
                        idx, num_samples
                    )));
                }
                Ok((None, idx))
            } else if index.is_instance_of::<PyTuple>() {
                let (chan, samp): (usize, usize) = index.extract()?;
                if chan >= num_channels {
                    return Err(PyIndexError::new_err(format!(
                        "Channel index {} out of bounds for audio with {} channels",
                        chan, num_channels
                    )));
                }
                if samp >= num_samples {
                    return Err(PyIndexError::new_err(format!(
                        "Sample index {} out of bounds for audio with {} samples per channel",
                        samp, num_samples
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
                        "Channel index {} out of bounds for audio with {} channels",
                        chan, num_channels
                    )));
                }
                if samp >= num_samples {
                    return Err(PyIndexError::new_err(format!(
                        "Sample index {} out of bounds for audio with {} samples per channel",
                        samp, num_samples
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
            PyAudioDataInner::I16(a) => a.with_view(py, |audio| {
                let (chan_opt, samp) = parse_index(
                    &index,
                    audio.is_mono(),
                    audio.num_channels(),
                    audio.samples_per_channel(),
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
                    audio.num_channels(),
                    audio.samples_per_channel(),
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
                    audio.num_channels(),
                    audio.samples_per_channel(),
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
                    audio.num_channels(),
                    audio.samples_per_channel(),
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
                    audio.num_channels(),
                    audio.samples_per_channel(),
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
        format!("{}", self)
    }

    fn __repr__(&self) -> String {
        format!("{:#}", self)
    }

    fn __add__(&self, other: &PyAudioSamples) -> Self {
        self + other
    }

    fn __mul__(&self, py: Python<'_>, factor: f64) -> Self {
        let mut c: PyAudioSamples = self.clone();
        c.scale(py, factor);
        c
    }

    fn __truediv__(&self, py: Python<'_>, divisor: f64) -> PyResult<Self> {
        if divisor == 0.0 {
            return Err(pyo3::exceptions::PyZeroDivisionError::new_err(
                "division by zero",
            ));
        }
        let mut c: PyAudioSamples = self.clone();
        c.scale(py, 1.0 / divisor);
        Ok(c)
    }

    #[staticmethod]
    #[pyo3(name = "zeros_mono")]
    fn py_zeros_mono_f32(length: usize, sample_rate: u32) -> Self {
        let data = Array1::<f32>::zeros(length);
        Self::new_mono(data, sample_rate)
    }

    #[staticmethod]
    #[pyo3(name = "zeros_mono_i16")]
    fn py_zeros_mono_i16(length: usize, sample_rate: u32) -> Self {
        let data = Array1::<i16>::zeros(length);
        Self::new_mono(data, sample_rate)
    }

    #[staticmethod]
    #[pyo3(name = "zeros_mono_i32")]
    fn py_zeros_mono_i32(length: usize, sample_rate: u32) -> Self {
        let data = Array1::<i32>::zeros(length);
        Self::new_mono(data, sample_rate)
    }

    #[staticmethod]
    #[pyo3(name = "zeros_mono_f64")]
    fn py_zeros_mono_f64(length: usize, sample_rate: u32) -> Self {
        let data = Array1::<f64>::zeros(length);
        Self::new_mono(data, sample_rate)
    }

    #[staticmethod]
    #[pyo3(name = "ones_mono")]
    fn py_ones_mono_f32(length: usize, sample_rate: u32) -> Self {
        let data = Array1::<f32>::ones(length);
        Self::new_mono(data, sample_rate)
    }

    #[staticmethod]
    #[pyo3(name = "ones_mono_i16")]
    fn py_ones_mono_i16(length: usize, sample_rate: u32) -> Self {
        let data = Array1::<i16>::ones(length);
        Self::new_mono(data, sample_rate)
    }

    #[staticmethod]
    #[pyo3(name = "ones_mono_i32")]
    fn py_ones_mono_i32(length: usize, sample_rate: u32) -> Self {
        let data = Array1::<i32>::ones(length);
        Self::new_mono(data, sample_rate)
    }

    #[staticmethod]
    #[pyo3(name = "ones_mono_f64")]
    fn py_ones_mono_f64(length: usize, sample_rate: u32) -> Self {
        let data = Array1::<f64>::ones(length);
        Self::new_mono(data, sample_rate)
    }

    #[staticmethod]
    #[pyo3(name = "uniform_mono")]
    fn py_uniform_mono_f32(length: usize, sample_rate: u32, value: f32) -> Self {
        let data = Array1::<f32>::from_elem(length, value);
        Self::new_mono(data, sample_rate)
    }

    #[staticmethod]
    #[pyo3(name = "uniform_mono_i16")]
    fn py_uniform_mono_i16(length: usize, sample_rate: u32, value: i16) -> Self {
        let data = Array1::<i16>::from_elem(length, value);
        Self::new_mono(data, sample_rate)
    }

    #[staticmethod]
    #[pyo3(name = "uniform_mono_i32")]
    fn py_uniform_mono_i32(length: usize, sample_rate: u32, value: i32) -> Self {
        let data = Array1::<i32>::from_elem(length, value);
        Self::new_mono(data, sample_rate)
    }

    #[staticmethod]
    #[pyo3(name = "uniform_mono_f64")]
    fn py_uniform_mono_f64(length: usize, sample_rate: u32, value: f64) -> Self {
        let data = Array1::<f64>::from_elem(length, value);
        Self::new_mono(data, sample_rate)
    }

    #[staticmethod]
    #[pyo3(name = "zeros_multi")]
    fn py_zeros_multi_f32(channels: usize, length: usize, sample_rate: u32) -> Self {
        let data = Array2::<f32>::zeros((channels, length));
        Self::new_multi(data, sample_rate)
    }

    #[staticmethod]
    #[pyo3(name = "zeros_multi_i16")]
    fn py_zeros_multi_i16(channels: usize, length: usize, sample_rate: u32) -> Self {
        let data = Array2::<i16>::zeros((channels, length));
        Self::new_multi(data, sample_rate)
    }

    #[staticmethod]
    #[pyo3(name = "zeros_multi_i32")]
    fn py_zeros_multi_i32(channels: usize, length: usize, sample_rate: u32) -> Self {
        let data = Array2::<i32>::zeros((channels, length));
        Self::new_multi(data, sample_rate)
    }

    #[staticmethod]
    #[pyo3(name = "zeros_multi_f64")]
    fn py_zeros_multi_f64(channels: usize, length: usize, sample_rate: u32) -> Self {
        let data = Array2::<f64>::zeros((channels, length));
        Self::new_multi(data, sample_rate)
    }

    #[staticmethod]
    #[pyo3(name = "ones_multi")]
    fn py_ones_multi_f32(channels: usize, length: usize, sample_rate: u32) -> Self {
        let data = Array2::<f32>::ones((channels, length));
        Self::new_multi(data, sample_rate)
    }

    #[staticmethod]
    #[pyo3(name = "ones_multi_i16")]
    fn py_ones_multi_i16(channels: usize, length: usize, sample_rate: u32) -> Self {
        let data = Array2::<i16>::ones((channels, length));
        Self::new_multi(data, sample_rate)
    }

    #[staticmethod]
    #[pyo3(name = "ones_multi_i32")]
    fn py_ones_multi_i32(channels: usize, length: usize, sample_rate: u32) -> Self {
        let data = Array2::<i32>::ones((channels, length));
        Self::new_multi(data, sample_rate)
    }

    #[staticmethod]
    #[pyo3(name = "ones_multi_f64")]
    fn py_ones_multi_f64(channels: usize, length: usize, sample_rate: u32) -> Self {
        let data = Array2::<f64>::ones((channels, length));
        Self::new_multi(data, sample_rate)
    }

    #[staticmethod]
    #[pyo3(name = "uniform_multi")]
    fn py_uniform_multi_f32(channels: usize, length: usize, sample_rate: u32, value: f32) -> Self {
        let data = Array2::<f32>::from_elem((channels, length), value);
        Self::new_multi(data, sample_rate)
    }

    #[staticmethod]
    #[pyo3(name = "uniform_multi_i16")]
    fn py_uniform_multi_i16(channels: usize, length: usize, sample_rate: u32, value: i16) -> Self {
        let data = Array2::<i16>::from_elem((channels, length), value);
        Self::new_multi(data, sample_rate)
    }

    #[staticmethod]
    #[pyo3(name = "uniform_multi_i32")]
    fn py_uniform_multi_i32(channels: usize, length: usize, sample_rate: u32, value: i32) -> Self {
        let data = Array2::<i32>::from_elem((channels, length), value);
        Self::new_multi(data, sample_rate)
    }

    #[staticmethod]
    #[pyo3(name = "uniform_multi_f64")]
    fn py_uniform_multi_f64(channels: usize, length: usize, sample_rate: u32, value: f64) -> Self {
        let data = Array2::<f64>::from_elem((channels, length), value);
        Self::new_multi(data, sample_rate)
    }

    // AudioChannelOps methods

    #[pyo3(name = "to_mono")]
    #[pyo3(signature = (method, weights=None), text_signature = "($self, method: str, weights: Optional[list[float]] = None) -> AudioSamples")]
    fn to_mono(&self, py: Python<'_>, method: &str, weights: Option<Vec<f64>>) -> PyResult<Self> {
        let conversion_method = match method.to_lowercase().as_str() {
            "average" => MonoConversionMethod::Average,
            "left" => MonoConversionMethod::Left,
            "right" => MonoConversionMethod::Right,
            "center" => MonoConversionMethod::Center,
            "weighted" => {
                if let Some(w) = weights {
                    MonoConversionMethod::Weighted(w)
                } else {
                    return Err(PyTypeError::new_err(
                        "weights parameter required for 'weighted' method",
                    ));
                }
            }
            _ => {
                return Err(PyTypeError::new_err(
                    "Invalid mono conversion method. Use 'average', 'left', 'right', 'center', or 'weighted'",
                ));
            }
        };

        match &self.inner {
            PyAudioDataInner::I16(a) => a.with_view(py, |audio| {
                let result = audio
                    .to_mono(conversion_method)
                    .map_err(|e| PyTypeError::new_err(format!("Audio error: {}", e)))?;
                Self::from_audio_samples(result)
            }),
            PyAudioDataInner::I24(a) => a.with_view(py, |audio| {
                let result = audio
                    .to_mono(conversion_method)
                    .map_err(|e| PyTypeError::new_err(format!("Audio error: {}", e)))?;
                Self::from_audio_samples(result)
            }),
            PyAudioDataInner::I32(a) => a.with_view(py, |audio| {
                let result = audio
                    .to_mono(conversion_method)
                    .map_err(|e| PyTypeError::new_err(format!("Audio error: {}", e)))?;
                Self::from_audio_samples(result)
            }),
            PyAudioDataInner::F32(a) => a.with_view(py, |audio| {
                let result = audio
                    .to_mono(conversion_method)
                    .map_err(|e| PyTypeError::new_err(format!("Audio error: {}", e)))?;
                Self::from_audio_samples(result)
            }),
            PyAudioDataInner::F64(a) => a.with_view(py, |audio| {
                let result = audio
                    .to_mono(conversion_method)
                    .map_err(|e| PyTypeError::new_err(format!("Audio error: {}", e)))?;
                Self::from_audio_samples(result)
            }),
        }
    }

    #[pyo3(signature = (method, pan=None), text_signature = "($self, method: str, pan: Optional[float] = None) -> AudioSamples")]
    fn to_stereo(&self, py: Python<'_>, method: &str, pan: Option<f64>) -> PyResult<Self> {
        let conversion_method = match method.to_lowercase().as_str() {
            "duplicate" => StereoConversionMethod::Duplicate,
            "left" => StereoConversionMethod::Left,
            "right" => StereoConversionMethod::Right,
            "pan" => {
                if let Some(p) = pan {
                    StereoConversionMethod::Pan(p)
                } else {
                    return Err(PyTypeError::new_err(
                        "pan parameter required for 'pan' method",
                    ));
                }
            }
            _ => {
                return Err(PyTypeError::new_err(
                    "Invalid stereo conversion method. Use 'duplicate', 'left', 'right', or 'pan'",
                ));
            }
        };

        match &self.inner {
            PyAudioDataInner::I16(a) => a.with_view(py, |audio| {
                let result = audio
                    .to_stereo(conversion_method)
                    .map_err(|e| PyTypeError::new_err(format!("Audio error: {}", e)))?;
                if result.is_mono() {
                    Err(PyTypeError::new_err(
                        "to_stereo should produce multi-channel data",
                    ))
                } else {
                    Self::from_audio_samples(result)
                }
            }),
            PyAudioDataInner::I24(a) => a.with_view(py, |audio| {
                let result = audio
                    .to_stereo(conversion_method)
                    .map_err(|e| PyTypeError::new_err(format!("Audio error: {}", e)))?;
                if result.is_mono() {
                    Err(PyTypeError::new_err(
                        "to_stereo should produce multi-channel data",
                    ))
                } else {
                    Self::from_audio_samples(result)
                }
            }),
            PyAudioDataInner::I32(a) => a.with_view(py, |audio| {
                let result = audio
                    .to_stereo(conversion_method)
                    .map_err(|e| PyTypeError::new_err(format!("Audio error: {}", e)))?;
                if result.is_mono() {
                    Err(PyTypeError::new_err(
                        "to_stereo should produce multi-channel data",
                    ))
                } else {
                    Self::from_audio_samples(result)
                }
            }),
            PyAudioDataInner::F32(a) => a.with_view(py, |audio| {
                let result = audio
                    .to_stereo(conversion_method)
                    .map_err(|e| PyTypeError::new_err(format!("Audio error: {}", e)))?;
                if result.is_mono() {
                    Err(PyTypeError::new_err(
                        "to_stereo should produce multi-channel data",
                    ))
                } else {
                    Self::from_audio_samples(result)
                }
            }),
            PyAudioDataInner::F64(a) => a.with_view(py, |audio| {
                let result = audio
                    .to_stereo(conversion_method)
                    .map_err(|e| PyTypeError::new_err(format!("Audio error: {}", e)))?;
                if result.is_mono() {
                    Err(PyTypeError::new_err(
                        "to_stereo should produce multi-channel data",
                    ))
                } else {
                    Self::from_audio_samples(result)
                }
            }),
        }
    }

    #[pyo3(signature = (channel_index), text_signature = "($self, channel_index: int) -> AudioSamples")]
    fn extract_channel(&self, py: Python<'_>, channel_index: usize) -> PyResult<Self> {
        match &self.inner {
            PyAudioDataInner::I16(a) => a.with_view(py, |audio| {
                let result = audio
                    .extract_channel(channel_index)
                    .map_err(|e| PyTypeError::new_err(format!("Audio error: {}", e)))?;
                if !result.is_mono() {
                    Err(PyTypeError::new_err(
                        "extract_channel should produce mono data",
                    ))
                } else {
                    Self::from_audio_samples(result)
                }
            }),
            PyAudioDataInner::I24(a) => a.with_view(py, |audio| {
                let result = audio
                    .extract_channel(channel_index)
                    .map_err(|e| PyTypeError::new_err(format!("Audio error: {}", e)))?;
                if !result.is_mono() {
                    Err(PyTypeError::new_err(
                        "extract_channel should produce mono data",
                    ))
                } else {
                    Self::from_audio_samples(result)
                }
            }),
            PyAudioDataInner::I32(a) => a.with_view(py, |audio| {
                let result = audio
                    .extract_channel(channel_index)
                    .map_err(|e| PyTypeError::new_err(format!("Audio error: {}", e)))?;
                if !result.is_mono() {
                    Err(PyTypeError::new_err(
                        "extract_channel should produce mono data",
                    ))
                } else {
                    Self::from_audio_samples(result)
                }
            }),
            PyAudioDataInner::F32(a) => a.with_view(py, |audio| {
                let result = audio
                    .extract_channel(channel_index)
                    .map_err(|e| PyTypeError::new_err(format!("Audio error: {}", e)))?;
                if !result.is_mono() {
                    Err(PyTypeError::new_err(
                        "extract_channel should produce mono data",
                    ))
                } else {
                    Self::from_audio_samples(result)
                }
            }),
            PyAudioDataInner::F64(a) => a.with_view(py, |audio| {
                let result = audio
                    .extract_channel(channel_index)
                    .map_err(|e| PyTypeError::new_err(format!("Audio error: {}", e)))?;
                if !result.is_mono() {
                    Err(PyTypeError::new_err(
                        "extract_channel should produce mono data",
                    ))
                } else {
                    Self::from_audio_samples(result)
                }
            }),
        }
    }

    #[pyo3(signature = (channel1, channel2), text_signature = "($self, channel1: int, channel2: int) -> None")]
    fn swap_channels(&mut self, py: Python<'_>, channel1: usize, channel2: usize) -> PyResult<()> {
        match &mut self.inner {
            PyAudioDataInner::I16(a) => a.with_view_mut(py, |mut audio| {
                audio
                    .swap_channels(channel1, channel2)
                    .map_err(|e| PyTypeError::new_err(format!("Audio error: {}", e)))?;
                Ok(())
            }),
            PyAudioDataInner::I24(a) => a.with_view_mut(py, |mut audio| {
                audio
                    .swap_channels(channel1, channel2)
                    .map_err(|e| PyTypeError::new_err(format!("Audio error: {}", e)))?;
                Ok(())
            }),
            PyAudioDataInner::I32(a) => a.with_view_mut(py, |mut audio| {
                audio
                    .swap_channels(channel1, channel2)
                    .map_err(|e| PyTypeError::new_err(format!("Audio error: {}", e)))?;
                Ok(())
            }),
            PyAudioDataInner::F32(a) => a.with_view_mut(py, |mut audio| {
                audio
                    .swap_channels(channel1, channel2)
                    .map_err(|e| PyTypeError::new_err(format!("Audio error: {}", e)))?;
                Ok(())
            }),
            PyAudioDataInner::F64(a) => a.with_view_mut(py, |mut audio| {
                audio
                    .swap_channels(channel1, channel2)
                    .map_err(|e| PyTypeError::new_err(format!("Audio error: {}", e)))?;
                Ok(())
            }),
        }
    }

    // AudioDynamicRange methods

    #[pyo3(signature = (threshold_db, ratio, attack_ms, release_ms, makeup_gain_db, sample_rate), text_signature = "($self, threshold_db: float, ratio: float, attack_ms: float, release_ms: float, makeup_gain_db: float, sample_rate: float) -> None")]
    fn apply_compressor(
        &mut self,
        py: Python<'_>,
        threshold_db: f64,
        ratio: f64,
        attack_ms: f64,
        release_ms: f64,
        makeup_gain_db: f64,
        sample_rate: f64,
    ) -> PyResult<()> {
        match &mut self.inner {
            PyAudioDataInner::I16(a) => {
                let config = CompressorConfig {
                    threshold_db,
                    ratio,
                    attack_ms,
                    release_ms,
                    makeup_gain_db,
                    knee_type: KneeType::Soft,
                    knee_width_db: 2.0,
                    detection_method: DynamicRangeMethod::Peak,
                    side_chain: SideChainConfig::disabled(),
                    lookahead_ms: 0.0,
                };
                a.with_view_mut(py, |mut audio| {
                    audio
                        .apply_compressor(&config, sample_rate)
                        .map_err(|e| PyTypeError::new_err(format!("Audio error: {}", e)))?;
                    Ok(())
                })
            }
            PyAudioDataInner::I24(a) => {
                let config = CompressorConfig {
                    threshold_db,
                    ratio,
                    attack_ms,
                    release_ms,
                    makeup_gain_db,
                    knee_type: KneeType::Soft,
                    knee_width_db: 2.0,
                    detection_method: DynamicRangeMethod::Peak,
                    side_chain: SideChainConfig::disabled(),
                    lookahead_ms: 0.0,
                };
                a.with_view_mut(py, |mut audio| {
                    audio
                        .apply_compressor(&config, sample_rate)
                        .map_err(|e| PyTypeError::new_err(format!("Audio error: {}", e)))?;
                    Ok(())
                })
            }
            PyAudioDataInner::I32(a) => {
                let config = CompressorConfig {
                    threshold_db,
                    ratio,
                    attack_ms,
                    release_ms,
                    makeup_gain_db,
                    knee_type: KneeType::Soft,
                    knee_width_db: 2.0,
                    detection_method: DynamicRangeMethod::Peak,
                    side_chain: SideChainConfig::disabled(),
                    lookahead_ms: 0.0,
                };
                a.with_view_mut(py, |mut audio| {
                    audio
                        .apply_compressor(&config, sample_rate)
                        .map_err(|e| PyTypeError::new_err(format!("Audio error: {}", e)))?;
                    Ok(())
                })
            }
            PyAudioDataInner::F32(a) => {
                let config = CompressorConfig {
                    threshold_db: threshold_db as f32,
                    ratio: ratio as f32,
                    attack_ms: attack_ms as f32,
                    release_ms: release_ms as f32,
                    makeup_gain_db: makeup_gain_db as f32,
                    knee_type: KneeType::Soft,
                    knee_width_db: 2.0,
                    detection_method: DynamicRangeMethod::Peak,
                    side_chain: SideChainConfig::disabled(),
                    lookahead_ms: 0.0,
                };
                a.with_view_mut(py, |mut audio| {
                    audio
                        .apply_compressor(&config, sample_rate as f32)
                        .map_err(|e| PyTypeError::new_err(format!("Audio error: {}", e)))?;
                    Ok(())
                })
            }
            PyAudioDataInner::F64(a) => {
                let config = CompressorConfig {
                    threshold_db,
                    ratio,
                    attack_ms,
                    release_ms,
                    makeup_gain_db,
                    knee_type: KneeType::Soft,
                    knee_width_db: 2.0,
                    detection_method: DynamicRangeMethod::Peak,
                    side_chain: SideChainConfig::disabled(),
                    lookahead_ms: 0.0,
                };
                a.with_view_mut(py, |mut audio| {
                    audio
                        .apply_compressor(&config, sample_rate)
                        .map_err(|e| PyTypeError::new_err(format!("Audio error: {}", e)))?;
                    Ok(())
                })
            }
        }
    }

    #[pyo3(signature = (ceiling_db, release_ms, lookahead_ms, sample_rate), text_signature = "($self, ceiling_db: float, release_ms: float, lookahead_ms: float, sample_rate: float) -> None")]
    fn apply_limiter(
        &mut self,
        py: Python<'_>,
        ceiling_db: f64,
        release_ms: f64,
        lookahead_ms: f64,
        sample_rate: f64,
    ) -> PyResult<()> {
        match &mut self.inner {
            PyAudioDataInner::I16(a) => {
                let config = LimiterConfig {
                    ceiling_db,
                    attack_ms: 0.5,
                    release_ms,
                    knee_type: KneeType::Soft,
                    knee_width_db: 1.0,
                    detection_method: DynamicRangeMethod::Peak,
                    side_chain: SideChainConfig::disabled(),
                    lookahead_ms,
                    isp_limiting: false,
                };
                a.with_view_mut(py, |mut audio| {
                    audio
                        .apply_limiter(&config, sample_rate)
                        .map_err(|e| PyTypeError::new_err(format!("Audio error: {}", e)))?;
                    Ok(())
                })
            }
            PyAudioDataInner::I24(a) => {
                let config = LimiterConfig {
                    ceiling_db,
                    attack_ms: 0.5,
                    release_ms,
                    knee_type: KneeType::Soft,
                    knee_width_db: 1.0,
                    detection_method: DynamicRangeMethod::Peak,
                    side_chain: SideChainConfig::disabled(),
                    lookahead_ms,
                    isp_limiting: false,
                };
                a.with_view_mut(py, |mut audio| {
                    audio
                        .apply_limiter(&config, sample_rate)
                        .map_err(|e| PyTypeError::new_err(format!("Audio error: {}", e)))?;
                    Ok(())
                })
            }
            PyAudioDataInner::I32(a) => {
                let config = LimiterConfig {
                    ceiling_db,
                    attack_ms: 0.5,
                    release_ms,
                    knee_type: KneeType::Soft,
                    knee_width_db: 1.0,
                    detection_method: DynamicRangeMethod::Peak,
                    side_chain: SideChainConfig::disabled(),
                    lookahead_ms,
                    isp_limiting: false,
                };
                a.with_view_mut(py, |mut audio| {
                    audio
                        .apply_limiter(&config, sample_rate)
                        .map_err(|e| PyTypeError::new_err(format!("Audio error: {}", e)))?;
                    Ok(())
                })
            }
            PyAudioDataInner::F32(a) => {
                let config = LimiterConfig {
                    ceiling_db: ceiling_db as f32,
                    attack_ms: 0.5,
                    release_ms: release_ms as f32,
                    knee_type: KneeType::Soft,
                    knee_width_db: 1.0,
                    detection_method: DynamicRangeMethod::Peak,
                    side_chain: SideChainConfig::disabled(),
                    lookahead_ms: lookahead_ms as f32,
                    isp_limiting: false,
                };
                a.with_view_mut(py, |mut audio| {
                    audio
                        .apply_limiter(&config, sample_rate as f32)
                        .map_err(|e| PyTypeError::new_err(format!("Audio error: {}", e)))?;
                    Ok(())
                })
            }
            PyAudioDataInner::F64(a) => {
                let config = LimiterConfig {
                    ceiling_db,
                    attack_ms: 0.5,
                    release_ms,
                    knee_type: KneeType::Soft,
                    knee_width_db: 1.0,
                    detection_method: DynamicRangeMethod::Peak,
                    side_chain: SideChainConfig::disabled(),
                    lookahead_ms,
                    isp_limiting: false,
                };
                a.with_view_mut(py, |mut audio| {
                    audio
                        .apply_limiter(&config, sample_rate)
                        .map_err(|e| PyTypeError::new_err(format!("Audio error: {}", e)))?;
                    Ok(())
                })
            }
        }
    }

    #[pyo3(signature = (threshold_db, ratio, attack_ms, release_ms, sample_rate), text_signature = "($self, threshold_db: float, ratio: float, attack_ms: float, release_ms: float, sample_rate: float) -> None")]
    fn apply_gate(
        &mut self,
        py: Python<'_>,
        threshold_db: f64,
        ratio: f64,
        attack_ms: f64,
        release_ms: f64,
        sample_rate: f64,
    ) -> PyResult<()> {
        match &mut self.inner {
            PyAudioDataInner::I16(a) => a.with_view_mut(py, |mut audio| {
                audio
                    .apply_gate(threshold_db, ratio, attack_ms, release_ms, sample_rate)
                    .map_err(|e| PyTypeError::new_err(format!("Audio error: {}", e)))?;
                Ok(())
            }),
            PyAudioDataInner::I24(a) => a.with_view_mut(py, |mut audio| {
                audio
                    .apply_gate(threshold_db, ratio, attack_ms, release_ms, sample_rate)
                    .map_err(|e| PyTypeError::new_err(format!("Audio error: {}", e)))?;
                Ok(())
            }),
            PyAudioDataInner::I32(a) => a.with_view_mut(py, |mut audio| {
                audio
                    .apply_gate(threshold_db, ratio, attack_ms, release_ms, sample_rate)
                    .map_err(|e| PyTypeError::new_err(format!("Audio error: {}", e)))?;
                Ok(())
            }),
            PyAudioDataInner::F32(a) => a.with_view_mut(py, |mut audio| {
                audio
                    .apply_gate(
                        threshold_db as f32,
                        ratio as f32,
                        attack_ms as f32,
                        release_ms as f32,
                        sample_rate as f32,
                    )
                    .map_err(|e| PyTypeError::new_err(format!("Audio error: {}", e)))?;
                Ok(())
            }),
            PyAudioDataInner::F64(a) => a.with_view_mut(py, |mut audio| {
                audio
                    .apply_gate(threshold_db, ratio, attack_ms, release_ms, sample_rate)
                    .map_err(|e| PyTypeError::new_err(format!("Audio error: {}", e)))?;
                Ok(())
            }),
        }
    }

    #[pyo3(signature = (threshold_db, ratio, attack_ms, release_ms, sample_rate), text_signature = "($self, threshold_db: float, ratio: float, attack_ms: float, release_ms: float, sample_rate: float) -> None")]
    fn apply_expander(
        &mut self,
        py: Python<'_>,
        threshold_db: f64,
        ratio: f64,
        attack_ms: f64,
        release_ms: f64,
        sample_rate: f64,
    ) -> PyResult<()> {
        match &mut self.inner {
            PyAudioDataInner::I16(a) => a.with_view_mut(py, |mut audio| {
                audio
                    .apply_expander(threshold_db, ratio, attack_ms, release_ms, sample_rate)
                    .map_err(|e| PyTypeError::new_err(format!("Audio error: {}", e)))?;
                Ok(())
            }),
            PyAudioDataInner::I24(a) => a.with_view_mut(py, |mut audio| {
                audio
                    .apply_expander(threshold_db, ratio, attack_ms, release_ms, sample_rate)
                    .map_err(|e| PyTypeError::new_err(format!("Audio error: {}", e)))?;
                Ok(())
            }),
            PyAudioDataInner::I32(a) => a.with_view_mut(py, |mut audio| {
                audio
                    .apply_expander(threshold_db, ratio, attack_ms, release_ms, sample_rate)
                    .map_err(|e| PyTypeError::new_err(format!("Audio error: {}", e)))?;
                Ok(())
            }),
            PyAudioDataInner::F32(a) => a.with_view_mut(py, |mut audio| {
                audio
                    .apply_expander(
                        threshold_db as f32,
                        ratio as f32,
                        attack_ms as f32,
                        release_ms as f32,
                        sample_rate as f32,
                    )
                    .map_err(|e| PyTypeError::new_err(format!("Audio error: {}", e)))?;
                Ok(())
            }),
            PyAudioDataInner::F64(a) => a.with_view_mut(py, |mut audio| {
                audio
                    .apply_expander(threshold_db, ratio, attack_ms, release_ms, sample_rate)
                    .map_err(|e| PyTypeError::new_err(format!("Audio error: {}", e)))?;
                Ok(())
            }),
        }
    }

    // AudioIirFiltering methods

    #[pyo3(signature = (design, sample_rate), text_signature = "($self, design: IirFilterDesign, sample_rate: float) -> None")]
    fn apply_iir_filter(
        &mut self,
        py: Python<'_>,
        design: &PyIirFilterDesign,
        sample_rate: f64,
    ) -> PyResult<()> {
        match &mut self.inner {
            PyAudioDataInner::I16(a) => a.with_view_mut(py, |mut audio| {
                audio
                    .apply_iir_filter(design.inner(), sample_rate)
                    .map_err(|e| PyTypeError::new_err(format!("Audio error: {}", e)))?;
                Ok(())
            }),
            PyAudioDataInner::I24(a) => a.with_view_mut(py, |mut audio| {
                audio
                    .apply_iir_filter(design.inner(), sample_rate)
                    .map_err(|e| PyTypeError::new_err(format!("Audio error: {}", e)))?;
                Ok(())
            }),
            PyAudioDataInner::I32(a) => a.with_view_mut(py, |mut audio| {
                audio
                    .apply_iir_filter(design.inner(), sample_rate)
                    .map_err(|e| PyTypeError::new_err(format!("Audio error: {}", e)))?;
                Ok(())
            }),
            PyAudioDataInner::F32(a) => a.with_view_mut(py, |mut audio| {
                audio
                    .apply_iir_filter(design.inner(), sample_rate)
                    .map_err(|e| PyTypeError::new_err(format!("Audio error: {}", e)))?;
                Ok(())
            }),
            PyAudioDataInner::F64(a) => a.with_view_mut(py, |mut audio| {
                audio
                    .apply_iir_filter(design.inner(), sample_rate)
                    .map_err(|e| PyTypeError::new_err(format!("Audio error: {}", e)))?;
                Ok(())
            }),
        }
    }

    #[pyo3(signature = (order, cutoff_frequency, sample_rate), text_signature = "($self, order: int, cutoff_frequency: float, sample_rate: float) -> None")]
    fn apply_butterworth_lowpass(
        &mut self,
        py: Python<'_>,
        order: usize,
        cutoff_frequency: f64,
        sample_rate: f64,
    ) -> PyResult<()> {
        match &mut self.inner {
            PyAudioDataInner::I16(a) => a.with_view_mut(py, |mut audio| {
                audio
                    .butterworth_lowpass(order, cutoff_frequency, sample_rate)
                    .map_err(|e| PyTypeError::new_err(format!("Audio error: {}", e)))?;
                Ok(())
            }),
            PyAudioDataInner::I24(a) => a.with_view_mut(py, |mut audio| {
                audio
                    .butterworth_lowpass(order, cutoff_frequency, sample_rate)
                    .map_err(|e| PyTypeError::new_err(format!("Audio error: {}", e)))?;
                Ok(())
            }),
            PyAudioDataInner::I32(a) => a.with_view_mut(py, |mut audio| {
                audio
                    .butterworth_lowpass(order, cutoff_frequency, sample_rate)
                    .map_err(|e| PyTypeError::new_err(format!("Audio error: {}", e)))?;
                Ok(())
            }),
            PyAudioDataInner::F32(a) => a.with_view_mut(py, |mut audio| {
                audio
                    .butterworth_lowpass(order, cutoff_frequency as f32, sample_rate as f32)
                    .map_err(|e| PyTypeError::new_err(format!("Audio error: {}", e)))?;
                Ok(())
            }),
            PyAudioDataInner::F64(a) => a.with_view_mut(py, |mut audio| {
                audio
                    .butterworth_lowpass(order, cutoff_frequency, sample_rate)
                    .map_err(|e| PyTypeError::new_err(format!("Audio error: {}", e)))?;
                Ok(())
            }),
        }
    }

    #[pyo3(signature = (order, cutoff_frequency, sample_rate), text_signature = "($self, order: int, cutoff_frequency: float, sample_rate: float) -> None")]
    fn apply_butterworth_highpass(
        &mut self,
        py: Python<'_>,
        order: usize,
        cutoff_frequency: f64,
        sample_rate: f64,
    ) -> PyResult<()> {
        match &mut self.inner {
            PyAudioDataInner::I16(a) => a.with_view_mut(py, |mut audio| {
                audio
                    .butterworth_highpass(order, cutoff_frequency, sample_rate)
                    .map_err(|e| PyTypeError::new_err(format!("Audio error: {}", e)))?;
                Ok(())
            }),
            PyAudioDataInner::I24(a) => a.with_view_mut(py, |mut audio| {
                audio
                    .butterworth_highpass(order, cutoff_frequency, sample_rate)
                    .map_err(|e| PyTypeError::new_err(format!("Audio error: {}", e)))?;
                Ok(())
            }),
            PyAudioDataInner::I32(a) => a.with_view_mut(py, |mut audio| {
                audio
                    .butterworth_highpass(order, cutoff_frequency, sample_rate)
                    .map_err(|e| PyTypeError::new_err(format!("Audio error: {}", e)))?;
                Ok(())
            }),
            PyAudioDataInner::F32(a) => a.with_view_mut(py, |mut audio| {
                audio
                    .butterworth_highpass(order, cutoff_frequency as f32, sample_rate as f32)
                    .map_err(|e| PyTypeError::new_err(format!("Audio error: {}", e)))?;
                Ok(())
            }),
            PyAudioDataInner::F64(a) => a.with_view_mut(py, |mut audio| {
                audio
                    .butterworth_highpass(order, cutoff_frequency, sample_rate)
                    .map_err(|e| PyTypeError::new_err(format!("Audio error: {}", e)))?;
                Ok(())
            }),
        }
    }

    #[pyo3(signature = (order, low_frequency, high_frequency, sample_rate), text_signature = "($self, order: int, low_frequency: float, high_frequency: float, sample_rate: float) -> None")]
    fn apply_butterworth_bandpass(
        &mut self,
        py: Python<'_>,
        order: usize,
        low_frequency: f64,
        high_frequency: f64,
        sample_rate: f64,
    ) -> PyResult<()> {
        match &mut self.inner {
            PyAudioDataInner::I16(a) => a.with_view_mut(py, |mut audio| {
                audio
                    .butterworth_bandpass(order, low_frequency, high_frequency, sample_rate)
                    .map_err(|e| PyTypeError::new_err(format!("Audio error: {}", e)))?;
                Ok(())
            }),
            PyAudioDataInner::I24(a) => a.with_view_mut(py, |mut audio| {
                audio
                    .butterworth_bandpass(order, low_frequency, high_frequency, sample_rate)
                    .map_err(|e| PyTypeError::new_err(format!("Audio error: {}", e)))?;
                Ok(())
            }),
            PyAudioDataInner::I32(a) => a.with_view_mut(py, |mut audio| {
                audio
                    .butterworth_bandpass(order, low_frequency, high_frequency, sample_rate)
                    .map_err(|e| PyTypeError::new_err(format!("Audio error: {}", e)))?;
                Ok(())
            }),
            PyAudioDataInner::F32(a) => a.with_view_mut(py, |mut audio| {
                audio
                    .butterworth_bandpass(
                        order,
                        low_frequency as f32,
                        high_frequency as f32,
                        sample_rate as f32,
                    )
                    .map_err(|e| PyTypeError::new_err(format!("Audio error: {}", e)))?;
                Ok(())
            }),
            PyAudioDataInner::F64(a) => a.with_view_mut(py, |mut audio| {
                audio
                    .butterworth_bandpass(order, low_frequency, high_frequency, sample_rate)
                    .map_err(|e| PyTypeError::new_err(format!("Audio error: {}", e)))?;
                Ok(())
            }),
        }
    }

    #[pyo3(signature = (order, cutoff_frequency, passband_ripple, sample_rate, response), text_signature = "($self, order: int, cutoff_frequency: float, passband_ripple: float, sample_rate: float, response: str) -> None")]
    fn apply_chebyshev_i(
        &mut self,
        py: Python<'_>,
        order: usize,
        cutoff_frequency: f64,
        passband_ripple: f64,
        sample_rate: f64,
        response: &str,
    ) -> PyResult<()> {
        let filter_response = match response.to_lowercase().as_str() {
            "lowpass" => FilterResponse::LowPass,
            "highpass" => FilterResponse::HighPass,
            "bandpass" => FilterResponse::BandPass,
            "bandstop" => FilterResponse::BandStop,
            _ => {
                return Err(PyTypeError::new_err(
                    "Invalid filter response. Use 'lowpass', 'highpass', 'bandpass', or 'bandstop'",
                ));
            }
        };

        match &mut self.inner {
            PyAudioDataInner::I16(a) => a.with_view_mut(py, |mut audio| {
                audio
                    .chebyshev_i(
                        order,
                        cutoff_frequency,
                        passband_ripple,
                        sample_rate,
                        filter_response,
                    )
                    .map_err(|e| PyTypeError::new_err(format!("Audio error: {}", e)))?;
                Ok(())
            }),
            PyAudioDataInner::I24(a) => a.with_view_mut(py, |mut audio| {
                audio
                    .chebyshev_i(
                        order,
                        cutoff_frequency,
                        passband_ripple,
                        sample_rate,
                        filter_response,
                    )
                    .map_err(|e| PyTypeError::new_err(format!("Audio error: {}", e)))?;
                Ok(())
            }),
            PyAudioDataInner::I32(a) => a.with_view_mut(py, |mut audio| {
                audio
                    .chebyshev_i(
                        order,
                        cutoff_frequency,
                        passband_ripple,
                        sample_rate,
                        filter_response,
                    )
                    .map_err(|e| PyTypeError::new_err(format!("Audio error: {}", e)))?;
                Ok(())
            }),
            PyAudioDataInner::F32(a) => a.with_view_mut(py, |mut audio| {
                audio
                    .chebyshev_i(
                        order,
                        cutoff_frequency as f32,
                        passband_ripple as f32,
                        sample_rate as f32,
                        filter_response,
                    )
                    .map_err(|e| PyTypeError::new_err(format!("Audio error: {}", e)))?;
                Ok(())
            }),
            PyAudioDataInner::F64(a) => a.with_view_mut(py, |mut audio| {
                audio
                    .chebyshev_i(
                        order,
                        cutoff_frequency,
                        passband_ripple,
                        sample_rate,
                        filter_response,
                    )
                    .map_err(|e| PyTypeError::new_err(format!("Audio error: {}", e)))?;
                Ok(())
            }),
        }
    }

    #[pyo3(signature = (frequencies, sample_rate), text_signature = "($self, frequencies: list[float], sample_rate: float) -> Tuple[list[float], list[float]]")]
    fn frequency_response(
        &self,
        py: Python<'_>,
        frequencies: Vec<f64>,
        sample_rate: f64,
    ) -> PyResult<(Vec<f64>, Vec<f64>)> {
        match &self.inner {
            PyAudioDataInner::I16(a) => a.with_view(py, |audio| {
                let (magnitude, phase) = audio
                    .frequency_response(&frequencies, sample_rate)
                    .map_err(|e| PyTypeError::new_err(format!("Audio error: {}", e)))?;
                Ok((magnitude, phase))
            }),
            PyAudioDataInner::I24(a) => a.with_view(py, |audio| {
                let (magnitude, phase) = audio
                    .frequency_response(&frequencies, sample_rate)
                    .map_err(|e| PyTypeError::new_err(format!("Audio error: {}", e)))?;
                Ok((magnitude, phase))
            }),
            PyAudioDataInner::I32(a) => a.with_view(py, |audio| {
                let (magnitude, phase) = audio
                    .frequency_response(&frequencies, sample_rate)
                    .map_err(|e| PyTypeError::new_err(format!("Audio error: {}", e)))?;
                Ok((magnitude, phase))
            }),
            PyAudioDataInner::F32(a) => a.with_view(py, |audio| {
                let frequencies_f32: Vec<f32> = frequencies.iter().map(|&f| f as f32).collect();
                let (magnitude, phase) = audio
                    .frequency_response(&frequencies_f32, sample_rate as f32)
                    .map_err(|e| PyTypeError::new_err(format!("Audio error: {}", e)))?;
                let magnitude_f64: Vec<f64> = magnitude.iter().map(|&m| m as f64).collect();
                let phase_f64: Vec<f64> = phase.iter().map(|&p| p as f64).collect();
                Ok((magnitude_f64, phase_f64))
            }),
            PyAudioDataInner::F64(a) => a.with_view(py, |audio| {
                let (magnitude, phase) = audio
                    .frequency_response(&frequencies, sample_rate)
                    .map_err(|e| PyTypeError::new_err(format!("Audio error: {}", e)))?;
                Ok((magnitude, phase))
            }),
        }
    }

    // AudioParametricEq trait methods
    #[pyo3(signature = (eq, sample_rate), text_signature = "($self, eq: ParametricEq, sample_rate: float) -> None")]
    fn apply_parametric_eq(
        &mut self,
        py: Python<'_>,
        eq: &PyParametricEq,
        sample_rate: f64,
    ) -> PyResult<()> {
        match &mut self.inner {
            PyAudioDataInner::I16(a) => a.with_view_mut(py, |mut audio| {
                audio
                    .apply_parametric_eq(eq.inner(), sample_rate)
                    .map_err(|e| PyTypeError::new_err(format!("Audio error: {}", e)))?;
                Ok(())
            }),
            PyAudioDataInner::I24(a) => a.with_view_mut(py, |mut audio| {
                audio
                    .apply_parametric_eq(eq.inner(), sample_rate)
                    .map_err(|e| PyTypeError::new_err(format!("Audio error: {}", e)))?;
                Ok(())
            }),
            PyAudioDataInner::I32(a) => a.with_view_mut(py, |mut audio| {
                audio
                    .apply_parametric_eq(eq.inner(), sample_rate)
                    .map_err(|e| PyTypeError::new_err(format!("Audio error: {}", e)))?;
                Ok(())
            }),
            PyAudioDataInner::F32(a) => {
                a.with_view_mut(py, |mut audio| {
                    // Convert f64 ParametricEq to f32 (we'll need to handle this differently)
                    audio
                        .apply_parametric_eq(eq.inner(), sample_rate)
                        .map_err(|e| PyTypeError::new_err(format!("Audio error: {}", e)))?;
                    Ok(())
                })
            }
            PyAudioDataInner::F64(a) => a.with_view_mut(py, |mut audio| {
                audio
                    .apply_parametric_eq(eq.inner(), sample_rate)
                    .map_err(|e| PyTypeError::new_err(format!("Audio error: {}", e)))?;
                Ok(())
            }),
        }
    }

    #[pyo3(signature = (band, sample_rate), text_signature = "($self, band: EqBand, sample_rate: float) -> None")]
    fn apply_eq_band(&mut self, py: Python<'_>, band: &PyEqBand, sample_rate: f64) -> PyResult<()> {
        match &mut self.inner {
            PyAudioDataInner::I16(a) => a.with_view_mut(py, |mut audio| {
                audio
                    .apply_eq_band(band.inner(), sample_rate)
                    .map_err(|e| PyTypeError::new_err(format!("Audio error: {}", e)))?;
                Ok(())
            }),
            PyAudioDataInner::I24(a) => a.with_view_mut(py, |mut audio| {
                audio
                    .apply_eq_band(band.inner(), sample_rate)
                    .map_err(|e| PyTypeError::new_err(format!("Audio error: {}", e)))?;
                Ok(())
            }),
            PyAudioDataInner::I32(a) => a.with_view_mut(py, |mut audio| {
                audio
                    .apply_eq_band(band.inner(), sample_rate)
                    .map_err(|e| PyTypeError::new_err(format!("Audio error: {}", e)))?;
                Ok(())
            }),
            PyAudioDataInner::F32(a) => a.with_view_mut(py, |mut audio| {
                audio
                    .apply_eq_band(band.inner(), sample_rate)
                    .map_err(|e| PyTypeError::new_err(format!("Audio error: {}", e)))?;
                Ok(())
            }),
            PyAudioDataInner::F64(a) => a.with_view_mut(py, |mut audio| {
                audio
                    .apply_eq_band(band.inner(), sample_rate)
                    .map_err(|e| PyTypeError::new_err(format!("Audio error: {}", e)))?;
                Ok(())
            }),
        }
    }

    #[pyo3(signature = (frequency, gain_db, q_factor, sample_rate), text_signature = "($self, frequency: float, gain_db: float, q_factor: float, sample_rate: float) -> None")]
    fn apply_peak_filter(
        &mut self,
        py: Python<'_>,
        frequency: f64,
        gain_db: f64,
        q_factor: f64,
        sample_rate: f64,
    ) -> PyResult<()> {
        match &mut self.inner {
            PyAudioDataInner::I16(a) => a.with_view_mut(py, |mut audio| {
                audio
                    .apply_peak_filter(frequency, gain_db, q_factor, sample_rate)
                    .map_err(|e| PyTypeError::new_err(format!("Audio error: {}", e)))?;
                Ok(())
            }),
            PyAudioDataInner::I24(a) => a.with_view_mut(py, |mut audio| {
                audio
                    .apply_peak_filter(frequency, gain_db, q_factor, sample_rate)
                    .map_err(|e| PyTypeError::new_err(format!("Audio error: {}", e)))?;
                Ok(())
            }),
            PyAudioDataInner::I32(a) => a.with_view_mut(py, |mut audio| {
                audio
                    .apply_peak_filter(frequency, gain_db, q_factor, sample_rate)
                    .map_err(|e| PyTypeError::new_err(format!("Audio error: {}", e)))?;
                Ok(())
            }),
            PyAudioDataInner::F32(a) => a.with_view_mut(py, |mut audio| {
                audio
                    .apply_peak_filter(
                        frequency as f32,
                        gain_db as f32,
                        q_factor as f32,
                        sample_rate as f32,
                    )
                    .map_err(|e| PyTypeError::new_err(format!("Audio error: {}", e)))?;
                Ok(())
            }),
            PyAudioDataInner::F64(a) => a.with_view_mut(py, |mut audio| {
                audio
                    .apply_peak_filter(frequency, gain_db, q_factor, sample_rate)
                    .map_err(|e| PyTypeError::new_err(format!("Audio error: {}", e)))?;
                Ok(())
            }),
        }
    }

    #[pyo3(signature = (frequency, gain_db, q_factor, sample_rate), text_signature = "($self, frequency: float, gain_db: float, q_factor: float, sample_rate: float) -> None")]
    fn apply_low_shelf(
        &mut self,
        py: Python<'_>,
        frequency: f64,
        gain_db: f64,
        q_factor: f64,
        sample_rate: f64,
    ) -> PyResult<()> {
        match &mut self.inner {
            PyAudioDataInner::I16(a) => a.with_view_mut(py, |mut audio| {
                audio
                    .apply_low_shelf(frequency, gain_db, q_factor, sample_rate)
                    .map_err(|e| PyTypeError::new_err(format!("Audio error: {}", e)))?;
                Ok(())
            }),
            PyAudioDataInner::I24(a) => a.with_view_mut(py, |mut audio| {
                audio
                    .apply_low_shelf(frequency, gain_db, q_factor, sample_rate)
                    .map_err(|e| PyTypeError::new_err(format!("Audio error: {}", e)))?;
                Ok(())
            }),
            PyAudioDataInner::I32(a) => a.with_view_mut(py, |mut audio| {
                audio
                    .apply_low_shelf(frequency, gain_db, q_factor, sample_rate)
                    .map_err(|e| PyTypeError::new_err(format!("Audio error: {}", e)))?;
                Ok(())
            }),
            PyAudioDataInner::F32(a) => a.with_view_mut(py, |mut audio| {
                audio
                    .apply_low_shelf(
                        frequency as f32,
                        gain_db as f32,
                        q_factor as f32,
                        sample_rate as f32,
                    )
                    .map_err(|e| PyTypeError::new_err(format!("Audio error: {}", e)))?;
                Ok(())
            }),
            PyAudioDataInner::F64(a) => a.with_view_mut(py, |mut audio| {
                audio
                    .apply_low_shelf(frequency, gain_db, q_factor, sample_rate)
                    .map_err(|e| PyTypeError::new_err(format!("Audio error: {}", e)))?;
                Ok(())
            }),
        }
    }

    #[pyo3(signature = (frequency, gain_db, q_factor, sample_rate), text_signature = "($self, frequency: float, gain_db: float, q_factor: float, sample_rate: float) -> None")]
    fn apply_high_shelf(
        &mut self,
        py: Python<'_>,
        frequency: f64,
        gain_db: f64,
        q_factor: f64,
        sample_rate: f64,
    ) -> PyResult<()> {
        match &mut self.inner {
            PyAudioDataInner::I16(a) => a.with_view_mut(py, |mut audio| {
                audio
                    .apply_high_shelf(frequency, gain_db, q_factor, sample_rate)
                    .map_err(|e| PyTypeError::new_err(format!("Audio error: {}", e)))?;
                Ok(())
            }),
            PyAudioDataInner::I24(a) => a.with_view_mut(py, |mut audio| {
                audio
                    .apply_high_shelf(frequency, gain_db, q_factor, sample_rate)
                    .map_err(|e| PyTypeError::new_err(format!("Audio error: {}", e)))?;
                Ok(())
            }),
            PyAudioDataInner::I32(a) => a.with_view_mut(py, |mut audio| {
                audio
                    .apply_high_shelf(frequency, gain_db, q_factor, sample_rate)
                    .map_err(|e| PyTypeError::new_err(format!("Audio error: {}", e)))?;
                Ok(())
            }),
            PyAudioDataInner::F32(a) => a.with_view_mut(py, |mut audio| {
                audio
                    .apply_high_shelf(
                        frequency as f32,
                        gain_db as f32,
                        q_factor as f32,
                        sample_rate as f32,
                    )
                    .map_err(|e| PyTypeError::new_err(format!("Audio error: {}", e)))?;
                Ok(())
            }),
            PyAudioDataInner::F64(a) => a.with_view_mut(py, |mut audio| {
                audio
                    .apply_high_shelf(frequency, gain_db, q_factor, sample_rate)
                    .map_err(|e| PyTypeError::new_err(format!("Audio error: {}", e)))?;
                Ok(())
            }),
        }
    }

    #[pyo3(signature = (low_freq, low_gain, mid_freq, mid_gain, mid_q, high_freq, high_gain, sample_rate), text_signature = "($self, low_freq: float, low_gain: float, mid_freq: float, mid_gain: float, mid_q: float, high_freq: float, high_gain: float, sample_rate: float) -> None")]
    fn apply_three_band_eq(
        &mut self,
        py: Python<'_>,
        low_freq: f64,
        low_gain: f64,
        mid_freq: f64,
        mid_gain: f64,
        mid_q: f64,
        high_freq: f64,
        high_gain: f64,
        sample_rate: f64,
    ) -> PyResult<()> {
        match &mut self.inner {
            PyAudioDataInner::I16(a) => a.with_view_mut(py, |mut audio| {
                audio
                    .apply_three_band_eq(
                        low_freq,
                        low_gain,
                        mid_freq,
                        mid_gain,
                        mid_q,
                        high_freq,
                        high_gain,
                        sample_rate,
                    )
                    .map_err(|e| PyTypeError::new_err(format!("Audio error: {}", e)))?;
                Ok(())
            }),
            PyAudioDataInner::I24(a) => a.with_view_mut(py, |mut audio| {
                audio
                    .apply_three_band_eq(
                        low_freq,
                        low_gain,
                        mid_freq,
                        mid_gain,
                        mid_q,
                        high_freq,
                        high_gain,
                        sample_rate,
                    )
                    .map_err(|e| PyTypeError::new_err(format!("Audio error: {}", e)))?;
                Ok(())
            }),
            PyAudioDataInner::I32(a) => a.with_view_mut(py, |mut audio| {
                audio
                    .apply_three_band_eq(
                        low_freq,
                        low_gain,
                        mid_freq,
                        mid_gain,
                        mid_q,
                        high_freq,
                        high_gain,
                        sample_rate,
                    )
                    .map_err(|e| PyTypeError::new_err(format!("Audio error: {}", e)))?;
                Ok(())
            }),
            PyAudioDataInner::F32(a) => a.with_view_mut(py, |mut audio| {
                audio
                    .apply_three_band_eq(
                        low_freq,
                        low_gain,
                        mid_freq,
                        mid_gain,
                        mid_q,
                        high_freq,
                        high_gain,
                        sample_rate,
                    )
                    .map_err(|e| PyTypeError::new_err(format!("Audio error: {}", e)))?;
                Ok(())
            }),
            PyAudioDataInner::F64(a) => a.with_view_mut(py, |mut audio| {
                audio
                    .apply_three_band_eq(
                        low_freq,
                        low_gain,
                        mid_freq,
                        mid_gain,
                        mid_q,
                        high_freq,
                        high_gain,
                        sample_rate,
                    )
                    .map_err(|e| PyTypeError::new_err(format!("Audio error: {}", e)))?;
                Ok(())
            }),
        }
    }

    // AudioTransforms trait methods
    #[pyo3(signature = (), text_signature = "($self) -> numpy.ndarray")]
    fn fft<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<Complex64>>> {
        match &self.inner {
            PyAudioDataInner::I16(a) => a.with_view(py, |audio| {
                let audio = audio
                    .fft()
                    .map_err(|e| PyTypeError::new_err(format!("Audio error: {}", e)))?;
                // fft returns Array2 with shape (channels, fft_bins), flatten for mono
                let flat: Vec<Complex64> = audio.into_iter().collect();
                Ok(PyArray::from_vec(py, flat))
            }),

            PyAudioDataInner::I24(a) => a.with_view(py, |audio| {
                let audio = audio
                    .fft()
                    .map_err(|e| PyTypeError::new_err(format!("Audio error: {}", e)))?;
                let flat: Vec<Complex64> = audio.into_iter().collect();
                Ok(PyArray::from_vec(py, flat))
            }),

            PyAudioDataInner::I32(a) => a.with_view(py, |audio| {
                let audio = audio
                    .fft()
                    .map_err(|e| PyTypeError::new_err(format!("Audio error: {}", e)))?;
                let flat: Vec<Complex64> = audio.into_iter().collect();
                Ok(PyArray::from_vec(py, flat))
            }),

            PyAudioDataInner::F32(a) => a.with_view(py, |audio| {
                let audio = audio
                    .fft::<f64>()
                    .map_err(|e| PyTypeError::new_err(format!("Audio error: {}", e)))?;
                let flat: Vec<Complex64> = audio.into_iter().collect();
                Ok(PyArray::from_vec(py, flat))
            }),

            PyAudioDataInner::F64(a) => a.with_view(py, |audio| {
                let audio = audio
                    .fft()
                    .map_err(|e| PyTypeError::new_err(format!("Audio error: {}", e)))?;
                let flat: Vec<Complex64> = audio.into_iter().collect();
                Ok(PyArray::from_vec(py, flat))
            }),
        }
    }

    #[pyo3(signature = (window_size=2048, overlap=0.5), text_signature = "($self, window_size: int = 2048, overlap: float = 0.5) -> numpy.ndarray")]
    fn power_spectral_density<'py>(
        &self,
        py: Python<'py>,
        window_size: usize,
        overlap: f64,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        match &self.inner {
            PyAudioDataInner::I16(a) => a.with_view(py, |audio| {
                let (_freqs, psd) = audio
                    .power_spectral_density(window_size, overlap)
                    .map_err(|e| PyTypeError::new_err(format!("Audio error: {}", e)))?;
                Ok(PyArray::from_vec(py, psd))
            }),
            PyAudioDataInner::I24(a) => a.with_view(py, |audio| {
                let (_freqs, psd) = audio
                    .power_spectral_density(window_size, overlap)
                    .map_err(|e| PyTypeError::new_err(format!("Audio error: {}", e)))?;
                Ok(PyArray::from_vec(py, psd))
            }),
            PyAudioDataInner::I32(a) => a.with_view(py, |audio| {
                let (_freqs, psd) = audio
                    .power_spectral_density(window_size, overlap)
                    .map_err(|e| PyTypeError::new_err(format!("Audio error: {}", e)))?;
                Ok(PyArray::from_vec(py, psd))
            }),
            PyAudioDataInner::F32(a) => a.with_view(py, |audio| {
                let (_freqs, psd) = audio
                    .power_spectral_density(window_size, overlap)
                    .map_err(|e| PyTypeError::new_err(format!("Audio error: {}", e)))?;
                Ok(PyArray::from_vec(py, psd))
            }),
            PyAudioDataInner::F64(a) => a.with_view(py, |audio| {
                let (_freqs, psd) = audio
                    .power_spectral_density(window_size, overlap)
                    .map_err(|e| PyTypeError::new_err(format!("Audio error: {}", e)))?;
                Ok(PyArray::from_vec(py, psd))
            }),
        }
    }

    #[pyo3(signature = (n_mels=128, fmin=0.0, fmax=None, window_size=2048, hop_size=512), text_signature = "($self, n_mels: int = 128, fmin: float = 0.0, fmax: Optional[float] = None, window_size: int = 2048, hop_size: int = 512) -> numpy.ndarray")]
    fn mel_spectrogram<'py>(
        &self,
        py: Python<'py>,
        n_mels: usize,
        fmin: f64,
        fmax: Option<f64>,
        window_size: usize,
        hop_size: usize,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        match &self.inner {
            PyAudioDataInner::I16(a) => a.with_view(py, |audio| {
                let fmax_val = fmax.unwrap_or(audio.sample_rate().get() as f64 / 2.0);
                let mel_spec: Array2<f64> = audio
                    .mel_spectrogram(n_mels, fmin, fmax_val, window_size, hop_size)
                    .map_err(|e| PyTypeError::new_err(format!("Audio error: {}", e)))?;
                Ok(mel_spec.into_pyarray(py))
            }),
            PyAudioDataInner::I24(a) => a.with_view(py, |audio| {
                let fmax_val = fmax.unwrap_or(audio.sample_rate().get() as f64 / 2.0);
                let mel_spec: Array2<f64> = audio
                    .mel_spectrogram(n_mels, fmin, fmax_val, window_size, hop_size)
                    .map_err(|e| PyTypeError::new_err(format!("Audio error: {}", e)))?;
                Ok(mel_spec.into_pyarray(py))
            }),
            PyAudioDataInner::I32(a) => a.with_view(py, |audio| {
                let fmax_val = fmax.unwrap_or(audio.sample_rate().get() as f64 / 2.0);
                let mel_spec: Array2<f64> = audio
                    .mel_spectrogram(n_mels, fmin, fmax_val, window_size, hop_size)
                    .map_err(|e| PyTypeError::new_err(format!("Audio error: {}", e)))?;
                Ok(mel_spec.into_pyarray(py))
            }),
            PyAudioDataInner::F32(a) => a.with_view(py, |audio| {
                let fmax_val = fmax.unwrap_or(audio.sample_rate().get() as f64 / 2.0);
                let mel_spec = audio
                    .mel_spectrogram(n_mels, fmin, fmax_val, window_size, hop_size)
                    .map_err(|e| PyTypeError::new_err(format!("Audio error: {}", e)))?;
                Ok(mel_spec.into_pyarray(py))
            }),
            PyAudioDataInner::F64(a) => a.with_view(py, |audio| {
                let fmax_val = fmax.unwrap_or(audio.sample_rate().get() as f64 / 2.0);
                let mel_spec: Array2<f64> = audio
                    .mel_spectrogram(n_mels, fmin, fmax_val, window_size, hop_size)
                    .map_err(|e| PyTypeError::new_err(format!("Audio error: {}", e)))?;
                Ok(mel_spec.into_pyarray(py))
            }),
        }
    }

    #[pyo3(signature = (n_mfcc=13, n_mels=128, fmin=0.0, fmax=None), text_signature = "($self, n_mfcc: int = 13, n_mels: int = 128, fmin: float = 0.0, fmax: Optional[float] = None) -> numpy.ndarray")]
    fn mfcc<'py>(
        &self,
        py: Python<'py>,
        n_mfcc: usize,
        n_mels: usize,
        fmin: f64,
        fmax: Option<f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        match &self.inner {
            PyAudioDataInner::I16(a) => a.with_view(py, |audio| {
                let fmax_val = fmax.unwrap_or(audio.sample_rate().get() as f64 / 2.0);
                let mfcc_result = audio
                    .mfcc(n_mfcc, n_mels, fmin, fmax_val)
                    .map_err(|e| PyTypeError::new_err(format!("Audio error: {}", e)))?;
                Ok(mfcc_result.into_pyarray(py))
            }),
            PyAudioDataInner::I24(a) => a.with_view(py, |audio| {
                let fmax_val = fmax.unwrap_or(audio.sample_rate().get() as f64 / 2.0);
                let mfcc_result = audio
                    .mfcc(n_mfcc, n_mels, fmin, fmax_val)
                    .map_err(|e| PyTypeError::new_err(format!("Audio error: {}", e)))?;
                Ok(mfcc_result.into_pyarray(py))
            }),
            PyAudioDataInner::I32(a) => a.with_view(py, |audio| {
                let fmax_val = fmax.unwrap_or(audio.sample_rate().get() as f64 / 2.0);
                let mfcc_result = audio
                    .mfcc(n_mfcc, n_mels, fmin, fmax_val)
                    .map_err(|e| PyTypeError::new_err(format!("Audio error: {}", e)))?;
                Ok(mfcc_result.into_pyarray(py))
            }),
            PyAudioDataInner::F32(a) => a.with_view(py, |audio| {
                let fmax_val = fmax.unwrap_or(audio.sample_rate().get() as f64 / 2.0);
                let mfcc_result = audio
                    .mfcc(n_mfcc, n_mels, fmin, fmax_val)
                    .map_err(|e| PyTypeError::new_err(format!("Audio error: {}", e)))?;
                Ok(mfcc_result.into_pyarray(py))
            }),
            PyAudioDataInner::F64(a) => a.with_view(py, |audio| {
                let fmax_val = fmax.unwrap_or(audio.sample_rate().get() as f64 / 2.0);
                let mfcc_result = audio
                    .mfcc(n_mfcc, n_mels, fmin, fmax_val)
                    .map_err(|e| PyTypeError::new_err(format!("Audio error: {}", e)))?;
                Ok(mfcc_result.into_pyarray(py))
            }),
        }
    }

    #[pyo3(signature = (n_chroma=12), text_signature = "($self, n_chroma: int = 12) -> numpy.ndarray")]
    fn chroma<'py>(&self, py: Python<'py>, n_chroma: usize) -> PyResult<Bound<'py, PyArray2<f64>>> {
        match &self.inner {
            PyAudioDataInner::I16(a) => a.with_view(py, |audio| {
                let chroma_result: Array2<f64> = audio
                    .chroma(n_chroma)
                    .map_err(|e| PyTypeError::new_err(format!("Audio error: {}", e)))?;
                Ok(chroma_result.into_pyarray(py))
            }),
            PyAudioDataInner::I24(a) => a.with_view(py, |audio| {
                let chroma_result: Array2<f64> = audio
                    .chroma(n_chroma)
                    .map_err(|e| PyTypeError::new_err(format!("Audio error: {}", e)))?;
                Ok(chroma_result.into_pyarray(py))
            }),
            PyAudioDataInner::I32(a) => a.with_view(py, |audio| {
                let chroma_result: Array2<f64> = audio
                    .chroma(n_chroma)
                    .map_err(|e| PyTypeError::new_err(format!("Audio error: {}", e)))?;
                Ok(chroma_result.into_pyarray(py))
            }),
            PyAudioDataInner::F32(a) => a.with_view(py, |audio| {
                let chroma_result: Array2<f32> = audio
                    .chroma(n_chroma)
                    .map_err(|e| PyTypeError::new_err(format!("Audio error: {}", e)))?;
                let chroma_result_f64 = chroma_result.mapv(|x| x as f64);
                Ok(chroma_result_f64.into_pyarray(py))
            }),
            PyAudioDataInner::F64(a) => a.with_view(py, |audio| {
                let chroma_result: Array2<f64> = audio
                    .chroma(n_chroma)
                    .map_err(|e| PyTypeError::new_err(format!("Audio error: {}", e)))?;
                Ok(chroma_result.into_pyarray(py))
            }),
        }
    }

    fn to_format(
        &self,
        py: Python<'_>,
        dtype: &Bound<'_, PyArrayDescr>,
    ) -> PyResult<PyAudioSamples> {
        if dtype.is_equiv_to(&self.dtype(py)) {
            return Ok(self.clone());
        }
        let out = match &self.inner {
            PyAudioDataInner::I16(a) => {
                let samples = a.with_view(py, |audio| {
                    if dtype.is_equiv_to(&numpy::dtype::<i16>(py)) {
                        let conv = audio.to_format::<i16>();
                        return PyAudioSamples::from_audio_samples(conv);
                    } else if dtype.is_equiv_to(&numpy::dtype::<i32>(py)) {
                        let conv = audio.to_format::<i32>();
                        return PyAudioSamples::from_audio_samples(conv);
                    } else if dtype.is_equiv_to(&numpy::dtype::<f32>(py)) {
                        let conv = audio.to_format::<f32>();
                        return PyAudioSamples::from_audio_samples(conv);
                    } else if dtype.is_equiv_to(&numpy::dtype::<f64>(py)) {
                        let conv = audio.to_format::<f64>();
                        return PyAudioSamples::from_audio_samples(conv);
                    } else {
                        return Err(PyTypeError::new_err("Unsupported target dtype"));
                    }
                })?;
                Ok(samples)
            }
            PyAudioDataInner::I24(a) => {
                let samples = a.with_view(py, |audio| {
                    if dtype.is_equiv_to(&numpy::dtype::<i32>(py)) {
                        let conv = audio.to_format::<i32>();
                        return PyAudioSamples::from_audio_samples(conv);
                    } else if dtype.is_equiv_to(&numpy::dtype::<f32>(py)) {
                        let conv = audio.to_format::<f32>();
                        return PyAudioSamples::from_audio_samples(conv);
                    } else if dtype.is_equiv_to(&numpy::dtype::<f64>(py)) {
                        let conv = audio.to_format::<f64>();
                        return PyAudioSamples::from_audio_samples(conv);
                    } else {
                        return Err(PyTypeError::new_err("Unsupported target dtype"));
                    }
                })?;
                Ok(samples)
            }

            PyAudioDataInner::I32(a) => {
                let samples = a.with_view(py, |audio| {
                    if dtype.is_equiv_to(&numpy::dtype::<i16>(py)) {
                        let conv = audio.to_format::<i16>();
                        return PyAudioSamples::from_audio_samples(conv);
                    } else if dtype.is_equiv_to(&numpy::dtype::<i32>(py)) {
                        let conv = audio.to_format::<i32>();
                        return PyAudioSamples::from_audio_samples(conv);
                    } else if dtype.is_equiv_to(&numpy::dtype::<f32>(py)) {
                        let conv = audio.to_format::<f32>();
                        return PyAudioSamples::from_audio_samples(conv);
                    } else if dtype.is_equiv_to(&numpy::dtype::<f64>(py)) {
                        let conv = audio.to_format::<f64>();
                        return PyAudioSamples::from_audio_samples(conv);
                    } else {
                        return Err(PyTypeError::new_err("Unsupported target dtype"));
                    }
                })?;
                Ok(samples)
            }

            PyAudioDataInner::F32(a) => {
                let samples = a.with_view(py, |audio| {
                    if dtype.is_equiv_to(&numpy::dtype::<i16>(py)) {
                        let conv = audio.to_format::<i16>();
                        return PyAudioSamples::from_audio_samples(conv);
                    } else if dtype.is_equiv_to(&numpy::dtype::<i32>(py)) {
                        let conv = audio.to_format::<i32>();
                        return PyAudioSamples::from_audio_samples(conv);
                    } else if dtype.is_equiv_to(&numpy::dtype::<f32>(py)) {
                        let conv = audio.to_format::<f32>();
                        return PyAudioSamples::from_audio_samples(conv);
                    } else if dtype.is_equiv_to(&numpy::dtype::<f64>(py)) {
                        let conv = audio.to_format::<f64>();
                        return PyAudioSamples::from_audio_samples(conv);
                    } else {
                        return Err(PyTypeError::new_err("Unsupported target dtype"));
                    }
                })?;
                Ok(samples)
            }

            PyAudioDataInner::F64(a) => {
                let samples = a.with_view(py, |audio| {
                    if dtype.is_equiv_to(&numpy::dtype::<i16>(py)) {
                        let conv = audio.to_format::<i16>();
                        return PyAudioSamples::from_audio_samples(conv);
                    } else if dtype.is_equiv_to(&numpy::dtype::<i32>(py)) {
                        let conv = audio.to_format::<i32>();
                        return PyAudioSamples::from_audio_samples(conv);
                    } else if dtype.is_equiv_to(&numpy::dtype::<f32>(py)) {
                        let conv = audio.to_format::<f32>();
                        return PyAudioSamples::from_audio_samples(conv);
                    } else if dtype.is_equiv_to(&numpy::dtype::<f64>(py)) {
                        let conv = audio.to_format::<f64>();
                        return PyAudioSamples::from_audio_samples(conv);
                    } else {
                        return Err(PyTypeError::new_err("Unsupported target dtype"));
                    }
                })?;
                Ok(samples)
            }
        };
        out
    }

    fn as_i16(&self, py: Python<'_>) -> PyResult<PyAudioSamples> {
        self.to_format(py, &numpy::dtype::<i16>(py))
    }

    fn as_i32(&self, py: Python<'_>) -> PyResult<PyAudioSamples> {
        self.to_format(py, &numpy::dtype::<i32>(py))
    }

    fn as_f32(&self, py: Python<'_>) -> PyResult<PyAudioSamples> {
        self.to_format(py, &numpy::dtype::<f32>(py))
    }
    fn as_f64(&self, py: Python<'_>) -> PyResult<PyAudioSamples> {
        self.to_format(py, &numpy::dtype::<f64>(py))
    }

    fn cast_as(&self, py: Python<'_>, dtype: &Bound<'_, PyArrayDescr>) -> PyResult<PyAudioSamples> {
        if dtype.is_equiv_to(&self.dtype(py)) {
            return Ok(self.clone());
        }
        let out = match &self.inner {
            PyAudioDataInner::I16(a) => {
                let samples = a.with_view(py, |audio| {
                    if dtype.is_equiv_to(&numpy::dtype::<i16>(py)) {
                        let conv = audio.cast_as::<i16>();
                        return PyAudioSamples::from_audio_samples(conv);
                    } else if dtype.is_equiv_to(&numpy::dtype::<i32>(py)) {
                        let conv = audio.cast_as::<i32>();
                        return PyAudioSamples::from_audio_samples(conv);
                    } else if dtype.is_equiv_to(&numpy::dtype::<f32>(py)) {
                        let conv = audio.cast_as::<f32>();
                        return PyAudioSamples::from_audio_samples(conv);
                    } else if dtype.is_equiv_to(&numpy::dtype::<f64>(py)) {
                        let conv = audio.cast_as::<f64>();
                        return PyAudioSamples::from_audio_samples(conv);
                    } else {
                        return Err(PyTypeError::new_err("Unsupported target dtype"));
                    }
                });
                samples
            }
            PyAudioDataInner::I24(a) => {
                let samples = a.with_view(py, |audio| {
                    if dtype.is_equiv_to(&numpy::dtype::<i16>(py)) {
                        let conv = audio.cast_as::<i16>();
                        return PyAudioSamples::from_audio_samples(conv);
                    } else if dtype.is_equiv_to(&numpy::dtype::<i32>(py)) {
                        let conv = audio.cast_as::<i32>();
                        return PyAudioSamples::from_audio_samples(conv);
                    } else if dtype.is_equiv_to(&numpy::dtype::<f32>(py)) {
                        let conv = audio.cast_as::<f32>();
                        return PyAudioSamples::from_audio_samples(conv);
                    } else if dtype.is_equiv_to(&numpy::dtype::<f64>(py)) {
                        let conv = audio.cast_as::<f64>();
                        return PyAudioSamples::from_audio_samples(conv);
                    } else {
                        return Err(PyTypeError::new_err("Unsupported target dtype"));
                    }
                });
                samples
            }

            PyAudioDataInner::I32(a) => {
                let samples = a.with_view(py, |audio| {
                    if dtype.is_equiv_to(&numpy::dtype::<i16>(py)) {
                        let conv = audio.cast_as::<i16>();
                        return PyAudioSamples::from_audio_samples(conv);
                    } else if dtype.is_equiv_to(&numpy::dtype::<i32>(py)) {
                        let conv = audio.cast_as::<i32>();
                        return PyAudioSamples::from_audio_samples(conv);
                    } else if dtype.is_equiv_to(&numpy::dtype::<f32>(py)) {
                        let conv = audio.cast_as::<f32>();
                        return PyAudioSamples::from_audio_samples(conv);
                    } else if dtype.is_equiv_to(&numpy::dtype::<f64>(py)) {
                        let conv = audio.cast_as::<f64>();
                        return PyAudioSamples::from_audio_samples(conv);
                    } else {
                        return Err(PyTypeError::new_err("Unsupported target dtype"));
                    }
                });
                samples
            }

            PyAudioDataInner::F32(a) => {
                let samples = a.with_view(py, |audio| {
                    if dtype.is_equiv_to(&numpy::dtype::<i16>(py)) {
                        let conv = audio.cast_as::<i16>();
                        return PyAudioSamples::from_audio_samples(conv);
                    } else if dtype.is_equiv_to(&numpy::dtype::<i32>(py)) {
                        let conv = audio.cast_as::<i32>();
                        return PyAudioSamples::from_audio_samples(conv);
                    } else if dtype.is_equiv_to(&numpy::dtype::<f32>(py)) {
                        let conv = audio.cast_as::<f32>();
                        return PyAudioSamples::from_audio_samples(conv);
                    } else if dtype.is_equiv_to(&numpy::dtype::<f64>(py)) {
                        let conv = audio.cast_as::<f64>();
                        return PyAudioSamples::from_audio_samples(conv);
                    } else {
                        return Err(PyTypeError::new_err("Unsupported target dtype"));
                    }
                });
                samples
            }

            PyAudioDataInner::F64(a) => {
                let samples = a.with_view(py, |audio| {
                    if dtype.is_equiv_to(&numpy::dtype::<i16>(py)) {
                        let conv = audio.cast_as::<i16>();
                        return PyAudioSamples::from_audio_samples(conv);
                    } else if dtype.is_equiv_to(&numpy::dtype::<i32>(py)) {
                        let conv = audio.cast_as::<i32>();
                        return PyAudioSamples::from_audio_samples(conv);
                    } else if dtype.is_equiv_to(&numpy::dtype::<f32>(py)) {
                        let conv = audio.cast_as::<f32>();
                        return PyAudioSamples::from_audio_samples(conv);
                    } else if dtype.is_equiv_to(&numpy::dtype::<f64>(py)) {
                        let conv = audio.cast_as::<f64>();
                        return PyAudioSamples::from_audio_samples(conv);
                    } else {
                        return Err(PyTypeError::new_err("Unsupported target dtype"));
                    }
                });
                samples
            }
        };
        out
    }

    fn cast_as_i16(&self, py: Python<'_>) -> PyResult<PyAudioSamples> {
        self.cast_as(py, &numpy::dtype::<i16>(py))
    }

    fn cast_as_i32(&self, py: Python<'_>) -> PyResult<PyAudioSamples> {
        self.cast_as(py, &numpy::dtype::<i32>(py))
    }

    fn cast_as_f32(&self, py: Python<'_>) -> PyResult<PyAudioSamples> {
        self.cast_as(py, &numpy::dtype::<f32>(py))
    }

    fn cast_as_f64(&self, py: Python<'_>) -> PyResult<PyAudioSamples> {
        self.cast_as(py, &numpy::dtype::<f64>(py))
    }

    /// NumPy array protocol: return array representation
    fn __array__(
        &self,
        py: Python<'_>,
        dtype: Option<&Bound<'_, PyArrayDescr>>,
    ) -> PyResult<Py<PyAny>> {
        match dtype {
            Some(dt) if !dt.is_equiv_to(&self.dtype(py)) => {
                // Cast to requested dtype first
                let casted = self.cast_as(py, dt)?;
                casted.__array__(py, None)
            }
            _ => {
                // Return array with current dtype
                Ok(self.to_numpy(py)?.into_any().unbind())
            }
        }
    }

    /// NumPy array protocol: expose memory layout for zero-copy operations
    #[getter]
    fn __array_interface__(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let dict = PyDict::new(py);

        match &self.inner {
            PyAudioDataInner::I16(typed) => typed.with_view(py, |audio| {
                let shape = match &audio.data {
                    AudioData::Mono(arr) => {
                        vec![(arr.as_view().len() * std::mem::size_of::<i16>()) as usize]
                    }
                    AudioData::Multi(arr) => {
                        let view = arr.as_view();
                        vec![view.nrows(), view.ncols()]
                    }
                };
                dict.set_item("shape", shape)?;
                dict.set_item("typestr", "<i2")?;
                self.set_array_interface_data(py, &dict, &audio.data)
            }),
            PyAudioDataInner::I24(_) => {
                // I24 doesn't have a direct numpy equivalent, so we'll expose as int32
                let shape = self.shape(py);
                dict.set_item("shape", shape)?;
                dict.set_item("typestr", "<i4")?;
                // For I24, we need to convert to get the data pointer
                let as_i32 = self.cast_as_i32(py)?;
                Ok(as_i32.__array_interface__(py)?)
            }
            PyAudioDataInner::I32(typed) => typed.with_view(py, |audio| {
                let shape = match &audio.data {
                    AudioData::Mono(arr) => {
                        vec![(arr.as_view().len() * std::mem::size_of::<i32>()) as usize]
                    }
                    AudioData::Multi(arr) => {
                        let view = arr.as_view();
                        vec![view.nrows(), view.ncols()]
                    }
                };
                dict.set_item("shape", shape)?;
                dict.set_item("typestr", "<i4")?;
                self.set_array_interface_data(py, &dict, &audio.data)
            }),
            PyAudioDataInner::F32(typed) => typed.with_view(py, |audio| {
                let shape = match &audio.data {
                    AudioData::Mono(arr) => {
                        vec![(arr.as_view().len() * std::mem::size_of::<f32>()) as usize]
                    }
                    AudioData::Multi(arr) => {
                        let view = arr.as_view();
                        vec![view.nrows(), view.ncols()]
                    }
                };
                dict.set_item("shape", shape)?;
                dict.set_item("typestr", "<f4")?;
                self.set_array_interface_data(py, &dict, &audio.data)
            }),
            PyAudioDataInner::F64(typed) => typed.with_view(py, |audio| {
                let shape = match &audio.data {
                    AudioData::Mono(arr) => {
                        vec![(arr.as_view().len() * std::mem::size_of::<f64>()) as usize]
                    }
                    AudioData::Multi(arr) => {
                        let view = arr.as_view();
                        vec![view.nrows(), view.ncols()]
                    }
                };
                dict.set_item("shape", shape)?;
                dict.set_item("typestr", "<f8")?;
                self.set_array_interface_data(py, &dict, &audio.data)
            }),
        }
    }

    /// Convert to NumPy array
    fn to_numpy<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        match &self.inner {
            PyAudioDataInner::I16(typed) => typed.with_view(py, |audio| match &audio.data {
                AudioData::Mono(arr) => Ok(arr.as_view().to_owned().into_pyarray(py).into_any()),
                AudioData::Multi(arr) => Ok(arr.as_view().to_owned().into_pyarray(py).into_any()),
            }),
            PyAudioDataInner::I24(_) => {
                // Convert I24 to i32 for numpy compatibility
                self.cast_as_i32(py)?.to_numpy(py)
            }
            PyAudioDataInner::I32(typed) => typed.with_view(py, |audio| match &audio.data {
                AudioData::Mono(arr) => Ok(arr.as_view().to_owned().into_pyarray(py).into_any()),
                AudioData::Multi(arr) => Ok(arr.as_view().to_owned().into_pyarray(py).into_any()),
            }),
            PyAudioDataInner::F32(typed) => typed.with_view(py, |audio| match &audio.data {
                AudioData::Mono(arr) => Ok(arr.as_view().to_owned().into_pyarray(py).into_any()),
                AudioData::Multi(arr) => Ok(arr.as_view().to_owned().into_pyarray(py).into_any()),
            }),
            PyAudioDataInner::F64(typed) => typed.with_view(py, |audio| match &audio.data {
                AudioData::Mono(arr) => Ok(arr.as_view().to_owned().into_pyarray(py).into_any()),
                AudioData::Multi(arr) => Ok(arr.as_view().to_owned().into_pyarray(py).into_any()),
            }),
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

impl<T: AudioSample + Element> Add<T> for PyAudioSamples {
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

impl Display for PyAudioSamples {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.inner.fmt(f)
    }
}

#[derive(Debug)]
enum PyAudioBacking<T: AudioSample + Element> {
    OwnedMono(Array1<T>),
    OwnedMulti(Array2<T>),
    NumpyMono(Py<PyArray1<T>>),
    NumpyMulti(Py<PyArray2<T>>),
}

impl<T: AudioSample + Element> Clone for PyAudioBacking<T> {
    fn clone(&self) -> Self {
        use PyAudioBacking::*;

        match self {
            OwnedMono(a) => OwnedMono(a.clone()),
            OwnedMulti(a) => OwnedMulti(a.clone()),
            NumpyMono(handle) => {
                let py = unsafe { Python::assume_attached() };
                let handle = handle.bind(py);
                let c = handle.clone();
                NumpyMono(c.unbind())
            }
            NumpyMulti(handle) => {
                let py = unsafe { Python::assume_attached() };
                let handle = handle.bind(py);
                let c = handle.clone();
                NumpyMulti(c.unbind())
            }
        }
    }
}

impl<T: AudioSample + Element> Add<Self> for PyAudioBacking<T> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        use PyAudioBacking::*;

        match (self, rhs) {
            (OwnedMono(a), OwnedMono(b)) => OwnedMono(a + b),
            (OwnedMulti(a), OwnedMulti(b)) => OwnedMulti(a + b),
            (NumpyMono(a), NumpyMono(b)) => {
                // Safety: If we have two numpy arrays then they must both be bound to the same Python interpreter which is running this code
                let py = unsafe { Python::assume_attached() };
                let a = unsafe { a.bind(py).as_array() };
                let b = unsafe { b.bind(py).as_array() };
                let result = &a + &b;
                NumpyMono(result.into_pyarray(py).unbind())
            }
            (NumpyMulti(a), NumpyMulti(b)) => {
                // Safety: If we have two numpy arrays then they must both be bound to the same Python interpreter which is running this code
                let py = unsafe { Python::assume_attached() };
                let a = unsafe { a.bind(py).as_array() };
                let b = unsafe { b.bind(py).as_array() };
                let result = &a + &b;
                NumpyMulti(result.into_pyarray(py).unbind())
            }
            _ => panic!("Addition not supported for mixed or numpy backings"),
        }
    }
}

impl<T: AudioSample + Element> Add<T> for PyAudioBacking<T> {
    type Output = Self;

    fn add(self, rhs: T) -> Self::Output {
        use PyAudioBacking::*;

        match self {
            OwnedMono(a) => OwnedMono(a + rhs),
            OwnedMulti(a) => OwnedMulti(a + rhs),
            NumpyMono(handle) => {
                // Safety: If we have a numpy array then it must be bound to the same Python interpreter which is running this code
                let py = unsafe { Python::assume_attached() };
                let a = unsafe { handle.bind(py).as_array() };
                let result = &a + rhs;
                NumpyMono(result.into_pyarray(py).unbind())
            }
            NumpyMulti(handle) => {
                // Safety: If we have a numpy array then it must be bound to the same Python interpreter which is running this code
                let py = unsafe { Python::assume_attached() };
                let a = unsafe { handle.bind(py).as_array() };
                let result = &a + rhs;
                NumpyMulti(result.into_pyarray(py).unbind())
            }
        }
    }
}

impl<T: AudioSample + Element> Sub<Self> for PyAudioBacking<T> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        use PyAudioBacking::*;

        match (self, rhs) {
            (OwnedMono(a), OwnedMono(b)) => OwnedMono(a - b),
            (OwnedMulti(a), OwnedMulti(b)) => OwnedMulti(a - b),
            (NumpyMono(a), NumpyMono(b)) => {
                // Safety: If we have two numpy arrays then they must both be bound to the same Python interpreter which is running this code
                let py = unsafe { Python::assume_attached() };
                let a = unsafe { a.bind(py).as_array() };
                let b = unsafe { b.bind(py).as_array() };
                let result = &a - &b;
                NumpyMono(result.into_pyarray(py).unbind())
            }
            (NumpyMulti(a), NumpyMulti(b)) => {
                // Safety: If we have two numpy arrays then they must both be bound to the same Python interpreter which is running this code
                let py = unsafe { Python::assume_attached() };
                let a = unsafe { a.bind(py).as_array() };
                let b = unsafe { b.bind(py).as_array() };
                let result = &a - &b;
                NumpyMulti(result.into_pyarray(py).unbind())
            }
            _ => panic!("Subtraction not supported for mixed or numpy backings"),
        }
    }
}

impl<T: AudioSample + Element> Sub<T> for PyAudioBacking<T> {
    type Output = Self;

    fn sub(self, rhs: T) -> Self::Output {
        use PyAudioBacking::*;

        match self {
            OwnedMono(a) => OwnedMono(a - rhs),
            OwnedMulti(a) => OwnedMulti(a - rhs),
            NumpyMono(handle) => {
                // Safety: If we have a numpy array then it must be bound to the same Python interpreter which is running this code
                let py = unsafe { Python::assume_attached() };
                let a = unsafe { handle.bind(py).as_array() };
                let result = &a - rhs;
                NumpyMono(result.into_pyarray(py).unbind())
            }
            NumpyMulti(handle) => {
                // Safety: If we have a numpy array then it must be bound to the same Python interpreter which is running this code
                let py = unsafe { Python::assume_attached() };
                let a = unsafe { handle.bind(py).as_array() };
                let result = &a - rhs;
                NumpyMulti(result.into_pyarray(py).unbind())
            }
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct TypedAudioSamples<T: AudioSample + Element> {
    backing: PyAudioBacking<T>,
    sample_rate: u32,
    layout: ChannelLayout,
}

impl<T: AudioSample + Element> TypedAudioSamples<T> {
    pub fn with_view<R>(
        &self,
        py: Python<'_>,
        f: impl for<'a> FnOnce(AudioSamples<'a, T>) -> R,
    ) -> R {
        use PyAudioBacking::*;
        let sr = NonZeroU32::new(self.sample_rate).expect("sample_rate must be non-zero");

        match &self.backing {
            OwnedMono(arr) => {
                let view = arr.view();
                let core = AudioSamples {
                    data: AudioData::from_borrowed_array1(view),
                    sample_rate: sr,
                    layout: self.layout,
                };
                f(core)
            }
            OwnedMulti(arr) => {
                let view = arr.view();
                let core = AudioSamples {
                    data: AudioData::from_borrowed_array2(view),
                    sample_rate: sr,
                    layout: self.layout,
                };
                f(core)
            }
            NumpyMono(handle) => {
                let bound = handle.bind(py);
                let view = unsafe { bound.as_array() };
                let core = AudioSamples {
                    data: AudioData::from_borrowed_array1(view),
                    sample_rate: sr,
                    layout: self.layout,
                };
                f(core)
            }
            NumpyMulti(handle) => {
                let bound = handle.bind(py);
                let view = unsafe { bound.as_array() };
                let core = AudioSamples {
                    data: AudioData::from_borrowed_array2(view),
                    sample_rate: sr,
                    layout: self.layout,
                };
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
        let sr = NonZeroU32::new(self.sample_rate).expect("sample_rate must be non-zero");

        match &mut self.backing {
            OwnedMono(arr) => {
                let view = arr.view_mut();
                let core = AudioSamples {
                    data: AudioData::from_borrowed_array1_mut(view),
                    sample_rate: sr,
                    layout: self.layout,
                };
                f(core)
            }
            OwnedMulti(arr) => {
                let view = arr.view_mut();
                let core = AudioSamples {
                    data: AudioData::from_borrowed_array2_mut(view),
                    sample_rate: sr,
                    layout: self.layout,
                };
                f(core)
            }
            NumpyMono(handle) => {
                let bound = handle.bind(py);
                let view = unsafe { bound.as_array_mut() };
                let core = AudioSamples {
                    data: AudioData::from_borrowed_array1_mut(view),
                    sample_rate: sr,
                    layout: self.layout,
                };
                f(core)
            }
            NumpyMulti(handle) => {
                let bound = handle.bind(py);
                let view = unsafe { bound.as_array_mut() };
                let core = AudioSamples {
                    data: AudioData::from_borrowed_array2_mut(view),
                    sample_rate: sr,
                    layout: self.layout,
                };
                f(core)
            }
        }
    }
}

impl<T: AudioSample + Element> Add<Self> for TypedAudioSamples<T> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        TypedAudioSamples {
            backing: self.backing + rhs.backing,
            sample_rate: self.sample_rate,
            layout: self.layout,
        }
    }
}

impl<T: AudioSample + Element> Add<T> for TypedAudioSamples<T> {
    type Output = Self;

    fn add(self, rhs: T) -> Self::Output {
        TypedAudioSamples {
            backing: self.backing + rhs,
            sample_rate: self.sample_rate,
            layout: self.layout,
        }
    }
}

impl<T: AudioSample + Element> Sub<Self> for TypedAudioSamples<T> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        TypedAudioSamples {
            backing: self.backing - rhs.backing,
            sample_rate: self.sample_rate,
            layout: self.layout,
        }
    }
}

impl<T: AudioSample + Element> Sub<T> for TypedAudioSamples<T> {
    type Output = Self;

    fn sub(self, rhs: T) -> Self::Output {
        TypedAudioSamples {
            backing: self.backing - rhs,
            sample_rate: self.sample_rate,
            layout: self.layout,
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) enum PyAudioDataInner {
    I16(TypedAudioSamples<i16>),
    I24(TypedAudioSamples<I24>),
    I32(TypedAudioSamples<i32>),
    F32(TypedAudioSamples<f32>),
    F64(TypedAudioSamples<f64>),
}

impl PyAudioDataInner {
    pub fn dtype<'py>(&'py self, py: Python<'py>) -> Bound<'py, PyArrayDescr> {
        match &self {
            PyAudioDataInner::I16(_) => numpy::dtype::<i16>(py),
            PyAudioDataInner::I24(_) => numpy::dtype::<I24>(py),
            PyAudioDataInner::I32(_) => numpy::dtype::<i32>(py),
            PyAudioDataInner::F32(_) => numpy::dtype::<f32>(py),
            PyAudioDataInner::F64(_) => numpy::dtype::<f64>(py),
        }
    }
}

impl Display for PyAudioDataInner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Python::attach(|py| match self {
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
            (PyAudioDataInner::I16(a), PyAudioDataInner::I16(b)) => PyAudioDataInner::I16(a + b),
            (PyAudioDataInner::I24(a), PyAudioDataInner::I24(b)) => PyAudioDataInner::I24(a + b),
            (PyAudioDataInner::I32(a), PyAudioDataInner::I32(b)) => PyAudioDataInner::I32(a + b),
            (PyAudioDataInner::F32(a), PyAudioDataInner::F32(b)) => PyAudioDataInner::F32(a + b),
            (PyAudioDataInner::F64(a), PyAudioDataInner::F64(b)) => PyAudioDataInner::F64(a + b),
            _ => panic!("Addition not supported for different audio data types"),
        }
    }
}

impl<T: AudioSample + Element> Add<T> for PyAudioDataInner
where
    T: ConvertTo<i16> + ConvertTo<I24> + ConvertTo<i32> + ConvertTo<f32> + ConvertTo<f64>,
{
    type Output = Self;

    fn add(self, rhs: T) -> Self::Output {
        match self {
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
            (PyAudioDataInner::I16(a), PyAudioDataInner::I16(b)) => PyAudioDataInner::I16(a - b),
            (PyAudioDataInner::I24(a), PyAudioDataInner::I24(b)) => PyAudioDataInner::I24(a - b),
            (PyAudioDataInner::I32(a), PyAudioDataInner::I32(b)) => PyAudioDataInner::I32(a - b),
            (PyAudioDataInner::F32(a), PyAudioDataInner::F32(b)) => PyAudioDataInner::F32(a - b),
            (PyAudioDataInner::F64(a), PyAudioDataInner::F64(b)) => PyAudioDataInner::F64(a - b),
            _ => panic!("Subtraction not supported for different audio data types"),
        }
    }
}

impl<T: AudioSample + Element> Sub<T> for PyAudioDataInner
where
    T: ConvertTo<i16> + ConvertTo<I24> + ConvertTo<i32> + ConvertTo<f32> + ConvertTo<f64>,
{
    type Output = Self;

    fn sub(self, rhs: T) -> Self::Output {
        match self {
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

// Configuration wrapper classes

#[pyclass(name = "IirFilterDesign")]
#[derive(Debug)]
pub struct PyIirFilterDesign {
    inner: IirFilterDesign<f64>,
}

impl PyIirFilterDesign {
    pub(crate) const fn inner(&self) -> &IirFilterDesign<f64> {
        &self.inner
    }
}

#[pymethods]
impl PyIirFilterDesign {
    #[new]
    fn new(
        filter_type: &str,
        response: &str,
        order: usize,
        cutoff_frequency: Option<f64>,
        low_frequency: Option<f64>,
        high_frequency: Option<f64>,
    ) -> PyResult<Self> {
        let filter_type = match filter_type.to_lowercase().as_str() {
            "butterworth" => IirFilterType::Butterworth,
            "chebyshev1" => IirFilterType::ChebyshevI,
            "chebyshev2" => IirFilterType::ChebyshevII,
            "elliptic" => IirFilterType::Elliptic,
            _ => {
                return Err(PyTypeError::new_err(
                    "Invalid filter type. Use 'butterworth', 'chebyshev1', 'chebyshev2', or 'elliptic'",
                ));
            }
        };

        let response = match response.to_lowercase().as_str() {
            "lowpass" => FilterResponse::LowPass,
            "highpass" => FilterResponse::HighPass,
            "bandpass" => FilterResponse::BandPass,
            "bandstop" => FilterResponse::BandStop,
            _ => {
                return Err(PyTypeError::new_err(
                    "Invalid response type. Use 'lowpass', 'highpass', 'bandpass', or 'bandstop'",
                ));
            }
        };

        Ok(Self {
            inner: IirFilterDesign {
                filter_type,
                response,
                order,
                cutoff_frequency,
                low_frequency,
                high_frequency,
                passband_ripple: None,
                stopband_attenuation: None,
            },
        })
    }

    #[staticmethod]
    const fn butterworth_lowpass(order: usize, cutoff_frequency: f64) -> Self {
        Self {
            inner: IirFilterDesign::butterworth_lowpass(order, cutoff_frequency),
        }
    }

    #[staticmethod]
    const fn butterworth_highpass(order: usize, cutoff_frequency: f64) -> Self {
        Self {
            inner: IirFilterDesign::butterworth_highpass(order, cutoff_frequency),
        }
    }

    #[staticmethod]
    const fn butterworth_bandpass(order: usize, low_frequency: f64, high_frequency: f64) -> Self {
        Self {
            inner: IirFilterDesign::butterworth_bandpass(order, low_frequency, high_frequency),
        }
    }
}

#[pyclass(name = "EqBand")]
#[derive(Debug)]
pub struct PyEqBand {
    inner: EqBand<f64>,
}

impl PyEqBand {
    pub(crate) const fn inner(&self) -> &EqBand<f64> {
        &self.inner
    }
}

#[pymethods]
impl PyEqBand {
    #[new]
    #[pyo3(signature = (band_type, frequency, gain_db, q_factor), text_signature = "($cls, band_type: str, frequency: float, gain_db: float, q_factor: float) -> EqBand")]
    fn new(band_type: &str, frequency: f64, gain_db: f64, q_factor: f64) -> PyResult<Self> {
        let band_type = match band_type.to_lowercase().as_str() {
            "peak" => EqBandType::Peak,
            "lowshelf" => EqBandType::LowShelf,
            "highshelf" => EqBandType::HighShelf,
            "lowpass" => EqBandType::LowPass,
            "highpass" => EqBandType::HighPass,
            "bandpass" => EqBandType::BandPass,
            "bandstop" => EqBandType::BandStop,
            _ => {
                return Err(PyTypeError::new_err(
                    "Invalid band type. Use 'peak', 'lowshelf', 'highshelf', 'lowpass', 'highpass', 'bandpass', or 'bandstop'",
                ));
            }
        };

        Ok(Self {
            inner: EqBand {
                band_type,
                frequency,
                gain_db,
                q_factor,
                enabled: true,
            },
        })
    }

    #[staticmethod]
    #[pyo3(signature = (frequency, gain_db, q_factor), text_signature = "($cls, frequency: float, gain_db: float, q_factor: float) -> EqBand")]
    const fn peak(frequency: f64, gain_db: f64, q_factor: f64) -> Self {
        Self {
            inner: EqBand::peak(frequency, gain_db, q_factor),
        }
    }

    #[staticmethod]
    #[pyo3(signature = (frequency, gain_db, q_factor), text_signature = "($cls, frequency: float, gain_db: float, q_factor: float) -> EqBand")]
    const fn low_shelf(frequency: f64, gain_db: f64, q_factor: f64) -> Self {
        Self {
            inner: EqBand::low_shelf(frequency, gain_db, q_factor),
        }
    }

    #[staticmethod]
    #[pyo3(signature = (frequency, gain_db, q_factor), text_signature = "($cls, frequency: float, gain_db: float, q_factor: float) -> EqBand")]
    const fn high_shelf(frequency: f64, gain_db: f64, q_factor: f64) -> Self {
        Self {
            inner: EqBand::high_shelf(frequency, gain_db, q_factor),
        }
    }

    #[getter]
    const fn frequency(&self) -> f64 {
        self.inner.frequency
    }

    #[setter]
    const fn set_frequency(&mut self, frequency: f64) {
        self.inner.frequency = frequency;
    }

    #[getter]
    const fn gain_db(&self) -> f64 {
        self.inner.gain_db
    }

    #[setter]
    const fn set_gain_db(&mut self, gain_db: f64) {
        self.inner.gain_db = gain_db;
    }

    #[getter]
    const fn q_factor(&self) -> f64 {
        self.inner.q_factor
    }

    #[setter]
    const fn set_q_factor(&mut self, q_factor: f64) {
        self.inner.q_factor = q_factor;
    }

    #[getter]
    const fn enabled(&self) -> bool {
        self.inner.enabled
    }

    #[setter]
    const fn set_enabled(&mut self, enabled: bool) {
        self.inner.enabled = enabled;
    }
}

#[pyclass(name = "ParametricEq")]
#[derive(Debug)]
pub struct PyParametricEq {
    inner: ParametricEq<f64>,
}

impl PyParametricEq {
    pub(crate) const fn inner(&self) -> &ParametricEq<f64> {
        &self.inner
    }
}

#[pymethods]
impl PyParametricEq {
    #[new]
    fn new() -> Self {
        Self {
            inner: ParametricEq::new(),
        }
    }

    fn add_band(&mut self, band: &PyEqBand) {
        self.inner.add_band(band.inner.clone());
    }

    fn remove_band(&mut self, index: usize) -> Option<PyEqBand> {
        self.inner
            .remove_band(index)
            .map(|inner| PyEqBand { inner })
    }

    #[getter]
    const fn output_gain_db(&self) -> f64 {
        self.inner.output_gain_db
    }

    #[setter]
    const fn set_output_gain_db(&mut self, gain_db: f64) {
        self.inner.output_gain_db = gain_db;
    }

    #[getter]
    const fn bypassed(&self) -> bool {
        self.inner.bypassed
    }

    #[setter]
    const fn set_bypassed(&mut self, bypassed: bool) {
        self.inner.bypassed = bypassed;
    }

    const fn __len__(&self) -> usize {
        self.inner.bands.len()
    }
}

/// Generate a sine wave audio signal.
///
/// Args:
///     frequency: Frequency of the sine wave in Hz
///     duration_secs: Duration of the signal in seconds
///     sample_rate: Sample rate in samples per second (default: 44100)
///     amplitude: Peak amplitude of the wave (default: 1.0)
///     dtype: NumPy dtype for the output array (default: f64)
///
/// Returns:
///     PyAudioSamples: Generated sine wave audio data
#[pyfunction]
#[pyo3(signature = (frequency, duration_secs, sample_rate=44100, amplitude=1.0, dtype=None), text_signature = "(frequency: float, duration_secs: float, sample_rate: int = 44100, amplitude: float = 1.0, dtype: Optional[numpy.dtype] = None) -> PyAudioSamples")]
pub fn sine_wave(
    py: Python<'_>,
    frequency: f64,
    duration_secs: f64,
    sample_rate: u32,
    amplitude: f64,
    dtype: Option<Bound<'_, PyArrayDescr>>,
) -> PyResult<PyAudioSamples> {
    let dtype = dtype.unwrap_or(numpy::dtype::<f64>(py));
    let duration = Duration::from_secs_f64(duration_secs);
    if dtype.is_equiv_to(&numpy::dtype::<i16>(py)) {
        let audio =
            audio_samples::sine_wave::<i16, f64>(frequency, duration, sample_rate, amplitude);
        PyAudioSamples::from_audio_samples(audio)
    } else if dtype.is_equiv_to(&numpy::dtype::<I24>(py)) {
        let audio =
            audio_samples::sine_wave::<I24, f64>(frequency, duration, sample_rate, amplitude);
        PyAudioSamples::from_audio_samples(audio)
    } else if dtype.is_equiv_to(&numpy::dtype::<i32>(py)) {
        let audio =
            audio_samples::sine_wave::<i32, f64>(frequency, duration, sample_rate, amplitude);
        PyAudioSamples::from_audio_samples(audio)
    } else if dtype.is_equiv_to(&numpy::dtype::<f32>(py)) {
        let audio =
            audio_samples::sine_wave::<f32, f64>(frequency, duration, sample_rate, amplitude);
        PyAudioSamples::from_audio_samples(audio)
    } else if dtype.is_equiv_to(&numpy::dtype::<f64>(py)) {
        let audio =
            audio_samples::sine_wave::<f64, f64>(frequency, duration, sample_rate, amplitude);
        PyAudioSamples::from_audio_samples(audio)
    } else {
        Err(PyTypeError::new_err(
            "Unsupported dtype for sine wave generation",
        ))
    }
}

/// Generate a cosine wave audio signal.
///
/// Args:
///     frequency: Frequency of the cosine wave in Hz
///     duration_secs: Duration of the signal in seconds
///     sample_rate: Sample rate in samples per second (default: 44100)
///     amplitude: Peak amplitude of the wave (default: 1.0)
///     dtype: NumPy dtype for the output array (default: f64)
///
/// Returns:
///     PyAudioSamples: Generated cosine wave audio data
#[pyfunction]
#[pyo3(signature = (frequency, duration_secs, sample_rate=44100, amplitude=1.0, dtype=None), text_signature = "(frequency: float, duration_secs: float, sample_rate: int = 44100, amplitude: float = 1.0, dtype: Optional[numpy.dtype] = None) -> PyAudioSamples")]
fn cosine_wave(
    py: Python<'_>,
    frequency: f64,
    duration_secs: f64,
    sample_rate: u32,
    amplitude: f64,
    dtype: Option<Bound<'_, PyArrayDescr>>,
) -> PyResult<PyAudioSamples> {
    let dtype = dtype.unwrap_or(numpy::dtype::<f64>(py));
    let duration = Duration::from_secs_f64(duration_secs);

    if dtype.is_equiv_to(&numpy::dtype::<i16>(py)) {
        let audio =
            audio_samples::cosine_wave::<i16, f64>(frequency, duration, sample_rate, amplitude);
        PyAudioSamples::from_audio_samples(audio)
    } else if dtype.is_equiv_to(&numpy::dtype::<I24>(py)) {
        let audio =
            audio_samples::cosine_wave::<I24, f64>(frequency, duration, sample_rate, amplitude);
        PyAudioSamples::from_audio_samples(audio)
    } else if dtype.is_equiv_to(&numpy::dtype::<i32>(py)) {
        let audio =
            audio_samples::cosine_wave::<i32, f64>(frequency, duration, sample_rate, amplitude);
        PyAudioSamples::from_audio_samples(audio)
    } else if dtype.is_equiv_to(&numpy::dtype::<f32>(py)) {
        let audio =
            audio_samples::cosine_wave::<f32, f64>(frequency, duration, sample_rate, amplitude);
        PyAudioSamples::from_audio_samples(audio)
    } else if dtype.is_equiv_to(&numpy::dtype::<f64>(py)) {
        let audio =
            audio_samples::cosine_wave::<f64, f64>(frequency, duration, sample_rate, amplitude);
        PyAudioSamples::from_audio_samples(audio)
    } else {
        Err(PyTypeError::new_err(
            "Unsupported dtype for cosine wave generation",
        ))
    }
}

/// Generate a sawtooth wave audio signal.
///
/// Args:
///     frequency: Frequency of the sawtooth wave in Hz
///     duration_secs: Duration of the signal in seconds
///     sample_rate: Sample rate in samples per second (default: 44100)
///     amplitude: Peak amplitude of the wave (default: 1.0)
///     dtype: NumPy dtype for the output array (default: f64)
///
/// Returns:
///     PyAudioSamples: Generated sawtooth wave audio data
#[pyfunction]
#[pyo3(signature = (frequency, duration_secs, sample_rate=44100, amplitude=1.0, dtype=None), text_signature = "(frequency: float, duration_secs: float, sample_rate: int = 44100, amplitude: float = 1.0, dtype: Optional[numpy.dtype] = None) -> PyAudioSamples")]
fn sawtooth_wave(
    py: Python<'_>,
    frequency: f64,
    duration_secs: f64,
    sample_rate: u32,
    amplitude: f64,
    dtype: Option<Bound<'_, PyArrayDescr>>,
) -> PyResult<PyAudioSamples> {
    let dtype = dtype.unwrap_or(numpy::dtype::<f64>(py));
    let duration = Duration::from_secs_f64(duration_secs);

    if dtype.is_equiv_to(&numpy::dtype::<i16>(py)) {
        let audio =
            audio_samples::sawtooth_wave::<i16, f64>(frequency, duration, sample_rate, amplitude);
        PyAudioSamples::from_audio_samples(audio)
    } else if dtype.is_equiv_to(&numpy::dtype::<I24>(py)) {
        let audio =
            audio_samples::sawtooth_wave::<I24, f64>(frequency, duration, sample_rate, amplitude);
        PyAudioSamples::from_audio_samples(audio)
    } else if dtype.is_equiv_to(&numpy::dtype::<i32>(py)) {
        let audio =
            audio_samples::sawtooth_wave::<i32, f64>(frequency, duration, sample_rate, amplitude);
        PyAudioSamples::from_audio_samples(audio)
    } else if dtype.is_equiv_to(&numpy::dtype::<f32>(py)) {
        let audio =
            audio_samples::sawtooth_wave::<f32, f64>(frequency, duration, sample_rate, amplitude);
        PyAudioSamples::from_audio_samples(audio)
    } else if dtype.is_equiv_to(&numpy::dtype::<f64>(py)) {
        let audio =
            audio_samples::sawtooth_wave::<f64, f64>(frequency, duration, sample_rate, amplitude);
        PyAudioSamples::from_audio_samples(audio)
    } else {
        Err(PyTypeError::new_err(
            "Unsupported dtype for sawtooth wave generation",
        ))
    }
}

/// Generate a square wave audio signal.
///
/// Args:
///     frequency: Frequency of the square wave in Hz
///     duration_secs: Duration of the signal in seconds
///     sample_rate: Sample rate in samples per second (default: 44100)
///     amplitude: Peak amplitude of the wave (default: 1.0)
///     dtype: NumPy dtype for the output array (default: f64)
///
/// Returns:
///     PyAudioSamples: Generated square wave audio data
#[pyfunction]
#[pyo3(signature = (frequency, duration_secs, sample_rate=44100, amplitude=1.0, dtype=None), text_signature = "(frequency: float, duration_secs: float, sample_rate: int = 44100, amplitude: float = 1.0, dtype: Optional[numpy.dtype] = None) -> PyAudioSamples")]
fn square_wave(
    py: Python<'_>,
    frequency: f64,
    duration_secs: f64,
    sample_rate: u32,
    amplitude: f64,
    dtype: Option<Bound<'_, PyArrayDescr>>,
) -> PyResult<PyAudioSamples> {
    let dtype = dtype.unwrap_or(numpy::dtype::<f64>(py));
    let duration = Duration::from_secs_f64(duration_secs);

    if dtype.is_equiv_to(&numpy::dtype::<i16>(py)) {
        let audio =
            audio_samples::square_wave::<i16, f64>(frequency, duration, sample_rate, amplitude);
        PyAudioSamples::from_audio_samples(audio)
    } else if dtype.is_equiv_to(&numpy::dtype::<I24>(py)) {
        let audio =
            audio_samples::square_wave::<I24, f64>(frequency, duration, sample_rate, amplitude);
        PyAudioSamples::from_audio_samples(audio)
    } else if dtype.is_equiv_to(&numpy::dtype::<i32>(py)) {
        let audio =
            audio_samples::square_wave::<i32, f64>(frequency, duration, sample_rate, amplitude);
        PyAudioSamples::from_audio_samples(audio)
    } else if dtype.is_equiv_to(&numpy::dtype::<f32>(py)) {
        let audio =
            audio_samples::square_wave::<f32, f64>(frequency, duration, sample_rate, amplitude);
        PyAudioSamples::from_audio_samples(audio)
    } else if dtype.is_equiv_to(&numpy::dtype::<f64>(py)) {
        let audio =
            audio_samples::square_wave::<f64, f64>(frequency, duration, sample_rate, amplitude);
        PyAudioSamples::from_audio_samples(audio)
    } else {
        Err(PyTypeError::new_err(
            "Unsupported dtype for square wave generation",
        ))
    }
}

/// Generate a triangle wave audio signal.
///
/// Args:
///     frequency: Frequency of the triangle wave in Hz
///     duration_secs: Duration of the signal in seconds
///     sample_rate: Sample rate in samples per second (default: 44100)
///     amplitude: Peak amplitude of the wave (default: 1.0)
///     dtype: NumPy dtype for the output array (default: f64)
///
/// Returns:
///     PyAudioSamples: Generated triangle wave audio data
#[pyfunction]
#[pyo3(signature = (frequency, duration_secs, sample_rate=44100, amplitude=1.0, dtype=None), text_signature = "(frequency: float, duration_secs: float, sample_rate: int = 44100, amplitude: float = 1.0, dtype: Optional[numpy.dtype] = None) -> PyAudioSamples")]
fn triangle_wave(
    py: Python<'_>,
    frequency: f64,
    duration_secs: f64,
    sample_rate: u32,
    amplitude: f64,
    dtype: Option<Bound<'_, PyArrayDescr>>,
) -> PyResult<PyAudioSamples> {
    let dtype = dtype.unwrap_or(numpy::dtype::<f64>(py));
    let duration = Duration::from_secs_f64(duration_secs);

    if dtype.is_equiv_to(&numpy::dtype::<i16>(py)) {
        let audio =
            audio_samples::triangle_wave::<i16, f64>(frequency, duration, sample_rate, amplitude);
        PyAudioSamples::from_audio_samples(audio)
    } else if dtype.is_equiv_to(&numpy::dtype::<I24>(py)) {
        let audio =
            audio_samples::triangle_wave::<I24, f64>(frequency, duration, sample_rate, amplitude);
        PyAudioSamples::from_audio_samples(audio)
    } else if dtype.is_equiv_to(&numpy::dtype::<i32>(py)) {
        let audio =
            audio_samples::triangle_wave::<i32, f64>(frequency, duration, sample_rate, amplitude);
        PyAudioSamples::from_audio_samples(audio)
    } else if dtype.is_equiv_to(&numpy::dtype::<f32>(py)) {
        let audio =
            audio_samples::triangle_wave::<f32, f64>(frequency, duration, sample_rate, amplitude);
        PyAudioSamples::from_audio_samples(audio)
    } else if dtype.is_equiv_to(&numpy::dtype::<f64>(py)) {
        let audio =
            audio_samples::triangle_wave::<f64, f64>(frequency, duration, sample_rate, amplitude);
        PyAudioSamples::from_audio_samples(audio)
    } else {
        Err(PyTypeError::new_err(
            "Unsupported dtype for triangle wave generation",
        ))
    }
}

/// Generate a frequency chirp (sweep) audio signal.
///
/// Args:
///     f0: Starting frequency in Hz
///     f1: Ending frequency in Hz
///     duration_secs: Duration of the signal in seconds
///     sample_rate: Sample rate in samples per second (default: 44100)
///     amplitude: Peak amplitude of the wave (default: 1.0)
///     dtype: NumPy dtype for the output array (default: f64)
///
/// Returns:
///     PyAudioSamples: Generated chirp audio data
#[pyfunction]
#[pyo3(signature = (f0, f1, duration_secs, sample_rate=44100, amplitude=1.0, dtype=None), text_signature = "(f0: float, f1: float, duration_secs: float, sample_rate: int = 44100, amplitude: float = 1.0, dtype: Optional[numpy.dtype] = None) -> PyAudioSamples")]
fn chirp(
    py: Python<'_>,
    f0: f64,
    f1: f64,
    duration_secs: f64,
    sample_rate: u32,
    amplitude: f64,
    dtype: Option<Bound<'_, PyArrayDescr>>,
) -> PyResult<PyAudioSamples> {
    let dtype = dtype.unwrap_or(numpy::dtype::<f64>(py));
    let duration = Duration::from_secs_f64(duration_secs);

    if dtype.is_equiv_to(&numpy::dtype::<i16>(py)) {
        let audio = audio_samples::chirp::<i16, f64>(f0, f1, duration, sample_rate, amplitude);
        PyAudioSamples::from_audio_samples(audio)
    } else if dtype.is_equiv_to(&numpy::dtype::<I24>(py)) {
        let audio = audio_samples::chirp::<I24, f64>(f0, f1, duration, sample_rate, amplitude);
        PyAudioSamples::from_audio_samples(audio)
    } else if dtype.is_equiv_to(&numpy::dtype::<i32>(py)) {
        let audio = audio_samples::chirp::<i32, f64>(f0, f1, duration, sample_rate, amplitude);
        PyAudioSamples::from_audio_samples(audio)
    } else if dtype.is_equiv_to(&numpy::dtype::<f32>(py)) {
        let audio = audio_samples::chirp::<f32, f64>(f0, f1, duration, sample_rate, amplitude);
        PyAudioSamples::from_audio_samples(audio)
    } else if dtype.is_equiv_to(&numpy::dtype::<f64>(py)) {
        let audio = audio_samples::chirp::<f64, f64>(f0, f1, duration, sample_rate, amplitude);
        PyAudioSamples::from_audio_samples(audio)
    } else {
        Err(PyTypeError::new_err(
            "Unsupported dtype for chirp generation",
        ))
    }
}

/// Generate white noise audio signal.
///
/// Args:
///     duration_secs: Duration of the signal in seconds
///     sample_rate: Sample rate in samples per second (default: 44100)
///     amplitude: Peak amplitude of the noise (default: 1.0)
///     dtype: NumPy dtype for the output array (default: f64)
///     seed: Optional seed for reproducible noise generation
///
/// Returns:
///     PyAudioSamples: Generated white noise audio data
#[pyfunction]
#[pyo3(signature = (duration_secs, sample_rate=44100, amplitude=1.0, dtype=None, seed=None), text_signature = "(duration_secs: float, sample_rate: int = 44100, amplitude: float = 1.0, dtype: Optional[numpy.dtype] = None, seed: Optional[int] = None) -> PyAudioSamples")]
fn white_noise(
    py: Python<'_>,
    duration_secs: f64,
    sample_rate: u32,
    amplitude: f64,
    dtype: Option<Bound<'_, PyArrayDescr>>,
    seed: Option<u64>,
) -> PyResult<PyAudioSamples> {
    let dtype = dtype.unwrap_or(numpy::dtype::<f64>(py));
    let duration = Duration::from_secs_f64(duration_secs);

    if dtype.is_equiv_to(&numpy::dtype::<i16>(py)) {
        let audio = audio_samples::white_noise::<i16, f64>(duration, sample_rate, amplitude, seed);
        PyAudioSamples::from_audio_samples(audio)
    } else if dtype.is_equiv_to(&numpy::dtype::<I24>(py)) {
        let audio = audio_samples::white_noise::<I24, f64>(duration, sample_rate, amplitude, seed);
        PyAudioSamples::from_audio_samples(audio)
    } else if dtype.is_equiv_to(&numpy::dtype::<i32>(py)) {
        let audio = audio_samples::white_noise::<i32, f64>(duration, sample_rate, amplitude, seed);
        PyAudioSamples::from_audio_samples(audio)
    } else if dtype.is_equiv_to(&numpy::dtype::<f32>(py)) {
        let audio = audio_samples::white_noise::<f32, f64>(duration, sample_rate, amplitude, seed);
        PyAudioSamples::from_audio_samples(audio)
    } else if dtype.is_equiv_to(&numpy::dtype::<f64>(py)) {
        let audio = audio_samples::white_noise::<f64, f64>(duration, sample_rate, amplitude, seed);
        PyAudioSamples::from_audio_samples(audio)
    } else {
        Err(PyTypeError::new_err(
            "Unsupported dtype for white noise generation",
        ))
    }
}

/// Generate pink noise audio signal.
///
/// Args:
///     duration_secs: Duration of the signal in seconds
///     sample_rate: Sample rate in samples per second (default: 44100)
///     amplitude: Peak amplitude of the noise (default: 1.0)
///     dtype: NumPy dtype for the output array (default: f64)
///
/// Returns:
///     PyAudioSamples: Generated pink noise audio data
#[pyfunction]
#[pyo3(signature = (duration_secs, sample_rate=44100, amplitude=1.0, dtype=None), text_signature = "(duration_secs: float, sample_rate: int = 44100, amplitude: float = 1.0, dtype: Optional[numpy.dtype] = None) -> PyAudioSamples")]
fn pink_noise(
    py: Python<'_>,
    duration_secs: f64,
    sample_rate: u32,
    amplitude: f64,
    dtype: Option<Bound<'_, PyArrayDescr>>,
) -> PyResult<PyAudioSamples> {
    let dtype = dtype.unwrap_or(numpy::dtype::<f64>(py));
    let duration = Duration::from_secs_f64(duration_secs);

    if dtype.is_equiv_to(&numpy::dtype::<i16>(py)) {
        let audio = audio_samples::pink_noise::<i16, f64>(duration, sample_rate, amplitude);
        PyAudioSamples::from_audio_samples(audio)
    } else if dtype.is_equiv_to(&numpy::dtype::<I24>(py)) {
        let audio = audio_samples::pink_noise::<I24, f64>(duration, sample_rate, amplitude);
        PyAudioSamples::from_audio_samples(audio)
    } else if dtype.is_equiv_to(&numpy::dtype::<i32>(py)) {
        let audio = audio_samples::pink_noise::<i32, f64>(duration, sample_rate, amplitude);
        PyAudioSamples::from_audio_samples(audio)
    } else if dtype.is_equiv_to(&numpy::dtype::<f32>(py)) {
        let audio = audio_samples::pink_noise::<f32, f64>(duration, sample_rate, amplitude);
        PyAudioSamples::from_audio_samples(audio)
    } else if dtype.is_equiv_to(&numpy::dtype::<f64>(py)) {
        let audio = audio_samples::pink_noise::<f64, f64>(duration, sample_rate, amplitude);
        PyAudioSamples::from_audio_samples(audio)
    } else {
        Err(PyTypeError::new_err(
            "Unsupported dtype for pink noise generation",
        ))
    }
}

/// Generate brown noise (Brownian/red noise) audio signal.
///
/// Args:
///     duration_secs: Duration of the signal in seconds
///     sample_rate: Sample rate in samples per second (default: 44100)
///     step: Step size for the random walk (default: 0.01)
///     amplitude: Peak amplitude of the noise (default: 1.0)
///     dtype: NumPy dtype for the output array (default: f64)
///
/// Returns:
///     PyAudioSamples: Generated brown noise audio data
#[pyfunction]
#[pyo3(signature = (duration_secs, sample_rate=44100, step=0.01, amplitude=1.0, dtype=None), text_signature = "(duration_secs: float, sample_rate: int = 44100, step: float = 0.01, amplitude: float = 1.0, dtype: Optional[numpy.dtype] = None) -> PyAudioSamples")]
fn brown_noise(
    py: Python<'_>,
    duration_secs: f64,
    sample_rate: u32,
    step: f64,
    amplitude: f64,
    dtype: Option<Bound<'_, PyArrayDescr>>,
) -> PyResult<PyAudioSamples> {
    let dtype = dtype.unwrap_or(numpy::dtype::<f64>(py));
    let duration = Duration::from_secs_f64(duration_secs);

    if dtype.is_equiv_to(&numpy::dtype::<i16>(py)) {
        let audio = audio_samples::brown_noise::<i16, f64>(duration, sample_rate, step, amplitude)
            .map_err(|err| {
                PyTypeError::new_err(format!("Error generating brown noise: {}", err))
            })?;
        PyAudioSamples::from_audio_samples(audio)
    } else if dtype.is_equiv_to(&numpy::dtype::<I24>(py)) {
        let audio = audio_samples::brown_noise::<I24, f64>(duration, sample_rate, step, amplitude)
            .map_err(|err| {
                PyTypeError::new_err(format!("Error generating brown noise: {}", err))
            })?;
        PyAudioSamples::from_audio_samples(audio)
    } else if dtype.is_equiv_to(&numpy::dtype::<i32>(py)) {
        let audio = audio_samples::brown_noise::<i32, f64>(duration, sample_rate, step, amplitude)
            .map_err(|err| {
                PyTypeError::new_err(format!("Error generating brown noise: {}", err))
            })?;

        PyAudioSamples::from_audio_samples(audio)
    } else if dtype.is_equiv_to(&numpy::dtype::<f32>(py)) {
        let audio = audio_samples::brown_noise::<f32, f64>(duration, sample_rate, step, amplitude)
            .map_err(|err| {
                PyTypeError::new_err(format!("Error generating brown noise: {}", err))
            })?;
        PyAudioSamples::from_audio_samples(audio)
    } else if dtype.is_equiv_to(&numpy::dtype::<f64>(py)) {
        let audio = audio_samples::brown_noise::<f64, f64>(duration, sample_rate, step, amplitude)
            .map_err(|err| {
                PyTypeError::new_err(format!("Error generating brown noise: {}", err))
            })?;
        PyAudioSamples::from_audio_samples(audio)
    } else {
        Err(PyTypeError::new_err(
            "Unsupported dtype for brown noise generation",
        ))
    }
}

/// Generate an impulse (delta function) audio signal.
///
/// Args:
///     duration_secs: Duration of the signal in seconds
///     sample_rate: Sample rate in samples per second (default: 44100)
///     amplitude: Peak amplitude of the impulse (default: 1.0)
///     position: Position of the impulse as fraction of duration (default: 0.5)
///     dtype: NumPy dtype for the output array (default: f64)
///
/// Returns:
///     PyAudioSamples: Generated impulse audio data
#[pyfunction]
#[pyo3(signature = (duration_secs, sample_rate=44100, amplitude=1.0, position=0.5, dtype=None), text_signature = "(duration_secs: float, sample_rate: int = 44100, amplitude: float = 1.0, position: float = 0.5, dtype: Optional[numpy.dtype] = None) -> PyAudioSamples")]
fn impulse(
    py: Python<'_>,
    duration_secs: f64,
    sample_rate: u32,
    amplitude: f64,
    position: f64,
    dtype: Option<Bound<'_, PyArrayDescr>>,
) -> PyResult<PyAudioSamples> {
    let dtype = dtype.unwrap_or(numpy::dtype::<f64>(py));
    let duration = Duration::from_secs_f64(duration_secs);

    if dtype.is_equiv_to(&numpy::dtype::<i16>(py)) {
        let audio = audio_samples::impulse::<i16, f64>(duration, sample_rate, amplitude, position);
        PyAudioSamples::from_audio_samples(audio)
    } else if dtype.is_equiv_to(&numpy::dtype::<I24>(py)) {
        let audio = audio_samples::impulse::<I24, f64>(duration, sample_rate, amplitude, position);
        PyAudioSamples::from_audio_samples(audio)
    } else if dtype.is_equiv_to(&numpy::dtype::<i32>(py)) {
        let audio = audio_samples::impulse::<i32, f64>(duration, sample_rate, amplitude, position);
        PyAudioSamples::from_audio_samples(audio)
    } else if dtype.is_equiv_to(&numpy::dtype::<f32>(py)) {
        let audio = audio_samples::impulse::<f32, f64>(duration, sample_rate, amplitude, position);
        PyAudioSamples::from_audio_samples(audio)
    } else if dtype.is_equiv_to(&numpy::dtype::<f64>(py)) {
        let audio = audio_samples::impulse::<f64, f64>(duration, sample_rate, amplitude, position);
        PyAudioSamples::from_audio_samples(audio)
    } else {
        Err(PyTypeError::new_err(
            "Unsupported dtype for impulse generation",
        ))
    }
}

/// Generate silence (zero amplitude) audio signal.
///
/// Args:
///     duration_secs: Duration of the signal in seconds
///     sample_rate: Sample rate in samples per second (default: 44100)
///     dtype: NumPy dtype for the output array (default: f64)
///
/// Returns:
///     PyAudioSamples: Generated silence audio data
#[pyfunction]
#[pyo3(signature = (duration_secs, sample_rate=44100, dtype=None), text_signature = "(duration_secs: float, sample_rate: int = 44100, dtype: Optional[numpy.dtype] = None) -> PyAudioSamples")]
fn silence(
    py: Python<'_>,
    duration_secs: f64,
    sample_rate: u32,
    dtype: Option<Bound<'_, PyArrayDescr>>,
) -> PyResult<PyAudioSamples> {
    let dtype = dtype.unwrap_or(numpy::dtype::<f64>(py));
    let duration = Duration::from_secs_f64(duration_secs);

    if dtype.is_equiv_to(&numpy::dtype::<i16>(py)) {
        let audio = audio_samples::silence::<i16, f64>(duration, sample_rate);
        PyAudioSamples::from_audio_samples(audio)
    } else if dtype.is_equiv_to(&numpy::dtype::<I24>(py)) {
        let audio = audio_samples::silence::<I24, f64>(duration, sample_rate);
        PyAudioSamples::from_audio_samples(audio)
    } else if dtype.is_equiv_to(&numpy::dtype::<i32>(py)) {
        let audio = audio_samples::silence::<i32, f64>(duration, sample_rate);
        PyAudioSamples::from_audio_samples(audio)
    } else if dtype.is_equiv_to(&numpy::dtype::<f32>(py)) {
        let audio = audio_samples::silence::<f32, f64>(duration, sample_rate);
        PyAudioSamples::from_audio_samples(audio)
    } else if dtype.is_equiv_to(&numpy::dtype::<f64>(py)) {
        let audio = audio_samples::silence::<f64, f64>(duration, sample_rate);
        PyAudioSamples::from_audio_samples(audio)
    } else {
        Err(PyTypeError::new_err(
            "Unsupported dtype for silence generation",
        ))
    }
}
