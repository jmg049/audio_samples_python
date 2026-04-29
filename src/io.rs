use crate::{
    PyAudioSamples, audio_io_err_to_py, dispatch_with_view, impl_py_repr, impl_py_wrapper_core,
    reexport, types::PySampleType,
};
use audio_samples::{AudioTypeConversion, I24, traits::StandardSample};
use audio_samples_io::types::BaseAudioInfo;
use numpy::{Element, PyArrayDescr, PyArrayMethods};
use pyo3::{
    Bound, PyResult, Python,
    exceptions::PyTypeError,
    pyclass, pyfunction, pymethods, pymodule,
    types::{PyAnyMethods, PyModule, PyModuleMethods},
};

macro_rules! dispatch_sample_type {
    ($ty:expr, |$T:ident| $body:expr) => {{
        match $ty {
            PySampleType::U8 => {
                type $T = u8;
                $body
            }
            PySampleType::I16 => {
                type $T = i16;
                $body
            }
            PySampleType::I24 => {
                type $T = I24;
                $body
            }
            PySampleType::I32 => {
                type $T = i32;
                $body
            }
            PySampleType::F32 => {
                type $T = f32;
                $body
            }
            PySampleType::F64 => {
                type $T = f64;
                $body
            }
        }
    }};
}

fn read_typed(py: Python<'_>, fp: &str, target: PySampleType) -> PyResult<PyAudioSamples> {
    dispatch_sample_type!(target, |T| read_with_numpy_backing::<T>(py, fp))
}

/// Convert a natively-typed array (from `read_pyarray_native`) into a `PyAudioSamples`.
/// Mono arrays are reshaped from (1, N) to (N,); multi-channel arrays are kept as-is.
#[cfg(target_endian = "little")]
fn native_array_to_py_audio_samples(
    py: Python<'_>,
    native: audio_samples_io::NativeAudioArray,
) -> PyResult<PyAudioSamples> {
    use audio_samples_io::NativeAudioArray;
    match native {
        NativeAudioArray::U8(arr, info) => native_to_audio_samples(py, arr, info),
        NativeAudioArray::I16(arr, info) => native_to_audio_samples(py, arr, info),
        NativeAudioArray::I32(arr, info) => native_to_audio_samples(py, arr, info),
        NativeAudioArray::F32(arr, info) => native_to_audio_samples(py, arr, info),
        NativeAudioArray::F64(arr, info) => native_to_audio_samples(py, arr, info),
    }
}

#[cfg(target_endian = "little")]
fn native_to_audio_samples<T>(
    py: Python<'_>,
    arr: pyo3::Py<numpy::PyArray2<T>>,
    info: audio_samples_io::types::BaseAudioInfo,
) -> PyResult<PyAudioSamples>
where
    T: audio_samples::traits::StandardSample + numpy::Element + 'static,
{
    use numpy::PyArrayMethods;
    let sample_rate = info.sample_rate;
    let channels = info.channels as usize;
    if channels == 1 {
        let bound = arr.bind(py);
        let array_1d = bound.reshape([info.total_samples])?;
        Ok(PyAudioSamples::new_mono_from_python(
            array_1d.to_owned(),
            sample_rate,
        ))
    } else {
        Ok(PyAudioSamples::new_multi_from_python_interleaved(
            arr.bind(py).to_owned(),
            sample_rate,
        ))
    }
}

#[pyfunction]
#[pyo3(signature = (fp: "str|Path", as_type:"SampleType"=None), text_signature = "(fp, as_type=None) -> AudioSamples")]
pub fn read(py: Python<'_>, fp: &str, as_type: Option<PySampleType>) -> PyResult<PyAudioSamples> {
    // Fast path: no type conversion requested → single-pass (one open, one header parse, one read).
    // This eliminates the extra File::open + header parse that peek_native_type() would add.
    #[cfg(target_endian = "little")]
    if as_type.is_none() {
        if let Some(result) = audio_samples_io::read_pyarray_native(py, std::path::Path::new(fp)) {
            return native_array_to_py_audio_samples(py, result?);
        }
        // Falls through for I24 or any format read_pyarray_native doesn't handle.
    }

    // Slow path: type conversion requested, or fast path not applicable.
    let target = match as_type {
        Some(t) => t,
        None => {
            let native = audio_samples_io::peek_native_type(fp).map_err(|e| {
                pyo3::exceptions::PyTypeError::new_err(format!("Failed to detect native type: {e}"))
            })?;
            let sample_type: audio_samples::SampleType = native.into();
            PySampleType::from_native(sample_type).ok_or_else(|| {
                pyo3::exceptions::PyTypeError::new_err("Unsupported native sample type")
            })?
        }
    };
    read_typed(py, fp, target)
}

#[pyfunction]
#[pyo3(signature = (fp: "str|Path", as_type: "SampleType" = None), text_signature = "(fp, as_type=None) -> (AudioSamples, AudioInfo)")]
pub fn read_with_info(
    py: Python<'_>,
    fp: &str,
    as_type: Option<Bound<'_, PyArrayDescr>>,
) -> PyResult<(PyAudioSamples, PyAudioInfo)> {
    let info = audio_samples_io::info(fp)
        .map_err(|e| PyTypeError::new_err(format!("Failed to get audio info: {e}")))?;

    let native = PySampleType::from_native(info.sample_type)
        .ok_or_else(|| PyTypeError::new_err("Unsupported native sample type"))?;

    let target = match as_type {
        Some(dt) => PySampleType::from_numpy(py, &dt)?,
        None => native,
    };

    let samples = read_typed(py, fp, target)?;
    Ok((samples, PyAudioInfo::from(info)))
}

/// Audio file information structure
#[pyclass(from_py_object, frozen, module = "audio_samples.io")]
#[pyo3(name = "AudioInfo")]
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct PyAudioInfo {
    pub(crate) inner: BaseAudioInfo,
}

#[pymethods]
impl PyAudioInfo {
    /// Sample rate in Hz
    #[getter]
    const fn sample_rate(&self) -> u32 {
        self.inner.sample_rate.get()
    }

    /// Number of audio channels
    #[getter]
    const fn channels(&self) -> u16 {
        self.inner.channels
    }

    /// Bits per sample
    #[getter]
    const fn bits_per_sample(&self) -> u16 {
        self.inner.bits_per_sample
    }

    /// Total number of samples per channel
    #[getter]
    const fn num_samples(&self) -> usize {
        self.inner.total_samples
    }

    /// Duration in seconds
    #[getter]
    const fn duration(&self) -> f64 {
        self.inner.duration.as_secs_f64()
    }

    /// Sample type as a string (e.g., "i16", "f32")
    #[getter]
    const fn sample_type(&self) -> &'static str {
        self.inner.sample_type.as_str()
    }
}

impl_py_wrapper_core!(PyAudioInfo, BaseAudioInfo);
impl_py_repr!(PyAudioInfo);

#[pyfunction]
#[pyo3(signature = (fp: "str|Path"), text_signature = "(fp: str | Path) -> AudioInfo")]
pub fn info(fp: &str) -> PyResult<PyAudioInfo> {
    let info = audio_samples_io::info(fp)
        .map_err(|e| PyTypeError::new_err(format!("Failed to get audio info: {e}")))?;
    Ok(PyAudioInfo::from(info))
}

fn read_with_numpy_backing<T>(py: Python<'_>, fp: &str) -> PyResult<PyAudioSamples>
where
    T: StandardSample + Element + 'static,
{
    let (pyarray, info) = audio_samples_io::read_pyarray::<_, T>(py, fp)?;

    let sample_rate = info.sample_rate;
    let channels = info.channels as usize;

    if channels == 1 {
        // Mono: reshape (1, samples) to (samples,) for PyArray1
        let pyarray_bound = pyarray.bind(py);
        let array_1d = pyarray_bound.reshape([info.total_samples])?;

        Ok(PyAudioSamples::new_mono_from_python(
            array_1d.to_owned(),
            sample_rate,
        ))
    } else {
        // Multi-channel: use Fortran-layout (interleaved) array directly
        Ok(PyAudioSamples::new_multi_from_python_interleaved(
            pyarray.bind(py).to_owned(),
            sample_rate,
        ))
    }
}

#[pyfunction]
#[pyo3(signature = (fp: "str|Path", samples: "AudioSamples", as_type: "SampleType" = None), text_signature = "(fp: str | Path, samples: AudioSamples, as_type: SampleType = None)")]
pub fn save(
    py: Python<'_>,
    fp: &str,
    samples: &PyAudioSamples,
    as_type: Option<PySampleType>,
) -> PyResult<()> {
    // Check if this is a WAV file
    let is_wav = fp.to_lowercase().ends_with(".wav");

    // Use a custom dispatch to handle f64->f32 conversion for WAV files
    use crate::PyAudioDataInner;
    match samples.inner() {
        PyAudioDataInner::F64(a) if is_wav && as_type.is_none() => {
            // Automatically convert f64 to f32 for WAV files for maximum compatibility
            a.with_view(py, |audio| {
                let audio = audio.to_format::<f32>();
                audio_samples_io::write(&fp, &audio).map_err(audio_io_err_to_py)
            })
        }
        _ => {
            dispatch_with_view!(samples, py, |audio| {
                if let Some(t) = as_type {
                    // If f64 is explicitly requested for a WAV file, convert to f32 for compatibility
                    let target_type = if is_wav && t == PySampleType::F64 {
                        PySampleType::F32
                    } else {
                        t
                    };

                    match target_type {
                        PySampleType::U8 => {
                            let audio = audio.to_format::<u8>();
                            audio_samples_io::write(&fp, &audio).map_err(audio_io_err_to_py)
                        }
                        PySampleType::I16 => {
                            let audio = audio.to_format::<i16>();
                            audio_samples_io::write(&fp, &audio).map_err(audio_io_err_to_py)
                        }
                        PySampleType::I24 => {
                            let audio = audio.to_format::<I24>();
                            audio_samples_io::write(&fp, &audio).map_err(audio_io_err_to_py)
                        }
                        PySampleType::I32 => {
                            let audio = audio.to_format::<i32>();
                            audio_samples_io::write(&fp, &audio).map_err(audio_io_err_to_py)
                        }
                        PySampleType::F32 => {
                            let audio = audio.to_format::<f32>();
                            audio_samples_io::write(&fp, &audio).map_err(audio_io_err_to_py)
                        }
                        PySampleType::F64 => {
                            let audio = audio.to_format::<f64>();
                            audio_samples_io::write(&fp, &audio).map_err(audio_io_err_to_py)
                        }
                    }
                } else {
                    audio_samples_io::write(&fp, &audio).map_err(audio_io_err_to_py)
                }
            })
        }
    }
}

#[pymodule]
pub fn io(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    let io = PyModule::new(py, "io")?;
    io.add_class::<PyAudioInfo>()?;
    io.add_function(pyo3::wrap_pyfunction!(info, &io)?)?;
    io.add_function(pyo3::wrap_pyfunction!(read, &io)?)?;
    io.add_function(pyo3::wrap_pyfunction!(read_with_info, &io)?)?;
    io.add_function(pyo3::wrap_pyfunction!(save, &io)?)?;

    reexport!(m, io, "AudioInfo", "info", "read", "read_with_info", "save");
    m.add_submodule(&io)?;
    Ok(())
}
