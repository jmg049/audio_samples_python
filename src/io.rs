use audio_io::types::BaseAudioInfo;
use audio_samples::{I24, SampleType};
use numpy::{PyArrayDescr, PyArrayDescrMethods};
use pyo3::{
    Bound, PyResult, Python,
    exceptions::PyTypeError,
    pyclass, pyfunction, pymethods,
    types::{PyModule, PyModuleMethods},
};

use crate::{PyAudioDataInner, PyAudioSamples};

#[pyclass]
pub struct PyAudioInfo {
    info: BaseAudioInfo,
}

impl From<BaseAudioInfo> for PyAudioInfo {
    fn from(info: BaseAudioInfo) -> Self {
        PyAudioInfo { info }
    }
}

#[pymethods]
impl PyAudioInfo {
    /// Sample rate in Hz
    #[getter]
    fn sample_rate(&self) -> u32 {
        self.info.sample_rate
    }

    /// Number of audio channels
    #[getter]
    fn channels(&self) -> u16 {
        self.info.channels
    }

    /// Bits per sample
    #[getter]
    fn bits_per_sample(&self) -> u16 {
        self.info.bits_per_sample
    }

    /// Total number of samples per channel
    #[getter]
    fn num_samples(&self) -> usize {
        self.info.total_samples
    }

    /// Duration in seconds
    #[getter]
    fn duration(&self) -> f64 {
        self.info.duration.as_secs_f64()
    }

    /// Sample type as a string (e.g., "i16", "f32")
    #[getter]
    fn sample_type(&self) -> &'static str {
        match self.info.sample_type {
            SampleType::I16 => "i16",
            SampleType::I24 => "i24",
            SampleType::I32 => "i32",
            SampleType::F32 => "f32",
            SampleType::F64 => "f64",
            SampleType::Unknown | _ => "unknown",
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "AudioInfo(sample_rate={}, channels={}, bits_per_sample={}, num_samples={}, duration={:.3}s, sample_type='{}')",
            self.sample_rate(),
            self.channels(),
            self.bits_per_sample(),
            self.num_samples(),
            self.duration(),
            self.sample_type()
        )
    }
}

use paste::paste;

macro_rules! impl_read_as {
    ($rust_typel:ident, $rust_type:ty) => {
        paste! {
            #[pyfunction]
            #[pyo3(signature = (fp), text_signature = "(fp: str) -> AudioSamples")]
            pub fn [<read_as_ $rust_typel>](fp: &str) -> PyResult<PyAudioSamples> {
                let audio = audio_io::read::<_, $rust_type>(fp)
                    .map_err(|e| PyTypeError::new_err(format!("Failed to read audio: {}", e)))?;
                PyAudioSamples::from_audio_samples(audio)
            }
        }
    };
}

impl_read_as!(i16, i16);
impl_read_as!(i24, I24);
impl_read_as!(i32, i32);
impl_read_as!(f32, f32);
impl_read_as!(f64, f64);

#[pyfunction]
#[pyo3(signature = (fp, as_type=None), text_signature = "(fp, as_type=None) -> (PyAudioSamples, PyAudioInfo)")]
pub fn read_with_info(
    py: Python<'_>,
    fp: &str,
    as_type: Option<Bound<'_, PyArrayDescr>>,
) -> PyResult<(PyAudioSamples, PyAudioInfo)> {
    let info = audio_io::info(fp)
        .map_err(|e| PyTypeError::new_err(format!("Failed to get audio info: {}", e)))?;
    let native_type = info.sample_type;
    let target_type = match as_type {
        Some(dt) => {
            if dt.is_equiv_to(&numpy::dtype::<i16>(py)) {
                SampleType::I16
            } else if dt.is_equiv_to(&numpy::dtype::<i32>(py)) {
                SampleType::I32
            } else if dt.is_equiv_to(&numpy::dtype::<f32>(py)) {
                SampleType::F32
            } else if dt.is_equiv_to(&numpy::dtype::<f64>(py)) {
                SampleType::F64
            } else {
                return Err(PyTypeError::new_err(
                    "Unsupported data type for reading audio samples",
                ));
            }
        }
        None => native_type,
    };
    let py_samples = match target_type {
        SampleType::I16 => read_as_i16(fp),
        SampleType::I32 => read_as_i32(fp),
        SampleType::F32 => read_as_f32(fp),
        SampleType::F64 => read_as_f64(fp),
        _ => Err(PyTypeError::new_err(
            "Unsupported data type for reading audio samples",
        )),
    }?;

    Ok((py_samples, PyAudioInfo::from(info)))
}

#[pyfunction]
#[pyo3(signature = (fp, as_type=None), text_signature = "(fp, as_type=None) -> PyAudioSamples")]
pub fn read(
    py: Python<'_>,
    fp: &str,
    as_type: Option<Bound<'_, PyArrayDescr>>,
) -> PyResult<PyAudioSamples> {
    let (samples, _) = read_with_info(py, fp, as_type)?;
    Ok(samples)
}

#[pyfunction]
#[pyo3(signature = (fp, samples), text_signature = "(fp: str, samples: AudioSamples) -> None")]
pub fn save(py: Python<'_>, fp: &str, samples: &PyAudioSamples) -> PyResult<()> {
    match samples.inner() {
        PyAudioDataInner::I16(typed) => typed.with_view(py, |audio| {
            audio_io::write(fp, &audio)
                .map_err(|e| PyTypeError::new_err(format!("Failed to save audio: {}", e)))
        }),
        PyAudioDataInner::I24(typed) => typed.with_view(py, |audio| {
            audio_io::write(fp, &audio)
                .map_err(|e| PyTypeError::new_err(format!("Failed to save audio: {}", e)))
        }),
        PyAudioDataInner::I32(typed) => typed.with_view(py, |audio| {
            audio_io::write(fp, &audio)
                .map_err(|e| PyTypeError::new_err(format!("Failed to save audio: {}", e)))
        }),
        PyAudioDataInner::F32(typed) => typed.with_view(py, |audio| {
            audio_io::write(fp, &audio)
                .map_err(|e| PyTypeError::new_err(format!("Failed to save audio: {}", e)))
        }),
        PyAudioDataInner::F64(typed) => typed.with_view(py, |audio| {
            audio_io::write(fp, &audio)
                .map_err(|e| PyTypeError::new_err(format!("Failed to save audio: {}", e)))
        }),
    }
}

#[pyfunction]
#[pyo3(signature = (fp, samples, as_type), text_signature = "(fp: str, samples: AudioSamples, as_type: numpy.dtype) -> None")]
pub fn save_as_type(
    py: Python<'_>,
    fp: &str,
    samples: &PyAudioSamples,
    as_type: Bound<'_, PyArrayDescr>,
) -> PyResult<()> {
    use audio_samples::AudioTypeConversion;

    // Determine target sample type
    let target_type = if as_type.is_equiv_to(&numpy::dtype::<i16>(py)) {
        SampleType::I16
    } else if as_type.is_equiv_to(&numpy::dtype::<i32>(py)) {
        SampleType::I32
    } else if as_type.is_equiv_to(&numpy::dtype::<f32>(py)) {
        SampleType::F32
    } else if as_type.is_equiv_to(&numpy::dtype::<f64>(py)) {
        SampleType::F64
    } else {
        return Err(PyTypeError::new_err(
            "Unsupported data type for saving audio. Supported types: i16, i32, f32, f64",
        ));
    };

    // Helper macro to convert and save
    macro_rules! convert_and_save {
        ($typed:expr, $target:ty) => {{
            $typed.with_view(py, |audio| {
                let converted = audio.to_format::<$target>();
                audio_io::write(fp, &converted)
                    .map_err(|e| PyTypeError::new_err(format!("Failed to save audio: {}", e)))
            })
        }};
    }

    match (samples.inner(), target_type) {
        // Convert to i16
        (PyAudioDataInner::I16(typed), SampleType::I16) => convert_and_save!(typed, i16),
        (PyAudioDataInner::I24(typed), SampleType::I16) => convert_and_save!(typed, i16),
        (PyAudioDataInner::I32(typed), SampleType::I16) => convert_and_save!(typed, i16),
        (PyAudioDataInner::F32(typed), SampleType::I16) => convert_and_save!(typed, i16),
        (PyAudioDataInner::F64(typed), SampleType::I16) => convert_and_save!(typed, i16),

        // Convert to i32
        (PyAudioDataInner::I16(typed), SampleType::I32) => convert_and_save!(typed, i32),
        (PyAudioDataInner::I24(typed), SampleType::I32) => convert_and_save!(typed, i32),
        (PyAudioDataInner::I32(typed), SampleType::I32) => convert_and_save!(typed, i32),
        (PyAudioDataInner::F32(typed), SampleType::I32) => convert_and_save!(typed, i32),
        (PyAudioDataInner::F64(typed), SampleType::I32) => convert_and_save!(typed, i32),

        // Convert to f32
        (PyAudioDataInner::I16(typed), SampleType::F32) => convert_and_save!(typed, f32),
        (PyAudioDataInner::I24(typed), SampleType::F32) => convert_and_save!(typed, f32),
        (PyAudioDataInner::I32(typed), SampleType::F32) => convert_and_save!(typed, f32),
        (PyAudioDataInner::F32(typed), SampleType::F32) => convert_and_save!(typed, f32),
        (PyAudioDataInner::F64(typed), SampleType::F32) => convert_and_save!(typed, f32),

        // Convert to f64
        (PyAudioDataInner::I16(typed), SampleType::F64) => convert_and_save!(typed, f64),
        (PyAudioDataInner::I24(typed), SampleType::F64) => convert_and_save!(typed, f64),
        (PyAudioDataInner::I32(typed), SampleType::F64) => convert_and_save!(typed, f64),
        (PyAudioDataInner::F32(typed), SampleType::F64) => convert_and_save!(typed, f64),
        (PyAudioDataInner::F64(typed), SampleType::F64) => convert_and_save!(typed, f64),

        // Unsupported combinations (I24 as target is not exposed to Python, Unknown sample type)
        _ => Err(PyTypeError::new_err("Unsupported conversion")),
    }
}

pub fn audio_io_module<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyModule>> {
    let io = PyModule::new(py, "io")?;
    io.add_class::<PyAudioInfo>()?;
    io.add_function(pyo3::wrap_pyfunction!(read, &io)?)?;
    io.add_function(pyo3::wrap_pyfunction!(read_with_info, &io)?)?;
    io.add_function(pyo3::wrap_pyfunction!(read_as_i16, &io)?)?;
    io.add_function(pyo3::wrap_pyfunction!(read_as_i32, &io)?)?;
    io.add_function(pyo3::wrap_pyfunction!(read_as_f32, &io)?)?;
    io.add_function(pyo3::wrap_pyfunction!(read_as_f64, &io)?)?;

    io.add_function(pyo3::wrap_pyfunction!(save, &io)?)?;
    io.add_function(pyo3::wrap_pyfunction!(save_as_type, &io)?)?;
    Ok(io)
}