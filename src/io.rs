use crate::{
    PyAudioSamples, audio_err_to_py, audio_io_err_to_py, audio_read_err_to_py, dispatch_with_view,
    dispatch_with_view_detached, impl_py_repr, impl_py_wrapper_core, nzu32_or_err, reexport,
    types::PySampleType,
};
use audio_samples::operations::ResamplingQuality;
use audio_samples::{I24, resample, traits::StandardSample};
use audio_samples_io::CompressionLevel;
use audio_samples_io::flac::{StreamInfo, VorbisComment};
use audio_samples_io::types::{BaseAudioInfo, FileType, WriteOptions};
use audio_samples_io::wav::{CuePoint, InfoMetadata, WavMetadata};
use numpy::{Element, PyArrayDescr, PyArrayMethods};
use pyo3::{
    Bound, PyAny, PyResult, Python,
    exceptions::{PyTypeError, PyValueError},
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
        NativeAudioArray::U8(arr, info) => native_to_audio_samples(py, &arr, info),
        NativeAudioArray::I16(arr, info) => native_to_audio_samples(py, &arr, info),
        NativeAudioArray::I32(arr, info) => native_to_audio_samples(py, &arr, info),
        NativeAudioArray::F32(arr, info) => native_to_audio_samples(py, &arr, info),
        NativeAudioArray::F64(arr, info) => native_to_audio_samples(py, &arr, info),
    }
}

#[cfg(target_endian = "little")]
fn native_to_audio_samples<T>(
    py: Python<'_>,
    arr: &pyo3::Py<numpy::PyArray2<T>>,
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
            array_1d.clone(),
            sample_rate,
        ))
    } else {
        Ok(PyAudioSamples::new_multi_from_python_interleaved(
            arr.bind(py).to_owned(),
            sample_rate,
        ))
    }
}

/// Read an audio file (WAV or FLAC) and return an ``AudioSamples``.
///
/// Args:
///
///     fp (str | Path | BinaryIO): Path to the audio file, or a file object opened in
///         binary mode with ``read``, ``seek`` and ``tell``, read from its current position.
///     as_type (SampleType | None): Sample type to convert to, for example
///         ``SampleType.F32`` for floats in [-1, 1]. Defaults to the file's
///         native sample type (16-bit files give raw ``int16`` values).
///
/// Returns:
///
///     AudioSamples: The decoded audio.
///
/// Raises:
///
///     FileNotFoundError: If ``fp`` does not exist (``AudioFileNotFoundError``).
///     TypeError: If the file cannot be decoded (``AudioDecodeError``), or ``fp`` is
///         neither a path nor a file object.
///     ValueError: If a file object holds anything other than WAV or FLAC.
#[pyfunction]
#[pyo3(signature = (fp: "str|Path|BinaryIO", as_type:"SampleType"=None))]
pub fn read(
    py: Python<'_>,
    fp: &Bound<'_, PyAny>,
    as_type: Option<PySampleType>,
) -> PyResult<PyAudioSamples> {
    let fp: std::path::PathBuf = match fp.extract() {
        Ok(path) => path,
        Err(not_a_path) => {
            if !fp.hasattr("read")? {
                return Err(not_a_path);
            }
            return crate::io_streaming::PyStreamedAudioReader::open_any(fp)?
                .read_rest(py, as_type);
        }
    };
    let fp = fp.to_string_lossy();
    let fp: &str = &fp;
    crate::ensure_file_exists(fp)?;
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
        None => native_sample_type(fp)?,
    };
    read_typed(py, fp, target)
}

#[pyfunction]
#[pyo3(signature = (fp: "str|Path", as_type: "SampleType" = None))]
#[allow(clippy::needless_pass_by_value)] // PyO3 extracts `PathBuf` by value
pub fn read_with_info(
    py: Python<'_>,
    fp: std::path::PathBuf,
    as_type: Option<&Bound<'_, PyArrayDescr>>,
) -> PyResult<(PyAudioSamples, PyAudioInfo)> {
    let fp = fp.to_string_lossy();
    let fp: &str = &fp;
    crate::ensure_file_exists(fp)?;
    let info = audio_samples_io::info(fp)
        .map_err(|e| audio_read_err_to_py(e, fp, "Failed to get audio info"))?;

    let native = PySampleType::from_native(info.sample_type)
        .ok_or_else(|| PyTypeError::new_err("Unsupported native sample type"))?;

    let target = match as_type {
        Some(dt) => PySampleType::from_numpy(py, dt)?,
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
#[pyo3(signature = (fp: "str|Path"))]
#[allow(clippy::needless_pass_by_value)] // PyO3 extracts `PathBuf` by value
pub fn info(fp: std::path::PathBuf) -> PyResult<PyAudioInfo> {
    let fp = fp.to_string_lossy();
    let fp: &str = &fp;
    crate::ensure_file_exists(fp)?;
    let info = audio_samples_io::info(fp)
        .map_err(|e| audio_read_err_to_py(e, fp, "Failed to get audio info"))?;
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
            array_1d.clone(),
            sample_rate,
        ))
    } else {
        let bound = pyarray.bind(py).to_owned();
        if numpy::PyUntypedArrayMethods::is_c_contiguous(&bound) {
            // FLAC decodes to planar samples, a C-order (channels, frames) array.
            Ok(PyAudioSamples::new_multi_from_python(bound, sample_rate))
        } else {
            // WAV: a Fortran-layout (interleaved) array, used directly.
            Ok(PyAudioSamples::new_multi_from_python_interleaved(
                bound,
                sample_rate,
            ))
        }
    }
}

/// Save audio to a file, with the format chosen from the extension.
///
/// WAV keeps every sample type, float64 included. FLAC stores 16-bit and 24-bit integers only, so
/// any other sample type raises ``ValueError`` unless ``as_type`` asks for ``SampleType.I16`` or
/// ``SampleType.I24``.
///
/// Args:
///
///     fp (str | Path): Output path.
///     samples (AudioSamples): The audio to write.
///     as_type (SampleType, optional): Convert to this sample type as the file is written, without
///         a copy of the signal. Defaults to the audio's native type.
///     flac_md5 (bool): Store the MD5 signature of the audio in a FLAC file, so that decoders can
///         verify it. ``False`` leaves it out, which encodes faster. Defaults to ``True``.
///
/// Raises:
///
///     ValueError: If the format is unsupported, the sample type cannot be stored in FLAC, or
///         ``flac_md5`` is false and the file is not FLAC.
///     OSError: If the file cannot be written.
#[pyfunction]
#[pyo3(signature = (fp: "str|Path", samples: "AudioSamples", as_type: "SampleType" = None, flac_md5: "bool" = true))]
#[allow(clippy::needless_pass_by_value)] // PyO3 extracts `PathBuf` by value
pub fn save(
    py: Python<'_>,
    fp: std::path::PathBuf,
    samples: &PyAudioSamples,
    as_type: Option<PySampleType>,
    flac_md5: bool,
) -> PyResult<()> {
    let fp = fp.to_string_lossy();
    write_samples(py, &fp, samples, WriteOptions::default(), as_type, flac_md5)
}

fn write_flac_without_md5<T: StandardSample + 'static>(
    fp: &str,
    audio: &audio_samples::AudioSamples<'_, T>,
    opts: WriteOptions,
) -> Result<(), audio_samples_io::AudioIOError> {
    audio_samples_io::write_flac_file_as::<T, _, T>(
        fp,
        audio,
        opts,
        CompressionLevel::default(),
        false,
    )
}

/// Write `samples` as `as_type`, or in their own type. Conversion happens block by block as the
/// file is written, and no Python objects are touched while it is.
fn write_samples(
    py: Python<'_>,
    fp: &str,
    samples: &PyAudioSamples,
    opts: WriteOptions,
    as_type: Option<PySampleType>,
    flac_md5: bool,
) -> PyResult<()> {
    check_flac_sample_type(fp, samples, as_type)?;
    let without_md5 = !flac_md5;
    if without_md5 && !fp.to_lowercase().ends_with(".flac") {
        return Err(PyValueError::new_err(
            "flac_md5=False applies to FLAC files only, and this path does not end in .flac",
        ));
    }
    macro_rules! write_as {
        ($audio:ident, $T:ty) => {
            if without_md5 {
                audio_samples_io::write_flac_file_as::<$T, _, _>(
                    fp,
                    &$audio,
                    opts,
                    CompressionLevel::default(),
                    false,
                )
            } else {
                audio_samples_io::write_as::<$T, _, _>(fp, &$audio, opts)
            }
        };
    }
    dispatch_with_view_detached!(samples, py, |audio| {
        match as_type {
            Some(PySampleType::U8) => write_as!(audio, u8),
            Some(PySampleType::I16) => write_as!(audio, i16),
            Some(PySampleType::I24) => write_as!(audio, I24),
            Some(PySampleType::I32) => write_as!(audio, i32),
            Some(PySampleType::F32) => write_as!(audio, f32),
            Some(PySampleType::F64) => write_as!(audio, f64),
            None if without_md5 => write_flac_without_md5(fp, &audio, opts),
            None => audio_samples_io::write_with_options(fp, &audio, opts),
        }
    })
    .map_err(audio_io_err_to_py)
}

/// FLAC holds 16-bit and 24-bit integers. Anything else would change the samples, so the conversion
/// has to be asked for: `as_type=SampleType.I16` or `as_type=SampleType.I24`.
fn check_flac_sample_type(
    fp: &str,
    samples: &PyAudioSamples,
    as_type: Option<PySampleType>,
) -> PyResult<()> {
    if !fp.to_lowercase().ends_with(".flac") {
        return Ok(());
    }
    let name = match (as_type, samples.inner()) {
        (Some(PySampleType::I16 | PySampleType::I24), _)
        | (None, crate::PyAudioDataInner::I16(_) | crate::PyAudioDataInner::I24(_)) => {
            return Ok(());
        }
        (Some(PySampleType::U8), _) | (None, crate::PyAudioDataInner::U8(_)) => "u8",
        (Some(PySampleType::I32), _) | (None, crate::PyAudioDataInner::I32(_)) => "i32",
        (Some(PySampleType::F32), _) | (None, crate::PyAudioDataInner::F32(_)) => "f32",
        (Some(PySampleType::F64), _) | (None, crate::PyAudioDataInner::F64(_)) => "f64",
    };
    Err(pyo3::exceptions::PyValueError::new_err(format!(
        "FLAC stores 16-bit or 24-bit integers, so {name} samples cannot be written unchanged. \
         Convert explicitly with as_type=SampleType.I16 or as_type=SampleType.I24."
    )))
}

/// The sample type stored in the file at `fp`, read from its header.
fn native_sample_type(fp: &str) -> PyResult<PySampleType> {
    let native = audio_samples_io::peek_native_type(fp)
        .map_err(|e| audio_read_err_to_py(e, fp, "Failed to detect native type"))?;
    let sample_type: audio_samples::SampleType = native.into();
    PySampleType::from_native(sample_type)
        .ok_or_else(|| pyo3::exceptions::PyTypeError::new_err("Unsupported native sample type"))
}

/// Read an audio file and resample it to a target sample rate in one call.
///
/// Reads the entire file (auto-detecting WAV/FLAC), then resamples to ``target_sr``.
/// The resampler is the same as ``AudioSamples.resample``: delay-free, exactly
/// ``ceil(n * target_sr / file_sr)`` samples per channel.
///
/// Args:
///
///     fp (str | Path): Path to the audio file.
///     target_sr (int): Target sample rate in Hz; must be greater than zero.
///     quality (ResamplingQuality, optional): Resampling quality/speed trade-off. Defaults to
///         ``ResamplingQuality.high`` for best fidelity.
///     as_type (SampleType, optional): Sample type for the returned audio. Defaults to the file's
///         sample type, as :func:`read` does; the resampler works on every sample type.
///
/// Returns:
///
///     AudioSamples: The resampled audio.
///
/// Raises:
///
///     ValueError: If ``target_sr`` is zero.
///     TypeError: If the file cannot be read or the format is unsupported.
///     RuntimeError: If resampling fails.
#[pyfunction]
#[pyo3(signature = (fp: "str | Path", target_sr: "int", quality: "ResamplingQuality" = None, as_type: "SampleType" = None))]
#[allow(clippy::needless_pass_by_value)] // PyO3 extracts `PathBuf` by value
pub fn read_and_resample(
    py: Python<'_>,
    fp: std::path::PathBuf,
    target_sr: u32,
    quality: Option<crate::types::PyResamplingQuality>,
    as_type: Option<PySampleType>,
) -> PyResult<PyAudioSamples> {
    let fp = fp.to_string_lossy();
    let fp: &str = &fp;
    crate::ensure_file_exists(fp)?;
    let target = nzu32_or_err(target_sr)?;
    let quality: ResamplingQuality = quality.map_or(ResamplingQuality::High, |q| q.inner);
    let as_type = match as_type {
        Some(t) => t,
        None => native_sample_type(fp)?,
    };

    // Read as `read` does (no second copy of the file), then resample without the GIL.
    let audio = read_typed(py, fp, as_type)?;
    dispatch_with_view_detached!(audio, py, |signal| resample(&signal, target, quality)
        .map(PyAudioSamples::from_audio_samples))
    .map_err(audio_err_to_py)
}

/// Peek at the native sample type of an audio file with minimal I/O.
///
/// Reads only the header (no full decode), making it much cheaper than :func:`info` when only
/// the sample type is needed.
///
/// Args:
///
///     fp (str | Path): Path to the audio file.
///
/// Returns:
///
///     str: The native sample type as a string (e.g. ``"i16"``, ``"f32"``).
///
/// Raises:
///
///     ValueError: If the format is unsupported or the type cannot be determined.
///     OSError: If the file cannot be opened.
#[pyfunction]
#[pyo3(signature = (fp: "str | Path"))]
#[allow(clippy::needless_pass_by_value)] // PyO3 extracts `PathBuf` by value
pub fn peek_native_type(fp: std::path::PathBuf) -> PyResult<&'static str> {
    let fp = fp.to_string_lossy();
    let fp: &str = &fp;
    crate::ensure_file_exists(fp)?;
    let validated = audio_samples_io::peek_native_type(fp).map_err(audio_io_err_to_py)?;
    let st: audio_samples::SampleType = validated.into();
    Ok(st.as_str())
}

// =============================================================================
// WRITE OPTIONS / METADATA / FORMAT TYPES
// =============================================================================

/// Options controlling how audio data is written.
///
/// Args:
///
///     write_buf_capacity (int): Size of the internal write buffer in bytes. The default of
///         256 holds only the file's header: samples go straight to the file where their memory
///         is already its layout, and otherwise in tiles of 64 KiB. A larger buffer copies them
///         first, and helps only when each write call to the destination is expensive.
#[pyclass(name = "WriteOptions", module = "audio_samples.io", from_py_object)]
#[derive(Clone, Copy)]
pub struct PyWriteOptions {
    pub(crate) inner: WriteOptions,
}

#[pymethods]
impl PyWriteOptions {
    #[new]
    #[pyo3(signature = (write_buf_capacity: "int" = None))]
    fn new(write_buf_capacity: Option<usize>) -> Self {
        let mut inner = WriteOptions::default();
        if let Some(cap) = write_buf_capacity {
            inner.write_buf_capacity = cap;
        }
        Self { inner }
    }

    /// Size of the internal write buffer in bytes.
    #[getter]
    #[allow(clippy::trivially_copy_pass_by_ref)] // PyO3 methods receive `&self`
    const fn write_buf_capacity(&self) -> usize {
        self.inner.write_buf_capacity
    }

    #[setter]
    const fn set_write_buf_capacity(&mut self, value: usize) {
        self.inner.write_buf_capacity = value;
    }
}

/// FLAC compression level (0-8): higher means smaller files but slower encoding.
///
/// Args:
///
///     level (int): Compression level, clamped to the range 0-8. Defaults to 5.
#[pyclass(name = "CompressionLevel", module = "audio_samples.io", from_py_object)]
#[derive(Clone, Copy)]
pub struct PyCompressionLevel {
    pub(crate) inner: CompressionLevel,
}

#[pymethods]
impl PyCompressionLevel {
    #[new]
    #[pyo3(signature = (level: "int" = None))]
    fn new(level: Option<u8>) -> Self {
        Self {
            inner: level.map_or_else(CompressionLevel::default, CompressionLevel::new),
        }
    }

    /// Fastest compression (level 0).
    #[staticmethod]
    const fn fastest() -> Self {
        Self {
            inner: CompressionLevel::new(0),
        }
    }

    /// Best compression (level 8).
    #[staticmethod]
    const fn best() -> Self {
        Self {
            inner: CompressionLevel::new(8),
        }
    }

    /// The numeric compression level (0-8).
    #[getter]
    #[allow(clippy::trivially_copy_pass_by_ref)] // PyO3 methods receive `&self`
    const fn level(&self) -> u8 {
        self.inner.level()
    }

    #[allow(clippy::trivially_copy_pass_by_ref)] // PyO3 methods receive `&self`
    fn __repr__(&self) -> String {
        format!("CompressionLevel(level={})", self.inner.level())
    }
}

/// LIST/INFO metadata tags for a WAV file (title, artist, etc.).
///
/// All fields are optional strings. Construct one and attach it to a :class:`WavMetadata` to
/// persist tags with :func:`write_with_metadata`.
#[pyclass(name = "InfoMetadata", module = "audio_samples.io", from_py_object)]
#[derive(Clone, Default)]
pub struct PyInfoMetadata {
    pub(crate) inner: InfoMetadata,
}

/// Generate getter+setter `#[pymethods]` for an `Option<String>` field of `PyInfoMetadata`.
macro_rules! info_string_field {
    ($get:ident, $set:ident, $field:ident, $doc:literal) => {
        #[pymethods]
        impl PyInfoMetadata {
            #[doc = $doc]
            #[getter]
            fn $get(&self) -> Option<String> {
                self.inner.$field.clone()
            }
            #[setter]
            fn $set(&mut self, value: Option<String>) {
                self.inner.$field = value;
            }
        }
    };
}

info_string_field!(title, set_title, title, "Track title.");
info_string_field!(artist, set_artist, artist, "Performing artist.");
info_string_field!(album, set_album, album, "Album name.");
info_string_field!(date, set_date, date, "Creation/release date.");
info_string_field!(comment, set_comment, comment, "Free-form comment.");
info_string_field!(genre, set_genre, genre, "Genre.");
info_string_field!(software, set_software, software, "Authoring software.");
info_string_field!(copyright, set_copyright, copyright, "Copyright notice.");
info_string_field!(engineer, set_engineer, engineer, "Engineer name.");
info_string_field!(subject, set_subject, subject, "Subject/description.");
info_string_field!(source, set_source, source, "Source of the material.");
info_string_field!(keywords, set_keywords, keywords, "Keywords.");

#[pymethods]
impl PyInfoMetadata {
    #[new]
    #[pyo3(signature = ())]
    fn new() -> Self {
        Self::default()
    }

    fn __repr__(&self) -> String {
        format!(
            "InfoMetadata(title={:?}, artist={:?})",
            self.inner.title, self.inner.artist
        )
    }
}

impl_py_wrapper_core!(PyInfoMetadata, InfoMetadata);

/// A WAV cue point (marker).
#[pyclass(name = "CuePoint", module = "audio_samples.io", from_py_object)]
#[derive(Clone)]
pub struct PyCuePoint {
    pub(crate) inner: CuePoint,
}

#[pymethods]
impl PyCuePoint {
    /// Cue point identifier.
    #[getter]
    const fn id(&self) -> u32 {
        self.inner.id
    }
    /// Play-order position.
    #[getter]
    const fn position(&self) -> u32 {
        self.inner.position
    }
    /// Sample offset within the data chunk.
    #[getter]
    const fn sample_offset(&self) -> u32 {
        self.inner.sample_offset
    }

    fn __repr__(&self) -> String {
        format!(
            "CuePoint(id={}, position={}, sample_offset={})",
            self.inner.id, self.inner.position, self.inner.sample_offset
        )
    }
}

impl_py_wrapper_core!(PyCuePoint, CuePoint);

/// Round-trippable WAV metadata: INFO tags plus cue points.
///
/// Pass to :func:`write_with_metadata` to persist tags that a plain read→write would drop.
#[pyclass(name = "WavMetadata", module = "audio_samples.io", from_py_object)]
#[derive(Clone, Default)]
pub struct PyWavMetadata {
    pub(crate) inner: WavMetadata,
}

#[pymethods]
impl PyWavMetadata {
    #[new]
    #[pyo3(signature = (info: "InfoMetadata" = None))]
    fn new(info: Option<PyInfoMetadata>) -> Self {
        let mut inner = WavMetadata::default();
        inner.info = info.map(|i| i.inner);
        Self { inner }
    }

    /// The INFO tags, or ``None`` if unset.
    #[getter]
    fn info(&self) -> Option<PyInfoMetadata> {
        self.inner
            .info
            .clone()
            .map(|inner| PyInfoMetadata { inner })
    }

    #[setter]
    fn set_info(&mut self, info: Option<PyInfoMetadata>) {
        self.inner.info = info.map(|i| i.inner);
    }

    /// The cue points (markers).
    #[getter]
    fn cue_points(&self) -> Vec<PyCuePoint> {
        self.inner
            .cue_points
            .iter()
            .cloned()
            .map(|inner| PyCuePoint { inner })
            .collect()
    }

    /// True if there are no tags and no cue points.
    #[getter]
    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
}

impl_py_wrapper_core!(PyWavMetadata, WavMetadata);

/// A FLAC Vorbis comment block: a vendor string plus key→values tags.
///
/// Construct one, populate it with :meth:`set`/:meth:`add`, and serialise with :meth:`to_bytes`
/// (or parse with :meth:`from_bytes`). Note: the public crate API does not expose reading the
/// Vorbis comment back out of a decoded FLAC file, so this class is primarily for constructing
/// and (de)serialising comment blocks.
#[pyclass(name = "VorbisComment", module = "audio_samples.io", from_py_object)]
#[derive(Clone, Default)]
pub struct PyVorbisComment {
    pub(crate) inner: VorbisComment,
}

#[pymethods]
impl PyVorbisComment {
    #[new]
    #[pyo3(signature = ())]
    fn new() -> Self {
        Self::default()
    }

    /// Parse a Vorbis comment block from its raw bytes.
    ///
    /// Args:
    ///
    ///     data (bytes): The raw Vorbis comment block.
    ///
    /// Returns:
    ///
    ///     VorbisComment: The parsed comment.
    ///
    /// Raises:
    ///
    ///     ValueError: If the bytes are malformed.
    #[staticmethod]
    #[pyo3(signature = (data: "bytes"))]
    fn from_bytes(data: &[u8]) -> PyResult<Self> {
        let inner = VorbisComment::from_bytes(data)
            .map_err(|e| PyValueError::new_err(format!("Invalid Vorbis comment: {e}")))?;
        Ok(Self { inner })
    }

    /// The vendor string.
    #[getter]
    fn vendor(&self) -> String {
        self.inner.vendor.clone()
    }

    #[setter]
    fn set_vendor(&mut self, value: String) {
        self.inner.vendor = value;
    }

    /// Get the first value for ``key``, or ``None``.
    #[pyo3(signature = (key: "str"))]
    fn get(&self, key: &str) -> Option<String> {
        self.inner.get(key).map(ToString::to_string)
    }

    /// Get all values for ``key`` (empty list if absent).
    #[pyo3(signature = (key: "str"))]
    fn get_all(&self, key: &str) -> Vec<String> {
        self.inner.get_all(key).cloned().unwrap_or_default()
    }

    /// Set ``key`` to a single ``value``, replacing any existing values.
    #[pyo3(signature = (key: "str", value: "str"))]
    fn set(&mut self, key: &str, value: String) {
        self.inner.set(key, value);
    }

    /// Append ``value`` to the values for ``key``.
    #[pyo3(signature = (key: "str", value: "str"))]
    fn add(&mut self, key: &str, value: String) {
        self.inner.add(key, value);
    }

    /// Serialise to a raw Vorbis comment block.
    #[pyo3(signature = ())]
    fn to_bytes<'py>(&self, py: Python<'py>) -> Bound<'py, pyo3::types::PyBytes> {
        pyo3::types::PyBytes::new(py, &self.inner.to_bytes())
    }
}

impl_py_wrapper_core!(PyVorbisComment, VorbisComment);

/// FLAC STREAMINFO metadata block (read-only).
#[pyclass(name = "StreamInfo", module = "audio_samples.io", from_py_object)]
#[derive(Clone, Copy)]
pub struct PyStreamInfo {
    pub(crate) inner: StreamInfo,
}

#[pymethods]
impl PyStreamInfo {
    /// Minimum block size (samples).
    #[getter]
    const fn min_block_size(&self) -> u16 {
        self.inner.min_block_size
    }
    /// Maximum block size (samples).
    #[getter]
    const fn max_block_size(&self) -> u16 {
        self.inner.max_block_size
    }
    /// Sample rate in Hz.
    #[getter]
    const fn sample_rate(&self) -> u32 {
        self.inner.sample_rate
    }
    /// Number of channels.
    #[getter]
    const fn channels(&self) -> u8 {
        self.inner.channels
    }
    /// Bits per sample.
    #[getter]
    const fn bits_per_sample(&self) -> u8 {
        self.inner.bits_per_sample
    }
    /// Total inter-channel samples (frames).
    #[getter]
    const fn total_samples(&self) -> u64 {
        self.inner.total_samples
    }
    /// Whether the MD5 signature of the decoded audio is present.
    #[getter]
    fn has_md5(&self) -> bool {
        self.inner.has_md5()
    }

    fn __repr__(&self) -> String {
        format!(
            "StreamInfo(sample_rate={}, channels={}, bits_per_sample={}, total_samples={})",
            self.inner.sample_rate,
            self.inner.channels,
            self.inner.bits_per_sample,
            self.inner.total_samples
        )
    }
}

impl_py_wrapper_core!(PyStreamInfo, StreamInfo);

// =============================================================================
// WRITE FUNCTIONS
// =============================================================================

/// Write audio samples to a file with explicit write options.
///
/// Like :func:`save` but lets you control the write-buffer size via :class:`WriteOptions`.
/// The format is chosen from the file extension (``.wav``/``.flac``/``.aiff``).
///
/// Args:
///
///     fp (str | Path): Output path.
///     samples (AudioSamples): The audio to write.
///     options (WriteOptions, optional): Write options. Defaults to a 256-byte buffer, which
///         holds the header.
///     as_type (SampleType, optional): Convert to this sample type as the file is written, without
///         a copy of the signal. Defaults to the audio's native type.
///     flac_md5 (bool): Store the MD5 signature of the audio in a FLAC file, so that decoders can
///         verify it. ``False`` leaves it out, which encodes faster. Defaults to ``True``.
///
/// Raises:
///
///     ValueError: If the format is unsupported, the file is FLAC and the sample type to write is
///         not I16 or I24 (FLAC stores integers of those widths only), or ``flac_md5`` is false
///         and the file is not FLAC.
///     OSError: If the file cannot be written.
#[pyfunction]
#[pyo3(signature = (fp: "str | Path", samples: "AudioSamples", options: "WriteOptions" = None, as_type: "SampleType" = None, flac_md5: "bool" = true))]
#[allow(clippy::needless_pass_by_value)] // PyO3 extracts `PathBuf` by value
pub fn write_with_options(
    py: Python<'_>,
    fp: std::path::PathBuf,
    samples: &PyAudioSamples,
    options: Option<PyWriteOptions>,
    as_type: Option<PySampleType>,
    flac_md5: bool,
) -> PyResult<()> {
    let fp = fp.to_string_lossy();
    let opts = options.map_or_else(WriteOptions::default, |o| o.inner);
    write_samples(py, &fp, samples, opts, as_type, flac_md5)
}

/// Limit how many threads a file read or decode may use.
///
/// Reads of large files and FLAC decodes run on a few short-lived threads, which end before the
/// call returns. By default their number follows the CPUs available to the process, with a cap,
/// or ``AUDIO_SAMPLES_IO_THREADS`` if that is set when the first read starts. This setting
/// replaces both.
///
/// Args:
///
///     limit (int | None): Most threads to use; ``1`` keeps everything on the calling thread.
///         ``None`` returns to the default.
///
/// Raises:
///
///     ValueError: If ``limit`` is zero.
#[pyfunction]
#[pyo3(signature = (limit: "int | None"))]
pub fn set_thread_limit(limit: Option<usize>) -> PyResult<()> {
    if limit == Some(0) {
        return Err(PyValueError::new_err("limit must be at least 1, or None"));
    }
    audio_samples_io::set_thread_limit(limit.and_then(std::num::NonZeroUsize::new));
    Ok(())
}

/// The most threads a read of a large file will use at present.
///
/// Returns:
///
///     int: See :func:`set_thread_limit`.
#[pyfunction]
pub fn thread_limit() -> usize {
    audio_samples_io::thread_limit()
}

/// Write a WAV file with trailing metadata chunks (LIST/INFO tags, cue points).
///
/// Like :func:`save`, but also serialises the given :class:`WavMetadata` after the audio data,
/// persisting tags/markers that a plain read→write round-trip would drop. WAV only.
///
/// Args:
///
///     fp (str | Path): Output ``.wav`` path.
///     samples (AudioSamples): The audio to write.
///     metadata (WavMetadata): Metadata (INFO tags / cue points) to serialise.
///
/// Raises:
///
///     OSError: If the file cannot be written.
#[pyfunction]
#[pyo3(signature = (fp: "str | Path", samples: "AudioSamples", metadata: "WavMetadata"))]
#[allow(clippy::needless_pass_by_value)] // PyO3 extracts `PathBuf` by value
pub fn write_with_metadata(
    py: Python<'_>,
    fp: std::path::PathBuf,
    samples: &PyAudioSamples,
    metadata: &PyWavMetadata,
) -> PyResult<()> {
    let fp = fp.to_string_lossy();
    let fp: &str = &fp;
    dispatch_with_view!(samples, py, |audio| {
        audio_samples_io::write_with_metadata(&fp, &audio, &metadata.inner)
            .map_err(audio_io_err_to_py)
    })
}

// =============================================================================
// METADATA READERS
// =============================================================================

/// Read the LIST/INFO tags from a WAV file.
///
/// Args:
///
///     fp (str | Path): Path to the WAV file.
///
/// Returns:
///
///     InfoMetadata | None: The INFO tags, or ``None`` if the file has no LIST/INFO chunk.
///
/// Raises:
///
///     OSError: If the file cannot be opened.
///     ValueError: If the file is not a WAV file or the chunk is malformed.
#[pyfunction]
#[pyo3(signature = (fp: "str | Path"))]
#[allow(clippy::needless_pass_by_value)] // PyO3 extracts `PathBuf` by value
pub fn read_wav_info_tags(fp: std::path::PathBuf) -> PyResult<Option<PyInfoMetadata>> {
    use audio_samples_io::traits::AudioFileMetadata;
    use audio_samples_io::wav::wav_file::WavFile;
    let fp = fp.to_string_lossy();
    let fp: &str = &fp;
    let path = std::path::Path::new(fp);
    if FileType::detect(path) != FileType::WAV {
        return Err(PyValueError::new_err(
            "read_wav_info_tags requires a WAV file",
        ));
    }
    let wav = WavFile::open_metadata(path).map_err(audio_io_err_to_py)?;
    let list = wav.list().map_err(audio_io_err_to_py)?;
    let Some(chunk) = list else {
        return Ok(None);
    };
    match chunk.info_metadata() {
        Some(Ok(inner)) => Ok(Some(PyInfoMetadata { inner })),
        Some(Err(e)) => Err(PyValueError::new_err(format!("Invalid INFO chunk: {e}"))),
        None => Ok(None),
    }
}

/// Read the cue points (markers) from a WAV file.
///
/// Args:
///
///     fp (str | Path): Path to the WAV file.
///
/// Returns:
///
///     list[CuePoint]: The cue points (empty if none).
///
/// Raises:
///
///     OSError: If the file cannot be opened.
///     ValueError: If the file is not a WAV file or the chunk is malformed.
#[pyfunction]
#[pyo3(signature = (fp: "str | Path"))]
#[allow(clippy::needless_pass_by_value)] // PyO3 extracts `PathBuf` by value
pub fn read_wav_cue_points(fp: std::path::PathBuf) -> PyResult<Vec<PyCuePoint>> {
    use audio_samples_io::traits::AudioFileMetadata;
    use audio_samples_io::wav::wav_file::WavFile;
    let fp = fp.to_string_lossy();
    let fp: &str = &fp;
    let path = std::path::Path::new(fp);
    if FileType::detect(path) != FileType::WAV {
        return Err(PyValueError::new_err(
            "read_wav_cue_points requires a WAV file",
        ));
    }
    let wav = WavFile::open_metadata(path).map_err(audio_io_err_to_py)?;
    let cue = wav.cue().map_err(audio_io_err_to_py)?;
    match cue {
        Some(chunk) => {
            let points = chunk
                .cue_points()
                .map_err(|e| PyValueError::new_err(format!("Invalid cue chunk: {e}")))?;
            Ok(points
                .into_iter()
                .map(|inner| PyCuePoint { inner })
                .collect())
        }
        None => Ok(Vec::new()),
    }
}

/// Read the STREAMINFO metadata block from a FLAC file.
///
/// Args:
///
///     fp (str | Path): Path to the FLAC file.
///
/// Returns:
///
///     StreamInfo: The FLAC STREAMINFO block.
///
/// Raises:
///
///     OSError: If the file cannot be opened.
///     ValueError: If the file is not a FLAC file.
#[pyfunction]
#[pyo3(signature = (fp: "str | Path"))]
#[allow(clippy::needless_pass_by_value)] // PyO3 extracts `PathBuf` by value
pub fn read_flac_stream_info(fp: std::path::PathBuf) -> PyResult<PyStreamInfo> {
    use audio_samples_io::flac::FlacFile;
    use audio_samples_io::traits::AudioFileMetadata;
    let fp = fp.to_string_lossy();
    let fp: &str = &fp;
    let path = std::path::Path::new(fp);
    if FileType::detect(path) != FileType::FLAC {
        return Err(PyValueError::new_err(
            "read_flac_stream_info requires a FLAC file",
        ));
    }
    let flac = FlacFile::open_metadata(path).map_err(audio_io_err_to_py)?;
    Ok(PyStreamInfo {
        inner: *flac.stream_info(),
    })
}

#[pymodule]
pub fn io(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    let io = PyModule::new(py, "io")?;
    io.add_class::<PyAudioInfo>()?;
    io.add_class::<PyWriteOptions>()?;
    io.add_class::<PyCompressionLevel>()?;
    io.add_class::<PyInfoMetadata>()?;
    io.add_class::<PyCuePoint>()?;
    io.add_class::<PyWavMetadata>()?;
    io.add_class::<PyVorbisComment>()?;
    io.add_class::<PyStreamInfo>()?;

    io.add_function(pyo3::wrap_pyfunction!(info, &io)?)?;
    io.add_function(pyo3::wrap_pyfunction!(read, &io)?)?;
    io.add_function(pyo3::wrap_pyfunction!(read_with_info, &io)?)?;
    io.add_function(pyo3::wrap_pyfunction!(read_and_resample, &io)?)?;
    io.add_function(pyo3::wrap_pyfunction!(peek_native_type, &io)?)?;
    io.add_function(pyo3::wrap_pyfunction!(save, &io)?)?;
    io.add_function(pyo3::wrap_pyfunction!(write_with_options, &io)?)?;
    io.add_function(pyo3::wrap_pyfunction!(set_thread_limit, &io)?)?;
    io.add_function(pyo3::wrap_pyfunction!(thread_limit, &io)?)?;
    io.add_function(pyo3::wrap_pyfunction!(write_with_metadata, &io)?)?;
    io.add_function(pyo3::wrap_pyfunction!(read_wav_info_tags, &io)?)?;
    io.add_function(pyo3::wrap_pyfunction!(read_wav_cue_points, &io)?)?;
    io.add_function(pyo3::wrap_pyfunction!(read_flac_stream_info, &io)?)?;

    // Streaming subsystem.
    crate::io_streaming::register(&io)?;

    reexport!(
        m,
        io,
        "AudioInfo",
        "WriteOptions",
        "CompressionLevel",
        "InfoMetadata",
        "CuePoint",
        "WavMetadata",
        "VorbisComment",
        "StreamInfo",
        "info",
        "read",
        "read_with_info",
        "read_and_resample",
        "peek_native_type",
        "save",
        "write_with_options",
        "write_with_metadata",
        "read_wav_info_tags",
        "read_wav_cue_points",
        "read_flac_stream_info"
    );
    crate::io_streaming::reexport_names(m, &io)?;
    m.add_submodule(&io)?;
    Ok(())
}
