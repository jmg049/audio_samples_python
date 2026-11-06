//! Python bindings for the `audio_samples_io` streaming read/write subsystem.
//!
//! The Rust streaming readers/writers are generic over an arbitrary `Read + Seek` /
//! `Write + Seek` source. The reader takes a path or a Python binary file object; the
//! writers take a path, which monomorphises to `BufWriter<File>`. Every class implements
//! the context-manager protocol (`__enter__`/`__exit__`) so it can be used with `with`.
//!
//! Frame data crosses the boundary as numpy arrays, matching the bulk
//! [`read`](crate::io::read) / [`save`](crate::io::save) idiom: mono audio is a 1-D array of
//! shape `(frames,)` and multi-channel audio is a 2-D array of shape `(channels, frames)`.

use std::fs::{File, OpenOptions as FsOpenOptions};
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom};
use std::num::NonZeroU32;
use std::path::Path;

use audio_samples::{I24, traits::StandardSample};
use audio_samples_io::traits::{
    AudioFileMetadata, AudioStreamReader, AudioStreamWrite, AudioStreamWriter,
};
use audio_samples_io::wav::{StreamedWavFile, StreamedWavWriter, WavSink};
use audio_samples_io::{
    StreamedFlacFile, StreamedFlacWriter, create_streamed_sink, create_streamed_writer,
};
use numpy::Element;
use pyo3::exceptions::{PyRuntimeError, PyStopIteration, PyTypeError, PyValueError};
use pyo3::{
    Bound, Py, PyAny, PyErr, PyResult, Python, pyclass, pymethods,
    types::{PyAnyMethods, PyBytes, PyBytesMethods, PyModule, PyModuleMethods, PyString},
};

use crate::io::PyCompressionLevel;
use crate::{
    PyAudioSamples, audio_io_err_to_py, dispatch_with_view, nzu_or_err, reexport,
    types::PySampleType,
};

// ============================================================================
// Helpers
// ============================================================================

/// The reader backing of the streaming constructors.
type FileReader = BufReader<ReaderSource>;
/// The writer backing used by the path-based streaming constructors.
type FileWriter = BufWriter<File>;

/// Bytes asked of a Python file object per call: each call allocates a `bytes` object of
/// this size at most, so a large read does not hold a second copy of the audio.
const PY_READ_CHUNK: usize = 64 * 1024;

/// What a streaming reader reads from.
enum ReaderSource {
    File(File),
    Python(PyFileSource),
}

impl Read for ReaderSource {
    #[inline]
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        match self {
            Self::File(f) => f.read(buf),
            Self::Python(f) => f.read(buf),
        }
    }
}

impl Seek for ReaderSource {
    #[inline]
    fn seek(&mut self, pos: SeekFrom) -> std::io::Result<u64> {
        match self {
            Self::File(f) => f.seek(pos),
            Self::Python(f) => f.seek(pos),
        }
    }
}

/// A Python binary file object, read through its `read` and `seek` with the thread
/// attached. Offsets count from the position the object had when it was opened.
struct PyFileSource {
    file: Py<PyAny>,
    base: u64,
}

impl PyFileSource {
    fn new(file: &Bound<'_, PyAny>) -> PyResult<Self> {
        for method in ["read", "seek", "tell"] {
            if !file.hasattr(method)? {
                return Err(PyTypeError::new_err(format!(
                    "fp: a file object needs read, seek and tell; this one has no {method}"
                )));
            }
        }
        Ok(Self {
            file: file.clone().unbind(),
            base: file.call_method0("tell")?.extract()?,
        })
    }
}

impl Read for PyFileSource {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        Python::attach(|py| {
            let want = buf.len().min(PY_READ_CHUNK);
            let data = self
                .file
                .bind(py)
                .call_method1("read", (want,))
                .map_err(std::io::Error::other)?;
            let mut copy = |bytes: &[u8]| {
                let src = bytes.get(..want).unwrap_or(bytes);
                let dst = buf.get_mut(..src.len()).ok_or_else(|| {
                    std::io::Error::other("read() returned more bytes than the buffer holds")
                })?;
                dst.copy_from_slice(src);
                Ok(src.len())
            };
            if let Ok(bytes) = data.cast::<PyBytes>() {
                return copy(bytes.as_bytes());
            }
            // Any other object with the buffer protocol; `str` from a text file is not.
            match pyo3::types::PyBytes::new(py, &[])
                .get_type()
                .call1((&data,))
            {
                Ok(bytes) if !data.is_instance_of::<PyString>() => copy(
                    bytes
                        .cast::<PyBytes>()
                        .map_err(|e| std::io::Error::other(PyErr::from(e)))?
                        .as_bytes(),
                ),
                _ => Err(std::io::Error::other(PyTypeError::new_err(
                    "fp: read() must return bytes; open the file in binary mode",
                ))),
            }
        })
    }
}

impl Seek for PyFileSource {
    fn seek(&mut self, pos: SeekFrom) -> std::io::Result<u64> {
        Python::attach(|py| {
            let (offset, whence) = match pos {
                SeekFrom::Start(n) => (
                    i64::try_from(n + self.base).map_err(std::io::Error::other)?,
                    0,
                ),
                SeekFrom::Current(n) => (n, 1),
                SeekFrom::End(n) => (n, 2),
            };
            let file = self.file.bind(py);
            file.call_method1("seek", (offset, whence))
                .map_err(std::io::Error::other)?;
            let at: u64 = file
                .call_method0("tell")
                .and_then(|p| p.extract())
                .map_err(std::io::Error::other)?;
            Ok(at.saturating_sub(self.base))
        })
    }
}

/// Read up to `frame_count` frames into a new NumPy array, which becomes the array of the
/// returned audio. WAV frames stay interleaved on disk order in a Fortran-layout
/// `(channels, frames)` array, as `read` gives; FLAC frames decode to a C-order one.
///
/// Returns `Ok(None)` at end-of-stream (zero frames read).
///
/// Decodes inline, with the thread attached: blocks are short and the result is a new
/// array written directly.
fn read_into_new_array<T>(
    py: Python<'_>,
    inner: &mut ReaderInner,
    channels: u16,
    sample_rate: NonZeroU32,
    frame_count: std::num::NonZeroUsize,
) -> PyResult<Option<PyAudioSamples>>
where
    T: StandardSample + Element + 'static,
{
    use numpy::{PyArray2, PyArrayMethods};

    let channels = usize::from(channels);
    let remaining = match inner {
        ReaderInner::Wav(r) => AudioStreamReader::remaining_frames(r),
        ReaderInner::Flac(r) => AudioStreamReader::remaining_frames(r),
    };
    let frames = frame_count.get().min(remaining);
    if frames == 0 {
        return Ok(None);
    }

    let interleaved = matches!(inner, ReaderInner::Wav(_));
    // SAFETY: the reader writes every value Python sees: the first `frames_read` frames are
    // read below, and a short read keeps only those.
    let array = unsafe { PyArray2::<T>::new(py, [channels, frames], interleaved) };
    // SAFETY: the fresh array is contiguous and no other reference to it exists; its values
    // are written before they are read.
    let slice =
        unsafe { array.as_slice_mut() }.map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let frames_read = match inner {
        ReaderInner::Wav(r) => r.read_interleaved_into(slice),
        ReaderInner::Flac(r) => r.read_planar_into(slice),
    }
    .map_err(audio_io_err_to_py)?;

    if frames_read == 0 {
        return Ok(None);
    }

    let array = if frames_read < frames {
        // Fewer frames than the header says: a truncated file.
        let order = if interleaved { "F" } else { "C" };
        let view = array.call_method1(
            "__getitem__",
            ((
                py.eval(c"slice(None)", None, None)?,
                pyo3::types::PySlice::new(py, 0, frames_read.cast_signed(), 1),
            ),),
        )?;
        view.call_method1("copy", (order,))?
            .cast_into::<PyArray2<T>>()?
    } else {
        array
    };

    Ok(Some(if channels == 1 {
        let mono = array.reshape([frames_read])?;
        PyAudioSamples::new_mono_from_python(mono, sample_rate)
    } else if interleaved {
        PyAudioSamples::new_multi_from_python_interleaved(array, sample_rate)
    } else {
        PyAudioSamples::new_multi_from_python(array, sample_rate)
    }))
}

// ============================================================================
// Streaming readers
// ============================================================================

/// Enum over the concrete file-backed streaming readers so the Python class is unified.
enum ReaderInner {
    Wav(StreamedWavFile<FileReader>),
    Flac(StreamedFlacFile<FileReader>),
}

macro_rules! reader_dispatch {
    ($self:expr, |$r:ident| $body:expr) => {
        match &mut $self.inner {
            Some(ReaderInner::Wav($r)) => $body,
            Some(ReaderInner::Flac($r)) => $body,
            None => {
                return Err(PyRuntimeError::new_err("streaming reader is closed"));
            }
        }
    };
}

/// A streaming audio file reader.
///
/// Opens a WAV or FLAC file and decodes frames on demand instead of loading the whole file
/// into memory. Use it as a context manager and pull frames with :meth:`read_frames`, or
/// iterate with :meth:`frames`, :meth:`windows`, or :meth:`samples`.
///
/// Args:
///
///     fp (str | Path | BinaryIO): Path to the audio file, or a binary file object with
///         ``read``, ``seek`` and ``tell``. The format is detected from the contents.
///
/// Raises:
///
///     OSError: If the file cannot be opened.
///     ValueError: If the format is unsupported or the header is corrupt.
#[pyclass(name = "StreamedAudioReader", module = "audio_samples.io")]
pub struct PyStreamedAudioReader {
    inner: Option<ReaderInner>,
}

impl PyStreamedAudioReader {
    /// The sample type the file stores.
    fn native_type(&self) -> PyResult<PySampleType> {
        let validated = match &self.inner {
            Some(ReaderInner::Wav(r)) => r.sample_type(),
            Some(ReaderInner::Flac(r)) => r.sample_type(),
            None => return Err(PyRuntimeError::new_err("streaming reader is closed")),
        };
        Ok(audio_samples::SampleType::from(validated).into())
    }

    fn open_path(fp: &str) -> PyResult<Self> {
        let path = Path::new(fp);
        // Re-use the crate's format detection by trying each opener based on extension/magic.
        let file_type = audio_samples_io::types::FileType::detect(path);
        if !matches!(
            file_type,
            audio_samples_io::types::FileType::WAV | audio_samples_io::types::FileType::FLAC
        ) {
            return Err(PyValueError::new_err(format!(
                "Unsupported file format for streaming: {file_type:?}"
            )));
        }
        let file = File::open(path)
            .map_err(audio_samples_io::error::AudioIOError::from)
            .map_err(audio_io_err_to_py)?;
        let reader = BufReader::new(ReaderSource::File(file));
        Self::open_reader(reader, file_type, path)
    }

    fn open_reader(
        reader: FileReader,
        file_type: audio_samples_io::types::FileType,
        path: &Path,
    ) -> PyResult<Self> {
        let inner = match file_type {
            audio_samples_io::types::FileType::WAV => ReaderInner::Wav(
                StreamedWavFile::new_with_path(reader, path.to_path_buf())
                    .map_err(audio_io_err_to_py)?,
            ),
            audio_samples_io::types::FileType::FLAC => ReaderInner::Flac(
                StreamedFlacFile::new_with_path(reader, path.to_path_buf())
                    .map_err(audio_io_err_to_py)?,
            ),
            other => {
                return Err(PyValueError::new_err(format!(
                    "Unsupported file format for streaming: {other:?}"
                )));
            }
        };
        Ok(Self { inner: Some(inner) })
    }

    /// Opens a Python binary file object at its current position. The format is detected
    /// from the first bytes.
    fn open_file_object(file: &Bound<'_, PyAny>) -> PyResult<Self> {
        use audio_samples_io::types::FileType;
        let mut source = PyFileSource::new(file)?;
        let mut magic = [0u8; FileType::MAGIC_LEN];
        let mut filled = 0;
        while let Some(rest) = magic.get_mut(filled..).filter(|rest| !rest.is_empty()) {
            match source.read(rest).map_err(file_err)? {
                0 => break,
                n => filled += n,
            }
        }
        source.seek(SeekFrom::Start(0)).map_err(file_err)?;
        let file_type = FileType::from_magic_bytes(magic.get(..filled).unwrap_or(&magic));
        if !matches!(file_type, FileType::WAV | FileType::FLAC) {
            return Err(PyValueError::new_err(
                "fp: a file object must hold WAV or FLAC data",
            ));
        }
        let reader = BufReader::with_capacity(PY_READ_CHUNK, ReaderSource::Python(source));
        Self::open_reader(reader, file_type, Path::new("<file object>"))
    }

    /// Opens what the `fp` of `read` and `StreamedAudioReader` can be: a path or a binary
    /// file object.
    pub(crate) fn open_any(fp: &Bound<'_, PyAny>) -> PyResult<Self> {
        match fp.extract::<std::path::PathBuf>() {
            Ok(path) => Self::open_path(&path.to_string_lossy()),
            Err(not_a_path) => {
                if fp.hasattr("read")? {
                    Self::open_file_object(fp)
                } else {
                    Err(not_a_path)
                }
            }
        }
    }

    /// All the frames that are left, as `read_frames` gives them.
    pub(crate) fn read_rest(
        &mut self,
        py: Python<'_>,
        dtype: Option<PySampleType>,
    ) -> PyResult<PyAudioSamples> {
        let frames = reader_dispatch!(self, |r| AudioStreamReader::remaining_frames(r));
        let frames = std::num::NonZeroUsize::new(frames)
            .ok_or_else(|| PyValueError::new_err("the file holds no audio"))?;
        self.read_frames(py, frames.get(), dtype)?
            .ok_or_else(|| PyValueError::new_err("the file holds no audio"))
    }
}

/// An I/O error of a Python file object, with the exception it came from when there is one.
fn file_err(e: std::io::Error) -> pyo3::PyErr {
    match e.downcast::<pyo3::PyErr>() {
        Ok(py_err) => py_err,
        Err(e) => pyo3::exceptions::PyOSError::new_err(e.to_string()),
    }
}

#[pymethods]
impl PyStreamedAudioReader {
    /// Open a WAV or FLAC file for streaming reads.
    ///
    /// Args:
    ///
    ///     fp (str | Path | BinaryIO): Path to the audio file, or a file object opened in
    ///         binary mode with ``read``, ``seek`` and ``tell``. A file object is read from
    ///         its current position and stays open; decoding calls its methods with the
    ///         GIL held.
    ///
    /// Raises:
    ///
    ///     OSError: If the file cannot be opened.
    ///     ValueError: If the format is unsupported or the header is corrupt.
    ///     TypeError: If ``fp`` is neither a path nor such a file object.
    #[new]
    #[pyo3(signature = (fp: "str | Path | BinaryIO"))]
    fn new(fp: &Bound<'_, pyo3::PyAny>) -> PyResult<Self> {
        Self::open_any(fp)
    }

    /// Sample rate in Hz.
    #[getter]
    fn sample_rate(&self) -> PyResult<u32> {
        match &self.inner {
            Some(ReaderInner::Wav(r)) => Ok(AudioStreamReader::sample_rate(r)),
            Some(ReaderInner::Flac(r)) => Ok(AudioStreamReader::sample_rate(r)),
            None => Err(PyRuntimeError::new_err("streaming reader is closed")),
        }
    }

    /// Number of audio channels.
    #[getter]
    fn num_channels(&self) -> PyResult<u16> {
        match &self.inner {
            Some(ReaderInner::Wav(r)) => Ok(AudioStreamReader::num_channels(r)),
            Some(ReaderInner::Flac(r)) => Ok(AudioStreamReader::num_channels(r)),
            None => Err(PyRuntimeError::new_err("streaming reader is closed")),
        }
    }

    /// Total number of frames in the stream.
    #[getter]
    fn total_frames(&self) -> PyResult<usize> {
        match &self.inner {
            Some(ReaderInner::Wav(r)) => Ok(AudioStreamReader::total_frames(r)),
            Some(ReaderInner::Flac(r)) => Ok(AudioStreamReader::total_frames(r)),
            None => Err(PyRuntimeError::new_err("streaming reader is closed")),
        }
    }

    /// Current frame position (0-indexed).
    #[getter]
    fn current_frame(&self) -> PyResult<usize> {
        match &self.inner {
            Some(ReaderInner::Wav(r)) => Ok(r.current_frame()),
            Some(ReaderInner::Flac(r)) => Ok(r.current_frame()),
            None => Err(PyRuntimeError::new_err("streaming reader is closed")),
        }
    }

    /// Number of frames remaining from the current position.
    #[getter]
    fn remaining_frames(&self) -> PyResult<usize> {
        match &self.inner {
            Some(ReaderInner::Wav(r)) => Ok(r.remaining_frames()),
            Some(ReaderInner::Flac(r)) => Ok(r.remaining_frames()),
            None => Err(PyRuntimeError::new_err("streaming reader is closed")),
        }
    }

    /// Read up to ``frame_count`` frames and return them as an :class:`AudioSamples`.
    ///
    /// Frames are decoded and converted to the requested ``dtype`` on the fly. Returns
    /// ``None`` once the end of the stream is reached (no frames left to read).
    ///
    /// Args:
    ///
    ///     frame_count (int): Maximum number of frames to read; must be greater than zero.
    ///     dtype (SampleType, optional): Target sample type for the returned array.
    ///         Defaults to the file's own sample type.
    ///
    /// Returns:
    ///
    ///     AudioSamples | None: The decoded chunk (mono shape ``(frames,)`` or multi-channel
    ///     shape ``(channels, frames)``), or ``None`` at end of stream.
    ///
    /// Raises:
    ///
    ///     ValueError: If ``frame_count`` is zero.
    ///     RuntimeError: If the reader is closed.
    ///     OSError: If reading fails or the data is corrupt.
    #[pyo3(signature = (frame_count: "int", dtype: "SampleType" = None))]
    fn read_frames(
        &mut self,
        py: Python<'_>,
        frame_count: usize,
        dtype: Option<PySampleType>,
    ) -> PyResult<Option<PyAudioSamples>> {
        let frame_count = nzu_or_err(frame_count)?;
        let dtype = dtype.map_or_else(|| self.native_type(), Ok)?;
        let channels = self.num_channels()?;
        let sample_rate = nzu32_sr(self.sample_rate()?)?;

        let inner = self
            .inner
            .as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("streaming reader is closed"))?;
        macro_rules! go {
            ($T:ty) => {
                read_into_new_array::<$T>(py, inner, channels, sample_rate, frame_count)
            };
        }
        match dtype {
            PySampleType::U8 => go!(u8),
            PySampleType::I16 => go!(i16),
            PySampleType::I24 => go!(I24),
            PySampleType::I32 => go!(i32),
            PySampleType::F32 => go!(f32),
            PySampleType::F64 => go!(f64),
        }
    }

    /// Seek so the next read returns frame ``frame``.
    ///
    /// Args:
    ///
    ///     frame (int): Target frame index (0-indexed).
    ///
    /// Raises:
    ///
    ///     OSError: If the frame is beyond the end of the stream or the seek fails.
    ///     RuntimeError: If the reader is closed.
    #[pyo3(signature = (frame: "int"))]
    fn seek_to_frame(&mut self, frame: usize) -> PyResult<()> {
        reader_dispatch!(self, |r| r.seek_to_frame(frame).map_err(audio_io_err_to_py))
    }

    /// Reset the stream to the beginning of the audio data.
    ///
    /// Raises:
    ///
    ///     OSError: If the underlying seek fails.
    ///     RuntimeError: If the reader is closed.
    #[pyo3(signature = ())]
    fn reset(&mut self) -> PyResult<()> {
        reader_dispatch!(self, |r| r.reset().map_err(audio_io_err_to_py))
    }

    /// Return an iterator yielding fixed-size chunks of ``chunk_frames`` frames.
    ///
    /// Each iteration calls :meth:`read_frames`; the final chunk may be shorter.
    ///
    /// Args:
    ///
    ///     chunk_frames (int): Frames per chunk; must be greater than zero. Defaults to 4096.
    ///     dtype (SampleType, optional): Target sample type. Defaults to the file's own
    ///         sample type.
    ///
    /// Returns:
    ///
    ///     StreamedFrameIterator: An iterator of :class:`AudioSamples` chunks.
    #[pyo3(signature = (chunk_frames: "int" = 4096, dtype: "SampleType" = None))]
    fn frames(
        slf: Py<Self>,
        chunk_frames: usize,
        dtype: Option<PySampleType>,
    ) -> PyResult<PyStreamedFrameIterator> {
        let _ = nzu_or_err(chunk_frames)?;
        Ok(PyStreamedFrameIterator {
            reader: slf,
            window: chunk_frames,
            hop: chunk_frames,
            dtype,
        })
    }

    /// Return an iterator yielding overlapping windows.
    ///
    /// Each window contains ``window`` frames; the read position advances by ``hop`` frames
    /// between windows (so ``hop < window`` produces overlap). Windows shorter than
    /// ``window`` at the end of the stream are still yielded.
    ///
    /// Args:
    ///
    ///     window (int): Frames per window; must be greater than zero.
    ///     hop (int): Frames to advance between windows; must be greater than zero.
    ///     dtype (SampleType, optional): Target sample type. Defaults to the file's own
    ///         sample type.
    ///
    /// Returns:
    ///
    ///     StreamedFrameIterator: An iterator of :class:`AudioSamples` windows.
    ///
    /// Raises:
    ///
    ///     ValueError: If ``window`` or ``hop`` is zero.
    #[pyo3(signature = (window: "int", hop: "int", dtype: "SampleType" = None))]
    fn windows(
        slf: Py<Self>,
        window: usize,
        hop: usize,
        dtype: Option<PySampleType>,
    ) -> PyResult<PyStreamedFrameIterator> {
        let _ = nzu_or_err(window)?;
        let _ = nzu_or_err(hop)?;
        Ok(PyStreamedFrameIterator {
            reader: slf,
            window,
            hop,
            dtype,
        })
    }

    /// Return an iterator yielding single frames (one frame per step).
    ///
    /// Each item is an :class:`AudioSamples` of one frame. This is convenient but slower than
    /// :meth:`frames`; prefer larger chunks for throughput.
    ///
    /// Args:
    ///
    ///     dtype (SampleType, optional): Target sample type. Defaults to the file's own
    ///         sample type.
    ///
    /// Returns:
    ///
    ///     StreamedFrameIterator: An iterator of single-frame :class:`AudioSamples`.
    #[pyo3(signature = (dtype: "SampleType" = None))]
    const fn samples(slf: Py<Self>, dtype: Option<PySampleType>) -> PyStreamedFrameIterator {
        PyStreamedFrameIterator {
            reader: slf,
            window: 1,
            hop: 1,
            dtype,
        }
    }

    /// Close the reader and release the underlying file handle.
    ///
    /// Further reads raise :class:`RuntimeError`. Idempotent.
    #[pyo3(signature = ())]
    fn close(&mut self) {
        self.inner = None;
    }

    /// Enter the context manager, returning ``self``.
    const fn __enter__(slf: Py<Self>) -> Py<Self> {
        slf
    }

    /// Exit the context manager, closing the reader.
    #[pyo3(signature = (_exc_type=None, _exc_value=None, _traceback=None))]
    fn __exit__(
        &mut self,
        _exc_type: Option<Bound<'_, pyo3::PyAny>>,
        _exc_value: Option<Bound<'_, pyo3::PyAny>>,
        _traceback: Option<Bound<'_, pyo3::PyAny>>,
    ) -> bool {
        self.close();
        false
    }
}

fn nzu32_sr(sr: u32) -> PyResult<NonZeroU32> {
    NonZeroU32::new(sr).ok_or_else(|| PyValueError::new_err("sample rate must be non-zero"))
}

/// Iterator over chunks / windows / single frames of a :class:`StreamedAudioReader`.
///
/// Created by :meth:`StreamedAudioReader.frames`, :meth:`StreamedAudioReader.windows`, and
/// :meth:`StreamedAudioReader.samples`. Yields :class:`AudioSamples`; for ``hop != window``
/// the iterator seeks the underlying reader between steps.
#[pyclass(name = "StreamedFrameIterator", module = "audio_samples.io")]
pub struct PyStreamedFrameIterator {
    reader: Py<PyStreamedAudioReader>,
    window: usize,
    hop: usize,
    dtype: Option<PySampleType>,
}

#[pymethods]
impl PyStreamedFrameIterator {
    const fn __iter__(slf: Py<Self>) -> Py<Self> {
        slf
    }

    fn __next__(&mut self, py: Python<'_>) -> PyResult<PyAudioSamples> {
        let mut reader = self.reader.borrow_mut(py);
        // Record position so overlapping windows can rewind by (window - hop).
        let start = reader.current_frame()?;
        let chunk = reader.read_frames(py, self.window, self.dtype)?;
        match chunk {
            Some(samples) => {
                // For overlapping windows seek back so the next window starts at start + hop.
                if self.hop != self.window {
                    let next = start + self.hop;
                    let total = reader.total_frames()?;
                    if next < total {
                        reader.seek_to_frame(next)?;
                    }
                }
                Ok(samples)
            }
            None => Err(PyStopIteration::new_err(())),
        }
    }
}

// ============================================================================
// Streaming writers
// ============================================================================

/// Enum over the concrete file-backed streaming writers (seekable).
enum WriterInner {
    Wav(StreamedWavWriter<FileWriter>),
    Flac(StreamedFlacWriter<FileWriter>),
}

macro_rules! writer_dispatch {
    ($self:expr, |$w:ident| $body:expr) => {
        match &mut $self.inner {
            Some(WriterInner::Wav($w)) => $body,
            Some(WriterInner::Flac($w)) => $body,
            None => return Err(PyRuntimeError::new_err("streaming writer is closed")),
        }
    };
}

/// A streaming audio file writer.
///
/// Encodes :class:`AudioSamples` chunks incrementally to a WAV or FLAC file (chosen from the
/// extension) without buffering the whole signal. The output sample type is fixed at creation
/// from ``dtype``. Always :meth:`finalize` (or use the ``with`` block) so format headers are
/// patched with the final sizes.
///
/// Args:
///
///     fp (str | Path): Output path; ``.wav`` or ``.flac`` selects the format.
///     channels (int): Number of channels; must be greater than zero.
///     sample_rate (int): Sample rate in Hz; must be greater than zero.
///     dtype (SampleType, optional): Output sample type. Defaults to ``f32``. FLAC supports
///         only integer types (``i16``/``i24``/``i32``).
///
/// Raises:
///
///     OSError: If the file cannot be created.
///     ValueError: If parameters are invalid or the type/format combination is unsupported.
#[pyclass(name = "StreamedAudioWriter", module = "audio_samples.io")]
pub struct PyStreamedAudioWriter {
    inner: Option<WriterInner>,
}

/// Dispatch on a sample type to a `create_streamed_writer::<_, T>` call (WAV).
fn make_wav_writer(
    writer: FileWriter,
    channels: u16,
    sample_rate: u32,
    dtype: PySampleType,
) -> PyResult<StreamedWavWriter<FileWriter>> {
    let r = match dtype {
        PySampleType::U8 | PySampleType::I16 => {
            create_streamed_writer::<_, i16>(writer, channels, sample_rate)
        }
        PySampleType::I24 => create_streamed_writer::<_, I24>(writer, channels, sample_rate),
        PySampleType::I32 => create_streamed_writer::<_, i32>(writer, channels, sample_rate),
        PySampleType::F32 => create_streamed_writer::<_, f32>(writer, channels, sample_rate),
        PySampleType::F64 => create_streamed_writer::<_, f64>(writer, channels, sample_rate),
    };
    r.map_err(audio_io_err_to_py)
}

#[pymethods]
impl PyStreamedAudioWriter {
    /// Create a streaming writer to ``fp`` (format chosen from the extension).
    #[new]
    #[pyo3(signature = (fp: "str | Path", channels: "int", sample_rate: "int", dtype: "SampleType" = None))]
    #[allow(clippy::needless_pass_by_value)] // PyO3 extracts `PathBuf` by value
    fn new(
        fp: std::path::PathBuf,
        channels: u16,
        sample_rate: u32,
        dtype: Option<PySampleType>,
    ) -> PyResult<Self> {
        let fp = fp.to_string_lossy();
        let fp: &str = &fp;
        let dtype = dtype.unwrap_or(PySampleType::F32);
        let path = Path::new(fp);
        let file = File::create(path).map_err(|e| audio_io_err_to_py(e.into()))?;
        let writer = BufWriter::with_capacity(256 * 1024, file);
        let inner = match audio_samples_io::types::FileType::from_path(path) {
            audio_samples_io::types::FileType::WAV => {
                WriterInner::Wav(make_wav_writer(writer, channels, sample_rate, dtype)?)
            }
            audio_samples_io::types::FileType::FLAC => {
                let w = create_streamed_flac_for(writer, channels, sample_rate, dtype)?;
                WriterInner::Flac(w)
            }
            other => {
                return Err(PyValueError::new_err(format!(
                    "Unsupported output format for streaming write: {other:?}"
                )));
            }
        };
        Ok(Self { inner: Some(inner) })
    }

    /// Create a streaming FLAC writer with an explicit compression level.
    ///
    /// Args:
    ///
    ///     fp (str | Path): Output ``.flac`` path.
    ///     channels (int): Number of channels; must be greater than zero.
    ///     sample_rate (int): Sample rate in Hz; must be greater than zero.
    ///     compression (CompressionLevel, optional): FLAC compression level (0-8).
    ///         Defaults to the codec default (5).
    ///     dtype (SampleType, optional): Integer output type (``i16``/``i24``/``i32``).
    ///         Defaults to ``i16``.
    ///
    /// Returns:
    ///
    ///     StreamedAudioWriter: A FLAC streaming writer.
    ///
    /// Raises:
    ///
    ///     OSError: If the file cannot be created.
    ///     ValueError: If the type is not a FLAC-supported integer type.
    #[staticmethod]
    #[pyo3(signature = (fp: "str | Path", channels: "int", sample_rate: "int", compression: "CompressionLevel" = None, dtype: "SampleType" = None))]
    #[allow(clippy::needless_pass_by_value)] // PyO3 extracts `PathBuf` by value
    fn create_flac(
        fp: std::path::PathBuf,
        channels: u16,
        sample_rate: u32,
        compression: Option<PyCompressionLevel>,
        dtype: Option<PySampleType>,
    ) -> PyResult<Self> {
        let fp = fp.to_string_lossy();
        let fp: &str = &fp;
        let dtype = dtype.unwrap_or(PySampleType::I16);
        let path = Path::new(fp);
        if audio_samples_io::types::FileType::from_path(path)
            != audio_samples_io::types::FileType::FLAC
        {
            return Err(PyValueError::new_err(
                "create_flac requires a .flac output path",
            ));
        }
        let file = File::create(path).map_err(|e| audio_io_err_to_py(e.into()))?;
        let writer = BufWriter::with_capacity(256 * 1024, file);
        let sample_type = flac_validated_type(dtype)?;
        let level =
            compression.map_or_else(audio_samples_io::CompressionLevel::default, |c| c.inner);
        let w = StreamedFlacWriter::new(writer, channels, sample_rate, sample_type, level)
            .map_err(audio_io_err_to_py)?;
        Ok(Self {
            inner: Some(WriterInner::Flac(w)),
        })
    }

    /// Write a chunk of audio frames to the stream.
    ///
    /// Samples are converted from the array's dtype to the writer's configured output type.
    /// The channel count must match the writer's configuration.
    ///
    /// Args:
    ///
    ///     samples (AudioSamples): Frames to write.
    ///
    /// Returns:
    ///
    ///     int: The number of frames written.
    ///
    /// Raises:
    ///
    ///     RuntimeError: If the writer is closed or already finalized.
    ///     ValueError: If the channel count does not match.
    ///     OSError: If the underlying write fails.
    #[pyo3(signature = (samples: "AudioSamples"))]
    fn write_frames(&mut self, py: Python<'_>, samples: &PyAudioSamples) -> PyResult<usize> {
        writer_dispatch!(self, |w| {
            dispatch_with_view!(samples, py, |audio| {
                w.write_frames(&audio).map_err(audio_io_err_to_py)
            })
        })
    }

    /// Flush buffered data to the underlying file without finalizing.
    ///
    /// Raises:
    ///
    ///     RuntimeError: If the writer is closed.
    ///     OSError: If the flush fails.
    #[pyo3(signature = ())]
    fn flush(&mut self) -> PyResult<()> {
        writer_dispatch!(self, |w| w.flush().map_err(audio_io_err_to_py))
    }

    /// Finalize the stream, patching format headers with final sizes.
    ///
    /// Must be called exactly once when writing is done (idempotent). After finalizing,
    /// further :meth:`write_frames` calls fail.
    ///
    /// Raises:
    ///
    ///     RuntimeError: If the writer is closed.
    ///     OSError: If the underlying seek/flush fails.
    #[pyo3(signature = ())]
    fn finalize(&mut self) -> PyResult<()> {
        writer_dispatch!(self, |w| w.finalize().map_err(audio_io_err_to_py))
    }

    /// Whether the stream has been finalized.
    #[getter]
    fn is_finalized(&self) -> PyResult<bool> {
        match &self.inner {
            Some(WriterInner::Wav(w)) => Ok(w.is_finalized()),
            Some(WriterInner::Flac(w)) => Ok(w.is_finalized()),
            None => Err(PyRuntimeError::new_err("streaming writer is closed")),
        }
    }

    /// Number of frames written so far.
    #[getter]
    fn frames_written(&self) -> PyResult<usize> {
        match &self.inner {
            Some(WriterInner::Wav(w)) => Ok(w.frames_written()),
            Some(WriterInner::Flac(w)) => Ok(w.frames_written()),
            None => Err(PyRuntimeError::new_err("streaming writer is closed")),
        }
    }

    /// Sample rate the writer was configured with, in Hz.
    #[getter]
    fn sample_rate(&self) -> PyResult<u32> {
        match &self.inner {
            Some(WriterInner::Wav(w)) => Ok(AudioStreamWriter::sample_rate(w)),
            Some(WriterInner::Flac(w)) => Ok(AudioStreamWriter::sample_rate(w)),
            None => Err(PyRuntimeError::new_err("streaming writer is closed")),
        }
    }

    /// Number of channels the writer was configured with.
    #[getter]
    fn num_channels(&self) -> PyResult<u16> {
        match &self.inner {
            Some(WriterInner::Wav(w)) => Ok(AudioStreamWriter::num_channels(w)),
            Some(WriterInner::Flac(w)) => Ok(AudioStreamWriter::num_channels(w)),
            None => Err(PyRuntimeError::new_err("streaming writer is closed")),
        }
    }

    /// Finalize (if not already) and release the file handle. Idempotent.
    ///
    /// Raises:
    ///
    ///     OSError: If finalization fails.
    #[pyo3(signature = ())]
    fn close(&mut self) -> PyResult<()> {
        if let Some(inner) = &mut self.inner {
            let res = match inner {
                WriterInner::Wav(w) => w.finalize(),
                WriterInner::Flac(w) => w.finalize(),
            };
            self.inner = None;
            res.map_err(audio_io_err_to_py)?;
        }
        Ok(())
    }

    /// Enter the context manager, returning ``self``.
    const fn __enter__(slf: Py<Self>) -> Py<Self> {
        slf
    }

    /// Exit the context manager, finalizing and closing the writer.
    #[pyo3(signature = (_exc_type=None, _exc_value=None, _traceback=None))]
    fn __exit__(
        &mut self,
        _exc_type: Option<Bound<'_, pyo3::PyAny>>,
        _exc_value: Option<Bound<'_, pyo3::PyAny>>,
        _traceback: Option<Bound<'_, pyo3::PyAny>>,
    ) -> PyResult<bool> {
        self.close()?;
        Ok(false)
    }
}

/// Map a `PySampleType` to a FLAC-compatible `ValidatedSampleType`.
fn flac_validated_type(dtype: PySampleType) -> PyResult<audio_samples_io::ValidatedSampleType> {
    use audio_samples_io::ValidatedSampleType as V;
    match dtype {
        PySampleType::I16 | PySampleType::U8 => Ok(V::I16),
        PySampleType::I24 => Ok(V::I24),
        PySampleType::I32 => Ok(V::I32),
        _ => Err(PyValueError::new_err(
            "FLAC supports only integer sample types (i16, i24, i32)",
        )),
    }
}

/// Build a default-compression FLAC streaming writer for the given dtype.
fn create_streamed_flac_for(
    writer: FileWriter,
    channels: u16,
    sample_rate: u32,
    dtype: PySampleType,
) -> PyResult<StreamedFlacWriter<FileWriter>> {
    let sample_type = flac_validated_type(dtype)?;
    StreamedFlacWriter::new(
        writer,
        channels,
        sample_rate,
        sample_type,
        audio_samples_io::CompressionLevel::default(),
    )
    .map_err(audio_io_err_to_py)
}

/// A non-seekable streaming WAV writer ("sink").
///
/// Writes a WAV stream to a file opened in append/write mode where seeking back to patch the
/// header is undesirable. Because the header cannot be back-patched, the final frame count is
/// either declared up front (``total_frames``) or left open-ended (streaming-size convention).
///
/// Args:
///
///     fp (str | Path): Output ``.wav`` path.
///     channels (int): Number of channels; must be greater than zero.
///     sample_rate (int): Sample rate in Hz; must be greater than zero.
///     dtype (SampleType, optional): Output sample type. Defaults to ``i16``.
///     total_frames (int, optional): Final frame count if known (produces a standard file and
///         verifies the count on :meth:`finalize`); ``None`` for an open-ended stream.
///
/// Raises:
///
///     OSError: If the file cannot be created.
///     ValueError: If parameters are invalid.
#[pyclass(name = "WavSink", module = "audio_samples.io")]
pub struct PyWavSink {
    inner: Option<WavSink<File>>,
}

#[pymethods]
impl PyWavSink {
    #[new]
    #[pyo3(signature = (fp: "str | Path", channels: "int", sample_rate: "int", dtype: "SampleType" = None, total_frames: "int" = None))]
    #[allow(clippy::needless_pass_by_value)] // PyO3 extracts `PathBuf` by value
    fn new(
        fp: std::path::PathBuf,
        channels: u16,
        sample_rate: u32,
        dtype: Option<PySampleType>,
        total_frames: Option<usize>,
    ) -> PyResult<Self> {
        let fp = fp.to_string_lossy();
        let fp: &str = &fp;
        let dtype = dtype.unwrap_or(PySampleType::I16);
        let sample_type = match dtype {
            PySampleType::U8 => audio_samples_io::ValidatedSampleType::U8,
            PySampleType::I16 => audio_samples_io::ValidatedSampleType::I16,
            PySampleType::I24 => audio_samples_io::ValidatedSampleType::I24,
            PySampleType::I32 => audio_samples_io::ValidatedSampleType::I32,
            PySampleType::F32 => audio_samples_io::ValidatedSampleType::F32,
            PySampleType::F64 => audio_samples_io::ValidatedSampleType::F64,
        };
        let file = FsOpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(Path::new(fp))
            .map_err(|e| audio_io_err_to_py(e.into()))?;
        let sink =
            create_streamed_sink_typed(file, channels, sample_rate, sample_type, total_frames)?;
        Ok(Self { inner: Some(sink) })
    }

    /// Write a chunk of audio frames to the sink.
    ///
    /// Args:
    ///
    ///     samples (AudioSamples): Frames to write; channel count must match.
    ///
    /// Returns:
    ///
    ///     int: The number of frames written.
    ///
    /// Raises:
    ///
    ///     RuntimeError: If the sink is closed or finalized.
    ///     ValueError: If the channel count mismatches or the declared length is exceeded.
    ///     OSError: If the write fails.
    #[pyo3(signature = (samples: "AudioSamples"))]
    fn write_frames(&mut self, py: Python<'_>, samples: &PyAudioSamples) -> PyResult<usize> {
        match &mut self.inner {
            Some(sink) => dispatch_with_view!(samples, py, |audio| {
                sink.write_frames(&audio).map_err(audio_io_err_to_py)
            }),
            None => Err(PyRuntimeError::new_err("sink is closed")),
        }
    }

    /// Flush buffered data to the underlying file.
    #[pyo3(signature = ())]
    fn flush(&mut self) -> PyResult<()> {
        let Some(sink) = &mut self.inner else {
            return Err(PyRuntimeError::new_err("sink is closed"));
        };
        sink.flush().map_err(audio_io_err_to_py)
    }

    /// Finalize the sink (idempotent). For a declared-length sink this verifies the count.
    #[pyo3(signature = ())]
    fn finalize(&mut self) -> PyResult<()> {
        let Some(sink) = &mut self.inner else {
            return Err(PyRuntimeError::new_err("sink is closed"));
        };
        sink.finalize().map_err(audio_io_err_to_py)
    }

    /// Number of frames written so far.
    #[getter]
    fn frames_written(&self) -> PyResult<usize> {
        self.inner.as_ref().map_or_else(
            || Err(PyRuntimeError::new_err("sink is closed")),
            |sink| Ok(sink.frames_written()),
        )
    }

    /// Finalize (if needed) and release the file handle. Idempotent.
    #[pyo3(signature = ())]
    fn close(&mut self) -> PyResult<()> {
        if let Some(sink) = &mut self.inner {
            let res = sink.finalize();
            self.inner = None;
            res.map_err(audio_io_err_to_py)?;
        }
        Ok(())
    }

    const fn __enter__(slf: Py<Self>) -> Py<Self> {
        slf
    }

    #[pyo3(signature = (_exc_type=None, _exc_value=None, _traceback=None))]
    fn __exit__(
        &mut self,
        _exc_type: Option<Bound<'_, pyo3::PyAny>>,
        _exc_value: Option<Bound<'_, pyo3::PyAny>>,
        _traceback: Option<Bound<'_, pyo3::PyAny>>,
    ) -> PyResult<bool> {
        self.close()?;
        Ok(false)
    }
}

/// Build a `WavSink` for a concrete dtype via the crate helper (which infers the type from a
/// generic param). We dispatch the generic call here.
fn create_streamed_sink_typed(
    file: File,
    channels: u16,
    sample_rate: u32,
    sample_type: audio_samples_io::ValidatedSampleType,
    total_frames: Option<usize>,
) -> PyResult<WavSink<File>> {
    use audio_samples_io::ValidatedSampleType as V;
    let r = match sample_type {
        V::U8 | V::I16 => create_streamed_sink::<_, i16>(file, channels, sample_rate, total_frames),
        V::I24 => create_streamed_sink::<_, I24>(file, channels, sample_rate, total_frames),
        V::I32 => create_streamed_sink::<_, i32>(file, channels, sample_rate, total_frames),
        V::F32 => create_streamed_sink::<_, f32>(file, channels, sample_rate, total_frames),
        V::F64 => create_streamed_sink::<_, f64>(file, channels, sample_rate, total_frames),
    };
    r.map_err(audio_io_err_to_py)
}

/// Register the streaming classes onto the `io` submodule.
pub fn register(io: &Bound<'_, PyModule>) -> PyResult<()> {
    io.add_class::<PyStreamedAudioReader>()?;
    io.add_class::<PyStreamedFrameIterator>()?;
    io.add_class::<PyStreamedAudioWriter>()?;
    io.add_class::<PyWavSink>()?;
    Ok(())
}

/// Re-export the streaming class names from the parent module.
pub fn reexport_names(parent: &Bound<'_, PyModule>, io: &Bound<'_, PyModule>) -> PyResult<()> {
    reexport!(
        parent,
        io,
        "StreamedAudioReader",
        "StreamedFrameIterator",
        "StreamedAudioWriter",
        "WavSink"
    );
    Ok(())
}
