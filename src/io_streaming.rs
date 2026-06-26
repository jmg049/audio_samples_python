//! Python bindings for the `audio_samples_io` streaming read/write subsystem.
//!
//! The Rust streaming readers/writers are generic over an arbitrary `Read + Seek` /
//! `Write + Seek` source. The Python bindings only expose the file-path constructors,
//! which monomorphise to `BufReader<File>` / `BufWriter<File>`. Every class implements the
//! context-manager protocol (`__enter__`/`__exit__`) so it can be used with `with`.
//!
//! Frame data crosses the boundary as numpy arrays, matching the bulk
//! [`read`](crate::io::read) / [`save`](crate::io::save) idiom: mono audio is a 1-D array of
//! shape `(frames,)` and multi-channel audio is a 2-D array of shape `(channels, frames)`.

use std::fs::{File, OpenOptions as FsOpenOptions};
use std::io::{BufReader, BufWriter};
use std::num::NonZeroU32;
use std::path::Path;

use audio_samples::{AudioSamples, ConvertFrom, ConvertTo, I24, traits::StandardSample};
use audio_samples_io::traits::{
    AudioStreamRead, AudioStreamReader, AudioStreamWrite, AudioStreamWriter,
};
use audio_samples_io::wav::{StreamedWavFile, StreamedWavWriter, WavSink};
use audio_samples_io::{
    StreamedFlacFile, StreamedFlacWriter, create_streamed_sink, create_streamed_writer,
    open_streamed, open_streamed_flac,
};
use numpy::Element;
use pyo3::exceptions::{PyRuntimeError, PyStopIteration, PyValueError};
use pyo3::{
    Bound, Py, PyResult, Python, pyclass, pymethods,
    types::{PyAnyMethods, PyModule, PyModuleMethods},
};

use crate::io::PyCompressionLevel;
use crate::{
    PyAudioSamples, audio_io_err_to_py, dispatch_with_view, nzu_or_err, reexport,
    types::PySampleType,
};

// ============================================================================
// Helpers
// ============================================================================

/// The reader backing used by the path-based streaming constructors.
type FileReader = BufReader<File>;
/// The writer backing used by the path-based streaming constructors.
type FileWriter = BufWriter<File>;

/// Build a fresh `AudioSamples<T>` of the right shape for `channels`, fill it from the
/// reader via `read_frames_into`, and return `(numpy_array, frames_read)`.
///
/// Returns `Ok(None)` at end-of-stream (zero frames read).
fn read_into_new_array<T, S>(
    py: Python<'_>,
    reader: &mut S,
    channels: u16,
    sample_rate: NonZeroU32,
    frame_count: std::num::NonZeroUsize,
) -> PyResult<Option<PyAudioSamples>>
where
    T: StandardSample + Element + ConvertTo<T> + ConvertFrom<T> + 'static,
    S: AudioStreamRead,
{
    let mut buffer = if channels == 1 {
        AudioSamples::<T>::zeros_mono(frame_count, sample_rate)
    } else {
        // SAFETY: channels validated > 0 by the reader during construction.
        let ch = NonZeroU32::new(u32::from(channels)).expect("channels must be non-zero");
        AudioSamples::<T>::zeros_multi(ch, frame_count, sample_rate)
    };

    // The streaming readers are !Send (they hold non-Send buffers), so we cannot release the
    // GIL across the decode the way the bulk resample path does. Decode inline.
    let _ = py;
    let frames_read = reader
        .read_frames_into::<T>(&mut buffer, frame_count)
        .map_err(audio_io_err_to_py)?;

    if frames_read == 0 {
        return Ok(None);
    }

    // Trim the buffer to the frames actually read, then materialise a numpy-backed
    // PyAudioSamples via the owned-array constructors.
    let py_samples = if channels == 1 {
        let arr = buffer
            .into_array1()
            .ok_or_else(|| PyRuntimeError::new_err("expected mono buffer"))?;
        let arr = arr.slice_move(numpy::ndarray::s![..frames_read]);
        PyAudioSamples::new_mono(arr, sample_rate)
    } else {
        let arr = buffer
            .into_array2()
            .ok_or_else(|| PyRuntimeError::new_err("expected multi-channel buffer"))?;
        let arr = arr.slice_move(numpy::ndarray::s![.., ..frames_read]).to_owned();
        PyAudioSamples::new_multi(arr, sample_rate)
    };
    Ok(Some(py_samples))
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
                return Err(PyRuntimeError::new_err(
                    "streaming reader is closed",
                ));
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
///     fp (str | Path): Path to the audio file. The format is detected from the contents.
///
/// Raises:
///     OSError: If the file cannot be opened.
///     ValueError: If the format is unsupported or the header is corrupt.
#[pyclass(name = "StreamedAudioReader", module = "audio_samples.io", unsendable)]
pub struct PyStreamedAudioReader {
    inner: Option<ReaderInner>,
}

impl PyStreamedAudioReader {
    fn open_path(fp: &str) -> PyResult<Self> {
        let path = Path::new(fp);
        // Re-use the crate's format detection by trying each opener based on extension/magic.
        match audio_samples_io::types::FileType::detect(path) {
            audio_samples_io::types::FileType::WAV => {
                let r = open_streamed(path).map_err(audio_io_err_to_py)?;
                Ok(Self {
                    inner: Some(ReaderInner::Wav(r)),
                })
            }
            audio_samples_io::types::FileType::FLAC => {
                let r = open_streamed_flac(path).map_err(audio_io_err_to_py)?;
                Ok(Self {
                    inner: Some(ReaderInner::Flac(r)),
                })
            }
            other => Err(PyValueError::new_err(format!(
                "Unsupported file format for streaming: {other:?}"
            ))),
        }
    }
}

#[pymethods]
impl PyStreamedAudioReader {
    /// Open a WAV or FLAC file for streaming reads.
    ///
    /// Args:
    ///     fp (str | Path): Path to the audio file.
    ///
    /// Raises:
    ///     OSError: If the file cannot be opened.
    ///     ValueError: If the format is unsupported or the header is corrupt.
    #[new]
    #[pyo3(signature = (fp: "str | Path"), text_signature = "(fp: str | Path)")]
    fn new(fp: &str) -> PyResult<Self> {
        Self::open_path(fp)
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
    ///     frame_count (int): Maximum number of frames to read; must be greater than zero.
    ///     dtype (SampleType, optional): Target sample type for the returned array.
    ///         Defaults to ``f32``.
    ///
    /// Returns:
    ///     AudioSamples | None: The decoded chunk (mono shape ``(frames,)`` or multi-channel
    ///     shape ``(channels, frames)``), or ``None`` at end of stream.
    ///
    /// Raises:
    ///     ValueError: If ``frame_count`` is zero.
    ///     RuntimeError: If the reader is closed.
    ///     OSError: If reading fails or the data is corrupt.
    #[pyo3(signature = (frame_count: "int", dtype: "SampleType" = None), text_signature = "($self, frame_count: int, dtype: SampleType = None) -> AudioSamples | None")]
    fn read_frames(
        &mut self,
        py: Python<'_>,
        frame_count: usize,
        dtype: Option<PySampleType>,
    ) -> PyResult<Option<PyAudioSamples>> {
        let frame_count = nzu_or_err(frame_count)?;
        let dtype = dtype.unwrap_or(PySampleType::F32);
        let channels = self.num_channels()?;
        let sample_rate = nzu32_sr(self.sample_rate()?)?;

        macro_rules! go {
            ($T:ty) => {
                reader_dispatch!(self, |r| read_into_new_array::<$T, _>(
                    py,
                    r,
                    channels,
                    sample_rate,
                    frame_count
                ))
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
    ///     frame (int): Target frame index (0-indexed).
    ///
    /// Raises:
    ///     OSError: If the frame is beyond the end of the stream or the seek fails.
    ///     RuntimeError: If the reader is closed.
    #[pyo3(signature = (frame: "int"), text_signature = "($self, frame: int) -> None")]
    fn seek_to_frame(&mut self, frame: usize) -> PyResult<()> {
        reader_dispatch!(self, |r| r.seek_to_frame(frame).map_err(audio_io_err_to_py))
    }

    /// Reset the stream to the beginning of the audio data.
    ///
    /// Raises:
    ///     OSError: If the underlying seek fails.
    ///     RuntimeError: If the reader is closed.
    #[pyo3(signature = (), text_signature = "($self) -> None")]
    fn reset(&mut self) -> PyResult<()> {
        reader_dispatch!(self, |r| r.reset().map_err(audio_io_err_to_py))
    }

    /// Return an iterator yielding fixed-size chunks of ``chunk_frames`` frames.
    ///
    /// Each iteration calls :meth:`read_frames`; the final chunk may be shorter.
    ///
    /// Args:
    ///     chunk_frames (int): Frames per chunk; must be greater than zero. Defaults to 4096.
    ///     dtype (SampleType, optional): Target sample type. Defaults to ``f32``.
    ///
    /// Returns:
    ///     StreamedFrameIterator: An iterator of :class:`AudioSamples` chunks.
    #[pyo3(signature = (chunk_frames: "int" = 4096, dtype: "SampleType" = None), text_signature = "($self, chunk_frames: int = 4096, dtype: SampleType = None) -> StreamedFrameIterator")]
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
            dtype: dtype.unwrap_or(PySampleType::F32),
        })
    }

    /// Return an iterator yielding overlapping windows.
    ///
    /// Each window contains ``window`` frames; the read position advances by ``hop`` frames
    /// between windows (so ``hop < window`` produces overlap). Windows shorter than
    /// ``window`` at the end of the stream are still yielded.
    ///
    /// Args:
    ///     window (int): Frames per window; must be greater than zero.
    ///     hop (int): Frames to advance between windows; must be greater than zero.
    ///     dtype (SampleType, optional): Target sample type. Defaults to ``f32``.
    ///
    /// Returns:
    ///     StreamedFrameIterator: An iterator of :class:`AudioSamples` windows.
    ///
    /// Raises:
    ///     ValueError: If ``window`` or ``hop`` is zero.
    #[pyo3(signature = (window: "int", hop: "int", dtype: "SampleType" = None), text_signature = "($self, window: int, hop: int, dtype: SampleType = None) -> StreamedFrameIterator")]
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
            dtype: dtype.unwrap_or(PySampleType::F32),
        })
    }

    /// Return an iterator yielding single frames (one frame per step).
    ///
    /// Each item is an :class:`AudioSamples` of one frame. This is convenient but slower than
    /// :meth:`frames`; prefer larger chunks for throughput.
    ///
    /// Args:
    ///     dtype (SampleType, optional): Target sample type. Defaults to ``f32``.
    ///
    /// Returns:
    ///     StreamedFrameIterator: An iterator of single-frame :class:`AudioSamples`.
    #[pyo3(signature = (dtype: "SampleType" = None), text_signature = "($self, dtype: SampleType = None) -> StreamedFrameIterator")]
    fn samples(slf: Py<Self>, dtype: Option<PySampleType>) -> PyStreamedFrameIterator {
        PyStreamedFrameIterator {
            reader: slf,
            window: 1,
            hop: 1,
            dtype: dtype.unwrap_or(PySampleType::F32),
        }
    }

    /// Close the reader and release the underlying file handle.
    ///
    /// Further reads raise :class:`RuntimeError`. Idempotent.
    #[pyo3(signature = (), text_signature = "($self) -> None")]
    fn close(&mut self) {
        self.inner = None;
    }

    /// Enter the context manager, returning ``self``.
    fn __enter__(slf: Py<Self>) -> Py<Self> {
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
#[pyclass(name = "StreamedFrameIterator", module = "audio_samples.io", unsendable)]
pub struct PyStreamedFrameIterator {
    reader: Py<PyStreamedAudioReader>,
    window: usize,
    hop: usize,
    dtype: PySampleType,
}

#[pymethods]
impl PyStreamedFrameIterator {
    fn __iter__(slf: Py<Self>) -> Py<Self> {
        slf
    }

    fn __next__(&mut self, py: Python<'_>) -> PyResult<PyAudioSamples> {
        let mut reader = self.reader.borrow_mut(py);
        // Record position so overlapping windows can rewind by (window - hop).
        let start = reader.current_frame()?;
        let chunk = reader.read_frames(py, self.window, Some(self.dtype))?;
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
///     fp (str | Path): Output path; ``.wav`` or ``.flac`` selects the format.
///     channels (int): Number of channels; must be greater than zero.
///     sample_rate (int): Sample rate in Hz; must be greater than zero.
///     dtype (SampleType, optional): Output sample type. Defaults to ``f32``. FLAC supports
///         only integer types (``i16``/``i24``/``i32``).
///
/// Raises:
///     OSError: If the file cannot be created.
///     ValueError: If parameters are invalid or the type/format combination is unsupported.
#[pyclass(name = "StreamedAudioWriter", module = "audio_samples.io", unsendable)]
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
    #[pyo3(signature = (fp: "str | Path", channels: "int", sample_rate: "int", dtype: "SampleType" = None), text_signature = "(fp: str | Path, channels: int, sample_rate: int, dtype: SampleType = None)")]
    fn new(
        fp: &str,
        channels: u16,
        sample_rate: u32,
        dtype: Option<PySampleType>,
    ) -> PyResult<Self> {
        let dtype = dtype.unwrap_or(PySampleType::F32);
        let path = Path::new(fp);
        let file = File::create(path).map_err(|e| audio_io_err_to_py(e.into()))?;
        let writer = BufWriter::with_capacity(256 * 1024, file);
        let inner = match audio_samples_io::types::FileType::from_path(path) {
            audio_samples_io::types::FileType::WAV => {
                WriterInner::Wav(make_wav_writer(writer, channels, sample_rate, dtype)?)
            }
            audio_samples_io::types::FileType::FLAC => {
                let w =
                    create_streamed_flac_for(writer, channels, sample_rate, dtype)?;
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
    ///     fp (str | Path): Output ``.flac`` path.
    ///     channels (int): Number of channels; must be greater than zero.
    ///     sample_rate (int): Sample rate in Hz; must be greater than zero.
    ///     compression (CompressionLevel, optional): FLAC compression level (0-8).
    ///         Defaults to the codec default (5).
    ///     dtype (SampleType, optional): Integer output type (``i16``/``i24``/``i32``).
    ///         Defaults to ``i16``.
    ///
    /// Returns:
    ///     StreamedAudioWriter: A FLAC streaming writer.
    ///
    /// Raises:
    ///     OSError: If the file cannot be created.
    ///     ValueError: If the type is not a FLAC-supported integer type.
    #[staticmethod]
    #[pyo3(signature = (fp: "str | Path", channels: "int", sample_rate: "int", compression: "CompressionLevel" = None, dtype: "SampleType" = None), text_signature = "(fp: str | Path, channels: int, sample_rate: int, compression: CompressionLevel = None, dtype: SampleType = None) -> StreamedAudioWriter")]
    fn create_flac(
        fp: &str,
        channels: u16,
        sample_rate: u32,
        compression: Option<PyCompressionLevel>,
        dtype: Option<PySampleType>,
    ) -> PyResult<Self> {
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
        let level = compression.map_or_else(
            audio_samples_io::CompressionLevel::default,
            |c| c.inner,
        );
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
    ///     samples (AudioSamples): Frames to write.
    ///
    /// Returns:
    ///     int: The number of frames written.
    ///
    /// Raises:
    ///     RuntimeError: If the writer is closed or already finalized.
    ///     ValueError: If the channel count does not match.
    ///     OSError: If the underlying write fails.
    #[pyo3(signature = (samples: "AudioSamples"), text_signature = "($self, samples: AudioSamples) -> int")]
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
    ///     RuntimeError: If the writer is closed.
    ///     OSError: If the flush fails.
    #[pyo3(signature = (), text_signature = "($self) -> None")]
    fn flush(&mut self) -> PyResult<()> {
        writer_dispatch!(self, |w| w.flush().map_err(audio_io_err_to_py))
    }

    /// Finalize the stream, patching format headers with final sizes.
    ///
    /// Must be called exactly once when writing is done (idempotent). After finalizing,
    /// further :meth:`write_frames` calls fail.
    ///
    /// Raises:
    ///     RuntimeError: If the writer is closed.
    ///     OSError: If the underlying seek/flush fails.
    #[pyo3(signature = (), text_signature = "($self) -> None")]
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
    ///     OSError: If finalization fails.
    #[pyo3(signature = (), text_signature = "($self) -> None")]
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
    fn __enter__(slf: Py<Self>) -> Py<Self> {
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
fn flac_validated_type(
    dtype: PySampleType,
) -> PyResult<audio_samples_io::ValidatedSampleType> {
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
///     fp (str | Path): Output ``.wav`` path.
///     channels (int): Number of channels; must be greater than zero.
///     sample_rate (int): Sample rate in Hz; must be greater than zero.
///     dtype (SampleType, optional): Output sample type. Defaults to ``i16``.
///     total_frames (int, optional): Final frame count if known (produces a standard file and
///         verifies the count on :meth:`finalize`); ``None`` for an open-ended stream.
///
/// Raises:
///     OSError: If the file cannot be created.
///     ValueError: If parameters are invalid.
#[pyclass(name = "WavSink", module = "audio_samples.io", unsendable)]
pub struct PyWavSink {
    inner: Option<WavSink<File>>,
}

#[pymethods]
impl PyWavSink {
    #[new]
    #[pyo3(signature = (fp: "str | Path", channels: "int", sample_rate: "int", dtype: "SampleType" = None, total_frames: "int" = None), text_signature = "(fp: str | Path, channels: int, sample_rate: int, dtype: SampleType = None, total_frames: int = None)")]
    fn new(
        fp: &str,
        channels: u16,
        sample_rate: u32,
        dtype: Option<PySampleType>,
        total_frames: Option<usize>,
    ) -> PyResult<Self> {
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
        let sink = create_streamed_sink_typed(file, channels, sample_rate, sample_type, total_frames)?;
        Ok(Self { inner: Some(sink) })
    }

    /// Write a chunk of audio frames to the sink.
    ///
    /// Args:
    ///     samples (AudioSamples): Frames to write; channel count must match.
    ///
    /// Returns:
    ///     int: The number of frames written.
    ///
    /// Raises:
    ///     RuntimeError: If the sink is closed or finalized.
    ///     ValueError: If the channel count mismatches or the declared length is exceeded.
    ///     OSError: If the write fails.
    #[pyo3(signature = (samples: "AudioSamples"), text_signature = "($self, samples: AudioSamples) -> int")]
    fn write_frames(&mut self, py: Python<'_>, samples: &PyAudioSamples) -> PyResult<usize> {
        match &mut self.inner {
            Some(sink) => dispatch_with_view!(samples, py, |audio| {
                sink.write_frames(&audio).map_err(audio_io_err_to_py)
            }),
            None => Err(PyRuntimeError::new_err("sink is closed")),
        }
    }

    /// Flush buffered data to the underlying file.
    #[pyo3(signature = (), text_signature = "($self) -> None")]
    fn flush(&mut self) -> PyResult<()> {
        match &mut self.inner {
            Some(sink) => sink.flush().map_err(audio_io_err_to_py),
            None => Err(PyRuntimeError::new_err("sink is closed")),
        }
    }

    /// Finalize the sink (idempotent). For a declared-length sink this verifies the count.
    #[pyo3(signature = (), text_signature = "($self) -> None")]
    fn finalize(&mut self) -> PyResult<()> {
        match &mut self.inner {
            Some(sink) => sink.finalize().map_err(audio_io_err_to_py),
            None => Err(PyRuntimeError::new_err("sink is closed")),
        }
    }

    /// Number of frames written so far.
    #[getter]
    fn frames_written(&self) -> PyResult<usize> {
        match &self.inner {
            Some(sink) => Ok(sink.frames_written()),
            None => Err(PyRuntimeError::new_err("sink is closed")),
        }
    }

    /// Finalize (if needed) and release the file handle. Idempotent.
    #[pyo3(signature = (), text_signature = "($self) -> None")]
    fn close(&mut self) -> PyResult<()> {
        if let Some(sink) = &mut self.inner {
            let res = sink.finalize();
            self.inner = None;
            res.map_err(audio_io_err_to_py)?;
        }
        Ok(())
    }

    fn __enter__(slf: Py<Self>) -> Py<Self> {
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
