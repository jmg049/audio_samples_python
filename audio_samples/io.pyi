"""
Type stubs for audio_samples.io: audio file I/O functions.

This module provides functions to read and write audio files in various formats.
"""
from typing import final
from typing_extensions import Self

from typing import BinaryIO, Optional, Any, Union
from pathlib import Path
import numpy as np
from . import AudioSamples
from .types import SampleType, ResamplingQuality

@final
class AudioInfo:
    """Information about an audio file."""

    @property
    def sample_rate(self) -> int:
        """Sample rate in Hz."""
        ...

    @property
    def channels(self) -> int:
        """Number of audio channels."""
        ...

    @property
    def bits_per_sample(self) -> int:
        """Bits per sample."""
        ...

    @property
    def num_samples(self) -> int:
        """Total number of samples per channel."""
        ...

    @property
    def duration(self) -> float:
        """Duration in seconds."""
        ...

    @property
    def sample_type(self) -> str:
        """Sample type as a string (e.g., 'i16', 'f32')."""
        ...

def info(fp: str) -> AudioInfo:
    """
    Get information about an audio file without loading the entire file.

    Args:
        fp: Path to the audio file

    Returns:
        AudioInfo: Information about the audio file (sample rate, channels, duration, etc.)

    Raises:
        FileNotFoundError: If the file does not exist (``AudioFileNotFoundError``,
            a subclass of ``AudioError`` and ``FileNotFoundError``).
        TypeError: If the file cannot be decoded or the format is unsupported
            (``AudioDecodeError``, a subclass of ``AudioError`` and ``TypeError``).
    """
    ...

def read(fp: Union[str, Path, BinaryIO], as_type: Optional[SampleType] = None) -> AudioSamples:
    """
    Read an audio file and return AudioSamples.

    Args:
        fp: Path to the audio file, or a file object opened in binary mode with
            ``read``, ``seek`` and ``tell``, read from its current position.
        as_type: Optional ``SampleType`` to convert the audio to, for example
            ``SampleType.F32`` for floats in [-1, 1] (default: the file's native
            sample type, so 16-bit files give raw ``int16`` values).

    Returns:
        AudioSamples: The loaded audio data

    Raises:
        FileNotFoundError: If the file does not exist (``AudioFileNotFoundError``,
            a subclass of ``AudioError`` and ``FileNotFoundError``).
        TypeError: If the file cannot be decoded or the format is unsupported
            (``AudioDecodeError``, a subclass of ``AudioError`` and ``TypeError``).
    """
    ...

def read_with_info(
    fp: str, as_type: Optional[np.dtype[Any]] = None
) -> tuple[AudioSamples, AudioInfo]:
    """
    Read an audio file and return AudioSamples along with file information.

    Args:
        fp: Path to the audio file
        as_type: Optional numpy dtype to convert the audio to (default: native format)

    Returns:
        tuple[AudioSamples, AudioInfo]: The loaded audio data and file information

    Raises:
        FileNotFoundError: If the file does not exist (``AudioFileNotFoundError``,
            a subclass of ``AudioError`` and ``FileNotFoundError``).
        TypeError: If the file cannot be decoded or the format is unsupported
            (``AudioDecodeError``, a subclass of ``AudioError`` and ``TypeError``).
    """
    ...

def save(
    fp: Union[str, Path],
    samples: AudioSamples,
    as_type: Optional[SampleType] = None,
    flac_md5: bool = True,
) -> None:
    """
    Save AudioSamples to an audio file.

    The format is determined by the file extension. By default the audio is
    saved using its native sample type; pass ``as_type`` to convert it to a
    specific sample type before writing.

    WAV keeps every sample type, float64 included. FLAC stores 16-bit and 24-bit
    integers only, so any other sample type raises ``ValueError`` unless ``as_type``
    asks for ``SampleType.I16`` or ``SampleType.I24``.

    Args:
        fp: Path to save the audio file
        samples: The audio data to save
        as_type: Optional sample type to convert to as the file is written, without a copy of
            the signal (default: native format)
        flac_md5: Store the MD5 signature of the audio in a FLAC file, so that decoders can
            verify it. ``False`` leaves it out, which encodes faster.

    Raises:
        ValueError: If the format is unsupported, the sample type cannot be stored in FLAC, or
            ``flac_md5`` is false and the file is not FLAC
        OSError: If the file cannot be written
    """
    ...


# -----------------------------------------------------------------------------
# Read / resample / peek
# -----------------------------------------------------------------------------

def read_and_resample(
    fp: Union[str, Path],
    target_sr: int,
    quality: Optional[ResamplingQuality] = None,
    as_type: Optional[SampleType] = None,
) -> AudioSamples:
    """
    Read an audio file and resample it to a target sample rate in one call.

    Reads the entire file (auto-detecting WAV/FLAC), then resamples to ``target_sr``.
    The resampler is the same as ``AudioSamples.resample``: delay-free, exactly
    ``ceil(n * target_sr / file_sr)`` samples per channel.

    Args:
        fp: Path to the audio file.
        target_sr: Target sample rate in Hz; must be greater than zero.
        quality: Resampling quality/speed trade-off. Defaults to ``ResamplingQuality.high``.
        as_type: Sample type for the returned audio. Defaults to the file's sample type,
            as ``read`` does; the resampler works on every sample type.

    Returns:
        The resampled audio.

    Raises:
        ValueError: If ``target_sr`` is zero.
        TypeError: If the file cannot be read or the format is unsupported.
        RuntimeError: If resampling fails.
    """
    ...

def peek_native_type(fp: Union[str, Path]) -> str:
    """
    Peek at the native sample type of an audio file with minimal I/O.

    Reads only the header (no full decode), making it much cheaper than ``info`` when
    only the sample type is needed.

    Args:
        fp: Path to the audio file.

    Returns:
        The native sample type as a string (e.g. ``"i16"``, ``"f32"``).

    Raises:
        ValueError: If the format is unsupported or the type cannot be determined.
        OSError: If the file cannot be opened.
    """
    ...


# -----------------------------------------------------------------------------
# Write options / format types
# -----------------------------------------------------------------------------

@final
class WriteOptions:
    """Options controlling how audio data is written."""

    def __new__(cls, write_buf_capacity: Optional[int] = None) -> Self:
        """
        Args:
            write_buf_capacity: Size of the internal write buffer in bytes. The default
                of 256 holds only the file's header: samples go straight to the file where
                their memory is already its layout, and otherwise in tiles of 64 KiB. A larger
                buffer copies them first, and helps only when each write call to the
                destination is expensive.
        """
        ...

    @property
    def write_buf_capacity(self) -> int:
        """Size of the internal write buffer in bytes."""
        ...

    @write_buf_capacity.setter
    def write_buf_capacity(self, value: int) -> None: ...

@final
class CompressionLevel:
    """FLAC compression level (0-8): higher means smaller files but slower encoding."""

    def __new__(cls, level: Optional[int] = None) -> Self:
        """
        Args:
            level: Compression level, clamped to the range 0-8. Defaults to 5.
        """
        ...

    @staticmethod
    def fastest() -> "CompressionLevel":
        """Fastest compression (level 0)."""
        ...

    @staticmethod
    def best() -> "CompressionLevel":
        """Best compression (level 8)."""
        ...

    @property
    def level(self) -> int:
        """The numeric compression level (0-8)."""
        ...


# -----------------------------------------------------------------------------
# WAV metadata
# -----------------------------------------------------------------------------

@final
class InfoMetadata:
    """
    LIST/INFO metadata tags for a WAV file (title, artist, etc.).

    All fields are optional strings. Construct one and attach it to a ``WavMetadata``
    to persist tags with ``write_with_metadata``.
    """

    def __init__(self) -> None: ...

    title: Optional[str]
    artist: Optional[str]
    album: Optional[str]
    date: Optional[str]
    comment: Optional[str]
    genre: Optional[str]
    software: Optional[str]
    copyright: Optional[str]
    engineer: Optional[str]
    subject: Optional[str]
    source: Optional[str]
    keywords: Optional[str]

@final
class CuePoint:
    """A WAV cue point (marker). Read-only."""

    @property
    def id(self) -> int:
        """Cue point identifier."""
        ...

    @property
    def position(self) -> int:
        """Play-order position."""
        ...

    @property
    def sample_offset(self) -> int:
        """Sample offset within the data chunk."""
        ...

@final
class WavMetadata:
    """
    Round-trippable WAV metadata: INFO tags plus cue points.

    Pass to ``write_with_metadata`` to persist tags that a plain read->write would drop.
    """


    def __new__(cls, info: Optional[InfoMetadata] = None) -> Self: ...

    info: Optional[InfoMetadata]

    @property
    def cue_points(self) -> list[CuePoint]:
        """The cue points (markers)."""
        ...

    @property
    def is_empty(self) -> bool:
        """True if there are no tags and no cue points."""
        ...


# -----------------------------------------------------------------------------
# FLAC metadata
# -----------------------------------------------------------------------------

@final
class VorbisComment:
    """
    A FLAC Vorbis comment block: a vendor string plus key->values tags.

    Construct one, populate it with ``set``/``add``, and serialise with ``to_bytes``
    (or parse with ``from_bytes``). Reading the Vorbis comment of a FLAC file is not
    supported, so this class is for building and (de)serialising comment blocks.
    """

    def __init__(self) -> None: ...

    @staticmethod
    def from_bytes(data: bytes) -> "VorbisComment":
        """
        Parse a Vorbis comment block from its raw bytes.

        Raises:
            ValueError: If the bytes are malformed.
        """
        ...

    vendor: str

    def get(self, key: str) -> Optional[str]:
        """Get the first value for ``key``, or ``None``."""
        ...

    def get_all(self, key: str) -> list[str]:
        """Get all values for ``key`` (empty list if absent)."""
        ...

    def set(self, key: str, value: str) -> None:
        """Set ``key`` to a single ``value``, replacing any existing values."""
        ...

    def add(self, key: str, value: str) -> None:
        """Append ``value`` to the values for ``key``."""
        ...

    def to_bytes(self) -> bytes:
        """Serialise to a raw Vorbis comment block."""
        ...

@final
class StreamInfo:
    """FLAC STREAMINFO metadata block (read-only)."""

    @property
    def min_block_size(self) -> int: ...
    @property
    def max_block_size(self) -> int: ...
    @property
    def sample_rate(self) -> int: ...
    @property
    def channels(self) -> int: ...
    @property
    def bits_per_sample(self) -> int: ...
    @property
    def total_samples(self) -> int:
        """Total inter-channel samples (frames)."""
        ...
    @property
    def has_md5(self) -> bool:
        """Whether the MD5 signature of the decoded audio is present."""
        ...


# -----------------------------------------------------------------------------
# Write functions
# -----------------------------------------------------------------------------

def set_thread_limit(limit: Optional[int]) -> None:
    """
    Limit how many threads a file read or decode may use.

    Reads of large files and FLAC decodes run on a few short-lived threads, which end
    before the call returns. By default their number follows the CPUs available to the
    process, with a cap, or ``AUDIO_SAMPLES_IO_THREADS`` if that is set when the first
    read starts. This setting overrides both.

    Args:
        limit: Most threads to use; ``1`` keeps everything on the calling thread.
            ``None`` returns to the default.

    Raises:
        ValueError: If ``limit`` is zero.
    """
    ...


def thread_limit() -> int:
    """The most threads a read of a large file will use at present."""
    ...


def write_with_options(
    fp: Union[str, Path],
    samples: AudioSamples,
    options: Optional[WriteOptions] = None,
    as_type: Optional[SampleType] = None,
    flac_md5: bool = True,
) -> None:
    """
    Write audio samples to a file with explicit write options.

    Like ``save`` but lets you control the write-buffer size via ``WriteOptions``.
    The format is chosen from the file extension (``.wav``/``.flac``/``.aiff``).

    Args:
        fp: Output path.
        samples: The audio to write.
        options: Write options. Defaults to a 256-byte buffer, which holds the header.
        as_type: Convert to this sample type as the file is written, without a copy of the
            signal. Defaults to the native type.
        flac_md5: Store the MD5 signature of the audio in a FLAC file, so that decoders can
            verify it. ``False`` leaves it out, which encodes faster.

    Raises:
        ValueError: If the format is unsupported.
        OSError: If the file cannot be written.
    """
    ...

def write_with_metadata(
    fp: Union[str, Path],
    samples: AudioSamples,
    metadata: WavMetadata,
) -> None:
    """
    Write a WAV file with trailing metadata chunks (LIST/INFO tags, cue points).

    Like ``save``, but also serialises the given ``WavMetadata`` after the audio data,
    persisting tags/markers that a plain read->write round-trip would drop. WAV only.

    Raises:
        OSError: If the file cannot be written.
    """
    ...


# -----------------------------------------------------------------------------
# Metadata readers
# -----------------------------------------------------------------------------

def read_wav_info_tags(fp: Union[str, Path]) -> Optional[InfoMetadata]:
    """
    Read the LIST/INFO tags from a WAV file.

    Returns:
        The INFO tags, or ``None`` if the file has no LIST/INFO chunk.

    Raises:
        OSError: If the file cannot be opened.
        ValueError: If the file is not a WAV file or the chunk is malformed.
    """
    ...

def read_wav_cue_points(fp: Union[str, Path]) -> list[CuePoint]:
    """
    Read the cue points (markers) from a WAV file.

    Returns:
        The cue points (empty list if none).

    Raises:
        OSError: If the file cannot be opened.
        ValueError: If the file is not a WAV file or the chunk is malformed.
    """
    ...

def read_flac_stream_info(fp: Union[str, Path]) -> StreamInfo:
    """
    Read the STREAMINFO metadata block from a FLAC file.

    Raises:
        OSError: If the file cannot be opened.
        ValueError: If the file is not a FLAC file.
    """
    ...


# -----------------------------------------------------------------------------
# Streaming subsystem
# -----------------------------------------------------------------------------

@final
class StreamedAudioReader:
    """
    A streaming audio file reader.

    Opens a WAV or FLAC file and decodes frames on demand instead of loading the whole
    file into memory. Use it as a context manager and pull frames with ``read_frames``,
    or iterate with ``frames``, ``windows``, or ``samples``.
    """

    def __new__(cls, fp: Union[str, Path, BinaryIO]) -> Self:
        """
        Open a WAV or FLAC file for streaming reads.

        Args:
            fp: Path to the audio file. The format is detected from the contents.

        Raises:
            OSError: If the file cannot be opened.
            ValueError: If the format is unsupported or the header is corrupt.
        """
        ...

    @property
    def sample_rate(self) -> int:
        """Sample rate in Hz."""
        ...

    @property
    def num_channels(self) -> int:
        """Number of audio channels."""
        ...

    @property
    def total_frames(self) -> int:
        """Total number of frames in the stream."""
        ...

    @property
    def current_frame(self) -> int:
        """Current frame position (0-indexed)."""
        ...

    @property
    def remaining_frames(self) -> int:
        """Number of frames remaining from the current position."""
        ...

    def read_frames(
        self, frame_count: int, dtype: Optional[SampleType] = None
    ) -> Optional[AudioSamples]:
        """
        Read up to ``frame_count`` frames and return them as an ``AudioSamples``.

        Frames are decoded and converted to the requested ``dtype`` on the fly. Returns
        ``None`` once the end of the stream is reached.

        Args:
            frame_count: Maximum number of frames to read; must be greater than zero.
            dtype: Target sample type for the returned array. Defaults to the file's
                own sample type.

        Returns:
            The decoded chunk (mono shape ``(frames,)`` or multi-channel
            ``(channels, frames)``), or ``None`` at end of stream.

        Raises:
            ValueError: If ``frame_count`` is zero.
            RuntimeError: If the reader is closed.
            OSError: If reading fails or the data is corrupt.
        """
        ...

    def seek_to_frame(self, frame: int) -> None:
        """
        Seek so the next read returns frame ``frame``.

        Raises:
            OSError: If the frame is beyond the end of the stream or the seek fails.
            RuntimeError: If the reader is closed.
        """
        ...

    def reset(self) -> None:
        """
        Reset the stream to the beginning of the audio data.

        Raises:
            OSError: If the underlying seek fails.
            RuntimeError: If the reader is closed.
        """
        ...

    def frames(
        self, chunk_frames: int = 4096, dtype: Optional[SampleType] = None
    ) -> "StreamedFrameIterator":
        """
        Return an iterator yielding fixed-size chunks of ``chunk_frames`` frames.

        The final chunk may be shorter.

        Args:
            chunk_frames: Frames per chunk; must be greater than zero. Defaults to 4096.
            dtype: Target sample type. Defaults to the file's own sample type.
        """
        ...

    def windows(
        self, window: int, hop: int, dtype: Optional[SampleType] = None
    ) -> "StreamedFrameIterator":
        """
        Return an iterator yielding overlapping windows.

        Each window contains ``window`` frames; the read position advances by ``hop``
        frames between windows (so ``hop < window`` produces overlap).

        Args:
            window: Frames per window; must be greater than zero.
            hop: Frames to advance between windows; must be greater than zero.
            dtype: Target sample type. Defaults to the file's own sample type.

        Raises:
            ValueError: If ``window`` or ``hop`` is zero.
        """
        ...

    def samples(self, dtype: Optional[SampleType] = None) -> "StreamedFrameIterator":
        """
        Return an iterator yielding single frames (one frame per step).

        Args:
            dtype: Target sample type. Defaults to the file's own sample type.
        """
        ...

    def close(self) -> None:
        """Close the reader and release the underlying file handle. Idempotent."""
        ...

    def __enter__(self) -> "StreamedAudioReader": ...
    def __exit__(
        self,
        _exc_type: object  =  None,
        _exc_value: object  =  None,
        _traceback: object  =  None,
    ) -> bool: ...

@final
class StreamedFrameIterator:
    """
    Iterator over chunks / windows / single frames of a ``StreamedAudioReader``.

    Created by ``StreamedAudioReader.frames``, ``StreamedAudioReader.windows``, and
    ``StreamedAudioReader.samples``. Yields ``AudioSamples``.
    """

    def __iter__(self) -> "StreamedFrameIterator": ...
    def __next__(self) -> AudioSamples: ...

@final
class StreamedAudioWriter:
    """
    A streaming audio file writer.

    Encodes ``AudioSamples`` chunks incrementally to a WAV or FLAC file (chosen from the
    extension) without buffering the whole signal. The output sample type is fixed at
    creation from ``dtype``. Always ``finalize`` (or use the ``with`` block) so format
    headers are patched with the final sizes.
    """

    def __new__(
        cls,
        fp: Union[str, Path],
        channels: int,
        sample_rate: int,
        dtype: Optional[SampleType] = None,
    ) -> Self:
        """
        Create a streaming writer to ``fp`` (format chosen from the extension).

        Args:
            fp: Output path; ``.wav`` or ``.flac`` selects the format.
            channels: Number of channels; must be greater than zero.
            sample_rate: Sample rate in Hz; must be greater than zero.
            dtype: Output sample type. Defaults to ``f32``. FLAC supports only integer
                types (``i16``/``i24``/``i32``).

        Raises:
            OSError: If the file cannot be created.
            ValueError: If parameters are invalid or the type/format combination is
                unsupported.
        """
        ...

    @staticmethod
    def create_flac(
        fp: Union[str, Path],
        channels: int,
        sample_rate: int,
        compression: Optional[CompressionLevel] = None,
        dtype: Optional[SampleType] = None,
    ) -> "StreamedAudioWriter":
        """
        Create a streaming FLAC writer with an explicit compression level.

        Args:
            fp: Output ``.flac`` path.
            channels: Number of channels; must be greater than zero.
            sample_rate: Sample rate in Hz; must be greater than zero.
            compression: FLAC compression level (0-8). Defaults to the codec default (5).
            dtype: Integer output type (``i16``/``i24``/``i32``). Defaults to ``i16``.

        Raises:
            OSError: If the file cannot be created.
            ValueError: If the type is not a FLAC-supported integer type.
        """
        ...

    def write_frames(self, samples: AudioSamples) -> int:
        """
        Write a chunk of audio frames to the stream.

        Samples are converted from the array's dtype to the writer's configured output
        type. The channel count must match the writer's configuration.

        Returns:
            The number of frames written.

        Raises:
            RuntimeError: If the writer is closed or already finalized.
            ValueError: If the channel count does not match.
            OSError: If the underlying write fails.
        """
        ...

    def flush(self) -> None:
        """Flush buffered data to the underlying file without finalizing."""
        ...

    def finalize(self) -> None:
        """
        Finalize the stream, patching format headers with final sizes.

        Must be called exactly once when writing is done (idempotent).
        """
        ...

    @property
    def is_finalized(self) -> bool:
        """Whether the stream has been finalized."""
        ...

    @property
    def frames_written(self) -> int:
        """Number of frames written so far."""
        ...

    @property
    def sample_rate(self) -> int:
        """Sample rate the writer was configured with, in Hz."""
        ...

    @property
    def num_channels(self) -> int:
        """Number of channels the writer was configured with."""
        ...

    def close(self) -> None:
        """Finalize (if not already) and release the file handle. Idempotent."""
        ...

    def __enter__(self) -> "StreamedAudioWriter": ...
    def __exit__(
        self,
        _exc_type: object  =  None,
        _exc_value: object  =  None,
        _traceback: object  =  None,
    ) -> bool: ...

@final
class WavSink:
    """
    A non-seekable streaming WAV writer ("sink").

    Writes a WAV stream where seeking back to patch the header is undesirable. Because
    the header cannot be back-patched, the final frame count is either declared up front
    (``total_frames``) or left open-ended (streaming-size convention).
    """

    def __new__(
        cls,
        fp: Union[str, Path],
        channels: int,
        sample_rate: int,
        dtype: Optional[SampleType] = None,
        total_frames: Optional[int] = None,
    ) -> Self:
        """
        Args:
            fp: Output ``.wav`` path.
            channels: Number of channels; must be greater than zero.
            sample_rate: Sample rate in Hz; must be greater than zero.
            dtype: Output sample type. Defaults to ``i16``.
            total_frames: Final frame count if known (produces a standard file and
                verifies the count on ``finalize``); ``None`` for an open-ended stream.

        Raises:
            OSError: If the file cannot be created.
            ValueError: If parameters are invalid.
        """
        ...

    def write_frames(self, samples: AudioSamples) -> int:
        """
        Write a chunk of audio frames to the sink.

        Returns:
            The number of frames written.

        Raises:
            RuntimeError: If the sink is closed or finalized.
            ValueError: If the channel count mismatches or the declared length is exceeded.
            OSError: If the write fails.
        """
        ...

    def flush(self) -> None:
        """Flush buffered data to the underlying file."""
        ...

    def finalize(self) -> None:
        """Finalize the sink (idempotent). For a declared-length sink this verifies the count."""
        ...

    @property
    def frames_written(self) -> int:
        """Number of frames written so far."""
        ...

    def close(self) -> None:
        """Finalize (if needed) and release the file handle. Idempotent."""
        ...

    def __enter__(self) -> "WavSink": ...
    def __exit__(
        self,
        _exc_type: object  =  None,
        _exc_value: object  =  None,
        _traceback: object  =  None,
    ) -> bool: ...

__all__ = [
    "AudioInfo",
    "CompressionLevel",
    "CuePoint",
    "InfoMetadata",
    "StreamInfo",
    "StreamedAudioReader",
    "StreamedAudioWriter",
    "StreamedFrameIterator",
    "VorbisComment",
    "WavMetadata",
    "WavSink",
    "WriteOptions",
    "info",
    "peek_native_type",
    "read",
    "read_and_resample",
    "read_flac_stream_info",
    "read_wav_cue_points",
    "read_wav_info_tags",
    "read_with_info",
    "save",
    "set_thread_limit",
    "thread_limit",
    "write_with_metadata",
    "write_with_options",
]
