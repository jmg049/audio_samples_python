"""
Reading, writing and streaming audio files (WAV and FLAC).
"""

from . import AudioSamples
from typing import Any, final

__all__ = ["AudioInfo", "CompressionLevel", "CuePoint", "InfoMetadata", "StreamInfo", "StreamedAudioReader", "StreamedAudioWriter", "StreamedFrameIterator", "VorbisComment", "WavMetadata", "WavSink", "WriteOptions", "info", "peek_native_type", "read", "read_and_resample", "read_flac_stream_info", "read_wav_cue_points", "read_wav_info_tags", "read_with_info", "save", "set_thread_limit", "thread_limit", "write_with_metadata", "write_with_options"]

@final
class AudioInfo:
    """
    Audio file information structure
    """
    @property
    def bits_per_sample(self, /) -> int:
        """
        Bits per sample
        """
    @property
    def channels(self, /) -> int:
        """
        Number of audio channels
        """
    @property
    def duration(self, /) -> float:
        """
        Duration in seconds
        """
    @property
    def num_samples(self, /) -> int:
        """
        Total number of samples per channel
        """
    @property
    def sample_rate(self, /) -> int:
        """
        Sample rate in Hz
        """
    @property
    def sample_type(self, /) -> str:
        """
        Sample type as a string (e.g., "i16", "f32")
        """

@final
class CompressionLevel:
    """
    FLAC compression level (0-8): higher means smaller files but slower encoding.

    Args:

        level (int): Compression level, clamped to the range 0-8. Defaults to 5.
    """
    def __new__(cls, /, level: "int | None" = None) -> CompressionLevel: ...
    def __repr__(self, /) -> str: ...
    @staticmethod
    def best() -> CompressionLevel:
        """
        Best compression (level 8).
        """
    @staticmethod
    def fastest() -> CompressionLevel:
        """
        Fastest compression (level 0).
        """
    @property
    def level(self, /) -> int:
        """
        The numeric compression level (0-8).
        """

@final
class CuePoint:
    """
    A WAV cue point (marker).
    """
    def __repr__(self, /) -> str: ...
    @property
    def id(self, /) -> int:
        """
        Cue point identifier.
        """
    @property
    def position(self, /) -> int:
        """
        Play-order position.
        """
    @property
    def sample_offset(self, /) -> int:
        """
        Sample offset within the data chunk.
        """

@final
class InfoMetadata:
    """
    LIST/INFO metadata tags for a WAV file (title, artist, etc.).

    All fields are optional strings. Construct one and attach it to a :class:`WavMetadata` to
    persist tags with :func:`write_with_metadata`.
    """
    def __new__(cls, /) -> InfoMetadata: ...
    def __repr__(self, /) -> str: ...
    @property
    def album(self, /) -> str | None:
        """
        Album name.
        """
    @album.setter
    def album(self, /, value: str | None) -> None: ...
    @property
    def artist(self, /) -> str | None:
        """
        Performing artist.
        """
    @artist.setter
    def artist(self, /, value: str | None) -> None: ...
    @property
    def comment(self, /) -> str | None:
        """
        Free-form comment.
        """
    @comment.setter
    def comment(self, /, value: str | None) -> None: ...
    @property
    def copyright(self, /) -> str | None:
        """
        Copyright notice.
        """
    @copyright.setter
    def copyright(self, /, value: str | None) -> None: ...
    @property
    def date(self, /) -> str | None:
        """
        Creation/release date.
        """
    @date.setter
    def date(self, /, value: str | None) -> None: ...
    @property
    def engineer(self, /) -> str | None:
        """
        Engineer name.
        """
    @engineer.setter
    def engineer(self, /, value: str | None) -> None: ...
    @property
    def genre(self, /) -> str | None:
        """
        Genre.
        """
    @genre.setter
    def genre(self, /, value: str | None) -> None: ...
    @property
    def keywords(self, /) -> str | None:
        """
        Keywords.
        """
    @keywords.setter
    def keywords(self, /, value: str | None) -> None: ...
    @property
    def software(self, /) -> str | None:
        """
        Authoring software.
        """
    @software.setter
    def software(self, /, value: str | None) -> None: ...
    @property
    def source(self, /) -> str | None:
        """
        Source of the material.
        """
    @source.setter
    def source(self, /, value: str | None) -> None: ...
    @property
    def subject(self, /) -> str | None:
        """
        Subject/description.
        """
    @subject.setter
    def subject(self, /, value: str | None) -> None: ...
    @property
    def title(self, /) -> str | None:
        """
        Track title.
        """
    @title.setter
    def title(self, /, value: str | None) -> None: ...

@final
class StreamInfo:
    """
    FLAC STREAMINFO metadata block (read-only).
    """
    def __repr__(self, /) -> str: ...
    @property
    def bits_per_sample(self, /) -> int:
        """
        Bits per sample.
        """
    @property
    def channels(self, /) -> int:
        """
        Number of channels.
        """
    @property
    def has_md5(self, /) -> bool:
        """
        Whether the MD5 signature of the decoded audio is present.
        """
    @property
    def max_block_size(self, /) -> int:
        """
        Maximum block size (samples).
        """
    @property
    def min_block_size(self, /) -> int:
        """
        Minimum block size (samples).
        """
    @property
    def sample_rate(self, /) -> int:
        """
        Sample rate in Hz.
        """
    @property
    def total_samples(self, /) -> int:
        """
        Total inter-channel samples (frames).
        """

@final
class StreamedAudioReader:
    """
    A streaming audio file reader.

    Opens a WAV or FLAC file and decodes frames on demand instead of loading the whole file
    into memory. Use it as a context manager and pull frames with :meth:`read_frames`, or
    iterate with :meth:`frames`, :meth:`windows`, or :meth:`samples`.

    Args:

        fp (str | Path | BinaryIO): Path to the audio file, or a binary file object with
            ``read``, ``seek`` and ``tell``. The format is detected from the contents.

    Raises:

        OSError: If the file cannot be opened.
        ValueError: If the format is unsupported or the header is corrupt.
    """
    def __enter__(self, /) -> StreamedAudioReader:
        """
        Enter the context manager, returning ``self``.
        """
    def __exit__(self, /, _exc_type: Any | None = None, _exc_value: Any | None = None, _traceback: Any | None = None) -> bool:
        """
        Exit the context manager, closing the reader.
        """
    def __new__(cls, /, fp: "str | Path | BinaryIO") -> StreamedAudioReader:
        """
        Open a WAV or FLAC file for streaming reads.

        Args:

            fp (str | Path | BinaryIO): Path to the audio file, or a file object opened in
                binary mode with ``read``, ``seek`` and ``tell``. A file object is read from
                its current position and stays open; decoding calls its methods with the
                GIL held.

        Raises:

            OSError: If the file cannot be opened.
            ValueError: If the format is unsupported or the header is corrupt.
            TypeError: If ``fp`` is neither a path nor such a file object.
        """
    def close(self, /) -> None:
        """
        Close the reader and release the underlying file handle.

        Further reads raise :class:`RuntimeError`. Idempotent.
        """
    @property
    def current_frame(self, /) -> int:
        """
        Current frame position (0-indexed).
        """
    def frames(self, /, chunk_frames: "int" = 4096, dtype: "SampleType | None" = None) -> StreamedFrameIterator:
        """
        Return an iterator yielding fixed-size chunks of ``chunk_frames`` frames.

        Each iteration calls :meth:`read_frames`; the final chunk may be shorter.

        Args:

            chunk_frames (int): Frames per chunk; must be greater than zero. Defaults to 4096.
            dtype (SampleType, optional): Target sample type. Defaults to the file's own
                sample type.

        Returns:

            StreamedFrameIterator: An iterator of :class:`AudioSamples` chunks.
        """
    @property
    def num_channels(self, /) -> int:
        """
        Number of audio channels.
        """
    def read_frames(self, /, frame_count: "int", dtype: "SampleType | None" = None) -> AudioSamples | None:
        """
        Read up to ``frame_count`` frames and return them as an :class:`AudioSamples`.

        Frames are decoded and converted to the requested ``dtype`` on the fly. Returns
        ``None`` once the end of the stream is reached (no frames left to read).

        Args:

            frame_count (int): Maximum number of frames to read; must be greater than zero.
            dtype (SampleType, optional): Target sample type for the returned array.
                Defaults to the file's own sample type.

        Returns:

            AudioSamples | None: The decoded chunk (mono shape ``(frames,)`` or multi-channel
            shape ``(channels, frames)``), or ``None`` at end of stream.

        Raises:

            ValueError: If ``frame_count`` is zero.
            RuntimeError: If the reader is closed.
            OSError: If reading fails or the data is corrupt.
        """
    @property
    def remaining_frames(self, /) -> int:
        """
        Number of frames remaining from the current position.
        """
    def reset(self, /) -> None:
        """
        Reset the stream to the beginning of the audio data.

        Raises:

            OSError: If the underlying seek fails.
            RuntimeError: If the reader is closed.
        """
    @property
    def sample_rate(self, /) -> int:
        """
        Sample rate in Hz.
        """
    def samples(self, /, dtype: "SampleType | None" = None) -> StreamedFrameIterator:
        """
        Return an iterator yielding single frames (one frame per step).

        Each item is an :class:`AudioSamples` of one frame. This is convenient but slower than
        :meth:`frames`; prefer larger chunks for throughput.

        Args:

            dtype (SampleType, optional): Target sample type. Defaults to the file's own
                sample type.

        Returns:

            StreamedFrameIterator: An iterator of single-frame :class:`AudioSamples`.
        """
    def seek_to_frame(self, /, frame: "int") -> None:
        """
        Seek so the next read returns frame ``frame``.

        Args:

            frame (int): Target frame index (0-indexed).

        Raises:

            OSError: If the frame is beyond the end of the stream or the seek fails.
            RuntimeError: If the reader is closed.
        """
    @property
    def total_frames(self, /) -> int:
        """
        Total number of frames in the stream.
        """
    def windows(self, /, window: "int", hop: "int", dtype: "SampleType | None" = None) -> StreamedFrameIterator:
        """
        Return an iterator yielding overlapping windows.

        Each window contains ``window`` frames; the read position advances by ``hop`` frames
        between windows (so ``hop < window`` produces overlap). Windows shorter than
        ``window`` at the end of the stream are still yielded.

        Args:

            window (int): Frames per window; must be greater than zero.
            hop (int): Frames to advance between windows; must be greater than zero.
            dtype (SampleType, optional): Target sample type. Defaults to the file's own
                sample type.

        Returns:

            StreamedFrameIterator: An iterator of :class:`AudioSamples` windows.

        Raises:

            ValueError: If ``window`` or ``hop`` is zero.
        """

@final
class StreamedAudioWriter:
    """
    A streaming audio file writer.

    Encodes :class:`AudioSamples` chunks incrementally to a WAV or FLAC file (chosen from the
    extension) without buffering the whole signal. The output sample type is fixed at creation
    from ``dtype``. Always :meth:`finalize` (or use the ``with`` block) so format headers are
    patched with the final sizes.

    Args:

        fp (str | Path): Output path; ``.wav`` or ``.flac`` selects the format.
        channels (int): Number of channels; must be greater than zero.
        sample_rate (int): Sample rate in Hz; must be greater than zero.
        dtype (SampleType, optional): Output sample type. Defaults to ``f32``. FLAC supports
            only integer types (``i16``/``i24``/``i32``).

    Raises:

        OSError: If the file cannot be created.
        ValueError: If parameters are invalid or the type/format combination is unsupported.
    """
    def __enter__(self, /) -> StreamedAudioWriter:
        """
        Enter the context manager, returning ``self``.
        """
    def __exit__(self, /, _exc_type: Any | None = None, _exc_value: Any | None = None, _traceback: Any | None = None) -> bool:
        """
        Exit the context manager, finalizing and closing the writer.
        """
    def __new__(cls, /, fp: "str | Path", channels: "int", sample_rate: "int", dtype: "SampleType | None" = None) -> StreamedAudioWriter:
        """
        Create a streaming writer to ``fp`` (format chosen from the extension).
        """
    def close(self, /) -> None:
        """
        Finalize (if not already) and release the file handle. Idempotent.

        Raises:

            OSError: If finalization fails.
        """
    @staticmethod
    def create_flac(fp: "str | Path", channels: "int", sample_rate: "int", compression: "CompressionLevel | None" = None, dtype: "SampleType | None" = None) -> StreamedAudioWriter:
        """
        Create a streaming FLAC writer with an explicit compression level.

        Args:

            fp (str | Path): Output ``.flac`` path.
            channels (int): Number of channels; must be greater than zero.
            sample_rate (int): Sample rate in Hz; must be greater than zero.
            compression (CompressionLevel, optional): FLAC compression level (0-8).
                Defaults to the codec default (5).
            dtype (SampleType, optional): Integer output type (``i16``/``i24``/``i32``).
                Defaults to ``i16``.

        Returns:

            StreamedAudioWriter: A FLAC streaming writer.

        Raises:

            OSError: If the file cannot be created.
            ValueError: If the type is not a FLAC-supported integer type.
        """
    def finalize(self, /) -> None:
        """
        Finalize the stream, patching format headers with final sizes.

        Must be called exactly once when writing is done (idempotent). After finalizing,
        further :meth:`write_frames` calls fail.

        Raises:

            RuntimeError: If the writer is closed.
            OSError: If the underlying seek/flush fails.
        """
    def flush(self, /) -> None:
        """
        Flush buffered data to the underlying file without finalizing.

        Raises:

            RuntimeError: If the writer is closed.
            OSError: If the flush fails.
        """
    @property
    def frames_written(self, /) -> int:
        """
        Number of frames written so far.
        """
    @property
    def is_finalized(self, /) -> bool:
        """
        Whether the stream has been finalized.
        """
    @property
    def num_channels(self, /) -> int:
        """
        Number of channels the writer was configured with.
        """
    @property
    def sample_rate(self, /) -> int:
        """
        Sample rate the writer was configured with, in Hz.
        """
    def write_frames(self, /, samples: "AudioSamples") -> int:
        """
        Write a chunk of audio frames to the stream.

        Samples are converted from the array's dtype to the writer's configured output type.
        The channel count must match the writer's configuration.

        Args:

            samples (AudioSamples): Frames to write.

        Returns:

            int: The number of frames written.

        Raises:

            RuntimeError: If the writer is closed or already finalized.
            ValueError: If the channel count does not match.
            OSError: If the underlying write fails.
        """

@final
class StreamedFrameIterator:
    """
    Iterator over chunks / windows / single frames of a :class:`StreamedAudioReader`.

    Created by :meth:`StreamedAudioReader.frames`, :meth:`StreamedAudioReader.windows`, and
    :meth:`StreamedAudioReader.samples`. Yields :class:`AudioSamples`; for ``hop != window``
    the iterator seeks the underlying reader between steps.
    """
    def __iter__(self, /) -> StreamedFrameIterator: ...
    def __next__(self, /) -> AudioSamples: ...

@final
class VorbisComment:
    """
    A FLAC Vorbis comment block: a vendor string plus key→values tags.

    Construct one, populate it with :meth:`set`/:meth:`add`, and serialise with :meth:`to_bytes`
    (or parse with :meth:`from_bytes`). The comment block of a decoded FLAC file cannot be read
    back through this class; it constructs and (de)serialises comment blocks.
    """
    def __new__(cls, /) -> VorbisComment: ...
    def add(self, /, key: "str", value: "str") -> None:
        """
        Append ``value`` to the values for ``key``.
        """
    @staticmethod
    def from_bytes(data: "bytes") -> VorbisComment:
        """
        Parse a Vorbis comment block from its raw bytes.

        Args:

            data (bytes): The raw Vorbis comment block.

        Returns:

            VorbisComment: The parsed comment.

        Raises:

            ValueError: If the bytes are malformed.
        """
    def get(self, /, key: "str") -> str | None:
        """
        Get the first value for ``key``, or ``None``.
        """
    def get_all(self, /, key: "str") -> list[str]:
        """
        Get all values for ``key`` (empty list if absent).
        """
    def set(self, /, key: "str", value: "str") -> None:
        """
        Set ``key`` to a single ``value``, replacing any existing values.
        """
    def to_bytes(self, /) -> bytes:
        """
        Serialise to a raw Vorbis comment block.
        """
    @property
    def vendor(self, /) -> str:
        """
        The vendor string.
        """
    @vendor.setter
    def vendor(self, /, value: str) -> None: ...

@final
class WavMetadata:
    """
    Round-trippable WAV metadata: INFO tags plus cue points.

    Pass to :func:`write_with_metadata` to persist tags that a plain read→write would drop.
    """
    def __new__(cls, /, info: "InfoMetadata | None" = None) -> WavMetadata: ...
    @property
    def cue_points(self, /) -> list[CuePoint]:
        """
        The cue points (markers).
        """
    @property
    def info(self, /) -> InfoMetadata | None:
        """
        The INFO tags, or ``None`` if unset.
        """
    @info.setter
    def info(self, /, info: InfoMetadata | None) -> None: ...
    @property
    def is_empty(self, /) -> bool:
        """
        True if there are no tags and no cue points.
        """

@final
class WavSink:
    """
    A non-seekable streaming WAV writer ("sink").

    Writes a WAV stream to a file opened in append/write mode where seeking back to patch the
    header is undesirable. Because the header cannot be back-patched, the final frame count is
    either declared up front (``total_frames``) or left open-ended (streaming-size convention).

    Args:

        fp (str | Path): Output ``.wav`` path.
        channels (int): Number of channels; must be greater than zero.
        sample_rate (int): Sample rate in Hz; must be greater than zero.
        dtype (SampleType, optional): Output sample type. Defaults to ``i16``.
        total_frames (int, optional): Final frame count if known (produces a standard file and
            verifies the count on :meth:`finalize`); ``None`` for an open-ended stream.

    Raises:

        OSError: If the file cannot be created.
        ValueError: If parameters are invalid.
    """
    def __enter__(self, /) -> WavSink: ...
    def __exit__(self, /, _exc_type: Any | None = None, _exc_value: Any | None = None, _traceback: Any | None = None) -> bool: ...
    def __new__(cls, /, fp: "str | Path", channels: "int", sample_rate: "int", dtype: "SampleType | None" = None, total_frames: "int | None" = None) -> WavSink: ...
    def close(self, /) -> None:
        """
        Finalize (if needed) and release the file handle. Idempotent.
        """
    def finalize(self, /) -> None:
        """
        Finalize the sink (idempotent). For a declared-length sink this verifies the count.
        """
    def flush(self, /) -> None:
        """
        Flush buffered data to the underlying file.
        """
    @property
    def frames_written(self, /) -> int:
        """
        Number of frames written so far.
        """
    def write_frames(self, /, samples: "AudioSamples") -> int:
        """
        Write a chunk of audio frames to the sink.

        Args:

            samples (AudioSamples): Frames to write; channel count must match.

        Returns:

            int: The number of frames written.

        Raises:

            RuntimeError: If the sink is closed or finalized.
            ValueError: If the channel count mismatches or the declared length is exceeded.
            OSError: If the write fails.
        """

@final
class WriteOptions:
    """
    Options controlling how audio data is written.

    Args:

        write_buf_capacity (int): Size of the internal write buffer in bytes. The default of
            256 holds only the file's header: samples go straight to the file where their memory
            is already its layout, and otherwise in tiles of 64 KiB. A larger buffer copies them
            first, and helps only when each write call to the destination is expensive.
    """
    def __new__(cls, /, write_buf_capacity: "int | None" = None) -> WriteOptions: ...
    @property
    def write_buf_capacity(self, /) -> int:
        """
        Size of the internal write buffer in bytes.
        """
    @write_buf_capacity.setter
    def write_buf_capacity(self, /, value: int) -> None: ...

def info(fp: "str|Path") -> AudioInfo: ...

def peek_native_type(fp: "str | Path") -> str:
    """
    Peek at the native sample type of an audio file with minimal I/O.

    Reads only the header (no full decode), making it much cheaper than :func:`info` when only
    the sample type is needed.

    Args:

        fp (str | Path): Path to the audio file.

    Returns:

        str: The native sample type as a string (e.g. ``"i16"``, ``"f32"``).

    Raises:

        ValueError: If the format is unsupported or the type cannot be determined.
        OSError: If the file cannot be opened.
    """

def read(fp: "str|Path|BinaryIO", as_type: "SampleType | None" = None) -> AudioSamples:
    """
    Read an audio file (WAV or FLAC) and return an ``AudioSamples``.

    Args:

        fp (str | Path | BinaryIO): Path to the audio file, or a file object opened in
            binary mode with ``read``, ``seek`` and ``tell``, read from its current position.
        as_type (SampleType | None): Sample type to convert to, for example
            ``SampleType.F32`` for floats in [-1, 1]. Defaults to the file's
            native sample type (16-bit files give raw ``int16`` values).

    Returns:

        AudioSamples: The decoded audio.

    Raises:

        FileNotFoundError: If ``fp`` does not exist (``AudioFileNotFoundError``).
        TypeError: If the file cannot be decoded (``AudioDecodeError``), or ``fp`` is
            neither a path nor a file object.
        ValueError: If a file object holds anything other than WAV or FLAC.
    """

def read_and_resample(fp: "str | Path", target_sr: "int", quality: "ResamplingQuality | None" = None, as_type: "SampleType | None" = None) -> AudioSamples:
    """
    Read an audio file and resample it to a target sample rate in one call.

    Reads the entire file (auto-detecting WAV/FLAC), then resamples to ``target_sr``.
    The resampler is the same as ``AudioSamples.resample``: delay-free, exactly
    ``ceil(n * target_sr / file_sr)`` samples per channel.

    Args:

        fp (str | Path): Path to the audio file.
        target_sr (int): Target sample rate in Hz; must be greater than zero.
        quality (ResamplingQuality, optional): Resampling quality/speed trade-off. Defaults to
            ``ResamplingQuality.high`` for best fidelity.
        as_type (SampleType, optional): Sample type for the returned audio. Defaults to the file's
            sample type, as :func:`read` does; the resampler works on every sample type.

    Returns:

        AudioSamples: The resampled audio.

    Raises:

        ValueError: If ``target_sr`` is zero.
        TypeError: If the file cannot be read or the format is unsupported.
        RuntimeError: If resampling fails.
    """

def read_flac_stream_info(fp: "str | Path") -> StreamInfo:
    """
    Read the STREAMINFO metadata block from a FLAC file.

    Args:

        fp (str | Path): Path to the FLAC file.

    Returns:

        StreamInfo: The FLAC STREAMINFO block.

    Raises:

        OSError: If the file cannot be opened.
        ValueError: If the file is not a FLAC file.
    """

def read_wav_cue_points(fp: "str | Path") -> list[CuePoint]:
    """
    Read the cue points (markers) from a WAV file.

    Args:

        fp (str | Path): Path to the WAV file.

    Returns:

        list[CuePoint]: The cue points (empty if none).

    Raises:

        OSError: If the file cannot be opened.
        ValueError: If the file is not a WAV file or the chunk is malformed.
    """

def read_wav_info_tags(fp: "str | Path") -> InfoMetadata | None:
    """
    Read the LIST/INFO tags from a WAV file.

    Args:

        fp (str | Path): Path to the WAV file.

    Returns:

        InfoMetadata | None: The INFO tags, or ``None`` if the file has no LIST/INFO chunk.

    Raises:

        OSError: If the file cannot be opened.
        ValueError: If the file is not a WAV file or the chunk is malformed.
    """

def read_with_info(fp: "str|Path", as_type: "np.dtype[Any] | None" = None) -> tuple[AudioSamples, AudioInfo]: ...

def save(fp: "str|Path", samples: "AudioSamples", as_type: "SampleType | None" = None, flac_md5: "bool" = True) -> None:
    """
    Save audio to a file, with the format chosen from the extension.

    WAV keeps every sample type, float64 included. FLAC stores 16-bit and 24-bit integers only, so
    any other sample type raises ``ValueError`` unless ``as_type`` asks for ``SampleType.I16`` or
    ``SampleType.I24``.

    Args:

        fp (str | Path): Output path.
        samples (AudioSamples): The audio to write.
        as_type (SampleType, optional): Convert to this sample type as the file is written, without
            a copy of the signal. Defaults to the audio's native type.
        flac_md5 (bool): Store the MD5 signature of the audio in a FLAC file, so that decoders can
            verify it. ``False`` leaves it out, which encodes faster. Defaults to ``True``.

    Raises:

        ValueError: If the format is unsupported, the sample type cannot be stored in FLAC, or
            ``flac_md5`` is false and the file is not FLAC.
        OSError: If the file cannot be written.
    """

def set_thread_limit(limit: "int | None") -> None:
    """
    Limit how many threads a file read or decode may use.

    Reads of large files and FLAC decodes run on a few short-lived threads, which end before the
    call returns. By default their number follows the CPUs available to the process, with a cap,
    or ``AUDIO_SAMPLES_IO_THREADS`` if that is set when the first read starts. This setting
    overrides both.

    Args:

        limit (int | None): Most threads to use; ``1`` keeps everything on the calling thread.
            ``None`` returns to the default.

    Raises:

        ValueError: If ``limit`` is zero.
    """

def thread_limit() -> int:
    """
    The most threads a read of a large file will use at present.

    Returns:

        int: See :func:`set_thread_limit`.
    """

def write_with_metadata(fp: "str | Path", samples: "AudioSamples", metadata: "WavMetadata") -> None:
    """
    Write a WAV file with trailing metadata chunks (LIST/INFO tags, cue points).

    Like :func:`save`, but also serialises the given :class:`WavMetadata` after the audio data,
    persisting tags/markers that a plain read→write round-trip would drop. WAV only.

    Args:

        fp (str | Path): Output ``.wav`` path.
        samples (AudioSamples): The audio to write.
        metadata (WavMetadata): Metadata (INFO tags / cue points) to serialise.

    Raises:

        OSError: If the file cannot be written.
    """

def write_with_options(fp: "str | Path", samples: "AudioSamples", options: "WriteOptions | None" = None, as_type: "SampleType | None" = None, flac_md5: "bool" = True) -> None:
    """
    Write audio samples to a file with explicit write options.

    Like :func:`save` but lets you control the write-buffer size via :class:`WriteOptions`.
    The format is chosen from the file extension (``.wav``/``.flac``/``.aiff``).

    Args:

        fp (str | Path): Output path.
        samples (AudioSamples): The audio to write.
        options (WriteOptions, optional): Write options. Defaults to a 256-byte buffer, which
            holds the header.
        as_type (SampleType, optional): Convert to this sample type as the file is written, without
            a copy of the signal. Defaults to the audio's native type.
        flac_md5 (bool): Store the MD5 signature of the audio in a FLAC file, so that decoders can
            verify it. ``False`` leaves it out, which encodes faster. Defaults to ``True``.

    Raises:

        ValueError: If the format is unsupported, the file is FLAC and the sample type to write is
            not I16 or I24 (FLAC stores integers of those widths only), or ``flac_md5`` is false
            and the file is not FLAC.
        OSError: If the file cannot be written.
    """
