"""
Type stubs for audio_python.io - Audio file I/O functions.

This module provides functions to read and write audio files in various formats.
"""

from typing import Optional, Any
import numpy as np
from . import AudioSamples


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


def read(fp: str, as_type: Optional[np.dtype[Any]] = None) -> AudioSamples:
    """
    Read an audio file and return AudioSamples.

    Args:
        fp: Path to the audio file
        as_type: Optional numpy dtype to convert the audio to (default: native format)

    Returns:
        AudioSamples: The loaded audio data

    Raises:
        TypeError: If the file cannot be read or the format is unsupported
    """
    ...


def read_with_info(fp: str, as_type: Optional[np.dtype[Any]] = None) -> tuple[AudioSamples, AudioInfo]:
    """
    Read an audio file and return AudioSamples along with file information.

    Args:
        fp: Path to the audio file
        as_type: Optional numpy dtype to convert the audio to (default: native format)

    Returns:
        tuple[AudioSamples, AudioInfo]: The loaded audio data and file information

    Raises:
        TypeError: If the file cannot be read or the format is unsupported
    """
    ...


def save(fp: str, samples: AudioSamples) -> None:
    """
    Save AudioSamples to an audio file.

    The format is determined by the file extension. The audio is saved
    using its native sample type.

    Args:
        fp: Path to save the audio file
        samples: The audio data to save

    Raises:
        TypeError: If the file cannot be written or the format is unsupported
    """
    ...


def save_as_type(fp: str, samples: AudioSamples, as_type: np.dtype[Any]) -> None:
    """
    Save AudioSamples to an audio file with type conversion.

    The format is determined by the file extension. The audio is converted
    to the specified sample type before saving.

    Args:
        fp: Path to save the audio file
        samples: The audio data to save
        as_type: The numpy dtype to convert to before saving (i16, i32, f32, f64)

    Raises:
        TypeError: If the file cannot be written, format is unsupported, or conversion fails
    """
    ...
