"""Exception hierarchy for audio_samples.

Every exception raised from the Rust core derives from :class:`AudioError`.
Each subclass also derives from the matching built-in exception
(``ValueError``, ``TypeError``, ``OSError``, ...), so ``except ValueError:``
and similar clauses catch it.
"""

from typing import Optional

class AudioError(Exception):
    """Base class of every error raised by audio_samples."""

    code: Optional[str]
    """Stable diagnostic code, e.g. ``"audio_samples::parameter::invalid_value"``."""
    help: Optional[str]
    """Hint on how to fix the call (also added as a note on Python 3.11+)."""

class AudioParameterError(AudioError, ValueError):
    """An argument has an invalid value (out of range, inconsistent, ...)."""

class AudioLayoutError(AudioError, ValueError):
    """Array shape, channel count or memory layout is incompatible."""

class AudioConversionError(AudioError, TypeError):
    """A sample-type conversion or cast failed."""

class AudioRuntimeError(AudioError, RuntimeError):
    """Base class for failures that happen while processing valid input."""

class AudioProcessingError(AudioRuntimeError):
    """A DSP operation failed while processing the audio."""

class AudioFeatureError(AudioRuntimeError):
    """The operation needs a feature that was not compiled into this build."""

class AudioSpectrogramError(AudioRuntimeError):
    """A spectrogram, mel, MFCC or CQT computation failed."""

class AudioIOError(AudioError, OSError):
    """Reading or writing an audio file failed at the operating-system level."""

class AudioFileNotFoundError(AudioIOError, FileNotFoundError):
    """The audio file does not exist."""

class AudioFormatError(AudioError, ValueError):
    """The file format is unsupported or the file contents are corrupt."""

class AudioDecodeError(AudioError, TypeError):
    """An existing file could not be decoded as audio. It is also a ``TypeError``,
    so ``except TypeError:`` catches it."""

__all__ = [
    "AudioError",
    "AudioParameterError",
    "AudioLayoutError",
    "AudioConversionError",
    "AudioRuntimeError",
    "AudioProcessingError",
    "AudioFeatureError",
    "AudioSpectrogramError",
    "AudioIOError",
    "AudioFileNotFoundError",
    "AudioFormatError",
    "AudioDecodeError",
]
