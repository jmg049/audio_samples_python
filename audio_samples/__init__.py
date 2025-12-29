"""
audio_samples - A Rust-based Python extension for audio processing.

This module provides high-performance audio processing capabilities with seamless numpy integration.
Supports multiple audio sample formats (i16, I24, i32, f32, f64) with zero-copy numpy integration.

Example usage:
    >>> import audio_samples as aus
    >>> from audio_samples import generation, io
    >>>
    >>> # Generate a sine wave
    >>> audio = generation.sine_wave(440.0, 1.0)
    >>>
    >>> # Read an audio file
    >>> audio = io.read("path/to/file.wav")
    >>>
    >>> # Process audio
    >>> audio.scale(0.5)
    >>> audio.apply_butterworth_lowpass(4, 1000.0, 44100.0)
"""

# The native extension is compiled as audio_samples.cpython-*.so
# When placed in the audio_samples/ directory, Python can import it
# but we need to use the correct module name for the PyInit function
import importlib.util
import sys
from pathlib import Path

# Find and load the .so file directly
_package_dir = Path(__file__).parent
_so_files = list(_package_dir.glob("audio_samples.cpython-*.so"))
if _so_files:
    # The module init function is PyInit_audio_samples, so we must use "audio_samples" as name
    _spec = importlib.util.spec_from_file_location("audio_samples", _so_files[0])
    _ext = importlib.util.module_from_spec(_spec)
    # Store the original module in sys.modules temporarily
    _old_module = sys.modules.get("audio_samples")
    try:
        _spec.loader.exec_module(_ext)
    finally:
        # Restore ourselves as the audio_samples module
        if _old_module is not None:
            sys.modules["audio_samples"] = _old_module

    # Import everything from the native extension
    for _name in dir(_ext):
        if not _name.startswith('_'):
            globals()[_name] = getattr(_ext, _name)

    # Re-export submodules for proper namespace access
    generation = _ext.generation
    io = _ext.io

    # Also make available the main class and config types
    AudioSamples = _ext.AudioSamples
    IirFilterDesign = _ext.IirFilterDesign
    EqBand = _ext.EqBand
    ParametricEq = _ext.ParametricEq

    # Expose generation functions at the top level for convenience
    _generation_functions = [
        "sine_wave", "cosine_wave", "sawtooth_wave", "square_wave", "triangle_wave",
        "chirp", "white_noise", "pink_noise", "brown_noise", "impulse", "silence"
    ]
    for _fname in _generation_functions:
        globals()[_fname] = getattr(generation, _fname)
else:
    raise ImportError("Could not find the audio_samples native extension (.so file)")

__all__ = [
    # Main class
    "AudioSamples",
    # Filter/EQ classes
    "IirFilterDesign",
    "EqBand",
    "ParametricEq",
    # Submodules
    "generation",
    "io",
    # Generation functions (also available at top level)
    "sine_wave",
    "cosine_wave",
    "sawtooth_wave",
    "square_wave",
    "triangle_wave",
    "chirp",
    "white_noise",
    "pink_noise",
    "brown_noise",
    "impulse",
    "silence",
]
