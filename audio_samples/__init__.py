"""
audio_samples - A Rust-based Python extension for audio processing.

Audio processing implemented in Rust, with conversion to and from NumPy arrays.
Supports multiple audio sample formats (i16, I24, i32, f32, f64) with zero-copy numpy integration.

Example usage:
    >>> import audio_samples as aus
    >>> from audio_samples import generation, io
    >>>
    >>> # Generate a 1 s, 440 Hz sine wave at 16 kHz
    >>> audio = generation.sine_wave(440.0, 1.0, sample_rate=16000)
    >>>
    >>> # Read an audio file
    >>> audio = io.read("path/to/file.wav")  # doctest: +SKIP
    >>>
    >>> # Process audio (both methods modify ``audio`` in place)
    >>> audio.scale(0.5)
    >>> audio.butterworth_lowpass(4, 1000.0)

Spectrogram, mel and MFCC parameter classes live in
``audio_samples.spectrograms`` and are re-exported at the top level::

    >>> from audio_samples import StftParams, WindowType
    >>> params = StftParams(n_fft=512, hop_size=160, window=WindowType.hanning)
    >>> result = audio.stft(params)

Errors raised from the Rust core derive from :class:`AudioError` (see
``audio_samples.exceptions``); each subclass also derives from the matching
built-in exception (``ValueError``, ``TypeError``, ...).
"""

import sys as _sys
from _thread import RLock as _RLock

# Importing this package runs this file and nothing else. The extension, NumPy,
# matplotlib and package metadata are loaded when first used: the extension on
# the first access to any of its names (``_load``).

_NATIVE_SUBMODULES = (
    "io",
    "utils",
    "utils.generation",
    "utils.detection",
    "generation",
    "types",
    "spectrograms",
    "exceptions",
    "explain",
    "spectral",
    "sequence",
    "features",
    "rhythm",
    "effects",
    "linalg",
    "inverse",
    "factorization",
    "pitch_filterbank",
    "segment",
    "arrays",
    "notation",
)

_load_lock = _RLock()
_loaded = False


def _load():
    """Load the native extension and publish its names in this module."""
    global _loaded
    if _loaded:
        return
    with _load_lock:
        if _loaded:
            return
        _load_extension()
        _loaded = True


def _load_extension():
    # The native extension is compiled as audio_samples.cpython-*.so
    # When placed in the audio_samples/ directory, Python can import it
    # but we need to use the correct module name for the PyInit function
    import importlib.machinery as _machinery
    import os as _os

    # Find the .so file. Prefer the exact build for this interpreter/platform
    # (multiple cpython-3XX builds can coexist in this directory); fall back to
    # any match only if that exact one is missing.
    _package_dir = _os.path.dirname(__file__)
    _exact_so = _os.path.join(_package_dir, f"audio_samples{_machinery.EXTENSION_SUFFIXES[0]}")
    if _os.path.exists(_exact_so):
        _so_files = [_exact_so]
    else:
        _so_files = [
            _os.path.join(_package_dir, _f)
            for _f in sorted(_os.listdir(_package_dir))
            if _f.startswith("audio_samples.cpython-") and _f.endswith(".so")
        ]
    if not _so_files:
        raise ImportError("Could not find the audio_samples native extension (.so file)")

    # The module init function is PyInit_audio_samples, so we must use "audio_samples" as name
    _loader = _machinery.ExtensionFileLoader("audio_samples", _so_files[0])
    _spec = _machinery.ModuleSpec("audio_samples", _loader, origin=_so_files[0])
    _ext = _loader.create_module(_spec)
    # Store the original module in sys.modules temporarily
    _old_module = _sys.modules.get("audio_samples")
    try:
        _loader.exec_module(_ext)
    finally:
        # Restore ourselves as the audio_samples module
        if _old_module is not None:
            _sys.modules["audio_samples"] = _old_module

    _names = globals()

    # Import everything from the native extension
    for _name in dir(_ext):
        if not _name.startswith("_"):
            _names[_name] = getattr(_ext, _name)

    # Note: generation is a submodule of utils
    _names["generation"] = _ext.utils.generation

    # Register the native submodules in sys.modules so that
    # ``from audio_samples.spectrograms import StftParams`` (and the same for
    # io, utils, types, ...) works like for a regular package.
    for _subname, _submod in {
        "io": _ext.io,
        "utils": _ext.utils,
        "utils.generation": _ext.utils.generation,
        "utils.detection": _ext.utils.detection,
        "generation": _ext.utils.generation,
        "types": _ext.types,
        "spectrograms": _ext.spectrograms,
        "exceptions": _ext.exceptions,
        "explain": _ext.explain,
        "spectral": _ext.spectral,
        "sequence": _ext.sequence,
        "features": _ext.features,
        "rhythm": _ext.rhythm,
        "effects": _ext.effects,
        "linalg": _ext.linalg,
        "inverse": _ext.inverse,
        "factorization": _ext.factorization,
        "pitch_filterbank": _ext.pitch_filterbank,
        "segment": _ext.segment,
        "arrays": _ext.arrays,
        "notation": _ext.notation,
    }.items():
        _sys.modules.setdefault(f"audio_samples.{_subname}", _submod)
        # ``exceptions`` is compiled from source text and has a spec naming a
        # file that does not exist. The import system reloads through that
        # spec, so point it at the finder below.
        if getattr(_submod, "__spec__", None) is not None:
            _submod.__spec__ = _machinery.ModuleSpec(f"audio_samples.{_subname}", _NativeSubmoduleFinder)

    # Spectrogram / mel / MFCC parameter classes, re-exported at top level.
    # ``WindowType`` is the spectrograms window class: it is the one accepted
    # by ``StftParams`` and by every method taking a window. The older
    # ``audio_samples.types.WindowType`` class (not accepted by any function)
    # stays available under that path for backwards compatibility.
    for _name in (
        "StftParams",
        "SpectrogramParams",
        "MelParams",
        "MfccParams",
        "LogParams",
        "ChromaParams",
        "CqtParams",
        "ErbParams",
        "MelNorm",
        "StftResult",
        "WindowType",
    ):
        _names[_name] = getattr(_ext.spectrograms, _name)

    # Plotting module (if available - currently not registered in Rust)
    _names["plotting"] = getattr(_ext, "plotting", None)

    # Expose generation functions at the top level for convenience
    for _fname in (
        "sine_wave",
        "cosine_wave",
        "sawtooth_wave",
        "square_wave",
        "triangle_wave",
        "white_noise",
        "pink_noise",
        "brown_noise",
        "impulse",
        "silence",
        "chirp",
        "stereo_sine_wave",
        "stereo_chirp",
        "stereo_silence",
    ):
        _names[_fname] = getattr(_ext.utils.generation, _fname)


class _NativeSubmoduleFinder:
    """Resolves ``import audio_samples.io`` and friends: those modules live in
    the extension, which may not be loaded yet."""

    @staticmethod
    def find_spec(fullname, path=None, target=None):
        prefix, _, rest = fullname.partition(".")
        if prefix != "audio_samples" or rest not in _NATIVE_SUBMODULES:
            return None
        import importlib.machinery as _machinery

        _load()
        return _machinery.ModuleSpec(fullname, _NativeSubmoduleFinder)

    @staticmethod
    def create_module(spec):
        return _sys.modules[spec.name]

    @staticmethod
    def exec_module(module):
        return None


# Ahead of the path finder: audio_samples/utils holds only type stubs, which that finder would
# take for a namespace package.
_sys.meta_path.insert(0, _NativeSubmoduleFinder)


def __getattr__(name):
    if name == "forms":
        import importlib

        module = importlib.import_module("audio_samples.forms")
        globals()["forms"] = module
        return module
    if name == "mpl":
        # Matplotlib visualisation extension (pure Python), None without matplotlib.
        import importlib

        try:
            module = importlib.import_module("audio_samples.mpl")
        except ImportError:
            module = None
        globals()["mpl"] = module
        return module
    if name == "__version__":
        from importlib.metadata import PackageNotFoundError, version

        try:
            value = version("audio_samples")
        except PackageNotFoundError:  # pragma: no cover - running from a source tree
            value = "unknown"
        globals()["__version__"] = value
        return value
    if not name.startswith("__") and not _loaded:
        _load()
        if name in globals():
            return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    _load()
    return sorted({*globals(), "mpl", "__version__"})


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
    "utils",
    "spectral",
    "sequence",
    "features",
    "rhythm",
    "effects",
    "linalg",
    "inverse",
    "factorization",
    "pitch_filterbank",
    "segment",
    "arrays",
    "notation",
    "spectrograms",
    "mpl",
    "forms",
    "spectrograms",
    "exceptions",
    "explain",
    "ExplainDoc",
    # Spectrogram parameter classes (from audio_samples.spectrograms)
    "StftParams",
    "SpectrogramParams",
    "MelParams",
    "MfccParams",
    "LogParams",
    "ChromaParams",
    "CqtParams",
    "ErbParams",
    "MelNorm",
    "StftResult",
    "WindowType",
    # Exceptions
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
    # Mu-law companding with quantised codes
    "mu_law_encode",
    "mu_law_decode",
    # Generation functions (also available at top level)
    "sine_wave",
    "cosine_wave",
    "sawtooth_wave",
    "square_wave",
    "triangle_wave",
    "white_noise",
    "pink_noise",
    "brown_noise",
    "impulse",
    "silence",
    "chirp",
    "stereo_sine_wave",
    "stereo_chirp",
    "stereo_silence",
]
