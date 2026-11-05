"""The ``.pyi`` stubs of the librosa-compatible API match the runtime signatures.

For every function in ``spectral``, ``features``, ``rhythm``, ``effects``,
``sequence`` and the librosa conversions of ``utils``, and for the
librosa-compatible ``AudioSamples`` methods, each stub parameter must exist at
runtime with the same default, and required parameters must be required in
both.
"""

from __future__ import annotations

import ast
import inspect
import math
from pathlib import Path

import pytest

import audio_samples as aus

PKG = Path(aus.__file__).parent
EMPTY = inspect.Parameter.empty

MODULES = {
    "spectral.pyi": aus.spectral,
    "features.pyi": aus.features,
    "rhythm.pyi": aus.rhythm,
    "effects.pyi": aus.effects,
    "sequence.pyi": aus.sequence,
}
UTILS = (
    "time_to_samples",
    "samples_to_frames",
    "frames_to_samples",
    "time_to_frames_floor",
    "times_like",
    "samples_like",
)
METHODS = (
    "stft_matrix", "istft_matrix", "spectrogram_matrix", "melspectrogram", "mfcc_frames",
    "spectral_centroid_frames", "spectral_bandwidth_frames", "spectral_rolloff_frames",
    "spectral_flatness_frames", "spectral_contrast_frames", "poly_features", "rms_frames",
    "zero_crossing_rate_frames", "reassigned_spectrogram", "chroma_stft", "piptrack",
    "estimate_tuning", "cqt", "vqt", "pseudo_cqt", "hybrid_cqt", "icqt", "griffinlim_cqt",
    "chroma_cqt", "chroma_cens", "chroma_vqt", "tonnetz", "yin", "pyin", "onset_strength",
    "onset_strength_multi", "onset_detect", "tempogram", "fourier_tempogram", "tempo",
    "beat_track", "plp", "tempogram_ratio", "hybrid_tempogram", "time_stretch",
    "pitch_shift", "preemphasis", "deemphasis", "trim_db", "trim_db_in_place", "split_db",
    "remix", "remix_in_place", "harmonic", "percussive", "harmonic_percussive", "lpc",
    "griffinlim_audio", "apply_window", "apply_filter",
)


def _stub_defs(path: Path, cls: str | None = None) -> dict[str, list[ast.FunctionDef]]:
    body = ast.parse(path.read_text()).body
    if cls is not None:
        body = next(n for n in body if isinstance(n, ast.ClassDef) and n.name == cls).body
    # All of a function's overloads, in order.
    defs: dict[str, list[ast.FunctionDef]] = {}
    for n in body:
        if isinstance(n, ast.FunctionDef):
            defs.setdefault(n.name, []).append(n)
    return defs


def _stub_params(fn: ast.FunctionDef):
    a = fn.args
    pos = a.posonlyargs + a.args
    defaults = [None] * (len(pos) - len(a.defaults)) + list(a.defaults)
    out = list(zip((p.arg for p in pos), defaults))
    out += list(zip((p.arg for p in a.kwonlyargs), a.kw_defaults))
    return [(n, d) for n, d in out if n not in ("self", "cls")]


def _value(node):
    if node is None:
        return EMPTY
    try:
        return ast.literal_eval(node)
    except ValueError:
        src = ast.unparse(node)
        if src in ("np.inf", "math.inf"):
            return math.inf
        if src in ("np.nan", "math.nan"):
            return math.nan
        return Ellipsis  # an expression the test does not evaluate


def _same(a, b) -> bool:
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return a == b


def _check(stubs: list[ast.FunctionDef], runtime) -> list[str]:
    """Each parameter must match the runtime. An overload may make a defaulted parameter
    required, or give it a narrower default, so a parameter is compared with every overload."""
    params = inspect.signature(runtime).parameters
    problems = []
    defaults: dict[str, list] = {}
    for stub in stubs:
        for name, default in _stub_params(stub):
            defaults.setdefault(name, []).append(default)
    for name, options in defaults.items():
        problem = _check_parameter(name, options, params)
        if problem:
            problems.append(problem)
    return problems


def _check_parameter(name, options, params):
    if name not in params:
        return f"{name}: not a runtime parameter"
    last = None
    for default in options:
        last = _parameter_problem(name, default, params)
        if last is None:
            return None
    return last


def _parameter_problem(name, default, params):
    sd, rd = _value(default), params[name].default
    if sd is Ellipsis:
        return None
    if (sd is EMPTY) != (rd is EMPTY):
        return f"{name}: required in {'stub' if sd is EMPTY else 'runtime'} only"
    if sd is not EMPTY and not _same(sd, rd):
        return f"{name}: stub default {sd!r}, runtime {rd!r}"
    return None


CASES = [(f, n) for f in MODULES for n in _stub_defs(PKG / f) if hasattr(MODULES[f], n)]
CASES += [("utils/__init__.pyi", n) for n in UTILS]


@pytest.mark.parametrize("stub,name", CASES)
def test_function_stub_matches_runtime(stub, name):
    module = MODULES.get(stub, aus.utils)
    assert _check(_stub_defs(PKG / stub)[name], getattr(module, name)) == []


@pytest.mark.parametrize("name", METHODS)
def test_method_stub_matches_runtime(name):
    stub = _stub_defs(PKG / "__init__.pyi", "AudioSamples")[name]
    assert _check(stub, getattr(aus.AudioSamples, name)) == []
