"""The stubs state the type the library returns.

Each expression below is evaluated, and its result's dtype compared with the expected one. The same
expression goes through ``reveal_type`` under mypy and pyright, which must report that dtype. The
rule under test is the precision rule: float32 / complex64 for uint8, int16 and float32 audio, float64 /
complex128 for int32 and float64 audio; float32 for a float32 or complex64 array and float64 for
anything else; and bool for the boolean results.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import audio_samples as aus
from audio_samples import effects, features, rhythm, sequence, spectral  # noqa: F401
from audio_samples import generation

ROOT = Path(__file__).resolve().parent.parent

SETUP = """\
from __future__ import annotations
from typing import Any
import numpy as np
import audio_samples as aus
from audio_samples import effects, features, generation, rhythm, sequence, spectral

i16 = generation.sine_wave(440.0, 1.0, 22050, 0.5, dtype=np.int16)
f32 = generation.sine_wave(440.0, 1.0, 22050, 0.5, dtype=np.float32)
i32 = generation.sine_wave(440.0, 1.0, 22050, 0.5, dtype=np.int32)
f64 = generation.sine_wave(440.0, 1.0, 22050, 0.5, dtype=np.float64)
y32 = np.sin(np.arange(4096) / 10).astype(np.float32)
y64 = np.sin(np.arange(4096) / 10).astype(np.float64)
s32 = np.abs(spectral.stft(y32)).astype(np.float32)
s64 = np.abs(spectral.stft(y64)).astype(np.float64)
d32 = spectral.stft(y32).astype(np.complex64)
d64 = spectral.stft(y64).astype(np.complex128)
e64 = rhythm.onset_strength(S=s64, sr=22050)
e32 = e64.astype(np.float32)
"""

# Methods whose real results follow the sample type, with the arguments they need.
FLOAT_METHODS = {
    "spectrogram_matrix": "",
    "melspectrogram": "",
    "mfcc_frames": "",
    "spectral_centroid_frames": "",
    "spectral_bandwidth_frames": "",
    "spectral_rolloff_frames": "",
    "spectral_flatness_frames": "",
    "spectral_contrast_frames": "",
    "poly_features": "",
    "rms_frames": "",
    "zero_crossing_rate_frames": "",
    "chroma_stft": "",
    "pseudo_cqt": "",
    "hybrid_cqt": "",
    "chroma_cqt": "",
    "chroma_cens": "",
    "tonnetz": "",
    "yin": "fmin=100.0, fmax=2000.0",
    "onset_strength": "",
    "onset_strength_multi": "",
    "tempogram": "",
    "plp": "",
    "tempogram_ratio": "",
    "hybrid_tempogram": "",
    "lpc": "order=4",
}
COMPLEX_METHODS = {"stft_matrix": "", "cqt": "", "vqt": "", "fourier_tempogram": ""}
SINGLE = ("i16", "f32")
DOUBLE = ("i32", "f64")

CASES: list[tuple[str, str]] = []
for name, arguments in FLOAT_METHODS.items():
    CASES += [(f"{audio}.{name}({arguments})", "float32") for audio in SINGLE]
    CASES += [(f"{audio}.{name}({arguments})", "float64") for audio in DOUBLE]
for name, arguments in COMPLEX_METHODS.items():
    CASES += [(f"{audio}.{name}({arguments})", "complex64") for audio in SINGLE]
    CASES += [(f"{audio}.{name}({arguments})", "complex128") for audio in DOUBLE]
CASES += [
    # Tuples: the element selected is the one the rule is about.
    ("f32.pyin(fmin=100.0, fmax=2000.0)[0]", "float32"),
    ("f32.pyin(fmin=100.0, fmax=2000.0)[1]", "bool"),
    ("f64.pyin(fmin=100.0, fmax=2000.0)[0]", "float64"),
    ("f64.pyin(fmin=100.0, fmax=2000.0)[1]", "bool"),
    ("f32.piptrack()[0]", "float32"),
    ("f64.reassigned_spectrogram()[2]", "float64"),
    # The envelopes and to_numpy keep the sample type.
    ("i16.to_numpy()", "int16"),
    ("f64.to_numpy()", "float64"),
    ("i32.amplitude_envelope()", "int32"),
    ("f32.analytic_envelope()", "float32"),
    ("f32.fft(44100)", "complex128"),
    ("i16.rfft(44100)", "float64"),
    ("f32.tempo()", "float64"),
    # Free functions.
    ("spectral.stft(y32)", "complex64"),
    ("spectral.stft(y64)", "complex128"),
    ("spectral.stft([0.0] * 4096)", "complex128"),
    ("spectral.istft(d32)", "float32"),
    ("spectral.istft(d64)", "float64"),
    ("spectral.power_to_db(s32)", "float32"),
    ("spectral.power_to_db(s64)", "float64"),
    ("spectral.db_to_amplitude(s32)", "float32"),
    ("spectral.delta(s32)", "float32"),
    ("spectral.frame(y32, frame_length=64, hop_length=32)", "float32"),
    ("spectral.window_sumsquare(window='hann', n_frames=4)", "float64"),
    ("spectral.window_sumsquare(window='hann', n_frames=4, dtype=np.float32)", "float32"),
    ("features.spectral_rolloff_from_spectrogram(s32, sr=22050)", "float32"),
    ("features.spectral_rolloff_from_spectrogram(s64, sr=22050)", "float64"),
    ("features.yin(y32, fmin=100.0, fmax=2000.0, sr=22050)", "float32"),
    ("features.pyin(y32, fmin=100.0, fmax=2000.0, sr=22050)[1]", "bool"),
    ("features.pcen(s32, sr=22050, hop_length=512)", "float32"),
    ("features.pcen(s64, sr=22050, hop_length=512, return_zf=True)[1]", "float64"),
    ("features.a_weighting(np.array([100.0, 1000.0]))", "float64"),
    ("rhythm.tempogram(onset_envelope=e32, sr=22050)", "float32"),
    ("rhythm.fourier_tempogram(onset_envelope=e64, sr=22050)", "complex128"),
    ("rhythm.tempo(onset_envelope=e32, sr=22050)", "float64"),
    ("rhythm.onset_detect(onset_envelope=e64, sr=22050)", "int64"),
    ("rhythm.onset_detect(onset_envelope=e64, sr=22050, units='time')", "float64"),
    ("rhythm.beat_track(onset_envelope=e64, sr=22050)[0]", "float64"),
    ("rhythm.beat_track(onset_envelope=e64, sr=22050)[1]", "int64"),
    ("rhythm.beat_track(onset_envelope=e64, sr=22050, units='time')[1]", "float64"),
    ("effects.griffinlim(s32, n_iter=1)", "float32"),
    ("effects.lpc(y32, order=4)", "float32"),
    ("effects.clicks(times=[0.1], sr=22050)", "float64"),
    ("effects.clicks(times=[0.1], sr=22050, dtype=np.float32)", "float32"),
    ("effects.phase_vocoder(d32, rate=1.5)", "complex64"),
    ("effects.preemphasis(y32)", "float32"),
    ("effects.preemphasis(y64, return_zf=True)[1]", "float64"),
    ("effects.hpss(s32)[0]", "float32"),
    ("effects.hpss(d64)[0]", "complex128"),
    ("effects.hpss(d64, mask=True)[0]", "float64"),
    ("effects.softmask(s32, s32)", "float32"),
    # Boolean results.
    ("sequence.recurrence_matrix(s32)", "bool"),
    ("sequence.recurrence_matrix(s64)", "bool"),
    ("sequence.recurrence_matrix(s32, mode='distance')", "float32"),
    ("sequence.recurrence_matrix(s64, mode='affinity')", "float64"),
    ("sequence.cross_similarity(s32, s32)", "bool"),
    ("sequence.cross_similarity(s64, s64, mode='distance')", "float64"),
    ("sequence.recurrence_to_lag(sequence.recurrence_matrix(s64))", "bool"),
    ("sequence.lag_to_recurrence(sequence.recurrence_to_lag(sequence.recurrence_matrix(s64)))", "bool"),
    ("sequence.recurrence_to_lag(sequence.recurrence_matrix(s32, mode='distance'))", "float32"),
    ("sequence.viterbi_binary(np.full((1, 5), 0.5), sequence.transition_loop(2, 0.5))", "bool"),
    ("sequence.viterbi_binary(np.full((1, 5), 0.5), sequence.transition_loop(2, 0.5), dtype=np.uint16)", "uint16"),
    ("sequence.viterbi_binary(np.full((1, 5), 0.5), sequence.transition_loop(2, 0.5), return_logp=True)[0]", "bool"),
    ("sequence.viterbi(np.full((3, 5), 1 / 3), sequence.transition_uniform(3))", "int64"),
    ("sequence.dtw(s32[:8, :16], s32[:8, :16], backtrack=False)", "float32"),
    ("sequence.dtw(s64[:8, :16], s64[:8, :16])[0]", "float64"),
    ("sequence.dtw(s64[:8, :16], s64[:8, :16])[1]", "int64"),
    ("sequence.dtw(C=s32[:8, :16], backtrack=False)", "float32"),
    ("sequence.rqa(s32[:6, :6], backtrack=False)", "float32"),
    ("sequence.sync(s32, [0, 3])", "float32"),
    # Audio built from arguments, by the array it holds.
    ("generation.white_noise(0.1, 8000, 1.0, np.float64, 1).to_numpy()", "float64"),
    ("generation.white_noise(0.1, 8000, dtype=np.int32).to_numpy()", "int32"),
    ("generation.example_clip('speech', 0.1, 8000, dtype=np.int16).to_numpy()", "int16"),
    ("aus.AudioSamples.istft_matrix(d32, sample_rate=22050).to_numpy()", "float32"),
    ("aus.AudioSamples.istft_matrix(d64, sample_rate=22050).to_numpy()", "float64"),
    ("aus.AudioSamples.magphase(d32)[0]", "float32"),
    ("aus.AudioSamples.magphase(d64)[1]", "complex128"),
    ("aus.AudioSamples.griffinlim_audio(s32, sample_rate=22050, n_iter=1).to_numpy()", "float32"),
    ("aus.AudioSamples.new_mono(y64, 8000).to_numpy()", "float64"),
    ("aus.AudioSamples.zeros_mono_i16(4, 8000).to_numpy()", "int16"),
    ("aus.AudioSamples.concatenate([f32, f32]).to_numpy()", "float32"),
    ("f64.as_i16().to_numpy()", "int16"),
    ("i16.as_f32().to_numpy()", "float32"),
    ("i16.resample(11025).to_numpy()", "int16"),
    ("i16.time_stretch(2.0).to_numpy()", "int16"),
    ("(i16 + 1.0).to_numpy()", "int16"),
    ("(1.0 + i16).to_numpy()", "float64"),
    ("(1.0 - i16).to_numpy()", "int16"),
    ("f32.harmonic_percussive()[0].to_numpy()", "float32"),
]

_NAMESPACE: dict = {}


@pytest.fixture(scope="module")
def namespace():
    if not _NAMESPACE:
        exec(compile(SETUP, "setup", "exec"), _NAMESPACE)
    return _NAMESPACE


@pytest.mark.parametrize("expression,dtype", CASES, ids=[c[0] for c in CASES])
def test_runtime_dtype(expression, dtype, namespace):
    result = eval(expression, namespace)
    assert np.asarray(result).dtype == np.dtype(dtype)


# How each dtype appears in a revealed type: numpy's aliases show as the precision classes.
_REVEALED = {
    "bool": ("bool",),
    "int16": ("int16", "signedinteger[_16Bit]"),
    "int32": ("int32", "signedinteger[_32Bit]"),
    "int64": ("int64", "signedinteger[_64Bit]"),
    "uint16": ("uint16", "unsignedinteger[_16Bit]"),
    "float32": ("float32", "floating[_32Bit]"),
    "float64": ("float64", "floating[_64Bit]"),
    "complex64": ("complex64", "complexfloating[_32Bit"),
    "complex128": ("complex128", "complexfloating[_64Bit"),
}


def _plain(text: str) -> str:
    """A revealed type without module paths."""
    return re.sub(r"\b(?:numpy|builtins)(?:\._\w+)*\.", "", text)


def _reveal_source() -> str:
    lines = [SETUP]
    for expression, _ in CASES:
        lines.append(f"reveal_type({expression})")
    return "\n".join(lines) + "\n"


def _first_reveal_line() -> int:
    return SETUP.count("\n") + 2


def _assert_reveals(revealed: dict[int, str], tool: str) -> None:
    problems = []
    for offset, (expression, dtype) in enumerate(CASES):
        line = _first_reveal_line() + offset
        text = revealed.get(line)
        if text is None:
            problems.append(f"{expression}: no revealed type")
        elif not any(option in text for option in _REVEALED[dtype]):
            problems.append(f"{expression}: {text}, expected {dtype}")
        elif "Any" in text.replace("tuple[Any, ...]", ""):
            problems.append(f"{expression}: {text} contains Any")
    assert problems == [], f"{tool}: " + "; ".join(problems)


@pytest.fixture(scope="module")
def reveal_file(tmp_path_factory):
    path = tmp_path_factory.mktemp("reveal") / "reveal.py"
    path.write_text(_reveal_source())
    return path


def test_mypy_reveals_the_dtype(reveal_file):
    pytest.importorskip("mypy")
    run = subprocess.run(
        [sys.executable, "-m", "mypy", "--config-file", str(ROOT / "pyproject.toml"), "--no-error-summary", str(reveal_file)],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    revealed = {}
    errors = []
    for line in run.stdout.splitlines():
        note = re.match(r".*reveal\.py:(\d+): note: Revealed type is \"(.*)\"", line)
        if note:
            revealed[int(note.group(1))] = _plain(note.group(2))
        elif ": error:" in line:
            errors.append(line)
    assert errors == []
    _assert_reveals(revealed, "mypy")


UNKNOWN_SOURCE = """\
from typing import Any
import numpy as np
from numpy.typing import NDArray
import audio_samples as aus
from audio_samples import spectral, sequence

audio = aus.io.read("x.wav")
array: NDArray[Any] = np.zeros(8)
reveal_type(audio)
reveal_type(audio.rms_frames())
reveal_type(audio.stft_matrix())
reveal_type(audio.pyin(fmin=100.0, fmax=2000.0)[0])
reveal_type(spectral.stft(array))
reveal_type(spectral.power_to_db(array))
reveal_type(sequence.recurrence_to_lag(array))
"""


def test_unknown_precision_is_not_guessed(tmp_path):
    """Audio read from a file and arrays of unknown dtype must not get the first overload's precision."""
    path = tmp_path / "unknown.py"
    path.write_text(UNKNOWN_SOURCE)
    run = subprocess.run(
        [sys.executable, "-m", "mypy", "--config-file", str(ROOT / "pyproject.toml"), str(path)],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    revealed = [_plain(m) for m in re.findall(r'Revealed type is "(.*)"', run.stdout)]
    assert len(revealed) == 7, run.stdout
    for text in revealed:
        assert "32Bit" not in text and "64Bit" not in text and "float" not in text and "complex" not in text, text
        assert "Any" in text, text


def _pyright() -> list[str] | None:
    path = shutil.which("pyright", path=str(Path(sys.executable).parent)) or shutil.which("pyright")
    return [path] if path else None


def test_pyright_reveals_the_dtype(reveal_file):
    command = _pyright()
    if command is None:
        pytest.skip("pyright is not installed")
    config = reveal_file.parent / "pyrightconfig.json"
    config.write_text(json.dumps({"reportOverlappingOverload": "none", "reportMissingModuleSource": "none"}))
    run = subprocess.run(
        [*command, "--outputjson", "--pythonpath", sys.executable, "-p", str(config)],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    report = json.loads(run.stdout)
    revealed = {}
    errors = []
    for diagnostic in report["generalDiagnostics"]:
        line = diagnostic["range"]["start"]["line"] + 1
        message = diagnostic["message"]
        found = re.match(r'Type of ".*" is "(.*)"', message, re.S)
        if found:
            revealed[line] = found.group(1)
        elif diagnostic["severity"] == "error":
            errors.append(f"{line}: {message}")
    assert errors == []
    _assert_reveals(revealed, "pyright")


def test_usage_file_runs():
    sys.path.insert(0, str(ROOT / "tests" / "typing"))
    try:
        import usage  # type: ignore[import-not-found]

        usage.run_all()
    finally:
        sys.path.pop(0)


@pytest.mark.parametrize("method", ["scale", "clip", "normalize", "remove_dc_offset"])
def test_in_place_methods_return_none(method):
    audio = generation.sine_wave(440.0, 0.1, 8000)
    arguments = {
        "scale": (0.5,),
        "clip": (-0.5, 0.5),
        "normalize": (-1.0, 1.0, aus.types.NormalizationMethod.minmax),
        "remove_dc_offset": (),
    }[method]
    assert getattr(audio, method)(*arguments) is None
