"""Usability regressions.

Covers NumPy interop, the spectrogram parameter classes, sample-rate checks,
panic-free construction, the exception hierarchy, introspectable signatures,
documented facts, stub drift, config reprs and the Jupyter HTML view.
"""

from __future__ import annotations

import ast
import inspect
import sys
import types as pytypes
from pathlib import Path

import numpy as np
import pytest

import audio_samples as aus

PKG_DIR = Path(aus.__file__).resolve().parent
SR = 16000


@pytest.fixture
def mono():
    return aus.sine_wave(440.0, 0.5, SR)


@pytest.fixture
def stereo(mono):
    return aus.AudioSamples.stack([mono, mono * 0.5])


# ---------------------------------------------------------------------------
# 1. NumPy interop
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ufunc", [np.sin, np.abs, np.square, np.exp])
def test_ufunc_matches_to_numpy(mono, stereo, ufunc):
    for audio in (mono, stereo):
        expected = ufunc(audio.to_numpy())
        result = ufunc(audio)
        assert isinstance(result, np.ndarray)
        np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize(
    "func", [np.mean, np.max, np.min, np.std, np.sum, np.fft.rfft, np.fft.fft]
)
def test_array_function_matches_to_numpy(mono, stereo, func):
    for audio in (mono, stereo):
        np.testing.assert_allclose(func(audio), func(audio.to_numpy()), rtol=1e-6)


def test_array_function_with_keywords_and_lists(mono, stereo):
    np.testing.assert_allclose(
        np.mean(stereo, axis=1), np.mean(stereo.to_numpy(), axis=1), rtol=1e-6
    )
    np.testing.assert_allclose(
        np.concatenate([mono, mono]),
        np.concatenate([mono.to_numpy(), mono.to_numpy()]),
    )


def test_ndarray_on_left_returns_audio(mono):
    arr = np.ones(mono.samples_per_channel(), dtype=np.float32)
    for result in (arr + mono, arr * mono, np.add(mono, arr)):
        assert isinstance(result, aus.AudioSamples)
        assert result.sample_rate == SR
    np.testing.assert_allclose((arr + mono).to_numpy(), arr + mono.to_numpy())


def test_reductions_and_comparisons_are_plain_numpy(mono):
    assert isinstance(np.add.reduce(mono), (np.floating, float))
    mask = np.abs(mono) >= 0.5
    assert mask.dtype == bool


def test_asarray_copy_true_returns_independent_copy(mono):
    if np.lib.NumpyVersion(np.__version__) >= "2.0.0":
        arr = np.asarray(mono, copy=True)
    else:
        arr = np.array(mono)
    arr[:] = 0
    assert np.any(mono.to_numpy() != 0)


# ---------------------------------------------------------------------------
# 2. Spectrogram parameter classes
# ---------------------------------------------------------------------------


def test_spectrograms_submodule_importable():
    from audio_samples.spectrograms import (  # noqa: F401
        LogParams,
        MelParams,
        MfccParams,
        SpectrogramParams,
        StftParams,
        WindowType,
    )

    assert sys.modules["audio_samples.spectrograms"] is aus.spectrograms
    assert aus.StftParams is StftParams
    assert aus.SpectrogramParams is SpectrogramParams
    assert aus.MelParams is MelParams
    assert aus.MfccParams is MfccParams
    assert aus.LogParams is LogParams
    assert aus.WindowType is WindowType


def test_other_submodules_importable():
    from audio_samples.io import read  # noqa: F401
    from audio_samples.utils import hz_to_mel, mel_to_hz  # noqa: F401
    from audio_samples.types import VadConfig  # noqa: F401
    from audio_samples.exceptions import AudioError  # noqa: F401
    from audio_samples.generation import sine_wave  # noqa: F401

    assert aus.AudioError is AudioError


def test_top_level_window_type_works_in_stft_params(mono):
    params = aus.StftParams(n_fft=512, hop_size=160, window=aus.WindowType.hanning)
    result = mono.stft(params)
    assert result.shape[0] == 257
    kaiser = aus.StftParams(512, 160, aus.WindowType.kaiser(8.0))
    assert mono.stft(kaiser).shape == result.shape


def test_legacy_types_window_type_still_available():
    assert hasattr(aus.types, "WindowType")
    assert aus.types.WindowType.hanning is not None


def test_mel_and_mfcc_via_top_level_classes(mono):
    sp = aus.StftParams(512, 160, aus.WindowType.hanning)
    mel = mono.mel_spectrogram(
        aus.SpectrogramParams(sp, sample_rate=SR), aus.MelParams(n_mels=40, f_min=0.0, f_max=8000.0)
    )
    assert mel.shape[0] == 40
    mfcc = mono.mfcc(sp, 40, aus.MfccParams(n_mfcc=13))
    assert mfcc.shape[0] == 13


def test_pyproject_does_not_require_spectrograms_pip_package():
    pyproject = PKG_DIR.parent / "pyproject.toml"
    if not pyproject.exists():
        pytest.skip("installed from a wheel, no pyproject.toml next to the package")
    text = pyproject.read_text()
    deps = text.split("dependencies = [", 1)[1].split("]", 1)[0]
    assert "spectrograms" not in deps


# ---------------------------------------------------------------------------
# 3. Sample-rate mismatches
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("op", ["+", "-", "*", "/"])
def test_arithmetic_rejects_sample_rate_mismatch(op):
    ns = {
        "a": aus.sine_wave(440.0, 0.5, 16000),  # 8000 samples
        "b": aus.sine_wave(440.0, 1.0, 8000),  # 8000 samples, different rate
    }
    with pytest.raises(ValueError, match=r"16000 Hz and 8000 Hz") as info:
        eval(f"a {op} b", ns)
    assert isinstance(info.value, aus.AudioParameterError)


@pytest.mark.parametrize("op", ["+=", "-=", "*=", "/="])
def test_inplace_arithmetic_rejects_sample_rate_mismatch(op):
    ns = {"a": aus.sine_wave(440.0, 0.5, 16000), "b": aus.sine_wave(440.0, 1.0, 8000)}
    with pytest.raises(ValueError, match=r"16000 Hz and 8000 Hz"):
        exec(f"a {op} b", ns)


def test_ufunc_rejects_sample_rate_mismatch():
    a = aus.sine_wave(440.0, 0.5, 16000)
    b = aus.sine_wave(440.0, 1.0, 8000)
    with pytest.raises(ValueError, match=r"16000 Hz and 8000 Hz"):
        np.add(a, b)


def test_same_rate_arithmetic_still_works(mono):
    other = aus.sine_wave(220.0, 0.5, SR)
    for result in (mono + other, mono - other, mono * other):
        assert result.sample_rate == SR
    ratio = mono / (other + 2.0)
    np.testing.assert_allclose(
        ratio.to_numpy(), mono.to_numpy() / (other.to_numpy() + 2.0), rtol=1e-6
    )


def test_spectrogram_params_sample_rate_must_match(mono):
    sp = aus.StftParams(512, 160, aus.WindowType.hanning)
    bad = aus.SpectrogramParams(sp, sample_rate=44100)
    with pytest.raises(ValueError, match=r"44100.*16000") as info:
        mono.mel_spectrogram(bad, aus.MelParams(n_mels=40, f_min=0.0, f_max=8000.0))
    assert isinstance(info.value, aus.AudioError)
    erb = aus.spectrograms.ErbParams(n_filters=32, f_min=50.0, f_max=7000.0)
    with pytest.raises(ValueError, match=r"44100.*16000"):
        mono.gammatone_spectrogram(bad, erb)


# ---------------------------------------------------------------------------
# 4. No Rust panics on invalid arguments
# ---------------------------------------------------------------------------

EMPTY_CONSTRUCTORS = [
    lambda: aus.AudioSamples.new_mono(np.array([], np.float32), SR),
    lambda: aus.AudioSamples.new_mono(np.array([], np.int16), SR),
    lambda: aus.AudioSamples.new_multi(np.zeros((2, 0), np.float32), SR),
    lambda: aus.AudioSamples.new_multi(np.zeros((0, 10), np.float32), SR),
    lambda: aus.AudioSamples.from_array(np.array([], np.float64), SR),
    lambda: aus.AudioSamples.zeros_mono(0, SR),
    lambda: aus.AudioSamples.zeros_multi(2, 0, SR),
    lambda: aus.AudioSamples.ones_multi(0, 10, SR),
    lambda: aus.AudioSamples.uniform_mono(0, SR, 1.0),
    lambda: aus.AudioSamples._new_mono_f32_from_np(np.array([], np.float32), SR),
]


@pytest.mark.parametrize("make", EMPTY_CONSTRUCTORS)
def test_empty_buffers_rejected_at_construction(make):
    with pytest.raises(ValueError, match="cannot be empty"):
        make()


def test_wrong_ndim_has_clear_message():
    with pytest.raises(TypeError, match="1-D"):
        aus.AudioSamples.new_mono(np.zeros((2, 100), np.float32), SR)
    with pytest.raises(TypeError, match="2-D"):
        aus.AudioSamples.new_multi(np.zeros(100, np.float32), SR)


def test_to_stereo_on_stereo_is_usable(stereo):
    result = stereo.to_stereo(aus.StereoConversionMethod.duplicate)
    assert result.channels == 2
    assert result.shape == [2, stereo.samples_per_channel()]
    np.testing.assert_array_equal(result.to_numpy(), stereo.to_numpy())
    assert "2 ch" in str(result)
    assert "2 ch" in repr(result)


def test_str_of_multichannel_does_not_panic():
    audio = aus.AudioSamples.new_multi(np.zeros((3, 100), np.float32), SR)
    assert "3 ch" in str(audio)


@pytest.mark.parametrize(
    "call",
    [
        lambda m, s: repr(aus.AudioSamples.new_mono(np.array([0.5], np.float32), SR)),
        lambda m, s: str(s),
        lambda m, s: repr(s),
        lambda m, s: s.to_stereo(aus.StereoConversionMethod.duplicate).rms(),
        lambda m, s: m.trim(0.1, 0.1),
        lambda m, s: aus.AudioSamples.concatenate([]),
        lambda m, s: aus.AudioSamples.stack([]),
        lambda m, s: s.extract_channel(5),
        lambda m, s: m.split(5.0),
    ],
)
def test_no_panic_exception_escapes(mono, stereo, call):
    try:
        call(mono, stereo)
    except Exception:
        pass
    except BaseException as exc:  # PanicException is a BaseException
        pytest.fail(f"{type(exc).__name__} escaped: {exc}")


def test_time_axis_has_one_value_per_sample(mono, stereo):
    for audio in (mono, stereo):
        t = audio.time_axis()
        n = audio.samples_per_channel()
        assert t.shape == (n,)
        np.testing.assert_allclose(t, np.arange(n) / audio.sample_rate, atol=1e-12)


# ---------------------------------------------------------------------------
# 5. Exceptions
# ---------------------------------------------------------------------------


def test_exception_hierarchy_keeps_builtin_bases():
    assert issubclass(aus.AudioParameterError, aus.AudioError)
    assert issubclass(aus.AudioParameterError, ValueError)
    assert issubclass(aus.AudioLayoutError, ValueError)
    assert issubclass(aus.AudioConversionError, TypeError)
    assert issubclass(aus.AudioRuntimeError, RuntimeError)
    assert issubclass(aus.AudioProcessingError, aus.AudioRuntimeError)
    assert issubclass(aus.AudioSpectrogramError, RuntimeError)
    assert issubclass(aus.AudioIOError, OSError)
    assert issubclass(aus.AudioFileNotFoundError, FileNotFoundError)
    assert issubclass(aus.AudioFormatError, ValueError)
    assert issubclass(aus.AudioDecodeError, TypeError)
    for name in aus.exceptions.__all__:
        assert issubclass(getattr(aus, name), aus.AudioError)


def test_core_error_carries_code_and_help(mono):
    with pytest.raises(ValueError) as info:
        mono.butterworth_lowpass(4, 9000.0)
    err = info.value
    assert isinstance(err, aus.AudioParameterError)
    assert "cutoff" in str(err)
    assert err.code and err.code.startswith("audio_samples::")
    assert err.help
    if sys.version_info >= (3, 11):
        assert any(err.help in note for note in err.__notes__)


def test_spectrogram_errors_are_single_wrapped(mono):
    sp = aus.StftParams(512, 160, aus.WindowType.hanning)
    with pytest.raises(RuntimeError) as info:
        mono.mfcc(sp, 10, aus.MfccParams(13))
    assert isinstance(info.value, aus.AudioError)
    assert isinstance(info.value.args[0], str)


def test_missing_file_raises_file_not_found(tmp_path):
    missing = str(tmp_path / "does_not_exist.wav")
    for fn in (aus.io.read, aus.io.info, aus.io.peek_native_type, aus.io.read_with_info):
        with pytest.raises(FileNotFoundError) as info:
            fn(missing)
        assert isinstance(info.value, aus.AudioError)
        assert info.value.filename == missing
        assert missing in str(info.value)


def test_non_audio_file_raises_type_error(tmp_path):
    bogus = tmp_path / "not_audio.wav"
    bogus.write_text("this is not a wav file")
    with pytest.raises(TypeError) as info:
        aus.io.read(str(bogus))
    assert isinstance(info.value, aus.AudioError)


def test_unsupported_save_format_message_not_doubled(tmp_path, mono):
    with pytest.raises(ValueError) as info:
        aus.io.save(str(tmp_path / "x.mp3"), mono)
    msg = str(info.value)
    assert msg.count("Unsupported format") == 1
    assert isinstance(info.value, aus.AudioError)


# ---------------------------------------------------------------------------
# 6. Signatures are introspectable
# ---------------------------------------------------------------------------


def _public_callables():
    modules = {
        "audio_samples": aus,
        "io": aus.io,
        "utils": aus.utils,
        "generation": aus.generation,
        "detection": aus.utils.detection,
        "types": aus.types,
    }
    seen = []
    for mod_name, mod in modules.items():
        for name in dir(mod):
            if name.startswith("_"):
                continue
            obj = getattr(mod, name)
            if isinstance(obj, (pytypes.BuiltinFunctionType, pytypes.FunctionType)):
                seen.append((f"{mod_name}.{name}", obj))
            elif (
                isinstance(obj, type)
                and obj.__module__.startswith("audio_samples")
                and not issubclass(obj, BaseException)
            ):
                for attr in dir(obj):
                    if attr.startswith("_"):
                        continue
                    member = getattr(obj, attr)
                    if callable(member) and not isinstance(member, obj):
                        seen.append((f"{mod_name}.{name}.{attr}", member))
    return seen


PUBLIC_CALLABLES = _public_callables()


def test_public_callables_found():
    names = {n for n, _ in PUBLIC_CALLABLES}
    assert "audio_samples.AudioSamples.stft" in names
    assert "io.read" in names
    assert len(PUBLIC_CALLABLES) > 300


def test_every_public_callable_has_a_parseable_signature():
    failures = []
    for name, obj in PUBLIC_CALLABLES:
        try:
            inspect.signature(obj)
        except (ValueError, TypeError) as exc:
            failures.append(f"{name}: {exc}")
    assert not failures, "\n".join(failures)


def test_no_rust_type_names_leak_into_docs():
    for name, obj in PUBLIC_CALLABLES:
        doc = obj.__doc__ or ""
        assert "PyAudioSamples" not in doc, name
        assert "audio_python" not in doc, name
        assert "PitchDetection.yin" not in doc, name


# ---------------------------------------------------------------------------
# 7. Documented facts
# ---------------------------------------------------------------------------


def test_zero_crossing_rate_is_per_second(mono):
    expected = mono.zero_crossings() / mono.duration_seconds
    assert mono.zero_crossing_rate() == pytest.approx(expected)
    assert mono.zero_crossing_rate() == pytest.approx(2 * 440.0, rel=0.01)
    assert "per second" in aus.AudioSamples.zero_crossing_rate.__doc__


def test_resample_default_quality_is_documented(mono):
    assert "ResamplingQuality.fast" in aus.AudioSamples.resample.__doc__
    np.testing.assert_array_equal(
        mono.resample(8000).to_numpy(),
        mono.resample(8000, aus.ResamplingQuality.fast).to_numpy(),
    )


def test_package_docstring_examples_run():
    ns: dict = {}
    for line in (aus.__doc__ or "").splitlines():
        line = line.strip()
        if line.startswith(">>> ") and "doctest: +SKIP" not in line:
            exec(line[4:], ns)
    assert "result" in ns


# ---------------------------------------------------------------------------
# 8. Stub drift
# ---------------------------------------------------------------------------


def _stub_tree(name: str) -> ast.Module:
    return ast.parse((PKG_DIR / name).read_text())


def test_audio_samples_stub_lists_every_runtime_member():
    tree = _stub_tree("__init__.pyi")
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "AudioSamples")
    names = set()
    for node in cls.body:
        if isinstance(node, ast.FunctionDef):
            names.add(node.name)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    runtime = {n for n in dir(aus.AudioSamples) if not n.startswith("_")}
    assert not runtime - names, sorted(runtime - names)
    stub_only = {n for n in names - runtime if not n.startswith("_")}
    assert not stub_only, sorted(stub_only)


def test_top_level_stub_lists_every_runtime_name():
    tree = _stub_tree("__init__.pyi")
    names = set()
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
            names.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update(a.asname or a.name for a in node.names)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    # audio_samples.migrate is a command and appears as an attribute only once it has been imported.
    runtime = {n for n in dir(aus) if not n.startswith("_")} - {"migrate"}
    assert not runtime - names, sorted(runtime - names)


def test_stub_defaults_match_runtime():
    sig = inspect.signature(aus.AudioSamples.detect_pitch_autocorr)
    assert all(p.default is p.empty for n, p in sig.parameters.items() if n != "self")
    tree = _stub_tree("__init__.pyi")
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "AudioSamples")
    funcs = {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}
    for name in ("detect_pitch_autocorr", "harmonic_to_noise_ratio", "harmonic_analysis"):
        assert not funcs[name].args.defaults, name


def test_stub_property_types_match_runtime(mono):
    assert isinstance(mono.dtype, str)
    assert isinstance(mono.shape, list)
    src = (PKG_DIR / "__init__.pyi").read_text()
    assert "def dtype(self) -> str:" in src
    assert "def shape(self) -> list[int]:" in src


def test_stubs_do_not_import_pip_spectrograms():
    for stub in PKG_DIR.glob("*.pyi"):
        for node in ast.parse(stub.read_text()).body:
            if isinstance(node, ast.ImportFrom):
                assert node.module != "spectrograms", stub.name
            if isinstance(node, ast.Import):
                assert all(a.name != "spectrograms" for a in node.names), stub.name
    assert (PKG_DIR / "spectrograms.pyi").exists()
    assert (PKG_DIR / "exceptions.pyi").exists()


def test_utils_doc_examples_use_importable_module():
    for name in ("hz_to_mel", "mel_to_hz", "power_to_db"):
        doc = getattr(aus.utils, name).__doc__
        assert "audio_samples.audio_math" not in doc
        assert "from audio_samples.utils import" in doc


# ---------------------------------------------------------------------------
# 9. Config reprs
# ---------------------------------------------------------------------------


def _configs():
    sp = aus.StftParams(1024, 256, aus.WindowType.hanning)
    return {
        "VadConfig": aus.VadConfig.default(),
        "CompressorConfig": aus.CompressorConfig.default(),
        "LimiterConfig": aus.LimiterConfig.default(),
        "GateConfig": aus.GateConfig.default(),
        "ExpanderConfig": aus.ExpanderConfig.default(),
        "IirFilterDesign": aus.IirFilterDesign.butterworth_lowpass(4, 1000.0),
        "EqBand": aus.EqBand.peak(1000.0, 3.0, 1.0),
        "ParametricEq": aus.ParametricEq(),
        "HpssConfig": aus.HpssConfig(stft_params=sp),
        "PeakPickingConfig": aus.types.PeakPickingConfig.default(),
        "OnsetDetectionConfig": aus.OnsetDetectionConfig.default(),
        "BeatTrackingConfig": aus.BeatTrackingConfig(120.0, aus.OnsetDetectionConfig.default()),
    }


def test_config_reprs_show_fields():
    for name, cfg in _configs().items():
        text = repr(cfg)
        assert "object at 0x" not in text, name
        assert "{" in text or "(" in text, (name, text)


def test_vad_config_repr_and_getters():
    cfg = aus.VadConfig.default()
    text = repr(cfg)
    assert "frame_size" in text and "energy_threshold_db" in text
    assert cfg.frame_size == 1024
    assert cfg.hop_size == 512
    assert isinstance(cfg.energy_threshold_db, float)
    assert repr(cfg.method)
    for field in (
        "pad_end",
        "zcr_min",
        "zcr_max",
        "min_speech_frames",
        "min_silence_frames",
        "hangover_frames",
        "smooth_frames",
        "speech_band_low_hz",
        "speech_band_high_hz",
        "spectral_ratio_threshold",
        "channel_policy",
    ):
        assert getattr(cfg, field) is not None, field
    with pytest.raises(AttributeError):
        cfg.frame_size = 3


def test_audio_samples_info_repr(mono):
    text = repr(mono.info())
    assert text.startswith("AudioSamplesInfo(")
    assert "sample_rate=16000" in text


# ---------------------------------------------------------------------------
# 10. Jupyter HTML view
# ---------------------------------------------------------------------------


def test_repr_html_mono_has_table_and_player(mono):
    html = mono._repr_html_()
    assert "<table>" in html
    for label in ("dtype", "channels", "sample rate", "duration", "peak", "RMS"):
        assert label in html
    assert "<audio controls" in html
    assert "data:audio/wav;base64," in html
    assert "<script" not in html


def test_repr_html_player_decodes_to_wav(stereo):
    import base64
    import io
    import re
    import wave

    html = stereo._repr_html_()
    b64 = re.search(r"base64,([A-Za-z0-9+/=]+)", html).group(1)
    with wave.open(io.BytesIO(base64.b64decode(b64))) as w:
        assert w.getnchannels() == 2
        assert w.getsampwidth() == 2
        assert w.getframerate() == SR
        assert w.getnframes() == stereo.samples_per_channel()


def test_repr_html_long_clip_has_no_player():
    long_clip = aus.AudioSamples.zeros_mono(31 * 8000, 8000)
    html = long_clip._repr_html_()
    assert "<table>" in html
    assert "<audio" not in html


def test_repr_html_never_raises():
    cases = [
        aus.AudioSamples.new_mono(np.array([np.nan, 1.0, -1.0], np.float32), SR),
        aus.AudioSamples.new_mono(np.array([1000, -1000, 0], np.int16), SR),
        aus.AudioSamples.new_multi(np.zeros((3, 10), np.float64), SR),
        aus.AudioSamples.new_mono(np.array([5.0, -5.0], np.float64), SR),
    ]
    for audio in cases:
        html = audio._repr_html_()
        assert html is None or "<" in html
