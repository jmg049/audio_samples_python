"""Documented behaviour holds: apply_filter, dtype-like generator arguments,
conversion docs, sample-rate checks on two-signal operations, stub and runtime
agreement, and the stated conventions (mel, MFCC, autocorrelation, FFT, presets,
memory sharing, import cost)."""

from __future__ import annotations

import ast
import doctest
import inspect
from pathlib import Path

import numpy as np
import pytest

import audio_samples as aus
from audio_samples import AudioSamples
from audio_samples.exceptions import AudioParameterError

PKG = Path(aus.__file__).resolve().parent


def _mono(x, sr=8000):
    return AudioSamples.new_mono(np.asarray(x, dtype=np.float64), sample_rate=sr)


# --------------------------------------------------------------------------
# apply_filter / apply_filter_in_place
# --------------------------------------------------------------------------


@pytest.mark.parametrize("m", [1, 2, 5, 31, 200])
def test_apply_filter_matches_numpy_valid(m):
    rng = np.random.default_rng(m)
    x = rng.standard_normal(1000)
    h = rng.standard_normal(m)
    y = _mono(x).apply_filter(h).to_numpy()
    expected = np.convolve(x, h, "valid")
    assert y.shape == (len(x) - m + 1,)
    np.testing.assert_allclose(y, expected, rtol=1e-9, atol=1e-9)


def test_apply_filter_multichannel_and_float32_taps():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((2, 300))
    h = np.array([0.25, 0.5, 0.25], dtype=np.float32)
    y = AudioSamples.new_multi(x, sample_rate=8000).apply_filter(h).to_numpy()
    for ch in range(2):
        np.testing.assert_allclose(y[ch], np.convolve(x[ch], h.astype(np.float64), "valid"))


def test_apply_filter_vs_convolve_lengths():
    x = np.arange(1.0, 6.0)
    h = np.array([1.0, 0.5])
    full = _mono(x).convolve(_mono(h)).to_numpy()
    valid = _mono(x).apply_filter(h).to_numpy()
    np.testing.assert_allclose(full, np.convolve(x, h, "full"))
    np.testing.assert_allclose(valid, np.convolve(x, h, "valid"))
    assert len(full) == 5 + 2 - 1 and len(valid) == 5 - 2 + 1


def test_apply_filter_in_place_and_errors():
    x = np.arange(1.0, 6.0)
    a = _mono(x.copy())
    assert a.apply_filter_in_place(np.array([1.0, -1.0])) is None
    np.testing.assert_allclose(a.to_numpy(), [1.0, 1.0, 1.0, 1.0])
    with pytest.raises(ValueError):
        _mono(x).apply_filter(np.array([]))
    with pytest.raises(AudioParameterError):
        _mono(x).apply_filter(np.ones(10))


def test_apply_window_accepts_float32():
    x = np.ones(16)
    w = np.hanning(16).astype(np.float32)
    out = _mono(x).apply_window(w).to_numpy()
    np.testing.assert_allclose(out, w.astype(np.float64), rtol=1e-6)
    with pytest.raises(TypeError, match="1-D"):
        _mono(x).apply_window(np.ones((2, 8)))


# --------------------------------------------------------------------------
# generator dtype-like arguments
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "dtype,expected",
    [
        (np.float64, np.float64),
        (np.dtype("float64"), np.float64),
        ("float64", np.float64),
        ("f4", np.float32),
        (np.int16, np.int16),
        ("int32", np.int32),
    ],
)
def test_generators_accept_dtype_like(dtype, expected):
    g = aus.generation
    assert g.sine_wave(440.0, 0.01, 8000, dtype=dtype).to_numpy().dtype == expected
    assert g.impulse(0.01, 8000, dtype=dtype).to_numpy().dtype == expected
    assert g.white_noise(0.01, 8000, dtype=dtype, seed=1).to_numpy().dtype == expected


def test_generator_default_dtype_is_documented():
    imp = aus.generation.impulse
    assert imp(0.01, 8000).to_numpy().dtype == np.float32
    assert "Default: float32" in imp.__doc__
    with pytest.raises(TypeError, match="Unsupported dtype"):
        aus.generation.sine_wave(440.0, 0.01, 8000, dtype=np.complex64)


# --------------------------------------------------------------------------
# conversion docs and docstring examples
# --------------------------------------------------------------------------


def test_conversion_methods_documented():
    A = AudioSamples
    for name in ["to_format", "as_u8", "as_i16", "as_i32", "as_f32", "as_f64"]:
        assert "udio-aware" in getattr(A, name).__doc__, name
    for name in ["cast_as", "cast_as_u8", "cast_as_i16", "cast_as_i32", "cast_as_f32", "cast_as_f64"]:
        assert "aw numeric cast" in getattr(A, name).__doc__, name
    assert "-32768" in A.to_format.__doc__ and "32767" in A.to_format.__doc__
    for name in ["new_mono", "new_multi"]:
        assert "arr.copy()" in getattr(A, name).__doc__, name


def test_asymmetric_int_scaling():
    x = _mono([-1.0, -0.5, 0.0, 0.5, 1.0])
    np.testing.assert_array_equal(x.as_i16().to_numpy(), [-32768, -16384, 0, 16384, 32767])
    raw = x.cast_as_i16().to_numpy()
    np.testing.assert_array_equal(raw, [-1, 0, 0, 0, 1])


def _run_doctests(obj, name):
    parser = doctest.DocTestParser()
    test = parser.get_doctest(obj.__doc__ or "", {}, name, None, 0)
    assert test.examples, f"{name} has no examples"
    runner = doctest.DocTestRunner(optionflags=doctest.NORMALIZE_WHITESPACE | doctest.ELLIPSIS)
    runner.run(test)
    res = runner.summarize(verbose=False)
    assert res.failed == 0, f"{name}: {res.failed} doctest failures"


@pytest.mark.parametrize(
    "obj,name",
    [
        (AudioSamples.apply_filter, "apply_filter"),
        (AudioSamples.to_format, "to_format"),
        (AudioSamples.cast_as, "cast_as"),
        (AudioSamples.new_mono, "new_mono"),
        (AudioSamples.new_multi, "new_multi"),
        (aus.utils.snr, "snr"),
        (aus.utils.mse, "mse"),
        (aus.utils.correlation, "correlation"),
        (aus.utils.align_signals, "align_signals"),
    ],
)
def test_docstring_examples_run(obj, name):
    _run_doctests(obj, name)


def test_no_nonexistent_constructors_in_docs():
    for fn in (aus.utils.snr, aus.utils.mse, aus.utils.correlation, aus.utils.align_signals):
        assert "from_mono_f32" not in fn.__doc__
        assert "zeros_mono_f32" not in fn.__doc__
    assert "from_mono_f32" not in (PKG / "utils" / "__init__.pyi").read_text()


# --------------------------------------------------------------------------
# two-signal ops reject mismatched sample rates
# --------------------------------------------------------------------------


def _rate_pair():
    rng = np.random.default_rng(0)
    return _mono(rng.standard_normal(200), 8000), _mono(rng.standard_normal(100), 16000)


@pytest.mark.parametrize(
    "op",
    [
        lambda a, b: a.convolve(b),
        lambda a, b: a.deconvolve(b),
        lambda a, b: a.cross_correlation(b, 5),
        lambda a, b: AudioSamples.mix([a, b]),
        lambda a, b: AudioSamples.concatenate([a, b]),
        lambda a, b: AudioSamples.stack([a, b]),
    ],
)
def test_two_signal_ops_name_both_rates(op):
    a, b = _rate_pair()
    with pytest.raises(AudioParameterError, match=r"8000 Hz and 16000 Hz"):
        op(a, b)


def test_cross_correlation_channel_mismatch_message():
    rng = np.random.default_rng(0)
    a = _mono(rng.standard_normal(100))
    s = AudioSamples.new_multi(rng.standard_normal((2, 100)), sample_rate=8000)
    with pytest.raises(AudioParameterError, match=r"channels, got 1 and 2"):
        a.cross_correlation(s, 5)


def test_cross_correlation_mixed_backing_and_unbiased():
    rng = np.random.default_rng(0)
    x, y = rng.standard_normal(500), rng.standard_normal(500)
    numpy_backed = _mono(x)
    owned = _mono(y).as_f64()  # conversion yields an owned (non-NumPy) buffer
    cc = np.array(numpy_backed.cross_correlation(owned, 3))
    n = len(x)
    expected = np.array([np.sum(x[: n - k] * y[k:]) / (n - k) for k in range(4)])
    np.testing.assert_allclose(cc, expected, rtol=1e-9)


# --------------------------------------------------------------------------
# stubs agree with runtime
# --------------------------------------------------------------------------


def _stub_class(path, cls_name):
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == cls_name:
            return node
    raise AssertionError(f"{cls_name} not in {path.name}")


def _stub_kinds(node):
    kinds = {}
    for n in node.body:
        if isinstance(n, ast.FunctionDef):
            decos = {getattr(d, "id", getattr(d, "attr", None)) for d in n.decorator_list}
            if "setter" in decos:
                continue
            kinds[n.name] = (
                "property" if "property" in decos
                else "static" if "staticmethod" in decos
                else "class" if "classmethod" in decos
                else "method"
            )
        elif isinstance(n, ast.AnnAssign):
            kinds[n.target.id] = "attr"
    return kinds


def _runtime_kind(cls, name):
    for k in cls.__mro__:
        if name in k.__dict__:
            raw = k.__dict__[name]
            break
    else:
        return None
    tname = type(raw).__name__
    if tname == "getset_descriptor" or isinstance(raw, property):
        return "property"
    if isinstance(raw, staticmethod):
        return "static"
    if tname == "classmethod_descriptor" or isinstance(raw, classmethod):
        return "class"
    if callable(raw):
        return "method"
    return "attr"


@pytest.mark.parametrize(
    "module,stub,classes",
    [
        (aus.spectrograms, "spectrograms.pyi",
         ["StftResult", "WindowType", "IldSpectrogram", "IlrSpectrogram", "IpdSpectrogram", "ItdSpectrogram"]),
        (aus.types, "types.pyi", ["PeakPickingConfig", "PadSide", "CompressorConfig"]),
    ],
)
def test_stub_member_kinds_match_runtime(module, stub, classes):
    for cls_name in classes:
        node = _stub_class(PKG / stub, cls_name)
        cls = getattr(module, cls_name)
        for name, kind in _stub_kinds(node).items():
            if name.startswith("__"):
                continue
            rk = _runtime_kind(cls, name)
            assert rk is not None, f"{cls_name}.{name} missing at runtime"
            ok = rk == kind or (kind == "attr" and rk in ("attr", "property"))
            assert ok, f"{cls_name}.{name}: stub {kind}, runtime {rk}"


# --------------------------------------------------------------------------
# documented conventions are numerically true
# --------------------------------------------------------------------------


def _mel_setup():
    sr = 22050
    x = np.random.default_rng(0).standard_normal(sr) * 0.1
    au = _mono(x, sr)
    stft = aus.StftParams(n_fft=1024, hop_size=256, window=aus.spectrograms.WindowType.hanning, centre=True)
    return x, sr, au, aus.SpectrogramParams(stft, sample_rate=sr), stft


def test_mel_spectrogram_conventions():
    librosa = pytest.importorskip("librosa")
    x, sr, au, sp, _ = _mel_setup()
    mp = aus.MelParams(n_mels=40, f_min=0.0, f_max=sr / 2)
    S = np.abs(librosa.stft(x, n_fft=1024, hop_length=256, window="hann", center=True, pad_mode="constant"))
    fb = librosa.filters.mel(sr=sr, n_fft=1024, n_mels=40, fmin=0, fmax=sr / 2, norm=None, htk=False)
    P = fb @ S**2
    M = au.mel_spectrogram(sp, mp)
    assert np.max(np.abs(M - np.sqrt(P))) / np.sqrt(P).max() < 5e-3  # sqrt(mel power)
    D = au.mel_spectrogram(sp, mp, aus.LogParams(floor_db=-80.0))
    assert np.max(np.abs(D - np.maximum(10 * np.log10(P), -80.0))) < 0.1
    tiny = _mono(x * 1e-4, sr).mel_spectrogram(sp, mp, aus.LogParams(floor_db=-80.0))
    assert tiny.min() == -80.0  # absolute floor, not relative to the maximum
    doc = AudioSamples.mel_spectrogram.__doc__
    assert "Slaney" in doc and "absolute" in doc and "sqrt(M @ |X|**2)" in doc


def test_mfcc_conventions():
    librosa = pytest.importorskip("librosa")
    scipy_fft = pytest.importorskip("scipy.fft")
    x, sr, au, _, stft = _mel_setup()
    C = au.mfcc(stft, 40, aus.MfccParams(n_mfcc=13))
    S = np.abs(librosa.stft(x, n_fft=1024, hop_length=256, window="hann", center=True, pad_mode="constant"))
    fb = librosa.filters.mel(sr=sr, n_fft=1024, n_mels=40, fmin=0, fmax=sr / 2, norm=None)
    L = 10 * np.log10(np.maximum(fb @ S**2, 1e-8))
    q = np.arange(13)[:, None]
    ref = scipy_fft.dct(L, type=2, axis=0)[:13] / 2 * (1 + 11 * np.sin(np.pi * q / 22))
    assert np.max(np.abs(C - ref)) / np.abs(ref).max() < 5e-3
    assert "lifter" in AudioSamples.mfcc.__doc__


def test_autocorrelation_is_unbiased_and_fft_one_sided():
    x = np.random.default_rng(3).standard_normal(400)
    au = _mono(x)
    n = len(x)
    ac = np.array(au.autocorrelation(4))
    np.testing.assert_allclose(ac, [np.sum(x[: n - k] * x[k:]) / (n - k) for k in range(5)])
    F = au.fft(512)
    assert F.shape == (1, 257)
    np.testing.assert_allclose(F[0], np.fft.rfft(x, 512))
    np.testing.assert_allclose(au.rfft(512)[0], np.abs(np.fft.rfft(x, 512)))
    assert "unbiased" in AudioSamples.autocorrelation.__doc__
    assert "one-sided" in AudioSamples.fft.__doc__


def test_not_implemented_methods_documented():
    assert "not implemented" in aus.types.PitchDetectionMethod.__doc__
    x = aus.generation.sine_wave(220.0, 0.3, 16000)
    with pytest.raises(AudioParameterError):
        x.track_pitch(1024, 256, aus.PitchDetectionMethod.cepstrum)


def test_onset_detection_function_returns_times_first():
    x = aus.generation.white_noise(1.0, 8000, dtype=np.float64, seed=0)
    times, values = x.onset_detection_function(aus.OnsetDetectionConfig.default())
    assert times[0] == 0.0 and np.all(np.diff(times) > 0)
    assert len(times) == len(values)
    assert "(times, values)" in AudioSamples.onset_detection_function.__doc__


# --------------------------------------------------------------------------
# presets, getters, copies, memory sharing, EnvelopeFollower
# --------------------------------------------------------------------------


def test_compressor_config_getters():
    c = aus.CompressorConfig.vocal  # class attribute, no parentheses
    assert isinstance(c.threshold_db, float) and c.ratio >= 1.0
    for name in ["attack_ms", "release_ms", "makeup_gain_db", "knee_width_db", "lookahead_ms"]:
        assert isinstance(getattr(c, name), float)
    c.knee_type, c.detection_method, c.side_chain  # readable
    d = aus.CompressorConfig.default()
    assert isinstance(d.ratio, float)


def test_peak_picking_getter_and_copy_semantics():
    pp = aus.PeakPickingConfig.drums(44100, 256)  # takes the sample rate and hop size
    assert isinstance(pp.min_peak_separation, int)
    cfg = aus.OnsetDetectionConfig.default()
    before = cfg.peak_picking.min_peak_separation
    cfg.peak_picking.min_peak_separation = before + 7  # modifies a copy only
    assert cfg.peak_picking.min_peak_separation == before
    copy = cfg.peak_picking
    copy.min_peak_separation = before + 7
    cfg.peak_picking = copy
    assert cfg.peak_picking.min_peak_separation == before + 7
    assert "copy" in type(cfg).peak_picking.__doc__


def test_new_mono_shares_memory_with_numpy_array():
    """new_mono wraps the array without copying, so in-place methods modify the caller's array."""
    a = np.random.default_rng(0).standard_normal(4000) * 0.5
    original = a.copy()
    x = AudioSamples.new_mono(a, sample_rate=8000)
    x.reverse_in_place()
    np.testing.assert_array_equal(a, original[::-1])  # caller's array changed
    b = np.random.default_rng(1).standard_normal(4000) * 0.5
    kept = b.copy()
    y = AudioSamples.new_mono(b.copy(), sample_rate=8000)
    y.reverse_in_place()
    np.testing.assert_array_equal(b, kept)  # passing a copy keeps the original


def test_envelope_follower_process():
    f = aus.types.EnvelopeFollower(1.0, 10.0, 8000.0, aus.types.DynamicRangeMethod.peak)
    assert f.process(0.0) == 0.0
    level = f.process(1.0)
    assert 0.0 < level <= 1.0
    f.reset()
    env = f.process_array(np.r_[np.ones(400), np.zeros(400)])
    assert env.shape == (800,)
    assert env[399] > 0.99 and env[-1] < env[399]
    g = aus.types.EnvelopeFollower(1.0, 10.0, 8000.0, aus.types.DynamicRangeMethod.peak)
    seq = np.array([g.process(v) for v in np.r_[np.ones(400), np.zeros(400)]])
    np.testing.assert_allclose(env, seq)


IN_PLACE = (
    "swap_channels pan balance apply_compressor apply_limiter apply_gate apply_expander "
    "apply_iir_filter filtfilt apply_butterworth_lowpass apply_chebyshev_i apply_elliptic_bandstop "
    "apply_bessel_lowpass apply_parametric_eq apply_eq_band apply_peak_filter apply_low_shelf "
    "apply_high_shelf apply_three_band_eq scale normalize clip remove_dc_offset "
    "apply_filter_in_place reverse_in_place fade_in fade_out perturb_in_place"
).split()


@pytest.mark.parametrize("name", IN_PLACE)
def test_in_place_methods_say_they_return_none(name):
    assert "returns ``None``" in getattr(AudioSamples, name).__doc__


def test_in_place_filter_returns_none():
    x = aus.generation.white_noise(0.1, 8000, dtype=np.float64, seed=0)
    design = aus.IirFilterDesign.butterworth_lowpass(4, 1000.0)
    assert x.apply_iir_filter(design) is None
    assert x.filtfilt(design) is None


def test_yin_threshold_wording():
    doc = AudioSamples.detect_pitch_yin.__doc__
    assert "stricter" in doc and "more permissive." not in doc.split("stricter")[0]


def test_adaptive_threshold_bounds_are_readable_properties():
    cfg = aus.AdaptiveThresholdConfig.combined(0.05, 0.9, 16)
    cfg.min_threshold = 0.2
    cfg.max_threshold = 0.8
    assert cfg.min_threshold == 0.2
    assert cfg.max_threshold == 0.8


# --------------------------------------------------------------------------
# importing the package
# --------------------------------------------------------------------------


def test_import_loads_nothing():
    import subprocess
    import sys

    code = (
        "import sys, audio_samples\n"
        "heavy = {'numpy', 'matplotlib', 'pathlib', 'email', 'importlib.metadata', 'audio_samples.io'} & set(sys.modules)\n"
        "assert not heavy, heavy\n"
        "assert 'mpl' in dir(audio_samples) and '__version__' in dir(audio_samples)\n"
        "assert isinstance(audio_samples.__version__, str)\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


@pytest.mark.parametrize(
    "code",
    [
        "from audio_samples import AudioSamples, StftParams, sine_wave, mu_law_encode",
        "import audio_samples.io as io; io.read",
        "from audio_samples.spectrograms import StftParams",
        "from audio_samples.utils.generation import sine_wave",
        "from audio_samples.utils import snr",
        "import audio_samples.exceptions as e; e.AudioError",
        "import audio_samples.notation as n; n.list_thaat",
        "from audio_samples.segment import agglomerative",
        "import audio_samples.arrays as a; a.localmax",
        "from audio_samples.linalg import nnls",
        "from audio_samples.inverse import mel_to_stft",
        "from audio_samples.factorization import decompose",
        "from audio_samples.pitch_filterbank import iirt",
        "from audio_samples import *; sine_wave, AudioError",
        "import audio_samples as a; a.generation, a.exceptions.AudioParameterError",
        "import pickle, audio_samples as a; pickle.loads(pickle.dumps(a.AudioParameterError('m')))",
        "import pickle; pickle.loads(b'\\x80\\x04c' + b'audio_samples.exceptions\\nAudioError\\n.')",
        "import audio_samples as a\ntry:\n    a.no_such_name\nexcept AttributeError:\n    pass\nelse:\n    raise SystemExit(1)",
    ],
)
def test_names_resolve_in_a_fresh_process(code):
    import subprocess
    import sys

    subprocess.run([sys.executable, "-c", code], check=True)


def test_mpl_is_available_on_first_access():
    pytest.importorskip("matplotlib")
    assert aus.mpl.__name__ == "audio_samples.mpl"
