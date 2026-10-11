"""Regression tests for Rust bug fixes seen through the bindings, and for the
binding/stub fixes listed in the librosa-parity audit (PY_BINDING_ISSUES).

Golden fixture: ``analysis_fixes.json`` (librosa 1.0.0).
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

import audio_samples as aus
from audio_samples import utils as U

from conftest import fixture_array, fixture_signal, fixture_sr, load_librosa_fixture


@pytest.fixture(scope="module")
def fx():
    return load_librosa_fixture("analysis_fixes")


# ---------------------------------------------------------------------------
# Rust fixes (golden fixtures)
# ---------------------------------------------------------------------------


def test_autocorrelation_scale(fx):
    """The autocorrelation is no longer divided by N twice: r[k] * (n - k)
    equals librosa.autocorrelate (exact FFT correlations of the same data with
    different FFT sizes; 7e-16 of the lag-0 value measured in Rust; bound 1e-12)."""
    case = fx["autocorrelate"]
    x = fixture_signal(case["signal"])
    n = len(x)
    m = case["max_size"]
    r = np.asarray(aus.AudioSamples.new_mono(x, fixture_sr()).autocorrelation(m - 1))
    assert len(r) == m
    raw = r * (n - np.arange(m))
    expected = fixture_array(case["values"])
    assert np.max(np.abs(raw - expected)) <= 1e-12 * np.max(np.abs(expected))


def test_yin_confident_frames(fx):
    """track_pitch(YIN) agrees with librosa.yin within 1 cent on frames whose whole
    window lies in one note (measured 0.20 cents in Rust: the difference functions
    differ slightly, see tests/librosa_reference/analysis_fixes.rs). Our frames are
    not centred: frame i covers librosa frame i + frame_length / 2 / hop."""
    case = fx["yin"]
    audio = aus.AudioSamples.new_mono(fixture_signal(case["signal"]), fixture_sr())
    contour = audio.track_pitch(
        case["frame_length"],
        case["hop_length"],
        aus.types.PitchDetectionMethod.yin,
        case["trough_threshold"],
        case["fmin"],
        case["fmax"],
    )
    frames = contour.frames()
    shift = case["frame_length"] // 2 // case["hop_length"]
    assert case["frames"]
    for t, f_lr in zip(case["frames"], case["f0"]):
        ours = frames[t - shift].frequency
        assert ours is not None, f"frame {t - shift} unvoiced"
        assert abs(1200 * np.log2(ours / f_lr)) < 1.0, (t, ours, f_lr)


def test_note_to_midi_grammar(fx):
    case = fx["note_to_midi"]
    for name, midi in zip(case["names"], case["midi"]):
        assert U.note_to_midi(name) == midi, name


# ---------------------------------------------------------------------------
# Binding / stub fixes
# ---------------------------------------------------------------------------


def test_midi_to_note_out_of_range_is_value_error():
    assert U.midi_to_note(69) == "A4"
    for bad in (-1, 128, 10**6):
        with pytest.raises(ValueError):
            U.midi_to_note(bad)


def test_conversions_accept_arrays():
    # The array forms give the scalar results element by element.
    hz = np.geomspace(30.0, 4000.0, 50)
    names, cents = U.frequency_to_note(hz)
    assert names == [U.frequency_to_note(f)[0] for f in hz.tolist()]
    assert np.array_equal(cents, [U.frequency_to_note(f)[1] for f in hz.tolist()])
    midi = np.arange(128)
    assert U.midi_to_note(midi) == [U.midi_to_note(int(m)) for m in midi]
    notes = U.midi_to_note(midi)
    assert np.array_equal(U.note_to_midi(notes), midi)
    assert np.array_equal(U.note_to_frequency(notes), [U.note_to_frequency(n) for n in notes])
    s = np.arange(0, 10**6, 997)
    assert np.array_equal(U.samples_to_time(s, 22050), [U.samples_to_time(int(x), 22050) for x in s])
    assert isinstance(U.samples_to_time(np.int64(5), 22050), float)
    with pytest.raises(ValueError, match="samples"):
        U.samples_to_time([3, -1], 22050)
    with pytest.raises(ValueError, match="midi_note"):
        U.midi_to_note([60, 128])


def test_clicks_dtype_keyword():
    from audio_samples import effects as E

    times = [0.1, 0.5, 1.2]
    y64 = E.clicks(times=times, sr=22050, length=30000)
    y32 = E.clicks(times=times, sr=22050, length=30000, dtype=np.float32)
    assert y64.dtype == np.float64 and y32.dtype == np.float32
    assert np.array_equal(y32, y64.astype(np.float32))
    assert E.clicks(times=times, sr=22050, dtype="float64").dtype == np.float64
    with pytest.raises(ValueError, match="dtype"):
        E.clicks(times=times, sr=22050, dtype=np.int16)


def test_viterbi_binary_dtype_keyword():
    from audio_samples import sequence as Q

    rng = np.random.default_rng(3)
    prob = rng.random((5, 200))
    t = np.array([[0.9, 0.1], [0.2, 0.8]])
    s64 = Q.viterbi_binary(prob, t)
    s16 = Q.viterbi_binary(prob, t, dtype=np.uint16)
    assert s64.dtype == np.int64 and s16.dtype == np.uint16
    assert np.array_equal(s64, s16)
    one, lp = Q.viterbi_binary(prob[0], t, dtype=np.uint16, return_logp=True)
    assert one.shape == (200,) and one.dtype == np.uint16 and isinstance(lp, float)
    with pytest.raises(ValueError, match="dtype"):
        Q.viterbi_binary(prob, t, dtype=np.float32)


def test_generator_dtype_accepts_scalar_types():
    for spec in (np.float64, np.dtype("float64"), "float64"):
        assert aus.sine_wave(440.0, 0.01, dtype=spec).dtype == "f64"
    assert aus.sine_wave(440.0, 0.01, dtype=np.int16).dtype == "i16"
    # The default is float32 (documented; unchanged).
    assert aus.sine_wave(440.0, 0.01).dtype == "f32"
    with pytest.raises(TypeError):
        aus.sine_wave(440.0, 0.01, dtype=np.complex64)


def test_hz_to_mel_default_stays_htk():
    # HTK: 2595 log10(1 + 1000/700); librosa's Slaney default needs htk=False.
    assert U.hz_to_mel(1000.0) == pytest.approx(2595 * np.log10(1 + 1000 / 700), rel=1e-14)
    assert U.hz_to_mel(1000.0, htk=False) == pytest.approx(15.0, rel=1e-14)
    assert U.mel_to_hz(U.hz_to_mel(440.0)) == pytest.approx(440.0, rel=1e-13)


def test_onset_detection_function_returns_times_first(sine_wave_mono):
    audio, _, params = sine_wave_mono
    times, odf = audio.onset_detection_function(aus.types.OnsetDetectionConfig.default())
    assert len(times) == len(odf) and len(times) > 2
    # times are strictly increasing seconds within the signal.
    t = np.asarray(times)
    assert np.all(np.diff(t) > 0) and t[-1] <= params["duration"] + 0.1


def test_hpss_accepts_hpss_config(sine_wave_mono):
    audio, _, _ = sine_wave_mono
    h, p = audio.hpss(aus.types.HpssConfig())
    assert h.samples_per_channel() > 0 and p.samples_per_channel() > 0
    h2, _ = audio.hpss(aus.types.HpssConfig.musical)
    assert h2.samples_per_channel() > 0
    with pytest.raises(TypeError):
        audio.hpss(aus.types.HpssConfig(), 17)
    with pytest.raises(TypeError):
        audio.hpss("nope")


def test_text_signatures_match_real_signatures():
    vad_sig = aus.types.VadConfig.__text_signature__
    assert vad_sig.startswith("(method, channel_policy, /, frame_size")
    assert "median_filter_harmonic" in aus.AudioSamples.hpss.__text_signature__
    assert "win_size" not in aus.AudioSamples.hpss.__text_signature__
    # New functions expose inspectable signatures.
    sig = inspect.signature(aus.spectral.power_to_db)
    assert list(sig.parameters)[:2] == ["S", "ref"]
    sig = inspect.signature(aus.AudioSamples.melspectrogram)
    assert sig.parameters["hop_length"].default == 512


def test_chroma_norm_is_exported():
    S = aus.spectrograms
    assert hasattr(S, "ChromaNorm")
    S.ChromaParams(norm=S.ChromaNorm.l1)


def test_submodules_importable():
    import importlib

    for name in ("audio_samples.spectral", "audio_samples.sequence", "audio_samples.utils", "audio_samples.types"):
        assert importlib.import_module(name) is not None
    from audio_samples.spectral import stft  # noqa: F401
    from audio_samples.sequence import dtw  # noqa: F401


def test_mu_law_top_level():
    assert aus.mu_law_encode(0.0) == 0
    np.testing.assert_array_equal(aus.mu_law_encode([[-1.0, 1.0]]), [[-128, 127]])
    assert aus.mu_law_decode(0) == 0.0
