"""librosa parity of the tonal bindings: ``AudioSamples.chroma_stft`` /
``piptrack`` / ``estimate_tuning``, ``audio_samples.features``
(``chroma_filterbank``, ``normalize``, ``pitch_tuning`` and the ``S=`` forms)
and the tuning / octave / key helpers in ``audio_samples.utils``.

Golden fixtures: ``tonal.json`` (``cases/tonal.py``). Tolerances are
peak-relative and equal to the Rust reference suite's
(``tests/librosa_reference/tonal.rs``). Discrete decisions (peak sets,
histogram argmax, key spelling) must match exactly.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

import audio_samples as aus
from audio_samples import features as F
from audio_samples import utils as U

from conftest import (
    assert_peak_close,
    fixture_array,
    fixture_signal,
    fixture_sr,
    load_librosa_fixture,
)

RTOL_FILTER = 1e-12  # same formula in f64: libm exp/log2/pow ulps
RTOL_PIPTRACK = 1e-12  # parabolic shift -b/a can amplify FFT round-off on flat peaks (measured ~3e-16)
ATOL_TUNING = 1e-12  # histogram argmax over identical peak sets: the same bin edge
RTOL_CHROMA = 1e-10  # same STFT, f64 basis, same per-frame norm: FFT / matmul round-off
RTOL_CHROMA_F32 = 1e-6  # vs librosa's default float32 filterbank (eps 1.2e-7 per weight)
RTOL_NORMALIZE = 1e-14  # elementwise division by a norm; summation order (a few ulp)
RTOL_CONV = 1e-14  # closed-form log2 / pow: libm ulps


@pytest.fixture(scope="module")
def fx():
    return load_librosa_fixture("tonal")


@pytest.fixture(scope="module")
def sr():
    return fixture_sr()


def _audio(name, sr):
    if name == "stereo":
        data = np.stack([fixture_signal("stereo_l"), fixture_signal("stereo_r")])
        return aus.AudioSamples.new_multi(data, sr)
    return aus.AudioSamples.new_mono(fixture_signal(name), sr)


def _detuned(fx, sr):
    codes = np.asarray(fx["estimate_tuning"]["detuned_pcm"], dtype=np.float64)
    return aus.AudioSamples.new_mono(codes / 32768.0, sr)


def _norm(v):
    return np.inf if v == "inf" else v


def test_chroma_filterbank(fx):
    for name, case in fx["chroma_filters"].items():
        p = dict(case["params"])
        p["norm"] = _norm(p["norm"])
        fb = F.chroma_filterbank(**p)
        assert list(fb.shape) == case["shape"], name
        step = case["col_step"]
        assert_peak_close(fb[:, ::step], fixture_array(case["subset"]), RTOL_FILTER, name)
        assert_peak_close(fb.sum(axis=1), fixture_array(case["row_sums"]), RTOL_FILTER, name + " sums")


def _check_piptrack(p, m, case, what):
    shape = list(case["shape"])
    if len(shape) == 2:
        shape.insert(0, 1)
    assert list(p.shape) == shape, what
    idx = np.flatnonzero(p)
    # Peak picking is a discrete decision: identical peak sets.
    np.testing.assert_array_equal(idx, np.asarray(case["index"], dtype=np.int64), err_msg=what)
    assert_peak_close(p.ravel()[idx], fixture_array(case["pitch"]), RTOL_PIPTRACK, what + " pitches")
    assert_peak_close(m.ravel()[idx], fixture_array(case["mag"]), RTOL_PIPTRACK, what + " mags")


def test_piptrack(fx, sr):
    f = fx["piptrack"]
    notes = _audio("notes", sr)
    _check_piptrack(*notes.piptrack(), f["notes_default"], "default")
    _check_piptrack(
        *notes.piptrack(n_fft=1024, fmin=100.0, fmax=1500.0, threshold=0.3), f["notes_range_thr"], "range"
    )
    _check_piptrack(*notes.piptrack(threshold=1.0, ref=np.mean), f["notes_ref_mean"], "ref=mean")
    _check_piptrack(*notes.piptrack(ref=2.0), f["notes_ref_value"], "ref=2")
    step = f["stereo_frame_step"]
    p, m = _audio("stereo", sr).piptrack()
    _check_piptrack(p[..., ::step], m[..., ::step], f["stereo_default"], "stereo")
    # S= form, librosa shapes (2-D in, 2-D out).
    s = notes.spectrogram_matrix(n_fft=2048, hop_length=512)
    p2, m2 = F.piptrack_from_spectrogram(s[0], sr=sr)
    _check_piptrack(p2[None], m2[None], f["notes_default"], "S=")


def test_estimate_tuning(fx, sr):
    f = fx["estimate_tuning"]
    notes = _audio("notes", sr)
    got = {
        "notes": notes.estimate_tuning(),
        "notes_res_0001": notes.estimate_tuning(resolution=1e-3),
        "notes_bpo24": notes.estimate_tuning(bins_per_octave=24),
        "notes_power_S": F.estimate_tuning_from_spectrogram(notes.spectrogram_matrix(power=2.0), sr=sr),
        "detuned": _detuned(fx, sr).estimate_tuning(),
        "detuned_nfft4096": _detuned(fx, sr).estimate_tuning(n_fft=4096),
        "mix": _audio("mix", sr).estimate_tuning(),
        "stereo": _audio("stereo", sr).estimate_tuning(),
    }
    for name, v in got.items():
        assert isinstance(v, float)
        assert abs(v - f[name]) < ATOL_TUNING, f"{name}: {v} vs {f[name]}"
    y = fixture_signal("notes")
    assert F.estimate_tuning_from_signal(y, sr=sr) == pytest.approx(f["notes"], abs=ATOL_TUNING)


def test_pitch_tuning(fx):
    for name, case in fx["pitch_tuning"].items():
        got = F.pitch_tuning(case["freqs"], resolution=case["res"], bins_per_octave=case["bpo"])
        assert abs(got - case["tuning"]) < ATOL_TUNING, name
    assert F.pitch_tuning([]) == 0.0
    with pytest.raises(ValueError):
        F.pitch_tuning([440.0], resolution=1.5)


def test_chroma_stft(fx, sr):
    f = fx["chroma_stft"]
    notes = _audio("notes", sr)
    # The "default" fixtures use librosa's default tuning=None (estimated); the
    # binding defaults to tuning=0.0.
    est = dict(tuning=None)
    cases = {
        "notes_default": (notes.chroma_stft(**est), RTOL_CHROMA),
        "notes_default_float32_basis": (notes.chroma_stft(**est), RTOL_CHROMA_F32),
        "notes_tuning0": (notes.chroma_stft(tuning=0.0), RTOL_CHROMA),
        "notes_norm_none": (notes.chroma_stft(norm=None, **est), RTOL_CHROMA),
        "notes_n24": (notes.chroma_stft(n_chroma=24, **est), RTOL_CHROMA),
        "notes_nfft4096_l2": (notes.chroma_stft(n_fft=4096, norm=2, **est), RTOL_CHROMA),
        "mix_default": (_audio("mix", sr).chroma_stft(**est), RTOL_CHROMA),
        "stereo_default": (_audio("stereo", sr).chroma_stft(**est), RTOL_CHROMA),
        "detuned_default": (_detuned(fx, sr).chroma_stft(**est), RTOL_CHROMA),
        "notes_magnitude_S": (
            F.chroma_stft_from_spectrogram(notes.spectrogram_matrix(), sr=sr, **est),
            RTOL_CHROMA,
        ),
    }
    for name, (got, rtol) in cases.items():
        e = fixture_array(f[name])
        e = e[None] if e.ndim == 2 else e
        assert got.shape == e.shape, name
        assert_peak_close(got, e, rtol, name)
    # 2-D S= input gives librosa's 2-D output.
    two_d = F.chroma_stft_from_spectrogram(notes.spectrogram_matrix(power=2.0)[0], sr=sr, **est)
    assert_peak_close(two_d, fixture_array(f["notes_default"]), RTOL_CHROMA, "S= 2-D")


def test_normalize(fx):
    f = fx["normalize"]
    x = fixture_array(f["input"])
    cases = {
        "inf_axis0": {},
        "l2_axis1": dict(norm=2, axis=1),
        "l1_axis0": dict(norm=1, axis=0),
        "l3_axis1": dict(norm=3, axis=1),
        "l0_axis0": dict(norm=0, axis=0),
        "neg_inf_axis1": dict(norm=-np.inf, axis=1),
        "none_axis_l2": dict(norm=2, axis=None),
        "thr_fill_true_l2_axis1": dict(norm=2, axis=1, threshold=1e-3, fill=True),
        "thr_fill_false_inf_axis1": dict(axis=1, threshold=1e-3, fill=False),
        "thr_fill_none_inf_axis1": dict(axis=1, threshold=1e-3),
        "fill_true_inf_axis0": dict(axis=0, fill=True),
    }
    for name, kw in cases.items():
        assert_peak_close(F.normalize(x, **kw), fixture_array(f[name]), RTOL_NORMALIZE, name)
    np.testing.assert_array_equal(F.normalize(x, norm=None), x)
    with pytest.raises(ValueError, match="axis"):
        F.normalize(x, axis=2)
    with pytest.raises(ValueError, match="fill"):
        F.normalize(x, norm=0, fill=True)


def test_conversions(fx):
    f = fx["conversions"]
    freqs, octs, a4, tunings = (fixture_array(f[k]) for k in ("freqs", "octs", "a4", "tunings"))
    assert_peak_close(U.hz_to_octs(freqs), fixture_array(f["hz_to_octs"]), RTOL_CONV, "hz_to_octs")
    assert_peak_close(
        U.hz_to_octs(freqs, tuning=0.3, bins_per_octave=24), fixture_array(f["hz_to_octs_t03_b24"]), RTOL_CONV, "h2o"
    )
    assert_peak_close(U.octs_to_hz(octs), fixture_array(f["octs_to_hz"]), RTOL_CONV, "octs_to_hz")
    assert_peak_close(
        U.octs_to_hz(octs, tuning=-0.2, bins_per_octave=36), fixture_array(f["octs_to_hz_tm02_b36"]), RTOL_CONV, "o2h"
    )
    assert_peak_close(U.a4_to_tuning(a4), fixture_array(f["a4_to_tuning"]), RTOL_CONV, "a4_to_tuning")
    assert_peak_close(U.a4_to_tuning(a4, bins_per_octave=36), fixture_array(f["a4_to_tuning_b36"]), RTOL_CONV, "a4")
    assert_peak_close(U.tuning_to_a4(tunings), fixture_array(f["tuning_to_a4"]), RTOL_CONV, "tuning_to_a4")
    assert_peak_close(
        U.tuning_to_a4(tunings, bins_per_octave=24), fixture_array(f["tuning_to_a4_b24"]), RTOL_CONV, "t2a"
    )
    assert_peak_close(
        U.cqt_frequencies(84, fmin=f["cqt_fmin_c1"]), fixture_array(f["cqt_default"]), RTOL_CONV, "cqt_frequencies"
    )
    assert_peak_close(
        U.cqt_frequencies(100, fmin=55.0, bins_per_octave=36, tuning=0.25),
        fixture_array(f["cqt_b36_t025"]),
        RTOL_CONV,
        "cqt_frequencies b36",
    )
    # Scalar in -> float out.
    assert isinstance(U.hz_to_octs(440.0), float) and U.hz_to_octs(440.0) == pytest.approx(4.0)
    assert isinstance(U.tuning_to_a4(0.0), float)
    with pytest.raises(ValueError, match="bins_per_octave"):
        U.hz_to_octs(440.0, bins_per_octave=0)


def test_keys(fx):
    for key, case in fx["keys"].items():
        assert U.key_to_notes(key) == case["unicode"], key
        assert U.key_to_notes(key, unicode=False) == case["ascii"], key
        assert U.key_to_notes(key, natural=True) == case["natural"], key
        d = U.key_to_degrees(key)
        assert d.dtype == np.int64
        np.testing.assert_array_equal(d, case["degrees"])
    with pytest.raises(ValueError):
        U.key_to_notes("H:maj")
    # Each call returns a new list (the names are shared, the list is not).
    a, b = U.key_to_notes("F#:maj"), U.key_to_notes("F#:maj")
    assert a == b and a is not b
    a.append("x")
    assert len(U.key_to_notes("F#:maj")) == 12


def test_shapes_and_validation(sr):
    x = _audio("notes", sr)
    st = _audio("stereo", sr)
    assert x.chroma_stft().shape[:2] == (1, 12)
    assert st.chroma_stft(n_chroma=24).shape[:2] == (2, 24)
    p, m = st.piptrack()
    assert p.shape == m.shape and p.shape[:2] == (2, 1025)
    i16 = aus.AudioSamples.new_mono((fixture_signal("notes") * 32767).astype(np.int16), sr)
    assert i16.chroma_stft().dtype == np.float32
    with pytest.raises(ValueError, match="n_chroma"):
        x.chroma_stft(n_chroma=0)
    with pytest.raises(ValueError, match="bins_per_octave"):
        x.estimate_tuning(bins_per_octave=-12)
    with pytest.raises(ValueError, match="ref"):
        x.piptrack(ref="median")
    with pytest.raises(ValueError):
        F.chroma_stft_from_spectrogram(np.ones((100, 4)), sr=sr, n_fft=2048)
    for name in ("chroma_stft", "piptrack", "estimate_tuning"):
        inspect.signature(getattr(aus.AudioSamples, name))
    assert inspect.signature(F.chroma_stft_from_spectrogram).parameters["norm"].default == np.inf


def test_against_librosa(sr):
    librosa = pytest.importorskip("librosa")
    y = fixture_signal("mix")
    x = aus.AudioSamples.new_mono(y, sr)
    assert_peak_close(
        x.chroma_stft(n_fft=1024, octwidth=None, tuning=None)[0],
        librosa.feature.chroma_stft(y=y, sr=sr, n_fft=1024, octwidth=None, dtype=np.float64),
        RTOL_CHROMA,
        "chroma_stft",
    )
    assert x.estimate_tuning(n_fft=1024, threshold=0.2) == pytest.approx(
        float(librosa.estimate_tuning(y=y, sr=sr, n_fft=1024, threshold=0.2)), abs=ATOL_TUNING
    )
    assert_peak_close(
        F.chroma_filterbank(sr=sr, n_fft=1024, n_chroma=36, base_c=False),
        librosa.filters.chroma(sr=sr, n_fft=1024, n_chroma=36, base_c=False, dtype=np.float64),
        RTOL_FILTER,
        "filters.chroma",
    )
    rng = np.random.default_rng(3)
    a = rng.normal(size=(4, 5, 6))
    assert_peak_close(F.normalize(a, norm=2, axis=-1), librosa.util.normalize(a, norm=2, axis=-1), RTOL_NORMALIZE, "n")
    assert U.key_to_notes("Eb:min") == librosa.key_to_notes("Eb:min")
