"""librosa parity of the rhythm bindings (``AudioSamples`` rhythm methods and
``audio_samples.rhythm``).

Golden fixtures: ``rhythm.json`` (librosa 1.0.0, generated in the audio_samples
repo by ``tests/reference/librosa/cases/rhythm.py``). The tolerances mirror the
Rust reference suite ``tests/librosa_reference/rhythm.rs``:

* onset envelopes 1e-10 peak-relative: means of positive dB differences of a
  mel power spectrogram that itself matches librosa to ~1e-12 (FFT and
  matrix-product order);
* tempograms, Fourier tempograms, PLP and the tempogram variants 1e-9: computed
  from librosa's own stored envelope, so only FFT round-off (pocketfft vs
  realfft) and libm ulps remain, amplified at most by column normalisation;
* events (peaks, onsets, beats) and tempi EXACT: decisions are taken on values
  that agree to ~1e-12 with no near-ties in these signals, and the kernels
  reproduce librosa's numba arithmetic order.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

import audio_samples as aus
from audio_samples import rhythm as R

from conftest import (
    assert_peak_close,
    fixture_array,
    fixture_signal,
    fixture_sr,
    load_librosa_fixture,
)

RTOL_ENV = 1e-10  # mel/dB round-off ~1e-13 dB inside the envelope (see module doc)
RTOL_TG = 1e-9  # FFT round-off on librosa's exact envelope, column-normalised
# hybrid_tempogram: sqrt(|F| * |A|) turns ~1e-16 FFT round-off where A == 0 in
# exact arithmetic into ~1e-7 absolute (Rust test uses rtol 1e-12 + atol 2e-7).
ATOL_HYBRID = 2e-7


@pytest.fixture(scope="module")
def fx():
    return load_librosa_fixture("rhythm")


@pytest.fixture(scope="module")
def sr():
    return fixture_sr()


def mono(name, sr):
    return aus.AudioSamples.new_mono(fixture_signal(name), sr)


def stereo(sr):
    data = np.stack([fixture_signal("stereo_l"), fixture_signal("stereo_r")])
    return aus.AudioSamples.new_multi(data, sr)


ONSET_CASES = {
    "default": {},
    "lag2": dict(lag=2),
    "max_size3": dict(max_size=3),
    "median": dict(aggregate=np.median),
    "nocenter": dict(center=False),
    "detrend": dict(detrend=True),
    "hop256_nfft1024": dict(hop_length=256, n_fft=1024),
}

SIGS = ("clicks120", "notes", "mix")


# ---------------------------------------------------------------------------
# Onset strength
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sig", SIGS)
@pytest.mark.parametrize("case", list(ONSET_CASES))
def test_onset_strength_fixture(fx, sr, sig, case):
    env = mono(sig, sr).onset_strength(**ONSET_CASES[case])
    assert env.shape[0] == 1 and env.dtype == np.float64
    assert_peak_close(env[0], fixture_array(fx["onset_strength"][sig][case]), RTOL_ENV, f"{sig} {case}")


def test_onset_strength_stereo(fx, sr):
    env = stereo(sr).onset_strength()
    assert env.shape == (2, 44)
    assert_peak_close(env, fixture_array(fx["onset_strength"]["stereo"]["default"]), RTOL_ENV)


def test_onset_strength_multi_fixture(fx, sr):
    f = fx["onset_strength_multi"]
    notes = mono("notes", sr)
    bounds = f["bounds"]
    assert_peak_close(notes.onset_strength_multi()[0], fixture_array(f["notes_default"]), RTOL_ENV)
    assert_peak_close(notes.onset_strength_multi(channels=bounds)[0], fixture_array(f["notes_bands"]), RTOL_ENV)
    messy = notes.onset_strength_multi(channels=[64, 0, 64, 200, 20], aggregate=np.max)
    assert_peak_close(messy[0], fixture_array(f["notes_bands_messy"]), RTOL_ENV)
    st = stereo(sr).onset_strength_multi(channels=bounds)
    assert_peak_close(st, fixture_array(f["stereo_bands"]), RTOL_ENV)
    full = notes.onset_strength_multi(aggregate=False)
    assert full.shape == (1, 128, 44)
    step = f["notes_no_aggregate_row_step"]
    assert_peak_close(full[0, ::step], fixture_array(f["notes_no_aggregate"]), RTOL_ENV)


def test_onset_strength_free_function_matches_method(sr):
    librosa = pytest.importorskip("librosa")
    y = fixture_signal("notes")
    S = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr, dtype=np.float64))
    a = aus.AudioSamples.new_mono(y, sr)
    for kw in ({}, dict(lag=2, max_size=3), dict(aggregate=np.median), dict(detrend=True, center=False)):
        free = R.onset_strength(S=S, **kw)
        assert free.ndim == 1
        # Same algorithm on librosa's dB spectrogram vs ours: spectrum round-off only.
        assert_peak_close(free, a.onset_strength(**kw)[0], RTOL_ENV, str(kw))
        assert_peak_close(free, librosa.onset.onset_strength(S=S, **kw), RTOL_ENV, f"librosa {kw}")
    multi = R.onset_strength_multi(S=S, channels=[0, 32, 128])
    assert multi.shape == (2, S.shape[1])
    assert_peak_close(multi, librosa.onset.onset_strength_multi(S=S, channels=[0, 32, 128]), RTOL_ENV)
    # 3-D input keeps the channel axis.
    assert R.onset_strength(S=np.stack([S, S])).shape == (2, S.shape[1])


def test_custom_aggregate_callable(sr):
    a = mono("notes", sr)
    # A Python callable goes through the binding's trampoline; np.mean natively.
    custom = a.onset_strength(aggregate=lambda x, axis=-1: np.mean(x, axis=axis))
    assert_peak_close(custom, a.onset_strength(), 1e-12)
    with pytest.raises(ZeroDivisionError):
        a.onset_strength(aggregate=lambda x, axis=-1: 1 / 0)


# ---------------------------------------------------------------------------
# Peak picking, onset detection, backtracking
# ---------------------------------------------------------------------------


def test_peak_pick_fixture(fx):
    f = fx["peak_pick"]
    x = fixture_array(f["x"])
    for name, case in f["cases"].items():
        peaks = R.peak_pick(x, **case["params"])
        assert peaks.dtype == np.int64
        np.testing.assert_array_equal(peaks, case["peaks"], err_msg=name)


@pytest.mark.parametrize("sig", SIGS)
def test_onset_detect_fixture(fx, sr, sig):
    g = fx["onset_detect"][sig]
    a = mono(sig, sr)
    env = a.onset_strength()[0]
    det = lambda **kw: R.onset_detect(onset_envelope=env, sr=sr, **kw)  # noqa: E731
    np.testing.assert_array_equal(det(), g["frames"])
    np.testing.assert_array_equal(det(backtrack=True), g["backtrack"])
    rms = fixture_array(g["rms"])
    np.testing.assert_array_equal(det(backtrack=True, energy=rms), g["backtrack_rms"])
    np.testing.assert_array_equal(det(normalize=False, delta=0.5), g["no_normalize"])
    custom = det(pre_max=3, post_max=3, pre_avg=5, post_avg=5, delta=0.2, wait=4)
    np.testing.assert_array_equal(custom, g["custom"])
    # Method: also equal to librosa's float32-mel-basis result.
    frames = a.onset_detect()
    assert frames.dtype == np.int64
    np.testing.assert_array_equal(frames, g["frames"])
    np.testing.assert_array_equal(frames, g["from_y_f32basis"])
    np.testing.assert_array_equal(a.onset_detect(units="samples"), g["samples"])
    # frames * hop / sr: one division, identical.
    np.testing.assert_array_equal(a.onset_detect(units="time"), fixture_array(g["time"]))
    np.testing.assert_array_equal(R.onset_backtrack(g["frames"], rms), g["backtrack_rms"])


def test_onset_detect_errors(sr):
    a = mono("notes", sr)
    with pytest.raises(ValueError, match="units"):
        a.onset_detect(units="beats")
    with pytest.raises(ValueError, match="sparse"):
        a.onset_detect(sparse=False)
    with pytest.raises(ValueError, match="mono"):
        stereo(sr).onset_detect()
    with pytest.raises(ValueError, match="hop_length"):
        a.onset_detect(hop_length=-1)
    assert R.onset_detect(onset_envelope=np.zeros(50)).size == 0
    assert R.onset_backtrack([], np.ones(4)).size == 0


# ---------------------------------------------------------------------------
# Tempograms
# ---------------------------------------------------------------------------


def test_tempo_frequencies_fixture(fx):
    f = fx["tempogram"]["tempo_frequencies"]
    t = R.tempo_frequencies(384, hop_length=512, sr=22050)
    assert np.isinf(t[0])
    np.testing.assert_array_equal(t[1:], fixture_array(f["t384_from1"]))
    np.testing.assert_array_equal(R.tempo_frequencies(10, hop_length=256, sr=16000)[1:], fixture_array(f["t10_hop256_sr16000_from1"]))
    np.testing.assert_array_equal(R.fourier_tempo_frequencies(sr=22050, win_length=384, hop_length=512), fixture_array(f["f384"]))
    np.testing.assert_array_equal(R.fourier_tempo_frequencies(sr=16000, win_length=63, hop_length=256), fixture_array(f["f63_hop256_sr16000"]))


@pytest.mark.parametrize("sig", ("clicks120", "notes"))
def test_tempograms_fixture(fx, sig):
    g = fx["tempogram"][sig]
    env = fixture_array(g["envelope"])
    bs, fs = g["bin_step"], g["frame_step"]
    tg = R.tempogram(onset_envelope=env)
    assert list(tg.shape) == g["tempogram_shape"]
    assert_peak_close(tg[::bs, ::fs], fixture_array(g["tempogram"]), RTOL_TG)
    small = R.tempogram(onset_envelope=env, win_length=32, center=False, norm=2.0)
    assert_peak_close(small, fixture_array(g["tempogram_w32_nocenter_l2"]), RTOL_TG)
    ham = R.tempogram(onset_envelope=env, win_length=33, window="hamming", norm=None)
    assert_peak_close(ham, fixture_array(g["tempogram_w33_hamming_nonorm"]), RTOL_TG)
    ft = R.fourier_tempogram(onset_envelope=env)
    assert ft.dtype == np.complex128 and list(ft.shape) == g["fourier_shape"]
    assert_peak_close(np.abs(ft[::bs, ::fs]), fixture_array(g["fourier_mag"]), RTOL_TG)
    assert_peak_close(ft[::24, ::8], fixture_array(g["fourier"]), RTOL_TG)
    ft40 = R.fourier_tempogram(onset_envelope=env, win_length=40, center=False)
    assert_peak_close(ft40[::4, ::4], fixture_array(g["fourier_w40_nocenter"]), RTOL_TG)


def test_tempogram_methods_stereo(fx, sr):
    f = fx["tempogram"]["stereo"]
    tg = stereo(sr).tempogram()
    assert list(tg.shape) == f["tempogram_shape"]
    # Onset envelope round-off adds ~1e-12 (Rust test: same bound).
    assert_peak_close(tg[:, ::16, ::4], fixture_array(f["tempogram"]), RTOL_TG)
    a = mono("clicks120", sr)
    g = fx["tempogram"]["clicks120"]
    assert_peak_close(a.tempogram()[0, ::8, ::4], fixture_array(g["tempogram"]), RTOL_TG)
    assert_peak_close(np.abs(a.fourier_tempogram()[0, ::8, ::4]), fixture_array(g["fourier_mag"]), RTOL_TG)


# ---------------------------------------------------------------------------
# Tempo
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sig", SIGS)
def test_tempo_fixture(fx, sr, sig):
    g = fx["tempo"][sig]
    a = mono(sig, sr)
    env = a.onset_strength()[0]
    t = lambda **kw: R.tempo(onset_envelope=env, sr=sr, **kw)  # noqa: E731
    np.testing.assert_array_equal(t(), fixture_array(g["default"]))
    np.testing.assert_array_equal(a.tempo()[0], fixture_array(g["default"]))
    np.testing.assert_array_equal(a.tempo()[0], fixture_array(g["from_y_f32basis"]))
    np.testing.assert_array_equal(t(aggregate=None), fixture_array(g["time_varying"]))
    np.testing.assert_array_equal(
        t(start_bpm=60, std_bpm=0.5, max_tempo=None, ac_size=4.0), fixture_array(g["start60_std05_nomax_ac4"])
    )
    np.testing.assert_array_equal(t(aggregate=np.median, max_tempo=200.0), fixture_array(g["median_max200"]))
    # From a tempogram (tg=): same grid and decision.
    tg = R.tempogram(onset_envelope=env, win_length=int(8.0 * sr) // 512)
    np.testing.assert_array_equal(R.tempo(tg=tg, sr=sr), fixture_array(g["default"]))


def test_tempo_stereo(fx, sr):
    t = stereo(sr).tempo()
    assert t.shape == (2, 1)
    np.testing.assert_array_equal(t, fixture_array(fx["tempo"]["stereo"]["default"]))
    np.testing.assert_array_equal(t, fixture_array(fx["tempo"]["stereo_f64"]["default"]))


def test_tempo_prior_matches_librosa(sr):
    librosa = pytest.importorskip("librosa")
    stats = pytest.importorskip("scipy.stats")
    env = mono("clicks120", sr).onset_strength()[0]
    for prior in (stats.uniform(30, 300), stats.lognorm(0.5, scale=90)):
        expected = librosa.feature.tempo(onset_envelope=env, sr=sr, prior=prior)
        np.testing.assert_array_equal(R.tempo(onset_envelope=env, sr=sr, prior=prior), expected)
    # A plain callable of the BPM is accepted as a log-density.
    flat = R.tempo(onset_envelope=env, sr=sr, prior=lambda bpm: 0.0)
    np.testing.assert_array_equal(flat, R.tempo(onset_envelope=env, sr=sr, prior=stats.uniform(0, 1e6)))


# ---------------------------------------------------------------------------
# Beat tracking
# ---------------------------------------------------------------------------


BEAT_CASES = {
    "default": {},
    "no_trim": dict(trim=False),
    "tightness400": dict(tightness=400),
    "bpm100": dict(bpm=100.0),
    "start90_tight50": dict(start_bpm=90.0, tightness=50),
}


@pytest.mark.parametrize("sig", SIGS)
def test_beat_track_fixture(fx, sr, sig):
    g = fx["beat_track"][sig]
    env = fixture_array(g["envelope"])
    for name, kw in BEAT_CASES.items():
        tempo, beats = R.beat_track(onset_envelope=env, sr=sr, **kw)
        assert beats.dtype == np.int64
        np.testing.assert_array_equal(beats, g[name]["beats"], err_msg=f"{sig} {name}")
        np.testing.assert_array_equal(np.atleast_1d(tempo), np.atleast_1d(fixture_array(g[name]["tempo"])))
    np.testing.assert_array_equal(
        R.beat_track(onset_envelope=env, sr=sr, units="time")[1], fixture_array(g["times"])
    )
    # From audio: our f64-basis envelope and librosa's float32-basis run agree.
    tempo, beats = mono(sig, sr).beat_track()
    np.testing.assert_array_equal(beats, g["default"]["beats"])
    np.testing.assert_array_equal(beats, g["default_f32basis"]["beats"])
    np.testing.assert_array_equal(np.atleast_1d(tempo), np.atleast_1d(fixture_array(g["default_f32basis"]["tempo"])))


def test_beat_track_random_and_curve(fx, sr):
    g = fx["beat_track"]["random"]
    env = fixture_array(g["envelope"])
    curve = fixture_array(g["curve"])
    tempo, beats = R.beat_track(onset_envelope=env, sr=sr)
    np.testing.assert_array_equal(beats, g["default"])
    np.testing.assert_array_equal(tempo, fixture_array(g["tempo"]))
    np.testing.assert_array_equal(R.beat_track(onset_envelope=env, sr=sr, bpm=curve)[1], g["bpm_curve"])
    np.testing.assert_array_equal(R.beat_track_with_bpm(env, curve, sr=sr), g["bpm_curve"])
    np.testing.assert_array_equal(R.beat_track_with_bpm(env, curve, sr=sr, trim=False), g["bpm_curve_no_trim"])
    np.testing.assert_array_equal(R.beat_track_with_bpm(env, 111.0, sr=sr), g["bpm_odd_fpb"])


def test_beat_track_return_conventions(sr):
    librosa = pytest.importorskip("librosa")
    y = fixture_signal("clicks120")
    env = librosa.onset.onset_strength(y=y, sr=sr, aggregate=np.median, dtype=np.float64)
    ours = R.beat_track(onset_envelope=env, sr=sr)
    theirs = librosa.beat.beat_track(onset_envelope=env, sr=sr)
    assert type(ours[0]) is type(theirs[0]) and ours[0].shape == theirs[0].shape
    t, b = R.beat_track(onset_envelope=env, sr=sr, bpm=100.0)
    assert t == 100.0
    np.testing.assert_array_equal(b, librosa.beat.beat_track(onset_envelope=env, sr=sr, bpm=100.0)[1])
    np.testing.assert_array_equal(
        R.beat_track(onset_envelope=env, sr=sr, units="samples")[1],
        librosa.beat.beat_track(onset_envelope=env, sr=sr, units="samples")[1],
    )
    silent = R.beat_track(onset_envelope=np.zeros(100), sr=sr)
    assert silent[0] == 0.0 and silent[1].size == 0
    with pytest.raises(ValueError, match="mono"):
        stereo(sr).beat_track()
    with pytest.raises(ValueError, match="units"):
        R.beat_track(onset_envelope=env, units="bars")


# ---------------------------------------------------------------------------
# PLP and tempogram variants
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sig", ("clicks120", "notes"))
def test_plp_fixture(fx, sr, sig):
    g = fx["plp"][sig]
    env = fixture_array(fx["beat_track"][sig]["envelope"])
    assert_peak_close(R.plp(onset_envelope=env, sr=sr), fixture_array(g["default"]), RTOL_TG)
    assert_peak_close(
        R.plp(onset_envelope=env, sr=sr, tempo_min=60, tempo_max=200, win_length=256),
        fixture_array(g["range_60_200_w256"]),
        RTOL_TG,
    )
    assert_peak_close(
        R.plp(onset_envelope=env, sr=sr, tempo_min=None, tempo_max=None), fixture_array(g["unbounded"]), RTOL_TG
    )
    p = mono(sig, sr).plp()
    assert p.shape == (1, env.size)
    assert_peak_close(p[0], fixture_array(g["default"]), RTOL_TG)


@pytest.mark.parametrize("sig", ("clicks120", "notes"))
def test_tempogram_variants_fixture(fx, sr, sig):
    g = fx["tempogram_variants"][sig]
    env = fixture_array(fx["tempogram"][sig]["envelope"])
    tg = R.tempogram(onset_envelope=env)
    assert_peak_close(R.tempogram_ratio(tg=tg, sr=sr), fixture_array(g["ratio"]), RTOL_TG)
    assert_peak_close(R.tempogram_ratio(onset_envelope=env, sr=sr), fixture_array(g["ratio"]), RTOL_TG)
    mean = R.tempogram_ratio(tg=tg, sr=sr, aggregate=np.mean)
    assert mean.shape == (13,)
    assert_peak_close(mean, fixture_array(g["ratio_mean"]), RTOL_TG)
    fac = R.tempogram_ratio(tg=tg, sr=sr, factors=np.array([1.0, 0.5, 2.0]), start_bpm=90.0, fill_value=-1.0)
    assert_peak_close(fac, fixture_array(g["ratio_factors"]), RTOL_TG)

    h = R.hybrid_tempogram(onset_envelope=env, sr=sr)
    assert list(h.shape) == g["hybrid_shape"]
    np.testing.assert_allclose(h[::8, ::4], fixture_array(g["hybrid"]), rtol=0, atol=ATOL_HYBRID)

    ftg = np.abs(R.fourier_tempogram(onset_envelope=env))[..., :-1]
    freqs = R.fourier_tempo_frequencies(sr=sr)
    assert_peak_close(R.metrogram(tg=ftg, freqs=freqs), fixture_array(g["metrogram"]), RTOL_TG)
    mmax = R.metrogram(tg=ftg, freqs=freqs, factors=np.array([0.5, 1.0 / 3.0]), aggregate=np.max)
    assert_peak_close(mmax, fixture_array(g["metrogram_max"]), RTOL_TG)


def test_tempogram_variant_methods(fx, sr):
    a = mono("clicks120", sr)
    g = fx["tempogram_variants"]["clicks120"]
    r = a.tempogram_ratio()
    assert r.shape == (1, 13, 130)
    # From our envelope (onset round-off ~1e-12) instead of librosa's.
    assert_peak_close(r[0], fixture_array(g["ratio"]), RTOL_TG)
    assert a.tempogram_ratio(aggregate=np.mean).shape == (1, 13, 1)
    h = a.hybrid_tempogram()
    assert list(h.shape[1:]) == g["hybrid_shape"]
    np.testing.assert_allclose(h[0, ::8, ::4], fixture_array(g["hybrid"]), rtol=0, atol=ATOL_HYBRID)


# ---------------------------------------------------------------------------
# Signatures, dtypes, validation
# ---------------------------------------------------------------------------


def test_signatures_are_introspectable():
    for f in (
        R.onset_strength, R.onset_strength_multi, R.onset_detect, R.onset_backtrack, R.peak_pick,
        R.tempo_frequencies, R.fourier_tempo_frequencies, R.tempogram, R.fourier_tempogram, R.tempo,
        R.beat_track, R.beat_track_with_bpm, R.plp, R.tempogram_ratio, R.hybrid_tempogram, R.metrogram,
    ):
        inspect.signature(f)
    for m in (
        "onset_strength", "onset_strength_multi", "onset_detect", "tempogram", "fourier_tempogram",
        "tempo", "beat_track", "plp", "tempogram_ratio", "hybrid_tempogram",
    ):
        inspect.signature(getattr(aus.AudioSamples, m))
    assert inspect.signature(R.tempogram).parameters["norm"].default == np.inf


def test_integer_and_float32_audio(sr):
    y = fixture_signal("clicks120")
    f32 = aus.AudioSamples.new_mono(y.astype(np.float32), sr)
    # float32 storage rounds the samples (~6e-8 relative): onset envelope ~1e-6.
    assert_peak_close(f32.onset_strength()[0], mono("clicks120", sr).onset_strength()[0], 1e-5)
    i16 = aus.AudioSamples.new_mono((y * 32767).astype(np.int16), sr)
    tempo, beats = i16.beat_track()
    assert beats.size > 0 and abs(tempo[0] - 120) < 5


def test_validation(sr):
    a = mono("notes", sr)
    with pytest.raises(ValueError, match="lag"):
        a.onset_strength(lag=0)
    with pytest.raises(ValueError, match="win_length"):
        a.tempogram(win_length=-3)
    with pytest.raises(ValueError, match="method"):
        R.peak_pick(np.ones(5), pre_max=1, post_max=1, pre_avg=1, post_avg=1, delta=0, wait=0, method="x")
    with pytest.raises(ValueError):
        R.tempo()
    with pytest.raises(ValueError, match="kind"):
        R.metrogram(tg=np.ones((4, 5)), freqs=np.arange(1.0, 5.0), kind="cubic")
    with pytest.raises(TypeError):
        R.tempo(onset_envelope=np.ones(100), prior=3)
    with pytest.raises(ValueError, match="channels"):
        R.onset_strength_multi(S=np.ones((8, 10)), channels=[slice(0, 4)])
