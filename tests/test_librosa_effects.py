"""librosa parity of the effects bindings (``AudioSamples`` effect methods and
``audio_samples.effects``).

Golden fixtures: ``effects.json`` (librosa 1.0.0, generated in the audio_samples
repo by ``tests/reference/librosa/cases/effects.py``). Tolerances are
peak-relative and mirror ``tests/librosa_reference/effects.rs``:

* phase vocoder / time stretch 1e-9: the output phase is a running sum of
  ``atan2`` differences, so STFT round-off in low-magnitude bins accumulates
  (measured ~1e-11);
* pre-/de-emphasis 1e-12: first-order recursions identical to ``lfilter``;
* HPSS components 1e-10 and masks 1e-9: exact median selection, inputs differ
  by STFT round-off;
* softmask 1e-14 (pow ulps), nn_filter 1e-12 (affinity average 1e-10), LPC 1e-9
  (bit-identical measured; margin for FMA), Griffin-Lim 1e-9 (round-off fed
  back through 8 iterations), clicks 1e-6 (librosa accumulates in float32);
* trim / split / remix exact (frame decisions and index arithmetic);
* pitch_shift: SNR > 55 dB against soxr_hq (different resampler; measured
  67-69 dB), its resampler-free time-stretch stage 1e-9.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

import audio_samples as aus
from audio_samples import effects as E

from conftest import (
    assert_peak_close,
    fixture_array,
    fixture_signal,
    fixture_sr,
    load_librosa_fixture,
)

RTOL_PV = 1e-9
RTOL_EMPH = 1e-12
RTOL_HPSS = 1e-10
RTOL_MASK = 1e-9
RTOL_SOFTMASK = 1e-14
RTOL_NN = 1e-12
RTOL_NN_AFF = 1e-10
RTOL_LPC = 1e-9
RTOL_GL = 1e-9
RTOL_CLICKS = 1e-6
MIN_SNR_PITCH = 55.0


@pytest.fixture(scope="module")
def fx():
    return load_librosa_fixture("effects")


@pytest.fixture(scope="module")
def sr():
    return fixture_sr()


def mono(name, sr):
    return aus.AudioSamples.new_mono(fixture_signal(name), sr)


def stereo_data():
    return np.stack([fixture_signal("stereo_l"), fixture_signal("stereo_r")])


def stereo(sr):
    return aus.AudioSamples.new_multi(stereo_data(), sr)


def arr(a):
    return np.asarray(a.to_numpy() if hasattr(a, "to_numpy") else a)


def subset(a, case):
    assert list(a.shape[-2:]) == case["shape"]
    return a[..., :: case["bin_step"], :: case["frame_step"]]


def snr_db(actual, ref):
    return 10 * np.log10(np.sum(ref**2) / max(np.sum((actual - ref) ** 2), 1e-300))


@pytest.fixture(scope="module")
def mix_stft(sr):
    return mono("mix", sr).stft_matrix()[0]


# ---------------------------------------------------------------------------
# Phase vocoder, time stretch, pitch shift
# ---------------------------------------------------------------------------


def test_phase_vocoder_fixture(fx, mix_stft):
    f = fx["phase_vocoder"]
    for rate in ("0.8", "1.25"):
        y = E.phase_vocoder(mix_stft, rate=float(rate))
        assert y.dtype == np.complex128
        assert_peak_close(subset(y, f[rate]), fixture_array(f[rate]["subset"]), RTOL_PV, rate)
    t_out = fixture_array(f["t_out"]["t_out"])
    y = E.phase_vocoder(mix_stft, t_out=t_out)
    assert_peak_close(subset(y, f["t_out"]), fixture_array(f["t_out"]["subset"]), RTOL_PV)
    # 3-D input keeps the channel axis.
    assert E.phase_vocoder(mix_stft[None], rate=2.0).shape[0] == 1
    with pytest.raises(ValueError):
        E.phase_vocoder(mix_stft)
    with pytest.raises(ValueError):
        E.phase_vocoder(mix_stft, rate=1.0, t_out=[0.0])


def test_time_stretch_fixture(fx, sr):
    f = fx["time_stretch"]
    k = int(fx["sig_step"])
    x = mono("mix", sr)
    for rate in ("1.5", "0.75"):
        y = x.time_stretch(float(rate))
        assert isinstance(y, aus.AudioSamples)
        assert y.samples_per_channel() == f[rate]["len"]
        assert_peak_close(arr(y)[::k], fixture_array(f[rate]["y"]), RTOL_PV, rate)
    fs = fx["time_stretch_stereo"]
    y = arr(stereo(sr).time_stretch(fs["rate"]))
    assert y.shape == (2, fs["len"])
    assert_peak_close(y[:, ::k], fixture_array(fs["y"]), RTOL_PV)
    z = mono("mix", sr)
    z.time_stretch_in_place(1.5)
    assert z.samples_per_channel() == f["1.5"]["len"]


def test_pitch_shift_fixture(fx, sr):
    f = fx["pitch_shift"]
    k = int(fx["sig_step"])
    x = mono("mix", sr)
    n = x.samples_per_channel()
    for n_steps in ("2", "-3"):
        case = f[n_steps]
        st = arr(x.time_stretch(case["rate"]))
        assert st.size == case["stretched_len"]
        assert_peak_close(st[::k], fixture_array(case["stretched"]), RTOL_PV, "stretched")
        y = arr(x.pitch_shift(float(n_steps)))
        assert y.size == n == case["len"]
        snr = snr_db(y[::k], fixture_array(case["y"]))
        assert snr > MIN_SNR_PITCH, f"pitch_shift {n_steps}: SNR {snr:.1f} dB"
    z = mono("mix", sr)
    z.pitch_shift_in_place(2, res_type="soxr_mq")
    assert z.samples_per_channel() == n
    with pytest.raises(ValueError, match="res_type"):
        x.pitch_shift(1, res_type="bogus")
    with pytest.raises(ValueError, match="scale"):
        x.pitch_shift(1, scale=True)
    y = x.pitch_shift(1, res_type=aus.types.ResamplingQuality.fast)
    assert y.samples_per_channel() == n


# ---------------------------------------------------------------------------
# Emphasis
# ---------------------------------------------------------------------------


def test_emphasis_fixture(fx, sr):
    f = fx["emphasis"]
    k = int(fx["sparse_step"])
    x = fixture_signal("mix")
    cases = [
        ("pre", E.preemphasis, dict()),
        ("pre_coef05_zi025", E.preemphasis, dict(coef=0.5, zi=[0.25])),
        ("de", E.deemphasis, dict()),
        ("de_coef08_zim01", E.deemphasis, dict(coef=0.8, zi=[-0.1])),
    ]
    for name, fn, kw in cases:
        y, zf = fn(x, return_zf=True, **kw)
        assert zf.shape == (1,)
        assert_peak_close(y[::k], fixture_array(f[name]), RTOL_EMPH, name)
        assert abs(zf[0] - f[f"{name}_zf"]) <= RTOL_EMPH * max(abs(f[f"{name}_zf"]), 1.0)
        assert_peak_close(fn(x, **kw), y, 0.0, name)
    st = stereo(sr)
    assert_peak_close(arr(st.preemphasis())[:, :50], fixture_array(f["pre_stereo_head"]), RTOL_EMPH)
    assert_peak_close(arr(st.deemphasis())[:, :50], fixture_array(f["de_stereo_head"]), RTOL_EMPH)
    y2, zf2 = E.preemphasis(stereo_data(), return_zf=True)
    assert zf2.shape == (2, 1)
    assert_peak_close(y2[:, :50], fixture_array(f["pre_stereo_head"]), RTOL_EMPH)
    assert_peak_close(arr(mono("mix", sr).preemphasis())[::k], fixture_array(f["pre"]), RTOL_EMPH)
    z = mono("mix", sr)
    z.preemphasis_in_place()
    z.deemphasis_in_place()
    assert_peak_close(arr(z), x, 1e-10)
    with pytest.raises(ValueError):
        E.deemphasis(x, coef=3.0)


def test_emphasis_blockwise_state():
    librosa = pytest.importorskip("librosa")
    x = fixture_signal("noise")[:1000]
    a, zf = E.preemphasis(x[:500], return_zf=True)
    b = E.preemphasis(x[500:], zi=zf)
    full = librosa.effects.preemphasis(x)
    np.testing.assert_allclose(np.concatenate([a, b]), full, rtol=0, atol=1e-12)


# ---------------------------------------------------------------------------
# Trim / split / remix
# ---------------------------------------------------------------------------


def test_trim_split_fixture(fx, sr):
    f = fx["trim_split"]
    burst = mono("burst", sr)
    for name in ("default", "top_db30", "frame1024_hop256", "ref_abs"):
        params = f[name].get("params", {})
        kw = {key: (int(v) if key in ("frame_length", "hop_length") else v) for key, v in params.items()}
        trimmed, idx = burst.trim_db(**kw)
        assert idx.dtype == np.int64
        np.testing.assert_array_equal(idx, f[name]["trim"], err_msg=name)
        assert trimmed.samples_per_channel() == idx[1] - idx[0]
        split = burst.split_db(**kw)
        assert split.dtype == np.int64 and split.shape[1] == 2
        np.testing.assert_array_equal(split, f[name]["split"], err_msg=name)
    np.testing.assert_array_equal(mono("mix", sr).split_db(top_db=20), f["mix_split"])
    np.testing.assert_array_equal(stereo(sr).split_db(top_db=20), f["stereo_split_top_db20"])
    z = mono("burst", sr)
    np.testing.assert_array_equal(z.trim_db_in_place(), f["default"]["trim"])
    assert z.samples_per_channel() == f["default"]["trim"][1] - f["default"]["trim"][0]


def test_trim_silence_and_aggregate(sr):
    silent = aus.AudioSamples.new_mono(np.zeros(4096), sr)
    # As librosa: with ref=max an all-zero signal is its own 0 dB reference
    # (one interval); with an absolute reference everything is silent.
    np.testing.assert_array_equal(silent.split_db(), [[0, 4096]])
    assert silent.split_db(ref=1.0).shape == (0, 2)
    with pytest.raises(ValueError, match="silent"):
        silent.trim_db(ref=1.0)
    st = stereo(sr)
    for agg in (np.max, np.min, np.mean, "mean"):
        assert st.split_db(top_db=20, aggregate=agg).ndim == 2
    with pytest.raises(TypeError):
        st.split_db(aggregate=np.median)


def test_remix_fixture(fx, sr):
    f = fx["remix"]
    k = int(fx["sig_step"])
    x = mono("mix", sr)
    for align in (True, False):
        case = f[f"align_{align}"]
        y = arr(x.remix(np.asarray(f["intervals"]), align_zeros=align))
        assert y.size == case["len"]
        np.testing.assert_array_equal(y[::k], fixture_array(case["y"]))
    z = mono("mix", sr)
    z.remix_in_place(f["intervals"], align_zeros=False)
    assert z.samples_per_channel() == f["align_False"]["len"]
    with pytest.raises(ValueError, match="intervals"):
        x.remix([1, 2, 3])


# ---------------------------------------------------------------------------
# HPSS, softmask, nn_filter
# ---------------------------------------------------------------------------

HPSS_CASES = {
    "default": {},
    "margin2": dict(margin=2.0),
    "kernel_17_9_power1": dict(kernel_size=(17, 9), power=1.0),
    "hard": dict(power=np.inf),
    "kernel_even": dict(kernel_size=(10, 12), margin=(1.0, 3.0)),
}


def test_hpss_spectrogram_fixture(fx, sr):
    f = fx["hpss_spectrogram"]
    d = mono("notes", sr).stft_matrix()[0]
    for name, kw in HPSS_CASES.items():
        case = f[name]
        h, p = E.hpss(d, **kw)
        assert h.dtype == np.complex128 and h.shape == d.shape
        assert_peak_close(subset(h, case["harm"]), fixture_array(case["harm"]["subset"]), RTOL_HPSS, name)
        assert_peak_close(subset(p, case["perc"]), fixture_array(case["perc"]["subset"]), RTOL_HPSS, name)
        mh, mp = E.hpss(d, mask=True, **kw)
        assert mh.dtype == (np.bool_ if np.isinf(kw.get("power", 2.0)) else np.float64)
        assert_peak_close(subset(mh, case["mask_harm"]).astype(np.float64), fixture_array(case["mask_harm"]["subset"]), RTOL_MASK)
        assert_peak_close(subset(mp, case["mask_perc"]).astype(np.float64), fixture_array(case["mask_perc"]["subset"]), RTOL_MASK)
    h, p = E.hpss(np.abs(d))
    assert h.dtype == np.float64
    assert_peak_close(subset(h, f["magnitude_default"]["harm"]), fixture_array(f["magnitude_default"]["harm"]["subset"]), RTOL_HPSS)
    assert_peak_close(subset(p, f["magnitude_default"]["perc"]), fixture_array(f["magnitude_default"]["perc"]["subset"]), RTOL_HPSS)
    assert E.hpss(np.abs(d)[None])[0].shape == (1,) + d.shape


def test_hpss_audio_fixture(fx, sr):
    f = fx["hpss_audio"]
    k = int(fx["sparse_step"])
    x = mono("notes", sr)
    h, p = x.harmonic_percussive()
    assert_peak_close(arr(h)[::k], fixture_array(f["harmonic"]), RTOL_HPSS)
    assert_peak_close(arr(p)[::k], fixture_array(f["percussive"]), RTOL_HPSS)
    assert_peak_close(arr(x.harmonic(margin=2.0))[::k], fixture_array(f["harmonic_margin2"]), RTOL_HPSS)
    assert_peak_close(arr(x.percussive(margin=2.0))[::k], fixture_array(f["percussive_margin2"]), RTOL_HPSS)
    z = mono("notes", sr)
    z.harmonic_in_place()
    assert_peak_close(arr(z)[::k], fixture_array(f["harmonic"]), RTOL_HPSS)
    z = mono("notes", sr)
    z.percussive_in_place()
    assert_peak_close(arr(z)[::k], fixture_array(f["percussive"]), RTOL_HPSS)
    with pytest.raises(ValueError, match="mask"):
        x.harmonic(mask=True)


def test_softmask_fixture(fx):
    f = fx["softmask"]
    x, r = fixture_array(f["x"]), fixture_array(f["x_ref"])
    assert_peak_close(E.softmask(x, r, power=1), fixture_array(f["power1"]), RTOL_SOFTMASK)
    assert_peak_close(E.softmask(x, r, power=2, split_zeros=True), fixture_array(f["power2_split"]), RTOL_SOFTMASK)
    assert_peak_close(E.softmask(x, r, power=3.5), fixture_array(f["power3_5"]), RTOL_SOFTMASK)
    hard = E.softmask(x, r, power=np.inf)
    assert hard.dtype == np.bool_
    np.testing.assert_array_equal(hard, fixture_array(f["inf"]).astype(bool))
    with pytest.raises(ValueError):
        E.softmask(-x, r)


def test_nn_filter_fixture(fx):
    f = fx["nn_filter"]
    chroma = fixture_array(f["chroma"])
    assert_peak_close(E.nn_filter(chroma), fixture_array(f["mean"]), RTOL_NN)
    assert_peak_close(E.nn_filter(chroma, aggregate=np.median), fixture_array(f["median"]), RTOL_NN)
    rec = aus.sequence.recurrence_matrix(chroma, mode="affinity", sym=True)
    assert_peak_close(E.nn_filter(chroma, rec=rec, aggregate=np.average), fixture_array(f["affinity_sym_average"]), RTOL_NN_AFF)
    # rec=None with recurrence keyword arguments.
    via_kwargs = E.nn_filter(chroma, aggregate=np.average, mode="affinity", sym=True)
    assert_peak_close(via_kwargs, fixture_array(f["affinity_sym_average"]), RTOL_NN_AFF)
    with pytest.raises(TypeError):
        E.nn_filter(chroma, bogus=1)
    with pytest.raises(TypeError):
        E.nn_filter(chroma, aggregate=np.max)


# ---------------------------------------------------------------------------
# Griffin-Lim, LPC, clicks
# ---------------------------------------------------------------------------


def test_griffinlim_fixture(fx, sr):
    f = fx["griffinlim"]
    k = int(fx["sig_step"])
    n = int(f["len"])
    x = fixture_signal("mix")[:n]
    S = np.abs(aus.spectral.stft(x, n_fft=512))
    base = dict(n_iter=int(f["n_iter"]), init=None, n_fft=512, length=n)
    assert_peak_close(E.griffinlim(S, **base)[::k], fixture_array(f["momentum099"]), RTOL_GL)
    assert_peak_close(E.griffinlim(S, momentum=0.0, **base)[::k], fixture_array(f["momentum0"]), RTOL_GL)
    nl = E.griffinlim(S, n_iter=4, init=None, n_fft=512)
    assert_peak_close(nl[::k], fixture_array(f["no_length"]), RTOL_GL)
    audio = aus.AudioSamples.griffinlim_audio(S, sample_rate=sr, **base)
    assert audio.sample_rate == sr
    assert_peak_close(arr(audio)[::k], fixture_array(f["momentum099"]), RTOL_GL)
    # Random init with a seed is reproducible (different RNG from librosa's).
    a = E.griffinlim(S, n_iter=2, n_fft=512, rng=3)
    np.testing.assert_array_equal(a, E.griffinlim(S, n_iter=2, n_fft=512, rng=3))
    with pytest.raises(TypeError):
        E.griffinlim(S, n_fft=512, rng=np.random.default_rng(0))


def test_lpc_fixture(fx, sr):
    f = fx["lpc"]
    for sig in ("mix", "noise"):
        for order in (4, 16):
            a = E.lpc(fixture_signal(sig), order=order)
            assert a.shape == (order + 1,)
            assert_peak_close(a, fixture_array(f[f"{sig}_{order}"]), RTOL_LPC)
    m = mono("mix", sr).lpc(order=4)
    assert m.shape == (1, 5)
    assert_peak_close(m[0], fixture_array(f["mix_4"]), RTOL_LPC)
    assert E.lpc(stereo_data(), order=3).shape == (2, 4)
    with pytest.raises(ValueError, match="order"):
        E.lpc(fixture_signal("mix"), order=0)


def test_clicks_fixture(fx, sr):
    f = fx["clicks"]
    k = int(fx["sig_step"])
    y = E.clicks(times=[0.1, 0.5, 0.52], sr=sr)
    assert y.size == f["times_len"]
    assert_peak_close(y[::k], fixture_array(f["times"]), RTOL_CLICKS)
    y = E.clicks(frames=[3, 20, 40], sr=sr, hop_length=256, click_freq=2000.0, click_duration=0.05, length=11000)
    assert_peak_close(y[::k], fixture_array(f["frames_len"]), RTOL_CLICKS)
    with pytest.raises(ValueError):
        E.clicks(sr=sr)


# ---------------------------------------------------------------------------
# Direct librosa comparisons, conventions
# ---------------------------------------------------------------------------


@pytest.mark.filterwarnings("ignore:The `(hop_length|n_fft)` parameter is deprecated:FutureWarning")
def test_against_librosa_directly(sr):
    librosa = pytest.importorskip("librosa")
    x = fixture_signal("notes")
    a = aus.AudioSamples.new_mono(x, sr)
    assert_peak_close(arr(a.time_stretch(1.3)), librosa.effects.time_stretch(x, rate=1.3), RTOL_PV)
    assert_peak_close(arr(a.harmonic(kernel_size=(11, 21))), librosa.effects.harmonic(x, kernel_size=(11, 21)), RTOL_HPSS)
    np.testing.assert_array_equal(a.split_db(top_db=30), librosa.effects.split(x, top_db=30))
    np.testing.assert_array_equal(a.trim_db(top_db=25)[1], librosa.effects.trim(x, top_db=25)[1])
    np.testing.assert_array_equal(
        a.trim_db(top_db=25, ref=0.1)[1], librosa.effects.trim(x, top_db=25, ref=0.1)[1]
    )
    assert_peak_close(E.lpc(x, order=8), librosa.lpc(x, order=8), RTOL_LPC)
    assert_peak_close(arr(a.deemphasis(coef=0.5)), librosa.effects.deemphasis(x, coef=0.5), RTOL_EMPH)


def test_types_and_signatures(sr):
    x = fixture_signal("mix")
    i16 = aus.AudioSamples.new_mono((x * 32767).astype(np.int16), sr)
    out = i16.time_stretch(2.0)
    assert out.dtype == i16.dtype  # input sample type kept
    f32 = aus.AudioSamples.new_mono(x.astype(np.float32), sr)
    assert f32.preemphasis().dtype == f32.dtype
    for fn in (E.phase_vocoder, E.preemphasis, E.deemphasis, E.softmask, E.hpss, E.nn_filter, E.griffinlim, E.lpc, E.clicks):
        inspect.signature(fn)
    for m in (
        "time_stretch", "time_stretch_in_place", "pitch_shift", "pitch_shift_in_place", "preemphasis",
        "preemphasis_in_place", "deemphasis", "deemphasis_in_place", "trim_db", "trim_db_in_place", "split_db",
        "remix", "remix_in_place", "harmonic", "harmonic_in_place", "percussive", "percussive_in_place",
        "harmonic_percussive", "lpc", "griffinlim_audio",
    ):
        inspect.signature(getattr(aus.AudioSamples, m))
    with pytest.raises(ValueError, match="rate"):
        mono("mix", sr).time_stretch(-1.0)
    with pytest.raises(ValueError, match="kernel_size"):
        mono("mix", sr).harmonic(kernel_size=(1, 2, 3))
