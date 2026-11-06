"""librosa parity of the constant-Q bindings: ``AudioSamples.cqt`` / ``vqt`` /
``pseudo_cqt`` / ``hybrid_cqt`` / ``icqt`` / ``griffinlim_cqt`` /
``chroma_cqt`` / ``chroma_cens`` / ``chroma_vqt`` / ``tonnetz`` and
``audio_samples.features`` (``cq_to_chroma``, ``wavelet``, ``wavelet_lengths``,
``chroma_cens_from_chroma``, ``tonnetz_from_chroma``).

Golden fixtures: ``constant_q.json`` (``cases/constant_q.py``). Bounds are
peak-relative and equal to the Rust reference suite's
(``tests/librosa_reference/constant_q.rs``), which documents two error sources:

1. Top octave (no resampling): a straight port; agrees with librosa's
   complex128-filter output to round-off and with stock (complex64-filter)
   librosa to ~4e-8 of the peak.
2. Lower octaves: librosa decimates with soxr (``soxr_hq``), the crate with its
   own windowed-sinc resampler. librosa's own ``soxr_hq`` vs ``soxr_vhq`` differ
   by up to 4.7e-4 of the peak; measured differences are below 5e-4 except where
   noted.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

import audio_samples as aus
from audio_samples import features as F

from conftest import (
    assert_peak_close,
    fixture_array,
    fixture_signal,
    fixture_sr,
    load_librosa_fixture,
)

RESAMPLED = 5e-4  # whole transforms through the resampling cascade (source 2)
NO_RESAMPLING = 1e-7  # vs stock librosa without resampling: complex64 filters (source 1)
TOP_OCTAVE_F64 = 1e-12  # top octave vs complex128-filter librosa: round-off
LOWER_OCTAVES = 2e-3  # each resampled octave of the default CQT (per-octave view of source 2)
# icqt: inverse per octave + upsampling by up to 64 (soxr vs ours). Interior agreement
# ~2e-7 .. 6e-5 (soxr_hq vs soxr_vhq: 6.3e-5); the abrupt signal start/end rings
# differently in the two interpolators (measured 8.7e-3 within ~1000 samples).
ICQT_WHOLE = 1.2e-2
ICQT_INTERIOR = 5e-5
ICQT_EDGE = 2048
# CENS quantises at 0.05..0.4: CQT differences ~1e-4 leave it unchanged (round-off).
CENS = 1e-12


def _cqt_bounds(name):
    # (whole matrix, middle half of the signal). The early-downsample cases decimate
    # by 32 / 16 in one step; mix's 330 Hz partial falls in the decimator's transition
    # band and its abrupt start/end leaks into the edge frames (librosa itself moves
    # by 0.33 switching to res_type="polyphase"); away from the edges 1e-5 .. 4e-4.
    return {"early_downsample": (0.3, 1e-3), "fmin65_bpo24_n48": (6e-3, 1e-3)}.get(name, (RESAMPLED, RESAMPLED))


@pytest.fixture(scope="module")
def fx():
    return load_librosa_fixture("constant_q")


@pytest.fixture(scope="module")
def sr():
    return fixture_sr()


def _audio(name, sr):
    if name == "stereo":
        data = np.stack([fixture_signal("stereo_l"), fixture_signal("stereo_r")])
        return aus.AudioSamples.new_multi(data, sr)
    return aus.AudioSamples.new_mono(fixture_signal(name), sr)


def _octave_errors(actual, expected, bpo):
    n = expected.shape[0]
    peak = np.max(np.abs(expected))
    out = []
    for i in range(-(-n // bpo)):
        hi = n - bpo * i
        lo = max(hi - bpo, 0)
        out.append(np.max(np.abs(actual[lo:hi] - expected[lo:hi])) / peak)
    return out


def test_cqt_default_per_octave(fx, sr):
    ours = _audio("mix", sr).cqt()[0]
    stock = fixture_array(fx["cqt_full_mix_default"], dtype=np.complex128)
    f64 = fixture_array(fx["cqt_f64_basis_mix_default"], dtype=np.complex128)
    assert ours.dtype == np.complex128 and ours.shape == stock.shape
    e64 = _octave_errors(ours, f64, 12)
    estock = _octave_errors(ours, stock, 12)
    assert e64[0] < TOP_OCTAVE_F64, e64
    assert estock[0] < NO_RESAMPLING, estock
    assert max(e64[1:]) < LOWER_OCTAVES, e64


def _cqt_params(params):
    kw = dict(params)
    return kw


def test_cqt_variants(fx, sr):
    step = fx["frame_step"]
    for name, case in fx["cqt"].items():
        kw = _cqt_params(case["params"])
        whole, interior = _cqt_bounds(name)
        exp = fixture_array(case["subset"], dtype=np.complex128)
        if name == "stereo_default":
            ours = _audio("stereo", sr).cqt(**kw)
            assert list(ours.shape) == case["shape"]
            assert_peak_close(ours[..., ::step], exp, whole, name)
            continue
        signal = "notes" if name.startswith("notes") else "mix"
        x = _audio(signal, sr)
        ours = x.cqt(**kw)
        assert list(ours.shape[1:]) == case["shape"], name
        sub = ours[0, :, ::step]
        assert_peak_close(sub, exp, whole, name)
        # Interior frames (middle half of the signal).
        hop = kw.get("hop_length", 512)
        n = x.samples_per_channel()
        cols = [j for j in range(exp.shape[1]) if n <= 4 * j * step * hop <= 3 * n]
        err = np.max(np.abs(sub[:, cols] - exp[:, cols])) / np.max(np.abs(exp))
        assert err <= interior, f"{name} interior {err:.3e}"


def test_vqt_variants(fx, sr):
    step = fx["frame_step"]
    x = _audio("mix", sr)
    for name, case in fx["vqt"].items():
        ours = x.vqt(**case["params"])
        assert list(ours.shape[1:]) == case["shape"]
        assert_peak_close(ours[0, :, ::step], fixture_array(case["subset"], dtype=np.complex128), RESAMPLED, name)


def test_pseudo_and_hybrid(fx, sr):
    step = fx["frame_step"]
    mix = _audio("mix", sr)
    # pseudo_cqt: one STFT, no resampling (source 1).
    p = mix.pseudo_cqt()
    assert p.dtype == np.float64
    assert_peak_close(p[0, :, ::step], fixture_array(fx["pseudo_cqt"]["default"]), NO_RESAMPLING, "pseudo")
    p = mix.pseudo_cqt(fmin=220.0, n_bins=48, scale=False)
    assert_peak_close(
        p[0, :, ::step], fixture_array(fx["pseudo_cqt"]["fmin220_scale_false"]), NO_RESAMPLING, "pseudo 220"
    )
    h = mix.hybrid_cqt()
    assert_peak_close(h[0, :, ::step], fixture_array(fx["hybrid_cqt"]["default"]), RESAMPLED, "hybrid")
    h = _audio("notes", sr).hybrid_cqt(bins_per_octave=36, n_bins=252)
    assert_peak_close(h[0, :, ::step], fixture_array(fx["hybrid_cqt"]["notes_bpo36"]), RESAMPLED, "hybrid notes")


def _check_icqt(y, expected, step, what):
    assert_peak_close(y, expected, ICQT_WHOLE, what)
    skip = ICQT_EDGE // step
    peak = np.max(np.abs(expected))
    inner = np.max(np.abs(y[skip:-skip] - expected[skip:-skip])) / peak
    assert inner <= ICQT_INTERIOR, f"{what} interior {inner:.3e}"


def test_icqt(fx, sr):
    istep = fx["icqt"]["step"]
    n = len(fixture_signal("mix"))
    # Input: librosa's own CQT (2-D, as librosa returns it), so only the inverse is compared.
    c = fixture_array(fx["cqt_full_mix_default"], dtype=np.complex128)
    y = aus.AudioSamples.icqt(c, sample_rate=sr, length=n)
    assert y.samples_per_channel() == n and y.sample_rate == sr
    _check_icqt(np.asarray(y.to_numpy())[::istep], fixture_array(fx["icqt"]["default"]), istep, "icqt")
    # Round trip through our own forward transform (scale=False), 3-D input.
    c = _audio("mix", sr).cqt(scale=False)
    y = aus.AudioSamples.icqt(c, sample_rate=sr, scale=False, length=n)
    _check_icqt(np.asarray(y.to_numpy())[::istep], fixture_array(fx["icqt"]["noscale"]), istep, "icqt noscale")


def test_griffinlim_cqt(sr):
    y = fixture_signal("notes")[: sr // 2]
    x = aus.AudioSamples.new_mono(y, sr)
    mag = np.abs(x.cqt())
    out = aus.AudioSamples.griffinlim_cqt(mag, sample_rate=sr, n_iter=4, length=len(y))
    assert out.samples_per_channel() == len(y)
    # The reconstruction's CQT magnitude approaches the target.
    rec = np.abs(out.cqt())
    assert np.max(np.abs(rec - mag)) / np.max(mag) < 0.5
    with pytest.raises(ValueError, match="init"):
        aus.AudioSamples.griffinlim_cqt(mag, sample_rate=sr, init="zeros")
    rand = aus.AudioSamples.griffinlim_cqt(mag, sample_rate=sr, n_iter=4, length=len(y), init="random")
    assert np.max(np.abs(np.abs(rand.cqt()) - mag)) / np.max(mag) < 0.5


def test_cq_to_chroma(fx):
    m = fx["cq_to_chroma"]
    cases = {
        "default_84": (84, {}),
        "bpo36_108": (108, dict(bins_per_octave=36)),
        "fmin65_bpo24_50": (50, dict(bins_per_octave=24, fmin=65.0)),
        "base_a_fmin100": (40, dict(fmin=100.0, base_c=False)),
        "n24_bpo24": (48, dict(bins_per_octave=24, n_chroma=24)),
    }
    for name, (n, kw) in cases.items():
        # Exact 0/1 matrices.
        np.testing.assert_array_equal(F.cq_to_chroma(n, **kw), fixture_array(m[name]), err_msg=name)
    assert F.cq_to_chroma(84, window=np.ones(3)).shape == (12, 84)


def test_chroma_features(fx, sr):
    notes = _audio("notes", sr)
    mix = _audio("mix", sr)
    t_notes = fx["estimated_tuning"]["notes"]
    cc, ce, cv, tz = fx["chroma_cqt"], fx["chroma_cens"], fx["chroma_vqt"], fx["tonnetz"]
    a = fixture_array
    assert_peak_close(notes.chroma_cqt(tuning=0.0)[0], a(cc["notes_tuning0"]), RESAMPLED, "chroma_cqt t0")
    # tuning=None: estimated from the signal (librosa's default; the binding's
    # default is tuning=0.0).
    assert_peak_close(notes.chroma_cqt(tuning=None)[0], a(cc["notes_tuning_none"]), RESAMPLED, "chroma_cqt est")
    np.testing.assert_array_equal(notes.chroma_cqt(), notes.chroma_cqt(tuning=0.0))
    assert_peak_close(notes.chroma_cqt(tuning=t_notes)[0], a(cc["notes_tuning_none"]), RESAMPLED, "chroma_cqt t")
    assert_peak_close(mix.chroma_cqt(tuning=0.0)[0], a(cc["mix_tuning0"]), RESAMPLED, "chroma_cqt mix")
    assert_peak_close(
        notes.chroma_cqt(tuning=0.0, cqt_mode="hybrid", norm=2, bins_per_octave=12)[0],
        a(cc["notes_hybrid_l2_bpo12"]),
        RESAMPLED,
        "chroma_cqt hybrid",
    )
    assert_peak_close(_audio("stereo", sr).chroma_cqt(tuning=0.0), a(cc["stereo_tuning0"]), RESAMPLED, "stereo")
    assert_peak_close(notes.chroma_cens(tuning=0.0)[0], a(ce["notes_tuning0"]), CENS, "cens t0")
    assert_peak_close(notes.chroma_cens(tuning=None)[0], a(ce["notes_tuning_none"]), CENS, "cens est")
    assert_peak_close(mix.chroma_cens(tuning=0.0, win_len_smooth=10)[0], a(ce["mix_tuning0_win10"]), CENS, "cens 10")
    assert_peak_close(mix.chroma_vqt()[0], a(cv["mix_equal"]), RESAMPLED, "chroma_vqt")
    assert_peak_close(notes.chroma_vqt(gamma=None)[0], a(cv["notes_gamma_none"]), RESAMPLED, "chroma_vqt gamma")
    assert_peak_close(notes.tonnetz(tuning=0.0)[0], a(tz["notes_tuning0"]), RESAMPLED, "tonnetz t0")
    assert_peak_close(notes.tonnetz(tuning=None)[0], a(tz["notes_tuning_none"]), RESAMPLED, "tonnetz est")


def test_chroma_array_forms(sr):
    notes = _audio("notes", sr)
    raw = notes.chroma_cqt(tuning=0.0, norm=None)
    # chroma_cens = CENS post-processing of the raw (norm=None, threshold=0) chroma.
    np.testing.assert_allclose(F.chroma_cens_from_chroma(raw), notes.chroma_cens(tuning=0.0), rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(F.chroma_cens_from_chroma(raw[0]), notes.chroma_cens(tuning=0.0)[0], atol=1e-15)
    chroma = notes.chroma_cqt(tuning=0.0)
    np.testing.assert_allclose(F.tonnetz_from_chroma(chroma), notes.tonnetz(tuning=0.0), atol=1e-15)
    assert F.tonnetz_from_chroma(chroma[0]).shape == (6, chroma.shape[2])


def test_wavelets(sr):
    freqs = aus.utils.cqt_frequencies(24, fmin=110.0)
    lengths, cutoff = F.wavelet_lengths(freqs=freqs, sr=sr)
    assert lengths.shape == (24,) and lengths.dtype == np.float64 and cutoff < sr / 2
    filters, l2 = F.wavelet(freqs=freqs, sr=sr)
    assert filters.dtype == np.complex128 and filters.shape[0] == 24
    assert filters.shape[1] & (filters.shape[1] - 1) == 0  # power of two (pad_fft)
    np.testing.assert_allclose(np.abs(filters).sum(axis=1), 1.0, rtol=1e-12)  # L1 norm
    np.testing.assert_array_equal(l2, lengths)
    with pytest.raises(ValueError):
        F.wavelet_lengths(freqs=[440.0], sr=sr)
    with pytest.raises(ValueError):
        F.wavelet(freqs=[440.0], sr=sr)
    # An explicit alpha gives the bandwidth, so one frequency is enough.
    one, one_cutoff = F.wavelet_lengths(freqs=[440.0], sr=sr, alpha=0.1)
    np.testing.assert_allclose(one, [10.0 * sr / 440.0], rtol=1e-12)
    assert 440.0 < one_cutoff < sr / 2
    single, single_lengths = F.wavelet(freqs=[440.0], sr=sr, alpha=[0.1])
    bank, _ = F.wavelet(freqs=[440.0, 880.0], sr=sr, alpha=0.1)
    np.testing.assert_array_equal(single_lengths, one)
    np.testing.assert_array_equal(single[0], bank[0])
    with pytest.raises(ValueError):
        F.wavelet_lengths(freqs=[], sr=sr, alpha=0.1)


def test_shapes_and_validation(sr):
    x = _audio("notes", sr)
    st = _audio("stereo", sr)
    assert x.cqt(n_bins=24, fmin=110.0).shape[:2] == (1, 24)
    assert st.cqt(n_bins=24, fmin=110.0).shape[:2] == (2, 24)
    assert st.chroma_cens(tuning=0.0).shape[:2] == (2, 12)
    assert x.tonnetz(tuning=0.0).shape[:2] == (1, 6)
    f32 = aus.AudioSamples.new_mono(fixture_signal("notes").astype(np.float32), sr)
    assert f32.cqt(n_bins=12, fmin=440.0).dtype == np.complex64
    with pytest.raises(ValueError, match="hop_length"):
        x.cqt(hop_length=0)
    with pytest.raises(ValueError, match="res_type"):
        x.cqt(res_type="kaiser_best")
    with pytest.raises(ValueError, match="intervals"):
        x.vqt(intervals="ji9")
    with pytest.raises(ValueError, match="cqt_mode"):
        x.chroma_cqt(cqt_mode="pseudo")
    with pytest.raises(ValueError, match="window"):
        x.chroma_cqt(window=np.ones(3))
    with pytest.raises(ValueError):
        x.cqt(n_bins=200)  # above Nyquist
    with pytest.raises(ValueError, match="C"):
        aus.AudioSamples.icqt(np.ones(5, dtype=complex), sample_rate=sr)
    for name in ("cqt", "vqt", "pseudo_cqt", "hybrid_cqt", "icqt", "griffinlim_cqt", "chroma_cqt",
                 "chroma_cens", "chroma_vqt", "tonnetz"):
        inspect.signature(getattr(aus.AudioSamples, name))


def test_against_librosa(sr):
    librosa = pytest.importorskip("librosa")
    y = fixture_signal("notes")
    x = aus.AudioSamples.new_mono(y, sr)
    # Top 2 octaves only (fmin=A4, 24 bins): one level of resampling at most.
    assert_peak_close(
        x.cqt(fmin=440.0, n_bins=24, bins_per_octave=12)[0],
        librosa.cqt(y, sr=sr, fmin=440.0, n_bins=24),
        RESAMPLED,
        "cqt",
    )
    assert_peak_close(
        x.pseudo_cqt(fmin=110.0, n_bins=60)[0],
        librosa.pseudo_cqt(y, sr=sr, fmin=110.0, n_bins=60),
        NO_RESAMPLING,
        "pseudo_cqt",
    )
    np.testing.assert_array_equal(F.cq_to_chroma(36, bins_per_octave=36), librosa.filters.cq_to_chroma(36, bins_per_octave=36))
    freqs = librosa.cqt_frequencies(n_bins=12, fmin=220.0)
    lengths, cutoff = F.wavelet_lengths(freqs=freqs, sr=sr)
    llen, lcut = librosa.filters.wavelet_lengths(freqs=freqs, sr=sr)
    np.testing.assert_allclose(lengths, llen, rtol=1e-12)
    assert cutoff == pytest.approx(lcut, rel=1e-12)
    one, one_cutoff = F.wavelet_lengths(freqs=[440.0], sr=sr, alpha=0.1)
    lone, lone_cutoff = librosa.filters.wavelet_lengths(freqs=np.array([440.0]), sr=sr, alpha=0.1)
    np.testing.assert_allclose(one, lone, rtol=1e-12)
    assert one_cutoff == pytest.approx(lone_cutoff, rel=1e-12)
    single, _ = F.wavelet(freqs=[440.0], sr=sr, alpha=0.1)
    lsingle, _ = librosa.filters.wavelet(freqs=np.array([440.0]), sr=sr, alpha=0.1)
    assert single.shape == lsingle.shape
    np.testing.assert_allclose(single, lsingle, rtol=0, atol=1e-6 * np.abs(lsingle).max())
