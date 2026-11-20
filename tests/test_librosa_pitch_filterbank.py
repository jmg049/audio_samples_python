"""``audio_samples.pitch_filterbank`` and ``spectral.mellin_transform`` against librosa
fixtures (``pitch_filterbank.json``, ``mellin.json``). Bounds mirror the Rust suite: see
``tests/analysis_reference/{pitch_filterbank,mellin}.rs`` for why each applies.
"""

from __future__ import annotations

import numpy as np
import pytest

import audio_samples as aus
from audio_samples import pitch_filterbank as P
from audio_samples import spectral as S

from conftest import assert_peak_close, load_reference_fixture


@pytest.fixture(scope="module")
def pf():
    return load_reference_fixture("pitch_filterbank")


@pytest.fixture(scope="module")
def me():
    return load_reference_fixture("mellin")


@pytest.fixture(scope="module")
def mix():
    sig = load_reference_fixture("signals")
    return np.array(sig["signals"]["mix"], dtype=np.float64) / sig["pcm_scale"]


@pytest.mark.parametrize("tuning", [0.0, 0.25])
def test_mr_frequencies(pf, tuning):
    f, r = P.mr_frequencies(tuning)
    want = pf[f"mr_{tuning}"]
    assert_peak_close(f, np.array(want["freqs"]), 1e-13, "freqs")
    assert r.tolist() == want["rates"]


@pytest.mark.parametrize(
    "key, kw",
    [
        ("ellip", {}),
        ("butter", {"ftype": "butter"}),
        ("cheby1", {"ftype": "cheby1"}),
        ("cheby2", {"ftype": "cheby2"}),
        ("q10", {"Q": 10.0, "passband_ripple": 0.5, "stopband_attenuation": 30.0}),
    ],
)
def test_designs(pf, key, kw):
    bank, rates = P.semitone_filterbank(**kw)
    assert len(bank) == 85 and rates.shape == (85,)
    for band, want in pf["bank"][key].items():
        assert_peak_close(bank[int(band)], np.array(want), 1e-9, f"{key} {band}")


def test_polynomials_and_custom_bands(pf):
    bank, _ = P.semitone_filterbank(flayout="ba")
    for band, want in pf["bank"]["ba"].items():
        b, a = bank[int(band)]
        assert_peak_close(b, np.array(want["b"]), 1e-6, band)
        assert_peak_close(a, np.array(want["a"]), 1e-6, band)
    c = pf["bank"]["custom"]
    bank, rates = P.semitone_filterbank(center_freqs=c["freqs"], sample_rates=c["rates"])
    assert rates.tolist() == c["rates"]
    for ours, want in zip(bank, c["sos"], strict=True):
        assert_peak_close(ours, np.array(want), 1e-9, "custom")
    for bad in (
        lambda: P.semitone_filterbank(center_freqs=[100.0]),
        lambda: P.semitone_filterbank(flayout="zpk"),
        lambda: P.semitone_filterbank(ftype="bessel"),
        lambda: P.semitone_filterbank(nope=1),
    ):
        with pytest.raises(ValueError):
            bad()


def _kw(kw):
    return {k: v for k, v in kw.items()}


def test_iirt(pf, mix):
    for name, c in pf["iirt"].items():
        p = P.iirt(mix, sr=22050, **_kw(c["kw"]))
        assert_peak_close(p[70:, :20], np.array(c["power"]), 1e-3 if name == "ba" else 2e-5, name)
    all_rates = P.iirt(mix, sr=22050)
    assert_peak_close(all_rates[:, ::8], np.array(pf["iirt"]["default"]["all"]), 5e-3, "all")


def test_iirt_types(mix):
    assert P.iirt(mix.astype(np.float32), sr=22050).dtype == np.float32
    assert P.iirt(np.stack([mix, mix]), sr=22050).ndim == 3
    audio = aus.AudioSamples.new_mono(mix.astype(np.float32), 22050)
    p = audio.iirt()
    assert p.dtype == np.float32 and p.shape[:2] == (1, 85)
    a16 = audio.to_format(aus.types.SampleType.I16) if hasattr(aus.types, "SampleType") else None
    if a16 is not None:
        assert a16.iirt().dtype == np.float32
    with pytest.raises(ValueError):
        P.iirt(np.array([0.0, np.nan] * 2000), sr=22050)
    with pytest.raises(ValueError):
        P.iirt(mix, sr=22050, flayout="zpk")
    with pytest.raises(ValueError):
        P.iirt(mix, sr=22050, pad_mode="nope")


def test_mellin(me):
    y = np.array(me["y"])
    for name, c in me["cases"].items():
        r = S.mellin_transform(y, **c["kw"])
        assert r.size == c["n"], name
        assert_peak_close(r[:64], _complex(c["out"]), 1e-9, name)
    r = S.mellin_transform(np.stack([y, y[::-1]]), n_fmt=128)
    assert_peak_close(r, _complex(me["stereo"]), 1e-9, "stereo")
    assert S.mellin_transform(y.astype(np.float32)).dtype == np.complex64
    for bad in (
        lambda: S.mellin_transform(y, t_min=0.0),
        lambda: S.mellin_transform(y, n_fmt=2),
        lambda: S.mellin_transform(y, over_sample=0.5),
        lambda: S.mellin_transform(y[:2]),
        lambda: S.mellin_transform(np.array([1.0, np.nan, 2.0])),
        lambda: S.mellin_transform(y, kind="nope"),
        lambda: S.mellin_transform(y, axis=3),
    ):
        with pytest.raises(ValueError):
            bad()


def _complex(v):
    return np.array(v["re"]) + 1j * np.array(v["im"])
