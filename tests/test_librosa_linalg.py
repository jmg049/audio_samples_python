"""``audio_samples.linalg``, ``inverse`` and ``factorization`` against librosa fixtures
(``linalg.json``, ``inverse.json``, ``factorization.json``). Bounds mirror the Rust suite;
see ``tests/librosa_reference/{linalg,inverse,factorization}.rs`` for why each applies.
"""

from __future__ import annotations

import numpy as np
import pytest

from audio_samples import factorization as Z
from audio_samples import inverse as I
from audio_samples import linalg as L
from audio_samples import spectral as S

from conftest import assert_peak_close, load_librosa_fixture


@pytest.fixture(scope="module")
def la():
    return load_librosa_fixture("linalg")


@pytest.fixture(scope="module")
def inv():
    return load_librosa_fixture("inverse")


@pytest.fixture(scope="module")
def nm():
    return load_librosa_fixture("factorization")


@pytest.mark.parametrize("key", ["nnls_vector", "nnls_vector_exact"])
def test_nnls_vector(la, key):
    c = la[key]
    assert_peak_close(L.nnls(np.array(c["a"]), np.array(c["b"])), np.array(c["x"]), 1e-9, key)


def test_nnls_matrix(la):
    c = la["nnls_matrix"]
    mel = S.mel_filterbank(sr=22050, n_fft=512, n_mels=32)
    b = np.array(c["b"])
    x = L.nnls(mel, b)
    assert (x >= 0).all()
    assert 0.5 * np.sum((mel @ x - b) ** 2) <= c["objective"] * 1.0001
    x3 = L.nnls(mel, np.stack([b, b]))
    assert x3.shape == (2, mel.shape[1], b.shape[1])
    np.testing.assert_allclose(x3[1], x, rtol=1e-12, atol=1e-12)
    assert L.nnls(mel.astype(np.float32), b.astype(np.float32)).dtype == np.float32
    assert L.nnls(mel, b.astype(np.float32)).dtype == np.float64


def test_decompositions(la):
    c = la["pinv"]
    assert_peak_close(L.pinv(np.array(c["a"])), np.array(c["p"]), 1e-11, "pinv")
    c = la["eigh"]
    w, v = L.eigh(np.array(c["a"]))
    assert_peak_close(w, np.array(c["w"]), 1e-13, "eigh")
    np.testing.assert_allclose(np.abs(np.sum(v * np.array(c["v"]), axis=0)), 1.0, atol=1e-11)
    c = la["svd"]
    a = np.array(c["a"])
    u, s, vt = L.svd(a)
    assert_peak_close(s, np.array(c["s"]), 1e-13, "svd")
    assert_peak_close(u * s @ vt, a, 1e-9, "svd product")
    u, s, vt = L.svd(a, k=2)
    assert s.shape == (2,) and u.shape == (15, 2) and vt.shape == (2, 7)
    assert_peak_close(s, np.array(c["s"])[:2], 1e-2, "truncated")


def test_errors():
    for call in (
        lambda: L.nnls(np.zeros((2, 2)), np.zeros(3)),
        lambda: L.nnls(np.zeros(2), np.zeros(2)),
        lambda: L.nnls(np.zeros((2, 2)), np.array([np.nan, 1.0])),
        lambda: L.pinv(np.zeros(3)),
        lambda: L.eigh(np.zeros((2, 3))),
    ):
        with pytest.raises(ValueError):
            call()


def test_mel_to_stft(inv):
    for name, c in inv["mel_to_stft"].items():
        kw = c["kw"]
        M = np.array(c["M"])
        mel_kw = {k: kw[k] for k in ("htk", "fmin", "fmax") if k in kw}
        s = I.mel_to_stft(M, sr=22050, n_fft=kw["n_fft"], power=kw["power"], **mel_kw)
        assert (s >= 0).all()
        basis = S.mel_filterbank(sr=22050, n_fft=kw["n_fft"], n_mels=kw["n_mels"], **mel_kw)
        assert 0.5 * np.sum((basis @ s ** kw["power"] - M) ** 2) <= c["objective"] * 1.001, name
        assert abs(np.sum(s**2) - c["energy"]) <= 0.15 * c["energy"], name
    assert I.mel_to_stft(M.astype(np.float32), sr=22050, n_fft=kw["n_fft"]).dtype == np.float32
    assert I.mel_to_stft(np.stack([M, M]), sr=22050, n_fft=kw["n_fft"]).ndim == 3


def test_mel_to_audio(inv):
    c = inv["mel_to_audio"]
    M = np.array(c["M"])
    kw = dict(sr=22050, n_fft=512, hop_length=128, n_iter=8, init=None, length=8192)
    y = I.mel_to_audio(M, **kw)
    assert y.shape == (8192,) and np.isfinite(y).all()
    assert I.mel_to_audio(M.astype(np.float32), **kw).dtype == np.float32
    assert I.mel_to_audio(np.stack([M, M]), **kw).shape == (2, 8192)
    a = I.mel_to_audio(M, sr=22050, n_fft=512, hop_length=128, n_iter=2, rng=3)
    b = I.mel_to_audio(M, sr=22050, n_fft=512, hop_length=128, n_iter=2, rng=3)
    assert np.array_equal(a, b)


def test_mfcc_to_mel(inv):
    c = inv["mfcc_to_mel"]
    mfcc = np.array(c["mfcc"])
    for name, case in c["cases"].items():
        m = I.mfcc_to_mel(mfcc * case["scale"], **case["kw"])
        assert_peak_close(m, np.array(case["mel"]), 1e-12, name)
    assert I.mfcc_to_mel(mfcc.astype(np.float32), n_mels=40).dtype == np.float32
    with pytest.raises(ValueError):
        I.mfcc_to_mel(mfcc, lifter=-1.0)


def test_mfcc_to_audio(inv):
    mfcc = np.array(inv["mfcc_to_mel"]["mfcc"])
    y = I.mfcc_to_audio(mfcc, sr=22050, n_mels=40, n_fft=512, hop_length=256, n_iter=2, init=None)
    assert y.ndim == 1 and np.isfinite(y).all()


def _cos(a, b):
    return np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-300)


def test_decompose(nm):
    s = np.array(nm["S"])
    for name, c in nm["cases"].items():
        kw = dict(c["kw"])
        comps, acts = Z.decompose(s, **kw)
        model = Z.Nmf(**kw)
        model.fit_transform(s.T)
        assert model.reconstruction_err_ <= c["err"] * 1.05, name
        assert np.linalg.norm(s - comps @ acts) > 0
        want = np.array(c["components"])
        used = set()
        for j in range(comps.shape[1]):
            scores = [(_cos(comps[:, j], want[:, i]), i) for i in range(want.shape[1]) if i not in used]
            score, best = max(scores)
            used.add(best)
            if name != "mu_is":
                assert score > 0.9, name


def test_decompose_options(nm):
    s = np.array(nm["S"])
    comps, acts = Z.decompose(s, n_components=3, sort=True)
    peaks = np.argmax(comps, axis=0)
    assert (np.diff(peaks) >= 0).all()
    assert np.linalg.norm(s - comps @ acts) <= nm["sorted"]["err"] * 1.05
    assert Z.decompose(s.astype(np.float32), n_components=2)[0].dtype == np.float32
    with pytest.raises(ValueError):
        Z.decompose(s, fit=False)
    with pytest.raises(ValueError):
        Z.decompose(np.stack([s, s]), n_components=2, sort=True)
    comps3, acts3 = Z.decompose(np.stack([s, s]), n_components=2)
    assert comps3.shape == (2, s.shape[0], 2) and acts3.shape == (2, s.shape[1])

    model = Z.Nmf(4)
    model.fit_transform(s.T)
    c2, a2 = Z.decompose(s, transformer=model, fit=False)
    assert c2.shape == (s.shape[0], 4) and a2.shape == (4, s.shape[1])

    class Fake:
        components_ = np.ones((2, s.shape[0]))

        def fit_transform(self, X):
            return np.ones((X.shape[0], 2))

    c3, a3 = Z.decompose(s, transformer=Fake())
    assert c3.shape == (s.shape[0], 2) and a3.shape == (2, s.shape[1])


def test_transform(nm):
    s = np.array(nm["S"])
    c = nm["transform"]
    h = np.array(c["H"])
    model = Z.Nmf(h.shape[0], tol=1e-8, max_iter=2000)
    model.fit_transform(s.T, W=np.array(c["W"]), H=h)
    w = model.transform(s.T)
    assert w.shape == np.array(c["W"]).shape
    with pytest.raises(ValueError):
        Z.Nmf(2).transform(s.T)
    with pytest.raises(ValueError):
        Z.Nmf(2).fit_transform(-s)
    with pytest.raises(ValueError):
        Z.Nmf(2, solver="cd", beta_loss=1.0)
