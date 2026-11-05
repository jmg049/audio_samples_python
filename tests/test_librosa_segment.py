"""``audio_samples.segment`` and ``sequence.path_to_steps`` against librosa fixtures
(``segment.json``). Bounds mirror ``tests/librosa_reference/segment.rs``: 1e-9 of the
peak for kernels and convolutions, exact for boundaries.
"""

from __future__ import annotations

import numpy as np
import pytest

from audio_samples import segment as G
from audio_samples import sequence as Q

from conftest import assert_peak_close, load_librosa_fixture


@pytest.fixture(scope="module")
def fx():
    return load_librosa_fixture("segment")


def test_diagonal_filter(fx):
    for name, c in fx["diagonal_filter"].items():
        win = c["window"] if isinstance(c["window"], str) else tuple(c["window"])
        k = G.diagonal_filter(win, c["n"], **c["kw"])
        assert_peak_close(k, np.array(c["k"]), 1e-9, name)
    with pytest.raises(ValueError):
        G.diagonal_filter("hann", 0)


def test_path_enhance(fx):
    c = fx["path_enhance"]
    r = np.array(c["R"])
    for name, case in c.items():
        if name == "R":
            continue
        kw = dict(case["kw"])
        n = kw.pop("n")
        assert_peak_close(G.path_enhance(r, n, **kw), np.array(case["out"]), 1e-9, name)
    assert G.path_enhance(r.astype(np.float32), 5).dtype == np.float32
    s = G.path_enhance(np.stack([r, r]), 5)
    np.testing.assert_array_equal(s[0], s[1])
    with pytest.raises(ValueError):
        G.path_enhance(r, 5, max_ratio=1.0, min_ratio=2.0)
    with pytest.raises(ValueError):
        G.path_enhance(r, 5, mode="nope")
    with pytest.raises(ValueError):
        G.path_enhance(r, 5, nope=1)


def test_agglomerative(fx):
    c = fx["agglomerative"]
    x = np.array(c["x"])
    for k in (1, 2, 3, 5, 12, 80):
        assert G.agglomerative(x, k).tolist() == c[str(k)], k
    assert G.agglomerative(x.T, 4, axis=0).tolist() == c["axis0"]
    assert G.agglomerative(x[0], 4).tolist() == c["one_d"]
    assert G.agglomerative(x.astype(np.float32), 3).tolist() == c["3"]
    with pytest.raises(ValueError):
        G.agglomerative(x, 0)
    with pytest.raises(ValueError):
        G.agglomerative(x, 81)
    with pytest.raises(ValueError):
        G.agglomerative(x, 2, axis=4)

    class Fake:
        labels_ = np.array([0, 0, 1, 1, 1, 2])

        def fit(self, data):
            self.shape = data.shape

    model = Fake()
    assert G.agglomerative(np.zeros((2, 6)), 3, clusterer=model).tolist() == [0, 2, 5]
    assert model.shape == (6, 2)


def test_subsegment(fx):
    x = np.array(fx["agglomerative"]["x"])
    s = fx["subsegment"]
    f = np.array(s["frames"])
    assert G.subsegment(x, f).tolist() == s["default"]
    assert G.subsegment(x, f, n_segments=2).tolist() == s["n2"]
    assert G.subsegment(x, np.array([3, 5]), n_segments=4).tolist() == s["many"]
    assert G.subsegment(x.T, f, axis=0).tolist() == s["default"]
    with pytest.raises(ValueError):
        G.subsegment(x, f, n_segments=0)


def test_timelag_filter(fx):
    scipy_ndimage = pytest.importorskip("scipy.ndimage")
    c = fx["timelag"]
    rec = np.array(c["rec"])
    med = G.timelag_filter(scipy_ndimage.median_filter)
    assert_peak_close(med(rec, size=(1, 3)), np.array(c["median"]), 1e-12, "median")
    nopad = G.timelag_filter(scipy_ndimage.gaussian_filter, pad=False)
    assert np.isclose(np.sum(nopad(rec, sigma=1.0)), c["lag_nopad_sum"])
    assert med.__wrapped__ is scipy_ndimage.median_filter

    def second(_, m):
        return m

    assert G.timelag_filter(second, index=1)(None, rec).shape == rec.shape
    with pytest.raises(ValueError):
        G.timelag_filter(second, index=5)(None, rec)


def test_path_to_steps(fx):
    c = fx["path_to_steps"]
    path = np.array(c["path"])
    assert Q.path_to_steps(path).tolist() == c["forward"]
    assert Q.path_to_steps(path, inverse=True).tolist() == c["inverse"]
    gap = np.array(c["gap_path"])
    np.testing.assert_allclose(Q.path_to_steps(gap), c["gap_forward"], atol=1e-12)
    np.testing.assert_allclose(Q.path_to_steps(gap, inverse=True), c["gap_inverse"], atol=1e-12)
    with pytest.raises(ValueError):
        Q.path_to_steps([[0, 0]])
