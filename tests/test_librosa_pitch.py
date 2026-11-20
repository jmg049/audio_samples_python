"""librosa parity of the pitch-tracking bindings: ``AudioSamples.yin`` /
``pyin`` and ``audio_samples.features`` (``yin``, ``pyin``, ``salience``,
``interp_harmonics``, ``f0_harmonics``).

Golden fixtures: ``pitch.json`` (``cases/pitch.py``). Tolerances are
peak-relative and equal to the Rust reference suite's
(``tests/analysis_reference/pitch.rs``): the CMND is computed as librosa does
(same FFT length and cumsum order), so only FFT round-off remains; decisions
(trough, pitch bin, voicing, Viterbi path) must match exactly.
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
    load_reference_fixture,
)

RTOL_F0 = 1e-11  # f0 from the same CMND up to FFT round-off (measured <= 3.3e-16)
RTOL_PROB = 1e-11  # Beta-prior sums: exact incomplete beta vs SciPy/Boost, dot order (measured 2.6e-15)
RTOL_INTERP = 1e-12  # identical interpolation formulas: summation order / ulps only


@pytest.fixture(scope="module")
def fx():
    return load_reference_fixture("pitch")


@pytest.fixture(scope="module")
def sr():
    return fixture_sr()


def _signal(name):
    if name == "stereo":
        return np.stack([fixture_signal("stereo_l"), fixture_signal("stereo_r")])
    return fixture_signal(name)


def _audio(name, sr):
    y = _signal(name)
    if y.ndim == 2:
        return aus.AudioSamples.new_multi(y, sr)
    return aus.AudioSamples.new_mono(y, sr)


def _nullable(v):
    """Fixture arrays where ``null`` encodes NaN."""

    def fix(x):
        if isinstance(x, list):
            return [fix(e) for e in x]
        return np.nan if x is None else x

    return np.asarray(fix(v), dtype=np.float64)


def _rows(a):
    a = np.asarray(a)
    return a[None] if a.ndim == 1 else a


def _assert_close_nan(actual, expected, rtol, what):
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected), err_msg=f"{what}: NaN pattern")
    assert_peak_close(np.nan_to_num(actual), np.nan_to_num(expected), rtol, what)


def test_yin(fx, sr):
    for name, case in fx["yin"].items():
        cfg = dict(case["config"])
        got = _audio(case["signal"], sr).yin(**cfg)
        want = _rows(fixture_array(case["f0"]))
        assert got.shape == want.shape, name
        assert got.dtype == np.float64
        assert_peak_close(got, want, RTOL_F0, f"yin {name}")
        # Array form with librosa shapes.
        arr = F.yin(_signal(case["signal"]), sr=sr, **cfg)
        np.testing.assert_array_equal(_rows(arr), got)


def test_pyin(fx, sr):
    for name, case in fx["pyin"].items():
        cfg = dict(case["config"])
        cfg["beta_parameters"] = tuple(cfg["beta_parameters"])
        x = _audio(case["signal"], sr)
        # The fixture uses fill_na=None (best guess), so every f0 is finite.
        f0, flag, prob = x.pyin(fill_na=None, **cfg)
        want_flag = _rows(np.asarray(case["voiced_flag"], dtype=bool))
        assert flag.dtype == np.bool_
        np.testing.assert_array_equal(flag, want_flag, err_msg=f"pyin {name} voiced_flag")
        assert_peak_close(f0, _rows(fixture_array(case["f0"])), RTOL_F0, f"pyin {name} f0")
        assert_peak_close(prob, _rows(fixture_array(case["voiced_prob"])), RTOL_PROB, f"pyin {name} prob")
        # Default NaN fill: voiced frames unchanged, unvoiced NaN.
        f0n, _, _ = x.pyin(**cfg)
        np.testing.assert_array_equal(f0n[flag], f0[flag])
        assert np.isnan(f0n[~flag]).all()
        # A numeric fill value.
        f0z, _, _ = x.pyin(fill_na=0.0, **cfg)
        assert (f0z[~flag] == 0.0).all()
        # Array form: librosa shapes and the same values.
        af0, aflag, aprob = F.pyin(_signal(case["signal"]), sr=sr, fill_na=None, **cfg)
        np.testing.assert_array_equal(_rows(af0), f0)
        np.testing.assert_array_equal(_rows(aflag), flag)
        np.testing.assert_array_equal(_rows(aprob), prob)


@pytest.fixture(scope="module")
def harm(fx):
    h = fx["harmonics"]
    return dict(
        S=fixture_array(h["S"]),
        freqs=fixture_array(h["freqs"]),
        freqs_tv=fixture_array(h["freqs_tv"]),
        f0=_nullable(h["f0"]),
        h=h,
    )


def test_interp_harmonics(harm):
    c = harm["h"]["interp_harmonics"]
    S, freqs, hs = harm["S"], harm["freqs"], c["harmonics"]
    out = F.interp_harmonics(S, freqs=freqs, harmonics=hs)
    assert out.shape == (len(hs),) + S.shape
    assert_peak_close(out, fixture_array(c["fixed"]), RTOL_INTERP, "fixed")
    assert_peak_close(
        F.interp_harmonics(S, freqs=freqs, harmonics=hs, fill_value=-1.0),
        fixture_array(c["fixed_fill"]),
        RTOL_INTERP,
        "fixed_fill",
    )
    assert_peak_close(
        F.interp_harmonics(S, freqs=harm["freqs_tv"], harmonics=hs),
        fixture_array(c["time_varying"]),
        RTOL_INTERP,
        "time_varying",
    )
    # 3-D input gives (channels, harmonics, F, T).
    assert F.interp_harmonics(np.stack([S, S]), freqs=freqs, harmonics=hs).shape == (2, len(hs)) + S.shape


def test_salience(harm):
    c = harm["h"]["salience"]
    S, freqs, hs, w = harm["S"], harm["freqs"], c["harmonics"], c["weights"]
    cases = {
        "default": (dict(freqs=freqs), True),
        "weighted": (dict(freqs=freqs, weights=w), True),
        "weighted_fill0": (dict(freqs=freqs, weights=w, fill_value=0.0), False),
        "max_nofilter": (dict(freqs=freqs, aggregate=np.max, filter_peaks=False), False),
        "sum_nofilter": (dict(freqs=freqs, aggregate=np.sum, filter_peaks=False), False),
        "median_nofilter": (dict(freqs=freqs, aggregate=np.median, filter_peaks=False), False),
        "min_nofilter": (dict(freqs=freqs, aggregate=np.min, filter_peaks=False), False),
        "time_varying": (dict(freqs=harm["freqs_tv"], weights=w), True),
    }
    for name, (kw, nullable) in cases.items():
        got = F.salience(S, harmonics=hs, **kw)
        want = _nullable(c[name]) if nullable else fixture_array(c[name])
        _assert_close_nan(got, want, RTOL_INTERP, f"salience {name}")
    # numpy.mean ignores the weights (unweighted average), as in librosa.
    np.testing.assert_allclose(
        F.salience(S, freqs=freqs, harmonics=hs, weights=w, aggregate=np.mean, filter_peaks=False),
        F.salience(S, freqs=freqs, harmonics=hs, filter_peaks=False),
        rtol=1e-14,
    )


def test_f0_harmonics(harm):
    c = harm["h"]["f0_harmonics"]
    S, freqs, f0, hs = harm["S"], harm["freqs"], harm["f0"], c["harmonics"]
    out = F.f0_harmonics(S, f0=f0, freqs=freqs, harmonics=hs)
    assert out.shape == (len(hs), S.shape[1])
    assert_peak_close(out, fixture_array(c["fixed"]), RTOL_INTERP, "fixed")
    assert_peak_close(
        F.f0_harmonics(S, f0=f0, freqs=freqs, harmonics=hs, fill_value=-2.0),
        fixture_array(c["fixed_fill"]),
        RTOL_INTERP,
        "fixed_fill",
    )
    assert_peak_close(
        F.f0_harmonics(S, f0=f0, freqs=harm["freqs_tv"], harmonics=hs),
        fixture_array(c["time_varying"]),
        RTOL_INTERP,
        "time_varying",
    )
    with pytest.raises(ValueError, match="f0"):
        F.f0_harmonics(S, f0=f0[None], freqs=freqs, harmonics=hs)


def test_shapes_and_validation(sr):
    x = _audio("notes", sr)
    st = _audio("stereo", sr)
    f0 = x.yin(fmin=65.0, fmax=2093.0)
    assert f0.ndim == 2 and f0.shape[0] == 1
    assert st.yin(fmin=65.0, fmax=2093.0).shape[0] == 2
    f, v, p = st.pyin(fmin=65.0, fmax=2093.0)
    assert f.shape == v.shape == p.shape and f.shape[0] == 2
    i16 = aus.AudioSamples.new_mono((fixture_signal("notes") * 32767).astype(np.int16), sr)
    assert i16.yin(fmin=65.0, fmax=2093.0).dtype == np.float32
    with pytest.raises(TypeError):
        x.yin()  # fmin / fmax are required, as in librosa
    with pytest.raises(ValueError):
        x.yin(fmin=65.0, fmax=20000.0)  # above Nyquist
    with pytest.raises(ValueError, match="frame_length"):
        x.yin(fmin=65.0, fmax=2093.0, frame_length=-1)
    with pytest.raises(ValueError, match="fill_na"):
        x.pyin(fmin=65.0, fmax=2093.0, fill_na="zero")
    with pytest.raises(ValueError, match="kind"):
        F.salience(np.ones((4, 2)), freqs=np.arange(4.0), harmonics=[1, 2], kind="cubic")
    with pytest.raises(TypeError, match="aggregate"):
        F.salience(np.ones((4, 2)), freqs=np.arange(4.0), harmonics=[1, 2], aggregate=len)
    sig = inspect.signature(aus.AudioSamples.pyin)
    assert np.isnan(sig.parameters["fill_na"].default)
    inspect.signature(F.pyin)


def test_against_librosa(sr):
    librosa = pytest.importorskip("librosa")
    y = fixture_signal("mix")
    x = aus.AudioSamples.new_mono(y, sr)
    assert_peak_close(
        x.yin(fmin=80.0, fmax=1000.0, frame_length=1024)[0],
        librosa.yin(y, fmin=80.0, fmax=1000.0, sr=sr, frame_length=1024),
        RTOL_F0,
        "yin",
    )
    f0, flag, prob = x.pyin(fmin=80.0, fmax=1000.0, frame_length=1024, fill_na=None)
    lf0, lflag, lprob = librosa.pyin(y, fmin=80.0, fmax=1000.0, sr=sr, frame_length=1024, fill_na=None)
    np.testing.assert_array_equal(flag[0], lflag)
    assert_peak_close(f0[0], lf0, RTOL_F0, "pyin f0")
    assert_peak_close(prob[0], lprob, RTOL_PROB, "pyin prob")
    S = np.abs(librosa.stft(y, n_fft=512))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=512)
    _assert_close_nan(
        F.salience(S, freqs=freqs, harmonics=[1, 2, 3]),
        librosa.salience(S, freqs=freqs, harmonics=[1, 2, 3]),
        RTOL_INTERP,
        "salience",
    )
