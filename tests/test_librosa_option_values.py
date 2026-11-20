"""Option values through the bindings, against the golden fixtures of ``option_values`` and
against the installed librosa directly.

The results are the same arithmetic as the reference on the same input and agree to 1e-9 of
the peak, except where a float32 reference (the frame RMS of ``trim`` and ``split``) or
another solver (``poly_features``) sets the tolerance. Random phases differ from librosa's,
so ``griffinlim_cqt`` is held to how close it comes to the target magnitude.
"""

from __future__ import annotations

import io

import librosa
import numpy as np
import pytest
import scipy.sparse

import audio_samples as aus
from audio_samples import features, rhythm, sequence
from audio_samples import effects as E
from conftest import load_reference_fixture

fixture = load_reference_fixture("option_values")
SR = fixture["sr"]


def _close(a, b, rtol=1e-9):
    a, b = np.asarray(a), np.asarray(b)
    assert a.shape == b.shape
    assert np.abs(a - b).max() <= rtol * max(np.abs(b).max(), 1e-300)


def _c(v):
    return np.asarray(v["re"]) + 1j * np.asarray(v["im"])


def _y():
    return np.asarray(fixture["y"])


# magphase --------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
@pytest.mark.parametrize("power", [0.5, 1.5, 0.25, -0.5, 2.0, 3, 1, 0.0])
def test_magphase_power_matches_librosa(dtype, power):
    D = librosa.stft(_y(), n_fft=256, hop_length=128).astype(dtype)
    D[3, 2] = 0.0
    with np.errstate(divide="ignore"):
        lm, lp = librosa.magphase(D, power=power)
    m, p = aus.AudioSamples.magphase(D, power=power)
    assert m.dtype == lm.dtype and p.dtype == lp.dtype
    finite = np.isfinite(lm)
    np.testing.assert_array_equal(np.isfinite(m), finite)
    _close(m[finite], lm[finite], 1e-5 if dtype == np.complex64 else 1e-12)
    _close(p, lp, 1e-5 if dtype == np.complex64 else 1e-12)


def test_magphase_fixture():
    f = fixture["magphase"]
    D = _c(f["D"])
    for name, power in (("mag_0_5", 0.5), ("mag_1_5", 1.5), ("mag_0_25", 0.25)):
        _close(aus.AudioSamples.magphase(D, power=power)[0], f[name], 1e-13)
    _close(aus.AudioSamples.magphase(D, power=1.5)[1], _c(f["phase"]), 1e-13)


# griffinlim and griffinlim_cqt ------------------------------------------------


def _cqt_mag():
    f = fixture["griffinlim_cqt_random"]
    return np.abs(librosa.cqt(_y(), sr=SR, n_bins=f["n_bins"], tuning=0.0)), f


def _convergence(y, mag, n_bins):
    rec = np.abs(librosa.cqt(np.asarray(y), sr=SR, n_bins=n_bins, tuning=0.0))
    return np.linalg.norm(rec - mag) / np.linalg.norm(mag)


def _gl_cqt(mag, f, **kw):
    out = aus.AudioSamples.griffinlim_cqt(mag, sample_rate=SR, n_iter=f["n_iter"], length=_y().size, init="random", **kw)
    return np.squeeze(out.to_numpy())


def test_griffinlim_cqt_random_converges_like_librosa():
    mag, f = _cqt_mag()
    y = _gl_cqt(mag, f, rng=0)
    assert _convergence(y, mag, f["n_bins"]) < 1.25 * f["spectral_convergence"]


def test_griffinlim_cqt_random_seeds():
    mag, f = _cqt_mag()
    np.testing.assert_array_equal(_gl_cqt(mag, f, rng=0), _gl_cqt(mag, f))
    assert not np.array_equal(_gl_cqt(mag, f, rng=0), _gl_cqt(mag, f, rng=1))
    assert not np.array_equal(_gl_cqt(mag, f, rng=None), _gl_cqt(mag, f, rng=None))
    np.testing.assert_array_equal(_gl_cqt(mag, f, rng=3), _gl_cqt(mag, f, random_state=3))
    with pytest.raises(ValueError, match="init"):
        aus.AudioSamples.griffinlim_cqt(mag, sample_rate=SR, init="zeros")


def test_griffinlim_cqt_default_is_zero_phase():
    mag, f = _cqt_mag()
    a = aus.AudioSamples.griffinlim_cqt(mag, sample_rate=SR, n_iter=2, length=_y().size)
    b = aus.AudioSamples.griffinlim_cqt(mag, sample_rate=SR, n_iter=2, length=_y().size, init=None)
    np.testing.assert_array_equal(a.to_numpy(), b.to_numpy())
    ref = librosa.griffinlim_cqt(mag, sr=SR, n_iter=2, length=_y().size, init=None, dtype=np.float64)
    _close(np.squeeze(a.to_numpy()), ref, 1e-2)


@pytest.mark.parametrize(
    "make",
    [
        lambda: np.random.default_rng(5),
        lambda: np.random.Generator(np.random.PCG64(5)),
        lambda: np.random.RandomState(5),
        lambda: [5, 6],
        lambda: np.random.SeedSequence(5),
        lambda: np.random.PCG64(5),
    ],
    ids=["default_rng", "Generator", "RandomState", "sequence", "SeedSequence", "BitGenerator"],
)
def test_griffinlim_rng_accepts_numpy_generators(make):
    S = np.abs(librosa.stft(_y(), n_fft=256))
    a = E.griffinlim(S, n_iter=2, rng=make())
    # The same generator state gives the same seed and so the same output.
    np.testing.assert_array_equal(a, E.griffinlim(S, n_iter=2, rng=make()))
    np.testing.assert_array_equal(a, E.griffinlim(S, n_iter=2, random_state=make()))
    assert not np.array_equal(a, E.griffinlim(S, n_iter=2, rng=0))


def test_griffinlim_rng_advances_the_generator():
    S = np.abs(librosa.stft(_y(), n_fft=256))
    g = np.random.default_rng(5)
    a, b = E.griffinlim(S, n_iter=1, rng=g), E.griffinlim(S, n_iter=1, rng=g)
    assert not np.array_equal(a, b)
    mag, f = _cqt_mag()
    g = np.random.default_rng(5)
    assert not np.array_equal(_gl_cqt(mag, f, rng=g), _gl_cqt(mag, f, rng=g))
    with pytest.raises(TypeError, match="rng"):
        E.griffinlim(S, n_iter=1, rng="nonsense")
    with pytest.raises(ValueError, match="rng"):
        E.griffinlim(S, n_iter=1, rng=-1)


# Time-varying freq ------------------------------------------------------------


def test_time_varying_freq_fixture():
    f = fixture["time_varying_freq"]
    S, freq = np.asarray(f["S"]), np.asarray(f["freq"])
    _close(features.spectral_centroid_from_spectrogram(S, sr=SR, freq=freq), f["centroid"], 1e-12)
    _close(features.spectral_bandwidth_from_spectrogram(S, sr=SR, freq=freq), f["bandwidth"], 1e-12)
    _close(features.spectral_bandwidth_from_spectrogram(S, sr=SR, freq=freq, p=3, norm=False), f["bandwidth_p3_nonorm"], 1e-11)
    _close(features.spectral_rolloff_from_spectrogram(S, sr=SR, freq=freq, roll_percent=0.7), f["rolloff"], 1e-12)
    p = features.poly_features_from_spectrogram(S, sr=SR, freq=freq, order=2)
    for k in range(3):
        _close(p[k], np.asarray(f["poly2"])[k], 1e-6)


def _reassigned(dtype, fill_nan=True):
    y = _y().astype(dtype)
    freqs, _, mags = librosa.reassigned_spectrogram(y, sr=SR, n_fft=256, hop_length=128, fill_nan=fill_nan)
    return np.abs(mags), freqs


@pytest.mark.parametrize("dtype,rtol", [(np.float32, 1e-5), (np.float64, 1e-11)])
def test_time_varying_freq_matches_librosa(dtype, rtol):
    S, freq = _reassigned(dtype)
    _close(features.spectral_centroid_from_spectrogram(S, sr=SR, freq=freq), librosa.feature.spectral_centroid(S=S, freq=freq), rtol)
    _close(features.spectral_bandwidth_from_spectrogram(S, sr=SR, freq=freq), librosa.feature.spectral_bandwidth(S=S, freq=freq), rtol)
    _close(features.spectral_rolloff_from_spectrogram(S, sr=SR, freq=freq), librosa.feature.spectral_rolloff(S=S, freq=freq), rtol)
    out = features.spectral_centroid_from_spectrogram(S, sr=SR, freq=freq)
    assert out.dtype == S.dtype


def test_time_varying_freq_channels_and_nan():
    S, freq = _reassigned(np.float64)
    both = np.stack([S, S * 0.5])
    _close(features.spectral_centroid_from_spectrogram(both, sr=SR, freq=freq), librosa.feature.spectral_centroid(S=both, freq=freq), 1e-11)
    per_channel = np.stack([freq, freq * 0.5])
    _close(features.spectral_centroid_from_spectrogram(both, sr=SR, freq=per_channel), librosa.feature.spectral_centroid(S=both, freq=per_channel), 1e-11)
    _close(features.spectral_rolloff_from_spectrogram(both, sr=SR, freq=per_channel), librosa.feature.spectral_rolloff(S=both, freq=per_channel), 1e-11)
    _close(features.spectral_bandwidth_from_spectrogram(both, sr=SR, freq=per_channel), librosa.feature.spectral_bandwidth(S=both, freq=per_channel), 1e-11)
    # Reassignment leaves NaN in quiet bins.
    S, freq = _reassigned(np.float64, fill_nan=False)
    assert np.isnan(freq).any()
    np.testing.assert_array_equal(
        np.isnan(features.spectral_centroid_from_spectrogram(S, sr=SR, freq=freq)),
        np.isnan(librosa.feature.spectral_centroid(S=S, freq=freq)),
    )
    a, b = features.spectral_rolloff_from_spectrogram(S, sr=SR, freq=freq), librosa.feature.spectral_rolloff(S=S, freq=freq)
    np.testing.assert_allclose(a, b, rtol=1e-11)
    with pytest.raises(ValueError, match="freq"):
        features.spectral_centroid_from_spectrogram(S, sr=SR, freq=freq[1:])
    with pytest.raises(ValueError, match="freq"):
        features.spectral_centroid_from_spectrogram(S, sr=SR, freq=freq[None, None])


# Slices -----------------------------------------------------------------------

_BANDS = [slice(0, 6), slice(4, 20, 3), slice(20, None), slice(10, 2, -2), slice(-4, None), slice(None, None, 5)]


def _sdb():
    return np.asarray(fixture["slices"]["Sdb"])


@pytest.mark.parametrize("aggregate", [None, np.median, np.max, np.min, np.sum, lambda x, axis: np.std(x, axis=axis)])
def test_onset_strength_multi_slices_matches_librosa(aggregate):
    S = _sdb()
    kw = dict(n_fft=512, hop_length=128, channels=_BANDS, aggregate=aggregate)
    ref = librosa.onset.onset_strength_multi(S=S, **kw)
    _close(rhythm.onset_strength_multi(S=S, sr=SR, **kw), ref, 1e-12)
    stereo = np.stack([S, S[::-1]])
    _close(rhythm.onset_strength_multi(S=stereo, sr=SR, **kw), librosa.onset.onset_strength_multi(S=stereo, **kw), 1e-12)


def test_onset_strength_multi_slices_fixture():
    f = fixture["slices"]
    S, slices = _sdb(), [slice(0, 6), slice(4, 20, 3), slice(20, None), slice(10, 2, -2)]
    kw = dict(S=S, sr=SR, n_fft=512, hop_length=128, channels=slices)
    _close(rhythm.onset_strength_multi(**kw), f["onset_mean"], 1e-12)
    _close(rhythm.onset_strength_multi(aggregate=np.median, **kw), f["onset_median"], 1e-12)
    _close(rhythm.onset_strength_multi(aggregate=np.max, detrend=True, lag=2, **kw), f["onset_max_detrend"], 1e-12)


def test_onset_strength_multi_slices_on_audio():
    y = _y()
    channels = [slice(0, 32), slice(32, 96, 2), slice(96, None)]
    a = aus.AudioSamples.new_mono(y, SR)
    ref = librosa.onset.onset_strength_multi(y=y, sr=SR, channels=channels)
    _close(a.onset_strength_multi(channels=channels)[0], ref, 1e-6)
    # Band edges given as integer bounds select the same bands as the equivalent slices.
    bounds = a.onset_strength_multi(channels=[0, 32, 96, 128])
    _close(a.onset_strength_multi(channels=[slice(0, 32), slice(32, 96), slice(96, 128)]), bounds, 1e-15)


def test_slices_edge_cases():
    S = _sdb()
    kw = dict(S=S, sr=SR, n_fft=512, hop_length=128)
    with np.errstate(all="ignore"), pytest.warns(RuntimeWarning):
        ref = librosa.onset.onset_strength_multi(S=S, n_fft=512, hop_length=128, channels=[slice(5, 5)])
    np.testing.assert_array_equal(rhythm.onset_strength_multi(channels=[slice(5, 5)], **kw)[0, 3:], ref[0, 3:])
    with pytest.raises(ValueError, match="empty"):
        rhythm.onset_strength_multi(channels=[slice(5, 5)], aggregate=np.max, **kw)
    with pytest.raises(ValueError, match="channels"):
        rhythm.onset_strength_multi(channels=[slice(0, 5), 7], **kw)
    with pytest.raises(ValueError):
        rhythm.onset_strength_multi(channels=[slice(0, 5, 0)], **kw)
    with pytest.raises(ValueError, match="idx"):
        sequence.sync(S, [slice(0, 5), 7])
    with pytest.raises(ValueError, match="empty"):
        sequence.sync(S, [slice(5, 5)], aggregate=np.max)


_SYNC = [slice(0, 5), slice(3, 12, 2), slice(30, 8, -3), slice(-3, None), slice(10**6, None, 2)]


@pytest.mark.parametrize("aggregate,name", [(None, "mean"), (np.median, "median"), (np.sum, "sum")])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_sync_slices_match_librosa(aggregate, name, dtype):
    S = _sdb().astype(dtype)
    with np.errstate(all="ignore"), pytest.warns(RuntimeWarning) if name != "sum" else _no_warning():
        ref = librosa.util.sync(S, _SYNC, aggregate=aggregate)
    out = sequence.sync(S, _SYNC, aggregate=aggregate)
    assert out.dtype == ref.dtype
    assert out.shape == ref.shape
    np.testing.assert_allclose(out, ref, rtol=2e-5 if dtype == np.float32 else 1e-12, equal_nan=True)


class _no_warning:
    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


def test_sync_slices_fixture_and_callable():
    f = fixture["slices"]
    S, slices = _sdb(), [slice(0, 5), slice(3, 12, 2), slice(30, 8, -3)]
    _close(sequence.sync(S, slices), f["sync_mean"], 1e-12)
    _close(sequence.sync(S, slices, aggregate=np.median), f["sync_median"], 1e-12)
    _close(sequence.sync(S, slices, aggregate=np.max), f["sync_max"], 1e-12)
    # A callable sees the frames in the slice's order, as in librosa.
    first = lambda x, axis: np.take(x, 0, axis=axis)
    _close(sequence.sync(S, slices, aggregate=first), librosa.util.sync(S, slices, aggregate=first), 1e-12)
    _close(sequence.sync(S, tuple(slices)), librosa.util.sync(S, tuple(slices)), 1e-12)
    three = np.stack([S, S * 2])
    _close(sequence.sync(three, slices), librosa.util.sync(three, slices), 1e-12)


# Metrogram --------------------------------------------------------------------


def test_metrogram_none_fixture():
    f = fixture["metrogram"]
    tg, freqs = np.asarray(f["tg"]), np.asarray(f["freqs"])
    _close(rhythm.metrogram(tg=tg, freqs=freqs, aggregate=None), f["per_bin"], 1e-12)
    _close(rhythm.metrogram(tg=tg, freqs=freqs, factors=[0.5, 0.25], aggregate=None, fill_value=0.25), f["per_bin_factors"], 1e-12)


@pytest.mark.parametrize("dtype,rtol", [(np.float32, 1e-5), (np.float64, 1e-12)])
def test_metrogram_none_matches_librosa(dtype, rtol):
    f = fixture["metrogram"]
    tg, freqs = np.asarray(f["tg"], dtype=dtype), np.asarray(f["freqs"])
    out = rhythm.metrogram(tg=tg, freqs=freqs, aggregate=None)
    ref = librosa.feature.metrogram(tg=tg, freqs=freqs, aggregate=None)
    assert out.dtype == dtype
    _close(out, ref, rtol)
    both = np.stack([tg, tg * 0.5])
    _close(rhythm.metrogram(tg=both, freqs=freqs, aggregate=None), librosa.feature.metrogram(tg=both, freqs=freqs, aggregate=None), rtol)
    _close(out.sum(axis=-2), rhythm.metrogram(tg=tg, freqs=freqs), rtol)
    with pytest.raises(ValueError, match="aggregate"):
        rhythm.metrogram(tg=tg, freqs=freqs, aggregate=False)


# harmonic and percussive ------------------------------------------------------


def test_harmonic_mask_is_documented_as_unsupported_and_why():
    # librosa documents mask=True as returning the masks; what it returns is the inverse
    # STFT of the mask taken as a spectrogram. The masks are effects.hpss(S, mask=True).
    y = _y().astype(np.float32)
    D = librosa.stft(y)
    mask = librosa.decompose.hpss(D, mask=True)[0]
    np.testing.assert_allclose(librosa.effects.harmonic(y, mask=True), librosa.istft(mask, length=y.size), rtol=1e-6, atol=1e-6)
    a = aus.AudioSamples.new_mono(y, SR)
    with pytest.raises(ValueError, match="hpss"):
        a.harmonic(mask=True)
    with pytest.raises(ValueError, match="hpss"):
        a.percussive(mask=True)
    _close(E.hpss(np.abs(D), mask=True)[0], mask, 1e-5)


# nn_filter --------------------------------------------------------------------


def _nn():
    f = fixture["nn_filter"]
    S = np.asarray(f["S"])
    n = S.shape[1]
    rec = scipy.sparse.csc_array((np.asarray(f["values"]), np.asarray(f["indices"]), np.asarray(f["indptr"])), shape=(n, n))
    return S, rec, f


def test_nn_filter_sparse_fixture():
    S, rec, f = _nn()
    _close(E.nn_filter(S, rec=rec), f["mean"], 1e-12)
    _close(E.nn_filter(S, rec=rec, aggregate=np.median), f["median"], 1e-12)
    _close(E.nn_filter(S, rec=rec, aggregate=np.average), f["average"], 1e-12)
    _close(E.nn_filter(S, rec=rec, aggregate=np.max), f["max"], 1e-12)
    _close(E.nn_filter(S, rec=rec, aggregate=np.min), f["min"], 1e-12)
    _close(E.nn_filter(S, rec=rec, aggregate=np.sum), f["sum"], 1e-12)
    _close(E.nn_filter(S, rec=rec, aggregate=lambda x, axis: np.percentile(x, 75, axis=axis)), f["percentile"], 1e-12)


@pytest.mark.parametrize("fmt", ["csc_array", "csr_array", "coo_array", "csc_matrix", "csr_matrix", "lil_array", "dok_array"])
@pytest.mark.parametrize("aggregate", [None, np.median, np.average, np.max])
def test_nn_filter_sparse_formats_match_librosa(fmt, aggregate):
    S, rec, _ = _nn()
    r = getattr(scipy.sparse, fmt)(rec)
    _close(E.nn_filter(S, rec=r, aggregate=aggregate), librosa.decompose.nn_filter(S, rec=r, aggregate=aggregate), 1e-12)


def test_nn_filter_sparse_details():
    S, rec, _ = _nn()
    # Boolean connectivity, and float32 features with float64 values.
    connected = rec.astype(bool)
    _close(E.nn_filter(S, rec=connected), librosa.decompose.nn_filter(S, rec=connected), 1e-12)
    S32 = S.astype(np.float32)
    out = E.nn_filter(S32, rec=rec, aggregate=np.average)
    assert out.dtype == np.float32
    _close(out, librosa.decompose.nn_filter(S32, rec=rec, aggregate=np.average), 1e-4)
    # 64-bit indices.
    big = rec.copy()
    big.indices = big.indices.astype(np.int64)
    big.indptr = big.indptr.astype(np.int64)
    _close(E.nn_filter(S, rec=big), librosa.decompose.nn_filter(S, rec=rec), 1e-12)
    # An explicit zero is a neighbour.
    explicit = rec.copy()
    explicit.data[:] = 0.0
    _close(E.nn_filter(S, rec=explicit), librosa.decompose.nn_filter(S, rec=explicit), 1e-12)
    with pytest.raises(ValueError, match="sum to zero"):
        E.nn_filter(S, rec=explicit, aggregate=np.average)
    with pytest.raises(ValueError, match="rec"):
        E.nn_filter(S, rec=scipy.sparse.csc_array(np.eye(S.shape[1] + 1)))


@pytest.mark.parametrize(
    "aggregate",
    [np.max, np.min, np.sum, np.amax, lambda x, axis: np.percentile(x, 75, axis=axis), lambda x, axis: 0.5],
    ids=["max", "min", "sum", "amax", "percentile", "scalar"],
)
def test_nn_filter_any_aggregate_matches_librosa(aggregate):
    S, rec, _ = _nn()
    dense = rec.toarray()
    ref = librosa.decompose.nn_filter(S, rec=dense, aggregate=aggregate)
    _close(E.nn_filter(S, rec=dense, aggregate=aggregate), ref, 1e-12)
    # With the recurrence matrix computed from S. The neighbour search differs from
    # scikit-learn's only by round-off.
    ref = librosa.decompose.nn_filter(S, aggregate=aggregate, k=5)
    _close(E.nn_filter(S, aggregate=aggregate, k=5), ref, 1e-9)


def test_nn_filter_aggregate_errors_and_names():
    S, rec, _ = _nn()
    _close(E.nn_filter(S, rec=rec, aggregate="max"), E.nn_filter(S, rec=rec, aggregate=np.max), 0.0)

    def fails(x, axis):
        raise KeyError("from the callable")

    with pytest.raises(KeyError, match="from the callable"):
        E.nn_filter(S, rec=rec, aggregate=fails)
    with pytest.raises(ValueError, match="aggregate"):
        E.nn_filter(S, rec=rec, aggregate=lambda x, axis: x)
    with pytest.raises(TypeError, match="aggregate"):
        E.nn_filter(S, rec=rec, aggregate=3)
    with pytest.raises(ValueError, match="aggregate"):
        E.nn_filter(S, rec=rec, aggregate="mode")
    # sparse is accepted as librosa accepts it: the links are sparse either way.
    _close(E.nn_filter(S, k=5, sparse=True), E.nn_filter(S, k=5), 0.0)


# trim_db and split_db ---------------------------------------------------------


def _burst(sr=SR):
    return np.asarray(fixture["silence"]["y"])


def test_silence_callables_fixture():
    f = fixture["silence"]
    y = _burst()
    a = aus.AudioSamples.new_mono(y, SR)
    np.testing.assert_array_equal(a.trim_db(top_db=20, ref=np.median)[1], f["trim_median_ref"])
    np.testing.assert_array_equal(a.split_db(top_db=20, ref=np.median), f["split_median_ref"])
    stereo = aus.AudioSamples.new_multi(np.stack([y, y[::-1] * 0.25]), SR)
    np.testing.assert_array_equal(stereo.split_db(top_db=25, aggregate=np.median), f["split_stereo_median"])
    percentile = lambda x, axis: np.percentile(x, 30, axis=axis)
    np.testing.assert_array_equal(stereo.split_db(top_db=25, aggregate=percentile), f["split_stereo_percentile"])
    np.testing.assert_array_equal(stereo.trim_db(top_db=25, ref=np.median, aggregate=np.median)[1], f["trim_stereo_both"])


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_silence_callables_match_librosa(dtype):
    y = _burst().astype(dtype)
    ref = lambda s, axis=None, keepdims=False: np.percentile(s, 80, axis=axis, keepdims=keepdims)
    a = aus.AudioSamples.new_mono(y, SR)
    np.testing.assert_array_equal(a.split_db(top_db=15, ref=ref), librosa.effects.split(y, top_db=15, ref=ref))
    trimmed, bounds = a.trim_db(top_db=15, ref=ref)
    lt, lb = librosa.effects.trim(y, top_db=15, ref=ref)
    np.testing.assert_array_equal(bounds, lb)
    np.testing.assert_array_equal(np.squeeze(trimmed.to_numpy()), lt)
    assert trimmed.dtype == ("f32" if dtype == np.float32 else "f64")
    stereo = np.stack([y, y[::-1] * 0.25])
    s = aus.AudioSamples.new_multi(stereo, SR)
    # keepdims result, as numpy reductions give it to apply_over_axes.
    keep = lambda x, axis: np.median(x, axis=axis, keepdims=True)
    np.testing.assert_array_equal(s.split_db(top_db=25, aggregate=keep), librosa.effects.split(stereo, top_db=25, aggregate=keep))
    np.testing.assert_array_equal(s.split_db(top_db=25, ref=ref, aggregate=keep), librosa.effects.split(stereo, top_db=25, ref=ref, aggregate=keep))


def test_silence_callables_edges():
    y = _burst()
    a = aus.AudioSamples.new_mono(y, SR)
    # The callable for a reference with the same value as numpy.max.
    wrapped = lambda s, axis=None, keepdims=False: np.max(s, axis=axis, keepdims=keepdims)
    np.testing.assert_array_equal(a.split_db(top_db=30, ref=wrapped), a.split_db(top_db=30))

    def fails(x, axis=None, keepdims=False):
        raise KeyError("from the callable")

    with pytest.raises(KeyError, match="from the callable"):
        a.trim_db(top_db=30, ref=fails)
    with pytest.raises(KeyError, match="from the callable"):
        a.split_db(top_db=30, ref=fails)
    stereo = aus.AudioSamples.new_multi(np.stack([y, y]), SR)
    with pytest.raises(KeyError, match="from the callable"):
        stereo.split_db(top_db=30, aggregate=lambda x, axis: fails(x))
    with pytest.raises(ValueError, match="aggregate"):
        stereo.split_db(top_db=30, aggregate=lambda x, axis: x)
    with pytest.raises(TypeError, match="ref"):
        a.split_db(top_db=30, ref=lambda s, axis, keepdims: "loud")
    with pytest.raises(TypeError, match="aggregate"):
        stereo.split_db(top_db=30, aggregate=3)
    # The in-place form takes callables too.
    b = aus.AudioSamples.new_mono(y.copy(), SR)
    bounds = b.trim_db_in_place(top_db=20, ref=np.median)
    np.testing.assert_array_equal(bounds, a.trim_db(top_db=20, ref=np.median)[1])
    silent = aus.AudioSamples.new_mono(np.zeros(4096), SR)
    with pytest.raises(ValueError, match="silent"):
        silent.trim_db(top_db=60, ref=lambda s, axis, keepdims: 1.0)


# io ---------------------------------------------------------------------------


def _clip(tmp_path_factory, suffix, dtype, channels=1):
    y = np.sin(np.arange(40000) * 0.03) * 0.5
    data = y if channels == 1 else np.stack([y, y[::-1]])
    a = aus.AudioSamples.new_mono(data, 22050) if channels == 1 else aus.AudioSamples.new_multi(data, 22050)
    path = tmp_path_factory.mktemp("clip") / f"clip{suffix}"
    aus.io.save(path, a.to_format(getattr(aus.SampleType, dtype.upper())))
    return path


@pytest.fixture(scope="module", params=[(".wav", "i16"), (".wav", "f32"), (".flac", "i16")], ids=["wav-i16", "wav-f32", "flac-i16"])
def clip(request, tmp_path_factory):
    suffix, dtype = request.param
    return _clip(tmp_path_factory, suffix, dtype), _clip(tmp_path_factory, suffix, dtype, channels=2)


def _same(a, b):
    assert a.dtype == b.dtype
    assert a.sample_rate == b.sample_rate
    np.testing.assert_array_equal(a.to_numpy(), b.to_numpy())


def test_read_file_objects_match_the_path(clip):
    for path in clip:
        with open(path, "rb") as f:
            _same(aus.io.read(f), aus.io.read(path))
            assert not f.closed
        _same(aus.io.read(io.BytesIO(path.read_bytes())), aus.io.read(path))
        for t in (aus.SampleType.F32, aus.SampleType.F64, aus.SampleType.I16):
            _same(aus.io.read(io.BytesIO(path.read_bytes()), as_type=t), aus.io.read(path, as_type=t))


def test_read_file_object_matches_librosa(clip):
    path = clip[0]
    y, sr = librosa.load(io.BytesIO(path.read_bytes()), sr=None)
    out = aus.io.read(io.BytesIO(path.read_bytes()), as_type=aus.SampleType.F32)
    assert sr == out.sample_rate
    # 16-bit samples convert to float with this library's scale (positive values over
    # 32767), as io.read of the path does; libsndfile divides by 32768.
    np.testing.assert_allclose(np.squeeze(out.to_numpy()), y, rtol=4e-5, atol=1e-6)


def test_read_file_object_from_its_position(clip):
    path = clip[0]
    prefixed = io.BytesIO(b"xxxx" + path.read_bytes())
    prefixed.seek(4)
    _same(aus.io.read(prefixed), aus.io.read(path))


class _Chunky:
    """A file object that gives short reads and only has what the reader needs."""

    def __init__(self, data):
        self._f = io.BytesIO(data)
        self.reads = 0

    def read(self, n=-1):
        self.reads += 1
        return self._f.read(min(n, 1000))

    def seek(self, *a):
        return self._f.seek(*a)

    def tell(self):
        return self._f.tell()


def test_read_duck_typed_file_with_short_reads(clip):
    path = clip[0]
    f = _Chunky(path.read_bytes())
    _same(aus.io.read(f), aus.io.read(path))
    assert f.reads > 1


def test_file_object_errors(tmp_path):
    with pytest.raises(ValueError, match="WAV or FLAC"):
        aus.io.read(io.BytesIO(b"ID3\x03" + b"\0" * 100))
    with pytest.raises(ValueError, match="WAV or FLAC"):
        aus.io.StreamedAudioReader(io.BytesIO(b"OggS" + b"\0" * 100))
    with pytest.raises(TypeError):
        aus.io.read(3)
    with pytest.raises(TypeError):
        aus.io.StreamedAudioReader(3)

    class NoSeek:
        def read(self, n=-1):
            return b""

    with pytest.raises(TypeError, match="seek"):
        aus.io.read(NoSeek())
    text = tmp_path / "t.txt"
    text.write_text("RIFFxxxxWAVE")
    with open(text) as f, pytest.raises(TypeError, match="binary"):
        aus.io.read(f)
    with pytest.raises(FileNotFoundError):
        aus.io.read(tmp_path / "missing.wav")

    class Fails(io.BytesIO):
        def read(self, n=-1):
            raise KeyError("from the file object")

    with pytest.raises(KeyError, match="from the file object"):
        aus.io.read(Fails(b"RIFF"))


def test_streamed_reader_on_a_file_object(clip):
    for path in clip:
        expected = aus.io.read(path).to_numpy()
        with aus.io.StreamedAudioReader(io.BytesIO(path.read_bytes())) as r, aus.io.StreamedAudioReader(path) as p:
            assert (r.sample_rate, r.num_channels, r.total_frames) == (p.sample_rate, p.num_channels, p.total_frames)
            assert r.total_frames == expected.shape[-1]
            first = r.read_frames(1000)
            np.testing.assert_array_equal(first.to_numpy(), p.read_frames(1000).to_numpy())
            r.seek_to_frame(20000)
            p.seek_to_frame(20000)
            np.testing.assert_array_equal(r.read_frames(500).to_numpy(), p.read_frames(500).to_numpy())
            r.reset()
            chunks = [c.to_numpy() for c in r.frames(4096)]
            np.testing.assert_array_equal(np.concatenate(chunks, axis=-1), expected)
            assert r.read_frames(10) is None
