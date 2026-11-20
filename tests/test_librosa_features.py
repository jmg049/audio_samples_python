"""librosa parity of the frame-wise feature bindings (``AudioSamples.*_frames``,
``poly_features``, ``reassigned_spectrogram``) and ``audio_samples.features``
(``*_from_spectrogram``, ``pcen``, weightings, ``perceptual_weighting``).

Golden fixtures: ``features.json`` (``cases/features.py`` in the audio_samples
repo). Tolerances are peak-relative and equal to the Rust reference suite's
(``tests/analysis_reference/features.rs``); for features with several rows
(contrast bands, polynomial coefficients) each row is checked against its own
peak, as there. The algorithms follow librosa operation by operation, so the
remaining differences are FFT round-off (realfft vs pocketfft), summation
order and libm ulps.
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

# Tolerances (peak-relative), as in tests/analysis_reference/features.rs.
RTOL_RATIO = 1e-12  # ratios / sums of STFT magnitudes: FFT round-off + summation order
RTOL_FLAT = 1e-11  # exp(mean(log P)) / mean(P): log/exp ulps, amplified in near-silent frames at amin
RTOL_CONTRAST_DB = 5e-10  # 10 log10 of valleys ~1e-5 of the peak: relative FFT error ~1e-11, x4.3 in dB
RTOL_CONTRAST_LIN = 1e-12  # linear contrast: means' round-off only
RTOL_POLY = 1e-11  # least squares: LAPACK SVD vs Householder QR, both backward stable (cond ~3..20)
RTOL_RMS = 1e-13  # mean of squares then sqrt: summation order only
RTOL_RMS_S = 1e-12  # Parseval sum over STFT magnitudes: FFT round-off
RTOL_PCEN = 1e-12  # same lfilter recursion; mel input round-off and log/exp ulps
RTOL_WEIGHT = 1e-14  # same closed-form expression; log10 ulps
RTOL_PERCEPTUAL = 1e-10  # dB of FFT power: abs error ~4.3 x relative error of near-empty bins
RTOL_REASSIGN_MAG = 1e-13  # magnitudes: FFT round-off
RTOL_REASSIGN = 1e-11  # ratios S_dh / S_h near the threshold amplify round-off (measured <= 6e-13)


@pytest.fixture(scope="module")
def fx():
    return load_reference_fixture("features")


@pytest.fixture(scope="module")
def sr():
    return fixture_sr()


def _audio(name, sr):
    if name == "stereo":
        data = np.stack([fixture_signal("stereo_l"), fixture_signal("stereo_r")])
        return aus.AudioSamples.new_multi(data, sr)
    return aus.AudioSamples.new_mono(fixture_signal(name), sr)


def _as3(expected):
    e = fixture_array(expected)
    return e[None] if e.ndim == 2 else e


def _check_rows(actual, expected, rtol, what):
    """Per-row peak-relative check of (channels, rows, frames) against a 2-D/3-D fixture."""
    e = _as3(expected)
    assert actual.shape == e.shape, f"{what}: shape {actual.shape} vs {e.shape}"
    worst = 0.0
    for c in range(e.shape[0]):
        for r in range(e.shape[1]):
            worst = max(worst, assert_peak_close(actual[c, r], e[c, r], rtol, f"{what} [{c},{r}]"))
    return worst


# ---------------------------------------------------------------------------
# Golden fixtures
# ---------------------------------------------------------------------------


def test_spectral_defaults(fx, sr):
    for name, case in fx["spectral"].items():
        x = _audio(name, sr)
        _check_rows(x.spectral_centroid_frames(), case["centroid"], RTOL_RATIO, f"{name} centroid")
        _check_rows(x.spectral_bandwidth_frames(), case["bandwidth"], RTOL_RATIO, f"{name} bandwidth")
        # Bin selection: exact.
        _check_rows(x.spectral_rolloff_frames(), case["rolloff"], 0.0, f"{name} rolloff")
        _check_rows(x.spectral_flatness_frames(), case["flatness"], RTOL_FLAT, f"{name} flatness")
        _check_rows(x.spectral_contrast_frames(), case["contrast"], RTOL_CONTRAST_DB, f"{name} contrast")
        _check_rows(x.poly_features(), case["poly1"], RTOL_POLY, f"{name} poly1")
        _check_rows(x.rms_frames(), case["rms"], RTOL_RMS, f"{name} rms")
        # Integer crossing counts / frame_length: exact.
        _check_rows(x.zero_crossing_rate_frames(), case["zcr"], 0.0, f"{name} zcr")


def test_spectral_nondefault(fx, sr):
    nc = dict(n_fft=1024, hop_length=256, center=False)
    for name, case in fx["nondefault"].items():
        x = _audio(name, sr)
        _check_rows(x.spectral_centroid_frames(**nc), case["centroid_nocenter"], RTOL_RATIO, "centroid_nocenter")
        _check_rows(
            x.spectral_centroid_frames(window="hamming", pad_mode="reflect"),
            case["centroid_hamming_reflect"],
            RTOL_RATIO,
            "centroid_hamming_reflect",
        )
        _check_rows(x.spectral_bandwidth_frames(p=1), case["bandwidth_p1"], RTOL_RATIO, "bandwidth_p1")
        # p = 3: pow(x, 3) and pow(x, 1/3) libm ulps on top.
        _check_rows(
            x.spectral_bandwidth_frames(p=3, norm=False), case["bandwidth_p3_nonorm"], RTOL_RATIO, "bandwidth_p3"
        )
        # S= path with an explicit centroid (n_fft 1024, hop 256).
        s = x.spectrogram_matrix(n_fft=1024, hop_length=256)
        freqs = aus.utils.fft_frequencies(1024, float(sr))
        cen = F.spectral_centroid_from_spectrogram(s, sr=sr, freq=freqs)
        bw = F.spectral_bandwidth_from_spectrogram(s, sr=sr, freq=freqs, centroid=cen)
        _check_rows(bw, case["bandwidth_S_centroid"], RTOL_RATIO, "bandwidth_S_centroid")
        _check_rows(x.spectral_rolloff_frames(roll_percent=0.5), case["rolloff_050"], 0.0, "rolloff_050")
        _check_rows(
            x.spectral_rolloff_frames(roll_percent=0.99, **nc), case["rolloff_099_nocenter"], 0.0, "rolloff_099"
        )
        _check_rows(x.spectral_flatness_frames(power=1.0), case["flatness_power1"], RTOL_FLAT, "flatness_power1")
        _check_rows(x.spectral_flatness_frames(amin=1e-4), case["flatness_amin"], RTOL_FLAT, "flatness_amin")
        _check_rows(
            x.spectral_contrast_frames(linear=True), case["contrast_linear"], RTOL_CONTRAST_LIN, "contrast_linear"
        )
        _check_rows(
            x.spectral_contrast_frames(n_bands=4, fmin=300.0, quantile=0.1),
            case["contrast_b4_f300_q01"],
            RTOL_CONTRAST_DB,
            "contrast_b4",
        )
        _check_rows(
            x.spectral_contrast_frames(n_fft=1024, hop_length=256),
            case["contrast_nfft1024"],
            RTOL_CONTRAST_DB,
            "contrast_nfft1024",
        )
        _check_rows(x.poly_features(order=0), case["poly0"], RTOL_POLY, "poly0")
        _check_rows(x.poly_features(order=2), case["poly2"], RTOL_POLY, "poly2")
        _check_rows(
            x.rms_frames(frame_length=1024, hop_length=256, center=False), case["rms_nocenter"], RTOL_RMS, "rms_nc"
        )
        _check_rows(
            x.rms_frames(frame_length=1001, hop_length=300, pad_mode="reflect"),
            case["rms_reflect_odd"],
            RTOL_RMS,
            "rms_reflect_odd",
        )
        sd = x.spectrogram_matrix()
        _check_rows(F.rms_from_spectrogram(sd), case["rms_S"], RTOL_RMS_S, "rms_S")
        _check_rows(
            x.zero_crossing_rate_frames(frame_length=1024, hop_length=256, center=False),
            case["zcr_nocenter"],
            0.0,
            "zcr_nocenter",
        )
        _check_rows(
            x.zero_crossing_rate_frames(threshold=0.05, ref_magnitude=np.max),
            case["zcr_thresh_max"],
            0.0,
            "zcr_thresh_max",
        )
        _check_rows(
            x.zero_crossing_rate_frames(threshold=0.01, zero_pos=False, pad=True),
            case["zcr_zero_neg_pad"],
            0.0,
            "zcr_zero_neg_pad",
        )


def test_spectrogram_forms_match_methods(sr):
    """The S= functions on 2-D input reproduce the methods (librosa shapes)."""
    x = _audio("mix", sr)
    s = x.spectrogram_matrix()
    s2 = s[0]
    np.testing.assert_allclose(F.spectral_centroid_from_spectrogram(s2, sr=sr), x.spectral_centroid_frames()[0])
    np.testing.assert_array_equal(F.spectral_rolloff_from_spectrogram(s2, sr=sr), x.spectral_rolloff_frames()[0])
    np.testing.assert_allclose(F.spectral_flatness_from_spectrogram(s2), x.spectral_flatness_frames()[0])
    np.testing.assert_allclose(
        F.spectral_contrast_from_spectrogram(s2, sr=sr), x.spectral_contrast_frames()[0], rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(F.poly_features_from_spectrogram(s2, sr=sr, order=2), x.poly_features(order=2)[0])
    assert F.spectral_centroid_from_spectrogram(s, sr=sr).shape == (1, 1, s.shape[2])


def test_pcen(fx, sr):
    f = fx["pcen"]
    rs = f["row_step"]
    mel = _audio("mix", sr).melspectrogram()[0] * 2.0**31
    cases = {
        "default": {},
        "max_size3": dict(max_size=3),
        "power0": dict(power=0.0),
        "bias0": dict(bias=0.0, gain=0.8),
        "b_explicit_hop256": dict(hop_length=256, b=0.2, time_constant=0.1),
    }
    for name, kw in cases.items():
        # librosa's default hop_length is 512; the binding requires it.
        p = F.pcen(mel, sr=sr, **{"hop_length": 512, **kw})
        assert p.shape == mel.shape
        _check_rows(p[::rs][None], f[name], RTOL_PCEN, f"pcen {name}")
    mel_st = _audio("stereo", sr).melspectrogram() * 2.0**31
    _check_rows(F.pcen(mel_st, sr=sr, hop_length=512)[:, ::rs], f["stereo"], RTOL_PCEN, "pcen stereo")
    ms = f["stft_notes_mag_step"]
    mag = _audio("notes", sr).spectrogram_matrix()[0] * 2.0**31
    _check_rows(F.pcen(mag, sr=sr, hop_length=512)[::ms][None], f["stft_notes"], RTOL_PCEN, "pcen stft notes")


def test_pcen_streaming_and_options(sr):
    rng = np.random.default_rng(1)
    s = rng.uniform(0, 1e3, (16, 40))
    whole = F.pcen(s, sr=sr, hop_length=512)
    a, zf = F.pcen(s[:, :15], sr=sr, hop_length=512, return_zf=True)
    assert zf.shape == (16, 1)
    b = F.pcen(s[:, 15:], sr=sr, hop_length=512, zi=zf)
    np.testing.assert_allclose(np.concatenate([a, b], axis=1), whole, rtol=1e-12)
    # ref replaces the smoothed input
    np.testing.assert_allclose(F.pcen(s, sr=sr, hop_length=512, ref=s), whole)
    with pytest.raises(ValueError, match="axis"):
        F.pcen(s, sr=sr, hop_length=512, axis=0)
    with pytest.raises(ValueError, match="max_size"):
        F.pcen(s, sr=sr, hop_length=512, max_size=0)
    with pytest.raises(TypeError):
        F.pcen(s + 0j, sr=sr, hop_length=512)


def test_frequency_weightings(fx):
    f = fx["weighting"]
    freqs = fixture_array(f["freqs"])
    special = fixture_array(f["special"])
    for k in "ABCD":
        # The fixtures use librosa's min_db=-80; frequency_weighting defaults to None.
        assert_peak_close(F.frequency_weighting(freqs, kind=k, min_db=-80.0), fixture_array(f[k]), RTOL_WEIGHT, k)
        assert_peak_close(
            F.frequency_weighting(special[1:], kind=k, min_db=None),
            fixture_array(f[k + "_special_nomin"]),
            RTOL_WEIGHT,
            k + " nomin",
        )
        assert_peak_close(
            F.frequency_weighting(special, kind=k, min_db=-40.0),
            fixture_array(f[k + "_special_min40"]),
            RTOL_WEIGHT,
            k + " min40",
        )
        fn = getattr(F, f"{k.lower()}_weighting")
        np.testing.assert_array_equal(fn(freqs), F.frequency_weighting(freqs, kind=k, min_db=-80.0))
        np.testing.assert_array_equal(fn(freqs, min_db=None), F.frequency_weighting(freqs, kind=k))
    assert_peak_close(
        F.multi_frequency_weighting(freqs, min_db=-80.0), fixture_array(f["multi_ZAC"]), RTOL_WEIGHT, "ZAC"
    )
    assert_peak_close(
        F.multi_frequency_weighting(freqs, kinds="DBA", min_db=-60.0),
        fixture_array(f["multi_DBA_min60"]),
        RTOL_WEIGHT,
        "DBA",
    )
    np.testing.assert_array_equal(
        F.multi_frequency_weighting(freqs, kinds=["D", "B", "A"], min_db=-60.0),
        F.multi_frequency_weighting(freqs, kinds="DBA", min_db=-60.0),
    )
    # Scalar in -> float out; array shape preserved; Z is flat.
    assert isinstance(F.a_weighting(1000.0), float)
    assert F.z_weighting(np.ones((2, 3))).shape == (2, 3)
    assert F.multi_frequency_weighting(np.ones((2, 3))).shape == (3, 2, 3)
    with pytest.raises(ValueError, match="kind"):
        F.frequency_weighting(100.0, kind="Q")


def test_perceptual_weighting(fx, sr):
    f = fx["weighting"]["perceptual"]
    bs = f["bin_step"]
    freqs = aus.utils.fft_frequencies(2048, float(sr))
    p = _audio("mix", sr).spectrogram_matrix(power=2.0)[0]
    pst = _audio("stereo", sr).spectrogram_matrix(power=2.0)
    cases = {
        # librosa's default top_db is 80; the binding defaults to None.
        "A_default": (p, dict(top_db=80.0)),
        "C_ref_max": (p, dict(kind="C", ref=np.max, top_db=80.0)),
        "D_no_top_db": (p, dict(kind="D")),
        "A_stereo": (pst, dict(top_db=80.0)),
    }
    for name, (spec, kw) in cases.items():
        w = F.perceptual_weighting(spec, freqs, **kw)
        assert w.shape == spec.shape
        assert_peak_close(w[..., ::bs, :], fixture_array(f[name]), RTOL_PERCEPTUAL, name)


def test_reassigned_spectrogram(fx, sr):
    f = fx["reassigned"]
    bs, fs = f["bin_step"], f["frame_step"]
    for name, case in f.items():
        if not isinstance(case, dict):
            continue
        kw = dict(case["params"])
        if kw.get("ref_power") == "max":
            kw["ref_power"] = np.max
        signal = "notes" if ("fill_nan" in name or name.startswith("freq_only")) else "mix"
        freqs, times, mags = _audio(signal, sr).reassigned_spectrogram(**kw)
        assert list(mags.shape[1:]) == case["shape"]
        sl = (0, slice(None, None, bs), slice(None, None, fs))
        assert_peak_close(mags[sl], fixture_array(case["mags"]), RTOL_REASSIGN_MAG, f"{name} mags")
        for what, ours in (("freqs", freqs[sl]), ("times", times[sl])):
            mask = np.asarray(case[f"{what}_nan"], dtype=bool)
            # The NaN pattern is a threshold decision: identical.
            np.testing.assert_array_equal(np.isnan(ours), mask, err_msg=f"{name} {what} NaN pattern")
            assert_peak_close(np.nan_to_num(ours, nan=0.0), fixture_array(case[what]), RTOL_REASSIGN, f"{name} {what}")


# ---------------------------------------------------------------------------
# Conventions
# ---------------------------------------------------------------------------


def test_shapes_dtypes_and_inputs(sr):
    y = fixture_signal("mix")
    mono = aus.AudioSamples.new_mono(y, sr)
    st = aus.AudioSamples.new_multi(np.stack([y, y[::-1].copy()]), sr)
    for meth in (
        "spectral_centroid_frames",
        "spectral_bandwidth_frames",
        "spectral_rolloff_frames",
        "spectral_flatness_frames",
        "rms_frames",
        "zero_crossing_rate_frames",
    ):
        a = getattr(mono, meth)()
        b = getattr(st, meth)()
        assert a.dtype == np.float64 and a.ndim == 3 and a.shape[:2] == (1, 1)
        assert b.shape[:2] == (2, 1)
        np.testing.assert_allclose(b[0], a[0], rtol=1e-12, atol=1e-12)
    assert mono.spectral_contrast_frames().shape[:2] == (1, 7)
    f, t, m = st.reassigned_spectrogram()
    assert f.shape == t.shape == m.shape and f.shape[:2] == (2, 1025)
    # float32 and int16 audio are analysed in float32.
    f32 = aus.AudioSamples.new_mono(y.astype(np.float32), sr)
    assert f32.spectral_centroid_frames().dtype == np.float32
    i16 = aus.AudioSamples.new_mono((y * 32767).astype(np.int16), sr)
    assert i16.rms_frames().shape == mono.rms_frames().shape


def test_argument_validation(sr):
    x = aus.AudioSamples.new_mono(fixture_signal("mix"), sr)
    with pytest.raises(ValueError, match="n_fft"):
        x.spectral_centroid_frames(n_fft=-1)
    with pytest.raises(ValueError, match="frame_length"):
        x.rms_frames(frame_length=0)
    with pytest.raises(ValueError, match="order"):
        x.poly_features(order=-1)
    with pytest.raises(ValueError, match="pad_mode"):
        x.rms_frames(pad_mode="wrap")
    with pytest.raises(ValueError):
        x.spectral_rolloff_frames(roll_percent=1.5)
    with pytest.raises(ValueError):
        F.spectral_centroid_from_spectrogram(-np.ones((5, 3)), sr=sr)
    with pytest.raises(ValueError, match="freq"):
        F.spectral_centroid_from_spectrogram(np.ones((5, 3)), sr=sr, freq=np.ones(4))
    with pytest.raises(ValueError, match="S"):
        F.spectral_centroid_from_spectrogram(np.ones(5), sr=sr)
    with pytest.raises(TypeError):
        x.reassigned_spectrogram(ref_power=np.mean)


def test_signatures_introspectable():
    for name in (
        "spectral_centroid_frames",
        "spectral_contrast_frames",
        "poly_features",
        "rms_frames",
        "zero_crossing_rate_frames",
        "reassigned_spectrogram",
    ):
        sig = inspect.signature(getattr(aus.AudioSamples, name))
        assert all(p.kind != p.POSITIONAL_OR_KEYWORD for n, p in sig.parameters.items() if n != "self")
    assert "freq" in inspect.signature(F.spectral_centroid_from_spectrogram).parameters
    assert inspect.signature(F.pcen).parameters["return_zf"].default is False


# ---------------------------------------------------------------------------
# Direct librosa comparisons
# ---------------------------------------------------------------------------


def test_against_librosa(sr):
    librosa = pytest.importorskip("librosa")
    y = fixture_signal("notes")
    x = aus.AudioSamples.new_mono(y, sr)
    assert_peak_close(x.spectral_centroid_frames()[0], librosa.feature.spectral_centroid(y=y, sr=sr), RTOL_RATIO, "c")
    assert_peak_close(
        x.spectral_contrast_frames(n_fft=1024)[0],
        librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=1024),
        RTOL_CONTRAST_DB,
        "contrast",
    )
    assert_peak_close(x.rms_frames()[0], librosa.feature.rms(y=y, dtype=np.float64), RTOL_RMS, "rms")
    s = np.abs(librosa.stft(y))
    assert_peak_close(
        F.spectral_bandwidth_from_spectrogram(s, sr=sr, p=1.5),
        librosa.feature.spectral_bandwidth(S=s, sr=sr, p=1.5),
        RTOL_RATIO,
        "bw S",
    )
    mel = librosa.feature.melspectrogram(y=y, sr=sr, dtype=np.float64) * 2**31
    ours, zf = F.pcen(mel, sr=sr, hop_length=512, max_size=3, return_zf=True)
    ref, lzf = librosa.pcen(mel, sr=sr, max_size=3, return_zf=True)
    assert_peak_close(ours, ref, RTOL_PCEN, "pcen")
    assert_peak_close(zf, lzf, RTOL_PCEN, "pcen zf")
    assert F.a_weighting(1000.0) == pytest.approx(float(librosa.A_weighting(1000.0)), rel=1e-14)
    f, t, m = x.reassigned_spectrogram(fill_nan=True)
    lf, lt, lm = librosa.reassigned_spectrogram(y=y, sr=sr, fill_nan=True)
    assert_peak_close(f[0], lf, RTOL_REASSIGN, "reassigned freqs")
    assert_peak_close(t[0], lt, RTOL_REASSIGN, "reassigned times")
