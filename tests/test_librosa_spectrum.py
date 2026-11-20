"""librosa parity of the spectral bindings.

Two kinds of checks:

* golden fixtures (``spectrum.json``, produced by librosa 1.0.0 in the
  audio_samples repo; see ``conftest.load_librosa_fixture``), so the tests run
  without librosa installed;
* direct comparisons against librosa when it is importable.

Tolerances are peak-relative (``max|a - e| <= rtol * max|e|``) and mirror the
Rust reference suite (``tests/librosa_reference/spectrum.rs``); the reason for
each is stated next to it. The algorithms are the same as librosa/SciPy
operation by operation, so differences are round-off: libm ulps, a different FFT
(realfft vs pocketfft) and a different summation order in matrix products.
"""

from __future__ import annotations

import numpy as np
import pytest

import audio_samples as aus
from audio_samples import sequence as _seq  # noqa: F401  (import check)
from audio_samples import spectral as S
from audio_samples import utils as U

from conftest import (
    assert_peak_close,
    fixture_array,
    fixture_signal,
    fixture_signal_pcm,
    fixture_sr,
    load_librosa_fixture,
)

# Tolerances (peak-relative), with the error source each one covers.
RTOL_ULP = 1e-14  # closed-form formulas: libm cos/exp/log/pow ulps only
RTOL_FFT = 1e-12  # FFT round-off (realfft vs pocketfft): ~1e-16 x peak per bin, summed
RTOL_ISTFT = 1e-11  # inverse FFT + division by the window envelope, which amplifies
#                     round-off where the envelope is small (edges, uncentred)
RTOL_MATMUL = 1e-12  # STFT round-off plus a 1025-term product in a different order
RTOL_DB = 1e-12  # exact inputs, log10 ulps; dB values are O(100)
RTOL_MFCC = 1e-11  # dB of a mel spectrogram carrying ~1e-14 error, then a direct DCT
#                    (SciPy uses an FFT-based DCT)


@pytest.fixture(scope="module")
def fx():
    return load_librosa_fixture("spectrum")


@pytest.fixture(scope="module")
def mix():
    return fixture_signal("mix")


@pytest.fixture(scope="module")
def sr():
    return fixture_sr()


@pytest.fixture(scope="module")
def mono(mix, sr):
    return aus.AudioSamples.new_mono(mix, sr)


@pytest.fixture(scope="module")
def stereo(sr):
    data = np.stack([fixture_signal("stereo_l"), fixture_signal("stereo_r")])
    return aus.AudioSamples.new_multi(data, sr)


def _norm_param(norm):
    if norm == "inf":
        return np.inf
    return norm


# ---------------------------------------------------------------------------
# Windows
# ---------------------------------------------------------------------------

WINDOW_SPECS = {
    "hann": "hann",
    "hamming": "hamming",
    "blackman": "blackman",
    "blackmanharris": "blackmanharris",
    "boxcar": "boxcar",
    "bartlett": "bartlett",
    "triang": "triang",
    "kaiser": ("kaiser", 8.6),
    "gaussian": ("gaussian", 7.0),
}


def test_get_window_matches_fixture(fx):
    n_checked = 0
    for key, expected in fx["windows"].items():
        name, n, kind = key.rsplit("_", 2)
        w = S.get_window(WINDOW_SPECS[name], int(n), fftbins=(kind == "periodic"))
        assert w.dtype == np.float64 and w.shape == (int(n),)
        # Same formulas as SciPy term by term; libm cos/exp/sqrt ulps only.
        assert_peak_close(w, fixture_array(expected), RTOL_ULP, key)
        n_checked += 1
    assert n_checked == 36


def test_window_aliases_and_arrays():
    np.testing.assert_array_equal(S.get_window("rectangular", 8), np.ones(8))
    np.testing.assert_array_equal(S.get_window(("hann",), 8), S.get_window("hann", 8))
    custom = np.array([0.5, 1.0, 0.5])
    np.testing.assert_array_equal(S.get_window(custom, 3), custom)
    # A bare number is a Kaiser beta, as in scipy.signal.get_window.
    np.testing.assert_array_equal(S.get_window(8.6, 16), S.get_window(("kaiser", 8.6), 16))
    with pytest.raises(ValueError, match="window"):
        S.get_window("nope", 8)
    with pytest.raises(ValueError, match="kaiser"):
        S.get_window("kaiser", 8)
    with pytest.raises(ValueError):
        S.get_window(custom, 4)  # wrong length


def test_window_bandwidth_formula(fx):
    for name, case in fx["window_bandwidth"].items():
        bw = S.window_bandwidth(name)
        # Sums of 1000 terms: numpy pairwise vs sequential summation.
        assert abs(bw - case["formula"]) <= 1e-13 * case["formula"], name
        # librosa returns a pre-computed table value (documented difference).
        assert abs(bw - case["table"]) < 1e-3, name


def test_window_sumsquare(fx):
    f = fx["window_sumsquare"]
    step = f["step"]
    a = S.window_sumsquare(window="hann", n_frames=10, hop_length=256, win_length=800, n_fft=1024)
    # Sums of at most 4 squared window values: round-off only.
    assert_peak_close(a[::step], fixture_array(f["hann_winlen"]), RTOL_ULP, "hann_winlen")
    b = S.window_sumsquare(window="hamming", n_frames=6, hop_length=512, n_fft=2048)
    assert_peak_close(b[::step], fixture_array(f["hamming"]), RTOL_ULP, "hamming")


# ---------------------------------------------------------------------------
# Mel filterbank and unit conversions
# ---------------------------------------------------------------------------


def test_mel_filterbank(fx):
    for name, case in fx["mel"].items():
        p = dict(case["params"])
        p["norm"] = _norm_param(p["norm"])
        m = S.mel_filterbank(**p)
        assert list(m.shape) == case["shape"], name
        r, c = case["row_step"], case["col_step"]
        # Same arithmetic as librosa's float64 basis; mel edges go through
        # log/exp, so the ramps carry a few ulps.
        assert_peak_close(m[::r, ::c], fixture_array(case["subset"]), 1e-12, name)
        assert_peak_close(m.sum(axis=1), fixture_array(case["row_sums"]), 1e-12, name)
        assert_peak_close(m.max(axis=1), fixture_array(case["row_max"]), 1e-12, name)


def test_conversions(fx, sr):
    f = fx["convert"]
    freqs = fixture_array(f["freqs"])
    # Identical formulas; libm log/exp/pow ulps only.
    assert_peak_close(U.hz_to_mel(freqs, htk=False), fixture_array(f["hz_to_mel_slaney"]), RTOL_ULP)
    assert_peak_close(U.hz_to_mel(freqs, htk=True), fixture_array(f["hz_to_mel_htk"]), RTOL_ULP)
    # The default scale is HTK.
    assert_peak_close(U.hz_to_mel(freqs), fixture_array(f["hz_to_mel_htk"]), RTOL_ULP)
    assert_peak_close(
        U.mel_to_hz(fixture_array(f["mels_htk_in"]), htk=True), fixture_array(f["mel_to_hz_htk"]), RTOL_ULP
    )
    assert_peak_close(
        U.mel_to_hz(fixture_array(f["mels_slaney_in"]), htk=False),
        fixture_array(f["mel_to_hz_slaney"]),
        RTOL_ULP,
    )
    assert_peak_close(
        U.mel_frequencies(128, 0.0, 11025.0, htk=False), fixture_array(f["mel_frequencies_slaney"]), RTOL_ULP
    )
    assert_peak_close(
        U.mel_frequencies(128, 0.0, 11025.0, htk=True), fixture_array(f["mel_frequencies_htk"]), RTOL_ULP
    )
    assert_peak_close(
        U.mel_frequencies(40, 133.0, 6854.0, htk=False), fixture_array(f["mel_frequencies_40_slaney"]), RTOL_ULP
    )
    assert_peak_close(
        U.mel_frequencies(40, 133.0, 6854.0, htk=True), fixture_array(f["mel_frequencies_40_htk"]), RTOL_ULP
    )

    times = fixture_array(f["times"])
    samples = np.asarray(f["samples"], dtype=np.int64)
    frames = np.asarray(f["frames"], dtype=np.int64)
    # Integer conversions must match exactly.
    np.testing.assert_array_equal(
        U.time_to_frames_floor(times, sample_rate=sr, hop_length=512), np.asarray(f["time_to_frames"])
    )
    np.testing.assert_array_equal(
        U.time_to_frames_floor(times, sample_rate=sr, hop_length=512, n_fft=2048),
        np.asarray(f["time_to_frames_nfft"]),
    )
    np.testing.assert_array_equal(U.time_to_samples(times, sample_rate=sr), np.asarray(f["time_to_samples"]))
    np.testing.assert_array_equal(
        U.samples_to_frames(samples, hop_length=512, n_fft=2047), np.asarray(f["samples_to_frames"])
    )
    np.testing.assert_array_equal(U.frames_to_samples(frames, hop_length=300), np.asarray(f["frames_to_samples"]))
    # int / float division: exact.
    np.testing.assert_array_equal(
        U.frames_to_time(frames, float(sr), 512, n_fft=2048), fixture_array(f["frames_to_time_nfft"])
    )


def test_conversions_scalar_and_shape():
    assert isinstance(U.hz_to_mel(1000.0), float)
    assert isinstance(U.time_to_frames_floor(0.5, sample_rate=22050, hop_length=512), int)
    out = U.hz_to_midi(np.full((2, 3), 440.0))
    assert out.shape == (2, 3) and np.all(out == 69.0)
    # time_to_frames rounds to the nearest frame.
    assert U.time_to_frames(1.0, 44100.0, 512) == 86
    assert U.frames_to_time(100, 44100.0, 512) == pytest.approx(100 * 512 / 44100)
    np.testing.assert_array_equal(U.samples_like(3, hop_length=512), [0, 512, 1024])
    np.testing.assert_allclose(U.times_like(np.zeros((5, 4)), sample_rate=1000.0, hop_length=100), [0, 0.1, 0.2, 0.3])


# ---------------------------------------------------------------------------
# STFT / ISTFT
# ---------------------------------------------------------------------------


def test_stft_matrix_cases(fx, mono, mix):
    for name, case in fx["stft"].items():
        kw = case["params"]
        d = mono.stft_matrix(**kw)
        assert d.dtype == np.complex128
        assert d.shape == (1, *case["shape"]), name
        b, t = case["bin_step"], case["frame_step"]
        expected = fixture_array(case["subset"])
        # Different FFT implementations (pocketfft vs realfft): ~1e-16 x peak per bin.
        assert_peak_close(d[0, ::b, ::t], expected, RTOL_FFT, f"stft_matrix/{name}")
        # The array function on a 1-D signal gives librosa's 2-D shape.
        d2 = S.stft(mix, **kw)
        assert d2.shape == tuple(case["shape"])
        assert_peak_close(d2[::b, ::t], expected, RTOL_FFT, f"spectral.stft/{name}")


def test_stft_stereo(fx, stereo):
    f = fx["stft_stereo"]
    d = stereo.stft_matrix()
    assert list(d.shape) == f["shape"]
    assert_peak_close(d[:, :: f["bin_step"], :: f["frame_step"]], fixture_array(f["subset"]), RTOL_FFT)


def test_stft_integer_audio_uses_audio_aware_conversion(sr):
    # Integer audio is converted with the crate's audio-aware conversion
    # (i16: negative codes / 32768, positive codes / 32767), not librosa's
    # usual k / 32768, so compare against the explicit float64 conversion.
    # int16 audio is analysed in float32 (every int16 value is exact in
    # float32), so it equals the float32 conversion's STFT exactly and agrees
    # with the float64 path to float32 FFT round-off.
    pcm = fixture_signal_pcm("mix")
    a16 = aus.AudioSamples.new_mono(pcm, sr)
    d16 = a16.stft_matrix(n_fft=512)
    assert d16.dtype == np.complex64
    np.testing.assert_array_equal(d16, a16.as_f32().stft_matrix(n_fft=512))
    # float32 FFT of a 512-point frame: a few float32 ulps (6e-8) of the peak per
    # bin; measured 1.2e-7 peak-relative.
    assert_peak_close(d16, a16.as_f64().stft_matrix(n_fft=512), 1e-6)
    # f32 audio of the same (exactly representable) values: float32 output,
    # equal to the float64 analysis to float32 round-off (measured 1.4e-7).
    a32 = aus.AudioSamples.new_mono((pcm / 32768.0).astype(np.float32), sr)
    a64 = aus.AudioSamples.new_mono(pcm / 32768.0, sr)
    d32 = a32.stft_matrix(n_fft=512)
    assert d32.dtype == np.complex64
    assert_peak_close(d32, a64.stft_matrix(n_fft=512), 1e-6)


def test_istft(fx, mix):
    f = fx["istft"]
    short = mix[:3000]
    d = S.stft(short)
    gain = 1.0 / (1.0 + np.arange(d.shape[0]) / 100.0)
    dm = d * gain[:, None]
    # Inverse FFT + overlap-add + division by the window envelope: FFT round-off.
    assert_peak_close(S.istft(dm, length=len(short))[::2], fixture_array(f["length_given_step2"]), 1e-12)
    y = S.istft(dm)
    assert len(y) == f["no_length_len"]
    assert_peak_close(y[::4], fixture_array(f["no_length_step4"]), 1e-12)
    # Tail near the end of the envelope: the division amplifies round-off (to about 2e-12).
    assert_peak_close(S.istft(dm, length=f["longer_len"])[-1500:], fixture_array(f["longer"]), RTOL_ISTFT)
    d2 = S.stft(short, n_fft=1024, hop_length=256, win_length=800, center=False)
    y2 = S.istft(d2 * gain[: d2.shape[0], None], n_fft=1024, hop_length=256, win_length=800, center=False)
    # Uncentred: first/last samples sit where the envelope is tiny.
    assert_peak_close(y2[::2], fixture_array(f["nocenter_winlen_step2"]), RTOL_ISTFT)

    # Method form returns AudioSamples.
    audio = aus.AudioSamples.istft_matrix(dm[None], sample_rate=22050, length=len(short))
    assert audio.sample_rate == 22050 and audio.num_channels == 1
    assert_peak_close(np.asarray(audio.to_numpy()).ravel()[::2], fixture_array(f["length_given_step2"]), 1e-12)


def test_istft_round_trip_stereo(stereo):
    d = stereo.stft_matrix(n_fft=512, window="hamming", pad_mode="reflect")
    y = aus.AudioSamples.istft_matrix(d, sample_rate=stereo.sample_rate, n_fft=512, window="hamming", length=stereo.samples_per_channel())
    assert y.num_channels == 2
    np.testing.assert_allclose(y.to_numpy(), stereo.to_numpy(), atol=1e-12)


# ---------------------------------------------------------------------------
# Mel spectrogram, dB, MFCC
# ---------------------------------------------------------------------------


def test_melspectrogram(fx, mono, stereo):
    f = fx["melspectrogram"]
    step = f["row_step"]
    m = mono.melspectrogram()
    assert m.shape[0] == 1 and m.shape[1] == 128
    # STFT round-off plus a 1025-term matrix product (BLAS vs matrixmultiply order).
    assert_peak_close(m[0, ::step], fixture_array(f["default"]), RTOL_MATMUL, "default")
    # librosa's default float32 mel basis: ~6e-8 relative rounding of the weights.
    assert_peak_close(m[0, ::step], fixture_array(f["default_float32_basis"]), 1e-6, "float32 basis")
    m2 = mono.melspectrogram(n_mels=64, fmin=50.0, fmax=8000.0, htk=True, power=1.0)
    assert_peak_close(m2[0, ::step], fixture_array(f["htk_power1"]), RTOL_MATMUL, "htk_power1")
    ms = stereo.melspectrogram()
    assert_peak_close(ms[:, :: f["stereo_row_step"], :], fixture_array(f["stereo"]), RTOL_MATMUL, "stereo")


def test_melspectrogram_from_spectrogram_matches_method(mono, sr):
    s = mono.spectrogram_matrix(power=2.0, hop_length=512)
    m = S.melspectrogram_from_spectrogram(s[0], sr=sr)
    np.testing.assert_allclose(m, mono.melspectrogram()[0], rtol=1e-13, atol=0)


def test_db_conversions(fx, stereo):
    f = fx["db"]
    mel_in = fixture_array(fx["melspectrogram"]["default"])
    # Inputs are librosa's exact inputs (stored); log10 ulps only.
    # librosa's default top_db is 80; the binding's default is no floor.
    assert_peak_close(S.power_to_db(mel_in, top_db=80.0), fixture_array(f["power_default"]), RTOL_DB)
    assert_peak_close(S.power_to_db(mel_in), fixture_array(f["power_no_top_db"]), RTOL_DB)
    assert_peak_close(S.power_to_db(mel_in, ref=np.max, top_db=80.0), fixture_array(f["power_ref_max"]), RTOL_DB)
    assert_peak_close(S.power_to_db(mel_in, ref="max", top_db=80.0), fixture_array(f["power_ref_max"]), RTOL_DB)
    assert_peak_close(S.power_to_db(mel_in, top_db=None), fixture_array(f["power_no_top_db"]), RTOL_DB)
    # Stereo: reference max and top_db are per channel (axes="auto"). The input is
    # computed here (melspectrogram round-off ~1e-14 rel) -> dB abs error ~1e-13.
    ms = stereo.melspectrogram()
    step = f["power_stereo_ref_max_row_step"]
    assert_peak_close(S.power_to_db(ms, ref=np.max, top_db=80.0)[:, ::step, :], fixture_array(f["power_stereo_ref_max"]), RTOL_DB)
    mag = np.abs(fixture_array(fx["stft"]["default"]["subset"]))
    assert_peak_close(S.amplitude_to_db(mag, ref=np.max, top_db=80.0), fixture_array(f["amplitude_ref_max"]), RTOL_DB)
    assert_peak_close(S.amplitude_to_db(mag, top_db=80.0), fixture_array(f["amplitude_default"]), RTOL_DB)
    # Complex input: magnitude is taken (as librosa, which warns).
    assert_peak_close(
        S.amplitude_to_db(fixture_array(fx["stft"]["default"]["subset"]), top_db=80.0),
        fixture_array(f["amplitude_default"]),
        RTOL_DB,
    )
    db_in = fixture_array(f["db_in"])
    assert_peak_close(S.db_to_power(db_in, ref=2.0), fixture_array(f["db_to_power_ref2"]), RTOL_ULP)
    assert_peak_close(S.db_to_amplitude(db_in, ref=2.0), fixture_array(f["db_to_amplitude_ref2"]), RTOL_ULP)
    assert isinstance(S.power_to_db(10.0), float)
    with pytest.raises(ValueError, match="amin"):
        S.power_to_db(mel_in, amin=0.0)
    with pytest.raises(TypeError):
        S.power_to_db(np.ones(3, dtype=complex))


MFCC_CASES = {
    "default": {},
    "lifter22": {"lifter": 22.0},
    "dct3": {"dct_type": 3},
    "dct1": {"dct_type": 1},
    "norm_none": {"norm": None},
    "htk_13_40": {"n_mfcc": 13, "n_mels": 40, "htk": True},
}


@pytest.mark.parametrize("name", sorted(MFCC_CASES))
def test_mfcc_frames(fx, mono, name):
    # The fixtures use librosa's default top_db=80 (the binding defaults to None).
    m = mono.mfcc_frames(top_db=80.0, **MFCC_CASES[name])
    # MFCC = DCT of dB values; dB errors are absolute (~1e-12 dB) and the DCT is a
    # direct product here vs FFT-based in SciPy. Floored bins agree exactly.
    assert_peak_close(m[0], fixture_array(fx["mfcc"][name]), RTOL_MFCC, name)


def test_mfcc_stereo_and_from_mel_db(fx, stereo, mono):
    assert_peak_close(stereo.mfcc_frames(top_db=80.0), fixture_array(fx["mfcc"]["stereo"]), RTOL_MFCC, "stereo")
    mel_db = S.power_to_db(mono.melspectrogram(), top_db=80.0)
    assert_peak_close(S.mfcc_from_mel_db(mel_db)[0], fixture_array(fx["mfcc"]["default"]), RTOL_MFCC, "from_mel_db")
    with pytest.raises(ValueError):
        mono.mfcc_frames(n_mfcc=200)  # > n_mels (librosa silently truncates)


# ---------------------------------------------------------------------------
# delta, stack_memory, frame
# ---------------------------------------------------------------------------

DELTA_CASES = {
    "order1": {},
    "order2": {"order": 2},
    "width5": {"width": 5},
    "nearest": {"mode": "nearest"},
    "mirror_w7_o2": {"width": 7, "order": 2, "mode": "mirror"},
    "wrap": {"mode": "wrap"},
    "constant": {"mode": "constant"},
    "axis0": {"axis": 0, "width": 5},
}


def test_delta(fx):
    mfcc = fixture_array(fx["mfcc"]["default"])
    for name, kw in DELTA_CASES.items():
        # Exact input. SciPy fits with lstsq, we use a normal-equation solve:
        # both exact up to a few ulps of the small-integer weights.
        assert_peak_close(S.delta(mfcc, **kw), fixture_array(fx["delta"][name]), 1e-12, name)
    with pytest.raises(ValueError):
        S.delta(mfcc, width=4)


def test_stack_memory_and_frame(fx, mix):
    mfcc5 = fixture_array(fx["mfcc"]["default"])[:5]
    # Pure copies: exact.
    np.testing.assert_array_equal(S.stack_memory(mfcc5, n_steps=3, delay=2), fixture_array(fx["stack_memory"]["n3_d2"]))
    np.testing.assert_array_equal(S.stack_memory(mfcc5, n_steps=2, delay=-1), fixture_array(fx["stack_memory"]["n2_dm1"]))
    np.testing.assert_array_equal(
        S.frame(mix[:3000], frame_length=512, hop_length=256)[::8, ::3], fixture_array(fx["frame"]["frames"])
    )


# ---------------------------------------------------------------------------
# Argument validation
# ---------------------------------------------------------------------------


def test_invalid_arguments_raise_value_error(mono):
    with pytest.raises(ValueError, match="n_fft"):
        mono.stft_matrix(n_fft=0)
    with pytest.raises(ValueError, match="hop_length"):
        mono.stft_matrix(hop_length=-1)
    with pytest.raises(ValueError, match="pad_mode"):
        mono.stft_matrix(pad_mode="wrap")
    with pytest.raises(ValueError, match="norm"):
        mono.melspectrogram(norm="area")
    with pytest.raises(ValueError, match="dct_type"):
        mono.mfcc_frames(dct_type=4)
    with pytest.raises(ValueError):
        mono.melspectrogram(fmin=5000.0, fmax=100.0)


# ---------------------------------------------------------------------------
# Direct comparison with librosa (when installed)
# ---------------------------------------------------------------------------


def test_direct_librosa_comparison(mono, mix, sr):
    librosa = pytest.importorskip("librosa")
    # Same tolerance rationale as the fixture tests above.
    assert_peak_close(mono.stft_matrix(n_fft=1024, window=("kaiser", 5.0))[0],
                      librosa.stft(mix, n_fft=1024, window=("kaiser", 5.0)), RTOL_FFT)
    assert_peak_close(
        mono.melspectrogram(n_fft=1024, hop_length=256, n_mels=40, norm=None)[0],
        librosa.feature.melspectrogram(y=mix, sr=sr, n_fft=1024, hop_length=256, n_mels=40, norm=None, dtype=np.float64),
        RTOL_MATMUL,
    )
    mel = librosa.feature.melspectrogram(y=mix, sr=sr, dtype=np.float64)
    assert_peak_close(
        mono.mfcc_frames(n_mfcc=13, lifter=10.0, top_db=None)[0],
        librosa.feature.mfcc(S=librosa.power_to_db(mel, top_db=None), n_mfcc=13, lifter=10.0),
        RTOL_MFCC,
    )
    assert_peak_close(
        S.mel_filterbank(sr=16000, n_fft=400, n_mels=23, fmin=20.0, htk=True, norm=2.0),
        librosa.filters.mel(sr=16000, n_fft=400, n_mels=23, fmin=20.0, htk=True, norm=2.0, dtype=np.float64),
        1e-12,
    )
    assert_peak_close(
        S.window_sumsquare(window="blackman", n_frames=7, hop_length=128, n_fft=512),
        librosa.filters.window_sumsquare(window="blackman", n_frames=7, hop_length=128, n_fft=512, dtype=np.float64),
        RTOL_ULP,
    )
    times = np.linspace(-1, 2, 97)
    np.testing.assert_array_equal(
        U.time_to_frames_floor(times, sample_rate=sr, hop_length=256, n_fft=1024),
        librosa.time_to_frames(times, sr=sr, hop_length=256, n_fft=1024),
    )
    np.testing.assert_array_equal(U.samples_like(np.zeros((2, 7)), hop_length=128), librosa.samples_like(np.zeros((2, 7)), hop_length=128))
    np.testing.assert_allclose(U.times_like(9, sample_rate=sr, hop_length=512, n_fft=2048), librosa.times_like(9, sr=sr, n_fft=2048), rtol=1e-15)


def test_mu_law_against_librosa():
    x = np.linspace(-1.0, 1.0, 1001)
    codes = aus.mu_law_encode(x)
    assert codes.dtype == np.int64 and codes.min() == -128 and codes.max() == 127
    back = aus.mu_law_decode(codes)
    assert np.max(np.abs(back - x)) < 0.05
    with pytest.raises(ValueError):
        aus.mu_law_encode([1.5])
    librosa = pytest.importorskip("librosa")
    # Integer codes: exact.
    np.testing.assert_array_equal(codes, librosa.mu_compress(x, mu=255, quantize=True))
    np.testing.assert_array_equal(aus.mu_law_encode(x, mu=15.5), librosa.mu_compress(x, mu=15.5, quantize=True))
    # Same closed form; pow ulps only.
    np.testing.assert_allclose(back, librosa.mu_expand(codes, mu=255, quantize=True), rtol=1e-14, atol=1e-15)
