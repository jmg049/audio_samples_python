"""Output dtypes of the analysis bindings.

Rule (the Rust crate's per-sample-type ``Float``):

* ``AudioSamples`` analysis methods return float32 / complex64 arrays for
  uint8, int16, int24 and float32 audio, and float64 / complex128 arrays for
  int32 and float64 audio.
* Free functions over NumPy arrays compute in float32 for float32 / complex64
  input and in float64 for everything else.
* Audio-returning methods keep the receiver's sample type; the static
  reconstructions (``istft_matrix``, ``icqt``, ``griffinlim_cqt``,
  ``griffinlim_audio``) take ``dtype=`` and otherwise follow the input matrix.

The float32 comparisons against librosa run librosa on the same float32
signal. Tolerances are peak-relative (``max|a - e| <= rtol * max|e|``); each
bound is set a few times above the error measured on these signals (stated
next to it). float32 has a unit round-off of 6e-8; the measured errors are a
few float32 ulps of the peak, accumulated over FFT butterflies, filterbank
sums and logarithms, with both libraries rounding at different points.
"""

from __future__ import annotations

import numpy as np
import pytest

import audio_samples as aus
from audio_samples import effects as E
from audio_samples import features as F
from audio_samples import rhythm as R
from audio_samples import sequence as Q
from audio_samples import spectral as S

from conftest import assert_peak_close, fixture_signal, fixture_signal_pcm, fixture_sr

ST = aus.SampleType
FLOAT32_TYPES = ("U8", "I16", "I24", "F32")
FLOAT64_TYPES = ("I32", "F64")


@pytest.fixture(scope="module")
def sr():
    return fixture_sr()


def _expected(sample_type: str, complex_out: bool):
    if sample_type in FLOAT32_TYPES:
        return np.complex64 if complex_out else np.float32
    return np.complex128 if complex_out else np.float64


ANALYSES = {
    "stft_matrix": (lambda a: a.stft_matrix(n_fft=512), True),
    "melspectrogram": (lambda a: a.melspectrogram(n_fft=512, n_mels=32), False),
    "mfcc_frames": (lambda a: a.mfcc_frames(n_fft=512, n_mels=32, n_mfcc=13), False),
    "chroma_stft": (lambda a: a.chroma_stft(n_fft=512), False),
    "cqt": (lambda a: a.cqt(fmin=440.0, n_bins=12), True),
    "onset_strength": (lambda a: a.onset_strength(), False),
    "yin": (lambda a: a.yin(fmin=110.0, fmax=1760.0), False),
    "spectral_centroid_frames": (lambda a: a.spectral_centroid_frames(n_fft=512), False),
}


@pytest.mark.parametrize("sample_type", FLOAT32_TYPES + FLOAT64_TYPES)
@pytest.mark.parametrize("method", sorted(ANALYSES))
def test_analysis_dtype_follows_sample_type(sr, sample_type, method):
    base = aus.AudioSamples.new_mono(fixture_signal("mix")[:8192], sr)
    audio = base.to_format(getattr(ST, sample_type))
    fn, complex_out = ANALYSES[method]
    out = fn(audio)
    assert out.dtype == _expected(sample_type, complex_out), f"{method} on {sample_type}"


# ---------------------------------------------------------------------------
# float32 audio against librosa on float32 input
# ---------------------------------------------------------------------------

# name -> (ours(audio) -> array, librosa(y, sr) -> array, rtol). Measured
# peak-relative errors on the "mix" and "notes" fixture signals are given as
# (mix, notes).
F32_CASES = {
    # FFT round-off; measured (1.2e-7, 1.0e-7).
    "stft": (lambda a: a.stft_matrix()[0], lambda lr, y, sr: lr.stft(y), 5e-7),
    # STFT power plus a 1025-term filterbank sum; measured (2.5e-7, 2.0e-7).
    "melspectrogram": (
        lambda a: a.melspectrogram()[0],
        lambda lr, y, sr: lr.feature.melspectrogram(y=y, sr=sr),
        1e-6,
    ),
    # dB values (log10 of the mel power) through a 128-term DCT; the peak is
    # the large C0 coefficient. Measured (1.2e-6, 1.0e-6).
    "mfcc": (
        lambda a: a.mfcc_frames(top_db=80.0)[0],
        lambda lr, y, sr: lr.feature.mfcc(y=y, sr=sr),
        5e-6,
    ),
    # Ratios of filterbank sums (inf-norm per frame); measured (2.4e-7, 6.0e-7).
    "chroma_stft": (
        lambda a: a.chroma_stft(tuning=0.0)[0],
        lambda lr, y, sr: lr.feature.chroma_stft(y=y, sr=sr, tuning=0.0),
        2e-6,
    ),
    # Top-octave CQT (no decimation); "notes" has little energy at C7, so its
    # peak is small and the relative error larger. Measured (2.1e-7, 1.3e-6).
    "cqt_top_octave": (
        lambda a: a.cqt(fmin=2093.004522404789, n_bins=12)[0],
        lambda lr, y, sr: lr.cqt(y, sr=sr, fmin=2093.004522404789, n_bins=12),
        5e-6,
    ),
    # Differences of dB values (cancellation) averaged over 128 mel bands;
    # measured (4.4e-6, 1.8e-7).
    "onset_strength": (
        lambda a: a.onset_strength()[0],
        lambda lr, y, sr: lr.onset.onset_strength(y=y, sr=sr),
        2e-5,
    ),
    # Parabolic interpolation of the CMND minimum; librosa returns float64 for
    # float32 input. Measured (7.2e-8, 1.0e-7).
    "yin": (
        lambda a: a.yin(fmin=65.0, fmax=2093.0)[0],
        lambda lr, y, sr: lr.yin(y, fmin=65.0, fmax=2093.0, sr=sr),
        5e-7,
    ),
    # Ratio of two 1025-term sums per frame; measured (1.9e-6, 1.9e-6).
    "spectral_centroid": (
        lambda a: a.spectral_centroid_frames()[0, 0],
        lambda lr, y, sr: lr.feature.spectral_centroid(y=y, sr=sr)[0],
        1e-5,
    ),
}


@pytest.mark.parametrize("signal", ("mix", "notes"))
@pytest.mark.parametrize("name", sorted(F32_CASES))
def test_float32_audio_matches_librosa_float32(sr, signal, name):
    librosa = pytest.importorskip("librosa")
    # The fixture signals are 16-bit PCM codes, exact in float32.
    y = fixture_signal(signal).astype(np.float32)
    audio = aus.AudioSamples.new_mono(y, sr)
    ours_fn, theirs_fn, rtol = F32_CASES[name]
    ours = ours_fn(audio)
    assert ours.dtype in (np.float32, np.complex64)
    assert_peak_close(ours, theirs_fn(librosa, y, sr), rtol, f"{signal}/{name}")


# ---------------------------------------------------------------------------
# Audio-returning operations keep the sample type
# ---------------------------------------------------------------------------


def _max_lsb(a, b) -> int:
    return int(np.max(np.abs(a.to_numpy().astype(np.int64) - b.to_numpy().astype(np.int64))))


AUDIO_OPS = {
    "time_stretch": (lambda a: a.time_stretch(1.25), "mix"),
    "preemphasis": (lambda a: a.preemphasis(), "mix"),
    "pitch_shift": (lambda a: a.pitch_shift(2), "mix"),
    "trim_db": (lambda a: a.trim_db(top_db=30.0)[0], "burst"),
}


@pytest.mark.parametrize("name", sorted(AUDIO_OPS))
def test_int16_audio_ops_return_int16(sr, name):
    fn, signal = AUDIO_OPS[name]
    a16 = aus.AudioSamples.new_mono(fixture_signal_pcm(signal), sr)
    out = fn(a16)
    assert out.dtype == "i16" and out.to_numpy().dtype == np.int16
    # The float32 computation agrees with the float64 path re-quantised to
    # int16 within one step.
    ref = fn(a16.as_f64()).to_format(ST.I16)
    assert out.samples_per_channel() == ref.samples_per_channel()
    assert _max_lsb(out, ref) <= 1


def test_istft_matrix_dtype(sr):
    a16 = aus.AudioSamples.new_mono(fixture_signal_pcm("mix"), sr)
    n = a16.samples_per_channel()
    d = a16.stft_matrix()
    assert d.dtype == np.complex64
    y16 = aus.AudioSamples.istft_matrix(d, sample_rate=sr, length=n, dtype=ST.I16)
    assert y16.dtype == "i16"
    # The round trip reproduces the int16 input exactly.
    assert _max_lsb(y16, a16) == 0
    # Without dtype=, the output follows the matrix precision.
    assert aus.AudioSamples.istft_matrix(d, sample_rate=sr).dtype == "f32"
    assert aus.AudioSamples.istft_matrix(d.astype(np.complex128), sample_rate=sr).dtype == "f64"
    assert aus.AudioSamples.istft_matrix(d, sample_rate=sr, dtype=ST.I32).dtype == "i32"


def test_constant_q_and_griffinlim_reconstruction_dtype(sr):
    y = fixture_signal("mix")[:8192]
    c = aus.AudioSamples.new_mono(y.astype(np.float32), sr).cqt(fmin=110.0, n_bins=48)
    assert c.dtype == np.complex64
    assert aus.AudioSamples.icqt(c, sample_rate=sr, fmin=110.0).dtype == "f32"
    assert aus.AudioSamples.icqt(c.astype(np.complex128), sample_rate=sr, fmin=110.0).dtype == "f64"
    assert aus.AudioSamples.icqt(c, sample_rate=sr, fmin=110.0, dtype=ST.I16).dtype == "i16"
    mag = np.abs(c)
    gl = aus.AudioSamples.griffinlim_cqt(mag, sample_rate=sr, fmin=110.0, n_iter=1)
    assert gl.dtype == "f32"
    assert aus.AudioSamples.griffinlim_cqt(mag, sample_rate=sr, fmin=110.0, n_iter=1, dtype=ST.I16).dtype == "i16"
    s = np.abs(S.stft(y.astype(np.float32), n_fft=512))
    assert aus.AudioSamples.griffinlim_audio(s, sample_rate=sr, n_iter=1).dtype == "f32"
    assert aus.AudioSamples.griffinlim_audio(s.astype(np.float64), sample_rate=sr, n_iter=1).dtype == "f64"
    assert aus.AudioSamples.griffinlim_audio(s, sample_rate=sr, n_iter=1, dtype=ST.U8).dtype == "u8"


# ---------------------------------------------------------------------------
# Free functions keep the input's precision
# ---------------------------------------------------------------------------


def test_free_functions_keep_input_precision(sr):
    y64 = fixture_signal("mix")[:8192]
    y32 = y64.astype(np.float32)
    calls = {
        "spectral.stft": lambda y: S.stft(y, n_fft=512),
        "spectral.power_to_db": lambda y: S.power_to_db(np.abs(y) + 1e-3),
        "spectral.frame": lambda y: S.frame(y, frame_length=256, hop_length=128),
        "features.yin": lambda y: F.yin(y, fmin=110.0, fmax=1760.0, sr=sr),
        "features.spectral_centroid_from_spectrogram": lambda y: F.spectral_centroid_from_spectrogram(
            np.abs(S.stft(y, n_fft=512)), sr=sr
        ),
        "rhythm.tempogram": lambda y: R.tempogram(onset_envelope=np.abs(y[:600]), sr=sr),
        "effects.preemphasis": lambda y: E.preemphasis(y),
        "effects.lpc": lambda y: E.lpc(y, order=8),
        "sequence.dtw": lambda y: Q.dtw(y[None, :40], y[None, 10:60])[0],
    }
    for name, fn in calls.items():
        lo, hi = fn(y32), fn(y64)
        assert lo.dtype in (np.float32, np.complex64), name
        assert hi.dtype in (np.float64, np.complex128), name
    # Lists and other dtypes are converted to float64.
    assert S.power_to_db([1.0, 10.0]).dtype == np.float64
    assert S.power_to_db(np.array([1, 10], dtype=np.int16)).dtype == np.float64
    assert S.power_to_db(np.array([1, 10], dtype=np.float16)).dtype == np.float64
    # complex64 input to istft gives float32.
    assert S.istft(S.stft(y32, n_fft=512)).dtype == np.float32


# ---------------------------------------------------------------------------
# Windows and FIR filters on integer audio
# ---------------------------------------------------------------------------


def test_apply_window_and_filter_on_int16_audio():
    x = aus.AudioSamples.new_mono(np.array([1000, 2000, 3000, -4000], dtype=np.int16), 8000)
    # Fractional coefficients: the products are rounded to nearest.
    w = x.apply_window(np.array([0.5, 0.25, 0.75, 0.1]))
    assert w.dtype == "i16"
    np.testing.assert_array_equal(w.to_numpy(), [500, 500, 2250, -400])
    # Taps outside [-1, 1]: y[n] = 1.5 x[n] - 0.5 x[n - 1], rounded and saturated.
    f = x.apply_filter([1.5, -0.5])
    assert f.dtype == "i16"
    np.testing.assert_array_equal(f.to_numpy(), [2500, 3500, -7500])
    # float64 audio: numpy.convolve(..., "valid").
    xf = x.as_f64()
    np.testing.assert_allclose(
        xf.apply_filter([1.5, -0.5]).to_numpy(), np.convolve(xf.to_numpy(), [1.5, -0.5], "valid"), rtol=1e-15
    )
