"""``AudioSamples.resample`` against ``librosa.resample`` (soxr_hq) and
``scipy.signal.resample_poly``, via the golden fixture ``resample.json``.

librosa's default resampler is the soxr C library, a different filter design
from ours (one Kaiser-windowed sinc stage), so the comparison is by SNR over the
stored windows (first/last 256 output samples and 1024 from the middle), with
the bounds of the Rust reference test ``tests/librosa_reference/resample.rs``:

* ``mix`` middle >= 125 dB: band-limited tones far from every Nyquist frequency,
  where both are near-ideal interpolators; this is the figure that shows time
  alignment (a 0.01-sample offset would cap it near 40 dB);
* ``mix`` head/tail >= 55 dB: the abrupt start/end is broadband, so the
  different transition bands show;
* ``noise`` >= 47 dB: white noise up to 11025 Hz exercises the transition band
  everywhere (measured 50.5-58.2 dB);
* ``scipy.signal.resample_poly`` middle window of ``mix`` >= 62 dB (limited by
  resample_poly's own Kaiser(5) passband ripple; measured 65.9-67.5 dB).

Lengths are exactly ``ceil(n * target / orig)`` (and equal librosa's here).
"""

from __future__ import annotations

import numpy as np
import pytest

import audio_samples as aus
from audio_samples.types import ResamplingQuality

from conftest import fixture_array, fixture_signal, load_librosa_fixture


def snr_db(actual, ref):
    return 10 * np.log10(np.sum(ref**2) / max(np.sum((actual - ref) ** 2), 1e-300))


def min_snr(signal: str, window: str) -> float:
    if signal == "mix":
        return 125.0 if window == "mid" else 55.0
    return 47.0


@pytest.fixture(scope="module")
def fx():
    return load_librosa_fixture("resample")


def test_resample_matches_librosa_soxr_hq(fx):
    for key, case in fx["resample"].items():
        signal = key.split("_")[0]
        x = fixture_signal(signal)
        audio = aus.AudioSamples.new_mono(x, int(case["orig_sr"]))
        y = audio.resample(int(case["target_sr"]), ResamplingQuality.high)
        assert y.sample_rate == case["target_sr"]
        y = np.asarray(y.to_numpy())
        assert y.size == case["len_exact"] == case["len"], key
        n = y.size
        edge = len(case["head"])
        for name, start in (("head", 0), ("mid", case["mid_start"]), ("tail", n - edge)):
            ref = fixture_array(case[name])
            snr = snr_db(y[start : start + ref.size], ref)
            assert snr >= min_snr(signal, name), f"{key} {name}: SNR {snr:.1f} dB"
        if signal == "mix":
            ref = fixture_array(case["scipy_mid"])
            start = case["mid_start"]
            snr = snr_db(y[start : start + ref.size], ref)
            assert snr >= 62.0, f"{key}: SNR vs resample_poly {snr:.1f} dB"


def test_resample_by_ratio_and_names(fx):
    case = fx["resample"]["mix_16000"]
    x = fixture_signal("mix")
    audio = aus.AudioSamples.new_mono(x, int(case["orig_sr"]))
    by_name = np.asarray(audio.resample(16000, "high").to_numpy())
    by_ratio = audio.resample_by_ratio(16000 / case["orig_sr"], quality="high")
    assert by_ratio.sample_rate == 16000
    np.testing.assert_array_equal(np.asarray(by_ratio.to_numpy()), by_name)


@pytest.mark.parametrize("quality", ["fast", "medium", "high", "very_high"])
@pytest.mark.parametrize("target", [8000, 16000, 44100, 48000])
def test_exact_length_and_alignment(quality, target):
    sr = 22050
    n = 22050 + 17
    t = np.arange(n) / sr
    x = 0.5 * np.sin(2 * np.pi * 440.0 * t)
    y = aus.AudioSamples.new_mono(x, sr).resample(target, quality)
    y = np.asarray(y.to_numpy())
    assert y.size == -(-n * target // sr)
    # Delay-free: in the interior the output is the same sinusoid sampled at
    # the new rate (no group delay). Fast upsampling is linear interpolation.
    tt = np.arange(y.size) / target
    ref = 0.5 * np.sin(2 * np.pi * 440.0 * tt)
    mid = slice(y.size // 4, 3 * y.size // 4)
    bound = 30.0 if quality == "fast" else 60.0
    assert snr_db(y[mid], ref[mid]) > bound


def test_resample_validation():
    a = aus.AudioSamples.new_mono(np.zeros(100), 22050)
    with pytest.raises(ValueError, match="target_sample_rate"):
        a.resample(0)
    with pytest.raises(ValueError, match="target_sample_rate"):
        a.resample(-1)
    with pytest.raises(ValueError, match="quality"):
        a.resample(16000, "ultra")
    with pytest.raises(TypeError):
        a.resample(16000, 3)
