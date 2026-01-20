"""
Resampling correctness tests.

Strategy
--------
Comparing sample-for-sample against librosa is not meaningful because each
library uses a different filter design.  Instead we test properties that any
correct resampler must satisfy:

1. Metadata: sample_rate and channel count are updated; sample count is right.
2. Integer-ratio Fast path: stride-N downsample matches numpy exactly; 2x
   upsample matches linear-interpolated reference.
3. Frequency preservation: a sine wave at f Hz, resampled to a new rate, still
   peaks at f Hz in the DFT (verified for all three quality levels).
4. Channel independence: each channel of a stereo signal is resampled
   correctly and independently.
5. Round-trip: resample up then back down (or vice versa) recovers the
   original within a reasonable tolerance.
6. Energy conservation: RMS changes by < 10 % after a 2:1 ratio resample.
"""

import numpy as np
import pytest

import audio_samples as aus
from audio_samples import AudioSamples, ResamplingQuality

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ALL_QUALITIES = [
    ResamplingQuality.fast,
    ResamplingQuality.medium,
    ResamplingQuality.high,
]


def make_sine(
    frequency: float, duration: float, sample_rate: int, amplitude: float = 0.5
) -> AudioSamples:
    n = int(duration * sample_rate)
    t = np.linspace(0, duration, n, endpoint=False)
    data = amplitude * np.sin(2 * np.pi * frequency * t)
    return aus.AudioSamples.new_mono(data.astype(np.float64), sample_rate=sample_rate)


def make_stereo(
    freq_l: float, freq_r: float, duration: float, sample_rate: int
) -> AudioSamples:
    n = int(duration * sample_rate)
    t = np.linspace(0, duration, n, endpoint=False)
    left = 0.5 * np.sin(2 * np.pi * freq_l * t)
    right = 0.5 * np.sin(2 * np.pi * freq_r * t)
    data = np.vstack([left, right]).astype(np.float64)
    return aus.AudioSamples.new_multi(data, sample_rate=sample_rate)


def dominant_frequency(samples: np.ndarray, sample_rate: int) -> float:
    """Return the frequency (Hz) of the largest DFT magnitude bin."""
    spectrum = np.abs(np.fft.rfft(samples))
    freqs = np.fft.rfftfreq(len(samples), d=1.0 / sample_rate)
    return float(freqs[np.argmax(spectrum)])


def expected_n_frames(in_frames: int, in_sr: int, out_sr: int) -> int:
    return (in_frames * out_sr + in_sr - 1) // in_sr


# ---------------------------------------------------------------------------
# 1. Metadata correctness
# ---------------------------------------------------------------------------


class TestMetadata:
    @pytest.mark.parametrize("quality", ALL_QUALITIES)
    @pytest.mark.parametrize(
        "src_sr,tgt_sr", [(44100, 22050), (22050, 44100), (44100, 48000)]
    )
    def test_sample_rate_updated(self, quality, src_sr, tgt_sr):
        audio = make_sine(440.0, 1.0, src_sr)
        result = audio.resample(target_sample_rate=tgt_sr, quality=quality)
        assert result.sample_rate == tgt_sr

    @pytest.mark.parametrize("quality", ALL_QUALITIES)
    def test_channel_count_preserved_mono(self, quality):
        audio = make_sine(440.0, 0.5, 44100)
        result = audio.resample(target_sample_rate=22050, quality=quality)
        assert result.channels == 1

    @pytest.mark.parametrize("quality", ALL_QUALITIES)
    def test_channel_count_preserved_stereo(self, quality):
        audio = make_stereo(440.0, 880.0, 0.5, 44100)
        result = audio.resample(target_sample_rate=22050, quality=quality)
        assert result.channels == 2

    @pytest.mark.parametrize("quality", ALL_QUALITIES)
    @pytest.mark.parametrize("src_sr,tgt_sr", [(44100, 22050), (22050, 44100)])
    def test_sample_count_correct(self, quality, src_sr, tgt_sr):
        duration = 1.0
        audio = make_sine(440.0, duration, src_sr)
        result = audio.resample(target_sample_rate=tgt_sr, quality=quality)
        expected = expected_n_frames(audio.samples_per_channel(), src_sr, tgt_sr)
        # Allow ±1 sample for rounding differences
        assert abs(result.samples_per_channel() - expected) <= 1, (
            f"{quality}: expected ~{expected} samples, got {result.samples_per_channel()}"
        )

    def test_identity_same_rate(self):
        audio = make_sine(440.0, 1.0, 44100)
        result = audio.resample(
            target_sample_rate=44100, quality=ResamplingQuality.fast
        )
        data_in = audio.to_numpy().ravel()
        data_out = result.to_numpy().ravel()
        np.testing.assert_array_equal(data_in, data_out)


# ---------------------------------------------------------------------------
# 2. Integer-ratio Fast path correctness
# ---------------------------------------------------------------------------


class TestFastIntegerRatio:
    def test_downsample_2x_matches_stride(self):
        """Fast 2x downsample must equal src[::2] (linear interp, frac=0)."""
        n = 44100
        data = np.sin(2 * np.pi * 440.0 * np.arange(n) / 44100).astype(np.float32)
        audio = aus.AudioSamples.new_mono(data.astype(np.float64), sample_rate=44100)
        result = audio.resample(
            target_sample_rate=22050, quality=ResamplingQuality.fast
        )
        out = result.to_numpy().ravel().astype(np.float32)
        ref = data[::2]
        np.testing.assert_allclose(
            out,
            ref[: len(out)],
            atol=1e-6,
            err_msg="Fast 2x downsample should equal stride-2 slice",
        )

    def test_upsample_2x_linear_interp(self):
        """Fast 2x upsample should be linear interpolation between samples."""
        # Use a simple ramp so we can compute the reference trivially
        n = 1000
        data = np.linspace(0.0, 1.0, n, endpoint=False).astype(np.float32)
        audio = aus.AudioSamples.new_mono(data.astype(np.float64), sample_rate=22050)
        result = audio.resample(
            target_sample_rate=44100, quality=ResamplingQuality.fast
        )
        out = result.to_numpy().ravel().astype(np.float32)

        # Build reference: interleave originals with midpoints
        ref = np.empty(2 * (n - 1), dtype=np.float32)
        ref[0::2] = data[:-1]
        ref[1::2] = 0.5 * (data[:-1] + data[1:])

        np.testing.assert_allclose(
            out[: len(ref)],
            ref,
            atol=1e-5,
            err_msg="Fast 2x upsample should be linear interpolation",
        )


# ---------------------------------------------------------------------------
# 3. Frequency preservation
# ---------------------------------------------------------------------------


class TestFrequencyPreservation:
    """
    A sine wave at f Hz should still peak at f Hz after resampling,
    provided f is well below the Nyquist of both rates.
    """

    @pytest.mark.parametrize("quality", ALL_QUALITIES)
    @pytest.mark.parametrize("src_sr,tgt_sr", [(44100, 22050), (22050, 44100)])
    def test_sine_frequency_preserved_mono(self, quality, src_sr, tgt_sr):
        freq = 440.0  # well below 11025 Hz Nyquist of 22050 rate
        audio = make_sine(freq, 2.0, src_sr)
        result = audio.resample(target_sample_rate=tgt_sr, quality=quality)
        samples = result.to_numpy().ravel()
        detected = dominant_frequency(samples, tgt_sr)
        assert abs(detected - freq) < 2.0, (
            f"{quality} ({src_sr}→{tgt_sr}): expected {freq} Hz, got {detected} Hz"
        )

    @pytest.mark.parametrize("quality", ALL_QUALITIES)
    def test_sine_frequency_preserved_stereo(self, quality):
        freq_l, freq_r = 440.0, 880.0
        audio = make_stereo(freq_l, freq_r, 2.0, 44100)
        result = audio.resample(target_sample_rate=22050, quality=quality)
        arr = result.to_numpy()  # shape (2, N) or (N, 2) — use numpy consistently
        # to_numpy returns (channels, samples) layout
        ch0 = arr[0] if arr.ndim == 2 else arr
        ch1 = arr[1] if arr.ndim == 2 else arr

        det_l = dominant_frequency(ch0, 22050)
        det_r = dominant_frequency(ch1, 22050)

        assert abs(det_l - freq_l) < 2.0, (
            f"Left channel: expected {freq_l} Hz, got {det_l} Hz"
        )
        assert abs(det_r - freq_r) < 2.0, (
            f"Right channel: expected {freq_r} Hz, got {det_r} Hz"
        )


# ---------------------------------------------------------------------------
# 4. Channel independence
# ---------------------------------------------------------------------------


class TestChannelIndependence:
    """Stereo result should match two independent mono resamples."""

    @pytest.mark.parametrize("quality", ALL_QUALITIES)
    def test_stereo_matches_independent_mono(self, quality):
        freq_l, freq_r = 300.0, 700.0
        n = int(1.0 * 44100)
        t = np.linspace(0, 1.0, n, endpoint=False)
        l = 0.5 * np.sin(2 * np.pi * freq_l * t)
        r = 0.5 * np.sin(2 * np.pi * freq_r * t)

        stereo = aus.AudioSamples.new_multi(
            np.vstack([l, r]).astype(np.float64), sample_rate=44100
        )
        mono_l = aus.AudioSamples.new_mono(l.astype(np.float64), sample_rate=44100)
        mono_r = aus.AudioSamples.new_mono(r.astype(np.float64), sample_rate=44100)

        stereo_rs = stereo.resample(target_sample_rate=22050, quality=quality)
        mono_l_rs = mono_l.resample(target_sample_rate=22050, quality=quality)
        mono_r_rs = mono_r.resample(target_sample_rate=22050, quality=quality)

        stereo_arr = stereo_rs.to_numpy()
        l_ref = mono_l_rs.to_numpy().ravel()
        r_ref = mono_r_rs.to_numpy().ravel()

        np.testing.assert_allclose(
            stereo_arr[0],
            l_ref,
            atol=1e-5,
            err_msg=f"{quality}: stereo left channel differs from mono resample",
        )
        np.testing.assert_allclose(
            stereo_arr[1],
            r_ref,
            atol=1e-5,
            err_msg=f"{quality}: stereo right channel differs from mono resample",
        )


# ---------------------------------------------------------------------------
# 5. Round-trip fidelity
# ---------------------------------------------------------------------------


class TestRoundTrip:
    """
    Downsample then upsample (or vice versa) should recover the original
    within the band that both rates support.
    """

    @pytest.mark.parametrize(
        "quality", [ResamplingQuality.medium, ResamplingQuality.high]
    )
    def test_roundtrip_mono_down_up(self, quality):
        """Down 44100→22050, then up 22050→44100 preserves the target frequency."""
        freq = 440.0  # well below both Nyquists
        audio = make_sine(freq, 2.0, 44100)
        down = audio.resample(target_sample_rate=22050, quality=quality)
        up = down.resample(target_sample_rate=44100, quality=quality)

        # Sinc filters introduce a group delay transient at both ends.
        # Trim 2000 samples (~45 ms) from each edge to avoid the boundary
        # region before analysing frequency content.
        samples = up.to_numpy().ravel()
        trim = 2000
        core = samples[trim:-trim]

        # Dominant frequency should still be 440 Hz
        detected = dominant_frequency(core, 44100)
        assert abs(detected - freq) < 2.0, (
            f"{quality} round-trip: expected {freq} Hz dominant, got {detected} Hz"
        )

        # Target frequency should hold > 95% of spectral power (pure sine)
        spectrum = np.abs(np.fft.rfft(core))
        peak_power = float(np.max(spectrum) ** 2)
        total_power = float(np.sum(spectrum**2))
        assert peak_power / total_power > 0.95, (
            f"{quality} round-trip SNR too low: {peak_power / total_power:.3f} (expected > 0.95)"
        )


# ---------------------------------------------------------------------------
# 6. Energy conservation
# ---------------------------------------------------------------------------


class TestEnergyConservation:
    @pytest.mark.parametrize("quality", ALL_QUALITIES)
    def test_rms_preserved_within_10pct(self, quality):
        audio = make_sine(440.0, 2.0, 44100)
        result = audio.resample(target_sample_rate=22050, quality=quality)

        rms_in = float(np.sqrt(np.mean(audio.to_numpy() ** 2)))
        rms_out = float(np.sqrt(np.mean(result.to_numpy() ** 2)))

        assert abs(rms_out - rms_in) / rms_in < 0.10, (
            f"{quality}: RMS changed by more than 10%: {rms_in:.4f} → {rms_out:.4f}"
        )
