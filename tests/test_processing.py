"""
Tests for processing operations - verifying audio_python transformations.

These tests verify that:
1. Normalization produces expected results
2. Scaling works correctly
3. Clipping behaves as expected
4. DC removal works correctly
5. Reverse operation is correct

API notes:
- Type conversion: as_f32, as_f64, as_i16, as_i32 (not to_*)
- Cast: cast_as_f32, cast_as_f64, etc.
- Statistics: peak() not abs_max(), std_dev() not std(), variance() not var()
"""

import pytest
import numpy as np

import audio_python as aus

# Tolerance constants
TIGHT_RTOL = 1e-6
TIGHT_ATOL = 1e-7
STANDARD_RTOL = 1e-5
STANDARD_ATOL = 1e-6


def assert_arrays_close(actual, expected, rtol=STANDARD_RTOL, atol=STANDARD_ATOL, msg=""):
    """Assert two arrays are close within tolerance."""
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    if actual.shape != expected.shape:
        raise AssertionError(f"Shape mismatch: {actual.shape} vs {expected.shape}. {msg}")
    if not np.allclose(actual, expected, rtol=rtol, atol=atol):
        diff = np.abs(actual - expected)
        max_diff = np.max(diff)
        raise AssertionError(f"Arrays not close. {msg}\n  Max difference: {max_diff}\n  Tolerance: rtol={rtol}, atol={atol}")


class TestNormalization:
    """Tests for audio normalization."""

    def test_normalize_minmax_range(self, sine_wave_mono):
        """Min-max normalization produces [-1, 1] range."""
        aus_audio, _, _ = sine_wave_mono
        
        # In-place normalization
        aus_audio.normalize(0.0, 1.0, "minmax")
        
        data = aus_audio.to_numpy()
        
        assert data.min() >= -1.0 - TIGHT_ATOL
        assert data.max() <= 1.0 + TIGHT_ATOL
        # After normalization, max or min should touch bounds
        assert abs(data.max() - 1.0) < TIGHT_ATOL or abs(data.min() + 1.0) < TIGHT_ATOL

    def test_normalize_zscore_statistics(self, white_noise):
        """Z-score normalization produces mean≈0, std≈1."""
        aus_audio, _ = white_noise
        
        aus_audio.normalize(0.0, 1.0, "zscore")
        
        data = aus_audio.to_numpy()
        
        assert abs(np.mean(data)) < 0.01, \
            f"Z-score normalized mean should be ~0, got {np.mean(data)}"
        assert abs(np.std(data) - 1.0) < 0.01, \
            f"Z-score normalized std should be ~1, got {np.std(data)}"

    def test_normalize_mean_centering(self, dc_offset_signal):
        """Mean normalization centers data around zero."""
        aus_audio, np_data, dc_offset = dc_offset_signal
        
        original_mean = aus_audio.mean()
        aus_audio.normalize(0.0, 1.0, "mean")
        new_mean = aus_audio.mean()
        
        assert abs(new_mean) < 0.01, \
            f"Mean-normalized signal should have mean ~0, got {new_mean}"

    def test_normalize_preserves_shape(self, sine_wave_stereo):
        """Normalization preserves audio shape."""
        aus_audio, np_data, params = sine_wave_stereo
        
        original_shape = aus_audio.shape
        original_sr = aus_audio.sample_rate
        original_channels = aus_audio.channels
        
        aus_audio.normalize(0.0, 1.0, "minmax")
        
        assert aus_audio.shape == original_shape
        assert aus_audio.sample_rate == original_sr
        assert aus_audio.channels == original_channels


class TestScaling:
    """Tests for amplitude scaling."""

    def test_scale_doubles_amplitude(self, sine_wave_mono):
        """Scaling by 2 doubles amplitude."""
        aus_audio, np_data, _ = sine_wave_mono
        
        original_max = aus_audio.peak()
        aus_audio.scale(2.0)
        new_max = aus_audio.peak()
        
        assert abs(new_max - 2.0 * original_max) < TIGHT_ATOL, \
            f"Expected max {2.0 * original_max}, got {new_max}"

    def test_scale_halves_amplitude(self, sine_wave_mono):
        """Scaling by 0.5 halves amplitude."""
        aus_audio, np_data, _ = sine_wave_mono
        
        original_max = aus_audio.peak()
        aus_audio.scale(0.5)
        new_max = aus_audio.peak()
        
        assert abs(new_max - 0.5 * original_max) < TIGHT_ATOL

    def test_scale_by_zero(self, sine_wave_mono):
        """Scaling by 0 produces silence."""
        aus_audio, _, _ = sine_wave_mono
        
        aus_audio.scale(0.0)
        
        data = aus_audio.to_numpy()
        assert np.all(data == 0), "Scaling by 0 should produce silence"

    def test_scale_negative_inverts(self, sample_rate):
        """Scaling by -1 inverts the signal."""
        data = np.array([1.0, 2.0, 3.0, -1.0, -2.0])
        aus_audio = aus.AudioSamples.new_mono(data, sample_rate=sample_rate)
        
        aus_audio.scale(-1.0)
        
        expected = np.array([-1.0, -2.0, -3.0, 1.0, 2.0])
        assert_arrays_close(aus_audio.to_numpy(), expected, rtol=TIGHT_RTOL, atol=TIGHT_ATOL)

    def test_scale_matches_numpy_multiply(self, sine_wave_mono):
        """Scaling matches numpy multiplication."""
        aus_audio, np_data, _ = sine_wave_mono
        
        scale_factor = 1.5
        aus_audio.scale(scale_factor)
        
        expected = np_data * scale_factor
        
        assert_arrays_close(
            aus_audio.to_numpy(), expected,
            rtol=TIGHT_RTOL, atol=TIGHT_ATOL,
            msg="Scale doesn't match numpy multiply"
        )


class TestClipping:
    """Tests for audio clipping."""

    def test_clip_symmetric(self, sine_wave_mono):
        """Symmetric clipping limits to [-threshold, threshold]."""
        aus_audio, _, _ = sine_wave_mono
        
        threshold = 0.3
        aus_audio.clip(-threshold, threshold)
        
        data = aus_audio.to_numpy()
        
        assert data.min() >= -threshold - TIGHT_ATOL
        assert data.max() <= threshold + TIGHT_ATOL

    def test_clip_asymmetric(self, sample_rate):
        """Asymmetric clipping uses different upper/lower bounds."""
        data = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        aus_audio = aus.AudioSamples.new_mono(data, sample_rate=sample_rate)
        
        aus_audio.clip(-0.5, 1.5)
        
        result = aus_audio.to_numpy()
        expected = np.array([-0.5, -0.5, 0.0, 1.0, 1.5])
        
        assert_arrays_close(result, expected, rtol=TIGHT_RTOL, atol=TIGHT_ATOL)

    def test_clip_matches_numpy_clip(self, sine_wave_mono):
        """Clipping matches numpy.clip()."""
        aus_audio, np_data, _ = sine_wave_mono
        
        low, high = -0.3, 0.4
        aus_audio.clip(low, high)
        
        expected = np.clip(np_data, low, high)
        
        assert_arrays_close(
            aus_audio.to_numpy(), expected,
            rtol=TIGHT_RTOL, atol=TIGHT_ATOL,
            msg="Clip doesn't match numpy.clip"
        )

    def test_clip_hard_limiter(self, sample_rate):
        """Clipping acts as hard limiter at ±1."""
        data = np.array([-2.0, -1.5, 0.0, 1.5, 2.0])
        aus_audio = aus.AudioSamples.new_mono(data, sample_rate=sample_rate)
        
        aus_audio.clip(-1.0, 1.0)
        
        result = aus_audio.to_numpy()
        expected = np.array([-1.0, -1.0, 0.0, 1.0, 1.0])
        
        assert_arrays_close(result, expected, rtol=TIGHT_RTOL, atol=TIGHT_ATOL)


class TestDCRemoval:
    """Tests for DC offset removal."""

    def test_remove_dc_offset(self, dc_offset_signal):
        """DC removal produces mean ≈ 0."""
        aus_audio, np_data, dc_offset = dc_offset_signal
        
        original_mean = aus_audio.mean()
        assert abs(original_mean - dc_offset) < 0.01, "Test signal should have DC offset"
        
        aus_audio.remove_dc_offset()
        
        new_mean = aus_audio.mean()
        assert abs(new_mean) < 0.001, \
            f"After DC removal, mean should be ~0, got {new_mean}"

    def test_remove_dc_constant_signal(self, sample_rate):
        """DC removal on constant signal produces silence."""
        dc_value = 0.5
        data = np.ones(1000) * dc_value
        aus_audio = aus.AudioSamples.new_mono(data, sample_rate=sample_rate)
        
        aus_audio.remove_dc_offset()
        
        result = aus_audio.to_numpy()
        assert np.allclose(result, 0, atol=TIGHT_ATOL), \
            "DC removal on constant signal should produce zeros"

    def test_remove_dc_preserves_ac(self, sample_rate):
        """DC removal preserves AC component shape."""
        num_samples = 4410
        t = np.linspace(0, 0.1, num_samples, endpoint=False)
        
        dc_offset = 0.3
        ac_amplitude = 0.5
        frequency = 440.0
        
        # Signal = DC + AC
        ac_component = ac_amplitude * np.sin(2 * np.pi * frequency * t)
        data = dc_offset + ac_component
        
        aus_audio = aus.AudioSamples.new_mono(data, sample_rate=sample_rate)
        aus_audio.remove_dc_offset()
        
        result = aus_audio.to_numpy()
        
        # Result should match AC component
        assert_arrays_close(
            result, ac_component,
            rtol=1e-4, atol=1e-5,
            msg="DC removal should preserve AC component"
        )


class TestReverse:
    """Tests for audio reversal."""

    def test_reverse_returns_reversed(self, sample_rate):
        """reverse() returns reversed copy."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        aus_audio = aus.AudioSamples.new_mono(data, sample_rate=sample_rate)
        
        reversed_audio = aus_audio.reverse()
        
        expected = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        assert_arrays_close(reversed_audio.to_numpy(), expected, rtol=TIGHT_RTOL, atol=TIGHT_ATOL)

    def test_reverse_in_place(self, sample_rate):
        """reverse_in_place() modifies original."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        aus_audio = aus.AudioSamples.new_mono(data, sample_rate=sample_rate)
        
        aus_audio.reverse_in_place()
        
        expected = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        assert_arrays_close(aus_audio.to_numpy(), expected, rtol=TIGHT_RTOL, atol=TIGHT_ATOL)

    def test_reverse_double_is_identity(self, sine_wave_mono):
        """Reversing twice returns original."""
        aus_audio, np_data, _ = sine_wave_mono
        
        aus_audio.reverse_in_place()
        aus_audio.reverse_in_place()
        
        assert_arrays_close(
            aus_audio.to_numpy(), np_data,
            rtol=TIGHT_RTOL, atol=TIGHT_ATOL,
            msg="Double reverse should be identity"
        )

    def test_reverse_preserves_metadata(self, sine_wave_mono):
        """Reverse preserves sample rate and duration."""
        aus_audio, _, params = sine_wave_mono
        
        reversed_audio = aus_audio.reverse()
        
        assert reversed_audio.sample_rate == params["sample_rate"]
        assert reversed_audio.samples_per_channel() == params["num_samples"]

    def test_reverse_stereo(self, sine_wave_stereo):
        """Reverse works on stereo audio."""
        aus_audio, np_data, _ = sine_wave_stereo
        
        reversed_audio = aus_audio.reverse()
        
        # Each channel should be reversed independently
        expected = np_data[:, ::-1]  # Reverse along sample axis
        
        assert_arrays_close(
            reversed_audio.to_numpy(), expected,
            rtol=TIGHT_RTOL, atol=TIGHT_ATOL,
            msg="Stereo reverse incorrect"
        )


class TestTrim:
    """Tests for audio trimming."""

    def test_trim_by_time(self, sine_wave_mono):
        """Trim by time extracts correct segment."""
        aus_audio, np_data, params = sine_wave_mono
        
        start_time = 0.25  # 250ms
        end_time = 0.75    # 750ms
        
        trimmed = aus_audio.trim(start_time, end_time)
        
        # Calculate expected samples
        sr = params["sample_rate"]
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        expected_samples = end_sample - start_sample
        
        assert trimmed.samples_per_channel() == expected_samples, \
            f"Expected {expected_samples} samples, got {trimmed.samples_per_channel}"

    def test_trim_preserves_sample_rate(self, sine_wave_mono):
        """Trim preserves sample rate."""
        aus_audio, _, params = sine_wave_mono
        
        trimmed = aus_audio.trim(0.1, 0.5)
        
        assert trimmed.sample_rate == params["sample_rate"]

    def test_trim_data_correct(self, sample_rate):
        """Trim extracts correct data values."""
        # Create predictable data: values = sample index
        num_samples = 44100  # 1 second
        data = np.arange(num_samples, dtype=np.float64) / num_samples
        aus_audio = aus.AudioSamples.new_mono(data, sample_rate=sample_rate)
        
        # Trim to 0.5-0.6 seconds
        trimmed = aus_audio.trim(0.5, 0.6)
        
        start_idx = int(0.5 * sample_rate)
        end_idx = int(0.6 * sample_rate)
        expected = data[start_idx:end_idx]
        
        assert_arrays_close(
            trimmed.to_numpy(), expected,
            rtol=TIGHT_RTOL, atol=TIGHT_ATOL,
            msg="Trim extracted wrong data"
        )


class TestTypeConversion:
    """Tests for sample type conversion."""

    def test_to_f32_from_f64(self, sine_wave_mono):
        """Convert f64 to f32."""
        aus_audio, np_data, _ = sine_wave_mono
        
        f32_audio = aus_audio.as_f32()
        
        assert f32_audio.dtype == "f32"
        # Values should be close (f32 has less precision)
        assert_arrays_close(
            f32_audio.to_numpy(), np_data.astype(np.float32),
            rtol=1e-5, atol=1e-6
        )

    def test_to_i16_scaling(self, sample_rate):
        """Convert float to i16 with proper scaling."""
        # Data in float range [-1, 1]
        data = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
        aus_audio = aus.AudioSamples.new_mono(data, sample_rate=sample_rate)
        
        i16_audio = aus_audio.as_i16()
        
        print(f"DEBUG: dtype attribute: {i16_audio.dtype}, type: {type(i16_audio.dtype)}")
        print(f"DEBUG: dir: {dir(i16_audio)}")
        
        assert i16_audio.dtype == "i16"
        
        # Check scaling (float 1.0 -> i16 32767, -1.0 -> -32768)
        i16_data = i16_audio.to_numpy()
        
        # Approximate expected values (may vary by 1 due to rounding)
        assert abs(i16_data[0] - (-32768)) <= 1  # -1.0
        assert abs(i16_data[4] - 32767) <= 1     # 1.0
        assert i16_data[2] == 0                   # 0.0

    def test_roundtrip_f64_i16_f64(self, sine_wave_mono):
        """Round-trip f64 -> i16 -> f64 preserves signal."""
        aus_audio, np_data, _ = sine_wave_mono
        
        i16_audio = aus_audio.as_i16()
        back_to_f64 = i16_audio.as_f64()
        
        # Some precision loss is expected due to quantization
        assert_arrays_close(
            back_to_f64.to_numpy(), np_data,
            rtol=1e-3, atol=1e-4,  # Looser tolerance for quantization
            msg="Round-trip f64->i16->f64 lost too much precision"
        )

    def test_cast_vs_convert(self, sample_rate):
        """cast_as_* does raw cast, as_* does audio-aware scaling."""
        data = np.array([0.5, -0.5], dtype=np.float64)
        aus_audio = aus.AudioSamples.new_mono(data, sample_rate=sample_rate)
        
        # as_i16 should scale: 0.5 -> ~16384
        converted = aus_audio.as_i16()
        converted_data = converted.to_numpy()
        
        # cast_as_i16 should truncate: 0.5 -> 0
        cast_audio = aus_audio.cast_as_i16()
        cast_data = cast_audio.to_numpy()
        
        # These should be different
        assert converted_data[0] != cast_data[0], \
            "as_i16 and cast_as_i16 should produce different results"
        
        # Converted should have scaled values
        assert abs(converted_data[0]) > 10000, \
            f"as_i16(0.5) should be ~16384, got {converted_data[0]}"


class TestArithmeticOperations:
    """Tests for arithmetic operations via Python operators."""

    def test_add_audio_samples(self, sample_rate):
        """Adding two AudioSamples works correctly."""
        data1 = np.array([1.0, 2.0, 3.0])
        data2 = np.array([0.5, 0.5, 0.5])
        
        audio1 = aus.AudioSamples.new_mono(data1, sample_rate=sample_rate)
        audio2 = aus.AudioSamples.new_mono(data2, sample_rate=sample_rate)
        
        result = audio1 + audio2
        
        expected = np.array([1.5, 2.5, 3.5])
        assert_arrays_close(result.to_numpy(), expected, rtol=TIGHT_RTOL, atol=TIGHT_ATOL)

    def test_multiply_by_scalar(self, sample_rate):
        """Multiplying AudioSamples by scalar works."""
        data = np.array([1.0, 2.0, 3.0])
        audio = aus.AudioSamples.new_mono(data, sample_rate=sample_rate)
        
        result = audio * 2.0
        
        expected = np.array([2.0, 4.0, 6.0])
        assert_arrays_close(result.to_numpy(), expected, rtol=TIGHT_RTOL, atol=TIGHT_ATOL)

    def test_divide_by_scalar(self, sample_rate):
        """Dividing AudioSamples by scalar works."""
        data = np.array([2.0, 4.0, 6.0])
        audio = aus.AudioSamples.new_mono(data, sample_rate=sample_rate)
        
        result = audio / 2.0
        
        expected = np.array([1.0, 2.0, 3.0])
        assert_arrays_close(result.to_numpy(), expected, rtol=TIGHT_RTOL, atol=TIGHT_ATOL)
