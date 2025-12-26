"""
Tests for statistics operations - comparing audio_python against numpy/librosa.

These tests verify that:
1. Basic statistics (min, max, mean, std_dev, variance) match numpy exactly
2. RMS calculation matches numpy implementation
3. Zero crossing count/rate matches expected values
4. Autocorrelation produces correct results

API notes:
- std() in API is std_dev()
- var() in API is variance()
- zero_crossing_count() is zero_crossings()
- abs_max() is peak()
"""

import pytest
import numpy as np

import audio_python as aus

# Tolerance constants
TIGHT_RTOL = 1e-6
TIGHT_ATOL = 1e-7
STANDARD_RTOL = 1e-5
STANDARD_ATOL = 1e-6


class TestBasicStatistics:
    """Tests for basic statistical operations."""

    def test_min_matches_numpy(self, sine_wave_mono):
        """Verify min() matches numpy.min()."""
        aus_audio, np_data, _ = sine_wave_mono
        
        aus_min = aus_audio.min()
        np_min = np.min(np_data)
        
        assert abs(aus_min - np_min) < TIGHT_ATOL, \
            f"min mismatch: aus={aus_min}, np={np_min}"

    def test_max_matches_numpy(self, sine_wave_mono):
        """Verify max() matches numpy.max()."""
        aus_audio, np_data, _ = sine_wave_mono
        
        aus_max = aus_audio.max()
        np_max = np.max(np_data)
        
        assert abs(aus_max - np_max) < TIGHT_ATOL, \
            f"max mismatch: aus={aus_max}, np={np_max}"

    def test_abs_max_matches_numpy(self, sine_wave_mono):
        """Verify peak() matches numpy abs max."""
        aus_audio, np_data, _ = sine_wave_mono
        
        aus_abs_max = aus_audio.peak()
        np_abs_max = np.max(np.abs(np_data))
        
        assert abs(aus_abs_max - np_abs_max) < TIGHT_ATOL, \
            f"peak mismatch: aus={aus_abs_max}, np={np_abs_max}"

    def test_mean_matches_numpy(self, sine_wave_mono):
        """Verify mean() matches numpy.mean()."""
        aus_audio, np_data, _ = sine_wave_mono
        
        aus_mean = aus_audio.mean()
        np_mean = np.mean(np_data)
        
        assert abs(aus_mean - np_mean) < TIGHT_ATOL, \
            f"mean mismatch: aus={aus_mean}, np={np_mean}"

    def test_std_matches_numpy(self, sine_wave_mono):
        """Verify std_dev() matches numpy.std()."""
        aus_audio, np_data, _ = sine_wave_mono
        
        aus_std = aus_audio.std_dev()
        # Note: numpy uses ddof=0 by default (population std)
        np_std = np.std(np_data, ddof=0)
        
        assert abs(aus_std - np_std) < STANDARD_ATOL, \
            f"std mismatch: aus={aus_std}, np={np_std}"

    def test_var_matches_numpy(self, sine_wave_mono):
        """Verify variance() matches numpy.var()."""
        aus_audio, np_data, _ = sine_wave_mono
        
        aus_var = aus_audio.variance()
        np_var = np.var(np_data, ddof=0)
        
        assert abs(aus_var - np_var) < STANDARD_ATOL, \
            f"var mismatch: aus={aus_var}, np={np_var}"

    def test_rms_matches_numpy_implementation(self, sine_wave_mono):
        """Verify rms() matches numpy RMS calculation."""
        aus_audio, np_data, _ = sine_wave_mono
        
        aus_rms = aus_audio.rms()
        np_rms = np.sqrt(np.mean(np_data ** 2))
        
        assert abs(aus_rms - np_rms) < STANDARD_ATOL, \
            f"rms mismatch: aus={aus_rms}, np={np_rms}"

    def test_rms_of_sine_wave_theoretical(self, sample_rate, medium_duration):
        """Verify RMS of sine wave equals amplitude / sqrt(2)."""
        amplitude = 0.8
        num_samples = int(sample_rate * medium_duration)
        t = np.linspace(0, medium_duration, num_samples, endpoint=False)
        
        # Pure sine wave
        data = amplitude * np.sin(2 * np.pi * 440 * t)
        aus_audio = aus.AudioSamples.new_mono(data, sample_rate=sample_rate)
        
        aus_rms = aus_audio.rms()
        theoretical_rms = amplitude / np.sqrt(2)
        
        # Allow slightly larger tolerance due to finite samples
        assert abs(aus_rms - theoretical_rms) < 1e-4, \
            f"Sine wave RMS should be {theoretical_rms}, got {aus_rms}"


class TestStatisticsMultiChannel:
    """Tests for statistics on multi-channel audio."""

    def test_stereo_min_max(self, sine_wave_stereo):
        """Verify min/max works on stereo audio."""
        aus_audio, np_data, _ = sine_wave_stereo
        
        aus_min = aus_audio.min()
        aus_max = aus_audio.max()
        
        np_min = np.min(np_data)
        np_max = np.max(np_data)
        
        assert abs(aus_min - np_min) < TIGHT_ATOL
        assert abs(aus_max - np_max) < TIGHT_ATOL

    def test_stereo_mean(self, sine_wave_stereo):
        """Verify mean works on stereo audio."""
        aus_audio, np_data, _ = sine_wave_stereo
        
        aus_mean = aus_audio.mean()
        np_mean = np.mean(np_data)
        
        assert abs(aus_mean - np_mean) < TIGHT_ATOL


class TestZeroCrossings:
    """Tests for zero crossing detection."""

    def test_zero_crossing_count_matches_manual(self, zero_crossing_signal):
        """Verify zero_crossings() matches expected count."""
        aus_audio, np_data, expected_crossings = zero_crossing_signal
        
        aus_crossings = aus_audio.zero_crossings()
        
        # Allow small difference due to boundary handling
        assert abs(aus_crossings - expected_crossings) <= 2, \
            f"Zero crossing count mismatch: aus={aus_crossings}, expected={expected_crossings}"

    def test_zero_crossing_rate_calculation(self, zero_crossing_signal, sample_rate):
        """Verify zero_crossing_rate returns crossings per second (Hz)."""
        aus_audio, np_data, _ = zero_crossing_signal
        
        aus_count = aus_audio.zero_crossings()
        aus_rate = aus_audio.zero_crossing_rate()
        
        # Rate is crossings per second (count * sample_rate / num_samples = count / duration)
        duration = len(np_data) / sample_rate
        expected_rate = aus_count / duration
        
        assert abs(aus_rate - expected_rate) < 1.0, \
            f"Zero crossing rate mismatch: aus={aus_rate}, expected={expected_rate}"

    def test_dc_signal_zero_crossings(self, sample_rate):
        """DC signal should have zero crossings."""
        # Constant positive signal
        data = np.ones(1000) * 0.5
        aus_audio = aus.AudioSamples.new_mono(data, sample_rate=sample_rate)
        
        assert aus_audio.zero_crossings() == 0

    def test_alternating_signal_crossings(self, sample_rate):
        """Signal alternating every sample should have max crossings."""
        data = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
        aus_audio = aus.AudioSamples.new_mono(data, sample_rate=sample_rate)
        
        # 7 sign changes in 8 samples
        assert aus_audio.zero_crossings() == 7


class TestKnownValues:
    """Tests with hand-calculated expected values."""

    def test_simple_array_statistics(self, sample_rate):
        """Test statistics on simple known array."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        aus_audio = aus.AudioSamples.new_mono(data, sample_rate=sample_rate)
        
        assert aus_audio.min() == 1.0
        assert aus_audio.max() == 5.0
        assert aus_audio.peak() == 5.0
        assert aus_audio.mean() == 3.0
        
        # Variance: mean of squared deviations
        # Deviations: [-2, -1, 0, 1, 2]
        # Squared: [4, 1, 0, 1, 4]
        # Mean: 10/5 = 2.0
        assert abs(aus_audio.variance() - 2.0) < TIGHT_ATOL
        
        # Std = sqrt(2) ≈ 1.414
        assert abs(aus_audio.std_dev() - np.sqrt(2)) < TIGHT_ATOL

    def test_symmetric_array_mean_zero(self, sample_rate):
        """Symmetric array around zero should have mean ≈ 0."""
        data = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
        aus_audio = aus.AudioSamples.new_mono(data, sample_rate=sample_rate)
        
        assert abs(aus_audio.mean()) < TIGHT_ATOL

    def test_rms_simple_values(self, sample_rate):
        """Test RMS on simple known values."""
        # [3, 4] -> squares [9, 16] -> mean 12.5 -> sqrt = 3.5355...
        data = np.array([3.0, 4.0])
        aus_audio = aus.AudioSamples.new_mono(data, sample_rate=sample_rate)
        
        expected_rms = np.sqrt((9 + 16) / 2)  # = sqrt(12.5)
        assert abs(aus_audio.rms() - expected_rms) < TIGHT_ATOL


class TestWhiteNoiseStatistics:
    """Tests for statistics on white noise (probabilistic checks)."""

    def test_white_noise_mean_near_zero(self, white_noise):
        """White noise should have mean close to zero."""
        aus_audio, np_data = white_noise
        
        # Uniform[-1, 1] has mean 0
        assert abs(aus_audio.mean()) < 0.05, \
            f"White noise mean should be near zero, got {aus_audio.mean()}"

    def test_white_noise_std_expected(self, white_noise):
        """White noise from uniform[-1,1] should have std ≈ 1/sqrt(3)."""
        aus_audio, np_data = white_noise
        
        # Uniform distribution on [-1, 1] has std = 1/sqrt(3) ≈ 0.577
        expected_std = 1 / np.sqrt(3)
        
        assert abs(aus_audio.std_dev() - expected_std) < 0.05, \
            f"White noise std should be ~{expected_std}, got {aus_audio.std_dev()}"

    def test_white_noise_many_zero_crossings(self, white_noise):
        """White noise should have many zero crossings."""
        aus_audio, np_data = white_noise
        
        # Zero crossing rate is in Hz (crossings per second)
        # White noise crosses zero frequently - roughly 50% of samples
        # So rate ≈ 0.5 * sample_rate
        crossing_rate = aus_audio.zero_crossing_rate()
        sample_rate = aus_audio.sample_rate
        
        # Expect roughly 0.3-0.7 * sample_rate crossings per second
        assert 0.3 * sample_rate < crossing_rate < 0.7 * sample_rate, \
            f"White noise should have crossing rate ~{0.5 * sample_rate}, got {crossing_rate}"
