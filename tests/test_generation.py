"""
Tests for signal generation - verifying audio_python.generation produces correct waveforms.

These tests verify that:
1. Generated signals have correct frequency (via FFT analysis)
2. Generated signals have correct amplitude
3. Generated signals have correct sample count and duration
4. All sample types (f32, f64) work correctly

API notes:
- Functions are named with _wave suffix: sine_wave, cosine_wave, square_wave, etc.
- Duration parameter is duration_secs (not duration)
- Chirp uses f0/f1 for start/end frequency
"""

import pytest
import numpy as np

import audio_python as aus
from audio_python import generation as gen

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


class TestSineWaveGeneration:
    """Tests for sine wave generation."""

    def test_sine_correct_sample_count(self, sample_rate, medium_duration):
        """Generated sine has correct number of samples."""
        expected_samples = int(sample_rate * medium_duration)
        
        audio = gen.sine_wave(
            frequency=440.0,
            duration_secs=medium_duration,
            sample_rate=sample_rate,
            amplitude=0.5
        )
        
        assert audio.samples_per_channel() == expected_samples, \
            f"Expected {expected_samples} samples, got {audio.samples_per_channel}"

    def test_sine_correct_sample_rate(self, sample_rate, medium_duration):
        """Generated sine has correct sample rate."""
        audio = gen.sine_wave(
            frequency=440.0,
            duration_secs=medium_duration,
            sample_rate=sample_rate,
            amplitude=0.5
        )
        
        assert audio.sample_rate == sample_rate

    def test_sine_correct_amplitude(self, sample_rate, medium_duration):
        """Generated sine has correct amplitude."""
        amplitude = 0.7
        
        audio = gen.sine_wave(
            frequency=440.0,
            duration_secs=medium_duration,
            sample_rate=sample_rate,
            amplitude=amplitude
        )
        
        data = audio.to_numpy()
        actual_amplitude = np.max(np.abs(data))
        
        assert abs(actual_amplitude - amplitude) < 0.01, \
            f"Expected amplitude {amplitude}, got {actual_amplitude}"

    def test_sine_frequency_via_fft(self, sample_rate, medium_duration):
        """Generated sine has correct frequency (verified via FFT)."""
        test_freq = 1000.0
        
        audio = gen.sine_wave(
            frequency=test_freq,
            duration_secs=medium_duration,
            sample_rate=sample_rate,
            amplitude=0.5
        )
        
        data = audio.to_numpy()
        fft = np.fft.rfft(data)
        magnitudes = np.abs(fft)
        
        # Find peak frequency
        peak_bin = np.argmax(magnitudes)
        freq_resolution = sample_rate / len(data)
        peak_freq = peak_bin * freq_resolution
        
        assert abs(peak_freq - test_freq) < freq_resolution * 2, \
            f"FFT shows peak at {peak_freq} Hz, expected {test_freq} Hz"

    def test_sine_matches_numpy_implementation(self, sample_rate, short_duration):
        """Generated sine matches numpy sin() implementation."""
        frequency = 440.0
        amplitude = 0.5
        num_samples = int(sample_rate * short_duration)
        
        audio = gen.sine_wave(
            frequency=frequency,
            duration_secs=short_duration,
            sample_rate=sample_rate,
            amplitude=amplitude
        )
        
        # Generate reference with numpy
        t = np.linspace(0, short_duration, num_samples, endpoint=False)
        expected = amplitude * np.sin(2 * np.pi * frequency * t)
        
        assert_arrays_close(
            audio.to_numpy(), expected,
            rtol=STANDARD_RTOL, atol=STANDARD_ATOL,
            msg="Sine wave doesn't match numpy implementation"
        )

    def test_sine_is_mono(self, sample_rate, short_duration):
        """Generated sine is mono."""
        audio = gen.sine_wave(
            frequency=440.0,
            duration_secs=short_duration,
            sample_rate=sample_rate,
            amplitude=0.5
        )
        
        assert audio.is_mono()
        assert audio.channels == 1


class TestCosineWaveGeneration:
    """Tests for cosine wave generation."""

    def test_cosine_correct_amplitude(self, sample_rate, medium_duration):
        """Generated cosine has correct amplitude."""
        amplitude = 0.6
        
        audio = gen.cosine_wave(
            frequency=440.0,
            duration_secs=medium_duration,
            sample_rate=sample_rate,
            amplitude=amplitude
        )
        
        data = audio.to_numpy()
        actual_amplitude = np.max(np.abs(data))
        
        assert abs(actual_amplitude - amplitude) < 0.01

    def test_cosine_starts_at_amplitude(self, sample_rate, short_duration):
        """Cosine wave starts at amplitude (cos(0) = 1)."""
        amplitude = 0.8
        
        audio = gen.cosine_wave(
            frequency=440.0,
            duration_secs=short_duration,
            sample_rate=sample_rate,
            amplitude=amplitude
        )
        
        data = audio.to_numpy()
        first_sample = data[0]
        
        assert abs(first_sample - amplitude) < 0.01, \
            f"Cosine should start at {amplitude}, got {first_sample}"

    def test_cosine_matches_numpy_implementation(self, sample_rate, short_duration):
        """Generated cosine matches numpy cos() implementation."""
        frequency = 440.0
        amplitude = 0.5
        num_samples = int(sample_rate * short_duration)
        
        audio = gen.cosine_wave(
            frequency=frequency,
            duration_secs=short_duration,
            sample_rate=sample_rate,
            amplitude=amplitude
        )
        
        t = np.linspace(0, short_duration, num_samples, endpoint=False)
        expected = amplitude * np.cos(2 * np.pi * frequency * t)
        
        assert_arrays_close(
            audio.to_numpy(), expected,
            rtol=STANDARD_RTOL, atol=STANDARD_ATOL,
            msg="Cosine wave doesn't match numpy implementation"
        )

    def test_sine_cosine_phase_difference(self, sample_rate, short_duration):
        """Sine and cosine should be 90 degrees out of phase."""
        frequency = 440.0
        amplitude = 0.5
        
        sine_audio = gen.sine_wave(
            frequency=frequency,
            duration_secs=short_duration,
            sample_rate=sample_rate,
            amplitude=amplitude
        )
        
        cosine_audio = gen.cosine_wave(
            frequency=frequency,
            duration_secs=short_duration,
            sample_rate=sample_rate,
            amplitude=amplitude
        )
        
        sine_data = sine_audio.to_numpy()
        cosine_data = cosine_audio.to_numpy()
        
        # sin^2 + cos^2 = amplitude^2
        sum_squares = sine_data**2 + cosine_data**2
        expected = amplitude**2 * np.ones_like(sum_squares)
        
        assert_arrays_close(
            sum_squares, expected,
            rtol=1e-4, atol=1e-5,
            msg="sin^2 + cos^2 should equal amplitude^2"
        )


class TestNoiseGeneration:
    """Tests for noise generation."""

    def test_white_noise_correct_sample_count(self, sample_rate, short_duration):
        """Generated white noise has correct sample count."""
        expected_samples = int(sample_rate * short_duration)
        
        audio = gen.white_noise(
            duration_secs=short_duration,
            sample_rate=sample_rate,
            amplitude=0.5
        )
        
        assert audio.samples_per_channel() == expected_samples

    def test_white_noise_matches_numpy_uniform(self, sample_rate, short_duration):
        """White noise has correct statistical properties and is reproducible with seed."""
        amplitude = 0.5
        num_samples = int(sample_rate * short_duration)

        # Test reproducibility with seed
        audio1 = gen.white_noise(
            duration_secs=short_duration,
            sample_rate=sample_rate,
            amplitude=amplitude,
            seed=1234
        )
        audio2 = gen.white_noise(
            duration_secs=short_duration,
            sample_rate=sample_rate,
            amplitude=amplitude,
            seed=1234
        )
        data1 = audio1.to_numpy()
        data2 = audio2.to_numpy()
        assert_arrays_close(data1, data2, rtol=1e-15, atol=1e-15, msg="White noise should be reproducible with same seed")

        # Test statistical properties
        # For uniform distribution in [-amplitude, amplitude), mean should be 0
        # and std should be amplitude / sqrt(3)
        expected_std = amplitude / np.sqrt(3)
        assert abs(np.mean(data1)) < 0.05  # Mean should be close to 0
        assert abs(np.std(data1) - expected_std) < 0.05  # Std should match theoretical value

        # All values should be in [-amplitude, amplitude]
        assert np.all(data1 >= -amplitude)
        assert np.all(data1 < amplitude)

    def test_white_noise_statistics(self, sample_rate, medium_duration):
        """White noise has expected statistical properties."""
        amplitude = 1.0
        
        audio = gen.white_noise(
            duration_secs=medium_duration,
            sample_rate=sample_rate,
            amplitude=amplitude
        )
        
        data = audio.to_numpy()
        
        # Mean should be near zero
        assert abs(np.mean(data)) < 0.05, \
            f"White noise mean should be near 0, got {np.mean(data)}"
        
        # Std should be approximately amplitude / sqrt(3) for uniform
        expected_std = amplitude / np.sqrt(3)
        assert abs(np.std(data) - expected_std) < 0.1, \
            f"White noise std should be near {expected_std}, got {np.std(data)}"


class TestSquareWaveGeneration:
    """Tests for square wave generation."""

    def test_square_wave_matches_scipy(self, sample_rate, short_duration):
        """Square wave matches scipy.signal.square."""
        from scipy import signal
        amplitude = 0.5
        frequency = 440.0
        num_samples = int(sample_rate * short_duration)
        t = np.linspace(0, short_duration, num_samples, endpoint=False)
        expected = amplitude * signal.square(2 * np.pi * frequency * t)
        audio = gen.square_wave(
            frequency=frequency,
            duration_secs=short_duration,
            sample_rate=sample_rate,
            amplitude=amplitude
        )
        data = audio.to_numpy()
        assert_arrays_close(data, expected, rtol=1e-4, atol=1e-5, msg="Square wave does not match scipy.signal.square")

    def test_square_wave_frequency(self, sample_rate, medium_duration):
        """Square wave has correct fundamental frequency."""
        test_freq = 100.0  # Low frequency for clear cycles
        
        audio = gen.square_wave(
            frequency=test_freq,
            duration_secs=medium_duration,
            sample_rate=sample_rate,
            amplitude=0.5
        )
        
        # Count zero crossings to verify frequency
        data = audio.to_numpy()
        crossings = np.sum(np.abs(np.diff(np.sign(data))) > 0)
        
        # Each cycle has 2 crossings
        expected_cycles = test_freq * medium_duration
        expected_crossings = expected_cycles * 2
        
        assert abs(crossings - expected_crossings) < 4, \
            f"Expected ~{expected_crossings} crossings, got {crossings}"


class TestSawtoothWaveGeneration:
    """Tests for sawtooth wave generation."""

    def test_sawtooth_matches_scipy(self, sample_rate, short_duration):
        """Sawtooth wave matches scipy.signal.sawtooth."""
        from scipy import signal
        amplitude = 0.5
        frequency = 440.0
        num_samples = int(sample_rate * short_duration)
        t = np.linspace(0, short_duration, num_samples, endpoint=False)
        expected = amplitude * signal.sawtooth(2 * np.pi * frequency * t)
        audio = gen.sawtooth_wave(
            frequency=frequency,
            duration_secs=short_duration,
            sample_rate=sample_rate,
            amplitude=amplitude
        )
        data = audio.to_numpy()
        assert_arrays_close(data, expected, rtol=1e-4, atol=1e-5, msg="Sawtooth wave does not match scipy.signal.sawtooth")


class TestTriangleWaveGeneration:
    """Tests for triangle wave generation."""

    def test_triangle_matches_scipy(self, sample_rate, short_duration):
        """Triangle wave matches scipy.signal.sawtooth with width=0.5."""
        from scipy import signal
        amplitude = 0.5
        frequency = 440.0
        num_samples = int(sample_rate * short_duration)
        t = np.linspace(0, short_duration, num_samples, endpoint=False)
        expected = amplitude * signal.sawtooth(2 * np.pi * frequency * t, width=0.5)
        audio = gen.triangle_wave(
            frequency=frequency,
            duration_secs=short_duration,
            sample_rate=sample_rate,
            amplitude=amplitude
        )
        data = audio.to_numpy()
        assert_arrays_close(data, expected, rtol=1e-4, atol=1e-5, msg="Triangle wave does not match scipy.signal.sawtooth (triangle)")


class TestSilenceGeneration:
    """Tests for silence generation."""

    def test_silence_matches_numpy_zeros(self, sample_rate, short_duration):
        """Silence matches numpy zeros."""
        num_samples = int(sample_rate * short_duration)
        expected = np.zeros(num_samples)
        audio = gen.silence(
            duration_secs=short_duration,
            sample_rate=sample_rate
        )
        data = audio.to_numpy()
        assert_arrays_close(data, expected, rtol=0, atol=0, msg="Silence does not match numpy zeros")

    def test_silence_correct_duration(self, sample_rate, medium_duration):
        """Silence has correct duration."""
        audio = gen.silence(
            duration_secs=medium_duration,
            sample_rate=sample_rate
        )
        
        expected_samples = int(sample_rate * medium_duration)
        assert audio.samples_per_channel() == expected_samples


class TestChirpGeneration:
    """Tests for chirp (frequency sweep) generation."""

    def test_chirp_correct_sample_count(self, sample_rate, medium_duration):
        """Generated chirp has correct sample count."""
        expected_samples = int(sample_rate * medium_duration)
        
        audio = gen.chirp(
            f0=100.0,
            f1=1000.0,
            duration_secs=medium_duration,
            sample_rate=sample_rate,
            amplitude=0.5
        )
        
        assert audio.samples_per_channel() == expected_samples

    def test_chirp_frequency_sweep(self, sample_rate, medium_duration):
        """Chirp frequency increases from start to end."""
        start_freq = 200.0
        end_freq = 2000.0
        
        audio = gen.chirp(
            f0=start_freq,
            f1=end_freq,
            duration_secs=medium_duration,
            sample_rate=sample_rate,
            amplitude=0.5
        )
        
        data = audio.to_numpy()
        
        # Split into beginning and end portions
        n_samples = len(data)
        window_size = n_samples // 10
        
        # Analyze frequency at start
        start_segment = data[:window_size]
        start_fft = np.fft.rfft(start_segment)
        start_magnitudes = np.abs(start_fft)
        start_peak_bin = np.argmax(start_magnitudes[1:]) + 1  # Skip DC
        start_freq_resolution = sample_rate / window_size
        detected_start_freq = start_peak_bin * start_freq_resolution
        
        # Analyze frequency at end
        end_segment = data[-window_size:]
        end_fft = np.fft.rfft(end_segment)
        end_magnitudes = np.abs(end_fft)
        end_peak_bin = np.argmax(end_magnitudes[1:]) + 1
        detected_end_freq = end_peak_bin * start_freq_resolution
        
        # End frequency should be higher than start
        assert detected_end_freq > detected_start_freq, \
            f"Chirp should sweep up: start={detected_start_freq}, end={detected_end_freq}"


class TestDifferentSampleRates:
    """Tests for generation at different sample rates."""

    @pytest.mark.parametrize("sr", [8000, 16000, 22050, 44100, 48000, 96000])
    def test_sine_at_various_sample_rates(self, sr):
        """Sine generation works at various sample rates."""
        audio = gen.sine_wave(
            frequency=440.0,
            duration_secs=0.1,
            sample_rate=sr,
            amplitude=0.5
        )
        
        assert audio.sample_rate == sr
        expected_samples = int(sr * 0.1)
        assert audio.samples_per_channel() == expected_samples

    @pytest.mark.parametrize("sr", [8000, 16000, 22050, 44100, 48000, 96000])
    def test_white_noise_at_various_sample_rates(self, sr):
        """White noise generation works at various sample rates."""
        audio = gen.white_noise(
            duration_secs=0.1,
            sample_rate=sr,
            amplitude=0.5
        )
        
        assert audio.sample_rate == sr
