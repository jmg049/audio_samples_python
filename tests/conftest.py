"""
Pytest configuration and shared fixtures for audio_python verification tests.

These tests verify audio_python against established libraries:
- soundfile: for I/O verification
- librosa: for spectral/feature verification
- numpy: for basic statistics verification
"""

import pytest
import numpy as np
import tempfile
import os
from pathlib import Path

# Import audio_samples_python - the module under test
import audio_samples as aus


# =============================================================================
# Tolerance constants
# =============================================================================

# Tight tolerance for exact operations (I/O round-trip for floats, basic math)
TIGHT_RTOL = 1e-6
TIGHT_ATOL = 1e-7

# Standard tolerance for most operations
STANDARD_RTOL = 1e-5
STANDARD_ATOL = 1e-6

# Loose tolerance for spectral features (inherent algorithmic differences)
SPECTRAL_RTOL = 1e-4
SPECTRAL_ATOL = 1e-5

# Very loose tolerance for features with known implementation differences
FEATURE_RTOL = 1e-3
FEATURE_ATOL = 1e-4


# =============================================================================
# Test audio generation fixtures
# =============================================================================

@pytest.fixture
def sample_rate():
    """Standard sample rate for tests."""
    return 44100


@pytest.fixture
def short_duration():
    """Short duration for quick tests (100ms)."""
    return 0.1


@pytest.fixture
def medium_duration():
    """Medium duration for more thorough tests (1 second)."""
    return 1.0


@pytest.fixture
def test_frequency():
    """Standard test frequency (440 Hz = A4)."""
    return 440.0


@pytest.fixture
def sine_wave_mono(sample_rate, medium_duration, test_frequency):
    """
    Generate a mono sine wave with known properties.
    
    Returns tuple: (audio_python AudioSamples, numpy array, params dict)
    """
    num_samples = int(sample_rate * medium_duration)
    t = np.linspace(0, medium_duration, num_samples, endpoint=False)
    amplitude = 0.5
    data = amplitude * np.sin(2 * np.pi * test_frequency * t)
    
    aus_audio = aus.AudioSamples.new_mono(data.astype(np.float64), sample_rate=sample_rate)
    
    params = {
        "frequency": test_frequency,
        "amplitude": amplitude,
        "sample_rate": sample_rate,
        "duration": medium_duration,
        "num_samples": num_samples,
    }
    
    return aus_audio, data, params


@pytest.fixture
def sine_wave_stereo(sample_rate, medium_duration, test_frequency):
    """
    Generate a stereo sine wave with different frequencies per channel.
    
    Left channel: test_frequency (440 Hz)
    Right channel: test_frequency * 1.5 (660 Hz, perfect fifth)
    
    Returns tuple: (audio_python AudioSamples, numpy array [2, N], params dict)
    """
    num_samples = int(sample_rate * medium_duration)
    t = np.linspace(0, medium_duration, num_samples, endpoint=False)
    amplitude = 0.5
    
    freq_left = test_frequency
    freq_right = test_frequency * 1.5
    
    left = amplitude * np.sin(2 * np.pi * freq_left * t)
    right = amplitude * np.sin(2 * np.pi * freq_right * t)
    
    # Shape: (2, num_samples) - channels as rows
    data = np.vstack([left, right])
    
    aus_audio = aus.AudioSamples.new_multi(data.astype(np.float64), sample_rate=sample_rate)
    
    params = {
        "frequency_left": freq_left,
        "frequency_right": freq_right,
        "amplitude": amplitude,
        "sample_rate": sample_rate,
        "duration": medium_duration,
        "num_samples": num_samples,
        "num_channels": 2,
    }
    
    return aus_audio, data, params


@pytest.fixture
def dc_offset_signal(sample_rate, short_duration):
    """
    Generate a signal with known DC offset for DC removal tests.
    
    Returns tuple: (audio_python AudioSamples, numpy array, dc_offset value)
    """
    num_samples = int(sample_rate * short_duration)
    t = np.linspace(0, short_duration, num_samples, endpoint=False)
    
    dc_offset = 0.25
    frequency = 440.0
    amplitude = 0.3
    
    # Sine wave with DC offset
    data = dc_offset + amplitude * np.sin(2 * np.pi * frequency * t)
    
    aus_audio = aus.AudioSamples.new_mono(data.astype(np.float64), sample_rate=sample_rate)
    
    return aus_audio, data, dc_offset


@pytest.fixture
def white_noise(sample_rate, short_duration):
    """
    Generate white noise with known seed for reproducibility.
    
    Returns tuple: (audio_python AudioSamples, numpy array)
    """
    np.random.seed(42)
    num_samples = int(sample_rate * short_duration)
    data = np.random.uniform(-1.0, 1.0, num_samples)
    
    aus_audio = aus.AudioSamples.new_mono(data.astype(np.float64), sample_rate=sample_rate)
    
    return aus_audio, data


@pytest.fixture
def impulse_signal(sample_rate):
    """
    Generate a unit impulse (Dirac delta) signal.
    
    Returns tuple: (audio_python AudioSamples, numpy array)
    """
    num_samples = 1024
    data = np.zeros(num_samples)
    data[0] = 1.0
    
    aus_audio = aus.AudioSamples.new_mono(data.astype(np.float64), sample_rate=sample_rate)
    
    return aus_audio, data


@pytest.fixture
def zero_crossing_signal(sample_rate):
    """
    Generate a signal with known zero crossing count.
    
    Square wave has predictable zero crossings.
    
    Returns tuple: (audio_python AudioSamples, numpy array, expected_crossings)
    """
    num_samples = 4410  # 0.1 seconds at 44100 Hz
    frequency = 100.0  # 100 Hz = 10 complete cycles in 0.1s
    t = np.linspace(0, num_samples / sample_rate, num_samples, endpoint=False)
    
    # Square wave via sign of sine
    data = np.sign(np.sin(2 * np.pi * frequency * t))
    
    # Each cycle has 2 zero crossings, 10 cycles = 20 crossings
    # But need to account for exact sample alignment
    expected_crossings = int(np.sum(np.abs(np.diff(np.sign(data))) > 0))
    
    aus_audio = aus.AudioSamples.new_mono(data.astype(np.float64), sample_rate=sample_rate)
    
    return aus_audio, data, expected_crossings


@pytest.fixture
def multi_frequency_signal(sample_rate, medium_duration):
    """
    Generate a signal with multiple frequency components for spectral analysis.
    
    Components: 440 Hz (A4), 880 Hz (A5), 1320 Hz (E6)
    
    Returns tuple: (audio_python AudioSamples, numpy array, frequency_list)
    """
    num_samples = int(sample_rate * medium_duration)
    t = np.linspace(0, medium_duration, num_samples, endpoint=False)
    
    frequencies = [440.0, 880.0, 1320.0]
    amplitudes = [0.5, 0.25, 0.125]
    
    data = np.zeros(num_samples)
    for freq, amp in zip(frequencies, amplitudes):
        data += amp * np.sin(2 * np.pi * freq * t)
    
    aus_audio = aus.AudioSamples.new_mono(data.astype(np.float64), sample_rate=sample_rate)
    
    return aus_audio, data, frequencies


# =============================================================================
# Temporary file fixtures
# =============================================================================

@pytest.fixture
def temp_wav_dir():
    """
    Create a temporary directory for WAV file tests.
    
    Cleaned up after test completes.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def temp_wav_file(temp_wav_dir):
    """
    Provide a temporary WAV file path.
    """
    return temp_wav_dir / "test.wav"


# =============================================================================
# Helper functions (available to all tests via conftest)
# =============================================================================

def assert_arrays_close(actual, expected, rtol=STANDARD_RTOL, atol=STANDARD_ATOL, msg=""):
    """
    Assert two arrays are close within tolerance.
    
    Provides informative error messages on failure.
    """
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    
    if actual.shape != expected.shape:
        raise AssertionError(
            f"Shape mismatch: {actual.shape} vs {expected.shape}. {msg}"
        )
    
    if not np.allclose(actual, expected, rtol=rtol, atol=atol):
        diff = np.abs(actual - expected)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        max_idx = np.unravel_index(np.argmax(diff), diff.shape)
        
        raise AssertionError(
            f"Arrays not close. {msg}\n"
            f"  Max difference: {max_diff} at index {max_idx}\n"
            f"  Mean difference: {mean_diff}\n"
            f"  Tolerance: rtol={rtol}, atol={atol}\n"
            f"  Actual[{max_idx}]: {actual[max_idx]}\n"
            f"  Expected[{max_idx}]: {expected[max_idx]}"
        )


def assert_metadata_equal(aus_audio, expected_sr, expected_channels, expected_samples):
    """
    Assert AudioSamples metadata matches expected values.
    """
    assert aus_audio.sample_rate == expected_sr, \
        f"Sample rate mismatch: {aus_audio.sample_rate} vs {expected_sr}"
    assert aus_audio.channels == expected_channels, \
        f"Channel count mismatch: {aus_audio.channels} vs {expected_channels}"
    assert aus_audio.samples_per_channel == expected_samples, \
        f"Sample count mismatch: {aus_audio.samples_per_channel} vs {expected_samples}"
