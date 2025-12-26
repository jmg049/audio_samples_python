"""
Tests for I/O operations - comparing audio_python against soundfile.

These tests verify that:
1. Reading WAV files produces identical data to soundfile
2. Writing and reading back produces identical data (round-trip)
3. Metadata (sample rate, channels, samples) is correctly preserved
4. Different sample formats are handled correctly

API notes:
- Write is aus.io.save() not aus.io.write()
- Metadata reading is via aus.io.read_with_info() which returns (AudioSamples, PyAudioInfo)
- Type conversion: as_f32, as_i16, etc. (not to_*)
"""

import pytest
import numpy as np
import soundfile as sf
from pathlib import Path

import audio_python as aus

# Import tolerance constants and helpers from conftest
# These are injected by pytest from conftest.py
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


def assert_metadata_equal(aus_audio, expected_sr, expected_channels, expected_samples):
    """Assert AudioSamples metadata matches expected values."""
    assert aus_audio.sample_rate == expected_sr, f"Sample rate mismatch: {aus_audio.sample_rate} vs {expected_sr}"
    assert aus_audio.channels == expected_channels, f"Channel count mismatch: {aus_audio.channels} vs {expected_channels}"
    assert aus_audio.samples_per_channel() == expected_samples, f"Sample count mismatch: {aus_audio.samples_per_channel()} vs {expected_samples}"


class TestReadWav:
    """Tests for reading WAV files."""

    def test_read_mono_f32_matches_soundfile(self, sine_wave_mono, temp_wav_file):
        """Verify mono f32 WAV read matches soundfile exactly."""
        aus_audio, np_data, params = sine_wave_mono
        
        # Write with soundfile as reference
        sf.write(temp_wav_file, np_data.astype(np.float32), params["sample_rate"], subtype="FLOAT")
        
        # Read with both libraries
        sf_data, sf_sr = sf.read(temp_wav_file, dtype="float32")
        aus_read = aus.io.read(str(temp_wav_file))
        aus_data = aus_read.to_numpy()
        
        # Compare
        assert aus_read.sample_rate == sf_sr
        assert aus_read.channels == 1
        assert_arrays_close(
            aus_data, sf_data,
            rtol=TIGHT_RTOL, atol=TIGHT_ATOL,
            msg="Mono f32 read mismatch"
        )

    def test_read_mono_f64_matches_soundfile(self, sine_wave_mono, temp_wav_file):
        """Verify mono f64 WAV read matches soundfile exactly."""
        aus_audio, np_data, params = sine_wave_mono
        
        # Write with soundfile as reference
        sf.write(temp_wav_file, np_data.astype(np.float64), params["sample_rate"], subtype="DOUBLE")
        
        # Read with both libraries
        sf_data, sf_sr = sf.read(temp_wav_file, dtype="float64")
        aus_read = aus.io.read(str(temp_wav_file))
        aus_data = aus_read.to_numpy()
        
        # Compare
        assert aus_read.sample_rate == sf_sr
        assert_arrays_close(
            aus_data, sf_data,
            rtol=TIGHT_RTOL, atol=TIGHT_ATOL,
            msg="Mono f64 read mismatch"
        )

    def test_read_mono_i16_matches_soundfile(self, sine_wave_mono, temp_wav_file):
        """Verify mono i16 WAV read matches soundfile."""
        aus_audio, np_data, params = sine_wave_mono
        
        # Convert to i16 range and write
        i16_data = (np_data * 32767).astype(np.int16)
        sf.write(temp_wav_file, i16_data, params["sample_rate"], subtype="PCM_16")
        
        # Read with both libraries
        sf_data, sf_sr = sf.read(temp_wav_file, dtype="int16")
        aus_read = aus.io.read(str(temp_wav_file))
        aus_data = aus_read.to_numpy()
        
        # Compare (integer should match exactly)
        assert aus_read.sample_rate == sf_sr
        np.testing.assert_array_equal(aus_data, sf_data, "Mono i16 read mismatch")

    def test_read_stereo_f32_matches_soundfile(self, sine_wave_stereo, temp_wav_file):
        """Verify stereo f32 WAV read matches soundfile."""
        aus_audio, np_data, params = sine_wave_stereo
        
        # soundfile expects (samples, channels), we have (channels, samples)
        sf_write_data = np_data.T.astype(np.float32)
        sf.write(temp_wav_file, sf_write_data, params["sample_rate"], subtype="FLOAT")
        
        # Read with both libraries
        sf_data, sf_sr = sf.read(temp_wav_file, dtype="float32")
        aus_read = aus.io.read(str(temp_wav_file))
        aus_data = aus_read.to_numpy()
        
        # soundfile returns (samples, channels), aus returns (channels, samples)
        # Transpose soundfile data for comparison
        sf_data_transposed = sf_data.T
        
        assert aus_read.sample_rate == sf_sr
        assert aus_read.channels == 2
        assert_arrays_close(
            aus_data, sf_data_transposed,
            rtol=TIGHT_RTOL, atol=TIGHT_ATOL,
            msg="Stereo f32 read mismatch"
        )

    def test_read_stereo_i16_matches_soundfile(self, sine_wave_stereo, temp_wav_file):
        """Verify stereo i16 WAV read matches soundfile."""
        aus_audio, np_data, params = sine_wave_stereo
        
        # Convert to i16 and transpose for soundfile
        i16_data = (np_data * 32767).astype(np.int16)
        sf_write_data = i16_data.T
        sf.write(temp_wav_file, sf_write_data, params["sample_rate"], subtype="PCM_16")
        
        # Read with both libraries
        sf_data, sf_sr = sf.read(temp_wav_file, dtype="int16")
        aus_read = aus.io.read(str(temp_wav_file))
        aus_data = aus_read.to_numpy()
        
        sf_data_transposed = sf_data.T
        
        assert aus_read.sample_rate == sf_sr
        assert aus_read.channels == 2
        np.testing.assert_array_equal(aus_data, sf_data_transposed, "Stereo i16 read mismatch")


class TestWriteWav:
    """Tests for writing WAV files."""

    def test_write_mono_i16_readable_by_soundfile(self, sine_wave_mono, temp_wav_file):
        """Verify mono i16 WAV written by aus can be read by soundfile."""
        aus_audio, np_data, params = sine_wave_mono
        
        # Create i16 audio (widely compatible format)
        i16_data = (np_data * 32767).astype(np.int16)
        aus_i16 = aus.AudioSamples.new_mono(i16_data, sample_rate=params["sample_rate"])
        aus.io.save(str(temp_wav_file), aus_i16)
        
        # Read with soundfile
        sf_data, sf_sr = sf.read(temp_wav_file, dtype="int16")
        
        assert sf_sr == params["sample_rate"]
        np.testing.assert_array_equal(sf_data, i16_data, "Write mono i16 verification failed")

    def test_write_stereo_i16_readable_by_soundfile(self, sine_wave_stereo, temp_wav_file):
        """Verify stereo i16 WAV written by aus can be read by soundfile."""
        aus_audio, np_data, params = sine_wave_stereo
        
        # Create i16 audio (widely compatible format)
        i16_data = (np_data * 32767).astype(np.int16)
        aus_i16 = aus.AudioSamples.new_multi(i16_data, sample_rate=params["sample_rate"])
        aus.io.save(str(temp_wav_file), aus_i16)
        
        # Read with soundfile
        sf_data, sf_sr = sf.read(temp_wav_file, dtype="int16")
        
        # Transpose sf_data (samples, channels) -> (channels, samples) for comparison
        assert sf_sr == params["sample_rate"]
        np.testing.assert_array_equal(sf_data.T, i16_data, "Write stereo i16 verification failed")


class TestRoundTrip:
    """Tests for write-then-read round-trip integrity."""

    def test_roundtrip_mono_f64(self, sine_wave_mono, temp_wav_file):
        """Mono f64 round-trip preserves data."""
        aus_audio, np_data, params = sine_wave_mono
        
        # Write and read back
        aus.io.save(str(temp_wav_file), aus_audio)
        aus_read = aus.io.read(str(temp_wav_file))
        
        assert_metadata_equal(
            aus_read,
            params["sample_rate"],
            1,
            params["num_samples"]
        )
        assert_arrays_close(
            aus_read.to_numpy(), np_data,
            rtol=TIGHT_RTOL, atol=TIGHT_ATOL,
            msg="Mono f64 round-trip mismatch"
        )

    def test_roundtrip_stereo_f64(self, sine_wave_stereo, temp_wav_file):
        """Stereo f64 round-trip preserves data."""
        aus_audio, np_data, params = sine_wave_stereo
        
        # Write and read back
        aus.io.save(str(temp_wav_file), aus_audio)
        aus_read = aus.io.read(str(temp_wav_file))
        
        assert_metadata_equal(
            aus_read,
            params["sample_rate"],
            2,
            params["num_samples"]
        )
        assert_arrays_close(
            aus_read.to_numpy(), np_data,
            rtol=TIGHT_RTOL, atol=TIGHT_ATOL,
            msg="Stereo f64 round-trip mismatch"
        )

    def test_roundtrip_mono_i16(self, sine_wave_mono, temp_wav_file):
        """Mono i16 round-trip preserves data exactly."""
        aus_audio, np_data, params = sine_wave_mono
        
        # Create i16 audio directly from scaled numpy array
        i16_data = (np_data * 32767).astype(np.int16)
        aus_i16 = aus.AudioSamples.new_mono(i16_data, sample_rate=params["sample_rate"])
        
        # Write and read back
        aus.io.save(str(temp_wav_file), aus_i16)
        aus_read = aus.io.read(str(temp_wav_file))
        
        # Integer round-trip should be exact
        np.testing.assert_array_equal(
            aus_read.to_numpy(),
            aus_i16.to_numpy(),
            "Mono i16 round-trip should be exact"
        )

    def test_roundtrip_preserves_different_sample_rates(self, temp_wav_file):
        """Round-trip preserves various sample rates."""
        sample_rates = [8000, 16000, 22050, 44100, 48000, 96000]
        
        for sr in sample_rates:
            data = np.sin(np.linspace(0, 2 * np.pi * 10, 1000))
            aus_audio = aus.AudioSamples.new_mono(data, sample_rate=sr)
            
            aus.io.save(str(temp_wav_file), aus_audio)
            aus_read = aus.io.read(str(temp_wav_file))
            
            assert aus_read.sample_rate == sr, f"Sample rate {sr} not preserved"


class TestMetadata:
    """Tests for file metadata reading."""

    def test_metadata_mono_wav(self, sine_wave_mono, temp_wav_file):
        """Verify metadata reading for mono WAV."""
        aus_audio, np_data, params = sine_wave_mono
        
        aus.io.save(str(temp_wav_file), aus_audio)
        audio, meta = aus.io.read_with_info(str(temp_wav_file))
        
        assert meta.sample_rate == params["sample_rate"]
        assert meta.channels == 1
        assert meta.num_samples == params["num_samples"]
        assert abs(meta.duration - params["duration"]) < 0.001

    def test_metadata_stereo_wav(self, sine_wave_stereo, temp_wav_file):
        """Verify metadata reading for stereo WAV."""
        aus_audio, np_data, params = sine_wave_stereo
        
        aus.io.save(str(temp_wav_file), aus_audio)
        audio, meta = aus.io.read_with_info(str(temp_wav_file))
        
        assert meta.sample_rate == params["sample_rate"]
        assert meta.channels == 2
        # num_samples is total samples (channels * samples_per_channel)
        assert meta.num_samples == params["num_samples"] * 2
        assert abs(meta.duration - params["duration"]) < 0.001

    def test_metadata_matches_soundfile_info(self, sine_wave_stereo, temp_wav_file):
        """Verify metadata matches soundfile's info."""
        aus_audio, np_data, params = sine_wave_stereo
        
        # Write with soundfile for reference
        sf.write(temp_wav_file, np_data.T.astype(np.float32), params["sample_rate"], subtype="FLOAT")
        
        # Compare metadata
        sf_info = sf.info(temp_wav_file)
        audio, aus_meta = aus.io.read_with_info(str(temp_wav_file))
        
        assert aus_meta.sample_rate == sf_info.samplerate
        assert aus_meta.channels == sf_info.channels
        # num_samples is total, sf_info.frames is per-channel
        assert aus_meta.num_samples == sf_info.frames * sf_info.channels
