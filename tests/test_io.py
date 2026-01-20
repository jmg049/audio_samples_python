import pytest
import numpy as np
import soundfile as sf

import audio_samples as aus

# Import tolerance constants and helpers from conftest
# These are injected by pytest from conftest.py
TIGHT_RTOL = 1e-6
TIGHT_ATOL = 1e-7
STANDARD_RTOL = 1e-5
STANDARD_ATOL = 1e-6


def assert_arrays_close(
    actual, expected, rtol=STANDARD_RTOL, atol=STANDARD_ATOL, msg=""
):
    """Assert two arrays are close within tolerance."""
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    if actual.shape != expected.shape:
        raise AssertionError(
            f"Shape mismatch: {actual.shape} vs {expected.shape}. {msg}"
        )
    if not np.allclose(actual, expected, rtol=rtol, atol=atol):
        diff = np.abs(actual - expected)
        max_diff = np.max(diff)
        raise AssertionError(
            f"Arrays not close. {msg}\n  Max difference: {max_diff}\n  Tolerance: rtol={rtol}, atol={atol}"
        )


def assert_metadata_equal(aus_audio, expected_sr, expected_channels, expected_samples):
    """Assert AudioSamples metadata matches expected values."""
    assert aus_audio.sample_rate == expected_sr, (
        f"Sample rate mismatch: {aus_audio.sample_rate} vs {expected_sr}"
    )
    assert aus_audio.channels == expected_channels, (
        f"Channel count mismatch: {aus_audio.channels} vs {expected_channels}"
    )
    assert aus_audio.samples_per_channel() == expected_samples, (
        f"Sample count mismatch: {aus_audio.samples_per_channel()} vs {expected_samples}"
    )


class TestReadWav:
    """Tests for reading WAV files."""

    def test_read_mono_f32_matches_soundfile(self, sine_wave_mono, temp_wav_file):
        """Verify mono f32 WAV read matches soundfile exactly."""
        aus_audio, np_data, params = sine_wave_mono

        # Write with soundfile as reference
        sf.write(
            temp_wav_file,
            np_data.astype(np.float32),
            params["sample_rate"],
            subtype="FLOAT",
        )

        # Read with both libraries
        sf_data, sf_sr = sf.read(temp_wav_file, dtype="float32")
        aus_read = aus.io.read(str(temp_wav_file))
        aus_data = aus_read.to_numpy()

        # Compare
        assert aus_read.sample_rate == sf_sr
        assert aus_read.num_channels == 1
        assert_arrays_close(
            aus_data,
            sf_data,
            rtol=TIGHT_RTOL,
            atol=TIGHT_ATOL,
            msg="Mono f32 read mismatch",
        )

    def test_read_mono_f64_matches_soundfile(self, sine_wave_mono, temp_wav_file):
        """Verify mono f64 WAV read matches soundfile exactly."""
        aus_audio, np_data, params = sine_wave_mono

        # Write with soundfile as reference
        sf.write(
            temp_wav_file,
            np_data.astype(np.float64),
            params["sample_rate"],
            subtype="DOUBLE",
        )

        # Read with both libraries
        sf_data, sf_sr = sf.read(temp_wav_file, dtype="float64")
        aus_read = aus.io.read(str(temp_wav_file))
        aus_data = aus_read.to_numpy()

        # Compare
        assert aus_read.sample_rate == sf_sr
        assert_arrays_close(
            aus_data,
            sf_data,
            rtol=TIGHT_RTOL,
            atol=TIGHT_ATOL,
            msg="Mono f64 read mismatch",
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
        assert aus_read.num_channels == 2
        assert_arrays_close(
            aus_data,
            sf_data_transposed,
            rtol=TIGHT_RTOL,
            atol=TIGHT_ATOL,
            msg="Stereo f32 read mismatch",
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
        assert aus_read.num_channels == 2
        np.testing.assert_array_equal(
            aus_data, sf_data_transposed, "Stereo i16 read mismatch"
        )


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
        np.testing.assert_array_equal(
            sf_data, i16_data, "Write mono i16 verification failed"
        )

    def test_write_stereo_i16_readable_by_soundfile(
        self, sine_wave_stereo, temp_wav_file
    ):
        """Verify stereo i16 WAV written by aus can be read by soundfile."""
        aus_audio, np_data, params = sine_wave_stereo

        # Create i16 audio (widely compatible format)
        i16_data = (np_data * 32767).astype(np.int16)
        aus_i16 = aus.AudioSamples.new_multi(
            i16_data, sample_rate=params["sample_rate"]
        )
        aus.io.save(str(temp_wav_file), aus_i16)

        # Read with soundfile
        sf_data, sf_sr = sf.read(temp_wav_file, dtype="int16")

        # Transpose sf_data (samples, channels) -> (channels, samples) for comparison
        assert sf_sr == params["sample_rate"]
        np.testing.assert_array_equal(
            sf_data.T, i16_data, "Write stereo i16 verification failed"
        )


class TestRoundTrip:
    """Tests for write-then-read round-trip integrity."""

    def test_roundtrip_mono_f64(self, sine_wave_mono, temp_wav_file):
        """Mono f64 round-trip preserves data."""
        aus_audio, np_data, params = sine_wave_mono

        # Write and read back
        aus.io.save(str(temp_wav_file), aus_audio)
        aus_read = aus.io.read(str(temp_wav_file))

        assert_metadata_equal(aus_read, params["sample_rate"], 1, params["num_samples"])
        assert_arrays_close(
            aus_read.to_numpy(),
            np_data,
            rtol=TIGHT_RTOL,
            atol=TIGHT_ATOL,
            msg="Mono f64 round-trip mismatch",
        )

    def test_roundtrip_stereo_f64(self, sine_wave_stereo, temp_wav_file):
        """Stereo f64 round-trip preserves data."""
        aus_audio, np_data, params = sine_wave_stereo

        # Write and read back
        aus.io.save(str(temp_wav_file), aus_audio)
        aus_read = aus.io.read(str(temp_wav_file))

        assert_metadata_equal(aus_read, params["sample_rate"], 2, params["num_samples"])
        assert_arrays_close(
            aus_read.to_numpy(),
            np_data,
            rtol=TIGHT_RTOL,
            atol=TIGHT_ATOL,
            msg="Stereo f64 round-trip mismatch",
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
            "Mono i16 round-trip should be exact",
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
        sf.write(
            temp_wav_file,
            np_data.T.astype(np.float32),
            params["sample_rate"],
            subtype="FLOAT",
        )

        # Compare metadata
        sf_info = sf.info(temp_wav_file)
        audio, aus_meta = aus.io.read_with_info(str(temp_wav_file))

        assert aus_meta.sample_rate == sf_info.samplerate
        assert aus_meta.channels == sf_info.channels
        # num_samples is total, sf_info.frames is per-channel
        assert aus_meta.num_samples == sf_info.frames * sf_info.channels


class TestDtypeSupport:
    """Tests for different dtype support."""

    def test_u8_mono_support(self, temp_wav_file):
        """Test that u8 (uint8) mono audio is supported."""
        # Create u8 audio data (0-255 range, centered at 128)
        data = np.array([128, 100, 156, 200, 50], dtype=np.uint8)
        audio = aus.AudioSamples.new_mono(data, sample_rate=44100)

        # Should be able to save and read
        aus.io.save(str(temp_wav_file), audio)
        audio_read = aus.io.read(str(temp_wav_file))

        assert audio_read.num_channels == 1
        assert audio_read.sample_rate == 44100
        assert audio_read.samples_per_channel() == len(data)

        # CRITICAL: Verify soundfile can read it (compatibility standard)
        sf_data, sf_sr = sf.read(temp_wav_file)
        assert sf_sr == 44100, "soundfile should read the same sample rate"
        assert len(sf_data) == len(data), (
            "soundfile should read the same number of samples"
        )

    def test_u8_stereo_support(self, temp_wav_file):
        """Test that u8 (uint8) stereo audio is supported."""
        # Create u8 stereo data
        data = np.array(
            [[128, 100, 156, 200, 50], [150, 120, 180, 220, 70]], dtype=np.uint8
        )
        audio = aus.AudioSamples.new_multi(data, sample_rate=44100)

        # Should be able to save and read
        aus.io.save(str(temp_wav_file), audio)
        audio_read = aus.io.read(str(temp_wav_file))

        assert audio_read.num_channels == 2
        assert audio_read.sample_rate == 44100
        assert audio_read.samples_per_channel() == data.shape[1]

        sf_data, sf_sr = sf.read(temp_wav_file)
        assert sf_sr == 44100, "soundfile should read the same sample rate"
        assert sf_data.shape[0] == data.shape[1], (
            "soundfile should read the same number of samples"
        )
        assert sf_data.shape[1] == 2, "soundfile should read 2 channels"

    def test_f64_auto_converts_to_f32_for_wav(self, sine_wave_mono, temp_wav_file):
        """Test that f64 audio is automatically converted to f32 for WAV files."""
        aus_audio, np_data, params = sine_wave_mono

        assert aus_audio.to_numpy().dtype == np.float64

        # Save to WAV (should auto-convert to f32)
        aus.io.save(str(temp_wav_file), aus_audio)

        try:
            sf_data, sf_sr = sf.read(temp_wav_file, dtype="float32")
            assert sf_sr == params["sample_rate"]
        except Exception as e:
            pytest.fail(f"soundfile must be able to read f64 audio saved as WAV: {e}")

        # Read back with audio_samples
        audio_read = aus.io.read(str(temp_wav_file))
        assert audio_read.sample_rate == params["sample_rate"]

        # Data should be very close (within f32 precision)
        assert_arrays_close(
            audio_read.to_numpy(),
            np_data,
            rtol=1e-6,  # f32 precision
            atol=1e-7,
            msg="f64->f32 conversion should preserve data within f32 precision",
        )

    def test_all_supported_dtypes_roundtrip(self, temp_wav_file):
        """Test roundtrip for all supported dtypes."""
        sample_rate = 44100
        test_data = np.array([0.0, 0.5, -0.5, 0.25, -0.25], dtype=np.float64)

        # Test each dtype
        dtypes_to_test = [
            (np.uint8, lambda x: ((x + 1.0) * 127.5).astype(np.uint8)),
            (np.int16, lambda x: (x * 32767).astype(np.int16)),
            (np.int32, lambda x: (x * 2147483647).astype(np.int32)),
            (np.float32, lambda x: x.astype(np.float32)),
            (np.float64, lambda x: x.astype(np.float64)),  # Will auto-convert to f32
        ]

        for dtype, converter in dtypes_to_test:
            converted_data = converter(test_data)
            audio = aus.AudioSamples.new_mono(converted_data, sample_rate=sample_rate)

            # Save and read
            aus.io.save(str(temp_wav_file), audio)
            audio_read = aus.io.read(str(temp_wav_file))

            # Check basic properties
            assert audio_read.sample_rate == sample_rate
            assert audio_read.num_channels == 1
            assert audio_read.samples_per_channel() == len(test_data)

            # CRITICAL: Verify soundfile compatibility for each dtype
            try:
                sf_data, sf_sr = sf.read(temp_wav_file)
                assert sf_sr == sample_rate
            except Exception as e:
                pytest.fail(
                    f"soundfile must be able to read {dtype.__name__} WAV files: {e}"
                )


class TestSoundfileCompatibility:
    """Tests specifically for soundfile compatibility - our gold standard."""

    def test_all_written_wavs_readable_by_soundfile(self, temp_wav_file):
        """Ensure all WAV files we write can be read by soundfile."""
        sample_rate = 44100

        # Test cases: (data, description)
        test_cases = [
            (np.array([128, 100, 200], dtype=np.uint8), "u8 mono"),
            (np.array([1000, 2000, -1000], dtype=np.int16), "i16 mono"),
            (np.array([100000, 200000, -100000], dtype=np.int32), "i32 mono"),
            (np.array([0.1, 0.5, -0.3], dtype=np.float32), "f32 mono"),
            (
                np.array([0.1, 0.5, -0.3], dtype=np.float64),
                "f64 mono (converts to f32)",
            ),
            (np.array([[128, 100], [200, 150]], dtype=np.uint8), "u8 stereo"),
            (np.array([[1000, 2000], [-1000, -2000]], dtype=np.int16), "i16 stereo"),
            (np.array([[0.1, 0.5], [-0.3, -0.7]], dtype=np.float32), "f32 stereo"),
        ]

        for data, description in test_cases:
            # Create audio
            if data.ndim == 1:
                audio = aus.AudioSamples.new_mono(data, sample_rate=sample_rate)
            else:
                audio = aus.AudioSamples.new_multi(data, sample_rate=sample_rate)

            # Save
            aus.io.save(str(temp_wav_file), audio)

            try:
                sf_data, sf_sr = sf.read(temp_wav_file)
                assert sf_sr == sample_rate, f"{description}: sample rate mismatch"
            except Exception as e:
                pytest.fail(f"soundfile MUST be able to read {description} files: {e}")

    def test_soundfile_can_read_multichannel_wavs(self, temp_wav_file):
        """Test that soundfile can read multi-channel WAV files we write."""
        sample_rate = 44100

        for num_channels in [1, 2, 3, 4, 5, 6, 8]:
            # Create multi-channel audio
            data = np.random.randn(num_channels, 100).astype(np.float32) * 0.1
            audio = aus.AudioSamples.new_multi(data, sample_rate=sample_rate)

            # Save
            aus.io.save(str(temp_wav_file), audio)

            # CRITICAL: Verify soundfile can read it
            try:
                sf_data, sf_sr = sf.read(temp_wav_file)
                assert sf_sr == sample_rate
                if num_channels == 1:
                    assert sf_data.ndim == 1
                else:
                    assert sf_data.shape[1] == num_channels, (
                        f"{num_channels} channel file should have {num_channels} channels"
                    )
            except Exception as e:
                pytest.fail(f"soundfile must read {num_channels}-channel WAV: {e}")
