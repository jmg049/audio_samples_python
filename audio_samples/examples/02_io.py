"""
Example 2: Audio I/O (Input/Output)

This example demonstrates:
- Reading audio files from disk
- Getting audio file information
- Saving audio files in different formats
- Working with different sample rates and bit depths
"""

import audio_samples as aus
from pathlib import Path
import tempfile


def main():
    print("=" * 60)
    print("Example 2: Audio I/O")
    print("=" * 60)

    # Create a temporary directory for our examples
    temp_dir = Path(tempfile.mkdtemp(prefix="audio_samples_io_"))
    print(f"\nUsing temporary directory: {temp_dir}")

    # =========================================================================
    # Part 1: Creating and Saving Audio Files
    # =========================================================================
    print("\n--- Part 1: Creating and Saving Audio Files ---")

    # Generate some test audio
    sine_wave = aus.sine_wave(440.0, duration_secs=2.0, sample_rate=44100)
    stereo_chirp = aus.stereo_chirp(100.0, 1000.0, duration_secs=3.0, sample_rate=48000)

    # Save as WAV file (most common format)
    wav_path = temp_dir / "sine_440hz.wav"
    aus.io.save(str(wav_path), sine_wave)
    print(f"[OK] Saved mono sine wave to: {wav_path.name}")

    # Save stereo audio
    stereo_path = temp_dir / "stereo_chirp.wav"
    aus.io.save(str(stereo_path), stereo_chirp)
    print(f"[OK] Saved stereo chirp to: {stereo_path.name}")

    # =========================================================================
    # Part 2: Reading Audio Files
    # =========================================================================
    print("\n--- Part 2: Reading Audio Files ---")

    # Read the audio file we just saved
    loaded_audio = aus.io.read(str(wav_path))
    print(f"[OK] Loaded audio from {wav_path.name}")
    print(f"  - Channels: {loaded_audio.num_channels}")
    print(f"  - Sample rate: {loaded_audio.sample_rate} Hz")
    duration = loaded_audio.samples_per_channel() / loaded_audio.sample_rate
    print(f"  - Duration: {duration:.2f} seconds")
    print(f"  - Samples: {loaded_audio.samples_per_channel()}")

    # Read with specific sample rate (resampling if needed)
    # Note: This depends on whether the read function supports resampling parameter
    loaded_stereo = aus.io.read(str(stereo_path))
    print(f"[OK] Loaded stereo audio from {stereo_path.name}")
    print(f"  - Channels: {loaded_stereo.num_channels}")
    print(f"  - Sample rate: {loaded_stereo.sample_rate} Hz")

    # =========================================================================
    # Part 3: Getting Audio File Information (Without Loading)
    # =========================================================================
    print("\n--- Part 3: Getting Audio File Information ---")

    # Get info without loading the entire file into memory
    info = aus.io.info(str(wav_path))
    print(f"File info for {wav_path.name}:")
    print(f"  - Sample rate: {info.sample_rate} Hz")
    print(f"  - Channels: {info.channels}")
    print(f"  - Duration: {info.duration:.2f} seconds")
    print(f"  - Total samples: {info.num_samples}")
    print(f"  - Sample type: {info.sample_type}")

    # =========================================================================
    # Part 4: Reading and Saving with Metadata
    # =========================================================================
    print("\n--- Part 4: Reading with Metadata ---")

    # Read audio and get info in one call
    audio, audio_info = aus.io.read_with_info(str(stereo_path))
    print(f"[OK] Loaded audio with info from {stereo_path.name}")
    print(f"  - Sample rate: {audio_info.sample_rate} Hz")
    print(f"  - Channels: {audio_info.channels}")
    print(f"  - Duration: {audio_info.duration:.2f} seconds")

    # =========================================================================
    # Part 5: Working with Different Sample Rates
    # =========================================================================
    print("\n--- Part 5: Working with Different Sample Rates ---")

    # Create audio at different sample rates
    audio_44k = aus.sine_wave(440.0, 1.0, sample_rate=44100)
    audio_48k = aus.sine_wave(440.0, 1.0, sample_rate=48000)
    audio_96k = aus.sine_wave(440.0, 1.0, sample_rate=96000)

    path_44k = temp_dir / "sine_44khz.wav"
    path_48k = temp_dir / "sine_48khz.wav"
    path_96k = temp_dir / "sine_96khz.wav"

    aus.io.save(str(path_44k), audio_44k)
    aus.io.save(str(path_48k), audio_48k)
    aus.io.save(str(path_96k), audio_96k)
    print("[OK] Saved audio at 44.1 kHz, 48 kHz, and 96 kHz")

    # Read them back and verify
    for path in [path_44k, path_48k, path_96k]:
        info = aus.io.info(str(path))
        print(
            f"  - {path.name}: {info.sample_rate} Hz, {info.num_samples} total samples"
        )

    # =========================================================================
    # Cleanup
    # =========================================================================
    print("\n--- Cleanup ---")
    import shutil

    shutil.rmtree(temp_dir)
    print("[OK] Cleaned up temporary directory")


if __name__ == "__main__":
    main()
