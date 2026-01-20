"""
Example 1: Basic Audio Creation and Operations

This example demonstrates:
- Creating mono and stereo audio signals
- Accessing audio properties
- Converting to/from NumPy arrays
- Basic signal processing operations
"""

import audio_samples as aus
import numpy as np


def main():
    print("=" * 60)
    print("Example 1: Basic Audio Creation and Operations")
    print("=" * 60)

    # =========================================================================
    # Part 1: Creating Mono Audio Signals
    # =========================================================================
    print("\n--- Part 1: Creating Mono Audio Signals ---")

    # Generate different waveforms
    sine = aus.sine_wave(440.0, duration_secs=1.0, sample_rate=44100, amplitude=0.8)
    square = aus.square_wave(440.0, duration_secs=0.5)
    triangle = aus.triangle_wave(440.0, duration_secs=0.5)
    sawtooth = aus.sawtooth_wave(440.0, duration_secs=0.5)
    chirp = aus.chirp(100.0, 2000.0, duration_secs=1.0)

    print(
        f"[OK] Created sine wave: {sine.num_channels} channel, {sine.samples_per_channel()} samples"
    )
    print("[OK] Created square, triangle, sawtooth, and chirp waves")

    # Generate noise
    white = aus.white_noise(duration_secs=0.5, amplitude=0.3)
    pink = aus.pink_noise(duration_secs=0.5, amplitude=0.3)
    brown = aus.brown_noise(duration_secs=0.5, amplitude=0.3)
    print("[OK] Created white, pink, and brown noise")

    # Utility signals
    impulse = aus.impulse(duration_secs=0.5, amplitude=1.0, position=0.5)
    silence = aus.silence(duration_secs=0.5)
    print("[OK] Created impulse and silence")

    # =========================================================================
    # Part 2: Creating Stereo Audio Signals
    # =========================================================================
    print("\n--- Part 2: Creating Stereo Audio Signals ---")

    stereo_sine = aus.stereo_sine_wave(440.0, duration_secs=1.0, amplitude=0.8)
    stereo_chirp = aus.stereo_chirp(100.0, 2000.0, duration_secs=1.0)
    stereo_silence = aus.stereo_silence(duration_secs=0.5)

    print(f"[OK] Created stereo sine: {stereo_sine.num_channels} channels")
    print("[OK] Created stereo chirp and silence")

    # =========================================================================
    # Part 3: Audio Properties
    # =========================================================================
    print("\n--- Part 3: Audio Properties ---")

    print(f"Sample rate: {sine.sample_rate} Hz")
    print(f"Channels: {sine.num_channels}")
    print(f"Samples per channel: {sine.samples_per_channel()}")
    duration = sine.samples_per_channel() / sine.sample_rate
    print(f"Duration: {duration:.2f} seconds")
    print(f"Is multi-channel: {sine.is_multi_channel()}")

    # =========================================================================
    # Part 4: NumPy Integration
    # =========================================================================
    print("\n--- Part 4: NumPy Integration ---")

    # Convert to NumPy array
    mono_array = np.array(sine, copy=False)
    print(f"Mono array shape: {mono_array.shape}")
    print(f"Min: {mono_array.min():.4f}, Max: {mono_array.max():.4f}")
    print(f"Mean: {mono_array.mean():.4f}, Std: {mono_array.std():.4f}")

    # Stereo arrays are 2D (channels x samples)
    stereo_array = np.array(stereo_sine, copy=False)
    print(f"Stereo array shape: {stereo_array.shape} (channels x samples)")
    print(f"Left channel: {stereo_array[0, :5]}")  # First 5 samples of left channel
    print(f"Right channel: {stereo_array[1, :5]}")  # First 5 samples of right channel

    # =========================================================================
    # Part 5: Basic Signal Processing
    # =========================================================================
    print("\n--- Part 5: Basic Signal Processing ---")

    # Scaling (amplitude adjustment)
    test_audio = aus.sine_wave(440.0, 0.5, amplitude=1.0)
    test_audio.scale(0.5)
    scaled_array = np.array(test_audio, copy=False)
    print(f"[OK] Scaled amplitude by 0.5 (new peak: {abs(scaled_array).max():.2f})")

    # Reversing
    test_audio2 = aus.sine_wave(440.0, 0.5)
    test_audio2.reverse()
    print("[OK] Reversed audio signal")

    # Extract channel from stereo
    stereo_test = aus.stereo_sine_wave(440.0, 0.5)
    left_channel = stereo_test.extract_channel(0)
    print(f"[OK] Extracted left channel: {left_channel.num_channels} channel")

    # Swap stereo channels (left ↔ right)
    stereo_test2 = aus.stereo_sine_wave(440.0, 0.5)
    stereo_test2.swap_channels(0, 1)  # Swap channel 0 and channel 1
    print("[OK] Swapped stereo channels (left ↔ right)")

    # =========================================================================
    # Part 6: Audio Measurements
    # =========================================================================
    print("\n--- Part 6: Audio Measurements ---")

    test_sine = aus.sine_wave(440.0, 1.0, amplitude=0.8)

    print(f"RMS amplitude: {test_sine.rms():.4f}")
    print(f"Peak amplitude: {test_sine.peak():.4f}")
    print(f"Zero crossing rate: {test_sine.zero_crossing_rate():.4f}")


if __name__ == "__main__":
    main()
