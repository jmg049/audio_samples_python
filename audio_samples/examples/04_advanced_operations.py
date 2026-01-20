"""
Example 4: Advanced Audio Operations

This example demonstrates advanced audio processing features including
onset detection, envelope extraction, filtering, FFT analysis, and
signal comparison utilities.
"""

import audio_samples as aus
from audio_samples import OnsetDetectionConfig
import numpy as np


def main():
    print("=" * 60)
    print("Example 4: Advanced Audio Operations")
    print("=" * 60)

    # =========================================================================
    # Part 1: Onset Detection
    # =========================================================================
    print("\n--- Part 1: Onset Detection ---")

    # Create audio with a clear onset
    audio = aus.chirp(100.0, 1000.0, duration_secs=2.0)

    # Detect onsets
    config = OnsetDetectionConfig.default()
    onsets = audio.detect_onsets(config)
    print(f"[OK] Detected {len(onsets)} onset(s)")
    if len(onsets) > 0:
        print(f"     First onset at: {onsets[0]:.3f}s")

    # =========================================================================
    # Part 2: Envelope Extraction
    # =========================================================================
    print("\n--- Part 2: Envelope Extraction ---")

    test_audio = aus.sine_wave(440.0, duration_secs=1.0)

    # Amplitude envelope
    amp_env = test_audio.amplitude_envelope()
    print(f"[OK] Amplitude envelope: {len(amp_env)} values")

    # RMS envelope
    rms_env = test_audio.rms_envelope(2048, 512)
    print(f"[OK] RMS envelope: {len(rms_env)} values")

    # create a stacked plot showing the envelopes

    # =========================================================================
    # Part 3: Audio Filtering
    # =========================================================================
    print("\n--- Part 3: Audio Filtering ---")

    # Lowpass filter
    test1 = aus.sine_wave(1000.0, 0.5)
    test1.apply_butterworth_lowpass(4, 500.0)
    print("[OK] Applied lowpass filter (cutoff: 500 Hz)")

    # Highpass filter
    test2 = aus.sine_wave(1000.0, 0.5)
    test2.apply_butterworth_highpass(4, 2000.0)
    print("[OK] Applied highpass filter (cutoff: 2000 Hz)")

    # Bandpass filter
    test3 = aus.sine_wave(1000.0, 0.5)
    test3.apply_butterworth_bandpass(4, 500.0, 2000.0)
    print("[OK] Applied bandpass filter (500-2000 Hz)")

    # =========================================================================
    # Part 4: Frequency Analysis (FFT)
    # =========================================================================
    print("\n--- Part 4: Frequency Analysis (FFT) ---")

    # Create a short signal for FFT
    tone = aus.sine_wave(440.0, 0.04)  # Short enough for 4096 FFT

    # Compute FFT
    fft_result = tone.fft(n_fft=4096)
    print(f"[OK] FFT computed: shape {fft_result.shape}")

    # Find peak frequency
    magnitude = np.abs(fft_result[0])
    peak_bin = np.argmax(magnitude)
    freq_res = 44100.0 / 4096
    peak_freq = peak_bin * freq_res
    print(f"     Peak frequency: {peak_freq:.1f} Hz (expected 440 Hz)")

    # =========================================================================
    # Part 5: Signal Comparison
    # =========================================================================
    print("\n--- Part 5: Signal Comparison ---")

    signal1 = aus.sine_wave(440.0, 1.0, amplitude=0.8)
    signal2 = aus.sine_wave(440.0, 1.0, amplitude=0.8)

    # Correlation
    corr = aus.utils.correlation(signal1, signal2)
    print(f"[OK] Correlation: {corr:.4f}")

    # Mean Squared Error
    mse_val = aus.utils.mse(signal1, signal2)
    print(f"[OK] Mean Squared Error: {mse_val:.6f}")

    # =========================================================================
    # Part 6: Utility Functions
    # =========================================================================
    print("\n--- Part 6: Utility Functions ---")

    # Frequency conversions
    mel = aus.utils.hz_to_mel(1000.0)
    print(f"[OK] 1000 Hz = {mel:.2f} mel")

    # MIDI conversions
    midi = aus.utils.note_to_midi("A4")
    print(f"[OK] A4 = MIDI {midi}")

    # Time conversions
    samps = aus.utils.seconds_to_samples(1.0, 44100)
    print(f"[OK] 1.0 second = {samps} samples")

    # Amplitude conversions
    db = aus.utils.amplitude_to_db(0.5)
    print(f"[OK] 0.5 amplitude = {db:.2f} dB")


if __name__ == "__main__":
    main()
