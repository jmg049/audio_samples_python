"""
Example 3: Audio Visualization and Plotting

This example demonstrates:
- Plotting waveforms (time domain)
- Creating spectrograms (time-frequency representation)
- Visualizing magnitude spectrum (frequency domain)
- Saving plots to files
"""

import audio_samples as aus
from pathlib import Path
import tempfile


def main():
    print("=" * 60)
    print("Example 3: Audio Visualization and Plotting")
    print("=" * 60)

    # Create temporary directory for saving plots
    temp_dir = Path(tempfile.mkdtemp(prefix="audio_samples_plots_"))
    print(f"\nPlots will be saved to: {temp_dir}")

    # =========================================================================
    # Part 1: Waveform Plotting (Time Domain)
    # =========================================================================
    print("\n--- Part 1: Waveform Plotting ---")

    # Create a simple sine wave
    sine = aus.sine_wave(440.0, duration_secs=1.0, sample_rate=44100)

    # Plot with default parameters
    print("Creating waveform plot...")
    waveform_plot = sine.plot_waveform()

    # Save as HTML (interactive)
    html_path = temp_dir / "waveform.html"
    waveform_plot.save(str(html_path))
    print("[OK] Saved interactive waveform: waveform.html")

    # =========================================================================
    # Part 2: Stereo Waveform Plotting
    # =========================================================================
    print("\n--- Part 2: Stereo Waveform Plotting ---")

    # Create stereo audio
    stereo = aus.stereo_chirp(100.0, 2000.0, duration_secs=2.0, sample_rate=44100)

    # Plot stereo waveform
    print("Plotting stereo waveform...")
    waveform_plot_params = aus.WaveformPlotParams(
        title="Stereo Chirp Waveform",
    )
    stereo_plot = stereo.plot_waveform(params=waveform_plot_params)
    # stereo_plot.show()
    stereo_plot.save(str(temp_dir / "stereo_waveform.html"))
    print("[OK] Saved stereo waveform")

    # =========================================================================
    # Part 3: Spectrogram Plotting (Time-Frequency)
    # =========================================================================
    print("\n--- Part 3: Spectrogram Plotting ---")

    # Create audio with changing frequency content
    chirp = aus.chirp(200.0, 2000.0, duration_secs=3.0, sample_rate=44100)

    # Plot spectrogram
    print("Creating spectrogram...")
    spectrogram_plot = chirp.plot_spectrogram()
    spectrogram_plot.save(str(temp_dir / "spectrogram.html"))
    print("[OK] Saved spectrogram")

    # =========================================================================
    # Part 4: Magnitude Spectrum (Frequency Domain)
    # =========================================================================
    print("\n--- Part 4: Magnitude Spectrum ---")

    # Create a signal with multiple frequency components
    # A4 (440 Hz) + E5 (659.25 Hz) chord
    note_a4 = aus.sine_wave(440.0, 1.0, amplitude=0.5)
    note_e5 = aus.sine_wave(659.25, 1.0, amplitude=0.5)

    # Mix them together
    chord = note_a4
    chord.mix([note_e5])

    # Plot magnitude spectrum
    print("Creating magnitude spectrum...")
    spectrum_plot = chord.plot_magnitude_spectrum()
    spectrum_plot.save(str(temp_dir / "spectrum.html"))
    print("[OK] Saved magnitude spectrum")

    # =========================================================================
    # Part 5: Comparing Different Waveforms
    # =========================================================================
    print("\n--- Part 5: Comparing Different Waveforms ---")

    # Create different waveforms
    waveforms = {
        "sine": aus.sine_wave(440.0, 0.5),
        "square": aus.square_wave(440.0, 0.5),
        "triangle": aus.triangle_wave(440.0, 0.5),
        "sawtooth": aus.sawtooth_wave(440.0, 0.5),
    }

    # Plot each waveform and spectrum
    for name, audio in waveforms.items():
        # Waveform
        waveform = audio.plot_waveform()
        waveform.save(str(temp_dir / f"waveform_{name}.html"))

        # Spectrum
        spectrum = audio.plot_magnitude_spectrum()
        spectrum.save(str(temp_dir / f"spectrum_{name}.html"))

    print("[OK] Saved waveforms and spectra for 4 wave types")

    # =========================================================================
    # Part 6: Working with Plot Objects
    # =========================================================================
    print("\n--- Part 6: Working with Plot Objects ---")

    # Get HTML for embedding in web pages
    plot = sine.plot_waveform()
    html_string = plot.html()
    print(f"[OK] Generated HTML string ({len(html_string)} chars)")
    print("     Can be embedded in web pages or Jupyter notebooks")

    # plot.show()


if __name__ == "__main__":
    main()
