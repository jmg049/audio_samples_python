"""Showcase audio_samples.mpl visualization capabilities.

This example demonstrates the matplotlib-based plotting functions from the
audio_samples.mpl module, including waveforms, spectrograms, envelopes,
and spectral analysis.
"""

import matplotlib.pyplot as plt
import audio_samples as aus
from audio_samples import mpl


if __name__ == "__main__":
    # Create a test signal with amplitude modulation for more interesting envelopes
    # Amplitude modulated tone: carrier at 440Hz, modulation at 5Hz
    t = aus.sine_wave(440.0, 1.0).time_axis()
    carrier = aus.sine_wave(440.0, 1.0)
    modulator = aus.sine_wave(5.0, 1.0) * 0.5 + 0.5  # 0-1 range
    am_signal = (carrier * modulator).trim(0.0, 0.5)

    # Also create a simple harmonic signal for spectrum analysis
    fundamental = aus.sine_wave(440.0, 1.0)
    harmonic2 = aus.sine_wave(880.0, 1.0) * 0.5
    harmonic3 = aus.sine_wave(1320.0, 1.0) * 0.3
    complex_signal = (fundamental + harmonic2 + harmonic3).trim(0.0, 0.5)

    # Set a publication-quality style
    style = mpl.MplStyle.paper()

    # 1. Waveform plot - just the basic waveform
    print("Plotting basic waveform...")
    fig1, ax1 = mpl.plot_waveform(
        am_signal,
        title="Amplitude Modulated Waveform (440Hz carrier, 5Hz modulation)",
        style=style,
    )
    plt.tight_layout()
    plt.show()

    # 2. Analytic envelope - shows the envelope extraction on AM signal
    print("Plotting analytic envelope...")
    fig2, ax2 = mpl.plot_analytic_envelope(
        am_signal,
        show_waveform=True,
        title="Analytic Envelope (Hilbert Transform)",
        style=style,
    )
    plt.tight_layout()
    plt.show()

    # 3. Magnitude spectrum
    print("Plotting magnitude spectrum...")
    fig3, ax3 = mpl.plot_magnitude_spectrum(
        complex_signal,
        n_fft=4096,
        db_scale=True,
        show_centroid=True,
        freq_max=3000,
        title="Magnitude Spectrum (dB)",
        style=style,
    )
    plt.tight_layout()
    plt.show()

    # 4. Spectrogram
    print("Plotting spectrogram...")
    fig4, ax4 = mpl.plot_spectrogram(
        complex_signal,
        db_range=(-80, 0),
        freq_max=3000,
        title="STFT Spectrogram",
        style=style,
    )
    plt.tight_layout()
    plt.show()

    # 5. Envelope comparison (amplitude, RMS, analytic, moving average)
    print("Plotting envelope comparison...")
    fig5, ax5 = mpl.plot_envelope_comparison(
        am_signal,
        window_size=2048,
        hop_size=512,
        title="Envelope Comparison (4 methods)",
        style=style,
    )
    plt.tight_layout()
    plt.show()

    # 6. Audio overview dashboard (comprehensive view)
    print("Plotting audio overview dashboard...")
    fig6, axes6 = mpl.plot_audio_overview(
        complex_signal, title="Audio Analysis Overview", style=style
    )
    plt.tight_layout()
    plt.show()

    print("\nAll plots displayed successfully!")
