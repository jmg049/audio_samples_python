"""
Parametric EQ
=============

A low shelf, a peak and a high shelf combined into one equaliser. ``frequency_response``
gives the curve and ``parametric_eq`` applies it.
"""

import matplotlib.pyplot as plt
import numpy as np

import audio_samples as aus
from audio_samples import mpl

# %%
eq = aus.ParametricEq()
eq.add_band(aus.EqBand.low_shelf(200.0, 6.0, 0.7))
eq.add_band(aus.EqBand.peak(1000.0, -9.0, 2.0))
eq.add_band(aus.EqBand.high_shelf(5000.0, 3.0, 0.7))

# %%
sr = 44100
audio = aus.generation.white_noise(2.0, sr, amplitude=0.25, seed=2).as_f64()
freqs = np.geomspace(20.0, 20000.0, 400)
mag_db, _ = audio.eq_frequency_response(eq, freqs)

fig, ax = plt.subplots(figsize=(7, 3))
ax.semilogx(freqs, mag_db)
ax.set(xlabel="Frequency (Hz)", ylabel="Gain (dB)", title="EQ curve")
ax.grid(True, which="both", alpha=0.3)

# %%
equalised = audio.parametric_eq(eq)
fig2, ax2 = mpl.plot_spectrum_comparison([audio, equalised], ["Input", "Equalised"], n_fft=2048)
