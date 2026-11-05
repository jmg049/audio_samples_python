"""
Low-pass and high-pass filters
==============================

Butterworth filters applied to noise. The measured spectra follow the designs:
down 3 dB at the cutoff frequency.
"""

import numpy as np

import audio_samples as aus
from audio_samples import mpl

# %%
sr = 22050
noise = aus.generation.white_noise(2.0, sr, amplitude=0.5, seed=1).as_f64()

# %%
# The ``butterworth_*`` methods filter in place, so each works on a copy.
low = noise.copy()
low.butterworth_lowpass(4, 1000.0)
high = noise.copy()
high.butterworth_highpass(4, 1000.0)

fig, ax = mpl.plot_spectrum_comparison(
    [noise, low, high], ["White noise", "Low-pass", "High-pass"], n_fft=1024, title="1 kHz Butterworth"
)

# %%
# The filter's own response, from its design.
design = aus.IirFilterDesign.butterworth_lowpass(4, 1000.0)
filt = aus.SosFilter.from_design(design, float(sr))
mag, _ = filt.frequency_response(np.array([1000.0]))
level = 20 * np.log10(mag[0])
print(f"{level:.2f} dB at the cutoff")
assert abs(level + 3.01) < 0.05

# %%
low
