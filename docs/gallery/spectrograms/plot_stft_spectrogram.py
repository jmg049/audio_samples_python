"""
STFT spectrogram
================

The complex STFT as an array, its magnitude in dB, and the inverse transform.
Reconstruction from the unmodified STFT returns the original signal.
"""

import matplotlib.pyplot as plt
import numpy as np

import audio_samples as aus
from audio_samples import spectral

# %%
# A rising chirp over a steady tone.
sr = 22050
chirp = aus.generation.chirp(200.0, 8000.0, 2.0, sr).as_f32()
tone = aus.generation.sine_wave(1000.0, 2.0, sr, amplitude=0.3).as_f32()
y = (chirp + tone).to_numpy()

# %%
# ``spectral.stft`` takes a NumPy array and returns a ``(bins, frames)`` array.
hop = 256
d = spectral.stft(y, n_fft=1024, hop_length=hop)
db = spectral.amplitude_to_db(np.abs(d), ref="max", top_db=80.0)

fig, ax = plt.subplots(figsize=(7, 3.5))
im = ax.imshow(db, origin="lower", aspect="auto", extent=[0, y.size / sr, 0, sr / 2])
ax.set(xlabel="Time (s)", ylabel="Frequency (Hz)", title="Magnitude (dB)")
fig.colorbar(im, ax=ax, label="dB")

# %%
# ``spectral.istft`` inverts it.
back = spectral.istft(d, hop_length=hop, length=y.size)
error = np.max(np.abs(back - y))
print(f"largest reconstruction error: {error:.2e}")
assert error < 1e-4
