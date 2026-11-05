"""
Onset detection
===============

The onset strength envelope and the peaks picked from it, as times in seconds.
"""

import matplotlib.pyplot as plt
import numpy as np

import audio_samples as aus
from audio_samples import effects

# %%
# Four decaying clicks half a second apart.
sr = 22050
times = np.array([0.25, 0.75, 1.25, 1.75])
y = effects.clicks(times=times, sr=sr, length=2 * sr, click_duration=0.05)
audio = aus.AudioSamples.new_mono(y.astype(np.float32), sr)
audio

# %%
envelope = audio.onset_strength()[0]
onsets = audio.onset_detect(units="time")
print(onsets)
assert np.max(np.abs(onsets - times)) < 0.05

t = np.arange(envelope.size) * 512 / sr
fig, ax = plt.subplots(figsize=(7, 3))
ax.plot(t, envelope, label="Onset strength")
ax.vlines(onsets, 0, envelope.max(), color="C3", ls="--", label="Detected onsets")
ax.set(xlabel="Time (s)", title="Onset detection")
ax.legend()
