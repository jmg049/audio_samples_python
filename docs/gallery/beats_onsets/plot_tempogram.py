"""
Tempogram
=========

Local tempo over time. A signal that changes speed shows up as a step.
"""

import matplotlib.pyplot as plt
import numpy as np

import audio_samples as aus
from audio_samples import effects

# %%
# Ten seconds at 90 BPM followed by ten at 150 BPM.
sr = 22050
times = np.concatenate([np.arange(0.0, 10.0, 60 / 90), np.arange(10.0, 20.0, 60 / 150)])
y = effects.clicks(times=times, sr=sr, length=20 * sr)
audio = aus.AudioSamples.new_mono(y.astype(np.float32), sr)

# %%
# ``tempogram`` returns ``(channels, lags, frames)``; lag 0 is excluded from the plot.
gram = audio.tempogram()[0]

fig, ax = plt.subplots(figsize=(7, 3.5))
ax.imshow(gram[1:], origin="lower", aspect="auto", extent=[0, 20, 1, gram.shape[0]])
ax.set(xlabel="Time (s)", ylabel="Lag (frames)", ylim=(1, 120), title="Autocorrelation tempogram")
