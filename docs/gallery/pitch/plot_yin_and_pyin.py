"""
Pitch tracking with YIN and pYIN
================================

``yin`` gives one estimate per frame. ``pyin`` adds a voiced or unvoiced decision
per frame, so silence is marked rather than given a meaningless pitch.
"""

import matplotlib.pyplot as plt
import numpy as np

import audio_samples as aus

# %%
# A vibrato tone around 220 Hz, then silence.
sr = 22050
t = np.arange(sr) / sr
phase = 2 * np.pi * (220.0 * t + 4.0 * np.sin(2 * np.pi * 5.0 * t) / (2 * np.pi * 5.0) * 2.0)
y = np.concatenate([0.5 * np.sin(phase), np.zeros(sr // 2)]).astype(np.float32)
audio = aus.AudioSamples.new_mono(y, sr)
audio

# %%
f0_yin = audio.yin(fmin=65.0, fmax=800.0)[0]
f0_pyin, voiced, _ = audio.pyin(fmin=65.0, fmax=800.0)
f0_pyin, voiced = f0_pyin[0], voiced[0]
frames = np.arange(f0_yin.size) * 512 / sr

fig, ax = plt.subplots(figsize=(7, 3.5))
ax.plot(frames, f0_yin, label="YIN", color="0.7")
ax.plot(frames, f0_pyin, label="pYIN (voiced frames)", lw=2)
ax.set(xlabel="Time (s)", ylabel="Frequency (Hz)", ylim=(0, 450), title="Pitch track")
ax.legend()

# %%
# pYIN marks the trailing silence as unvoiced and finds 220 Hz in the tone.
assert not voiced[-5:].any()
assert abs(np.nanmedian(f0_pyin) - 220.0) < 5.0
