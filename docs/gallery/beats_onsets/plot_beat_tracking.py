"""
Beat tracking and tempo
=======================

``beat_track`` estimates the tempo and places beats on the onset envelope. The
helper ``mpl.plot_beats`` draws them over the waveform.
"""

import numpy as np

import audio_samples as aus
from audio_samples import effects, mpl

# %%
# A click track at 120 beats per minute.
sr = 22050
times = np.arange(0.5, 8.0, 0.5)
y = effects.clicks(times=times, sr=sr, length=8 * sr)
audio = aus.AudioSamples.new_mono(y.astype(np.float32), sr)

# %%
tempo, beats = audio.beat_track(units="time")
print(f"tempo {float(np.ravel(tempo)[0]):.1f} BPM, {beats.size} beats")
assert abs(float(np.ravel(tempo)[0]) - 120.0) < 3.0

gaps = np.diff(beats)
assert np.max(np.abs(gaps - 0.5)) < 0.03

fig, ax = mpl.plot_beats(audio)
