"""
Time stretch and pitch shift
============================

``time_stretch`` changes the length and keeps the pitch; ``pitch_shift`` changes
the pitch and keeps the length.
"""

import numpy as np

import audio_samples as aus
from audio_samples import mpl

# %%
sr = 22050
audio = aus.generation.compound_tone([(220.0, 0.6), (440.0, 0.3), (660.0, 0.1)], 2.0, sr).as_f32()
audio

# %%
# ``rate=1.5`` plays 1.5 times as fast, so the result is two thirds as long.
faster = audio.time_stretch(1.5)
assert abs(faster.samples_per_channel() - audio.samples_per_channel() / 1.5) < 2048
faster

# %%
# Twelve semitones up is one octave: 220 Hz becomes 440 Hz.
octave_up = audio.pitch_shift(12.0)
assert octave_up.samples_per_channel() == audio.samples_per_channel()
f0 = octave_up.detect_pitch_yin(threshold=0.1, min_frequency=65.0, max_frequency=1000.0)
print(f"{f0:.1f} Hz")
assert abs(f0 - 440.0) < 5.0
octave_up

# %%
fig, ax = mpl.plot_spectrum_comparison(
    [audio, octave_up], ["Original", "Pitch shifted +12"], freq_max=2000.0
)
