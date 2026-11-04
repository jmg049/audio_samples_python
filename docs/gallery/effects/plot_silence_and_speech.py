"""
Trim silence and find active regions
====================================

``trim_db`` removes leading and trailing quiet, ``split_db`` lists the sounding
intervals, and ``speech_regions`` runs voice activity detection.
"""

import numpy as np

import audio_samples as aus
from audio_samples import mpl

# %%
# Two bursts of a tone with silence around and between them.
sr = 16000
tone = aus.generation.sine_wave(300.0, 0.5, sr, amplitude=0.5).as_f32()
gap = aus.generation.silence(0.5, sr).as_f32()
audio = aus.AudioSamples.concatenate([gap, tone, gap, tone, gap])

# %%
trimmed, interval = audio.trim_db(top_db=40.0)
print(f"kept samples {interval[0]} to {interval[1]} of {audio.samples_per_channel()}")
assert trimmed.samples_per_channel() < audio.samples_per_channel()

intervals = audio.split_db(top_db=40.0)
print(intervals / sr)
assert len(intervals) == 2

# %%
config = aus.VadConfig.default()
regions = audio.speech_regions(config)
print(regions)

fig, ax = mpl.plot_speech_regions(audio, config=config)
