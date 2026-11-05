"""
One-call overview
=================

``plot_audio_overview`` draws the waveform, spectrum, spectrogram and level
statistics of a signal in one figure.
"""

import audio_samples as aus
from audio_samples import mpl

# %%
sr = 22050
audio = aus.generation.chirp(100.0, 6000.0, 2.0, sr).as_f32()
fig = mpl.plot_audio_overview(audio)
