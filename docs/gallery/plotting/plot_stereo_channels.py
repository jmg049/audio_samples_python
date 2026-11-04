"""
Stereo channels
===============

A channel per axis, and the panning methods that act on a stereo signal.
"""

import audio_samples as aus
from audio_samples import mpl

# %%
audio = aus.generation.stereo_sine_wave(440.0, 0.02, 22050)
audio.pan(-0.6)
print(f"left {audio.extract_channel(0).peak():.2f}, right {audio.extract_channel(1).peak():.2f}")
assert audio.extract_channel(0).peak() > audio.extract_channel(1).peak()

fig, axes = mpl.plot_waveform_channels(audio, title="Panned left")
