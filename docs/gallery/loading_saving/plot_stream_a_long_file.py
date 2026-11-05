"""
Read a long file in blocks
==========================

``io.StreamedAudioReader`` decodes on demand, so a long file does not have to
fit in memory. This example computes the RMS level of each second from a
one-minute file and reads only one block at a time.
"""

import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import audio_samples as aus

# %%
# Write a minute of noise whose level rises and falls.
sr = 8000
t = np.arange(60 * sr) / sr
level = 0.1 + 0.4 * (1 + np.sin(2 * np.pi * t / 30.0)) / 2
y = (level * np.random.default_rng(0).standard_normal(t.size)).astype(np.float32)
path = Path(tempfile.mkdtemp()) / "long.flac"
aus.io.save(str(path), aus.AudioSamples.new_mono(y, sr).as_i16())

# %%
# Read one second at a time. ``read_frames`` returns an array, or ``None`` at the
# end of the file.
levels = []
with aus.io.StreamedAudioReader(str(path)) as reader:
    while True:
        block = reader.read_frames(sr, dtype=aus.SampleType.F32)
        if block is None:
            break
        levels.append(float(np.sqrt(np.mean(block**2))))

assert len(levels) == 60

fig, ax = plt.subplots(figsize=(7, 3))
ax.plot(levels)
ax.set(xlabel="Time (s)", ylabel="RMS", title="RMS per second, read in blocks")
