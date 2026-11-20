"""
Resampling quality
==================

``resample`` takes a target rate and a quality. A chirp that reaches past the target
rate's Nyquist frequency shows how much each quality setting attenuates what cannot be kept.
"""

import numpy as np

import audio_samples as aus
from audio_samples import mpl

# %%
# Sweep up to 20 kHz at 44.1 kHz.
audio = aus.generation.chirp(100.0, 20000.0, 2.0, 44100).as_f32()

# %%
fast = audio.resample(16000, "fast")
high = audio.resample(16000, "high")
print(fast, high, sep="\n")
assert fast.sample_rate == high.sample_rate == 16000
assert abs(high.samples_per_channel() - 32000) <= 1

fig, ax = mpl.plot_spectrum_comparison(
    [fast, high], ['"fast"', '"high"'], n_fft=4096, title="Chirp resampled to 16 kHz"
)

# %%
# ``io.read_and_resample`` does the same while reading a file. To build a fixed-rate
# dataset, this is the one call that is needed.
high
