"""
Process many clips with threads
===============================

File reads, ``resample`` and the analysis methods run without holding the GIL, so
a thread pool processes clips in parallel on a regular Python build as well as on a
free-threaded one. The results are the same as a plain loop.
"""

from concurrent.futures import ThreadPoolExecutor

import matplotlib.pyplot as plt
import numpy as np

import audio_samples as aus

# %%
# Twelve clips at different pitches, at different sample rates.
rates = [8000, 16000, 22050, 44100]
clips = [
    aus.generation.compound_tone([(110.0 * (i + 1), 0.7), (220.0 * (i + 1), 0.3)], 1.0, rates[i % 4]).as_f32()
    for i in range(12)
]


def features(clip):
    """The same 16 kHz MFCC summary for any input rate."""
    mfcc = clip.resample(16000, "high").mfcc_frames(n_mfcc=13, n_fft=512, hop_length=160)
    return mfcc[0].mean(axis=1)


# %%
with ThreadPoolExecutor(4) as pool:
    rows = list(pool.map(features, clips))

in_a_loop = [features(c) for c in clips]
assert all(np.array_equal(a, b) for a, b in zip(rows, in_a_loop))

fig, ax = plt.subplots(figsize=(7, 3))
ax.imshow(np.stack(rows), aspect="auto", origin="lower")
ax.set(xlabel="MFCC coefficient", ylabel="Clip", title="Mean MFCC per clip")
