"""
MFCC
====

Mel-frequency cepstral coefficients with first-order deltas, the usual input to
classic speech and music classifiers.
"""

import matplotlib.pyplot as plt

import audio_samples as aus
from audio_samples import spectral

# %%
# Two different harmonic tones in succession give two different coefficient sets.
sr = 22050
first = aus.generation.compound_tone([(220.0, 0.5), (440.0, 0.3), (660.0, 0.2)], 1.0, sr).as_f32()
second = aus.generation.compound_tone([(660.0, 0.8), (1320.0, 0.2)], 1.0, sr).as_f32()
audio = aus.AudioSamples.concatenate([first, second])

# %%
# ``mfcc_frames`` returns ``(channels, n_mfcc, frames)``.
mfcc = audio.mfcc_frames(n_mfcc=13, top_db=80.0)[0]
delta = spectral.delta(mfcc)
print(mfcc.shape)

fig, axes = plt.subplots(2, 1, figsize=(7, 4.5), sharex=True)
axes[0].imshow(mfcc, origin="lower", aspect="auto")
axes[0].set(ylabel="Coefficient", title="MFCC")
axes[1].imshow(delta, origin="lower", aspect="auto")
axes[1].set(xlabel="Frame", ylabel="Coefficient", title="Delta")
fig.tight_layout()
