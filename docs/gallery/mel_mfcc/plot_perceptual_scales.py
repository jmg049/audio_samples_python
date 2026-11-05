"""
Perceptual scales, loudness weighting and mu-law
================================================

Three ways audio processing follows hearing: the mel scale spaces frequencies as
pitch is perceived, A-weighting follows the ear's sensitivity at moderate levels,
and mu-law companding spends its resolution on quiet samples.
"""

import matplotlib.pyplot as plt
import numpy as np

import audio_samples as aus
from audio_samples import features, utils

# %%
freqs = np.geomspace(20.0, 16000.0, 300)
mel = np.array([utils.hz_to_mel(float(f)) for f in freqs])
weight = features.a_weighting(freqs)
assert abs(utils.hz_to_mel(1000.0) - 1000.0) < 0.02
assert abs(weight[np.argmin(np.abs(freqs - 1000.0))]) < 0.1

# %%
# mu-law keeps the sign and lifts small values.
x = np.linspace(-1.0, 1.0, 513, dtype=np.float32)
curve = aus.AudioSamples.new_mono(x, 8000).mu_compress(255.0).to_numpy()
assert curve[256] == 0.0
assert curve[300] > x[300]

fig, axes = plt.subplots(3, 1, figsize=(7, 7))
axes[0].semilogx(freqs, mel)
axes[0].set(xlabel="Frequency (Hz)", ylabel="Mel", title="Mel scale")
axes[1].semilogx(freqs, weight)
axes[1].set(xlabel="Frequency (Hz)", ylabel="Gain (dB)", ylim=(-70, 5), title="A-weighting")
axes[2].plot(x, curve)
axes[2].set(xlabel="Input", ylabel="Output", title="Mu-law, mu = 255")
for ax in axes:
    ax.grid(True, which="both", alpha=0.3)
fig.tight_layout()
