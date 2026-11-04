"""
Compressor and limiter
======================

A compressor reduces the level above a threshold and a limiter holds the peak
under a ceiling. Both are in-place methods, so the examples work on copies.
"""

import numpy as np

import audio_samples as aus
from audio_samples import mpl

# %%
# A quiet tone, then the same tone 14 dB louder.
sr = 22050
t = np.arange(2 * sr) / sr
level = np.where(t < 1.0, 0.1, 0.5)
audio = aus.AudioSamples.new_mono((level * np.sin(2 * np.pi * 330.0 * t)).astype(np.float32), sr)

# %%
compressed = audio.copy()
config = aus.CompressorConfig.vocal
print(f"threshold {config.threshold_db} dB, ratio {config.ratio}, makeup {config.makeup_gain_db} dB")
compressed.apply_compressor(config)

limited = audio.copy()
ceiling = aus.LimiterConfig.mastering()
print(f"ceiling {ceiling.ceiling_db} dB")
limited.apply_limiter(ceiling)

print(f"peak: original {audio.peak():.3f}, compressed {compressed.peak():.3f}, limited {limited.peak():.3f}")
loud = compressed.to_numpy()[sr + sr // 2 :]
quiet = compressed.to_numpy()[sr // 2 : sr]
assert loud.max() / quiet.max() < 5.0  # the 14 dB step was 5 times, now less
assert limited.peak() <= 10 ** (ceiling.ceiling_db / 20) * 1.01

fig, axes = mpl.plot_waveform_comparison(
    [audio, compressed, limited], ["Original", "Compressor", "Limiter"], title="Dynamics"
)
