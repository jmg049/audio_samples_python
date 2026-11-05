"""
Harmonic and percussive separation
==================================

Median-filtering HPSS: sustained tones go to the harmonic part and clicks to the
percussive part. ``harmonic_percussive`` returns one ``AudioSamples`` for each.
"""

import numpy as np

import audio_samples as aus
from audio_samples import effects, mpl

# %%
# A sustained tone mixed with clicks.
sr = 22050
tone = 0.4 * np.sin(2 * np.pi * 440.0 * np.arange(2 * sr) / sr)
clicks = effects.clicks(times=np.arange(0.25, 2.0, 0.25), sr=sr, length=2 * sr)
audio = aus.AudioSamples.new_mono((tone + 0.5 * clicks).astype(np.float32), sr)
audio

# %%
harmonic, percussive = audio.harmonic_percussive()
fig, axes = mpl.plot_waveform_comparison(
    [audio, harmonic, percussive], ["Mix", "Harmonic", "Percussive"], title="HPSS"
)

# %%
# The harmonic part sounds like the tone.
harmonic

# %%
# The percussive part sounds like the clicks.
percussive

# %%
# The two parts add back up to the mix.
residual = audio.to_numpy() - (harmonic + percussive).to_numpy()
print(f"largest residual: {np.max(np.abs(residual)):.2e}")
assert np.max(np.abs(residual)) < 1e-3
