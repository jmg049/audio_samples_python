"""
Harmonics of a tone
===================

``detect_pitch_yin`` gives one number for a whole signal, and ``mpl.plot_harmonics_bar``
shows the level of each harmonic at that pitch.
"""

import audio_samples as aus
from audio_samples import mpl

# %%
# A sawtooth has every harmonic, each weaker than the one before.
sr = 22050
audio = aus.generation.sawtooth_wave_bandlimited(220.0, 1.0, sr).as_f32()

f0 = audio.detect_pitch_yin(threshold=0.1, min_frequency=65.0, max_frequency=800.0)
print(f"f0 = {f0:.1f} Hz")
assert abs(f0 - 220.0) < 2.0

fig, ax = mpl.plot_harmonics_bar(audio, fundamental_freq=f0, num_harmonics=10)

# %%
audio
