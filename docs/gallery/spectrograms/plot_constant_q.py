"""
Constant-Q transform
====================

A log-frequency transform with the same number of bins per octave, which suits
music: a scale becomes evenly spaced steps.
"""

import matplotlib.pyplot as plt
import numpy as np

import audio_samples as aus
from audio_samples import spectral, utils

# %%
# A one-octave C major scale of harmonic notes, half a second per note.
sr = 22050
notes = ["C4", "D4", "E4", "F4", "G4", "A4", "B4", "C5"]
freqs = [utils.note_to_frequency(n) for n in notes]
scale = aus.AudioSamples.concatenate(
    [
        aus.generation.compound_tone([(f, 0.6), (2 * f, 0.3), (3 * f, 0.1)], 0.5, sr).as_f32()
        for f in freqs
    ]
)
scale

# %%
# ``cqt`` returns ``(channels, bins, frames)``.
fmin = 65.41
c = scale.cqt(fmin=fmin, n_bins=60, bins_per_octave=12)[0]
db = spectral.amplitude_to_db(np.abs(c), ref="max", top_db=60.0)

fig, ax = plt.subplots(figsize=(7, 3.5))
ax.imshow(db, origin="lower", aspect="auto", extent=[0, scale.duration_seconds, 0, 60])
ax.set(xlabel="Time (s)", ylabel="Semitones above C2", title="Constant-Q transform (dB)")

# %%
# The strongest bin of the first note is C4, 24 semitones above C2.
first_note = np.abs(c[:, 5:15]).mean(axis=1)
assert int(first_note.argmax()) == 24
