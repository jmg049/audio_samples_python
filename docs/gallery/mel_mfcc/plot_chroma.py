"""
Chroma
======

Energy per pitch class, folded across octaves. A chord shows up as its three
notes whatever the octave. ``estimate_key`` uses the same idea.
"""

import matplotlib.pyplot as plt

import audio_samples as aus
from audio_samples import utils

# %%
# A C major chord followed by an A minor chord.
sr = 22050


def chord(names, seconds):
    tones = [aus.generation.sine_wave(utils.note_to_frequency(n), seconds, sr).as_f32() for n in names]
    return aus.AudioSamples.mix(tones)


audio = aus.AudioSamples.concatenate([chord(["C4", "E4", "G4"], 1.0), chord(["A3", "C4", "E4"], 1.0)])
audio

# %%
# ``chroma_stft`` returns ``(channels, 12, frames)``, row 0 being C.
chroma = audio.chroma_stft()[0]

fig, ax = plt.subplots(figsize=(7, 3))
ax.imshow(chroma, origin="lower", aspect="auto", extent=[0, audio.duration_seconds, 0, 12])
ax.set_yticks([i + 0.5 for i in range(12)])
ax.set_yticklabels(["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"])
ax.set(xlabel="Time (s)", title="Chroma")

# %%
# The three strongest pitch classes of the first second are C, E and G.
first = chroma[:, : chroma.shape[1] // 2 - 2].mean(axis=1)
assert set(first.argsort()[-3:]) == {0, 4, 7}
