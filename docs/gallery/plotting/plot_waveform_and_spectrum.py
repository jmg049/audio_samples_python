"""
Waveform, spectrum and spectrogram
==================================

The three most used figures, drawn into one Matplotlib figure with ``ax=``.
"""

import matplotlib.pyplot as plt

import audio_samples as aus
from audio_samples import mpl

# %%
sr = 22050
audio = aus.generation.compound_tone([(330.0, 0.6), (660.0, 0.3), (990.0, 0.1)], 1.0, sr).as_f32()
audio.fade_in(0.1, aus.FadeCurve.linear)
audio.fade_out(0.3, aus.FadeCurve.linear)

fig, axes = plt.subplots(3, 1, figsize=(7, 7))
mpl.plot_waveform(audio, show_envelope=True, ax=axes[0])
mpl.plot_magnitude_spectrum(audio, freq_max=4000.0, ax=axes[1])
mpl.plot_spectrogram(audio, freq_max=4000.0, ax=axes[2])
fig.tight_layout()
