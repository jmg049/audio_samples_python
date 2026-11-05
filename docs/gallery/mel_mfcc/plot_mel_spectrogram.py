"""
Mel spectrogram
===============

The power spectrogram projected onto mel bands, computed in one method call and
then again from an existing spectrogram.
"""

import matplotlib.pyplot as plt
import numpy as np

import audio_samples as aus
from audio_samples import spectral

# %%
sr = 22050
audio = aus.generation.chirp(100.0, 9000.0, 2.0, sr).as_f32()

# %%
# ``melspectrogram`` returns ``(channels, n_mels, frames)``.
mel = audio.melspectrogram(n_mels=64, n_fft=2048, hop_length=512)
print(mel.shape)
db = spectral.power_to_db(mel[0], ref="max", top_db=80.0)

fig, ax = plt.subplots(figsize=(7, 3.5))
im = ax.imshow(db, origin="lower", aspect="auto", extent=[0, audio.duration_seconds, 0, 64])
ax.set(xlabel="Time (s)", ylabel="Mel band", title="Mel spectrogram (dB)")
fig.colorbar(im, ax=ax, label="dB")

# %%
# With a spectrogram already computed, project it directly. The result matches.
power = np.abs(audio.stft_matrix(n_fft=2048, hop_length=512)) ** 2
projected = spectral.melspectrogram_from_spectrogram(power[0], sr=sr, n_fft=2048, n_mels=64)
assert np.allclose(projected, mel[0], rtol=1e-4, atol=1e-6)
