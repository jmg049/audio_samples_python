"""
Read and write WAV and FLAC
===========================

Saves a signal as 16-bit WAV and as FLAC, reads both back, and checks that the
lossless files return the same samples. ``io.info`` reads the header alone.
"""

import tempfile
from pathlib import Path

import numpy as np

import audio_samples as aus
from audio_samples import mpl

# %%
# Generate one second of a chirp and convert it to 16-bit integers, the sample
# type most WAV files use.
audio = aus.generation.chirp(200.0, 4000.0, 1.0, 22050).as_i16()

folder = Path(tempfile.mkdtemp())
wav_path = folder / "chirp.wav"
flac_path = folder / "chirp.flac"
aus.io.save(str(wav_path), audio)
aus.io.save(str(flac_path), audio)

# %%
# ``io.info`` reads the header without decoding the samples.
info = aus.io.info(str(flac_path))
print(info.sample_rate, info.channels, info.bits_per_sample, info.num_samples)

# %%
# ``io.read`` keeps the file's sample type unless ``as_type`` asks for another.
from_wav = aus.io.read(str(wav_path))
from_flac = aus.io.read(str(flac_path))
assert from_wav.dtype == audio.dtype
assert np.array_equal(from_wav.to_numpy(), audio.to_numpy())
assert np.array_equal(from_flac.to_numpy(), audio.to_numpy())

print(f"WAV {wav_path.stat().st_size} bytes, FLAC {flac_path.stat().st_size} bytes")

# %%
# ``as_type=SampleType.F32`` maps the integers to floats in [-1, 1].
as_float = aus.io.read(str(flac_path), as_type=aus.SampleType.F32)
assert abs(as_float.peak()) <= 1.0

fig, ax = mpl.plot_waveform(as_float, title="Read back from FLAC, as float32")

# %%
# An ``AudioSamples`` shows a player in a notebook or on this page.
as_float
