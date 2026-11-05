"""Smoke test of an installed wheel: no Rust toolchain, no source tree.

    python wheel_smoke.py

Imports the package from site-packages (it fails if it is imported from a source
checkout), then exercises WAV and FLAC reading and writing, the STFT, MFCCs and
resampling, and checks the results against independently computed values.
"""

import math
import os
import pickle
import platform
import sys
import tempfile

import audio_samples as aus
import numpy as np


def check(condition, message):
    if not condition:
        raise SystemExit(f"FAILED: {message}")


def main():
    location = os.path.abspath(aus.__file__)
    check("site-packages" in location, f"imported from {location}")

    print(f"audio_samples {aus.__version__} from {location}")
    print(f"python {platform.python_version()} on {platform.machine()} {sys.platform}")
    print(f"numpy {np.__version__}")

    sr = 22050
    n = sr * 2
    t = np.arange(n) / sr
    x = (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    audio = aus.AudioSamples.new_mono(x, sr)
    check(audio.sample_rate == sr, "sample rate")

    error = aus.AudioParameterError("message")
    check(pickle.loads(pickle.dumps(error)).args == ("message",), "pickle of an exception")

    with tempfile.TemporaryDirectory() as d:
        pcm = np.round(x * 32767.0).astype(np.int16)
        source = aus.AudioSamples.new_mono(pcm, sr)
        for ext in ("wav", "flac"):
            path = os.path.join(d, f"tone.{ext}")
            aus.io.save(path, source)
            back = aus.io.read(path, as_type=aus.SampleType.I16)
            check(back.sample_rate == sr, f"{ext} sample rate")
            check(np.array_equal(np.asarray(back.to_numpy()).reshape(-1), pcm), f"{ext} PCM round trip")

    stft = np.asarray(audio.stft_matrix())
    bin_hz = sr / 2048
    peak_hz = float(np.abs(stft).reshape(1025, -1).mean(axis=1).argmax()) * bin_hz
    check(abs(peak_hz - 440.0) < bin_hz, f"STFT peak at {peak_hz} Hz")

    mfcc = np.asarray(audio.mfcc_frames(top_db=80.0))
    check(np.isfinite(mfcc).all() and mfcc.size > 0, "MFCC values")

    resampled = audio.resample(16000, "high")
    expected = math.ceil(n * 16000 / sr)
    y = np.asarray(resampled.to_numpy()).reshape(-1)
    check(resampled.sample_rate == 16000, "resampled rate")
    check(abs(y.size - expected) <= 1, f"resampled length {y.size}, expected {expected}")
    tr = np.arange(y.size) / 16000
    a, b = 2000, y.size - 2000
    fit = np.sum(y[a:b] * np.exp(-2j * np.pi * 440.0 * tr[a:b])) * 2 / (b - a)
    check(abs(abs(fit) - 0.5) < 1e-3, f"resampled amplitude {abs(fit)}")

    from audio_samples import StftParams, WindowType  # noqa: F401

    print("smoke test passed")


if __name__ == "__main__":
    main()
