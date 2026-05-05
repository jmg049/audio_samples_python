<div align="center">

# AudioSamples Python

## Fast, simple, and expressive audio processing and IO in Python

<img src="logo.png" title="AudioSamples Logo -- Ferrous' Mustachioed Cousin From East Berlin, Eisenhaltig" width="200"/>

[![PyPI][pypi-img]][pypi] [![Docs][docs-img]][docs] [![License: MIT][license-img]][license]
</div>

---

## Overview

Python bindings for the high-performance AudioSamples Rust ecosystem. AudioSamples eliminates the manual metadata coordination burden that plagues existing audio processing libraries by treating audio as a first-class data type with intrinsically embedded properties.

Current audio processing workflows suffer from artificial complexity inherited from C-era design patterns. Libraries like librosa, soundfile, and torchaudio force researchers to manually coordinate sample rates, channel layouts, and format information across every function call, creating cognitive overhead and error-prone workflows. AudioSamples eliminates this coordination burden by embedding audio properties intrinsically within the data structure, enabling automatic property preservation through processing pipelines.


```text
audio_samples
├── core types, signal generation, DSP, analysis
├── audio_samples_io
│   └── file I/O: read and write WAV and FLAC
├── audio_samples_python
│   └── python bindings via PyO3
├── audio_samples_streaming
│   └── streaming pipeline, device I/O, rodio, async
└── audio_samples_ml
    └── STT and TTS integrations
```

| Crate | What it provides | Start here if... |
|---|---|---|
| [`audio_samples`](https://crates.io/crates/audio_samples) | `AudioSamples<T>`, core types, signal generation, DSP, analysis | You need in-memory audio representations or signal processing primitives |
| [`audio_samples_io`](https://crates.io/crates/audio_samples_io) | Read and write WAV and FLAC files | You need to load or save audio files with minimal setup |
| [`audio_samples_python`](https://crates.io/crates/audio_samples_python) | Python bindings via PyO3 and NumPy interop | You want to use the library from Python or integrate with Python workflows |
| [`audio_samples_streaming`](https://crates.io/crates/audio_samples_streaming) | Chunk-based streaming, real-time device I/O, async pipelines | You need real-time audio processing or streaming pipelines |
| [`audio_samples_ml`](https://crates.io/crates/audio_samples_ml) | Speech-to-text (STT) and text-to-speech (TTS) integrations | You want to integrate transcription or synthesis into your audio pipeline |

## Why AudioSamples?

### The Problem with Existing Libraries

**Manual Metadata Coordination**: Every operation requires passing sample rates and format information manually:

```python
# Traditional approach - error-prone manual coordination
data, sr = soundfile.read('audio.wav') # silently converts all samples in 'audio.wav' to 'float64'
stft = librosa.stft(data, sr=sr)  # Must pass sr manually
freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)  # Must pass sr again
```

**Hidden Operations**: Libraries like librosa perform transformative operations without user awareness, automatically resampling to non-standard sample rates (22050Hz) and converting formats, compromising research reproducibility.

**Performance Bottlenecks**: Popular frameworks like TorchAudio suffer from inefficient I/O implementations that deliver 4-10x slower performance than focused audio libraries, despite corporate backing.

### The AudioSamples Solution

**Audio-First Design**: Audio objects carry sample rates, channel configurations, and format information intrinsically:

```python
# AudioSamples approach - automatic coordination
audio = aus.io.read('audio.wav') # Reads samples in the same encoding as the file itself. Can pass a dtype argument to specify the target type if desired. YOU make that choice.
stft_matrix, freqs = audio.stft_with_freqs(window_size=2048, hop_size=512)
```

**Explicit Operations Only**: The aim is that no implicit behavior occurs without user consent (trying to find any and all remainging). AudioSamples preserves original audio properties unless explicitly requested to change them.

**Performance by Default**: Rust foundations deliver 2-4x faster I/O operations compared to established Python libraries while maintaining full ecosystem interoperability.

## Installation

```bash
pip install audio_samples
```

## Quick Start

### Basic Audio Generation and Processing

```python
import audio_samples as aus
import numpy as np

# Audio parameters
duration = 1.0  # seconds
sample_rate = 44100

# Generate audio signals
sine = aus.sine_wave(440.0, duration, sample_rate)
cosine = aus.cosine_wave(880.0, duration, sample_rate)
white_noise = aus.white_noise(duration, sample_rate)

# Mix signals with operator overloading
mixed = sine + cosine * 0.3 + white_noise * 0.05

# Audio analysis (built-in, no external libraries needed)
print(f"RMS level: {mixed.rms():.4f}")
print(f"Mean: {mixed.mean():.6f}")
print(f"Zero crossings: {mixed.zero_crossings()}")
print(f"Spectral centroid: {mixed.spectral_centroid():.2f} Hz")
```

### Audio I/O Operations

```python
### Audio I/O Operations
# create some audio to work with and save it
sine = aus.sine_wave(440.0, 1.0, 44100)
aus.io.save("sine.wav", sine)

# read audio file
audio = aus.io.read("sine.wav")
# audio = aus.io.read("sine.wav", dtype=np.float32)  # Optionally specify target dtype (default is the file's native encoding)

# Apply processing
audio.normalize(-1.0, 1.0, aus.NormalizationMethod.peak)
audio.fade_in(0.1, aus.FadeCurve.exponential)
audio.fade_out(0.1, aus.FadeCurve.logarithmic)

# Save processed audio
aus.io.save("output.wav", audio)

# Display audio information
print(repr(audio))
```

## Performance Benchmarks

The I/O layer is built also built in Rust: a streaming WAV header parser feeds a single `read_exact` call directly into a NumPy-owned buffer — matching the speed of `np.fromfile`. Writes use a Fortran-layout fast path (F-order interleaved arrays write in one `write_all` call) and a tiled streaming interleave for C-order arrays, eliminating all intermediate allocations.

Benchmarked against 80 configurations (4 sample rates × 2 dtypes × 5 durations × 2 channel counts) with 100 iterations each, warmed up over 20 runs:

- **Load**: faster than scipy in 80/80 configurations, faster than librosa in 78/80
- **Save**: faster than scipy in 73/80 configurations, faster than librosa in 80/80

### Load times — 44,100 Hz (ms, lower is better)

| Config | audio_samples | scipy | librosa | wave |
|---|---|---|---|---|
| i16, mono, 1s | **0.008** | 0.012 | 0.024 | 0.010 |
| i16, mono, 30s | **0.101** | 0.171 | 0.122 | 0.178 |
| i16, mono, 60s | **0.187** | 0.277 | 0.202 | 0.282 |
| i16, stereo, 1s | **0.009** | 0.015 | 0.027 | 0.013 |
| i16, stereo, 30s | **0.178** | 0.265 | 0.218 | 0.285 |
| i16, stereo, 60s | **0.450** | 0.612 | 0.518 | 0.577 |
| f32, mono, 1s | **0.010** | 0.016 | 0.025 | — |
| f32, mono, 30s | **0.192** | 0.279 | 0.228 | — |
| f32, mono, 60s | **0.393** | 0.491 | 0.420 | — |
| f32, stereo, 1s | **0.013** | 0.021 | 0.029 | — |
| f32, stereo, 30s | **0.436** | 0.530 | 0.435 | — |
| f32, stereo, 60s | **1.696** | 1.955 | 1.703 | — |

### Save times — 44,100 Hz (ms, lower is better)

| Config | audio_samples | scipy | librosa | wave |
|---|---|---|---|---|
| i16, mono, 1s | **0.007** | 0.012 | 0.022 | 0.010 |
| i16, mono, 30s | **0.114** | 0.128 | 0.133 | 0.128 |
| i16, mono, 60s | **0.264** | 0.255 | 0.276 | 0.263 |
| i16, stereo, 1s | **0.010** | 0.015 | 0.025 | 0.013 |
| i16, stereo, 30s | **0.267** | 0.297 | 0.295 | 0.267 |
| i16, stereo, 60s | **0.597** | 0.593 | 0.606 | 0.596 |
| f32, mono, 1s | **0.010** | 0.015 | 0.149 | — |
| f32, mono, 30s | **0.265** | 0.268 | 4.204 | — |
| f32, mono, 60s | **0.579** | 0.625 | 8.484 | — |
| f32, stereo, 1s | **0.017** | 0.022 | 0.306 | — |
| f32, stereo, 30s | **0.583** | 0.612 | 8.369 | — |
| f32, stereo, 60s | **2.101** | 2.030 | 17.085 | — |

*Times in milliseconds (mean over 100 iterations). `—` = library does not support this format natively.*

## Feature Showcase

### 1. Signal Generation

AudioSamples provides precise waveform generators:

```python
# Basic waveforms
sine = aus.generation.sine_wave(440.0, duration, sample_rate)
sawtooth = aus.generation.sawtooth_wave(220.0, duration, sample_rate)
square = aus.generation.square_wave(110.0, duration, sample_rate)
triangle = aus.generation.triangle_wave(660.0, duration, sample_rate)

# Advanced signals
chirp = aus.generation.chirp(100.0, 2000.0, duration, sample_rate)
impulse = aus.generation.impulse(100, sample_rate)  # 100ms impulse

# Noise generators
pink = aus.generation.pink_noise(duration, sample_rate)
brown = aus.generation.brown_noise(duration, sample_rate)
```

### 2. Multi-Channel Operations

```python
# Create stereo using the proper AudioSamples methods
left_channel = sine
right_channel = cosine * 0.7
stereo = aus.AudioSamples.stack([left_channel, right_channel])

# Channel operations
stereo.pan(0.3)           # Pan 30% to the right
stereo.balance(0.2)       # 20% balance adjustment
stereo.swap_channels(0, 1) # Swap left and right

# Channel extraction and conversion
mono = stereo.to_mono(aus.MonoConversionMethod.average)
stereo_from_mono = sine.to_stereo(aus.StereoConversionMethod.duplicate)
```

### 3. Audio Analysis and Statistics

```python
# Built-in audio statistics - no external libraries needed
test_signal = sine + cosine * 0.3 + white_noise * 0.05

stats = {
    'mean': test_signal.mean(),
    'rms': test_signal.rms(),
    'variance': test_signal.variance(),
    'std_dev': test_signal.std_dev(),
    'zero_crossings': test_signal.zero_crossings(),
    'crossing_rate': test_signal.zero_crossing_rate()
}

# Spectral analysis
centroid = test_signal.spectral_centroid()
rolloff = test_signal.spectral_rolloff(0.85)  # 85th percentile

# Correlation analysis
autocorr = test_signal.autocorrelation(1000)   # 1000 samples max lag
cross_corr = sine.cross_correlation(cosine, 500)
```

### 4. Audio Editing and Effects

```python
# Non-destructive editing operations
trimmed = audio.trim(0.2, 0.8)              # Keep 0.2s to 0.8s
silence_trimmed = audio.trim_silence(-40.0)  # -40dB threshold

# Audio manipulation
repeated = sine.repeat(3)                    # Repeat 3 times
padded = sine.pad(0.5, 0.5, 0.0)           # 0.5s padding each side

# Concatenation and segmentation
segments = [sine, square, triangle]
concatenated = aus.AudioSamples.concatenate(segments)
split_segments = concatenated.split(0.5)    # 0.5s segments

# Audio processing
audio.scale(0.5)                            # 50% volume
audio.normalize(-1.0, 1.0, 'minmax')        # Normalize range
audio.remove_dc_offset()                    # Remove DC bias
audio.clip(-0.8, 0.8)                       # Soft clipping
```

### 5. Resampling and Format Conversion

```python
# High-quality resampling
original = aus.generation.sine_wave(440.0, 0.5, 44100)

# Resample to different rates
upsampled = original.resample(88200, 'high')     # 2x upsampling
downsampled = original.resample(22050, 'high')   # 2x downsampling
ratio_resampled = original.resample_by_ratio(1.5, 'high')  # 1.5x rate

# Multiple sample format support
# i16, i32, f32, f64 with type-safe conversions

audio_i16 = original.to_format(SampleType.I16)
audio_i32  = original.to_format(SampleType.I32)
audio_f32 = original.to_format(SampleType.F32)
audio_f64 = original.to_format(SampleType.F64)
```

### 6. Digital Filtering

```python
# Built-in digital filters
audio = aus.generation.sine_wave(440.0, 0.5, sample_rate)

# Apply filters in-place
audio.low_pass_filter(1000.0)              # Low-pass at 1kHz
audio.high_pass_filter(100.0)              # High-pass at 100Hz
audio.band_pass_filter(200.0, 2000.0)      # Band-pass 200Hz-2kHz
```

## NumPy Integration

AudioSamples provides seamless NumPy interoperability while encouraging the use of audio-specific methods:

```python
left = aus.sine_wave(440.0, duration, sample_rate)
right = aus.sine_wave(880.0, duration, sample_rate)

# GOOD: Use AudioSamples methods for audio operations
stereo = aus.AudioSamples.stack([left, right])           # Proper multi-channel
concatenated = aus.AudioSamples.concatenate([left, right])    # Proper concatenation
audio.fade_in(0.1, aus.FadeCurve.linear)                            # Built-in audio fades
resampled = audio.resample(48000, aus.ResamplingQuality.high)                # Proper resampling

# ALSO GOOD: Use NumPy for mathematical operations
gain_curve = np.linspace(0.1, 1.0, len(audio))
gained_audio = audio * gain_curve                        # Custom gain curves

window = np.hanning(len(audio))
windowed_audio = audio * window                          # Windowing

# Custom mathematical transformations
distortion = np.tanh(audio.to_numpy() * 3.0)            # Custom distortion
distorted_audio = aus.AudioSamples.new_mono(distortion, sample_rate)
```

## Requirements

- Python >= 3.8
- NumPy >= 1.24.4

## License

MIT License

## Contributing

Contributions are welcome! This package is part of the broader AudioSamples ecosystem:

- [`audio_samples`](https://github.com/jmg049/audio_samples) - Core Rust library
- [`audio_samples_io`](https://github.com/jmg049/audio_samples_io) - Audio I/O for Rust
- [`audio_samples_python`](https://github.com/jmg049/audio_samples_python) - This package

Read [Contributing](CONTRIBUTING.md) for more details.

## Citing

If you use AudioSamples in research, please cite:

```bibtex
@inproceedings{geraghty2026audio,
  author    = {Geraghty, Jack and Golpayegani, Fatemeh and Hines, Andrew},
  title     = {Audio Made Simple: A Modern Framework for Audio Processing},
  booktitle = {ACM Multimedia Systems Conference 2026 (MMSys '26)},
  year      = {2026},
  month     = apr,
  publisher = {ACM},
  address   = {Hong Kong, Hong Kong},
  doi       = {10.1145/3793853.3799811},
  note      = {Accepted for publication}
}
```

[pypi]: https://pypi.org/project/audio_samples/
[pypi-img]: https://img.shields.io/pypi/v/audio_samples?style=for-the-badge&color=009E73&label=PyPI

[docs]: https://jmg049.github.io/audio_samples_python/
[docs-img]: https://img.shields.io/pypi/v/audio_samples?style=for-the-badge&color=009E73&label=Docs

[license-img]: https://img.shields.io/crates/l/audio_samples?style=for-the-badge&label=license&labelColor=gray
[license]: https://github.com/jmg049/audio_samples_python/blob/main/LICENSE