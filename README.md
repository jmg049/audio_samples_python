<div align="center">

# AudioSamples Python

## Fast, simple, and expressive audio processing and IO in Python

<img src="logo.png" title="AudioSamples Logo -- Ferrous' Mustachioed Cousin From East Berlin, Eisenhaltig" width="200"/>

[![PyPI][pypi-img]][pypi] [![License: MIT][license-img]][license]
</div>

---

## Overview

Python bindings for the high-performance AudioSamples Rust ecosystem. AudioSamples eliminates the manual metadata coordination burden that plagues existing audio processing libraries by treating audio as a first-class data type with intrinsically embedded properties.

Current audio processing workflows suffer from artificial complexity inherited from C-era design patterns. Libraries like librosa, soundfile, and torchaudio force researchers to manually coordinate sample rates, channel layouts, and format information across every function call, creating cognitive overhead and error-prone workflows. AudioSamples eliminates this coordination burden by embedding audio properties intrinsically within the data structure, enabling automatic property preservation through processing pipelines.

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
sine = aus.generation.sine_wave(440.0, duration, sample_rate)
cosine = aus.generation.cosine_wave(880.0, duration, sample_rate)
white_noise = aus.generation.white_noise(duration, sample_rate)

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
import audio_samples as aus

# Read audio file (auto-detects format), but you can still specify the dtype is necessary
audio = aus.io.read("input.wav")

# Apply processing
audio.normalize(-1.0, 1.0, 'peak')
audio.fade_in(0.1, 'exponential')
audio.fade_out(0.1, 'logarithmic')

# Save processed audio
aus.io.save("output.wav", audio)

# Display audio information
print(audio.info())
# Sample rate: 44100 Hz
# Channels: 2
# Duration: 10.5s
# Samples per channel: 462000
```

## Performance Benchmarks

AudioSamples consistently outperforms other Python audio libraries across tested read and write scenarios:

### Reading Performance

> Note: All audio was sampled at 44,100Hz with f32 sample encoding.

| **Library**    | **audio_samples** | **scipy** | **soundfile** | **torchaudio** |
|----------------|-------------------|-----------|---------------|----------------|
| **0.1s, 1ch**  | 6.09e-05          | 1.46e-04  | 2.41e-04      | 2.35e-01       |
| **0.1s, 2ch**  | 6.75e-05          | 1.62e-04  | 2.55e-04      | 5.72e-04       |
| **0.5s, 1ch**  | 7.79e-05          | 1.65e-04  | 2.66e-04      | 6.45e-04       |
| **0.5s, 2ch**  | 8.73e-05          | 1.77e-04  | 2.84e-04      | 9.04e-04       |
| **1.0s, 1ch**  | 8.66e-05          | 1.73e-04  | 2.79e-04      | 7.81e-04       |
| **1.0s, 2ch**  | 1.13e-04          | 1.99e-04  | 3.01e-04      | 9.89e-04       |
| **2.0s, 1ch**  | 1.08e-04          | 1.90e-04  | 3.04e-04      | 9.73e-04       |
| **2.0s, 2ch**  | 1.61e-04          | 2.50e-04  | 3.54e-04      | 1.37e-03       |
| **5.0s, 1ch**  | 1.86e-04          | 2.70e-04  | 3.75e-04      | 1.51e-03       |
| **5.0s, 2ch**  | 3.24e-04          | 3.97e-04  | 5.06e-04      | 2.54e-03       |
| **10.0s, 1ch** | 3.30e-04          | 3.97e-04  | 5.06e-04      | 2.46e-03       |
| **10.0s, 2ch** | 6.05e-04          | 6.32e-04  | 7.38e-04      | 4.32e-03       |
| **30.0s, 1ch** | 8.27e-04          | 8.25e-04  | 9.37e-04      | 5.97e-03       |
| **30.0s, 2ch** | 1.62e-03          | 1.48e-03  | 1.61e-03      | 1.12e-02       |
| **60.0s, 1ch** | 1.54e-03          | 1.44e-03  | 1.55e-03      | 1.09e-02       |
| **60.0s, 2ch** | 2.19e-03          | 2.77e-03  | 2.89e-03      | 2.03e-02       |

### Writing Performance

> Note: All audio was sampled at 44,100Hz with f32 sample encoding.

| **Library**    | **audio_samples** | **scipy** | **soundfile** | **torchaudio** |
|----------------|-------------------|-----------|---------------|----------------|
| **0.1s, 1ch**  | 6.21e-05          | 2.23e-04  | 2.71e-04      | 2.93e-04       |
| **0.1s, 2ch**  | 8.72e-05          | 2.58e-04  | 2.95e-04      | 2.67e-04       |
| **0.5s, 1ch**  | 7.71e-05          | 2.34e-04  | 2.99e-04      | 3.56e-04       |
| **0.5s, 2ch**  | 1.49e-04          | 3.47e-04  | 3.98e-04      | 3.57e-04       |
| **1.0s, 1ch**  | 9.52e-05          | 2.61e-04  | 3.38e-04      | 4.57e-04       |
| **1.0s, 2ch**  | 2.16e-04          | 4.43e-04  | 5.36e-04      | 4.68e-04       |
| **2.0s, 1ch**  | 1.38e-04          | 2.90e-04  | 4.21e-04      | 6.59e-04       |
| **2.0s, 2ch**  | 3.46e-04          | 6.26e-04  | 7.96e-04      | 6.71e-04       |
| **5.0s, 1ch**  | 2.51e-04          | 4.16e-04  | 6.77e-04      | 1.26e-03       |
| **5.0s, 2ch**  | 7.67e-04          | 1.21e-03  | 1.63e-03      | 1.28e-03       |
| **10.0s, 1ch** | 4.45e-04          | 6.14e-04  | 1.11e-03      | 2.34e-03       |
| **10.0s, 2ch** | 1.51e-03          | 2.18e-03  | 2.93e-03      | 2.29e-03       |
| **30.0s, 1ch** | 1.20e-03          | 1.46e-03  | 2.91e-03      | 6.25e-03       |
| **30.0s, 2ch** | 4.61e-03          | 6.23e-03  | 7.90e-03      | 6.32e-03       |
| **60.0s, 1ch** | 2.25e-03          | 2.81e-03  | 5.60e-03      | 1.16e-02       |
| **60.0s, 2ch** | 1.63e-02          | 1.24e-02  | 1.59e-02      | 1.16e-02       |

*Times in seconds, lower is better*

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
mono = stereo.to_mono('average')
stereo_from_mono = sine.to_stereo('duplicate')
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
# i16, i24, i32, f32, f64 with type-safe conversions
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
# GOOD: Use AudioSamples methods for audio operations
stereo = aus.AudioSamples.stack([left, right])           # Proper multi-channel
concatenated = aus.AudioSamples.concatenate([a1, a2])    # Proper concatenation
audio.fade_in(0.1, 'linear')                            # Built-in audio fades
resampled = audio.resample(48000, 'high')                # Proper resampling

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

## Documentation

Full API documentation will be available at the project homepage once published.

## License

MIT License

## Contributing

Contributions are welcome! This package is part of the broader AudioSamples ecosystem:

- [`audio_samples`](https://github.com/jmg049/audio_samples) - Core Rust library
- [`audio_samples_io`](https://github.com/jmg049/audio_samples_io) - Audio I/O for Rust
- [`audio_samples_python`](https://github.com/jmg049/audio_samples_python) - This package

Read [Contributing](CONTRIBUTING.md) for more details.

[pypi]: https://pypi.org/project/audio_samples/
[pypi-img]: https://img.shields.io/pypi/v/audio_samples?style=for-the-badge&color=009E73&label=PyPI

[license-img]: https://img.shields.io/crates/l/audio_samples?style=for-the-badge&label=license&labelColor=gray
[license]: https://github.com/jmg049/audio_samples_python/blob/main/LICENSE