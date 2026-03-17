import audio_samples as aus
import numpy as np

from audio_samples import SampleType

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

### Audio I/O Operations
sine = aus.sine_wave(440.0, 1.0, 44100)
aus.io.save("sine.wav", sine)

audio = aus.io.read("sine.wav")

# Apply processing
audio.normalize(-1.0, 1.0, aus.NormalizationMethod.peak)
audio.fade_in(0.1, aus.FadeCurve.exponential)
audio.fade_out(0.1, aus.FadeCurve.logarithmic)

# Save processed audio
aus.io.save("output.wav", audio)

# Display audio information
print(repr(audio))

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

print(f"Mono RMS: {mono.rms():.4f}")
print(f"Stereo from Mono RMS: {stereo_from_mono.rms():.4f}")

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
assert autocorr is not None, "Autocorrelation should return a valid array"

cross_corr = sine.cross_correlation(cosine, 500)  # 500 samples max lag

print("Audio Statistics:")
for key, value in stats.items():
    print(f"{key}: {value:.6f}")

print(f"Spectral Centroid: {centroid:.2f} Hz")
print(f"Spectral Rolloff (85%): {rolloff:.2f} Hz")
print(f"Autocorrelation (lag 0): {autocorr[0]:.4f}")
print(f"Cross-correlation (lag 0): {cross_corr[0]:.4f}")

# Non-destructive editing operations
trimmed = audio.trim(0.2, 0.8)              # Keep 0.2s to 0.8s
silence_trimmed = audio.trim_silence(-40.0)  # -40dB threshold

# Audio manipulation
repeated = sine.repeat(3)                    # Repeat 3 times
padded = sine.pad(0.5, 0.5, 0.0)           # 0.5s padding each side

square = aus.square_wave(220.0, duration, sample_rate)
triangle = aus.triangle_wave(330.0, duration, sample_rate)

# Concatenation and segmentation
segments = [sine, square, triangle]
concatenated = aus.AudioSamples.concatenate(segments)
split_segments = concatenated.split(0.5)    # 0.5s segments

# Audio processing
audio.scale(0.5)                            # 50% volume
audio.normalize(-1.0, 1.0, aus.NormalizationMethod.minmax)        # Normalize range
audio.remove_dc_offset()                    # Remove DC bias
audio.clip(-0.8, 0.8)                       # Soft clipping

print(f"Trimmed duration: {trimmed.duration_seconds:.2f} seconds")
print(f"Silence-trimmed duration: {silence_trimmed.duration_seconds:.2f} seconds")
print(f"Repeated duration: {repeated.duration_seconds:.2f} seconds")
print(f"Padded duration: {padded.duration_seconds:.2f} seconds")
print(f"Concatenated duration: {concatenated.duration_seconds:.2f} seconds")
print(f"Number of split segments: {len(split_segments)}")


# Performant resampling
original = aus.sine_wave(440.0, 0.5, 44100)

# Resample to different rates
upsampled = original.resample(88200, aus.ResamplingQuality.fast)     # 2x upsampling
downsampled = original.resample(22050, aus.ResamplingQuality.medium)   # 2x downsampling
ratio_resampled = original.resample_by_ratio(1.5, aus.ResamplingQuality.high)  # 1.5x rate

# Multiple sample format support
# i16, i32, f32, f64 with type-safe conversions

audio_i16 = original.to_format(SampleType.I16)
audio_i32  = original.to_format(SampleType.I32)
audio_f32 = original.to_format(SampleType.F32)
audio_f64 = original.to_format(SampleType.F64)

# Built-in digital filters
audio = aus.sine_wave(440.0, 0.5, sample_rate)

# Apply filters in-place
audio.low_pass_filter(1000.0)              # Low-pass at 1kHz
audio.high_pass_filter(100.0)              # High-pass at 100Hz
audio.band_pass_filter(200.0, 2000.0)      # Band-pass 200Hz-2kHz

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