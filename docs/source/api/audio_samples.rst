AudioSamples Core
================

The AudioSamples class is the core of the library, providing a type-safe audio representation with
intrinsically embedded properties (sample rate, channel layout, format) that eliminates manual
metadata coordination.

AudioSamples Class
------------------

The main audio processing class supporting multiple sample formats and comprehensive audio operations.
All methods preserve audio properties automatically and support operator overloading for natural
audio mathematics.

Key Features
~~~~~~~~~~~~

**Constructors and Factory Methods**
   - Create audio from NumPy arrays with automatic type detection
   - Generate zero-filled or ones-filled audio in multiple formats (i16, i32, f32, f64)
   - Create uniform-value audio with specified sample values

**Audio Editing and Manipulation**
   - Concatenate multiple audio segments end-to-end
   - Stack mono sources into multi-channel audio
   - Repeat, trim, pad, and split audio with precise timing
   - Mix multiple audio sources with optional weights
   - Apply fade-in and fade-out effects with multiple curve types

**Channel Operations**
   - Pan stereo audio left/right with precise control
   - Balance stereo channels independently
   - Convert between mono and stereo with various methods
   - Extract individual channels or swap channel positions

**Spectral Analysis and Transforms**
   - Short-Time Fourier Transform (STFT) with configurable windows
   - Inverse STFT for signal reconstruction
   - Magnitude and phase spectrograms with multiple scaling options
   - Mel-frequency spectrograms and MFCC features
   - Chroma features for harmonic analysis

**Audio Processing**
   - High-quality resampling with configurable quality levels
   - Window function application for signal processing
   - Sample rate conversion by exact ratios

**Pitch Analysis**
   - YIN algorithm for fundamental frequency detection
   - Time-based pitch tracking with multiple detection methods
   - Configurable frequency range and threshold parameters

**Audio Decomposition**
   - Harmonic-Percussive Source Separation (HPSS)
   - Configurable median filter sizes and mask softness

**Digital Filtering**
   - Butterworth filters (lowpass, highpass, bandpass)
   - Simple filters with cutoff frequency control
   - IIR filter design and application
   - Chebyshev Type I filters with passband ripple control

**Statistics and Analysis**
   - Peak, minimum, and maximum sample values
   - Mean, RMS, variance, and standard deviation
   - Zero crossing detection and rate calculation
   - Autocorrelation analysis with configurable lag
   - Spectral centroid and rolloff calculations

**Amplitude Processing**
   - Amplitude scaling with precise factor control
   - Normalization with multiple methods (peak, minmax, RMS)
   - Soft clipping to prevent distortion
   - DC offset removal for signal cleanup

**Dynamic Range Processing**
   - Compression with configurable threshold, ratio, and timing
   - Limiting with lookahead and release control
   - Noise gating for signal cleanup
   - Expansion for dynamic range enhancement

**Equalization**
   - Parametric equalizer with multiple band support
   - Peak, low-shelf, and high-shelf filters
   - Three-band EQ with configurable frequencies
   - Real-time frequency response calculation

**File Format Support**
   - Multiple sample formats: i16, i24, i32, f32, f64
   - Automatic format detection and conversion
   - Type-safe casting between sample formats

**NumPy Integration**
   - Seamless operator overloading for mathematical operations
   - Support for numpy universal functions
   - Automatic property preservation through all operations

Filter and EQ Configuration Classes
-----------------------------------

**IirFilterDesign**
   Configuration class for Infinite Impulse Response filters supporting:

   - Butterworth, Chebyshev Type I/II, and Elliptic filter types
   - Lowpass, highpass, bandpass, and bandstop responses
   - Configurable filter order and cutoff frequencies

**EqBand**
   Individual equalizer band configuration with:

   - Peak, lowshelf, highshelf, lowpass, highpass, and bandpass types
   - Center/cutoff frequency specification in Hz
   - Gain control in dB with positive/negative values
   - Q factor for bandwidth control

**ParametricEq**
   Multi-band parametric equalizer supporting:

   - Multiple EQ bands with independent configuration
   - Real-time frequency response calculation
   - Cascaded filter application for natural sound