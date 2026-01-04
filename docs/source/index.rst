AudioSamples Python Documentation
===================================

**Fast, simple, and expressive audio processing for Python**

AudioSamples Python provides high-performance audio processing capabilities through Rust-powered Python bindings.
It eliminates the manual metadata coordination burden that plagues existing audio processing libraries by treating
audio as a first-class data type with intrinsically embedded properties.

.. note::
   AudioSamples consistently outperforms other Python audio libraries, delivering 2-4x faster I/O operations
   compared to established libraries like scipy, soundfile, and torchaudio.

Quick Start
-----------

Install AudioSamples:

.. code-block:: bash

   pip install audio_samples

Generate and process audio:

.. code-block:: python

   import audio_samples as aus
   import numpy as np

   # Generate audio signals
   sine = aus.generation.sine_wave(440.0, 1.0, 44100)
   cosine = aus.generation.cosine_wave(880.0, 1.0, 44100)

   # Mix signals with operator overloading
   mixed = sine + cosine * 0.3

   # Audio analysis (built-in, no external libraries needed)
   print(f"RMS level: {mixed.rms():.4f}")
   print(f"Spectral centroid: {mixed.spectral_centroid():.2f} Hz")

Key Features
------------

- **High Performance**: Rust-powered audio processing with minimal Python overhead
- **Type Safety**: Audio semantics encoded in the data structure, not external metadata
- **Rich Audio Operations**: Built-in signal generation, analysis, and processing
- **NumPy Integration**: Seamless interoperability with the NumPy ecosystem
- **Multi-channel Support**: First-class support for mono, stereo, and multi-channel audio
- **Real-time Analysis**: Spectral analysis, statistics, and audio metrics
- **Audio Effects**: Fading, normalization, filtering, and more
- **Fast I/O**: High-performance audio file reading and writing

Why AudioSamples?
-----------------

**The Problem with Existing Libraries**

Current Python audio libraries force manual metadata coordination:

.. code-block:: python

   # Traditional approach - error-prone manual coordination
   data, sr = soundfile.read('audio.wav')
   stft = librosa.stft(data, sr=sr)  # Must pass sr manually
   freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)  # Must pass sr again

**The AudioSamples Solution**

Audio objects carry properties intrinsically:

.. code-block:: python

   # AudioSamples approach - automatic coordination
   audio = aus.io.read('audio.wav')
   stft_matrix = audio.stft(window_size=2048, hop_size=512)
   # Time/frequency coordinates computed automatically

Table of Contents
-----------------

.. toctree::
   :maxdepth: 2
   :caption: Documentation

   examples/index
   api/index

.. toctree::
   :maxdepth: 1
   :caption: Development

   GitHub Repository <https://github.com/jmg049/audio_samples_python>

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`