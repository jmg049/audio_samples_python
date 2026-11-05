AudioSamples Python Documentation
===================================

**Fast, simple, and expressive audio processing for Python**

AudioSamples Python is a set of Python bindings to the AudioSamples Rust crates for audio processing.
An ``AudioSamples`` object stores its sample rate, channel layout and sample type together with the samples,
so these properties do not have to be passed separately to each function, as they do in most existing audio
processing libraries.

.. note::
   AudioSamples file I/O is 2-4x faster than that of scipy, soundfile and torchaudio.

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

- **Rust implementation**: processing runs in Rust, with little Python overhead
- **Type Safety**: Audio semantics encoded in the data structure, not external metadata
- **Audio Operations**: signal generation, analysis and processing
- **NumPy Integration**: conversion to and from NumPy arrays; NumPy functions and ufuncs accept ``AudioSamples``
- **Multi-channel Support**: mono, stereo and multi-channel audio
- **Real-time Analysis**: Spectral analysis, statistics, and audio metrics
- **Audio Effects**: Fading, normalization, filtering, and more
- **Fast I/O**: WAV and FLAC reading and writing

Why AudioSamples?
-----------------

**The Problem with Existing Libraries**

With most Python audio libraries, the caller passes the sample rate to each function:

.. code-block:: python

   # Traditional approach - error-prone manual coordination
   data, sr = soundfile.read('audio.wav')
   stft = librosa.stft(data, sr=sr)  # Must pass sr manually
   freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)  # Must pass sr again

**The AudioSamples Solution**

Audio objects store their own properties:

.. code-block:: python

   # AudioSamples approach - automatic coordination
   audio = aus.io.read('audio.wav')
   stft = audio.stft(aus.StftParams(n_fft=2048, hop_size=512, window=aus.WindowType.hanning))
   freqs = stft.frequencies  # frequency axis from the audio's own sample rate

Learning guide
--------------

:doc:`tutorials/index` is a short course in speech and audio processing: fourteen
notebooks that check every number they state. The same course exists for the
Rust crate as an `illustrated book <https://jmg049.github.io/audio_samples/>`_ with figures computed by
the library.

Coming from librosa
--------------------

:doc:`moving_from_librosa` is for someone who knows librosa and is deciding whether
to move: measurements on both sides, a pipeline side by side, the differences that
trip people up, and what is still better in librosa. :doc:`librosa_migration` is the
lookup table: every librosa 1.0 function, its ``audio_samples`` equivalent and every
documented difference.

Table of Contents
-----------------

.. toctree::
   :maxdepth: 2
   :caption: Documentation

   moving_from_librosa
   librosa_migration
   tutorials/index
   for_instructors
   examples/index
   free_threading
   api/index

.. toctree::
   :maxdepth: 1
   :caption: Development

   changelog
   GitHub Repository <https://github.com/jmg049/audio_samples_python>

Citing
======

If you use AudioSamples in research, please cite
`Audio Made Simple: A Modern Framework for Audio Processing
<https://doi.org/10.1145/3793853.3799811>`_ (Geraghty, Golpayegani and Hines,
MMSys '26). The ``CITATION.cff`` file in the repository has the full reference.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`