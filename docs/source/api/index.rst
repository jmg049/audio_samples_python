API Reference
=============

This section contains the complete API reference for AudioSamples Python, automatically generated from the
comprehensive type stubs that provide full type safety and IDE support.

.. toctree::
   :maxdepth: 2

   audio_samples
   spectral
   features
   sequence
   rhythm
   effects
   generation
   io
   types

Core Classes
============

- :doc:`audio_samples` - Main AudioSamples class with comprehensive audio processing
- :doc:`spectral` - librosa-compatible STFT, mel, MFCC, dB and unit conversions
- :doc:`features` - librosa-compatible features, chroma/tuning, constant-Q and pitch tracking
- :doc:`sequence` - DTW, Viterbi, recurrence matrices and sync (librosa-compatible)
- :doc:`rhythm` - onset strength/detection, tempograms, tempo, beat tracking, PLP (librosa-compatible)
- :doc:`effects` - time stretch, pitch shift, emphasis, trim/split, HPSS, Griffin-Lim, LPC (librosa-compatible)
- :doc:`generation` - Signal generation functions
- :doc:`io` - File I/O operations
- :doc:`types` - Audio processing configurations and enumerations

Quick Reference
===============

The complete API documentation is organized into three main sections:

**AudioSamples Class**
   The core audio processing class with 100+ methods for signal processing, analysis, and manipulation.

**Signal Generation**
   Functions to generate waveforms, noise, and test signals.

**File I/O**
   High-performance audio file reading and writing with format detection.

**Types Module**
   Audio processing configurations and enumerations. Used throughout the library for specifying parameters and options.