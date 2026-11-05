API Reference
=============

This section is the API reference for AudioSamples Python, generated from the type stubs (which also
provide type checking and IDE completion).

.. toctree::
   :maxdepth: 2

   audio_samples
   spectral
   features
   sequence
   segment
   arrays
   forms
   linalg
   inverse
   factorization
   pitch_filterbank
   notation
   rhythm
   effects
   generation
   io
   types

Core Classes
============

- :doc:`audio_samples` - Main AudioSamples class and its processing methods
- :doc:`spectral` - librosa-compatible STFT, mel, MFCC, dB and unit conversions
- :doc:`features` - librosa-compatible features, chroma/tuning, constant-Q and pitch tracking
- :doc:`sequence` - DTW, Viterbi, recurrence matrices and sync (librosa-compatible)
- :doc:`segment` - path enhancement, time-lag filtering, clustering and subsegmentation
- :doc:`arrays` - array helpers, zero crossings, tones and sweeps
- :doc:`forms` - array forms of methods, with their extra options
- :doc:`linalg` - NNLS, pseudo-inverse, eigendecomposition and SVD
- :doc:`inverse` - mel and MFCC inversion
- :doc:`factorization` - non-negative matrix factorisation
- :doc:`pitch_filterbank` - semitone filterbank and IIR time-frequency representation
- :doc:`notation` - tuning systems, note, FJS, svara, mela and thaat names
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
   Audio file reading and writing with format detection.

**Types Module**
   Audio processing configurations and enumerations. Used throughout the library for specifying parameters and options.