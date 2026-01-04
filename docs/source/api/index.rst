API Reference
=============

This section contains the complete API reference for AudioSamples Python, automatically generated from the
comprehensive type stubs that provide full type safety and IDE support.

.. toctree::
   :maxdepth: 2

   audio_samples
   generation
   io

Core Classes
============

.. autosummary::
   :toctree: generated/
   :nosignatures:

   audio_samples.AudioSamples
   audio_samples.IirFilterDesign
   audio_samples.EqBand
   audio_samples.ParametricEq

Quick Reference
===============

Signal Generation
-----------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   audio_samples.generation.sine_wave
   audio_samples.generation.cosine_wave
   audio_samples.generation.sawtooth_wave
   audio_samples.generation.square_wave
   audio_samples.generation.triangle_wave
   audio_samples.generation.chirp
   audio_samples.generation.white_noise
   audio_samples.generation.pink_noise
   audio_samples.generation.brown_noise
   audio_samples.generation.impulse
   audio_samples.generation.silence

File I/O
--------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   audio_samples.io.read
   audio_samples.io.read_with_info
   audio_samples.io.save
   audio_samples.io.save_as_type
   audio_samples.io.AudioInfo