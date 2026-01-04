Signal Generation
=================

.. currentmodule:: audio_samples.generation

The generation module provides functions to generate various audio waveforms and noise types with
precise control over frequency, amplitude, and duration. All functions return AudioSamples objects
with embedded metadata.

Basic Waveforms
---------------

.. autofunction:: sine_wave

.. autofunction:: cosine_wave

.. autofunction:: sawtooth_wave

.. autofunction:: square_wave

.. autofunction:: triangle_wave

Advanced Signals
----------------

.. autofunction:: chirp

.. autofunction:: impulse

.. autofunction:: silence

Noise Generators
----------------

.. autofunction:: white_noise

.. autofunction:: pink_noise

.. autofunction:: brown_noise

Examples
--------

Basic Waveform Generation
~~~~~~~~~~~~~~~~~~~~~~~~~

Generate a 440 Hz sine wave for 1 second:

.. code-block:: python

   import audio_samples as aus

   # Generate a 440 Hz sine wave
   sine = aus.generation.sine_wave(440.0, 1.0, 44100)

   # Properties are embedded automatically
   print(f"Sample rate: {sine.sample_rate()} Hz")
   print(f"Duration: {sine.duration_seconds():.2f} seconds")
   print(f"Samples: {sine.samples_per_channel()}")

Multi-tone Generation
~~~~~~~~~~~~~~~~~~~~~

Create complex signals by mixing waveforms:

.. code-block:: python

   # Generate harmonic series
   fundamental = aus.generation.sine_wave(220.0, 1.0, 44100, 0.8)
   second_harmonic = aus.generation.sine_wave(440.0, 1.0, 44100, 0.4)
   third_harmonic = aus.generation.sine_wave(660.0, 1.0, 44100, 0.2)

   # Mix harmonics
   complex_tone = fundamental + second_harmonic + third_harmonic

   # Normalize to prevent clipping
   complex_tone.normalize(-1.0, 1.0, 'peak')

Noise Generation
~~~~~~~~~~~~~~~~

Generate different types of noise for testing and synthesis:

.. code-block:: python

   # White noise (equal energy per frequency)
   white = aus.generation.white_noise(2.0, 44100, 0.3)

   # Pink noise (1/f power spectrum)
   pink = aus.generation.pink_noise(2.0, 44100, 0.3)

   # Brown noise (1/f² power spectrum)
   brown = aus.generation.brown_noise(2.0, 44100, 0.01, 0.3)

Frequency Sweeps
~~~~~~~~~~~~~~~~

Create frequency sweeps for testing and analysis:

.. code-block:: python

   # Linear chirp from 100 Hz to 2000 Hz
   sweep = aus.generation.chirp(100.0, 2000.0, 5.0, 44100, 0.5)

   # Can be used for impulse response measurement
   print(f"Sweep range: 100 Hz to 2000 Hz over {sweep.duration_seconds():.1f} seconds")