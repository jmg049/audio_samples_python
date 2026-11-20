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

Band-limited Waveforms
----------------------

.. autofunction:: sawtooth_wave_bandlimited

.. autofunction:: square_wave_bandlimited

.. autofunction:: triangle_wave_bandlimited

Modulated and Compound Signals
------------------------------

.. autofunction:: am_signal

.. autofunction:: fm_signal

.. autofunction:: compound_tone

.. autofunction:: exponential_chirp

.. autofunction:: exponential_bursts

Stereo Signals
--------------

.. autofunction:: stereo_chirp

.. autofunction:: stereo_silence

.. autofunction:: stereo_sine_wave

Noise Generators
----------------

.. autofunction:: white_noise

.. autofunction:: pink_noise

.. autofunction:: brown_noise

Example Clips
-------------

.. autofunction:: example_clip

.. autofunction:: example_clip_names

.. autofunction:: example_speech_f0

``example_clip`` gives realistic audio for documentation and experiments without a file or a
download. The clip is computed when you ask for it, with the sample rate, duration and sample type
you choose; the defaults are 44100 Hz, 8 s and ``float32``, and the returned audio carries them.

.. code-block:: python

   import numpy as np
   import audio_samples as aus

   drums = aus.generation.example_clip("drum_loop", 8.0, 22050, dtype=np.float32)
   tempo, beats = drums.beat_track(units="time")
   assert abs(float(np.ravel(tempo)[0]) - aus.generation.EXAMPLE_TEMPO_BPM) < 3.0

Each clip has known contents, so analysis results can be checked against them:

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Name
     - Contents
   * - ``"chords"``
     - C, G, Am, F in equal temperament (A4 = 440 Hz), one chord per four beats at
       ``EXAMPLE_TEMPO_BPM`` (110), with soft attacks and harmonic partials only
   * - ``"drum_loop"``
     - Kick on beats 1 and 3, snare on beats 2 and 4, hi-hat on every eighth note at
       ``EXAMPLE_TEMPO_BPM``, with the first kick at time zero
   * - ``"chords_and_drums"``
     - Exactly ``0.5 * chords + 0.5 * drum_loop`` before conversion to the sample type
   * - ``"speech"``
     - A voiced source following ``example_speech_f0(times)`` through five formant resonators:
       a 2.4 s phrase starting every 3 s. The vowels last 0.6 s each, /a e i o/ in the first
       phrase and /u a i e/ in the second, with first three formants /a/ (730, 1090, 2440),
       /e/ (530, 1840, 2480), /i/ (270, 2290, 3010), /o/ (570, 840, 2410), /u/ (300, 870, 2240) Hz
   * - ``"noisy_recording"``
     - ``"speech"`` plus low-passed noise and 50 Hz mains hum with harmonics, at
       ``EXAMPLE_NOISY_SNR_DB`` (15 dB) over the voiced parts

A shorter clip is a prefix of a longer one. Components above 95 % of the Nyquist frequency are
omitted, and the peak stays below full scale, so integer sample types do not clip.

**Reproducibility.** Rendering uses only IEEE 754 addition, subtraction, multiplication, division,
square root and ``floor``, plus a hand-written random number generator, so it does not depend on a
platform's maths library. The same arguments give the same bytes on any platform with IEEE 754
``float64`` arithmetic, which excludes x87 builds such as ``i586``. The test suites check digests
of the 16-bit output, which match on x86-64 with glibc and with musl, in debug and release builds and
with ``-C target-cpu=native``, and on aarch64.

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

.. automodule:: audio_samples.utils.generation
   :members:

.. currentmodule:: audio_samples.generation

