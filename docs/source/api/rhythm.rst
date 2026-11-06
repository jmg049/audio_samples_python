Rhythm: Onsets, Tempo and Beats
===============================

Rhythm analysis as ``AudioSamples`` methods that start from audio, and as free
functions in ``audio_samples.rhythm`` that take precomputed inputs (a
spectrogram ``S``, an ``onset_envelope`` or a tempogram ``tg``).

- Onsets: ``onset_strength``, ``onset_strength_multi``, ``onset_detect``
  (methods and functions); ``rhythm.onset_backtrack``, ``rhythm.peak_pick``.
- Tempograms and tempo: ``tempogram``, ``fourier_tempogram``,
  ``hybrid_tempogram``, ``tempogram_ratio``, ``tempo`` (methods and
  functions); ``rhythm.metrogram``, ``rhythm.tempo_frequencies``,
  ``rhythm.fourier_tempo_frequencies``.
- Beats: ``beat_track`` and ``plp`` (methods and functions);
  ``rhythm.beat_track_with_bpm`` tracks beats at a given tempo.

Shapes: the ``AudioSamples`` methods always return a leading channel axis
(``(channels, n_frames)`` envelopes, ``(channels, F, T)`` tempograms); index
``[0]`` for mono audio. Event detectors need mono audio and return 1-D int64
frames/samples or float64 times. ``beat_track`` returns ``(tempo, beats)``.

Event output is sparse only (``sparse=False`` raises); custom
``tempogram_ratio(freqs=...)`` and non-linear ``kind`` are not supported;
``onset_backtrack`` with no events returns an empty array.
``beat_track`` defaults to ``trim=False``.

.. automodule:: audio_samples.rhythm
   :no-members:

.. currentmodule:: audio_samples

AudioSamples methods
--------------------

.. automethod:: AudioSamples.onset_strength
   :no-index:

.. automethod:: AudioSamples.onset_strength_multi
   :no-index:

.. automethod:: AudioSamples.onset_detect
   :no-index:

.. automethod:: AudioSamples.tempogram
   :no-index:

.. automethod:: AudioSamples.fourier_tempogram
   :no-index:

.. automethod:: AudioSamples.tempo
   :no-index:

.. automethod:: AudioSamples.beat_track
   :no-index:

.. automethod:: AudioSamples.plp
   :no-index:

.. automethod:: AudioSamples.tempogram_ratio
   :no-index:

.. automethod:: AudioSamples.hybrid_tempogram
   :no-index:

.. currentmodule:: audio_samples.rhythm

Onsets
------

.. autofunction:: onset_strength

.. autofunction:: onset_strength_multi

.. autofunction:: onset_detect

.. autofunction:: onset_backtrack

.. autofunction:: peak_pick

Tempograms and tempo
--------------------

.. autofunction:: tempo_frequencies

.. autofunction:: fourier_tempo_frequencies

.. autofunction:: tempogram

.. autofunction:: fourier_tempogram

.. autofunction:: tempo

.. autofunction:: tempogram_ratio

.. autofunction:: hybrid_tempogram

.. autofunction:: metrogram

Beats
-----

.. autofunction:: beat_track

.. autofunction:: beat_track_with_bpm

.. autofunction:: plp
