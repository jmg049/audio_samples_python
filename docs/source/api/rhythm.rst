Rhythm: Onsets, Tempo and Beats
===============================

librosa-compatible rhythm analysis (librosa 1.0), as ``AudioSamples`` methods
that start from audio and as free functions in ``audio_samples.rhythm`` that
take librosa's precomputed inputs (``S=``, ``onset_envelope=``, ``tg=``).
Keyword names and defaults are librosa's.

======================================  ============================================================
librosa                                 audio_samples
======================================  ============================================================
``onset.onset_strength``                ``AudioSamples.onset_strength`` / ``rhythm.onset_strength(S=...)``
``onset.onset_strength_multi``          ``AudioSamples.onset_strength_multi`` / ``rhythm.onset_strength_multi(S=...)``
``onset.onset_detect``                  ``AudioSamples.onset_detect`` / ``rhythm.onset_detect(onset_envelope=...)``
``onset.onset_backtrack``               ``rhythm.onset_backtrack``
``util.peak_pick``                      ``rhythm.peak_pick``
``tempo_frequencies``                   ``rhythm.tempo_frequencies``
``fourier_tempo_frequencies``           ``rhythm.fourier_tempo_frequencies``
``feature.tempogram``                   ``AudioSamples.tempogram`` / ``rhythm.tempogram``
``feature.fourier_tempogram``           ``AudioSamples.fourier_tempogram`` / ``rhythm.fourier_tempogram``
``feature.tempo``                       ``AudioSamples.tempo`` / ``rhythm.tempo``
``beat.beat_track``                     ``AudioSamples.beat_track`` / ``rhythm.beat_track``
(internal ``__beat_tracker``)           ``rhythm.beat_track_with_bpm``
``beat.plp``                            ``AudioSamples.plp`` / ``rhythm.plp``
``feature.tempogram_ratio``             ``AudioSamples.tempogram_ratio`` / ``rhythm.tempogram_ratio``
``feature.hybrid_tempogram``            ``AudioSamples.hybrid_tempogram`` / ``rhythm.hybrid_tempogram``
``feature.metrogram``                   ``rhythm.metrogram``
======================================  ============================================================

Shapes: the ``AudioSamples`` methods always return a leading channel axis
(``(channels, n_frames)`` envelopes, ``(channels, F, T)`` tempograms); index
``[0]`` for librosa's 1-D-input shape. Event detectors need mono audio and
return 1-D int64 frames/samples or float64 times. ``beat_track`` returns
``(tempo, beats)`` exactly like librosa.

Differences from librosa: the onset mel basis is float64 (librosa's default
float32 basis differs by ~6e-8 relative; event and tempo outputs agree on the
reference signals); only ``sparse=True`` event output; ``channels`` as slices,
``metrogram(aggregate=None)``, custom ``tempogram_ratio(freqs=...)`` and
non-linear ``kind`` are not supported; ``onset_backtrack`` with no events
returns an empty array instead of raising.

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
