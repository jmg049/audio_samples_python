Effects and Decomposition
=========================

librosa-compatible effects (librosa 1.0). Audio-to-audio effects are
``AudioSamples`` methods returning ``AudioSamples`` of the input's sample type,
each with an ``_in_place`` variant; array-level helpers live in
``audio_samples.effects``.

======================================  ============================================================
librosa                                 audio_samples
======================================  ============================================================
``effects.time_stretch``                ``AudioSamples.time_stretch`` (+ ``_in_place``)
``effects.pitch_shift``                 ``AudioSamples.pitch_shift`` (+ ``_in_place``)
``effects.preemphasis``                 ``AudioSamples.preemphasis`` (+ ``_in_place``) / ``effects.preemphasis`` (``zi``, ``return_zf``)
``effects.deemphasis``                  ``AudioSamples.deemphasis`` (+ ``_in_place``) / ``effects.deemphasis``
``effects.trim``                        ``AudioSamples.trim_db`` (+ ``_in_place``)
``effects.split``                       ``AudioSamples.split_db``
``effects.remix``                       ``AudioSamples.remix`` (+ ``_in_place``)
``effects.harmonic`` / ``percussive``   ``AudioSamples.harmonic`` / ``percussive`` (+ ``_in_place``)
``effects.hpss``                        ``AudioSamples.harmonic_percussive``
``decompose.hpss``                      ``effects.hpss``
``util.softmask``                       ``effects.softmask``
``decompose.nn_filter``                 ``effects.nn_filter``
``phase_vocoder``                       ``effects.phase_vocoder``
``griffinlim``                          ``effects.griffinlim`` / ``AudioSamples.griffinlim_audio``
``lpc``                                 ``AudioSamples.lpc`` / ``effects.lpc``
``clicks``                              ``effects.clicks``
``resample``                            ``AudioSamples.resample`` (``quality="high"`` is closest to ``soxr_hq``)
``mu_compress`` / ``mu_expand``         ``AudioSamples.mu_compress`` / ``mu_expand``, ``mu_law_encode`` / ``mu_law_decode``
======================================  ============================================================

``trim``/``split`` are named ``trim_db``/``split_db`` and ``effects.hpss`` is
``harmonic_percussive`` because ``trim``, ``split`` and ``hpss`` already exist on
``AudioSamples`` with different (non-librosa) semantics.

Differences from librosa: ``pitch_shift`` and ``resample`` use this library's
delay-free windowed-sinc resampler instead of soxr (``res_type="soxr_hq"`` maps
to ``ResamplingQuality.high``; agreement ~68 dB SNR on band-limited audio);
``trim_db`` raises ``ValueError`` when everything is silent (an
``AudioSamples`` cannot be empty); ``harmonic``/``percussive`` use the
``window`` argument (librosa ignores it) and do not accept ``mask=True``;
Griffin-Lim's random initial phases come from this library's RNG (integer
seeds only; ``init=None`` matches librosa to round-off); ``clicks`` returns
float64; ``nn_filter`` takes dense recurrence matrices and the mean, median or
weighted-average aggregates only.

.. automodule:: audio_samples.effects
   :no-members:

.. currentmodule:: audio_samples

AudioSamples methods
--------------------

.. automethod:: AudioSamples.time_stretch
   :no-index:

.. automethod:: AudioSamples.pitch_shift
   :no-index:

.. automethod:: AudioSamples.preemphasis
   :no-index:

.. automethod:: AudioSamples.deemphasis
   :no-index:

.. automethod:: AudioSamples.trim_db
   :no-index:

.. automethod:: AudioSamples.split_db
   :no-index:

.. automethod:: AudioSamples.remix
   :no-index:

.. automethod:: AudioSamples.harmonic
   :no-index:

.. automethod:: AudioSamples.percussive
   :no-index:

.. automethod:: AudioSamples.harmonic_percussive
   :no-index:

.. automethod:: AudioSamples.lpc
   :no-index:

.. automethod:: AudioSamples.griffinlim_audio
   :no-index:

.. automethod:: AudioSamples.resample
   :no-index:

.. automethod:: AudioSamples.resample_by_ratio
   :no-index:

.. currentmodule:: audio_samples.effects

Array functions
---------------

.. autofunction:: phase_vocoder

.. autofunction:: preemphasis

.. autofunction:: deemphasis

.. autofunction:: softmask

.. autofunction:: hpss

.. autofunction:: nn_filter

.. autofunction:: griffinlim

.. autofunction:: lpc

.. autofunction:: clicks
