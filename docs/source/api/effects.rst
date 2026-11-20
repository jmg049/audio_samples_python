Effects and Decomposition
=========================

Audio-to-audio effects are ``AudioSamples`` methods returning ``AudioSamples``
of the input's sample type, each with an ``_in_place`` variant; array-level
functions live in ``audio_samples.effects``.

- Time and pitch: ``AudioSamples.time_stretch``, ``pitch_shift``,
  ``resample``; ``effects.phase_vocoder`` on an STFT.
- Emphasis: ``AudioSamples.preemphasis`` / ``deemphasis``;
  ``effects.preemphasis`` / ``deemphasis`` with filter state (``zi``,
  ``return_zf``).
- Silence: ``AudioSamples.trim_db`` and ``split_db`` (frames within
  ``top_db`` of the reference), ``remix`` (concatenate intervals).
- Harmonic/percussive separation: ``AudioSamples.harmonic``, ``percussive``,
  ``harmonic_percussive``; ``effects.hpss`` and ``effects.softmask`` on
  spectrograms; ``effects.nn_filter``.
- Reconstruction and synthesis: ``effects.griffinlim`` /
  ``AudioSamples.griffinlim_audio``, ``effects.clicks``.
- Linear prediction and companding: ``AudioSamples.lpc`` / ``effects.lpc``,
  ``AudioSamples.mu_compress`` / ``mu_expand``, ``mu_law_encode`` /
  ``mu_law_decode``.

``trim_db``, ``split_db`` and ``harmonic_percussive`` are separate from
``AudioSamples.trim``, ``split`` and ``hpss``, which work differently.

``pitch_shift`` and ``resample`` use this library's delay-free windowed-sinc
resampler (``res_type="soxr_hq"`` maps to ``ResamplingQuality.high``);
``trim_db`` raises ``ValueError`` when everything is silent (an
``AudioSamples`` cannot be empty); ``harmonic``/``percussive`` use the
``window`` argument and do not accept ``mask=True``; Griffin-Lim's random
initial phases come from this library's RNG and ``init=None`` starts from
zero phase; ``clicks`` returns float64 unless asked for float32; ``nn_filter``
takes dense recurrence matrices and the mean, median or weighted-average
aggregates only.

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
