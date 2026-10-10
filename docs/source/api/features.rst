librosa-compatible Features, Tonal, Constant-Q and Pitch Analysis
=================================================================

These bindings reproduce librosa 1.0's frame-wise features
(``librosa.feature.*``), chroma and tuning estimation, the constant-Q family
(``cqt``, ``vqt``, ``icqt``, chroma/tonnetz) and the YIN / pYIN pitch
trackers, with librosa's keyword names and defaults. Agreement with librosa is
measured by the golden-fixture tests ``tests/test_librosa_features.py``,
``test_librosa_tonal.py``, ``test_librosa_constant_q.py`` and
``test_librosa_pitch.py`` (round-off level, except the constant-Q octaves below
the top one, which use the crate's resampler instead of soxr: ~1e-4 of the peak).

Entry points:

* methods on :class:`audio_samples.AudioSamples` (librosa's ``y=`` forms; the
  sample rate comes from the audio), always returning
  ``(channels, n_features, n_frames)`` arrays (pitch trackers:
  ``(channels, n_frames)``); index ``[0]`` for librosa's 1-D-input shape;
* ``audio_samples.features``: array functions (librosa's ``S=`` / ``C=`` /
  ``chroma=`` forms, filters and helpers) following librosa's shapes;
* ``audio_samples.utils``: tuning, octave and key conversions.

Name mapping
------------

=========================================  ==============================================
librosa                                    audio_samples
=========================================  ==============================================
``feature.spectral_centroid(y=...)``       ``AudioSamples.spectral_centroid_frames``
``feature.spectral_bandwidth(y=...)``      ``AudioSamples.spectral_bandwidth_frames``
``feature.spectral_rolloff(y=...)``        ``AudioSamples.spectral_rolloff_frames``
``feature.spectral_flatness(y=...)``       ``AudioSamples.spectral_flatness_frames``
``feature.spectral_contrast(y=...)``       ``AudioSamples.spectral_contrast_frames``
``feature.poly_features(y=...)``           ``AudioSamples.poly_features``
``feature.rms(y=...)``                     ``AudioSamples.rms_frames``
``feature.zero_crossing_rate(y)``          ``AudioSamples.zero_crossing_rate_frames``
``reassigned_spectrogram``                 ``AudioSamples.reassigned_spectrogram``
``feature.chroma_stft(y=...)``             ``AudioSamples.chroma_stft``
``piptrack(y=...)`` / ``estimate_tuning``  ``AudioSamples.piptrack`` / ``estimate_tuning``
``cqt`` / ``vqt``                          ``AudioSamples.cqt`` / ``vqt``
``pseudo_cqt`` / ``hybrid_cqt``            ``AudioSamples.pseudo_cqt`` / ``hybrid_cqt``
``icqt`` / ``griffinlim_cqt``              ``AudioSamples.icqt`` / ``griffinlim_cqt`` (static)
``feature.chroma_cqt`` / ``chroma_cens``   ``AudioSamples.chroma_cqt`` / ``chroma_cens``
``feature.chroma_vqt`` / ``tonnetz``       ``AudioSamples.chroma_vqt`` / ``tonnetz``
``yin`` / ``pyin``                         ``AudioSamples.yin`` / ``pyin``; ``features.yin`` / ``pyin``
``feature.*(S=...)``                       ``features.*_from_spectrogram``
``pcen``, ``*_weighting``                  ``features.pcen``, ``features.a_weighting`` ...
``filters.chroma``                         ``features.chroma_filterbank``
``util.normalize``                         ``features.normalize``
``filters.cq_to_chroma`` / ``wavelet``     ``features.cq_to_chroma`` / ``wavelet``
``salience``, ``*_harmonics``              ``features.salience`` / ``interp_harmonics`` / ``f0_harmonics``
``hz_to_octs``, ``A4_to_tuning``, ...      ``utils.hz_to_octs``, ``utils.a4_to_tuning``, ...
=========================================  ==============================================

Differences: outputs and filterbanks are always float64 / complex128
(librosa defaults to float32 bases and ``rms(dtype=float32)``); ``res_type``
selects the crate's resampler quality (soxr is not used); only
``intervals="equal"`` (VQT), ``kind="linear"`` (harmonics), the last axis as
time and ``init=None`` (``griffinlim_cqt``) are supported; callables are
limited to ``numpy.max`` / ``numpy.mean`` / ``numpy.median`` etc. where librosa
accepts them.

AudioSamples methods
--------------------

.. currentmodule:: audio_samples

Frame-wise features
~~~~~~~~~~~~~~~~~~~

.. automethod:: AudioSamples.spectral_centroid_frames
   :no-index:

.. automethod:: AudioSamples.spectral_bandwidth_frames
   :no-index:

.. automethod:: AudioSamples.spectral_rolloff_frames
   :no-index:

.. automethod:: AudioSamples.spectral_flatness_frames
   :no-index:

.. automethod:: AudioSamples.spectral_contrast_frames
   :no-index:

.. automethod:: AudioSamples.poly_features
   :no-index:

.. automethod:: AudioSamples.rms_frames
   :no-index:

.. automethod:: AudioSamples.zero_crossing_rate_frames
   :no-index:

.. automethod:: AudioSamples.reassigned_spectrogram
   :no-index:

Tonal
~~~~~

.. automethod:: AudioSamples.chroma_stft
   :no-index:

.. automethod:: AudioSamples.piptrack
   :no-index:

.. automethod:: AudioSamples.estimate_tuning
   :no-index:

Constant-Q
~~~~~~~~~~

.. automethod:: AudioSamples.cqt
   :no-index:

.. automethod:: AudioSamples.vqt
   :no-index:

.. automethod:: AudioSamples.pseudo_cqt
   :no-index:

.. automethod:: AudioSamples.hybrid_cqt
   :no-index:

.. automethod:: AudioSamples.icqt
   :no-index:

.. automethod:: AudioSamples.griffinlim_cqt
   :no-index:

.. automethod:: AudioSamples.chroma_cqt
   :no-index:

.. automethod:: AudioSamples.chroma_cens
   :no-index:

.. automethod:: AudioSamples.chroma_vqt
   :no-index:

.. automethod:: AudioSamples.tonnetz
   :no-index:

Pitch tracking
~~~~~~~~~~~~~~

.. automethod:: AudioSamples.yin
   :no-index:

.. automethod:: AudioSamples.pyin
   :no-index:

Array functions (``audio_samples.features``)
--------------------------------------------

.. automodule:: audio_samples.features
   :no-members:

.. currentmodule:: audio_samples.features

Spectral descriptors
~~~~~~~~~~~~~~~~~~~~

.. autofunction:: spectral_centroid_from_spectrogram

.. autofunction:: spectral_bandwidth_from_spectrogram

.. autofunction:: spectral_rolloff_from_spectrogram

.. autofunction:: spectral_flatness_from_spectrogram

.. autofunction:: spectral_contrast_from_spectrogram

.. autofunction:: poly_features_from_spectrogram

.. autofunction:: rms_from_spectrogram

.. autofunction:: pcen

Frequency weighting
~~~~~~~~~~~~~~~~~~~

.. autofunction:: a_weighting

.. autofunction:: b_weighting

.. autofunction:: c_weighting

.. autofunction:: d_weighting

.. autofunction:: z_weighting

.. autofunction:: frequency_weighting

.. autofunction:: multi_frequency_weighting

.. autofunction:: perceptual_weighting

Tonal
~~~~~

.. autofunction:: chroma_filterbank

.. autofunction:: normalize

.. autofunction:: pitch_tuning

.. autofunction:: chroma_stft_from_spectrogram

.. autofunction:: piptrack_from_spectrogram

.. autofunction:: estimate_tuning_from_spectrogram

.. autofunction:: estimate_tuning_from_signal

Constant-Q
~~~~~~~~~~

.. autofunction:: cq_to_chroma

.. autofunction:: wavelet

.. autofunction:: wavelet_lengths

.. autofunction:: chroma_cens_from_chroma

.. autofunction:: tonnetz_from_chroma

Pitch tracking and harmonics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: yin

.. autofunction:: pyin

.. autofunction:: salience

.. autofunction:: interp_harmonics

.. autofunction:: f0_harmonics

Tuning, octave and key conversions (``audio_samples.utils``)
------------------------------------------------------------

Vectorised like librosa: a scalar gives a Python float, an array-like an array
of the same shape.

.. currentmodule:: audio_samples.utils

.. autofunction:: hz_to_octs

.. autofunction:: octs_to_hz

.. autofunction:: a4_to_tuning

.. autofunction:: tuning_to_a4

.. autofunction:: cqt_frequencies

.. autofunction:: key_to_notes

.. autofunction:: key_to_degrees
