Features, Tonal, Constant-Q and Pitch Analysis
==============================================

Frame-wise spectral features, chroma and tuning estimation, the constant-Q
family (``cqt``, ``vqt``, ``icqt``, chroma and tonnetz) and the YIN / pYIN
pitch trackers.

Entry points:

* methods on :class:`audio_samples.AudioSamples`, which take the sample rate
  from the audio and always return ``(channels, n_features, n_frames)`` arrays
  (pitch trackers: ``(channels, n_frames)``); index ``[0]`` for mono audio;
* ``audio_samples.features``: array functions that take a spectrogram ``S``,
  a constant-Q transform ``C`` or a chromagram, plus the filterbanks and
  helpers behind them; a 2-D input gives 2-D output and a 3-D
  ``(channels, ...)`` input 3-D output;
* ``audio_samples.utils``: tuning, octave and key conversions.

Output follows the input's precision: float32 / complex64 for uint8, int16,
int24 and float32 audio (or float32 arrays), float64 / complex128 otherwise.
Filterbanks, wavelets and frequency weightings are float64 tables.
``res_type`` selects this library's resampler quality for the constant-Q
octave decimation. Only ``intervals="equal"`` (VQT) and ``kind="linear"``
(harmonics) are supported, and time is the last axis.

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

Vectorised: a scalar gives a Python float, an array-like an array of the same
shape.

.. currentmodule:: audio_samples.utils

.. autofunction:: hz_to_octs

.. autofunction:: octs_to_hz

.. autofunction:: a4_to_tuning

.. autofunction:: tuning_to_a4

.. autofunction:: cqt_frequencies

.. autofunction:: key_to_notes

.. autofunction:: key_to_degrees
