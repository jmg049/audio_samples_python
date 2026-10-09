librosa-compatible Spectral Analysis
====================================

These functions reproduce librosa 1.0's spectral pipeline (STFT/ISTFT, dB
conversions, mel filterbank, MFCC, delta features) with librosa's defaults and
keyword arguments, and agree with librosa to floating-point round-off (see the
``tests/test_librosa_spectrum.py`` tolerances).

Two entry points are provided:

* methods on :class:`audio_samples.AudioSamples`, which analyse every channel
  and always return arrays shaped ``(channels, n_features, n_frames)`` (mono
  audio gives a leading axis of length 1; librosa returns ``(n_features,
  n_frames)`` for 1-D input, so index ``[0]``);
* free functions in ``audio_samples.spectral`` over NumPy arrays, which follow
  librosa's shapes exactly.

Windows are given as SciPy-style specifications: ``"hann"`` (default),
``"hamming"``, ``"blackman"``, ``"blackmanharris"``, ``"boxcar"`` /
``"rectangular"``, ``"bartlett"``, ``"triang"``, ``("kaiser", beta)``,
``("gaussian", std)``, a number (Kaiser beta) or an array of coefficients.

AudioSamples methods
--------------------

.. currentmodule:: audio_samples

.. automethod:: AudioSamples.stft_matrix
   :no-index:

.. automethod:: AudioSamples.istft_matrix
   :no-index:

.. automethod:: AudioSamples.spectrogram_matrix
   :no-index:

.. automethod:: AudioSamples.melspectrogram
   :no-index:

.. automethod:: AudioSamples.mfcc_frames
   :no-index:

Array functions (``audio_samples.spectral``)
--------------------------------------------

.. automodule:: audio_samples.spectral
   :no-members:

.. currentmodule:: audio_samples.spectral

Transforms
~~~~~~~~~~

.. autofunction:: stft

.. autofunction:: istft

.. autofunction:: frame

Decibel conversions
~~~~~~~~~~~~~~~~~~~

.. autofunction:: power_to_db

.. autofunction:: amplitude_to_db

.. autofunction:: db_to_power

.. autofunction:: db_to_amplitude

Filterbanks and windows
~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: mel_filterbank

.. autofunction:: get_window

.. autofunction:: window_sumsquare

.. autofunction:: window_bandwidth

Features from spectrograms
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: melspectrogram_from_spectrogram

.. autofunction:: mfcc_from_mel_db

.. autofunction:: delta

.. autofunction:: stack_memory

Unit conversions (``audio_samples.utils``)
------------------------------------------

Vectorised like librosa: a scalar gives a Python number, an array-like gives a
NumPy array of the same shape. ``hz_to_mel``, ``mel_to_hz`` and
``mel_frequencies`` keep their historical **HTK** default; pass ``htk=False``
for librosa's Slaney default. ``time_to_frames`` keeps its round-to-nearest
semantics; ``time_to_frames_floor`` is librosa's ``time_to_frames``.

.. currentmodule:: audio_samples.utils

.. autofunction:: hz_to_mel

.. autofunction:: mel_to_hz

.. autofunction:: mel_frequencies

.. autofunction:: hz_to_midi

.. autofunction:: midi_to_hz

.. autofunction:: time_to_samples

.. autofunction:: samples_to_frames

.. autofunction:: frames_to_samples

.. autofunction:: time_to_frames_floor

.. autofunction:: frames_to_time

.. autofunction:: time_to_frames

.. autofunction:: times_like

.. autofunction:: samples_like

mu-law codes
~~~~~~~~~~~~

Also available as ``audio_samples.mu_law_encode`` / ``audio_samples.mu_law_decode``.

.. autofunction:: mu_law_encode

.. autofunction:: mu_law_decode

Example
-------

.. code-block:: python

   import numpy as np
   import audio_samples as aus
   from audio_samples import spectral

   audio = aus.io.read("speech.wav")
   S = audio.melspectrogram(n_mels=64)          # (channels, 64, n_frames)
   S_db = spectral.power_to_db(S, ref=np.max)   # per-channel reference
   mfcc = audio.mfcc_frames(n_mfcc=13)          # (channels, 13, n_frames)
   d_mfcc = spectral.delta(mfcc)                # along the frame axis
