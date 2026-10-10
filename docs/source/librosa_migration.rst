Coming from librosa
===================

This page maps `librosa <https://librosa.org>`_ 1.0 onto the ``audio_samples``
Python API. It covers every public function of ``librosa`` (the core,
including ``librosa.convert``), ``librosa.feature``, ``effects``, ``onset``,
``beat``, ``decompose``, ``filters``, ``util``, ``segment``, ``sequence`` and
``display``. For each one it gives the Python equivalent and how closely it
matches, then collects every documented difference.

The librosa-compatible API is tested against golden fixtures produced by
librosa 1.0.0 and, where librosa is installed, against librosa directly (the
``tests/test_librosa_*.py`` suites). The tolerances, and the reasons for them,
are in those tests.

.. contents:: On this page
   :local:
   :depth: 1

Conventions
-----------

**The audio carries its sample rate.** librosa passes ``y`` and ``sr``
separately. An :class:`~audio_samples.AudioSamples` holds the samples, the
sample rate and the channel layout, so its methods take no ``sr`` argument.
Free functions over NumPy arrays (``audio_samples.spectral``, ``features``,
``rhythm``, ``effects``, ``sequence``, ``utils``) keep librosa's ``sr``
keyword where librosa has one (the older ``utils`` conversions call it
``sample_rate``).

**Two entry points.** librosa's ``y=`` forms are ``AudioSamples`` methods.
librosa's ``S=``, ``onset_envelope=``, ``chroma=`` and ``tg=`` forms, and the
filterbanks and helpers, are free functions in the submodules:

=================================  =============================================================
Submodule                          Contents
=================================  =============================================================
``audio_samples.spectral``         STFT/ISTFT, dB conversions, mel filterbank, windows, MFCC from
                                   a mel spectrogram, ``delta``, ``stack_memory``, ``frame``
``audio_samples.features``         ``feature.*(S=...)`` forms, PCEN, weightings, chroma and
                                   constant-Q filters, ``normalize``, tuning, YIN/pYIN, harmonics
``audio_samples.rhythm``           onset strength and detection, peak picking, tempograms,
                                   tempo, beat tracking, PLP
``audio_samples.effects``          phase vocoder, emphasis filters, HPSS, ``softmask``,
                                   ``nn_filter``, Griffin-Lim, LPC, ``clicks``
``audio_samples.sequence``         DTW, Viterbi, transition matrices, recurrence, RQA, ``sync``
``audio_samples.utils``            unit, frame/time, tuning, octave and key conversions
``audio_samples.generation``       tones, chirps and other test signals
``audio_samples.io``               reading and writing audio files
=================================  =============================================================

**Methods return** ``(channels, F, T)``. Every frame-wise method returns a
NumPy array with a leading channel axis, also for mono audio:
``(channels, n_features, n_frames)``, and ``(channels, 1, n_frames)`` for
per-frame scalars such as the centroid or RMS. Onset envelopes and pitch tracks
are ``(channels, n_frames)``. librosa drops the leading axis for 1-D input, so
index ``[0]`` to get librosa's mono shape:

.. code-block:: python

   S = audio.melspectrogram()        # (1, 128, n_frames) for mono audio
   S_mono = S[0]                     # librosa.feature.melspectrogram(y=y, sr=sr).shape

Free functions follow librosa's shapes exactly (a 1-D signal gives a 2-D STFT,
a scalar gives a Python float).

**Keyword names follow librosa 1.0.** Parameter names and keyword-only markers
are librosa's, so ``audio.mfcc_frames(n_mfcc=13, lifter=22, top_db=80.0)``
means what ``librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, lifter=22)`` means.
Defaults equal librosa's except where librosa's default changes the data
without being asked (an 80 dB dynamic-range floor, tuning estimation, beat
trimming, unseeded random phases, snapping to zero crossings) and where the
value has to come from the caller (the ``sr`` of the free functions, the hop
length of frames computed elsewhere, the ``trim`` threshold); see
`Defaults that differ from librosa`_. ``inspect.signature`` and ``help()``
work on every function. Unsupported librosa options raise ``ValueError`` or
``TypeError``; they are never silently ignored.

**Output precision follows the input.** ``AudioSamples`` analysis methods
return ``float32`` / ``complex64`` arrays for ``uint8``, ``int16``, ``int24``
and ``float32`` audio, and ``float64`` / ``complex128`` arrays for ``int32``
and ``float64`` audio (the smallest float type that holds every sample value
exactly). Free functions over NumPy arrays compute in ``float32`` for a
``float32`` or ``complex64`` array and in ``float64`` for any other input
(lists, integer arrays, ``float16`` and ``float64`` arrays are converted to
``float64``). Audio-returning methods (``time_stretch``, ``preemphasis``,
``trim_db``, ...) return the receiver's sample type; the static
reconstructions (``istft_matrix``, ``icqt``, ``griffinlim_cqt``,
``griffinlim_audio``) take ``dtype=`` and otherwise return ``float32`` audio
for a single-precision matrix and ``float64`` audio for anything else.
librosa also keeps ``float32`` for ``float32`` input, but several of its bases
(mel, chroma, CQT wavelets) default to single precision even for ``float64``
input; pass ``dtype=np.float64`` to librosa to compare ``float64`` output to
round-off. Integer audio is converted with this library's audio-aware scaling:
negative values are divided by ``2**(bits-1)`` and positive ones by
``2**(bits-1) - 1`` (librosa's ``buf_to_float`` divides both by
``2**(bits-1)``), so compare with float audio.

**Non-finite audio raises.** As librosa's ``valid_audio`` does ("Audio buffer
is not finite everywhere"), the librosa-compatible methods raise
``ValueError`` when the audio contains a NaN or an infinity. This is slightly
stricter than librosa, which skips the check in a few functions (e.g.
``preemphasis``). Free functions over arrays do not check.

**Errors.** Invalid parameters raise ``ValueError`` naming the parameter
(``"n_fft: must be a positive integer, got -1"``), wrong types ``TypeError``,
and failures inside a computation ``RuntimeError``.

**The GIL is released.** The heavy computations (STFT, CQT, features, pitch
tracking, rhythm, effects, sequence alignment) run without the GIL, so several
Python threads can analyse different signals in parallel.

**Older methods keep their behaviour.** Where a librosa name clashed with an
existing method, the librosa-compatible version takes a new name and the old
method is unchanged (see :ref:`librosa-existing-apis`): ``stft_matrix`` (not
``stft``), ``mfcc_frames`` (not ``mfcc``), ``spectral_centroid_frames`` (not
``spectral_centroid``), ``trim_db`` (not ``trim``), ``split_db`` (not
``split``), ``harmonic_percussive`` (not ``hpss``).

Examples
--------

Each example runs as written (librosa's call is in the comment). A generated
tone takes the place of ``librosa.load``; read a file with
``audio_samples.io.read("file.wav")``.

STFT and inverse STFT
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   import audio_samples as aus

   # librosa: y, sr = librosa.load(...); here the audio carries its rate.
   y = aus.generation.sine_wave(440.0, 1.0, 22050, 0.5, dtype=np.float64)
   assert y.sample_rate == 22050

   # D = librosa.stft(y)          -> librosa (1025, 44); here (channels, 1025, 44)
   D = y.stft_matrix()
   assert D.shape == (1, 1025, 44) and D.dtype == np.complex128

   # y_hat = librosa.istft(D, length=len(y))
   y_hat = aus.AudioSamples.istft_matrix(D, sample_rate=22050, length=y.samples_per_channel())
   assert np.allclose(y_hat.to_numpy(), y.to_numpy(), atol=1e-9)

Mel spectrogram in dB, MFCC and deltas
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   import audio_samples as aus
   from audio_samples import spectral

   y = aus.generation.sine_wave(440.0, 1.0, 22050, 0.5, dtype=np.float64)

   # S = librosa.feature.melspectrogram(y=y, sr=sr)
   S = y.melspectrogram()
   assert S.shape == (1, 128, 44)

   # S_db = librosa.power_to_db(S, ref=np.max)  (librosa's default top_db is 80)
   S_db = spectral.power_to_db(S, ref=np.max, top_db=80.0)
   assert abs(S_db.max()) < 1e-9 and S_db.min() >= -80.0

   # mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13); librosa.feature.delta(mfcc, order=2)
   mfcc = y.mfcc_frames(n_mfcc=13, top_db=80.0)
   d2 = spectral.delta(mfcc, order=2)
   assert mfcc.shape == d2.shape == (1, 13, 44)

Spectral features and chroma
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   import audio_samples as aus

   y = aus.generation.sine_wave(440.0, 1.0, 22050, 0.5, dtype=np.float64)

   centroid = y.spectral_centroid_frames()   # librosa.feature.spectral_centroid(y=y, sr=sr)
   rms = y.rms_frames()                      # librosa.feature.rms(y=y)
   zcr = y.zero_crossing_rate_frames()       # librosa.feature.zero_crossing_rate(y)
   assert centroid.shape == rms.shape == zcr.shape == (1, 1, 44)
   assert abs(centroid[0, 0, 20] - 440.0) < 20.0

   chroma = y.chroma_stft(tuning=None)       # librosa.feature.chroma_stft(y=y, sr=sr)
   assert chroma.shape == (1, 12, 44)
   assert chroma[0, 9, 20] == 1.0            # pitch class A

Onsets, tempo and beats
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   import audio_samples as aus

   # A decaying click every 0.5 s (120 BPM) for 4 s.
   sr = 22050
   x = np.zeros(4 * sr)
   for k in range(8):
       x[k * sr // 2 : k * sr // 2 + 200] = np.exp(-np.arange(200) / 30.0)
   y = aus.AudioSamples.new_mono(x, sr)

   env = y.onset_strength()                  # librosa.onset.onset_strength(y=y, sr=sr)
   assert env.shape[0] == 1                  # (channels, n_frames); env[0] is librosa's

   onset_times = y.onset_detect(units="time")  # librosa.onset.onset_detect(..., units="time")
   assert len(onset_times) >= 7

   # librosa.beat.beat_track(y=y, sr=sr, units="time")  (librosa trims by default)
   tempo, beats = y.beat_track(units="time", trim=True)
   assert abs(float(tempo[0]) - 120.0) < 5.0 and len(beats) > 0

Effects and resampling
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   import audio_samples as aus

   y = aus.generation.sine_wave(440.0, 1.0, 22050, 0.5, dtype=np.float64)
   n = y.samples_per_channel()

   fast = y.time_stretch(2.0)                # librosa.effects.time_stretch(y, rate=2.0)
   assert fast.samples_per_channel() == n // 2

   up = y.pitch_shift(2.0)                   # librosa.effects.pitch_shift(y, sr=sr, n_steps=2)
   assert up.samples_per_channel() == n

   trimmed, (start, end) = y.trim_db(top_db=60.0)  # librosa.effects.trim(y)
   assert trimmed.samples_per_channel() == end - start

   y16 = y.resample(16000, "high")           # librosa.resample(y, orig_sr=22050, target_sr=16000)
   assert y16.samples_per_channel() == 16000

Pitch tracking
~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   import audio_samples as aus

   y = aus.generation.sine_wave(220.0, 0.5, 22050, 0.5, dtype=np.float64)

   f0 = y.yin(fmin=65.4, fmax=2093.0)        # librosa.yin(y, fmin=65.4, fmax=2093.0, sr=sr)
   assert abs(f0[0, 11] - 220.0) < 1.0

   f0, voiced_flag, voiced_prob = y.pyin(fmin=65.4, fmax=2093.0)  # librosa.pyin(...)
   assert voiced_flag[0, 11] and abs(f0[0, 11] - 220.0) < 2.0

Equivalence tables
------------------

Status values:

- **equivalent**: same results as librosa with the same arguments, to
  floating-point round-off; event outputs (frames, beats, paths) identical.
- **with differences**: the same algorithm with a documented difference in
  dtype, options or edge cases (see the notes and `Differences from librosa`_).
- **not provided**: no counterpart, with the reason.

``AudioSamples.x`` is a method (``audio.x(...)``); ``spectral.x``,
``features.x``, ``rhythm.x``, ``effects.x``, ``sequence.x``, ``utils.x``,
``generation.x`` and ``io.x`` are functions in the ``audio_samples``
submodule of that name.

Core: audio
~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 18 32 14 36

   * - librosa
     - audio_samples
     - status
     - notes
   * - ``load``
     - ``io.read``, ``io.read_and_resample``
     - with differences
     - No automatic resampling to 22050 Hz or mixing to mono: the file's own
       rate, channels and sample type are kept (``to_mono``, ``resample``)
   * - ``loadx``
     - --
     - not provided
     - librosa's experimental extended loader
   * - ``stream``
     - ``io.StreamedAudioReader``
     - with differences
     - Block reader over WAV/FLAC with its own interface (no ``block_length`` /
       ``frame_length`` framing)
   * - ``get_samplerate``
     - ``io.info``
     - with differences
     - ``io.info(path).sample_rate``; for loaded audio ``AudioSamples.sample_rate``
   * - ``get_duration``
     - ``AudioSamples.duration_seconds``
     - with differences
     - Audio only; no ``S=`` or ``path=`` forms
   * - ``resample``
     - ``AudioSamples.resample``, ``AudioSamples.resample_by_ratio``
     - with differences
     - Own windowed-sinc resampler, not soxr; ``"high"`` is closest to
       ``soxr_hq`` (the method's default is ``"fast"``); exact integer length
   * - ``to_mono``
     - ``AudioSamples.to_mono``
     - equivalent
     - Default method is the channel average
   * - ``to_stereo``
     - ``AudioSamples.to_stereo``
     - with differences
     - ``StereoConversionMethod``; no two-signal form (use ``AudioSamples.stack``)
   * - ``to_multi``
     - ``AudioSamples.stack``
     - with differences
     - No pad or trim to a common length
   * - ``autocorrelate``
     - ``AudioSamples.autocorrelation``
     - with differences
     - First channel only; unbiased estimate: ``r[k] * (n - k)`` equals
       librosa's raw sum
   * - ``lpc``
     - ``AudioSamples.lpc``, ``effects.lpc``
     - equivalent
     - Burg's method; one row per channel
   * - ``zero_crossings``
     - ``AudioSamples.zero_crossings``
     - with differences
     - Returns a count, not a boolean mask; use ``zero_crossing_rate_frames``
       for librosa's frame-wise rate
   * - ``clicks``
     - ``effects.clicks``
     - with differences
     - float64 output
   * - ``tone``
     - ``generation.sine_wave``, ``generation.cosine_wave``
     - with differences
     - No ``phi``; returns ``AudioSamples``
   * - ``chirp``
     - ``generation.exponential_chirp`` (librosa's default ``linear=False``),
       ``generation.chirp`` (linear)
     - with differences
     - No ``phi``; returns ``AudioSamples``
   * - ``mu_compress``
     - ``audio_samples.mu_law_encode`` (``quantize=True``),
       ``AudioSamples.mu_compress`` (``quantize=False``)
     - equivalent
     - ``mu_law_encode`` returns librosa's default integer codes
   * - ``mu_expand``
     - ``audio_samples.mu_law_decode``, ``AudioSamples.mu_expand``
     - equivalent
     - As ``mu_compress``

Core: spectral representations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 18 32 14 36

   * - librosa
     - audio_samples
     - status
     - notes
   * - ``stft``
     - ``AudioSamples.stft_matrix``, ``spectral.stft``
     - equivalent
     - All ``pad_mode`` values, ``win_length``, any window.
       ``AudioSamples.spectrogram_matrix`` gives ``abs(stft) ** power``
   * - ``istft``
     - ``AudioSamples.istft_matrix`` (static), ``spectral.istft``
     - with differences
     - A bin count that does not match ``n_fft`` and ``length=0`` raise
   * - ``magphase``
     - ``AudioSamples.magphase`` (static)
     - with differences
     - 2-D input; ``power`` is an integer
   * - ``reassigned_spectrogram``
     - ``AudioSamples.reassigned_spectrogram``
     - equivalent
     -
   * - ``phase_vocoder``
     - ``effects.phase_vocoder``
     - equivalent
     - The deprecated ``hop_length`` / ``n_fft`` are accepted with a
       ``FutureWarning`` and unused, as in librosa 1.0
   * - ``griffinlim``
     - ``effects.griffinlim``, ``AudioSamples.griffinlim_audio`` (static)
     - with differences
     - ``init=None`` matches; random initial phases come from this library's
       RNG (integer seeds only); ``rng`` defaults to the seed 0 (librosa:
       unseeded)
   * - ``iirt``
     - --
     - not provided
     - Deferred: needs elliptic order selection (``ellipord``)
   * - ``fmt``
     - --
     - not provided
     - Deferred: rarely used; needs a cubic-spline exponential resampler
   * - ``pcen``
     - ``features.pcen``
     - equivalent
     - ``zi`` / ``return_zf`` streaming state; ``sr`` and ``hop_length`` are
       required (librosa: 22050 and 512)
   * - ``amplitude_to_db``
     - ``spectral.amplitude_to_db``
     - equivalent
     - ``ref`` as a number, ``"max"`` or ``numpy.max``; per-channel reference;
       ``top_db`` defaults to ``None`` (librosa: 80)
   * - ``power_to_db``
     - ``spectral.power_to_db``
     - equivalent
     - As above
   * - ``db_to_amplitude``
     - ``spectral.db_to_amplitude``
     - equivalent
     -
   * - ``db_to_power``
     - ``spectral.db_to_power``
     - equivalent
     -
   * - ``perceptual_weighting``
     - ``features.perceptual_weighting``
     - equivalent
     - ``top_db`` defaults to ``None`` (librosa: 80)
   * - ``frequency_weighting``
     - ``features.frequency_weighting``
     - equivalent
     - ``min_db`` defaults to ``None`` (librosa: -80)
   * - ``multi_frequency_weighting``
     - ``features.multi_frequency_weighting``
     - equivalent
     - ``min_db`` defaults to ``None`` (librosa: -80)
   * - ``A_weighting`` ... ``Z_weighting``
     - ``features.a_weighting``, ``b_weighting``, ``c_weighting``,
       ``d_weighting``, ``z_weighting``
     - equivalent
     - Lower-case names

Core: constant-Q
~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 18 32 14 36

   * - librosa
     - audio_samples
     - status
     - notes
   * - ``cqt``
     - ``AudioSamples.cqt``
     - with differences
     - Octave decimation uses this library's resampler, not soxr: the top
       octave matches to round-off, lower octaves to ~1e-4 of the peak
   * - ``vqt``
     - ``AudioSamples.vqt``
     - with differences
     - As ``cqt``; only ``intervals="equal"``
   * - ``pseudo_cqt``
     - ``AudioSamples.pseudo_cqt``
     - equivalent
     - No resampling involved (librosa's ``complex64`` filters: ~3e-8)
   * - ``hybrid_cqt``
     - ``AudioSamples.hybrid_cqt``
     - with differences
     - Low octaves as ``cqt``
   * - ``icqt``
     - ``AudioSamples.icqt`` (static)
     - with differences
     - Upsampling uses this library's resampler
   * - ``griffinlim_cqt``
     - ``AudioSamples.griffinlim_cqt`` (static)
     - with differences
     - Only ``init=None``

Core: pitch, tuning and harmonics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 18 32 14 36

   * - librosa
     - audio_samples
     - status
     - notes
   * - ``yin``
     - ``AudioSamples.yin``, ``features.yin``
     - equivalent
     - ``fmin`` / ``fmax`` required, as in librosa
   * - ``pyin``
     - ``AudioSamples.pyin``, ``features.pyin``
     - equivalent
     - Returns ``(f0, voiced_flag, voiced_prob)``
   * - ``piptrack``
     - ``AudioSamples.piptrack``, ``features.piptrack_from_spectrogram``
     - equivalent
     -
   * - ``estimate_tuning``
     - ``AudioSamples.estimate_tuning``, ``features.estimate_tuning_from_signal``,
       ``features.estimate_tuning_from_spectrogram``
     - equivalent
     - Pools all channels, as librosa does
   * - ``pitch_tuning``
     - ``features.pitch_tuning``
     - equivalent
     -
   * - ``salience``
     - ``features.salience``
     - equivalent
     -
   * - ``interp_harmonics``
     - ``features.interp_harmonics``
     - with differences
     - Only ``kind="linear"``
   * - ``f0_harmonics``
     - ``features.f0_harmonics``
     - with differences
     - Only ``kind="linear"``

Core: unit conversions (``librosa.convert``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The librosa-compatible conversions in ``audio_samples.utils`` are vectorised
like librosa's: a scalar gives a Python number, an array-like an array of the
same shape. They take librosa's ``sr`` keyword or the older ``sample_rate``;
one of them is required (librosa defaults to 22050), and so is ``hop_length``
where librosa defaults it to 512. The note-name functions are scalar.

.. list-table::
   :header-rows: 1
   :widths: 18 32 14 36

   * - librosa
     - audio_samples
     - status
     - notes
   * - ``frames_to_samples``
     - ``utils.frames_to_samples``
     - equivalent
     -
   * - ``frames_to_time``
     - ``utils.frames_to_time``
     - with differences
     - Positional ``(frames, sample_rate, hop_size)`` (librosa: ``sr=``,
       ``hop_length=`` keywords); ``n_fft`` offset and negative frames supported
   * - ``samples_to_frames``
     - ``utils.samples_to_frames``
     - equivalent
     -
   * - ``samples_to_time``
     - ``utils.samples_to_time``
     - with differences
     - Scalar only; ``(samples, sample_rate)``
   * - ``time_to_frames``
     - ``utils.time_to_frames_floor``
     - equivalent
     - ``utils.time_to_frames`` keeps its older round-to-nearest behaviour
   * - ``time_to_samples``
     - ``utils.time_to_samples``
     - equivalent
     - ``utils.seconds_to_samples`` rounds instead of truncating
   * - ``blocks_to_frames``, ``blocks_to_samples``, ``blocks_to_time``
     - --
     - not provided
     - Block streaming belongs to file I/O; one multiplication each
   * - ``times_like``
     - ``utils.times_like``
     - equivalent
     -
   * - ``samples_like``
     - ``utils.samples_like``
     - equivalent
     -
   * - ``fft_frequencies``
     - ``utils.fft_frequencies``
     - with differences
     - Positional ``(n_fft, sample_rate)`` with no defaults (librosa:
       ``sr=22050``, ``n_fft=2048`` keywords); same values
   * - ``mel_frequencies``
     - ``utils.mel_frequencies`` with ``htk=False``
     - with differences
     - Default is HTK (kept for compatibility); librosa's default is Slaney
   * - ``hz_to_mel``
     - ``utils.hz_to_mel`` with ``htk=False``
     - with differences
     - As above
   * - ``mel_to_hz``
     - ``utils.mel_to_hz`` with ``htk=False``
     - with differences
     - As above
   * - ``cqt_frequencies``
     - ``utils.cqt_frequencies``
     - equivalent
     -
   * - ``tempo_frequencies``
     - ``rhythm.tempo_frequencies``
     - equivalent
     -
   * - ``fourier_tempo_frequencies``
     - ``rhythm.fourier_tempo_frequencies``
     - equivalent
     -
   * - ``hz_to_midi``
     - ``utils.hz_to_midi``
     - equivalent
     -
   * - ``midi_to_hz``
     - ``utils.midi_to_hz``
     - equivalent
     -
   * - ``hz_to_note``
     - ``utils.frequency_to_note``
     - with differences
     - Scalar; returns ``(name, cents)``; ASCII ``#``; MIDI 0..127 only
   * - ``midi_to_note``
     - ``utils.midi_to_note``
     - with differences
     - Scalar integer MIDI 0..127; ASCII ``#``; no cents, no ``octave=False``
   * - ``note_to_hz``
     - ``utils.note_to_frequency``
     - with differences
     - Scalar; grammar and limits as ``note_to_midi``
   * - ``note_to_midi``
     - ``utils.note_to_midi``
     - with differences
     - Scalar; librosa's grammar (accidentals, cents suffix); the octave is
       required and the result is in 0..127
   * - ``hz_to_octs``
     - ``utils.hz_to_octs``
     - equivalent
     -
   * - ``octs_to_hz``
     - ``utils.octs_to_hz``
     - equivalent
     -
   * - ``A4_to_tuning``
     - ``utils.a4_to_tuning``
     - equivalent
     - Lower-case name
   * - ``tuning_to_A4``
     - ``utils.tuning_to_a4``
     - equivalent
     - Lower-case name
   * - ``key_to_notes``
     - ``utils.key_to_notes``
     - equivalent
     -
   * - ``key_to_degrees``
     - ``utils.key_to_degrees``
     - equivalent
     -
   * - ``interval_frequencies``, ``pythagorean_intervals``, ``plimit_intervals``
     - --
     - not provided
     - Deferred: tuning-system generators, low use
   * - ``interval_to_fjs``, ``hz_to_fjs``, ``fifths_to_note``
     - --
     - not provided
     - Deferred: Functional Just System notation, low use
   * - ``hz_to_svara_c``, ``hz_to_svara_h``, ``midi_to_svara_c``,
       ``midi_to_svara_h``, ``note_to_svara_c``, ``note_to_svara_h``,
       ``list_mela``, ``list_thaat``, ``mela_to_degrees``, ``mela_to_svara``,
       ``thaat_to_degrees``
     - --
     - not provided
     - Deferred: Carnatic / Hindustani notation tables, low use

``librosa.feature``
~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 18 32 14 36

   * - librosa
     - audio_samples
     - status
     - notes
   * - ``melspectrogram``
     - ``AudioSamples.melspectrogram``, ``spectral.melspectrogram_from_spectrogram``
     - with differences
     - float64 mel basis (librosa's default float32 basis: ~1e-7 relative)
   * - ``mfcc``
     - ``AudioSamples.mfcc_frames``, ``spectral.mfcc_from_mel_db``
     - with differences
     - float64 mel basis; ``n_mfcc > n_mels`` raises; ``top_db`` defaults to
       ``None`` (librosa: 80)
   * - ``delta``
     - ``spectral.delta``
     - equivalent
     -
   * - ``stack_memory``
     - ``spectral.stack_memory``
     - equivalent
     -
   * - ``spectral_centroid``
     - ``AudioSamples.spectral_centroid_frames``,
       ``features.spectral_centroid_from_spectrogram``
     - equivalent
     - A time-varying 2-D ``freq`` is not supported
   * - ``spectral_bandwidth``
     - ``AudioSamples.spectral_bandwidth_frames``,
       ``features.spectral_bandwidth_from_spectrogram``
     - equivalent
     -
   * - ``spectral_contrast``
     - ``AudioSamples.spectral_contrast_frames``,
       ``features.spectral_contrast_from_spectrogram``
     - with differences
     - A band with no FFT bin raises
   * - ``spectral_flatness``
     - ``AudioSamples.spectral_flatness_frames``,
       ``features.spectral_flatness_from_spectrogram``
     - equivalent
     -
   * - ``spectral_rolloff``
     - ``AudioSamples.spectral_rolloff_frames``,
       ``features.spectral_rolloff_from_spectrogram``
     - equivalent
     -
   * - ``poly_features``
     - ``AudioSamples.poly_features``, ``features.poly_features_from_spectrogram``
     - with differences
     - A rank-deficient fit raises (NumPy warns and returns a minimum-norm
       solution)
   * - ``rms``
     - ``AudioSamples.rms_frames``, ``features.rms_from_spectrogram``
     - with differences
     - Output precision follows the sample type (librosa's default
       ``dtype=float32``)
   * - ``zero_crossing_rate``
     - ``AudioSamples.zero_crossing_rate_frames``
     - equivalent
     -
   * - ``chroma_stft``
     - ``AudioSamples.chroma_stft``, ``features.chroma_stft_from_spectrogram``
     - with differences
     - float64 chroma basis; ``tuning`` defaults to ``0.0`` (librosa: ``None``,
       estimated); with odd ``n_fft`` the given ``n_fft`` is used for tuning
   * - ``chroma_cqt``
     - ``AudioSamples.chroma_cqt``
     - with differences
     - Inherits the CQT resampler difference; ``window`` must be ``None``;
       ``tuning`` defaults to ``0.0`` (librosa: ``None``)
   * - ``chroma_cens``
     - ``AudioSamples.chroma_cens``, ``features.chroma_cens_from_chroma``
     - with differences
     - As ``chroma_cqt`` (from audio); equivalent from a chromagram
   * - ``chroma_vqt``
     - ``AudioSamples.chroma_vqt``
     - with differences
     - As ``vqt``
   * - ``tonnetz``
     - ``AudioSamples.tonnetz``, ``features.tonnetz_from_chroma``
     - with differences
     - As ``chroma_cqt`` (from audio); equivalent from a chromagram
   * - ``tempogram``
     - ``AudioSamples.tempogram``, ``rhythm.tempogram``
     - equivalent
     -
   * - ``fourier_tempogram``
     - ``AudioSamples.fourier_tempogram``, ``rhythm.fourier_tempogram``
     - equivalent
     -
   * - ``tempo``
     - ``AudioSamples.tempo``, ``rhythm.tempo``
     - equivalent
     - The method returns ``(channels, 1)``, or ``(channels, n_frames)`` when
       time-varying
   * - ``tempogram_ratio``
     - ``AudioSamples.tempogram_ratio``, ``rhythm.tempogram_ratio``
     - with differences
     - Custom ``freqs`` not supported
   * - ``hybrid_tempogram``
     - ``AudioSamples.hybrid_tempogram``, ``rhythm.hybrid_tempogram``
     - equivalent
     -
   * - ``metrogram``
     - ``rhythm.metrogram``
     - with differences
     - ``aggregate=None`` (the 4-D per-bin output) not supported
   * - ``inverse.mel_to_stft``, ``inverse.mel_to_audio``,
       ``inverse.mfcc_to_mel``, ``inverse.mfcc_to_audio``
     - --
     - not provided
     - Deferred: needs a non-negative least-squares solver

``librosa.effects``
~~~~~~~~~~~~~~~~~~~

Audio-to-audio effects return a new ``AudioSamples`` of the input's sample
type; each has an ``_in_place`` twin (``time_stretch_in_place``, ...).

.. list-table::
   :header-rows: 1
   :widths: 18 32 14 36

   * - librosa
     - audio_samples
     - status
     - notes
   * - ``time_stretch``
     - ``AudioSamples.time_stretch``
     - equivalent
     - Exactly ``round(n / rate)`` samples
   * - ``pitch_shift``
     - ``AudioSamples.pitch_shift``
     - with differences
     - Final resampling by this library's resampler, not soxr (~68 dB SNR);
       ``scale=True`` not supported
   * - ``preemphasis``
     - ``AudioSamples.preemphasis``, ``effects.preemphasis``
     - equivalent
     - The function takes ``zi`` / ``return_zf``
   * - ``deemphasis``
     - ``AudioSamples.deemphasis``, ``effects.deemphasis``
     - equivalent
     -
   * - ``trim``
     - ``AudioSamples.trim_db``
     - with differences
     - Returns ``(trimmed, array([start, end]))``; raises when everything is
       silent (librosa returns an empty array); ``top_db`` is required
       (librosa: 60); frame RMS in the sample type's float (librosa: float32)
   * - ``split``
     - ``AudioSamples.split_db``
     - with differences
     - int64 ``(n, 2)`` intervals; ``top_db`` is required (librosa: 60); frame
       RMS in the sample type's float (librosa: float32)
   * - ``remix``
     - ``AudioSamples.remix``
     - equivalent
     - ``align_zeros`` defaults to ``False`` (librosa: ``True``)
   * - ``hpss``
     - ``AudioSamples.harmonic_percussive``
     - with differences
     - ``window`` is used (librosa ignores it)
   * - ``harmonic``
     - ``AudioSamples.harmonic``
     - with differences
     - As ``hpss``; ``mask=True`` not supported
   * - ``percussive``
     - ``AudioSamples.percussive``
     - with differences
     - As ``harmonic``

``librosa.onset`` and ``librosa.beat``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 18 32 14 36

   * - librosa
     - audio_samples
     - status
     - notes
   * - ``onset.onset_strength``
     - ``AudioSamples.onset_strength``, ``rhythm.onset_strength``
     - equivalent
     - float64 mel basis
   * - ``onset.onset_strength_multi``
     - ``AudioSamples.onset_strength_multi``, ``rhythm.onset_strength_multi``
     - with differences
     - ``channels`` as boundaries only (no slices)
   * - ``onset.onset_detect``
     - ``AudioSamples.onset_detect``, ``rhythm.onset_detect``
     - with differences
     - Mono audio only; ``sparse=True`` only
   * - ``onset.onset_backtrack``
     - ``rhythm.onset_backtrack``
     - with differences
     - No events give an empty array (librosa raises)
   * - ``beat.beat_track``
     - ``AudioSamples.beat_track``, ``rhythm.beat_track``,
       ``rhythm.beat_track_with_bpm``
     - with differences
     - Mono audio only; ``sparse=True`` only; ``trim`` defaults to ``False``
       (librosa: ``True``)
   * - ``beat.plp``
     - ``AudioSamples.plp``, ``rhythm.plp``
     - equivalent
     -

``librosa.decompose``
~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 18 32 14 36

   * - librosa
     - audio_samples
     - status
     - notes
   * - ``decompose``
     - --
     - not provided
     - Deferred: NMF through scikit-learn
   * - ``hpss``
     - ``effects.hpss``
     - equivalent
     - Kernel sizes, power, margins, ``mask=True``
   * - ``nn_filter``
     - ``effects.nn_filter``
     - with differences
     - Dense recurrence matrices; mean, median or weighted-average aggregates

``librosa.filters``
~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 18 32 14 36

   * - librosa
     - audio_samples
     - status
     - notes
   * - ``get_window``
     - ``spectral.get_window``
     - equivalent
     - Periodic (``fftbins=True``) or symmetric
   * - ``mel``
     - ``spectral.mel_filterbank``
     - with differences
     - float64 weights (librosa ``dtype=float32``); empty filters are zero rows
       without a warning
   * - ``chroma``
     - ``features.chroma_filterbank``
     - with differences
     - float64 weights
   * - ``cq_to_chroma``
     - ``features.cq_to_chroma``
     - with differences
     - The smoothing ``window`` must be ``None``
   * - ``wavelet``
     - ``features.wavelet``
     - with differences
     - complex128 filters (librosa: complex64); no explicit ``alpha``
   * - ``wavelet_lengths``
     - ``features.wavelet_lengths``
     - with differences
     - No explicit ``alpha`` (at least two ascending ``freqs``)
   * - ``window_bandwidth``
     - ``spectral.window_bandwidth``
     - with differences
     - Always evaluates the formula; librosa returns a table value for named
       windows (up to ~1e-4 relative apart)
   * - ``window_sumsquare``
     - ``spectral.window_sumsquare``
     - equivalent
     - float64 output
   * - ``diagonal_filter``
     - --
     - not provided
     - Deferred: needs spline image rotation; only used by ``path_enhance``
   * - ``mr_frequencies``
     - --
     - not provided
     - Deferred: only used by ``iirt``
   * - ``semitone_filterbank``
     - --
     - not provided
     - Deferred: needs ``ellipord``

``librosa.util``
~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 18 32 14 36

   * - librosa
     - audio_samples
     - status
     - notes
   * - ``frame``
     - ``spectral.frame``
     - with differences
     - 1-D input, last-axis framing
   * - ``normalize``
     - ``features.normalize``
     - equivalent
     -
   * - ``peak_pick``
     - ``rhythm.peak_pick``
     - with differences
     - 1-D input
   * - ``sync``
     - ``sequence.sync``
     - with differences
     - Frame indices only (no slices); an empty index list gives one segment
   * - ``softmask``
     - ``effects.softmask``
     - equivalent
     -
   * - ``fix_length``
     - ``AudioSamples.pad_samples_right``
     - with differences
     - Pads only; a longer signal is returned unchanged
   * - ``localmax``, ``localmin``
     - --
     - not provided
     - Internal helpers of peak picking and beat tracking
   * - ``match_events``
     - --
     - not provided
     - Internal helper of ``remix(align_zeros=True)``
   * - ``nnls``
     - --
     - not provided
     - Deferred with ``feature.inverse``
   * - ``pad_center``, ``fix_frames``, ``abs2``, ``phasor``, ``tiny``,
       ``expand_to``, ``stack``, ``dtype_c2r``, ``dtype_r2c``,
       ``MAX_MEM_BLOCK``
     - --
     - not provided
     - NumPy one-liners and dtype plumbing
   * - ``buf_to_float``, ``valid_audio``
     - --
     - not provided
     - Decoding is file I/O; ``AudioSamples`` enforces its invariants and the
       librosa-compatible methods reject non-finite audio
   * - ``axis_sort``, ``sparsify_rows``, ``shear``, ``cyclic_gradient``,
       ``fill_off_diagonal``, ``index_to_slice``, ``count_unique``,
       ``is_unique``, ``interp_broadcast``, ``match_intervals``
     - --
     - not provided
     - Generic array utilities
   * - ``is_positive_int``, ``valid_int``, ``valid_intervals``,
       ``rename_kw``, ``Deprecated``
     - --
     - not provided
     - Argument-validation and deprecation plumbing
   * - ``example``, ``ex``, ``example_info``, ``list_examples``,
       ``find_files``, ``cite``
     - --
     - not provided
     - Example data, file discovery and citation helpers

``librosa.segment`` and ``librosa.sequence``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All in ``audio_samples.sequence``, on ``(n_features, n_frames)`` arrays.

.. list-table::
   :header-rows: 1
   :widths: 18 32 14 36

   * - librosa
     - audio_samples
     - status
     - notes
   * - ``segment.recurrence_matrix``
     - ``sequence.recurrence_matrix``
     - with differences
     - Dense only (``sparse=True`` raises); ``axis=-1`` only; exact
       neighbour search
   * - ``segment.cross_similarity``
     - ``sequence.cross_similarity``
     - with differences
     - As ``recurrence_matrix``
   * - ``segment.recurrence_to_lag``
     - ``sequence.recurrence_to_lag``
     - equivalent
     - Dense input
   * - ``segment.lag_to_recurrence``
     - ``sequence.lag_to_recurrence``
     - equivalent
     - Dense input
   * - ``segment.path_enhance``
     - --
     - not provided
     - Deferred: needs ``filters.diagonal_filter``
   * - ``segment.timelag_filter``
     - --
     - not provided
     - A decorator; compose ``recurrence_to_lag``, a filter and
       ``lag_to_recurrence``
   * - ``segment.agglomerative``, ``segment.subsegment``
     - --
     - not provided
     - scikit-learn clustering
   * - ``sequence.dtw``
     - ``sequence.dtw``
     - equivalent
     - int64 paths
   * - ``sequence.dtw_backtracking``
     - ``sequence.dtw_backtracking``
     - equivalent
     -
   * - ``sequence.rqa``
     - ``sequence.rqa``
     - with differences
     - Same scores; the path through the second column follows the move
       actually taken (librosa stores a wrong back-pointer there)
   * - ``sequence.viterbi``
     - ``sequence.viterbi``
     - equivalent
     - int64 states (librosa: uint16)
   * - ``sequence.viterbi_discriminative``
     - ``sequence.viterbi_discriminative``
     - equivalent
     - As ``viterbi``
   * - ``sequence.viterbi_binary``
     - ``sequence.viterbi_binary``
     - equivalent
     - As ``viterbi``
   * - ``sequence.transition_uniform``
     - ``sequence.transition_uniform``
     - equivalent
     -
   * - ``sequence.transition_loop``
     - ``sequence.transition_loop``
     - equivalent
     -
   * - ``sequence.transition_cycle``
     - ``sequence.transition_cycle``
     - equivalent
     -
   * - ``sequence.transition_local``
     - ``sequence.transition_local``
     - equivalent
     -
   * - ``sequence.path_to_steps``
     - --
     - not provided
     - Differences of consecutive path points

``librosa.display``
~~~~~~~~~~~~~~~~~~~

``librosa.display`` is matplotlib plotting and is not provided. The arrays
returned here are plain NumPy arrays, so they can be passed to
``librosa.display.specshow`` or to matplotlib directly (with ``[0]`` for mono
audio).

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - librosa
     - status
     - notes
   * - ``specshow``, ``waveshow``, ``AdaptiveWaveplot``, ``wavebars``, ``wavef0``
     - not provided
     - matplotlib plotting
   * - ``multiplot``, ``highlight``, ``legend_for_axes``, ``colorbar_db``,
       ``colorbar_phase``, ``infer_cmap``
     - not provided
     - matplotlib helpers
   * - ``TimeFormatter``, ``NoteFormatter``, ``LogHzFormatter``,
       ``ChromaFormatter``, ``ChromaFJSFormatter``, ``ChromaSvaraFormatter``,
       ``FJSFormatter``, ``TonnetzFormatter``, ``Transformf0``
     - not provided
     - matplotlib axis formatters

Differences from librosa
------------------------

These are the differences documented in the API reference
(:doc:`api/spectral`, :doc:`api/features`, :doc:`api/rhythm`,
:doc:`api/effects`, :doc:`api/sequence`) and in the Rust crate's migration
guide, gathered by theme.

Shapes and types
~~~~~~~~~~~~~~~~

- **The channel axis is always present** on ``AudioSamples`` methods:
  ``(channels, F, T)`` for frame-wise features, ``(channels, T)`` for onset
  envelopes and pitch tracks, ``(channels, 1)`` for a static tempo. Index
  ``[0]`` for librosa's mono shape. ``tempogram_ratio`` with ``aggregate`` keeps
  a ``(channels, n_factors, 1)`` shape where librosa drops the last axis.
- **Output precision follows the input** (see `Conventions`_). Several
  librosa bases default to single precision even for ``float64`` input. Here
  they are computed in ``float64`` and rounded once to the output type, so
  ``float64`` output differs from librosa's defaults by about 1e-7 relative:

  - the mel basis (``filters.mel``, hence ``melspectrogram``, ``mfcc``,
    ``onset_strength``);
  - the chroma basis (``filters.chroma``, ``chroma_stft``);
  - CQT wavelets (``filters.wavelet``): librosa's ``complex64`` filters move
    its top-octave CQT by up to ~4e-8 of the peak;
  - ``rms`` and the RMS behind ``trim_db`` / ``split_db``: librosa computes the
    mean square in ``float32``, so for ``float64`` audio a frame within ~1e-6
    dB of ``top_db`` can be classified differently.

  ``clicks``, ``window_sumsquare``, ``cq_to_chroma``, ``wavelet``, the
  filterbanks and the frequency weightings are tables and return ``float64``;
  tempo estimates and tuning are always ``float64``. ``yin``, ``pyin``,
  ``tempogram``, ``tempogram_ratio`` and ``hybrid_tempogram`` return
  ``float32`` for ``float32`` input (librosa promotes them to ``float64``).
  ``dtype=`` arguments that select a librosa output precision are accepted and
  ignored where documented (``griffinlim``).
- **Integer outputs are int64** (librosa: ``uint16`` Viterbi states, ``intp``
  paths). Event detectors return 1-D ``int64`` frames or samples, or
  ``float64`` times, chosen with librosa's ``units=``.
- **Integer audio scaling.** ``int16`` / ``int32`` audio is converted with this
  library's asymmetric scaling (negative values by ``2**(bits-1)``, positive by
  ``2**(bits-1) - 1``); librosa's ``buf_to_float`` divides both by
  ``2**(bits-1)``: up to one LSB apart.
- **Sequence and recurrence functions** take 2-D ``(n_features, n_frames)``
  input with the default ``axis=-1`` layout and return dense arrays
  (``sparse=True`` raises).

Defaults that differ from librosa
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pass librosa's value explicitly to reproduce a librosa call.

.. list-table::
   :header-rows: 1
   :widths: 40 20 40

   * - Argument
     - librosa
     - audio_samples
   * - ``top_db`` of ``power_to_db``, ``amplitude_to_db``, ``mfcc_frames``,
       ``perceptual_weighting``
     - 80
     - ``None``: an 80 dB floor discards dynamic range
   * - ``min_db`` of ``frequency_weighting``, ``multi_frequency_weighting``
     - -80
     - ``None`` (``a_weighting`` ... ``z_weighting`` keep -80)
   * - ``tuning`` of ``chroma_stft``, ``chroma_cqt``, ``chroma_cens``,
       ``tonnetz``, ``chroma_stft_from_spectrogram``
     - ``None`` (estimated)
     - ``0.0``; ``None`` still estimates
   * - ``trim`` of ``beat_track``, ``beat_track_with_bpm``
     - ``True``
     - ``False``: weak leading and trailing beats are kept
   * - ``rng`` of ``griffinlim``, ``griffinlim_audio``
     - unseeded
     - ``0``: repeated calls give the same output
   * - ``align_zeros`` of ``remix``
     - ``True``
     - ``False``: interval boundaries are used as given
   * - ``top_db`` of ``trim_db``, ``trim_db_in_place``, ``split_db``
     - 60
     - required
   * - ``sr`` of the free functions in ``spectral``, ``features``, ``rhythm``,
       ``effects`` (and ``sample_rate`` / ``sr`` of the ``utils`` frame and
       time conversions)
     - 22050
     - required
   * - ``hop_length`` of ``pcen``, ``utils.samples_to_frames``,
       ``frames_to_samples``, ``time_to_frames_floor``, ``times_like``,
       ``samples_like``, and of ``effects.clicks`` with ``frames=``
     - 512
     - required

The onset envelope (``onset_strength`` and everything built on it) and
``spectral_contrast`` keep librosa's 80 dB floor, which is part of those
features' definitions. All other defaults equal librosa's.

Resampling (no soxr)
~~~~~~~~~~~~~~~~~~~~

- ``librosa.resample`` defaults to soxr (``res_type="soxr_hq"``). This library
  has its own delay-free Kaiser-windowed sinc resampler;
  ``AudioSamples.resample(sr, "high")`` is designed after soxr_hq's passband
  and attenuation. Measured against ``librosa.resample`` from 22050 Hz: 129 to
  135 dB SNR in the middle of a band-limited signal, 59 to 104 dB at its abrupt
  edges, 50 to 58 dB on white noise. ``AudioSamples.resample`` defaults to
  ``"fast"``; pass ``"high"`` for librosa-like quality.
- The output length is exactly ``ceil(n * target / orig)`` (librosa computes it
  in floating point and can be one sample longer). Fractional rates and
  librosa's ``scale`` are not supported.
- Operations that resample internally inherit the difference:
  ``pitch_shift`` (~68 dB SNR; its ``res_type`` selects this library's
  quality, ``"soxr_hq"`` maps to ``"high"``); ``cqt``, ``vqt``,
  ``hybrid_cqt``, ``chroma_cqt``, ``chroma_cens``, ``chroma_vqt``, ``tonnetz``
  (top octave to round-off, decimated octaves ~1e-4 of the peak, the size of
  the difference between librosa's own ``soxr_hq`` and ``soxr_vhq``);
  ``icqt`` and ``griffinlim_cqt`` (upsampling stages). ``res_type`` on these
  selects this library's resampler quality.

Errors where librosa warns or fails late
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - Situation
     - librosa
     - audio_samples
   * - ``mfcc`` with ``n_mfcc > n_mels``
     - silently returns ``n_mels`` rows
     - ``ValueError``
   * - ``istft`` whose bin count does not match ``n_fft``
     - infers ``n_fft`` from the bins
     - ``ValueError``
   * - ``istft(length=0)``
     - treated as ``None``
     - ``ValueError``
   * - ``spectral_contrast`` band with no FFT bin
     - ``IndexError`` or NaN with a warning
     - ``ValueError``
   * - rank-deficient ``poly_features`` fit
     - ``RankWarning``, minimum-norm solution
     - error
   * - ``onset_backtrack`` with no events
     - raises
     - empty array
   * - ``trim`` on all-silent audio
     - empty array
     - ``ValueError`` (an ``AudioSamples`` cannot be empty)
   * - ``onset_detect`` / ``beat_track`` on multichannel audio
     - dense (``sparse=False``) broadcasting
     - ``ValueError`` (mono only)
   * - ``griffinlim`` with ``momentum > 1``
     - warns
     - accepted silently
   * - ``filters.mel`` with empty filters
     - warns
     - zero rows, silently
   * - ``hz_to_note`` outside MIDI 0..127
     - names any frequency
     - ``ValueError``

Randomness
~~~~~~~~~~

- ``griffinlim`` with ``init="random"`` (librosa's default) draws phases from
  this library's RNG, not NumPy's, so results differ from librosa's for the
  same seed (both are valid reconstructions). ``rng`` defaults to the seed 0,
  so repeated calls give the same output; ``rng=None`` draws a new seed on
  every call (librosa's default). Only integer seeds (or ``None``) are
  accepted as ``rng`` / ``random_state``; NumPy generators raise
  ``TypeError``. ``init=None`` is deterministic and matches librosa to
  round-off.
- ``griffinlim_cqt`` offers only ``init=None``.

Options that are not supported
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each raises ``ValueError`` or ``TypeError`` instead of being ignored:

- ``sparse=True`` in ``recurrence_matrix`` / ``cross_similarity`` (dense only),
  and ``sparse=False`` in ``onset_detect`` / ``beat_track`` (sparse only);
- ``vqt`` / ``chroma_vqt``: ``intervals`` other than ``"equal"``;
- ``interp_harmonics`` / ``f0_harmonics`` / ``salience``: ``kind`` other than
  ``"linear"`` and a frequency ``axis`` other than ``-2``;
- ``griffinlim_cqt``: ``init="random"``;
- ``chroma_cqt`` / ``cq_to_chroma``: the smoothing ``window``;
- ``wavelet`` / ``wavelet_lengths``: an explicit ``alpha``;
- ``spectral_centroid_from_spectrogram``: a time-varying 2-D ``freq``;
- ``onset_strength_multi``: ``channels`` given as slices;
- ``tempogram_ratio``: custom ``freqs``;
- ``metrogram``: ``aggregate=None``;
- ``harmonic`` / ``percussive``: ``mask=True``;
- ``pitch_shift``: ``scale=True``;
- ``sync``: slice indices (frame indices only);
- ``nn_filter``: sparse recurrence matrices and aggregates other than mean,
  median or a weighted average;
- callables: only ``numpy.max`` / ``numpy.mean`` / ``numpy.median`` (and the
  others named in each docstring) where librosa accepts any callable;
- ``magphase``: a fractional ``power``.

Deliberate behavioural differences
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``window_bandwidth``: librosa returns a hard-coded table value for named
  windows and ignores ``n``; here the formula is always evaluated (up to ~1e-4
  relative apart, e.g. Blackman 1.72697 vs 1.72676).
- ``harmonic`` / ``percussive`` / ``harmonic_percussive``: librosa accepts a
  ``window`` argument but always uses Hann; here ``window`` is used.
- ``rqa``: librosa stores the wrong back-pointer for a ``(-2, -1)`` move into
  the second column, so its backtracking can step outside the matrix. The
  scores are identical; the path here follows the move actually taken.
- ``sync`` with an empty index list returns one segment ``[0, n_frames)``;
  librosa returns zero columns.
- ``chroma_stft`` with an odd ``n_fft``: librosa's tuning estimate re-infers the
  FFT length as ``2 * (bins - 1)``; here the given ``n_fft`` is used.
- ``recurrence_matrix``: exact brute-force neighbour search (librosa uses
  scikit-learn, whose distances can differ by round-off); librosa's
  connectivity tie-breaking is reproduced.
- ``time_stretch``: on input containing exact digital silence the phases of
  near-zero bins are ill-conditioned and round-off is amplified (~30 dB SNR
  there, as for librosa against itself with a 1e-15 perturbation).

.. _librosa-existing-apis:

Existing methods that are not librosa-compatible
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These older methods keep their behaviour for compatibility. Use the
librosa-compatible name in the last column.

.. list-table::
   :header-rows: 1
   :widths: 30 40 30

   * - Existing API
     - Difference from librosa
     - librosa-compatible API
   * - ``AudioSamples.stft``, ``istft``, ``mel_spectrogram``,
       ``constant_q_transform``, ``chroma``
     - Use the ``spectrograms`` crate: symmetric windows, mono only, other
       normalisations and kernels
     - ``stft_matrix``, ``istft_matrix``, ``melspectrogram``, ``cqt``,
       ``chroma_stft``
   * - ``AudioSamples.mfcc``
     - Symmetric window, no mel norm, absolute dB floor, unnormalised DCT,
       lifter 22
     - ``mfcc_frames``
   * - ``AudioSamples.spectral_centroid``, ``spectral_rolloff``,
       ``spectral_bandwidth``, ``spectral_flatness``, ``spectral_contrast``
     - One value per signal from a single FFT; power-weighted
     - ``spectral_*_frames``
   * - ``AudioSamples.zero_crossing_rate``
     - One whole-signal rate in Hz
     - ``zero_crossing_rate_frames``
   * - ``AudioSamples.rms``
     - One whole-signal value
     - ``rms_frames``
   * - ``AudioSamples.detect_onsets``, ``onset_detection_function``,
       ``spectral_flux``, ``complex_onset_detection``
     - Different detection functions and peak picker
     - ``onset_strength``, ``onset_detect``
   * - ``AudioSamples.detect_beats``, ``estimate_tempo``
     - Beat grid at a given tempo; global-autocorrelation tempo
     - ``beat_track``, ``tempo``
   * - ``AudioSamples.hpss``
     - Its own mask blend and window conventions
     - ``harmonic_percussive``, ``effects.hpss``
   * - ``AudioSamples.trim``, ``trim_silence``
     - Threshold individual samples in absolute dBFS
     - ``trim_db``
   * - ``AudioSamples.split``
     - Fixed-length chunking
     - ``split_db``
   * - ``AudioSamples.detect_pitch_yin``, ``track_pitch``
     - One estimate per buffer or uncentred frames
     - ``yin``, ``pyin``
   * - ``utils.hz_to_mel``, ``mel_to_hz``, ``mel_frequencies`` (defaults)
     - HTK scale by default
     - the same functions with ``htk=False``
   * - ``utils.time_to_frames``
     - Rounds to the nearest frame
     - ``utils.time_to_frames_floor``
   * - ``utils.seconds_to_samples``
     - Rounds (librosa truncates)
     - ``utils.time_to_samples``
   * - ``AudioSamples.mu_compress``, ``mu_expand``
     - ``quantize=False`` (librosa's default is ``quantize=True``)
     - ``mu_law_encode``, ``mu_law_decode``
   * - ``AudioSamples.windows``
     - Zero-pads trailing windows
     - ``spectral.frame``
   * - ``AudioSamples.as_f64`` / ``to_format`` on integer audio
     - Asymmetric integer scaling
     - --
