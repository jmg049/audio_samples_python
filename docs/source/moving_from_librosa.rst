Moving from librosa
===================

This page is for someone who knows `librosa <https://librosa.org>`_ and is deciding
whether to switch to ``audio_samples``. It gives measurements on both sides, a pipeline
written in both libraries, the differences that catch people out, recipes for common
jobs, a way to migrate in steps, and the cases where librosa is still the better choice.
:doc:`librosa_migration` is the function-by-function lookup. Rust users and anyone porting
a pipeline to Rust should also read
`the Rust companion <https://github.com/jmg049/audio_samples/blob/main/documentation/MOVING_FROM_LIBROSA_RUST.md>`_.

``audio_samples`` is not a wrapper around librosa or a copy of it. It is a Rust library
with Python bindings. Its librosa-compatible layer reproduces librosa 1.0 results where
that makes sense and departs from librosa's defaults where those change the data
without being asked.

.. contents:: On this page
   :local:
   :depth: 1

Install it from source for now
------------------------------

.. important::

   The API on this page is not on PyPI yet. ``pip install audio_samples`` gives version
   0.2.4 (checked 2026-11-04), which has no ``stft_matrix``, ``mfcc_frames``,
   ``spectral``, ``features``, ``rhythm``, ``effects`` or ``sequence``. That release has
   one wheel, CPython 3.12 on x86-64 Linux; on anything else pip tries to build the
   source distribution. Until a release includes this API, build from two checkouts side
   by side, which needs a Rust toolchain (1.87 or newer):

.. code-block:: bash

   git clone https://github.com/jmg049/audio_samples
   git clone https://github.com/jmg049/audio_samples_python
   cd audio_samples_python
   pip install maturin
   maturin develop --release

The bindings have one required Python dependency, NumPy. Every example below has run
only on x86-64 Linux, on CPython 3.13.

Evidence
--------

@@EVIDENCE@@

A complete pipeline, side by side
---------------------------------

Every Python block on this page runs, in order, in the test suite
(``tests/test_moving_from_librosa.py``), and the ``assert`` lines are the comparison: if
a result stopped matching, the test would fail. The first block writes the file the rest
use: 20 s of decaying notes at 120 BPM, stereo, 44100 Hz, float32 WAV.

.. code-block:: python

   import numpy as np
   import audio_samples as aus

   sr_file = 44100
   n = 20 * sr_file
   noise = np.random.default_rng(0).standard_normal((2, n))
   x = 0.002 * noise
   for k in range(40):
       start, length = k * sr_file // 2, 2 * sr_file // 5
       t = np.arange(length) / sr_file
       f = 110.0 * 2.0 ** (((k * 7) % 24) / 12.0)
       note = np.exp(-8.0 * t) * (0.5 * np.sin(2 * np.pi * f * t) + 0.25 * np.sin(4 * np.pi * f * t))
       x[0, start : start + length] += note
       x[1, start : start + length] += 0.8 * note
   song = aus.AudioSamples.new_multi(x.astype(np.float32), sr_file)
   aus.io.save("song.wav", song)


   def relative_error(ours, theirs):
       """Largest difference as a fraction of the largest value."""
       return np.abs(ours - theirs).max() / np.abs(theirs).max()

The pipeline in librosa:

.. code-block:: python

   import librosa

   y, sr = librosa.load("song.wav")                                # mono, 22050 Hz, float32
   D = librosa.stft(y)
   S = librosa.feature.melspectrogram(y=y, sr=sr)
   S_db = librosa.power_to_db(S, ref=np.max)
   mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
   onsets = librosa.onset.onset_detect(y=y, sr=sr, units="time")
   tempo, beats = librosa.beat.beat_track(y=y, sr=sr, units="time")

The same in ``audio_samples``. Each thing librosa does by default is written out: the
type, the mix-down, the rate, the 80 dB floor and beat trimming.

.. code-block:: python

   a = (
       aus.io.read("song.wav", as_type=aus.SampleType.F32)
       .to_mono()
       .resample(22050, "high")
   )
   a_D = a.stft_matrix()                                           # (channels, bins, frames)
   a_S = a.melspectrogram()
   a_S_db = aus.spectral.power_to_db(a_S, ref=np.max, top_db=80.0)
   a_mfcc = a.mfcc_frames(n_mfcc=13, top_db=80.0)
   a_onsets = a.onset_detect(units="time")
   a_tempo, a_beats = a.beat_track(units="time", trim=True)

The shapes differ by the leading channel axis. Onsets, tempo and beats are identical on
this file. The resamplers are different algorithms (see `Resampling is not soxr`_), which
is where the differences in the arrays come from:

.. code-block:: python

   assert a.to_numpy().dtype == y.dtype == np.float32
   assert a.sample_rate == sr == 22050
   assert a_D.shape == (1, *D.shape) and a_mfcc.shape == (1, *mfcc.shape)

   # The two resamplers: about 95 dB apart here.
   snr_db = 10 * np.log10(np.sum(y**2) / np.sum((a.to_numpy() - y) ** 2))
   assert snr_db > 90

   assert relative_error(a_D[0], D) < 2e-5
   assert relative_error(a_S[0], S) < 2e-6
   assert relative_error(a_mfcc[0], mfcc) < 2e-4
   assert np.array_equal(a_onsets, onsets)
   assert a_tempo[0] == tempo[0]
   assert np.array_equal(a_beats, beats)

Feed both libraries the same samples and the resampler is out of the comparison. The
rest then agrees to float32 round-off, and to about 1e-15 in float64 when librosa is
told to build its mel basis in float64:

.. code-block:: python

   b = aus.AudioSamples.new_mono(y, sr)
   assert relative_error(b.stft_matrix()[0], D) < 1e-6
   assert relative_error(b.melspectrogram()[0], S) < 1e-6
   assert relative_error(b.mfcc_frames(n_mfcc=13, top_db=80.0)[0], mfcc) < 5e-6

   y64 = y.astype(np.float64)
   b64 = aus.AudioSamples.new_mono(y64, sr)
   S64 = librosa.feature.melspectrogram(y=y64, sr=sr, dtype=np.float64)
   assert relative_error(b64.melspectrogram()[0], S64) < 1e-14
   mfcc64 = librosa.feature.mfcc(y=y64, sr=sr, dtype=np.float64)
   assert relative_error(b64.mfcc_frames(top_db=80.0)[0], mfcc64) < 1e-13

The differences that will trip you up
-------------------------------------

Each item shows the librosa code, what a direct translation does, and the correct form.
``raises`` below returns the error message, and fails if nothing is raised.

.. code-block:: python

   def raises(exception, call):
       try:
           call()
       except exception as error:
           return str(error)
       raise AssertionError("nothing was raised")

The audio carries its rate
~~~~~~~~~~~~~~~~~~~~~~~~~~

librosa passes ``y`` and ``sr`` around. Here the audio holds its rate, and methods have
neither argument. Passing either is a ``TypeError``, not something ignored:

.. code-block:: python

   # librosa.feature.melspectrogram(y=y, sr=sr)
   assert "unexpected keyword argument 'sr'" in raises(TypeError, lambda: b.melspectrogram(sr=sr))
   assert "unexpected keyword argument 'y'" in raises(TypeError, lambda: b.melspectrogram(y=y))
   # librosa.effects.pitch_shift(y, sr=sr, n_steps=2)
   up = b.pitch_shift(2.0)
   assert up.sample_rate == sr

Free functions over arrays do take ``sr``, and it is required where librosa defaults it
to 22050. A wrong default rate gives wrong results with no error, so there is no default:

.. code-block:: python

   # librosa.onset.onset_strength(S=S_db, sr=sr)
   env = aus.rhythm.onset_strength(S=S_db, sr=sr)
   assert raises(TypeError, lambda: aus.rhythm.onset_strength(S=S_db))
   assert relative_error(env, librosa.onset.onset_strength(S=S_db, sr=sr)) < 1e-5

The older unit conversions name it differently. ``utils.frames_to_time`` takes
``(frames, sample_rate, hop_size)``:

.. code-block:: python

   # librosa.frames_to_time(10, sr=22050, hop_length=512)
   assert aus.utils.frames_to_time(10, 22050, 512) == librosa.frames_to_time(10, sr=22050, hop_length=512)

Reading a file changes nothing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``librosa.load(path)`` returns mono float32 at 22050 Hz whatever the file holds.
``aus.io.read(path)`` returns what the file holds: rate, channels and sample type. Code
that assumes librosa's result keeps running on different data, so check this first when
porting.

.. code-block:: python

   aus.io.save("song16.wav", song, as_type=aus.SampleType.I16)

   y16, sr16 = librosa.load("song16.wav")
   assert (y16.dtype, y16.ndim, sr16) == (np.float32, 1, 22050)

   naive = aus.io.read("song16.wav")                               # the direct translation
   assert naive.to_numpy().dtype == np.int16
   assert naive.num_channels == 2 and naive.sample_rate == 44100

   # What librosa.load(path) did:
   same = aus.io.read("song16.wav", as_type=aus.SampleType.F32).to_mono().resample(22050, "high")
   assert same.to_numpy().dtype == np.float32 and same.sample_rate == 22050

16-bit values are raw integers until you ask for floats. Analysis methods convert them
themselves, so this is only a trap if you take the array out. The scaling is also not
quite librosa's: negative values are divided by 32768 and positive ones by 32767, where
libsndfile divides both by 32768, so a 16-bit file read as float differs from
``librosa.load(path, sr=None)`` by at most about 3e-5:

.. code-block:: python

   ours16 = aus.io.read("song16.wav", as_type=aus.SampleType.F32).to_numpy()
   theirs16, _ = librosa.load("song16.wav", sr=None, mono=False)
   assert 0 < np.abs(ours16 - theirs16).max() < 3.1e-5

Precision follows the input
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Analysis of ``uint8``, ``int16``, ``int24`` and ``float32`` audio returns ``float32`` or
``complex64``; ``int32`` and ``float64`` audio returns ``float64`` or ``complex128``.
Audio-returning operations keep the sample type. librosa's float32 input stays float32 for
most results, but it also promotes some (``yin``, ``pyin``, ``tempogram``) and it builds
mel and chroma bases in float32 even for float64 input.

.. code-block:: python

   assert naive.mfcc_frames().dtype == np.float32                  # int16 audio
   assert naive.time_stretch(2.0).to_numpy().dtype == np.int16     # stays int16
   assert b64.stft_matrix().dtype == np.complex128

   f0_ours = b.yin(fmin=65.4, fmax=2093.0)
   f0_theirs = librosa.yin(y, fmin=65.4, fmax=2093.0, sr=sr)
   assert f0_ours.dtype == np.float32 and f0_theirs.dtype == np.float64

Free functions over arrays compute in ``float32`` for a ``float32`` or ``complex64``
array, and for ``float32`` ``AudioSamples``; everything else is converted to ``float64``.
Integer arrays are converted by value, without audio scaling, so give them floats. Passing
``dtype=`` to ``griffinlim`` is accepted and ignored: the input's precision decides.

.. _changed defaults:

Some defaults are different on purpose
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Where librosa's default changes the data without being asked, this library does not
follow it. These calls run without a complaint and give different numbers, so they are the
most important ones to check. In Rust, ``X::librosa()`` builds a config with librosa's
settings; in Python there are no presets, so pass the keyword.

.. list-table::
   :header-rows: 1
   :widths: 44 18 38

   * - Argument
     - librosa
     - audio_samples, and what to pass
   * - ``top_db`` of ``power_to_db``, ``amplitude_to_db``, ``mfcc_frames``, ``perceptual_weighting``
     - 80
     - ``None``; pass ``top_db=80.0``
   * - ``min_db`` of ``frequency_weighting``, ``multi_frequency_weighting``
     - -80
     - ``None``; pass ``min_db=-80.0``
   * - ``tuning`` of ``chroma_stft``, ``chroma_cqt``, ``chroma_cens``, ``tonnetz``
     - ``None`` (estimated)
     - ``0.0``; pass ``tuning=None``
   * - ``trim`` of ``beat_track``
     - ``True``
     - ``False``; pass ``trim=True``
   * - ``rng`` of ``griffinlim``
     - unseeded
     - ``0``; pass ``rng=None`` for a new seed each call
   * - ``align_zeros`` of ``remix``
     - ``True``
     - ``False``; pass ``align_zeros=True``
   * - ``quality`` of ``AudioSamples.resample``
     - ``soxr_hq``
     - ``"fast"``; pass ``"high"``
   * - ``top_db`` of ``trim_db``, ``split_db``
     - 60
     - required
   * - ``sr`` of free functions and ``hop_length`` of ``pcen``, ``utils`` conversions
     - 22050, 512
     - required
   * - ``htk`` of ``utils.hz_to_mel``, ``mel_to_hz``, ``mel_frequencies``
     - ``False``
     - ``True``; pass ``htk=False``

.. code-block:: python

   # librosa.power_to_db(S, ref=np.max) floors at -80 dB; the direct translation does not.
   assert librosa.power_to_db(S, ref=np.max).min() == -80.0
   assert aus.spectral.power_to_db(S, ref=np.max).min() < -80.0
   assert aus.spectral.power_to_db(S, ref=np.max, top_db=80.0).min() == -80.0

   # librosa.beat.beat_track keeps fewer beats than the direct translation.
   assert len(b.beat_track(units="time")[1]) > len(beats)
   assert np.array_equal(b.beat_track(units="time", trim=True)[1], beats)

   # librosa.feature.chroma_stft estimates the tuning; the direct translation assumes 0.
   chroma = librosa.feature.chroma_stft(y=y, sr=sr)
   assert np.abs(b.chroma_stft()[0] - chroma).max() > 1e-3
   assert np.abs(b.chroma_stft(tuning=None)[0] - chroma).max() < 1e-5

   # librosa.hz_to_mel uses the Slaney scale; utils.hz_to_mel defaults to HTK.
   assert aus.utils.hz_to_mel(440.0) != librosa.hz_to_mel(440.0)
   assert aus.utils.hz_to_mel(440.0, htk=False) == librosa.hz_to_mel(440.0)

The onset envelope and ``spectral_contrast`` keep librosa's 80 dB floor, which is part of
how those features are defined.

Boolean results
~~~~~~~~~~~~~~~

``recurrence_matrix`` and ``cross_similarity`` in connectivity mode return ``bool``, as in
librosa. ``viterbi_binary`` returns ``bool`` states where librosa returns ``uint16``; ask
for librosa's type if later code does arithmetic on the states.

.. code-block:: python

   features = np.random.default_rng(1).random((4, 30))
   assert aus.sequence.recurrence_matrix(features).dtype == np.bool_
   assert librosa.segment.recurrence_matrix(features).dtype == np.bool_

   prob = np.random.default_rng(2).random((3, 30))
   transition = np.array([[0.8, 0.2], [0.2, 0.8]])
   states = librosa.sequence.viterbi_binary(prob, transition)
   assert states.dtype == np.uint16
   assert aus.sequence.viterbi_binary(prob, transition).dtype == np.bool_
   assert np.array_equal(aus.sequence.viterbi_binary(prob, transition, dtype=np.uint16), states)

Shapes and channel layout
~~~~~~~~~~~~~~~~~~~~~~~~~

Methods always return a leading channel axis, also for mono: ``(channels, n_features,
n_frames)``, and ``(channels, n_frames)`` for onset envelopes and pitch tracks. librosa
drops the axis for 1-D input. Index ``[0]`` to get its mono shape. Free functions follow
librosa's shapes.

.. code-block:: python

   assert b.melspectrogram().shape == (1, *S.shape)
   assert b.onset_strength().shape[0] == 1
   assert aus.spectral.stft(y).shape == D.shape

   stereo = aus.io.read("song.wav")
   assert stereo.shape == [2, n]
   assert stereo.melspectrogram().shape[0] == 2

Some older methods have the name you would reach for and do something else. They predate
the librosa-compatible layer and keep their behaviour. The ones that take arguments fail
loudly. These do not: they return one number for the whole signal.

.. code-block:: python

   assert isinstance(b.rms(), float)                               # use rms_frames()
   assert isinstance(b.spectral_centroid(), float)                 # use spectral_centroid_frames()
   assert isinstance(b.zero_crossing_rate(), float)                # in Hz; use zero_crossing_rate_frames()

   assert "params" in raises(TypeError, lambda: b.stft())          # use stft_matrix()
   assert "mfcc_params" in raises(TypeError, lambda: b.mfcc())     # use mfcc_frames()
   assert "config" in raises(TypeError, lambda: b.hpss())          # use harmonic_percussive()
   raises(TypeError, lambda: b.trim())                             # use trim_db(top_db=60.0)

Errors where librosa warns or carries on
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # librosa returns n_mels rows; here it is an error.
   assert librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40, n_mels=20).shape[0] == 20
   assert "n_mfcc" in raises(ValueError, lambda: b.mfcc_frames(n_mfcc=40, n_mels=20))

   # librosa detects onsets per channel only with sparse=False; here mono is required.
   assert "mono" in raises(ValueError, lambda: stereo.onset_detect())
   assert "sparse" in raises(ValueError, lambda: b.onset_detect(sparse=False))

   # Options that are not implemented raise instead of being ignored.
   assert "sparse" in raises(ValueError, lambda: aus.sequence.recurrence_matrix(features, sparse=True))
   assert "scale" in raises(ValueError, lambda: b.pitch_shift(2.0, scale=True))

   # trim with a fixed reference when every frame is below it.
   quiet = np.zeros(4096, dtype=np.float32)
   assert librosa.effects.trim(quiet, top_db=60, ref=1.0)[0].size == 0
   assert "silent" in raises(ValueError, lambda: aus.AudioSamples.new_mono(quiet, sr).trim_db(top_db=60.0, ref=1.0))

Resampling is not soxr
~~~~~~~~~~~~~~~~~~~~~~

``librosa.resample`` uses soxr. This library has its own delay-free Kaiser-windowed sinc
resampler. ``"high"`` is designed after ``soxr_hq``'s passband and attenuation but does not
reproduce it, so anything downstream of resampling agrees less closely than everything
else on this page. Measured against librosa from 22050 Hz to 11025, 16000 and 44100 Hz: 129
to 135 dB SNR in the middle of a band-limited signal, 59 to 104 dB at its abrupt edges, 50
to 58 dB on white noise. There is no ``res_type`` choice of filter design (``soxr_vhq``,
``kaiser_best``, ``scipy`` and so on), and fractional rates are not accepted.

``AudioSamples.resample`` defaults to ``"fast"``, which is a much lower quality:

.. code-block:: python

   ours_fast = b.resample(16000).to_numpy()
   ours_high = b.resample(16000, "high").to_numpy()
   theirs = librosa.resample(y, orig_sr=sr, target_sr=16000)
   snr = lambda u, v: 10 * np.log10(np.sum(v**2) / np.sum((u - v) ** 2))
   assert snr(ours_high, theirs) > 80 > snr(ours_fast, theirs)
   assert b.resample(16000, "high").samples_per_channel() == int(np.ceil(y.size * 16000 / sr))

Everything that resamples inside inherits the difference: ``pitch_shift``, ``cqt``,
``vqt``, ``hybrid_cqt``, ``chroma_cqt``, ``chroma_cens``, ``tonnetz``, ``icqt``.

Keyword names
~~~~~~~~~~~~~

Parameter names follow librosa 1.0 in the librosa-compatible layer, with these exceptions.
Some method names differ because the plain name was taken: ``stft_matrix``, ``mfcc_frames``,
``spectral_centroid_frames`` and the other ``_frames`` features, ``trim_db``, ``split_db``,
``harmonic_percussive``. The weightings are lower case (``features.a_weighting``). Unit
conversions: ``hz_to_note`` is ``utils.frequency_to_note`` and returns ``(name, cents)``;
``note_to_hz`` is ``utils.note_to_frequency``; ``time_to_frames`` is
``utils.time_to_frames_floor``, because ``utils.time_to_frames`` rounds to nearest. Sharps
are ASCII ``#``.

.. code-block:: python

   assert aus.utils.frequency_to_note(440.0)[0] == librosa.hz_to_note(440.0, unicode=False)
   assert aus.utils.note_to_frequency("A4") == librosa.note_to_hz("A4")
   times = [0.1, 0.7, 1.3]
   assert np.array_equal(
       aus.utils.time_to_frames_floor(times, sr=22050, hop_length=512),
       librosa.time_to_frames(times, sr=22050, hop_length=512),
   )

Getting librosa's exact results
-------------------------------

Pass librosa's value for each default in `changed defaults`_, feed both libraries the same
samples, and compare in the precision you care about. Rust configs have ``X::librosa()``
for this; the Python bindings have no presets. This is what agrees and how closely, from the
golden suite (fixtures generated by librosa 1.0.0; ``cargo test --release --features
full_no_plotting --test librosa_reference -- --nocapture``), in float64, as a fraction of
the peak:

.. list-table::
   :header-rows: 1
   :widths: 50 20 30

   * - Area
     - Worst measured
     - Notes
   * - Windows, mel and chroma filterbanks, unit conversions, ``normalize``
     - 6e-16
     - librosa's ``dtype=np.float64``
   * - STFT, ISTFT, dB, melspectrogram, MFCC, delta
     - 2e-12
     - 3.3e-8 against librosa's default float32 mel basis
   * - Spectral features, PCEN, weightings, reassigned spectrogram
     - 5e-11
     -
   * - ``chroma_stft``, ``piptrack``
     - 1.5e-15
     - Tuning estimates identical
   * - ``yin``, ``pyin``, ``salience``
     - 3.3e-16
     - pYIN voiced flags identical
   * - Onset strength, tempograms, PLP
     - 3.1e-14
     - Onset and beat frames and tempi identical
   * - HPSS, softmask, emphasis, remix, LPC, ``nn_filter``
     - 6e-13
     - ``trim`` and ``split`` boundaries identical
   * - Phase vocoder, ``time_stretch``, Griffin-Lim from zero phase
     - 7e-11
     -
   * - DTW, Viterbi, RQA, recurrence, ``sync``
     - 3.5e-16
     - Paths and state sequences identical
   * - CQT top octave
     - 1.7e-16
     - Lower octaves up to 1.3e-4, through the resampler
   * - ``pitch_shift``
     - 68 dB SNR
     - Through the resampler; depends on the signal
   * - ``resample`` (``"high"``)
     - 129 to 135 dB SNR
     - Middle of a band-limited signal; 50 to 58 dB on white noise

float32 input agrees to float32 round-off, 1e-7 to 5e-6 depending on the function. The
exception is ``spectral_contrast`` on a tonal signal, up to 9.2e-3 of its range (about
0.2 dB), from float32 FFT noise in the valleys.

These do not agree, on purpose or because the implementations differ:

- ``griffinlim`` with ``init="random"``: phases come from a different generator. Both are
  valid reconstructions. ``init=None`` agrees.
- ``window_bandwidth``: librosa returns a table value for named windows; this evaluates the
  formula (up to 1e-4 apart).
- ``rqa``: librosa stores a wrong back-pointer for a ``(-2, -1)`` move into the second
  column. Scores are identical; the path here follows the move taken.
- ``harmonic`` / ``percussive`` / ``harmonic_percussive``: ``window`` is used; librosa
  ignores it and uses Hann.
- ``time_stretch`` on input with exact digital silence: phases of near-zero bins are
  ill-conditioned, about 30 dB SNR there, as librosa against itself with a 1e-15 change.
- ``recurrence_matrix``: exact neighbour search here; librosa's scikit-learn search can
  differ by round-off in distances. Connectivity tie-breaking matches.

.. code-block:: python

   gl_ours = aus.effects.griffinlim(np.abs(D), n_iter=4, init=None)
   gl_theirs = librosa.griffinlim(np.abs(D), n_iter=4, init=None)
   assert relative_error(gl_ours, gl_theirs) < 1e-3
   f0, flag, prob_voiced = b.pyin(fmin=65.4, fmax=2093.0)
   lf0, lflag, lprob = librosa.pyin(y, fmin=65.4, fmax=2093.0, sr=sr)
   assert np.array_equal(flag[0], lflag)
   assert np.nanmax(np.abs(f0[0] - lf0) / lf0) < 1e-6

Recipes
-------

Features for machine learning
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A fixed-length vector per file. Method results are ordinary NumPy arrays:

.. code-block:: python

   def feature_vector(path):
       audio = aus.io.read(path, as_type=aus.SampleType.F32).to_mono().resample(22050, "high")
       mfcc = audio.mfcc_frames(n_mfcc=13, top_db=80.0)[0]
       delta = aus.spectral.delta(mfcc)
       centroid = audio.spectral_centroid_frames()[0]
       rms = audio.rms_frames()[0]
       zcr = audio.zero_crossing_rate_frames()[0]
       chroma = audio.chroma_stft(tuning=None)[0]
       parts = [mfcc, delta, centroid, rms, zcr, chroma]
       return np.concatenate([np.r_[p.mean(axis=1), p.std(axis=1)] for p in parts])

   vector = feature_vector("song.wav")
   assert vector.dtype == np.float32 and vector.shape == (2 * (13 + 13 + 1 + 1 + 1 + 12),)

Batch processing a directory, with threads
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

File reads, ``to_mono``, ``resample`` and the analysis methods release the GIL, so a thread
pool scales without the pickling and start-up of a process pool. librosa's load and
numba-compiled and NumPy steps also run in threads. See the batch table under `Evidence`_.
``tempo``'s estimate, a Python callable given as ``aggregate`` and ``StreamedAudioReader`` hold
the GIL.

.. code-block:: python

   from concurrent.futures import ThreadPoolExecutor
   from pathlib import Path

   for i in range(4):
       aus.io.save(f"take{i}.wav", song)

   files = sorted(Path(".").glob("take*.wav"))
   with ThreadPoolExecutor(max_workers=4) as pool:
       matrix = np.stack(list(pool.map(feature_vector, files)))
   assert matrix.shape == (4, vector.size)

A directory of formats other than WAV and FLAC needs a decoder first; see `Other file
formats`_. There is no equivalent of ``librosa.util.find_files``; ``pathlib`` does it.

Beat tracking
~~~~~~~~~~~~~

.. code-block:: python

   tempo_ours, beat_times = b.beat_track(units="time", trim=True)
   beat_samples = b.beat_track(units="samples", trim=True)[1]
   assert np.array_equal(beat_times, beats) and beat_samples.dtype == np.int64

   # A precomputed envelope, as librosa's onset_envelope= form.
   envelope = b.onset_strength()[0]
   tempo_env, beat_frames = aus.rhythm.beat_track(onset_envelope=envelope, sr=sr, trim=True)
   theirs_env = librosa.beat.beat_track(onset_envelope=envelope, sr=sr)[1]
   assert np.array_equal(beat_frames, theirs_env)

Beat tracking and onset detection need mono audio, so mix down first or loop over
``extract_channel``.

Harmonic-percussive separation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   harmonic, percussive = b.harmonic_percussive()
   h_theirs, p_theirs = librosa.effects.hpss(y)
   assert relative_error(harmonic.to_numpy(), h_theirs) < 1e-5
   assert relative_error(percussive.to_numpy(), p_theirs) < 1e-5

   # On a spectrogram, with masks if you want them.
   H, P = aus.effects.hpss(np.abs(D))
   assert relative_error(H, librosa.decompose.hpss(np.abs(D))[0]) < 1e-5
   mask_h, mask_p = aus.effects.hpss(np.abs(D), mask=True)

Pitch tracking with pYIN
~~~~~~~~~~~~~~~~~~~~~~~~

``fmin`` and ``fmax`` are required, as in librosa. The arrays have the channel axis, and
the float type follows the input (librosa returns float64 for float32 input):

.. code-block:: python

   note = aus.generation.sine_wave(220.0, 0.5, 22050, 0.5, dtype=np.float32)
   f0, voiced, voiced_prob = note.pyin(fmin=65.4, fmax=2093.0)
   assert f0.dtype == np.float32 and f0.shape[0] == 1
   assert voiced[0, 11] and abs(f0[0, 11] - 220.0) < 2.0

Time-stretch and pitch-shift
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   faster = b.time_stretch(1.25)
   assert relative_error(faster.to_numpy(), librosa.effects.time_stretch(y, rate=1.25)) < 1e-5

   higher = b.pitch_shift(2.0)
   assert higher.samples_per_channel() == y.size

   # Stereo works on both; a sample type is kept.
   assert stereo.time_stretch(1.25).num_channels == 2
   assert naive.pitch_shift(2.0).to_numpy().dtype == np.int16

``pitch_shift`` ends in a resampling step, so it agrees only as far as the resamplers do.

Griffin-Lim
~~~~~~~~~~~

From a magnitude spectrogram. ``rng`` is seeded by default, so repeated calls give the
same result:

.. code-block:: python

   magnitude = np.abs(D)
   first = aus.effects.griffinlim(magnitude, n_iter=8)
   second = aus.effects.griffinlim(magnitude, n_iter=8)
   assert np.array_equal(first, second)
   assert first.dtype == np.float32

   as_audio = aus.AudioSamples.griffinlim_audio(magnitude, sample_rate=sr, n_iter=8)
   assert as_audio.sample_rate == sr

``griffinlim_cqt`` only offers the deterministic start, and ``librosa.feature.inverse``
(``mel_to_audio``, ``mfcc_to_audio`` and friends) has no equivalent.

Plotting
~~~~~~~~

Results are NumPy arrays, so matplotlib works directly. ``librosa.display`` is not
reproduced, but nothing stops you using it on these arrays: it only needs the matrix and the
rate. ``audio_samples.mpl`` has its own plots on top of matplotlib (``pip install
matplotlib seaborn``), and is imported on first use.

.. code-block:: python

   import matplotlib

   matplotlib.use("Agg")
   import matplotlib.pyplot as plt
   import librosa.display

   figure, axes = plt.subplots(3, 1)
   axes[0].imshow(a_S_db[0], origin="lower", aspect="auto")
   librosa.display.specshow(a_S_db[0], sr=a.sample_rate, x_axis="time", y_axis="mel", ax=axes[1])
   aus.mpl.plot_waveform(a, ax=axes[2])
   plt.close(figure)

Other file formats
~~~~~~~~~~~~~~~~~~

``aus.io`` reads WAV and FLAC only. MP3, Ogg Vorbis, Opus and AIFF raise an error, where
``librosa.load`` handles them through libsndfile. Decode with soundfile and wrap the array,
which is what ``librosa.load`` does. soundfile returns ``(frames, channels)``; these take
``(channels, frames)``:

.. code-block:: python

   import soundfile

   soundfile.write("song.ogg", x.T, sr_file, format="OGG", subtype="VORBIS")
   assert raises(TypeError, lambda: aus.io.read("song.ogg"))

   data, rate = soundfile.read("song.ogg", dtype="float32", always_2d=True)
   ogg = aus.AudioSamples.new_multi(data.T, rate)
   assert ogg.num_channels == 2 and ogg.sample_rate == 44100

File objects (``io.BytesIO``) are not accepted either. Read the bytes with soundfile in the
same way.

To NumPy, PyTorch and scikit-learn
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # NumPy, either way, without a copy when the audio wraps a NumPy array.
   wrapped = aus.AudioSamples.new_mono(y, sr)
   assert np.shares_memory(wrapped.to_numpy(), y)

   # In-place methods therefore change the array you passed.
   private = aus.AudioSamples.new_mono(y.copy(), sr)
   assert not np.shares_memory(private.to_numpy(), y)

.. code-block:: python

   import torch

   tensor = torch.from_numpy(a_mfcc.copy())
   assert tensor.dtype == torch.float32 and tuple(tensor.shape) == a_mfcc.shape

.. code-block:: python

   from sklearn.preprocessing import StandardScaler

   scaled = StandardScaler().fit_transform(matrix)
   assert scaled.shape == matrix.shape

Migrating gradually
-------------------

Both libraries can live in one process, and they exchange plain arrays. ``import
audio_samples`` loads only its extension, so adding it costs little until it is used.

**What crossing the boundary costs.** Wrapping a NumPy array in an ``AudioSamples`` and
taking it back do not copy. Audio produced by a method (``resample``, ``to_mono``,
``time_stretch``) is held by Rust: ``to_numpy()`` then copies on each call, and
``numpy.asarray`` keeps one shared array. Feature methods return NumPy arrays that nothing
else refers to. Measured for 60 s of float32 (one machine, one core):

@@INTEROP@@

.. code-block:: python

   # librosa features from audio_samples audio, and the reverse.
   resampled = aus.io.read("song.wav", as_type=aus.SampleType.F32).to_mono().resample(22050, "high")
   librosa_on_ours = librosa.feature.spectral_flatness(y=np.asarray(resampled))
   ours_on_librosa = aus.AudioSamples.new_mono(y, sr).spectral_flatness_frames()

**A sensible order.**

#. Leave ``librosa.load`` for last if you have formats other than WAV and FLAC. Otherwise
   convert it first and write out the rate, mix-down and type; the rest of the code keeps
   working on the same array through ``np.asarray``.
#. Move leaf features with exact agreement next: STFT, mel, MFCC, spectral features, chroma
   from an STFT. Index ``[0]`` until the code downstream handles a channel axis.
#. Then rhythm (onsets, tempo, beats) and pitch, whose results are identical or nearly so
   given identical samples.
#. Then effects and constant-Q transforms, which go through the resampler or iterate,
   so expect smaller agreement.
#. Replace plotting last, or keep ``librosa.display``.

**Verifying a converted pipeline.** Run the old and new paths on a set of your own files and
compare each stage, feeding the new stage from the old one, so that a difference is pinned
to one step. Keep this as a test until librosa is gone.

.. code-block:: python

   def compare_stage(name, ours, theirs, tolerance):
       ours, theirs = np.asarray(ours), np.asarray(theirs)
       if ours.ndim == theirs.ndim + 1:
           ours = ours[0]
       if ours.shape != theirs.shape:
           raise AssertionError(f"{name}: shape {ours.shape} against {theirs.shape}")
       error = relative_error(ours, theirs)
       if error > tolerance:
           raise AssertionError(f"{name}: relative error {error:.3g} above {tolerance:.3g}")
       return error

   y_old, sr_old = librosa.load("song.wav")
   audio_new = aus.AudioSamples.new_mono(y_old, sr_old)              # the old samples
   compare_stage("stft", audio_new.stft_matrix(), librosa.stft(y_old), 1e-6)
   compare_stage("mel", audio_new.melspectrogram(), librosa.feature.melspectrogram(y=y_old, sr=sr_old), 1e-6)
   compare_stage("onsets", audio_new.onset_strength(), librosa.onset.onset_strength(y=y_old, sr=sr_old), 1e-5)

float32 comparisons bottom out around 1e-7 to 5e-6. To check an algorithm rather than float32
round-off, convert to float64 first and pass ``dtype=np.float64`` to librosa's mel and chroma
functions.

When to stay on librosa
-----------------------

Stay, or keep librosa alongside, if any of these matter to you. Each was checked on the
versions under `Evidence`_.

**Availability.** The API described here is unreleased, and has only been run on x86-64
Linux with CPython 3.13. The current PyPI release lacks it. There are no wheels for macOS or
Windows, and no continuous integration on those platforms in the repository. librosa is pure
Python with wheels for its compiled dependencies on all three.

**Maturity and breadth.** ``audio_samples`` is at version 0.2.4, classified as beta. It has
one main author. librosa has years of published use, a citation record, an example gallery
and many answers online. If you need a result your field will recognise as "computed with
librosa", that counts for more than speed.

**Install size of the wheel.** librosa's own wheel is pure Python, so it is smaller than this
library's compiled one. The dependencies it pulls in are a different matter.

@@FOOTPRINT@@

**File formats and sources.** WAV and FLAC only: no MP3, Ogg Vorbis, Opus or AIFF, and no file
objects. ``librosa.load`` takes all of these. The workaround is under `Other file formats`_.

**One silent type change.** ``io.save`` of float64 audio to a ``.wav`` file writes float32, also
with ``as_type=SampleType.F64``. It is the one place this library changes a sample type without
being asked. To keep float64 samples on disk, use another format or ``soundfile.write``.

**Functions with no equivalent:**

- ``librosa.display``: no ``specshow`` or formatters here, though it works on these arrays;
- ``feature.inverse``: ``mel_to_stft``, ``mel_to_audio``, ``mfcc_to_mel``, ``mfcc_to_audio``, and
  ``util.nnls``;
- ``decompose.decompose`` (NMF through scikit-learn);
- ``iirt``, ``fmt``, ``filters.semitone_filterbank``, ``filters.mr_frequencies``,
  ``filters.diagonal_filter``;
- ``segment.path_enhance``, ``segment.agglomerative``, ``segment.subsegment``,
  ``segment.timelag_filter``, ``sequence.path_to_steps``;
- notation and tuning systems: ``interval_frequencies``, ``pythagorean_intervals``,
  ``plimit_intervals``, ``interval_to_fjs``, ``hz_to_fjs``, ``fifths_to_note``, the svara, mela and
  thaat functions;
- ``blocks_to_frames``, ``blocks_to_samples``, ``blocks_to_time``;
- the example data (``ex``, ``example_info``, ``list_examples``), ``util.find_files``,
  ``cache`` and ``loadx``;
- helpers such as ``util.localmax``, ``util.match_events``, ``util.pad_center``,
  ``util.fix_frames``, ``util.axis_sort``, ``util.sparsify_rows``, ``util.shear``. Most are a
  NumPy line or two.

**Options that raise here.** Each was run and raises ``ValueError`` or ``TypeError``:

- ``sparse=True`` in ``recurrence_matrix`` and ``cross_similarity``, ``sparse=False`` in
  ``onset_detect`` and ``beat_track``;
- multichannel audio in ``onset_detect`` and ``beat_track``;
- ``intervals`` other than ``"equal"`` in ``vqt`` and ``chroma_vqt``;
- ``kind`` other than ``"linear"`` in ``interp_harmonics``, ``f0_harmonics``, ``salience``;
- ``init="random"`` in ``griffinlim_cqt``; a NumPy generator as ``rng`` in ``griffinlim``;
- a smoothing ``window`` in ``chroma_cqt`` and ``cq_to_chroma``;
- ``alpha`` in ``wavelet`` and ``wavelet_lengths``;
- a time-varying 2-D ``freq`` in ``spectral_centroid_from_spectrogram``;
- slices as ``channels`` in ``onset_strength_multi``, and as indices in ``sync``;
- custom ``freqs`` in ``tempogram_ratio``, ``aggregate=None`` in ``metrogram``;
- ``mask=True`` in ``harmonic``, ``percussive`` and ``harmonic_percussive`` (it is available in
  ``effects.hpss``);
- ``scale=True`` in ``pitch_shift``;
- a fractional ``power`` in ``magphase``, a 3-D matrix too (index ``[0]``);
- a callable other than ``numpy.max``, ``numpy.min`` or ``numpy.mean`` as ``aggregate`` in
  ``trim_db`` and ``split_db``; other than ``numpy.mean``, ``numpy.median`` or ``numpy.average`` in
  ``nn_filter``;
- fractional sample rates.

**Accepted, not applied.** ``griffinlim(dtype=...)`` is ignored, and ``griffinlim`` accepts a
``momentum`` above 1 silently where librosa warns.

**Where the implementations differ.** Anything downstream of soxr: if your results must match
librosa's ``resample``, ``pitch_shift`` or lower CQT octaves to better than 1e-4, keep librosa
for those steps.

**The API is split.** Older methods with the names you expect (``rms``, ``stft``, ``hpss``,
``trim``) coexist with the librosa-compatible ones under different names. See
:ref:`librosa-existing-apis`.
