Threads and free-threaded Python
================================

``audio_samples`` can be called from many Python threads, on a regular build of
CPython and on a free-threaded build (for example ``python3.14t``). Importing it on a
free-threaded build leaves the GIL off: the extension declares that it does not
need it, and ``sys._is_gil_enabled()`` stays ``False`` after the import.

Free-threaded support is tested on Linux x86-64 with CPython 3.14.2 and NumPy 2.5.4.
``tests/test_threads.py`` runs on every interpreter.

What you can do
---------------

* Call any read-only function or method from any number of threads, on the same
  or on different inputs. This includes one ``AudioSamples`` object or one NumPy
  array shared by every thread. The results are the same as on one thread.
* Read files, resample and analyse in a ``ThreadPoolExecutor``.
* Pass a Python callable as ``aggregate`` or ``prior``. It runs on the thread that made
  the call. It has to be safe to run at the same time as itself if several threads
  use it.

What raises an exception
------------------------

These are errors you can catch, not races:

* An in-place method (``scale``, ``apply_compressor``, ``butterworth_lowpass``, ...)
  needs the object to itself. If another thread is using the object, the
  in-place call raises ``RuntimeError: Already borrowed``, and a read that starts
  during an in-place call raises ``RuntimeError: Already mutably borrowed``. Neither
  waits for the other: guard the object with a lock if threads share it this way, or
  give each thread a copy. The same holds for other objects with state, such as a
  ``SosFilter``.
* ``SosFilter.process_samples_in_place`` and ``process_block`` hold a write borrow on the
  array they are given. A call that reads the same array on another thread raises
  ``TypeError: The given array is already borrowed``.
* ``StreamedAudioReader``, ``StreamedAudioWriter`` and ``WavSink`` can be used from any
  thread, one call at a time. Two calls at once on one of them raise the same
  ``RuntimeError`` as other objects with state.
* On a free-threaded build, an in-place method on an ``AudioSamples`` built on a NumPy array
  takes a write borrow on that array, and reads of it through any ``AudioSamples`` or
  array argument take a read borrow. The call that meets the other raises
  ``TypeError: The given array is already borrowed``. Each borrow goes through one lock shared
  by all arrays: on an efficiency core a call on such audio cost about 60 ns more, and calls that do
  almost nothing on 8 threads wait on the lock (``rms`` of 64 samples of one shared array, 3.4 us
  per call against 1.6 us without the borrow). With the GIL an in-place method runs
  with the GIL held, so a reader holding the GIL cannot overlap it, and these borrows
  are not taken.

What you must not do
--------------------

* Write to a NumPy array from Python, for example ``y *= 2``, while another thread
  passes it to this package. Neither NumPy nor this package detects that: the
  result of the read is undefined, as it is for a NumPy function in the same situation.
* Draw with ``audio_samples.mpl`` from several threads. Matplotlib is not thread-safe;
  compute in threads and plot on one.

Thread pools
------------

FFT plans and resampling kernels are cached per thread, so that threads do not
wait on each other. A new thread starts with empty caches, so keep one pool for a
batch rather than making a thread per file.

Scaling
-------

Measured on 2026-11-04 and 2026-11-05 on an Intel i7-13700KF under Linux, with audio_samples 0.2.4, CPython
3.14.2 (regular and free-threaded), NumPy 2.5.4, librosa 1.0.0, SciPy 1.18.1 and numba 0.68.1.
Native libraries were limited to one thread each, so the threads are all Python's. The process
was pinned to the eight efficiency cores (four CPUs in each of two clusters, each cluster
sharing 4 MB of L2), so 16 threads run on 8 CPUs. Each cell is the fastest of at least five
passes, after a warm-up pass; a pass waited for a quiet moment first. Other jobs were sharing the machine:
the last column gives how many CPUs of the eight they kept busy during the passes reported, so
a row with a high number is less reliable. Process lists taken during the runs showed compilers and test suites.

The script is ``benches/librosa_compare/thread_scaling.py`` in the ``audio_samples`` repository.
The MFCC job reads a 30 s, 44100 Hz, stereo, 16-bit WAV file (from the page cache), mixes to mono,
resamples to 22050 Hz and computes MFCCs. The STFT and resample jobs work on 60 s of float32
at 22050 Hz already in memory; resampling goes to 16000 Hz at each library's highest quality.

**MFCC of 128 file reads**, seconds for the batch (speed-up over one thread)

.. list-table::
   :header-rows: 1

   * - Library, Python
     - 1 thread
     - 2 threads
     - 4 threads
     - 8 threads
     - 16 threads
     - Others busy (CPUs)
   * - audio_samples, 3.14
     - 1.60 (1.0x)
     - 0.86 (1.9x)
     - 0.58 (2.8x)
     - 0.40 (4.0x)
     - 0.38 (4.2x)
     - 0.0 to 0.1
   * - audio_samples, 3.14t
     - 1.64 (1.0x)
     - 0.86 (1.9x)
     - 0.57 (2.9x)
     - 0.47 (3.5x)
     - 0.41 (4.0x)
     - 0.0 to 0.1
   * - librosa, 3.14
     - 6.87 (1.0x)
     - 3.58 (1.9x)
     - 2.19 (3.1x)
     - 1.58 (4.4x)
     - 1.79 (3.9x)
     - 0.0 to 1.0
   * - librosa, 3.14t
     - 7.66 (1.0x)
     - 3.80 (2.0x)
     - 2.15 (3.6x)
     - 1.55 (5.0x)
     - 1.66 (4.6x)
     - 0.1 to 2.6

**STFT of 128 arrays**, seconds for the batch (speed-up over one thread)

.. list-table::
   :header-rows: 1

   * - Library, Python
     - 1 thread
     - 2 threads
     - 4 threads
     - 8 threads
     - 16 threads
     - Others busy (CPUs)
   * - audio_samples, 3.14
     - 1.59 (1.0x)
     - 0.81 (2.0x)
     - 0.43 (3.7x)
     - 0.31 (5.1x)
     - 0.34 (4.7x)
     - 0.0 to 0.1
   * - audio_samples, 3.14t
     - 1.58 (1.0x)
     - 0.81 (2.0x)
     - 0.43 (3.7x)
     - 0.32 (4.9x)
     - 0.33 (4.9x)
     - 0.1 to 0.5
   * - librosa, 3.14
     - 3.39 (1.0x)
     - 1.96 (1.7x)
     - 1.34 (2.5x)
     - 1.27 (2.7x)
     - 1.35 (2.5x)
     - 0.0 to 1.1
   * - librosa, 3.14t
     - 3.43 (1.0x)
     - 1.94 (1.8x)
     - 1.32 (2.6x)
     - 1.22 (2.8x)
     - 1.15 (3.0x)
     - 0.1 to 2.0

**Resample 128 arrays**, seconds for the batch (speed-up over one thread)

.. list-table::
   :header-rows: 1

   * - Library, Python
     - 1 thread
     - 2 threads
     - 4 threads
     - 8 threads
     - 16 threads
     - Others busy (CPUs)
   * - audio_samples, 3.14
     - 1.10 (1.0x)
     - 0.55 (2.0x)
     - 0.28 (3.9x)
     - 0.16 (6.8x)
     - 0.17 (6.6x)
     - 0.0 to 0.2
   * - audio_samples, 3.14t
     - 1.11 (1.0x)
     - 0.55 (2.0x)
     - 0.29 (3.9x)
     - 0.16 (6.9x)
     - 0.16 (6.7x)
     - 0.0 to 0.1
   * - librosa, 3.14
     - 1.41 (1.0x)
     - 0.71 (2.0x)
     - 0.39 (3.7x)
     - 0.28 (5.0x)
     - 0.26 (5.3x)
     - 0.0 to 1.0
   * - librosa, 3.14t
     - 1.47 (1.0x)
     - 0.76 (1.9x)
     - 0.44 (3.3x)
     - 0.31 (4.7x)
     - 0.32 (4.6x)
     - 0.0 to 0.2

What this shows:

* For ``audio_samples`` the free-threaded build gives no extra scaling over the regular build,
  because these calls already run without the GIL. One-thread times are within 4 %.
* Scaling is close to linear up to 4 threads for resampling and stops at about 4 to 7
  times on these 8 cores. More threads than CPUs do not help.
* The limit for MFCC is the memory system, not a lock. With ``perf stat`` on 3.14t, 8 threads
  executed the same number of instructions as one (114 billion) but took 68 % more cycles, with
  more last-level cache misses. System time was about 10 % of CPU time. Resampling, with
  a small working set, scales furthest.
* Rayon is not used on these paths. glibc's allocator returns large buffers to the kernel, which costs page
  faults: setting ``MALLOC_MMAP_THRESHOLD_=4294967296``, ``MALLOC_TRIM_THRESHOLD_=4294967296`` and
  ``MALLOC_TOP_PAD_=268435456`` took the 8-thread STFT batch from 0.31 s to 0.28 s and left MFCC at 0.44 to
  0.45 s. The package does not change these process-wide settings.
* With 16 threads on 14 logical CPUs (efficiency cores plus three performance cores with
  hyper-threading), measured before the MFCC path was reduced, the batches took: MFCC
  ``audio_samples`` 0.35 s on 3.14 and 0.39 s on 3.14t, librosa 1.37 s and 1.28 s; STFT ``audio_samples`` 0.28 s and 0.30 s, librosa 0.70 s and 0.52 s; resample ``audio_samples``
  0.08 s and 0.09 s, librosa 0.20 s and 0.20 s.
* librosa 1.0.0 installs from wheels on 3.14t, including numba and llvmlite, and does not turn the GIL
  back on. Its 8-thread MFCC batch took 1.58 s on 3.14 and 1.55 s on 3.14t.
