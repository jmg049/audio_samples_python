"""Concurrent use of the bindings from many Python threads.

These tests run on every interpreter. On a regular build the threads overlap
wherever the bindings release the GIL; on a free-threaded build (3.14t) they
overlap everywhere, and the module must not switch the GIL back on.

Each worker compares its results byte for byte with the results of the same
call made on one thread beforehand, so a corrupted cache or a shared scratch
buffer shows up as a wrong answer, not only as a crash.

``AUDIO_SAMPLES_THREAD_STRESS_SECONDS`` lengthens the mixed workload (default:
a few rounds, about a second).
"""

from __future__ import annotations

import os
import subprocess
import sys
import sysconfig
import textwrap
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

import audio_samples as aus

N_THREADS = 8
STRESS_SECONDS = float(os.environ.get("AUDIO_SAMPLES_THREAD_STRESS_SECONDS", "0"))

free_threaded = bool(sysconfig.get_config_var("Py_GIL_DISABLED"))


def run_threads(fn, n=N_THREADS):
    """Run ``fn(thread_index)`` on ``n`` threads released together; return the results."""
    barrier = threading.Barrier(n)

    def task(i):
        barrier.wait()
        return fn(i)

    with ThreadPoolExecutor(n) as pool:
        return [f.result() for f in [pool.submit(task, i) for i in range(n)]]


def signal(seed=0, seconds=1.0, sr=22050, dtype=np.float32):
    t = np.arange(int(seconds * sr), dtype=np.float64) / sr
    rng = np.random.default_rng(seed)
    f0 = 220.0 * (1 + seed % 5)
    y = 0.5 * np.sin(2 * np.pi * f0 * t) + 0.05 * rng.standard_normal(t.size)
    y[:: sr // 2] += 0.5
    return y.astype(dtype)


def digest(value):
    """Comparable bytes for a result: arrays, tuples, lists, numbers, AudioSamples."""
    if isinstance(value, aus.AudioSamples):
        return ("audio", value.sample_rate, digest(value.to_numpy()))
    if isinstance(value, np.ndarray):
        return (value.dtype.str, value.shape, value.tobytes())
    if isinstance(value, (tuple, list)):
        return tuple(digest(v) for v in value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def array_calls(y):
    """Calls on a NumPy array across the function modules."""
    s = np.abs(aus.spectral.stft(y)) ** 2
    mel_db = aus.spectral.power_to_db(aus.spectral.melspectrogram_from_spectrogram(s, sr=22050))
    env = aus.rhythm.onset_strength(S=mel_db, sr=22050)
    chroma = aus.features.chroma_stft_from_spectrogram(s, sr=22050)
    c = aus.sequence.recurrence_matrix(chroma[..., :64], k=3)
    return {
        "stft": lambda: aus.spectral.stft(y, n_fft=1024, hop_length=256),
        "istft": lambda: aus.spectral.istft(aus.spectral.stft(y)),
        "power_to_db": lambda: aus.spectral.power_to_db(s),
        "frame": lambda: aus.spectral.frame(y, frame_length=512, hop_length=128),
        "frame_copy": lambda: aus.spectral.frame(y, frame_length=512, hop_length=128, copy=True),
        "mel": lambda: aus.spectral.melspectrogram_from_spectrogram(s, sr=22050),
        "delta": lambda: aus.spectral.delta(s[:16]),
        "window_bandwidth": lambda: aus.spectral.window_bandwidth("hann", 1024),
        "get_window": lambda: aus.spectral.get_window("hann", 1024),
        "centroid": lambda: aus.features.spectral_centroid_from_spectrogram(s, sr=22050),
        "contrast": lambda: aus.features.spectral_contrast_from_spectrogram(s, sr=22050),
        "pcen": lambda: aus.features.pcen(s, sr=22050, hop_length=512),
        "yin": lambda: aus.features.yin(y, fmin=65.0, fmax=2093.0, sr=22050),
        "pyin": lambda: aus.features.pyin(y[:11025], fmin=65.0, fmax=1000.0, sr=22050),
        "onset_strength": lambda: aus.rhythm.onset_strength(S=mel_db, sr=22050),
        "peak_pick": lambda: aus.rhythm.peak_pick(env, pre_max=3, post_max=3, pre_avg=3, post_avg=5, delta=0.1, wait=10),
        "tempogram": lambda: aus.rhythm.tempogram(onset_envelope=env, sr=22050),
        "tempo": lambda: aus.rhythm.tempo(onset_envelope=env, sr=22050),
        "beat_track": lambda: aus.rhythm.beat_track(onset_envelope=env, sr=22050),
        "hpss": lambda: aus.effects.hpss(np.abs(aus.spectral.stft(y))),
        "griffinlim": lambda: aus.effects.griffinlim(np.abs(aus.spectral.stft(y[:4096])), n_iter=4, init=None),
        "lpc": lambda: aus.effects.lpc(y, order=12),
        "phase_vocoder": lambda: aus.effects.phase_vocoder(aus.spectral.stft(y), rate=1.3),
        "dtw": lambda: aus.sequence.dtw(chroma[..., :40], chroma[..., 10:50])[0],
        "recurrence": lambda: aus.sequence.recurrence_to_lag(c),
        "viterbi": lambda: aus.sequence.viterbi(
            np.full((3, 40), 1.0 / 3), aus.sequence.transition_uniform(3)
        ),
    }


def audio_calls(a):
    """Read-only methods on an ``AudioSamples``."""
    return {
        "rms": a.rms,
        "stats": lambda: (a.peak(), a.mean(), a.variance(), a.zero_crossings()),
        "stft_matrix": lambda: a.stft_matrix(),
        "melspectrogram": lambda: a.melspectrogram(),
        "mfcc_frames": lambda: a.mfcc_frames(),
        "chroma_stft": lambda: a.chroma_stft(),
        "cqt": lambda: a.cqt(),
        "resample": lambda: a.resample(16000, "high"),
        "resample_fast": lambda: a.resample(16000, "fast"),
        "to_numpy": lambda: a.to_numpy(),
        "copy": lambda: a.copy(),
        "scaled_copy": lambda: a * 0.5,
        "pitch_shift": lambda: a.pitch_shift(2.0),
        "time_stretch": lambda: a.time_stretch(1.25),
        "yin": lambda: a.yin(fmin=65.0, fmax=2093.0),
        "beat_track": lambda: a.beat_track(),
        "onset_strength": lambda: a.onset_strength(),
        "harmonic": lambda: a.harmonic(),
    }


def test_import_keeps_the_gil_off():
    """Importing the package and its lazy parts leaves the GIL as it was."""
    code = textwrap.dedent(
        """
        import sys
        before = sys._is_gil_enabled() if hasattr(sys, "_is_gil_enabled") else True
        import audio_samples as aus
        aus.spectrograms, aus.generation, aus.io, aus.utils, aus.spectral, aus.sequence
        aus.features, aus.rhythm, aus.effects, aus.explain, aus.__version__
        after = sys._is_gil_enabled() if hasattr(sys, "_is_gil_enabled") else True
        assert after == before, (before, after)
        """
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert "RuntimeWarning" not in out.stderr, out.stderr
    assert "GIL" not in out.stderr, out.stderr


def test_concurrent_lazy_attributes_and_first_import():
    """The lazy module attributes and the first import race cleanly."""
    code = textwrap.dedent(
        """
        import threading
        barrier = threading.Barrier(8)
        results, errors = [], []
        def work():
            barrier.wait()
            try:
                import audio_samples as aus
                results.append((aus.__version__, aus.mpl is None, aus.SampleType.F32, id(aus)))
            except BaseException as e:
                errors.append(repr(e))
        threads = [threading.Thread(target=work) for _ in range(8)]
        [t.start() for t in threads]
        [t.join() for t in threads]
        assert not errors, errors
        assert len(set(map(str, results))) == 1, results
        """
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr


def _compare_concurrently(calls, rounds=2):
    expected = {name: digest(fn()) for name, fn in calls.items()}
    names = sorted(calls)

    def work(i):
        bad = []
        # Each thread walks the calls from a different start, so different
        # calls overlap in each round.
        for r in range(rounds):
            for k in range(len(names)):
                name = names[(k + i * 3 + r) % len(names)]
                if digest(calls[name]()) != expected[name]:
                    bad.append(name)
        return bad

    return [name for bad in run_threads(work) for name in bad]


def test_shared_numpy_input_gives_identical_results():
    """Many threads use one float32 array and one float64 array."""
    for dtype in (np.float32, np.float64):
        y = signal(dtype=dtype, seconds=0.5)
        assert _compare_concurrently(array_calls(y)) == []


def test_separate_numpy_inputs_give_identical_results():
    """Each thread has its own array, so thread-local caches see different sizes."""
    inputs = [signal(seed=i, seconds=0.25 + 0.05 * i) for i in range(N_THREADS)]
    expected = [digest(aus.spectral.stft(y, n_fft=512 + 64 * i)) for i, y in enumerate(inputs)]

    def work(i):
        for _ in range(10):
            if digest(aus.spectral.stft(inputs[i], n_fft=512 + 64 * i)) != expected[i]:
                return False
        return True

    assert all(run_threads(work))


def test_shared_audio_object_read_methods():
    """One ``AudioSamples``, owned or wrapping a NumPy array, read from many threads."""
    y = signal(seconds=0.5)
    wrapped = aus.AudioSamples.new_mono(y, 22050)
    owned = wrapped.copy()
    assert _compare_concurrently(audio_calls(wrapped)) == []
    assert _compare_concurrently(audio_calls(owned)) == []


def test_shared_filter_design():
    """One design object used by every thread; each filters its own signal."""
    design = aus.IirFilterDesign.butterworth_lowpass(4, 2000.0)
    y = signal(seconds=0.25)

    def filtered():
        a = aus.AudioSamples.new_mono(y.copy(), 22050)
        a.apply_iir_filter(design)
        return a.to_numpy()

    expected = digest(filtered())

    def work(_):
        return all(digest(filtered()) == expected for _ in range(20))

    assert all(run_threads(work))


def test_in_place_method_on_a_shared_object_fails_cleanly():
    """An in-place call that meets another call on the object raises ``RuntimeError``.

    The other thread's computation is not disturbed: every read that does
    return a result gets either the data before or after a whole in-place call.
    """
    y = np.ones(22050, dtype=np.float32)
    a = aus.AudioSamples.new_mono(y.copy(), 22050)
    stop = threading.Event()
    seen = {"writes": 0, "busy": 0, "reads": 0}
    lock = threading.Lock()

    def writer():
        while not stop.is_set():
            try:
                a.scale(1.0)
                with lock:
                    seen["writes"] += 1
            except RuntimeError as e:
                assert "borrowed" in str(e).lower(), e
                with lock:
                    seen["busy"] += 1

    def reader():
        while not stop.is_set():
            try:
                peak = a.stft_matrix().size
                assert peak > 0
                with lock:
                    seen["reads"] += 1
            except RuntimeError as e:
                assert "borrowed" in str(e).lower(), e

    errors = []

    def guard(t):
        def run():
            try:
                t()
            except BaseException as e:  # noqa: BLE001 - a PanicException must fail the test
                errors.append(e)

        return run

    threads = [threading.Thread(target=guard(t)) for t in [writer] * 2 + [reader] * 4]
    [t.start() for t in threads]
    time.sleep(0.5)
    stop.set()
    [t.join() for t in threads]
    assert not errors, errors
    assert seen["reads"] > 0
    assert seen["writes"] + seen["busy"] > 0
    assert np.array_equal(a.to_numpy(), y)


def test_array_written_in_place_while_read_never_panics():
    """A filter writes an array in place while other threads read it.

    The writer holds a NumPy borrow. A reader that meets it gets a Python
    exception that names the borrow, and a reader that does not gets a result.
    Neither side may panic. The same goes for a second writer.
    """
    y = signal(seconds=0.5, dtype=np.float64)
    design = aus.IirFilterDesign.butterworth_lowpass(2, 1000.0)
    errors = []
    stop = threading.Event()
    reads = [
        lambda: aus.spectral.stft(y),
        lambda: aus.spectral.frame(y, frame_length=512, hop_length=128),
        lambda: aus.features.yin(y, fmin=65.0, fmax=2093.0, sr=22050),
        lambda: aus.rhythm.peak_pick(y, pre_max=3, post_max=3, pre_avg=3, post_avg=5, delta=0.1, wait=10),
        lambda: aus.effects.lpc(y, order=4),
    ]

    def guarded(body):
        def run():
            try:
                body()
            except BaseException as e:  # noqa: BLE001 - a PanicException must fail the test
                errors.append(e)

        return run

    def reader():
        while not stop.is_set():
            for read in reads:
                try:
                    read()
                except TypeError as e:
                    assert "borrowed" in str(e).lower(), e

    def writer():
        filt = aus.SosFilter.from_design(design, 22050.0)
        while not stop.is_set():
            try:
                filt.process_samples_in_place(y)
            except (TypeError, ValueError) as e:
                assert "borrowed" in str(e).lower(), e

    threads = [threading.Thread(target=guarded(reader)) for _ in range(4)]
    threads += [threading.Thread(target=guarded(writer)) for _ in range(2)]
    [t.start() for t in threads]
    time.sleep(0.5)
    stop.set()
    [t.join() for t in threads]
    assert not errors, errors


def test_python_callbacks_stay_with_their_thread():
    """A Python ``aggregate`` runs on the calling thread and belongs to that call."""
    y = signal(seconds=1.0)
    a = aus.AudioSamples.new_mono(y, 22050)

    def work(i):
        calls = []

        def aggregate(values, axis=-1):
            calls.append(threading.get_ident())
            return np.mean(values, axis=axis)

        r = a.onset_strength(aggregate=aggregate)
        assert calls and set(calls) == {threading.get_ident()}

        def boom(values, axis=-1):
            raise KeyError(i)

        with pytest.raises(KeyError) as exc:
            a.onset_strength(aggregate=boom)
        assert exc.value.args == (i,)
        return digest(r)

    results = run_threads(work)
    assert len(set(results)) == 1


def test_first_use_of_static_caches_race():
    """``key_to_notes`` and the kernel, window and filterbank caches on first use."""
    code = textwrap.dedent(
        """
        import threading
        import numpy as np
        import audio_samples as aus
        keys = ["C:maj", "F#:min", "Bb:dorian", "Eb:maj", "A:min", "G:mixolydian"]
        barrier = threading.Barrier(8)
        out, errors = [], []
        def work(i):
            barrier.wait()
            try:
                out.append(tuple(tuple(aus.utils.key_to_notes(k)) for k in keys))
                out.append(tuple(tuple(aus.utils.key_to_notes(k, unicode=False)) for k in keys))
                aus.spectral.window_bandwidth("hann", 512 + i)
                aus.spectral.mel_filterbank(sr=22050, n_fft=512)
                a = aus.AudioSamples.new_mono(np.ones(4410, dtype=np.float32), 22050)
                a.resample(16000, "high")
            except BaseException as e:
                errors.append(repr(e))
        threads = [threading.Thread(target=work, args=(i,)) for i in range(8)]
        [t.start() for t in threads]
        [t.join() for t in threads]
        assert not errors, errors
        assert len(set(out)) == 2, out
        """
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr


def test_short_lived_threads_release_thread_local_state():
    """Threads that exit with cached state (``window_bandwidth``, FFT plans) leave no crash behind."""
    name = "hann"
    y = signal(seconds=0.1)
    expected = digest(aus.spectral.stft(y))
    results = []

    def work():
        results.append(
            (digest(aus.spectral.stft(y)) == expected, aus.spectral.window_bandwidth(name, 1024))
        )

    for _ in range(6):
        ts = [threading.Thread(target=work) for _ in range(8)]
        [t.start() for t in ts]
        [t.join() for t in ts]
    assert len(results) == 48
    assert all(ok for ok, _ in results)
    assert len({bw for _, bw in results}) == 1


def test_reads_and_resampling_at_once(tmp_path):
    """File reads, ``read_and_resample``, writes and streaming on many threads."""
    wav = tmp_path / "a.wav"
    flac = tmp_path / "a.flac"
    source = aus.AudioSamples.new_mono(signal(seconds=1.0), 22050).as_i16()
    aus.io.save(str(wav), source)
    aus.io.save(str(flac), source)
    expected = {
        "wav": digest(aus.io.read(str(wav), as_type=aus.SampleType.F32)),
        "flac": digest(aus.io.read(str(flac), as_type=aus.SampleType.F32)),
        "resampled": digest(aus.io.read_and_resample(str(wav), 16000, as_type=aus.SampleType.F32)),
    }

    def work(i):
        ok = []
        for r in range(3):
            ok.append(digest(aus.io.read(str(wav), as_type=aus.SampleType.F32)) == expected["wav"])
            ok.append(digest(aus.io.read(str(flac), as_type=aus.SampleType.F32)) == expected["flac"])
            ok.append(
                digest(aus.io.read_and_resample(str(wav), 16000, as_type=aus.SampleType.F32))
                == expected["resampled"]
            )
            out = tmp_path / f"{i}_{r}.wav"
            aus.io.save(str(out), source)
            ok.append(digest(aus.io.read(str(out))) == digest(aus.io.read(str(wav))))
            with aus.io.StreamedAudioReader(str(wav)) as reader:
                ok.append(reader.sample_rate == 22050)
                ok.append(reader.read_frames(1024).shape[0] == 1024)
        return all(ok)

    assert all(run_threads(work))


def test_streaming_objects_work_on_another_thread(tmp_path):
    """A reader, writer or sink made on one thread can be used and closed on another."""
    path = tmp_path / "a.wav"
    audio = aus.AudioSamples.new_mono(signal(seconds=0.1), 22050)
    aus.io.save(str(path), audio)
    reader = aus.io.StreamedAudioReader(str(path))
    writer = aus.io.StreamedAudioWriter(str(tmp_path / "b.wav"), 1, 22050, dtype=aus.SampleType.F32)
    results = []

    def other():
        block = reader.read_frames(16)
        writer.write_frames(block)
        writer.close()
        results.append(block.shape)

    t = threading.Thread(target=other)
    t.start()
    t.join()
    assert results, "the other thread raised"
    assert reader.read_frames(16) is not None
    assert aus.io.info(str(tmp_path / "b.wav")).num_samples == 16


def test_shared_reader_serialises_or_raises_a_normal_exception(tmp_path):
    """Concurrent calls on one reader raise RuntimeError at worst, and never lose a frame."""
    path = tmp_path / "a.wav"
    aus.io.save(str(path), aus.AudioSamples.new_mono(signal(seconds=1.0), 22050))
    reader = aus.io.StreamedAudioReader(str(path))
    lock = threading.Lock()
    frames = [0]

    def work(_):
        while True:
            try:
                block = reader.read_frames(64)
            except RuntimeError as e:
                assert "borrowed" in str(e).lower(), e
                continue
            if block is None:
                return True
            with lock:
                frames[0] += block.shape[-1]

    assert all(run_threads(work))
    assert frames[0] == 22050


def test_in_place_write_on_a_wrapped_array_meets_readers():
    """An in-place method on audio wrapping an array, against readers of that array.

    The write and the reads go through different objects, so only the array's
    borrow connects them. A reader gets a result or a TypeError that names the
    borrow, and without a GIL the two do meet.
    """
    y = np.ones(400_000, dtype=np.float32)
    writer_audio = aus.AudioSamples.new_mono(y, 22050)
    reader_audio = aus.AudioSamples.new_mono(y, 22050)
    stop = threading.Event()
    errors = []
    met = [0]

    def guarded(body):
        def run():
            try:
                body()
            except BaseException as e:  # noqa: BLE001 - a PanicException must fail the test
                errors.append(e)

        return run

    def reader():
        while not stop.is_set():
            for read in (reader_audio.rms, lambda: aus.spectral.frame(y, frame_length=64, hop_length=64)):
                try:
                    read()
                except TypeError as e:
                    assert "borrowed" in str(e).lower(), e
                    met[0] += 1

    def writer():
        while not stop.is_set():
            try:
                writer_audio.butterworth_lowpass(4, 2000.0)
            except TypeError as e:
                assert "borrowed" in str(e).lower(), e
                met[0] += 1

    threads = [threading.Thread(target=guarded(reader)) for _ in range(3)]
    threads.append(threading.Thread(target=guarded(writer)))
    [t.start() for t in threads]
    time.sleep(1.0)
    stop.set()
    [t.join() for t in threads]
    assert not errors, errors
    if free_threaded:
        assert met[0] > 0, "the write and the reads never met"


@pytest.mark.skipif(not free_threaded, reason="needs a free-threaded interpreter")
def test_gil_stays_off_in_this_process():
    """Not even the numerical dependencies used by this test file turn it on."""
    assert sys._is_gil_enabled() is False


def test_mixed_workload():
    """Mixed calls on shared and separate inputs, reads and resampling, for a while."""
    y = signal(seconds=0.5)
    shared = aus.AudioSamples.new_mono(y, 22050)
    calls = {("a", k): v for k, v in array_calls(y).items()}
    calls.update({("o", k): v for k, v in audio_calls(shared).items()})
    expected = {k: digest(fn()) for k, fn in calls.items()}
    keys = sorted(calls)
    deadline = time.monotonic() + STRESS_SECONDS

    def work(i):
        rng = np.random.default_rng(i)
        mine = aus.AudioSamples.new_mono(signal(seed=i, seconds=0.25), 22050)
        mine_stft = digest(mine.stft_matrix())
        bad = []
        count = 0
        while True:
            key = keys[int(rng.integers(len(keys)))]
            if digest(calls[key]()) != expected[key]:
                bad.append(key)
            if digest(mine.stft_matrix()) != mine_stft:
                bad.append("own stft")
            count += 1
            if count >= 12 and time.monotonic() > deadline:
                return bad

    assert [k for bad in run_threads(work, 16) for k in bad] == []
