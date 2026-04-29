"""
profile_io.py — Profiling harness for WAV load/save comparison.

Usage:
    # Quick comparison for the default scenario (60s stereo i16):
    python profile_io.py --compare

    # Sweep across multiple scenarios (durations, dtypes, channel counts):
    python profile_io.py --sweep

    # Profile a single library in a tight loop (intended for py-spy):
    python profile_io.py --lib audio_samples --op load
    python profile_io.py --lib librosa --op load

    # Generate flamegraph (requires py-spy: pip install py-spy):
    py-spy record -o flamegraph_aus_load.svg -- python profile_io.py --lib audio_samples --op load
    py-spy record --native -o flamegraph_aus_load_native.svg -- python profile_io.py --lib audio_samples --op load

    # Use your own WAV file:
    python profile_io.py --compare --wav /path/to/your.wav

Notes on the soundfile baseline:
    soundfile's default output dtype for PCM files is float64, which allocates 2x the
    memory and does more conversion work than librosa (which returns float32).  The
    benchmarks use dtype='float32' for soundfile so both are measured on equal footing.
"""

from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path

import numpy as np
import scipy.io.wavfile
import soundfile as sf
import librosa

import audio_samples as aus

# ---------------------------------------------------------------------------
# Default single-scenario settings
# ---------------------------------------------------------------------------
SAMPLE_RATE = 44100
DTYPE = "i16"
DURATION_S = 60.0
CHANNELS = 2

DEFAULT_ITERS = 2000
WARMUP_ITERS = 50

# ---------------------------------------------------------------------------
# Sweep scenarios: (sample_rate, dtype, duration_s, channels, label)
# ---------------------------------------------------------------------------
SWEEP_SCENARIOS = [
    (44100, "i16",  1.0,  1, "mono  1s  i16"),
    (44100, "i16",  5.0,  1, "mono  5s  i16"),
    (44100, "i16", 30.0,  1, "mono 30s  i16"),
    (44100, "i16",  5.0,  2, "stereo 5s  i16"),
    (44100, "i16", 30.0,  2, "stereo 30s i16"),
    (44100, "i16", 60.0,  2, "stereo 60s i16"),
    (44100, "f32",  5.0,  1, "mono  5s  f32"),
    (44100, "f32", 30.0,  2, "stereo 30s f32"),
    (48000, "i16", 30.0,  2, "stereo 30s i16 48k"),
]

SWEEP_ITERS = 2000
SWEEP_WARMUP = 50


# ---------------------------------------------------------------------------
# WAV file helpers
# ---------------------------------------------------------------------------

def _wav_path(sr: int, dtype: str, dur: float, ch: int) -> str:
    return f"/tmp/profile_io_{sr}_{dtype}_{dur:.0f}s_{ch}ch.wav"


def make_wav(path: str, sr: int, dtype: str, dur: float, ch: int) -> None:
    np_dtype = np.dtype("int16") if dtype == "i16" else np.dtype("float32")
    if ch == 1:
        audio = aus.sine_wave(440.0, dur, sr, dtype=np_dtype)
    else:
        left  = aus.sine_wave(440.0, dur, sr, dtype=np_dtype)
        right = aus.sine_wave(880.0, dur, sr, dtype=np_dtype)
        audio = aus.AudioSamples.stack([left, right])
    aus.io.save(path, audio)
    size_mb = Path(path).stat().st_size / (1024 * 1024)
    print(f"  Written: {path}  ({size_mb:.1f} MB)")


def ensure_wav(sr: int, dtype: str, dur: float, ch: int) -> str:
    path = _wav_path(sr, dtype, dur, ch)
    if not Path(path).exists():
        make_wav(path, sr, dtype, dur, ch)
    return path


# ---------------------------------------------------------------------------
# Load functions
# ---------------------------------------------------------------------------

def load_aus(path: str) -> None:
    aus.io.read(path)


def load_librosa(path: str) -> None:
    librosa.load(path, sr=None, mono=False, dtype=np.float32)


def load_scipy(path: str) -> None:
    scipy.io.wavfile.read(path)


def load_sf(path: str) -> None:
    # dtype='float32' matches librosa's output — fair comparison.
    # soundfile's default (float64) would allocate 2x memory and convert more,
    # making it look slower than librosa even though librosa uses soundfile internally.
    sf.read(path, always_2d=True, dtype="float32")


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------

def _build_save_payloads(path: str):
    aus_audio = aus.io.read(path)
    raw = aus_audio.to_numpy()
    sf_data = raw.T.astype(np.int16)
    return aus_audio, sf_data


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

def timed_loop(fn, n_iters: int, warmup: int) -> list[float]:
    for _ in range(warmup):
        fn()
    gc.collect()
    times: list[float] = []
    for _ in range(n_iters):
        gc.disable()
        try:
            t0 = time.perf_counter()
            fn()
            t1 = time.perf_counter()
        finally:
            gc.enable()
        times.append(t1 - t0)
    return times


def stats(times: list[float]) -> dict[str, float]:
    s = sorted(times)
    n = len(s)
    return {
        "mean": sum(s) / n * 1e3,
        "p50":  s[n // 2] * 1e3,
        "p90":  s[int(0.9 * n)] * 1e3,
        "min":  s[0] * 1e3,
    }


def print_stats(label: str, times: list[float]) -> None:
    st = stats(times)
    print(f"  {label:<20s}  mean={st['mean']:.3f}ms  p50={st['p50']:.3f}ms  "
          f"p90={st['p90']:.3f}ms  min={st['min']:.3f}ms")


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

def run_single(lib: str, op: str, wav_path: str, n_iters: int) -> None:
    print(f"\n[{lib}] {op}  —  {n_iters} iters  (warmup={WARMUP_ITERS})")
    print(f"  File: {wav_path}")

    if op == "load":
        loaders = {
            "audio_samples": load_aus,
            "librosa":        load_librosa,
            "scipy":          load_scipy,
            "soundfile":      load_sf,
        }
        fn = loaders.get(lib)
        if fn is None:
            raise ValueError(f"Unknown lib: {lib}")
        times = timed_loop(lambda: fn(wav_path), n_iters, WARMUP_ITERS)
        print_stats(f"{lib} load", times)

    elif op == "save":
        aus_audio, sf_data = _build_save_payloads(wav_path)
        save_path = wav_path.replace(".wav", f"_{lib}_out.wav")
        if lib == "audio_samples":
            times = timed_loop(lambda: aus.io.save(save_path, aus_audio), n_iters, WARMUP_ITERS)
        elif lib in ("librosa", "soundfile"):
            times = timed_loop(lambda: sf.write(save_path, sf_data, SAMPLE_RATE), n_iters, WARMUP_ITERS)
        elif lib == "scipy":
            times = timed_loop(lambda: scipy.io.wavfile.write(save_path, SAMPLE_RATE, sf_data), n_iters, WARMUP_ITERS)
        else:
            raise ValueError(f"Unknown lib: {lib}")
        print_stats(f"{lib} save", times)

    else:
        raise ValueError(f"Unknown op: {op}")


def run_compare(wav_path: str, n_iters: int) -> None:
    print(f"\nComparison: sr={SAMPLE_RATE} {DTYPE} {DURATION_S:.0f}s {'stereo' if CHANNELS == 2 else 'mono'}")
    print(f"  File: {wav_path}")
    print(f"  Iters: {n_iters}  warmup: {WARMUP_ITERS}\n")

    aus_audio, sf_data = _build_save_payloads(wav_path)
    save_path = wav_path.replace(".wav", "_out.wav")

    load_cases = [
        ("audio_samples", lambda: load_aus(wav_path)),
        ("librosa",        lambda: load_librosa(wav_path)),
        ("scipy",          lambda: load_scipy(wav_path)),
        ("soundfile",      lambda: load_sf(wav_path)),
    ]
    save_cases = [
        ("audio_samples", lambda: aus.io.save(save_path, aus_audio)),
        ("librosa/sf",     lambda: sf.write(save_path, sf_data, SAMPLE_RATE)),
        ("scipy",          lambda: scipy.io.wavfile.write(save_path, SAMPLE_RATE, sf_data)),
    ]

    print("LOAD:")
    for label, fn in load_cases:
        times = timed_loop(fn, n_iters, WARMUP_ITERS)
        print_stats(label, times)

    print("\nSAVE:")
    for label, fn in save_cases:
        times = timed_loop(fn, n_iters, WARMUP_ITERS)
        print_stats(label, times)


def run_sweep(n_iters: int, warmup: int) -> None:
    print(f"\nSweep — {n_iters} iters, {warmup} warmup  (p50 ms)\n")

    paths = [ensure_wav(sr, dtype, dur, ch) for sr, dtype, dur, ch, _ in SWEEP_SCENARIOS]

    # Global pre-warmup: one pass through every scenario for every library.
    # This warms glibc's mmap allocator for all file sizes before measuring any of them.
    # Without this, early-in-the-sweep scenarios pay a cold-allocator tax (~5x slower)
    # that later scenarios avoid because the heap is warmed by prior iterations.
    print("Pre-warming allocator across all scenarios... ", end="", flush=True)
    GLOBAL_WARMUP = 100
    for _ in range(GLOBAL_WARMUP):
        for path in paths:
            load_aus(path)
            load_scipy(path)
    print("done\n")

    col = 10
    load_header = f"{'Scenario':<22} {'aus':>{col}} {'librosa':>{col}} {'scipy':>{col}} {'soundfile':>{col}}  aus/lib  aus/scipy"
    print("LOAD:")
    print(load_header)
    print("-" * len(load_header))

    for (_, _, _, _, label), path in zip(SWEEP_SCENARIOS, paths):
        aus_t  = stats(timed_loop(lambda p=path: load_aus(p),     n_iters, warmup))["p50"]
        lib_t  = stats(timed_loop(lambda p=path: load_librosa(p), n_iters, warmup))["p50"]
        scp_t  = stats(timed_loop(lambda p=path: load_scipy(p),   n_iters, warmup))["p50"]
        sf_t   = stats(timed_loop(lambda p=path: load_sf(p),      n_iters, warmup))["p50"]

        ratio_lib = lib_t / aus_t
        ratio_scp = scp_t / aus_t

        print(
            f"{label:<22} {aus_t:>{col}.3f} {lib_t:>{col}.3f} {scp_t:>{col}.3f} {sf_t:>{col}.3f}"
            f"  {ratio_lib:>6.2f}x  {ratio_scp:>6.2f}x"
        )

    print()
    save_header = f"{'Scenario':<22} {'aus':>{col}} {'sf/librosa':>{col}} {'scipy':>{col}}  aus/sf  aus/scipy"
    print("SAVE:")
    print(save_header)
    print("-" * len(save_header))

    for (sr, _, _, _, label), path in zip(SWEEP_SCENARIOS, paths):
        aus_audio, sf_data = _build_save_payloads(path)
        save_path = path.replace(".wav", "_sweep_out.wav")

        aus_t = stats(timed_loop(lambda a=aus_audio, sp=save_path: aus.io.save(sp, a),                            n_iters, warmup))["p50"]
        sf_t  = stats(timed_loop(lambda d=sf_data,  sp=save_path, r=sr: sf.write(sp, d, r),                       n_iters, warmup))["p50"]
        scp_t = stats(timed_loop(lambda d=sf_data,  sp=save_path, r=sr: scipy.io.wavfile.write(sp, r, d),         n_iters, warmup))["p50"]

        ratio_sf  = sf_t  / aus_t
        ratio_scp = scp_t / aus_t

        print(
            f"{label:<22} {aus_t:>{col}.3f} {sf_t:>{col}.3f} {scp_t:>{col}.3f}"
            f"  {ratio_sf:>6.2f}x  {ratio_scp:>6.2f}x"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="IO profiling harness")
    p.add_argument("--lib", default="audio_samples",
                   choices=["audio_samples", "librosa", "scipy", "soundfile"])
    p.add_argument("--op", default="load", choices=["load", "save"])
    p.add_argument("--iters", type=int, default=DEFAULT_ITERS)
    p.add_argument("--compare", action="store_true",
                   help="Compare all libraries for the default scenario")
    p.add_argument("--sweep", action="store_true",
                   help="Compare all libraries across multiple scenarios")
    p.add_argument("--wav", type=str, default=None,
                   help="Path to an existing WAV file (generated if not provided)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.sweep:
        run_sweep(SWEEP_ITERS, SWEEP_WARMUP)
        return

    if args.wav:
        wav_path = args.wav
        print(f"Using existing file: {wav_path}")
    else:
        wav_path = _wav_path(SAMPLE_RATE, DTYPE, DURATION_S, CHANNELS)
        if not Path(wav_path).exists():
            print("Generating reference WAV file...")
            make_wav(wav_path, SAMPLE_RATE, DTYPE, DURATION_S, CHANNELS)
        else:
            size_mb = Path(wav_path).stat().st_size / (1024 * 1024)
            print(f"Reusing existing file: {wav_path}  ({size_mb:.1f} MB)")

    if args.compare:
        run_compare(wav_path, args.iters)
    else:
        run_single(args.lib, args.op, wav_path, args.iters)


if __name__ == "__main__":
    main()
