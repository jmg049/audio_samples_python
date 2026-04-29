"""
profile_flac.py — Profiling harness for FLAC load/save comparison.

Usage:
    # Compare all libraries across multiple scenarios:
    python profile_flac.py --sweep

    # Quick comparison for the default scenario (30s stereo i16):
    python profile_flac.py --compare

    # Profile a single library in a tight loop (intended for py-spy):
    python profile_flac.py --lib audio_samples --op load
    python profile_flac.py --lib soundfile --op load

    # Generate flamegraph (requires py-spy: pip install py-spy):
    py-spy record -o flamegraph_aus_flac_load.svg -- python profile_flac.py --lib audio_samples --op load
    py-spy record --native -o flamegraph_aus_flac_load_native.svg -- python profile_flac.py --lib audio_samples --op load

    # Use your own FLAC file:
    python profile_flac.py --compare --flac /path/to/your.flac

Notes:
    - scipy does not support FLAC, so only audio_samples, soundfile, and librosa are compared.
    - soundfile uses libsndfile (C library) for FLAC decoding.
    - librosa uses soundfile internally and adds a float32 conversion step.
    - audio_samples uses a pure-Rust FLAC decoder.
    - All FLAC files are written at the default compression level (5) unless noted.
    - soundfile dtype='float32' is used so librosa and soundfile are on equal footing
      (soundfile defaults to float64 for PCM, which allocates 2x memory and is slower).
"""

from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa

import audio_samples as aus

# ---------------------------------------------------------------------------
# Default single-scenario settings
# ---------------------------------------------------------------------------
SAMPLE_RATE = 44100
DTYPE = "i16"
DURATION_S = 30.0
CHANNELS = 2

DEFAULT_ITERS = 1000
WARMUP_ITERS = 50

# ---------------------------------------------------------------------------
# Sweep scenarios: (sample_rate, dtype, duration_s, channels, label)
# ---------------------------------------------------------------------------
# FLAC only supports integer PCM (no float32).
# Subtypes used: i16 → PCM_16, i24 → PCM_24.
SWEEP_SCENARIOS = [
    (44100, "i16",  1.0,  1, "mono  1s  i16"),
    (44100, "i16",  5.0,  1, "mono  5s  i16"),
    (44100, "i16", 30.0,  1, "mono 30s  i16"),
    (44100, "i16",  5.0,  2, "stereo 5s  i16"),
    (44100, "i16", 30.0,  2, "stereo 30s i16"),
    (44100, "i16", 60.0,  2, "stereo 60s i16"),
    (44100, "i24",  5.0,  1, "mono  5s  i24"),
    (44100, "i24", 30.0,  2, "stereo 30s i24"),
    (48000, "i16", 30.0,  2, "stereo 30s i16 48k"),
]

SWEEP_ITERS = 2000
SWEEP_WARMUP = 50


# ---------------------------------------------------------------------------
# FLAC file helpers
# ---------------------------------------------------------------------------

def _flac_path(sr: int, dtype: str, dur: float, ch: int) -> str:
    return f"/tmp/profile_flac_{sr}_{dtype}_{dur:.0f}s_{ch}ch.flac"


_SF_SUBTYPE = {"i16": "PCM_16", "i24": "PCM_24", "i32": "PCM_32"}
_NP_DTYPE   = {"i16": np.dtype("int16"), "i24": np.dtype("int32"), "i32": np.dtype("int32")}


def make_flac(path: str, sr: int, dtype: str, dur: float, ch: int) -> None:
    np_dtype   = _NP_DTYPE[dtype]
    sf_subtype = _SF_SUBTYPE[dtype]

    if ch == 1:
        data = aus.sine_wave(440.0, dur, sr, dtype=np_dtype).to_numpy()
        sf.write(path, data, sr, subtype=sf_subtype, format="FLAC")
    else:
        left  = aus.sine_wave(440.0, dur, sr, dtype=np_dtype).to_numpy()
        right = aus.sine_wave(880.0, dur, sr, dtype=np_dtype).to_numpy()
        stereo = np.stack([left, right], axis=1)  # (N, 2) — soundfile layout
        sf.write(path, stereo, sr, subtype=sf_subtype, format="FLAC")

    size_kb = Path(path).stat().st_size / 1024
    print(f"  Written: {path}  ({size_kb:.0f} KB)")


def ensure_flac(sr: int, dtype: str, dur: float, ch: int) -> str:
    path = _flac_path(sr, dtype, dur, ch)
    if not Path(path).exists():
        make_flac(path, sr, dtype, dur, ch)
    return path


# ---------------------------------------------------------------------------
# Load functions
# ---------------------------------------------------------------------------

def load_aus(path: str) -> None:
    aus.io.read(path)


def load_librosa(path: str) -> None:
    librosa.load(path, sr=None, mono=False, dtype=np.float32)


def load_sf(path: str) -> None:
    # dtype='float32' matches librosa's output — fair comparison.
    # soundfile's default (float64) allocates 2x memory and converts more.
    sf.read(path, always_2d=True, dtype="float32")


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------

def _build_save_payloads(flac_path: str):
    aus_audio = aus.io.read(flac_path)
    raw = aus_audio.to_numpy()
    sf_data = raw.T.astype(np.int16) if raw.ndim == 2 else raw.astype(np.int16)
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

def run_single(lib: str, op: str, flac_path: str, n_iters: int) -> None:
    print(f"\n[{lib}] {op}  —  {n_iters} iters  (warmup={WARMUP_ITERS})")
    print(f"  File: {flac_path}")

    if op == "load":
        loaders = {
            "audio_samples": load_aus,
            "librosa":        load_librosa,
            "soundfile":      load_sf,
        }
        fn = loaders.get(lib)
        if fn is None:
            raise ValueError(f"Unknown lib '{lib}'. Choices: audio_samples, librosa, soundfile")
        times = timed_loop(lambda: fn(flac_path), n_iters, WARMUP_ITERS)
        print_stats(f"{lib} load", times)

    elif op == "save":
        aus_audio, sf_data = _build_save_payloads(flac_path)
        save_path = flac_path.replace(".flac", f"_{lib}_out.flac")

        if lib == "audio_samples":
            times = timed_loop(lambda: aus.io.save(save_path, aus_audio), n_iters, WARMUP_ITERS)
        elif lib in ("soundfile", "librosa"):
            times = timed_loop(
                lambda: sf.write(save_path, sf_data, SAMPLE_RATE, format="FLAC"),
                n_iters, WARMUP_ITERS,
            )
        else:
            raise ValueError(f"Unknown lib '{lib}'.")
        print_stats(f"{lib} save", times)

    else:
        raise ValueError(f"Unknown op: {op}")


def run_compare(flac_path: str, n_iters: int) -> None:
    print(f"\nComparison: sr={SAMPLE_RATE} {DTYPE} {DURATION_S:.0f}s {'stereo' if CHANNELS == 2 else 'mono'}")
    print(f"  File: {flac_path}  ({Path(flac_path).stat().st_size / 1024:.0f} KB)")
    print(f"  Iters: {n_iters}  warmup: {WARMUP_ITERS}\n")

    aus_audio, sf_data = _build_save_payloads(flac_path)
    save_path = flac_path.replace(".flac", "_out.flac")

    load_cases = [
        ("audio_samples", lambda: load_aus(flac_path)),
        ("librosa",        lambda: load_librosa(flac_path)),
        ("soundfile",      lambda: load_sf(flac_path)),
    ]
    save_cases = [
        ("audio_samples", lambda: aus.io.save(save_path, aus_audio)),
        ("soundfile",      lambda: sf.write(save_path, sf_data, SAMPLE_RATE, format="FLAC")),
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

    paths = [ensure_flac(sr, dtype, dur, ch) for sr, dtype, dur, ch, _ in SWEEP_SCENARIOS]

    # Global pre-warmup: one pass through every scenario for every library.
    # Warms glibc's mmap allocator for all FLAC file sizes before measuring.
    print("Pre-warming allocator across all scenarios... ", end="", flush=True)
    GLOBAL_WARMUP = 100
    for _ in range(GLOBAL_WARMUP):
        for path in paths:
            load_aus(path)
            load_sf(path)
    print("done\n")

    col = 10
    load_header = (
        f"{'Scenario':<22} {'aus':>{col}} {'librosa':>{col}} {'soundfile':>{col}}"
        f"  aus/lib  aus/sf"
    )
    print("LOAD:")
    print(load_header)
    print("-" * len(load_header))

    for (_, _, _, _, label), path in zip(SWEEP_SCENARIOS, paths):
        aus_t = stats(timed_loop(lambda p=path: load_aus(p),     n_iters, warmup))["p50"]
        lib_t = stats(timed_loop(lambda p=path: load_librosa(p), n_iters, warmup))["p50"]
        sf_t  = stats(timed_loop(lambda p=path: load_sf(p),      n_iters, warmup))["p50"]

        ratio_lib = lib_t / aus_t
        ratio_sf  = sf_t  / aus_t

        print(
            f"{label:<22} {aus_t:>{col}.3f} {lib_t:>{col}.3f} {sf_t:>{col}.3f}"
            f"  {ratio_lib:>6.2f}x  {ratio_sf:>5.2f}x"
        )

    print()
    save_header = f"{'Scenario':<22} {'aus':>{col}} {'soundfile':>{col}}  aus/sf"
    print("SAVE:")
    print(save_header)
    print("-" * len(save_header))

    for (sr, dtype, _, _, label), path in zip(SWEEP_SCENARIOS, paths):
        aus_audio, sf_data = _build_save_payloads(path)
        save_path = path.replace(".flac", "_sweep_out.flac")
        sf_subtype = _SF_SUBTYPE[dtype]

        aus_t = stats(timed_loop(lambda a=aus_audio, sp=save_path: aus.io.save(sp, a),                                                           n_iters, warmup))["p50"]
        sf_t  = stats(timed_loop(lambda d=sf_data, sp=save_path, r=sr, st=sf_subtype: sf.write(sp, d, r, subtype=st, format="FLAC"),             n_iters, warmup))["p50"]

        ratio_sf = sf_t / aus_t

        print(
            f"{label:<22} {aus_t:>{col}.3f} {sf_t:>{col}.3f}"
            f"  {ratio_sf:>5.2f}x"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FLAC IO profiling harness")
    p.add_argument("--lib", default="audio_samples",
                   choices=["audio_samples", "librosa", "soundfile"])
    p.add_argument("--op", default="load", choices=["load", "save"])
    p.add_argument("--iters", type=int, default=DEFAULT_ITERS)
    p.add_argument("--compare", action="store_true",
                   help="Compare all libraries for the default scenario")
    p.add_argument("--sweep", action="store_true",
                   help="Compare all libraries across multiple scenarios")
    p.add_argument("--flac", type=str, default=None,
                   help="Path to an existing FLAC file (generated if not provided)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.sweep:
        run_sweep(SWEEP_ITERS, SWEEP_WARMUP)
        return

    if args.flac:
        flac_path = args.flac
        print(f"Using existing file: {flac_path}")
    else:
        flac_path = _flac_path(SAMPLE_RATE, DTYPE, DURATION_S, CHANNELS)
        if not Path(flac_path).exists():
            print("Generating reference FLAC file...")
            make_flac(flac_path, SAMPLE_RATE, DTYPE, DURATION_S, CHANNELS)
        else:
            size_kb = Path(flac_path).stat().st_size / 1024
            print(f"Reusing existing file: {flac_path}  ({size_kb:.0f} KB)")

    if args.compare:
        run_compare(flac_path, args.iters)
    else:
        run_single(args.lib, args.op, flac_path, args.iters)


if __name__ == "__main__":
    main()
