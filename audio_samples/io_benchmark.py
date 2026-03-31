"""
io_benchmark.py - Benchmark audio file I/O across multiple Python libraries.

Libraries: audio_samples, librosa, wave (stdlib), scipy, torchaudio

Usage:
    python -m audio_samples.io_benchmark
    python -m audio_samples.io_benchmark --iters 30 --warmup 5 --out results.csv
"""

from __future__ import annotations

import argparse
import csv
import gc
import math
import tempfile
import time
import wave as wave_mod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import scipy.io.wavfile
import soundfile as sf
import librosa
import torch
import torchaudio
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich import box

import audio_samples as aus

console = Console()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SAMPLE_RATES: list[int] = [8000, 16000, 22100, 44100]
DTYPES: list[str] = ["i16", "f32"]
DURATIONS: list[float] = [1.0, 5.0, 10.0, 30.0, 60.0]
CHANNELS: list[int] = [1, 2]

LIBS: list[str] = ["audio_samples", "librosa", "wave", "scipy", "torchaudio"]

DEFAULT_ITERS = 100
DEFAULT_WARMUP = 10

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BenchmarkConfig:
    sample_rate: int
    dtype: str          # "i16" | "f32"
    duration: float     # seconds
    channels: int       # 1 = mono, 2 = stereo

    def label(self) -> str:
        ch = "mono" if self.channels == 1 else "stereo"
        return f"sr={self.sample_rate} {self.dtype} {self.duration:.0f}s {ch}"

    def n_samples(self) -> int:
        return int(self.sample_rate * self.duration)

    def file_size_bytes(self) -> int:
        bytes_per_sample = 2 if self.dtype == "i16" else 4
        return self.n_samples() * self.channels * bytes_per_sample


@dataclass
class LibResult:
    lib: str
    config: BenchmarkConfig
    load_times: list[float] = field(default_factory=list)   # seconds
    save_times: list[float] = field(default_factory=list)   # seconds
    error: Optional[str] = None

    # ---- statistics --------------------------------------------------------

    def _stats(self, times: list[float]) -> dict[str, float]:
        if not times:
            nan = float("nan")
            return {"mean": nan, "median": nan, "std": nan, "min": nan, "p90": nan}
        s = sorted(times)
        n = len(s)
        mean = sum(s) / n
        median = s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2.0
        variance = sum((x - mean) ** 2 for x in s) / n
        std = math.sqrt(variance)
        p90 = s[max(0, int(0.9 * n) - 1)]
        return {"mean": mean, "median": median, "std": std, "min": s[0], "p90": p90}

    def load_stats(self) -> dict[str, float]:
        return self._stats(self.load_times)

    def save_stats(self) -> dict[str, float]:
        return self._stats(self.save_times)


# ---------------------------------------------------------------------------
# Audio generation
# ---------------------------------------------------------------------------

NP_DTYPE = {"i16": np.dtype("int16"), "f32": np.dtype("float32")}


def generate_audio(cfg: BenchmarkConfig) -> aus.AudioSamples:
    """Generate test audio using audio_samples generation module."""
    np_dtype = NP_DTYPE[cfg.dtype]
    if cfg.channels == 1:
        return aus.sine_wave(
            frequency=440.0,
            duration_secs=cfg.duration,
            sample_rate=cfg.sample_rate,
            dtype=np_dtype,
        )
    # Stereo: stack two mono signals
    left = aus.sine_wave(
        frequency=440.0,
        duration_secs=cfg.duration,
        sample_rate=cfg.sample_rate,
        dtype=np_dtype,
    )
    right = aus.sine_wave(
        frequency=880.0,
        duration_secs=cfg.duration,
        sample_rate=cfg.sample_rate,
        dtype=np_dtype,
    )
    return aus.AudioSamples.stack([left, right])


# ---------------------------------------------------------------------------
# Library adapters
# ---------------------------------------------------------------------------
# Each load_* / save_* function is designed to be called as a zero-argument
# closure produced by prepare_library_fns() below, so that all data is
# captured by value and NO conversion work happens inside the timed region.
# ---------------------------------------------------------------------------


def _prepare_numpy_views(
    cfg: BenchmarkConfig, aus_audio: aus.AudioSamples
) -> dict[str, object]:
    """Pre-compute the numpy layouts each library expects for saves."""
    raw: np.ndarray = aus_audio.to_numpy()

    # Normalise to (n_channels, n_samples) regardless of what to_numpy() returns.
    # Mono comes back as (N,); stereo as (2, N).
    if raw.ndim == 1:
        np_ch_first: np.ndarray = raw[np.newaxis, :]   # (1, N)
    else:
        np_ch_first = raw                               # (C, N)

    # scipy / soundfile expect (N,) for mono and (N, C) for multi-channel
    if cfg.channels == 1:
        np_sc_mono: np.ndarray = np_ch_first[0]        # (N,)
    else:
        np_sc_mono = np_ch_first.T                      # (N, 2)

    # wave expects raw interleaved frames as bytes: [s0_L, s0_R, s1_L, s1_R, …]
    if cfg.dtype == "i16":
        wave_bytes: bytes = np_ch_first.T.astype(np.int16, copy=False).tobytes()
    else:
        wave_bytes = b""  # wave doesn't support f32

    # torchaudio expects (n_channels, n_samples) with channels_first=True (default)
    torch_tensor: torch.Tensor = torch.from_numpy(np.ascontiguousarray(np_ch_first))

    return {
        "ch_first": np_ch_first,
        "sc_mono": np_sc_mono,
        "wave_bytes": wave_bytes,
        "torch_tensor": torch_tensor,
    }


def prepare_library_fns(
    cfg: BenchmarkConfig,
    aus_audio: aus.AudioSamples,
    ref_path: str,
    save_dir: Path,
) -> dict[str, tuple[Callable[[], None], Callable[[], None]]]:
    """
    Return {lib_name: (load_fn, save_fn)} for all libraries.
    Both callables are zero-argument and ready to time.
    Data is captured in closures to avoid conversion overhead during timing.
    """
    views = _prepare_numpy_views(cfg, aus_audio)
    sr = cfg.sample_rate
    n_ch = cfg.channels

    # Pre-allocate save paths per library so we don't create directories in the
    # timed region and repeated writes always hit the same inode.
    save_paths = {lib: str(save_dir / f"{lib}_save.wav") for lib in LIBS}

    fns: dict[str, tuple[Callable[[], None], Callable[[], None]]] = {}

    # ---- audio_samples -----------------------------------------------------
    def _aus_load(p=ref_path):
        aus.io.read(p)

    def _aus_save(p=save_paths["audio_samples"], a=aus_audio):
        aus.io.save(p, a)

    fns["audio_samples"] = (_aus_load, _aus_save)

    # ---- librosa ------------------------------------------------------------
    # sr=None → no resampling; mono=False → preserve channels.
    # librosa passes dtype through to soundfile, so we can request the native
    # dtype (int16 or float32) and avoid any silent conversion overhead.
    _lib_np_dtype = NP_DTYPE[cfg.dtype]

    def _librosa_load(p=ref_path, dt=_lib_np_dtype):
        librosa.load(p, sr=None, mono=False, dtype=dt)

    # librosa has no write function; use soundfile (the same backend it uses).
    # Pass the native dtype data so sf.write picks the correct WAV subtype
    # automatically (PCM_16 for int16, FLOAT for float32).
    _sf_data = views["sc_mono"]  # already the correct dtype from _prepare_numpy_views

    def _librosa_save(p=save_paths["librosa"], d=_sf_data, s=sr):
        sf.write(p, d, s)

    fns["librosa"] = (_librosa_load, _librosa_save)

    # ---- wave (stdlib) -------------------------------------------------------
    # wave only handles PCM integer formats — skip f32 entirely
    if cfg.dtype == "i16":
        _wave_bytes: bytes = views["wave_bytes"]  # type: ignore[assignment]

        def _wave_load(p=ref_path):
            with wave_mod.open(p, "rb") as wf:
                wf.readframes(wf.getnframes())

        def _wave_save(
            p=save_paths["wave"],
            b=_wave_bytes,
            s=sr,
            c=n_ch,
        ):
            with wave_mod.open(p, "wb") as wf:
                wf.setnchannels(c)
                wf.setsampwidth(2)   # int16 = 2 bytes
                wf.setframerate(s)
                wf.writeframes(b)

        fns["wave"] = (_wave_load, _wave_save)
    else:
        fns["wave"] = None  # type: ignore[assignment]

    # ---- scipy --------------------------------------------------------------
    _scipy_data = views["sc_mono"]

    def _scipy_load(p=ref_path):
        scipy.io.wavfile.read(p)

    def _scipy_save(p=save_paths["scipy"], d=_scipy_data, s=sr):
        scipy.io.wavfile.write(p, s, d)

    fns["scipy"] = (_scipy_load, _scipy_save)

    # ---- torchaudio ---------------------------------------------------------
    _torch_tensor: torch.Tensor = views["torch_tensor"]  # type: ignore[assignment]

    def _torchaudio_load(p=ref_path):
        # TorchCodec backend always returns normalised float32 regardless of
        # the normalize flag — just call the standard load API.
        torchaudio.load(p)

    def _torchaudio_save(p=save_paths["torchaudio"], t=_torch_tensor, s=sr):
        torchaudio.save(p, t, s)

    fns["torchaudio"] = (_torchaudio_load, _torchaudio_save)

    return fns


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------


def time_fn(fn: Callable[[], None], n_iters: int, warmup_iters: int) -> list[float]:
    """
    Time *fn* with warmup.  GC is disabled for each individual timed call to
    prevent collection pauses from inflating measurements.  Returns raw sample
    times in seconds.
    """
    for _ in range(warmup_iters):
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


# ---------------------------------------------------------------------------
# Benchmark one configuration
# ---------------------------------------------------------------------------


def benchmark_one(
    cfg: BenchmarkConfig,
    ref_path: Path,
    save_dir: Path,
    n_iters: int,
    warmup_iters: int,
) -> dict[str, LibResult]:
    """Run load + save benchmarks for all libraries for a single config."""
    aus_audio = aus.io.read(str(ref_path))
    lib_fns = prepare_library_fns(cfg, aus_audio, str(ref_path), save_dir)

    results: dict[str, LibResult] = {}
    for lib in LIBS:
        res = LibResult(lib=lib, config=cfg)
        pair = lib_fns.get(lib)
        if pair is None:
            res.error = f"{lib} does not support {cfg.dtype}"
            results[lib] = res
            continue
        load_fn, save_fn = pair
        try:
            res.load_times = time_fn(load_fn, n_iters, warmup_iters)
        except Exception as exc:
            res.error = f"load error: {exc}"
        try:
            res.save_times = time_fn(save_fn, n_iters, warmup_iters)
        except Exception as exc:
            existing = res.error or ""
            res.error = (existing + f" | save error: {exc}").lstrip(" |")
        results[lib] = res

    return results


# ---------------------------------------------------------------------------
# CSV persistence
# ---------------------------------------------------------------------------


_CSV_FIELDS = [
    "lib",
    "sample_rate",
    "dtype",
    "duration",
    "channels",
    "n_samples",
    "file_size_bytes",
    "load_mean_ms",
    "load_median_ms",
    "load_std_ms",
    "load_min_ms",
    "load_p90_ms",
    "save_mean_ms",
    "save_median_ms",
    "save_std_ms",
    "save_min_ms",
    "save_p90_ms",
    "error",
]


def results_to_rows(
    all_results: list[tuple[BenchmarkConfig, dict[str, LibResult]]]
) -> list[dict]:
    rows = []
    for cfg, lib_results in all_results:
        for lib in LIBS:
            res = lib_results.get(lib)
            if res is None:
                continue
            ls = res.load_stats()
            ss = res.save_stats()

            def ms(v: float) -> str:
                return "" if math.isnan(v) else f"{v * 1e3:.4f}"

            rows.append(
                {
                    "lib": lib,
                    "sample_rate": cfg.sample_rate,
                    "dtype": cfg.dtype,
                    "duration": cfg.duration,
                    "channels": cfg.channels,
                    "n_samples": cfg.n_samples(),
                    "file_size_bytes": cfg.file_size_bytes(),
                    "load_mean_ms": ms(ls["mean"]),
                    "load_median_ms": ms(ls["median"]),
                    "load_std_ms": ms(ls["std"]),
                    "load_min_ms": ms(ls["min"]),
                    "load_p90_ms": ms(ls["p90"]),
                    "save_mean_ms": ms(ss["mean"]),
                    "save_median_ms": ms(ss["median"]),
                    "save_std_ms": ms(ss["std"]),
                    "save_min_ms": ms(ss["min"]),
                    "save_p90_ms": ms(ss["p90"]),
                    "error": res.error or "",
                }
            )
    return rows


def save_csv(rows: list[dict], path: str) -> None:
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Rich display helpers
# ---------------------------------------------------------------------------

LIB_COLORS = {
    "audio_samples": "green",
    "librosa": "cyan",
    "wave": "yellow",
    "scipy": "magenta",
    "torchaudio": "blue",
}


def _fmt_ms(v: float) -> str:
    if math.isnan(v):
        return "[dim]-[/dim]"
    if v * 1e3 < 1.0:
        return f"{v * 1e6:.1f}µs"
    return f"{v * 1e3:.2f}ms"


def print_scenario_table(
    cfg: BenchmarkConfig, lib_results: dict[str, LibResult]
) -> None:
    title = (
        f"[bold]{cfg.label()}[/bold]  "
        f"[dim]({cfg.file_size_bytes() / 1024:.1f} KiB)[/dim]"
    )
    tbl = Table(title=title, box=box.SIMPLE_HEAVY, show_lines=False)
    tbl.add_column("Library", style="bold", min_width=14)
    tbl.add_column("Load mean", justify="right", min_width=10)
    tbl.add_column("Load p90", justify="right", min_width=10)
    tbl.add_column("Load std", justify="right", min_width=10)
    tbl.add_column("Save mean", justify="right", min_width=10)
    tbl.add_column("Save p90", justify="right", min_width=10)
    tbl.add_column("Save std", justify="right", min_width=10)
    tbl.add_column("Notes", justify="left")

    for lib in LIBS:
        res = lib_results.get(lib)
        if res is None:
            continue
        color = LIB_COLORS.get(lib, "white")
        ls = res.load_stats()
        ss = res.save_stats()
        note = f"[dim]{res.error}[/dim]" if res.error else ""
        tbl.add_row(
            f"[{color}]{lib}[/{color}]",
            _fmt_ms(ls["mean"]),
            _fmt_ms(ls["p90"]),
            _fmt_ms(ls["std"]),
            _fmt_ms(ss["mean"]),
            _fmt_ms(ss["p90"]),
            _fmt_ms(ss["std"]),
            note,
        )
    console.print(tbl)


def print_summary_table(
    all_results: list[tuple[BenchmarkConfig, dict[str, LibResult]]]
) -> None:
    """Print one summary table per (dtype, channels) combination."""
    for dtype in DTYPES:
        for channels in CHANNELS:
            ch_label = "mono" if channels == 1 else "stereo"
            tbl = Table(
                title=f"[bold]Summary — {dtype} {ch_label}[/bold]  "
                      f"[dim](mean load / mean save across all SR & duration)[/dim]",
                box=box.ROUNDED,
            )
            tbl.add_column("Library", style="bold", min_width=14)
            for sr in SAMPLE_RATES:
                for dur in DURATIONS:
                    tbl.add_column(
                        f"{sr}Hz\n{dur:.0f}s",
                        justify="right",
                        min_width=9,
                        no_wrap=True,
                    )

            for lib in LIBS:
                color = LIB_COLORS.get(lib, "white")
                row: list[str] = [f"[{color}]{lib}[/{color}]"]
                for sr in SAMPLE_RATES:
                    for dur in DURATIONS:
                        # find matching result
                        match = next(
                            (
                                lr[lib]
                                for cfg, lr in all_results
                                if cfg.dtype == dtype
                                and cfg.channels == channels
                                and cfg.sample_rate == sr
                                and cfg.duration == dur
                            ),
                            None,
                        )
                        if match is None or not match.load_times:
                            cell = "[dim]-[/dim]"
                        else:
                            ls = match.load_stats()
                            ss = match.save_stats()
                            lm = ls["mean"]
                            sm = ss["mean"]
                            cell = (
                                f"{_fmt_ms(lm)}\n[dim]{_fmt_ms(sm)}[/dim]"
                            )
                        row.append(cell)
                tbl.add_row(*row)

            console.print(tbl)
            console.print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Benchmark audio I/O across multiple Python libraries."
    )
    p.add_argument(
        "--iters", type=int, default=DEFAULT_ITERS,
        help=f"Number of timed iterations per benchmark (default: {DEFAULT_ITERS})",
    )
    p.add_argument(
        "--warmup", type=int, default=DEFAULT_WARMUP,
        help=f"Number of warmup iterations (default: {DEFAULT_WARMUP})",
    )
    p.add_argument(
        "--out", type=str, default="bench_io.csv",
        help="Output CSV path (default: bench_io.csv)",
    )
    p.add_argument(
        "--verbose", action="store_true",
        help="Print a detailed table for every scenario (not just the summary)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    n_iters: int = args.iters
    warmup_iters: int = args.warmup
    csv_out: str = args.out
    verbose: bool = args.verbose

    configs = [
        BenchmarkConfig(sr, dtype, dur, ch)
        for sr in SAMPLE_RATES
        for dtype in DTYPES
        for dur in DURATIONS
        for ch in CHANNELS
    ]
    total = len(configs)

    console.rule("[bold blue]Audio I/O Benchmark[/bold blue]")
    console.print(f"  Libraries  : {', '.join(LIBS)}")
    console.print(f"  Sample rates: {SAMPLE_RATES}")
    console.print(f"  Dtypes      : {DTYPES}")
    console.print(f"  Durations   : {DURATIONS}s")
    console.print(f"  Channels    : {CHANNELS}")
    console.print(f"  Scenarios   : {total}")
    console.print(f"  Iterations  : {n_iters}  (warmup: {warmup_iters})")
    console.print()

    all_results: list[tuple[BenchmarkConfig, dict[str, LibResult]]] = []

    with tempfile.TemporaryDirectory(prefix="audio_io_bench_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        ref_dir = tmp_path / "ref"
        save_dir = tmp_path / "saves"
        ref_dir.mkdir()
        save_dir.mkdir()

        # ── Phase 1: generate & persist all reference WAV files ──────────────
        console.print("[bold]Phase 1/2 — Generating reference WAV files[/bold]")
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task("Generating...", total=total)
            ref_files: dict[BenchmarkConfig, Path] = {}
            for cfg in configs:
                progress.update(task, description=f"[dim]{cfg.label()}[/dim]")
                audio = generate_audio(cfg)
                ref_path = ref_dir / f"{cfg.sample_rate}_{cfg.dtype}_{cfg.duration}_{cfg.channels}.wav"
                aus.io.save(str(ref_path), audio)
                ref_files[cfg] = ref_path
                progress.advance(task)
        console.print(f"  [green]✓[/green] {total} reference files written to {ref_dir}\n")

        # ── Phase 2: run benchmarks ───────────────────────────────────────────
        console.print("[bold]Phase 2/2 — Running benchmarks[/bold]")
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Benchmarking...", total=total)
            for cfg in configs:
                progress.update(task, description=f"[dim]{cfg.label()}[/dim]")
                lib_results = benchmark_one(
                    cfg,
                    ref_files[cfg],
                    save_dir,
                    n_iters,
                    warmup_iters,
                )
                all_results.append((cfg, lib_results))
                progress.advance(task)

        console.print("  [green]✓[/green] Benchmarks complete\n")

    # ── Results ───────────────────────────────────────────────────────────────
    if verbose:
        console.print("\n[bold]Per-scenario results[/bold]\n")
        for cfg, lib_results in all_results:
            print_scenario_table(cfg, lib_results)
            console.print()

    console.print("[bold]Summary tables[/bold] [dim](load mean / save mean)[/dim]\n")
    print_summary_table(all_results)

    # ── CSV ──────────────────────────────────────────────────────────────────
    rows = results_to_rows(all_results)
    save_csv(rows, csv_out)
    console.print(f"[green]Results saved →[/green] {csv_out}")


if __name__ == "__main__":
    main()
