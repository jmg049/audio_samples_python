#!/usr/bin/env python3
"""
Benchmarks comparing audio_samples vs librosa for common operations.

Run with:
    python -m audio_samples.benchmarks
    python -m audio_samples.benchmarks --fast       # quick test
    python -m audio_samples.benchmarks -o out.json  # save results

Quality mapping
---------------
aus Fast   ↔  librosa res_type='linear'        (zero-overhead linear interp)
aus Medium ↔  librosa res_type='kaiser_fast'   (sinc, moderate quality)
aus High   ↔  librosa res_type='kaiser_best'   (sinc, highest quality, default)
"""

import pyperf
import numpy.typing as npt
import librosa
import audio_samples as aus
from audio_samples import AudioSamples, ResamplingQuality


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def generate_audio(
    duration_seconds: float = 5.0,
    sample_rate: int = 44100,
    num_channels: int = 1,
) -> AudioSamples:
    if num_channels == 1:
        return aus.sine_wave(
            frequency=440.0,
            duration_secs=duration_seconds,
            sample_rate=sample_rate,
        )
    waves = [
        aus.sine_wave(
            frequency=440.0 * (i + 1),
            duration_secs=duration_seconds,
            sample_rate=sample_rate,
        )
        for i in range(num_channels)
    ]
    return aus.AudioSamples.stack(waves)


def audio_samples_to_librosa_compat(audio: AudioSamples) -> tuple[npt.NDArray, int]:
    return audio.to_numpy().T, audio.sample_rate


def resample_librosa(
    audio: npt.NDArray, orig_sr: int, target_sr: int, res_type: str
) -> npt.NDArray:
    return librosa.resample(
        audio, orig_sr=orig_sr, target_sr=target_sr, res_type=res_type
    )


def resample_aus(
    audio: AudioSamples, target_sr: int, quality: ResamplingQuality
) -> AudioSamples:
    return audio.resample(target_sample_rate=target_sr, quality=quality)


# ---------------------------------------------------------------------------
# Benchmark scenarios
#
# Each entry: (label_suffix, duration_secs, src_sr, target_sr, channels)
# ---------------------------------------------------------------------------

SCENARIOS = [
    # (label,            duration, src_sr, target_sr, channels)
    # ── duration sweep (mono, downsampling 44.1k → 22.05k) ──────────────────
    ("1s_mono_down", 1.0, 44100, 22050, 1),
    ("5s_mono_down", 5.0, 44100, 22050, 1),
    ("10s_mono_down", 10.0, 44100, 22050, 1),
    # ── upsampling (mono, 22.05k → 44.1k) ───────────────────────────────────
    ("1s_mono_up", 1.0, 22050, 44100, 1),
    ("5s_mono_up", 5.0, 22050, 44100, 1),
    ("10s_mono_up", 10.0, 22050, 44100, 1),
    # ── multi-channel downsampling ───────────────────────────────────────────
    ("5s_stereo_down", 5.0, 44100, 22050, 2),
    ("5s_4ch_down", 5.0, 44100, 22050, 4),
    # ── multi-channel upsampling ─────────────────────────────────────────────
    ("5s_stereo_up", 5.0, 22050, 44100, 2),
    ("5s_4ch_up", 5.0, 22050, 44100, 4),
]

# Quality tiers: (aus_quality, librosa_res_type, tier_label)
QUALITY_TIERS = [
    (ResamplingQuality.fast, "linear", "fast"),
    (ResamplingQuality.medium, "kaiser_fast", "medium"),
    (ResamplingQuality.high, "kaiser_best", "high"),
]


def _build_inputs(duration, src_sr, target_sr, channels):
    """Pre-build both input formats so timing excludes setup."""
    aus_audio = generate_audio(duration, src_sr, channels)
    librosa_audio, _ = audio_samples_to_librosa_compat(aus_audio)
    return aus_audio, librosa_audio, src_sr, target_sr


if __name__ == "__main__":
    runner = pyperf.Runner()

    for label, duration, src_sr, target_sr, channels in SCENARIOS:
        aus_audio, librosa_audio, orig_sr, tgt_sr = _build_inputs(
            duration, src_sr, target_sr, channels
        )

        for aus_quality, librosa_res_type, tier in QUALITY_TIERS:
            runner.bench_func(
                f"resample_aus[{tier}]/{label}",
                resample_aus,
                aus_audio,
                tgt_sr,
                aus_quality,
            )
            runner.bench_func(
                f"resample_librosa[{tier}]/{label}",
                resample_librosa,
                librosa_audio,
                orig_sr,
                tgt_sr,
                librosa_res_type,
            )
