import time
import statistics
import numpy as np
import librosa
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import audio_samples as aus
from audio_samples import ResamplingQuality


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------


def t(fn, n=100):
    fn()  # warm-up
    times = []
    for _ in range(n):
        s = time.perf_counter()
        fn()
        times.append(time.perf_counter() - s)
    times.sort()
    return statistics.mean(times[1:-1]) * 1e6  # µs, trimmed mean


# ---------------------------------------------------------------------------
# Scenarios and quality tiers
# ---------------------------------------------------------------------------

SCENARIOS = [
    # (label, duration, src_sr, tgt_sr, channels)
    ("1s_mono_dn", 1.0, 44100, 22050, 1),
    ("5s_mono_dn", 5.0, 44100, 22050, 1),
    ("10s_mono_dn", 10.0, 44100, 22050, 1),
    ("1s_mono_up", 1.0, 22050, 44100, 1),
    ("5s_mono_up", 5.0, 22050, 44100, 1),
    ("10s_mono_up", 10.0, 22050, 44100, 1),
    ("1s_st_dn", 1.0, 44100, 22050, 2),
    ("5s_st_dn", 5.0, 44100, 22050, 2),
    ("5s_st_up", 5.0, 22050, 44100, 2),
    ("10s_st_dn", 10.0, 44100, 22050, 2),
    ("10s_st_up", 10.0, 22050, 44100, 2),
]

TIERS = [
    # (aus_quality, librosa_res_type, label)
    (ResamplingQuality.fast, "linear", "fast"),
    (ResamplingQuality.medium, "kaiser_fast", "medium"),
    (ResamplingQuality.high, "kaiser_best", "high"),
]

TIER_COLORS = {
    "fast": "#1f77b4",
    "medium": "#ff7f0e",
    "high": "#2ca02c",
}


# ---------------------------------------------------------------------------
# Run benchmarks
# ---------------------------------------------------------------------------

# results[scenario_label][tier_label] = {"aus": µs, "lib": µs}
results: dict[str, dict[str, dict[str, float]]] = {}

header = (
    f"{'Scenario':<13} {'Tier':<8} {'aus (µs)':>10} {'librosa (µs)':>13} {'Speedup':>8}"
)
print(header)
print("-" * len(header))

for label, dur, src_sr, tgt_sr, channels in SCENARIOS:
    if channels == 1:
        a = aus.sine_wave(frequency=440.0, duration_secs=dur, sample_rate=src_sr)
    else:
        waves = [
            aus.sine_wave(
                frequency=440.0 * (i + 1), duration_secs=dur, sample_rate=src_sr
            )
            for i in range(channels)
        ]
        a = aus.AudioSamples.stack(waves)
    # librosa expects (n_samples,) for mono or (n_channels, n_samples) for multi-channel.
    # to_numpy() already returns (n_channels, n_samples), so no transpose.
    # The previous .T was turning stereo (2, N) into (N, 2), which resampy
    # misinterprets as N 2-sample signals — effectively free and meaningless.
    arr = a.to_numpy()

    results[label] = {}
    for aus_q, lib_r, tier in TIERS:
        aus_t = t(lambda q=aus_q: a.resample(target_sample_rate=tgt_sr, quality=q))
        lib_t = t(
            lambda r=lib_r: librosa.resample(
                arr, orig_sr=src_sr, target_sr=tgt_sr, res_type=r
            )
        )
        results[label][tier] = {"aus": aus_t, "lib": lib_t}
        print(
            f"{label:<13} {tier:<8} {aus_t:>10.1f} {lib_t:>13.1f} {lib_t / aus_t:>7.1f}x"
        )
    print()


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

scenario_labels = [s[0] for s in SCENARIOS]
tier_labels = [t[2] for t in TIERS]
n_scenarios = len(scenario_labels)
x = np.arange(n_scenarios)

fig = plt.figure(figsize=(17, 10))
fig.suptitle(
    "audio_samples vs librosa — resampling benchmark", fontsize=13, fontweight="bold"
)
gs = fig.add_gridspec(
    2, 3, hspace=0.55, wspace=0.35, left=0.07, right=0.97, top=0.91, bottom=0.12
)

# ── Top row: absolute times (log scale), one subplot per quality tier ──────

for col, (_, _, tier) in enumerate(TIERS):
    ax = fig.add_subplot(gs[0, col])
    color = TIER_COLORS[tier]

    aus_times = [results[s][tier]["aus"] for s in scenario_labels]
    lib_times = [results[s][tier]["lib"] for s in scenario_labels]

    w = 0.38
    ax.bar(x - w / 2, aus_times, w, label="audio_samples", color=color, zorder=3)
    ax.bar(
        x + w / 2,
        lib_times,
        w,
        label="librosa",
        color=color,
        alpha=0.45,
        hatch="///",
        zorder=3,
    )

    ax.set_yscale("log")
    ax.set_title(f"{tier.capitalize()} quality", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(scenario_labels, rotation=40, ha="right", fontsize=7.5)
    ax.set_ylabel("Time (µs)")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3, zorder=0)

# ── Bottom row: speedup (librosa / audio_samples), all tiers together ──────

ax_sp = fig.add_subplot(gs[1, :])
bar_w = 0.22
offsets = np.array([-bar_w, 0, bar_w])

for i, (_, _, tier) in enumerate(TIERS):
    speedups = [
        results[s][tier]["lib"] / results[s][tier]["aus"] for s in scenario_labels
    ]
    bars = ax_sp.bar(
        x + offsets[i],
        speedups,
        bar_w,
        label=tier.capitalize(),
        color=TIER_COLORS[tier],
        zorder=3,
    )
    # Label bars that are very large (> 100×) to keep y-axis readable
    for bar, sp in zip(bars, speedups):
        if sp > 100:
            ax_sp.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.05,
                f"{sp:.0f}×",
                ha="center",
                va="bottom",
                fontsize=6.5,
                fontweight="bold",
                color=TIER_COLORS[tier],
            )

ax_sp.axhline(
    1.0, color="crimson", linestyle="--", linewidth=1.2, label="1× (parity)", zorder=4
)
ax_sp.set_yscale("log")
ax_sp.set_xticks(x)
ax_sp.set_xticklabels(scenario_labels, rotation=40, ha="right")
ax_sp.set_ylabel("Speedup  (librosa ÷ audio_samples,  higher = faster)")
ax_sp.set_title("Speedup over librosa by quality tier", fontweight="bold")
ax_sp.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:g}×"))
ax_sp.legend(fontsize=9)
ax_sp.grid(axis="y", alpha=0.3, zorder=0)

out = "bench_resampling.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Plot saved → {out}")
