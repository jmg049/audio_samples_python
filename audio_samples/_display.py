"""Jupyter rich display for :class:`audio_samples.AudioSamples`.

``AudioSamples._repr_html_`` calls :func:`repr_html`. It renders a summary
table and, for short mono or stereo clips, an inline ``<audio>`` player that
embeds the samples as a base64-encoded 16-bit PCM WAV file. No JavaScript and
no external resources are used.
"""

from __future__ import annotations

import base64
import html
import io
import wave

import numpy as np

#: Clips longer than this (in seconds) get no inline player, to keep notebooks small.
MAX_PLAYER_SECONDS = 30.0


def _to_float(data: np.ndarray, dtype: str) -> np.ndarray:
    """Map samples to float64 in [-1, 1] using the sample format's full scale."""
    x = np.asarray(data)
    if np.issubdtype(x.dtype, np.floating):
        return x.astype(np.float64, copy=False)
    if dtype == "u8":
        return (x.astype(np.float64) - 128.0) / 128.0
    if dtype == "I24":
        return x.astype(np.float64) / float(1 << 23)
    if np.issubdtype(x.dtype, np.integer):
        return x.astype(np.float64) / float(np.iinfo(x.dtype).max + 1)
    return x.astype(np.float64)


def _wav_base64(samples: np.ndarray, sample_rate: int) -> str:
    """Encode (channels, samples) floats in [-1, 1] as a base64 16-bit WAV."""
    clipped = np.clip(samples, -1.0, 1.0)
    pcm = np.round(clipped * 32767.0).astype("<i2")
    interleaved = pcm.T.reshape(-1)  # (samples, channels) frame order
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(samples.shape[0])
        w.setsampwidth(2)
        w.setframerate(int(sample_rate))
        w.writeframes(interleaved.tobytes())
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _fmt_db(value: float) -> str:
    if value <= 0.0 or not np.isfinite(value):
        return "-inf dBFS" if value == 0.0 else "n/a"
    return f"{20.0 * np.log10(value):.1f} dBFS"


def repr_html(audio) -> str:
    """Return an HTML summary (and player) for ``audio``; never raises."""
    try:
        return _render(audio)
    except Exception:  # pragma: no cover - defensive fallback
        try:
            text = repr(audio)
        except Exception:
            text = "AudioSamples"
        return f"<pre>{html.escape(text)}</pre>"


def _render(audio) -> str:
    sr = int(audio.sample_rate)
    channels = int(audio.channels)
    n = int(audio.samples_per_channel())
    duration = n / sr
    dtype = str(audio.dtype)

    data = np.asarray(audio.to_numpy())
    data2d = data.reshape(1, -1) if data.ndim == 1 else data
    x = _to_float(data2d, dtype)
    finite = np.isfinite(x)
    peak = float(np.max(np.abs(x[finite]))) if finite.any() else float("nan")
    rms = float(np.sqrt(np.mean(np.square(x[finite])))) if finite.any() else float("nan")

    layout = {1: "mono", 2: "stereo"}.get(channels, f"{channels} channels")
    rows = [
        ("dtype", dtype),
        ("channels", f"{channels} ({layout})"),
        ("sample rate", f"{sr} Hz"),
        ("duration", f"{duration:.3f} s ({n} samples per channel)"),
        ("peak", f"{peak:.4f} ({_fmt_db(peak)})"),
        ("RMS", f"{rms:.4f} ({_fmt_db(rms)})"),
    ]
    if not finite.all():
        rows.append(("non-finite samples", str(int((~finite).sum()))))
    table = "".join(
        f"<tr><th style='text-align:left;padding-right:1em'>{html.escape(k)}</th>"
        f"<td style='text-align:left'>{html.escape(v)}</td></tr>"
        for k, v in rows
    )

    if channels > 2:
        player = "<p><em>No player: inline playback supports mono or stereo only.</em></p>"
    elif duration > MAX_PLAYER_SECONDS:
        player = (
            f"<p><em>No player: clip is longer than {MAX_PLAYER_SECONDS:g} s.</em></p>"
        )
    elif not finite.all():
        player = "<p><em>No player: the signal contains NaN or infinite samples.</em></p>"
    else:
        b64 = _wav_base64(x, sr)
        player = (
            "<audio controls preload='none' "
            f"src='data:audio/wav;base64,{b64}'>"
            "Your browser does not support the audio element.</audio>"
        )
        if peak > 1.0:
            player += "<p><em>Player clips samples outside [-1, 1].</em></p>"

    return (
        "<div class='audio-samples-repr'>"
        "<strong>AudioSamples</strong>"
        f"<table>{table}</table>{player}</div>"
    )
