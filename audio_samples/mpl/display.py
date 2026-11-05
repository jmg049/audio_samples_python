"""Matplotlib display of feature matrices and waveforms.

:func:`plot_matrix` draws any time-frequency or feature matrix with axes that know what
they show: seconds, lag, Hz, mel, constant-Q bins, notes, svara, Functional Just System
names, chroma, tempo, or tonnetz dimensions. :func:`plot_waveform_adaptive` draws a
waveform that follows the zoom: an envelope far out and samples close in.
:func:`plot_waveform_bars` and :func:`plot_waveform_f0` give simplified and
pitch-displaced waveforms. The rest are the tick formatters, colormap selection, colorbars,
multi-panel layout, legends and highlights these need.

All plotting functions take ``ax=`` and ``style=`` like the rest of ``audio_samples.mpl``.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, Sequence

import matplotlib
import numpy as np
from matplotlib import colors as mcolors
from matplotlib import patheffects as mpe
from matplotlib import ticker as mticker
from matplotlib import transforms as mtransforms
from matplotlib.collections import PatchCollection
from matplotlib.patches import FancyBboxPatch
from numpy.typing import NDArray

from audio_samples import notation as _notation
from audio_samples import utils as _utils

from ._utils import _apply_style, _ensure_axes, _resolve_style

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from .style import MplStyle

_TIME_AXES = ("time", "h", "m", "s", "ms")
_LAG_AXES = ("lag", "lag_h", "lag_m", "lag_s", "lag_ms")
_FFT_AXES = ("linear", "fft", "hz", "fft_note", "fft_svara", "oct3", "log", "log_oct3")
_MEL_AXES = ("mel", "mel_oct3")
_CQT_AXES = ("cqt", "cqt_hz", "cqt_note", "cqt_svara", "cqt_oct3")
_VQT_AXES = ("vqt_hz", "vqt_note", "vqt_oct3", "vqt_fjs")
_CHROMA_AXES = ("chroma", "chroma_h", "chroma_c", "chroma_fjs")
_OTHER_AXES = ("tempo", "fourier_tempo", "tonnetz", "frames", "off", "none", None)
AXIS_TYPES = (*_TIME_AXES, *_LAG_AXES, *_FFT_AXES, *_MEL_AXES, *_CQT_AXES, *_VQT_AXES, *_CHROMA_AXES, *_OTHER_AXES)
"""Every axis type :func:`plot_matrix` accepts."""


def _midi_name(midi: float, key: str, octave: bool, unicode: bool, cents: bool = False) -> str:
    if not np.isfinite(midi):
        return ""
    names = _utils.key_to_notes(key, unicode=unicode)
    number = int(np.round(midi))
    name = names[number % 12]
    if octave:
        name = f"{name}{int(number / 12) - 1}"
    if cents:
        name = f"{name}{int(100 * np.around(midi - number, 2)):+02d}"
    return name


def _hz_to_midi(hz: float) -> float:
    return 69.0 + 12.0 * np.log2(hz / 440.0)


class TimeFormatter(mticker.Formatter):
    """Formats seconds as a time or a lag.

    Args:
        lag: Wrap values past half the axis to negative lags.
        unit: ``"h"``, ``"m"``, ``"s"``, ``"ms"``, or None for ``h:mm:ss`` scaled to the view.
    """

    def __init__(self, lag: bool = False, unit: str | None = None) -> None:
        if unit not in (None, "h", "m", "s", "ms"):
            raise ValueError(f"unit: unknown {unit!r}")
        self.lag = lag
        self.unit = unit

    def __call__(self, x: float, pos: int | None = None) -> str:
        axis: Any = self.axis
        vmin, vmax = axis.get_view_interval()
        _, dmax = axis.get_data_interval()
        value = x
        sign = ""
        if self.lag and x > 0.5 * dmax and dmax > 0:
            value = np.abs(x - dmax)
            sign = "-"
        elif value < 0:
            sign, value = "-", -value
        if self.unit == "h":
            return f"{sign}{value / 3600:.3g}"
        if self.unit == "m":
            return f"{sign}{value / 60:.3g}"
        if self.unit == "s":
            return f"{sign}{value:.3g}"
        if self.unit == "ms":
            return f"{sign}{value * 1000:.3g}"
        span = vmax - vmin
        if span >= 3600:
            h, rest = divmod(value, 3600)
            m, s = divmod(rest, 60)
            return f"{sign}{int(h):d}:{int(m):02d}:{int(s):02d}"
        if span >= 60:
            m, s = divmod(value, 60)
            return f"{sign}{int(m):d}:{int(s):02d}"
        if span >= 1:
            return f"{sign}{value:.2g}" if value < 10 else f"{sign}{value:.0f}"
        return f"{sign}{value:.3g}"


class _SpanFormatter(mticker.Formatter):
    """Base of the formatters that hide minor labels when the view covers many octaves."""

    def __init__(self, major: bool = True) -> None:
        self.major = major

    def _skip(self) -> bool:
        axis: Any = self.axis
        vmin, vmax = sorted(axis.get_view_interval())
        return not self.major and vmin > 0 and np.log2(vmax / vmin) > 2.0

    def _name(self, x: float) -> str:
        raise NotImplementedError

    def __call__(self, x: float, pos: int | None = None) -> str:
        if x <= 0 or self._skip():
            return ""
        return self._name(x)


class LogHzFormatter(_SpanFormatter):
    """Formats frequencies on a logarithmic axis."""

    def _name(self, x: float) -> str:
        return f"{x:g}"


class NoteFormatter(_SpanFormatter):
    """Formats frequencies as note names.

    Args:
        octave: Include the octave.
        major: These are the major ticks; minor ticks hide when the view spans many octaves.
        key: The key that decides the spelling, such as ``"Eb:min"``.
        unicode: Unicode accidentals; otherwise ASCII.
    """

    def __init__(self, octave: bool = True, major: bool = True, key: str = "C:maj", unicode: bool = True) -> None:
        super().__init__(major)
        self.octave = octave
        self.key = key
        self.unicode = unicode

    def _name(self, x: float) -> str:
        axis: Any = self.axis
        vmin, vmax = sorted(axis.get_view_interval())
        cents = vmin > 0 and np.log2(vmax / vmin) < 1.0 / 6.0
        return _midi_name(_hz_to_midi(x), self.key, self.octave, self.unicode, cents)


class SvaraFormatter(_SpanFormatter):
    """Formats frequencies as svara.

    Args:
        Sa: The frequency of Sa, in Hz.
        octave: Mark the octaves next to Sa's.
        major: As for :class:`NoteFormatter`.
        abbr: Short forms.
        mela: A melakarta raga, by number or name, for Carnatic names; None for Hindustani.
        unicode: Dots and subscripts; otherwise ASCII.
    """

    def __init__(
        self,
        Sa: float,
        octave: bool = True,
        major: bool = True,
        abbr: bool = False,
        mela: int | str | None = None,
        unicode: bool = True,
    ) -> None:
        super().__init__(major)
        self.Sa = Sa
        self.octave = octave
        self.abbr = abbr
        self.mela = mela
        self.unicode = unicode

    def _name(self, x: float) -> str:
        kw: dict[str, Any] = dict(Sa=self.Sa, abbr=self.abbr, octave=self.octave, unicode=self.unicode)
        if self.mela is None:
            return _notation.hz_to_svara_h(x, **kw)
        return _notation.hz_to_svara_c(x, mela=self.mela, **kw)


class FJSFormatter(_SpanFormatter):
    """Formats frequencies with Functional Just System names.

    Args:
        fmin: The lowest frequency, in Hz.
        n_bins: Bins of the axis.
        bins_per_octave: Bins per octave.
        intervals: A tuning system name or the ratios of one octave.
        major: As for :class:`NoteFormatter`.
        unison: The note of ``fmin``. Default: the nearest equal-tempered note.
        unicode: Unicode accidentals and scripts; otherwise ASCII.
    """

    def __init__(
        self,
        *,
        fmin: float,
        n_bins: int,
        bins_per_octave: int,
        intervals: str | Sequence[float],
        major: bool = True,
        unison: str | None = None,
        unicode: bool = True,
    ) -> None:
        super().__init__(major)
        self.fmin = fmin
        self.unison = unison
        self.unicode = unicode
        self.frequencies = _notation.interval_frequencies(
            n_bins, fmin=fmin, intervals=intervals, bins_per_octave=bins_per_octave
        )

    def _name(self, x: float) -> str:
        if not (self.frequencies[0] <= x <= self.frequencies[-1] * 1.001):
            return ""
        nearest = self.frequencies[np.argmin(np.abs(np.log2(self.frequencies / x)))]
        return _notation.hz_to_fjs(float(nearest), fmin=self.fmin, unison=self.unison, unicode=self.unicode)


class EngFormatter(_SpanFormatter):
    """Formats values with an engineering prefix, as ``1.2k``."""

    def __init__(self, unit: str = "", places: int | None = None, sep: str = " ", major: bool = True) -> None:
        super().__init__(major)
        self._inner = mticker.EngFormatter(unit=unit, places=places, sep=sep)

    def set_axis(self, axis: Any) -> None:
        super().set_axis(axis)
        self._inner.set_axis(axis)

    def _name(self, x: float) -> str:
        return self._inner(x)


class ChromaFormatter(mticker.Formatter):
    """Formats pitch-class numbers as note names in a key."""

    def __init__(self, key: str = "C:maj", unicode: bool = True) -> None:
        self.key = key
        self.unicode = unicode

    def __call__(self, x: float, pos: int | None = None) -> str:
        return _midi_name(x, self.key, False, self.unicode)


class ChromaSvaraFormatter(mticker.Formatter):
    """Formats pitch-class numbers as svara of a scale on Sa.

    Args:
        Sa: Pitch class of Sa. Default 0.
        mela: A melakarta raga, by number or name, for Carnatic names; None for Hindustani.
        abbr: Short forms.
        unicode: Subscripts; otherwise ASCII.
    """

    def __init__(self, Sa: float | None = None, mela: int | str | None = None, abbr: bool = True, unicode: bool = True) -> None:
        self.Sa = 0.0 if Sa is None else Sa
        self.mela = mela
        self.abbr = abbr
        self.unicode = unicode

    def __call__(self, x: float, pos: int | None = None) -> str:
        kw: dict[str, Any] = dict(Sa=self.Sa, abbr=self.abbr, octave=False, unicode=self.unicode)
        if self.mela is None:
            return _notation.midi_to_svara_h(x, **kw)
        return _notation.midi_to_svara_c(x, mela=self.mela, **kw)


class ChromaFJSFormatter(mticker.Formatter):
    """Formats chroma bins with Functional Just System names.

    Args:
        intervals: A tuning system name or the ratios of one octave.
        unison: The note of ratio 1.
        unicode: Unicode accidentals and scripts; otherwise ASCII.
        bins_per_octave: Needed with a tuning system name.
    """

    def __init__(
        self,
        *,
        intervals: str | Sequence[float],
        unison: str = "C",
        unicode: bool = True,
        bins_per_octave: int | None = None,
    ) -> None:
        if isinstance(intervals, str):
            if bins_per_octave is None:
                raise ValueError("bins_per_octave: needed with a tuning system name")
            ratios = _notation.interval_frequencies(bins_per_octave, fmin=1.0, intervals=intervals, bins_per_octave=bins_per_octave)
        else:
            ratios = np.sort(np.asarray(intervals, dtype=np.float64))
        self.names = _notation.interval_to_fjs(ratios, unison=unison, unicode=unicode)

    def __call__(self, x: float, pos: int | None = None) -> str:
        index = int(np.round(x))
        if not 0 <= index < len(self.names):
            return ""
        return str(self.names[index])


class TonnetzFormatter(mticker.Formatter):
    """Formats the six tonal-centroid dimensions."""

    LABELS = ("5$_x$", "5$_y$", "m3$_x$", "m3$_y$", "M3$_x$", "M3$_y$")

    def __call__(self, x: float, pos: int | None = None) -> str:
        index = int(np.round(x))
        return self.LABELS[index] if 0 <= index < 6 else ""


class F0Transform(mtransforms.Transform):
    """Moves a waveform on a frequency axis so that it rides on a pitch track.

    A point ``(t, a)`` maps to the frequency ``f0(t) * 2 ** (a * scale)``, so the waveform's
    amplitude displaces the pitch contour by a fraction of an octave.

    Args:
        f0: Fundamental frequency per frame, in Hz; NaN where unvoiced.
        sample_rate: Sample rate of the frames.
        hop_length: Samples between frames.
        scale: Octaves of displacement per unit amplitude.
        offset: Time of the first frame, in seconds.
        transpose: The time axis is vertical.
    """

    input_dims = 2
    output_dims = 2
    is_separable = False

    def __init__(
        self,
        f0: NDArray[Any],
        *,
        sample_rate: float = 22050,
        hop_length: int = 512,
        scale: float = 1.0 / 12.0,
        offset: float = 0.0,
        transpose: bool = False,
    ) -> None:
        super().__init__()
        self.f0 = np.asarray(f0, dtype=np.float64)
        self.times = offset + np.arange(self.f0.size) * hop_length / sample_rate
        self.scale = scale
        self.transpose = transpose

    def transform_non_affine(self, values: Any) -> Any:  # noqa: D102
        points = np.asarray(values, dtype=np.float64)
        t = points[:, 1] if self.transpose else points[:, 0]
        a = points[:, 0] if self.transpose else points[:, 1]
        f = np.interp(t, self.times, self.f0) * 2.0 ** (a * self.scale)
        return np.column_stack((f, t) if self.transpose else (t, f))


def infer_colormap(
    data: NDArray[Any],
    *,
    robust: bool = True,
    sequential: Any = "magma",
    boolean: Any = "gray_r",
    diverging: Any = "coolwarm",
    threshold: float = 0.0,
) -> mcolors.Colormap:
    """Chooses a colormap for an array.

    Boolean data gets ``boolean``; data on both sides of ``threshold`` gets ``diverging``;
    anything else gets ``sequential``.

    Args:
        data: The array; NaNs are ignored.
        robust: Judge by the 2nd and 98th percentiles rather than the extremes.
        sequential: Colormap for one-signed data.
        boolean: Colormap for boolean data.
        diverging: Colormap for two-signed data.
        threshold: The value that decides the sign.

    Returns:
        The colormap.
    """
    from matplotlib import colormaps

    def resolve(c: Any) -> mcolors.Colormap:
        return colormaps[c] if isinstance(c, str) else c

    arr = np.atleast_1d(data)
    if arr.dtype == np.bool_:
        return resolve(boolean)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return resolve(sequential)
    # ``finite`` is a copy made above, so the percentile may reorder it.
    lo, hi = np.percentile(finite, [2, 98], overwrite_input=True) if robust else (finite.min(), finite.max())
    if lo >= threshold or hi <= threshold:
        return resolve(sequential)
    return resolve(diverging)


_DB_RE = re.compile(r"^dB(FS)?(\[(power)?(,?)([^\]]*)\])?$")


def _scale_values(
    data: NDArray[Any],
    scale: str | None,
    top_db: float | None,
    x: NDArray[Any] | None,
    y: NDArray[Any] | None,
    sequential: Any,
    boolean: Any,
    diverging: Any,
    cyclic: Any,
    threshold: float,
) -> tuple[NDArray[Any], Any, dict[str, float]]:
    limits: dict[str, float] = {}
    if scale is None:
        if np.iscomplexobj(data):
            raise ValueError("data: complex input needs value_scale='phase' or 'dB'; take a magnitude for anything else")
        return data, infer_colormap(data, sequential=sequential, boolean=boolean, diverging=diverging, threshold=threshold), limits
    if scale in ("phase", "dphase", "dphase_t"):
        if not np.iscomplexobj(data):
            raise ValueError(f"data: value_scale={scale!r} needs complex input")
        out = np.angle(data)
        if scale != "phase":
            along_t = scale == "dphase"
            coords = x if along_t else y
            other = y if along_t else x
            if coords is None or other is None:
                raise ValueError(f"value_scale={scale!r} needs a time and a frequency axis")
            axis = 1 if along_t else 0
            dt = np.diff(coords)
            freq = other if along_t else other
            expected = 2.0 * np.pi * np.expand_dims(freq, 1 if along_t else 0)
            expected = expected * np.expand_dims(dt, 0 if along_t else 1)
            delta = np.diff(out, axis=axis) - expected
            out = np.angle(np.exp(1j * delta))
            first = [slice(None)] * 2
            first[axis] = slice(0, 1)
            out = np.concatenate([np.zeros_like(out[tuple(first)]), out], axis=axis)
        limits = {"vmin": -np.pi, "vmax": np.pi}
        return out, cyclic, limits
    m = _DB_RE.match(scale)
    if m is None:
        raise ValueError(f"value_scale: unknown {scale!r}")
    full_scale, _, power, _, value = m.groups()
    magnitude = np.abs(data)
    if full_scale:
        if value:
            raise ValueError(f"value_scale: {scale!r} takes no reference value")
        reference = float(np.max(magnitude)) if magnitude.size else 1.0
    else:
        reference = float(value) if value else 1.0
    factor = 10.0 if power else 20.0
    floor = 1e-10 if power else 1e-5
    out = factor * np.log10(np.maximum(magnitude, floor * max(reference, 1e-300)) / reference)
    if top_db is not None:
        if top_db < 0:
            raise ValueError("top_db: must not be negative")
        out = np.maximum(out, out.max() - top_db)
    return out, infer_colormap(out, sequential=sequential, boolean=boolean, diverging=diverging, threshold=threshold), limits


def _axis_coords(kind: str | None, n: int, **kw: Any) -> NDArray[Any]:
    sr = kw["sample_rate"]
    if kind in _FFT_AXES:
        n_fft = kw["n_fft"] if kw["n_fft"] is not None else 2 * (n - 1)
        return np.arange(n) * sr / max(n_fft, 1)
    if kind in _MEL_AXES:
        fmax = kw["fmax"] if kw["fmax"] is not None else sr / 2
        fmin = kw["fmin"] if kw["fmin"] is not None else 0.0
        return _utils.mel_frequencies(n, fmin, fmax, htk=kw["htk"])
    if kind in _CQT_AXES:
        fmin = kw["fmin"] if kw["fmin"] is not None else 32.70319566257483
        return _utils.cqt_frequencies(n, fmin=fmin, bins_per_octave=kw["bins_per_octave"], tuning=kw["tuning"])
    if kind in _VQT_AXES:
        if kw["intervals"] is None:
            raise ValueError(f"intervals: needed for the axis {kind!r}")
        fmin = kw["fmin"] if kw["fmin"] is not None else 32.70319566257483
        return _notation.interval_frequencies(n, fmin=fmin, intervals=kw["intervals"], bins_per_octave=kw["bins_per_octave"])
    if kind in ("chroma", "chroma_h", "chroma_c"):
        return np.arange(n) * 12.0 / kw["bins_per_octave"]
    if kind in _TIME_AXES or kind in _LAG_AXES:
        return np.arange(n) * kw["hop_length"] / sr
    if kind == "tempo":
        # Lags to tempo: lag 0 has no tempo, so it takes the next octave up.
        lags = np.arange(n, dtype=np.float64)
        lags[0] = 0.5
        return 60.0 * sr / (kw["hop_length"] * lags)
    if kind == "fourier_tempo":
        win = kw["win_length"] if kw["win_length"] is not None else 2 * (n - 1)
        return np.arange(n) * (sr * 60.0 / kw["hop_length"]) / max(win, 1)
    if kind in ("chroma_fjs", "tonnetz", "frames", "off", "none", None):
        return np.arange(n, dtype=np.float64)
    raise ValueError(f"axis: unknown type {kind!r}; expected one of {AXIS_TYPES}")


def _third_octave_ticks(lo: float, hi: float) -> NDArray[Any]:
    k = np.arange(-40, 41)
    centres = 1000.0 * 2.0 ** (k / 3.0)
    return centres[(centres >= max(lo, 1e-6)) & (centres <= hi)]


def _decorate(axis: Any, kind: str | None, coords: NDArray[Any], **kw: Any) -> None:
    """Sets the scale, ticks and label of one axis."""
    ax = axis.axes
    which = "y" if axis is ax.yaxis else "x"
    set_scale = ax.set_yscale if which == "y" else ax.set_xscale
    set_limit = ax.set_ylim if which == "y" else ax.set_xlim
    label = axis.set_label_text
    key, unicode = kw["key"], kw["unicode"]
    if kind in ("off", "none", None):
        axis.set_major_locator(mticker.NullLocator())
        axis.set_minor_locator(mticker.NullLocator())
        label("")
        if kind in ("off", "none"):
            axis.set_visible(kind != "off")
        return
    if kind in _TIME_AXES or kind in _LAG_AXES:
        lag = kind in _LAG_AXES
        unit = {"h": "h", "m": "m", "s": "s", "ms": "ms"}.get(kind.replace("lag_", ""))
        axis.set_major_formatter(TimeFormatter(lag=lag, unit=unit))
        names = {"h": "Hours", "m": "Minutes", "s": "Seconds", "ms": "Milliseconds"}
        prefix = "Lag" if lag else "Time"
        label(f"{prefix} ({names[unit]})" if unit else prefix)
        return
    if kind == "frames":
        axis.set_major_locator(mticker.MaxNLocator(integer=True))
        label("Frames")
        return
    if kind == "tonnetz":
        axis.set_major_locator(mticker.FixedLocator(np.asarray(np.arange(6)).tolist()))
        axis.set_major_formatter(TonnetzFormatter())
        label("Tonnetz")
        return
    if kind in ("chroma", "chroma_h", "chroma_c"):
        axis.set_major_locator(mticker.FixedLocator(np.asarray(coords).tolist()))
        if kind == "chroma":
            axis.set_major_formatter(ChromaFormatter(key=key, unicode=unicode))
        else:
            sa = kw["Sa"]
            if kind == "chroma_h":
                sa = 0.0 if sa is None else sa
                axis.set_major_formatter(ChromaSvaraFormatter(Sa=sa, unicode=unicode))
            else:
                axis.set_major_formatter(ChromaSvaraFormatter(Sa=sa, mela=kw["mela"], unicode=unicode))
        label("Pitch class" if kind == "chroma" else "Svara")
        return
    if kind == "chroma_fjs":
        axis.set_major_locator(mticker.FixedLocator(np.asarray(coords).tolist()))
        axis.set_major_formatter(
            ChromaFJSFormatter(intervals=kw["intervals"], unison=kw["unison"] or "C", unicode=unicode, bins_per_octave=kw["bins_per_octave"])
        )
        label("Pitch class")
        return
    if kind in ("tempo", "fourier_tempo"):
        if kind == "tempo":
            set_scale("log", base=2)
        set_limit(kw["tempo_min"], kw["tempo_max"])
        axis.set_major_formatter(mticker.ScalarFormatter())
        label("BPM")
        return
    logarithmic = kind in _CQT_AXES or kind in _VQT_AXES or kind in ("log", "log_oct3")
    if kind in _MEL_AXES:
        set_scale("symlog", base=2, linthresh=1000.0, linscale=0.5)
    elif kind in ("log", "log_oct3"):
        set_scale("symlog", base=2, linthresh=kw["sample_rate"] / max(kw["n_fft"] or 2 * (coords.size - 1), 1), linscale=0.5)
    elif logarithmic:
        set_scale("log", base=2)
    if kind.endswith("oct3"):
        axis.set_major_locator(mticker.FixedLocator(np.asarray(_third_octave_ticks(float(coords[1 if coords[0] <= 0 else 0]), float(coords[-1]))).tolist()))
        axis.set_major_formatter(EngFormatter(unit="Hz"))
        label("Hz")
        return
    if kind.endswith("_note"):
        axis.set_major_locator(mticker.FixedLocator(np.asarray(_utils.cqt_frequencies(12 * 12, fmin=8.175798915643707)).tolist()))
        axis.set_major_formatter(NoteFormatter(key=key, unicode=unicode))
        label("Note")
    elif kind.endswith("_svara"):
        sa = kw["Sa"] if kw["Sa"] is not None else (float(coords[1 if coords[0] <= 0 else 0]))
        axis.set_major_locator(mticker.FixedLocator(np.asarray(sa * 2.0 ** (np.arange(-48, 96) / 12.0)).tolist()))
        axis.set_major_formatter(SvaraFormatter(Sa=sa, mela=kw["mela"], unicode=unicode))
        label("Svara")
    elif kind == "vqt_fjs":
        axis.set_major_locator(mticker.FixedLocator(np.asarray(coords).tolist()))
        axis.set_major_formatter(
            FJSFormatter(fmin=float(coords[0]), n_bins=coords.size, bins_per_octave=kw["bins_per_octave"], intervals=kw["intervals"], unison=kw["unison"], unicode=unicode)
        )
        label("Note")
    else:
        axis.set_major_formatter(mticker.ScalarFormatter() if not logarithmic else LogHzFormatter())
        label("Hz")
    if kind in ("linear", "fft", "hz", "fft_note", "fft_svara", "oct3"):
        set_limit(float(coords[0]), float(coords[-1]))


def plot_matrix(
    data: NDArray[Any],
    *,
    x_axis: str | None = None,
    y_axis: str | None = None,
    x_coords: NDArray[Any] | None = None,
    y_coords: NDArray[Any] | None = None,
    value_scale: str | None = None,
    sample_rate: float = 22050,
    hop_length: int = 512,
    n_fft: int | None = None,
    win_length: int | None = None,
    fmin: float | None = None,
    fmax: float | None = None,
    tempo_min: float | None = 16,
    tempo_max: float | None = 480,
    tuning: float = 0.0,
    bins_per_octave: int = 12,
    key: str = "C:maj",
    Sa: float | None = None,
    mela: int | str | None = None,
    thaat: str | None = None,
    equal_aspect: bool = True,
    htk: bool = False,
    unicode: bool = True,
    intervals: str | Sequence[float] | None = None,
    unison: str | None = None,
    top_db: float | None = 80.0,
    sequential: Any = "magma",
    boolean: Any = "gray_r",
    diverging: Any = "coolwarm",
    cyclic: Any = "twilight_shifted",
    threshold: float = 0.0,
    title: str | None = None,
    colorbar: bool = False,
    ax: "Axes | None" = None,
    style: "MplStyle | None" = None,
    **mesh_kwargs: Any,
) -> "tuple[Any, Axes, Any]":
    """Draws a feature matrix with axes that know what they show.

    Axis types: ``time``, ``h``, ``m``, ``s``, ``ms``; ``lag``, ``lag_h``, ``lag_m``, ``lag_s``,
    ``lag_ms``; ``linear``, ``fft``, ``hz``, ``log``, ``oct3``, ``log_oct3``, ``fft_note``,
    ``fft_svara``; ``mel``, ``mel_oct3``; ``cqt``, ``cqt_hz``, ``cqt_note``, ``cqt_svara``,
    ``cqt_oct3``; ``vqt_hz``, ``vqt_note``, ``vqt_oct3``, ``vqt_fjs``; ``chroma``, ``chroma_h``,
    ``chroma_c``, ``chroma_fjs``; ``tempo``, ``fourier_tempo``; ``tonnetz``; ``frames``; ``off``.

    Args:
        data: ``(rows, columns)``, real or, for phase and dB scales, complex.
        x_axis: Type of the horizontal axis.
        y_axis: Type of the vertical axis.
        x_coords: Coordinates in place of those of the axis type.
        y_coords: Likewise.
        value_scale: ``"dB"``, ``"dB[0.1]"``, ``"dB[power]"``, ``"dB[power,0.1]"``, ``"dBFS"``,
            ``"dBFS[power]"``, ``"phase"``, ``"dphase"``, ``"dphase_t"``, or None for the values.
        sample_rate: Sample rate behind the time and frequency axes.
        hop_length: Samples between frames.
        n_fft: FFT size. Default ``2 * (rows - 1)``.
        win_length: Window of a Fourier tempogram. Default ``2 * (rows - 1)``.
        fmin: Lowest frequency of mel, constant-Q and variable-Q axes.
        fmax: Highest frequency of mel axes. Default ``sample_rate / 2``.
        tempo_min: Lowest BPM shown on tempo axes.
        tempo_max: Highest BPM shown on tempo axes.
        tuning: Offset of constant-Q bins in fractions of a bin.
        bins_per_octave: Bins per octave.
        key: Key for note and chroma names.
        Sa: Sa for svara names: a frequency on frequency axes, a pitch class on chroma axes.
        mela: Melakarta raga for Carnatic names.
        thaat: Thaat for Hindustani chroma names.
        equal_aspect: With the same time-like type on both axes, make the pixels square.
        htk: Mel axes use the HTK formula.
        unicode: Unicode symbols in names.
        intervals: Tuning system of variable-Q and FJS axes.
        unison: Note of the lowest bin for FJS names.
        top_db: Dynamic range of dB scales.
        sequential: Colormap for one-signed data.
        boolean: Colormap for boolean data.
        diverging: Colormap for two-signed data.
        cyclic: Colormap for phase.
        threshold: The value that decides whether data has two signs.
        title: Axes title.
        colorbar: Add a colorbar suited to the value scale.
        ax: Existing axes.
        style: Visual style.
        **mesh_kwargs: Passed to ``pcolormesh``; ``cmap``, ``vmin`` and ``vmax`` override.

    Returns:
        ``(fig, ax, mesh)``.
    """
    array = np.asarray(data)
    if array.ndim != 2:
        raise ValueError(f"data: expected 2 dimensions, got {array.ndim}")
    resolved = _resolve_style(style)
    fig: Any
    fig, ax = _ensure_axes(ax, resolved)
    if thaat is not None:
        degrees = _notation.thaat_to_degrees(thaat)
        del degrees
    kw: dict[str, Any] = dict(
        sample_rate=sample_rate,
        hop_length=hop_length,
        n_fft=n_fft,
        win_length=win_length,
        fmin=fmin,
        fmax=fmax,
        tempo_min=tempo_min,
        tempo_max=tempo_max,
        tuning=tuning,
        bins_per_octave=bins_per_octave,
        key=key,
        Sa=Sa,
        mela=mela,
        htk=htk,
        unicode=unicode,
        intervals=intervals,
        unison=unison,
    )
    rows, cols = array.shape
    xc = np.asarray(x_coords) if x_coords is not None else _axis_coords(x_axis, cols, **kw)
    yc = np.asarray(y_coords) if y_coords is not None else _axis_coords(y_axis, rows, **kw)
    known_x = x_coords is not None or x_axis is not None
    known_y = y_coords is not None or y_axis is not None
    values, cmap, limits = _scale_values(array, value_scale, top_db, xc if known_x else None, yc if known_y else None, sequential, boolean, diverging, cyclic, threshold)
    if values.dtype == np.bool_:
        values = values.astype(np.float64)
    mesh_kwargs.setdefault("cmap", cmap)
    mesh_kwargs.setdefault("rasterized", True)
    mesh_kwargs.setdefault("shading", "auto")
    for name, value in limits.items():
        mesh_kwargs.setdefault(name, value)
    mesh = ax.pcolormesh(xc, yc, values, **mesh_kwargs)
    _decorate(ax.xaxis, x_axis, xc, **kw)
    _decorate(ax.yaxis, y_axis, yc, **kw)
    if equal_aspect and x_axis == y_axis and x_axis in (*_TIME_AXES, *_LAG_AXES, *_CHROMA_AXES):
        ax.set_aspect("equal")
    if colorbar:
        if value_scale and value_scale.startswith("dB"):
            add_db_colorbar(mesh, ax=ax)
        elif value_scale:
            add_phase_colorbar(mesh, ax=ax)
        else:
            ax.figure.colorbar(mesh, ax=ax)
    if title is not None:
        ax.set_title(title)
    _apply_style(fig, ax, resolved)
    return fig, ax, mesh


def _as_samples(y: Any, sample_rate: float | None) -> tuple[NDArray[Any], float]:
    if hasattr(y, "sample_rate") and hasattr(y, "to_numpy"):
        return np.asarray(y.to_numpy()), float(y.sample_rate)
    if sample_rate is None:
        raise ValueError("sample_rate: needed with an array")
    return np.asarray(y), float(sample_rate)


def _envelope(x: NDArray[Any], hop: int) -> NDArray[Any]:
    """The largest magnitude of each block of ``hop`` samples, repeated for fill plots."""
    n = x.size // hop if hop > 0 else 0
    if n == 0:
        return np.abs(x)
    blocks = x[: n * hop].reshape(n, hop)
    return np.maximum(blocks.max(axis=1), -blocks.min(axis=1))


class AdaptiveWaveform:
    """A waveform that shows samples when zoomed in and an envelope otherwise.

    It listens to the axes' limits. Attributes: ``steps`` and ``envelope``, the artists.
    Call :meth:`disconnect` to stop updating.
    """

    def __init__(self, ax: "Axes", times: NDArray[Any], y: NDArray[Any], steps: Any, envelope: Any, sample_rate: float, max_samples: int, transpose: bool) -> None:
        self.times = times
        self.samples = y
        self.steps = steps
        self.envelope = envelope
        self.sample_rate = sample_rate
        self.max_samples = max_samples
        self.transpose = transpose
        self.ax = ax
        self._callback = ax.callbacks.connect("ylim_changed" if transpose else "xlim_changed", self.update)
        self.update(ax)

    def update(self, ax: "Axes") -> None:
        """Chooses the artist for the current view."""
        lo, hi = ax.get_ylim() if self.transpose else ax.get_xlim()
        lo, hi = sorted((lo, hi))
        if (hi - lo) * self.sample_rate < self.max_samples:
            first = max(0, int(np.searchsorted(self.times, lo)) - 1)
            last = min(self.times.size, int(np.searchsorted(self.times, hi)) + 1)
            xs, ys = self.times[first:last], self.samples[first:last]
            if self.transpose:
                self.steps.set_data(ys, xs)
            else:
                self.steps.set_data(xs, ys)
            self.steps.set_visible(True)
            self.envelope.set_visible(False)
        else:
            self.steps.set_visible(False)
            self.envelope.set_visible(True)
        ax.figure.canvas.draw_idle()

    def disconnect(self) -> None:
        """Stops following the axes."""
        self.ax.callbacks.disconnect(self._callback)


def plot_waveform_adaptive(
    y: Any,
    *,
    sample_rate: float | None = None,
    max_points: int = 11025,
    axis: str | None = "time",
    offset: float = 0.0,
    marker: Any = "",
    where: str = "post",
    label: str | None = None,
    transpose: bool = False,
    mask: Any = None,
    invert: bool = False,
    invert_color: Any = None,
    title: str | None = None,
    ax: "Axes | None" = None,
    style: "MplStyle | None" = None,
    **kwargs: Any,
) -> "tuple[Any, Axes, AdaptiveWaveform]":
    """Draws a waveform that shows samples when zoomed in and an envelope otherwise.

    Args:
        y: ``AudioSamples`` or an array, mono or ``(channels, samples)``; channels are averaged.
        sample_rate: Needed with an array.
        max_points: The most samples drawn at once; a wider view shows the envelope.
        axis: Type of the time axis, as for :func:`plot_matrix`, or None.
        offset: Time of the first sample, in seconds.
        marker: Marker for the sample view.
        where: Step placement for the sample view: ``"pre"``, ``"post"`` or ``"mid"``.
        label: Legend label.
        transpose: Draw time on the vertical axis.
        mask: Boolean per sample; only true samples are drawn.
        invert: Draw the negative half in ``invert_color``.
        invert_color: Colour of the negative half. Default: the waveform's colour with half
            its opacity.
        title: Axes title.
        ax: Existing axes.
        style: Visual style.
        **kwargs: Colour and other properties for both artists.

    Returns:
        ``(fig, ax, waveform)``.
    """
    samples, sr = _as_samples(y, sample_rate)
    if samples.ndim > 1:
        samples = samples.mean(axis=0)
    samples = samples.astype(np.float64, copy=False)
    if max_points <= 0:
        raise ValueError("max_points: must be positive")
    if mask is not None:
        samples = np.where(np.asarray(mask, dtype=bool), samples, np.nan)
    resolved = _resolve_style(style)
    fig: Any
    fig, ax = _ensure_axes(ax, resolved)
    times = offset + np.arange(samples.size) / sr
    step_kwargs = {k: v for k, v in kwargs.items() if k not in ("alpha",)}
    plot = ax.plot if not transpose else ax.plot
    steps = plot(times, samples, drawstyle=f"steps-{where}", marker=marker, label=None, **step_kwargs)[0]
    if transpose:
        steps.set_data(samples, times)
    color = steps.get_color()
    hop = max(1, samples.size // max(1, max_points // 2))
    env = _envelope(np.nan_to_num(samples), hop)
    env_times = offset + (np.arange(env.size) + 0.5) * hop / sr
    fill_kwargs = dict(kwargs)
    fill_kwargs.setdefault("color", color)
    negative = invert_color if invert_color is not None else None
    if invert:
        pos = (env_times, np.zeros_like(env), env)
        neg = (env_times, -env, np.zeros_like(env))
        if transpose:
            envelope = ax.fill_betweenx(*pos, label=label, **fill_kwargs)
            lower = dict(fill_kwargs)
            lower["color"] = negative if negative is not None else color
            lower.setdefault("alpha", 0.5)
            ax.fill_betweenx(*neg, **lower)
        else:
            envelope = ax.fill_between(*pos, label=label, **fill_kwargs)
            lower = dict(fill_kwargs)
            lower["color"] = negative if negative is not None else color
            lower.setdefault("alpha", 0.5)
            ax.fill_between(*neg, **lower)
    elif transpose:
        envelope = ax.fill_betweenx(env_times, -env, env, label=label, **fill_kwargs)
    else:
        envelope = ax.fill_between(env_times, -env, env, label=label, **fill_kwargs)
    span = (offset, offset + samples.size / sr)
    if transpose:
        ax.set_ylim(*span)
    else:
        ax.set_xlim(*span)
    if axis is not None:
        _decorate(ax.yaxis if transpose else ax.xaxis, axis, times, sample_rate=sr, n_fft=None, key="C:maj", unicode=True)
    waveform = AdaptiveWaveform(ax, times, samples, steps, envelope, sr, max_points, transpose)
    if title is not None:
        ax.set_title(title)
    _apply_style(fig, ax, resolved)
    return fig, ax, waveform


def plot_waveform_bars(
    y: Any,
    *,
    sample_rate: float | None = None,
    n_bars: int = 100,
    gap_ratio: float = 0.4,
    rounding_ratio: float = 0.5,
    axis: str | None = "time",
    offset: float = 0.0,
    invert: bool = False,
    invert_color: Any = None,
    transpose: bool = False,
    label: str | None = None,
    title: str | None = None,
    ax: "Axes | None" = None,
    style: "MplStyle | None" = None,
    **patch_kwargs: Any,
) -> "tuple[Any, Axes, PatchCollection]":
    """Draws a waveform as bars of its envelope, for small figures.

    Args:
        y: ``AudioSamples`` or an array, mono or ``(channels, samples)``; channels are averaged.
        sample_rate: Needed with an array.
        n_bars: Number of bars.
        gap_ratio: Fraction of each bar's slot that stays empty, in ``[0, 1)``.
        rounding_ratio: Rounding of the corners, from 0 to 1 of half the bar's width.
        axis: Type of the time axis, or None.
        offset: Time of the first sample, in seconds.
        invert: Draw only the positive half, and the negative half in ``invert_color``.
        invert_color: Colour of the negative half.
        transpose: Draw time on the vertical axis.
        label: Legend label.
        title: Axes title.
        ax: Existing axes.
        style: Visual style.
        **patch_kwargs: Properties of the bars.

    Returns:
        ``(fig, ax, bars)``.
    """
    samples, sr = _as_samples(y, sample_rate)
    if samples.ndim > 1:
        samples = samples.mean(axis=0)
    if n_bars < 1:
        raise ValueError("n_bars: must be positive")
    if not 0.0 <= gap_ratio < 1.0:
        raise ValueError("gap_ratio: must lie in [0, 1)")
    if not 0.0 <= rounding_ratio <= 1.0:
        raise ValueError("rounding_ratio: must lie in [0, 1]")
    resolved = _resolve_style(style)
    fig: Any
    fig, ax = _ensure_axes(ax, resolved)
    hop = max(1, samples.size // n_bars)
    env = _envelope(samples.astype(np.float64, copy=False), hop)[:n_bars]
    slot = hop / sr
    width = slot * (1.0 - gap_ratio)
    heights = np.maximum(env, 1e-6)
    patches = []
    colors = []
    default_color = matplotlib.rcParams["axes.prop_cycle"].by_key()["color"][0]
    color = patch_kwargs.pop("color", patch_kwargs.pop("facecolor", default_color))
    radius = 0.5 * width * rounding_ratio
    for i, h in enumerate(heights):
        start = offset + i * slot + 0.5 * (slot - width)
        spans = [(0.0, h, color), (-h, h, invert_color if invert_color is not None else color)] if invert else [(-h, 2 * h, color)]
        for low, tall, c in spans:
            if transpose:
                box = FancyBboxPatch((low, start), tall, width, boxstyle=f"round,pad=0,rounding_size={radius}", mutation_aspect=1.0)
            else:
                box = FancyBboxPatch((start, low), width, tall, boxstyle=f"round,pad=0,rounding_size={radius}", mutation_aspect=1.0)
            patches.append(box)
            colors.append(c)
    bars = PatchCollection(patches, facecolors=colors, label=label, **patch_kwargs)
    ax.add_collection(bars)
    span = (offset, offset + env.size * slot)
    peak = float(heights.max())
    if transpose:
        ax.set_ylim(*span)
        ax.set_xlim(-peak * 1.05, peak * 1.05)
    else:
        ax.set_xlim(*span)
        ax.set_ylim(-peak * 1.05, peak * 1.05)
    if axis is not None:
        _decorate(ax.yaxis if transpose else ax.xaxis, axis, np.array(span), sample_rate=sr, n_fft=None, key="C:maj", unicode=True)
    if title is not None:
        ax.set_title(title)
    _apply_style(fig, ax, resolved)
    return fig, ax, bars


def plot_waveform_f0(
    y: Any,
    *,
    f0: NDArray[Any],
    sample_rate: float | None = None,
    hop_length: int = 512,
    bins_per_octave: int = 12,
    time_axis: str = "time",
    freq_axis: str = "cqt_note",
    offset: float = 0.0,
    key: str = "C:maj",
    Sa: float | None = None,
    mela: int | str | None = None,
    thaat: str | None = None,
    unicode: bool = True,
    method: str = "adaptive",
    transpose: bool = False,
    title: str | None = None,
    ax: "Axes | None" = None,
    style: "MplStyle | None" = None,
    **kwargs: Any,
) -> "tuple[Any, Axes, Any]":
    """Draws a waveform riding on its pitch track.

    The amplitude displaces the fundamental frequency by up to a bin, so the waveform's
    envelope and its pitch contour can be read together.

    Args:
        y: ``AudioSamples`` or an array.
        f0: Fundamental frequency per frame, in Hz; NaN where unvoiced.
        sample_rate: Needed with an array.
        hop_length: Samples between frames of ``f0``.
        bins_per_octave: Displacement per unit amplitude: one bin.
        time_axis: Type of the time axis.
        freq_axis: Type of the frequency axis, as for :func:`plot_matrix`.
        offset: Time of the first sample, in seconds.
        key: Key for note names.
        Sa: Sa for svara names.
        mela: Melakarta raga for Carnatic names.
        thaat: Thaat for Hindustani names.
        unicode: Unicode symbols in names.
        method: ``"adaptive"`` or ``"bars"``.
        transpose: Draw time on the vertical axis.
        title: Axes title.
        ax: Existing axes.
        style: Visual style.
        **kwargs: Passed to the waveform.

    Returns:
        ``(fig, ax, artist)``.
    """
    samples, sr = _as_samples(y, sample_rate)
    if method not in ("adaptive", "bars"):
        raise ValueError(f"method: unknown {method!r}; expected 'adaptive' or 'bars'")
    pitch = np.asarray(f0, dtype=np.float64)
    if samples.ndim > 1:
        samples = samples.mean(axis=0)
    peak = float(np.max(np.abs(samples))) if samples.size else 1.0
    scaled = samples / peak if peak > 0 else samples
    transform = F0Transform(
        pitch, sample_rate=sr, hop_length=hop_length, scale=1.0 / bins_per_octave, offset=offset, transpose=transpose
    )
    resolved = _resolve_style(style)
    fig: Any
    fig, ax = _ensure_axes(ax, resolved)
    artist: Any
    if method == "adaptive":
        fig, ax, artist = plot_waveform_adaptive(scaled, sample_rate=sr, axis=None, offset=offset, transpose=transpose, ax=ax, style=resolved, transform=transform + ax.transData, **kwargs)
    else:
        fig, ax, artist = plot_waveform_bars(scaled, sample_rate=sr, axis=None, offset=offset, transpose=transpose, ax=ax, style=resolved, transform=transform + ax.transData, **kwargs)
    voiced = pitch[np.isfinite(pitch) & (pitch > 0)]
    frequency_axis = ax.xaxis if transpose else ax.yaxis
    time_ax = ax.yaxis if transpose else ax.xaxis
    if voiced.size:
        span = (float(voiced.min()) * 2.0 ** (-2.0 / 12), float(voiced.max()) * 2.0 ** (2.0 / 12))
        coords = np.geomspace(span[0], span[1], 12)
        kw: dict[str, Any] = dict(sample_rate=sr, n_fft=None, key=key, unicode=unicode, Sa=Sa, mela=mela, bins_per_octave=bins_per_octave, intervals=None, unison=None)
        _decorate(frequency_axis, freq_axis, coords, **kw)
        (ax.set_xlim if transpose else ax.set_ylim)(*span)
    if thaat is not None:
        _notation.thaat_to_degrees(thaat)
    _decorate(time_ax, time_axis, np.array([offset]), sample_rate=sr, n_fft=None, key=key, unicode=unicode)
    if title is not None:
        ax.set_title(title)
    return fig, ax, artist


def add_db_colorbar(mappable: Any, *, ax: "Axes | None" = None, fig: Any = None, format: Any = "% -3.f", **kwargs: Any) -> Any:
    """Adds a colorbar labelled in decibels.

    Args:
        mappable: The image or mesh.
        ax: The axes to take room from. Default: the mappable's.
        fig: The figure. Default: the mappable's.
        format: Tick format.
        **kwargs: Passed to ``Figure.colorbar``.

    Returns:
        The colorbar.
    """
    target = ax if ax is not None else mappable.axes
    figure = fig if fig is not None else target.figure
    bar = figure.colorbar(mappable, ax=target, format=format, **kwargs)
    bar.set_label("dB")
    return bar


def add_phase_colorbar(mappable: Any, *, numticks: int = 9, ax: "Axes | None" = None, fig: Any = None, **kwargs: Any) -> Any:
    """Adds a colorbar with ticks at multiples of π.

    Args:
        mappable: The image or mesh, in radians.
        numticks: Number of ticks.
        ax: The axes to take room from. Default: the mappable's.
        fig: The figure. Default: the mappable's.
        **kwargs: Passed to ``Figure.colorbar``.

    Returns:
        The colorbar.
    """
    if numticks < 2:
        raise ValueError("numticks: must be at least 2")
    target = ax if ax is not None else mappable.axes
    figure = fig if fig is not None else target.figure

    def name(x: float, pos: int | None = None) -> str:
        fraction = x / np.pi
        if abs(fraction) < 1e-9:
            return "0"
        from fractions import Fraction

        f = Fraction(fraction).limit_denominator(16)
        sign = "-" if f < 0 else ""
        n, d = abs(f.numerator), f.denominator
        top = "" if n == 1 else str(n)
        return f"{sign}{top}π" if d == 1 else f"{sign}{top}π/{d}"

    bar = figure.colorbar(mappable, ax=target, ticks=np.linspace(-np.pi, np.pi, numticks), format=mticker.FuncFormatter(name), **kwargs)
    return bar


def plot_grid(
    kind: str,
    *data: Any,
    axes: Any = None,
    fig: Any = None,
    orient: str = "v",
    shared: Any = None,
    fig_kw: dict[str, Any] | None = None,
    sharex: bool = True,
    sharey: bool = True,
    label_outer: bool = True,
    labels: Sequence[str | None] | None = None,
    titles: Sequence[str | None] | None = None,
    prop_cycle: Any = None,
    **kwargs: Any,
) -> NDArray[np.object_]:
    """Draws related waveforms or matrices on an array of subplots.

    Each input gets one panel; with one input, its leading axis is split. Keywords that are
    lists with one element per panel go to each panel in turn; with ``shared`` they name the
    properties to make common, such as a colour range.

    Args:
        kind: ``"waveform"``, ``"bars"`` or ``"matrix"``.
        *data: The inputs.
        axes: Existing axes. Default: a new figure.
        fig: A figure to draw in.
        orient: ``"v"`` stacks panels vertically; ``"h"`` side by side.
        shared: For matrices, True gives every panel one colour range.
        fig_kw: Passed when creating the figure.
        sharex: Share the horizontal axis.
        sharey: Share the vertical axis.
        label_outer: Label only the outer axes.
        labels: Legend label per panel.
        titles: Title per panel.
        prop_cycle: A ``cycler`` of properties per panel.
        **kwargs: Passed to the plotting function.

    Returns:
        The array of axes.
    """
    import matplotlib.pyplot as plt

    if kind not in ("waveform", "bars", "matrix"):
        raise ValueError(f"kind: unknown {kind!r}; expected 'waveform', 'bars' or 'matrix'")
    if orient not in ("v", "h"):
        raise ValueError(f"orient: unknown {orient!r}")
    items: list[Any] = []
    if len(data) == 1:
        first = data[0]
        items = list(first) if isinstance(first, (list, tuple)) or (hasattr(first, "ndim") and first.ndim > (2 if kind == "matrix" else 1)) else [first]
    else:
        items = list(data)
    if not items:
        raise ValueError("data: nothing to draw")
    count = len(items)
    if axes is None:
        options = dict(fig_kw or {})
        shape = (count, 1) if orient == "v" else (1, count)
        if fig is None:
            _, axes = plt.subplots(*shape, sharex=sharex, sharey=sharey, squeeze=False, **options)
        else:
            axes = fig.subplots(*shape, sharex=sharex, sharey=sharey, squeeze=False, **options)
    grid = np.atleast_1d(axes)
    flat = grid.ravel()
    if flat.size != count:
        raise ValueError(f"axes: {flat.size} axes for {count} inputs")
    func: Any = {"waveform": plot_waveform_adaptive, "bars": plot_waveform_bars, "matrix": plot_matrix}[kind]
    per_panel = {k: v for k, v in kwargs.items() if isinstance(v, list) and len(v) == count}
    common = {k: v for k, v in kwargs.items() if k not in per_panel}
    if shared and kind == "matrix" and "vmin" not in common and "vmax" not in common:
        common["vmin"] = min(float(np.nanmin(np.real(i))) for i in items)
        common["vmax"] = max(float(np.nanmax(np.real(i))) for i in items)
    cycle = list(prop_cycle) if prop_cycle is not None else None
    for i, (item, panel) in enumerate(zip(items, flat, strict=True)):
        options = dict(common)
        options.update({k: v[i] for k, v in per_panel.items()})
        if cycle is not None:
            options.update(cycle[i % len(cycle)])
        if labels is not None and kind != "matrix":
            options["label"] = labels[i]
        func(item, ax=panel, **options)
        if labels is not None and kind == "matrix" and labels[i] is not None:
            panel.set_ylabel(labels[i])
        if titles is not None and titles[i] is not None:
            panel.set_title(titles[i])
        if label_outer:
            panel.label_outer()
    return grid


def legend_for_axes(axes: Any = None, *, fig: Any = None, **kwargs: Any) -> Any:
    """Makes one legend for the labelled artists of several axes.

    Args:
        axes: An ``Axes``, a list or an array. Default: every axes of ``fig``, or of the
            current figure.
        fig: The figure to hold the legend. Default: the figure of the axes.
        **kwargs: Passed to ``Figure.legend``.

    Returns:
        The legend.
    """
    import matplotlib.pyplot as plt

    if axes is None:
        figure = fig if fig is not None else plt.gcf()
        items = list(figure.axes)
    else:
        items = list(np.atleast_1d(axes).ravel())
        figure = fig if fig is not None else items[0].figure
    handles: list[Any] = []
    names: list[str] = []
    for ax in items:
        for handle, name in zip(*ax.get_legend_handles_labels(), strict=True):
            if name not in names:
                handles.append(handle)
                names.append(name)
    return figure.legend(handles, names, **kwargs)


def highlight_effects(
    *,
    artist: Any = None,
    ax: "Axes | None" = None,
    color: Any = None,
    bright_color: Any = "white",
    dark_color: Any = "black",
    luminance_threshold: float = 0.5,
    **kwargs: Any,
) -> list[Any]:
    """Makes a line or text stand out from what lies under it.

    The outline is ``bright_color`` over dark content and ``dark_color`` over bright content,
    judged from the axes' image, if any, otherwise from its background.

    Args:
        artist: The artist to change. Default: only return the effects.
        ax: The axes to judge. Default: the artist's, or the current axes.
        color: The outline colour, in place of judging.
        bright_color: Outline over dark content.
        dark_color: Outline over bright content.
        luminance_threshold: Luminance from 0 to 1 that separates dark from bright.
        **kwargs: Passed to ``withStroke``; ``linewidth`` defaults to 3.

    Returns:
        The path effects.
    """
    import matplotlib.pyplot as plt

    target = ax if ax is not None else (artist.axes if artist is not None else plt.gca())
    chosen = color
    if chosen is None:
        rgb = None
        mappables: list[Any] = list(target.get_images())
        for collection in target.collections:
            candidate: Any = collection
            array: Any = getattr(candidate, "get_array", None)
            if array is not None and array() is not None:
                mappables.append(candidate)
        if mappables:
            m = mappables[-1]
            rgb = np.asarray(m.to_rgba(np.ma.masked_invalid(m.get_array())))[..., :3].reshape(-1, 3).mean(axis=0)
        if rgb is None:
            rgb = np.asarray(mcolors.to_rgb(target.get_facecolor()))
        luminance = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
        chosen = bright_color if luminance < luminance_threshold else dark_color
    kwargs.setdefault("linewidth", 3)
    effects = [mpe.withStroke(foreground=chosen, **kwargs)]
    if artist is not None:
        artist.set_path_effects(effects)
    return effects
