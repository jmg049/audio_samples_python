"""Explanations of configurations and concepts, computed from your settings.

``explain(obj, ...)`` describes what a configuration (``StftParams``,
``MelParams``, ``IirFilterDesign``, ``CompressorConfig``, ...) will do: the
formulas the code evaluates, the symbols with their values, quantities derived
from your settings, trade-offs and warnings about suspicious settings. The
topic functions (``sampling``, ``aliasing``, ``quantisation``, ``dft``,
``window``, ``yin``, ``autocorrelation_pitch``, ``lpc``, ``cepstrum``,
``resampling``, ``fir``) explain concepts that have no configuration class.
All return an :class:`ExplainDoc`.

Examples:
    >>> import audio_samples as aus
    >>> from audio_samples.explain import explain, aliasing
    >>> aliasing(7000.0, 8000.0).quantity("Alias frequency")
    1000.0
    >>> params = aus.StftParams(n_fft=256, hop_size=128, window=aus.WindowType.hanning)
    >>> doc = explain(params, sample_rate=16000, f0_range=(80.0, 300.0))
    >>> doc.has_warning("caution")
    True
"""

from __future__ import annotations
from typing import final

from typing import Literal, Optional, Sequence, TypedDict, Union

import numpy as np
from numpy.typing import NDArray

from audio_samples.spectrograms import (
    MelParams,
    MfccParams,
    SpectrogramParams,
    StftParams,
    WindowType,
)
from audio_samples.types import (
    CompressorConfig,
    EqBand,
    IirFilterDesign,
    LimiterConfig,
    OnsetDetectionConfig,
    VadConfig,
)

__all__ = [
    "ExplainDoc",
    "explain",
    "sampling",
    "aliasing",
    "quantisation",
    "dft",
    "window",
    "yin",
    "autocorrelation_pitch",
    "lpc",
    "cepstrum",
    "resampling",
    "fir",
]

_WarningLevelName = Literal["info", "caution", "problem"]

class _FormulaDict(TypedDict):
    latex: str
    caption: str

class _SymbolDict(TypedDict):
    symbol: str
    meaning: str
    value: Optional[str]
    unit: Optional[str]

class _QuantityDict(TypedDict):
    name: str
    symbol: str
    value: float
    unit: str
    how: str

class _TableDict(TypedDict):
    caption: str
    headers: list[str]
    rows: list[list[str]]

class _WarningDict(TypedDict):
    level: _WarningLevelName
    message: str
    hint: Optional[str]

class _SeeAlsoDict(TypedDict):
    label: str
    target: str

_Explainable = Union[
    StftParams,
    SpectrogramParams,
    MelParams,
    MfccParams,
    WindowType,
    IirFilterDesign,
    EqBand,
    CompressorConfig,
    LimiterConfig,
    VadConfig,
    OnsetDetectionConfig,
]
"""Types accepted by :func:`explain` and :meth:`AudioSamples.explain`."""

@final
class ExplainDoc:
    """A structured explanation of a configuration or concept.

    Read-only. Render with :meth:`text` (also ``str(doc)``), :meth:`markdown`
    (Jupyter, mdBook, MyST; also the notebook display) or :meth:`html`.

    Examples:
        >>> from audio_samples.explain import aliasing
        >>> doc = aliasing(7000.0, 8000.0)
        >>> doc.quantity("Alias frequency")
        1000.0
    """

    @property
    def title(self) -> str:
        """Title, e.g. ``"Short-time Fourier transform (STFT)"``."""
        ...
    @property
    def topic(self) -> str:
        """Syllabus topic label, e.g. ``"Windowing"`` or ``"IIR filters"``."""
        ...
    @property
    def summary(self) -> str:
        """One to three sentences saying what will happen."""
        ...
    @property
    def formulas(self) -> list[_FormulaDict]:
        """Formulas the code evaluates: ``{latex, caption}`` (LaTeX without ``$``)."""
        ...
    @property
    def symbols(self) -> list[_SymbolDict]:
        """Symbols in the formulas: ``{symbol, meaning, value, unit}``."""
        ...
    @property
    def derived(self) -> list[_QuantityDict]:
        """Quantities computed from the settings: ``{name, symbol, value, unit, how}``."""
        ...
    @property
    def tables(self) -> list[_TableDict]:
        """Tables: ``{caption, headers, rows}``; cells wrapped in ``$`` are LaTeX."""
        ...
    @property
    def trade_offs(self) -> list[str]:
        """Resolution, latency and leakage trade-offs for these settings."""
        ...
    @property
    def warnings(self) -> list[_WarningDict]:
        """Computed observations: ``{level, message, hint}``."""
        ...
    @property
    def see_also(self) -> list[_SeeAlsoDict]:
        """Related reading: ``{label, target}``."""
        ...
    def text(self) -> str:
        """Render for a terminal (Unicode maths, no raw LaTeX)."""
        ...
    def markdown(self) -> str:
        """Render as Markdown with ``$$...$$`` LaTeX blocks."""
        ...
    def html(self) -> str:
        """Render as a self-contained HTML fragment (no JavaScript)."""
        ...
    def quantity(self, name: str) -> Optional[float]:
        """Value of the first derived quantity called ``name``, or None.

        Args:
            name: Quantity name exactly as in :attr:`derived`.

        Returns:
            The value, or None if there is no such quantity.
        """
        ...
    def has_warning(self, level: Optional[_WarningLevelName] = None) -> bool:
        """Whether a warning of exactly ``level`` exists (any warning if None).

        Args:
            level: ``"info"``, ``"caution"`` or ``"problem"`` (case-insensitive).

        Returns:
            True if such a warning exists.

        Raises:
            ValueError: If ``level`` is not a known warning level.
        """
        ...
    def has_warning_containing(self, text: str) -> bool:
        """Whether any warning message contains ``text``."""
        ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...
    def _repr_markdown_(self) -> str:
        """Jupyter display: Markdown with ``$$`` maths (typeset by MathJax)."""
        ...

def explain(
    obj: _Explainable,
    *,
    sample_rate: Optional[float] = None,
    n_samples: Optional[int] = None,
    n_channels: Optional[int] = None,
    f0_range: Optional[tuple[float, float]] = None,
    n_fft: Optional[int] = None,
    n_mels: Optional[int] = None,
) -> ExplainDoc:
    """Explain a configuration object.

    Args:
        obj: The configuration to explain.
        sample_rate: Sample rate in Hz of the signal it will be applied to.
        n_samples: Signal length in samples per channel.
        n_channels: Number of channels.
        f0_range: Expected fundamental-frequency range ``(lowest, highest)`` in Hz.
        n_fft: FFT size a ``MelParams`` will be combined with.
        n_mels: Number of mel bands an ``MfccParams`` will be combined with.

    Returns:
        The explanation. Missing context is not an error: quantities that need
        it are omitted and an ``info`` warning says so.

    Raises:
        TypeError: If ``obj`` is not a supported type.

    Examples:
        >>> import audio_samples as aus
        >>> params = aus.StftParams(n_fft=256, hop_size=128, window=aus.WindowType.hanning)
        >>> explain(params, sample_rate=16000).quantity("Frequency resolution")
        62.5
    """
    ...

def sampling(sample_rate: float) -> ExplainDoc:
    """Explain sampling at ``sample_rate`` Hz (sampling period, Nyquist frequency).

    Examples:
        >>> sampling(16000.0).quantity("Nyquist frequency")
        8000.0
    """
    ...

def aliasing(frequency: float, sample_rate: float) -> ExplainDoc:
    """Explain where a ``frequency`` Hz tone lands when sampled at ``sample_rate`` Hz.

    Examples:
        >>> aliasing(7000.0, 8000.0).quantity("Alias frequency")
        1000.0
    """
    ...

def quantisation(bits: int) -> ExplainDoc:
    """Explain uniform quantisation at ``bits`` bits per sample (step size, SQNR)."""
    ...

def dft(n_fft: int, sample_rate: float) -> ExplainDoc:
    """Explain an ``n_fft``-point DFT at ``sample_rate`` Hz (bin spacing, bins)."""
    ...

def window(
    window_type: WindowType, n: int, sample_rate: float
) -> ExplainDoc:
    """Explain a window of ``n`` samples, measured from its coefficients.

    Raises:
        TypeError: If ``window_type`` is not a ``WindowType``.
    """
    ...

def yin(
    threshold: float,
    min_frequency: float,
    max_frequency: float,
    sample_rate: float,
    frame_length: Optional[int] = None,
) -> ExplainDoc:
    """Explain YIN pitch estimation (lag range, minimum frame length, ...)."""
    ...

def autocorrelation_pitch(
    min_frequency: float, max_frequency: float, sample_rate: float
) -> ExplainDoc:
    """Explain autocorrelation pitch estimation for a search range."""
    ...

def lpc(order: int, sample_rate: float) -> ExplainDoc:
    """Explain linear predictive coding of a given order."""
    ...

def cepstrum(n_fft: int, sample_rate: float) -> ExplainDoc:
    """Explain the real cepstrum for an ``n_fft``-point FFT."""
    ...

def resampling(from_rate: float, to_rate: float) -> ExplainDoc:
    """Explain sample-rate conversion from ``from_rate`` to ``to_rate`` Hz."""
    ...

def fir(
    coefficients: Union[Sequence[float], NDArray[np.floating]], sample_rate: float
) -> ExplainDoc:
    """Explain an FIR filter from its taps (linear phase, group delay)."""
    ...
