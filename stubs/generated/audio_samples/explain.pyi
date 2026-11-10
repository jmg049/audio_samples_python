"""
Explanations of configurations and concepts, computed from your settings.

``explain(obj, ...)`` describes what a configuration (``StftParams``, ``MelParams``,
``IirFilterDesign``, ``CompressorConfig``, ...) will do; the topic functions
(``sampling``, ``aliasing``, ``quantisation``, ``dft``, ``window``, ``yin``,
``autocorrelation_pitch``, ``lpc``, ``cepstrum``, ``resampling``, ``fir``) explain concepts
that have no configuration class. All return an ``ExplainDoc``.
"""

from _typeshed import Incomplete
from typing import Any, final

@final
class ExplainDoc:
    """
    A structured explanation of a configuration or concept.

    An ``ExplainDoc`` says what a configuration will do (summary), the formulas
    the code actually evaluates, the symbols in those formulas with their
    values, quantities derived from your settings (frequency resolution, group
    delay, gain at the cutoff, ...), tables, trade-offs and warnings about
    suspicious settings. Every number is computed from the settings.

    Obtain one from :func:`audio_samples.explain.explain`, from a topic
    function such as :func:`audio_samples.explain.aliasing`, from
    :meth:`AudioSamples.explain`, or from ``config.explain(...)`` on a filter,
    dynamics or detection configuration. It is read-only.

    Rendering: ``str(doc)`` / :meth:`text` for a terminal (Unicode maths),
    :meth:`markdown` for Jupyter, mdBook or MyST (``$$`` LaTeX blocks), and
    :meth:`html` for a self-contained HTML fragment. In a notebook the last
    expression ``doc`` renders as Markdown (maths typeset by MathJax).

    Examples:

        >>> from audio_samples.explain import aliasing
        >>> doc = aliasing(7000.0, 8000.0)
        >>> doc.quantity("Alias frequency")
        1000.0
    """
    def __repr__(self, /) -> str: ...
    def __str__(self, /) -> str: ...
    def _repr_markdown_(self, /) -> str:
        """
        Jupyter rich display: Markdown with ``$$`` maths (typeset by MathJax).

        No ``_repr_html_`` is defined on purpose: Jupyter prefers HTML over
        Markdown when both exist, and the HTML fragment carries no MathJax
        delimiters.
        """
    @property
    def derived(self, /) -> list:
        """
        Quantities computed from the settings.

        Returns:

            list[dict]: One dict per quantity with keys ``name``, ``symbol``
                (LaTeX), ``value`` (float, always finite), ``unit`` and ``how``
                (the LaTeX formula used to compute it).
        """
    @property
    def formulas(self, /) -> list:
        """
        Formulas the explained code evaluates.

        Returns:

            list[dict]: One dict per formula with keys ``latex`` (math-mode
                LaTeX without ``$`` delimiters) and ``caption``.
        """
    def has_warning(self, /, level: "str | None" = None) -> bool:
        """
        Whether the explanation carries a warning of the given level.

        Args:

            level (str | None): ``"info"``, ``"caution"`` or ``"problem"``
                (case-insensitive), matched exactly. None (the default) matches
                any warning.

        Returns:

            bool: True if at least one such warning exists.

        Raises:

            ValueError: If ``level`` is not a known warning level.
        """
    def has_warning_containing(self, /, text: "str") -> bool:
        """
        Whether any warning message contains ``text``.

        Args:

            text (str): Substring to search for (case-sensitive).

        Returns:

            bool: True if a matching warning exists.
        """
    def html(self, /) -> str:
        """
        Render as a self-contained HTML fragment (no JavaScript).

        Returns:

            str: HTML source.
        """
    def markdown(self, /) -> str:
        """
        Render as Markdown with ``$$...$$`` LaTeX blocks.

        Renders in Jupyter (MathJax), mdBook and MyST. This is what a notebook
        shows for a bare ``doc`` expression.

        Returns:

            str: Markdown source.
        """
    def quantity(self, /, name: "str") -> float | None:
        """
        Look up a derived quantity by name.

        Args:

            name (str): Quantity name exactly as listed in :attr:`derived`,
                e.g. ``"Frequency resolution"``.

        Returns:

            float | None: The value of the first quantity with that name, or
                None if there is none.

        Examples:

            >>> from audio_samples.explain import sampling
            >>> sampling(48000.0).quantity("Nyquist frequency")
            24000.0
        """
    @property
    def see_also(self, /) -> list:
        """
        Related reading: Rust paths, Python names, guide chapters.

        Returns:

            list[dict]: One dict per reference with keys ``label`` and ``target``.
        """
    @property
    def summary(self, /) -> str:
        """
        One to three sentences saying what will happen, in plain words.
        """
    @property
    def symbols(self, /) -> list:
        """
        Symbols used in the formulas.

        Returns:

            list[dict]: One dict per symbol with keys ``symbol`` (LaTeX),
                ``meaning``, ``value`` (str or None) and ``unit`` (str or None).
        """
    @property
    def tables(self, /) -> list:
        """
        Tables (pole/zero lists, static curves, band edges, ...).

        Returns:

            list[dict]: One dict per table with keys ``caption`` (str),
                ``headers`` (list[str]) and ``rows`` (list[list[str]]). Cells
                that start and end with ``$`` are LaTeX.
        """
    def text(self, /) -> str:
        """
        Render for a terminal.

        Formulas are drawn as Unicode maths; no raw LaTeX reaches the output.

        Returns:

            str: A multi-line string (also ``str(doc)``).
        """
    @property
    def title(self, /) -> str:
        """
        Title of the explanation, e.g. ``"Short-time Fourier transform (STFT)"``.
        """
    @property
    def topic(self, /) -> str:
        """
        Syllabus topic label, e.g. ``"Windowing"`` or ``"IIR filters"``.
        """
    @property
    def trade_offs(self, /) -> list[str]:
        """
        Resolution, latency and leakage trade-offs, stated for these settings.

        Returns:

            list[str]: One sentence or short paragraph per trade-off.
        """
    @property
    def warnings(self, /) -> list:
        """
        Computed observations about the settings.

        Returns:

            list[dict]: One dict per warning with keys ``level`` (``"info"``,
                ``"caution"`` or ``"problem"``), ``message`` and ``hint``
                (str or None).
        """

def aliasing(frequency: "float", sample_rate: "float") -> ExplainDoc:
    """
    Explain where a tone of a given frequency lands after sampling (aliasing).

    Args:

        frequency (float): Frequency of the continuous-time tone in Hz.
        sample_rate (float): Sample rate in Hz.

    Returns:

        ExplainDoc: The explanation, with the ``"Alias frequency"`` quantity.

    Examples:

        >>> from audio_samples.explain import aliasing
        >>> aliasing(7000.0, 8000.0).quantity("Alias frequency")
        1000.0
    """

def autocorrelation_pitch(min_frequency: "float", max_frequency: "float", sample_rate: "float") -> ExplainDoc:
    """
    Explain autocorrelation pitch estimation for a search range.

    Args:

        min_frequency (float): Lowest frequency searched, in Hz.
        max_frequency (float): Highest frequency searched, in Hz.
        sample_rate (float): Sample rate in Hz.

    Returns:

        ExplainDoc: The explanation (lag range, lag resolution, ...).
    """

def cepstrum(n_fft: "int", sample_rate: "float") -> ExplainDoc:
    """
    Explain the real cepstrum for an N-point FFT.

    Args:

        n_fft (int): FFT length in samples.
        sample_rate (float): Sample rate in Hz.

    Returns:

        ExplainDoc: The explanation (quefrency resolution and range, ...).
    """

def dft(n_fft: "int", sample_rate: "float") -> ExplainDoc:
    """
    Explain an N-point discrete Fourier transform at a given sample rate.

    Args:

        n_fft (int): Transform length in samples.
        sample_rate (float): Sample rate in Hz.

    Returns:

        ExplainDoc: The explanation (bin spacing, number of bins, ...).
    """

def explain(obj: Any, *, sample_rate: "float | None" = None, n_samples: "int | None" = None, n_channels: "int | None" = None, f0_range: "tuple[float, float] | None" = None, n_fft: "int | None" = None, n_mels: "int | None" = None) -> ExplainDoc:
    """
    Explain a configuration object.

    Dispatches on the type of ``obj`` and returns an :class:`ExplainDoc`
    describing what the configuration will do, with numbers computed from its
    settings and the optional context below. Missing context is not an error:
    quantities that need it are left out and an ``info`` warning says so.

    Args:

        obj: The configuration to explain. One of ``StftParams``,
            ``SpectrogramParams``, ``MelParams``, ``MfccParams``,
            ``WindowType``, ``IirFilterDesign``, ``EqBand``,
            ``CompressorConfig``, ``LimiterConfig``, ``VadConfig`` or
            ``OnsetDetectionConfig``.
        sample_rate (float | None): Sample rate in Hz of the signal the
            configuration will be applied to.
        n_samples (int | None): Signal length in samples per channel.
        n_channels (int | None): Number of channels.
        f0_range (tuple[float, float] | None): Expected fundamental-frequency
            range ``(lowest, highest)`` in Hz; enables checks such as "window
            shorter than two pitch periods".
        n_fft (int | None): FFT size a ``MelParams`` will be combined with
            (enables the empty-mel-band check).
        n_mels (int | None): Number of mel bands an ``MfccParams`` will be
            combined with (enables the ``n_mfcc > n_mels`` check).

    Returns:

        ExplainDoc: The explanation.

    Raises:

        TypeError: If ``obj`` is not one of the supported types.

    Examples:

        >>> import audio_samples as aus
        >>> from audio_samples.explain import explain
        >>> params = aus.StftParams(n_fft=256, hop_size=128, window=aus.WindowType.hanning)
        >>> doc = explain(params, sample_rate=16000, f0_range=(80.0, 300.0))
        >>> doc.quantity("Frequency resolution")
        62.5
        >>> doc.has_warning("caution")
        True
    """

def fir(coefficients: "Sequence[float] | numpy.ndarray", sample_rate: "float") -> ExplainDoc:
    """
    Explain an FIR filter given its coefficients.

    Reports the number of taps, whether the filter is linear phase
    (symmetric or antisymmetric coefficients) and its group delay.

    Args:

        coefficients (Sequence[float] | numpy.ndarray): Filter taps ``h[0..N-1]``.
        sample_rate (float): Sample rate in Hz.

    Returns:

        ExplainDoc: The explanation.
    """

def lpc(order: "int", sample_rate: "float") -> ExplainDoc:
    """
    Explain linear predictive coding (LPC) of a given order.

    Args:

        order (int): Prediction order (number of coefficients).
        sample_rate (float): Sample rate in Hz.

    Returns:

        ExplainDoc: The explanation (formants resolvable, rule-of-thumb order, ...).
    """

def quantisation(bits: "int") -> ExplainDoc:
    """
    Explain uniform quantisation at a given bit depth (step size, SQNR).

    Args:

        bits (int): Bits per sample.

    Returns:

        ExplainDoc: The explanation.

    Examples:

        >>> from audio_samples.explain import quantisation
        >>> round(quantisation(16).quantity("SQNR (full-scale sine)"), 2)
        98.09
    """

def resampling(from_rate: "float", to_rate: "float") -> ExplainDoc:
    """
    Explain sample-rate conversion between two rates.

    Args:

        from_rate (float): Source sample rate in Hz.
        to_rate (float): Target sample rate in Hz.

    Returns:

        ExplainDoc: The explanation (ratio, new Nyquist, anti-alias warnings, ...).
    """

def sampling(sample_rate: "float") -> ExplainDoc:
    """
    Explain sampling at a given rate: sampling period and Nyquist frequency.

    Args:

        sample_rate (float): Sample rate in Hz.

    Returns:

        ExplainDoc: The explanation (a ``problem`` warning if the rate is not
            finite and positive).

    Examples:

        >>> from audio_samples.explain import sampling
        >>> sampling(16000.0).quantity("Nyquist frequency")
        8000.0
    """

def window(window_type: "WindowType", n: "int", sample_rate: "float") -> ExplainDoc:
    """
    Explain a window function of a given length, measured from its coefficients.

    Args:

        window_type (WindowType): The window (``audio_samples.WindowType``).
        n (int): Window length in samples.
        sample_rate (float): Sample rate in Hz.

    Returns:

        ExplainDoc: The explanation (ENBW, main-lobe width, side-lobe level, ...).

    Raises:

        TypeError: If ``window_type`` is not a ``WindowType``.
    """

def yin(threshold: "float", min_frequency: "float", max_frequency: "float", sample_rate: "float", frame_length: "int | None" = None) -> ExplainDoc:
    """
    Explain YIN pitch estimation for a search range and frame length.

    Args:

        threshold (float): CMND threshold (typically 0.1 to 0.2).
        min_frequency (float): Lowest frequency searched, in Hz.
        max_frequency (float): Highest frequency searched, in Hz.
        sample_rate (float): Sample rate in Hz.
        frame_length (int | None): Analysis frame length in samples; when given,
            a ``problem`` warning fires if it is too short for ``min_frequency``.

    Returns:

        ExplainDoc: The explanation (lag range, minimum frame length, ...).
    """

def __getattr__(name: str) -> Incomplete: ...
