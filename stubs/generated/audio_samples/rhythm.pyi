"""
Rhythm analysis over NumPy arrays (onset strength and detection, peak picking, tempograms, tempo, beat tracking, PLP). float32 / complex64 arrays are processed in float32 and give float32 output; any other input is converted to float64.
"""

from numpy.typing import ArrayLike
from typing import Any

__all__ = ["beat_track", "beat_track_with_bpm", "fourier_tempo_frequencies", "fourier_tempogram", "hybrid_tempogram", "metrogram", "onset_backtrack", "onset_detect", "onset_strength", "onset_strength_multi", "peak_pick", "plp", "tempo", "tempo_frequencies", "tempogram", "tempogram_ratio"]

def beat_track(*, onset_envelope: "ArrayLike", sr: float, hop_length: int = ..., start_bpm: float = 120.0, tightness: float = 100.0, trim: bool = False, bpm: Any | None = None, prior: Any | None = None, units: str = "frames", sparse: bool = True) -> tuple[Any, Any]:
    """
    Tracks beats in an onset strength envelope.

    Ellis's dynamic-programming tracker: beats maximise onset strength while
    keeping inter-beat intervals close to the tempo's period.

    Args:

        onset_envelope (array_like): 1-D onset envelope (``AudioSamples.beat_track``
            computes it with ``aggregate=numpy.median``).
        sr (float): Sampling rate. Required.
        hop_length (int): Hop in samples. Default 512.
        start_bpm (float): Prior centre of the tempo estimate. Default 120.
        tightness (float): Tempo-consistency weight. Default 100.
        trim (bool): Remove weak leading/trailing beats. Default False.
        bpm (Optional[float | array_like]): Tempo to track at (a number, or one
            value per frame for a time-varying tempo); None estimates it.
        prior: Tempo prior (see ``tempo``).
        units (str): ``"frames"`` (default), ``"samples"`` or ``"time"``.
        sparse (bool): Must be True.

    Returns:

        tuple: ``(tempo, beats)``: ``tempo`` is the estimate as a
        float64 array of shape ``(1,)``, or ``bpm`` unchanged when given, or
        ``0.0`` when the envelope is all zeros; ``beats`` are int64 frames or
        samples, or float64 times.

    Raises:

        ValueError: For invalid parameters, a 2-D envelope or ``sparse=False``.
    """

def beat_track_with_bpm(onset_envelope: "ArrayLike", bpm: "float | ArrayLike", *, sr: float, hop_length: int = ..., tightness: float = 100.0, trim: bool = False) -> "NDArray[np.int64]":
    """
    Runs the dynamic-programming beat tracker at a given tempo, without tempo
    estimation (what ``beat_track(onset_envelope=..., bpm=...)`` runs).

    Args:

        onset_envelope (array_like): 1-D onset envelope.
        bpm (float | array_like): Tempo in BPM: one value, or one per frame.
        sr (float): Sampling rate. Required.
        hop_length (int): Hop in samples (frame rate ``sr / hop_length``).
            Default 512.
        tightness (float): Tempo-consistency weight. Default 100.
        trim (bool): Remove weak leading/trailing beats. Default False.

    Returns:

        numpy.ndarray[int64]: ascending beat frames.

    Raises:

        ValueError: For a wrong-length or non-positive ``bpm``, or a tempo so
            fast that its period is under one frame.
    """

def fourier_tempo_frequencies(*, sr: float, win_length: int = ..., hop_length: int = ...) -> "NDArray[np.float64]":
    """
    Tempo (BPM) of each bin of a Fourier tempogram.

    Args:

        sr (float): Sampling rate. Required.
        win_length (int): Fourier tempogram window in frames. Default 384.
        hop_length (int): Hop in samples. Default 512.

    Returns:

        numpy.ndarray[float64]: shape ``(win_length // 2 + 1,)``.
    """

def fourier_tempogram(*, onset_envelope: "ArrayLike", sr: float, hop_length: int = ..., win_length: int = ..., center: bool = True, window: str | tuple[str, float] | float | ArrayLike = ...) -> "NDArray[Any]":
    """
    Fourier tempogram: short-time Fourier transform of an onset strength
    envelope.

    Args:

        onset_envelope (array_like): ``(n_frames,)`` or ``(channels, n_frames)``.
        sr, hop_length: Unused (the envelope is already computed).
        win_length (int): STFT length in frames. Default 384.
        center (bool): Centre the frames (zero padding). Default True.
        window: Analysis window. Default ``"hann"``.

    Returns:

        numpy.ndarray[complex64 | complex128]: shape ``(win_length // 2 + 1, n_frames + 1)``
        (centred) for 1-D input, with a leading channel axis for 2-D.
    """

def hybrid_tempogram(*, onset_envelope: "ArrayLike", sr: float, hop_length: int = ..., win_length: int = ..., center: bool = True, window: str | tuple[str, float] | float | ArrayLike = ...) -> "NDArray[Any]":
    """
    Hybrid tempogram: the Fourier and autocorrelation tempograms combined on
    the Fourier tempo grid.

    Args:

        onset_envelope (array_like): ``(n_frames,)`` or ``(channels, n_frames)``.
        sr (float): Sampling rate. Required.
        hop_length (int): Hop in samples. Default 512.
        win_length (int): Window of both tempograms in frames (>= 3). Default 384.
        center (bool): Centred frames. Default True.
        window: Window of both tempograms. Default ``"hann"``.

    Returns:

        numpy.ndarray[float32 | float64]: shape ``(win_length // 2 + 1, n_frames)`` on the
        ``fourier_tempo_frequencies`` grid, with a leading channel axis for 2-D input.
    """

def metrogram(*, tg: "ArrayLike", freqs: "ArrayLike", factors: Any | None = None, aggregate: "Callable[..., Any] | Literal['mean', 'median', 'max', 'min', 'sum'] | None" = ..., kind: str = "linear", fill_value: float = 0.0) -> "NDArray[Any]":
    """
    Metrogram (metrical subharmonic salience) of a tempogram: for each factor
    ``h``, the tempogram interpolated at ``h * freqs``, multiplied by the
    tempogram and aggregated over the tempo axis.

    Args:

        tg (array_like): Tempogram magnitude ``(n_freqs, n_frames)`` or
            ``(channels, n_freqs, n_frames)``.
        freqs (array_like): Tempo (BPM) of each bin; at least two finite values
            (drop the infinite zero-lag bin of ``tempo_frequencies`` first).
        factors (Optional[array_like]): Subharmonic factors; None uses
            ``[1/3, 1/4, 1/5, 1/7]``.
        aggregate: Reduction over the tempo axis (default ``numpy.sum``). None keeps
            every bin.
        kind (str): Must be ``"linear"``.
        fill_value (float): Value outside the frequency grid. Default 0.

    Returns:

        numpy.ndarray[float32 | float64]: shape ``(n_factors, n_frames)``, or
        ``(n_factors, n_freqs, n_frames)`` with ``aggregate=None``, with a leading
        channel axis for 3-D input.
    """

def onset_backtrack(events: "ArrayLike", energy: "ArrayLike") -> "NDArray[np.int64]":
    """
    Moves events back to the preceding local minimum of ``energy``.

    Args:

        events (array_like): Event frame indices (non-negative integers).
        energy (array_like): 1-D function to search for minima (onset envelope
            or RMS energy).

    Returns:

        numpy.ndarray[int64]: one backtracked frame per event. An empty
        ``events`` gives an empty array.
    """

def onset_detect(*, onset_envelope: "ArrayLike", sr: float, hop_length: int = ..., backtrack: bool = False, energy: Any | None = None, units: str = "frames", normalize: bool = True, sparse: bool = True, pre_max: int | None = None, post_max: int | None = None, pre_avg: int | None = None, post_avg: int | None = None, delta: float | None = None, wait: int | None = None, method: str = "greedy") -> "NDArray[np.float64] | NDArray[np.int64]":
    """
    Detects onsets by peak picking on a precomputed onset strength envelope.

    The envelope is normalised to ``[0, 1]`` (``normalize=True``), peak-picked
    with the parameters below (defaults derived from the frame rate: 30 ms
    ``pre_max`` and ``wait``, 100 ms averaging windows, ``delta=0.07``) and
    optionally backtracked.

    Args:

        onset_envelope (array_like): 1-D onset strength envelope.
        sr (float): Sampling rate of the analysed audio. Required.
        hop_length (int): Hop of the envelope in samples. Default 512.
        backtrack (bool): Move onsets back to the preceding local minimum of
            ``energy``. Default False.
        energy (Optional[array_like]): 1-D function to backtrack on; None uses
            the (normalised) envelope.
        units (str): ``"frames"`` (default), ``"samples"`` or ``"time"``.
        normalize (bool): Scale the envelope to ``[0, 1]`` first. Default True.
        sparse (bool): Must be True.
        pre_max, post_max, pre_avg, post_avg, wait (Optional[int]),
        delta (Optional[float]), method (str): Peak-picking parameters (see
            ``peak_pick``); None derives the value from the frame rate.

    Returns:

        numpy.ndarray: onset frames or samples (int64), or times (float64).

    Raises:

        ValueError: For invalid parameters, a 2-D envelope or ``sparse=False``.
    """

def onset_strength(*, S: "ArrayLike", sr: float, lag: int = ..., max_size: int = ..., ref: Any | None = None, detrend: bool = False, center: bool = True, aggregate: Any | None = None, n_fft: int = ..., hop_length: int = ...) -> "NDArray[Any]":
    """
    Onset strength of a log-power spectrogram: positive differences between
    frames, aggregated across bins.

    ``S`` is used as given (compute it with ``spectral.power_to_db`` of a mel
    spectrogram, or use ``AudioSamples.onset_strength``).

    Args:

        S (array_like): Log-power spectrogram, shape ``(n_bins, n_frames)`` or
            ``(channels, n_bins, n_frames)``.
        sr (float): Required. Unused (``S`` is already computed).
        lag (int): Time lag of the difference in frames. Default 1.
        max_size (int): Size of the maximum filter across frequency applied to
            the reference (SuperFlux); 1 disables it. Default 1.
        ref (Optional[array_like]): Reference spectrogram of the same shape as
            ``S``; None uses ``S`` (maximum-filtered when ``max_size > 1``).
        detrend (bool): High-pass filter the envelope. Default False.
        center (bool): Shift by ``n_fft // (2 * hop_length)`` frames to undo the
            centred-STFT delay. Default True.
        aggregate: Reduction across bins: None (mean), ``numpy.mean``,
            ``numpy.median``, ``numpy.max``, ``numpy.min``, ``numpy.sum``, a
            name, or a callable ``f(x, axis=-1)``. ``False`` returns the first
            bin's flux.
        n_fft (int): FFT size behind ``S`` (centring shift). Default 2048.
        hop_length (int): Hop behind ``S`` (centring shift). Default 512.

    Returns:

        numpy.ndarray[float32 | float64]: shape ``(n_frames,)`` for 2-D ``S``,
        ``(channels, n_frames)`` for 3-D.

    Raises:

        ValueError: For invalid parameters or an empty spectrogram.
    """

def onset_strength_multi(*, S: "ArrayLike", sr: float, n_fft: int = ..., hop_length: int = ..., lag: int = ..., max_size: int = ..., ref: Any | None = None, detrend: bool = False, center: bool = True, aggregate: Any | None = None, channels: Any | None = None) -> "NDArray[Any]":
    """
    Multi-band onset strength of a precomputed spectrogram: one envelope per
    group of bins.

    Args:

        S (array_like): Log-power spectrogram ``(n_bins, n_frames)`` or
            ``(channels, n_bins, n_frames)``, used as given.
        sr (float): Required. Unused (see ``onset_strength``).
        n_fft (int): FFT size behind ``S`` (centring shift). Default 2048.
        hop_length (int): Hop behind ``S`` (centring shift). Default 512.
        lag, max_size, ref, detrend, center: As ``onset_strength``.
        aggregate: As ``onset_strength``; ``False`` keeps every bin.
        channels (Optional[Sequence[int] | Sequence[slice]]): Sub-band boundaries in
            bins: values above ``n_bins`` are dropped, the rest sorted and
            de-duplicated, and each consecutive pair is one band. Or one slice of the
            bins per band, in any order. None gives one band.

    Returns:

        numpy.ndarray[float32 | float64]: shape ``(n_bands, n_frames)`` for 2-D ``S``,
        ``(channels, n_bands, n_frames)`` for 3-D.

    Raises:

        ValueError: For invalid parameters or boundaries.
    """

def peak_pick(x: "ArrayLike", *, pre_max: int, post_max: int, pre_avg: int, post_avg: int, delta: float, wait: int, sparse: bool = True, method: str = "greedy", axis: int = ...) -> "NDArray[np.int64]":
    """
    Picks peaks against a local maximum and a local mean threshold (1-D input,
    sparse output only).

    A sample ``n`` is a peak if ``x[n] == max(x[n - pre_max : n + post_max])``,
    ``x[n] >= mean(x[n - pre_avg : n + post_avg]) + delta`` and (greedy method)
    it is more than ``wait`` samples after the previous peak.

    Args:

        x (array_like): 1-D finite signal.
        pre_max, post_max, pre_avg, post_avg (int): Window extents in samples
            (``post_max`` and ``post_avg`` at least 1).
        delta (float): Threshold offset above the local mean (>= 0).
        wait (int): Minimum number of samples between peaks.
        sparse (bool): Must be True (indices are returned).
        method (str): ``"greedy"`` (default), ``"dp_count"`` or ``"dp_value"``.
        axis (int): Must be -1 (1-D input only).

    Returns:

        numpy.ndarray[int64]: ascending peak indices.

    Raises:

        ValueError: For invalid parameters, non-finite input or ``sparse=False``.
    """

def plp(*, onset_envelope: "ArrayLike", sr: float, hop_length: int = ..., win_length: int = ..., tempo_min: float | None = ..., tempo_max: float | None = ..., prior: Any | None = None) -> "NDArray[Any]":
    """
    Predominant local pulse from the Fourier tempogram of an onset envelope.

    Args:

        onset_envelope (array_like): ``(n_frames,)`` or ``(channels, n_frames)``
            (``AudioSamples.plp`` computes it with ``aggregate=numpy.median``).
        sr (float): Sampling rate. Required.
        hop_length (int): Hop in samples. Default 512.
        win_length (int): Fourier tempogram window in frames. Default 384.
        tempo_min (Optional[float]): Lowest tempo (BPM). Default 30.
        tempo_max (Optional[float]): Highest tempo (BPM). Default 300.
        prior: Tempo prior (see ``tempo``), added in the log domain.

    Returns:

        numpy.ndarray[float32 | float64]: pulse curve, same shape as the envelope, in
        ``[0, 1]``.

    Raises:

        ValueError: If ``tempo_max <= tempo_min`` or the envelope is empty.
    """

def tempo(*, onset_envelope: Any | None = None, tg: Any | None = None, sr: float, hop_length: int = ..., start_bpm: float = 120.0, std_bpm: float = 1.0, ac_size: float = 8.0, max_tempo: float | None = ..., aggregate: "Callable[..., Any] | Literal['mean', 'median', 'max', 'min', 'sum'] | None" = ..., prior: Any | None = None) -> "NDArray[np.float64]":
    """
    Estimates the tempo in BPM from an onset envelope or a precomputed
    tempogram ``tg``.

    The tempo is the lag bin maximising ``log1p(1e6 * tempogram) + log prior``
    (default prior: log-normal around ``start_bpm`` with ``std_bpm`` octaves).

    Args:

        onset_envelope (Optional[array_like]): ``(n_frames,)`` or
            ``(channels, n_frames)``. Exactly one of ``onset_envelope`` / ``tg``.
        tg (Optional[array_like]): Autocorrelation tempogram
            ``(win_length, n_frames)`` or ``(channels, win_length, n_frames)``.
        sr (float): Sampling rate. Required.
        hop_length (int): Hop in samples. Default 512.
        start_bpm (float): Prior centre. Default 120.
        std_bpm (float): Prior spread in octaves. Default 1.
        ac_size (float): Autocorrelation window in seconds (envelope input).
            Default 8.
        max_tempo (Optional[float]): Tempi at or above this are excluded.
            Default 320; None allows any.
        aggregate: Reduction over time (default ``numpy.mean``); None estimates
            one tempo per frame.
        prior: None (log-normal default), a ``scipy.stats`` distribution (its
            ``logpdf`` is used) or a callable of the BPM.

    Returns:

        numpy.ndarray[float64]: shape ``(1,)`` (or ``(n_frames,)`` with
        ``aggregate=None``) for 1-D input, with a leading channel axis for 2-D.

    Raises:

        ValueError: For invalid parameters, or neither/both inputs given.
    """

def tempo_frequencies(n_bins: int, *, hop_length: int = ..., sr: float) -> "NDArray[np.float64]":
    """
    Tempo (BPM) of each lag bin of an autocorrelation tempogram.

    Args:

        n_bins (int): Number of lag bins.
        hop_length (int): Hop in samples. Default 512.
        sr (float): Sampling rate. Required.

    Returns:

        numpy.ndarray[float64]: shape ``(n_bins,)``; bin 0 (zero lag) is ``inf``.
    """

def tempogram(*, onset_envelope: "ArrayLike", sr: float, hop_length: int = ..., win_length: int = ..., center: bool = True, window: str | tuple[str, float] | float | ArrayLike = ..., norm: float | None = ...) -> "NDArray[Any]":
    """
    Autocorrelation tempogram: windowed autocorrelation of an onset strength
    envelope.

    Args:

        onset_envelope (array_like): ``(n_frames,)`` or ``(channels, n_frames)``.
        sr (float): Required. Unused (the envelope is already computed).
        hop_length (int): Unused (as above).
        win_length (int): Autocorrelation window in frames (number of lag
            bins). Default 384.
        center (bool): Centre the windows (linear-ramp padding). Default True.
        window: Window applied before the autocorrelation. Default ``"hann"``.
        norm (Optional[float]): Column normalisation: ``numpy.inf`` (default,
            divide by the maximum), a p-norm, or None.

    Returns:

        numpy.ndarray[float32 | float64]: shape ``(win_length, n_frames)`` for 1-D input,
        ``(channels, win_length, n_frames)`` for 2-D (``n_frames - win_length + 1``
        frames when not centred).

    Raises:

        ValueError: For invalid parameters or (uncentred) a too-short envelope.
    """

def tempogram_ratio(*, onset_envelope: Any | None = None, tg: Any | None = None, bpm: Any | None = None, sr: float, hop_length: int = ..., win_length: int = ..., start_bpm: float = 120.0, std_bpm: float = 1.0, max_tempo: float | None = ..., freqs: Any | None = None, factors: Any | None = None, aggregate: Any | None = None, prior: Any | None = None, center: bool = True, window: str | tuple[str, float] | float | ArrayLike = ..., kind: str = "linear", fill_value: float = 0.0, norm: float | None = ...) -> "NDArray[Any]":
    """
    Tempogram ratio features from an onset envelope or a precomputed tempogram
    ``tg``.

    The tempogram is sampled at ``factors`` times the per-frame tempo estimate
    (Prockup et al. 2015).

    Args:

        onset_envelope (Optional[array_like]): ``(n_frames,)`` or
            ``(channels, n_frames)``; used to compute the tempogram when ``tg``
            is None.
        tg (Optional[array_like]): Autocorrelation tempogram
            ``(win_length, n_frames)`` or ``(channels, win_length, n_frames)``.
        bpm (Optional[array_like]): Per-frame tempo ``(n_frames,)`` /
            ``(channels, n_frames)``; None estimates it from the tempogram.
        sr (float): Sampling rate. Required.
        hop_length (int): Hop in samples. Default 512.
        win_length, center, window, norm: Tempogram parameters (envelope input).
        start_bpm, std_bpm, max_tempo, prior: Per-frame tempo estimation.
        freqs: Must be None (the ``tempo_frequencies`` grid is used).
        factors (Optional[array_like]): Tempo multiples; None uses the 13
            default factors.
        aggregate: Optional reduction over time (default None).
        kind (str): Must be ``"linear"``.
        fill_value (float): Value outside the tempogram's range. Default 0.

    Returns:

        numpy.ndarray[float32 | float64]: shape ``(n_factors, n_frames)`` for 1-D input
        (``(n_factors,)`` when aggregated), with a leading channel axis for 2-D.

    Raises:

        ValueError: For invalid parameters, unsupported ``freqs`` / ``kind``,
            or neither input given.
    """
