"""librosa-compatible rhythm analysis over NumPy arrays.

Free functions mirroring ``librosa.onset`` (``onset_strength(S=...)``,
``onset_strength_multi(S=...)``, ``onset_detect(onset_envelope=...)``,
``onset_backtrack``), ``librosa.util.peak_pick``, ``librosa.feature``
(``tempogram``, ``fourier_tempogram``, ``tempo``, ``tempogram_ratio``,
``hybrid_tempogram``, ``metrogram``), ``librosa.beat`` (``beat_track``,
``plp``) and ``librosa.tempo_frequencies`` / ``fourier_tempo_frequencies``.
Defaults are librosa 1.0.0's. They take precomputed inputs (librosa's ``S=``,
``onset_envelope=``, ``tg=``); the ``AudioSamples`` methods of the same names
compute them from audio.

Shapes follow librosa: a 1-D envelope (2-D spectrogram) gives librosa's result,
a ``(channels, n)`` envelope (3-D spectrogram) adds a leading channel axis.
Event outputs are 1-D arrays (int64 frames/samples or float64 times).

``aggregate`` accepts ``numpy.mean`` / ``median`` / ``max`` / ``min`` / ``sum``
(and their names, run natively) or any callable ``f(x, axis=-1)``. ``prior``
accepts a ``scipy.stats`` distribution (its ``logpdf``) or a callable of the BPM
returning a log-density.
"""

from __future__ import annotations

from typing import Any, Callable, Literal, Optional, Sequence, Union

import numpy as np
import numpy.typing as npt
from numpy.typing import NDArray

#: Window specification: a SciPy window name, ``(name, parameter)``, a number
#: (Kaiser beta) or an array of coefficients.
WindowSpec = Union[str, tuple[str, float], float, npt.ArrayLike]
#: Reduction: a NumPy reduction, its name, or a callable ``f(x, axis=-1)``.
AggregateSpec = Union[Callable[..., Any], Literal["mean", "median", "max", "min", "sum"]]
#: Tempo prior: a ``scipy.stats`` distribution (``.logpdf`` used) or a callable of the BPM.
PriorSpec = Union[Any, Callable[[float], float], None]
#: Event units.
Units = Literal["frames", "samples", "time"]

def onset_strength(
    *,
    S: npt.ArrayLike,
    sr: float = 22050,
    lag: int = 1,
    max_size: int = 1,
    ref: Optional[npt.ArrayLike] = None,
    detrend: bool = False,
    center: bool = True,
    aggregate: Union[AggregateSpec, Literal[False], None] = None,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> NDArray[np.float64]:
    """Onset strength of a log-power spectrogram, ``librosa.onset.onset_strength(S=S)``.

    ``S`` is used as given (compute it with ``spectral.power_to_db`` of a mel
    spectrogram, or use ``AudioSamples.onset_strength``).

    Args:
        S: ``(n_bins, n_frames)`` or ``(channels, n_bins, n_frames)``.
        sr: Unused (librosa only uses it to compute ``S``).
        lag: Time lag of the difference in frames.
        max_size: Maximum filter across frequency on the reference (1 = off).
        ref: Reference spectrogram of ``S``'s shape; None uses ``S``.
        detrend: High-pass filter the envelope.
        center: Compensate the centred-STFT delay of ``n_fft // (2 * hop_length)`` frames.
        aggregate: Reduction across bins; None = mean. ``False`` returns the
            first bin's flux (as librosa).
        n_fft: FFT size behind ``S`` (centring shift).
        hop_length: Hop behind ``S`` (centring shift).

    Returns:
        numpy.ndarray[float64]: shape ``(n_frames,)`` (2-D ``S``) or
        ``(channels, n_frames)`` (3-D).

    Raises:
        ValueError: Invalid parameters or an empty spectrogram.
    """
    ...

def onset_strength_multi(
    *,
    S: npt.ArrayLike,
    sr: float = 22050,
    n_fft: int = 2048,
    hop_length: int = 512,
    lag: int = 1,
    max_size: int = 1,
    ref: Optional[npt.ArrayLike] = None,
    detrend: bool = False,
    center: bool = True,
    aggregate: Union[AggregateSpec, Literal[False], None] = None,
    channels: Optional[Sequence[int]] = None,
) -> NDArray[np.float64]:
    """Multi-band onset strength, ``librosa.onset.onset_strength_multi(S=S)``.

    Args:
        S: ``(n_bins, n_frames)`` or ``(channels, n_bins, n_frames)`` log-power spectrogram.
        sr, n_fft, hop_length, lag, max_size, ref, detrend, center: As ``onset_strength``.
        aggregate: As ``onset_strength``; ``False`` keeps every bin.
        channels: Sub-band boundaries in bins (sorted, de-duplicated, values
            above ``n_bins`` dropped); None = one band. Slices are not supported.

    Returns:
        numpy.ndarray[float64]: shape ``(n_bands, n_frames)`` or
        ``(channels, n_bands, n_frames)``.
    """
    ...

def peak_pick(
    x: npt.ArrayLike,
    *,
    pre_max: int,
    post_max: int,
    pre_avg: int,
    post_avg: int,
    delta: float,
    wait: int,
    sparse: bool = True,
    method: Literal["greedy", "dp_count", "dp_value"] = "greedy",
    axis: int = -1,
) -> NDArray[np.int64]:
    """Peak picking, ``librosa.util.peak_pick`` (1-D input, sparse output only).

    Returns:
        numpy.ndarray[int64]: ascending peak indices.

    Raises:
        ValueError: Invalid parameters, non-finite input or ``sparse=False``.
    """
    ...

def onset_backtrack(events: npt.ArrayLike, energy: npt.ArrayLike) -> NDArray[np.int64]:
    """Move events back to the preceding local minimum of ``energy``,
    ``librosa.onset.onset_backtrack``. Empty ``events`` give an empty array
    (librosa raises).

    Returns:
        numpy.ndarray[int64]: one frame per event.
    """
    ...

def onset_detect(
    *,
    onset_envelope: npt.ArrayLike,
    sr: float = 22050,
    hop_length: int = 512,
    backtrack: bool = False,
    energy: Optional[npt.ArrayLike] = None,
    units: Units = "frames",
    normalize: bool = True,
    sparse: bool = True,
    pre_max: Optional[int] = None,
    post_max: Optional[int] = None,
    pre_avg: Optional[int] = None,
    post_avg: Optional[int] = None,
    delta: Optional[float] = None,
    wait: Optional[int] = None,
    method: Literal["greedy", "dp_count", "dp_value"] = "greedy",
) -> NDArray[Any]:
    """Onset detection on an envelope, ``librosa.onset.onset_detect(onset_envelope=...)``.

    Peak-picking parameters left as None get librosa's frame-rate defaults
    (30 ms ``pre_max``/``wait``, 100 ms averaging, ``delta=0.07``).

    Returns:
        numpy.ndarray: 1-D int64 frames/samples or float64 times (per ``units``).

    Raises:
        ValueError: Invalid parameters, 2-D envelope or ``sparse=False``.
    """
    ...

def tempo_frequencies(n_bins: int, *, hop_length: int = 512, sr: float = 22050) -> NDArray[np.float64]:
    """BPM of each autocorrelation-tempogram lag bin, ``librosa.tempo_frequencies``
    (bin 0 is ``inf``). Shape ``(n_bins,)``."""
    ...

def fourier_tempo_frequencies(
    *, sr: float = 22050, win_length: int = 384, hop_length: int = 512
) -> NDArray[np.float64]:
    """BPM of each Fourier-tempogram bin, ``librosa.fourier_tempo_frequencies``.
    Shape ``(win_length // 2 + 1,)``."""
    ...

def tempogram(
    *,
    onset_envelope: npt.ArrayLike,
    sr: float = 22050,
    hop_length: int = 512,
    win_length: int = 384,
    center: bool = True,
    window: WindowSpec = "hann",
    norm: Optional[float] = np.inf,
) -> NDArray[np.float64]:
    """Autocorrelation tempogram, ``librosa.feature.tempogram(onset_envelope=...)``.

    Args:
        onset_envelope: ``(n_frames,)`` or ``(channels, n_frames)``.
        sr, hop_length: Unused (librosa uses them only to compute the envelope).
        win_length: Window / number of lag bins.
        center: Centre windows (linear-ramp padding).
        window: Window before the autocorrelation.
        norm: Column normalisation (``numpy.inf`` = max, a p-norm, or None).

    Returns:
        numpy.ndarray[float64]: ``(win_length, n_frames)`` (with a leading channel
        axis for 2-D input); ``n_frames - win_length + 1`` frames when not centred.
    """
    ...

def fourier_tempogram(
    *,
    onset_envelope: npt.ArrayLike,
    sr: float = 22050,
    hop_length: int = 512,
    win_length: int = 384,
    center: bool = True,
    window: WindowSpec = "hann",
) -> NDArray[np.complex128]:
    """Fourier tempogram, ``librosa.feature.fourier_tempogram(onset_envelope=...)``.

    Returns:
        numpy.ndarray[complex128]: ``(win_length // 2 + 1, n_frames + 1)`` when
        centred (leading channel axis for 2-D input).
    """
    ...

def tempo(
    *,
    onset_envelope: Optional[npt.ArrayLike] = None,
    tg: Optional[npt.ArrayLike] = None,
    sr: float = 22050,
    hop_length: int = 512,
    start_bpm: float = 120,
    std_bpm: float = 1.0,
    ac_size: float = 8.0,
    max_tempo: Optional[float] = 320.0,
    aggregate: Optional[AggregateSpec] = "mean",
    prior: PriorSpec = None,
) -> NDArray[np.float64]:
    """Tempo estimate, ``librosa.feature.tempo(onset_envelope=... | tg=...)``.

    Args:
        onset_envelope: ``(n_frames,)`` / ``(channels, n_frames)``; exactly one
            of ``onset_envelope`` and ``tg``.
        tg: Autocorrelation tempogram ``(win_length, n_frames)`` / 3-D.
        aggregate: Reduction over time (default mean); None = per-frame tempo.
        prior: None (log-normal around ``start_bpm``), a ``scipy.stats``
            distribution or a callable of the BPM.

    Returns:
        numpy.ndarray[float64]: ``(1,)`` (``(n_frames,)`` with ``aggregate=None``),
        with a leading channel axis for multi-channel input.
    """
    ...

def beat_track(
    *,
    onset_envelope: npt.ArrayLike,
    sr: float = 22050,
    hop_length: int = 512,
    start_bpm: float = 120.0,
    tightness: float = 100,
    trim: bool = True,
    bpm: Optional[Union[float, npt.ArrayLike]] = None,
    prior: PriorSpec = None,
    units: Units = "frames",
    sparse: bool = True,
) -> tuple[Any, NDArray[Any]]:
    """Dynamic-programming beat tracker, ``librosa.beat.beat_track(onset_envelope=...)``.

    Returns:
        tuple: ``(tempo, beats)`` as librosa: ``tempo`` is a float64 array of
        shape ``(1,)`` (``bpm`` unchanged when given; ``0.0`` for an all-zero
        envelope); ``beats`` 1-D int64 frames/samples or float64 times.

    Raises:
        ValueError: Invalid parameters, a 2-D envelope or ``sparse=False``.
    """
    ...

def beat_track_with_bpm(
    onset_envelope: npt.ArrayLike,
    bpm: Union[float, npt.ArrayLike],
    *,
    sr: float = 22050,
    hop_length: int = 512,
    tightness: float = 100.0,
    trim: bool = True,
) -> NDArray[np.int64]:
    """Beat tracking at a given tempo (one value or one per frame); librosa's
    internal ``__beat_tracker``. Returns ascending int64 beat frames."""
    ...

def plp(
    *,
    onset_envelope: npt.ArrayLike,
    sr: float = 22050,
    hop_length: int = 512,
    win_length: int = 384,
    tempo_min: Optional[float] = 30,
    tempo_max: Optional[float] = 300,
    prior: PriorSpec = None,
) -> NDArray[np.float64]:
    """Predominant local pulse, ``librosa.beat.plp(onset_envelope=...)``.

    Returns:
        numpy.ndarray[float64]: same shape as the envelope, values in ``[0, 1]``.
    """
    ...

def tempogram_ratio(
    *,
    onset_envelope: Optional[npt.ArrayLike] = None,
    tg: Optional[npt.ArrayLike] = None,
    bpm: Optional[npt.ArrayLike] = None,
    sr: float = 22050,
    hop_length: int = 512,
    win_length: int = 384,
    start_bpm: float = 120,
    std_bpm: float = 1.0,
    max_tempo: Optional[float] = 320.0,
    freqs: None = None,
    factors: Optional[npt.ArrayLike] = None,
    aggregate: Optional[AggregateSpec] = None,
    prior: PriorSpec = None,
    center: bool = True,
    window: WindowSpec = "hann",
    kind: Literal["linear"] = "linear",
    fill_value: float = 0,
    norm: Optional[float] = np.inf,
) -> NDArray[np.float64]:
    """Tempogram ratio features, ``librosa.feature.tempogram_ratio``.

    ``freqs`` must be None and ``kind`` ``"linear"``.

    Returns:
        numpy.ndarray[float64]: ``(n_factors, n_frames)`` (``(n_factors,)`` when
        aggregated), with a leading channel axis for multi-channel input.
    """
    ...

def hybrid_tempogram(
    *,
    onset_envelope: npt.ArrayLike,
    sr: float = 22050,
    hop_length: int = 512,
    win_length: int = 384,
    center: bool = True,
    window: WindowSpec = "hann",
) -> NDArray[np.float64]:
    """Hybrid tempogram, ``librosa.feature.hybrid_tempogram(onset_envelope=...)``.

    Returns:
        numpy.ndarray[float64]: ``(win_length // 2 + 1, n_frames)`` on the
        ``fourier_tempo_frequencies`` grid (leading channel axis for 2-D input).
    """
    ...

def metrogram(
    *,
    tg: npt.ArrayLike,
    freqs: npt.ArrayLike,
    factors: Optional[npt.ArrayLike] = None,
    aggregate: AggregateSpec = "sum",
    kind: Literal["linear"] = "linear",
    fill_value: float = 0,
) -> NDArray[np.float64]:
    """Metrogram, ``librosa.feature.metrogram``. ``aggregate=None`` (librosa's
    per-bin 4-D output) is not supported.

    Returns:
        numpy.ndarray[float64]: ``(n_factors, n_frames)`` (leading channel axis
        for 3-D ``tg``).
    """
    ...
