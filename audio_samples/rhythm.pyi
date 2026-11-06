"""Rhythm analysis over NumPy arrays.

Free functions for onsets (``onset_strength``, ``onset_strength_multi``,
``onset_detect``, ``onset_backtrack``, ``peak_pick``), tempograms and tempo
(``tempogram``, ``fourier_tempogram``, ``tempo``, ``tempogram_ratio``,
``hybrid_tempogram``, ``metrogram``, ``tempo_frequencies``,
``fourier_tempo_frequencies``) and beats (``beat_track``,
``beat_track_with_bpm``, ``plp``). ``sr`` is required and ``beat_track`` /
``beat_track_with_bpm`` default to ``trim=False``. They take precomputed
inputs (a spectrogram ``S``, an ``onset_envelope`` or a tempogram ``tg``);
the ``AudioSamples`` methods of the same names compute them from audio.

float32 / complex64 NumPy arrays are processed in float32 and give float32 /
complex64 output; any other array-like (lists, integer, float16 or float64
arrays) is converted to float64 and gives float64 / complex128 output. Tempo
estimates are always float64.

A 1-D envelope (2-D spectrogram) gives a result without a channel axis; a
``(channels, n)`` envelope (3-D spectrogram) adds a leading channel axis.
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
from typing import Any, overload, type_check_only
from typing_extensions import TypeAlias

#: Matches only an array whose dtype is unknown to the type checker (Any). The overloads use it so that
#: such an array gives a result of unknown precision instead of the first overload's.
@type_check_only
class _UnknownDtype(np.generic):
    def __new__(cls) -> _UnknownDtype: ...

#: float32 and complex64 arrays are processed in float32. Anything else is converted to float64.
_SingleArray: TypeAlias = NDArray[np.float32] | NDArray[np.complex64]


#: Window specification: a SciPy window name, ``(name, parameter)``, a number
#: (Kaiser beta) or an array of coefficients.
_WindowSpec = Union[str, tuple[str, float], float, npt.ArrayLike]
#: Reduction: a NumPy reduction, its name, or a callable ``f(x, axis=-1)``.
_AggregateSpec = Union[Callable[..., Any], Literal["mean", "median", "max", "min", "sum"]]
#: Tempo prior: a ``scipy.stats`` distribution (``.logpdf`` used) or a callable of the BPM.
_PriorSpec = Union[Any, Callable[[float], float], None]
#: Event units.
_Units = Literal["frames", "samples", "time"]

@overload
def onset_strength(
    *,
    S: NDArray[_UnknownDtype],
    sr: float,
    lag: int = 1,
    max_size: int = 1,
    ref: Optional[npt.ArrayLike] = None,
    detrend: bool = False,
    center: bool = True,
    aggregate: Union[_AggregateSpec, Literal[False], None] = None,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> NDArray[Any]: ...
@overload
def onset_strength(  # type: ignore[overload-overlap]
    *,
    S: _SingleArray,
    sr: float,
    lag: int = 1,
    max_size: int = 1,
    ref: Optional[npt.ArrayLike] = None,
    detrend: bool = False,
    center: bool = True,
    aggregate: Union[_AggregateSpec, Literal[False], None] = None,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> NDArray[np.float32]: ...
@overload
def onset_strength(
    *,
    S: npt.ArrayLike,
    sr: float,
    lag: int = 1,
    max_size: int = 1,
    ref: Optional[npt.ArrayLike] = None,
    detrend: bool = False,
    center: bool = True,
    aggregate: Union[_AggregateSpec, Literal[False], None] = None,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> NDArray[np.float64]:
    """Onset strength of a log-power spectrogram: positive differences between
    frames, aggregated across bins.

    ``S`` is used as given (compute it with ``spectral.power_to_db`` of a mel
    spectrogram, or use ``AudioSamples.onset_strength``).

    Args:
        S: ``(n_bins, n_frames)`` or ``(channels, n_bins, n_frames)``.
        sr: Unused (``S`` is already computed).
        lag: Time lag of the difference in frames.
        max_size: Maximum filter across frequency on the reference (1 = off).
        ref: Reference spectrogram of ``S``'s shape; None uses ``S``.
        detrend: High-pass filter the envelope.
        center: Compensate the centred-STFT delay of ``n_fft // (2 * hop_length)`` frames.
        aggregate: Reduction across bins; None = mean. ``False`` returns the
            first bin's flux.
        n_fft: FFT size behind ``S`` (centring shift).
        hop_length: Hop behind ``S`` (centring shift).

    Returns:
        numpy.ndarray[float32 | float64]: shape ``(n_frames,)`` (2-D ``S``) or
        ``(channels, n_frames)`` (3-D).

    Raises:
        ValueError: Invalid parameters or an empty spectrogram.
    """
    ...

@overload
def onset_strength_multi(
    *,
    S: NDArray[_UnknownDtype],
    sr: float,
    n_fft: int = 2048,
    hop_length: int = 512,
    lag: int = 1,
    max_size: int = 1,
    ref: Optional[npt.ArrayLike] = None,
    detrend: bool = False,
    center: bool = True,
    aggregate: Union[_AggregateSpec, Literal[False], None] = None,
    channels: Optional[Union[Sequence[int], Sequence[slice]]] = None,
) -> NDArray[Any]: ...
@overload
def onset_strength_multi(  # type: ignore[overload-overlap]
    *,
    S: _SingleArray,
    sr: float,
    n_fft: int = 2048,
    hop_length: int = 512,
    lag: int = 1,
    max_size: int = 1,
    ref: Optional[npt.ArrayLike] = None,
    detrend: bool = False,
    center: bool = True,
    aggregate: Union[_AggregateSpec, Literal[False], None] = None,
    channels: Optional[Union[Sequence[int], Sequence[slice]]] = None,
) -> NDArray[np.float32]: ...
@overload
def onset_strength_multi(
    *,
    S: npt.ArrayLike,
    sr: float,
    n_fft: int = 2048,
    hop_length: int = 512,
    lag: int = 1,
    max_size: int = 1,
    ref: Optional[npt.ArrayLike] = None,
    detrend: bool = False,
    center: bool = True,
    aggregate: Union[_AggregateSpec, Literal[False], None] = None,
    channels: Optional[Union[Sequence[int], Sequence[slice]]] = None,
) -> NDArray[np.float64]:
    """Multi-band onset strength: one envelope per group of bins.

    Args:
        S: ``(n_bins, n_frames)`` or ``(channels, n_bins, n_frames)`` log-power spectrogram.
        sr, n_fft, hop_length, lag, max_size, ref, detrend, center: As ``onset_strength``.
        aggregate: As ``onset_strength``; ``False`` keeps every bin.
        channels: Sub-band boundaries in bins (sorted, de-duplicated, values
            above ``n_bins`` dropped), or one slice of the bins per band; None = one
            band.

    Returns:
        numpy.ndarray[float32 | float64]: shape ``(n_bands, n_frames)`` or
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
    """Peak picking against a local maximum and a local mean threshold (1-D
    input, sparse output only).

    Returns:
        numpy.ndarray[int64]: ascending peak indices.

    Raises:
        ValueError: Invalid parameters, non-finite input or ``sparse=False``.
    """
    ...

def onset_backtrack(events: npt.ArrayLike, energy: npt.ArrayLike) -> NDArray[np.int64]:
    """Move events back to the preceding local minimum of ``energy``.
    Empty ``events`` give an empty array.

    Returns:
        numpy.ndarray[int64]: one frame per event.
    """
    ...

@overload
def onset_detect(
    *,
    onset_envelope: npt.ArrayLike,
    sr: float,
    hop_length: int = 512,
    backtrack: bool = False,
    energy: Optional[npt.ArrayLike] = None,
    units: Literal['time'],
    normalize: bool = True,
    sparse: bool = True,
    pre_max: Optional[int] = None,
    post_max: Optional[int] = None,
    pre_avg: Optional[int] = None,
    post_avg: Optional[int] = None,
    delta: Optional[float] = None,
    wait: Optional[int] = None,
    method: Literal['greedy', 'dp_count', 'dp_value'] = 'greedy',
) -> NDArray[np.float64]: ...
@overload
def onset_detect(
    *,
    onset_envelope: npt.ArrayLike,
    sr: float,
    hop_length: int = 512,
    backtrack: bool = False,
    energy: Optional[npt.ArrayLike] = None,
    units: Literal['frames', 'samples'] = 'frames',
    normalize: bool = True,
    sparse: bool = True,
    pre_max: Optional[int] = None,
    post_max: Optional[int] = None,
    pre_avg: Optional[int] = None,
    post_avg: Optional[int] = None,
    delta: Optional[float] = None,
    wait: Optional[int] = None,
    method: Literal['greedy', 'dp_count', 'dp_value'] = 'greedy',
) -> NDArray[np.int64]:
    """Onset detection by peak picking on an onset strength envelope.

    Peak-picking parameters left as None are set from the frame rate
    (30 ms ``pre_max``/``wait``, 100 ms averaging, ``delta=0.07``).

    Returns:
        numpy.ndarray: 1-D int64 frames/samples or float64 times (per ``units``).

    Raises:
        ValueError: Invalid parameters, 2-D envelope or ``sparse=False``.
    """
    ...

def tempo_frequencies(n_bins: int, *, hop_length: int = 512, sr: float) -> NDArray[np.float64]:
    """BPM of each autocorrelation-tempogram lag bin (bin 0 is ``inf``).
    Shape ``(n_bins,)``."""
    ...

def fourier_tempo_frequencies(
    *, sr: float, win_length: int = 384, hop_length: int = 512
) -> NDArray[np.float64]:
    """BPM of each Fourier-tempogram bin. Shape ``(win_length // 2 + 1,)``."""
    ...

@overload
def tempogram(
    *,
    onset_envelope: NDArray[_UnknownDtype],
    sr: float,
    hop_length: int = 512,
    win_length: int = 384,
    center: bool = True,
    window: _WindowSpec = 'hann',
    norm: Optional[float] = np.inf,
) -> NDArray[Any]: ...
@overload
def tempogram(  # type: ignore[overload-overlap]
    *,
    onset_envelope: _SingleArray,
    sr: float,
    hop_length: int = 512,
    win_length: int = 384,
    center: bool = True,
    window: _WindowSpec = 'hann',
    norm: Optional[float] = np.inf,
) -> NDArray[np.float32]: ...
@overload
def tempogram(
    *,
    onset_envelope: npt.ArrayLike,
    sr: float,
    hop_length: int = 512,
    win_length: int = 384,
    center: bool = True,
    window: _WindowSpec = 'hann',
    norm: Optional[float] = np.inf,
) -> NDArray[np.float64]:
    """Autocorrelation tempogram: windowed autocorrelation of an onset strength envelope.

    Args:
        onset_envelope: ``(n_frames,)`` or ``(channels, n_frames)``.
        sr, hop_length: Unused (the envelope is already computed).
        win_length: Window / number of lag bins.
        center: Centre windows (linear-ramp padding).
        window: Window before the autocorrelation.
        norm: Column normalisation (``numpy.inf`` = max, a p-norm, or None).

    Returns:
        numpy.ndarray[float32 | float64]: ``(win_length, n_frames)`` (with a leading channel
        axis for 2-D input); ``n_frames - win_length + 1`` frames when not centred.
    """
    ...

@overload
def fourier_tempogram(
    *,
    onset_envelope: NDArray[_UnknownDtype],
    sr: float,
    hop_length: int = 512,
    win_length: int = 384,
    center: bool = True,
    window: _WindowSpec = 'hann',
) -> NDArray[Any]: ...
@overload
def fourier_tempogram(  # type: ignore[overload-overlap]
    *,
    onset_envelope: _SingleArray,
    sr: float,
    hop_length: int = 512,
    win_length: int = 384,
    center: bool = True,
    window: _WindowSpec = 'hann',
) -> NDArray[np.complex64]: ...
@overload
def fourier_tempogram(
    *,
    onset_envelope: npt.ArrayLike,
    sr: float,
    hop_length: int = 512,
    win_length: int = 384,
    center: bool = True,
    window: _WindowSpec = 'hann',
) -> NDArray[np.complex128]:
    """Fourier tempogram: short-time Fourier transform of an onset strength envelope.

    Returns:
        numpy.ndarray[complex64 | complex128]: ``(win_length // 2 + 1, n_frames + 1)`` when
        centred (leading channel axis for 2-D input).
    """
    ...

def tempo(
    *,
    onset_envelope: Optional[npt.ArrayLike] = None,
    tg: Optional[npt.ArrayLike] = None,
    sr: float,
    hop_length: int = 512,
    start_bpm: float = 120,
    std_bpm: float = 1.0,
    ac_size: float = 8.0,
    max_tempo: Optional[float] = 320.0,
    aggregate: Optional[_AggregateSpec] = "mean",
    prior: _PriorSpec = None,
) -> NDArray[np.float64]:
    """Tempo estimate in BPM from an onset envelope or a precomputed tempogram ``tg``.

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

@overload
def beat_track(
    *,
    onset_envelope: npt.ArrayLike,
    sr: float,
    hop_length: int = 512,
    start_bpm: float = 120.0,
    tightness: float = 100,
    trim: bool = False,
    bpm: Optional[Union[float, npt.ArrayLike]] = None,
    prior: _PriorSpec = None,
    units: Literal['time'],
    sparse: bool = True,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...
@overload
def beat_track(
    *,
    onset_envelope: npt.ArrayLike,
    sr: float,
    hop_length: int = 512,
    start_bpm: float = 120.0,
    tightness: float = 100,
    trim: bool = False,
    bpm: Optional[Union[float, npt.ArrayLike]] = None,
    prior: _PriorSpec = None,
    units: Literal['frames', 'samples'] = 'frames',
    sparse: bool = True,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Dynamic-programming beat tracker on an onset strength envelope.

    Returns:
        tuple: ``(tempo, beats)``: ``tempo`` is a float64 array of
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
    sr: float,
    hop_length: int = 512,
    tightness: float = 100.0,
    trim: bool = False,
) -> NDArray[np.int64]:
    """Dynamic-programming beat tracking at a given tempo (one value or one per
    frame), without tempo estimation. Returns ascending int64 beat frames."""
    ...

@overload
def plp(
    *,
    onset_envelope: NDArray[_UnknownDtype],
    sr: float,
    hop_length: int = 512,
    win_length: int = 384,
    tempo_min: Optional[float] = 30,
    tempo_max: Optional[float] = 300,
    prior: _PriorSpec = None,
) -> NDArray[Any]: ...
@overload
def plp(  # type: ignore[overload-overlap]
    *,
    onset_envelope: _SingleArray,
    sr: float,
    hop_length: int = 512,
    win_length: int = 384,
    tempo_min: Optional[float] = 30,
    tempo_max: Optional[float] = 300,
    prior: _PriorSpec = None,
) -> NDArray[np.float32]: ...
@overload
def plp(
    *,
    onset_envelope: npt.ArrayLike,
    sr: float,
    hop_length: int = 512,
    win_length: int = 384,
    tempo_min: Optional[float] = 30,
    tempo_max: Optional[float] = 300,
    prior: _PriorSpec = None,
) -> NDArray[np.float64]:
    """Predominant local pulse from the Fourier tempogram of an onset envelope.

    Returns:
        numpy.ndarray[float32 | float64]: same shape as the envelope, values in ``[0, 1]``.
    """
    ...

@overload
def tempogram_ratio(
    *,
    onset_envelope: NDArray[_UnknownDtype],
    tg: None = None,
    bpm: Optional[npt.ArrayLike] = None,
    sr: float,
    hop_length: int = 512,
    win_length: int = 384,
    start_bpm: float = 120,
    std_bpm: float = 1.0,
    max_tempo: Optional[float] = 320.0,
    freqs: None = None,
    factors: Optional[npt.ArrayLike] = None,
    aggregate: Optional[_AggregateSpec] = None,
    prior: _PriorSpec = None,
    center: bool = True,
    window: _WindowSpec = 'hann',
    kind: Literal['linear'] = 'linear',
    fill_value: float = 0,
    norm: Optional[float] = np.inf,
) -> NDArray[Any]: ...
@overload
def tempogram_ratio(
    *,
    onset_envelope: None = None,
    tg: NDArray[_UnknownDtype],
    bpm: Optional[npt.ArrayLike] = None,
    sr: float,
    hop_length: int = 512,
    win_length: int = 384,
    start_bpm: float = 120,
    std_bpm: float = 1.0,
    max_tempo: Optional[float] = 320.0,
    freqs: None = None,
    factors: Optional[npt.ArrayLike] = None,
    aggregate: Optional[_AggregateSpec] = None,
    prior: _PriorSpec = None,
    center: bool = True,
    window: _WindowSpec = 'hann',
    kind: Literal['linear'] = 'linear',
    fill_value: float = 0,
    norm: Optional[float] = np.inf,
) -> NDArray[Any]: ...
@overload
def tempogram_ratio(  # type: ignore[overload-overlap]
    *,
    onset_envelope: _SingleArray,
    tg: None = None,
    bpm: Optional[npt.ArrayLike] = None,
    sr: float,
    hop_length: int = 512,
    win_length: int = 384,
    start_bpm: float = 120,
    std_bpm: float = 1.0,
    max_tempo: Optional[float] = 320.0,
    freqs: None = None,
    factors: Optional[npt.ArrayLike] = None,
    aggregate: Optional[_AggregateSpec] = None,
    prior: _PriorSpec = None,
    center: bool = True,
    window: _WindowSpec = 'hann',
    kind: Literal['linear'] = 'linear',
    fill_value: float = 0,
    norm: Optional[float] = np.inf,
) -> NDArray[np.float32]: ...
@overload
def tempogram_ratio(  # type: ignore[overload-overlap]
    *,
    onset_envelope: None = None,
    tg: _SingleArray,
    bpm: Optional[npt.ArrayLike] = None,
    sr: float,
    hop_length: int = 512,
    win_length: int = 384,
    start_bpm: float = 120,
    std_bpm: float = 1.0,
    max_tempo: Optional[float] = 320.0,
    freqs: None = None,
    factors: Optional[npt.ArrayLike] = None,
    aggregate: Optional[_AggregateSpec] = None,
    prior: _PriorSpec = None,
    center: bool = True,
    window: _WindowSpec = 'hann',
    kind: Literal['linear'] = 'linear',
    fill_value: float = 0,
    norm: Optional[float] = np.inf,
) -> NDArray[np.float32]: ...
@overload
def tempogram_ratio(
    *,
    onset_envelope: Optional[npt.ArrayLike] = None,
    tg: Optional[npt.ArrayLike] = None,
    bpm: Optional[npt.ArrayLike] = None,
    sr: float,
    hop_length: int = 512,
    win_length: int = 384,
    start_bpm: float = 120,
    std_bpm: float = 1.0,
    max_tempo: Optional[float] = 320.0,
    freqs: None = None,
    factors: Optional[npt.ArrayLike] = None,
    aggregate: Optional[_AggregateSpec] = None,
    prior: _PriorSpec = None,
    center: bool = True,
    window: _WindowSpec = 'hann',
    kind: Literal['linear'] = 'linear',
    fill_value: float = 0,
    norm: Optional[float] = np.inf,
) -> NDArray[np.float64]:
    """Tempogram ratio features: the tempogram sampled at fixed multiples of the
    per-frame tempo estimate.

    ``freqs`` must be None and ``kind`` ``"linear"``.

    Returns:
        numpy.ndarray[float32 | float64]: ``(n_factors, n_frames)`` (``(n_factors,)`` when
        aggregated), with a leading channel axis for multi-channel input.
    """
    ...

@overload
def hybrid_tempogram(
    *,
    onset_envelope: NDArray[_UnknownDtype],
    sr: float,
    hop_length: int = 512,
    win_length: int = 384,
    center: bool = True,
    window: _WindowSpec = 'hann',
) -> NDArray[Any]: ...
@overload
def hybrid_tempogram(  # type: ignore[overload-overlap]
    *,
    onset_envelope: _SingleArray,
    sr: float,
    hop_length: int = 512,
    win_length: int = 384,
    center: bool = True,
    window: _WindowSpec = 'hann',
) -> NDArray[np.float32]: ...
@overload
def hybrid_tempogram(
    *,
    onset_envelope: npt.ArrayLike,
    sr: float,
    hop_length: int = 512,
    win_length: int = 384,
    center: bool = True,
    window: _WindowSpec = 'hann',
) -> NDArray[np.float64]:
    """Hybrid tempogram: the Fourier and autocorrelation tempograms combined on
    the Fourier tempo grid.

    Returns:
        numpy.ndarray[float32 | float64]: ``(win_length // 2 + 1, n_frames)`` on the
        ``fourier_tempo_frequencies`` grid (leading channel axis for 2-D input).
    """
    ...

@overload
def metrogram(
    *,
    tg: NDArray[_UnknownDtype],
    freqs: npt.ArrayLike,
    factors: Optional[npt.ArrayLike] = None,
    aggregate: Optional[_AggregateSpec] = 'sum',
    kind: Literal['linear'] = 'linear',
    fill_value: float = 0,
) -> NDArray[Any]: ...
@overload
def metrogram(  # type: ignore[overload-overlap]
    *,
    tg: _SingleArray,
    freqs: npt.ArrayLike,
    factors: Optional[npt.ArrayLike] = None,
    aggregate: Optional[_AggregateSpec] = 'sum',
    kind: Literal['linear'] = 'linear',
    fill_value: float = 0,
) -> NDArray[np.float32]: ...
@overload
def metrogram(
    *,
    tg: npt.ArrayLike,
    freqs: npt.ArrayLike,
    factors: Optional[npt.ArrayLike] = None,
    aggregate: Optional[_AggregateSpec] = 'sum',
    kind: Literal['linear'] = 'linear',
    fill_value: float = 0,
) -> NDArray[np.float64]:
    """Metrogram (metrical-subharmonic salience): for each factor ``h``, the
    tempogram interpolated at ``h * freqs``, multiplied by the tempogram and
    aggregated over the tempo axis.

    Returns:
        numpy.ndarray[float32 | float64]: ``(n_factors, n_frames)``, or
        ``(n_factors, n_freqs, n_frames)`` with ``aggregate=None`` (leading channel axis
        for 3-D ``tg``).
    """
    ...

__all__ = [
    "__doc__",
    "beat_track",
    "beat_track_with_bpm",
    "fourier_tempo_frequencies",
    "fourier_tempogram",
    "hybrid_tempogram",
    "metrogram",
    "onset_backtrack",
    "onset_detect",
    "onset_strength",
    "onset_strength_multi",
    "peak_pick",
    "plp",
    "tempo",
    "tempo_frequencies",
    "tempogram",
    "tempogram_ratio",
]
