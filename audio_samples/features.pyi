"""librosa-compatible feature, tonal, constant-Q and pitch helpers over NumPy arrays.

``audio_samples.features`` collects the array-level (``S=`` / ``y=``) forms of
``librosa.feature``, the tonal and harmonic helpers of librosa's core, and the
``librosa.filters`` / ``librosa.util`` functions they rely on. Mapping:

=====================================  ==========================================
librosa                                audio_samples.features
=====================================  ==========================================
``feature.spectral_centroid(S=...)``   ``spectral_centroid_from_spectrogram``
``feature.spectral_bandwidth(S=...)``  ``spectral_bandwidth_from_spectrogram``
``feature.spectral_rolloff(S=...)``    ``spectral_rolloff_from_spectrogram``
``feature.spectral_flatness(S=...)``   ``spectral_flatness_from_spectrogram``
``feature.spectral_contrast(S=...)``   ``spectral_contrast_from_spectrogram``
``feature.poly_features(S=...)``       ``poly_features_from_spectrogram``
``feature.rms(S=...)``                 ``rms_from_spectrogram``
``pcen``                               ``pcen``
``A_weighting`` ... ``Z_weighting``    ``a_weighting`` ... ``z_weighting``
``frequency_weighting``                ``frequency_weighting``
``multi_frequency_weighting``          ``multi_frequency_weighting``
``perceptual_weighting``               ``perceptual_weighting``
``filters.chroma``                     ``chroma_filterbank``
``util.normalize``                     ``normalize``
``pitch_tuning``                       ``pitch_tuning``
``feature.chroma_stft(S=...)``         ``chroma_stft_from_spectrogram``
``piptrack(S=...)``                    ``piptrack_from_spectrogram``
``estimate_tuning(S=...)``             ``estimate_tuning_from_spectrogram``
``estimate_tuning(y=...)``             ``estimate_tuning_from_signal``
``filters.cq_to_chroma``               ``cq_to_chroma``
``filters.wavelet``                    ``wavelet``
``filters.wavelet_lengths``            ``wavelet_lengths``
``feature.chroma_cens(C=...)``         ``chroma_cens_from_chroma`` (chroma input)
``feature.tonnetz(chroma=...)``        ``tonnetz_from_chroma``
``yin`` / ``pyin``                     ``yin`` / ``pyin``
``salience``                           ``salience``
``interp_harmonics``                   ``interp_harmonics``
``f0_harmonics``                       ``f0_harmonics``
=====================================  ==========================================

The ``y=`` forms of the frame-wise features, chroma, CQT and pitch trackers
are ``AudioSamples`` methods (``spectral_centroid_frames``, ``chroma_stft``,
``cqt``, ``pyin``, ...); the tuning / octave / key conversions are in
``audio_samples.utils`` (``hz_to_octs``, ``cqt_frequencies``,
``key_to_notes``, ...).

float32 / complex64 NumPy arrays are processed in float32 and give float32 /
complex64 output; any other array-like (lists, integer, float16 or float64
arrays) is converted to float64 and gives float64 / complex128 output.
Shapes follow librosa: a 2-D ``(F, T)`` spectrogram gives 2-D output and a 3-D
``(channels, F, T)`` spectrogram 3-D output (a 1-D / 2-D signal for ``yin`` /
``pyin``). Filterbanks, wavelets and frequency weightings are float64 /
complex128 tables (librosa defaults to float32 / complex64).

``sr`` is required wherever librosa defaults it to 22050, and ``pcen`` also
requires ``hop_length`` (librosa: 512). Other defaults are librosa 1.0.0's,
except ``tuning=0.0`` in ``chroma_stft_from_spectrogram`` (librosa: None,
estimated), ``top_db=None`` in ``perceptual_weighting`` (librosa: 80) and
``min_db=None`` in ``frequency_weighting`` / ``multi_frequency_weighting``
(librosa: -80).
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


#: Window specification: a SciPy window name, ``(name, param)`` tuple, a
#: number (Kaiser beta) or an array of coefficients.
_WindowSpec = Union[str, tuple[str, float], float, npt.ArrayLike]
#: Padding mode used when ``center=True``.
_PadMode = Literal["constant", "reflect", "edge", "symmetric"]
#: dB reference: a number, ``"max"`` or ``numpy.max``.
_DbRef = Union[float, Literal["max"], Callable[..., Any]]
#: Weighting curve name.
_WeightingKind = Literal["A", "B", "C", "D", "Z", "a", "b", "c", "d", "z"]

# ---------------------------------------------------------------------------
# Spectral descriptors from a spectrogram (librosa.feature.*(S=...))
# ---------------------------------------------------------------------------

@overload
def spectral_centroid_from_spectrogram(
    S: NDArray[_UnknownDtype],
    *,
    sr: float,
    freq: Optional[npt.ArrayLike] = None,
) -> NDArray[Any]: ...
@overload
def spectral_centroid_from_spectrogram(  # type: ignore[overload-overlap]
    S: _SingleArray,
    *,
    sr: float,
    freq: Optional[npt.ArrayLike] = None,
) -> NDArray[np.float32]: ...
@overload
def spectral_centroid_from_spectrogram(
    S: npt.ArrayLike,
    *,
    sr: float,
    freq: Optional[npt.ArrayLike] = None,
) -> NDArray[np.float64]:
    """Spectral centroid per frame, ``librosa.feature.spectral_centroid(S=S, sr=sr, freq=freq)``.

    Args:
        S: Non-negative magnitude spectrogram, ``(F, T)`` or ``(channels, F, T)``.
        sr: Sample rate for the default ``freq``.
        freq: 1-D bin centre frequencies (Hz); None = ``fft_frequencies(sr, 2 * (F - 1))``.
            Time-varying 2-D ``freq`` (accepted by librosa) is not supported.

    Returns:
        numpy.ndarray[float32 | float64], shape ``(1, T)`` or ``(channels, 1, T)`` in Hz.

    Raises:
        ValueError: Negative / non-finite ``S`` or a ``freq`` of the wrong length.
    """
    ...

@overload
def spectral_bandwidth_from_spectrogram(
    S: NDArray[_UnknownDtype],
    *,
    sr: float,
    freq: Optional[npt.ArrayLike] = None,
    centroid: Optional[npt.ArrayLike] = None,
    norm: bool = True,
    p: float = 2.0,
) -> NDArray[Any]: ...
@overload
def spectral_bandwidth_from_spectrogram(  # type: ignore[overload-overlap]
    S: _SingleArray,
    *,
    sr: float,
    freq: Optional[npt.ArrayLike] = None,
    centroid: Optional[npt.ArrayLike] = None,
    norm: bool = True,
    p: float = 2.0,
) -> NDArray[np.float32]: ...
@overload
def spectral_bandwidth_from_spectrogram(
    S: npt.ArrayLike,
    *,
    sr: float,
    freq: Optional[npt.ArrayLike] = None,
    centroid: Optional[npt.ArrayLike] = None,
    norm: bool = True,
    p: float = 2.0,
) -> NDArray[np.float64]:
    """Order-``p`` spectral bandwidth per frame, ``librosa.feature.spectral_bandwidth(S=S, ...)``.

    Args:
        S: Non-negative spectrogram, ``(F, T)`` or ``(channels, F, T)``.
        sr: Sample rate for the default ``freq``.
        freq: 1-D bin frequencies; None = FFT bin centres.
        centroid: Precomputed centroid ``(1, T)`` / ``(channels, 1, T)``; None computes it.
        norm: Normalise each frame to unit sum first.
        p: Order of the deviation (finite, non-zero).

    Returns:
        numpy.ndarray[float32 | float64], shape ``(1, T)`` or ``(channels, 1, T)`` in Hz.
    """
    ...

@overload
def spectral_rolloff_from_spectrogram(
    S: NDArray[_UnknownDtype],
    *,
    sr: float,
    freq: Optional[npt.ArrayLike] = None,
    roll_percent: float = 0.85,
) -> NDArray[Any]: ...
@overload
def spectral_rolloff_from_spectrogram(  # type: ignore[overload-overlap]
    S: _SingleArray,
    *,
    sr: float,
    freq: Optional[npt.ArrayLike] = None,
    roll_percent: float = 0.85,
) -> NDArray[np.float32]: ...
@overload
def spectral_rolloff_from_spectrogram(
    S: npt.ArrayLike,
    *,
    sr: float,
    freq: Optional[npt.ArrayLike] = None,
    roll_percent: float = 0.85,
) -> NDArray[np.float64]:
    """Roll-off frequency per frame, ``librosa.feature.spectral_rolloff(S=S, ...)``.

    Returns:
        numpy.ndarray[float32 | float64], shape ``(1, T)`` or ``(channels, 1, T)`` in Hz.

    Raises:
        ValueError: ``roll_percent`` outside ``(0, 1)`` or invalid ``S``.
    """
    ...

@overload
def spectral_flatness_from_spectrogram(
    S: NDArray[_UnknownDtype],
    *,
    amin: float = 1e-10,
    power: float = 2.0,
) -> NDArray[Any]: ...
@overload
def spectral_flatness_from_spectrogram(  # type: ignore[overload-overlap]
    S: _SingleArray,
    *,
    amin: float = 1e-10,
    power: float = 2.0,
) -> NDArray[np.float32]: ...
@overload
def spectral_flatness_from_spectrogram(
    S: npt.ArrayLike,
    *,
    amin: float = 1e-10,
    power: float = 2.0,
) -> NDArray[np.float64]:
    """Spectral flatness per frame, ``librosa.feature.spectral_flatness(S=S, amin=amin, power=power)``.

    Args:
        S: Non-negative magnitude spectrogram, ``(F, T)`` or ``(channels, F, T)``.
        amin: Floor applied to ``S ** power``.
        power: Exponent applied to ``S``.

    Returns:
        numpy.ndarray[float32 | float64], shape ``(1, T)`` or ``(channels, 1, T)``, values in ``[0, 1]``.
    """
    ...

@overload
def spectral_contrast_from_spectrogram(
    S: NDArray[_UnknownDtype],
    *,
    sr: float,
    freq: Optional[npt.ArrayLike] = None,
    fmin: float = 200.0,
    n_bands: int = 6,
    quantile: float = 0.02,
    linear: bool = False,
) -> NDArray[Any]: ...
@overload
def spectral_contrast_from_spectrogram(  # type: ignore[overload-overlap]
    S: _SingleArray,
    *,
    sr: float,
    freq: Optional[npt.ArrayLike] = None,
    fmin: float = 200.0,
    n_bands: int = 6,
    quantile: float = 0.02,
    linear: bool = False,
) -> NDArray[np.float32]: ...
@overload
def spectral_contrast_from_spectrogram(
    S: npt.ArrayLike,
    *,
    sr: float,
    freq: Optional[npt.ArrayLike] = None,
    fmin: float = 200.0,
    n_bands: int = 6,
    quantile: float = 0.02,
    linear: bool = False,
) -> NDArray[np.float64]:
    """Octave-band spectral contrast, ``librosa.feature.spectral_contrast(S=S, ...)``.

    Returns:
        numpy.ndarray[float32 | float64], shape ``(n_bands + 1, T)`` or ``(channels, n_bands + 1, T)``
        (dB differences, or linear differences with ``linear=True``).

    Raises:
        ValueError: Invalid parameters (e.g. the top band above Nyquist).
    """
    ...

@overload
def poly_features_from_spectrogram(
    S: NDArray[_UnknownDtype],
    *,
    sr: float,
    freq: Optional[npt.ArrayLike] = None,
    order: int = 1,
) -> NDArray[Any]: ...
@overload
def poly_features_from_spectrogram(  # type: ignore[overload-overlap]
    S: _SingleArray,
    *,
    sr: float,
    freq: Optional[npt.ArrayLike] = None,
    order: int = 1,
) -> NDArray[np.float32]: ...
@overload
def poly_features_from_spectrogram(
    S: npt.ArrayLike,
    *,
    sr: float,
    freq: Optional[npt.ArrayLike] = None,
    order: int = 1,
) -> NDArray[np.float64]:
    """Polynomial fit per frame, ``librosa.feature.poly_features(S=S, ...)``.

    Returns:
        numpy.ndarray[float32 | float64], shape ``(order + 1, T)`` or ``(channels, order + 1, T)``,
        highest power first (as ``numpy.polyfit``).
    """
    ...

@overload
def rms_from_spectrogram(S: NDArray[_UnknownDtype], *, frame_length: int=2048) -> NDArray[Any]: ...
@overload
def rms_from_spectrogram(  # type: ignore[overload-overlap]
    S: _SingleArray,
    *,
    frame_length: int = 2048,
) -> NDArray[np.float32]: ...
@overload
def rms_from_spectrogram(S: npt.ArrayLike, *, frame_length: int=2048) -> NDArray[np.float64]:
    """RMS per frame from STFT magnitudes (Parseval), ``librosa.feature.rms(S=S, frame_length=...)``.

    Args:
        S: STFT magnitudes with ``1 + frame_length // 2`` bins, 2-D or 3-D.
        frame_length: FFT length used for ``S``.

    Returns:
        numpy.ndarray[float32 | float64], shape ``(1, T)`` or ``(channels, 1, T)``
        (librosa's default dtype is float32).
    """
    ...

@overload
def pcen(
    S: NDArray[_UnknownDtype],
    *,
    sr: float,
    hop_length: int,
    gain: float = 0.98,
    bias: float = 2.0,
    power: float = 0.5,
    time_constant: float = 0.4,
    eps: float = 1e-06,
    b: Optional[float] = None,
    max_size: int = 1,
    ref: Optional[npt.ArrayLike] = None,
    axis: int = -1,
    max_axis: Optional[int] = None,
    zi: Optional[npt.ArrayLike] = None,
    return_zf: Literal[False] = False,
) -> NDArray[Any]: ...
@overload
def pcen(
    S: NDArray[_UnknownDtype],
    *,
    sr: float,
    hop_length: int,
    gain: float = 0.98,
    bias: float = 2.0,
    power: float = 0.5,
    time_constant: float = 0.4,
    eps: float = 1e-06,
    b: Optional[float] = None,
    max_size: int = 1,
    ref: Optional[npt.ArrayLike] = None,
    axis: int = -1,
    max_axis: Optional[int] = None,
    zi: Optional[npt.ArrayLike] = None,
    return_zf: Literal[True],
) -> tuple[NDArray[Any], NDArray[Any]]: ...
@overload
def pcen(  # type: ignore[overload-overlap]
    S: _SingleArray,
    *,
    sr: float,
    hop_length: int,
    gain: float = 0.98,
    bias: float = 2.0,
    power: float = 0.5,
    time_constant: float = 0.4,
    eps: float = 1e-06,
    b: Optional[float] = None,
    max_size: int = 1,
    ref: Optional[npt.ArrayLike] = None,
    axis: int = -1,
    max_axis: Optional[int] = None,
    zi: Optional[npt.ArrayLike] = None,
    return_zf: Literal[False] = False,
) -> NDArray[np.float32]: ...
@overload
def pcen(  # type: ignore[overload-overlap]
    S: _SingleArray,
    *,
    sr: float,
    hop_length: int,
    gain: float = 0.98,
    bias: float = 2.0,
    power: float = 0.5,
    time_constant: float = 0.4,
    eps: float = 1e-06,
    b: Optional[float] = None,
    max_size: int = 1,
    ref: Optional[npt.ArrayLike] = None,
    axis: int = -1,
    max_axis: Optional[int] = None,
    zi: Optional[npt.ArrayLike] = None,
    return_zf: Literal[True],
) -> tuple[NDArray[np.float32], NDArray[np.float32]]: ...
@overload
def pcen(
    S: npt.ArrayLike,
    *,
    sr: float,
    hop_length: int,
    gain: float = 0.98,
    bias: float = 2.0,
    power: float = 0.5,
    time_constant: float = 0.4,
    eps: float = 1e-06,
    b: Optional[float] = None,
    max_size: int = 1,
    ref: Optional[npt.ArrayLike] = None,
    axis: int = -1,
    max_axis: Optional[int] = None,
    zi: Optional[npt.ArrayLike] = None,
    return_zf: Literal[False] = False,
) -> NDArray[np.float64]: ...
@overload
def pcen(
    S: npt.ArrayLike,
    *,
    sr: float,
    hop_length: int,
    gain: float = 0.98,
    bias: float = 2.0,
    power: float = 0.5,
    time_constant: float = 0.4,
    eps: float = 1e-06,
    b: Optional[float] = None,
    max_size: int = 1,
    ref: Optional[npt.ArrayLike] = None,
    axis: int = -1,
    max_axis: Optional[int] = None,
    zi: Optional[npt.ArrayLike] = None,
    return_zf: Literal[True],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Per-channel energy normalisation, ``librosa.pcen``.

    Time is the last axis and the max filter runs along frequency (axis -2); other
    ``axis`` / ``max_axis`` values raise ValueError.

    Args:
        S: Non-negative spectrogram, ``(F, T)`` or ``(channels, F, T)``. Complex input
            raises TypeError (pass ``numpy.abs(S)``).
        sr, hop_length: Convert ``time_constant`` to frames.
        gain, bias, power, time_constant, eps, b, max_size: As librosa.
        ref: Array (shape of ``S``) to smooth instead of ``S``.
        zi: Initial filter state ``(..., F, 1)``, e.g. a previous ``zf``.
        return_zf: Also return the final state.

    Returns:
        numpy.ndarray[float32 | float64] shaped like ``S``; with ``return_zf=True`` a tuple
        ``(pcen, zf)`` with ``zf`` of shape ``(..., F, 1)``.
    """
    ...

@overload
def a_weighting(frequencies: float, *, min_db: Optional[float]=-80.0) -> float: ...
@overload
def a_weighting(
    frequencies: npt.ArrayLike,
    *,
    min_db: Optional[float] = -80.0,
) -> NDArray[np.float64]:
    """A-weighting in dB, ``librosa.A_weighting``. Returns a float for scalar input,
    otherwise numpy.ndarray[float64] of the input's shape."""
    ...

@overload
def b_weighting(frequencies: float, *, min_db: Optional[float]=-80.0) -> float: ...
@overload
def b_weighting(
    frequencies: npt.ArrayLike,
    *,
    min_db: Optional[float] = -80.0,
) -> NDArray[np.float64]:
    """B-weighting in dB, ``librosa.B_weighting`` (float or array, like the input)."""
    ...

@overload
def c_weighting(frequencies: float, *, min_db: Optional[float]=-80.0) -> float: ...
@overload
def c_weighting(
    frequencies: npt.ArrayLike,
    *,
    min_db: Optional[float] = -80.0,
) -> NDArray[np.float64]:
    """C-weighting in dB, ``librosa.C_weighting`` (float or array, like the input)."""
    ...

@overload
def d_weighting(frequencies: float, *, min_db: Optional[float]=-80.0) -> float: ...
@overload
def d_weighting(
    frequencies: npt.ArrayLike,
    *,
    min_db: Optional[float] = -80.0,
) -> NDArray[np.float64]:
    """D-weighting in dB, ``librosa.D_weighting`` (float or array, like the input)."""
    ...

@overload
def z_weighting(frequencies: float, *, min_db: Optional[float]=-80.0) -> float: ...
@overload
def z_weighting(
    frequencies: npt.ArrayLike,
    *,
    min_db: Optional[float] = -80.0,
) -> NDArray[np.float64]:
    """Z-weighting (flat, 0 dB), ``librosa.Z_weighting`` (float or array, like the input)."""
    ...

@overload
def frequency_weighting(
    frequencies: float,
    *,
    kind: _WeightingKind = 'A',
    min_db: Optional[float] = None,
) -> float: ...
@overload
def frequency_weighting(
    frequencies: npt.ArrayLike,
    *,
    kind: _WeightingKind = 'A',
    min_db: Optional[float] = None,
) -> NDArray[np.float64]:
    """Weighting curve by name, ``librosa.frequency_weighting``.

    Returns:
        float for scalar input, otherwise numpy.ndarray[float64] of the input's shape.

    Raises:
        ValueError: Unknown ``kind``.
    """
    ...

def multi_frequency_weighting(
    frequencies: npt.ArrayLike,
    *,
    kinds: Union[str, Sequence[str]] = "ZAC",
    min_db: Optional[float] = None,
) -> NDArray[np.float64]:
    """Stacked weighting curves, ``librosa.multi_frequency_weighting``.

    Returns:
        numpy.ndarray[float64], shape ``(len(kinds), *frequencies.shape)``.
    """
    ...

@overload
def perceptual_weighting(
    S: NDArray[_UnknownDtype],
    frequencies: npt.ArrayLike,
    *,
    kind: _WeightingKind = 'A',
    ref: _DbRef = 1.0,
    amin: float = 1e-10,
    top_db: Optional[float] = None,
) -> NDArray[Any]: ...
@overload
def perceptual_weighting(  # type: ignore[overload-overlap]
    S: _SingleArray,
    frequencies: npt.ArrayLike,
    *,
    kind: _WeightingKind = 'A',
    ref: _DbRef = 1.0,
    amin: float = 1e-10,
    top_db: Optional[float] = None,
) -> NDArray[np.float32]: ...
@overload
def perceptual_weighting(
    S: npt.ArrayLike,
    frequencies: npt.ArrayLike,
    *,
    kind: _WeightingKind = 'A',
    ref: _DbRef = 1.0,
    amin: float = 1e-10,
    top_db: Optional[float] = None,
) -> NDArray[np.float64]:
    """``power_to_db(S, ref, amin, top_db) + weighting(frequencies)``, ``librosa.perceptual_weighting``.

    Args:
        S: Power spectrogram, ``(F, T)`` or ``(channels, F, T)``.
        frequencies: 1-D bin frequencies (length ``F``).

    Returns:
        numpy.ndarray[float32 | float64] shaped like ``S`` (dB).
    """
    ...

# ---------------------------------------------------------------------------
# Tonal
# ---------------------------------------------------------------------------

def chroma_filterbank(
    *,
    sr: float,
    n_fft: int,
    n_chroma: int = 12,
    tuning: float = 0.0,
    ctroct: float = 5.0,
    octwidth: Optional[float] = 2.0,
    norm: Optional[float] = 2.0,
    base_c: bool = True,
) -> NDArray[np.float64]:
    """Chroma filterbank, ``librosa.filters.chroma(..., dtype=numpy.float64)``.

    Returns:
        numpy.ndarray[float64], shape ``(n_chroma, 1 + n_fft // 2)``.
    """
    ...

@overload
def normalize(
    S: NDArray[_UnknownDtype],
    *,
    norm: Optional[float] = np.inf,
    axis: Optional[int] = 0,
    threshold: Optional[float] = None,
    fill: Optional[bool] = None,
) -> NDArray[Any]: ...
@overload
def normalize(  # type: ignore[overload-overlap]
    S: _SingleArray,
    *,
    norm: Optional[float] = np.inf,
    axis: Optional[int] = 0,
    threshold: Optional[float] = None,
    fill: Optional[bool] = None,
) -> NDArray[np.float32]: ...
@overload
def normalize(
    S: npt.ArrayLike,
    *,
    norm: Optional[float] = np.inf,
    axis: Optional[int] = 0,
    threshold: Optional[float] = None,
    fill: Optional[bool] = None,
) -> NDArray[np.float64]:
    """Unit-norm slices along an axis, ``librosa.util.normalize``.

    Args:
        S: Array of any shape (finite values).
        norm: ``numpy.inf`` (max-abs), ``-numpy.inf``, 0 (non-zero count), ``p > 0``, or None.
        axis: Axis of the norm; None = whole array.
        threshold: Norms below this count as "small" (default: smallest normal float64).
        fill: Small slices: None = unchanged, False = zero, True = unit-norm constant.

    Returns:
        numpy.ndarray[float32 | float64] of the same shape.

    Raises:
        ValueError: Invalid norm / threshold / axis, ``fill=True`` with ``norm=0``,
            or non-finite input.
    """
    ...

def pitch_tuning(
    frequencies: npt.ArrayLike,
    *,
    resolution: float  =  0.01,
    bins_per_octave: int  =  12,
) -> float:
    """Tuning offset of a set of frequencies, ``librosa.pitch_tuning``.

    Returns:
        float in ``[-0.5, 0.5)`` fractions of a bin (0.0 when no positive frequency is given).
    """
    ...

@overload
def chroma_stft_from_spectrogram(
    S: NDArray[_UnknownDtype],
    *,
    sr: float,
    norm: Optional[float] = np.inf,
    n_fft: Optional[int] = None,
    tuning: Optional[float] = 0.0,
    n_chroma: int = 12,
    ctroct: float = 5.0,
    octwidth: Optional[float] = 2.0,
    base_c: bool = True,
) -> NDArray[Any]: ...
@overload
def chroma_stft_from_spectrogram(  # type: ignore[overload-overlap]
    S: _SingleArray,
    *,
    sr: float,
    norm: Optional[float] = np.inf,
    n_fft: Optional[int] = None,
    tuning: Optional[float] = 0.0,
    n_chroma: int = 12,
    ctroct: float = 5.0,
    octwidth: Optional[float] = 2.0,
    base_c: bool = True,
) -> NDArray[np.float32]: ...
@overload
def chroma_stft_from_spectrogram(
    S: npt.ArrayLike,
    *,
    sr: float,
    norm: Optional[float] = np.inf,
    n_fft: Optional[int] = None,
    tuning: Optional[float] = 0.0,
    n_chroma: int = 12,
    ctroct: float = 5.0,
    octwidth: Optional[float] = 2.0,
    base_c: bool = True,
) -> NDArray[np.float64]:
    """Chromagram of a (power) spectrogram, ``librosa.feature.chroma_stft(S=S, ...)``.

    Args:
        S: Power spectrogram ``(1 + n_fft // 2, T)`` or ``(channels, ...)``.
        n_fft: None infers ``2 * (F - 1)``.
        tuning: Tuning offset in fractions of a chroma bin. Default 0.0; None
            estimates it from ``S`` (librosa's default).

    Returns:
        numpy.ndarray[float32 | float64], shape ``(n_chroma, T)`` or ``(channels, n_chroma, T)``.
    """
    ...

@overload
def piptrack_from_spectrogram(
    S: NDArray[_UnknownDtype],
    *,
    sr: float,
    n_fft: Optional[int] = None,
    fmin: float = 150.0,
    fmax: float = 4000.0,
    threshold: float = 0.1,
    ref: Union[None, float, Callable[..., Any], Literal['max', 'mean']] = None,
) -> tuple[NDArray[Any], NDArray[Any]]: ...
@overload
def piptrack_from_spectrogram(  # type: ignore[overload-overlap]
    S: _SingleArray,
    *,
    sr: float,
    n_fft: Optional[int] = None,
    fmin: float = 150.0,
    fmax: float = 4000.0,
    threshold: float = 0.1,
    ref: Union[None, float, Callable[..., Any], Literal['max', 'mean']] = None,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]: ...
@overload
def piptrack_from_spectrogram(
    S: npt.ArrayLike,
    *,
    sr: float,
    n_fft: Optional[int] = None,
    fmin: float = 150.0,
    fmax: float = 4000.0,
    threshold: float = 0.1,
    ref: Union[None, float, Callable[..., Any], Literal['max', 'mean']] = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Spectral peak tracking, ``librosa.piptrack(S=S, ...)``.

    Args:
        ref: None / ``numpy.max`` (per-frame max, default), ``numpy.mean``, or a
            number (absolute threshold).

    Returns:
        ``(pitches, magnitudes)``, both shaped like ``S``, in ``S``'s precision
        (float32 for a float32 ``S``, float64 otherwise).
    """
    ...

def estimate_tuning_from_spectrogram(
    S: npt.ArrayLike,
    *,
    sr: float,
    n_fft: Optional[int] = None,
    resolution: float = 0.01,
    bins_per_octave: int = 12,
    fmin: float = 150.0,
    fmax: float = 4000.0,
    threshold: float = 0.1,
    ref: Union[None, float, Callable[..., Any], Literal["max", "mean"]] = None,
) -> float:
    """Tuning estimate from a spectrogram, ``librosa.estimate_tuning(S=S, ...)``.

    Returns:
        float in ``[-0.5, 0.5)`` fractions of a bin (channels are pooled).
    """
    ...

def estimate_tuning_from_signal(
    y: npt.ArrayLike,
    *,
    sr: float,
    n_fft: int = 2048,
    resolution: float = 0.01,
    bins_per_octave: int = 12,
    hop_length: Optional[int] = None,
    win_length: Optional[int] = None,
    window: _WindowSpec = "hann",
    center: bool = True,
    pad_mode: _PadMode = "constant",
    fmin: float = 150.0,
    fmax: float = 4000.0,
    threshold: float = 0.1,
    ref: Union[None, float, Callable[..., Any], Literal["max", "mean"]] = None,
) -> float:
    """Tuning estimate from a signal ``(n,)`` or ``(channels, n)``, ``librosa.estimate_tuning(y=y, ...)``.

    Returns:
        float in ``[-0.5, 0.5)`` fractions of a bin.
    """
    ...

# ---------------------------------------------------------------------------
# Constant-Q
# ---------------------------------------------------------------------------

def cq_to_chroma(
    n_input: int,
    *,
    bins_per_octave: int = 12,
    n_chroma: int = 12,
    fmin: Optional[float] = None,
    window: Optional[npt.ArrayLike] = None,
    base_c: bool = True,
) -> NDArray[np.float64]:
    """CQT-to-chroma folding matrix, ``librosa.filters.cq_to_chroma(..., dtype=numpy.float64)``.

    ``window`` smooths each row by convolution with these weights.

    Returns:
        numpy.ndarray[float64] of zeros and ones, shape ``(n_chroma, n_input)``.
    """
    ...

def wavelet_lengths(
    *,
    freqs: npt.ArrayLike,
    sr: float,
    window: _WindowSpec = "hann",
    filter_scale: float = 1.0,
    gamma: Optional[float] = 0.0,
    alpha: float | npt.ArrayLike | None = None,
) -> tuple[NDArray[np.float64], float]:
    """Filter lengths of a CQT/VQT filterbank, ``librosa.filters.wavelet_lengths``.

    Returns:
        ``(lengths, cutoff)``: float64 lengths in samples, highest cut-off frequency (Hz).
        librosa's explicit ``alpha`` is not supported (at least two ascending ``freqs``).
    """
    ...

def wavelet(
    *,
    freqs: npt.ArrayLike,
    sr: float,
    window: _WindowSpec = "hann",
    filter_scale: float = 1.0,
    pad_fft: bool = True,
    norm: Optional[float] = 1.0,
    gamma: Optional[float] = 0.0,
    alpha: float | npt.ArrayLike | None = None,
) -> tuple[NDArray[np.complex128], NDArray[np.float64]]:
    """CQT/VQT wavelet filters, ``librosa.filters.wavelet(..., dtype=numpy.complex128)``.

    Returns:
        ``(filters, lengths)``: complex128 ``(len(freqs), width)`` and float64 lengths.
        librosa's explicit ``alpha`` (and ``**kwargs`` passed to the window) are not
        supported; ``alpha`` is derived from ``freqs`` as librosa does by default.
    """
    ...

@overload
def chroma_cens_from_chroma(
    chroma: NDArray[_UnknownDtype],
    *,
    norm: Optional[float] = 2.0,
    win_len_smooth: Optional[int] = 41,
    smoothing_window: _WindowSpec = 'hann',
) -> NDArray[Any]: ...
@overload
def chroma_cens_from_chroma(  # type: ignore[overload-overlap]
    chroma: _SingleArray,
    *,
    norm: Optional[float] = 2.0,
    win_len_smooth: Optional[int] = 41,
    smoothing_window: _WindowSpec = 'hann',
) -> NDArray[np.float32]: ...
@overload
def chroma_cens_from_chroma(
    chroma: npt.ArrayLike,
    *,
    norm: Optional[float] = 2.0,
    win_len_smooth: Optional[int] = 41,
    smoothing_window: _WindowSpec = 'hann',
) -> NDArray[np.float64]:
    """CENS post-processing of a chromagram (the second half of ``librosa.feature.chroma_cens``).

    Args:
        chroma: ``(n_chroma, T)`` or ``(channels, n_chroma, T)``; librosa computes it
            with ``chroma_cqt(norm=None, threshold=0)``.

    Returns:
        numpy.ndarray[float32 | float64] shaped like ``chroma``.
    """
    ...

@overload
def tonnetz_from_chroma(chroma: NDArray[_UnknownDtype]) -> NDArray[Any]: ...
@overload
def tonnetz_from_chroma(  # type: ignore[overload-overlap]
    chroma: _SingleArray,
) -> NDArray[np.float32]: ...
@overload
def tonnetz_from_chroma(chroma: npt.ArrayLike) -> NDArray[np.float64]:
    """Tonal centroid features, ``librosa.feature.tonnetz(chroma=chroma)``.

    Returns:
        numpy.ndarray[float32 | float64], shape ``(6, T)`` or ``(channels, 6, T)``.
    """
    ...

# ---------------------------------------------------------------------------
# Pitch tracking and harmonics
# ---------------------------------------------------------------------------

@overload
def yin(
    y: NDArray[_UnknownDtype],
    *,
    fmin: float,
    fmax: float,
    sr: float,
    frame_length: int = 2048,
    hop_length: Optional[int] = None,
    trough_threshold: float = 0.1,
    center: bool = True,
    pad_mode: _PadMode = 'constant',
) -> NDArray[Any]: ...
@overload
def yin(  # type: ignore[overload-overlap]
    y: _SingleArray,
    *,
    fmin: float,
    fmax: float,
    sr: float,
    frame_length: int = 2048,
    hop_length: Optional[int] = None,
    trough_threshold: float = 0.1,
    center: bool = True,
    pad_mode: _PadMode = 'constant',
) -> NDArray[np.float32]: ...
@overload
def yin(
    y: npt.ArrayLike,
    *,
    fmin: float,
    fmax: float,
    sr: float,
    frame_length: int = 2048,
    hop_length: Optional[int] = None,
    trough_threshold: float = 0.1,
    center: bool = True,
    pad_mode: _PadMode = 'constant',
) -> NDArray[np.float64]:
    """YIN f0 per frame, ``librosa.yin``.

    Returns:
        numpy.ndarray[float32 | float64] in Hz, shape ``(n_frames,)`` or ``(channels, n_frames)``.
    """
    ...

@overload
def pyin(
    y: NDArray[_UnknownDtype],
    *,
    fmin: float,
    fmax: float,
    sr: float,
    frame_length: int = 2048,
    hop_length: Optional[int] = None,
    n_thresholds: int = 100,
    beta_parameters: tuple[float, float] = (2, 18),
    boltzmann_parameter: float = 2,
    resolution: float = 0.1,
    max_transition_rate: float = 35.92,
    switch_prob: float = 0.01,
    no_trough_prob: float = 0.01,
    fill_na: Optional[float] = np.nan,
    center: bool = True,
    pad_mode: _PadMode = 'constant',
    transition_min_prob: Optional[float] = 0.0001,
) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any]]: ...
@overload
def pyin(  # type: ignore[overload-overlap]
    y: _SingleArray,
    *,
    fmin: float,
    fmax: float,
    sr: float,
    frame_length: int = 2048,
    hop_length: Optional[int] = None,
    n_thresholds: int = 100,
    beta_parameters: tuple[float, float] = (2, 18),
    boltzmann_parameter: float = 2,
    resolution: float = 0.1,
    max_transition_rate: float = 35.92,
    switch_prob: float = 0.01,
    no_trough_prob: float = 0.01,
    fill_na: Optional[float] = np.nan,
    center: bool = True,
    pad_mode: _PadMode = 'constant',
    transition_min_prob: Optional[float] = 0.0001,
) -> tuple[NDArray[np.float32], NDArray[np.bool_], NDArray[np.float32]]: ...
@overload
def pyin(
    y: npt.ArrayLike,
    *,
    fmin: float,
    fmax: float,
    sr: float,
    frame_length: int = 2048,
    hop_length: Optional[int] = None,
    n_thresholds: int = 100,
    beta_parameters: tuple[float, float] = (2, 18),
    boltzmann_parameter: float = 2,
    resolution: float = 0.1,
    max_transition_rate: float = 35.92,
    switch_prob: float = 0.01,
    no_trough_prob: float = 0.01,
    fill_na: Optional[float] = np.nan,
    center: bool = True,
    pad_mode: _PadMode = 'constant',
    transition_min_prob: Optional[float] = 0.0001,
) -> tuple[NDArray[np.float64], NDArray[np.bool_], NDArray[np.float64]]:
    """Probabilistic YIN, ``librosa.pyin``.

    Args:
        fill_na: f0 of unvoiced frames: NaN (default), None (best-guess pitch) or a number.

    Returns:
        ``(f0, voiced_flag, voiced_prob)``, each ``(n_frames,)`` or ``(channels, n_frames)``.
    """
    ...

@overload
def salience(
    S: NDArray[_UnknownDtype],
    *,
    freqs: npt.ArrayLike,
    harmonics: Sequence[float],
    weights: Optional[npt.ArrayLike] = None,
    aggregate: Optional[Callable[..., Any]] = None,
    filter_peaks: bool = True,
    fill_value: float = np.nan,
    kind: Literal['linear'] = 'linear',
    axis: int = -2,
) -> NDArray[Any]: ...
@overload
def salience(  # type: ignore[overload-overlap]
    S: _SingleArray,
    *,
    freqs: npt.ArrayLike,
    harmonics: Sequence[float],
    weights: Optional[npt.ArrayLike] = None,
    aggregate: Optional[Callable[..., Any]] = None,
    filter_peaks: bool = True,
    fill_value: float = np.nan,
    kind: Literal['linear'] = 'linear',
    axis: int = -2,
) -> NDArray[np.float32]: ...
@overload
def salience(
    S: npt.ArrayLike,
    *,
    freqs: npt.ArrayLike,
    harmonics: Sequence[float],
    weights: Optional[npt.ArrayLike] = None,
    aggregate: Optional[Callable[..., Any]] = None,
    filter_peaks: bool = True,
    fill_value: float = np.nan,
    kind: Literal['linear'] = 'linear',
    axis: int = -2,
) -> NDArray[np.float64]:
    """Harmonic salience, ``librosa.salience``.

    Args:
        S: Magnitudes ``(F, T)`` or ``(channels, F, T)``.
        freqs: 1-D bin frequencies or an array shaped like ``S``.
        aggregate: None / ``numpy.average`` (weighted, default), ``numpy.mean``,
            ``numpy.sum``, ``numpy.max``, ``numpy.min``, ``numpy.median``.

    Returns:
        numpy.ndarray[float32 | float64] shaped like ``S``.
    """
    ...

@overload
def interp_harmonics(
    x: NDArray[_UnknownDtype],
    *,
    freqs: npt.ArrayLike,
    harmonics: npt.ArrayLike,
    kind: Literal['linear'] = 'linear',
    fill_value: float = 0,
    axis: int = -2,
) -> NDArray[Any]: ...
@overload
def interp_harmonics(  # type: ignore[overload-overlap]
    x: _SingleArray,
    *,
    freqs: npt.ArrayLike,
    harmonics: npt.ArrayLike,
    kind: Literal['linear'] = 'linear',
    fill_value: float = 0,
    axis: int = -2,
) -> NDArray[np.float32]: ...
@overload
def interp_harmonics(
    x: npt.ArrayLike,
    *,
    freqs: npt.ArrayLike,
    harmonics: npt.ArrayLike,
    kind: Literal['linear'] = 'linear',
    fill_value: float = 0,
    axis: int = -2,
) -> NDArray[np.float64]:
    """Energy interpolated at harmonics of every bin, ``librosa.interp_harmonics``.

    Returns:
        numpy.ndarray[float32 | float64], shape ``(len(harmonics), F, T)`` or
        ``(channels, len(harmonics), F, T)``.
    """
    ...

@overload
def f0_harmonics(
    x: NDArray[_UnknownDtype],
    *,
    f0: npt.ArrayLike,
    freqs: npt.ArrayLike,
    harmonics: npt.ArrayLike,
    kind: Literal['linear'] = 'linear',
    fill_value: float = 0,
    axis: int = -2,
) -> NDArray[Any]: ...
@overload
def f0_harmonics(  # type: ignore[overload-overlap]
    x: _SingleArray,
    *,
    f0: npt.ArrayLike,
    freqs: npt.ArrayLike,
    harmonics: npt.ArrayLike,
    kind: Literal['linear'] = 'linear',
    fill_value: float = 0,
    axis: int = -2,
) -> NDArray[np.float32]: ...
@overload
def f0_harmonics(
    x: npt.ArrayLike,
    *,
    f0: npt.ArrayLike,
    freqs: npt.ArrayLike,
    harmonics: npt.ArrayLike,
    kind: Literal['linear'] = 'linear',
    fill_value: float = 0,
    axis: int = -2,
) -> NDArray[np.float64]:
    """Energy at the harmonics of a time-varying f0, ``librosa.f0_harmonics``.

    Args:
        f0: ``(T,)`` or ``(channels, T)``; NaN frames get ``fill_value``.

    Returns:
        numpy.ndarray[float32 | float64], shape ``(len(harmonics), T)`` or ``(channels, len(harmonics), T)``.
    """
    ...

__all__ = [
    "__doc__",
    "a_weighting",
    "b_weighting",
    "c_weighting",
    "chroma_cens_from_chroma",
    "chroma_filterbank",
    "chroma_stft_from_spectrogram",
    "cq_to_chroma",
    "d_weighting",
    "estimate_tuning_from_signal",
    "estimate_tuning_from_spectrogram",
    "f0_harmonics",
    "frequency_weighting",
    "interp_harmonics",
    "multi_frequency_weighting",
    "normalize",
    "pcen",
    "perceptual_weighting",
    "piptrack_from_spectrogram",
    "pitch_tuning",
    "poly_features_from_spectrogram",
    "pyin",
    "rms_from_spectrogram",
    "salience",
    "spectral_bandwidth_from_spectrogram",
    "spectral_centroid_from_spectrogram",
    "spectral_contrast_from_spectrogram",
    "spectral_flatness_from_spectrogram",
    "spectral_rolloff_from_spectrogram",
    "tonnetz_from_chroma",
    "wavelet",
    "wavelet_lengths",
    "yin",
    "z_weighting",
]
