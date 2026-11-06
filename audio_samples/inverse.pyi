"""Inversion of mel spectrograms and MFCCs to spectrograms and audio.

Arrays are ``(bands, frames)`` or ``(channels, bands, frames)``. ``float32`` input stays
in ``float32``; anything else is computed in ``float64``.
"""

from __future__ import annotations

from typing import Sequence

from typing import Any, Literal, Optional, Union, overload

import numpy as np
import numpy.typing as npt
from numpy.typing import NDArray

_RngLike = Union[int, None, np.random.Generator, np.random.RandomState, np.random.SeedSequence, np.random.BitGenerator, Sequence[int]]

_SingleArray = NDArray[np.float32]
_Norm = Union[Literal["slaney"], float, None]


@overload
def mel_to_stft(
    M: _SingleArray, *, sr: float, n_fft: int = 2048, power: float = 2.0, fmin: float = 0.0,
    fmax: Optional[float] = None, htk: bool = False, norm: _Norm = "slaney",
) -> NDArray[np.float32]: ...
@overload
def mel_to_stft(
    M: npt.ArrayLike, *, sr: float, n_fft: int = 2048, power: float = 2.0, fmin: float = 0.0,
    fmax: Optional[float] = None, htk: bool = False, norm: _Norm = "slaney",
) -> NDArray[Any]: ...
def mel_to_audio(
    M: npt.ArrayLike,
    *,
    sr: float,
    n_fft: int = 2048,
    hop_length: Optional[int] = None,
    win_length: Optional[int] = None,
    window: Any = "hann",
    center: bool = True,
    pad_mode: str = "constant",
    power: float = 2.0,
    n_iter: int = 32,
    length: Optional[int] = None,
    momentum: float = 0.99,
    init: Optional[Literal["random"]] = "random",
    rng: _RngLike = 0,
    fmin: float = 0.0,
    fmax: Optional[float] = None,
    htk: bool = False,
    norm: _Norm = "slaney",
) -> NDArray[Any]:
    """Estimates audio from a mel spectrogram: least squares, then fast Griffin-Lim."""
@overload
def mfcc_to_mel(
    mfcc: _SingleArray, *, n_mels: int = 128, dct_type: int = 2, norm: Optional[str] = "ortho",
    ref: float = 1.0, lifter: float = 0.0,
) -> NDArray[np.float32]: ...
@overload
def mfcc_to_mel(
    mfcc: npt.ArrayLike, *, n_mels: int = 128, dct_type: int = 2, norm: Optional[str] = "ortho",
    ref: float = 1.0, lifter: float = 0.0,
) -> NDArray[Any]: ...
def mfcc_to_audio(
    mfcc: npt.ArrayLike,
    *,
    sr: float,
    n_mels: int = 128,
    dct_type: int = 2,
    norm: Optional[str] = "ortho",
    ref: float = 1.0,
    lifter: float = 0.0,
    n_fft: int = 2048,
    hop_length: Optional[int] = None,
    win_length: Optional[int] = None,
    window: Any = "hann",
    center: bool = True,
    pad_mode: str = "constant",
    power: float = 2.0,
    n_iter: int = 32,
    length: Optional[int] = None,
    momentum: float = 0.99,
    init: Optional[Literal["random"]] = "random",
    rng: _RngLike = 0,
    fmin: float = 0.0,
    fmax: Optional[float] = None,
    htk: bool = False,
    mel_norm: _Norm = "slaney",
) -> NDArray[Any]:
    """Estimates audio from MFCCs."""

__all__ = ["__doc__", "mel_to_audio", "mel_to_stft", "mfcc_to_audio", "mfcc_to_mel"]
