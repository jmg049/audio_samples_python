"""Multirate semitone filterbank and the IIR time-frequency representation."""

from __future__ import annotations

from typing import Any, Literal, Optional, overload

import numpy as np
import numpy.typing as npt
from numpy.typing import NDArray


def mr_frequencies(tuning: float) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Centre frequencies and sample rates of the semitones from MIDI 24 to 108."""
def semitone_filterbank(
    *,
    center_freqs: Optional[npt.ArrayLike] = None,
    tuning: float = 0.0,
    sample_rates: Optional[npt.ArrayLike] = None,
    flayout: Literal["sos", "ba"] = "sos",
    **kwargs: Any,
) -> tuple[list[Any], NDArray[np.float64]]:
    """One band-pass filter per semitone, of minimum order for the specified response."""
@overload
def iirt(
    y: NDArray[np.float32], *, sr: int, win_length: int = 2048, hop_length: Optional[int] = None,
    center: bool = True, tuning: float = 0.0, pad_mode: str = "constant", flayout: Literal["sos", "ba"] = "sos",
    res_type: str = "soxr_hq", **kwargs: Any,
) -> NDArray[np.float32]: ...
@overload
def iirt(
    y: npt.ArrayLike, *, sr: int, win_length: int = 2048, hop_length: Optional[int] = None,
    center: bool = True, tuning: float = 0.0, pad_mode: str = "constant", flayout: Literal["sos", "ba"] = "sos",
    res_type: str = "soxr_hq", **kwargs: Any,
) -> NDArray[Any]: ...

__all__ = ["iirt", "mr_frequencies", "semitone_filterbank"]
