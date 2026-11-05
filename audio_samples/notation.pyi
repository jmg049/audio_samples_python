"""Tuning systems and musical notation: interval generators, Functional Just System, svara, mela and thaat names."""

from __future__ import annotations

from typing import Any, Sequence, Union, overload, Literal, Optional

import numpy as np
import numpy.typing as npt
from numpy.typing import NDArray

_Mela = Union[int, str]


def pythagorean_intervals(*, bins_per_octave: int = 12, sort: bool = True, return_factors: bool = False) -> Any:
    """One octave of the Pythagorean tuning."""
def plimit_intervals(*, primes: npt.ArrayLike, bins_per_octave: int = 12, sort: bool = True, return_factors: bool = False) -> Any:
    """One octave of a just-intonation system over the given primes."""
def interval_frequencies(
    n_bins: int, *, fmin: float, intervals: Union[str, npt.ArrayLike], bins_per_octave: int = 12, tuning: float = 0.0, sort: bool = True
) -> NDArray[np.float64]:
    """Frequencies of a tuning system over several octaves."""
def fifths_to_note(*, unison: str, fifths: int, unicode: bool = True) -> str:
    """The note a number of perfect fifths from a unison."""
def interval_to_fjs(interval: Any, *, unison: str = "C", tolerance: float = 1.0317460317460319, unicode: bool = True) -> Any:
    """The Functional Just System spelling of an interval."""
def hz_to_fjs(frequencies: Any, *, fmin: Optional[float] = None, unison: Optional[str] = None, unicode: bool = False) -> Any:
    """The Functional Just System spelling of frequencies relative to a reference."""
def list_thaat() -> list[str]:
    """The ten thaats, by name."""
def thaat_to_degrees(thaat: str) -> NDArray[np.int64]:
    """The scale degrees of a thaat."""
def list_mela() -> dict[str, int]:
    """The 72 melakarta ragas and their numbers."""
def mela_to_degrees(mela: _Mela) -> NDArray[np.int64]:
    """The scale degrees of a melakarta raga."""
def mela_to_svara(mela: _Mela, *, abbr: bool = True, unicode: bool = True) -> list[str]:
    """The names of the twelve svaras of a melakarta raga."""
def midi_to_svara_h(midi: Any, *, Sa: float, abbr: bool = True, octave: bool = True, unicode: bool = True) -> Any:
    """The Hindustani svara of MIDI numbers."""
def midi_to_svara_c(midi: Any, *, Sa: float, mela: _Mela, abbr: bool = True, octave: bool = True, unicode: bool = True) -> Any:
    """The Carnatic svara of MIDI numbers."""
def hz_to_svara_h(frequencies: Any, *, Sa: float, abbr: bool = True, octave: bool = True, unicode: bool = True) -> Any:
    """The Hindustani svara of frequencies."""
def hz_to_svara_c(frequencies: Any, *, Sa: float, mela: _Mela, abbr: bool = True, octave: bool = True, unicode: bool = True) -> Any:
    """The Carnatic svara of frequencies."""
def note_to_svara_h(notes: Union[str, Sequence[str]], *, Sa: str, abbr: bool = True, octave: bool = True, unicode: bool = True) -> Any:
    """The Hindustani svara of note names."""
def note_to_svara_c(notes: Union[str, Sequence[str]], *, Sa: str, mela: _Mela, abbr: bool = True, octave: bool = True, unicode: bool = True) -> Any:
    """The Carnatic svara of note names."""

__all__ = [
    "__doc__",
    "fifths_to_note",
    "hz_to_fjs",
    "hz_to_svara_c",
    "hz_to_svara_h",
    "interval_frequencies",
    "interval_to_fjs",
    "list_mela",
    "list_thaat",
    "mela_to_degrees",
    "mela_to_svara",
    "midi_to_svara_c",
    "midi_to_svara_h",
    "note_to_svara_c",
    "note_to_svara_h",
    "plimit_intervals",
    "pythagorean_intervals",
    "thaat_to_degrees",
]
