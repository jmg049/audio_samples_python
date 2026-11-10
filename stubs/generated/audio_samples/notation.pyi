"""
Tuning systems and musical notation: interval generators, Functional Just System, svara, mela and thaat names.
"""

from _typeshed import Incomplete
from typing import Any

def fifths_to_note(*, unison: str, fifths: int, unicode: bool = True) -> str:
    """
    The note a number of perfect fifths from a unison.

    Args:

        unison (str): The starting note.
        fifths (int): Signed number of fifths.
        unicode (bool): Unicode accidentals; otherwise ASCII. Default true.

    Returns:

        str: The note's name, without an octave.
    """

def hz_to_fjs(frequencies: Any, *, fmin: float | None = None, unison: str | None = None, unicode: bool = False) -> Any:
    """
    The Functional Just System spelling of frequencies relative to a reference.

    Args:

        frequencies (float | ``array_like``): In Hz.
        fmin (Optional[float]): The frequency of ratio 1. Default: the smallest input.
        unison (Optional[str]): The note of ratio 1. Default: nearest note to ``fmin``.
        unicode (bool): As for :func:`interval_to_fjs`. Default false.

    Returns:

        str | numpy.ndarray[str]: The spelling, the shape of the input.
    """

def hz_to_note(hz: "float | ArrayLike", *, octave: bool = True, cents: bool = False, key: str = "C:maj", unicode: bool = False) -> Any:
    """
    The name of a frequency.

    Args:

        hz (float | ``array_like``): Frequencies in Hz.
        octave, cents, key, unicode: As ``midi_to_note``.

    Returns:

        str | numpy.ndarray: The names.
    """

def hz_to_svara_c(frequencies: Any, *, Sa: float, mela: "int | str", abbr: bool = True, octave: bool = True, unicode: bool = True) -> Any:
    """
    The Carnatic svara of frequencies within a melakarta raga.

    Args:

        frequencies (float | ``array_like``): In Hz.
        Sa (float): The frequency of Sa, in Hz.
        mela (int | str): The raga's number or name.
        abbr, octave, unicode: As for :func:`midi_to_svara_h`.

    Returns:

        str | numpy.ndarray[str]: The svara, the shape of the input.
    """

def hz_to_svara_h(frequencies: Any, *, Sa: float, abbr: bool = True, octave: bool = True, unicode: bool = True) -> Any:
    """
    The Hindustani svara of frequencies.

    Args:

        frequencies (float | ``array_like``): In Hz.
        Sa (float): The frequency of Sa, in Hz.
        abbr, octave, unicode: As for :func:`midi_to_svara_h`.

    Returns:

        str | numpy.ndarray[str]: The svara, the shape of the input.
    """

def interval_frequencies(n_bins: int, *, fmin: float, intervals: "str | ArrayLike", bins_per_octave: int = ..., tuning: float = 0.0, sort: bool = True) -> "NDArray[np.float64]":
    """
    Frequencies of a tuning system over several octaves.

    Args:

        ``n_bins`` (int): Number of frequencies.
        fmin (float): The lowest frequency, in Hz.
        intervals (str | ``array_like``): ``"equal"``, ``"pythagorean"``, ``"ji3"``, ``"ji5"``,
            ``"ji7"``, or the ratios of one octave.
        ``bins_per_octave`` (int): Intervals per octave. Default 12.
        tuning (float): Offset of equal temperament in fractions of a bin.
        sort (bool): In increasing order. Default true.

    Returns:

        numpy.ndarray[float64]: The frequencies.
    """

def interval_to_fjs(interval: Any, *, unison: str = "C", tolerance: float = ..., unicode: bool = True) -> Any:
    """
    The Functional Just System spelling of an interval.

    Ratios built from the primes up to 23 are recognised, to six decimal places.

    Args:

        interval (float | ``array_like``): The ratio or ratios, positive.
        unison (str): The note of ratio 1. Default ``"C"``.
        tolerance (float): How close in ratio a stack of fifths must come. Default 65/63.
        unicode (bool): Unicode accidentals and scripts; otherwise ``^n`` and ``_n``.

    Returns:

        str | numpy.ndarray[str]: The spelling, the shape of the input.
    """

def list_thaat() -> list:
    """
    The ten thaats, by name.

    Returns:

        list[str]: The names.
    """

def mela_to_degrees(mela: "int | str") -> "NDArray[np.int64]":
    """
    The scale degrees of a melakarta raga, in semitones above Sa.

    Args:

        mela (int | str): The raga's number, from 1 to 72, or its name.

    Returns:

        numpy.ndarray[int64]: Seven degrees.
    """

def mela_to_svara(mela: "int | str", *, abbr: bool = True, unicode: bool = True) -> list:
    """
    The names of the twelve svaras of a melakarta raga, from Sa.

    Args:

        mela (int | str): The raga's number or name.
        abbr (bool): Short forms such as ``R1``. Default true.
        unicode (bool): Subscript digits; otherwise plain digits. Default true.

    Returns:

        list[str]: Twelve names.
    """

def midi_to_note(midi: "float | ArrayLike", *, octave: bool = True, cents: bool = False, key: str = "C:maj", unicode: bool = False) -> Any:
    """
    The name of a MIDI number.

    Args:

        midi (float | ``array_like``): MIDI numbers. A value that is not finite gives ``""``.
        octave (bool): Include the octave. Default True.
        cents (bool): Include the deviation in cents, such as ``"A4+25"``. Needs ``octave``.
        key (str): The key, such as ``"F:maj"``, that decides the spelling. Default ``"C:maj"``.
        unicode (bool): Unicode accidentals; otherwise ASCII ``#`` and ``b``. Default False.

    Returns:

        str | numpy.ndarray: The names.
    """

def midi_to_svara_c(midi: Any, *, Sa: float, mela: "int | str", abbr: bool = True, octave: bool = True, unicode: bool = True) -> Any:
    """
    The Carnatic svara of MIDI numbers within a melakarta raga.

    Args:

        midi (float | ``array_like``): MIDI numbers; a non-finite value gives ``""``.
        Sa (float): The MIDI number of Sa.
        mela (int | str): The raga's number or name.
        abbr, octave, unicode: As for :func:`midi_to_svara_h`.

    Returns:

        str | numpy.ndarray[str]: The svara, the shape of the input.
    """

def midi_to_svara_h(midi: Any, *, Sa: float, abbr: bool = True, octave: bool = True, unicode: bool = True) -> Any:
    """
    The Hindustani svara of MIDI numbers.

    Args:

        midi (float | ``array_like``): MIDI numbers; a non-finite value gives ``""``.
        Sa (float): The MIDI number of Sa.
        abbr (bool): Short forms such as ``S``. Default true.
        octave (bool): Mark the octaves next to Sa's. Default true.
        unicode (bool): Dots; otherwise ``'`` and ``,``. Default true.

    Returns:

        str | numpy.ndarray[str]: The svara, the shape of the input.
    """

def note_to_hz(note: "str | Iterable[str]", *, round_midi: bool = False) -> Any:
    """
    The frequency of a note name.

    Args:

        note (str | Iterable[str]): Names such as ``"C#4"`` or ``"A4+25"``.
        ``round_midi`` (bool): Round to a whole note first. Default False.

    Returns:

        float | numpy.ndarray: The frequencies in Hz.
    """

def note_to_midi(note: "str | Iterable[str]", *, round_midi: bool = True) -> Any:
    """
    The MIDI number of a note name.

    Args:

        note (str | Iterable[str]): Names such as ``"C#4"`` or ``"A4+25"``. An empty name
            gives NaN.
        ``round_midi`` (bool): Round to a whole number. Default True.

    Returns:

        float | numpy.ndarray: The numbers.
    """

def note_to_svara_c(notes: "str | Sequence[str]", *, Sa: str, mela: "int | str", abbr: bool = True, octave: bool = True, unicode: bool = True) -> Any:
    """
    The Carnatic svara of note names within a melakarta raga.

    Args:

        notes (str | sequence of str): The notes.
        Sa (str): The note of Sa.
        mela (int | str): The raga's number or name.
        abbr, octave, unicode: As for :func:`midi_to_svara_h`.

    Returns:

        str | numpy.ndarray[str]: The svara.
    """

def note_to_svara_h(notes: "str | Sequence[str]", *, Sa: str, abbr: bool = True, octave: bool = True, unicode: bool = True) -> Any:
    """
    The Hindustani svara of note names.

    Args:

        notes (str | sequence of str): The notes.
        Sa (str): The note of Sa.
        abbr, octave, unicode: As for :func:`midi_to_svara_h`.

    Returns:

        str | numpy.ndarray[str]: The svara.
    """

def plimit_intervals(*, primes: "ArrayLike", bins_per_octave: int = ..., sort: bool = True, return_factors: bool = False) -> Any:
    """
    One octave of a just-intonation system over the given primes.

    Args:

        primes (``array_like[int]``): The primes, such as ``[3, 5]``.
        ``bins_per_octave`` (int): Number of intervals. Default 12.
        sort (bool): In increasing order; otherwise in the order chosen. Default true.
        ``return_factors`` (bool): Return prime factorisations, as dicts, instead of ratios.

    Returns:

        numpy.ndarray[float64] | list[dict[int, int]]: Ratios in ``[1, 2)``, or factors.
    """

def pythagorean_intervals(*, bins_per_octave: int = ..., sort: bool = True, return_factors: bool = False) -> Any:
    """
    One octave of the Pythagorean tuning.

    Args:

        ``bins_per_octave`` (int): Number of intervals. Default 12.
        sort (bool): In increasing order; otherwise by power of 3. Default true.
        ``return_factors`` (bool): Return prime factorisations, as dicts, instead of ratios.

    Returns:

        numpy.ndarray[float64] | list[dict[int, int]]: Ratios in ``[1, 2)``, or factors.
    """

def thaat_to_degrees(thaat: str) -> "NDArray[np.int64]":
    """
    The scale degrees of a thaat, in semitones above Sa.

    Args:

        thaat (str): Its name, in any letter case.

    Returns:

        numpy.ndarray[int64]: Seven degrees.
    """

def __getattr__(name: str) -> Incomplete: ...
