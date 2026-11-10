"""
Multirate semitone filterbank and the IIR time-frequency representation.
"""

from typing import Any

__all__ = ["iirt", "mr_frequencies", "semitone_filterbank"]

def iirt(y: "ArrayLike", *, sr: int, win_length: int = ..., hop_length: int | None = None, center: bool = True, tuning: float = 0.0, pad_mode: str = "constant", flayout: str = "sos", res_type: str = "soxr_hq", **kwargs) -> "NDArray[Any]":
    """
    The energy of each semitone band per frame.

    Args:

        y (``array_like``): Signal ``(n,)`` or ``(channels, n)``, finite.
        sr (int): Sample rate in Hz.
        ``win_length`` (int): Frame length in samples. Default 2048.
        ``hop_length`` (Optional[int]): Samples between frames. Default ``win_length // 4``.
        center (bool): Pad so that frames are centred. Default true.
        tuning (float): Deviation from A440 in fractions of a semitone.
        ``pad_mode`` (str): ``"constant"``, ``"edge"``, ``"reflect"``, ``"symmetric"``, ``"wrap"``.
        flayout (str): ``"sos"`` (default) or ``"ba"``.
        ``res_type`` (str): Resampler quality, as for ``AudioSamples.resample``.
        **kwargs: The filterbank's parameters; see :func:`semitone_filterbank`.

    Returns:

        numpy.ndarray: ``(85, n_frames)``, with a leading channel axis for 2-D input, of
        ``y``'s precision.
    """

def mr_frequencies(tuning: float) -> "tuple[NDArray[np.float64], NDArray[np.float64]]":
    """
    Centre frequencies and sample rates of the semitones from MIDI 24 to 108.

    Args:

        tuning (float): Deviation from A440 in fractions of a semitone.

    Returns:

        tuple[numpy.ndarray[float64], numpy.ndarray[float64]]: Centre frequencies and sample
        rates in Hz, 85 of each.
    """

def semitone_filterbank(*, center_freqs: Any | None = None, tuning: float = 0.0, sample_rates: Any | None = None, flayout: str = "sos", **kwargs) -> "tuple[list[Any], NDArray[np.float64]]":
    """
    One band-pass filter per semitone, of minimum order for the specified response.

    Args:

        ``center_freqs`` (Optional[array_like]): Centre frequencies in Hz; give together with
            ``sample_rates``. Default: :func:`mr_frequencies`.
        tuning (float): Used when ``center_freqs`` is not given.
        ``sample_rates`` (Optional[array_like]): The sample rate of each band in Hz.
        flayout (str): ``"sos"`` (default) or ``"ba"``. Polynomials lose precision at the
            orders these narrow filters reach.
        **kwargs: ``Q`` (25), ``passband_ripple`` (1 dB), ``stopband_attenuation`` (50 dB),
            ``ftype`` (``"ellip"``, ``"butter"``, ``"cheby1"``, ``"cheby2"``).

    Returns:

        tuple[list, numpy.ndarray[float64]]: One filter per band: an ``(n, 6)`` array of
        sections, or a ``(b, a)`` tuple; and the sample rates.
    """
