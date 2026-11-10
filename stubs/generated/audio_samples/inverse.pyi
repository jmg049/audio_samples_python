"""
Inversion of mel spectrograms and MFCCs to spectrograms and audio.
"""

from collections.abc import Sequence
from numpy.random import BitGenerator, Generator, RandomState, SeedSequence
from numpy.typing import ArrayLike
from typing import Literal

__all__ = ["mel_to_audio", "mel_to_stft", "mfcc_to_audio", "mfcc_to_mel"]

def mel_to_audio(M: "ArrayLike", *, sr: float, n_fft: int = ..., hop_length: int | None = None, win_length: int | None = None, window: str | tuple[str, float] | float | ArrayLike = ..., center: bool = True, pad_mode: str = "constant", power: float = 2.0, n_iter: int = ..., length: int | None = None, momentum: float = 0.99, init: str | None = ..., rng: int | None | Generator | RandomState | SeedSequence | BitGenerator | Sequence[int] = ..., fmin: float = 0.0, fmax: float | None = None, htk: bool = False, norm: Literal["slaney"] | float | None = ...) -> "NDArray[Any]":
    """
    Estimates audio from a mel spectrogram: least squares, then fast Griffin-Lim.

    Args:

        M (``array_like``): Mel spectrogram ``(n_mels, n_frames)`` or
            ``(channels, n_mels, n_frames)``.
        sr (float): Sample rate in Hz.
        ``n_fft``, ``hop_length``, ``win_length``, window, center, ``pad_mode``: STFT parameters.
        power (float): Exponent that was applied to the magnitude. Default 2.
        ``n_iter`` (int): Griffin-Lim iterations. Default 32.
        length (Optional[int]): Exact output length.
        momentum (float): Fast Griffin-Lim momentum. Default 0.99.
        init (Optional[str]): ``"random"`` (default) or None (zero phase).
        rng: Seed for ``init="random"``: an integer (default 0), None, or a NumPy
            generator, from which one seed is drawn.
        fmin, fmax, htk, norm: The filterbank.

    Returns:

        numpy.ndarray: ``(n,)``, or ``(channels, n)`` for 3-D input, of ``M``'s precision.
    """

def mel_to_stft(M: "ArrayLike", *, sr: float, n_fft: int = ..., power: float = 2.0, fmin: float = 0.0, fmax: float | None = None, htk: bool = False, norm: Literal["slaney"] | float | None = ...) -> "NDArray[Any]":
    """
    Estimates an STFT magnitude from a mel spectrogram by non-negative least squares.

    Args:

        M (``array_like``): Mel spectrogram ``(n_mels, n_frames)`` or
            ``(channels, n_mels, n_frames)``.
        sr (float): Sample rate in Hz.
        ``n_fft`` (int): FFT size of the spectrogram to estimate. Default 2048.
        power (float): Exponent that was applied to the magnitude. Default 2.
        fmin, fmax, htk, norm: The filterbank, as ``spectral.mel_filterbank``.

    Returns:

        numpy.ndarray: ``(1 + n_fft // 2, n_frames)``, with a leading channel axis for 3-D
        input, of ``M``'s precision.
    """

def mfcc_to_audio(mfcc: "ArrayLike", *, sr: float, n_mels: int = ..., dct_type: int = ..., norm: str | None = ..., ref: float = 1.0, lifter: float = 0.0, n_fft: int = ..., hop_length: int | None = None, win_length: int | None = None, window: str | tuple[str, float] | float | ArrayLike = ..., center: bool = True, pad_mode: str = "constant", power: float = 2.0, n_iter: int = ..., length: int | None = None, momentum: float = 0.99, init: str | None = ..., rng: int | None | Generator | RandomState | SeedSequence | BitGenerator | Sequence[int] = ..., fmin: float = 0.0, fmax: float | None = None, htk: bool = False, mel_norm: Literal["slaney"] | float | None = ...) -> "NDArray[Any]":
    """
    Estimates audio from MFCCs: :func:`mfcc_to_mel`, then :func:`mel_to_audio`.

    ``n_mels``, ``dct_type``, ``norm``, ``ref`` and ``lifter`` are as for :func:`mfcc_to_mel`.
    ``n_fft``, ``hop_length``, ``win_length``, ``window``, ``center``, ``pad_mode``, ``power``,
    ``n_iter``, ``length``, ``momentum``, ``init``, ``rng``, ``fmin``, ``fmax``, ``htk`` and
    ``mel_norm`` are as for :func:`mel_to_audio`, where ``mel_norm`` is its ``norm``.

    Args:

        mfcc (``array_like``): ``(n_mfcc, n_frames)`` or ``(channels, n_mfcc, n_frames)``.
        sr (float): Sample rate in Hz.

    Returns:

        numpy.ndarray: ``(n,)``, or ``(channels, n)`` for 3-D input, of ``mfcc``'s precision.
    """

def mfcc_to_mel(mfcc: "ArrayLike", *, n_mels: int = ..., dct_type: int = ..., norm: str | None = ..., ref: float = 1.0, lifter: float = 0.0) -> "NDArray[Any]":
    """
    Recovers a mel power spectrogram from MFCCs.

    Args:

        mfcc (``array_like``): ``(n_mfcc, n_frames)`` or ``(channels, n_mfcc, n_frames)``.
        ``n_mels`` (int): Mel bands to recover. Default 128.
        ``dct_type`` (int): 1, 2 or 3. Default 2.
        norm (Optional[str]): ``"ortho"`` or None. Default ``"ortho"``.
        ref (float): The reference of the decibel scale. Default 1.
        lifter (float): The liftering coefficient to undo. Default 0.

    Returns:

        numpy.ndarray: ``(n_mels, n_frames)``, with a leading channel axis for 3-D input,
        of ``mfcc``'s precision.
    """
