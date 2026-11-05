"""Element-wise arithmetic on the samples of integer audio, in the audio's own sample type.

NumPy on integer arrays wraps on overflow and promotes when dividing. Here the
result stays in the sample type and saturates at its limits. ``u8`` samples are
offset binary, where 128 is silence, so the work is done on the value relative to
silence and the offset put back afterwards.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

_OPERATIONS: dict[str, Callable[..., np.ndarray]] = {
    "add": np.add,
    "sub": np.subtract,
    "mul": np.multiply,
    "div": np.divide,
    "rsub": lambda left, right: np.subtract(right, left),
    "rdiv": lambda left, right: np.divide(right, left),
}
_LIMITS = {
    "u8": (0, 255),
    "i16": (-32768, 32767),
    "I24": (-8388608, 8388607),
    "i32": (-2147483648, 2147483647),
}
_NAMES = {"u8": "uint8", "i16": "int16", "I24": "int24", "i32": "int32"}


def _offset(sample_type: str) -> int:
    return 128 if sample_type == "u8" else 0


def _finish(result: np.ndarray, sample_type: str, dtype: np.dtype) -> np.ndarray:
    low, high = _LIMITS[sample_type]
    return np.clip(np.rint(result + _offset(sample_type)), low, high).astype(dtype)


def scalar_to_samples(value: float, sample_type: str, dtype: np.dtype) -> np.ndarray:
    """A value in the floating-point range (1.0 is full scale) in integer sample units."""
    low, high = _LIMITS[sample_type]
    offset = _offset(sample_type)
    scaled = offset + value * (offset - low if value < 0 else high - offset)
    return np.asarray(np.clip(np.rint(scaled), low, high), dtype=dtype)


def _describe(value: int) -> str:
    # str() refuses very long ints.
    return str(value) if value.bit_length() <= 256 else f"an integer of {value.bit_length()} bits"


def with_integer(samples: np.ndarray, value, operation: str, sample_type: str) -> np.ndarray:
    """``samples + value``, ``samples - value`` or, for ``rsub``, ``value - samples``.

    ``value`` is a number of sample steps relative to silence. A NumPy integer scalar with the
    dtype of ``u8`` samples is a sample itself, so its offset comes off first.
    """
    low, high = _LIMITS[sample_type]
    offset = _offset(sample_type)
    number = int(value)
    if offset and getattr(value, "dtype", None) == np.uint8:
        number -= offset
    if not low - offset <= number <= high - offset:
        raise OverflowError(
            f"{_describe(number)} does not fit {_NAMES[sample_type]} audio: an int scalar is a number "
            f"of sample steps from {low - offset} to {high - offset}"
        )
    centred = samples.astype(np.int64) - offset
    if operation == "add":
        result = centred + number
    elif operation == "sub":
        result = centred - number
    else:
        result = number - centred
    return np.clip(result + offset, low, high).astype(samples.dtype)


def apply(samples: np.ndarray, other, operation: str, sample_type: str) -> np.ndarray:
    """``samples <operation> other``, with the dtype of ``samples``.

    On integer audio a non-finite value in ``other`` raises ``ValueError``. A divisor sample of
    silence gives the limit with the sign of the dividend, and silence when the dividend is silence.
    """
    function = _OPERATIONS[operation]
    if sample_type in ("f32", "f64"):
        with np.errstate(divide="ignore", invalid="ignore"):
            return function(samples, other).astype(samples.dtype, copy=False)

    other = np.asarray(other)
    if other.dtype.kind not in "biuf":
        raise TypeError(f"integer audio cannot be combined with an array of dtype {other.dtype}")
    if other.dtype.kind == "f" and not np.isfinite(other).all():
        raise ValueError("the array has a value that is not finite, which integer audio cannot use")
    offset = _offset(sample_type)
    left = samples.astype(np.float64) - offset
    right = other.astype(np.float64)
    if offset and other.dtype == np.uint8:
        right = right - offset
    with np.errstate(divide="ignore", invalid="ignore"):
        result = function(left, right)
    if operation in ("div", "rdiv"):
        dividend, divisor = (left, right) if operation == "div" else (right, left)
        low, high = _LIMITS[sample_type]
        result = np.where(divisor == 0, np.sign(dividend) * (high - low + 1), result)
    return _finish(result, sample_type, samples.dtype)


def negative(samples: np.ndarray, sample_type: str) -> np.ndarray:
    """``-samples``, saturating, and mirrored about silence for ``u8``."""
    if sample_type in ("f32", "f64"):
        return np.negative(samples)
    return _finish(-(samples.astype(np.float64) - _offset(sample_type)), sample_type, samples.dtype)


def absolute(samples: np.ndarray, sample_type: str) -> np.ndarray:
    """``abs(samples)``, saturating, and about silence for ``u8``."""
    if sample_type in ("f32", "f64"):
        return np.abs(samples)
    return _finish(np.abs(samples.astype(np.float64) - _offset(sample_type)), sample_type, samples.dtype)
