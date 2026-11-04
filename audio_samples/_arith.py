"""Element-wise arithmetic on the samples of integer audio, in the audio's own sample type.

NumPy on integer arrays wraps on overflow and promotes when dividing. Here the
result stays in the sample type and saturates at its limits. ``u8`` samples are
offset binary, where 128 is silence, so their offset is removed for the sum and
put back afterwards.
"""

from __future__ import annotations

import numpy as np

_OPERATIONS = {"add": np.add, "sub": np.subtract, "mul": np.multiply, "div": np.divide}
_LIMITS = {
    "u8": (0.0, 255.0),
    "i16": (-32768.0, 32767.0),
    "I24": (-8388608.0, 8388607.0),
    "i32": (-2147483648.0, 2147483647.0),
}


def scalar_to_samples(value: float, sample_type: str, dtype: np.dtype) -> np.ndarray:
    """A value in the floating-point range (1.0 is full scale) in integer sample units."""
    low, high = _LIMITS[sample_type]
    if sample_type == "u8":
        offset = 128.0
        scaled = offset + value * (128.0 if value < 0 else 127.0)
    else:
        scaled = value * (-low if value < 0 else high)
    return np.asarray(np.clip(np.rint(scaled), low, high), dtype=dtype)


def with_raw_integer(samples: np.ndarray, value: int, operation: str, sample_type: str) -> np.ndarray:
    """``samples + value``, ``samples - value`` or, for ``rsub``, ``value - samples``.

    ``value`` is in the stored units of the sample type, ``u8`` offset included. An int too large
    for the sample type saturates the result, so it is first limited to what can reach the limits.
    """
    low, high = (int(limit) for limit in _LIMITS[sample_type])
    reach = 2 * (high - low)
    value = max(-reach, min(reach, int(value)))
    wide = samples.astype(np.int64)
    if operation == "add":
        result = wide + value
    elif operation == "sub":
        result = wide - value
    else:
        result = value - wide
    return np.clip(result, low, high).astype(samples.dtype)


def apply(samples: np.ndarray, other, operation: str, sample_type: str) -> np.ndarray:
    """``samples <operation> other``, with the dtype of ``samples``."""
    function = _OPERATIONS[operation]
    if sample_type in ("f32", "f64"):
        with np.errstate(divide="ignore", invalid="ignore"):
            return function(samples, other).astype(samples.dtype, copy=False)

    low, high = _LIMITS[sample_type]
    other = np.asarray(other)
    left = samples.astype(np.float64)
    right = other.astype(np.float64)
    offset = 0.0
    if sample_type == "u8":
        offset = 128.0
        left -= offset
        if other.dtype == np.uint8:
            right -= offset
    with np.errstate(divide="ignore", invalid="ignore"):
        result = function(left, right)
    result = np.nan_to_num(result, nan=0.0, posinf=high - offset, neginf=low - offset) + offset
    return np.clip(np.rint(result), low, high).astype(samples.dtype)
