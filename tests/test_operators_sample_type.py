"""Every arithmetic operator, reflected and in-place ones included, on every sample type.

The sample type never changes, integer results saturate instead of wrapping, and an in-place
operator gives the same result as its binary form, also when the operand is the audio itself.
"""

from __future__ import annotations

import numpy as np
import pytest

import audio_samples as aus
from audio_samples.types import SampleType

SAMPLE_TYPES = {
    "u8": SampleType.U8,
    "i16": SampleType.I16,
    "I24": SampleType.I24,
    "i32": SampleType.I32,
    "f32": SampleType.F32,
    "f64": SampleType.F64,
}
INTEGER_LIMITS = {
    "u8": (0, 255),
    "i16": (-32768, 32767),
    "I24": (-(2**23), 2**23 - 1),
    "i32": (-(2**31), 2**31 - 1),
}
TYPES = list(SAMPLE_TYPES)


def make(sample_type: str, amplitude: float = 0.5) -> aus.AudioSamples:
    base = aus.generation.sine_wave(100.0, 0.05, 8000, amplitude, dtype=np.float64)
    return base.to_format(SAMPLE_TYPES[sample_type])


def binary_operators(x, n):
    return {
        "x + 0.25": lambda: x + 0.25,
        "0.25 + x": lambda: 0.25 + x,
        "x - 0.25": lambda: x - 0.25,
        "0.25 - x": lambda: 0.25 - x,
        "x * 0.5": lambda: x * 0.5,
        "0.5 * x": lambda: 0.5 * x,
        "x / 2.0": lambda: x / 2.0,
        "x ** 2.0": lambda: x**2.0,
        "x + x": lambda: x + x,
        "x - x": lambda: x - x,
        "x * x": lambda: x * x,
        "x / x": lambda: x / x,
        "x + n": lambda: x + n,
        "x - n": lambda: x - n,
        "x * n": lambda: x * n,
        "x / n": lambda: x / n,
        "x + float array": lambda: x + np.full(n.shape, 1.0),
        "x * float array": lambda: x * np.full(n.shape, 0.5),
        "x / float array": lambda: x / np.full(n.shape, 2.0),
    }


@pytest.mark.parametrize("sample_type", TYPES)
@pytest.mark.parametrize(
    "operator",
    list(binary_operators(None, np.zeros(1))),
)
def test_binary_operator_keeps_the_sample_type(sample_type, operator):
    x = make(sample_type)
    result = binary_operators(x, x.to_numpy())[operator]()
    assert result.dtype == sample_type
    assert result.sample_rate == x.sample_rate
    assert result.samples_per_channel() == x.samples_per_channel()


IN_PLACE = {
    "+= scalar": lambda y, x, n: y.__iadd__(0.25),
    "-= scalar": lambda y, x, n: y.__isub__(0.25),
    "*= scalar": lambda y, x, n: y.__imul__(0.5),
    "/= scalar": lambda y, x, n: y.__itruediv__(2.0),
    "+= audio": lambda y, x, n: y.__iadd__(x),
    "-= audio": lambda y, x, n: y.__isub__(x),
    "*= audio": lambda y, x, n: y.__imul__(x),
    "/= audio": lambda y, x, n: y.__itruediv__(x),
    "+= array": lambda y, x, n: y.__iadd__(n),
    "-= array": lambda y, x, n: y.__isub__(n),
    "*= array": lambda y, x, n: y.__imul__(n),
    "/= array": lambda y, x, n: y.__itruediv__(n),
    "+= itself": lambda y, x, n: y.__iadd__(y),
    "-= itself": lambda y, x, n: y.__isub__(y),
    "*= itself": lambda y, x, n: y.__imul__(y),
    "/= itself": lambda y, x, n: y.__itruediv__(y),
}
BINARY_OF_IN_PLACE = {
    "+= scalar": lambda x, n: x + 0.25,
    "-= scalar": lambda x, n: x - 0.25,
    "*= scalar": lambda x, n: x * 0.5,
    "/= scalar": lambda x, n: x / 2.0,
    "+= audio": lambda x, n: x + x,
    "-= audio": lambda x, n: x - x,
    "*= audio": lambda x, n: x * x,
    "/= audio": lambda x, n: x / x,
    "+= array": lambda x, n: x + n,
    "-= array": lambda x, n: x - n,
    "*= array": lambda x, n: x * n,
    "/= array": lambda x, n: x / n,
    "+= itself": lambda x, n: x + x,
    "-= itself": lambda x, n: x - x,
    "*= itself": lambda x, n: x * x,
    "/= itself": lambda x, n: x / x,
}


@pytest.mark.parametrize("sample_type", TYPES)
@pytest.mark.parametrize("operator", list(IN_PLACE))
def test_in_place_operator_matches_the_binary_one(sample_type, operator):
    x = make(sample_type)
    n = x.to_numpy()
    y = x.copy()
    returned = IN_PLACE[operator](y, x, n)
    assert returned is None or returned is y
    assert y.dtype == sample_type
    expected = BINARY_OF_IN_PLACE[operator](x, n)
    np.testing.assert_array_equal(y.to_numpy(), expected.to_numpy())


@pytest.mark.parametrize("sample_type", TYPES)
def test_in_place_statements(sample_type):
    y = make(sample_type)
    identity = y
    y += y
    y -= 0.0
    y *= 1.0
    y /= 1.0
    assert y is identity
    assert y.dtype == sample_type
    y *= y
    y /= y
    assert y is identity and y.dtype == sample_type


@pytest.mark.parametrize("sample_type", TYPES)
def test_reflected_operators_commute(sample_type):
    x = make(sample_type)
    np.testing.assert_array_equal((0.25 + x).to_numpy(), (x + 0.25).to_numpy())
    np.testing.assert_array_equal((0.5 * x).to_numpy(), (x * 0.5).to_numpy())


@pytest.mark.parametrize("sample_type", TYPES)
def test_adding_silence_changes_nothing(sample_type):
    x = make(sample_type)
    np.testing.assert_array_equal((x + 0.0).to_numpy(), x.to_numpy())
    np.testing.assert_array_equal((0.0 + x).to_numpy(), x.to_numpy())
    np.testing.assert_array_equal((x - 0.0).to_numpy(), x.to_numpy())


@pytest.mark.parametrize("sample_type", TYPES)
def test_subtracting_audio_from_itself_gives_silence(sample_type):
    x = make(sample_type)
    silence = make(sample_type, 0.0).to_numpy()
    np.testing.assert_array_equal((x - x).to_numpy(), silence)


@pytest.mark.parametrize("sample_type", ["u8", "i16", "I24", "i32"])
def test_integer_results_saturate(sample_type):
    low, high = INTEGER_LIMITS[sample_type]
    x = make(sample_type, 0.9)
    samples = x.to_numpy()
    for result in (x + 1.0, 1.0 + x, x + x, x * x):
        values = result.to_numpy()
        assert values.max() == high, result
        assert values.min() >= low
    # Going over the top must not wrap round to the bottom.
    loud = (x + 1.0).to_numpy()
    assert np.all(loud[samples > samples.mean()] == high)
    assert (x - 1.0).to_numpy().min() == low
    assert (x + x).to_numpy()[np.argmax(samples)] == high


@pytest.mark.parametrize("sample_type", ["i16", "I24", "i32"])
def test_scalar_add_is_in_floating_point_units(sample_type):
    x = make(sample_type, 0.25)
    expected = (make("f64", 0.25) + 0.25).to_format(SAMPLE_TYPES[sample_type]).to_numpy()
    result = (x + 0.25).to_numpy()
    assert np.max(np.abs(result.astype(np.float64) - expected.astype(np.float64))) <= 2


@pytest.mark.parametrize("sample_type", ["u8", "i16", "I24", "i32"])
def test_integer_division_by_zero_saturates(sample_type):
    low, high = INTEGER_LIMITS[sample_type]
    x = make(sample_type)
    result = (x / (x - x)).to_numpy()
    assert result.min() >= low and result.max() <= high
    assert (x / x).dtype == sample_type


@pytest.mark.parametrize("sample_type", TYPES)
def test_mixed_sample_types_convert_to_the_left_operand(sample_type):
    x = make(sample_type)
    other = make("f64" if sample_type != "f64" else "i16")
    for result in (x + other, x - other, x * other, x / other):
        assert result.dtype == sample_type
    y = x.copy()
    y += other
    assert y.dtype == sample_type
    expected = other.to_format(SAMPLE_TYPES[sample_type])
    np.testing.assert_array_equal((x + other).to_numpy(), (x + expected).to_numpy())


@pytest.mark.parametrize("sample_type", TYPES)
def test_multi_channel_keeps_type_and_shape(sample_type):
    mono = make(sample_type)
    stereo = aus.AudioSamples.stack([mono, mono])
    for result in (0.25 + stereo, stereo - stereo, stereo * stereo, stereo / stereo, 0.25 - stereo):
        assert result.dtype == sample_type
        assert result.shape == stereo.shape
    y = stereo.copy()
    y *= y
    assert y.dtype == sample_type and y.shape == stereo.shape


@pytest.mark.parametrize("sample_type", TYPES)
def test_scalar_division_and_power_of_a_scalar_raise(sample_type):
    x = make(sample_type)
    with pytest.raises(TypeError):
        1.0 / x
    with pytest.raises(TypeError):
        2.0**x
    with pytest.raises(ZeroDivisionError):
        x / 0.0
    with pytest.raises(ZeroDivisionError):
        x.copy().__itruediv__(0.0)
    with pytest.raises(TypeError):
        x + "text"


@pytest.mark.parametrize("operator", ["__add__", "__sub__", "__mul__", "__truediv__", "__iadd__", "__imul__"])
def test_incompatible_operands_raise_rather_than_panic(operator):
    x = make("f32")
    with pytest.raises(ValueError):
        getattr(x.copy(), operator)(aus.generation.sine_wave(100.0, 0.1, 8000))
    with pytest.raises(ValueError):
        getattr(x.copy(), operator)(aus.generation.sine_wave(100.0, 0.05, 16000))
