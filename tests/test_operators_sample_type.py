"""Every arithmetic operator, reflected and in-place ones included, on every sample type.

The sample type never changes, integer results saturate instead of wrapping, and an in-place
operator gives the same result as its binary form, also when the operand is the audio itself.
"""

from __future__ import annotations

import subprocess
import sys

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


def from_values(values, dtype, sample_type="") -> aus.AudioSamples:
    audio = aus.AudioSamples.new_mono(np.array(values, dtype=dtype), 8000)
    return audio if not sample_type else audio.cast_as(SAMPLE_TYPES[sample_type])


INT_OPERATORS = {
    "x + 1": lambda x: x + 1,
    "1 + x": lambda x: 1 + x,
    "x - 1": lambda x: x - 1,
    "1 - x": lambda x: 1 - x,
}
# Stored values before, and after each of the operators above.
RAW_INTEGER_CASES = {
    "i16": (np.int16, [100, -100, 30000], [101, -99, 30001], [101, -99, 30001], [99, -101, 29999], [-99, 101, -29999]),
    "I24": (np.int32, [100, -100, 8000000], [101, -99, 8000001], [101, -99, 8000001], [99, -101, 7999999], [-99, 101, -7999999]),
    "i32": (np.int32, [100, -100, 2000000000], [101, -99, 2000000001], [101, -99, 2000000001], [99, -101, 1999999999], [-99, 101, -1999999999]),
    # The stored value, so 128 is not special and 1 - x saturates at the bottom of the range.
    "u8": (np.uint8, [100, 128, 254], [101, 129, 255], [101, 129, 255], [99, 127, 253], [0, 0, 0]),
}


@pytest.mark.parametrize("sample_type", list(RAW_INTEGER_CASES))
def test_int_scalar_is_a_raw_sample_value(sample_type):
    dtype, values, *expected = RAW_INTEGER_CASES[sample_type]
    x = from_values(values, dtype, "I24" if sample_type == "I24" else "")
    assert x.dtype == sample_type
    for (name, operator), want in zip(INT_OPERATORS.items(), expected):
        result = operator(x)
        assert result.dtype == sample_type, name
        assert result.to_numpy().tolist() == want, name


@pytest.mark.parametrize("sample_type", list(RAW_INTEGER_CASES))
@pytest.mark.parametrize("statement", ["y += 1", "y -= 1"])
def test_in_place_int_scalar_is_a_raw_sample_value(sample_type, statement):
    dtype, values, plus, _, minus, _ = RAW_INTEGER_CASES[sample_type]
    namespace = {"y": from_values(values, dtype, "I24" if sample_type == "I24" else "")}
    exec(statement, namespace)
    assert namespace["y"].dtype == sample_type
    assert namespace["y"].to_numpy().tolist() == (plus if "+" in statement else minus)


def test_int_and_float_scalars_differ_on_integer_audio():
    x = from_values([100, -100, 0], np.int16)
    assert (x + 1.0).to_numpy().tolist() == [32767, 32667, 32767]
    assert (1.0 + x).to_numpy().tolist() == [32767, 32667, 32767]
    assert (x + 1).to_numpy().tolist() == [101, -99, 1]


@pytest.mark.parametrize("sample_type", TYPES)
@pytest.mark.parametrize("gain", [2, -3, 0, 7])
def test_int_gains_match_float_gains(sample_type, gain):
    x = make(sample_type)
    want_mul = (x * float(gain)).to_numpy()
    np.testing.assert_array_equal((x * gain).to_numpy(), want_mul)
    np.testing.assert_array_equal((gain * x).to_numpy(), want_mul)
    y = x.copy()
    y *= gain
    np.testing.assert_array_equal(y.to_numpy(), want_mul)
    assert y.dtype == sample_type
    if gain != 0:
        want_div = (x / float(gain)).to_numpy()
        np.testing.assert_array_equal((x / gain).to_numpy(), want_div)
        y = x.copy()
        y /= gain
        np.testing.assert_array_equal(y.to_numpy(), want_div)
    else:
        with pytest.raises(ZeroDivisionError):
            x / gain


@pytest.mark.parametrize("sample_type", TYPES)
def test_every_operator_with_int_and_float_scalars_on_both_sides_keeps_the_type(sample_type):
    x = make(sample_type)
    for scalar in (1, 1.0, np.int16(1), np.int64(1), np.uint8(1), np.float32(1.0), np.float64(1.0)):
        for result in (x + scalar, x - scalar, x * scalar, x / scalar, x.__radd__(scalar), x.__rsub__(scalar), x.__rmul__(scalar)):
            assert result.dtype == sample_type, type(scalar)
        y = x.copy()
        y += scalar
        y -= scalar
        y *= scalar
        y /= scalar
        assert y.dtype == sample_type, type(scalar)
    for result in (1 + x, 1 - x, 1 * x, 1.0 + x, 1.0 - x, 1.0 * x):
        assert result.dtype == sample_type


@pytest.mark.parametrize("sample_type", ["u8", "i16", "I24", "i32"])
@pytest.mark.parametrize("scalar", [np.int8(1), np.int16(1), np.int32(1), np.int64(1), np.uint8(1), np.uint64(1)])
def test_numpy_integer_scalars_behave_as_ints(sample_type, scalar):
    x = make(sample_type)
    np.testing.assert_array_equal((x + scalar).to_numpy(), (x + 1).to_numpy())
    np.testing.assert_array_equal((x - scalar).to_numpy(), (x - 1).to_numpy())
    np.testing.assert_array_equal((x * scalar).to_numpy(), (x * 1).to_numpy())
    np.testing.assert_array_equal(x.__radd__(scalar).to_numpy(), (1 + x).to_numpy())
    np.testing.assert_array_equal(x.__rsub__(scalar).to_numpy(), (1 - x).to_numpy())


@pytest.mark.parametrize("sample_type", ["u8", "i16", "I24", "i32"])
@pytest.mark.parametrize("scalar", [np.float32(0.25), np.float64(0.25)])
def test_numpy_float_scalars_behave_as_floats(sample_type, scalar):
    x = make(sample_type)
    np.testing.assert_array_equal((x + scalar).to_numpy(), (x + 0.25).to_numpy())
    np.testing.assert_array_equal((x - scalar).to_numpy(), (x - 0.25).to_numpy())


@pytest.mark.parametrize("sample_type", TYPES)
@pytest.mark.parametrize("scalar", [True, False, np.True_, np.False_, np.array(True)])
def test_bool_scalar_raises(sample_type, scalar):
    x = make(sample_type)
    for operation in (
        lambda: x + scalar,
        lambda: x - scalar,
        lambda: x * scalar,
        lambda: x / scalar,
        lambda: x.__radd__(scalar),
        lambda: x.__rsub__(scalar),
        lambda: x.__rmul__(scalar),
        lambda: x.copy().__iadd__(scalar),
        lambda: x.copy().__imul__(scalar),
    ):
        with pytest.raises(TypeError, match="bool"):
            operation()


def test_bool_array_is_still_an_array():
    x = from_values([100, -100, 7], np.int16)
    assert (x * np.array([True, False, True])).to_numpy().tolist() == [100, 0, 7]


@pytest.mark.parametrize("sample_type", ["u8", "i16", "I24", "i32"])
@pytest.mark.parametrize("huge", [2**40, 2**64, 10**400])
def test_int_too_large_for_the_sample_type_saturates(sample_type, huge):
    low, high = INTEGER_LIMITS[sample_type]
    x = make(sample_type)
    for operator, want in (
        (lambda: x + huge, high),
        (lambda: huge + x, high),
        (lambda: x - huge, low),
        (lambda: huge - x, high),
        (lambda: x + -huge, low),
        (lambda: -huge - x, low),
    ):
        result = operator().to_numpy()
        assert result.min() == want and result.max() == want
    y = x.copy()
    y += huge
    assert y.to_numpy().min() == high
    assert (x * huge).dtype == sample_type
    assert (x / huge).dtype == sample_type
    assert (x * huge).to_numpy().max() == high


@pytest.mark.parametrize("sample_type", ["i16", "I24", "i32"])
def test_int_results_saturate_at_the_limits(sample_type):
    low, high = INTEGER_LIMITS[sample_type]
    dtype = {"i16": np.int16}.get(sample_type, np.int32)
    x = from_values([high, low, 0], dtype, "I24" if sample_type == "I24" else "")
    assert (x + 1).to_numpy().tolist() == [high, low + 1, 1]
    assert (x - 1).to_numpy().tolist() == [high - 1, low, -1]
    assert (1 - x).to_numpy().tolist() == [1 - high, high, 1]


@pytest.mark.parametrize("sample_type", ["f32", "f64"])
def test_float_audio_takes_an_int_as_the_same_number(sample_type):
    x = make(sample_type)
    for integer in (1, 2, -3, np.int16(2)):
        number = float(integer)
        np.testing.assert_array_equal((x + integer).to_numpy(), (x + number).to_numpy())
        np.testing.assert_array_equal((x - integer).to_numpy(), (x - number).to_numpy())
        np.testing.assert_array_equal((x * integer).to_numpy(), (x * number).to_numpy())
        np.testing.assert_array_equal((x / integer).to_numpy(), (x / number).to_numpy())
        np.testing.assert_array_equal(x.__radd__(integer).to_numpy(), (number + x).to_numpy())
        np.testing.assert_array_equal(x.__rsub__(integer).to_numpy(), (number - x).to_numpy())
        np.testing.assert_array_equal(x.__rmul__(integer).to_numpy(), (number * x).to_numpy())


def test_float_audio_with_an_enormous_int_does_not_raise():
    x = make("f64")
    assert (x * 10**400).dtype == "f64"
    assert (x + 10**400).dtype == "f64"


@pytest.mark.parametrize("sample_type", ["i16", "I24", "i32"])
def test_int_scalar_on_multi_channel_audio(sample_type):
    mono = make(sample_type)
    stereo = aus.AudioSamples.stack([mono, mono])
    samples = stereo.to_numpy()
    np.testing.assert_array_equal((stereo + 1).to_numpy(), samples + 1)
    np.testing.assert_array_equal((1 - stereo).to_numpy(), 1 - samples.astype(np.int64))
    assert (stereo + 1).dtype == sample_type and (stereo + 1).shape == stereo.shape


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


def test_import_does_not_load_the_arithmetic_helper():
    code = "import sys, audio_samples; assert 'audio_samples._arith' not in sys.modules; assert 'numpy' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)
