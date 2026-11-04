"""Every arithmetic operator, reflected, in-place and unary ones included, on every sample type.

The sample type never changes and integer results saturate. On integer audio a float scalar
is in full-scale units and an int scalar is a number of sample steps relative to silence, and
operands that have no meaning for the sample type raise.
"""

from __future__ import annotations

import subprocess
import sys
import warnings

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
LIMITS = {
    "u8": (0, 255),
    "i16": (-32768, 32767),
    "I24": (-(2**23), 2**23 - 1),
    "i32": (-(2**31), 2**31 - 1),
}
# The steps relative to silence that an int scalar can be.
STEP_RANGE = {
    "u8": (-128, 127),
    "i16": (-32768, 32767),
    "I24": (-(2**23), 2**23 - 1),
    "i32": (-(2**31), 2**31 - 1),
}
TYPES = list(SAMPLE_TYPES)
INTEGER_TYPES = list(LIMITS)
FLOAT_TYPES = ["f32", "f64"]


def make(sample_type: str, amplitude: float = 0.5, offset: float = 0.0) -> aus.AudioSamples:
    base = aus.generation.sine_wave(100.0, 0.05, 8000, amplitude, dtype=np.float64)
    values = base.to_numpy() + offset
    return aus.AudioSamples.new_mono(values, 8000).to_format(SAMPLE_TYPES[sample_type])


def nonzero(sample_type: str) -> aus.AudioSamples:
    """Audio with no silent sample, so that it can be a divisor."""
    return make(sample_type, 0.25, 0.5)


def values(data, sample_type: str) -> aus.AudioSamples:
    """Audio holding exactly these stored values."""
    if sample_type == "I24":
        return aus.AudioSamples.new_mono(np.array(data, dtype=np.int32), 8000).cast_as(SampleType.I24)
    dtype = {"u8": np.uint8, "i16": np.int16, "i32": np.int32, "f32": np.float32, "f64": np.float64}[sample_type]
    return aus.AudioSamples.new_mono(np.array(data, dtype=dtype), 8000)


def stored(audio) -> list:
    return audio.to_numpy().tolist()


def assert_same(a, b) -> None:
    np.testing.assert_array_equal(a.to_numpy(), b.to_numpy())


# --- Every operator keeps the sample type -------------------------------------------------


def binary_operators(x, n):
    return {
        "x + 0.25": lambda: x + 0.25,
        "0.25 + x": lambda: 0.25 + x,
        "x - 0.25": lambda: x - 0.25,
        "0.25 - x": lambda: 0.25 - x,
        "x + 1": lambda: x + 1,
        "1 + x": lambda: 1 + x,
        "x - 1": lambda: x - 1,
        "1 - x": lambda: 1 - x,
        "x * 0.5": lambda: x * 0.5,
        "0.5 * x": lambda: 0.5 * x,
        "x * 2": lambda: x * 2,
        "2 * x": lambda: 2 * x,
        "x / 2.0": lambda: x / 2.0,
        "x / 2": lambda: x / 2,
        "x ** 2.0": lambda: x**2.0,
        "x + x": lambda: x + x,
        "x - x": lambda: x - x,
        "x * x": lambda: x * x,
        "x / x": lambda: x / x,
        "x + n": lambda: x + n,
        "x - n": lambda: x - n,
        "x * n": lambda: x * n,
        "x / n": lambda: x / n,
        "n + x": lambda: n + x,
        "n - x": lambda: n - x,
        "n * x": lambda: n * x,
        "n / x": lambda: n / x,
        "x + float array": lambda: x + np.full(n.shape, 1.0),
        "x * float array": lambda: x * np.full(n.shape, 0.5),
        "x / float array": lambda: x / np.full(n.shape, 2.0),
        "float array + x": lambda: np.full(n.shape, 1.0) + x,
        "np.int64(1) + x": lambda: np.int64(1) + x,
        "np.int64(1) - x": lambda: np.int64(1) - x,
        "np.int64(2) * x": lambda: np.int64(2) * x,
        "np.float64(0.5) + x": lambda: np.float64(0.5) + x,
        "np.float32(0.5) - x": lambda: np.float32(0.5) - x,
        "np.float64(0.5) * x": lambda: np.float64(0.5) * x,
        "np.add(x, 1)": lambda: np.add(x, 1),
        "np.multiply(x, x)": lambda: np.multiply(x, x),
        "-x": lambda: -x,
        "+x": lambda: +x,
        "abs(x)": lambda: abs(x),
    }


@pytest.mark.parametrize("sample_type", TYPES)
@pytest.mark.parametrize("operator", list(binary_operators(None, np.zeros(1))))
def test_operator_keeps_the_sample_type(sample_type, operator):
    x = nonzero(sample_type)
    result = binary_operators(x, x.to_numpy())[operator]()
    assert isinstance(result, aus.AudioSamples)
    assert result.dtype == sample_type
    assert result.sample_rate == x.sample_rate
    assert result.samples_per_channel() == x.samples_per_channel()


def _binary_forms(x, n, y):
    """(statement on y, the binary expression it should equal)."""
    return {
        "+= 0.25": (lambda: y.__iadd__(0.25), lambda: x + 0.25),
        "-= 0.25": (lambda: y.__isub__(0.25), lambda: x - 0.25),
        "+= 1": (lambda: y.__iadd__(1), lambda: x + 1),
        "-= 1": (lambda: y.__isub__(1), lambda: x - 1),
        "*= 0.5": (lambda: y.__imul__(0.5), lambda: x * 0.5),
        "/= 2.0": (lambda: y.__itruediv__(2.0), lambda: x / 2.0),
        "*= 2": (lambda: y.__imul__(2), lambda: x * 2),
        "/= 2": (lambda: y.__itruediv__(2), lambda: x / 2),
        "+= audio": (lambda: y.__iadd__(x), lambda: x + x),
        "-= audio": (lambda: y.__isub__(x), lambda: x - x),
        "*= audio": (lambda: y.__imul__(x), lambda: x * x),
        "/= audio": (lambda: y.__itruediv__(x), lambda: x / x),
        "+= array": (lambda: y.__iadd__(n), lambda: x + n),
        "-= array": (lambda: y.__isub__(n), lambda: x - n),
        "*= array": (lambda: y.__imul__(n), lambda: x * n),
        "/= array": (lambda: y.__itruediv__(n), lambda: x / n),
        "+= itself": (lambda: y.__iadd__(y), lambda: x + x),
        "-= itself": (lambda: y.__isub__(y), lambda: x - x),
        "*= itself": (lambda: y.__imul__(y), lambda: x * x),
        "/= itself": (lambda: y.__itruediv__(y), lambda: x / x),
    }


@pytest.mark.parametrize("sample_type", TYPES)
@pytest.mark.parametrize("operator", list(_binary_forms(None, None, None)))
def test_in_place_operator_matches_the_binary_one(sample_type, operator):
    x = nonzero(sample_type)
    y = x.copy()
    in_place, binary = _binary_forms(x, x.to_numpy(), y)[operator]
    in_place()
    assert y.dtype == sample_type
    assert_same(y, binary())


@pytest.mark.parametrize("sample_type", TYPES)
def test_in_place_statements_keep_the_object(sample_type):
    y = nonzero(sample_type)
    identity = y
    y += y
    y -= 0.0
    y += 1
    y -= 1
    y *= 1.0
    y *= 1
    y /= 1.0
    y /= 1
    assert y is identity and y.dtype == sample_type
    y = nonzero(sample_type)
    identity = y
    y *= y
    y = nonzero(sample_type)
    y /= y
    assert y.dtype == sample_type


@pytest.mark.parametrize("sample_type", TYPES)
def test_reflected_operators_commute(sample_type):
    x = make(sample_type)
    assert_same(0.25 + x, x + 0.25)
    assert_same(0.5 * x, x * 0.5)
    assert_same(1 + x, x + 1)
    assert_same(2 * x, x * 2)


@pytest.mark.parametrize("sample_type", TYPES)
def test_numpy_on_the_left_is_the_same_as_on_the_right(sample_type):
    x = make(sample_type)
    n = x.to_numpy()
    assert_same(np.int64(1) + x, 1 + x)
    assert_same(np.int64(1) - x, 1 - x)
    assert_same(np.int64(3) * x, 3 * x)
    assert_same(np.float64(0.25) + x, 0.25 + x)
    assert_same(np.float32(0.25) - x, 0.25 - x)
    assert_same(np.float64(0.5) * x, 0.5 * x)
    assert_same(n + x, x + n)
    assert_same(n * x, x * n)
    with pytest.raises(TypeError):
        np.float64(1.0) / x


@pytest.mark.parametrize("sample_type", TYPES)
def test_adding_silence_changes_nothing(sample_type):
    x = make(sample_type)
    for result in (x + 0.0, 0.0 + x, x - 0.0, x + 0, 0 + x, x - 0, +x):
        assert_same(result, x)


@pytest.mark.parametrize("sample_type", TYPES)
def test_subtracting_audio_from_itself_gives_silence(sample_type):
    x = make(sample_type)
    assert_same(x - x, make(sample_type, 0.0))


# --- Int scalars -------------------------------------------------------------------------

# Stored values, then x + 1, 1 + x, x - 1, 1 - x.
INT_CASES = {
    "i16": ([100, -100, 30000], [101, -99, 30001], [101, -99, 30001], [99, -101, 29999], [-99, 101, -29999]),
    "I24": ([100, -100, 8000000], [101, -99, 8000001], [101, -99, 8000001], [99, -101, 7999999], [-99, 101, -7999999]),
    "i32": (
        [100, -100, 2000000000],
        [101, -99, 2000000001],
        [101, -99, 2000000001],
        [99, -101, 1999999999],
        [-99, 101, -1999999999],
    ),
    "u8": ([100, 128, 254], [101, 129, 255], [101, 129, 255], [99, 127, 253], [157, 129, 3]),
}


@pytest.mark.parametrize("sample_type", INTEGER_TYPES)
def test_int_scalar_is_a_number_of_steps_from_silence(sample_type):
    data, plus, reflected_plus, minus, reflected_minus = INT_CASES[sample_type]
    x = values(data, sample_type)
    assert x.dtype == sample_type
    assert stored(x + 1) == plus
    assert stored(1 + x) == reflected_plus
    assert stored(x - 1) == minus
    assert stored(1 - x) == reflected_minus
    y = x.copy()
    y += 1
    assert stored(y) == plus and y.dtype == sample_type
    y = x.copy()
    y -= 1
    assert stored(y) == minus and y.dtype == sample_type


def test_int_and_float_scalars_differ_on_integer_audio():
    x = values([100, -100, 0], "i16")
    assert stored(x + 1.0) == [32767, 32667, 32767]
    assert stored(1.0 + x) == [32767, 32667, 32767]
    assert stored(x + 1) == [101, -99, 1]


@pytest.mark.parametrize("sample_type", INTEGER_TYPES)
def test_largest_int_scalars_are_accepted(sample_type):
    low, high = STEP_RANGE[sample_type]
    x = make(sample_type)
    assert (x + high).dtype == sample_type
    assert (low + x).dtype == sample_type
    assert (x - low).dtype == sample_type


@pytest.mark.parametrize("sample_type", INTEGER_TYPES)
@pytest.mark.parametrize("beyond", ["high", "low", "huge", "enormous", "negative enormous"])
def test_int_scalar_that_does_not_fit_raises(sample_type, beyond):
    low, high = STEP_RANGE[sample_type]
    value = {"high": high + 1, "low": low - 1, "huge": 2**64, "enormous": 10**400, "negative enormous": -(10**400)}[beyond]
    x = make(sample_type)
    y = x.copy()
    before = stored(y)
    for operation in (
        lambda: x + value,
        lambda: value + x,
        lambda: x - value,
        lambda: value - x,
        lambda: y.__iadd__(value),
        lambda: y.__isub__(value),
    ):
        with pytest.raises(OverflowError, match=f"from {low} to {high}"):
            operation()
    assert stored(y) == before


def test_overflow_error_names_value_type_and_range():
    with pytest.raises(OverflowError, match="40000 does not fit int16 audio.*-32768 to 32767"):
        values([1], "i16") + 40000
    with pytest.raises(OverflowError, match="200 does not fit uint8 audio.*-128 to 127"):
        values([1], "u8") + 200
    with pytest.raises(OverflowError, match="int24 audio.*-8388608 to 8388607"):
        values([1], "I24") + 2**23
    with pytest.raises(OverflowError, match="int32 audio"):
        values([1], "i32") + 2**31


@pytest.mark.parametrize("sample_type", INTEGER_TYPES)
def test_valid_int_scalar_whose_result_overflows_saturates(sample_type):
    low, high = LIMITS[sample_type]
    x = values([high, low], sample_type)
    assert stored(x + 1) == [high, low + 1]
    assert stored(x - 1) == [high - 1, low]


@pytest.mark.parametrize("sample_type", TYPES)
@pytest.mark.parametrize("gain", [2, -3, 0, 7, 40000, 2**40])
def test_int_gains_match_float_gains(sample_type, gain):
    x = make(sample_type)
    want_mul = x * float(gain)
    assert_same(x * gain, want_mul)
    assert_same(gain * x, want_mul)
    y = x.copy()
    y *= gain
    assert_same(y, want_mul)
    if gain != 0:
        want_div = x / float(gain)
        assert_same(x / gain, want_div)
        y = x.copy()
        y /= gain
        assert_same(y, want_div)


@pytest.mark.parametrize("sample_type", TYPES)
def test_int_gain_beyond_float_raises(sample_type):
    x = make(sample_type)
    for operation in (lambda: x * 10**400, lambda: 10**400 * x, lambda: x / 10**400):
        with pytest.raises(OverflowError):
            operation()


# --- NumPy integer scalars ---------------------------------------------------------------


def test_numpy_scalar_with_the_sample_dtype_is_a_sample_on_u8():
    u = values([100, 128, 254], "u8")
    assert stored(u + np.uint8(128)) == [100, 128, 254]
    assert stored(u - np.uint8(128)) == [100, 128, 254]
    assert stored(np.uint8(128) + u) == [100, 128, 254]
    assert stored(u + np.uint8(129)) == [101, 129, 255]
    assert stored(u + np.uint8(0)) == [0, 0, 126]
    assert stored(np.uint8(129) - u) == [157, 129, 3]
    y = u.copy()
    y += np.uint8(128)
    assert stored(y) == [100, 128, 254]


def test_other_numpy_ints_on_u8_are_steps():
    u = values([100, 128, 254], "u8")
    for scalar in (np.int8(1), np.int16(1), np.int32(1), np.int64(1), np.uint16(1), np.uint64(1)):
        assert stored(u + scalar) == [101, 129, 255], type(scalar)
        assert stored(u - scalar) == [99, 127, 253], type(scalar)
    with pytest.raises(OverflowError):
        u + np.int16(200)
    with pytest.raises(OverflowError):
        u + np.uint16(128)


@pytest.mark.parametrize("sample_type", ["i16", "I24", "i32"])
@pytest.mark.parametrize(
    "scalar", [np.int8(1), np.int16(1), np.int32(1), np.int64(1), np.uint8(1), np.uint16(1), np.uint32(1), np.uint64(1)]
)
def test_numpy_integer_scalars_are_steps_on_signed_audio(sample_type, scalar):
    x = make(sample_type)
    assert_same(x + scalar, x + 1)
    assert_same(x - scalar, x - 1)
    assert_same(x * scalar, x * 1)
    assert_same(x.__radd__(scalar), 1 + x)
    assert_same(x.__rsub__(scalar), 1 - x)


def test_numpy_uint8_on_signed_audio_has_no_offset():
    assert stored(values([100], "i16") + np.uint8(200)) == [300]


@pytest.mark.parametrize("sample_type", INTEGER_TYPES)
def test_numpy_scalar_out_of_range_raises(sample_type):
    with pytest.raises(OverflowError):
        make(sample_type) + np.uint64(2**64 - 1)
    with pytest.raises(OverflowError):
        make(sample_type) + np.int64(2**40)


def test_numpy_int32_on_int24_audio_is_range_checked():
    with pytest.raises(OverflowError, match="int24"):
        make("I24") + np.int32(2**23)
    assert stored(values([5], "I24") + np.int32(2)) == [7]


# --- Float scalars on integer audio ------------------------------------------------------


@pytest.mark.parametrize("sample_type", INTEGER_TYPES)
@pytest.mark.parametrize("scalar", [np.float32(0.25), np.float64(0.25)])
def test_numpy_float_scalars_behave_as_floats(sample_type, scalar):
    x = make(sample_type)
    assert_same(x + scalar, x + 0.25)
    assert_same(x - scalar, x - 0.25)


NON_FINITE = [float("nan"), float("inf"), -float("inf"), np.float64("nan"), np.float32("inf")]


@pytest.mark.parametrize("sample_type", INTEGER_TYPES)
@pytest.mark.parametrize("scalar", NON_FINITE)
def test_non_finite_scalar_raises_on_integer_audio_without_a_warning(sample_type, scalar):
    x = make(sample_type)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for operation in (
            lambda: x + scalar,
            lambda: x.__radd__(scalar),
            lambda: x - scalar,
            lambda: x.__rsub__(scalar),
            lambda: x * scalar,
            lambda: x.__rmul__(scalar),
            lambda: x / scalar,
            lambda: x.copy().__iadd__(scalar),
            lambda: x.copy().__isub__(scalar),
            lambda: x.copy().__imul__(scalar),
            lambda: x.copy().__itruediv__(scalar),
        ):
            with pytest.raises(ValueError, match="finite"):
                operation()


@pytest.mark.parametrize("sample_type", INTEGER_TYPES)
@pytest.mark.parametrize("scalar", [10.0, -10.0, 1.0000001, -1.5, 1e300, np.float32(2.0), np.float64(2.0)])
def test_additive_float_outside_the_sample_range_raises(sample_type, scalar):
    x = make(sample_type)
    for operation in (
        lambda: x + scalar,
        lambda: x.__radd__(scalar),
        lambda: x - scalar,
        lambda: x.__rsub__(scalar),
        lambda: x.copy().__iadd__(scalar),
        lambda: x.copy().__isub__(scalar),
    ):
        with pytest.raises(ValueError, match="sample range"):
            operation()


@pytest.mark.parametrize("sample_type", INTEGER_TYPES)
def test_additive_float_at_full_scale_is_accepted(sample_type):
    low, high = LIMITS[sample_type]
    x = make(sample_type, 0.9)
    assert (x + 1.0).to_numpy().max() == high
    assert (x - 1.0).to_numpy().min() == low
    assert (1.0 + x).dtype == sample_type
    assert (-1.0 + x).dtype == sample_type


@pytest.mark.parametrize("sample_type", INTEGER_TYPES)
def test_finite_gain_saturates(sample_type):
    low, high = LIMITS[sample_type]
    x = make(sample_type)
    for gain in (10.0, 1e300, -10.0, 40000):
        result = (x * gain).to_numpy()
        assert result.max() == high and result.min() == low, gain
    assert_same(x * 10.0, 10.0 * x)
    assert (x / 1e-3).to_numpy().max() == high


@pytest.mark.parametrize("sample_type", INTEGER_TYPES)
@pytest.mark.parametrize("zero", [0, 0.0, -0.0, np.int16(0), np.float32(0.0)])
def test_scalar_division_by_zero_raises(sample_type, zero):
    x = make(sample_type)
    with pytest.raises(ZeroDivisionError):
        x / zero
    with pytest.raises(ZeroDivisionError):
        x.copy().__itruediv__(zero)


@pytest.mark.parametrize("sample_type", INTEGER_TYPES)
def test_division_by_a_zero_in_audio_or_array_raises(sample_type):
    x = nonzero(sample_type)
    silence = make(sample_type, 0.0)
    n = x.to_numpy()
    zeros = np.zeros(n.shape)
    with_a_zero = np.ones(n.shape)
    with_a_zero[3] = 0
    for operation in (
        lambda: x / silence,
        lambda: x / zeros,
        lambda: x / with_a_zero,
        lambda: silence / silence,
        lambda: x.copy().__itruediv__(silence),
        lambda: x.copy().__itruediv__(zeros),
        lambda: n / silence,
        lambda: silence.__itruediv__(silence),
    ):
        with pytest.raises(ZeroDivisionError):
            operation()


@pytest.mark.parametrize("sample_type", INTEGER_TYPES)
@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_non_finite_array_raises_on_integer_audio(sample_type, bad):
    x = nonzero(sample_type)
    n = np.ones(x.to_numpy().shape)
    n[2] = bad
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for operation in (lambda: x + n, lambda: x - n, lambda: x * n, lambda: x / n, lambda: n + x, lambda: n - x, lambda: n * x):
            with pytest.raises(ValueError, match="finite"):
                operation()


def test_complex_array_raises_on_integer_audio():
    x = make("i16")
    with pytest.raises(TypeError):
        x * np.ones(x.to_numpy().shape, dtype=np.complex64)


@pytest.mark.parametrize("sample_type", INTEGER_TYPES)
def test_valid_operands_saturate(sample_type):
    low, high = LIMITS[sample_type]
    x = make(sample_type, 0.9)
    samples = x.to_numpy()
    for result in (x + 1.0, 1.0 + x, x + x, x * x):
        result_values = result.to_numpy()
        assert result_values.max() == high
        assert result_values.min() >= low
    assert (x - 1.0).to_numpy().min() == low
    assert (x + x).to_numpy()[np.argmax(samples)] == high


@pytest.mark.parametrize("sample_type", ["i16", "I24", "i32"])
def test_float_scalar_add_is_in_full_scale_units(sample_type):
    x = make(sample_type, 0.25)
    expected = (make("f64", 0.25) + 0.25).to_format(SAMPLE_TYPES[sample_type]).to_numpy()
    assert np.max(np.abs((x + 0.25).to_numpy().astype(np.float64) - expected.astype(np.float64))) <= 2


# --- Float audio is unchanged ------------------------------------------------------------


@pytest.mark.parametrize("sample_type", FLOAT_TYPES)
def test_float_audio_takes_an_int_as_the_same_number(sample_type):
    x = make(sample_type)
    for integer in (1, 2, -3, 40000, np.int16(2), np.uint8(128)):
        number = float(integer)
        assert_same(x + integer, x + number)
        assert_same(x - integer, x - number)
        assert_same(x * integer, x * number)
        assert_same(x / integer, x / number)
        assert_same(x.__radd__(integer), number + x)
        assert_same(x.__rsub__(integer), number - x)
        assert_same(x.__rmul__(integer), number * x)


@pytest.mark.parametrize("sample_type", FLOAT_TYPES)
def test_float_audio_follows_float_rules(sample_type):
    x = make(sample_type)
    samples = x.to_numpy()
    assert np.isnan((x + float("nan")).to_numpy()).all()
    assert np.isinf((x + float("inf")).to_numpy()).all()
    np.testing.assert_array_equal((x + 10.0).to_numpy(), samples + samples.dtype.type(10.0))
    with pytest.raises(ZeroDivisionError):
        x / 0.0
    with pytest.raises(OverflowError):  # as for 1.0 + 10**400
        x + 10**400
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        quotient = (x / (x - x)).to_numpy()
    assert np.isnan(quotient).any()


# --- Unary operators ---------------------------------------------------------------------


UNARY_CASES = {
    "u8": ([100, 128, 254, 0, 255], [156, 128, 2, 255, 1], [156, 128, 254, 255, 255]),
    "i16": ([100, -100, 32767, -32768, 0], [-100, 100, -32767, 32767, 0], [100, 100, 32767, 32767, 0]),
    "I24": ([100, -100, 2**23 - 1, -(2**23), 0], [-100, 100, -(2**23 - 1), 2**23 - 1, 0], [100, 100, 2**23 - 1, 2**23 - 1, 0]),
    "i32": ([100, -100, 2**31 - 1, -(2**31), 0], [-100, 100, -(2**31 - 1), 2**31 - 1, 0], [100, 100, 2**31 - 1, 2**31 - 1, 0]),
    "f32": ([0.5, -0.25, 0.0], [-0.5, 0.25, 0.0], [0.5, 0.25, 0.0]),
    "f64": ([0.5, -0.25, 0.0], [-0.5, 0.25, 0.0], [0.5, 0.25, 0.0]),
}


@pytest.mark.parametrize("sample_type", TYPES)
def test_unary_operators(sample_type):
    data, negated, absolute = UNARY_CASES[sample_type]
    x = values(data, sample_type)
    for result, want in ((-x, negated), (+x, data), (abs(x), absolute)):
        assert result.dtype == sample_type
        assert result is not x
        assert stored(result) == want
    assert stored(x) == data


@pytest.mark.parametrize("sample_type", TYPES)
def test_positive_is_a_copy(sample_type):
    x = make(sample_type)
    y = +x
    y *= 0
    assert not np.array_equal(x.to_numpy(), y.to_numpy())


@pytest.mark.parametrize("sample_type", TYPES)
def test_unary_operators_on_multi_channel_audio(sample_type):
    mono = make(sample_type)
    stereo = aus.AudioSamples.stack([mono, mono])
    for result in (-stereo, +stereo, abs(stereo)):
        assert result.dtype == sample_type and result.shape == stereo.shape
        assert result.num_channels == 2


# --- The rest ----------------------------------------------------------------------------


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
    x = values([100, -100, 7], "i16")
    assert stored(x * np.array([True, False, True])) == [100, 0, 7]


@pytest.mark.parametrize("sample_type", TYPES)
def test_mixed_sample_types_convert_to_the_left_operand(sample_type):
    x = nonzero(sample_type)
    other = nonzero("f64" if sample_type != "f64" else "i16")
    for result in (x + other, x - other, x * other, x / other):
        assert result.dtype == sample_type
    y = x.copy()
    y += other
    assert y.dtype == sample_type
    assert_same(x + other, x + other.to_format(SAMPLE_TYPES[sample_type]))


@pytest.mark.parametrize("sample_type", TYPES)
def test_multi_channel_keeps_type_and_shape(sample_type):
    mono = nonzero(sample_type)
    stereo = aus.AudioSamples.stack([mono, mono])
    for result in (0.25 + stereo, stereo - stereo, stereo * stereo, stereo / stereo, 0.25 - stereo, 1 + stereo, 1 - stereo):
        assert result.dtype == sample_type
        assert result.shape == stereo.shape
    y = stereo.copy()
    y *= y
    assert y.dtype == sample_type and y.shape == stereo.shape


@pytest.mark.parametrize("sample_type", ["i16", "I24", "i32"])
def test_int_scalar_on_multi_channel_audio(sample_type):
    mono = make(sample_type)
    stereo = aus.AudioSamples.stack([mono, mono])
    samples = stereo.to_numpy()
    np.testing.assert_array_equal((stereo + 1).to_numpy(), samples + 1)
    np.testing.assert_array_equal((1 - stereo).to_numpy(), 1 - samples.astype(np.int64))


@pytest.mark.parametrize("sample_type", TYPES)
def test_scalar_on_the_left_of_division_and_power_raises(sample_type):
    x = make(sample_type)
    with pytest.raises(TypeError):
        1.0 / x
    with pytest.raises(TypeError):
        2.0**x
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
