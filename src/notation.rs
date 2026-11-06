//! `audio_samples.notation`: tuning systems and musical notation, binding
//! `audio_samples::utils::notation`.

use std::collections::BTreeMap;

use audio_samples::utils::notation::{self as nt, TuningSystem};
use numpy::{PyArray1, PyUntypedArray, PyUntypedArrayMethods};
use pyo3::IntoPyObjectExt;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyString};

use crate::args::{Int, is_scalar, non_negative, value_err};
use crate::audio_err_to_py;

fn factors_out(py: Python<'_>, factors: Vec<Vec<(u32, i32)>>) -> PyResult<Bound<'_, PyAny>> {
    let out = PyList::empty(py);
    for f in factors {
        let d = PyDict::new(py);
        for (p, e) in f {
            d.set_item(p, e)?;
        }
        out.append(d)?;
    }
    Ok(out.into_any())
}

/// One octave of the Pythagorean tuning.
///
/// Args:
///
///     ``bins_per_octave`` (int): Number of intervals. Default 12.
///     sort (bool): In increasing order; otherwise by power of 3. Default true.
///     ``return_factors`` (bool): Return prime factorisations, as dicts, instead of ratios.
///
/// Returns:
///
///     numpy.ndarray[float64] | list[dict[int, int]]: Ratios in ``[1, 2)``, or factors.
#[pyfunction]
#[pyo3(
    signature = (*, bins_per_octave = Int(12), sort = true, return_factors = false),
    text_signature = "(*, bins_per_octave=12, sort=True, return_factors=False)"
)]
#[inline]
pub fn pythagorean_intervals(
    py: Python<'_>,
    bins_per_octave: Int,
    sort: bool,
    return_factors: bool,
) -> PyResult<Bound<'_, PyAny>> {
    let b = non_negative("bins_per_octave", bins_per_octave)?;
    if return_factors {
        return factors_out(py, nt::pythagorean_factors(b, sort));
    }
    Ok(PyArray1::from_vec(py, nt::pythagorean_intervals(b, sort)).into_any())
}

/// One octave of a just-intonation system over the given primes.
///
/// Args:
///
///     primes (``array_like[int]``): The primes, such as ``[3, 5]``.
///     ``bins_per_octave`` (int): Number of intervals. Default 12.
///     sort (bool): In increasing order; otherwise in the order chosen. Default true.
///     ``return_factors`` (bool): Return prime factorisations, as dicts, instead of ratios.
///
/// Returns:
///
///     numpy.ndarray[float64] | list[dict[int, int]]: Ratios in ``[1, 2)``, or factors.
#[pyfunction]
#[pyo3(
    signature = (*, primes, bins_per_octave = Int(12), sort = true, return_factors = false),
    text_signature = "(*, primes, bins_per_octave=12, sort=True, return_factors=False)"
)]
#[inline]
pub fn plimit_intervals<'py>(
    py: Python<'py>,
    primes: &Bound<'py, PyAny>,
    bins_per_octave: Int,
    sort: bool,
    return_factors: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let list = crate::args::i64_vec(primes, "primes")?;
    let list: Vec<u32> = list
        .into_iter()
        .map(|p| u32::try_from(p).map_err(|_| value_err("primes", "must be positive")))
        .collect::<PyResult<_>>()?;
    let b = non_negative("bins_per_octave", bins_per_octave)?;
    if return_factors {
        let f = nt::plimit_factors(&list, b, sort).map_err(audio_err_to_py)?;
        return factors_out(py, f);
    }
    let v = nt::plimit_intervals(&list, b, sort).map_err(audio_err_to_py)?;
    Ok(PyArray1::from_vec(py, v).into_any())
}

pub fn parse_tuning(intervals: &Bound<'_, PyAny>) -> PyResult<TuningSystem> {
    if let Ok(name) = intervals.cast::<PyString>() {
        let name = name.to_cow()?;
        return TuningSystem::from_name(&name).ok_or_else(|| {
            value_err(
                "intervals",
                format!("unknown {name:?}; expected 'equal', 'pythagorean', 'ji3', 'ji5', 'ji7' or ratios"),
            )
        });
    }
    Ok(TuningSystem::Custom(crate::features::args::f64_vec(
        intervals,
        "intervals",
    )?))
}

/// Maps a function over a scalar or a sequence, returning a value or an array.
fn map_each<'py, R: pyo3::IntoPyObject<'py>>(
    input: &Bound<'py, PyAny>,
    f: impl Fn(&Bound<'py, PyAny>) -> PyResult<R>,
) -> PyResult<Bound<'py, PyAny>> {
    let py = input.py();
    if input.is_instance_of::<PyString>() || is_scalar(input)? {
        return f(input)?.into_bound_py_any(py);
    }
    let mut out = Vec::new();
    for item in input.try_iter()? {
        out.push(f(&item?)?.into_bound_py_any(py)?);
    }
    py.import("numpy")?
        .getattr("array")?
        .call1((PyList::new(py, out)?,))
}

/// The name of a MIDI number.
///
/// Args:
///
///     midi (float | ``array_like``): MIDI numbers. A value that is not finite gives ``""``.
///     octave (bool): Include the octave. Default True.
///     cents (bool): Include the deviation in cents, such as ``"A4+25"``. Needs ``octave``.
///     key (str): The key, such as ``"F:maj"``, that decides the spelling. Default ``"C:maj"``.
///     unicode (bool): Unicode accidentals; otherwise ASCII ``#`` and ``b``. Default False.
///
/// Returns:
///
///     str | numpy.ndarray: The names.
#[pyfunction]
#[pyo3(
    signature = (midi, *, octave = true, cents = false, key = "C:maj", unicode = false),
    text_signature = "(midi, *, octave=True, cents=False, key='C:maj', unicode=False)"
)]
#[inline]
pub fn midi_to_note<'py>(
    midi: &Bound<'py, PyAny>,
    octave: bool,
    cents: bool,
    key: &str,
    unicode: bool,
) -> PyResult<Bound<'py, PyAny>> {
    map_each(midi, |m| {
        nt::midi_name(m.extract()?, octave, cents, key, unicode).map_err(audio_err_to_py)
    })
}

/// The name of a frequency.
///
/// Args:
///
///     hz (float | ``array_like``): Frequencies in Hz.
///     octave, cents, key, unicode: As ``midi_to_note``.
///
/// Returns:
///
///     str | numpy.ndarray: The names.
#[pyfunction]
#[pyo3(
    signature = (hz, *, octave = true, cents = false, key = "C:maj", unicode = false),
    text_signature = "(hz, *, octave=True, cents=False, key='C:maj', unicode=False)"
)]
#[inline]
pub fn hz_to_note<'py>(
    hz: &Bound<'py, PyAny>,
    octave: bool,
    cents: bool,
    key: &str,
    unicode: bool,
) -> PyResult<Bound<'py, PyAny>> {
    map_each(hz, |f| {
        let f: f64 = f.extract()?;
        #[allow(
            clippy::suboptimal_flops,
            reason = "the unfused form is the reference rounding"
        )]
        let midi = 12.0 * (f.log2() - 440f64.log2()) + 69.0;
        nt::midi_name(midi, octave, cents, key, unicode).map_err(audio_err_to_py)
    })
}

fn note_number(note: &str, round_midi: bool) -> PyResult<f64> {
    if note.is_empty() {
        return Ok(f64::NAN);
    }
    let midi =
        audio_samples::utils::audio_math::note_to_midi_fractional(note).map_err(audio_err_to_py)?;
    Ok(if round_midi {
        let rounded = midi.round();
        // Only an exact half rounds to even; any other fraction rounds to nearest.
        #[allow(
            clippy::float_cmp,
            reason = "an exact half decides the rounding direction"
        )]
        let is_half = (midi - midi.trunc()).abs() == 0.5;
        if is_half && rounded % 2.0 != 0.0 {
            rounded - midi.signum()
        } else {
            rounded
        }
    } else {
        midi
    })
}

/// The MIDI number of a note name.
///
/// Args:
///
///     note (str | Iterable[str]): Names such as ``"C#4"`` or ``"A4+25"``. An empty name
///         gives NaN.
///     ``round_midi`` (bool): Round to a whole number. Default True.
///
/// Returns:
///
///     float | numpy.ndarray: The numbers.
#[pyfunction]
#[pyo3(signature = (note, *, round_midi = true), text_signature = "(note, *, round_midi=True)")]
#[inline]
pub fn note_to_midi<'py>(
    note: &Bound<'py, PyAny>,
    round_midi: bool,
) -> PyResult<Bound<'py, PyAny>> {
    map_each(note, |n| {
        note_number(n.cast::<PyString>()?.to_cow()?.as_ref(), round_midi)
    })
}

/// The frequency of a note name.
///
/// Args:
///
///     note (str | Iterable[str]): Names such as ``"C#4"`` or ``"A4+25"``.
///     ``round_midi`` (bool): Round to a whole note first. Default False.
///
/// Returns:
///
///     float | numpy.ndarray: The frequencies in Hz.
#[pyfunction]
#[pyo3(signature = (note, *, round_midi = false), text_signature = "(note, *, round_midi=False)")]
#[inline]
pub fn note_to_hz<'py>(note: &Bound<'py, PyAny>, round_midi: bool) -> PyResult<Bound<'py, PyAny>> {
    map_each(note, |n| {
        note_number(n.cast::<PyString>()?.to_cow()?.as_ref(), round_midi)
            .map(|m| 440.0 * ((m - 69.0) / 12.0).exp2())
    })
}

/// Frequencies of a tuning system over several octaves.
///
/// Args:
///
///     ``n_bins`` (int): Number of frequencies.
///     fmin (float): The lowest frequency, in Hz.
///     intervals (str | ``array_like``): ``"equal"``, ``"pythagorean"``, ``"ji3"``, ``"ji5"``,
///         ``"ji7"``, or the ratios of one octave.
///     ``bins_per_octave`` (int): Intervals per octave. Default 12.
///     tuning (float): Offset of equal temperament in fractions of a bin.
///     sort (bool): In increasing order. Default true.
///
/// Returns:
///
///     numpy.ndarray[float64]: The frequencies.
#[pyfunction]
#[pyo3(
    signature = (n_bins, *, fmin, intervals, bins_per_octave = Int(12), tuning = 0.0, sort = true),
    text_signature = "(n_bins, *, fmin, intervals, bins_per_octave=12, tuning=0.0, sort=True)"
)]
#[inline]
pub fn interval_frequencies<'py>(
    py: Python<'py>,
    n_bins: Int,
    fmin: f64,
    intervals: &Bound<'py, PyAny>,
    bins_per_octave: Int,
    tuning: f64,
    sort: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let system = parse_tuning(intervals)?;
    let v = nt::interval_frequencies(
        non_negative("n_bins", n_bins)?,
        fmin,
        &system,
        non_negative("bins_per_octave", bins_per_octave)?,
        tuning,
        sort,
    )
    .map_err(audio_err_to_py)?;
    Ok(PyArray1::from_vec(py, v).into_any())
}

/// The note a number of perfect fifths from a unison.
///
/// Args:
///
///     unison (str): The starting note.
///     fifths (int): Signed number of fifths.
///     unicode (bool): Unicode accidentals; otherwise ASCII. Default true.
///
/// Returns:
///
///     str: The note's name, without an octave.
#[pyfunction]
#[pyo3(signature = (*, unison, fifths, unicode = true), text_signature = "(*, unison, fifths, unicode=True)")]
#[inline]
pub fn fifths_to_note(unison: &str, fifths: Int, unicode: bool) -> PyResult<String> {
    nt::fifths_to_note(unison, fifths.0, unicode).map_err(audio_err_to_py)
}

/// Maps a scalar or array of floats to a string, or an array of strings of the same shape.
fn map_names<'py>(
    values: &Bound<'py, PyAny>,
    name: &str,
    f: impl Fn(f64) -> PyResult<String>,
) -> PyResult<Bound<'py, PyAny>> {
    let py = values.py();
    if is_scalar(values)? {
        let v: f64 = values.extract().or_else(|_| {
            crate::args::with_float_view::<f64, _>(values, name, |a| {
                a.first()
                    .copied()
                    .ok_or_else(|| value_err(name, "is empty"))
            })
        })?;
        return f(v)?.into_bound_py_any(py);
    }
    crate::args::with_float_view::<f64, _>(values, name, |a| {
        let names = a.iter().map(|&v| f(v)).collect::<PyResult<Vec<String>>>()?;
        strings_out(py, names, a.shape())
    })
}

/// A new Unicode array of the given shape holding the names, as wide as the widest one.
///
/// `names` must yield one name per element, in C order.
fn svara_array<'py>(
    py: Python<'py>,
    shape: &[usize],
    widest: usize,
    names: impl Iterator<Item = PyResult<nt::Svara>>,
) -> PyResult<Bound<'py, PyAny>> {
    let width = widest.max(1);
    let array = py
        .import("numpy")?
        .getattr("empty")?
        .call1((shape.to_vec(), format!("<U{width}")))?;
    let cells = array.cast::<PyUntypedArray>()?;
    let total = shape.iter().product::<usize>() * width;
    // SAFETY: `cells` is a live NumPy array, so its object pointer is valid.
    let data = unsafe { (*cells.as_array_ptr()).data };
    #[allow(clippy::cast_ptr_alignment)] // NumPy aligns `<U` data to 4 bytes.
    let data = data.cast::<u32>();
    // SAFETY: the array was just created with `total` code points of 4-byte aligned,
    // contiguous storage, and nothing else refers to it yet.
    let cells = unsafe { std::slice::from_raw_parts_mut(data, total) };
    for (cell, name) in cells.chunks_exact_mut(width).zip(names) {
        let name = name?;
        let mut chars = name.chars();
        for slot in cell {
            *slot = chars.next().map_or(0, u32::from);
        }
    }
    Ok(array)
}

/// Maps a function giving a svara over a scalar or an array, returning a `str` or an array
/// of the input's shape. Each array element is named twice: once to find the widest name,
/// once to write it, so that no list of names is held.
fn map_svaras<'py>(
    values: &Bound<'py, PyAny>,
    name: &str,
    f: impl Fn(f64) -> PyResult<nt::Svara>,
) -> PyResult<Bound<'py, PyAny>> {
    let py = values.py();
    if is_scalar(values)? {
        let v: f64 = values.extract().or_else(|_| {
            crate::args::with_float_view::<f64, _>(values, name, |a| {
                a.first()
                    .copied()
                    .ok_or_else(|| value_err(name, "is empty"))
            })
        })?;
        return Ok(PyString::new(py, f(v)?.as_str()).into_any());
    }
    crate::args::with_float_view::<f64, _>(values, name, |a| {
        if a.is_empty() {
            return strings_out(py, Vec::new(), a.shape());
        }
        let mut widest = 0;
        for &v in &a {
            widest = widest.max(f(v)?.chars().count());
        }
        svara_array(py, a.shape(), widest, a.iter().map(|&v| f(v)))
    })
}

fn strings_out<'py>(
    py: Python<'py>,
    names: Vec<String>,
    shape: &[usize],
) -> PyResult<Bound<'py, PyAny>> {
    let list = PyList::new(py, names)?;
    let arr = py.import("numpy")?.getattr("array")?.call1((list,))?;
    arr.call_method1("reshape", (shape.to_vec(),))
}

/// The Functional Just System spelling of an interval.
///
/// Ratios built from the primes up to 23 are recognised, to six decimal places.
///
/// Args:
///
///     interval (float | ``array_like``): The ratio or ratios, positive.
///     unison (str): The note of ratio 1. Default ``"C"``.
///     tolerance (float): How close in ratio a stack of fifths must come. Default 65/63.
///     unicode (bool): Unicode accidentals and scripts; otherwise ``^n`` and ``_n``.
///
/// Returns:
///
///     str | numpy.ndarray[str]: The spelling, the shape of the input.
#[pyfunction]
#[pyo3(
    signature = (interval, *, unison = "C", tolerance = 65.0 / 63.0, unicode = true),
    text_signature = "(interval, *, unison='C', tolerance=1.0317460317460319, unicode=True)"
)]
#[inline]
pub fn interval_to_fjs<'py>(
    interval: &Bound<'py, PyAny>,
    unison: &str,
    tolerance: f64,
    unicode: bool,
) -> PyResult<Bound<'py, PyAny>> {
    map_names(interval, "interval", |r| {
        nt::interval_to_fjs(r, unison, tolerance, unicode).map_err(audio_err_to_py)
    })
}

/// The Functional Just System spelling of frequencies relative to a reference.
///
/// Args:
///
///     frequencies (float | ``array_like``): In Hz.
///     fmin (Optional[float]): The frequency of ratio 1. Default: the smallest input.
///     unison (Optional[str]): The note of ratio 1. Default: nearest note to ``fmin``.
///     unicode (bool): As for :func:`interval_to_fjs`. Default false.
///
/// Returns:
///
///     str | numpy.ndarray[str]: The spelling, the shape of the input.
#[pyfunction]
#[pyo3(
    signature = (frequencies, *, fmin = None, unison = None, unicode = false),
    text_signature = "(frequencies, *, fmin=None, unison=None, unicode=False)"
)]
#[inline]
pub fn hz_to_fjs<'py>(
    frequencies: &Bound<'py, PyAny>,
    fmin: Option<f64>,
    unison: Option<&str>,
    unicode: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let reference = match fmin {
        Some(f) => f,
        None => crate::args::with_float_view::<f64, _>(frequencies, "frequencies", |a| {
            Ok(a.iter().copied().fold(f64::INFINITY, f64::min))
        })?,
    };
    map_names(frequencies, "frequencies", |h| {
        nt::hz_to_fjs(h, reference, unison, unicode).map_err(audio_err_to_py)
    })
}

/// The ten thaats, by name.
///
/// Returns:
///
///     list[str]: The names.
#[pyfunction]
#[inline]
pub fn list_thaat(py: Python<'_>) -> PyResult<Bound<'_, PyList>> {
    static NAMES: pyo3::sync::PyOnceLock<Py<PyList>> = pyo3::sync::PyOnceLock::new();
    let names =
        NAMES.get_or_try_init(py, || PyList::new(py, nt::list_thaat()).map(Bound::unbind))?;
    let copy = names.bind(py).get_slice(0, usize::MAX);
    Ok(copy)
}

/// The scale degrees of a thaat, in semitones above Sa.
///
/// Args:
///
///     thaat (str): Its name, in any letter case.
///
/// Returns:
///
///     numpy.ndarray[int64]: Seven degrees.
#[pyfunction]
#[inline]
pub fn thaat_to_degrees<'py>(py: Python<'py>, thaat: &str) -> PyResult<Bound<'py, PyAny>> {
    let d = nt::thaat_to_degrees(thaat).map_err(audio_err_to_py)?;
    Ok(PyArray1::from_slice(py, &d.map(i64::from)).into_any())
}

/// The 72 melakarta ragas and their numbers.
///
/// Returns:
///
///     dict[str, int]: Name to number, from 1 to 72.
#[pyfunction]
#[inline]
pub fn list_mela(py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
    static MELAS: pyo3::sync::PyOnceLock<Py<PyDict>> = pyo3::sync::PyOnceLock::new();
    let melas = MELAS.get_or_try_init(py, || -> PyResult<Py<PyDict>> {
        let ordered: BTreeMap<&str, u8> = nt::list_mela()
            .iter()
            .zip(1_u8..)
            .map(|(m, i)| (*m, i))
            .collect();
        // Compiled as a literal: the names in a code object are immortal, so copying the
        // dict does not touch their reference counts.
        let items: Vec<String> = ordered.iter().map(|(k, v)| format!("{k:?}: {v}")).collect();
        let source = std::ffi::CString::new(format!("{{{}}}", items.join(", ")))?;
        Ok(py
            .eval(&source, None, None)?
            .cast_into::<PyDict>()?
            .unbind())
    })?;
    melas.bind(py).copy()
}

fn mela_arg(mela: &Bound<'_, PyAny>) -> PyResult<u8> {
    if let Ok(name) = mela.cast::<PyString>() {
        return nt::mela_number(&name.to_cow()?).map_err(audio_err_to_py);
    }
    let n: i64 = mela.extract()?;
    u8::try_from(n)
        .ok()
        .filter(|v| (1..=72).contains(v))
        .ok_or_else(|| value_err("mela", format!("{n} must be in the range 1 to 72")))
}

/// The scale degrees of a melakarta raga, in semitones above Sa.
///
/// Args:
///
///     mela (int | str): The raga's number, from 1 to 72, or its name.
///
/// Returns:
///
///     numpy.ndarray[int64]: Seven degrees.
#[pyfunction]
#[inline]
pub fn mela_to_degrees<'py>(mela: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let d = nt::mela_to_degrees(mela_arg(mela)?).map_err(audio_err_to_py)?;
    Ok(PyArray1::from_slice(mela.py(), &d.map(i64::from)).into_any())
}

/// The names of the twelve svaras of a melakarta raga, from Sa.
///
/// Args:
///
///     mela (int | str): The raga's number or name.
///     abbr (bool): Short forms such as ``R1``. Default true.
///     unicode (bool): Subscript digits; otherwise plain digits. Default true.
///
/// Returns:
///
///     list[str]: Twelve names.
#[pyfunction]
#[pyo3(signature = (mela, *, abbr = true, unicode = true), text_signature = "(mela, *, abbr=True, unicode=True)")]
#[inline]
pub fn mela_to_svara<'py>(
    mela: &Bound<'py, PyAny>,
    abbr: bool,
    unicode: bool,
) -> PyResult<Bound<'py, PyList>> {
    let names = nt::mela_to_svara(mela_arg(mela)?, abbr, unicode).map_err(audio_err_to_py)?;
    PyList::new(mela.py(), names.iter().map(nt::Svara::as_str))
}

/// The Hindustani svara of MIDI numbers.
///
/// Args:
///
///     midi (float | ``array_like``): MIDI numbers; a non-finite value gives ``""``.
///     Sa (float): The MIDI number of Sa.
///     abbr (bool): Short forms such as ``S``. Default true.
///     octave (bool): Mark the octaves next to Sa's. Default true.
///     unicode (bool): Dots; otherwise ``'`` and ``,``. Default true.
///
/// Returns:
///
///     str | numpy.ndarray[str]: The svara, the shape of the input.
#[pyfunction]
#[pyo3(
    signature = (midi, *, Sa, abbr = true, octave = true, unicode = true),
    text_signature = "(midi, *, Sa, abbr=True, octave=True, unicode=True)"
)]
#[allow(non_snake_case)]
#[inline]
pub fn midi_to_svara_h<'py>(
    midi: &Bound<'py, PyAny>,
    Sa: f64,
    abbr: bool,
    octave: bool,
    unicode: bool,
) -> PyResult<Bound<'py, PyAny>> {
    map_svaras(midi, "midi", |m| {
        Ok(nt::midi_to_svara_h(m, Sa, abbr, octave, unicode))
    })
}

/// The Carnatic svara of MIDI numbers within a melakarta raga.
///
/// Args:
///
///     midi (float | ``array_like``): MIDI numbers; a non-finite value gives ``""``.
///     Sa (float): The MIDI number of Sa.
///     mela (int | str): The raga's number or name.
///     abbr, octave, unicode: As for :func:`midi_to_svara_h`.
///
/// Returns:
///
///     str | numpy.ndarray[str]: The svara, the shape of the input.
#[pyfunction]
#[pyo3(
    signature = (midi, *, Sa, mela, abbr = true, octave = true, unicode = true),
    text_signature = "(midi, *, Sa, mela, abbr=True, octave=True, unicode=True)"
)]
#[allow(non_snake_case)]
#[inline]
pub fn midi_to_svara_c<'py>(
    midi: &Bound<'py, PyAny>,
    Sa: f64,
    mela: &Bound<'py, PyAny>,
    abbr: bool,
    octave: bool,
    unicode: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let m = mela_arg(mela)?;
    map_svaras(midi, "midi", |v| {
        nt::midi_to_svara_c(v, Sa, m, abbr, octave, unicode).map_err(audio_err_to_py)
    })
}

/// The Hindustani svara of frequencies.
///
/// Args:
///
///     frequencies (float | ``array_like``): In Hz.
///     Sa (float): The frequency of Sa, in Hz.
///     abbr, octave, unicode: As for :func:`midi_to_svara_h`.
///
/// Returns:
///
///     str | numpy.ndarray[str]: The svara, the shape of the input.
#[pyfunction]
#[pyo3(
    signature = (frequencies, *, Sa, abbr = true, octave = true, unicode = true),
    text_signature = "(frequencies, *, Sa, abbr=True, octave=True, unicode=True)"
)]
#[allow(non_snake_case)]
#[inline]
pub fn hz_to_svara_h<'py>(
    frequencies: &Bound<'py, PyAny>,
    Sa: f64,
    abbr: bool,
    octave: bool,
    unicode: bool,
) -> PyResult<Bound<'py, PyAny>> {
    map_svaras(frequencies, "frequencies", |h| {
        Ok(nt::hz_to_svara_h(h, Sa, abbr, octave, unicode))
    })
}

/// The Carnatic svara of frequencies within a melakarta raga.
///
/// Args:
///
///     frequencies (float | ``array_like``): In Hz.
///     Sa (float): The frequency of Sa, in Hz.
///     mela (int | str): The raga's number or name.
///     abbr, octave, unicode: As for :func:`midi_to_svara_h`.
///
/// Returns:
///
///     str | numpy.ndarray[str]: The svara, the shape of the input.
#[pyfunction]
#[pyo3(
    signature = (frequencies, *, Sa, mela, abbr = true, octave = true, unicode = true),
    text_signature = "(frequencies, *, Sa, mela, abbr=True, octave=True, unicode=True)"
)]
#[allow(non_snake_case)]
#[inline]
pub fn hz_to_svara_c<'py>(
    frequencies: &Bound<'py, PyAny>,
    Sa: f64,
    mela: &Bound<'py, PyAny>,
    abbr: bool,
    octave: bool,
    unicode: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let m = mela_arg(mela)?;
    map_svaras(frequencies, "frequencies", |h| {
        nt::hz_to_svara_c(h, Sa, m, abbr, octave, unicode).map_err(audio_err_to_py)
    })
}

/// Maps a note name or sequence of names to a `str`, or an array of them.
fn map_notes<'py>(
    notes: &Bound<'py, PyAny>,
    f: impl Fn(&str) -> PyResult<nt::Svara>,
) -> PyResult<Bound<'py, PyAny>> {
    let py = notes.py();
    if let Ok(s) = notes.cast::<PyString>() {
        return Ok(PyString::new(py, f(&s.to_cow()?)?.as_str()).into_any());
    }
    let mut out = Vec::new();
    for n in notes.try_iter()? {
        let n = n?;
        out.push(f(&n.cast::<PyString>()?.to_cow()?)?);
    }
    if out.is_empty() {
        return strings_out(py, Vec::new(), &[0]);
    }
    let widest = out.iter().map(|n| n.chars().count()).max().unwrap_or(0);
    svara_array(py, &[out.len()], widest, out.iter().map(|n| Ok(*n)))
}

/// The Hindustani svara of note names.
///
/// Args:
///
///     notes (str | sequence of str): The notes.
///     Sa (str): The note of Sa.
///     abbr, octave, unicode: As for :func:`midi_to_svara_h`.
///
/// Returns:
///
///     str | numpy.ndarray[str]: The svara.
#[pyfunction]
#[pyo3(
    signature = (notes, *, Sa, abbr = true, octave = true, unicode = true),
    text_signature = "(notes, *, Sa, abbr=True, octave=True, unicode=True)"
)]
#[allow(non_snake_case)]
#[inline]
pub fn note_to_svara_h<'py>(
    notes: &Bound<'py, PyAny>,
    Sa: &str,
    abbr: bool,
    octave: bool,
    unicode: bool,
) -> PyResult<Bound<'py, PyAny>> {
    map_notes(notes, |n| {
        nt::note_to_svara_h(n, Sa, abbr, octave, unicode).map_err(audio_err_to_py)
    })
}

/// The Carnatic svara of note names within a melakarta raga.
///
/// Args:
///
///     notes (str | sequence of str): The notes.
///     Sa (str): The note of Sa.
///     mela (int | str): The raga's number or name.
///     abbr, octave, unicode: As for :func:`midi_to_svara_h`.
///
/// Returns:
///
///     str | numpy.ndarray[str]: The svara.
#[pyfunction]
#[pyo3(
    signature = (notes, *, Sa, mela, abbr = true, octave = true, unicode = true),
    text_signature = "(notes, *, Sa, mela, abbr=True, octave=True, unicode=True)"
)]
#[allow(non_snake_case)]
#[inline]
pub fn note_to_svara_c<'py>(
    notes: &Bound<'py, PyAny>,
    Sa: &str,
    mela: &Bound<'py, PyAny>,
    abbr: bool,
    octave: bool,
    unicode: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let m = mela_arg(mela)?;
    map_notes(notes, |n| {
        nt::note_to_svara_c(n, Sa, m, abbr, octave, unicode).map_err(audio_err_to_py)
    })
}

#[inline]
pub fn register(py: Python<'_>, parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(py, "notation")?;
    m.add(
        "__doc__",
        "Tuning systems and musical notation: interval generators, Functional Just System, svara, mela and thaat names.",
    )?;
    m.add_function(wrap_pyfunction!(pythagorean_intervals, &m)?)?;
    m.add_function(wrap_pyfunction!(plimit_intervals, &m)?)?;
    m.add_function(wrap_pyfunction!(midi_to_note, &m)?)?;
    m.add_function(wrap_pyfunction!(hz_to_note, &m)?)?;
    m.add_function(wrap_pyfunction!(note_to_midi, &m)?)?;
    m.add_function(wrap_pyfunction!(note_to_hz, &m)?)?;
    m.add_function(wrap_pyfunction!(interval_frequencies, &m)?)?;
    m.add_function(wrap_pyfunction!(fifths_to_note, &m)?)?;
    m.add_function(wrap_pyfunction!(interval_to_fjs, &m)?)?;
    m.add_function(wrap_pyfunction!(hz_to_fjs, &m)?)?;
    m.add_function(wrap_pyfunction!(list_thaat, &m)?)?;
    m.add_function(wrap_pyfunction!(thaat_to_degrees, &m)?)?;
    m.add_function(wrap_pyfunction!(list_mela, &m)?)?;
    m.add_function(wrap_pyfunction!(mela_to_degrees, &m)?)?;
    m.add_function(wrap_pyfunction!(mela_to_svara, &m)?)?;
    m.add_function(wrap_pyfunction!(midi_to_svara_h, &m)?)?;
    m.add_function(wrap_pyfunction!(midi_to_svara_c, &m)?)?;
    m.add_function(wrap_pyfunction!(hz_to_svara_h, &m)?)?;
    m.add_function(wrap_pyfunction!(hz_to_svara_c, &m)?)?;
    m.add_function(wrap_pyfunction!(note_to_svara_h, &m)?)?;
    m.add_function(wrap_pyfunction!(note_to_svara_c, &m)?)?;
    parent.add_submodule(&m)?;
    Ok(())
}
