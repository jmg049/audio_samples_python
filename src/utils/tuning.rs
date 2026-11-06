//! Tuning, octave and key helpers of `audio_samples.utils`, binding the
//! functions of `audio_samples::utils::audio_math` (`hz_to_octs`,
//! `octs_to_hz`, `A4_to_tuning`, `tuning_to_A4`, `cqt_frequencies`,
//! `key_to_notes`, `key_to_degrees`).

use crate::args::Int;
use audio_samples::utils::audio_math;
use numpy::PyArray1;
use pyo3::prelude::*;

use crate::args::{map_f64, non_negative, positive};
use crate::audio_err_to_py;

/// Converts frequencies (Hz) to fractional octave numbers. Octave 0 is A0
/// (27.5 Hz for A440 tuning), so A4 is octave 4.0.
///
/// Args:
///
///     frequencies (float | array_like): Frequencies in Hz.
///     tuning (float): Tuning deviation from A440 in fractions of a bin. Default 0.
///     bins_per_octave (int): Bins per octave the tuning refers to. Default 12.
///
/// Returns:
///
///     float | numpy.ndarray[float64]: octave numbers, same shape as the input.
#[pyfunction]
#[pyo3(
    signature = (frequencies, *, tuning = 0.0, bins_per_octave = Int(12)),
    text_signature = "(frequencies, *, tuning=0.0, bins_per_octave=12)"
)]
pub fn hz_to_octs<'py>(
    frequencies: &Bound<'py, PyAny>,
    tuning: f64,
    bins_per_octave: Int,
) -> PyResult<Bound<'py, PyAny>> {
    let b = positive("bins_per_octave", bins_per_octave)?;
    crate::args::map_f64_slice(
        frequencies,
        "frequencies",
        |f| audio_math::hz_to_octs(f, tuning, b),
        |fs| audio_math::hz_to_octs_slice(fs, tuning, b),
    )
}

/// Converts fractional octave numbers to frequencies (Hz), the inverse of
/// ``hz_to_octs``.
///
/// Args:
///
///     octs (float | array_like): Octave numbers.
///     tuning (float): Tuning deviation in fractions of a bin. Default 0.
///     bins_per_octave (int): Default 12.
///
/// Returns:
///
///     float | numpy.ndarray[float64]: frequencies in Hz, same shape as the input.
#[pyfunction]
#[pyo3(
    signature = (octs, *, tuning = 0.0, bins_per_octave = Int(12)),
    text_signature = "(octs, *, tuning=0.0, bins_per_octave=12)"
)]
pub fn octs_to_hz<'py>(
    octs: &Bound<'py, PyAny>,
    tuning: f64,
    bins_per_octave: Int,
) -> PyResult<Bound<'py, PyAny>> {
    let b = positive("bins_per_octave", bins_per_octave)?;
    map_f64(octs, "octs", |o| audio_math::octs_to_hz(o, tuning, b))
}

/// Converts an A4 reference frequency to a tuning deviation in fractions of a
/// bin.
///
/// Args:
///
///     A4 (float | array_like): Reference frequency of A4 in Hz.
///     bins_per_octave (int): Default 12.
///
/// Returns:
///
///     float | numpy.ndarray[float64]: tuning, same shape as the input.
#[pyfunction]
#[pyo3(signature = (A4, *, bins_per_octave = Int(12)), text_signature = "(A4, *, bins_per_octave=12)")]
#[allow(non_snake_case)]
pub fn a4_to_tuning<'py>(
    A4: &Bound<'py, PyAny>,
    bins_per_octave: Int,
) -> PyResult<Bound<'py, PyAny>> {
    let b = positive("bins_per_octave", bins_per_octave)?;
    crate::args::map_f64_slice(
        A4,
        "A4",
        |a| audio_math::a4_to_tuning(a, b),
        |a4| audio_math::a4_to_tuning_slice(a4, b),
    )
}

/// Converts a tuning deviation (fractions of a bin) to the A4 reference
/// frequency.
///
/// Args:
///
///     tuning (float | array_like): Tuning deviation.
///     bins_per_octave (int): Default 12.
///
/// Returns:
///
///     float | numpy.ndarray[float64]: A4 in Hz, same shape as the input.
#[pyfunction]
#[pyo3(signature = (tuning, *, bins_per_octave = Int(12)), text_signature = "(tuning, *, bins_per_octave=12)")]
pub fn tuning_to_a4<'py>(
    tuning: &Bound<'py, PyAny>,
    bins_per_octave: Int,
) -> PyResult<Bound<'py, PyAny>> {
    let b = positive("bins_per_octave", bins_per_octave)?;
    map_f64(tuning, "tuning", |t| audio_math::tuning_to_a4(t, b))
}

/// Centre frequencies of a constant-Q filterbank.
///
/// Args:
///
///     n_bins (int): Number of bins.
///     fmin (float): Frequency of the first bin (Hz, before tuning). Required.
///     bins_per_octave (int): Default 12.
///     tuning (float): Tuning deviation in fractions of a bin. Default 0.
///
/// Returns:
///
///     numpy.ndarray[float64]: shape ``(n_bins,)``.
#[pyfunction]
#[pyo3(
    signature = (n_bins, *, fmin, bins_per_octave = Int(12), tuning = 0.0),
    text_signature = "(n_bins, *, fmin, bins_per_octave=12, tuning=0.0)"
)]
pub fn cqt_frequencies(
    py: Python<'_>,
    n_bins: Int,
    fmin: f64,
    bins_per_octave: Int,
    tuning: f64,
) -> PyResult<Bound<'_, PyArray1<f64>>> {
    use numpy::{PyArray1, PyArrayMethods};
    // Computed straight into a NumPy-allocated array (no Rust buffer).
    let n = non_negative("n_bins", n_bins)?;
    let bpo = positive("bins_per_octave", bins_per_octave)?;
    let out = PyArray1::<f64>::zeros(py, n, false);
    // SAFETY: the array was just created and is not shared yet, so nothing
    // else can borrow it. Writing without a NumPy borrow avoids an entry in
    // rust-numpy's borrow table (an allocation per call).
    let slice = unsafe { out.as_slice_mut() }.expect("a new array is contiguous");
    audio_math::cqt_frequencies_into(slice, fmin, bpo, tuning);
    Ok(out)
}

/// Spells the 12 pitch classes for a key.
///
/// Args:
///
///     key (str): A key such as ``"C:maj"``, ``"F#:min"``, ``"Bb:dorian"``.
///     unicode (bool): Use ``♯`` / ``♭`` (default True) instead of ``#`` / ``b``.
///     natural (bool): Mark naturals with ``♮`` (or ``n``). Default False.
///
/// Returns:
///
///     list[str]: 12 note names, starting from C.
///
/// Raises:
///
///     ValueError: For an unparsable key.
#[pyfunction]
#[pyo3(signature = (key, *, unicode = true, natural = false), text_signature = "(key, *, unicode=True, natural=False)")]
pub fn key_to_notes<'py>(
    py: Python<'py>,
    key: &str,
    unicode: bool,
    natural: bool,
) -> PyResult<Bound<'py, pyo3::types::PyList>> {
    use pyo3::sync::MutexExt;
    use pyo3::types::{PyList, PySequenceMethods, PyString, PyTuple};
    use std::collections::HashMap;
    use std::sync::{LazyLock, Mutex, PoisonError};
    // The 12 names come from a static table. Each table's Python strings are
    // made once and kept in a tuple; a call returns a new list copied from
    // it, with no string objects created or looked up per call.
    static TUPLES: LazyLock<Mutex<HashMap<usize, Py<PyTuple>>>> = LazyLock::new(Default::default);
    let names = audio_math::key_to_notes_static(key, unicode, natural).map_err(audio_err_to_py)?;
    let id = std::ptr::from_ref(names) as usize;
    let cached = TUPLES
        .lock_py_attached(py)
        .unwrap_or_else(PoisonError::into_inner)
        .get(&id)
        .map(|t| t.clone_ref(py));
    let tuple = if let Some(t) = cached {
        t.into_bound(py)
    } else {
        let t = PyTuple::new(py, names.iter().map(|n| PyString::intern(py, n)))?;
        TUPLES
            .lock_py_attached(py)
            .unwrap_or_else(PoisonError::into_inner)
            .entry(id)
            .or_insert_with(|| t.clone().unbind());
        t
    };
    let list: Bound<'py, PyList> = tuple.as_sequence().to_list()?;
    Ok(list)
}

/// Pitch classes (0 = C) of the 7 scale degrees of a key.
///
/// Args:
///
///     key (str): A key such as ``"C:maj"`` or ``"A:min"``.
///
/// Returns:
///
///     numpy.ndarray[int64]: shape ``(7,)``.
///
/// Raises:
///
///     ValueError: For an unparsable key.
#[pyfunction]
#[pyo3(signature = (key), text_signature = "(key)")]
pub fn key_to_degrees<'py>(py: Python<'py>, key: &str) -> PyResult<Bound<'py, PyArray1<i64>>> {
    use numpy::PyArrayMethods;
    let d = audio_math::key_to_degrees(key).map_err(audio_err_to_py)?;
    // Written straight into a NumPy-allocated array: no Rust vector and no
    // owner object per call.
    let out = PyArray1::<i64>::zeros(py, d.len(), false);
    // SAFETY: the array was just created and is not shared yet.
    let slice = unsafe { out.as_slice_mut() }.expect("a new array is contiguous");
    for (o, &x) in slice.iter_mut().zip(d.iter()) {
        *o = i64::from(x);
    }
    Ok(out)
}

/// Registers the tuning helpers on `audio_samples.utils`.
pub fn add(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(hz_to_octs, m)?)?;
    m.add_function(wrap_pyfunction!(octs_to_hz, m)?)?;
    m.add_function(wrap_pyfunction!(a4_to_tuning, m)?)?;
    m.add_function(wrap_pyfunction!(tuning_to_a4, m)?)?;
    m.add_function(wrap_pyfunction!(cqt_frequencies, m)?)?;
    m.add_function(wrap_pyfunction!(key_to_notes, m)?)?;
    m.add_function(wrap_pyfunction!(key_to_degrees, m)?)?;
    Ok(())
}
