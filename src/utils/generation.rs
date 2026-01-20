use std::time::Duration;

use audio_samples::I24;
use numpy::{PyArrayDescr, PyArrayDescrMethods};
use pyo3::{
    exceptions::{PyTypeError, PyValueError},
    prelude::*,
};

use crate::{PyAudioSamples, nzu32_or_err, reexport};

#[derive(Debug, Clone, Copy)]
pub enum Generator {
    Sine { frequency: f64 },
    Cosine { frequency: f64 },
    Sawtooth { frequency: f64 },
    Square { frequency: f64 },
    Triangle { frequency: f64 },
    Chirp { f0: f64, f1: f64 },
    WhiteNoise { seed: Option<u64> },
    PinkNoise { seed: Option<u64> },
    BrownNoise { step: f64, seed: Option<u64> },
    Impulse { position: f64 },
    Silence,
}

macro_rules! dispatch_audio {
    ($py:expr, $dtype:expr, |$T:ident| $body:expr) => {{
        let dtype = $dtype.unwrap_or_else(|| numpy::dtype::<f32>($py));

        if dtype.is_equiv_to(&numpy::dtype::<i16>($py)) {
            type $T = i16;
            $body
        } else if dtype.is_equiv_to(&numpy::dtype::<I24>($py)) {
            type $T = I24;
            $body
        } else if dtype.is_equiv_to(&numpy::dtype::<i32>($py)) {
            type $T = i32;
            $body
        } else if dtype.is_equiv_to(&numpy::dtype::<f32>($py)) {
            type $T = f32;
            $body
        } else if dtype.is_equiv_to(&numpy::dtype::<f64>($py)) {
            type $T = f64;
            $body
        } else {
            Err(PyTypeError::new_err("Unsupported dtype"))
        }
    }};
}

fn generate_audio(
    py: Python<'_>,
    generator: Generator,
    duration_secs: f64,
    sample_rate: u32,
    amplitude: f64,
    dtype: Option<Bound<'_, PyArrayDescr>>,
) -> PyResult<PyAudioSamples> {
    if duration_secs <= 0.0 {
        return Err(PyValueError::new_err("duration_secs must be positive"));
    }
    let sample_rate = nzu32_or_err(sample_rate)?;

    let duration = Duration::from_secs_f64(duration_secs);

    dispatch_audio!(py, dtype, |T| {
        let audio = match generator {
            Generator::Sine { frequency } => {
                audio_samples::sine_wave::<T>(frequency, duration, sample_rate, amplitude)
            }
            Generator::Cosine { frequency } => {
                audio_samples::cosine_wave::<T>(frequency, duration, sample_rate, amplitude)
            }
            Generator::Sawtooth { frequency } => {
                audio_samples::sawtooth_wave::<T>(frequency, duration, sample_rate, amplitude)
            }
            Generator::Square { frequency } => {
                audio_samples::square_wave::<T>(frequency, duration, sample_rate, amplitude)
            }
            Generator::Triangle { frequency } => {
                audio_samples::triangle_wave::<T>(frequency, duration, sample_rate, amplitude)
            }
            Generator::Chirp { f0, f1 } => {
                audio_samples::chirp::<T>(f0, f1, duration, sample_rate, amplitude)
            }
            Generator::WhiteNoise { seed } => {
                audio_samples::white_noise::<T>(duration, sample_rate, amplitude, seed)
            }
            Generator::PinkNoise { seed } => {
                audio_samples::pink_noise::<T>(duration, sample_rate, amplitude, seed)
            }
            Generator::BrownNoise { step, seed } => {
                audio_samples::brown_noise::<T>(duration, sample_rate, step, amplitude, seed)
            }
            Generator::Impulse { position } => {
                audio_samples::impulse::<T>(duration, sample_rate, amplitude, position)
            }
            Generator::Silence => audio_samples::silence::<T>(duration, sample_rate),
        };

        Ok(PyAudioSamples::from_audio_samples(audio))
    })
}

#[pymodule]
pub fn generation(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    let generation_mod = PyModule::new(py, "generation")?;

    // Mono generation functions
    generation_mod.add_function(wrap_pyfunction!(sine_wave, &generation_mod)?)?;
    generation_mod.add_function(wrap_pyfunction!(cosine_wave, &generation_mod)?)?;
    generation_mod.add_function(wrap_pyfunction!(sawtooth_wave, &generation_mod)?)?;
    generation_mod.add_function(wrap_pyfunction!(triangle_wave, &generation_mod)?)?;
    generation_mod.add_function(wrap_pyfunction!(square_wave, &generation_mod)?)?;
    generation_mod.add_function(wrap_pyfunction!(white_noise, &generation_mod)?)?;
    generation_mod.add_function(wrap_pyfunction!(pink_noise, &generation_mod)?)?;
    generation_mod.add_function(wrap_pyfunction!(brown_noise, &generation_mod)?)?;
    generation_mod.add_function(wrap_pyfunction!(impulse, &generation_mod)?)?;
    generation_mod.add_function(wrap_pyfunction!(silence, &generation_mod)?)?;
    generation_mod.add_function(wrap_pyfunction!(chirp, &generation_mod)?)?;

    // Stereo generation functions
    generation_mod.add_function(wrap_pyfunction!(stereo_sine_wave, &generation_mod)?)?;
    generation_mod.add_function(wrap_pyfunction!(stereo_chirp, &generation_mod)?)?;
    generation_mod.add_function(wrap_pyfunction!(stereo_silence, &generation_mod)?)?;

    reexport!(
        m,
        generation_mod,
        "sine_wave",
        "cosine_wave",
        "sawtooth_wave",
        "triangle_wave",
        "square_wave",
        "white_noise",
        "pink_noise",
        "brown_noise",
        "impulse",
        "silence",
        "chirp",
        "stereo_sine_wave",
        "stereo_chirp",
        "stereo_silence"
    );
    m.add_submodule(&generation_mod)?;
    Ok(())
}

/// Generate a sine wave audio signal.
///
/// Args:
///     frequency: Frequency of the sine wave in Hz
///     duration_secs: Duration of the signal in seconds
///     sample_rate: Sample rate in samples per second (default: 44100)
///     amplitude: Peak amplitude of the wave (default: 1.0)
///     dtype: NumPy dtype for the output array (default: f64)
///
/// Returns:
///     PyAudioSamples: Generated sine wave audio data
#[pyfunction]
#[pyo3(signature = (frequency, duration_secs, sample_rate=44100, amplitude=1.0, dtype=None), text_signature = "(frequency: float, duration_secs: float, sample_rate: int = 44100, amplitude: float = 1.0, dtype: Optional[numpy.dtype] = None) -> PyAudioSamples")]
pub fn sine_wave(
    py: Python<'_>,
    frequency: f64,
    duration_secs: f64,
    sample_rate: u32,
    amplitude: f64,
    dtype: Option<Bound<'_, PyArrayDescr>>,
) -> PyResult<PyAudioSamples> {
    generate_audio(
        py,
        Generator::Sine { frequency },
        duration_secs,
        sample_rate,
        amplitude,
        dtype,
    )
}

/// Generate a cosine wave audio signal.
///
/// Args:
///     frequency: Frequency of the cosine wave in Hz
///     duration_secs: Duration of the signal in seconds
///     sample_rate: Sample rate in samples per second (default: 44100)
///     amplitude: Peak amplitude of the wave (default: 1.0)
///     dtype: NumPy dtype for the output array (default: f64)
///
/// Returns:
///     PyAudioSamples: Generated cosine wave audio data
#[pyfunction]
#[pyo3(signature = (frequency, duration_secs, sample_rate=44100, amplitude=1.0, dtype=None), text_signature = "(frequency: float, duration_secs: float, sample_rate: int = 44100, amplitude: float = 1.0, dtype: Optional[numpy.dtype] = None) -> PyAudioSamples")]
fn cosine_wave(
    py: Python<'_>,
    frequency: f64,
    duration_secs: f64,
    sample_rate: u32,
    amplitude: f64,
    dtype: Option<Bound<'_, PyArrayDescr>>,
) -> PyResult<PyAudioSamples> {
    generate_audio(
        py,
        Generator::Cosine { frequency },
        duration_secs,
        sample_rate,
        amplitude,
        dtype,
    )
}

/// Generate a sawtooth wave audio signal.
///
/// Args:
///     frequency: Frequency of the sawtooth wave in Hz
///     duration_secs: Duration of the signal in seconds
///     sample_rate: Sample rate in samples per second (default: 44100)
///     amplitude: Peak amplitude of the wave (default: 1.0)
///     dtype: NumPy dtype for the output array (default: f64)
///
/// Returns:
///     PyAudioSamples: Generated sawtooth wave audio data
#[pyfunction]
#[pyo3(signature = (frequency, duration_secs, sample_rate=44100, amplitude=1.0, dtype=None), text_signature = "(frequency: float, duration_secs: float, sample_rate: int = 44100, amplitude: float = 1.0, dtype: Optional[numpy.dtype] = None) -> PyAudioSamples")]
fn sawtooth_wave(
    py: Python<'_>,
    frequency: f64,
    duration_secs: f64,
    sample_rate: u32,
    amplitude: f64,
    dtype: Option<Bound<'_, PyArrayDescr>>,
) -> PyResult<PyAudioSamples> {
    generate_audio(
        py,
        Generator::Sawtooth { frequency },
        duration_secs,
        sample_rate,
        amplitude,
        dtype,
    )
}

/// Generate a square wave audio signal.
///
/// Args:
///     frequency: Frequency of the square wave in Hz
///     duration_secs: Duration of the signal in seconds
///     sample_rate: Sample rate in samples per second (default: 44100)
///     amplitude: Peak amplitude of the wave (default: 1.0)
///     dtype: NumPy dtype for the output array (default: f64)
///
/// Returns:
///     PyAudioSamples: Generated square wave audio data
#[pyfunction]
#[pyo3(signature = (frequency, duration_secs, sample_rate=44100, amplitude=1.0, dtype=None), text_signature = "(frequency: float, duration_secs: float, sample_rate: int = 44100, amplitude: float = 1.0, dtype: Optional[numpy.dtype] = None) -> PyAudioSamples")]
fn square_wave(
    py: Python<'_>,
    frequency: f64,
    duration_secs: f64,
    sample_rate: u32,
    amplitude: f64,
    dtype: Option<Bound<'_, PyArrayDescr>>,
) -> PyResult<PyAudioSamples> {
    generate_audio(
        py,
        Generator::Square { frequency },
        duration_secs,
        sample_rate,
        amplitude,
        dtype,
    )
}

/// Generate a triangle wave audio signal.
///
/// Args:
///     frequency: Frequency of the triangle wave in Hz
///     duration_secs: Duration of the signal in seconds
///     sample_rate: Sample rate in samples per second (default: 44100)
///     amplitude: Peak amplitude of the wave (default: 1.0)
///     dtype: NumPy dtype for the output array (default: f64)
///
/// Returns:
///     PyAudioSamples: Generated triangle wave audio data
#[pyfunction]
#[pyo3(signature = (frequency, duration_secs, sample_rate=44100, amplitude=1.0, dtype=None), text_signature = "(frequency: float, duration_secs: float, sample_rate: int = 44100, amplitude: float = 1.0, dtype: Optional[numpy.dtype] = None) -> PyAudioSamples")]
fn triangle_wave(
    py: Python<'_>,
    frequency: f64,
    duration_secs: f64,
    sample_rate: u32,
    amplitude: f64,
    dtype: Option<Bound<'_, PyArrayDescr>>,
) -> PyResult<PyAudioSamples> {
    generate_audio(
        py,
        Generator::Triangle { frequency },
        duration_secs,
        sample_rate,
        amplitude,
        dtype,
    )
}

/// Generate a frequency chirp (sweep) audio signal.
///
/// Args:
///     f0: Starting frequency in Hz
///     f1: Ending frequency in Hz
///     duration_secs: Duration of the signal in seconds
///     sample_rate: Sample rate in samples per second (default: 44100)
///     amplitude: Peak amplitude of the wave (default: 1.0)
///     dtype: NumPy dtype for the output array (default: f64)
///
/// Returns:
///     PyAudioSamples: Generated chirp audio data
#[pyfunction]
#[pyo3(signature = (f0, f1, duration_secs, sample_rate=44100, amplitude=1.0, dtype=None), text_signature = "(f0: float, f1: float, duration_secs: float, sample_rate: int = 44100, amplitude: float = 1.0, dtype: Optional[numpy.dtype] = None) -> PyAudioSamples")]
fn chirp(
    py: Python<'_>,
    f0: f64,
    f1: f64,
    duration_secs: f64,
    sample_rate: u32,
    amplitude: f64,
    dtype: Option<Bound<'_, PyArrayDescr>>,
) -> PyResult<PyAudioSamples> {
    generate_audio(
        py,
        Generator::Chirp { f0, f1 },
        duration_secs,
        sample_rate,
        amplitude,
        dtype,
    )
}

/// Generate white noise audio signal.
///
/// Args:
///     duration_secs: Duration of the signal in seconds
///     sample_rate: Sample rate in samples per second (default: 44100)
///     amplitude: Peak amplitude of the noise (default: 1.0)
///     dtype: NumPy dtype for the output array (default: f64)
///     seed: Optional seed for reproducible noise generation
///
/// Returns:
///     PyAudioSamples: Generated white noise audio data
#[pyfunction]
#[pyo3(signature = (duration_secs, sample_rate=44100, amplitude=1.0, dtype=None, seed=None), text_signature = "(duration_secs: float, sample_rate: int = 44100, amplitude: float = 1.0, dtype: Optional[numpy.dtype] = None, seed: Optional[int] = None) -> PyAudioSamples")]
fn white_noise(
    py: Python<'_>,
    duration_secs: f64,
    sample_rate: u32,
    amplitude: f64,
    dtype: Option<Bound<'_, PyArrayDescr>>,
    seed: Option<u64>,
) -> PyResult<PyAudioSamples> {
    generate_audio(
        py,
        Generator::WhiteNoise { seed },
        duration_secs,
        sample_rate,
        amplitude,
        dtype,
    )
}

/// Generate pink noise audio signal.
///
/// Args:
///     duration_secs: Duration of the signal in seconds
///     sample_rate: Sample rate in samples per second (default: 44100)
///     amplitude: Peak amplitude of the noise (default: 1.0)
///     dtype: NumPy dtype for the output array (default: f64)
///
/// Returns:
///     PyAudioSamples: Generated pink noise audio data
#[pyfunction]
#[pyo3(signature = (duration_secs, sample_rate=44100, amplitude=1.0, dtype=None,seed=None), text_signature = "(duration_secs: float, sample_rate: int = 44100, amplitude: float = 1.0, dtype: Optional[numpy.dtype] = None, seed: Optional[int] = None) -> PyAudioSamples")]
fn pink_noise(
    py: Python<'_>,
    duration_secs: f64,
    sample_rate: u32,
    amplitude: f64,
    dtype: Option<Bound<'_, PyArrayDescr>>,
    seed: Option<u64>,
) -> PyResult<PyAudioSamples> {
    generate_audio(
        py,
        Generator::PinkNoise { seed },
        duration_secs,
        sample_rate,
        amplitude,
        dtype,
    )
}

/// Generate brown noise (Brownian/red noise) audio signal.
///
/// Args:
///     duration_secs: Duration of the signal in seconds
///     sample_rate: Sample rate in samples per second (default: 44100)
///     step: Step size for the random walk (default: 0.01)
///     amplitude: Peak amplitude of the noise (default: 1.0)
///     dtype: NumPy dtype for the output array (default: f64)
///
/// Returns:
///     PyAudioSamples: Generated brown noise audio data
#[pyfunction]
#[pyo3(signature = (duration_secs, sample_rate=44100, step=0.01, amplitude=1.0, dtype=None, seed=None), text_signature = "(duration_secs: float, sample_rate: int = 44100, step: float = 0.01, amplitude: float = 1.0, dtype: Optional[numpy.dtype] = None, seed: Optional[int] = None) -> PyAudioSamples")]
fn brown_noise(
    py: Python<'_>,
    duration_secs: f64,
    sample_rate: u32,
    step: f64,
    amplitude: f64,
    dtype: Option<Bound<'_, PyArrayDescr>>,
    seed: Option<u64>,
) -> PyResult<PyAudioSamples> {
    generate_audio(
        py,
        Generator::BrownNoise { step, seed },
        duration_secs,
        sample_rate,
        amplitude,
        dtype,
    )
}

/// Generate an impulse (delta function) audio signal.
///
/// Args:
///     duration_secs: Duration of the signal in seconds
///     sample_rate: Sample rate in samples per second (default: 44100)
///     amplitude: Peak amplitude of the impulse (default: 1.0)
///     position: Position of the impulse as fraction of duration (default: 0.5)
///     dtype: NumPy dtype for the output array (default: f64)
///
/// Returns:
///     PyAudioSamples: Generated impulse audio data
#[pyfunction]
#[pyo3(signature = (duration_secs, sample_rate=44100, amplitude=1.0, position=0.5, dtype=None), text_signature = "(duration_secs: float, sample_rate: int = 44100, amplitude: float = 1.0, position: float = 0.5, dtype: Optional[numpy.dtype] = None) -> PyAudioSamples")]
fn impulse(
    py: Python<'_>,
    duration_secs: f64,
    sample_rate: u32,
    amplitude: f64,
    position: f64,
    dtype: Option<Bound<'_, PyArrayDescr>>,
) -> PyResult<PyAudioSamples> {
    generate_audio(
        py,
        Generator::Impulse { position },
        duration_secs,
        sample_rate,
        amplitude,
        dtype,
    )
}

/// Generate silence (zero amplitude) audio signal.
///
/// Args:
///     duration_secs: Duration of the signal in seconds
///     sample_rate: Sample rate in samples per second (default: 44100)
///     dtype: NumPy dtype for the output array (default: f64)
///
/// Returns:
///     PyAudioSamples: Generated silence audio data
#[pyfunction]
#[pyo3(signature = (duration_secs, sample_rate=44100, dtype=None), text_signature = "(duration_secs: float, sample_rate: int = 44100, dtype: Optional[numpy.dtype] = None) -> PyAudioSamples")]
fn silence(
    py: Python<'_>,
    duration_secs: f64,
    sample_rate: u32,
    dtype: Option<Bound<'_, PyArrayDescr>>,
) -> PyResult<PyAudioSamples> {
    generate_audio(
        py,
        Generator::Silence,
        duration_secs,
        sample_rate,
        0.0,
        dtype,
    )
}

/// Generate a stereo sine wave audio signal (same frequency on both channels).
///
/// Args:
///     frequency: Frequency of the sine wave in Hz
///     duration_secs: Duration of the signal in seconds
///     sample_rate: Sample rate in samples per second (default: 44100)
///     amplitude: Peak amplitude of the wave (default: 1.0)
///     dtype: NumPy dtype for the output array (default: f64)
///
/// Returns:
///     PyAudioSamples: Generated stereo sine wave audio data
#[pyfunction]
#[pyo3(signature = (frequency, duration_secs, sample_rate=44100, amplitude=1.0, dtype=None), text_signature = "(frequency: float, duration_secs: float, sample_rate: int = 44100, amplitude: float = 1.0, dtype: Optional[numpy.dtype] = None) -> PyAudioSamples")]
pub fn stereo_sine_wave(
    py: Python<'_>,
    frequency: f64,
    duration_secs: f64,
    sample_rate: u32,
    amplitude: f64,
    dtype: Option<Bound<'_, PyArrayDescr>>,
) -> PyResult<PyAudioSamples> {
    if duration_secs <= 0.0 {
        return Err(PyValueError::new_err("duration_secs must be positive"));
    }
    let sample_rate = nzu32_or_err(sample_rate)?;
    let duration = Duration::from_secs_f64(duration_secs);

    dispatch_audio!(py, dtype, |T| {
        let audio =
            audio_samples::stereo_sine_wave::<T>(frequency, duration, sample_rate, amplitude);
        Ok(PyAudioSamples::from_audio_samples(audio))
    })
}

/// Generate a stereo chirp signal (frequency sweep on both channels).
///
/// Args:
///     start_freq: Starting frequency in Hz
///     end_freq: Ending frequency in Hz
///     duration_secs: Duration of the signal in seconds
///     sample_rate: Sample rate in samples per second (default: 44100)
///     amplitude: Peak amplitude of the chirp (default: 1.0)
///     dtype: NumPy dtype for the output array (default: f64)
///
/// Returns:
///     PyAudioSamples: Generated stereo chirp audio data
#[pyfunction]
#[pyo3(signature = (start_freq, end_freq, duration_secs, sample_rate=44100, amplitude=1.0, dtype=None), text_signature = "(start_freq: float, end_freq: float, duration_secs: float, sample_rate: int = 44100, amplitude: float = 1.0, dtype: Optional[numpy.dtype] = None) -> PyAudioSamples")]
pub fn stereo_chirp(
    py: Python<'_>,
    start_freq: f64,
    end_freq: f64,
    duration_secs: f64,
    sample_rate: u32,
    amplitude: f64,
    dtype: Option<Bound<'_, PyArrayDescr>>,
) -> PyResult<PyAudioSamples> {
    if duration_secs <= 0.0 {
        return Err(PyValueError::new_err("duration_secs must be positive"));
    }
    let sample_rate = nzu32_or_err(sample_rate)?;
    let duration = Duration::from_secs_f64(duration_secs);

    dispatch_audio!(py, dtype, |T| {
        let audio = audio_samples::stereo_chirp::<T>(
            start_freq,
            end_freq,
            duration,
            sample_rate,
            amplitude,
        );
        Ok(PyAudioSamples::from_audio_samples(audio))
    })
}

/// Generate stereo silence.
///
/// Args:
///     duration_secs: Duration of the signal in seconds
///     sample_rate: Sample rate in samples per second (default: 44100)
///     dtype: NumPy dtype for the output array (default: f64)
///
/// Returns:
///     PyAudioSamples: Stereo silence audio data
#[pyfunction]
#[pyo3(signature = (duration_secs, sample_rate=44100, dtype=None), text_signature = "(duration_secs: float, sample_rate: int = 44100, dtype: Optional[numpy.dtype] = None) -> PyAudioSamples")]
pub fn stereo_silence(
    py: Python<'_>,
    duration_secs: f64,
    sample_rate: u32,
    dtype: Option<Bound<'_, PyArrayDescr>>,
) -> PyResult<PyAudioSamples> {
    if duration_secs <= 0.0 {
        return Err(PyValueError::new_err("duration_secs must be positive"));
    }
    let sample_rate = nzu32_or_err(sample_rate)?;
    let duration = Duration::from_secs_f64(duration_secs);

    dispatch_audio!(py, dtype, |T| {
        let audio = audio_samples::stereo_silence::<T>(duration, sample_rate);
        Ok(PyAudioSamples::from_audio_samples(audio))
    })
}
