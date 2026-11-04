use std::time::Duration;

use audio_samples::I24;
use audio_samples::utils::generation::{ExampleClip, ToneComponent};
use non_empty_slice::NonEmptyVec;
use numpy::{PyArray1, PyArrayDescr, PyArrayDescrMethods};
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
    ExponentialChirp { f0: f64, f1: f64 },
    SquareBandlimited { frequency: f64 },
    SawtoothBandlimited { frequency: f64 },
    TriangleBandlimited { frequency: f64 },
    WhiteNoise { seed: Option<u64> },
    PinkNoise { seed: Option<u64> },
    BrownNoise { step: f64, seed: Option<u64> },
    Impulse { position: f64 },
    Silence,
}

macro_rules! dispatch_audio {
    ($py:expr, $dtype:expr, |$T:ident| $body:expr) => {{
        // Accept anything `numpy.dtype(...)` accepts: a dtype instance, a
        // scalar type such as `np.float64`, or a string such as "float64".
        let dtype: Bound<'_, PyArrayDescr> = match $dtype {
            Some(d) => PyArrayDescr::new($py, &d)?,
            None => numpy::dtype::<f32>($py),
        };

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
            Err(PyTypeError::new_err(format!(
                "Unsupported dtype {dtype}; generators support int16, int24, int32, float32 and float64"
            )))
        }
    }};
}

fn generate_audio(
    py: Python<'_>,
    generator: Generator,
    duration_secs: f64,
    sample_rate: u32,
    amplitude: f64,
    dtype: Option<Bound<'_, PyAny>>,
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
            Generator::ExponentialChirp { f0, f1 } => {
                audio_samples::exponential_chirp::<T>(f0, f1, duration, sample_rate, amplitude)
            }
            Generator::SquareBandlimited { frequency } => {
                audio_samples::square_wave_bandlimited::<T>(
                    frequency,
                    duration,
                    sample_rate,
                    amplitude,
                )
            }
            Generator::SawtoothBandlimited { frequency } => {
                audio_samples::sawtooth_wave_bandlimited::<T>(
                    frequency,
                    duration,
                    sample_rate,
                    amplitude,
                )
            }
            Generator::TriangleBandlimited { frequency } => {
                audio_samples::triangle_wave_bandlimited::<T>(
                    frequency,
                    duration,
                    sample_rate,
                    amplitude,
                )
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

    // Band-limited / modulated / compound generators (New in 2.0)
    generation_mod.add_function(wrap_pyfunction!(square_wave_bandlimited, &generation_mod)?)?;
    generation_mod.add_function(wrap_pyfunction!(
        sawtooth_wave_bandlimited,
        &generation_mod
    )?)?;
    generation_mod.add_function(wrap_pyfunction!(
        triangle_wave_bandlimited,
        &generation_mod
    )?)?;
    generation_mod.add_function(wrap_pyfunction!(exponential_chirp, &generation_mod)?)?;
    generation_mod.add_function(wrap_pyfunction!(fm_signal, &generation_mod)?)?;
    generation_mod.add_function(wrap_pyfunction!(am_signal, &generation_mod)?)?;
    generation_mod.add_function(wrap_pyfunction!(compound_tone, &generation_mod)?)?;
    generation_mod.add_function(wrap_pyfunction!(exponential_bursts, &generation_mod)?)?;

    // Stereo generation functions
    generation_mod.add_function(wrap_pyfunction!(stereo_sine_wave, &generation_mod)?)?;
    generation_mod.add_function(wrap_pyfunction!(stereo_chirp, &generation_mod)?)?;
    generation_mod.add_function(wrap_pyfunction!(stereo_silence, &generation_mod)?)?;

    // Example clips
    generation_mod.add_function(wrap_pyfunction!(example_clip, &generation_mod)?)?;
    generation_mod.add_function(wrap_pyfunction!(example_clip_names, &generation_mod)?)?;
    generation_mod.add_function(wrap_pyfunction!(example_speech_f0, &generation_mod)?)?;
    generation_mod.add("EXAMPLE_TEMPO_BPM", audio_samples::utils::generation::EXAMPLE_TEMPO_BPM)?;
    generation_mod.add(
        "EXAMPLE_NOISY_SNR_DB",
        audio_samples::utils::generation::EXAMPLE_NOISY_SNR_DB,
    )?;

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
        "square_wave_bandlimited",
        "sawtooth_wave_bandlimited",
        "triangle_wave_bandlimited",
        "exponential_chirp",
        "fm_signal",
        "am_signal",
        "compound_tone",
        "exponential_bursts",
        "stereo_sine_wave",
        "stereo_chirp",
        "stereo_silence"
    );
    m.add_submodule(&generation_mod)?;
    Ok(())
}

/// Render a named example clip of realistic audio.
///
/// The clip is synthesised on each call: nothing is stored or downloaded. The same arguments
/// give the same samples on any platform, and a shorter clip is a prefix of a longer one.
///
/// ======================  ==================================================================
/// Name                    Contents
/// ======================  ==================================================================
/// ``"chords"``            C, G, Am, F in equal temperament (A4 = 440 Hz), one chord per four
///                         beats at ``EXAMPLE_TEMPO_BPM``, soft attacks
/// ``"drum_loop"``         Kick on beats 1 and 3, snare on 2 and 4, hi-hat on every eighth
///                         note at ``EXAMPLE_TEMPO_BPM``, first kick at time zero
/// ``"chords_and_drums"``  ``0.5 * chords + 0.5 * drum_loop``
/// ``"speech"``            Voiced speech-like signal with formants, following
///                         ``example_speech_f0``; a 2.4 s phrase every 3 s, pauses between
/// ``"noisy_recording"``   ``speech`` with low-passed noise and 50 Hz mains hum, at
///                         ``EXAMPLE_NOISY_SNR_DB`` over the voiced parts
/// ======================  ==================================================================
///
/// Args:
///     name: The clip, one of the names above (see ``example_clip_names``)
///     duration_secs: Duration in seconds (default: 8.0); the pattern repeats for as long as
///         requested
///     sample_rate: Sample rate in samples per second (default: 44100)
///     dtype: Output sample type as a NumPy dtype, scalar type (``np.float64``)
///         or string (``"float64"``). Default: float32
///
/// Returns:
///     AudioSamples: Mono audio of the given sample type at the given sample rate
///
/// Raises:
///     ValueError: Unknown name, or a duration that gives no samples
#[pyfunction]
#[pyo3(signature = (name, duration_secs=8.0, sample_rate=44100, dtype=None))]
fn example_clip(
    py: Python<'_>,
    name: &str,
    duration_secs: f64,
    sample_rate: u32,
    dtype: Option<Bound<'_, PyAny>>,
) -> PyResult<PyAudioSamples> {
    let clip = ExampleClip::from_name(name).ok_or_else(|| {
        let names = ExampleClip::ALL.map(ExampleClip::name).join(", ");
        PyValueError::new_err(format!("name: unknown example clip {name:?}; the clips are {names}"))
    })?;
    let sample_rate = nzu32_or_err(sample_rate)?;
    if !duration_secs.is_finite() || duration_secs * f64::from(sample_rate.get()) < 1.0 {
        return Err(PyValueError::new_err(format!(
            "duration_secs: must give at least one sample at {sample_rate} Hz, got {duration_secs}"
        )));
    }
    let duration = Duration::from_secs_f64(duration_secs);
    dispatch_audio!(py, dtype, |T| {
        let audio = audio_samples::example_clip::<T>(clip, duration, sample_rate);
        Ok(PyAudioSamples::from_audio_samples(audio))
    })
}

/// The names ``example_clip`` accepts.
///
/// Returns:
///     list[str]: The clip names
#[pyfunction]
fn example_clip_names() -> Vec<&'static str> {
    ExampleClip::ALL.map(ExampleClip::name).to_vec()
}

/// The F0 of the ``"speech"`` and ``"noisy_recording"`` clips at the given times.
///
/// Args:
///     times: Times in seconds from the start of the clip
///
/// Returns:
///     numpy.ndarray: F0 in Hz as 1-D float64, NaN during pauses
#[pyfunction]
fn example_speech_f0<'py>(py: Python<'py>, times: Vec<f64>) -> Bound<'py, PyArray1<f64>> {
    let f0: Vec<f64> = times
        .into_iter()
        .map(|t| audio_samples::utils::generation::example_speech_f0(t).unwrap_or(f64::NAN))
        .collect();
    PyArray1::from_vec(py, f0)
}

/// Generate a sine wave audio signal.
///
/// Args:
///     frequency: Frequency of the sine wave in Hz
///     duration_secs: Duration of the signal in seconds
///     sample_rate: Sample rate in samples per second (default: 44100)
///     amplitude: Peak amplitude of the wave (default: 1.0)
///     dtype: Output sample type as a NumPy dtype, scalar type (``np.float64``)
///         or string (``"float64"``). Default: float32
///
/// Returns:
///     AudioSamples: Generated sine wave audio data
#[pyfunction]
#[pyo3(signature = (frequency, duration_secs, sample_rate=44100, amplitude=1.0, dtype=None))]
pub fn sine_wave(
    py: Python<'_>,
    frequency: f64,
    duration_secs: f64,
    sample_rate: u32,
    amplitude: f64,
    dtype: Option<Bound<'_, PyAny>>,
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
///     dtype: Output sample type as a NumPy dtype, scalar type (``np.float64``)
///         or string (``"float64"``). Default: float32
///
/// Returns:
///     AudioSamples: Generated cosine wave audio data
#[pyfunction]
#[pyo3(signature = (frequency, duration_secs, sample_rate=44100, amplitude=1.0, dtype=None))]
fn cosine_wave(
    py: Python<'_>,
    frequency: f64,
    duration_secs: f64,
    sample_rate: u32,
    amplitude: f64,
    dtype: Option<Bound<'_, PyAny>>,
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
///     dtype: Output sample type as a NumPy dtype, scalar type (``np.float64``)
///         or string (``"float64"``). Default: float32
///
/// Returns:
///     AudioSamples: Generated sawtooth wave audio data
#[pyfunction]
#[pyo3(signature = (frequency, duration_secs, sample_rate=44100, amplitude=1.0, dtype=None))]
fn sawtooth_wave(
    py: Python<'_>,
    frequency: f64,
    duration_secs: f64,
    sample_rate: u32,
    amplitude: f64,
    dtype: Option<Bound<'_, PyAny>>,
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
///     dtype: Output sample type as a NumPy dtype, scalar type (``np.float64``)
///         or string (``"float64"``). Default: float32
///
/// Returns:
///     AudioSamples: Generated square wave audio data
#[pyfunction]
#[pyo3(signature = (frequency, duration_secs, sample_rate=44100, amplitude=1.0, dtype=None))]
fn square_wave(
    py: Python<'_>,
    frequency: f64,
    duration_secs: f64,
    sample_rate: u32,
    amplitude: f64,
    dtype: Option<Bound<'_, PyAny>>,
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
///     dtype: Output sample type as a NumPy dtype, scalar type (``np.float64``)
///         or string (``"float64"``). Default: float32
///
/// Returns:
///     AudioSamples: Generated triangle wave audio data
#[pyfunction]
#[pyo3(signature = (frequency, duration_secs, sample_rate=44100, amplitude=1.0, dtype=None))]
fn triangle_wave(
    py: Python<'_>,
    frequency: f64,
    duration_secs: f64,
    sample_rate: u32,
    amplitude: f64,
    dtype: Option<Bound<'_, PyAny>>,
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
///     dtype: Output sample type as a NumPy dtype, scalar type (``np.float64``)
///         or string (``"float64"``). Default: float32
///
/// Returns:
///     AudioSamples: Generated chirp audio data
#[pyfunction]
#[pyo3(signature = (f0, f1, duration_secs, sample_rate=44100, amplitude=1.0, dtype=None))]
fn chirp(
    py: Python<'_>,
    f0: f64,
    f1: f64,
    duration_secs: f64,
    sample_rate: u32,
    amplitude: f64,
    dtype: Option<Bound<'_, PyAny>>,
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
///     dtype: Output sample type as a NumPy dtype, scalar type (``np.float64``)
///         or string (``"float64"``). Default: float32
///     seed: Optional seed for reproducible noise generation
///
/// Returns:
///     AudioSamples: Generated white noise audio data
#[pyfunction]
#[pyo3(signature = (duration_secs, sample_rate=44100, amplitude=1.0, dtype=None, seed=None))]
fn white_noise(
    py: Python<'_>,
    duration_secs: f64,
    sample_rate: u32,
    amplitude: f64,
    dtype: Option<Bound<'_, PyAny>>,
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
///     dtype: Output sample type as a NumPy dtype, scalar type (``np.float64``)
///         or string (``"float64"``). Default: float32
///
/// Returns:
///     AudioSamples: Generated pink noise audio data
#[pyfunction]
#[pyo3(signature = (duration_secs, sample_rate=44100, amplitude=1.0, dtype=None,seed=None))]
fn pink_noise(
    py: Python<'_>,
    duration_secs: f64,
    sample_rate: u32,
    amplitude: f64,
    dtype: Option<Bound<'_, PyAny>>,
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
///     dtype: Output sample type as a NumPy dtype, scalar type (``np.float64``)
///         or string (``"float64"``). Default: float32
///
/// Returns:
///     AudioSamples: Generated brown noise audio data
#[pyfunction]
#[pyo3(signature = (duration_secs, sample_rate=44100, step=0.01, amplitude=1.0, dtype=None, seed=None))]
fn brown_noise(
    py: Python<'_>,
    duration_secs: f64,
    sample_rate: u32,
    step: f64,
    amplitude: f64,
    dtype: Option<Bound<'_, PyAny>>,
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
///     dtype: Output sample type as a NumPy dtype, scalar type (``np.float64``)
///         or string (``"float64"``). Default: float32
///
/// Returns:
///     AudioSamples: Generated impulse audio data
#[pyfunction]
#[pyo3(signature = (duration_secs, sample_rate=44100, amplitude=1.0, position=0.5, dtype=None))]
fn impulse(
    py: Python<'_>,
    duration_secs: f64,
    sample_rate: u32,
    amplitude: f64,
    position: f64,
    dtype: Option<Bound<'_, PyAny>>,
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
///     dtype: Output sample type as a NumPy dtype, scalar type (``np.float64``)
///         or string (``"float64"``). Default: float32
///
/// Returns:
///     AudioSamples: Generated silence audio data
#[pyfunction]
#[pyo3(signature = (duration_secs, sample_rate=44100, dtype=None))]
fn silence(
    py: Python<'_>,
    duration_secs: f64,
    sample_rate: u32,
    dtype: Option<Bound<'_, PyAny>>,
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
///     dtype: Output sample type as a NumPy dtype, scalar type (``np.float64``)
///         or string (``"float64"``). Default: float32
///
/// Returns:
///     AudioSamples: Generated stereo sine wave audio data
#[pyfunction]
#[pyo3(signature = (frequency, duration_secs, sample_rate=44100, amplitude=1.0, dtype=None))]
pub fn stereo_sine_wave(
    py: Python<'_>,
    frequency: f64,
    duration_secs: f64,
    sample_rate: u32,
    amplitude: f64,
    dtype: Option<Bound<'_, PyAny>>,
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
///     dtype: Output sample type as a NumPy dtype, scalar type (``np.float64``)
///         or string (``"float64"``). Default: float32
///
/// Returns:
///     AudioSamples: Generated stereo chirp audio data
#[pyfunction]
#[pyo3(signature = (start_freq, end_freq, duration_secs, sample_rate=44100, amplitude=1.0, dtype=None))]
pub fn stereo_chirp(
    py: Python<'_>,
    start_freq: f64,
    end_freq: f64,
    duration_secs: f64,
    sample_rate: u32,
    amplitude: f64,
    dtype: Option<Bound<'_, PyAny>>,
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
///     dtype: Output sample type as a NumPy dtype, scalar type (``np.float64``)
///         or string (``"float64"``). Default: float32
///
/// Returns:
///     AudioSamples: Stereo silence audio data
#[pyfunction]
#[pyo3(signature = (duration_secs, sample_rate=44100, dtype=None))]
pub fn stereo_silence(
    py: Python<'_>,
    duration_secs: f64,
    sample_rate: u32,
    dtype: Option<Bound<'_, PyAny>>,
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

/// Generate a band-limited (alias-free) square wave via additive synthesis.
///
/// Unlike a hard-clipped square wave, this sums only the odd harmonics that lie
/// strictly below the Nyquist frequency, producing an exactly band-limited signal
/// with no aliasing.
///
/// Args:
///     frequency: Fundamental frequency of the square wave in Hz
///     duration_secs: Duration of the signal in seconds
///     sample_rate: Sample rate in samples per second (default: 44100)
///     amplitude: Peak amplitude of the underlying ideal square wave (default: 1.0)
///     dtype: Output sample type as a NumPy dtype, scalar type (``np.float64``)
///         or string (``"float64"``). Default: float32
///
/// Returns:
///     AudioSamples: Generated band-limited square wave audio data
#[pyfunction]
#[pyo3(signature = (frequency, duration_secs, sample_rate=44100, amplitude=1.0, dtype=None))]
fn square_wave_bandlimited(
    py: Python<'_>,
    frequency: f64,
    duration_secs: f64,
    sample_rate: u32,
    amplitude: f64,
    dtype: Option<Bound<'_, PyAny>>,
) -> PyResult<PyAudioSamples> {
    generate_audio(
        py,
        Generator::SquareBandlimited { frequency },
        duration_secs,
        sample_rate,
        amplitude,
        dtype,
    )
}

/// Generate a band-limited (alias-free) sawtooth wave via additive synthesis.
///
/// Sums only the harmonics that lie strictly below the Nyquist frequency, producing
/// an exactly band-limited signal with no aliasing.
///
/// Args:
///     frequency: Fundamental frequency of the sawtooth wave in Hz
///     duration_secs: Duration of the signal in seconds
///     sample_rate: Sample rate in samples per second (default: 44100)
///     amplitude: Peak amplitude of the underlying ideal sawtooth wave (default: 1.0)
///     dtype: Output sample type as a NumPy dtype, scalar type (``np.float64``)
///         or string (``"float64"``). Default: float32
///
/// Returns:
///     AudioSamples: Generated band-limited sawtooth wave audio data
#[pyfunction]
#[pyo3(signature = (frequency, duration_secs, sample_rate=44100, amplitude=1.0, dtype=None))]
fn sawtooth_wave_bandlimited(
    py: Python<'_>,
    frequency: f64,
    duration_secs: f64,
    sample_rate: u32,
    amplitude: f64,
    dtype: Option<Bound<'_, PyAny>>,
) -> PyResult<PyAudioSamples> {
    generate_audio(
        py,
        Generator::SawtoothBandlimited { frequency },
        duration_secs,
        sample_rate,
        amplitude,
        dtype,
    )
}

/// Generate a band-limited (alias-free) triangle wave via additive synthesis.
///
/// Sums only the odd harmonics that lie strictly below the Nyquist frequency,
/// producing an exactly band-limited signal with no aliasing.
///
/// Args:
///     frequency: Fundamental frequency of the triangle wave in Hz
///     duration_secs: Duration of the signal in seconds
///     sample_rate: Sample rate in samples per second (default: 44100)
///     amplitude: Peak amplitude of the underlying ideal triangle wave (default: 1.0)
///     dtype: Output sample type as a NumPy dtype, scalar type (``np.float64``)
///         or string (``"float64"``). Default: float32
///
/// Returns:
///     AudioSamples: Generated band-limited triangle wave audio data
#[pyfunction]
#[pyo3(signature = (frequency, duration_secs, sample_rate=44100, amplitude=1.0, dtype=None))]
fn triangle_wave_bandlimited(
    py: Python<'_>,
    frequency: f64,
    duration_secs: f64,
    sample_rate: u32,
    amplitude: f64,
    dtype: Option<Bound<'_, PyAny>>,
) -> PyResult<PyAudioSamples> {
    generate_audio(
        py,
        Generator::TriangleBandlimited { frequency },
        duration_secs,
        sample_rate,
        amplitude,
        dtype,
    )
}

/// Generate an exponential (logarithmic) frequency chirp (sweep) audio signal.
///
/// The instantaneous frequency sweeps geometrically from `start_freq` to `end_freq`.
/// Both frequencies must be strictly positive.
///
/// Args:
///     start_freq: Starting frequency in Hz (must be > 0)
///     end_freq: Ending frequency in Hz (must be > 0)
///     duration_secs: Duration of the signal in seconds
///     sample_rate: Sample rate in samples per second (default: 44100)
///     amplitude: Peak amplitude of the wave (default: 1.0)
///     dtype: Output sample type as a NumPy dtype, scalar type (``np.float64``)
///         or string (``"float64"``). Default: float32
///
/// Returns:
///     AudioSamples: Generated exponential chirp audio data
#[pyfunction]
#[pyo3(signature = (start_freq, end_freq, duration_secs, sample_rate=44100, amplitude=1.0, dtype=None))]
fn exponential_chirp(
    py: Python<'_>,
    start_freq: f64,
    end_freq: f64,
    duration_secs: f64,
    sample_rate: u32,
    amplitude: f64,
    dtype: Option<Bound<'_, PyAny>>,
) -> PyResult<PyAudioSamples> {
    if start_freq <= 0.0 || end_freq <= 0.0 {
        return Err(PyValueError::new_err(
            "exponential_chirp requires start_freq > 0 and end_freq > 0",
        ));
    }
    generate_audio(
        py,
        Generator::ExponentialChirp {
            f0: start_freq,
            f1: end_freq,
        },
        duration_secs,
        sample_rate,
        amplitude,
        dtype,
    )
}

/// Generate a frequency-modulated (FM) signal.
///
/// A sinusoidal carrier whose instantaneous phase is modulated by a sinusoidal
/// modulator. With `modulation_index == 0` this reduces to a pure sine at the
/// carrier frequency.
///
/// Args:
///     carrier_freq: Carrier frequency in Hz
///     modulator_freq: Modulator frequency in Hz
///     modulation_index: Modulation index (peak phase deviation in radians)
///     duration_secs: Duration of the signal in seconds
///     sample_rate: Sample rate in samples per second (default: 44100)
///     amplitude: Peak amplitude (default: 1.0)
///     dtype: Output sample type as a NumPy dtype, scalar type (``np.float64``)
///         or string (``"float64"``). Default: float32
///
/// Returns:
///     AudioSamples: Generated FM signal audio data
#[pyfunction]
#[pyo3(signature = (carrier_freq, modulator_freq, modulation_index, duration_secs, sample_rate=44100, amplitude=1.0, dtype=None))]
fn fm_signal(
    py: Python<'_>,
    carrier_freq: f64,
    modulator_freq: f64,
    modulation_index: f64,
    duration_secs: f64,
    sample_rate: u32,
    amplitude: f64,
    dtype: Option<Bound<'_, PyAny>>,
) -> PyResult<PyAudioSamples> {
    if duration_secs <= 0.0 {
        return Err(PyValueError::new_err("duration_secs must be positive"));
    }
    let sample_rate = nzu32_or_err(sample_rate)?;
    let duration = Duration::from_secs_f64(duration_secs);

    dispatch_audio!(py, dtype, |T| {
        let audio = audio_samples::fm_signal::<T>(
            carrier_freq,
            modulator_freq,
            modulation_index,
            duration,
            sample_rate,
            amplitude,
        );
        Ok(PyAudioSamples::from_audio_samples(audio))
    })
}

/// Generate an amplitude-modulated (AM) signal.
///
/// A carrier signal modulated by a low-frequency envelope.
///
/// Args:
///     carrier_freq: Frequency of the carrier signal in Hz
///     modulator_freq: Frequency of the modulating envelope in Hz
///     modulation_depth: Depth of modulation (0.0 = none, 1.0 = full)
///     duration_secs: Duration of the signal in seconds
///     sample_rate: Sample rate in samples per second (default: 44100)
///     amplitude: Overall amplitude (default: 1.0)
///     dtype: Output sample type as a NumPy dtype, scalar type (``np.float64``)
///         or string (``"float64"``). Default: float32
///
/// Returns:
///     AudioSamples: Generated AM signal audio data
#[pyfunction]
#[pyo3(signature = (carrier_freq, modulator_freq, modulation_depth, duration_secs, sample_rate=44100, amplitude=1.0, dtype=None))]
fn am_signal(
    py: Python<'_>,
    carrier_freq: f64,
    modulator_freq: f64,
    modulation_depth: f64,
    duration_secs: f64,
    sample_rate: u32,
    amplitude: f64,
    dtype: Option<Bound<'_, PyAny>>,
) -> PyResult<PyAudioSamples> {
    if duration_secs <= 0.0 {
        return Err(PyValueError::new_err("duration_secs must be positive"));
    }
    let sample_rate = nzu32_or_err(sample_rate)?;
    let duration = Duration::from_secs_f64(duration_secs);

    dispatch_audio!(py, dtype, |T| {
        let audio = audio_samples::am_signal::<T>(
            carrier_freq,
            modulator_freq,
            modulation_depth,
            duration,
            sample_rate,
            amplitude,
        );
        Ok(PyAudioSamples::from_audio_samples(audio))
    })
}

/// Generate a compound tone from multiple frequency components.
///
/// Useful for creating signals with harmonics or multiple simultaneous tones. Each
/// component is given as a ``(frequency_hz, amplitude)`` tuple.
///
/// Args:
///     components: List of ``(frequency_hz, amplitude)`` tuples. Must be non-empty.
///     duration_secs: Duration of the signal in seconds
///     sample_rate: Sample rate in samples per second (default: 44100)
///     dtype: Output sample type as a NumPy dtype, scalar type (``np.float64``)
///         or string (``"float64"``). Default: float32
///
/// Returns:
///     AudioSamples: Generated compound tone audio data
///
/// Raises:
///     ValueError: If `components` is empty or `duration_secs` is not positive.
#[pyfunction]
#[pyo3(signature = (components, duration_secs, sample_rate=44100, dtype=None))]
fn compound_tone(
    py: Python<'_>,
    components: Vec<(f64, f64)>,
    duration_secs: f64,
    sample_rate: u32,
    dtype: Option<Bound<'_, PyAny>>,
) -> PyResult<PyAudioSamples> {
    if duration_secs <= 0.0 {
        return Err(PyValueError::new_err("duration_secs must be positive"));
    }
    if components.is_empty() {
        return Err(PyValueError::new_err(
            "components must contain at least one (frequency, amplitude) tuple",
        ));
    }
    let sample_rate = nzu32_or_err(sample_rate)?;
    let duration = Duration::from_secs_f64(duration_secs);

    let tone_components: Vec<ToneComponent> = components
        .into_iter()
        .map(|(frequency, amplitude)| ToneComponent::new(frequency, amplitude))
        .collect();
    let components = NonEmptyVec::new(tone_components).map_err(|_| {
        PyValueError::new_err("components must contain at least one (frequency, amplitude) tuple")
    })?;

    dispatch_audio!(py, dtype, |T| {
        let audio = audio_samples::compound_tone::<T>(&components, duration, sample_rate);
        Ok(PyAudioSamples::from_audio_samples(audio))
    })
}

/// Generate a signal with periodic exponential-decay bursts.
///
/// Creates percussive-like transients useful for testing onset detection.
///
/// Args:
///     burst_rate: Number of bursts per second
///     decay_rate: Exponential decay rate (higher = faster decay)
///     duration_secs: Duration of the signal in seconds
///     sample_rate: Sample rate in samples per second (default: 44100)
///     amplitude: Peak amplitude of bursts (default: 1.0)
///     dtype: Output sample type as a NumPy dtype, scalar type (``np.float64``)
///         or string (``"float64"``). Default: float32
///
/// Returns:
///     AudioSamples: Generated burst signal audio data
#[pyfunction]
#[pyo3(signature = (burst_rate, decay_rate, duration_secs, sample_rate=44100, amplitude=1.0, dtype=None))]
fn exponential_bursts(
    py: Python<'_>,
    burst_rate: f64,
    decay_rate: f64,
    duration_secs: f64,
    sample_rate: u32,
    amplitude: f64,
    dtype: Option<Bound<'_, PyAny>>,
) -> PyResult<PyAudioSamples> {
    if duration_secs <= 0.0 {
        return Err(PyValueError::new_err("duration_secs must be positive"));
    }
    let sample_rate = nzu32_or_err(sample_rate)?;
    let duration = Duration::from_secs_f64(duration_secs);

    dispatch_audio!(py, dtype, |T| {
        let audio = audio_samples::utils::generation::exponential_bursts::<T>(
            burst_rate,
            decay_rate,
            duration,
            sample_rate,
            amplitude,
        );
        Ok(PyAudioSamples::from_audio_samples(audio))
    })
}
