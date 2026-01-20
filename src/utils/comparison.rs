//! Python bindings for audio signal comparison and similarity metrics.
//!
//! This module exposes the audio_samples::utils::comparison functions to Python,
//! providing signal comparison functions like correlation, MSE, SNR, and signal alignment.

use audio_samples::utils::comparison;
use pyo3::prelude::*;

use crate::{PyAudioSamples, audio_err_to_py};

#[pyfunction]
#[pyo3(signature = (a, b), text_signature = "(a: AudioSamples, b: AudioSamples) -> float")]
/// Computes the Pearson correlation coefficient between two audio signals.
///
/// This function measures linear similarity between two signals on a per-sample basis.
/// Returns a scalar correlation coefficient in the range [-1, 1].
///
/// For mono signals, the correlation is computed directly over the single channel.
/// For multi-channel signals, the correlation is computed independently for each
/// channel and the results are averaged.
///
/// Args:
///     a: The first audio signal.
///     b: The second audio signal.
///
/// Returns:
///     A scalar correlation coefficient in the range [-1, 1].
///
/// Raises:
///     ValueError: If the signals have different dimensions or channel configurations.
///
/// Example:
///     >>> from audio_samples import AudioSamples
///     >>> from audio_samples.utils import correlation
///     >>> audio1 = AudioSamples.zeros_mono_f32(44100, 44100)
///     >>> corr = correlation(audio1, audio1)  # Returns 1.0 for identical signals
#[inline]
pub fn correlation(py: Python, a: &PyAudioSamples, b: &PyAudioSamples) -> PyResult<f64> {
    // We need to get both AudioSamples and call the comparison function
    // The comparison functions in audio_samples work with AudioSamples<T>
    // We need to ensure both have the same type

    use crate::PyAudioDataInner;

    // First check if they're the same type
    let a_inner = a.inner();
    let b_inner = b.inner();

    match (a_inner, b_inner) {
        (PyAudioDataInner::U8(a_data), PyAudioDataInner::U8(b_data)) => {
            a_data.with_view(py, |a_view| {
                b_data.with_view(py, |b_view| {
                    comparison::correlation(&a_view, &b_view).map_err(audio_err_to_py)
                })
            })
        }
        (PyAudioDataInner::I16(a_data), PyAudioDataInner::I16(b_data)) => {
            a_data.with_view(py, |a_view| {
                b_data.with_view(py, |b_view| {
                    comparison::correlation(&a_view, &b_view).map_err(audio_err_to_py)
                })
            })
        }
        (PyAudioDataInner::I24(a_data), PyAudioDataInner::I24(b_data)) => {
            a_data.with_view(py, |a_view| {
                b_data.with_view(py, |b_view| {
                    comparison::correlation(&a_view, &b_view).map_err(audio_err_to_py)
                })
            })
        }
        (PyAudioDataInner::I32(a_data), PyAudioDataInner::I32(b_data)) => {
            a_data.with_view(py, |a_view| {
                b_data.with_view(py, |b_view| {
                    comparison::correlation(&a_view, &b_view).map_err(audio_err_to_py)
                })
            })
        }
        (PyAudioDataInner::F32(a_data), PyAudioDataInner::F32(b_data)) => {
            a_data.with_view(py, |a_view| {
                b_data.with_view(py, |b_view| {
                    comparison::correlation(&a_view, &b_view).map_err(audio_err_to_py)
                })
            })
        }
        (PyAudioDataInner::F64(a_data), PyAudioDataInner::F64(b_data)) => {
            a_data.with_view(py, |a_view| {
                b_data.with_view(py, |b_view| {
                    comparison::correlation(&a_view, &b_view).map_err(audio_err_to_py)
                })
            })
        }
        _ => Err(pyo3::exceptions::PyTypeError::new_err(
            "Audio samples must have the same data type for comparison",
        )),
    }
}

#[pyfunction]
#[pyo3(signature = (a, b), text_signature = "(a: AudioSamples, b: AudioSamples) -> float")]
/// Computes the mean squared error (MSE) between two audio signals.
///
/// This function measures the average squared per-sample difference between two signals.
/// Lower values indicate higher similarity, with 0.0 indicating identical signals.
///
/// For mono signals, the MSE is computed directly over the single channel.
/// For multi-channel signals, the MSE is computed independently for each channel
/// and the results are averaged.
///
/// Args:
///     a: The first audio signal.
///     b: The second audio signal.
///
/// Returns:
///     The mean squared error as a non-negative scalar value.
///
/// Raises:
///     ValueError: If the signals have different dimensions or channel configurations.
///
/// Example:
///     >>> from audio_samples import AudioSamples
///     >>> from audio_samples.utils import mse
///     >>> audio1 = AudioSamples.zeros_mono_f32(44100, 44100)
///     >>> error = mse(audio1, audio1)  # Returns 0.0 for identical signals
#[inline]
pub fn mse(py: Python, a: &PyAudioSamples, b: &PyAudioSamples) -> PyResult<f64> {
    use crate::PyAudioDataInner;

    let a_inner = a.inner();
    let b_inner = b.inner();

    match (a_inner, b_inner) {
        (PyAudioDataInner::U8(a_data), PyAudioDataInner::U8(b_data)) => {
            a_data.with_view(py, |a_view| {
                b_data.with_view(py, |b_view| {
                    comparison::mse(&a_view, &b_view).map_err(audio_err_to_py)
                })
            })
        }
        (PyAudioDataInner::I16(a_data), PyAudioDataInner::I16(b_data)) => {
            a_data.with_view(py, |a_view| {
                b_data.with_view(py, |b_view| {
                    comparison::mse(&a_view, &b_view).map_err(audio_err_to_py)
                })
            })
        }
        (PyAudioDataInner::I24(a_data), PyAudioDataInner::I24(b_data)) => {
            a_data.with_view(py, |a_view| {
                b_data.with_view(py, |b_view| {
                    comparison::mse(&a_view, &b_view).map_err(audio_err_to_py)
                })
            })
        }
        (PyAudioDataInner::I32(a_data), PyAudioDataInner::I32(b_data)) => {
            a_data.with_view(py, |a_view| {
                b_data.with_view(py, |b_view| {
                    comparison::mse(&a_view, &b_view).map_err(audio_err_to_py)
                })
            })
        }
        (PyAudioDataInner::F32(a_data), PyAudioDataInner::F32(b_data)) => {
            a_data.with_view(py, |a_view| {
                b_data.with_view(py, |b_view| {
                    comparison::mse(&a_view, &b_view).map_err(audio_err_to_py)
                })
            })
        }
        (PyAudioDataInner::F64(a_data), PyAudioDataInner::F64(b_data)) => {
            a_data.with_view(py, |a_view| {
                b_data.with_view(py, |b_view| {
                    comparison::mse(&a_view, &b_view).map_err(audio_err_to_py)
                })
            })
        }
        _ => Err(pyo3::exceptions::PyTypeError::new_err(
            "Audio samples must have the same data type for comparison",
        )),
    }
}

#[pyfunction]
#[pyo3(signature = (signal, noise), text_signature = "(signal: AudioSamples, noise: AudioSamples) -> float")]
/// Computes the signal-to-noise ratio (SNR) in decibels between a signal and noise.
///
/// This function measures the ratio between the average power of a signal and the
/// average power of a noise signal, expressed in decibels. Higher values indicate
/// greater dominance of the signal relative to noise.
///
/// For mono inputs, power is computed over the single channel.
/// For multi-channel inputs, power is computed over all samples across all channels.
///
/// Args:
///     signal: The signal component.
///     noise: The noise component.
///
/// Returns:
///     The signal-to-noise ratio in decibels. Returns positive infinity if noise
///     power is zero.
///
/// Raises:
///     ValueError: If the signals have different dimensions or channel configurations.
///
/// Example:
///     >>> from audio_samples import AudioSamples
///     >>> from audio_samples.utils import snr
///     >>> signal = AudioSamples.from_mono_f32([1.0, 2.0, 3.0], 44100)
///     >>> noise = AudioSamples.from_mono_f32([0.1, 0.2, 0.1], 44100)
///     >>> ratio = snr(signal, noise)  # Returns positive value in dB
#[inline]
pub fn snr(py: Python, signal: &PyAudioSamples, noise: &PyAudioSamples) -> PyResult<f64> {
    use crate::PyAudioDataInner;

    let signal_inner = signal.inner();
    let noise_inner = noise.inner();

    match (signal_inner, noise_inner) {
        (PyAudioDataInner::U8(s_data), PyAudioDataInner::U8(n_data)) => {
            s_data.with_view(py, |s_view| {
                n_data.with_view(py, |n_view| {
                    comparison::snr(&s_view, &n_view).map_err(audio_err_to_py)
                })
            })
        }
        (PyAudioDataInner::I16(s_data), PyAudioDataInner::I16(n_data)) => {
            s_data.with_view(py, |s_view| {
                n_data.with_view(py, |n_view| {
                    comparison::snr(&s_view, &n_view).map_err(audio_err_to_py)
                })
            })
        }
        (PyAudioDataInner::I24(s_data), PyAudioDataInner::I24(n_data)) => {
            s_data.with_view(py, |s_view| {
                n_data.with_view(py, |n_view| {
                    comparison::snr(&s_view, &n_view).map_err(audio_err_to_py)
                })
            })
        }
        (PyAudioDataInner::I32(s_data), PyAudioDataInner::I32(n_data)) => {
            s_data.with_view(py, |s_view| {
                n_data.with_view(py, |n_view| {
                    comparison::snr(&s_view, &n_view).map_err(audio_err_to_py)
                })
            })
        }
        (PyAudioDataInner::F32(s_data), PyAudioDataInner::F32(n_data)) => {
            s_data.with_view(py, |s_view| {
                n_data.with_view(py, |n_view| {
                    comparison::snr(&s_view, &n_view).map_err(audio_err_to_py)
                })
            })
        }
        (PyAudioDataInner::F64(s_data), PyAudioDataInner::F64(n_data)) => {
            s_data.with_view(py, |s_view| {
                n_data.with_view(py, |n_view| {
                    comparison::snr(&s_view, &n_view).map_err(audio_err_to_py)
                })
            })
        }
        _ => Err(pyo3::exceptions::PyTypeError::new_err(
            "Audio samples must have the same data type for comparison",
        )),
    }
}

#[pyfunction]
#[pyo3(signature = (reference, signal), text_signature = "(reference: AudioSamples, signal: AudioSamples) -> tuple[AudioSamples, int]")]
/// Aligns two audio signals by estimating a time offset that maximizes correlation.
///
/// This function temporally aligns `signal` to `reference` by searching for a
/// non-negative sample offset that maximizes cross-correlation. Returns a shifted
/// version of the signal padded at the start, together with the estimated offset.
///
/// For mono signals, alignment is computed directly on the single channel.
/// For multi-channel signals, all channels are averaged to a single signal before
/// alignment. The estimated offset is then applied uniformly to all channels.
///
/// Only non-negative offsets are considered (signal is shifted forward in time only).
/// The maximum offset searched is half of the shorter signal length.
///
/// Args:
///     reference: The reference signal that defines the target alignment.
///     signal: The signal to be aligned to the reference.
///
/// Returns:
///     A tuple (aligned_signal, offset_samples) where aligned_signal is the shifted
///     signal and offset_samples is the number of samples by which it was shifted.
///
/// Raises:
///     ValueError: If the signals have different channel counts or configurations.
///
/// Example:
///     >>> from audio_samples import AudioSamples
///     >>> from audio_samples.utils import align_signals
///     >>> reference = AudioSamples.from_mono_f32([1.0, 2.0, 3.0, 4.0], 44100)
///     >>> signal = AudioSamples.from_mono_f32([2.0, 3.0, 4.0, 5.0], 44100)
///     >>> aligned, offset = align_signals(reference, signal)
///     >>> print(f"Offset: {offset} samples")
#[inline]
pub fn align_signals(
    py: Python,
    reference: &PyAudioSamples,
    signal: &PyAudioSamples,
) -> PyResult<(PyAudioSamples, usize)> {
    use crate::PyAudioDataInner;

    let ref_inner = reference.inner();
    let sig_inner = signal.inner();

    match (ref_inner, sig_inner) {
        (PyAudioDataInner::U8(r_data), PyAudioDataInner::U8(s_data)) => {
            r_data.with_view(py, |r_view| {
                s_data.with_view(py, |s_view| {
                    let (aligned, offset) =
                        comparison::align_signals(&r_view, &s_view).map_err(audio_err_to_py)?;
                    Ok((PyAudioSamples::from_audio_samples(aligned), offset))
                })
            })
        }
        (PyAudioDataInner::I16(r_data), PyAudioDataInner::I16(s_data)) => {
            r_data.with_view(py, |r_view| {
                s_data.with_view(py, |s_view| {
                    let (aligned, offset) =
                        comparison::align_signals(&r_view, &s_view).map_err(audio_err_to_py)?;
                    Ok((PyAudioSamples::from_audio_samples(aligned), offset))
                })
            })
        }
        (PyAudioDataInner::I24(r_data), PyAudioDataInner::I24(s_data)) => {
            r_data.with_view(py, |r_view| {
                s_data.with_view(py, |s_view| {
                    let (aligned, offset) =
                        comparison::align_signals(&r_view, &s_view).map_err(audio_err_to_py)?;
                    Ok((PyAudioSamples::from_audio_samples(aligned), offset))
                })
            })
        }
        (PyAudioDataInner::I32(r_data), PyAudioDataInner::I32(s_data)) => {
            r_data.with_view(py, |r_view| {
                s_data.with_view(py, |s_view| {
                    let (aligned, offset) =
                        comparison::align_signals(&r_view, &s_view).map_err(audio_err_to_py)?;
                    Ok((PyAudioSamples::from_audio_samples(aligned), offset))
                })
            })
        }
        (PyAudioDataInner::F32(r_data), PyAudioDataInner::F32(s_data)) => {
            r_data.with_view(py, |r_view| {
                s_data.with_view(py, |s_view| {
                    let (aligned, offset) =
                        comparison::align_signals(&r_view, &s_view).map_err(audio_err_to_py)?;
                    Ok((PyAudioSamples::from_audio_samples(aligned), offset))
                })
            })
        }
        (PyAudioDataInner::F64(r_data), PyAudioDataInner::F64(s_data)) => {
            r_data.with_view(py, |r_view| {
                s_data.with_view(py, |s_view| {
                    let (aligned, offset) =
                        comparison::align_signals(&r_view, &s_view).map_err(audio_err_to_py)?;
                    Ok((PyAudioSamples::from_audio_samples(aligned), offset))
                })
            })
        }
        _ => Err(pyo3::exceptions::PyTypeError::new_err(
            "Audio samples must have the same data type for alignment",
        )),
    }
}
