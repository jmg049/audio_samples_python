use audio_samples::{AudioEditing, AudioSamples, ConvertTo, I24};
use non_empty_slice::{NonEmptySlice, NonEmptyVec};
use numpy::PyArrayDescrMethods;
use pyo3::{
    exceptions::{PyTypeError, PyValueError},
    prelude::*,
    types::{PyList, PyType},
};

use crate::{
    PyAudioDataInner, PyAudioSamples, audio_err_to_py, dispatch_with_view, dispatch_with_view_mut,
    types::{PyFadeCurve, PyPadSide, PyPerturbationConfig},
};

#[pymethods]
impl PyAudioSamples {
    #[pyo3(signature = (), text_signature = "($self) -> AudioSamples")]
    /// Reverse the samples and return a new buffer.
    ///
    /// Returns:
    ///     AudioSamples: Buffer containing the reversed samples.
    fn reverse(&self, py: Python<'_>) -> PyResult<Self> {
        dispatch_with_view!(self, py, |audio| {
            let reversed = audio.reverse();
            Ok(Self::from_audio_samples(reversed))
        })
    }

    #[pyo3(signature = (), text_signature = "($self)")]
    /// Reverse the samples in place.
    ///
    /// Returns:
    ///     None: Operation mutates the current buffer.
    ///
    /// Raises:
    ///     AudioError: If reversal fails.
    fn reverse_in_place(&mut self, py: Python<'_>) -> PyResult<()> {
        dispatch_with_view_mut!(self, py, |mut audio| audio
            .reverse_in_place()
            .map_err(audio_err_to_py))
    }

    #[pyo3(signature = (start_seconds: "float", end_seconds: "float"), text_signature = "($self, start_seconds: float, end_seconds: float) -> AudioSamples")]
    /// Trim the buffer to the given time range.
    ///
    /// Args:
    ///     start_seconds (float): Inclusive start offset in seconds.
    ///     end_seconds (float): Exclusive end offset in seconds.
    ///
    /// Returns:
    ///     AudioSamples: Buffer containing only the requested region.
    ///
    /// Raises:
    ///     AudioError: If the range is invalid or trimming fails.
    fn trim(&self, py: Python<'_>, start_seconds: f64, end_seconds: f64) -> PyResult<Self> {
        dispatch_with_view!(self, py, |audio| {
            let trimmed = audio
                .trim(start_seconds, end_seconds)
                .map_err(audio_err_to_py)?;
            Ok(Self::from_audio_samples(trimmed))
        })
    }

    #[pyo3(signature = (pad_start_seconds: "float", pad_end_seconds: "float", pad_value: "float"), text_signature = "($self, pad_start_seconds: float, pad_end_seconds: float, pad_value: float) -> AudioSamples")]
    /// Pad the buffer with constant values on both sides.
    ///
    /// Args:
    ///     pad_start_seconds (float): Duration to pad before the signal.
    ///     pad_end_seconds (float): Duration to pad after the signal.
    ///     pad_value (float): Value used for padding.
    ///
    /// Returns:
    ///     AudioSamples: New buffer with the added padding.
    ///
    /// Raises:
    ///     AudioError: If padding fails or durations are invalid.
    fn pad(
        &self,
        py: Python<'_>,
        pad_start_seconds: f64,
        pad_end_seconds: f64,
        pad_value: f64,
    ) -> PyResult<Self> {
        dispatch_with_view!(self, py, |audio| {
            let audio = audio
                .pad(pad_start_seconds, pad_end_seconds, pad_value.convert_to())
                .map_err(audio_err_to_py)?;
            Ok(Self::from_audio_samples(audio))
        })
    }

    #[pyo3(signature = (target_num_samples: "int", pad_value: "float"), text_signature = "($self, target_num_samples: int, pad_value: float) -> AudioSamples")]
    /// Right-pad the buffer to a fixed sample count.
    ///
    /// Args:
    ///     target_num_samples (int): Desired total number of samples.
    ///     pad_value (float): Value used for padding.
    ///
    /// Returns:
    ///     AudioSamples: New buffer with the requested length.
    ///
    /// Raises:
    ///     AudioError: If the target length is shorter than the input or padding fails.
    fn pad_samples_right(
        &self,
        py: Python<'_>,
        target_num_samples: usize,
        pad_value: f64,
    ) -> PyResult<Self> {
        dispatch_with_view!(self, py, |audio| {
            let audio = audio
                .pad_samples_right(target_num_samples, pad_value.convert_to())
                .map_err(audio_err_to_py)?;
            Ok(Self::from_audio_samples(audio))
        })
    }

    #[pyo3(signature = (target_duration_seconds: "float", pad_value: "float", pad_side: "PadSide"), text_signature = "($self, target_duration_seconds: float, pad_value: float, pad_side: PadSide) -> AudioSamples")]
    /// Pad the buffer to reach a target duration.
    ///
    /// Args:
    ///     target_duration_seconds (float): Desired total duration.
    ///     pad_value (float): Value used for padding.
    ///     pad_side (PadSide): Side to apply the padding.
    ///
    /// Returns:
    ///     AudioSamples: New buffer matching the requested duration.
    ///
    /// Raises:
    ///     AudioError: If padding fails or the target duration is invalid.
    fn pad_to_duration(
        &self,
        py: Python<'_>,
        target_duration_seconds: f64,
        pad_value: f64,
        pad_side: PyPadSide,
    ) -> PyResult<Self> {
        dispatch_with_view!(self, py, |audio| {
            let audio = audio
                .pad_to_duration(
                    target_duration_seconds,
                    pad_value.convert_to(),
                    pad_side.into(),
                )
                .map_err(audio_err_to_py)?;
            Ok(Self::from_audio_samples(audio))
        })
    }

    #[pyo3(signature = (segment_duration_seconds: "float"), text_signature = "($self, segment_duration_seconds: float) -> list[AudioSamples]")]
    /// Split the buffer into equal-length segments.
    ///
    /// Args:
    ///     segment_duration_seconds (float): Duration of each segment.
    ///
    /// Returns:
    ///     list[AudioSamples]: List of segments that cover the original buffer.
    ///
    /// Raises:
    ///     AudioError: If segmentation fails or the duration is invalid.
    fn split(&self, py: Python<'_>, segment_duration_seconds: f64) -> PyResult<Vec<Self>> {
        dispatch_with_view!(self, py, |audio| {
            let segments = audio
                .split(segment_duration_seconds)
                .map_err(audio_err_to_py)?;
            Ok(segments
                .into_iter()
                .map(Self::from_audio_samples)
                .collect::<Vec<_>>())
        })
    }

    #[classmethod]
    #[pyo3(signature = (segments: "list[AudioSamples]"), text_signature = "($cls, segments: list[AudioSamples]) -> AudioSamples")]
    /// Concatenate multiple buffers end to end.
    ///
    /// Args:
    ///     segments (list[AudioSamples]): Non-empty list of buffers to join.
    ///
    /// Returns:
    ///     AudioSamples: Combined buffer spanning all segments.
    ///
    /// Raises:
    ///     ValueError: If the segment list is empty.
    ///     TypeError: If segments differ in dtype or are not AudioSamples.
    ///     AudioError: If concatenation fails.
    fn concatenate(
        _cls: &Bound<'_, PyType>,
        py: Python<'_>,
        segments: &Bound<'_, PyList>,
    ) -> PyResult<Self> {
        if segments.is_empty() {
            return Err(PyValueError::new_err(
                "Cannot concatenate empty segment list",
            ));
        }

        // Extract PyAudioSamples from the list
        let segments_vec: Vec<PyRef<PyAudioSamples>> = segments
            .iter()
            .map(|item| {
                item.extract::<PyRef<PyAudioSamples>>()
                    .map_err(|e| PyTypeError::new_err(format!("Expected AudioSamples, got: {}", e)))
            })
            .collect::<PyResult<Vec<_>>>()?;
        // safety: We just checked that segments_vec is not empty, so this is safe
        let segments_vec = unsafe { NonEmptyVec::new_unchecked(segments_vec) };

        // Validate all segments have the same dtype
        let first_dtype = segments_vec[0].dtype(py);
        for segment in segments_vec.iter().skip(1) {
            if !segment.dtype(py).is_equiv_to(&first_dtype) {
                return Err(PyTypeError::new_err(
                    "All segments must have the same dtype for concatenation",
                ));
            }
        }

        // Dispatch based on the dtype
        match &segments_vec[0].inner {
            PyAudioDataInner::U8(_) => {
                let audio_segments: Vec<_> = segments_vec
                    .iter()
                    .map(|seg| {
                        if let PyAudioDataInner::U8(typed) = &seg.inner {
                            typed.with_view(py, |audio| audio.clone().into_owned())
                        } else {
                            unreachable!()
                        }
                    })
                    .collect();
                // safety: have checked for non-empty already
                let audio_segments = unsafe { NonEmptyVec::new_unchecked(audio_segments) };
                let concatenated = <AudioSamples<u8> as AudioEditing>::concatenate(&audio_segments)
                    .map_err(audio_err_to_py)?
                    .into_owned();
                Ok(Self::from_audio_samples(concatenated))
            }
            PyAudioDataInner::I16(_) => {
                let audio_segments: Vec<_> = segments_vec
                    .iter()
                    .map(|seg| {
                        if let PyAudioDataInner::I16(typed) = &seg.inner {
                            typed.with_view(py, |audio| audio.clone().into_owned())
                        } else {
                            unreachable!()
                        }
                    })
                    .collect();
                // safety: have checked for non-empty already
                let audio_segments = unsafe { NonEmptyVec::new_unchecked(audio_segments) };
                let concatenated =
                    <AudioSamples<i16> as AudioEditing>::concatenate(&audio_segments)
                        .map_err(audio_err_to_py)?
                        .into_owned();
                Ok(Self::from_audio_samples(concatenated))
            }
            PyAudioDataInner::I24(_) => {
                let audio_segments: Vec<_> = segments_vec
                    .iter()
                    .map(|seg| {
                        if let PyAudioDataInner::I24(typed) = &seg.inner {
                            typed.with_view(py, |audio| audio.clone().into_owned())
                        } else {
                            unreachable!()
                        }
                    })
                    .collect();
                // safety: have checked for non-empty already
                let audio_segments = unsafe { NonEmptyVec::new_unchecked(audio_segments) };
                let concatenated =
                    <AudioSamples<I24> as AudioEditing>::concatenate(&audio_segments)
                        .map_err(audio_err_to_py)?
                        .into_owned();
                Ok(Self::from_audio_samples(concatenated))
            }
            PyAudioDataInner::I32(_) => {
                let audio_segments: Vec<_> = segments_vec
                    .iter()
                    .map(|seg| {
                        if let PyAudioDataInner::I32(typed) = &seg.inner {
                            typed.with_view(py, |audio| audio.clone().into_owned())
                        } else {
                            unreachable!()
                        }
                    })
                    .collect();
                // safety: have checked for non-empty already
                let audio_segments = unsafe { NonEmptyVec::new_unchecked(audio_segments) };
                let concatenated =
                    <AudioSamples<i32> as AudioEditing>::concatenate(&audio_segments)
                        .map_err(audio_err_to_py)?
                        .into_owned();
                Ok(Self::from_audio_samples(concatenated))
            }
            PyAudioDataInner::F32(_) => {
                let audio_segments: Vec<_> = segments_vec
                    .iter()
                    .map(|seg| {
                        if let PyAudioDataInner::F32(typed) = &seg.inner {
                            typed.with_view(py, |audio| audio.clone().into_owned())
                        } else {
                            unreachable!()
                        }
                    })
                    .collect();
                // safety: have checked for non-empty already
                let audio_segments = unsafe { NonEmptyVec::new_unchecked(audio_segments) };
                let concatenated =
                    <AudioSamples<f32> as AudioEditing>::concatenate(&audio_segments)
                        .map_err(audio_err_to_py)?
                        .into_owned();
                Ok(Self::from_audio_samples(concatenated))
            }
            PyAudioDataInner::F64(_) => {
                let audio_segments: Vec<_> = segments_vec
                    .iter()
                    .map(|seg| {
                        if let PyAudioDataInner::F64(typed) = &seg.inner {
                            typed.with_view(py, |audio| audio.clone().into_owned())
                        } else {
                            unreachable!()
                        }
                    })
                    .collect();
                // safety: have checked for non-empty already
                let audio_segments = unsafe { NonEmptyVec::new_unchecked(audio_segments) };
                let concatenated =
                    <AudioSamples<f64> as AudioEditing>::concatenate(&audio_segments)
                        .map_err(audio_err_to_py)?
                        .into_owned();
                Ok(Self::from_audio_samples(concatenated))
            }
        }
    }

    #[classmethod]
    #[pyo3(signature = (sources: "list[AudioSamples]", weights: "Optional[list[float]]"=None), text_signature = "($cls, sources: list[AudioSamples], weights: Optional[list[float]] = None) -> AudioSamples")]
    /// Mix multiple buffers together with optional weights.
    ///
    /// Args:
    ///     sources (list[AudioSamples]): Non-empty list of buffers to mix.
    ///     weights (Optional[list[float]]): Optional mixing weights, matching the number of sources.
    ///
    /// Returns:
    ///     AudioSamples: Mixed buffer containing the weighted sum.
    ///
    /// Raises:
    ///     ValueError: If the source list is empty.
    ///     TypeError: If sources differ in dtype or are not AudioSamples.
    ///     AudioError: If mixing fails.
    fn mix(
        _cls: &Bound<'_, PyType>,
        py: Python<'_>,
        sources: &Bound<'_, PyList>,
        weights: Option<Vec<f64>>,
    ) -> PyResult<Self> {
        if sources.is_empty() {
            return Err(PyValueError::new_err("Cannot mix empty source list"));
        }

        let weights_slice: Option<&NonEmptySlice<f64>> = if let Some(w) = &weights
            && !w.is_empty()
        {
            // safety: just checked for empty
            Some(unsafe { NonEmptySlice::new_unchecked(w) })
        } else {
            None
        };
        // Extract PyAudioSamples from the list
        let sources_vec: Vec<PyRef<PyAudioSamples>> = sources
            .iter()
            .map(|item| {
                item.extract::<PyRef<PyAudioSamples>>()
                    .map_err(|e| PyTypeError::new_err(format!("Expected AudioSamples, got: {}", e)))
            })
            .collect::<PyResult<Vec<_>>>()?;

        // Validate all sources have the same dtype
        let first_dtype = sources_vec[0].dtype(py);
        for source in sources_vec.iter().skip(1) {
            if !source.dtype(py).is_equiv_to(&first_dtype) {
                return Err(PyTypeError::new_err(
                    "All sources must have the same dtype for mixing",
                ));
            }
        }

        // Dispatch based on the dtype
        match &sources_vec[0].inner {
            PyAudioDataInner::U8(_) => {
                let audio_sources: Vec<_> = sources_vec
                    .iter()
                    .map(|src| {
                        if let PyAudioDataInner::U8(typed) = &src.inner {
                            typed.with_view(py, |audio| audio.clone().into_owned())
                        } else {
                            unreachable!()
                        }
                    })
                    .collect();
                // safety: have checked for non-empty already
                let audio_sources = unsafe { NonEmptyVec::new_unchecked(audio_sources) };
                let mixed = if let Some(w) = weights_slice {
                    <AudioSamples<u8> as AudioEditing>::mix(&audio_sources, Some(w))
                        .map_err(audio_err_to_py)?
                } else {
                    <AudioSamples<u8> as AudioEditing>::mix(&audio_sources, None)
                        .map_err(audio_err_to_py)?
                }
                .into_owned();
                Ok(Self::from_audio_samples(mixed))
            }
            PyAudioDataInner::I16(_) => {
                let audio_sources: Vec<_> = sources_vec
                    .iter()
                    .map(|src| {
                        if let PyAudioDataInner::I16(typed) = &src.inner {
                            typed.with_view(py, |audio| audio.clone().into_owned())
                        } else {
                            unreachable!()
                        }
                    })
                    .collect();
                // safety: have checked for non-empty already
                let audio_sources = unsafe { NonEmptyVec::new_unchecked(audio_sources) };
                let mixed = if let Some(w) = weights_slice {
                    <AudioSamples<i16> as AudioEditing>::mix(&audio_sources, Some(w))
                        .map_err(audio_err_to_py)?
                } else {
                    <AudioSamples<i16> as AudioEditing>::mix(&audio_sources, None)
                        .map_err(audio_err_to_py)?
                }
                .into_owned();
                Ok(Self::from_audio_samples(mixed))
            }
            PyAudioDataInner::I24(_) => {
                let audio_sources: Vec<_> = sources_vec
                    .iter()
                    .map(|src| {
                        if let PyAudioDataInner::I24(typed) = &src.inner {
                            typed.with_view(py, |audio| audio.clone().into_owned())
                        } else {
                            unreachable!()
                        }
                    })
                    .collect();
                // safety: have checked for non-empty already
                let audio_sources = unsafe { NonEmptyVec::new_unchecked(audio_sources) };
                let mixed = if let Some(w) = weights_slice {
                    <AudioSamples<I24> as AudioEditing>::mix(&audio_sources, Some(w))
                        .map_err(audio_err_to_py)?
                } else {
                    <AudioSamples<I24> as AudioEditing>::mix(&audio_sources, None)
                        .map_err(audio_err_to_py)?
                }
                .into_owned();
                Ok(Self::from_audio_samples(mixed))
            }
            PyAudioDataInner::I32(_) => {
                let audio_sources: Vec<_> = sources_vec
                    .iter()
                    .map(|src| {
                        if let PyAudioDataInner::I32(typed) = &src.inner {
                            typed.with_view(py, |audio| audio.clone().into_owned())
                        } else {
                            unreachable!()
                        }
                    })
                    .collect();
                // safety: have checked for non-empty already
                let audio_sources = unsafe { NonEmptyVec::new_unchecked(audio_sources) };
                let mixed = if let Some(w) = weights_slice {
                    <AudioSamples<i32> as AudioEditing>::mix(&audio_sources, Some(w))
                        .map_err(audio_err_to_py)?
                } else {
                    <AudioSamples<i32> as AudioEditing>::mix(&audio_sources, None)
                        .map_err(audio_err_to_py)?
                }
                .into_owned();
                Ok(Self::from_audio_samples(mixed))
            }
            PyAudioDataInner::F32(_) => {
                let audio_sources: Vec<_> = sources_vec
                    .iter()
                    .map(|src| {
                        if let PyAudioDataInner::F32(typed) = &src.inner {
                            typed.with_view(py, |audio| audio.clone().into_owned())
                        } else {
                            unreachable!()
                        }
                    })
                    .collect();

                // safety: have checked for non-empty already
                let weights_slice = weights
                    .as_ref()
                    .map(|w| unsafe { NonEmptySlice::new_unchecked(w) });
                // safety: have checked for non-empty already
                let audio_sources = unsafe { NonEmptyVec::new_unchecked(audio_sources) };
                let mixed = if let Some(&w) = weights_slice.as_ref() {
                    <AudioSamples<f32> as AudioEditing>::mix(&audio_sources, Some(w))
                        .map_err(audio_err_to_py)?
                } else {
                    <AudioSamples<f32> as AudioEditing>::mix(&audio_sources, None)
                        .map_err(audio_err_to_py)?
                }
                .into_owned();
                Ok(Self::from_audio_samples(mixed))
            }
            PyAudioDataInner::F64(_) => {
                let audio_sources: Vec<_> = sources_vec
                    .iter()
                    .map(|src| {
                        if let PyAudioDataInner::F64(typed) = &src.inner {
                            typed.with_view(py, |audio| audio.clone().into_owned())
                        } else {
                            unreachable!()
                        }
                    })
                    .collect();
                // safety: have checked for non-empty already
                let audio_sources = unsafe { NonEmptyVec::new_unchecked(audio_sources) };
                let mixed = if let Some(w) = weights_slice {
                    <AudioSamples<f64> as AudioEditing>::mix(&audio_sources, Some(w))
                        .map_err(audio_err_to_py)?
                } else {
                    <AudioSamples<f64> as AudioEditing>::mix(&audio_sources, None)
                        .map_err(audio_err_to_py)?
                }
                .into_owned();
                Ok(Self::from_audio_samples(mixed))
            }
        }
    }

    #[pyo3(signature = (duration_seconds: "float", curve: "FadeCurve"), text_signature = "($self, duration_seconds: float, curve: FadeCurve) -> None")]
    /// Apply an in-place fade-in envelope.
    ///
    /// Args:
    ///     duration_seconds (float): Fade duration.
    ///     curve (FadeCurve): Curve controlling the fade shape.
    ///
    /// Returns:
    ///     None: Operation mutates the current buffer.
    ///
    /// Raises:
    ///     AudioError: If the fade fails or parameters are invalid.
    fn fade_in(
        &mut self,
        py: Python<'_>,
        duration_seconds: f64,
        curve: &PyFadeCurve,
    ) -> PyResult<()> {
        dispatch_with_view_mut!(self, py, |mut audio| {
            audio
                .fade_in(duration_seconds, curve.inner)
                .map_err(audio_err_to_py)
        })
    }

    #[pyo3(signature = (duration_seconds: "float", curve: "FadeCurve"), text_signature = "($self, duration_seconds: float, curve: FadeCurve) -> None")]
    /// Apply an in-place fade-out envelope.
    ///
    /// Args:
    ///     duration_seconds (float): Fade duration.
    ///     curve (FadeCurve): Curve controlling the fade shape.
    ///
    /// Returns:
    ///     None: Operation mutates the current buffer.
    ///
    /// Raises:
    ///     AudioError: If the fade fails or parameters are invalid.
    fn fade_out(
        &mut self,
        py: Python<'_>,
        duration_seconds: f64,
        curve: &PyFadeCurve,
    ) -> PyResult<()> {
        dispatch_with_view_mut!(self, py, |mut audio| {
            audio
                .fade_out(duration_seconds, curve.inner)
                .map_err(audio_err_to_py)
        })
    }

    #[pyo3(signature = (count: "int"), text_signature = "($self, count: int) -> AudioSamples")]
    /// Repeat the buffer a fixed number of times.
    ///
    /// Args:
    ///     count (int): Number of repetitions; must be greater than zero.
    ///
    /// Returns:
    ///     AudioSamples: Concatenated buffer containing the repeated audio.
    ///
    /// Raises:
    ///     AudioError: If repetition fails.
    fn repeat(&self, py: Python<'_>, count: usize) -> PyResult<Self> {
        dispatch_with_view!(self, py, |audio| {
            let repeated = audio.repeat(count).map_err(audio_err_to_py)?;
            Ok(Self::from_audio_samples(repeated))
        })
    }

    #[pyo3(signature = (threshold_db: "float"), text_signature = "($self, threshold_db: float) -> AudioSamples")]
    /// Trim leading and trailing silence below a threshold.
    ///
    /// Args:
    ///     threshold_db (float): Silence threshold in decibels.
    ///
    /// Returns:
    ///     AudioSamples: Buffer with silence removed.
    ///
    /// Raises:
    ///     AudioError: If trimming fails.
    fn trim_silence(&self, py: Python<'_>, threshold_db: f64) -> PyResult<Self> {
        dispatch_with_view!(self, py, |audio| {
            let trimmed = audio.trim_silence(threshold_db).map_err(audio_err_to_py)?;
            Ok(Self::from_audio_samples(trimmed))
        })
    }

    #[pyo3(signature = (config: "PerturbationConfig"), text_signature = "($self, config: PerturbationConfig) -> AudioSamples")]
    /// Apply stochastic perturbations and return a new buffer.
    ///
    /// Args:
    ///     config (PerturbationConfig): Configuration describing perturbations to apply.
    ///
    /// Returns:
    ///     AudioSamples: Perturbed buffer.
    ///
    /// Raises:
    ///     AudioError: If perturbation fails.
    fn perturb(&self, py: Python<'_>, config: &PyPerturbationConfig) -> PyResult<Self> {
        dispatch_with_view!(self, py, |audio| {
            let perturbed = audio.perturb(&config.inner).map_err(audio_err_to_py)?;
            Ok(Self::from_audio_samples(perturbed))
        })
    }

    #[pyo3(signature = (config: "PerturbationConfig"), text_signature = "($self, config: PerturbationConfig) -> None")]
    /// Apply stochastic perturbations in place.
    ///
    /// Args:
    ///     config (PerturbationConfig): Configuration describing perturbations to apply.
    ///
    /// Returns:
    ///     None: Operation mutates the current buffer.
    ///
    /// Raises:
    ///     AudioError: If perturbation fails.
    fn perturb_in_place(&mut self, py: Python<'_>, config: &PyPerturbationConfig) -> PyResult<()> {
        dispatch_with_view_mut!(self, py, |mut audio| {
            audio
                .perturb_in_place(&config.inner)
                .map_err(audio_err_to_py)
        })
    }

    #[classmethod]
    #[pyo3(name = "stack", signature = (sources: "list[AudioSamples]"), text_signature = "($cls, sources: list[AudioSamples]) -> AudioSamples")]
    /// Stack multiple buffers along the channel axis.
    ///
    /// Args:
    ///     sources (list[AudioSamples]): Buffers to stack; must share dtype and sample rate.
    ///
    /// Returns:
    ///     AudioSamples: Combined buffer with channels from each source.
    ///
    /// Raises:
    ///     AudioError: If stacking fails or sources are incompatible.
    fn py_stack(
        _cls: &Bound<'_, PyType>,
        py: Python<'_>,
        sources: Vec<Bound<'_, PyAudioSamples>>,
    ) -> PyResult<Self> {
        Self::stack(py, sources)
    }
}
