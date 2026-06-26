use audio_samples::{AudioChannelOps, AudioSamples, ConvertTo};
use non_empty_slice::NonEmptySlice;
use numpy::PyArrayDescrMethods;
use pyo3::{exceptions::PyValueError, prelude::*, types::PyList};

use crate::{
    PyAudioDataInner, PyAudioSamples, audio_err_to_py, dispatch_with_view, dispatch_with_view_mut,
    types::{PyMonoConversionMethod, PyStereoConversionMethod},
};

#[pymethods]
impl PyAudioSamples {
    #[pyo3(signature = (method: "MonoConversionMethod" = PyMonoConversionMethod::default()), text_signature = "($self, method: MonoConversionMethod=MonoConversionMethod.average) -> AudioSamples")]
    /// Convert the audio buffer to mono using the provided strategy.
    ///
    /// Args:
    ///     method (MonoConversionMethod): Conversion method describing how to combine channels.
    ///
    /// Returns:
    ///     AudioSamples: New buffer with a single channel.
    ///
    /// Raises:
    ///     AudioError: If channel conversion fails.
    fn to_mono(&self, py: Python<'_>, method: Option<PyMonoConversionMethod>) -> PyResult<Self> {
        dispatch_with_view!(self, py, |audio| {
            audio
                .to_mono(method.unwrap_or_default().inner)
                .map_err(audio_err_to_py)
                .map(Self::from_audio_samples)
        })
    }

    #[pyo3(signature = (method: "StereoConversionMethod" = PyStereoConversionMethod::default()), text_signature = "($self, method: StereoConversionMethod=StereoConversionMethod.duplicate) -> AudioSamples")]
    /// Convert the audio buffer to stereo using the provided strategy.
    ///
    /// Args:
    ///     method (StereoConversionMethod): Conversion method describing how to expand channels.
    ///
    /// Returns:
    ///     AudioSamples: New buffer with two channels.
    ///
    /// Raises:
    ///     AudioError: If channel conversion fails.
    fn to_stereo(
        &self,
        py: Python<'_>,
        method: Option<PyStereoConversionMethod>,
    ) -> PyResult<Self> {
        dispatch_with_view!(self, py, |audio| {
            audio
                .to_stereo(method.unwrap_or_default().inner)
                .map_err(audio_err_to_py)
                .map(Self::from_audio_samples)
        })
    }
    #[pyo3(signature = (n_channels: "int"), text_signature = "($self, n_channels: int) -> AudioSamples")]
    /// Duplicate the buffer into the requested number of channels.
    ///
    /// Args:
    ///     n_channels (int): Target channel count; must be greater than zero.
    ///
    /// Returns:
    ///     AudioSamples: New buffer with the duplicated channels.
    ///
    /// Raises:
    ///     AudioError: If duplication fails or the channel count is invalid.
    fn duplicate_to_channels(&self, py: Python<'_>, n_channels: usize) -> PyResult<Self> {
        dispatch_with_view!(self, py, |audio| {
            audio
                .duplicate_to_channels(n_channels)
                .map_err(audio_err_to_py)
                .map(Self::from_audio_samples)
        })
    }

    #[pyo3(signature = (channel_index: "int"), text_signature = "($self, channel_index: int) -> AudioSamples")]
    /// Extract a single channel from the buffer.
    ///
    /// Args:
    ///     channel_index (int): Zero-based index of the channel to extract.
    ///
    /// Returns:
    ///     AudioSamples: Buffer representing the selected channel.
    ///
    /// Raises:
    ///     AudioError: If the channel index is out of range or extraction fails.
    fn extract_channel(&self, py: Python<'_>, channel_index: usize) -> PyResult<Self> {
        dispatch_with_view!(self, py, |audio| {
            let result = audio
                .extract_channel(channel_index)
                .map_err(audio_err_to_py)?;
            Ok(Self::from_audio_samples(result))
        })
    }

    #[pyo3(signature = (channel1: "int", channel2: "int"), text_signature = "($self, channel1: int, channel2: int) -> None")]
    /// Swap two channels in place.
    ///
    /// Args:
    ///     channel1 (int): Zero-based index of the first channel.
    ///     channel2 (int): Zero-based index of the second channel.
    ///
    /// Returns:
    ///     None: Operation mutates the current buffer.
    ///
    /// Raises:
    ///     AudioError: If swapping fails or indices are invalid.
    fn swap_channels(&mut self, py: Python<'_>, channel1: usize, channel2: usize) -> PyResult<()> {
        dispatch_with_view_mut!(self, py, |mut audio| {
            audio
                .swap_channels_in_place(channel1, channel2)
                .map_err(audio_err_to_py)?;
            Ok(())
        })
    }

    #[pyo3(signature = (pan_value: "float"), text_signature = "($self, pan_value: float)")]
    /// Apply stereo panning to the buffer in place.
    ///
    /// Args:
    ///     pan_value (float): Normalized pan value; negative favors left, positive favors right.
    ///
    /// Returns:
    ///     None: Operation mutates the current buffer.
    ///
    /// Raises:
    ///     AudioError: If the pan adjustment fails.
    fn pan(&mut self, py: Python<'_>, pan_value: f64) -> PyResult<()> {
        dispatch_with_view_mut!(self, py, |mut audio| {
            audio
                .pan_in_place(pan_value.convert_to())
                .map_err(audio_err_to_py)
        })
    }

    #[pyo3(signature = (balance: "float"), text_signature = "($self, balance: float)")]
    /// Adjust the stereo balance in place.
    ///
    /// Args:
    ///     balance (float): Balance offset; negative favors left, positive favors right.
    ///
    /// Returns:
    ///     None: Operation mutates the current buffer.
    ///
    /// Raises:
    ///     AudioError: If the balance adjustment fails.
    fn balance(&mut self, py: Python<'_>, balance: f64) -> PyResult<()> {
        dispatch_with_view_mut!(self, py, |mut audio| {
            audio.balance_in_place(balance).map_err(audio_err_to_py)
        })
    }

    #[pyo3(signature = (channel_index: "int"), text_signature = "($self, channel_index: int) -> AudioSamples")]
    /// Return a single channel as a new mono buffer.
    ///
    /// The underlying crate method ``borrow_channel`` returns a borrowed,
    /// lifetime-bound view into the buffer. Such views cannot be exposed to
    /// Python safely, so this binding copies the selected channel into an owned
    /// buffer before returning it.
    ///
    /// Args:
    ///     channel_index (int): Zero-based index of the channel to borrow.
    ///         Ignored for mono audio.
    ///
    /// Returns:
    ///     AudioSamples: An owned mono buffer containing the selected channel.
    ///
    /// Raises:
    ///     ValueError: If the channel index is out of range.
    fn borrow_channel(&self, py: Python<'_>, channel_index: usize) -> PyResult<Self> {
        dispatch_with_view!(self, py, |audio| {
            let channel = audio
                .borrow_channel(channel_index)
                .map_err(audio_err_to_py)?;
            Ok(Self::from_audio_samples(channel.into_owned()))
        })
    }

    #[staticmethod]
    #[pyo3(signature = (channels: "list[AudioSamples]"), text_signature = "(channels: list[AudioSamples]) -> AudioSamples")]
    /// Combine multiple mono buffers into a single multi-channel buffer.
    ///
    /// The first buffer becomes channel 0, the second channel 1, and so on. All
    /// inputs must share the same dtype and number of samples; the output sample
    /// rate is taken from the first input.
    ///
    /// Args:
    ///     channels (list[AudioSamples]): Non-empty list of mono buffers, all of
    ///         the same dtype and length.
    ///
    /// Returns:
    ///     AudioSamples: A multi-channel buffer with one channel per input.
    ///
    /// Raises:
    ///     ValueError: If the list is empty, the dtypes differ, or the inputs do
    ///         not all have the same sample count.
    fn interleave_channels(py: Python<'_>, channels: &Bound<'_, PyList>) -> PyResult<Self> {
        if channels.is_empty() {
            return Err(PyValueError::new_err("Cannot interleave empty channel list"));
        }

        let channels_vec: Vec<PyRef<PyAudioSamples>> = channels
            .iter()
            .map(|item| {
                item.extract::<PyRef<PyAudioSamples>>().map_err(|e| {
                    PyValueError::new_err(format!("Expected AudioSamples, got: {}", e))
                })
            })
            .collect::<PyResult<Vec<_>>>()?;

        // Validate all channels share the same dtype.
        let first_dtype = channels_vec[0].dtype(py);
        for ch in channels_vec.iter().skip(1) {
            if !ch.dtype(py).is_equiv_to(&first_dtype) {
                return Err(PyValueError::new_err(
                    "All channels must have the same dtype for interleaving",
                ));
            }
        }

        macro_rules! interleave_for {
            ($variant:ident, $ty:ty) => {{
                let owned: Vec<AudioSamples<'static, $ty>> = channels_vec
                    .iter()
                    .map(|seg| {
                        if let PyAudioDataInner::$variant(typed) = &seg.inner {
                            typed.with_view(py, |audio| audio.clone().into_owned())
                        } else {
                            unreachable!()
                        }
                    })
                    .collect();
                let slice = NonEmptySlice::new(&owned)
                    .ok_or_else(|| PyValueError::new_err("Cannot interleave empty channel list"))?;
                let interleaved =
                    <AudioSamples<$ty> as AudioChannelOps>::interleave_channels(slice)
                        .map_err(audio_err_to_py)?;
                Ok(Self::from_audio_samples(interleaved))
            }};
        }

        match &channels_vec[0].inner {
            PyAudioDataInner::U8(_) => interleave_for!(U8, u8),
            PyAudioDataInner::I16(_) => interleave_for!(I16, i16),
            PyAudioDataInner::I24(_) => interleave_for!(I24, audio_samples::I24),
            PyAudioDataInner::I32(_) => interleave_for!(I32, i32),
            PyAudioDataInner::F32(_) => interleave_for!(F32, f32),
            PyAudioDataInner::F64(_) => interleave_for!(F64, f64),
        }
    }

    #[pyo3(signature = (), text_signature = "($self) -> list[AudioSamples]")]
    /// Split a multi-channel buffer into individual mono buffers.
    ///
    /// Channel 0 becomes element 0, channel 1 becomes element 1, and so on. For
    /// mono input a single-element list is returned. This is the inverse of
    /// :meth:`interleave_channels`.
    ///
    /// Returns:
    ///     list[AudioSamples]: One owned mono buffer per input channel.
    ///
    /// Raises:
    ///     AudioError: If channel separation fails.
    fn deinterleave_channels(&self, py: Python<'_>) -> PyResult<Vec<Self>> {
        dispatch_with_view!(self, py, |audio| {
            let channels = audio.deinterleave_channels().map_err(audio_err_to_py)?;
            Ok(channels
                .into_iter()
                .map(Self::from_audio_samples)
                .collect::<Vec<_>>())
        })
    }
}
