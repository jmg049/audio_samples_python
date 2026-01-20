use audio_samples::{AudioChannelOps, ConvertTo};
use pyo3::prelude::*;

use crate::{
    PyAudioSamples, audio_err_to_py, dispatch_with_view, dispatch_with_view_mut,
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
    fn extract_channel(&self, py: Python<'_>, channel_index: u32) -> PyResult<Self> {
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
    fn swap_channels(&mut self, py: Python<'_>, channel1: u32, channel2: u32) -> PyResult<()> {
        dispatch_with_view_mut!(self, py, |mut audio| {
            audio
                .swap_channels(channel1, channel2)
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
            audio.pan(pan_value.convert_to()).map_err(audio_err_to_py)
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
            audio.balance(balance).map_err(audio_err_to_py)
        })
    }
}
