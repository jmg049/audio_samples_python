//! Structured iteration over audio sample data.
//!
//! These bindings expose the crate's windowing and iteration APIs to Python.
//!
//! # Zero-copy limitation
//!
//! The crate also provides a zero-copy, borrowing iterator
//! (``windows_ref``) that yields views borrowing directly into the audio
//! buffer. Such borrowed, lifetime-bound views cannot be exposed to Python
//! safely, so they are intentionally **not** bound here. Every method in this
//! module instead materialises each window/frame/channel as an owned NumPy
//! array, copying data at the Python boundary.

use pyo3::{prelude::*, types::PyList};

use crate::{PyAudioSamples, dispatch_with_view, nzu_or_err};

#[pymethods]
impl PyAudioSamples {
    #[pyo3(signature = (window_size: "int", hop_size: "int"), text_signature = "($self, window_size: int, hop_size: int) -> list[numpy.ndarray]")]
    /// Split the signal into fixed-size, optionally overlapping windows.
    ///
    /// Each window covers ``window_size`` samples per channel and successive
    /// windows start ``hop_size`` samples apart, so windows overlap when
    /// ``hop_size < window_size``. Trailing data that does not fill a complete
    /// window is zero-padded (the crate's default ``PaddingMode.Zero``).
    ///
    /// Note:
    ///     The crate's zero-copy ``windows_ref`` iterator yields views that
    ///     borrow into the buffer and cannot be exposed to Python safely. This
    ///     method therefore copies each window into a new NumPy array.
    ///
    /// Args:
    ///     window_size (int): Samples per channel in each window; must be > 0.
    ///     hop_size (int): Samples between successive window starts; must be > 0.
    ///
    /// Returns:
    ///     list[numpy.ndarray]: One array per window, each shaped ``(samples,)``
    ///         for mono audio or ``(channels, samples)`` for multi-channel audio.
    ///
    /// Raises:
    ///     ValueError: If window_size or hop_size is zero.
    fn windows<'py>(
        &self,
        py: Python<'py>,
        window_size: usize,
        hop_size: usize,
    ) -> PyResult<Bound<'py, PyList>> {
        let window_size = nzu_or_err(window_size)?;
        let hop_size = nzu_or_err(hop_size)?;
        let arrays: Vec<Bound<'py, PyAny>> = dispatch_with_view!(self, py, |audio| {
            audio
                .windows(window_size, hop_size)
                .map(|w| PyAudioSamples::from_audio_samples(w).to_numpy(py))
                .collect::<PyResult<Vec<_>>>()
        })?;
        PyList::new(py, arrays)
    }

    #[pyo3(signature = (), text_signature = "($self) -> list[numpy.ndarray]")]
    /// Iterate over time-aligned frames of the signal.
    ///
    /// Each frame is a snapshot across all channels at a single time index, so
    /// the number of frames equals the number of samples per channel.
    ///
    /// Note:
    ///     The crate yields each frame as a borrowed view; this method copies
    ///     each frame into a new owned NumPy array.
    ///
    /// Returns:
    ///     list[numpy.ndarray]: One array per time index, each shaped
    ///         ``(1,)`` for mono audio or ``(channels, 1)`` for multi-channel
    ///         audio.
    fn frames<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let arrays: Vec<Bound<'py, PyAny>> = dispatch_with_view!(self, py, |audio| {
            audio
                .frames()
                .map(|f| PyAudioSamples::from_audio_samples(f.into_owned()).to_numpy(py))
                .collect::<PyResult<Vec<_>>>()
        })?;
        PyList::new(py, arrays)
    }

    #[pyo3(name = "iter_channels", signature = (), text_signature = "($self) -> list[numpy.ndarray]")]
    /// Iterate over the complete signal of each channel.
    ///
    /// Each item is the full temporal sequence of one channel, yielded in
    /// increasing channel-index order. Named ``iter_channels`` to avoid clashing
    /// with :meth:`channels`, which returns the channel count.
    ///
    /// Note:
    ///     Channel iteration copies each channel into a new owned NumPy array.
    ///
    /// Returns:
    ///     list[numpy.ndarray]: One mono array per channel, each shaped
    ///         ``(samples,)``.
    fn iter_channels<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let arrays: Vec<Bound<'py, PyAny>> = dispatch_with_view!(self, py, |audio| {
            audio
                .channels()
                .map(|c| PyAudioSamples::from_audio_samples(c).to_numpy(py))
                .collect::<PyResult<Vec<_>>>()
        })?;
        PyList::new(py, arrays)
    }
}
