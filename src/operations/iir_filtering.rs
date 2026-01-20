use std::ops::Deref;

use audio_samples::{AudioIirFiltering, operations::types::IirFilterDesign};
use pyo3::{PyResult, Python, exceptions::PyTypeError, pymethods};

use crate::{
    PyAudioSamples, audio_err_to_py, dispatch_with_view, dispatch_with_view_mut, nzu_or_err,
    types::{PyFilterResponse, PyIirFilterDesign},
};

#[pymethods]
impl PyAudioSamples {
    /// Apply an arbitrary IIR filter design in place.
    ///
    /// Args:
    ///     design (IirFilterDesign): Filter definition including coefficients and structure.
    ///
    /// Returns:
    ///     None: Operation mutates the current buffer.
    ///
    /// Raises:
    ///     AudioError: If filter application fails.
    #[pyo3(signature = (design: "IirFilterDesign"), text_signature = "($self, design: IirFilterDesign) -> None")]
    fn apply_iir_filter(&mut self, py: Python<'_>, design: &PyIirFilterDesign) -> PyResult<()> {
        dispatch_with_view_mut!(self, py, |mut audio| {
            let rs_design: &IirFilterDesign = design.deref();
            audio.apply_iir_filter(rs_design).map_err(audio_err_to_py)
        })
    }

    /// Apply a Butterworth low-pass filter in place.
    ///
    /// Args:
    ///     order (int): Filter order; must be greater than zero.
    ///     cutoff_frequency (float): Cutoff frequency in hertz.
    ///
    /// Returns:
    ///     None: Operation mutates the current buffer.
    ///
    /// Raises:
    ///     ValueError: If order is zero.
    ///     AudioError: If filter design or application fails.
    #[pyo3(signature = (order: "int", cutoff_frequency: "float"), text_signature = "($self, order: int, cutoff_frequency: float) -> None")]
    fn apply_butterworth_lowpass(
        &mut self,
        py: Python<'_>,
        order: usize,
        cutoff_frequency: f64,
    ) -> PyResult<()> {
        let order = nzu_or_err(order)?;
        dispatch_with_view_mut!(self, py, |mut audio| {
            audio
                .butterworth_lowpass(order, cutoff_frequency)
                .map_err(|e| PyTypeError::new_err(format!("Audio error: {e}")))?;
            Ok(())
        })
    }

    /// Apply a Butterworth high-pass filter in place.
    ///
    /// Args:
    ///     order (int): Filter order; must be greater than zero.
    ///     cutoff_frequency (float): Cutoff frequency in hertz.
    ///
    /// Returns:
    ///     None: Operation mutates the current buffer.
    ///
    /// Raises:
    ///     ValueError: If order is zero.
    ///     AudioError: If filter design or application fails.
    #[pyo3(signature = (order: "int", cutoff_frequency: "float"), text_signature = "($self, order: int, cutoff_frequency: float) -> None")]
    fn apply_butterworth_highpass(
        &mut self,
        py: Python<'_>,
        order: usize,
        cutoff_frequency: f64,
    ) -> PyResult<()> {
        let order = nzu_or_err(order)?;
        dispatch_with_view_mut!(self, py, |mut audio| {
            audio
                .butterworth_highpass(order, cutoff_frequency)
                .map_err(|e| PyTypeError::new_err(format!("Audio error: {e}")))?;
            Ok(())
        })
    }

    /// Apply a Butterworth band-pass filter in place.
    ///
    /// Args:
    ///     order (int): Filter order; must be greater than zero.
    ///     low_frequency (float): Lower cutoff frequency in hertz.
    ///     high_frequency (float): Upper cutoff frequency in hertz.
    ///
    /// Returns:
    ///     None: Operation mutates the current buffer.
    ///
    /// Raises:
    ///     ValueError: If order is zero.
    ///     AudioError: If filter design or application fails.
    #[pyo3(signature = (order: "int", low_frequency: "float", high_frequency: "float"), text_signature = "($self, order: int, low_frequency: float, high_frequency: float) -> None")]
    fn apply_butterworth_bandpass(
        &mut self,
        py: Python<'_>,
        order: usize,
        low_frequency: f64,
        high_frequency: f64,
    ) -> PyResult<()> {
        let order = nzu_or_err(order)?;
        dispatch_with_view_mut!(self, py, |mut audio| {
            audio
                .butterworth_bandpass(order, low_frequency, high_frequency)
                .map_err(|e| PyTypeError::new_err(format!("Audio error: {e}")))?;
            Ok(())
        })
    }

    /// Apply a Chebyshev Type I filter in place.
    ///
    /// Args:
    ///     order (int): Filter order; must be greater than zero.
    ///     cutoff_frequency (float): Cutoff frequency in hertz.
    ///     passband_ripple (float): Maximum ripple within the passband in decibels.
    ///     response (str): Filter response type ('lowpass', 'highpass', 'bandpass', 'bandstop').
    ///
    /// Returns:
    ///     None: Operation mutates the current buffer.
    ///
    /// Raises:
    ///     ValueError: If order is zero.
    ///     AudioError: If filter design or application fails.
    #[pyo3(signature = (order: "int", cutoff_frequency: "float", passband_ripple: "float", response: "Literal['lowpass', 'highpass', 'bandpass', 'bandstop']"), text_signature = "($self, order: int, cutoff_frequency: float, passband_ripple: float, response: Literal['lowpass', 'highpass', 'bandpass', 'bandstop']) -> None")]
    fn apply_chebyshev_i(
        &mut self,
        py: Python<'_>,
        order: usize,
        cutoff_frequency: f64,
        passband_ripple: f64,
        response: PyFilterResponse,
    ) -> PyResult<()> {
        let filter_response = response.into();
        let order = nzu_or_err(order)?;
        dispatch_with_view_mut!(self, py, |mut audio| {
            audio
                .chebyshev_i(order, cutoff_frequency, passband_ripple, filter_response)
                .map_err(|e| PyTypeError::new_err(format!("Audio error: {e}")))?;
            Ok(())
        })
    }

    /// Evaluate the frequency response of the current filter configuration.
    ///
    /// Args:
    ///     frequencies (list[float]): Frequencies in hertz at which to evaluate the response.
    ///
    /// Returns:
    ///     tuple[list[float], list[float]]: Magnitude and phase arrays at the requested frequencies.
    ///
    /// Raises:
    ///     AudioError: If the response computation fails.
    #[pyo3(signature = (frequencies: "list[float]"), text_signature = "($self, frequencies: list[float]) -> tuple[list[float], list[float]]")]
    fn frequency_response(
        &self,
        py: Python<'_>,
        frequencies: Vec<f64>,
    ) -> PyResult<(Vec<f64>, Vec<f64>)> {
        dispatch_with_view!(self, py, |audio| {
            audio
                .frequency_response(&frequencies)
                .map_err(audio_err_to_py)
        })
    }
    /// Apply a Butterworth low-pass filter in place.
    ///
    /// Args:
    ///     order (int): Filter order; must be greater than zero.
    ///     cutoff_frequency (float): Cutoff frequency in hertz.
    ///
    /// Returns:
    ///     None: Operation mutates the current buffer.
    ///
    /// Raises:
    ///     ValueError: If order is zero.
    ///     AudioError: If filter design or application fails.
    #[pyo3(signature = (order: "int", cutoff_frequency: "float"), text_signature = "($self, order: int, cutoff_frequency: float) -> None")]
    fn butterworth_lowpass(
        &mut self,
        py: Python<'_>,
        order: usize,
        cutoff_frequency: f64,
    ) -> PyResult<()> {
        let order = nzu_or_err(order)?;
        dispatch_with_view_mut!(self, py, |mut audio| {
            audio
                .butterworth_lowpass(order, cutoff_frequency)
                .map_err(audio_err_to_py)
        })
    }

    /// Apply a Butterworth high-pass filter in place.
    ///
    /// Args:
    ///     order (int): Filter order; must be greater than zero.
    ///     cutoff_frequency (float): Cutoff frequency in hertz.
    ///
    /// Returns:
    ///     None: Operation mutates the current buffer.
    ///
    /// Raises:
    ///     ValueError: If order is zero.
    ///     AudioError: If filter design or application fails.
    #[pyo3(signature = (order: "int", cutoff_frequency: "float"), text_signature = "($self, order: int, cutoff_frequency: float) -> None")]
    fn butterworth_highpass(
        &mut self,
        py: Python<'_>,
        order: usize,
        cutoff_frequency: f64,
    ) -> PyResult<()> {
        let order = nzu_or_err(order)?;
        dispatch_with_view_mut!(self, py, |mut audio| {
            audio
                .butterworth_highpass(order, cutoff_frequency)
                .map_err(audio_err_to_py)
        })
    }

    /// Apply a Butterworth band-pass filter in place.
    ///
    /// Args:
    ///     order (int): Filter order; must be greater than zero.
    ///     low_frequency (float): Lower cutoff frequency in hertz.
    ///     high_frequency (float): Upper cutoff frequency in hertz.
    ///
    /// Returns:
    ///     None: Operation mutates the current buffer.
    ///
    /// Raises:
    ///     ValueError: If order is zero.
    ///     AudioError: If filter design or application fails.
    #[pyo3(signature = (order: "int", low_frequency: "float", high_frequency: "float"), text_signature = "($self, order: int, low_frequency: float, high_frequency: float) -> None")]
    fn butterworth_bandpass(
        &mut self,
        py: Python<'_>,
        order: usize,
        low_frequency: f64,
        high_frequency: f64,
    ) -> PyResult<()> {
        let order = nzu_or_err(order)?;
        dispatch_with_view_mut!(self, py, |mut audio| {
            audio
                .butterworth_bandpass(order, low_frequency, high_frequency)
                .map_err(audio_err_to_py)
        })
    }
}
