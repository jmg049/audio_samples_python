use std::ops::Deref;

use audio_samples::operations::iir_filtering::SosFilter;
use audio_samples::{
    AudioIirFiltering,
    operations::types::{FilterResponse, IirFilterDesign},
};
use numpy::{IntoPyArray, PyArray1, PyArrayMethods};
use pyo3::{Bound, PyResult, Python, exceptions::PyTypeError, pyclass, pymethods};

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
            audio.apply_iir_filter_in_place(rs_design).map_err(audio_err_to_py)
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
                .butterworth_lowpass_in_place(order, cutoff_frequency)
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
                .butterworth_highpass_in_place(order, cutoff_frequency)
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
                .butterworth_bandpass_in_place(order, low_frequency, high_frequency)
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
                .chebyshev_i_in_place(order, cutoff_frequency, passband_ripple, filter_response)
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
                .butterworth_lowpass_in_place(order, cutoff_frequency)
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
                .butterworth_highpass_in_place(order, cutoff_frequency)
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
                .butterworth_bandpass_in_place(order, low_frequency, high_frequency)
                .map_err(audio_err_to_py)
        })
    }

    // -------------------------------------------------------------------------
    // Chebyshev Type II convenience filters
    // -------------------------------------------------------------------------

    /// Apply a Chebyshev Type II low-pass filter in place.
    ///
    /// Chebyshev Type II (inverse Chebyshev) filters have a maximally flat
    /// passband and equiripple stopband. They are specified by the stopband
    /// attenuation rather than passband ripple.
    ///
    /// Args:
    ///     order (int): Filter order; must be greater than zero.
    ///     cutoff_frequency (float): Stopband edge frequency in hertz.
    ///     stopband_attenuation (float): Minimum stopband attenuation in decibels
    ///         (typically 20-80 dB); must be positive.
    ///
    /// Returns:
    ///     None: Operation mutates the current buffer.
    ///
    /// Raises:
    ///     ValueError: If order is zero or a parameter is invalid.
    ///     AudioError: If filter design or application fails.
    #[pyo3(signature = (order: "int", cutoff_frequency: "float", stopband_attenuation: "float"), text_signature = "($self, order: int, cutoff_frequency: float, stopband_attenuation: float) -> None")]
    fn apply_chebyshev_ii_lowpass(
        &mut self,
        py: Python<'_>,
        order: usize,
        cutoff_frequency: f64,
        stopband_attenuation: f64,
    ) -> PyResult<()> {
        let order = nzu_or_err(order)?;
        let design = IirFilterDesign::chebyshev_ii(
            FilterResponse::LowPass,
            order,
            cutoff_frequency,
            stopband_attenuation,
        );
        dispatch_with_view_mut!(self, py, |mut audio| {
            audio.apply_iir_filter_in_place(&design).map_err(audio_err_to_py)
        })
    }

    /// Apply a Chebyshev Type II high-pass filter in place.
    ///
    /// Args:
    ///     order (int): Filter order; must be greater than zero.
    ///     cutoff_frequency (float): Stopband edge frequency in hertz.
    ///     stopband_attenuation (float): Minimum stopband attenuation in decibels;
    ///         must be positive.
    ///
    /// Returns:
    ///     None: Operation mutates the current buffer.
    ///
    /// Raises:
    ///     ValueError: If order is zero or a parameter is invalid.
    ///     AudioError: If filter design or application fails.
    #[pyo3(signature = (order: "int", cutoff_frequency: "float", stopband_attenuation: "float"), text_signature = "($self, order: int, cutoff_frequency: float, stopband_attenuation: float) -> None")]
    fn apply_chebyshev_ii_highpass(
        &mut self,
        py: Python<'_>,
        order: usize,
        cutoff_frequency: f64,
        stopband_attenuation: f64,
    ) -> PyResult<()> {
        let order = nzu_or_err(order)?;
        let design = IirFilterDesign::chebyshev_ii(
            FilterResponse::HighPass,
            order,
            cutoff_frequency,
            stopband_attenuation,
        );
        dispatch_with_view_mut!(self, py, |mut audio| {
            audio.apply_iir_filter_in_place(&design).map_err(audio_err_to_py)
        })
    }

    /// Apply a Chebyshev Type II band-pass filter in place.
    ///
    /// Args:
    ///     order (int): Prototype filter order per edge; must be greater than zero.
    ///     low_frequency (float): Lower band edge in hertz.
    ///     high_frequency (float): Upper band edge in hertz.
    ///     stopband_attenuation (float): Minimum stopband attenuation in decibels;
    ///         must be positive.
    ///
    /// Returns:
    ///     None: Operation mutates the current buffer.
    ///
    /// Raises:
    ///     ValueError: If order is zero or a parameter is invalid.
    ///     AudioError: If filter design or application fails.
    #[pyo3(signature = (order: "int", low_frequency: "float", high_frequency: "float", stopband_attenuation: "float"), text_signature = "($self, order: int, low_frequency: float, high_frequency: float, stopband_attenuation: float) -> None")]
    fn apply_chebyshev_ii_bandpass(
        &mut self,
        py: Python<'_>,
        order: usize,
        low_frequency: f64,
        high_frequency: f64,
        stopband_attenuation: f64,
    ) -> PyResult<()> {
        let order = nzu_or_err(order)?;
        let design = IirFilterDesign::chebyshev_ii_band(
            FilterResponse::BandPass,
            order,
            low_frequency,
            high_frequency,
            stopband_attenuation,
        );
        dispatch_with_view_mut!(self, py, |mut audio| {
            audio.apply_iir_filter_in_place(&design).map_err(audio_err_to_py)
        })
    }

    /// Apply a Chebyshev Type II band-stop filter in place.
    ///
    /// Args:
    ///     order (int): Prototype filter order per edge; must be greater than zero.
    ///     low_frequency (float): Lower band edge in hertz.
    ///     high_frequency (float): Upper band edge in hertz.
    ///     stopband_attenuation (float): Minimum stopband attenuation in decibels;
    ///         must be positive.
    ///
    /// Returns:
    ///     None: Operation mutates the current buffer.
    ///
    /// Raises:
    ///     ValueError: If order is zero or a parameter is invalid.
    ///     AudioError: If filter design or application fails.
    #[pyo3(signature = (order: "int", low_frequency: "float", high_frequency: "float", stopband_attenuation: "float"), text_signature = "($self, order: int, low_frequency: float, high_frequency: float, stopband_attenuation: float) -> None")]
    fn apply_chebyshev_ii_bandstop(
        &mut self,
        py: Python<'_>,
        order: usize,
        low_frequency: f64,
        high_frequency: f64,
        stopband_attenuation: f64,
    ) -> PyResult<()> {
        let order = nzu_or_err(order)?;
        let design = IirFilterDesign::chebyshev_ii_band(
            FilterResponse::BandStop,
            order,
            low_frequency,
            high_frequency,
            stopband_attenuation,
        );
        dispatch_with_view_mut!(self, py, |mut audio| {
            audio.apply_iir_filter_in_place(&design).map_err(audio_err_to_py)
        })
    }

    // -------------------------------------------------------------------------
    // Elliptic (Cauer) convenience filters
    // -------------------------------------------------------------------------

    /// Apply an elliptic (Cauer) low-pass filter in place.
    ///
    /// Elliptic filters are equiripple in both passband and stopband, giving the
    /// steepest transition for a given order. They are specified by both the
    /// passband ripple and the stopband attenuation.
    ///
    /// Args:
    ///     order (int): Filter order; must be greater than zero.
    ///     cutoff_frequency (float): Passband edge frequency in hertz.
    ///     passband_ripple (float): Peak passband ripple in decibels; must be positive.
    ///     stopband_attenuation (float): Minimum stopband attenuation in decibels;
    ///         must be positive and greater than ``passband_ripple``.
    ///
    /// Returns:
    ///     None: Operation mutates the current buffer.
    ///
    /// Raises:
    ///     ValueError: If order is zero or a parameter is invalid.
    ///     AudioError: If filter design or application fails.
    #[pyo3(signature = (order: "int", cutoff_frequency: "float", passband_ripple: "float", stopband_attenuation: "float"), text_signature = "($self, order: int, cutoff_frequency: float, passband_ripple: float, stopband_attenuation: float) -> None")]
    fn apply_elliptic_lowpass(
        &mut self,
        py: Python<'_>,
        order: usize,
        cutoff_frequency: f64,
        passband_ripple: f64,
        stopband_attenuation: f64,
    ) -> PyResult<()> {
        let order = nzu_or_err(order)?;
        let design = IirFilterDesign::elliptic(
            FilterResponse::LowPass,
            order,
            cutoff_frequency,
            passband_ripple,
            stopband_attenuation,
        );
        dispatch_with_view_mut!(self, py, |mut audio| {
            audio.apply_iir_filter_in_place(&design).map_err(audio_err_to_py)
        })
    }

    /// Apply an elliptic (Cauer) high-pass filter in place.
    ///
    /// Args:
    ///     order (int): Filter order; must be greater than zero.
    ///     cutoff_frequency (float): Passband edge frequency in hertz.
    ///     passband_ripple (float): Peak passband ripple in decibels; must be positive.
    ///     stopband_attenuation (float): Minimum stopband attenuation in decibels;
    ///         must be positive and greater than ``passband_ripple``.
    ///
    /// Returns:
    ///     None: Operation mutates the current buffer.
    ///
    /// Raises:
    ///     ValueError: If order is zero or a parameter is invalid.
    ///     AudioError: If filter design or application fails.
    #[pyo3(signature = (order: "int", cutoff_frequency: "float", passband_ripple: "float", stopband_attenuation: "float"), text_signature = "($self, order: int, cutoff_frequency: float, passband_ripple: float, stopband_attenuation: float) -> None")]
    fn apply_elliptic_highpass(
        &mut self,
        py: Python<'_>,
        order: usize,
        cutoff_frequency: f64,
        passband_ripple: f64,
        stopband_attenuation: f64,
    ) -> PyResult<()> {
        let order = nzu_or_err(order)?;
        let design = IirFilterDesign::elliptic(
            FilterResponse::HighPass,
            order,
            cutoff_frequency,
            passband_ripple,
            stopband_attenuation,
        );
        dispatch_with_view_mut!(self, py, |mut audio| {
            audio.apply_iir_filter_in_place(&design).map_err(audio_err_to_py)
        })
    }

    /// Apply an elliptic (Cauer) band-pass filter in place.
    ///
    /// Args:
    ///     order (int): Prototype filter order per edge; must be greater than zero.
    ///     low_frequency (float): Lower band edge in hertz.
    ///     high_frequency (float): Upper band edge in hertz.
    ///     passband_ripple (float): Peak passband ripple in decibels; must be positive.
    ///     stopband_attenuation (float): Minimum stopband attenuation in decibels;
    ///         must be positive and greater than ``passband_ripple``.
    ///
    /// Returns:
    ///     None: Operation mutates the current buffer.
    ///
    /// Raises:
    ///     ValueError: If order is zero or a parameter is invalid.
    ///     AudioError: If filter design or application fails.
    #[pyo3(signature = (order: "int", low_frequency: "float", high_frequency: "float", passband_ripple: "float", stopband_attenuation: "float"), text_signature = "($self, order: int, low_frequency: float, high_frequency: float, passband_ripple: float, stopband_attenuation: float) -> None")]
    fn apply_elliptic_bandpass(
        &mut self,
        py: Python<'_>,
        order: usize,
        low_frequency: f64,
        high_frequency: f64,
        passband_ripple: f64,
        stopband_attenuation: f64,
    ) -> PyResult<()> {
        let order = nzu_or_err(order)?;
        let design = IirFilterDesign::elliptic_band(
            FilterResponse::BandPass,
            order,
            low_frequency,
            high_frequency,
            passband_ripple,
            stopband_attenuation,
        );
        dispatch_with_view_mut!(self, py, |mut audio| {
            audio.apply_iir_filter_in_place(&design).map_err(audio_err_to_py)
        })
    }

    /// Apply an elliptic (Cauer) band-stop filter in place.
    ///
    /// Args:
    ///     order (int): Prototype filter order per edge; must be greater than zero.
    ///     low_frequency (float): Lower band edge in hertz.
    ///     high_frequency (float): Upper band edge in hertz.
    ///     passband_ripple (float): Peak passband ripple in decibels; must be positive.
    ///     stopband_attenuation (float): Minimum stopband attenuation in decibels;
    ///         must be positive and greater than ``passband_ripple``.
    ///
    /// Returns:
    ///     None: Operation mutates the current buffer.
    ///
    /// Raises:
    ///     ValueError: If order is zero or a parameter is invalid.
    ///     AudioError: If filter design or application fails.
    #[pyo3(signature = (order: "int", low_frequency: "float", high_frequency: "float", passband_ripple: "float", stopband_attenuation: "float"), text_signature = "($self, order: int, low_frequency: float, high_frequency: float, passband_ripple: float, stopband_attenuation: float) -> None")]
    fn apply_elliptic_bandstop(
        &mut self,
        py: Python<'_>,
        order: usize,
        low_frequency: f64,
        high_frequency: f64,
        passband_ripple: f64,
        stopband_attenuation: f64,
    ) -> PyResult<()> {
        let order = nzu_or_err(order)?;
        let design = IirFilterDesign::elliptic_band(
            FilterResponse::BandStop,
            order,
            low_frequency,
            high_frequency,
            passband_ripple,
            stopband_attenuation,
        );
        dispatch_with_view_mut!(self, py, |mut audio| {
            audio.apply_iir_filter_in_place(&design).map_err(audio_err_to_py)
        })
    }

    // -------------------------------------------------------------------------
    // Bessel (Bessel-Thomson) convenience filters
    // -------------------------------------------------------------------------

    /// Apply a Bessel low-pass filter in place.
    ///
    /// Bessel filters have a maximally flat group delay in the passband,
    /// preserving the wave shape of in-band signals. The cutoff is the -3 dB point.
    ///
    /// Args:
    ///     order (int): Filter order; must be greater than zero.
    ///     cutoff_frequency (float): -3 dB cutoff frequency in hertz.
    ///
    /// Returns:
    ///     None: Operation mutates the current buffer.
    ///
    /// Raises:
    ///     ValueError: If order is zero or a parameter is invalid.
    ///     AudioError: If filter design or application fails.
    #[pyo3(signature = (order: "int", cutoff_frequency: "float"), text_signature = "($self, order: int, cutoff_frequency: float) -> None")]
    fn apply_bessel_lowpass(
        &mut self,
        py: Python<'_>,
        order: usize,
        cutoff_frequency: f64,
    ) -> PyResult<()> {
        let order = nzu_or_err(order)?;
        let design = IirFilterDesign::bessel(FilterResponse::LowPass, order, cutoff_frequency);
        dispatch_with_view_mut!(self, py, |mut audio| {
            audio.apply_iir_filter_in_place(&design).map_err(audio_err_to_py)
        })
    }

    /// Apply a Bessel high-pass filter in place.
    ///
    /// Args:
    ///     order (int): Filter order; must be greater than zero.
    ///     cutoff_frequency (float): -3 dB cutoff frequency in hertz.
    ///
    /// Returns:
    ///     None: Operation mutates the current buffer.
    ///
    /// Raises:
    ///     ValueError: If order is zero or a parameter is invalid.
    ///     AudioError: If filter design or application fails.
    #[pyo3(signature = (order: "int", cutoff_frequency: "float"), text_signature = "($self, order: int, cutoff_frequency: float) -> None")]
    fn apply_bessel_highpass(
        &mut self,
        py: Python<'_>,
        order: usize,
        cutoff_frequency: f64,
    ) -> PyResult<()> {
        let order = nzu_or_err(order)?;
        let design = IirFilterDesign::bessel(FilterResponse::HighPass, order, cutoff_frequency);
        dispatch_with_view_mut!(self, py, |mut audio| {
            audio.apply_iir_filter_in_place(&design).map_err(audio_err_to_py)
        })
    }

    /// Apply a Bessel band-pass filter in place.
    ///
    /// Args:
    ///     order (int): Prototype filter order per edge; must be greater than zero.
    ///     low_frequency (float): Lower band edge in hertz.
    ///     high_frequency (float): Upper band edge in hertz.
    ///
    /// Returns:
    ///     None: Operation mutates the current buffer.
    ///
    /// Raises:
    ///     ValueError: If order is zero or a parameter is invalid.
    ///     AudioError: If filter design or application fails.
    #[pyo3(signature = (order: "int", low_frequency: "float", high_frequency: "float"), text_signature = "($self, order: int, low_frequency: float, high_frequency: float) -> None")]
    fn apply_bessel_bandpass(
        &mut self,
        py: Python<'_>,
        order: usize,
        low_frequency: f64,
        high_frequency: f64,
    ) -> PyResult<()> {
        let order = nzu_or_err(order)?;
        let design = IirFilterDesign::bessel_band(
            FilterResponse::BandPass,
            order,
            low_frequency,
            high_frequency,
        );
        dispatch_with_view_mut!(self, py, |mut audio| {
            audio.apply_iir_filter_in_place(&design).map_err(audio_err_to_py)
        })
    }

    /// Apply a Bessel band-stop filter in place.
    ///
    /// Args:
    ///     order (int): Prototype filter order per edge; must be greater than zero.
    ///     low_frequency (float): Lower band edge in hertz.
    ///     high_frequency (float): Upper band edge in hertz.
    ///
    /// Returns:
    ///     None: Operation mutates the current buffer.
    ///
    /// Raises:
    ///     ValueError: If order is zero or a parameter is invalid.
    ///     AudioError: If filter design or application fails.
    #[pyo3(signature = (order: "int", low_frequency: "float", high_frequency: "float"), text_signature = "($self, order: int, low_frequency: float, high_frequency: float) -> None")]
    fn apply_bessel_bandstop(
        &mut self,
        py: Python<'_>,
        order: usize,
        low_frequency: f64,
        high_frequency: f64,
    ) -> PyResult<()> {
        let order = nzu_or_err(order)?;
        let design = IirFilterDesign::bessel_band(
            FilterResponse::BandStop,
            order,
            low_frequency,
            high_frequency,
        );
        dispatch_with_view_mut!(self, py, |mut audio| {
            audio.apply_iir_filter_in_place(&design).map_err(audio_err_to_py)
        })
    }

    /// Apply a zero-phase (forward-backward) IIR filter in place.
    ///
    /// Filters the signal twice -- once forward and once backward -- so the net
    /// phase response is zero. This doubles the effective filter order and
    /// squares the magnitude response. Useful when phase distortion must be
    /// avoided (e.g. analysis or offline processing).
    ///
    /// Args:
    ///     design (IirFilterDesign): Filter definition to apply forward and backward.
    ///
    /// Returns:
    ///     None: Operation mutates the current buffer.
    ///
    /// Raises:
    ///     AudioError: If filter design or application fails.
    #[pyo3(signature = (design: "IirFilterDesign"), text_signature = "($self, design: IirFilterDesign) -> None")]
    fn filtfilt(&mut self, py: Python<'_>, design: &PyIirFilterDesign) -> PyResult<()> {
        let rs_design: &IirFilterDesign = design.deref();
        dispatch_with_view_mut!(self, py, |mut audio| {
            audio.filtfilt_in_place(rs_design).map_err(audio_err_to_py)
        })
    }
}

/// Streaming second-order-sections (SOS) IIR filter.
///
/// A `SosFilter` is a cascade of biquad sections built once from an
/// :class:`IirFilterDesign` and then driven sample-by-sample or block-by-block.
/// Internal delay-line state persists across calls, so processing consecutive
/// blocks produces exactly the same result as processing the whole signal at
/// once. This makes it suitable for real-time and streaming use where redesigning
/// the filter for every block would be wasteful.
///
/// Build one with :meth:`from_design`; the sample rate is fixed at construction
/// time and used for all frequency-dependent computations.
#[pyclass(name = "SosFilter", module = "audio_samples.types")]
pub struct PySosFilter {
    inner: SosFilter,
    sample_rate: f64,
}

#[pymethods]
impl PySosFilter {
    /// Build a streaming SOS filter from a filter design.
    ///
    /// Designs the filter once and returns a stateful cascade. The returned
    /// filter starts with zeroed delay lines.
    ///
    /// Args:
    ///     design (IirFilterDesign): Filter specification (type, order,
    ///         frequencies, ripple/attenuation).
    ///     sample_rate (float): Sample rate of the signal in hertz.
    ///
    /// Returns:
    ///     SosFilter: A freshly-reset streaming filter implementing the design.
    ///
    /// Raises:
    ///     AudioError: If the design is invalid (out-of-range frequency,
    ///         unsupported response, order too high, etc.).
    #[staticmethod]
    #[pyo3(signature = (design: "IirFilterDesign", sample_rate: "float"), text_signature = "(design: IirFilterDesign, sample_rate: float) -> SosFilter")]
    fn from_design(design: &PyIirFilterDesign, sample_rate: f64) -> PyResult<Self> {
        let rs_design: &IirFilterDesign = design.deref();
        let inner = SosFilter::from_design(rs_design, sample_rate).map_err(audio_err_to_py)?;
        Ok(Self { inner, sample_rate })
    }

    /// Process a single sample through the cascade.
    ///
    /// Feeds ``x`` through each section in order; internal state is updated.
    ///
    /// Args:
    ///     x (float): Input sample.
    ///
    /// Returns:
    ///     float: The filtered output sample.
    #[pyo3(signature = (x: "float"), text_signature = "($self, x: float) -> float")]
    fn process_sample(&mut self, x: f64) -> f64 {
        self.inner.process_sample(x)
    }

    /// Process an array of samples, returning a new array.
    ///
    /// Equivalent to calling :meth:`process_sample` for each input in order;
    /// state carries across the whole array.
    ///
    /// Args:
    ///     samples (np.typing.NDArray[np.float64]): Input samples.
    ///
    /// Returns:
    ///     np.typing.NDArray[np.float64]: Filtered output, same length as input.
    #[pyo3(signature = (samples: "np.typing.NDArray[np.float64]"), text_signature = "($self, samples: np.typing.NDArray[np.float64]) -> np.typing.NDArray[np.float64]")]
    fn process_samples<'py>(
        &mut self,
        py: Python<'py>,
        samples: &Bound<'py, PyArray1<f64>>,
    ) -> Bound<'py, PyArray1<f64>> {
        let input = samples.readonly().as_array().to_vec();
        let output = self.inner.process_samples(&input);
        output.into_pyarray(py)
    }

    /// Process an array of samples in place.
    ///
    /// Overwrites each element of ``samples`` with its filtered value. The input
    /// array is modified directly.
    ///
    /// Args:
    ///     samples (np.typing.NDArray[np.float64]): Input/output buffer; modified in place.
    ///
    /// Returns:
    ///     None: The provided array is modified in place.
    #[pyo3(signature = (samples: "np.typing.NDArray[np.float64]"), text_signature = "($self, samples: np.typing.NDArray[np.float64]) -> None")]
    fn process_samples_in_place(&mut self, samples: &Bound<'_, PyArray1<f64>>) {
        let mut rw = samples.readwrite();
        let slice = rw.as_slice_mut().expect("contiguous numpy array");
        self.inner.process_samples_in_place(slice);
    }

    /// Process a block of samples in place, retaining state across calls.
    ///
    /// Alias of :meth:`process_samples_in_place` with a name that signals the
    /// design-once / stream-many usage. Because delay lines persist, calling
    /// ``process_block`` on consecutive blocks of a signal yields exactly the
    /// same result as one call over the whole signal.
    ///
    /// Args:
    ///     block (np.typing.NDArray[np.float64]): Input/output block; modified in place.
    ///
    /// Returns:
    ///     None: The provided array is modified in place.
    #[pyo3(signature = (block: "np.typing.NDArray[np.float64]"), text_signature = "($self, block: np.typing.NDArray[np.float64]) -> None")]
    fn process_block(&mut self, block: &Bound<'_, PyArray1<f64>>) {
        let mut rw = block.readwrite();
        let slice = rw.as_slice_mut().expect("contiguous numpy array");
        self.inner.process_block(slice);
    }

    /// Reset all sections' delay lines to zero.
    ///
    /// After a reset the cascade behaves identically to a freshly built filter
    /// with the same coefficients.
    ///
    /// Returns:
    ///     None.
    #[pyo3(signature = (), text_signature = "($self) -> None")]
    fn reset(&mut self) {
        self.inner.reset();
    }

    /// Compute the frequency response of the cascade.
    ///
    /// Evaluates the combined transfer function of all sections. The magnitude
    /// is the product of section magnitudes; the phase is the sum of section
    /// phases. The sample rate fixed at construction is used.
    ///
    /// Args:
    ///     frequencies (np.typing.NDArray[np.float64]): Frequencies in hertz.
    ///
    /// Returns:
    ///     tuple[np.typing.NDArray[np.float64], np.typing.NDArray[np.float64]]:
    ///     Magnitude and phase (radians) arrays, each the same length as
    ///     ``frequencies``.
    #[pyo3(signature = (frequencies: "np.typing.NDArray[np.float64]"), text_signature = "($self, frequencies: np.typing.NDArray[np.float64]) -> tuple[np.typing.NDArray[np.float64], np.typing.NDArray[np.float64]]")]
    fn frequency_response<'py>(
        &self,
        py: Python<'py>,
        frequencies: &Bound<'py, PyArray1<f64>>,
    ) -> (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>) {
        let freqs = frequencies.readonly().as_array().to_vec();
        let (mag, phase) = self.inner.frequency_response(&freqs, self.sample_rate);
        (mag.into_pyarray(py), phase.into_pyarray(py))
    }
}
