use audio_samples::operations::iir_filtering::SosFilter;
use audio_samples::{
    AudioIirFiltering,
    operations::types::{FilterResponse, IirFilterDesign},
};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::{Bound, PyAny, PyErr, PyResult, Python, pyclass, pymethods};

use crate::{
    PyAudioSamples, audio_err_to_py, dispatch_with_view_mut, nzu_or_err,
    types::{PyIirFilterDesign, filter_response_arg},
};

#[pymethods]
impl PyAudioSamples {
    /// Apply an arbitrary IIR filter design in place.
    ///
    /// Modifies the signal in place and returns ``None``. If the ``AudioSamples``
    /// was created from a NumPy array (``new_mono``/``new_multi``), that array
    /// is modified too; pass ``arr.copy()`` to keep the original.
    ///
    /// Args:
    ///
    ///     design (IirFilterDesign): Filter definition including coefficients and structure.
    ///
    /// Returns:
    ///
    ///     None: Operation mutates the current buffer.
    ///
    /// Raises:
    ///
    ///     AudioError: If filter application fails.
    #[pyo3(signature = (design: "IirFilterDesign"))]
    fn apply_iir_filter(&mut self, py: Python<'_>, design: &PyIirFilterDesign) -> PyResult<()> {
        dispatch_with_view_mut!(self, py, |mut audio| {
            let rs_design: &IirFilterDesign = design;
            audio
                .apply_iir_filter_in_place(rs_design)
                .map_err(audio_err_to_py)
        })
    }

    /// Apply a Butterworth low-pass filter in place.
    ///
    /// Modifies the signal in place and returns ``None``. If the ``AudioSamples``
    /// was created from a NumPy array (``new_mono``/``new_multi``), that array
    /// is modified too; pass ``arr.copy()`` to keep the original.
    ///
    /// Args:
    ///
    ///     order (int): Filter order; must be greater than zero.
    ///     cutoff_frequency (float): Cutoff frequency in hertz.
    ///
    /// Returns:
    ///
    ///     None: Operation mutates the current buffer.
    ///
    /// Raises:
    ///
    ///     ValueError: If order is zero.
    ///     AudioError: If filter design or application fails.
    #[pyo3(signature = (order: "int", cutoff_frequency: "float"))]
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
                .map_err(audio_err_to_py)?;
            Ok(())
        })
    }

    /// Apply a Butterworth high-pass filter in place.
    ///
    /// Modifies the signal in place and returns ``None``. If the ``AudioSamples``
    /// was created from a NumPy array (``new_mono``/``new_multi``), that array
    /// is modified too; pass ``arr.copy()`` to keep the original.
    ///
    /// Args:
    ///
    ///     order (int): Filter order; must be greater than zero.
    ///     cutoff_frequency (float): Cutoff frequency in hertz.
    ///
    /// Returns:
    ///
    ///     None: Operation mutates the current buffer.
    ///
    /// Raises:
    ///
    ///     ValueError: If order is zero.
    ///     AudioError: If filter design or application fails.
    #[pyo3(signature = (order: "int", cutoff_frequency: "float"))]
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
                .map_err(audio_err_to_py)?;
            Ok(())
        })
    }

    /// Apply a Butterworth band-pass filter in place.
    ///
    /// Modifies the signal in place and returns ``None``. If the ``AudioSamples``
    /// was created from a NumPy array (``new_mono``/``new_multi``), that array
    /// is modified too; pass ``arr.copy()`` to keep the original.
    ///
    /// Args:
    ///
    ///     order (int): Filter order; must be greater than zero.
    ///     low_frequency (float): Lower cutoff frequency in hertz.
    ///     high_frequency (float): Upper cutoff frequency in hertz.
    ///
    /// Returns:
    ///
    ///     None: Operation mutates the current buffer.
    ///
    /// Raises:
    ///
    ///     ValueError: If order is zero.
    ///     AudioError: If filter design or application fails.
    #[pyo3(signature = (order: "int", low_frequency: "float", high_frequency: "float"))]
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
                .map_err(audio_err_to_py)?;
            Ok(())
        })
    }

    /// Apply a Chebyshev Type I filter in place.
    ///
    /// Modifies the signal in place and returns ``None``. If the ``AudioSamples``
    /// was created from a NumPy array (``new_mono``/``new_multi``), that array
    /// is modified too; pass ``arr.copy()`` to keep the original.
    ///
    /// Args:
    ///
    ///     order (int): Filter order; must be greater than zero.
    ///     cutoff_frequency (float): Cutoff frequency in hertz.
    ///     passband_ripple (float): Maximum ripple within the passband in decibels.
    ///     response (FilterResponse | str): ``'lowpass'`` or ``'highpass'``. For
    ///         band designs use :meth:`IirFilterDesign.chebyshev_i_band` with
    ///         :meth:`apply_iir_filter`.
    ///
    /// Returns:
    ///
    ///     None: Operation mutates the current buffer.
    ///
    /// Raises:
    ///
    ///     ValueError: If order is zero.
    ///     AudioError: If filter design or application fails.
    #[pyo3(signature = (order: "int", cutoff_frequency: "float", passband_ripple: "float", response: "FilterResponse | str"))]
    fn apply_chebyshev_i(
        &mut self,
        py: Python<'_>,
        order: usize,
        cutoff_frequency: f64,
        passband_ripple: f64,
        response: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let filter_response = filter_response_arg(response)?;
        let order = nzu_or_err(order)?;
        dispatch_with_view_mut!(self, py, |mut audio| {
            audio
                .chebyshev_i_in_place(order, cutoff_frequency, passband_ripple, filter_response)
                .map_err(audio_err_to_py)?;
            Ok(())
        })
    }

    /// Deprecated: always returns a flat response.
    ///
    /// Audio samples do not carry a filter. Use
    /// :meth:`IirFilterDesign.frequency_response` or
    /// :meth:`SosFilter.frequency_response` instead. Emits a
    /// ``DeprecationWarning``.
    ///
    /// Args:
    ///
    ///     frequencies (list[float]): Frequencies in hertz.
    ///
    /// Returns:
    ///
    ///     tuple[list[float], list[float]]: All-ones magnitudes and all-zero phases.
    #[pyo3(signature = (frequencies: "list[float]"))]
    #[allow(
        clippy::unused_self,
        reason = "a Python instance method, kept for compatibility"
    )]
    #[allow(
        clippy::needless_pass_by_value,
        reason = "PyO3 extracts a Python sequence into an owned Vec"
    )]
    fn frequency_response(
        &self,
        py: Python<'_>,
        frequencies: Vec<f64>,
    ) -> PyResult<(Vec<f64>, Vec<f64>)> {
        let category = py.get_type::<pyo3::exceptions::PyDeprecationWarning>();
        PyErr::warn(
            py,
            &category,
            c"AudioSamples.frequency_response() always returns a flat response; use \
IirFilterDesign.frequency_response(frequencies, sample_rate) or SosFilter.frequency_response()",
            1,
        )?;
        Ok((vec![1.0; frequencies.len()], vec![0.0; frequencies.len()]))
    }
    /// Apply a Butterworth low-pass filter in place.
    ///
    /// Modifies the signal in place and returns ``None``. If the ``AudioSamples``
    /// was created from a NumPy array (``new_mono``/``new_multi``), that array
    /// is modified too; pass ``arr.copy()`` to keep the original.
    ///
    /// Args:
    ///
    ///     order (int): Filter order; must be greater than zero.
    ///     cutoff_frequency (float): Cutoff frequency in hertz.
    ///
    /// Returns:
    ///
    ///     None: Operation mutates the current buffer.
    ///
    /// Raises:
    ///
    ///     ValueError: If order is zero.
    ///     AudioError: If filter design or application fails.
    #[pyo3(signature = (order: "int", cutoff_frequency: "float"))]
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
    /// Modifies the signal in place and returns ``None``. If the ``AudioSamples``
    /// was created from a NumPy array (``new_mono``/``new_multi``), that array
    /// is modified too; pass ``arr.copy()`` to keep the original.
    ///
    /// Args:
    ///
    ///     order (int): Filter order; must be greater than zero.
    ///     cutoff_frequency (float): Cutoff frequency in hertz.
    ///
    /// Returns:
    ///
    ///     None: Operation mutates the current buffer.
    ///
    /// Raises:
    ///
    ///     ValueError: If order is zero.
    ///     AudioError: If filter design or application fails.
    #[pyo3(signature = (order: "int", cutoff_frequency: "float"))]
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
    /// Modifies the signal in place and returns ``None``. If the ``AudioSamples``
    /// was created from a NumPy array (``new_mono``/``new_multi``), that array
    /// is modified too; pass ``arr.copy()`` to keep the original.
    ///
    /// Args:
    ///
    ///     order (int): Filter order; must be greater than zero.
    ///     low_frequency (float): Lower cutoff frequency in hertz.
    ///     high_frequency (float): Upper cutoff frequency in hertz.
    ///
    /// Returns:
    ///
    ///     None: Operation mutates the current buffer.
    ///
    /// Raises:
    ///
    ///     ValueError: If order is zero.
    ///     AudioError: If filter design or application fails.
    #[pyo3(signature = (order: "int", low_frequency: "float", high_frequency: "float"))]
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
    /// Modifies the signal in place and returns ``None``. If the ``AudioSamples``
    /// was created from a NumPy array (``new_mono``/``new_multi``), that array
    /// is modified too; pass ``arr.copy()`` to keep the original.
    ///
    /// Chebyshev Type II (inverse Chebyshev) filters have a maximally flat
    /// passband and equiripple stopband. They are specified by the stopband
    /// attenuation rather than passband ripple.
    ///
    /// Args:
    ///
    ///     order (int): Filter order; must be greater than zero.
    ///     cutoff_frequency (float): Stopband edge frequency in hertz.
    ///     stopband_attenuation (float): Minimum stopband attenuation in decibels
    ///         (typically 20-80 dB); must be positive.
    ///
    /// Returns:
    ///
    ///     None: Operation mutates the current buffer.
    ///
    /// Raises:
    ///
    ///     ValueError: If order is zero or a parameter is invalid.
    ///     AudioError: If filter design or application fails.
    #[pyo3(signature = (order: "int", cutoff_frequency: "float", stopband_attenuation: "float"))]
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
            audio
                .apply_iir_filter_in_place(&design)
                .map_err(audio_err_to_py)
        })
    }

    /// Apply a Chebyshev Type II high-pass filter in place.
    ///
    /// Modifies the signal in place and returns ``None``. If the ``AudioSamples``
    /// was created from a NumPy array (``new_mono``/``new_multi``), that array
    /// is modified too; pass ``arr.copy()`` to keep the original.
    ///
    /// Args:
    ///
    ///     order (int): Filter order; must be greater than zero.
    ///     cutoff_frequency (float): Stopband edge frequency in hertz.
    ///     stopband_attenuation (float): Minimum stopband attenuation in decibels;
    ///         must be positive.
    ///
    /// Returns:
    ///
    ///     None: Operation mutates the current buffer.
    ///
    /// Raises:
    ///
    ///     ValueError: If order is zero or a parameter is invalid.
    ///     AudioError: If filter design or application fails.
    #[pyo3(signature = (order: "int", cutoff_frequency: "float", stopband_attenuation: "float"))]
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
            audio
                .apply_iir_filter_in_place(&design)
                .map_err(audio_err_to_py)
        })
    }

    /// Apply a Chebyshev Type II band-pass filter in place.
    ///
    /// Modifies the signal in place and returns ``None``. If the ``AudioSamples``
    /// was created from a NumPy array (``new_mono``/``new_multi``), that array
    /// is modified too; pass ``arr.copy()`` to keep the original.
    ///
    /// Args:
    ///
    ///     order (int): Prototype filter order per edge; must be greater than zero.
    ///     low_frequency (float): Lower band edge in hertz.
    ///     high_frequency (float): Upper band edge in hertz.
    ///     stopband_attenuation (float): Minimum stopband attenuation in decibels;
    ///         must be positive.
    ///
    /// Returns:
    ///
    ///     None: Operation mutates the current buffer.
    ///
    /// Raises:
    ///
    ///     ValueError: If order is zero or a parameter is invalid.
    ///     AudioError: If filter design or application fails.
    #[pyo3(signature = (order: "int", low_frequency: "float", high_frequency: "float", stopband_attenuation: "float"))]
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
            audio
                .apply_iir_filter_in_place(&design)
                .map_err(audio_err_to_py)
        })
    }

    /// Apply a Chebyshev Type II band-stop filter in place.
    ///
    /// Modifies the signal in place and returns ``None``. If the ``AudioSamples``
    /// was created from a NumPy array (``new_mono``/``new_multi``), that array
    /// is modified too; pass ``arr.copy()`` to keep the original.
    ///
    /// Args:
    ///
    ///     order (int): Prototype filter order per edge; must be greater than zero.
    ///     low_frequency (float): Lower band edge in hertz.
    ///     high_frequency (float): Upper band edge in hertz.
    ///     stopband_attenuation (float): Minimum stopband attenuation in decibels;
    ///         must be positive.
    ///
    /// Returns:
    ///
    ///     None: Operation mutates the current buffer.
    ///
    /// Raises:
    ///
    ///     ValueError: If order is zero or a parameter is invalid.
    ///     AudioError: If filter design or application fails.
    #[pyo3(signature = (order: "int", low_frequency: "float", high_frequency: "float", stopband_attenuation: "float"))]
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
            audio
                .apply_iir_filter_in_place(&design)
                .map_err(audio_err_to_py)
        })
    }

    // -------------------------------------------------------------------------
    // Elliptic (Cauer) convenience filters
    // -------------------------------------------------------------------------

    /// Apply an elliptic (Cauer) low-pass filter in place.
    ///
    /// Modifies the signal in place and returns ``None``. If the ``AudioSamples``
    /// was created from a NumPy array (``new_mono``/``new_multi``), that array
    /// is modified too; pass ``arr.copy()`` to keep the original.
    ///
    /// Elliptic filters are equiripple in both passband and stopband, giving the
    /// steepest transition for a given order. They are specified by both the
    /// passband ripple and the stopband attenuation.
    ///
    /// Args:
    ///
    ///     order (int): Filter order; must be greater than zero.
    ///     cutoff_frequency (float): Passband edge frequency in hertz.
    ///     passband_ripple (float): Peak passband ripple in decibels; must be positive.
    ///     stopband_attenuation (float): Minimum stopband attenuation in decibels;
    ///         must be positive and greater than ``passband_ripple``.
    ///
    /// Returns:
    ///
    ///     None: Operation mutates the current buffer.
    ///
    /// Raises:
    ///
    ///     ValueError: If order is zero or a parameter is invalid.
    ///     AudioError: If filter design or application fails.
    #[pyo3(signature = (order: "int", cutoff_frequency: "float", passband_ripple: "float", stopband_attenuation: "float"))]
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
            audio
                .apply_iir_filter_in_place(&design)
                .map_err(audio_err_to_py)
        })
    }

    /// Apply an elliptic (Cauer) high-pass filter in place.
    ///
    /// Modifies the signal in place and returns ``None``. If the ``AudioSamples``
    /// was created from a NumPy array (``new_mono``/``new_multi``), that array
    /// is modified too; pass ``arr.copy()`` to keep the original.
    ///
    /// Args:
    ///
    ///     order (int): Filter order; must be greater than zero.
    ///     cutoff_frequency (float): Passband edge frequency in hertz.
    ///     passband_ripple (float): Peak passband ripple in decibels; must be positive.
    ///     stopband_attenuation (float): Minimum stopband attenuation in decibels;
    ///         must be positive and greater than ``passband_ripple``.
    ///
    /// Returns:
    ///
    ///     None: Operation mutates the current buffer.
    ///
    /// Raises:
    ///
    ///     ValueError: If order is zero or a parameter is invalid.
    ///     AudioError: If filter design or application fails.
    #[pyo3(signature = (order: "int", cutoff_frequency: "float", passband_ripple: "float", stopband_attenuation: "float"))]
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
            audio
                .apply_iir_filter_in_place(&design)
                .map_err(audio_err_to_py)
        })
    }

    /// Apply an elliptic (Cauer) band-pass filter in place.
    ///
    /// Modifies the signal in place and returns ``None``. If the ``AudioSamples``
    /// was created from a NumPy array (``new_mono``/``new_multi``), that array
    /// is modified too; pass ``arr.copy()`` to keep the original.
    ///
    /// Args:
    ///
    ///     order (int): Prototype filter order per edge; must be greater than zero.
    ///     low_frequency (float): Lower band edge in hertz.
    ///     high_frequency (float): Upper band edge in hertz.
    ///     passband_ripple (float): Peak passband ripple in decibels; must be positive.
    ///     stopband_attenuation (float): Minimum stopband attenuation in decibels;
    ///         must be positive and greater than ``passband_ripple``.
    ///
    /// Returns:
    ///
    ///     None: Operation mutates the current buffer.
    ///
    /// Raises:
    ///
    ///     ValueError: If order is zero or a parameter is invalid.
    ///     AudioError: If filter design or application fails.
    #[pyo3(signature = (order: "int", low_frequency: "float", high_frequency: "float", passband_ripple: "float", stopband_attenuation: "float"))]
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
            audio
                .apply_iir_filter_in_place(&design)
                .map_err(audio_err_to_py)
        })
    }

    /// Apply an elliptic (Cauer) band-stop filter in place.
    ///
    /// Modifies the signal in place and returns ``None``. If the ``AudioSamples``
    /// was created from a NumPy array (``new_mono``/``new_multi``), that array
    /// is modified too; pass ``arr.copy()`` to keep the original.
    ///
    /// Args:
    ///
    ///     order (int): Prototype filter order per edge; must be greater than zero.
    ///     low_frequency (float): Lower band edge in hertz.
    ///     high_frequency (float): Upper band edge in hertz.
    ///     passband_ripple (float): Peak passband ripple in decibels; must be positive.
    ///     stopband_attenuation (float): Minimum stopband attenuation in decibels;
    ///         must be positive and greater than ``passband_ripple``.
    ///
    /// Returns:
    ///
    ///     None: Operation mutates the current buffer.
    ///
    /// Raises:
    ///
    ///     ValueError: If order is zero or a parameter is invalid.
    ///     AudioError: If filter design or application fails.
    #[pyo3(signature = (order: "int", low_frequency: "float", high_frequency: "float", passband_ripple: "float", stopband_attenuation: "float"))]
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
            audio
                .apply_iir_filter_in_place(&design)
                .map_err(audio_err_to_py)
        })
    }

    // -------------------------------------------------------------------------
    // Bessel (Bessel-Thomson) convenience filters
    // -------------------------------------------------------------------------

    /// Apply a Bessel low-pass filter in place.
    ///
    /// Modifies the signal in place and returns ``None``. If the ``AudioSamples``
    /// was created from a NumPy array (``new_mono``/``new_multi``), that array
    /// is modified too; pass ``arr.copy()`` to keep the original.
    ///
    /// Bessel filters have a maximally flat group delay in the passband,
    /// preserving the wave shape of in-band signals. The cutoff is the -3 dB point.
    ///
    /// Args:
    ///
    ///     order (int): Filter order; must be greater than zero.
    ///     cutoff_frequency (float): -3 dB cutoff frequency in hertz.
    ///
    /// Returns:
    ///
    ///     None: Operation mutates the current buffer.
    ///
    /// Raises:
    ///
    ///     ValueError: If order is zero or a parameter is invalid.
    ///     AudioError: If filter design or application fails.
    #[pyo3(signature = (order: "int", cutoff_frequency: "float"))]
    fn apply_bessel_lowpass(
        &mut self,
        py: Python<'_>,
        order: usize,
        cutoff_frequency: f64,
    ) -> PyResult<()> {
        let order = nzu_or_err(order)?;
        let design = IirFilterDesign::bessel(FilterResponse::LowPass, order, cutoff_frequency);
        dispatch_with_view_mut!(self, py, |mut audio| {
            audio
                .apply_iir_filter_in_place(&design)
                .map_err(audio_err_to_py)
        })
    }

    /// Apply a Bessel high-pass filter in place.
    ///
    /// Modifies the signal in place and returns ``None``. If the ``AudioSamples``
    /// was created from a NumPy array (``new_mono``/``new_multi``), that array
    /// is modified too; pass ``arr.copy()`` to keep the original.
    ///
    /// Args:
    ///
    ///     order (int): Filter order; must be greater than zero.
    ///     cutoff_frequency (float): -3 dB cutoff frequency in hertz.
    ///
    /// Returns:
    ///
    ///     None: Operation mutates the current buffer.
    ///
    /// Raises:
    ///
    ///     ValueError: If order is zero or a parameter is invalid.
    ///     AudioError: If filter design or application fails.
    #[pyo3(signature = (order: "int", cutoff_frequency: "float"))]
    fn apply_bessel_highpass(
        &mut self,
        py: Python<'_>,
        order: usize,
        cutoff_frequency: f64,
    ) -> PyResult<()> {
        let order = nzu_or_err(order)?;
        let design = IirFilterDesign::bessel(FilterResponse::HighPass, order, cutoff_frequency);
        dispatch_with_view_mut!(self, py, |mut audio| {
            audio
                .apply_iir_filter_in_place(&design)
                .map_err(audio_err_to_py)
        })
    }

    /// Apply a Bessel band-pass filter in place.
    ///
    /// Modifies the signal in place and returns ``None``. If the ``AudioSamples``
    /// was created from a NumPy array (``new_mono``/``new_multi``), that array
    /// is modified too; pass ``arr.copy()`` to keep the original.
    ///
    /// Args:
    ///
    ///     order (int): Prototype filter order per edge; must be greater than zero.
    ///     low_frequency (float): Lower band edge in hertz.
    ///     high_frequency (float): Upper band edge in hertz.
    ///
    /// Returns:
    ///
    ///     None: Operation mutates the current buffer.
    ///
    /// Raises:
    ///
    ///     ValueError: If order is zero or a parameter is invalid.
    ///     AudioError: If filter design or application fails.
    #[pyo3(signature = (order: "int", low_frequency: "float", high_frequency: "float"))]
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
            audio
                .apply_iir_filter_in_place(&design)
                .map_err(audio_err_to_py)
        })
    }

    /// Apply a Bessel band-stop filter in place.
    ///
    /// Modifies the signal in place and returns ``None``. If the ``AudioSamples``
    /// was created from a NumPy array (``new_mono``/``new_multi``), that array
    /// is modified too; pass ``arr.copy()`` to keep the original.
    ///
    /// Args:
    ///
    ///     order (int): Prototype filter order per edge; must be greater than zero.
    ///     low_frequency (float): Lower band edge in hertz.
    ///     high_frequency (float): Upper band edge in hertz.
    ///
    /// Returns:
    ///
    ///     None: Operation mutates the current buffer.
    ///
    /// Raises:
    ///
    ///     ValueError: If order is zero or a parameter is invalid.
    ///     AudioError: If filter design or application fails.
    #[pyo3(signature = (order: "int", low_frequency: "float", high_frequency: "float"))]
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
            audio
                .apply_iir_filter_in_place(&design)
                .map_err(audio_err_to_py)
        })
    }

    /// Apply a zero-phase (forward-backward) IIR filter in place.
    ///
    /// Modifies the signal in place and returns ``None``. If the ``AudioSamples``
    /// was created from a NumPy array (``new_mono``/``new_multi``), that array
    /// is modified too; pass ``arr.copy()`` to keep the original.
    ///
    /// Filters the signal twice -- once forward and once backward -- so the net
    /// phase response is zero. This doubles the effective filter order and
    /// squares the magnitude response. Useful when phase distortion must be
    /// avoided (e.g. analysis or offline processing).
    ///
    /// Args:
    ///
    ///     design (IirFilterDesign): Filter definition to apply forward and backward.
    ///
    /// Returns:
    ///
    ///     None: Operation mutates the current buffer.
    ///
    /// Raises:
    ///
    ///     AudioError: If filter design or application fails.
    #[pyo3(signature = (design: "IirFilterDesign"))]
    fn filtfilt(&mut self, py: Python<'_>, design: &PyIirFilterDesign) -> PyResult<()> {
        let rs_design: &IirFilterDesign = design;
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
    ///
    ///     design (IirFilterDesign): Filter specification (type, order,
    ///         frequencies, ripple/attenuation).
    ///     sample_rate (float): Sample rate of the signal in hertz.
    ///
    /// Returns:
    ///
    ///     SosFilter: A freshly-reset streaming filter implementing the design.
    ///
    /// Raises:
    ///
    ///     AudioError: If the design is invalid (out-of-range frequency,
    ///         unsupported response, order too high, etc.).
    #[staticmethod]
    #[pyo3(signature = (design: "IirFilterDesign", sample_rate: "float"))]
    fn from_design(design: &PyIirFilterDesign, sample_rate: f64) -> PyResult<Self> {
        let rs_design: &IirFilterDesign = design;
        let inner = SosFilter::from_design(rs_design, sample_rate).map_err(audio_err_to_py)?;
        Ok(Self { inner, sample_rate })
    }

    /// Build a streaming filter from second-order-section coefficients.
    ///
    /// Accepts the ``(n_sections, 6)`` array returned by
    /// ``scipy.signal.butter(..., output='sos')`` (and friends) or by
    /// :meth:`IirFilterDesign.sos`, with rows ``[b0, b1, b2, a0, a1, a2]``.
    ///
    /// Args:
    ///
    ///     sos (np.typing.NDArray[np.float64]): Section coefficients, shape ``(n, 6)``.
    ///     sample_rate (float): Sample rate in hertz (used by :meth:`frequency_response`).
    ///
    /// Raises:
    ///
    ///     ValueError: If the array is not ``(n, 6)``, is empty, contains
    ///         non-finite values, or has a zero ``a0``.
    #[staticmethod]
    #[pyo3(signature = (sos: "np.typing.NDArray[np.float64]", sample_rate: "float"))]
    fn from_sos(sos: &Bound<'_, PyArray2<f64>>, sample_rate: f64) -> PyResult<Self> {
        let arr = sos.try_readonly()?;
        let view = arr.as_array();
        if view.ncols() != 6 {
            return Err(PyValueError::new_err(format!(
                "sos must have shape (n_sections, 6), got {:?}",
                view.shape()
            )));
        }
        let rows: Vec<[f64; 6]> = view
            .rows()
            .into_iter()
            .map(|r| {
                let mut section = [0.0; 6];
                for (dst, &src) in section.iter_mut().zip(r) {
                    *dst = src;
                }
                section
            })
            .collect();
        let inner = SosFilter::from_sos_coefficients(&rows).map_err(audio_err_to_py)?;
        Ok(Self { inner, sample_rate })
    }

    /// The filter's second-order sections as a ``(n_sections, 6)`` array
    /// (SciPy ``sos`` layout, ``a0`` normalised to 1).
    #[pyo3(signature = ())]
    fn sos<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let rows = self.inner.sos_coefficients().map_err(audio_err_to_py)?;
        let flat: Vec<f64> = rows.iter().flat_map(|r| r.iter().copied()).collect();
        let arr = ndarray::Array2::from_shape_vec((rows.len(), 6), flat)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(arr.into_pyarray(py))
    }

    /// Zero-phase forward-backward filtering of a whole signal.
    ///
    /// The signal is extended at both ends by odd reflection and each pass
    /// starts from the steady-state initial conditions for its first sample.
    /// The streaming state is neither used nor modified. Returns a new array.
    ///
    /// Args:
    ///
    ///     samples (np.typing.NDArray[np.float64]): The whole signal.
    ///
    /// Returns:
    ///
    ///     np.typing.NDArray[np.float64]: Filtered signal, same length.
    #[pyo3(signature = (samples: "np.typing.NDArray[np.float64]"))]
    fn filtfilt<'py>(
        &self,
        py: Python<'py>,
        samples: &Bound<'py, PyArray1<f64>>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let mut buf = samples.try_readonly()?.as_array().to_vec();
        self.inner
            .filtfilt_in_place(&mut buf)
            .map_err(audio_err_to_py)?;
        Ok(buf.into_pyarray(py))
    }

    /// Process a single sample through the cascade.
    ///
    /// Feeds ``x`` through each section in order; internal state is updated.
    ///
    /// Args:
    ///
    ///     x (float): Input sample.
    ///
    /// Returns:
    ///
    ///     float: The filtered output sample.
    #[pyo3(signature = (x: "float"))]
    fn process_sample(&mut self, x: f64) -> f64 {
        self.inner.process_sample(x)
    }

    /// Process an array of samples, returning a new array.
    ///
    /// Equivalent to calling :meth:`process_sample` for each input in order;
    /// state carries across the whole array.
    ///
    /// Args:
    ///
    ///     samples (np.typing.NDArray[np.float64]): Input samples.
    ///
    /// Returns:
    ///
    ///     np.typing.NDArray[np.float64]: Filtered output, same length as input.
    #[pyo3(signature = (samples: "np.typing.NDArray[np.float64]"))]
    fn process_samples<'py>(
        &mut self,
        py: Python<'py>,
        samples: &Bound<'py, PyArray1<f64>>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let input = samples.try_readonly()?.as_array().to_vec();
        let output = self.inner.process_samples(&input);
        Ok(output.into_pyarray(py))
    }

    /// Process an array of samples in place.
    ///
    /// Overwrites each element of ``samples`` with its filtered value. The input
    /// array is modified directly.
    ///
    /// Args:
    ///
    ///     samples (np.typing.NDArray[np.float64]): Input/output buffer; modified in place.
    ///
    /// Returns:
    ///
    ///     None: The provided array is modified in place.
    #[pyo3(signature = (samples: "np.typing.NDArray[np.float64]"))]
    fn process_samples_in_place(&mut self, samples: &Bound<'_, PyArray1<f64>>) -> PyResult<()> {
        process_array_in_place(&mut self.inner, samples)
    }

    /// Process a block of samples in place, retaining state across calls.
    ///
    /// Alias of :meth:`process_samples_in_place` with a name that signals the
    /// design-once / stream-many usage. Because delay lines persist, calling
    /// ``process_block`` on consecutive blocks of a signal yields exactly the
    /// same result as one call over the whole signal.
    ///
    /// Args:
    ///
    ///     block (np.typing.NDArray[np.float64]): Input/output block; modified in place.
    ///
    /// Returns:
    ///
    ///     None: The provided array is modified in place.
    #[pyo3(signature = (block: "np.typing.NDArray[np.float64]"))]
    fn process_block(&mut self, block: &Bound<'_, PyArray1<f64>>) -> PyResult<()> {
        process_array_in_place(&mut self.inner, block)
    }

    /// Reset all sections' delay lines to zero.
    ///
    /// After a reset the cascade behaves identically to a freshly built filter
    /// with the same coefficients.
    ///
    /// Returns:
    ///
    ///     None.
    #[pyo3(signature = ())]
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
    ///
    ///     frequencies (np.typing.NDArray[np.float64]): Frequencies in hertz.
    ///
    /// Returns:
    ///
    ///     tuple[np.typing.NDArray[np.float64], np.typing.NDArray[np.float64]]:
    ///     Magnitude and phase (radians) arrays, each the same length as
    ///     ``frequencies``.
    #[pyo3(signature = (frequencies: "np.typing.NDArray[np.float64]"))]
    #[allow(
        clippy::type_complexity,
        reason = "a pair of NumPy arrays, as Python receives it"
    )]
    fn frequency_response<'py>(
        &self,
        py: Python<'py>,
        frequencies: &Bound<'py, PyArray1<f64>>,
    ) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
        let freqs = frequencies.try_readonly()?.as_array().to_vec();
        let (mag, phase) = self.inner.frequency_response(&freqs, self.sample_rate);
        Ok((mag.into_pyarray(py), phase.into_pyarray(py)))
    }
}

/// Filter a 1-D float64 numpy array in place with `filter`, handling
/// non-contiguous (strided) arrays and rejecting read-only ones.
fn process_array_in_place(
    filter: &mut SosFilter,
    samples: &Bound<'_, PyArray1<f64>>,
) -> PyResult<()> {
    let mut rw = samples
        .try_readwrite()
        .map_err(|e| PyValueError::new_err(format!("cannot write to array: {e}")))?;
    if let Ok(slice) = rw.as_slice_mut() {
        filter.process_block(slice);
    } else {
        let mut view = rw.as_array_mut();
        let mut buf = view.to_vec();
        filter.process_block(&mut buf);
        for (dst, v) in view.iter_mut().zip(buf) {
            *dst = v;
        }
    }
    Ok(())
}
