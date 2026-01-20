use audio_samples::AudioParametricEq;
use pyo3::{PyResult, Python, pymethods};

use crate::{
    PyAudioSamples, audio_err_to_py, dispatch_with_view_mut,
    types::{PyEqBand, PyParametricEq},
};

#[pymethods]
impl PyAudioSamples {
    /// Apply a parametric equalizer configuration in place.
    ///
    /// Args:
    ///     eq (ParametricEq): Equalizer definition containing one or more bands.
    ///
    /// Returns:
    ///     None: Operation mutates the current buffer.
    ///
    /// Raises:
    ///     AudioError: If filter application fails.
    #[pyo3(signature = (eq: "ParametricEq"), text_signature = "($self, eq: ParametricEq) -> None")]
    fn apply_parametric_eq(&mut self, py: Python<'_>, eq: PyParametricEq) -> PyResult<()> {
        dispatch_with_view_mut!(self, py, |mut audio| {
            audio.apply_parametric_eq(&(*eq)).map_err(audio_err_to_py)
        })
    }

    /// Apply a single EQ band in place.
    ///
    /// Args:
    ///     band (EqBand): Parametric band definition to apply.
    ///
    /// Returns:
    ///     None: Operation mutates the current buffer.
    ///
    /// Raises:
    ///     AudioError: If filter application fails.
    #[pyo3(signature = (band: "EqBand"), text_signature = "($self, band: EqBand) -> None")]
    fn apply_eq_band(&mut self, py: Python<'_>, band: &PyEqBand) -> PyResult<()> {
        dispatch_with_view_mut!(self, py, |mut audio| {
            audio.apply_eq_band(&(*band)).map_err(audio_err_to_py)
        })
    }

    /// Apply a parametric peak filter in place.
    ///
    /// Args:
    ///     frequency (float): Center frequency in hertz.
    ///     gain_db (float): Gain adjustment in decibels.
    ///     q_factor (float): Quality factor controlling bandwidth.
    ///
    /// Returns:
    ///     None: Operation mutates the current buffer.
    ///
    /// Raises:
    ///     AudioError: If filter application fails.
    #[pyo3(signature = (frequency: "float", gain_db: "float", q_factor: "float"), text_signature = "($self, frequency: float, gain_db: float, q_factor: float) -> None")]
    fn apply_peak_filter(
        &mut self,
        py: Python<'_>,
        frequency: f64,
        gain_db: f64,
        q_factor: f64,
    ) -> PyResult<()> {
        dispatch_with_view_mut!(self, py, |mut audio| {
            audio
                .apply_peak_filter(frequency, gain_db, q_factor)
                .map_err(audio_err_to_py)
        })
    }

    /// Apply a low-shelf filter in place.
    ///
    /// Args:
    ///     frequency (float): Shelf corner frequency in hertz.
    ///     gain_db (float): Gain adjustment in decibels.
    ///     q_factor (float): Shelf slope parameter.
    ///
    /// Returns:
    ///     None: Operation mutates the current buffer.
    ///
    /// Raises:
    ///     AudioError: If filter application fails.
    #[pyo3(signature = (frequency: "float", gain_db: "float", q_factor: "float"), text_signature = "($self, frequency: float, gain_db: float, q_factor: float) -> None")]
    fn apply_low_shelf(
        &mut self,
        py: Python<'_>,
        frequency: f64,
        gain_db: f64,
        q_factor: f64,
    ) -> PyResult<()> {
        dispatch_with_view_mut!(self, py, |mut audio| {
            audio
                .apply_low_shelf(frequency, gain_db, q_factor)
                .map_err(audio_err_to_py)
        })
    }

    /// Apply a high-shelf filter in place.
    ///
    /// Args:
    ///     frequency (float): Shelf corner frequency in hertz.
    ///     gain_db (float): Gain adjustment in decibels.
    ///     q_factor (float): Shelf slope parameter.
    ///
    /// Returns:
    ///     None: Operation mutates the current buffer.
    ///
    /// Raises:
    ///     AudioError: If filter application fails.
    #[pyo3(signature = (frequency: "float", gain_db: "float", q_factor: "float"), text_signature = "($self, frequency: float, gain_db: float, q_factor: float) -> None")]
    fn apply_high_shelf(
        &mut self,
        py: Python<'_>,
        frequency: f64,
        gain_db: f64,
        q_factor: f64,
    ) -> PyResult<()> {
        dispatch_with_view_mut!(self, py, |mut audio| {
            audio
                .apply_high_shelf(frequency, gain_db, q_factor)
                .map_err(audio_err_to_py)
        })
    }

    /// Apply a three-band equalizer in place.
    ///
    /// Args:
    ///     low_freq (float): Low-shelf corner frequency in hertz.
    ///     low_gain (float): Low-shelf gain in decibels.
    ///     mid_freq (float): Mid-band center frequency in hertz.
    ///     mid_gain (float): Mid-band gain in decibels.
    ///     mid_q (float): Mid-band quality factor.
    ///     high_freq (float): High-shelf corner frequency in hertz.
    ///     high_gain (float): High-shelf gain in decibels.
    ///
    /// Returns:
    ///     None: Operation mutates the current buffer.
    ///
    /// Raises:
    ///     AudioError: If filter application fails.
    #[pyo3(signature = (low_freq: "float", low_gain:"float", mid_freq:"float", mid_gain:"float", mid_q:"float", high_freq: "float", high_gain: "float"), text_signature = "($self, low_freq: float, low_gain: float, mid_freq: float, mid_gain: float, mid_q: float, high_freq: float, high_gain: float) -> None")]
    fn apply_three_band_eq(
        &mut self,
        py: Python<'_>,
        low_freq: f64,
        low_gain: f64,
        mid_freq: f64,
        mid_gain: f64,
        mid_q: f64,
        high_freq: f64,
        high_gain: f64,
    ) -> PyResult<()> {
        dispatch_with_view_mut!(self, py, |mut audio| {
            audio
                .apply_three_band_eq(
                    low_freq, low_gain, mid_freq, mid_gain, mid_q, high_freq, high_gain,
                )
                .map_err(audio_err_to_py)
        })
    }
}
