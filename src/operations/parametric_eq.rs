use audio_samples::AudioParametricEq;
use numpy::{IntoPyArray, PyArray1, PyArrayMethods};
use pyo3::{Bound, PyResult, Python, pymethods};

use crate::{
    PyAudioSamples, audio_err_to_py, dispatch_with_view, dispatch_with_view_mut,
    types::{PyEqBand, PyParametricEq, PyThreeBandEqConfig},
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
            audio.apply_parametric_eq_in_place(&(*eq)).map_err(audio_err_to_py)
        })
    }

    /// Apply a parametric equalizer configuration, returning a new buffer.
    ///
    /// Non-mutating twin of :meth:`apply_parametric_eq`; leaves the current
    /// buffer unchanged.
    ///
    /// Args:
    ///     eq (ParametricEq): Equalizer definition containing one or more bands.
    ///
    /// Returns:
    ///     AudioSamples: A new equalized buffer.
    ///
    /// Raises:
    ///     AudioError: If filter application fails.
    #[pyo3(signature = (eq: "ParametricEq"), text_signature = "($self, eq: ParametricEq) -> AudioSamples")]
    fn parametric_eq(&self, py: Python<'_>, eq: PyParametricEq) -> PyResult<Self> {
        dispatch_with_view!(self, py, |audio| {
            audio
                .apply_parametric_eq(&(*eq))
                .map_err(audio_err_to_py)
                .map(|a| Self::from_audio_samples(a.into_owned()))
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
            audio.apply_eq_band_in_place(&(*band)).map_err(audio_err_to_py)
        })
    }

    /// Apply a single EQ band, returning a new buffer.
    ///
    /// Non-mutating twin of :meth:`apply_eq_band`; leaves the current buffer
    /// unchanged.
    ///
    /// Args:
    ///     band (EqBand): Parametric band definition to apply.
    ///
    /// Returns:
    ///     AudioSamples: A new buffer with the band applied.
    ///
    /// Raises:
    ///     AudioError: If filter application fails.
    #[pyo3(signature = (band: "EqBand"), text_signature = "($self, band: EqBand) -> AudioSamples")]
    fn eq_band(&self, py: Python<'_>, band: &PyEqBand) -> PyResult<Self> {
        dispatch_with_view!(self, py, |audio| {
            audio
                .apply_eq_band(&(*band))
                .map_err(audio_err_to_py)
                .map(|a| Self::from_audio_samples(a.into_owned()))
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
                .apply_peak_filter_in_place(frequency, gain_db, q_factor)
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
                .apply_low_shelf_in_place(frequency, gain_db, q_factor)
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
                .apply_high_shelf_in_place(frequency, gain_db, q_factor)
                .map_err(audio_err_to_py)
        })
    }

    /// Apply a three-band equalizer in place.
    ///
    /// Args:
    ///     config (ThreeBandEqConfig): Three-band EQ configuration describing the
    ///         low shelf, mid peak, and high shelf bands.
    ///
    /// Returns:
    ///     None: Operation mutates the current buffer.
    ///
    /// Raises:
    ///     AudioError: If filter application fails.
    #[pyo3(signature = (config: "ThreeBandEqConfig"), text_signature = "($self, config: ThreeBandEqConfig) -> None")]
    fn apply_three_band_eq(
        &mut self,
        py: Python<'_>,
        config: PyThreeBandEqConfig,
    ) -> PyResult<()> {
        dispatch_with_view_mut!(self, py, |mut audio| {
            audio
                .apply_three_band_eq_in_place(&config.inner)
                .map_err(audio_err_to_py)
        })
    }

    /// Compute the combined magnitude and phase response of a parametric EQ.
    ///
    /// Evaluates each enabled band's biquad filter at every requested frequency
    /// and combines the results (magnitudes multiplied, phases summed), applying
    /// the EQ's output gain to the combined magnitude. Disabled bands are skipped.
    /// This is purely analytical and does not modify the audio.
    ///
    /// Args:
    ///     eq (ParametricEq): The equalizer whose response to evaluate.
    ///     frequencies (np.typing.NDArray[np.float64]): Frequencies in hertz at
    ///         which to evaluate the response.
    ///
    /// Returns:
    ///     tuple[np.typing.NDArray[np.float64], np.typing.NDArray[np.float64]]:
    ///     Linear magnitude (1.0 = unity gain) and phase (radians) arrays, each
    ///     the same length as ``frequencies``.
    ///
    /// Raises:
    ///     AudioError: If any enabled band fails to design a filter (e.g. a
    ///         frequency above the Nyquist limit).
    #[pyo3(signature = (eq: "ParametricEq", frequencies: "np.typing.NDArray[np.float64]"), text_signature = "($self, eq: ParametricEq, frequencies: np.typing.NDArray[np.float64]) -> tuple[np.typing.NDArray[np.float64], np.typing.NDArray[np.float64]]")]
    fn eq_frequency_response<'py>(
        &self,
        py: Python<'py>,
        eq: PyParametricEq,
        frequencies: &Bound<'py, PyArray1<f64>>,
    ) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
        let freqs = frequencies.readonly().as_array().to_vec();
        let (magnitudes, phases) = dispatch_with_view!(self, py, |audio| {
            audio
                .eq_frequency_response(&(*eq), &freqs)
                .map_err(audio_err_to_py)
        })?;
        Ok((magnitudes.into_pyarray(py), phases.into_pyarray(py)))
    }
}
