use audio_samples::AudioDynamicRange;
use non_empty_slice::NonEmptySlice;
use numpy::{IntoPyArray, PyArray1, PyArrayMethods};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::{Bound, PyResult, Python, pymethods};

use crate::{
    PyAudioDataInner, PyAudioSamples, audio_err_to_py, dispatch_with_view, dispatch_with_view_mut,
    types::{PyCompressorConfig, PyExpanderConfig, PyGateConfig, PyLimiterConfig},
};

/// Apply a sidechain dynamics operation, matching the dtype of the main and
/// sidechain buffers and dispatching to the appropriate `AudioSamples<T>` impl.
///
/// Both the main and sidechain signals are materialised into owned
/// (`'static`-lifetime) `AudioSamples` before processing. This is required
/// because the crate's sidechain methods take `sidechain_signal: &Self` and
/// `AudioSamples<'a, T>` is *invariant* over `'a`: a borrowed numpy view and a
/// separately-borrowed sidechain view cannot be unified to the same lifetime,
/// so the operation cannot run on borrowed views in place. The closure receives
/// `(audio, &sidechain_audio)` (both owned, same concrete sample type) and must
/// produce an owned result that is wrapped into a new `PyAudioSamples`.
macro_rules! dispatch_sidechain {
    ($self_expr:expr, $sidechain:expr, $py:expr, |$audio:ident, $sc:ident| $body:expr) => {{
        let __self = $self_expr;
        let __sc = $sidechain;
        match (__self.inner(), __sc.inner()) {
            (PyAudioDataInner::U8(a), PyAudioDataInner::U8(s)) => {
                let $audio = a.with_view($py, |v| v.clone().into_owned());
                let $sc = s.with_view($py, |v| v.clone().into_owned());
                $body.map(PyAudioSamples::from_audio_samples)
            }
            (PyAudioDataInner::I16(a), PyAudioDataInner::I16(s)) => {
                let $audio = a.with_view($py, |v| v.clone().into_owned());
                let $sc = s.with_view($py, |v| v.clone().into_owned());
                $body.map(PyAudioSamples::from_audio_samples)
            }
            (PyAudioDataInner::I24(a), PyAudioDataInner::I24(s)) => {
                let $audio = a.with_view($py, |v| v.clone().into_owned());
                let $sc = s.with_view($py, |v| v.clone().into_owned());
                $body.map(PyAudioSamples::from_audio_samples)
            }
            (PyAudioDataInner::I32(a), PyAudioDataInner::I32(s)) => {
                let $audio = a.with_view($py, |v| v.clone().into_owned());
                let $sc = s.with_view($py, |v| v.clone().into_owned());
                $body.map(PyAudioSamples::from_audio_samples)
            }
            (PyAudioDataInner::F32(a), PyAudioDataInner::F32(s)) => {
                let $audio = a.with_view($py, |v| v.clone().into_owned());
                let $sc = s.with_view($py, |v| v.clone().into_owned());
                $body.map(PyAudioSamples::from_audio_samples)
            }
            (PyAudioDataInner::F64(a), PyAudioDataInner::F64(s)) => {
                let $audio = a.with_view($py, |v| v.clone().into_owned());
                let $sc = s.with_view($py, |v| v.clone().into_owned());
                $body.map(PyAudioSamples::from_audio_samples)
            }
            _ => Err(PyTypeError::new_err(
                "Main and sidechain signals must have the same dtype",
            )),
        }
    }};
}

#[pymethods]
impl PyAudioSamples {
    /// Apply a compressor configuration in place.
    ///
    /// Args:
    ///     compressor_config (CompressorConfig): Parameters controlling threshold, ratio, and timing.
    ///
    /// Returns:
    ///     None: Operation mutates the current buffer.
    ///
    /// Raises:
    ///     AudioError: If compression fails.
    #[pyo3(signature = (compressor_config: "CompressorConfig"), text_signature = "($self, compressor_config: CompressorConfig) -> None")]
    fn apply_compressor(
        &mut self,
        py: Python<'_>,
        compressor_config: PyCompressorConfig,
    ) -> PyResult<()> {
        dispatch_with_view_mut!(self, py, |mut audio| {
            audio
                .apply_compressor_in_place(&compressor_config.inner)
                .map_err(audio_err_to_py)
        })
    }

    /// Apply a limiter configuration in place.
    ///
    /// Args:
    ///     config (LimiterConfig): Parameters controlling ceiling, attack, and release.
    ///
    /// Returns:
    ///     None: Operation mutates the current buffer.
    ///
    /// Raises:
    ///     AudioError: If limiting fails.
    #[pyo3(signature = (config: "LimiterConfig"), text_signature = "($self, config: LimiterConfig) -> None")]
    fn apply_limiter(&mut self, py: Python<'_>, config: PyLimiterConfig) -> PyResult<()> {
        dispatch_with_view_mut!(self, py, |mut audio| {
            audio
                .apply_limiter_in_place(&config.into())
                .map_err(audio_err_to_py)
        })
    }

    /// Apply sidechain compression, returning a new buffer.
    ///
    /// The compressor's gain reduction is driven by the level of ``sidechain``
    /// rather than the main signal. Sidechain processing must be enabled on the
    /// configuration (set ``config.side_chain`` to an enabled SideChainConfig).
    /// Only mono-to-mono sidechain processing is supported.
    ///
    /// Note:
    ///     Unlike the other dynamics operations in this module (which mutate in
    ///     place), this returns a new buffer. The crate's sidechain method takes
    ///     ``sidechain_signal: &Self`` and ``AudioSamples`` is invariant over its
    ///     lifetime, so a borrowed numpy view and a borrowed sidechain view cannot
    ///     be unified to one lifetime; both operands are materialised as owned
    ///     copies and a new result is produced.
    ///
    /// Args:
    ///     config (CompressorConfig): Compressor parameters with sidechain enabled.
    ///     sidechain (AudioSamples): External control signal. Must match the main
    ///         signal's dtype and length.
    ///
    /// Returns:
    ///     AudioSamples: A new sidechain-compressed buffer.
    ///
    /// Raises:
    ///     TypeError: If the main and sidechain dtypes differ.
    ///     ValueError: If sidechain is not enabled, lengths differ, or either
    ///         signal is multi-channel.
    ///     AudioError: If compression fails.
    #[pyo3(signature = (config: "CompressorConfig", sidechain: "AudioSamples"), text_signature = "($self, config: CompressorConfig, sidechain: AudioSamples) -> AudioSamples")]
    fn apply_compressor_sidechain(
        &self,
        py: Python<'_>,
        config: PyCompressorConfig,
        sidechain: &PyAudioSamples,
    ) -> PyResult<Self> {
        dispatch_sidechain!(self, sidechain, py, |audio, sc| {
            audio
                .apply_compressor_sidechain(&config.inner, &sc)
                .map_err(audio_err_to_py)
        })
    }

    /// Apply sidechain limiting, returning a new buffer.
    ///
    /// The limiter's gain reduction ceiling is enforced based on the level of
    /// ``sidechain`` rather than the main signal. Sidechain processing must be
    /// enabled on the configuration. Only mono-to-mono processing is supported.
    ///
    /// Note:
    ///     Returns a new buffer rather than mutating in place, for the same
    ///     lifetime-invariance reason described on
    ///     :meth:`apply_compressor_sidechain`.
    ///
    /// Args:
    ///     config (LimiterConfig): Limiter parameters with sidechain enabled.
    ///     sidechain (AudioSamples): External control signal. Must match the main
    ///         signal's dtype and length.
    ///
    /// Returns:
    ///     AudioSamples: A new sidechain-limited buffer.
    ///
    /// Raises:
    ///     TypeError: If the main and sidechain dtypes differ.
    ///     ValueError: If sidechain is not enabled, lengths differ, or either
    ///         signal is multi-channel.
    ///     AudioError: If limiting fails.
    #[pyo3(signature = (config: "LimiterConfig", sidechain: "AudioSamples"), text_signature = "($self, config: LimiterConfig, sidechain: AudioSamples) -> AudioSamples")]
    fn apply_limiter_sidechain(
        &self,
        py: Python<'_>,
        config: PyLimiterConfig,
        sidechain: &PyAudioSamples,
    ) -> PyResult<Self> {
        let limiter_config = config.into();
        dispatch_sidechain!(self, sidechain, py, |audio, sc| {
            audio
                .apply_limiter_sidechain(&limiter_config, &sc)
                .map_err(audio_err_to_py)
        })
    }

    /// Compute the static compression input/output curve.
    ///
    /// Maps each input level through the compressor's static gain characteristic
    /// (threshold, ratio, knee) plus makeup gain. This is purely analytical and
    /// does not modify or use the audio content (no envelope following).
    ///
    /// Args:
    ///     config (CompressorConfig): Compressor parameters.
    ///     input_levels_db (np.typing.NDArray[np.float64]): Non-empty array of
    ///         input levels in dBFS to evaluate.
    ///
    /// Returns:
    ///     np.typing.NDArray[np.float64]: Output levels in dBFS, one per input,
    ///     in the same order.
    ///
    /// Raises:
    ///     ValueError: If ``input_levels_db`` is empty or the configuration is invalid.
    ///     AudioError: If the computation fails.
    #[pyo3(signature = (config: "CompressorConfig", input_levels_db: "np.typing.NDArray[np.float64]"), text_signature = "($self, config: CompressorConfig, input_levels_db: np.typing.NDArray[np.float64]) -> np.typing.NDArray[np.float64]")]
    fn get_compression_curve<'py>(
        &self,
        py: Python<'py>,
        config: PyCompressorConfig,
        input_levels_db: &Bound<'py, PyArray1<f64>>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let levels = input_levels_db.readonly().as_array().to_vec();
        let levels = NonEmptySlice::new(&levels)
            .ok_or_else(|| PyValueError::new_err("input_levels_db must be non-empty"))?;
        let curve = dispatch_with_view!(self, py, |audio| {
            audio
                .get_compression_curve(&config.inner, levels)
                .map_err(audio_err_to_py)
        })?;
        Ok(curve.into_pyarray(py))
    }

    /// Compute the per-sample gain reduction the compressor would apply.
    ///
    /// Passes the audio through the envelope follower and compression gain
    /// calculation, collecting the gain reduction (in dB) at every sample
    /// without modifying the signal. For multi-channel audio, only the first
    /// channel is analysed.
    ///
    /// Args:
    ///     config (CompressorConfig): Compressor parameters.
    ///
    /// Returns:
    ///     np.typing.NDArray[np.float64]: Gain reduction values in dB (each >= 0.0),
    ///     one per sample (first channel for multi-channel audio).
    ///
    /// Raises:
    ///     AudioError: If the configuration is invalid or processing fails.
    #[pyo3(signature = (config: "CompressorConfig"), text_signature = "($self, config: CompressorConfig) -> np.typing.NDArray[np.float64]")]
    fn get_gain_reduction<'py>(
        &self,
        py: Python<'py>,
        config: PyCompressorConfig,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let reduction = dispatch_with_view!(self, py, |audio| {
            audio
                .get_gain_reduction(&config.inner)
                .map_err(audio_err_to_py)
        })?;
        Ok(reduction.into_pyarray(py))
    }

    /// Apply a downward noise gate in place.
    ///
    /// Args:
    ///     config (GateConfig): Gate parameters controlling threshold, ratio, and timing.
    ///
    /// Returns:
    ///     None: Operation mutates the current buffer.
    ///
    /// Raises:
    ///     AudioError: If gating fails.
    #[pyo3(signature = (config: "GateConfig"), text_signature = "($self, config: GateConfig) -> None")]
    fn apply_gate(&mut self, py: Python<'_>, config: PyGateConfig) -> PyResult<()> {
        dispatch_with_view_mut!(self, py, |mut audio| {
            audio
                .apply_gate_in_place(&config.inner)
                .map_err(audio_err_to_py)
        })
    }

    /// Apply a downward expander in place.
    ///
    /// Args:
    ///     config (ExpanderConfig): Expander parameters controlling threshold, ratio,
    ///         and timing.
    ///
    /// Returns:
    ///     None: Operation mutates the current buffer.
    ///
    /// Raises:
    ///     AudioError: If expansion fails.
    #[pyo3(signature = (config: "ExpanderConfig"), text_signature = "($self, config: ExpanderConfig) -> None")]
    fn apply_expander(&mut self, py: Python<'_>, config: PyExpanderConfig) -> PyResult<()> {
        dispatch_with_view_mut!(self, py, |mut audio| {
            audio
                .apply_expander_in_place(&config.inner)
                .map_err(audio_err_to_py)
        })
    }
}
