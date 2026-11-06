use crate::PyAudioSamples;
use numpy::PyArray1;
use pyo3::{Bound, Python, pymethods};

#[pymethods]
impl PyAudioSamples {
    /// Time of each sample in seconds, one value per sample of a channel:
    /// ``numpy.arange(samples_per_channel()) / sample_rate``.
    fn time_axis<'py>(&'py self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        // n samples at times 0, 1/sr, ..., (n-1)/sr (the last sample starts one
        // period before `duration_seconds`; multi-channel audio shares the axis).
        let num = self.samples_per_channel(py);
        let sr = f64::from(self.sample_rate(py).get());
        let end = crate::args::size_f64(num.saturating_sub(1)) / sr;
        crate::utils::audio_math::linspace(py, 0.0, end, num)
    }
}
