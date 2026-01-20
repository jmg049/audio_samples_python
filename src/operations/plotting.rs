use crate::PyAudioSamples;
use numpy::PyArray1;
use pyo3::{Bound, Python, pymethods};

#[pymethods]
impl PyAudioSamples {
    fn time_axis<'py>(&'py self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        let start = 0.0;
        let end = self.duration_seconds(py);
        let num = self.total_samples(py);
        crate::utils::audio_math::linspace(py, start, end, num)
    }
}
