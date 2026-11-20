pub mod audio_math;
pub mod comparison;
pub mod detection;
pub mod generation;
pub mod tuning;

use pyo3::prelude::*;

/// Unit conversions (Hz, mel, MIDI, notes, octaves, frames, samples, time, dB), signal
/// comparison metrics and tuning helpers, with the ``generation`` and ``detection``
/// submodules.
#[pymodule(submodule, name = "utils")]
pub mod py_module {
    use pyo3::prelude::*;

    #[pymodule_export]
    use super::audio_math::{
        amplitude_to_db, blocks_to_frames, blocks_to_samples, blocks_to_time, cents_to_ratio,
        db_to_amplitude, db_to_power, fft_frequencies, frames_to_samples, frames_to_time,
        frequency_to_note, hz_to_mel, hz_to_midi, linspace, mel_frequencies, mel_scale, mel_to_hz,
        midi_to_hz, midi_to_note, ms_to_samples, mu_law_decode, mu_law_encode, note_to_frequency,
        note_to_midi, power_to_db, ratio_to_cents, samples_like, samples_to_frames,
        samples_to_time, seconds_to_samples, time_to_frames, time_to_frames_floor, time_to_samples,
        times_like,
    };
    #[pymodule_export]
    use super::comparison::{
        align_signals, correlation, correlation_per_channel, log_spectral_distance, mse,
        mse_per_channel, psnr, segmental_snr, snr, snr_per_channel,
    };
    #[pymodule_export]
    use super::detection::py_module as detection;
    #[pymodule_export]
    use super::generation::py_module as generation;
    #[pymodule_export]
    use super::tuning::{
        a4_to_tuning, cqt_frequencies, hz_to_octs, key_to_degrees, key_to_notes, octs_to_hz,
        tuning_to_a4,
    };

    #[pymodule_init]
    fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {
        // The generation and detection functions are also reachable from
        // `audio_samples.utils`, as the same objects.
        for (sub, names) in [
            ("generation", super::generation::UTILS_NAMES),
            ("detection", super::detection::UTILS_NAMES),
        ] {
            let sub = m.getattr(sub)?;
            for name in names {
                m.add(*name, sub.getattr(*name)?)?;
            }
        }
        Ok(())
    }
}
