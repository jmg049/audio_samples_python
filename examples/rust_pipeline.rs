//! A librosa pipeline in Rust on a real file: `y, sr = librosa.load(path)`, mel spectrogram,
//! MFCC, onsets, tempo and beats.
//!
//! ```text
//! cargo run --release --example rust_pipeline -- file.wav
//! ```
//!
//! Python's equivalent is in `docs/source/moving_from_librosa.rst`.

use audio_samples::operations::spectrum::power_to_db;
use audio_samples::operations::types::{MonoConversionMethod, ResamplingQuality};
use audio_samples::{
    AudioChannelOps, AudioProcessing, AudioRhythm, AudioSamples, AudioSpectrum, BeatTrackConfig,
    DbConfig, DbReference, MelSpectrogramConfig, MfccConfig, OnsetDetectConfig, sample_rate,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args().nth(1).ok_or("usage: rust_pipeline FILE")?;

    // y, sr = librosa.load(path): the file's rate and channels are kept unless you change them.
    let file: AudioSamples<'static, f32> = audio_samples_io::read(&path)?;
    println!(
        "{path}: {} channels, {} Hz, {} samples per channel",
        file.num_channels().get(),
        file.sample_rate().get(),
        file.samples_per_channel().get()
    );
    let y = file
        .to_mono(MonoConversionMethod::Average)?
        .resample(sample_rate!(22050), ResamplingQuality::High)?;

    // S = librosa.feature.melspectrogram(y=y, sr=sr); librosa.power_to_db(S, ref=np.max)
    let s = y.melspectrogram(&MelSpectrogramConfig::librosa())?;
    let db = DbConfig {
        reference: DbReference::Max,
        ..DbConfig::librosa_power()
    };
    let s_db = power_to_db(&s, &db)?;
    println!("mel dB: {:?}", s_db.dim());

    // librosa.feature.mfcc(y=y, sr=sr)
    let mfcc = y.mfcc_frames(&MfccConfig::librosa())?;
    println!("mfcc: {:?}", mfcc.dim());

    // librosa.onset.onset_detect(y=y, sr=sr, units="time")
    let onsets = y.onset_detect(&OnsetDetectConfig::librosa())?.times();
    println!("{} onsets", onsets.len());

    // tempo, beats = librosa.beat.beat_track(y=y, sr=sr, units="time")
    let track = y.beat_track(&BeatTrackConfig::librosa())?;
    println!(
        "tempo {:.1} BPM, {} beats",
        track.tempo,
        track.beats.times().len()
    );
    Ok(())
}
