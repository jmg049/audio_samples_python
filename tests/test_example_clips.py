"""The example clips: determinism, what the audio carries, and the contents real analysis recovers."""

from __future__ import annotations

import time

import numpy as np
import pytest

import audio_samples as aus
from audio_samples import generation

NAMES = ["chords", "drum_loop", "chords_and_drums", "speech", "noisy_recording"]

# 64-bit FNV-1a of 2 s of 16-bit little-endian PCM at 22050 Hz. The Rust test suite stores the same
# values, and the same digests were obtained on x86-64 (glibc and musl) and aarch64 builds.
PCM16_DIGESTS = {
    "chords": 0x4007_76DF_E116_B59F,
    "drum_loop": 0x29B7_D1A7_CCAD_0015,
    "chords_and_drums": 0xAE00_60E6_9853_5762,
    "speech": 0x2E0A_B50F_7D7C_1305,
    "noisy_recording": 0x7A46_1BDD_55B8_ACD0,
}


def fnv1a(data: bytes) -> int:
    value = 0xCBF2_9CE4_8422_2325
    for byte in data:
        value = ((value ^ byte) * 0x0000_0100_0000_01B3) & 0xFFFF_FFFF_FFFF_FFFF
    return value


def clip(name: str, seconds: float = 8.0, rate: int = 22050, dtype=np.float64) -> aus.AudioSamples:
    return generation.example_clip(name, seconds, rate, dtype=dtype)


def samples(audio) -> np.ndarray:
    return audio.to_numpy()


@pytest.mark.parametrize("name", NAMES)
def test_pcm_digest(name):
    pcm = samples(generation.example_clip(name, 2.0, 22050, dtype=np.int16))
    assert pcm.dtype == np.int16
    assert fnv1a(pcm.astype("<i2").tobytes()) == PCM16_DIGESTS[name]


def test_names():
    assert generation.example_clip_names() == NAMES


@pytest.mark.parametrize("name", NAMES)
def test_defaults_are_as_documented(name):
    audio = generation.example_clip(name)
    assert audio.sample_rate == 44100
    assert audio.samples_per_channel() == 8 * 44100
    assert audio.num_channels == 1
    assert samples(audio).dtype == np.float32


@pytest.mark.parametrize("dtype", [np.int16, np.int32, np.float32, np.float64, "float64"])
@pytest.mark.parametrize("name", NAMES)
def test_caller_chooses_rate_duration_and_type(name, dtype):
    audio = generation.example_clip(name, 0.5, 16000, dtype=dtype)
    assert audio.sample_rate == 16000
    assert audio.samples_per_channel() == 8000
    assert samples(audio).dtype == np.dtype(dtype)


@pytest.mark.parametrize("name", NAMES)
def test_integer_and_float_audio_are_the_same_clip(name):
    as_int = samples(generation.example_clip(name, 1.0, 16000, dtype=np.int16))
    as_float = samples(generation.example_clip(name, 1.0, 16000, dtype=np.float64))
    assert np.abs(as_int / 32767.0 - as_float).max() < 2 / 32767


@pytest.mark.parametrize("name", NAMES)
def test_shorter_clip_is_a_prefix(name):
    short, long = samples(clip(name, 3.0)), samples(clip(name, 7.0))
    assert np.array_equal(short, long[: short.size])


@pytest.mark.parametrize("rate", [8000, 16000, 22050, 44100, 48000, 96000])
@pytest.mark.parametrize("name", NAMES)
def test_peak_is_below_full_scale(name, rate):
    peak = np.abs(samples(clip(name, 7.0, rate))).max()
    assert 0.3 < peak < 0.98


@pytest.mark.parametrize(
    "kwargs",
    [
        {"name": "trumpet"},
        {"name": "chords", "duration_secs": 0.0},
        {"name": "chords", "duration_secs": -1.0},
        {"name": "chords", "duration_secs": float("nan")},
        {"name": "chords", "duration_secs": float("inf")},
        {"name": "chords", "duration_secs": 1e-6},
    ],
)
def test_bad_arguments_raise(kwargs):
    with pytest.raises(ValueError):
        generation.example_clip(**kwargs)


def test_zero_rate_and_bad_dtype_raise():
    with pytest.raises(ValueError):
        generation.example_clip("chords", 1.0, 0)
    with pytest.raises(TypeError):
        generation.example_clip("chords", 1.0, 8000, dtype=np.complex64)


def test_generation_is_fast():
    start = time.perf_counter()
    for name in NAMES:
        generation.example_clip(name)
    # Five 8 s clips at 44100 Hz. About 0.1 s on one core; the limit allows for a loaded machine.
    assert time.perf_counter() - start < 2.0


def test_drum_loop_tempo_and_onsets():
    audio = clip("drum_loop", 16.0)
    tempo, beats = audio.beat_track(units="time")
    assert abs(float(np.ravel(tempo)[0]) - generation.EXAMPLE_TEMPO_BPM) < 3.0
    expected = 60.0 / generation.EXAMPLE_TEMPO_BPM
    assert abs((beats[-1] - beats[0]) / (len(beats) - 1) - expected) < 0.002

    onsets = audio.onset_detect(units="time")
    eighth = expected / 2
    assert abs(len(onsets) - 16.0 / eighth) <= 2
    assert np.max(np.abs(onsets / eighth - np.round(onsets / eighth))) * eighth < 0.06


def test_chords_pitch_classes():
    chroma = clip("chords", 9.0).chroma_stft()
    frames_per_second = 22050 / 512
    bar = 4 * 60 / generation.EXAMPLE_TEMPO_BPM
    for i, expected in enumerate([{0, 4, 7}, {2, 7, 11}, {0, 4, 9}, {0, 5, 9}]):
        start = int((i * bar + 0.3) * frames_per_second)
        stop = int(((i + 1) * bar - 0.3) * frames_per_second)
        strongest = np.argsort(chroma[0, :, start:stop].mean(axis=1))[-3:]
        assert set(strongest.tolist()) == expected, i
    assert abs(clip("chords", 9.0).estimate_tuning()) < 0.03


def test_mix_is_the_stems_and_hpss_separates_them():
    chords, drums = samples(clip("chords", 9.0)), samples(clip("drum_loop", 9.0))
    mix = clip("chords_and_drums", 9.0)
    assert np.array_equal(samples(mix), 0.5 * chords + 0.5 * drums)

    harmonic, percussive = mix.harmonic_percussive()
    assert _snr_db(samples(harmonic), chords) > 10.0
    assert _snr_db(samples(percussive), drums) > 4.0
    assert _snr_db(samples(harmonic), chords) > _snr_db(samples(harmonic), drums) + 10.0


def _snr_db(estimate, reference):
    n = min(estimate.size, reference.size)
    estimate, reference = estimate[:n], reference[:n]
    gain = float(estimate @ reference) / float(reference @ reference)
    residual = estimate - gain * reference
    return 10 * np.log10(gain**2 * float(reference @ reference) / float(residual @ residual))


def _f0_errors(audio):
    f0, voiced, _ = audio.pyin(fmin=60.0, fmax=400.0)
    times = aus.utils.times_like(f0[0], sr=22050, hop_length=512)
    half_window = 1024 / 22050
    expected = generation.example_speech_f0(times)
    inside = ~np.isnan(generation.example_speech_f0(times - half_window)) & ~np.isnan(
        generation.example_speech_f0(times + half_window)
    )
    hits = inside & voiced[0]
    cents = 1200 * np.log2(f0[0, hits] / expected[hits])
    return np.sqrt(np.mean(cents**2)), hits.sum() / inside.sum()


def test_pyin_recovers_the_contour():
    rms_cents, coverage = _f0_errors(clip("speech", 7.0))
    assert rms_cents < 15.0
    assert coverage > 0.98


def test_pyin_recovers_the_contour_through_noise():
    rms_cents, coverage = _f0_errors(clip("noisy_recording", 7.0))
    assert rms_cents < 20.0
    assert coverage > 0.95


def test_f0_contour_marks_pauses():
    f0 = generation.example_speech_f0([0.0, 1.0, 2.7, 3.5, 5.7, -1.0])
    assert f0.dtype == np.float64
    assert np.array_equal(np.isnan(f0), [False, False, True, False, True, True])
    assert np.all((f0[~np.isnan(f0)] > 80) & (f0[~np.isnan(f0)] < 200))
    assert generation.example_speech_f0(np.array([1.0]))[0] == generation.example_speech_f0([1.0])[0]


@pytest.mark.parametrize("rate", [16000, 22050, 48000])
def test_noisy_recording_has_the_stated_snr(rate):
    speech, noisy = samples(clip("speech", 6.0, rate)), samples(clip("noisy_recording", 6.0, rate))
    noise = noisy - speech
    voiced = ~np.isnan(generation.example_speech_f0(np.arange(speech.size) / rate))
    snr = 20 * np.log10(np.sqrt(np.mean(speech[voiced] ** 2)) / np.sqrt(np.mean(noise**2)))
    assert abs(snr - generation.EXAMPLE_NOISY_SNR_DB) < 1.0

    spectrum = np.abs(np.fft.rfft(noise)) / noise.size
    frequency = np.fft.rfftfreq(noise.size, 1 / rate)
    around_hum = (frequency > 20) & (frequency < 200)
    assert abs(frequency[around_hum][np.argmax(spectrum[around_hum])] - 50.0) < 1.0


def test_speech_formants():
    # Peaks of the all-pole envelope at the centre of each vowel.
    audio = clip("speech", 4.0, 16000)
    x = samples(audio)
    cases = {0.3: (730, 1090, 2440), 0.9: (530, 1840, 2480), 1.5: (270, 2290, 3010), 2.1: (570, 840, 2410), 3.3: (300, 870, 2240)}
    for centre, formants in cases.items():
        start = int((centre - 0.1) * 16000)
        segment = aus.AudioSamples.new_mono(x[start : start + 3200] * np.hanning(3200), 16000)
        a = segment.preemphasis(coef=0.97).lpc(order=14)[0]
        response = np.abs(np.fft.rfft(a, 8192))
        envelope = 1 / response
        peaks = np.flatnonzero((envelope[1:-1] > envelope[:-2]) & (envelope[1:-1] > envelope[2:])) + 1
        peak_hz = peaks * 16000 / 8192
        for formant in formants:
            assert np.min(np.abs(peak_hz - formant)) < 0.1 * formant, (centre, formant, peak_hz)
