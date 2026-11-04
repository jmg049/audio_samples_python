"""Typed use of the public API. mypy and pyright check this file, and pytest runs it.

``assert_type`` makes the checkers compare the static type, and at run time ``check`` compares the dtype,
so the stubs cannot claim a precision that the library does not deliver.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from typing_extensions import assert_type

import audio_samples as aus
from audio_samples import effects, features, generation, rhythm, sequence, spectral, utils


def check(array: NDArray[Any], dtype: type[np.generic]) -> None:
    assert array.dtype == np.dtype(dtype), (array.dtype, dtype)


def audio_keeps_its_sample_type() -> None:
    default = generation.sine_wave(440.0, 0.5, 22050)
    assert_type(default, aus.AudioSamples[np.float32])
    check(default.to_numpy(), np.float32)

    f64 = generation.sine_wave(440.0, 0.5, 22050, 0.5, dtype=np.float64)
    assert_type(f64, aus.AudioSamples[np.float64])
    check(f64.to_numpy(), np.float64)
    assert_type(f64.to_numpy(), NDArray[np.float64])

    positional = generation.sine_wave(440.0, 0.5, 22050, 0.5, np.int16)
    assert_type(positional, aus.AudioSamples[np.int16])
    check(positional.to_numpy(), np.int16)

    wrapped = aus.AudioSamples.new_mono(np.zeros(8, dtype=np.int32), 8000)
    assert_type(wrapped, aus.AudioSamples[np.int32])
    check(wrapped.to_numpy(), np.int32)

    resampled = f64.resample(11025)
    assert_type(resampled, aus.AudioSamples[np.float64])
    check(resampled.to_numpy(), np.float64)

    converted = f64.as_i16()
    assert_type(converted, aus.AudioSamples[np.int16])
    check(converted.to_numpy(), np.int16)
    assert_type(f64 + 1.0, aus.AudioSamples[np.float64])

    # The sample type comes from the dtype argument, which a string does not state statically.
    named = generation.sine_wave(440.0, 0.5, 22050, dtype="float64")
    assert_type(named, aus.AudioSamples[Any])
    check(named.to_numpy(), np.float64)


def clips_follow_the_dtype() -> None:
    clip = generation.example_clip("speech", 0.5, 8000, dtype=np.int16)
    assert_type(clip, aus.AudioSamples[np.int16])
    check(clip.to_numpy(), np.int16)
    assert_type(generation.example_clip("chords"), aus.AudioSamples[np.float32])
    assert_type(generation.example_clip_names(), list[str])
    assert_type(generation.example_speech_f0([0.5]), NDArray[np.float64])


def analysis_precision_follows_the_sample_type() -> None:
    i16 = generation.sine_wave(440.0, 0.5, 22050, 0.5, dtype=np.int16)
    f32 = generation.sine_wave(440.0, 0.5, 22050, 0.5, dtype=np.float32)
    i32 = generation.sine_wave(440.0, 0.5, 22050, 0.5, dtype=np.int32)
    f64 = generation.sine_wave(440.0, 0.5, 22050, 0.5, dtype=np.float64)

    single_i16 = i16.melspectrogram()
    assert_type(single_i16, NDArray[np.float32])
    check(single_i16, np.float32)
    single_f32 = f32.rms_frames()
    assert_type(single_f32, NDArray[np.float32])
    check(single_f32, np.float32)
    complex_single = f32.stft_matrix()
    assert_type(complex_single, NDArray[np.complex64])
    check(complex_single, np.complex64)

    double_i32 = i32.melspectrogram()
    assert_type(double_i32, NDArray[np.float64])
    check(double_i32, np.float64)
    double_f64 = f64.rms_frames()
    assert_type(double_f64, NDArray[np.float64])
    check(double_f64, np.float64)
    complex_double = f64.stft_matrix()
    assert_type(complex_double, NDArray[np.complex128])
    check(complex_double, np.complex128)

    # The envelopes keep the sample type.
    assert_type(i16.amplitude_envelope(), NDArray[np.int16])
    check(i16.amplitude_envelope(), np.int16)

    # The sample type of audio from a file is known only at run time.
    unknown: aus.AudioSamples = f64
    assert_type(unknown.to_numpy(), NDArray[Any])
    # Its results have unknown precision, so they fit either annotation.
    either_single: NDArray[np.float32] = unknown.rms_frames()
    either_double: NDArray[np.float64] = unknown.rms_frames()
    assert either_single is not either_double


def bool_results() -> None:
    f64 = generation.sine_wave(220.0, 0.5, 22050, 0.5, dtype=np.float64)
    f0, voiced, probability = f64.pyin(fmin=65.0, fmax=2000.0)
    assert_type(f0, NDArray[np.float64])
    assert_type(voiced, NDArray[np.bool_])
    assert_type(probability, NDArray[np.float64])
    check(voiced, np.bool_)

    y32: NDArray[np.float32] = np.sin(np.arange(4096) / 10).astype(np.float32)
    single_f0, single_voiced, _ = features.pyin(y32, fmin=65.0, fmax=2000.0, sr=22050)
    assert_type(single_f0, NDArray[np.float32])
    assert_type(single_voiced, NDArray[np.bool_])
    check(single_f0, np.float32)
    check(single_voiced, np.bool_)

    frames = np.random.default_rng(0).random((8, 32))
    connectivity = sequence.recurrence_matrix(frames)
    assert_type(connectivity, NDArray[np.bool_])
    check(connectivity, np.bool_)
    distance = sequence.recurrence_matrix(frames, mode="distance")
    assert_type(distance, NDArray[np.float64])
    check(distance, np.float64)
    frames32: NDArray[np.float32] = frames.astype(np.float32)
    single_affinity = sequence.recurrence_matrix(frames32, mode="affinity")
    assert_type(single_affinity, NDArray[np.float32])
    check(single_affinity, np.float32)
    cross = sequence.cross_similarity(frames, frames)
    assert_type(cross, NDArray[np.bool_])
    check(cross, np.bool_)
    lag = sequence.recurrence_to_lag(connectivity)
    assert_type(lag, NDArray[np.bool_])
    check(lag, np.bool_)

    probabilities = np.clip(frames[:1], 0.1, 0.9)
    transition = sequence.transition_loop(2, 0.5)
    states = sequence.viterbi_binary(probabilities, transition)
    assert_type(states, NDArray[np.bool_])
    check(states, np.bool_)
    wide_states = sequence.viterbi_binary(probabilities, transition, dtype=np.uint16)
    assert_type(wide_states, NDArray[np.uint16])
    check(wide_states, np.uint16)


def free_functions_follow_the_array() -> None:
    y32: NDArray[np.float32] = np.sin(np.arange(4096) / 10).astype(np.float32)
    y64: NDArray[np.float64] = y32.astype(np.float64)

    d32 = spectral.stft(y32)
    assert_type(d32, NDArray[np.complex64])
    check(d32, np.complex64)
    d64 = spectral.stft(y64)
    assert_type(d64, NDArray[np.complex128])
    check(d64, np.complex128)
    from_list = spectral.stft([0.0] * 4096)
    assert_type(from_list, NDArray[np.complex128])
    check(from_list, np.complex128)

    s32 = np.abs(d32).astype(np.float32)
    s64 = np.abs(d64).astype(np.float64)

    db32 = spectral.power_to_db(s32)
    assert_type(db32, NDArray[np.float32])
    check(db32, np.float32)
    db64 = spectral.power_to_db(s64)
    assert_type(db64, NDArray[np.float64])
    check(db64, np.float64)
    assert_type(spectral.power_to_db(2.0), float)
    assert isinstance(spectral.power_to_db(2.0), float)

    centroid = features.spectral_centroid_from_spectrogram(s32, sr=22050)
    assert_type(centroid, NDArray[np.float32])
    check(centroid, np.float32)

    envelope = rhythm.onset_strength(S=s64, sr=22050, n_fft=2048, hop_length=512)
    assert_type(envelope, NDArray[np.float64])
    check(envelope, np.float64)

    frames = rhythm.onset_detect(onset_envelope=envelope, sr=22050)
    assert_type(frames, NDArray[np.int64])
    check(frames, np.int64)
    times = rhythm.onset_detect(onset_envelope=envelope, sr=22050, units="time")
    assert_type(times, NDArray[np.float64])
    check(times, np.float64)

    emphasised = effects.preemphasis(y32)
    assert_type(emphasised, NDArray[np.float32])
    check(emphasised, np.float32)
    with_state, state = effects.preemphasis(y32, return_zf=True)
    assert_type(with_state, NDArray[np.float32])
    assert_type(state, NDArray[np.float32])
    check(state, np.float32)

    harmonic, percussive = effects.hpss(d32)
    assert_type(harmonic, NDArray[np.complex64])
    check(percussive, np.complex64)

    cost, path = sequence.dtw(s64[:8, :16], s64[:8, :16])
    assert_type(cost, NDArray[np.float64])
    assert_type(path, NDArray[np.int64])
    check(path, np.int64)
    assert_type(sequence.dtw(s32[:8, :16], s32[:8, :16], backtrack=False), NDArray[np.float32])


def scalars_and_arrays() -> None:
    assert_type(utils.hz_to_midi(440.0), float)
    assert isinstance(utils.hz_to_midi(440.0), float)
    midi = utils.hz_to_midi([440.0, 880.0])
    assert_type(midi, NDArray[np.float64])
    check(midi, np.float64)
    assert_type(utils.time_to_samples(1.0, sr=8000), int)
    assert isinstance(utils.time_to_samples(1.0, sr=8000), int)
    assert_type(aus.mu_law_encode(0.5), int)
    assert_type(aus.mu_law_decode(10), float)


def annotations_work_at_run_time() -> None:
    alias = aus.AudioSamples[np.float32]
    assert alias.__origin__ is aus.AudioSamples  # type: ignore[attr-defined]


def run_all() -> None:
    audio_keeps_its_sample_type()
    clips_follow_the_dtype()
    analysis_precision_follows_the_sample_type()
    bool_results()
    free_functions_follow_the_array()
    scalars_and_arrays()
    annotations_work_at_run_time()
