"""Robustness of the librosa-style bindings against extreme arguments.

Regression tests for the adversarial review of the librosa-parity bindings:
huge or out-of-range integers must raise ``ValueError`` naming the parameter
(never pyo3's ``OverflowError``, a ``PanicException`` or a process abort on a
failed allocation).
"""

from __future__ import annotations

import numpy as np
import pytest

import audio_samples as aus
from audio_samples import effects, features, rhythm, sequence, spectral
from audio_samples import utils as U

RNG = np.random.default_rng(0)
Y = RNG.standard_normal(4096) * 0.1
FEAT = RNG.standard_normal((5, 30))
ENV = np.abs(RNG.standard_normal(200))


# (callable taking the bad integer, parameter name expected in the message)
INT_CASES = [
    (lambda v: spectral.delta(FEAT, width=v), "width"),
    (lambda v: spectral.delta(FEAT, order=v), "order"),
    (lambda v: spectral.delta(FEAT, axis=v), "axis"),
    (lambda v: spectral.frame(Y, frame_length=v, hop_length=128), "frame_length"),
    (lambda v: spectral.stft(Y, n_fft=v), "n_fft"),
    (lambda v: spectral.get_window("hann", v), "Nx"),
    (lambda v: spectral.window_sumsquare(window="hann", n_frames=v), "n_frames"),
    (lambda v: sequence.recurrence_matrix(FEAT, k=v), "k"),
    (lambda v: sequence.cross_similarity(FEAT, FEAT, k=v), "k"),
    (lambda v: sequence.sync(FEAT, [0, 5, 10], axis=v), "axis"),
    (lambda v: sequence.transition_uniform(v), "n_states"),
    (lambda v: sequence.transition_loop(v, 0.5), "n_states"),
    (lambda v: features.chroma_cens_from_chroma(np.abs(FEAT[:4].repeat(3, 0)), win_len_smooth=v), "win_len_smooth"),
    (lambda v: features.chroma_filterbank(sr=22050, n_fft=2048, n_chroma=v), "n_chroma"),
    (lambda v: effects.clicks(times=[0.1], sr=22050, hop_length=v), "hop_length"),
    (lambda v: effects.clicks(times=[0.1], sr=22050, length=v), "length"),
    (lambda v: rhythm.tempo_frequencies(v), "n_bins"),
    (lambda v: U.midi_to_note(v), "midi_note"),
]


@pytest.mark.parametrize("value", [2**40, 2**63, 2**70, -(2**70)])
@pytest.mark.parametrize("case", range(len(INT_CASES)))
def test_huge_ints_raise_value_error(case, value):
    f, name = INT_CASES[case]
    with pytest.raises(ValueError, match=name):
        f(value)


def test_huge_delay_is_handled():
    # librosa pads by the delay; the Rust implementation shifts with zero fill,
    # so a huge delay (either sign) gives zeros instead of an allocation.
    for v in (2**40, 2**70, -(2**70)):
        out = spectral.stack_memory(FEAT, n_steps=2, delay=v)
        assert out.shape == (10, 30)
        np.testing.assert_array_equal(out[5:], 0.0)


def test_non_integers_still_type_error():
    with pytest.raises(TypeError):
        spectral.stft(Y, n_fft=2.5)


def test_size_cap_message():
    with pytest.raises(ValueError, match=r"n_fft: must be at most 2147483648 \(2\*\*31\), got 1099511627776"):
        spectral.stft(Y, n_fft=2**40)
    with pytest.raises(ValueError, match=r"got >= 2\*\*63"):
        spectral.stft(Y, n_fft=2**70)


def test_hpss_filter_lengths_are_validated():
    a = aus.AudioSamples.new_mono(Y, sample_rate=22050)
    p = aus.spectrograms.StftParams(1024, 256, aus.spectrograms.WindowType.hanning, True)
    for bad in (-1, 0, 2**70):
        with pytest.raises(ValueError, match="median_filter_harmonic"):
            a.hpss(p, median_filter_harmonic=bad)


def test_sample_rate_keyword_out_of_range():
    D = spectral.stft(Y, n_fft=512)
    for bad in (-1, 2**40, 2**70):
        with pytest.raises(ValueError, match="sample_rate"):
            aus.AudioSamples.istft_matrix(D, sample_rate=bad)


def test_clicks_far_time_is_value_error():
    for t in (1e10, 1e20):
        with pytest.raises(ValueError, match="times"):
            effects.clicks(times=[t], sr=22050)
