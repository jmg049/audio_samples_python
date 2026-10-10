"""Regression tests for the adversarial review of the librosa-parity bindings.

Huge or out-of-range integers raise ``ValueError`` naming the parameter (never
pyo3's ``OverflowError``, a ``PanicException`` or a process abort on a failed
allocation); the ``hpss(stft_params=)`` alias; stub and
submodule fixes; librosa keywords that used to raise ``TypeError``.
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


def test_transition_matrices_at_the_cap_are_value_errors_not_panics():
    # 2**31 passes the binding's size cap; the Rust side rejects the
    # 2**62-element matrix (transition_uniform used to panic, the others abort).
    for f in (sequence.transition_uniform, lambda n: sequence.transition_loop(n, 0.5)):
        with pytest.raises(ValueError, match="n_states"):
            f(2**31)


def test_other_rust_side_size_checks():
    with pytest.raises(ValueError, match="frame_length"):
        spectral.frame(np.zeros(100_000), frame_length=50_000, hop_length=1)
    with pytest.raises(ValueError, match="sample_rate"):
        features.wavelet(freqs=np.array([100.0, 200.0, 400.0]), sr=1e15)
    with pytest.raises(ValueError, match="frame_rate"):
        rhythm.beat_track_with_bpm(onset_envelope=ENV, bpm=120.0, sr=1e15)
    with pytest.raises(ValueError, match="order"):
        effects.lpc(Y[:10], order=2**31)


def test_huge_max_size_is_fast():
    S = np.abs(RNG.standard_normal((128, 20)))
    small = rhythm.onset_strength(S=S, max_size=2 * 128)
    np.testing.assert_allclose(rhythm.onset_strength(S=S, max_size=2**31), small)


def test_clicks_far_time_is_value_error():
    for t in (1e10, 1e20):
        with pytest.raises(ValueError, match="times"):
            effects.clicks(times=[t], sr=22050)


def test_hpss_stft_params_keyword_is_a_deprecated_alias():
    a = aus.AudioSamples.new_mono(Y, sample_rate=22050)
    p = aus.spectrograms.StftParams(1024, 256, aus.spectrograms.WindowType.hanning, True)
    h_new, p_new = a.hpss(p)
    with pytest.warns(DeprecationWarning, match="stft_params"):
        h_old, p_old = a.hpss(stft_params=p, median_filter_harmonic=17)
    np.testing.assert_array_equal(h_old.to_numpy(), h_new.to_numpy())
    np.testing.assert_array_equal(p_old.to_numpy(), p_new.to_numpy())
    with pytest.raises(TypeError, match="not both"):
        a.hpss(p, stft_params=p)
    with pytest.raises(TypeError, match="missing"):
        a.hpss()


def test_spectrograms_submodule_importable():
    from audio_samples.spectrograms import ChromaNorm, StftParams

    assert StftParams is aus.spectrograms.StftParams
    assert ChromaNorm.l2 is not None


def test_recurrence_matrix_signature_lists_self_and_self_links():
    import inspect

    params = inspect.signature(sequence.recurrence_matrix).parameters
    assert params["self"].default is False and params["self_links"].default is False
    X = FEAT
    np.testing.assert_array_equal(
        sequence.recurrence_matrix(X, k=5, self_links=True),
        sequence.recurrence_matrix(X, k=5, **{"self": True}),
    )


# ---------------------------------------------------------------------------
# librosa keywords that used to raise TypeError
# ---------------------------------------------------------------------------


def test_lag_transforms_axis_matches_librosa():
    librosa = pytest.importorskip("librosa")
    R = np.random.default_rng(1).random((12, 12)) > 0.6
    for pad in (True, False):
        for axis in (0, 1, -1):
            lag = librosa.segment.recurrence_to_lag(R, pad=pad, axis=axis)
            np.testing.assert_array_equal(sequence.recurrence_to_lag(R, pad=pad, axis=axis), lag)
            np.testing.assert_array_equal(
                sequence.lag_to_recurrence(lag, axis=axis), librosa.segment.lag_to_recurrence(lag, axis=axis)
            )
    # -2 is the same axis as 0 for these 2-D matrices.
    np.testing.assert_array_equal(sequence.recurrence_to_lag(R, axis=-2), sequence.recurrence_to_lag(R, axis=0))
    with pytest.raises(ValueError, match="axis"):
        sequence.recurrence_to_lag(R, axis=2)


@pytest.mark.parametrize("norm", [None, 1, 2, np.inf, -np.inf, 0, 0.5])
def test_window_sumsquare_norm_matches_librosa(norm):
    librosa = pytest.importorskip("librosa")
    kw = dict(window="hann", n_frames=7, hop_length=128, win_length=400, n_fft=512, norm=norm)
    ours = spectral.window_sumsquare(**kw)
    ref = librosa.filters.window_sumsquare(**kw, dtype=np.float64)
    # Same window, same normalisation; only summation-order round-off.
    np.testing.assert_allclose(ours, ref, rtol=0, atol=1e-15)


def test_phase_vocoder_accepts_deprecated_keywords():
    D = spectral.stft(Y, n_fft=512)
    ref = effects.phase_vocoder(D, rate=1.5)
    with pytest.warns(FutureWarning, match="hop_length"):
        out = effects.phase_vocoder(D, rate=1.5, hop_length=128)
    np.testing.assert_array_equal(out, ref)
    with pytest.warns(FutureWarning, match="n_fft"):
        effects.phase_vocoder(D, rate=1.5, n_fft=512)


def test_times_like_axis_and_sr_alias():
    librosa = pytest.importorskip("librosa")
    X = np.zeros((3, 7, 5))
    np.testing.assert_allclose(U.times_like(X, sr=8000, axis=-2), librosa.times_like(X, sr=8000, axis=-2))
    np.testing.assert_array_equal(U.samples_like(X, axis=0), librosa.samples_like(X, axis=0))
    np.testing.assert_allclose(U.times_like(X), librosa.times_like(X))
    np.testing.assert_array_equal(U.time_to_samples([0.5, 1.25], sr=100), librosa.time_to_samples([0.5, 1.25], sr=100))
    assert U.time_to_frames_floor(0.5, sr=1000, hop_length=10) == librosa.time_to_frames(0.5, sr=1000, hop_length=10)
    assert U.time_to_samples(1.0, sample_rate=100) == U.time_to_samples(1.0, sr=100) == 100
    with pytest.raises(TypeError, match="not both"):
        U.time_to_samples(1.0, sr=1, sample_rate=2)
    with pytest.raises(ValueError, match="axis"):
        U.times_like(X, axis=3)
