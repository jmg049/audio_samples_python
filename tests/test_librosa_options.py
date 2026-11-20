"""Golden-reference checks for options reached through array forms and explicit parameters.

Most results are the same arithmetic as the reference, on the same spectrograms, and agree
to 1e-9 of the peak. Interpolated values use a different solver and are held to 1e-8.
Variable-Q octaves below the first are resampled with a different filter; with energy near
their cut-off the difference reaches 3e-3 of the peak, so they are held to 5e-3.
Reassigned times and frequencies divide two small numbers in the quietest bins, so they are
compared only where the power is within 60 dB of the peak, to 1e-4 of the range.
"""

from __future__ import annotations

import numpy as np
import pytest

import audio_samples as aus
from audio_samples import arrays, features, forms, notation
from conftest import load_reference_fixture

fixture = load_reference_fixture("options")
array_fixture = load_reference_fixture("array_ops")


def _c(v):
    return np.asarray(v["re"]) + 1j * np.asarray(v["im"])


def _close(a, b, rtol=1e-9):
    a, b = np.asarray(a), np.asarray(b)
    assert a.shape == b.shape
    assert np.abs(a - b).max() <= rtol * max(np.abs(b).max(), 1e-300)


def test_wavelet_alpha():
    f = fixture["wavelet_alpha"]
    lengths, cutoff = features.wavelet_lengths(freqs=f["freqs"], sr=22050, alpha=f["alpha"])
    _close(lengths, f["lengths"], 1e-12)
    assert cutoff == pytest.approx(f["cutoff"])
    filters, _ = features.wavelet(freqs=f["freqs"], sr=22050, alpha=f["alpha"])
    _close(filters, _c(f["filters"]))


def _mono():
    return aus.AudioSamples.new_mono(np.asarray(fixture["y"]), fixture["sr"])


def test_vqt_intervals():
    f = fixture["vqt_intervals"]
    v = _mono().vqt(fmin=110.0, n_bins=48, gamma=0.0, tuning=0.0, intervals=np.asarray(f["ratios"]))
    _close(np.squeeze(v), _c(f["v"]), 1e-5)


@pytest.mark.parametrize("name", ["pythagorean", "ji3", "ji5", "ji7"])
def test_vqt_named_intervals(name):
    v = _mono().vqt(fmin=220.0, n_bins=24, gamma=0.0, tuning=0.0, intervals=name)
    _close(np.squeeze(v), _c(fixture["vqt_named"][name]), 5e-3)


def test_chroma_forms():
    f = fixture["chroma"]
    C, window = np.asarray(f["C"]), np.asarray(f["window"])
    basis = features.cq_to_chroma(72, bins_per_octave=36, fmin=55.0, window=window)
    _close(basis, f["cq_to_chroma_window"])
    _close(forms.chroma_cqt(C=C, fmin=55.0, bins_per_octave=36, window=window), f["chroma_cqt"])
    _close(forms.chroma_cens(C=C, fmin=55.0, bins_per_octave=36, window=window), f["chroma_cens"])
    _close(forms.chroma_vqt(V=C, fmin=55.0, bins_per_octave=36), f["chroma_vqt"])


@pytest.mark.parametrize("kind", ["linear", "nearest", "cubic"])
def test_harmonic_kinds_and_axis(kind):
    f = fixture["harmonics"]
    S = np.asarray(f["S"])
    got = features.interp_harmonics(S, freqs=f["freqs"], harmonics=f["harmonics"], kind=kind, axis=0)
    _close(got, f["kinds"][kind]["interp"], 1e-8)
    got = features.f0_harmonics(S, freqs=f["freqs"], f0=f["f0"], harmonics=f["harmonics"], kind=kind, axis=0)
    _close(got, f["kinds"][kind]["f0"], 1e-8)


@pytest.mark.parametrize("kind", ["nearest", "cubic"])
def test_salience_with_other_interpolants(kind):
    f = fixture["harmonics"]
    S = np.asarray(f["S"])
    got = forms.salience(S, freqs=np.asarray(f["freqs"]), harmonics=f["harmonics"], kind=kind, fill_value=0.0)
    _close(got, f["kinds"][kind]["salience"], 1e-8)


def test_tempogram_ratio_custom_frequencies():
    f = fixture["tempogram_ratio"]
    got = forms.tempogram_ratio(tg=np.asarray(f["tg"]), bpm=np.asarray(f["bpm"]), freqs=np.asarray(f["custom_freqs"]))
    _close(got, f["result"])


def test_onset_feature_and_reference():
    f = fixture["onset_options"]
    y = np.asarray(fixture["y"])

    def feature(y, sr, n_fft, hop_length):
        audio = aus.AudioSamples.new_mono(y, int(sr))
        return np.abs(np.squeeze(audio.stft_matrix(n_fft=n_fft, hop_length=hop_length))) ** 2

    got = forms.onset_strength(y=y, sr=fixture["sr"], n_fft=1024, hop_length=256, feature=feature)
    _close(np.squeeze(got), f["power_feature"], 1e-6)


def test_reassigned_with_spectrogram():
    f = fixture["reassigned"]
    y = np.asarray(fixture["y"])
    audio = aus.AudioSamples.new_mono(y, fixture["sr"])
    S = np.squeeze(audio.stft_matrix(n_fft=512, hop_length=128))
    freqs, times, mags = forms.reassigned_spectrogram(y, sr=fixture["sr"], S=S * 2.0, n_fft=512, hop_length=128, fill_nan=True)
    _close(mags[:64, :16], f["mags"], 1e-9)
    keep = np.asarray(f["mags"]) ** 2 > np.asarray(f["mags"]).max() ** 2 * 1e-6
    assert np.abs(freqs[:64, :16] - np.asarray(f["freqs"]))[keep].max() < 1e-4 * fixture["sr"] / 2
    assert np.abs(times[:64, :16] - np.asarray(f["times"]))[keep].max() < 1e-4 * y.size / fixture["sr"]


def test_stft_and_istft_buffers():
    y = np.asarray(fixture["y"], dtype=np.float32)
    plain = forms.stft(y, n_fft=512, hop_length=128)
    assert plain.dtype == np.complex64
    assert forms.stft(y, n_fft=512, hop_length=128, dtype=np.complex128).dtype == np.complex128
    buffer = np.zeros((257, plain.shape[-1] + 4), dtype=np.complex64)
    written = forms.stft(y, n_fft=512, hop_length=128, out=buffer)
    assert np.shares_memory(written, buffer)
    assert np.array_equal(written, plain)
    samples = np.zeros(y.size + 8, dtype=np.float32)
    back = forms.istft(plain, hop_length=128, n_fft=512, length=y.size, out=samples)
    assert np.shares_memory(back, samples)
    assert np.abs(back - y).max() < 1e-4


def test_zero_crossings():
    f = array_fixture["zero_crossings"]
    y = np.asarray(f["y"])
    _close(arrays.zero_crossings(y).astype(float), f["default"], 0.0)
    _close(arrays.zero_crossings(y, pad=False).astype(float), f["no_pad"], 0.0)
    _close(arrays.zero_crossings(y, zero_pos=False, threshold=0.5).astype(float), f["zero_sign"], 0.0)
    _close(arrays.zero_crossings(y, threshold=0.1, ref_magnitude=np.max).astype(float), f["ref"], 0.0)
    _close(arrays.zero_crossings(np.stack([y, -y]), axis=0).astype(float), f["axis0"], 0.0)


def test_tone_and_chirp():
    f = array_fixture["generators"]
    _close(arrays.tone(441.0, sr=22050, length=512), f["tone"], 1e-12)
    _close(arrays.tone(441.0, sr=22050, length=512, phi=0.7), f["tone_phi"], 1e-12)
    _close(arrays.chirp(fmin=110.0, fmax=3520.0, sr=22050, length=2048), f["chirp_log"], 1e-12)
    _close(arrays.chirp(fmin=110.0, fmax=3520.0, sr=22050, length=2048, linear=True, phi=0.3), f["chirp_lin"], 1e-12)


def test_note_names():
    f = array_fixture["note_names"]
    assert list(notation.midi_to_note(f["midi"])) == f["c"]
    assert list(notation.midi_to_note([60.0, 61.0, 70.0], octave=False, key="F:maj")) == f["f_major_no_octave"]
    assert list(notation.midi_to_note([61.0, 70.0], key="Db:maj", unicode=True)) == f["unicode"]
    assert list(notation.midi_to_note([69.25, 69.0, 68.75], cents=True)) == f["cents"]
    notes = f["fractional_notes"]
    _close(notation.note_to_midi(notes, round_midi=False), f["fractional"], 1e-14)
    _close(notation.note_to_midi(notes), f["rounded"], 0.0)
    _close(notation.note_to_hz(notes), f["hz"], 1e-14)
    _close(notation.note_to_hz(notes, round_midi=True), f["hz_rounded"], 1e-14)
    assert notation.hz_to_note(440.0) == "A4"
    assert notation.midi_to_note(float("nan")) == ""


def test_sparse_forms_of_similarity():
    data = np.random.default_rng(0).normal(size=(4, 40))
    from audio_samples import sequence

    dense = sequence.recurrence_matrix(data, k=3, sym=True)
    rows = sequence.recurrence_matrix(data, k=3, sym=True, sparse=True)
    assert np.array_equal(rows.toarray(), dense)
    moved = sequence.recurrence_matrix(data.T, k=3, sym=True, axis=0)
    assert np.array_equal(moved, dense)
    other = sequence.cross_similarity(data, data[:, ::2], k=3)
    assert np.array_equal(sequence.cross_similarity(data, data[:, ::2], k=3, sparse=True).toarray(), other)


def _forms_inputs():
    y = np.asarray(fixture["y"])
    return y, np.stack([y, y[::-1]]), y[: y.size // 2], fixture["sr"]


def test_channel_forms():
    f = fixture["forms"]
    y, stereo, short, _ = _forms_inputs()
    _close(forms.to_mono(stereo), f["to_mono"])
    _close(forms.to_mono(y, short, pad=True, norm=False), f["to_mono_sum"])
    _close(forms.to_stereo(left=y, right=short, pad=True), f["to_stereo"])
    _close(forms.to_multi(y, short, stereo, pad=False, norm=False), f["to_multi"])
    buffer = np.zeros_like(np.asarray(f["to_mono"]))
    assert forms.to_mono(stereo, out=buffer) is buffer
    _close(forms.to_mono(stereo, y), f["to_mono_mixed"])
    with pytest.raises(ValueError):
        forms.to_mono(np.array([1.0, np.nan]))


def test_scaled_resampling_and_pitch_shift():
    f = fixture["forms"]
    y, _, _, sr = _forms_inputs()
    # The resampler's filter differs from the reference's; the scale must not.
    got = forms.resample(y, orig_sr=sr, target_sr=8000, scale=True)
    assert got.shape == np.shape(f["resample_scale"])
    assert np.abs(got - np.asarray(f["resample_scale"]))[200:-200].max() < 5e-3 * np.abs(got).max()
    shifted = forms.pitch_shift(y, sr=sr, n_steps=2.0, scale=True)
    assert shifted.shape == np.shape(f["pitch_shift_scale"])
    energy, reference = np.sum(shifted**2), np.sum(np.asarray(f["pitch_shift_scale"]) ** 2)
    assert abs(energy / reference - 1.0) < 0.02


def test_frame_axes_and_duration():
    f = fixture["forms"]
    y, _, _, sr = _forms_inputs()
    _close(forms.frame(np.stack([y[:64], y[:64]], axis=1), frame_length=8, hop_length=4, axis=0), f["frame_axis0"], 0.0)
    _close(forms.frame(y[:64], frame_length=8, hop_length=4), f["frame_axis1"], 0.0)
    S = np.zeros((513, 40))
    assert forms.get_duration(S=S, sr=sr, n_fft=1024, hop_length=256) == pytest.approx(f["duration_S"])
    assert forms.get_duration(S=S, sr=sr, n_fft=1024, hop_length=256, center=False) == pytest.approx(f["duration_S_uncentred"])


def test_dense_event_masks():
    f = fixture["forms"]
    clicks = np.asarray(load_reference_fixture("signals")["signals"]["clicks120"], dtype=np.float64)
    sr = fixture["sr"]
    onset = forms.onset_detect(y=clicks, sr=sr, sparse=False)
    assert onset.dtype == np.bool_
    assert onset.shape == np.shape(f["onset_mask"])
    # Frame positions may move by one: the mel spectrogram has a different round-off.
    wanted = np.flatnonzero(f["onset_mask"])
    assert np.abs(np.flatnonzero(onset)[: wanted.size] - wanted).max() <= 1
    _, beats = forms.beat_track(y=clicks, sr=sr, sparse=False, trim=True)
    assert np.abs(np.flatnonzero(beats) - np.flatnonzero(f["beat_mask"])).max() <= 1


def test_read_segment_and_stream(tmp_path):
    from audio_samples import io

    y, _, _, sr = _forms_inputs()
    path = tmp_path / "clip.wav"
    audio = aus.AudioSamples.new_mono(y.astype(np.float32), sr)
    io.save(str(path), audio)
    piece = forms.read_segment(path, offset=0.1, duration=0.2)
    first = int(round(0.1 * sr))
    _close(piece.to_numpy(), y.astype(np.float32)[first : first + int(round(0.2 * sr))], 1e-6)
    blocks = list(forms.stream_blocks(path, block_length=2, frame_length=1024, hop_length=512))
    data = y.astype(np.float32)
    assert all(b.shape[-1] <= 1024 + 512 for b in blocks)
    assert blocks[0].dtype == np.float32
    for index, block in enumerate(blocks):
        _close(block, data[index * 1024 : index * 1024 + block.shape[-1]], 1e-6)
    filled = list(forms.stream_blocks(path, block_length=2, frame_length=1024, hop_length=512, fill_value=0.0))
    assert filled[-1].shape[-1] == 1024 + 512
    skipping = list(forms.stream_blocks(path, block_length=1, frame_length=256, hop_length=1024))
    for index, block in enumerate(skipping):
        _close(block, data[index * 1024 : index * 1024 + block.shape[-1]], 1e-6)
    assert np.isclose(forms.get_duration(path=path), y.size / sr)
