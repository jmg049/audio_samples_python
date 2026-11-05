"""Results written into a given array, and strings built without an intermediate list."""

import numpy as np
import pytest

from audio_samples import forms, notation


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("order", ["C", "F"])
def test_stft_out_matches_the_returned_matrix(dtype, order):
    y = np.sin(np.arange(4096) * 0.05).astype(dtype)
    want = forms.stft(y, n_fft=512)
    ctype = np.complex64 if dtype == np.float32 else np.complex128
    out = np.full((257, want.shape[1] + 3), 7 + 7j, dtype=ctype, order=order)
    got = forms.stft(y, n_fft=512, out=out)
    assert np.shares_memory(got, out)
    assert got.shape == want.shape
    np.testing.assert_array_equal(got, want)
    assert np.all(out[..., want.shape[1] :] == 7 + 7j)


def test_stft_out_multichannel_and_a_strided_buffer():
    y = np.stack([np.sin(np.arange(2048) * 0.05), np.cos(np.arange(2048) * 0.03)])
    want = forms.stft(y, n_fft=256)
    store = np.zeros((2, 129, 2 * want.shape[2]), dtype=np.complex128)
    got = forms.stft(y, n_fft=256, out=store[..., ::2])
    np.testing.assert_array_equal(got, want)
    np.testing.assert_array_equal(store[..., ::2], want)
    assert np.all(store[..., 1::2] == 0)


def test_stft_out_of_another_type_is_converted():
    y = np.sin(np.arange(2048) * 0.05)
    out = np.zeros((129, 40), dtype=np.complex64)
    got = forms.stft(y, n_fft=256, out=out)
    np.testing.assert_allclose(got, forms.stft(y, n_fft=256), atol=1e-4)


def test_stft_out_too_short_or_not_finite_raises():
    y = np.sin(np.arange(2048) * 0.05)
    with pytest.raises(ValueError, match="out"):
        forms.stft(y, n_fft=256, out=np.zeros((129, 2), dtype=np.complex128))
    with pytest.raises(ValueError, match="out"):
        forms.stft(y, n_fft=256, out=np.zeros((128, 40), dtype=np.complex128))
    y[3] = np.nan
    with pytest.raises(ValueError, match="finite"):
        forms.stft(y, n_fft=256, out=np.zeros((129, 40), dtype=np.complex128))


def test_svara_arrays_keep_shape_and_width():
    midi = np.array([[60.0, 73.0], [np.nan, 48.0]])
    got = notation.midi_to_svara_h(midi, Sa=60.0, abbr=False, unicode=True)
    assert got.shape == (2, 2)
    assert got.dtype == np.dtype("<U3")
    assert got.tolist() == [["Sa", "r\u0307e"], ["", "S\u0323a"]]
    assert notation.midi_to_svara_h(np.array([np.nan]), Sa=60.0).dtype == np.dtype("<U1")
    assert notation.midi_to_svara_h(67.0, Sa=60.0) == "P"
    assert isinstance(notation.mela_to_svara(29), list)
    assert notation.note_to_svara_h(["C4", "G4"], Sa="C4").tolist() == ["S", "P"]
