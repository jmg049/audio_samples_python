"""``audio_samples.arrays`` and the block conversions against librosa fixtures
(``array_ops.json``). Tolerances mirror ``tests/analysis_reference/array_ops.rs``:
exact for indices and booleans, 1e-15 of the peak for single arithmetic steps,
1e-10 for splines, which solve a linear system by a different elimination.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from audio_samples import arrays as A
from audio_samples import generation, utils

from conftest import assert_peak_close, load_reference_fixture

RTOL = 1e-15
SPLINE_RTOL = 1e-10


@pytest.fixture(scope="module")
def fx():
    return load_reference_fixture("array_ops")


def test_local_extrema(fx):
    f = fx["local"]
    x = np.array(f["x"])
    assert np.array_equal(A.localmax(x, axis=0), np.array(f["max_axis0"], bool))
    assert np.array_equal(A.localmax(x, axis=1), np.array(f["max_axis1"], bool))
    assert np.array_equal(A.localmin(x, axis=0), np.array(f["min_axis0"], bool))
    assert np.array_equal(A.localmin(x, axis=-1), np.array(f["min_axis1"], bool))
    assert A.localmax(x.astype(np.float32)).dtype == np.bool_
    assert A.localmax([1, 3, 2]).tolist() == [False, True, False]
    with pytest.raises(ValueError):
        A.localmax(x, axis=3)


def test_match_events(fx):
    f = fx["match_events"]
    assert A.match_events(f["from"], f["to"]).tolist() == f["both"]
    assert A.match_events(f["from_inner"], f["to"], left=False).tolist() == f["no_left"]
    assert A.match_events(f["from_inner"], f["to"], right=False).tolist() == f["no_right"]
    assert A.match_events(f["subset_from"], f["subset_to"], left=False, right=False).tolist() == f["exact"]
    with pytest.raises(ValueError):
        A.match_events([], [1.0])


def test_match_intervals(fx):
    f = fx["match_intervals"]
    assert A.match_intervals(np.array(f["from"]), np.array(f["to"])).tolist() == f["strict"]
    got = A.match_intervals(np.array(f["gap_from"]), np.array(f["gap_to"]), strict=False)
    assert got.tolist() == f["gap_non_strict"]
    with pytest.raises(ValueError):
        A.match_intervals(np.array([[3.5, 4.0]]), np.array([[0.0, 1.0], [5.0, 6.0]]))


def test_complex_helpers(fx):
    f = fx["complex"]
    z = np.array(f["z_re"]) + 1j * np.array(f["z_im"])
    assert_peak_close(A.abs2(z), np.array(f["abs2"]), 4 * RTOL, "abs2")
    assert A.abs2(z.astype(np.complex64)).dtype == np.float32
    assert_peak_close(A.abs2(z.real), np.array(f["abs2_real"]), RTOL, "abs2 real")
    assert A.abs2(3 + 4j) == 25.0
    ang = np.array(f["angles"])
    p = A.phasor(ang)
    assert_peak_close(p.real, np.array(f["phasor_re"]), RTOL, "phasor")
    assert_peak_close(p.imag, np.array(f["phasor_im"]), RTOL, "phasor")
    assert A.phasor(ang.astype(np.float32)).dtype == np.complex64
    pm = A.phasor(ang, mag=np.array(f["mag"]))
    assert_peak_close(pm.real, np.array(f["phasor_mag_re"]), RTOL, "phasor mag")
    assert_peak_close(pm.imag, np.array(f["phasor_mag_im"]), RTOL, "phasor mag")
    assert np.isclose(A.phasor(0.0), 1.0)
    assert A.phasor(ang, mag=2.0 + 0j).dtype == np.complex128


def test_tiny(fx):
    f = fx["tiny"]
    assert float(A.tiny(np.float32(1))) == f["f32"]
    assert A.tiny(np.float32(1)).dtype == np.float32
    assert float(A.tiny(np.float64(1))) == f["f64"]
    assert float(A.tiny(np.complex64(1))) == f["c64"]
    assert float(A.tiny(1)) == f["int"]
    assert float(A.tiny(1.0)) == f["f64"]
    assert float(A.tiny(np.zeros(3, np.float16))) == float(np.finfo(np.float16).tiny)


def test_pad_center(fx):
    f = fx["pad_center"]
    x = np.array(f["x"])
    for mode in ("constant", "edge", "reflect", "symmetric", "wrap"):
        assert A.pad_center(x, size=18, mode=mode).tolist() == f[mode], mode
    # Modes without a native path go through numpy.pad, as do non-float dtypes.
    assert A.pad_center(x, size=7, mode="linear_ramp", end_values=1.0).shape == (7,)
    assert A.pad_center(np.arange(3), size=5).dtype == np.arange(3).dtype
    assert A.pad_center(x, size=7, constant_values=2.0)[0] == 2.0
    m = np.array(f["matrix"])
    assert A.pad_center(m, size=5, axis=0).tolist() == f["matrix_axis0"]
    assert A.pad_center(m, size=6, axis=-1).tolist() == f["matrix_axis1"]
    assert A.pad_center(x.astype(np.complex64), size=7).dtype == np.complex64
    with pytest.raises(ValueError):
        A.pad_center(x, size=1)


def test_frames_and_slices(fx):
    f = fx["fix_frames"]
    assert A.fix_frames(f["frames"]).tolist() == f["default"]
    assert A.fix_frames(f["frames"], x_min=5, x_max=25).tolist() == f["bounded"]
    assert A.fix_frames(f["frames"], x_min=5, x_max=25, pad=False).tolist() == f["no_pad"]
    assert A.fix_frames(f["frames"], x_min=None, x_max=None).tolist() == f["none"]
    assert [[s.start, s.stop] for s in A.index_to_slice(f["frames"], idx_min=2, idx_max=30)] == f["slices"]
    assert A.index_to_slice([0, 2], step=3)[0] == slice(0, 2, 3)
    with pytest.raises(ValueError):
        A.fix_frames([-1])


def test_axis_sort(fx):
    f = fx["axis_sort"]
    s = np.array(f["s"])
    got, idx = A.axis_sort(s, index=True)
    assert got.tolist() == f["columns"] and idx.tolist() == f["columns_idx"]
    got, idx = A.axis_sort(s, axis=0, index=True)
    assert got.tolist() == f["rows"] and idx.tolist() == f["rows_idx"]
    # Equal keys are stable here and unspecified in librosa, so compare the keys.
    by_min = A.axis_sort(s, value=np.argmin)
    assert np.argmin(by_min, axis=0).tolist() == np.argmin(np.array(f["argmin"]), axis=0).tolist()
    with pytest.raises(ValueError):
        A.axis_sort(np.zeros(3))


def test_shear(fx):
    f = fx["shear"]
    x = np.array(f["x"])
    assert A.shear(x, factor=1).tolist() == f["columns"]
    assert A.shear(x, factor=-2, axis=0).tolist() == f["rows"]
    assert A.shear(x.astype(np.complex64)).dtype == np.complex64
    assert A.shear(x > 0).dtype == np.bool_


def test_shear_sparse_stays_sparse(fx):
    sparse = pytest.importorskip("scipy.sparse")
    x = np.round(np.array(fx["shear"]["x"]))
    for cls in (sparse.csr_array, sparse.csc_array, sparse.csr_matrix):
        for axis in (-1, 0):
            got = A.shear(cls(x), factor=2, axis=axis)
            assert type(got) is cls
            assert np.array_equal(got.toarray(), A.shear(x, factor=2, axis=axis))


def test_sparsify_rows(fx):
    f = fx["sparsify_rows"]
    x = np.array(f["x"])
    assert A.sparsify_rows(x, quantile=0.01).toarray().tolist() == f["q01"]
    q = A.sparsify_rows(x, quantile=0.25)
    assert q.toarray().tolist() == f["q25"]
    assert q.nnz == np.count_nonzero(f["q25"])
    assert A.sparsify_rows(x[0], quantile=0.1).toarray().tolist() == f["one_d"]
    assert A.sparsify_rows(x.astype(np.float32)).data.dtype == np.float32
    assert A.sparsify_rows(x, dtype=np.float32).data.dtype == np.float32
    assert A.sparsify_rows(np.ones((2, 2), np.int64)).data.dtype == np.int64
    pytest.importorskip("scipy.sparse")
    assert np.array_equal(q.tocsr().toarray(), q.toarray())
    with pytest.raises(ValueError):
        A.sparsify_rows(x, quantile=1.0)


def test_cyclic_gradient(fx):
    f = fx["cyclic_gradient"]
    x = np.array(f["x"])
    assert_peak_close(A.cyclic_gradient(x), np.array(f["axis1"]), RTOL, "axis1")
    assert_peak_close(A.cyclic_gradient(x, axis=0, edge_order=2), np.array(f["axis0_order2"]), RTOL, "axis0")
    assert A.cyclic_gradient(x.astype(np.float32)).dtype == np.float32
    z = A.cyclic_gradient(x + 1j * x)
    assert_peak_close(z.imag, np.array(f["axis1"]), RTOL, "complex")
    with pytest.raises(ValueError):
        A.cyclic_gradient(x, edge_order=3)


def test_fill_off_diagonal(fx):
    for key, want in fx["fill_off_diagonal"].items():
        want = np.array(want)
        radius = float(key.split("_")[1])
        for dtype in (np.float64, np.float32, np.int64):
            m = np.ones(want.shape, dtype)
            assert A.fill_off_diagonal(m, radius=radius, value=-1) is None
            assert np.array_equal(m, want.astype(dtype)), key
    m = np.ones((3, 3))
    A.fill_off_diagonal(m, radius=0.0)
    assert m.sum() == 0.0
    with pytest.raises(TypeError):
        A.fill_off_diagonal([[1.0]], radius=0.1)


def test_unique(fx):
    f = fx["unique"]
    x = np.array(f["x"])
    assert A.count_unique(x).tolist() == f["count_axis1"]
    assert A.count_unique(x, axis=0).tolist() == f["count_axis0"]
    assert A.is_unique(x).astype(int).tolist() == f["is_axis1"]
    assert A.is_unique(x, axis=0).astype(int).tolist() == f["is_axis0"]
    assert A.count_unique(np.array([np.nan, np.nan, 1.0])).tolist() == 2


@pytest.mark.parametrize(
    "kind", ["linear", "nearest", "nearest-up", "zero", "slinear", "quadratic", "cubic", "previous", "next"]
)
def test_interp_kinds(fx, kind):
    f = fx["interp"]
    want = f["kinds"][kind]
    y1, y2 = A.interp_broadcast(
        x1=np.array(f["x1"]), x1_pos=f["x1_pos"], x2=np.array(f["x2"]), x2_pos=f["x2_pos"],
        interp_pos=f["q"], op=None, kind=kind, fill_value=-7.0, axis=-1,
    )
    assert_peak_close(y1, np.array(want["y1"]), SPLINE_RTOL, kind)
    assert_peak_close(y2, np.array(want["y2"]), SPLINE_RTOL, kind)


def test_interp_broadcast_ops(fx):
    f = fx["interp"]
    kw = dict(x1=np.array(f["x1"]), x1_pos=f["x1_pos"], x2=np.array(f["x2"]), x2_pos=f["x2_pos"], interp_pos=f["q"], axis=-1)
    want = np.array(f["product"])
    assert_peak_close(A.interp_broadcast(**kw), want, 1e-14, "multiply")
    assert_peak_close(A.interp_broadcast(op=np.multiply, **kw), want, 1e-14, "numpy multiply")
    assert_peak_close(A.interp_broadcast(op=lambda a, b: a * b, **kw), want, 1e-14, "callable")
    assert A.interp_broadcast(**{**kw, "x1": kw["x1"].astype(np.float32), "x2": kw["x2"].astype(np.float32)}).dtype == np.float32
    with pytest.raises(ValueError):
        A.interp_broadcast(**{**kw, "axis": -3})


def test_interp1d_shapes():
    y = np.arange(12.0).reshape(3, 4)
    q = np.array([[0.5, 1.5], [2.5, 9.0]])
    out = A.interp1d(np.arange(4.0), y, q, fill_value=-1.0)
    assert out.shape == (3, 2, 2)
    assert out[0, 1, 1] == -1.0
    assert A.interp1d([0.0, 1.0], [0.0, 1.0], [0.5], kind=1).tolist() == [0.5]
    with pytest.raises(ValueError):
        A.interp1d([0.0, 1.0], [0.0, 1.0], [0.5], kind="spline")


def test_dtypes():
    assert A.dtype_c2r(np.complex64) == np.float32
    assert A.dtype_c2r(np.complex128) == np.float64
    assert A.dtype_c2r(np.float64) == np.float64
    assert A.dtype_c2r(np.int32) == np.float32
    assert A.dtype_c2r(np.int32, default=np.float64) == np.float64
    assert A.dtype_r2c(np.float32) == np.complex64
    assert A.dtype_r2c(np.float64) == np.complex128
    assert A.dtype_r2c(np.complex128) == np.complex128
    assert A.dtype_r2c(np.int32) == np.complex64
    assert A.dtype_r2c("f8") == np.complex128


def test_buf_to_float(fx):
    f = fx["buf_to_float"]
    raw = bytes(f["bytes"])
    for n in (1, 2, 4, 8):
        assert A.buf_to_float(raw, n_bytes=n, dtype=np.float64).tolist() == f[f"n{n}"], n
    y = A.buf_to_float(raw)
    assert y.dtype == np.float32
    assert y.tolist() == np.array(f["n2_f32"], np.float32).tolist()
    assert A.buf_to_float(raw, n_bytes=4).dtype == np.float64
    assert A.buf_to_float(np.array([0, -32768], np.int16)).tolist() == [0.0, -1.0]
    assert A.buf_to_float(bytearray(raw), n_bytes=2).shape == (32,)
    assert A.buf_to_float(b"\x00\x00\x80", n_bytes=3).tolist() == [-1.0]
    with pytest.raises(ValueError):
        A.buf_to_float(raw, n_bytes=5)
    with pytest.raises(ValueError):
        A.buf_to_float(raw, dtype=np.int16)


def test_blocks(fx):
    f = fx["blocks"]
    assert utils.blocks_to_frames(f["blocks"], block_length=16).tolist() == f["frames"]
    assert utils.blocks_to_samples(f["blocks"], block_length=16, hop_length=512).tolist() == f["samples"]
    assert utils.blocks_to_time(f["blocks"], block_length=16, hop_length=512, sample_rate=22050).tolist() == f["time"]
    assert utils.blocks_to_frames(2, block_length=16) == 32
    with pytest.raises(ValueError):
        utils.blocks_to_frames(2, block_length=0)


def test_expand_and_stack():
    x = np.zeros((2, 3))
    assert A.expand_to(x, ndim=4, axes=[1, 3]).shape == (1, 2, 1, 3)
    assert A.expand_to(x, ndim=4, axes=(1, -1)).shape == (1, 2, 1, 3)
    assert A.expand_to(np.zeros(3), ndim=3, axes=1).shape == (1, 3, 1)
    assert np.shares_memory(A.expand_to(x, ndim=3, axes=[0, 2]), x)
    assert A.expand_to(np.arange(6).reshape(2, 3), ndim=2, axes=[1, 0]).tolist() == [[0, 3], [1, 4], [2, 5]]
    with pytest.raises(ValueError):
        A.expand_to(x, ndim=1, axes=[0, 1])
    assert A.stack([x, x]).shape == (2, 2, 3)
    assert A.stack([x, x], axis=-1).shape == (2, 3, 2)
    with pytest.raises(ValueError):
        A.stack([x, np.zeros(2)])
    with pytest.raises(ValueError):
        A.stack([])


def test_checks():
    assert A.valid_audio(np.zeros(3))
    assert A.valid_audio(np.zeros(3, np.float16))
    for bad in (np.array([np.nan]), np.array(1.0)):
        with pytest.raises(ValueError):
            A.valid_audio(bad)
    for bad in ([0.0], np.zeros(3, np.int16)):
        with pytest.raises(TypeError):
            A.valid_audio(bad)
    assert A.valid_intervals(np.array([[0.0, 1.0]]))
    for bad in (np.array([[1.0, 0.0]]), np.zeros((2, 3))):
        with pytest.raises(ValueError):
            A.valid_intervals(bad)
    assert A.is_positive_int(3) and A.is_positive_int(np.int16(3))
    assert not (A.is_positive_int(0) or A.is_positive_int(2.0) or A.is_positive_int(None))
    assert A.valid_int(2.7) == 2 and A.valid_int(-2.5) == -3 and A.valid_int(3) == 3
    assert A.valid_int(2.5, cast=np.ceil) == 3
    with pytest.raises(TypeError):
        A.valid_int(2.5, cast=1)
    assert A.MAX_MEM_BLOCK == 2**18


def test_find_files(tmp_path):
    (tmp_path / "sub").mkdir()
    for name in ("a.wav", "B.WAV", "c.Wav", "d.txt", ".hidden.wav", "sub/e.flac"):
        (tmp_path / name).touch()
    names = lambda v: [os.path.basename(p) for p in v]  # noqa: E731
    assert names(A.find_files(tmp_path)) == ["B.WAV", "a.wav", "c.Wav", "e.flac"]
    assert names(A.find_files(str(tmp_path), recurse=False)) == ["B.WAV", "a.wav", "c.Wav"]
    assert names(A.find_files(tmp_path, ext="wav", case_sensitive=True)) == ["a.wav"]
    assert names(A.find_files(tmp_path, ext=["flac", "txt"])) == ["d.txt", "e.flac"]
    assert names(A.find_files(tmp_path, offset=1, limit=2)) == ["a.wav", "c.Wav"]
    assert names(A.find_files(tmp_path, offset=-1)) == ["e.flac"]
    assert names(A.find_files(tmp_path, limit=-1)) == ["B.WAV", "a.wav", "c.Wav"]
    assert A.find_files(tmp_path / "missing") == []
    assert all(os.path.isabs(p) for p in A.find_files(tmp_path))


def test_example_clip_info():
    for name in generation.example_clip_names():
        assert generation.example_clip_info(name).endswith(".")
    with pytest.raises(ValueError):
        generation.example_clip_info("nope")
