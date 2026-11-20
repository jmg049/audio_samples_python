"""librosa parity of ``audio_samples.sequence`` (DTW, Viterbi, transition
matrices, RQA, recurrence / cross-similarity, lag conversions, sync).

Golden fixtures: ``sequence.json`` (librosa 1.0.0; inputs are stored next to the
outputs, so every case is the same computation on exactly the same inputs). The
dynamic programmes sum the same terms in the same order as librosa's numba
kernels; the only round-off sources are the local costs / logs (scipy ``cdist``
or numpy ``log`` may differ by an ulp), so values are compared at 1e-12
relative to the peak and paths / state sequences must match exactly. These
tolerances mirror ``tests/analysis_reference/sequence.rs`` in the Rust crate.
"""

from __future__ import annotations

import numpy as np
import pytest

from audio_samples import sequence as Q

from conftest import assert_peak_close, fixture_array, load_reference_fixture

RTOL_DP = 1e-12  # DP outputs: identical summation order, ulp-level local-cost differences
RTOL_KNN = 1e-12  # k-NN distances feed medians/means and exp(): ulp-level, amplified slightly


@pytest.fixture(scope="module")
def fx():
    return load_reference_fixture("sequence")


def _assert_inf_matrix(actual, expected, what):
    """+inf cells (``null`` in the fixture) must coincide; finite cells within RTOL_DP."""
    assert actual.shape == expected.shape, what
    np.testing.assert_array_equal(np.isinf(actual), np.isinf(expected), err_msg=what)
    fin = np.isfinite(expected)
    assert_peak_close(actual[fin], expected[fin], RTOL_DP, what)


# ---------------------------------------------------------------------------
# DTW
# ---------------------------------------------------------------------------


def test_dtw_cases(fx):
    inputs = {k: fixture_array(v) for k, v in fx["dtw"]["inputs"].items()}
    n = 0
    for name, case in fx["dtw"]["cases"].items():
        cfg = {k: (np.asarray(v) if isinstance(v, list) else v) for k, v in case["config"].items()}
        if "C" in case:
            D, wp, steps = Q.dtw(C=inputs[case["C"]], return_steps=True, **cfg)
        else:
            D, wp, steps = Q.dtw(inputs[case["X"]], inputs[case["Y"]], return_steps=True, **cfg)
        _assert_inf_matrix(D, fixture_array(case["D"]), f"dtw/{name}: D")
        assert wp.dtype == np.int64
        np.testing.assert_array_equal(wp, np.asarray(case["wp"]), err_msg=f"dtw/{name}: wp")
        np.testing.assert_array_equal(steps, np.asarray(case["steps"]), err_msg=f"dtw/{name}: steps")
        n += 1
    assert n >= 18


def test_dtw_return_conventions(fx):
    inputs = fx["dtw"]["inputs"]
    x, y = fixture_array(inputs["xa"]), fixture_array(inputs["ya"])
    D = Q.dtw(x, y, backtrack=False)
    assert isinstance(D, np.ndarray) and D.shape == (14, 19)
    D2, wp = Q.dtw(x, y)
    np.testing.assert_array_equal(D, D2)
    assert tuple(wp[0]) == (13, 18) and tuple(wp[-1]) == (0, 0)
    # 1-D sequences are one feature, as librosa's atleast_2d.
    D1, _ = Q.dtw(x[0], y[0])
    D1b, _ = Q.dtw(x[:1], y[:1])
    np.testing.assert_array_equal(D1, D1b)
    with pytest.raises(ValueError):
        Q.dtw(x)  # Y missing
    with pytest.raises(ValueError):
        Q.dtw(x, y, C=np.ones((2, 2)))
    with pytest.raises(ValueError, match="metric"):
        Q.dtw(x, y, metric="mahalanobis")


def test_dtw_backtracking(fx):
    for name, case in fx["dtw_backtracking"].items():
        sigma = case["step_sizes_sigma"]
        wp = Q.dtw_backtracking(
            np.asarray(case["steps"]),
            step_sizes_sigma=None if sigma is None else np.asarray(sigma),
            subseq=case["subseq"],
            start=case["start"],
        )
        np.testing.assert_array_equal(wp, np.asarray(case["wp"]), err_msg=name)


# ---------------------------------------------------------------------------
# Viterbi
# ---------------------------------------------------------------------------


def _opt(case, key):
    v = case[key]
    return None if v is None else fixture_array(v)


def test_viterbi(fx):
    for name, case in fx["viterbi"].items():
        states, logp = Q.viterbi(
            fixture_array(case["prob"]),
            fixture_array(case["transition"]),
            p_init=_opt(case, "p_init"),
            transition_min_prob=case["min_prob"],
            return_logp=True,
        )
        np.testing.assert_array_equal(states, np.asarray(case["states"]), err_msg=name)
        assert abs(logp - case["logp"]) <= RTOL_DP * abs(case["logp"]), name
        only = Q.viterbi(fixture_array(case["prob"]), fixture_array(case["transition"]), p_init=_opt(case, "p_init"),
                         transition_min_prob=case["min_prob"])
        np.testing.assert_array_equal(only, states)


def test_viterbi_discriminative(fx):
    for name, case in fx["viterbi_discriminative"].items():
        states, logp = Q.viterbi_discriminative(
            fixture_array(case["prob"]),
            fixture_array(case["transition"]),
            p_state=_opt(case, "p_state"),
            p_init=_opt(case, "p_init"),
            transition_min_prob=case["min_prob"],
            return_logp=True,
        )
        np.testing.assert_array_equal(states, np.asarray(case["states"]), err_msg=name)
        assert abs(logp - case["logp"]) <= RTOL_DP * abs(case["logp"]), name


def test_viterbi_binary(fx):
    for name, case in fx["viterbi_binary"].items():
        states, logp = Q.viterbi_binary(
            fixture_array(case["prob"]),
            fixture_array(case["transition"]),
            p_state=_opt(case, "p_state"),
            p_init=_opt(case, "p_init"),
            transition_min_prob=case["min_prob"],
            return_logp=True,
        )
        np.testing.assert_array_equal(states, np.asarray(case["states"]), err_msg=name)
        assert_peak_close(logp, fixture_array(case["logp"]), RTOL_DP, name)
    # A plain (2, 2) transition is accepted (shared by all states).
    case = fx["viterbi_binary"]["shared"]
    s2 = Q.viterbi_binary(fixture_array(case["prob"]), fixture_array(case["transition"])[0])
    np.testing.assert_array_equal(s2, np.asarray(case["states"]))


def test_viterbi_validation():
    with pytest.raises(ValueError, match="transition"):
        Q.viterbi(np.full((2, 3), 0.5), np.ones((2, 2)))
    with pytest.raises(ValueError):
        Q.viterbi(np.full((2, 3), 0.5), np.eye(3))


# ---------------------------------------------------------------------------
# Transition matrices
# ---------------------------------------------------------------------------


def test_transitions(fx):
    for name, case in fx["transition"].items():
        a = case["args"]
        if a["kind"] == "uniform":
            t = Q.transition_uniform(a["n"])
        elif a["kind"] == "loop":
            p = a["prob"][0] if len(a["prob"]) == 1 else a["prob"]
            t = Q.transition_loop(a["n"], p)
        elif a["kind"] == "cycle":
            p = a["prob"][0] if len(a["prob"]) == 1 else a["prob"]
            t = Q.transition_cycle(a["n"], p)
        else:
            w = a["width"][0] if len(a["width"]) == 1 else a["width"]
            t = Q.transition_local(a["n"], w, window=a["window"], wrap=a["wrap"])
        # Closed-form window values and normalisation; cosine ulps only.
        assert_peak_close(t, fixture_array(case["T"]), 1e-15, name)
    with pytest.raises(ValueError):
        Q.transition_loop(1, 0.5)
    with pytest.raises(ValueError):
        Q.transition_uniform(0)


# ---------------------------------------------------------------------------
# RQA
# ---------------------------------------------------------------------------


def test_rqa(fx):
    sims = {k: fixture_array(v) for k, v in fx["rqa"]["sims"].items()}
    for name, case in fx["rqa"]["cases"].items():
        score, path = Q.rqa(
            sims[case["sim"]],
            gap_onset=case["gap_onset"],
            gap_extend=case["gap_extend"],
            knight_moves=case["knight_moves"],
        )
        # Same additions in the same order; numba differs by an ulp in a few cells.
        assert_peak_close(score, fixture_array(case["score"]), 1e-15, f"rqa/{name}")
        np.testing.assert_array_equal(path, np.asarray(case["path"]).reshape(-1, 2), err_msg=name)
    s = Q.rqa(sims["binary"], backtrack=False)
    assert isinstance(s, np.ndarray)


# ---------------------------------------------------------------------------
# Recurrence / cross-similarity / lag
# ---------------------------------------------------------------------------


def _translate(cfg):
    cfg = dict(cfg)
    cfg.pop("ref_cols", None)
    return cfg


def test_recurrence_matrix(fx):
    X = fixture_array(fx["segment_input"])
    for name, case in fx["recurrence"].items():
        R = Q.recurrence_matrix(X, **case["config"])  # "self" passes through **kwargs
        expected = fixture_array(case["R"])
        if case["config"].get("mode", "connectivity") == "connectivity":
            assert R.dtype == np.bool_
            np.testing.assert_array_equal(R.astype(float), expected, err_msg=name)
        else:
            assert_peak_close(R, expected, RTOL_KNN, f"recurrence/{name}")
    np.testing.assert_array_equal(
        Q.recurrence_matrix(X, k=5, sparse=True).toarray(), Q.recurrence_matrix(X, k=5)
    )
    # self_links= is the keyword-friendly spelling of librosa's self=.
    np.testing.assert_array_equal(
        Q.recurrence_matrix(X, k=5, self_links=True), Q.recurrence_matrix(X, k=5, **{"self": True})
    )


def test_cross_similarity(fx):
    data = fixture_array(fx["cross_data"])
    ref = fixture_array(fx["cross_ref"])
    for name, case in fx["cross_similarity"].items():
        cfg = case["config"]
        r = ref[:, : cfg["ref_cols"]] if "ref_cols" in cfg else ref
        R = Q.cross_similarity(data, r, **_translate(cfg))
        expected = fixture_array(case["R"])
        if cfg.get("mode", "connectivity") == "connectivity":
            np.testing.assert_array_equal(R.astype(float), expected, err_msg=name)
        else:
            assert_peak_close(R, expected, RTOL_KNN, f"cross_similarity/{name}")


def test_lag_conversions(fx):
    rec = fixture_array(fx["lag"]["rec"])
    for key in ("pad_True", "pad_False"):
        case = fx["lag"][key]
        lag = Q.recurrence_to_lag(rec, pad=case["pad"])
        # Pure index permutation: exact.
        np.testing.assert_array_equal(lag, fixture_array(case["lag"]))
        np.testing.assert_array_equal(Q.lag_to_recurrence(lag), rec)
    # bool in, bool out (as librosa).
    b = rec > 0
    assert Q.recurrence_to_lag(b).dtype == np.bool_


# ---------------------------------------------------------------------------
# sync
# ---------------------------------------------------------------------------

AGGREGATES = {"mean": np.mean, "median": np.median, "max": np.max, "min": np.min, "sum": np.sum}


def test_sync(fx):
    data = fixture_array(fx["sync_input"])
    n = 0
    for name, case in fx["sync"].items():
        agg = AGGREGATES[case["aggregate"]]
        out = Q.sync(data, case["idx"], aggregate=agg, pad=case["pad"])
        if not case["idx"]:
            # Documented difference: librosa treats an empty index list as an empty
            # list of slices (0 columns); we aggregate the single segment [0, n).
            assert out.shape == (data.shape[0], 1)
            continue
        # numpy's pairwise summation vs our sequential sum over <= 20 values.
        assert_peak_close(out, fixture_array(case["out"]), 1e-15, f"sync/{name}")
        # Names and generic callables give the same result.
        np.testing.assert_allclose(Q.sync(data, case["idx"], aggregate=case["aggregate"], pad=case["pad"]), out, rtol=0, atol=0)
        n += 1
    assert n > 30
    custom = Q.sync(data, [5, 10], aggregate=lambda x, axis: float(np.ptp(x)))
    np.testing.assert_allclose(custom[:, 0], np.ptp(data[:, :5], axis=1))


def test_sync_index_forms_agree():
    data = np.arange(60.0).reshape(3, 20)
    expected = Q.sync(data, np.array([4, 9, 15]))
    # Lists and tuples of ints are read directly; other forms convert.
    for idx in ([4, 9, 15], (4, 9, 15), [4.0, 9.0, 15.0], np.array([4, 9, 15], dtype=np.int32),
                [15, 4, 9, 9], [4, True * 9, 15]):
        np.testing.assert_array_equal(Q.sync(data, idx), expected)
    with pytest.raises(ValueError, match=">= 0"):
        Q.sync(data, [4, -1])
    with pytest.raises(ValueError, match=">= 0"):
        Q.sync(data, (4, -1))
    with pytest.raises(ValueError, match="integer"):
        Q.sync(data, [4, 2.5])


# ---------------------------------------------------------------------------
# Direct comparison with librosa (when installed)
# ---------------------------------------------------------------------------


def test_direct_librosa_comparison():
    librosa = pytest.importorskip("librosa")
    rng = np.random.default_rng(7)
    X = rng.standard_normal((6, 25))
    Y = rng.standard_normal((6, 31))
    for metric in ("euclidean", "cosine", "cityblock"):
        D, wp = Q.dtw(X, Y, metric=metric, subseq=True)
        LD, lwp = librosa.sequence.dtw(X, Y, metric=metric, subseq=True)
        assert_peak_close(D, LD, RTOL_DP, f"dtw {metric}")
        np.testing.assert_array_equal(wp, lwp)
    prob = rng.uniform(size=(5, 50))
    trans = librosa.sequence.transition_local(5, 3, window="hamming", wrap=True)
    # Window values via cos: ulp-level (8e-16 relative measured).
    assert_peak_close(Q.transition_local(5, 3, window="hamming", wrap=True), trans, 1e-15, "transition_local")
    s, lp = Q.viterbi(prob, trans, return_logp=True)
    ls, llp = librosa.sequence.viterbi(prob, trans, return_logp=True)
    np.testing.assert_array_equal(s, ls)
    assert abs(lp - float(np.asarray(llp).ravel()[0])) <= RTOL_DP * abs(lp)
    R = Q.recurrence_matrix(X, mode="affinity", sym=True)
    LR = librosa.segment.recurrence_matrix(X, mode="affinity", sym=True)
    assert_peak_close(R, LR, RTOL_KNN, "recurrence affinity")
    beats = [3, 8, 15, 21]
    np.testing.assert_allclose(Q.sync(X, beats, aggregate=np.median), librosa.util.sync(X, beats, aggregate=np.median), rtol=1e-15)
