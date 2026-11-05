"""``audio_samples.notation`` against librosa fixtures (``notation.json``). Names and
factorisations must match exactly; ratios agree to 1e-14 (see
``tests/librosa_reference/notation.rs``).
"""

from __future__ import annotations

import numpy as np
import pytest

from audio_samples import notation as N

from conftest import load_librosa_fixture


@pytest.fixture(scope="module")
def fx():
    return load_librosa_fixture("notation")


def _factors(values):
    return [{str(k): v for k, v in d.items()} for d in values]


def test_tuning(fx):
    t = fx["tuning"]
    for b in (7, 12, 24, 53):
        for sort in (True, False):
            np.testing.assert_allclose(N.pythagorean_intervals(bins_per_octave=b, sort=sort), t[f"pyth_{b}_{sort}"], rtol=1e-14)
            assert _factors(N.pythagorean_intervals(bins_per_octave=b, sort=sort, return_factors=True)) == t[f"pyth_factors_{b}_{sort}"]
    for primes, b in (([3], 12), ([3, 5], 12), ([3, 5], 24), ([3, 5, 7], 16), ([3, 11], 9), ([5], 5)):
        for sort in (True, False):
            key = "_".join(map(str, primes)) + f"_{b}_{sort}"
            np.testing.assert_allclose(N.plimit_intervals(primes=primes, bins_per_octave=b, sort=sort), t[f"plimit_{key}"], rtol=1e-14)
            assert _factors(N.plimit_intervals(primes=primes, bins_per_octave=b, sort=sort, return_factors=True)) == t[f"plimit_factors_{key}"]


def test_frequencies(fx):
    c = fx["frequencies"]
    for name in ("equal", "pythagorean", "ji3", "ji5", "ji7"):
        for sort in (True, False):
            np.testing.assert_allclose(N.interval_frequencies(30, fmin=55.0, intervals=name, sort=sort), c[f"{name}_{sort}"], rtol=1e-14)
    np.testing.assert_allclose(N.interval_frequencies(30, fmin=55.0, intervals="equal", tuning=0.25), c["equal_tuned"], rtol=1e-14)
    np.testing.assert_allclose(N.interval_frequencies(7, fmin=100.0, intervals=[1.0, 1.25, 1.5]), c["custom"], rtol=1e-14)
    with pytest.raises(ValueError):
        N.interval_frequencies(3, fmin=55.0, intervals="nope")


def test_fjs(fx):
    for c in fx["fifths"]:
        assert N.fifths_to_note(unison=c["unison"], fifths=c["fifths"], unicode=c["unicode"]) == c["out"]
    t = fx["fjs_table"]
    r = np.array(t["intervals"])
    assert N.interval_to_fjs(r).tolist() == t["unicode"]
    assert N.interval_to_fjs(r, unison="F#", unicode=False).tolist() == t["ascii"]
    assert N.interval_to_fjs(1.5) == "G"
    c = fx["fjs_octaves"]
    assert N.interval_to_fjs(np.array(c["intervals"])).tolist() == c["out"]
    c = fx["hz_to_fjs"]
    assert N.hz_to_fjs(c["hz"]).tolist() == c["default"]
    assert N.hz_to_fjs(c["hz"], fmin=110.0, unison="D", unicode=True).tolist() == c["fmin"]
    with pytest.raises(ValueError):
        N.interval_to_fjs(-1.0)
    with pytest.raises(ValueError):
        N.fifths_to_note(unison="X", fifths=1)


def test_tables(fx):
    assert N.list_thaat() == fx["list_thaat"]
    for t, d in fx["thaat_degrees"].items():
        assert N.thaat_to_degrees(t).tolist() == d
    assert N.list_mela() == fx["list_mela"]
    for i, d in enumerate(fx["mela_degrees"], start=1):
        assert N.mela_to_degrees(i).tolist() == d
    assert N.mela_to_degrees("Kanakangi").tolist() == fx["mela_degrees"][0]
    for key, table in fx["mela_svara"].items():
        abbr, uni = (k == "True" for k in key.split("_"))
        for i, want in enumerate(table, start=1):
            assert N.mela_to_svara(i, abbr=abbr, unicode=uni) == want
    for bad in (lambda: N.mela_to_degrees(0), lambda: N.mela_to_svara("nope"), lambda: N.thaat_to_degrees("nope")):
        with pytest.raises(ValueError):
            bad()


def test_svara(fx):
    c = fx["svara"]
    midi = np.array([np.nan if m is None else m for m in c["midi"]])
    for abbr in (True, False):
        for octave in (True, False):
            for uni in (True, False):
                kw = dict(abbr=abbr, octave=octave, unicode=uni)
                key = f"{abbr}_{octave}_{uni}"
                assert N.midi_to_svara_h(midi, Sa=60.0, **kw).tolist() == c[f"h_{key}"]
                assert N.midi_to_svara_c(midi, Sa=60.0, mela=31, **kw).tolist() == c[f"c_{key}"]
    assert N.midi_to_svara_h(60.0, Sa=60.0) == "S"
    hz = np.array(c["hz"])
    assert N.hz_to_svara_h(hz, Sa=261.63).tolist() == c["hz_h"]
    assert N.hz_to_svara_c(hz, Sa=261.63, mela="kosalam").tolist() == c["hz_c"]
    assert N.note_to_svara_h(c["notes"], Sa="C4").tolist() == c["note_h"]
    assert N.note_to_svara_c(c["notes"], Sa="C4", mela=1).tolist() == c["note_c"]
    assert N.note_to_svara_h("C4", Sa="C4") == "S"
