"""The migration checker: its table against reality, and its advice on real source text."""

from __future__ import annotations

import ast
import inspect
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import audio_samples as aus
from audio_samples._migrate_table import (
    NOT_PROVIDED,
    REMOVED,
    TABLE,
    Default,
    Renamed,
)
from audio_samples.migrate import SHAPE_DIFFERS, check_paths, check_source, main

EMPTY = inspect.Parameter.empty
GUIDE = Path(__file__).resolve().parent.parent / "docs" / "source" / "librosa_migration.rst"

# Public names that are not functions of the listed modules' __all__ but are worth keeping.
_NON_FUNCTION_ENTRIES = {
    "librosa.__version__",
    "librosa.version.version",
    "librosa.version.short_version",
    "librosa.version.show_versions",
    "librosa.util.MAX_MEM_BLOCK",
    "librosa.util.valid_intervals",
    "librosa.util.match_intervals",
    "librosa.util.Deprecated",
}
_MODULES = (
    "librosa",
    "librosa.feature",
    "librosa.feature.inverse",
    "librosa.effects",
    "librosa.onset",
    "librosa.beat",
    "librosa.decompose",
    "librosa.filters",
    "librosa.util",
    "librosa.segment",
    "librosa.sequence",
    "librosa.display",
)


def _resolve_ours(name: str):
    """The attribute of audio_samples that a table name refers to."""
    head, _, rest = name.partition(".")
    if head == "AudioSamples":
        return getattr(aus.AudioSamples, rest)
    if head == "audio_samples":
        return getattr(aus, rest)
    return getattr(getattr(aus, head), rest)


def _librosa_object(name: str):
    import importlib

    parts = name.split(".")
    for end in range(len(parts), 0, -1):
        try:
            obj = importlib.import_module(".".join(parts[:end]))
        except ImportError:
            continue
        for attr in parts[end:]:
            obj = getattr(obj, attr)
        return obj
    raise LookupError(name)


def _public_librosa_names():
    librosa = pytest.importorskip("librosa")
    import importlib

    assert librosa.__version__.startswith("1."), "the table describes librosa 1.x"
    names = set()
    for module_name in _MODULES:
        module = importlib.import_module(module_name)
        for attr in module.__all__:
            value = getattr(module, attr)
            if inspect.ismodule(value):
                continue
            names.add(f"{module_name}.{attr}")
    return names


def test_every_public_librosa_function_is_in_the_table():
    missing = sorted(_public_librosa_names() - TABLE.keys())
    assert missing == [], f"not in the table: {missing}"


def test_every_table_name_exists_in_librosa():
    pytest.importorskip("librosa")
    unknown = []
    for name in TABLE:
        try:
            _librosa_object(name)
        except (AttributeError, LookupError):
            unknown.append(name)
    assert unknown == []


def test_old_names_are_gone_and_new_names_are_in_the_table():
    pytest.importorskip("librosa")
    for old, new in REMOVED.items():
        with pytest.raises((AttributeError, LookupError)):
            _librosa_object(old)
        if new.startswith("librosa."):
            assert new in TABLE


def _signature(name: str):
    obj = _librosa_object(name)
    if not callable(obj) or name in _NON_FUNCTION_ENTRIES:
        return None
    try:
        return inspect.signature(obj)
    except (TypeError, ValueError):
        return None


_ENTRIES = sorted(TABLE)


@pytest.mark.parametrize("name", _ENTRIES)
def test_parameters_match_librosa(name):
    pytest.importorskip("librosa")
    signature = _signature(name)
    if signature is None:
        return
    entry = TABLE[name]
    if inspect.isclass(_librosa_object(name)) or name.startswith("librosa.display."):
        return  # constructors and plotting: no advice depends on their parameters
    positional, named, variadic, takes_keywords = [], [], False, False
    for parameter in signature.parameters.values():
        if parameter.kind in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD):
            positional.append(parameter.name)
        elif parameter.kind == parameter.VAR_POSITIONAL:
            positional.append(parameter.name)
            variadic = True
        elif parameter.kind == parameter.KEYWORD_ONLY:
            named.append(parameter.name)
        else:
            takes_keywords = True
    assert entry.positional == tuple(positional), name
    assert entry.parameters == frozenset([*positional, *named]), sorted(entry.parameters ^ set(positional + named))
    assert entry.var_positional == variadic, name
    assert entry.takes_keywords == takes_keywords, name


def _same_default(a, b) -> bool:
    if a is b or repr(a) == repr(b):
        return True
    if callable(a) and isinstance(b, str):
        return a.__name__ == b  # numpy.max and "max"
    try:
        return bool(a == b)
    except (TypeError, ValueError):
        return False


def _forms(entry):
    """(primary function, options, keywords that select the form or another form)."""
    selecting = {keyword for form in entry.forms for keyword in form.when}
    if entry.ours:
        yield entry.ours[0], entry.options, selecting
    for form in entry.forms:
        yield form.ours[0], form.options, selecting


_WITH_EQUIVALENT = [n for n in _ENTRIES if TABLE[n].ours and not n.startswith("librosa.display.")]


@pytest.mark.parametrize("name", _WITH_EQUIVALENT)
def test_each_keyword_is_accounted_for(name):
    """Every librosa parameter exists in the equivalent with the same default, or the table says what differs."""
    pytest.importorskip("librosa")
    entry = TABLE[name]
    if not callable(_librosa_object(name)):
        pytest.skip("a constant has no parameters")
    theirs = inspect.signature(_librosa_object(name)).parameters
    problems = []
    for primary, options, selecting in _forms(entry):
        try:
            ours = inspect.signature(_resolve_ours(primary)).parameters
        except (TypeError, ValueError):
            continue  # no text signature (a class or a static helper)
        for parameter in theirs.values():
            key = parameter.name
            if parameter.kind == parameter.VAR_KEYWORD or key in options or key in selecting:
                continue
            if key not in ours:
                problems.append(f"{primary}: no {key}")
                continue
            if parameter.default is not EMPTY and repr(parameter.default) == "<DEPRECATED parameter>":
                continue
            if not _same_default(parameter.default, ours[key].default):
                problems.append(f"{primary}: {key} default {parameter.default!r} vs {ours[key].default!r}")
    assert problems == []


@pytest.mark.parametrize("name", _WITH_EQUIVALENT)
def test_options_describe_real_keywords_and_real_defaults(name):
    pytest.importorskip("librosa")
    entry = TABLE[name]
    if not callable(_librosa_object(name)):
        pytest.skip("a constant has no parameters")
    theirs = inspect.signature(_librosa_object(name)).parameters
    for _, options, _ in _forms(entry):
        for key, option in options.items():
            if key not in theirs:
                # Forwarded through librosa's **kwargs.
                assert entry.takes_keywords, f"{key} is not a parameter of {name}"
                continue
            if isinstance(option, Default):
                actual = theirs[key].default
                if callable(actual):
                    assert repr(option.librosa).endswith(actual.__name__), key
                elif actual is EMPTY:
                    pytest.fail(f"{key} of {name} has no default in librosa")
                else:
                    assert repr(option.librosa) == repr(actual), key


def test_renamed_keywords_exist_in_an_equivalent():
    for name, entry in TABLE.items():
        for primary, options, _ in _forms(entry):
            candidates = [primary, *entry.ours]
            parameters = set()
            for candidate in candidates:
                try:
                    parameters |= set(inspect.signature(_resolve_ours(candidate)).parameters)
                except (TypeError, ValueError):
                    parameters.add("*no signature*")
            for key, option in options.items():
                if isinstance(option, Renamed) and "*no signature*" not in parameters:
                    assert option.to in parameters, f"{name}: {key} -> {option.to} in {candidates}"


def test_every_equivalent_exists_in_audio_samples():
    for name, entry in TABLE.items():
        for ours in entry.ours:
            assert _resolve_ours(ours) is not None, f"{name}: {ours}"
        for form in entry.forms:
            for ours in form.ours:
                assert _resolve_ours(ours) is not None, f"{name}: {ours}"


def _table_text():
    for entry in TABLE.values():
        yield entry.note
        yield entry.shape or ""
        yield from entry.behaviour
        groups = [entry.options, *(form.options for form in entry.forms)]
        for form in entry.forms:
            yield form.note or ""
            yield form.shape or ""
            yield from form.behaviour
        for options in groups:
            for option in options.values():
                for name in ("note", "reason", "advice", "value_note"):
                    yield str(getattr(option, name, ""))


def test_names_in_the_advice_exist_in_audio_samples():
    pattern = re.compile(r"(?<![\w.])(AudioSamples|io|spectral|features|rhythm|effects|sequence|utils|generation|forms|arrays|notation)\.([A-Za-z_]\w*)")
    missing = set()
    for text in _table_text():
        for head, attribute in pattern.findall(text):
            try:
                _resolve_ours(f"{head}.{attribute}")
            except AttributeError:
                missing.add(f"{head}.{attribute}")
    assert missing == set()


def test_not_provided_entries_have_a_reason_and_no_equivalent():
    for name, entry in TABLE.items():
        if entry.match == NOT_PROVIDED:
            assert entry.ours == () and entry.note, name
        else:
            assert entry.ours, name


# --- The table against the document ---------------------------------------


def _document_rows():
    """(section, cells) for each row of the equivalence tables in the guide."""
    lines = GUIDE.read_text().split("\n")
    section = ""
    rows = []
    for i, line in enumerate(lines):
        if i + 1 < len(lines) and re.fullmatch(r"[~-]{3,}", lines[i + 1]):
            section = line.strip()
        if line.startswith("   * - "):
            cells = [line[7:]]
            j = i + 1
            while j < len(lines) and not lines[j].startswith("   * - ") and lines[j].startswith("     "):
                text = lines[j]
                if text.startswith("     - "):
                    cells.append(text[7:])
                elif text.strip() == "-":
                    cells.append("")
                else:
                    cells[-1] += " " + text.strip()
                j += 1
            rows.append((section, cells))
    return rows


def _prefix(section: str) -> str | None:
    if section.startswith("Core"):
        return "librosa"
    return {
        "``librosa.feature``": "librosa.feature",
        "``librosa.effects``": "librosa.effects",
        "``librosa.onset`` and ``librosa.beat``": "librosa",
        "``librosa.decompose``": "librosa.decompose",
        "``librosa.filters``": "librosa.filters",
        "``librosa.util``": "librosa.util",
        "``librosa.segment`` and ``librosa.sequence``": "librosa",
        "``librosa.display``": "librosa.display",
    }.get(section)


def _document_entries():
    entries = {}
    for section, cells in _document_rows():
        prefix = _prefix(section)
        if prefix is None or cells[0] == "librosa":
            continue
        names = re.findall(r"``([^`]+)``", cells[0])
        if "..." in cells[0]:
            names = [f"{letter}_weighting" for letter in "ABCDZ"]
        if len(cells) == 3:
            ours_tokens, status = [], cells[1]
        else:
            ours_tokens = [t for t in re.findall(r"``([^`]+)``", cells[1]) if "=" not in t]
            status = cells[2]
        ours, previous = [], ""
        for token in ours_tokens:
            if "." not in token and previous:
                token = f"{previous}.{token}"
            previous = token.rpartition(".")[0]
            ours.append(token)
        for name in names:
            entries[f"{prefix}.{name}"] = (status, tuple(ours))
    return entries


def test_table_matches_the_document():
    document = _document_entries()
    assert len(document) > 200
    problems = []
    for name, (status, ours) in document.items():
        entry = TABLE.get(name)
        if entry is None:
            problems.append(f"{name}: in the document, not in the table")
            continue
        if entry.match != status:
            problems.append(f"{name}: document {status!r}, table {entry.match!r}")
        wanted = set(ours)
        if name.endswith("_weighting") and name[8] in "ABCDZ":
            wanted = {f"features.{name[8].lower()}_weighting"}
        if set(entry.ours) != wanted:
            problems.append(f"{name}: document {sorted(wanted)}, table {sorted(entry.ours)}")
    assert problems == []


# --- Resolving names -------------------------------------------------------


def _names(source: str) -> list[str]:
    return [f.librosa for f in check_source(source)]


@pytest.mark.parametrize(
    "source, expected",
    [
        ("import librosa\nlibrosa.stft(y)", ["librosa.stft"]),
        ("import librosa as lb\nlb.stft(y)", ["librosa.stft"]),
        ("from librosa.feature import mfcc\nmfcc(y=y)", ["librosa.feature.mfcc"]),
        ("from librosa import stft as s\ns(y)", ["librosa.stft"]),
        ("import librosa.feature as lf\nlf.mfcc(y=y)", ["librosa.feature.mfcc"]),
        ("from librosa import feature as F\nF.rms(y=y)", ["librosa.feature.rms"]),
        ("import librosa.feature\nlibrosa.feature.rms(y=y)", ["librosa.feature.rms"]),
        ("from librosa.core import stft\nstft(y)", ["librosa.stft"]),
        ("import librosa\nlibrosa.core.stft(y)", ["librosa.stft"]),
        ("import librosa\nlibrosa.core.spectrum.stft(y)", ["librosa.stft"]),
        ("from librosa.feature.spectral import mfcc\nmfcc(y=y)", ["librosa.feature.mfcc"]),
        ("from librosa import *\nstft(y)", ["librosa.stft"]),
        ("import librosa\ns = librosa.stft\ns(y)", ["librosa.stft"]),
        ("import librosa as lb\nmfcc = lb.feature.mfcc\nmfcc(y=y)", ["librosa.feature.mfcc"]),
        # Imported in a function, and used there.
        ("def f(y):\n    import librosa\n    return librosa.stft(y)", ["librosa.stft"]),
        # An optional import.
        ("try:\n    import librosa\nexcept ImportError:\n    librosa = None\nlibrosa.stft(y)", ["librosa.stft"]),
        # A lazy global.
        ("librosa = None\ndef f():\n    global librosa\n    import librosa\ndef g(y):\n    return librosa.stft(y)", ["librosa.stft"]),
        # A reference rather than a call.
        ("import librosa\nfrom functools import partial\np = partial(librosa.stft, n_fft=512)", ["librosa.stft"]),
        # Nested in another call and in a method chain.
        ("import librosa\nprint(librosa.power_to_db(librosa.stft(y)).T)", ["librosa.power_to_db", "librosa.stft"]),
        # Multi-line call.
        ("import librosa\nlibrosa.stft(\n    y,\n    n_fft=512,\n)", ["librosa.stft"]),
    ],
)
def test_resolves_imports_and_aliases(source, expected):
    assert sorted(_names(source)) == sorted(expected)


@pytest.mark.parametrize(
    "source",
    [
        # Same names, not librosa.
        "def stft(y):\n    return y\nstft(1)",
        "import scipy.signal as librosa\nlibrosa.stft(y)",
        "from librosa.feature import mfcc\ndef f(mfcc):\n    return mfcc(1)",
        "from librosa import stft\n[stft(x) for stft in fns]",
        "from librosa import stft\nf = lambda stft: stft(1)",
        "import librosa\nclass A:\n    def librosa(self):\n        return 1\n    def b(self):\n        return self.librosa()",
        "# librosa.stft(y)\nx = 'librosa.stft(y)'",
        # An unused import is not usage to migrate.
        "import librosa",
        # A local variable in another scope.
        "from librosa import stft\ndef f():\n    stft = 3\n    return stft(1)",
    ],
)
def test_does_not_report_other_names(source):
    assert _names(source) == []


def test_class_attribute_does_not_hide_the_import_inside_methods():
    source = "import librosa\nclass A:\n    librosa = 1\n    def f(self, y):\n        return librosa.stft(y)"
    assert _names(source) == ["librosa.stft"]


# --- Advice specific to the call -------------------------------------------


def _single(source: str):
    findings = [f for f in check_source("import librosa\n" + source) if f.is_call]
    assert len(findings) == 1, findings
    return findings[0]


def _watch_keywords(finding, kind):
    return {w.keyword for w in finding.watch if w.kind == kind}


def test_a_default_is_flagged_only_when_the_call_relies_on_it():
    relying = _single("librosa.power_to_db(S)")
    assert _watch_keywords(relying, "default differs") == {"top_db"}
    assert "80.0" in relying.watch[0].message
    passing = _single("librosa.power_to_db(S, top_db=60)")
    assert _watch_keywords(passing, "default differs") == set()
    none = _single("librosa.power_to_db(S, top_db=None)")
    assert _watch_keywords(none, "default differs") == set()


def test_positional_argument_counts_as_passed():
    finding = _single("librosa.load('a.wav')")
    assert _watch_keywords(finding, "keyword changed") == {"path"}
    assert _watch_keywords(finding, "default differs") == {"sr", "mono", "dtype"}
    finding = _single("librosa.load('a.wav', sr=None, mono=False)")
    assert _watch_keywords(finding, "default differs") == {"dtype"}


def test_unsupported_option_is_flagged_whatever_its_form():
    finding = _single("librosa.load('a.wav', offset=1.0)")
    assert _watch_keywords(finding, "unsupported option") == {"offset"}
    assert finding.blocking


def test_a_value_only_rejected_when_it_is_the_unsupported_one():
    dense = _single("librosa.onset.onset_detect(y=y, sr=sr, sparse=False)")
    assert _watch_keywords(dense, "unsupported option") == {"sparse"}
    sparse = _single("librosa.onset.onset_detect(y=y, sr=sr, sparse=True)")
    assert _watch_keywords(sparse, "unsupported option") == set()
    assert not sparse.blocking
    variable = _single("librosa.onset.onset_detect(y=y, sr=sr, sparse=flag)")
    assert _watch_keywords(variable, "unsupported option") == set()
    assert _watch_keywords(variable, "not checked") == {"sparse"}


def test_a_spelled_out_value_that_is_the_only_one_passes():
    assert not _single("librosa.filters.mel(sr=sr, n_fft=512, dtype=np.float64)").blocking
    other = _single("librosa.filters.mel(sr=sr, n_fft=512, dtype=np.float32)")
    assert _watch_keywords(other, "not checked") == {"dtype"}


def test_the_form_follows_the_keywords():
    from_audio = _single("librosa.feature.spectral_centroid(y=y, sr=sr)")
    assert from_audio.equivalent[0] == "AudioSamples.spectral_centroid_frames"
    assert any(w.kind == SHAPE_DIFFERS for w in from_audio.watch)

    from_spectrogram = _single("librosa.feature.spectral_centroid(S=S)")
    assert from_spectrogram.equivalent == ("features.spectral_centroid_from_spectrogram",)
    assert _watch_keywords(from_spectrogram, "default differs") == {"sr"}

    explicit_none = _single("librosa.feature.spectral_centroid(y=y, S=None)")
    assert explicit_none.equivalent[0] == "AudioSamples.spectral_centroid_frames"

    beats = _single("librosa.beat.beat_track(onset_envelope=e, sr=sr, trim=False)")
    assert beats.equivalent == ("rhythm.beat_track",)
    assert _watch_keywords(beats, "default differs") == set()


def test_keyword_unpacking():
    literal = _single("librosa.power_to_db(S, **{'top_db': 60})")
    assert _watch_keywords(literal, "default differs") == set()
    opaque = _single("librosa.power_to_db(S, **options)")
    assert _watch_keywords(opaque, "default differs") == set()
    assert any(w.kind == "not checked" for w in opaque.watch)


def test_a_keyword_librosa_does_not_have_is_reported():
    finding = _single("librosa.get_duration(filename='a.wav')")
    assert _watch_keywords(finding, "not checked") == {"filename"}


def test_too_many_positional_arguments_are_reported():
    finding = _single("librosa.load('a.wav', 16000)")
    assert any("positional arguments" in w.message for w in finding.watch)


def test_no_counterpart_blocks():
    finding = _single("librosa.cite()")
    assert finding.equivalent == ()
    assert finding.match == NOT_PROVIDED
    assert finding.blocking


def test_old_names_map_to_the_current_function():
    finding = _single("librosa.beat.tempo(y=y)")
    assert finding.librosa == "librosa.feature.tempo"
    assert finding.equivalent[0] == "AudioSamples.tempo"
    old = _single("librosa.output.write_wav('a.wav', y, sr)")
    assert old.match == NOT_PROVIDED
    assert "io.save" in old.note


def test_unknown_attribute_is_reported_as_unknown():
    finding = _single("librosa.not_a_function(y)")
    assert finding.match == "unknown"
    assert finding.blocking


def test_location_and_call_text():
    findings = check_source("import librosa as lb\n\ny = lb.tone(\n    440,\n    sr=8000)\n", "x.py")
    assert findings[0].location == "x.py:3"
    assert findings[0].call == "lb.tone( 440, sr=8000)"


# --- Files and notebooks ---------------------------------------------------


def _notebook(*cells):
    return json.dumps(
        {
            "cells": [
                {"cell_type": kind, "source": source, "metadata": {}, **({"outputs": [], "execution_count": None} if kind == "code" else {})}
                for kind, source in cells
            ],
            "metadata": {},
            "nbformat": 4,
            "nbformat_minor": 5,
        }
    )


def test_notebook_cells_share_imports_and_tolerate_ipython_syntax(tmp_path):
    notebook = tmp_path / "a.ipynb"
    notebook.write_text(
        _notebook(
            ("markdown", "librosa.stft(y) in text"),
            ("code", ["%matplotlib inline\n", "!pip install librosa\n", "import librosa as lb\n"]),
            ("code", "files = !ls\ny, sr = lb.load('a.wav')\nlb.stft?\n"),
            ("code", "%%time\nS = lb.stft(y)\n"),
            ("code", "%%bash\nlibrosa.stft > not_python\n"),
        )
    )
    findings, errors, files = check_paths([notebook])
    assert errors == [] and files == 1
    assert [(f.librosa, f.cell, f.line) for f in findings] == [("librosa.load", 2, 2), ("librosa.stft", 3, 2)]
    assert findings[0].location == f"{notebook}:cell2:2"


def test_directories_are_searched_and_installed_code_skipped(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "a.py").write_text("import librosa\nlibrosa.stft(y)\n")
    (tmp_path / ".venv" / "lib").mkdir(parents=True)
    (tmp_path / ".venv" / "lib" / "b.py").write_text("import librosa\nlibrosa.stft(y)\n")
    findings, errors, files = check_paths([tmp_path])
    assert errors == [] and files == 1 and len(findings) == 1


def test_encoding_declaration_is_honoured(tmp_path):
    path = tmp_path / "latin.py"
    path.write_bytes("# -*- coding: latin-1 -*-\nimport librosa\ns = 'caf\xe9'; librosa.stft(y)\n".encode("latin-1"))
    findings, errors, _ = check_paths([path])
    assert errors == [] and findings[0].call == "librosa.stft(y)"


def _run(*args, tmp_path=None):
    return subprocess.run(
        [sys.executable, "-m", "audio_samples.migrate", *map(str, args)],
        capture_output=True,
        text=True,
    )


def test_exit_status(tmp_path):
    clean = tmp_path / "clean.py"
    clean.write_text("import numpy\n")
    used = tmp_path / "used.py"
    used.write_text("import librosa\nlibrosa.power_to_db(S)\n")
    blocked = tmp_path / "blocked.py"
    blocked.write_text("import librosa\nlibrosa.cite()\n")
    broken = tmp_path / "broken.py"
    broken.write_text("def (:\n")

    assert _run(clean).returncode == 0
    assert _run(used).returncode == 1
    assert _run(used, "--fail-on", "blocking").returncode == 0
    assert _run(blocked, "--fail-on", "blocking").returncode == 1
    assert _run(blocked, "--fail-on", "never").returncode == 0
    assert _run(broken).returncode == 2
    assert _run(broken, "--fail-on", "never").returncode == 2
    assert _run(tmp_path / "missing.py").returncode == 2
    assert _run(used, broken).returncode == 2


def test_json_output_with_summary(tmp_path, capsys):
    path = tmp_path / "a.py"
    path.write_text("import librosa\nlibrosa.power_to_db(S)\nlibrosa.cite()\n")
    status = main(["--format", "json", str(path)])
    report = json.loads(capsys.readouterr().out)
    assert status == 1
    assert report["summary"]["uses"] == 2
    assert report["summary"]["by_match"]["not provided"] == 1
    assert report["summary"]["by_watch"]["default differs"] == 1
    assert report["summary"]["blocking"] == 1
    first = report["findings"][0]
    assert first["librosa"] == "librosa.power_to_db"
    assert first["line"] == 2
    assert first["equivalent"] == ["spectral.power_to_db"]
    assert first["watch"][0]["keyword"] == "top_db"


def test_text_output_has_the_summary(tmp_path, capsys):
    path = tmp_path / "a.py"
    path.write_text("import librosa\nlibrosa.power_to_db(S)\n")
    main([str(path)])
    out = capsys.readouterr().out
    assert f"{path}:2: librosa.power_to_db(S)" in out
    assert "use:   spectral.power_to_db" in out
    assert "match: equivalent" in out
    assert "1 librosa uses of 1 functions in 1 of 1 files" in out


def test_files_are_never_changed(tmp_path):
    path = tmp_path / "a.py"
    source = "import librosa\nlibrosa.stft(y)\n"
    path.write_text(source)
    before = path.stat().st_mtime_ns
    _run(path)
    assert path.read_text() == source
    assert path.stat().st_mtime_ns == before


def test_checker_needs_neither_librosa_nor_numpy_and_import_does_not_load_it(tmp_path):
    code = (
        "import sys\n"
        "sys.modules['librosa'] = None\n"
        "import audio_samples\n"
        "assert 'audio_samples.migrate' not in sys.modules\n"
        "assert 'audio_samples._migrate_table' not in sys.modules\n"
        "assert 'numpy' not in sys.modules\n"
        "from audio_samples.migrate import check_source\n"
        "assert 'numpy' not in sys.modules\n"
        "assert check_source('import librosa\\nlibrosa.stft(y)')[0].librosa == 'librosa.stft'\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True, cwd=tmp_path)
