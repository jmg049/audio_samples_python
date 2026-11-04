"""Execute the Python code blocks of the guide pages, so that they cannot drift.

Every ``.. code-block:: python`` of ``docs/source/moving_from_librosa.rst`` runs in order
in one shared namespace, in a temporary directory, as a reader would run them. Where the
guide says two results match, its ``assert`` lines check it numerically. Blocks of
``docs/source/librosa_migration.rst`` are self-contained and run separately.

A block that uses librosa, soundfile, matplotlib or scikit-learn is skipped when that package
is missing; with fixtures required (``AUDIO_SAMPLES_REQUIRE_FIXTURES=1``, see conftest) that
fails the test instead. The test dependencies include them. torch is not a test dependency:
its block is always skipped when torch is missing.
"""

from __future__ import annotations

import importlib.util
import os
import re
from pathlib import Path

import pytest

from conftest import fixtures_required

DOCS = Path(__file__).resolve().parent.parent / "docs" / "source"
OPTIONAL = ("librosa", "soundfile", "matplotlib", "sklearn", "torch")


def python_blocks(path: Path) -> list[tuple[int, str]]:
    """``(first_line, dedented_code)`` of each python code block."""
    lines = path.read_text(encoding="utf-8").splitlines()
    blocks = []
    i = 0
    while i < len(lines):
        if lines[i].strip() == ".. code-block:: python":
            base = len(lines[i]) - len(lines[i].lstrip())
            j = i + 1
            body = []
            while j < len(lines) and (not lines[j].strip() or len(lines[j]) - len(lines[j].lstrip()) > base):
                body.append(lines[j])
                j += 1
            nonblank = [ln for ln in body if ln.strip()]
            indent = min(len(ln) - len(ln.lstrip()) for ln in nonblank)
            blocks.append((i + 1, "\n".join(ln[indent:] for ln in body).strip("\n") + "\n"))
            i = j
        else:
            i += 1
    return blocks


def needed_packages(code: str) -> list[str]:
    no_comments = "\n".join(ln.split("#", 1)[0] for ln in code.splitlines())
    return [m for m in OPTIONAL if re.search(rf"\b{m}\b", no_comments)]


def require(packages: list[str]) -> None:
    missing = [p for p in packages if importlib.util.find_spec(p) is None]
    if missing:
        msg = f"needs {missing}"
        if fixtures_required() and missing != ["torch"]:
            pytest.fail(msg + "; required under AUDIO_SAMPLES_REQUIRE_FIXTURES")
        pytest.skip(msg)


GUIDE = python_blocks(DOCS / "moving_from_librosa.rst")
LOOKUP = python_blocks(DOCS / "librosa_migration.rst")
_NAMESPACE: dict = {}
_FAILED: list[int] = []


@pytest.fixture(scope="module")
def guide_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("guide")


def test_the_guide_has_blocks():
    assert len(GUIDE) > 30 and len(LOOKUP) >= 6


@pytest.mark.parametrize("line,code", GUIDE, ids=[f"line{n}" for n, _ in GUIDE])
def test_guide_block(line, code, guide_dir, monkeypatch):
    require(needed_packages(code))
    if _FAILED:
        pytest.fail(f"an earlier block failed (line {_FAILED[0]}); the blocks share a namespace")
    monkeypatch.chdir(guide_dir)
    monkeypatch.setenv("MPLBACKEND", "Agg")
    try:
        exec(compile("\n" * (line - 1) + code, "moving_from_librosa.rst", "exec"), _NAMESPACE)
    except Exception:
        _FAILED.append(line)
        raise


@pytest.mark.parametrize("line,code", LOOKUP, ids=[f"line{n}" for n, _ in LOOKUP])
def test_lookup_block(line, code, tmp_path, monkeypatch):
    require(needed_packages(code))
    monkeypatch.chdir(tmp_path)
    exec(compile("\n" * (line - 1) + code, "librosa_migration.rst", "exec"), {})


def test_internal_links_and_references():
    """Local targets the guide refers to exist."""
    text = (DOCS / "moving_from_librosa.rst").read_text(encoding="utf-8")
    titles = {
        lines[i].strip()
        for lines in [text.splitlines()]
        for i in range(len(lines) - 1)
        if lines[i].strip() and set(lines[i + 1].strip()) <= {"-", "~", "="} and lines[i + 1].strip()
    }
    for target in re.findall(r"(?<![`\w])`([^`<>]+)`_(?!_)", text):
        assert " ".join(target.split()) in titles or target == "changed defaults", target
    assert os.path.exists(DOCS / "librosa_migration.rst")
