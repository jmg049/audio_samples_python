"""Execute the Python code blocks of README.md so that README drift is caught.

Every ```python block is run, in order, in one shared namespace (later blocks
reuse variables from earlier ones, as a reader would). A block is skipped when
the line right before its opening fence is an HTML comment starting with
``<!-- readme-test: skip`` (used for blocks that need external files or other
libraries). The blocks run inside a temporary directory, so files they write
do not land in the repository.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

README = Path(__file__).resolve().parent.parent / "README.md"
SKIP_MARKER = "<!-- readme-test: skip"


def _python_blocks() -> list[tuple[int, str, bool]]:
    """Return ``(line_number, code, skipped)`` for each python block."""
    lines = README.read_text(encoding="utf-8").splitlines()
    blocks = []
    i = 0
    while i < len(lines):
        if re.match(r"^```python\s*$", lines[i]):
            start = i + 1
            j = start
            while j < len(lines) and not lines[j].startswith("```"):
                j += 1
            prev = lines[i - 1].strip() if i > 0 else ""
            blocks.append((start + 1, "\n".join(lines[start:j]), prev.startswith(SKIP_MARKER)))
            i = j + 1
        else:
            i += 1
    return blocks


BLOCKS = _python_blocks()
_NAMESPACE: dict = {}


def test_readme_has_python_blocks():
    assert len(BLOCKS) >= 10
    assert any(not skipped for _, _, skipped in BLOCKS)


@pytest.mark.parametrize(
    "lineno,code,skipped", BLOCKS, ids=[f"README.md:{ln}" for ln, _, _ in BLOCKS]
)
def test_readme_block(lineno, code, skipped, tmp_path, monkeypatch):
    if skipped:
        pytest.skip(f"README block at line {lineno} is marked readme-test: skip")
    monkeypatch.chdir(tmp_path)
    if not _NAMESPACE:
        exec("import audio_samples as aus\nimport numpy as np", _NAMESPACE)
    exec(compile(code, f"README.md:{lineno}", "exec"), _NAMESPACE)
