"""Fail when a stub change alters what any overload gives, under mypy or pyright.

``tests/typing/gen_reveals.py`` writes a ``reveal_type`` call for every overload, property and
function of the stubs into ``tests/typing/reveals/``. This script type-checks those files with mypy
and pyright, collects every revealed type and compares them with ``tests/typing/reveals.snapshot``.

    python tests/typing/check_precision.py           # check
    python tests/typing/check_precision.py --update  # accept the current reveals

A difference fails the check. Lost entries and reveals that gained ``Any`` are reported as
precision losses; other changes are listed for review. Update the snapshot only for intended
changes; the snapshot diff then shows the change in review.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REVEALS = ROOT / "tests" / "typing" / "reveals"
SNAPSHOT = ROOT / "tests" / "typing" / "reveals.snapshot"


def tags_by_line(path: Path) -> dict[int, str]:
    out, current = {}, None
    for i, line in enumerate(path.read_text().splitlines(), 1):
        m = re.match(r"def (r_\w+)\(", line)
        if m:
            current = m.group(1)
        elif "reveal_type(" in line and current:
            out[i] = current
    return out


def tool(name: str) -> str:
    exe = Path(sys.executable).parent / name
    return str(exe) if exe.exists() else (shutil.which(name) or name)


def mypy_reveals(files: list[Path]) -> dict[str, str]:
    proc = subprocess.run(
        [sys.executable, "-m", "mypy", "--config-file", str(ROOT / "pyproject.toml"), "--no-incremental",
         "--hide-error-context", "--no-error-summary", "--show-column-numbers", *map(str, files)],
        cwd=ROOT, capture_output=True, text=True,
    )
    lines = {p.resolve(): tags_by_line(p) for p in files}
    out, errors = {}, []
    for line in proc.stdout.splitlines():
        m = re.match(r"(.+?):(\d+):\d+: (\w+): (.*)", line)
        if not m:
            continue
        path, lineno, kind, msg = Path(ROOT / m.group(1)).resolve(), int(m.group(2)), m.group(3), m.group(4)
        tag = lines.get(path, {}).get(lineno)
        if kind == "note" and msg.startswith("Revealed type is ") and tag:
            out[f"{path.stem}.{tag}"] = msg[len("Revealed type is "):].strip('"')
        elif kind == "error":
            errors.append(line)
    if errors:
        print("mypy reports errors in the generated calls (a stub overload cannot be called with its own types):")
        print("\n".join(errors))
        sys.exit(1)
    return out


def pyright_reveals(files: list[Path]) -> dict[str, str]:
    proc = subprocess.run(
        [tool("pyright"), "--outputjson", "--pythonpath", sys.executable, *map(str, files)],
        cwd=ROOT, capture_output=True, text=True,
    )
    data = json.loads(proc.stdout)
    lines = {p.resolve(): tags_by_line(p) for p in files}
    out, errors = {}, []
    for d in data.get("generalDiagnostics", []):
        path = Path(d["file"]).resolve()
        lineno = d["range"]["start"]["line"] + 1
        tag = lines.get(path, {}).get(lineno)
        msg = d["message"]
        if d["severity"] == "information" and msg.startswith("Type of ") and tag:
            out[f"{path.stem}.{tag}"] = msg.split('" is "', 1)[1].rstrip('"')
        elif d["severity"] == "error":
            errors.append(f"{d['file']}:{lineno}: {msg}")
    if errors:
        print("pyright reports errors in the generated calls (a stub overload cannot be called with its own types):")
        print("\n".join(errors))
        sys.exit(1)
    return out


def current() -> dict[str, str]:
    files = sorted(REVEALS.glob("*.py"))
    m, p = mypy_reveals(files), pyright_reveals(files)
    keys = sorted(set(m) | set(p))
    return {k: f"mypy: {m.get(k, '<none>')} | pyright: {p.get(k, '<none>')}" for k in keys}


def versions() -> str:
    """The versions that decide how revealed types are written."""
    import mypy.version
    import numpy
    out = subprocess.run([tool("pyright"), "--version"], capture_output=True, text=True).stdout
    pyright = out.split()[-1] if out.split() else "?"
    return f"mypy {mypy.version.__version__}, pyright {pyright}, numpy {numpy.__version__}"


def load() -> tuple[dict[str, str], str]:
    out, recorded = {}, ""
    if SNAPSHOT.exists():
        for line in SNAPSHOT.read_text().splitlines():
            if line.startswith("# versions: "):
                recorded = line[len("# versions: "):]
            elif line and not line.startswith("#"):
                k, v = line.split(" = ", 1)
                out[k] = v
    return out, recorded


def save(reveals: dict[str, str]) -> None:
    head = ("# What every stub overload gives under mypy and pyright. Written by\n"
            "# tests/typing/check_precision.py --update; checked by scripts/stubs.sh.\n"
            f"# versions: {versions()}\n")
    SNAPSHOT.write_text(head + "".join(f"{k} = {v}\n" for k, v in sorted(reveals.items())))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--update", action="store_true")
    opts = ap.parse_args()
    if not opts.update:
        _old, recorded = load()
        have = versions()
        if recorded and recorded != have:
            print(f"tests/typing/reveals.snapshot was recorded with {recorded}; this environment has {have}.\n"
                  "Revealed types are written differently by other versions. Install the pinned ones:\n"
                  "  pip install --group typing   (or: uv pip install --group typing)")
            return 1
    now = current()
    if opts.update:
        save(now)
        print(f"{SNAPSHOT.relative_to(ROOT)}: {len(now)} reveals")
        return 0
    old, _recorded = load()
    lost = sorted(set(old) - set(now))
    added = sorted(set(now) - set(old))
    changed = sorted(k for k in set(old) & set(now) if old[k] != now[k])
    gained_any = [k for k in changed if now[k].count("Any") > old[k].count("Any")]
    if not (lost or added or changed):
        print(f"precision: {len(now)} reveals unchanged")
        return 0
    if lost:
        print("Precision lost: these overloads, methods or functions no longer exist in the stubs:")
        print("".join(f"  {k}\n      was {old[k]}\n" for k in lost))
    if gained_any:
        print("Precision lost: these results became less precise (more Any):")
        print("".join(f"  {k}\n      was {old[k]}\n      now {now[k]}\n" for k in gained_any))
    other = [k for k in changed if k not in gained_any]
    if other:
        print("Changed results:")
        print("".join(f"  {k}\n      was {old[k]}\n      now {now[k]}\n" for k in other))
    if added:
        print("New overloads, methods or functions:")
        print("".join(f"  {k}\n      now {now[k]}\n" for k in added))
    print("The stubs' types changed. If every change above is intended, run:\n"
          "  python tests/typing/gen_reveals.py && python tests/typing/check_precision.py --update\n"
          "and commit tests/typing/reveals/ and tests/typing/reveals.snapshot with the stub change.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
