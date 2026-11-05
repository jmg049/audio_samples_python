#!/usr/bin/env python3
"""Make the source tree self-contained before `maturin sdist`.

    python tests/ci/bundle_siblings.py

maturin does not package the targets of `[patch.crates-io]`, so an sdist of this
repository cannot find `../audio_samples` and `../audio_samples_io`. This script
copies the tracked sources of those checkouts into `bundled/` and points the
patch table there. It changes the working tree: run it on a throw-away checkout.
Remove it, and the patch table, once the matching releases are on crates.io.
"""

import re
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# Not needed to build, and most of the size.
SKIP = ("tests/fixtures/", "book/", "benchmarks/", "logo.png")


def tracked(repo):
    out = subprocess.run(["git", "ls-files", "-z"], cwd=repo, check=True, capture_output=True, text=True).stdout
    return [f for f in out.split("\0") if f and not f.startswith(SKIP)]


def main():
    manifest = ROOT / "Cargo.toml"
    text = manifest.read_text()
    for name in ("audio_samples", "audio_samples_io"):
        src = ROOT.parent / name
        dst = ROOT / "bundled" / name
        for rel in tracked(src):
            target = dst / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src / rel, target)
        text, n = re.subn(rf'^{name} = \{{ path = "\.\./{name}" \}}$', f'{name} = {{ path = "bundled/{name}" }}', text, flags=re.M)
        if n != 1:
            raise SystemExit(f"no [patch.crates-io] entry for {name}")
    manifest.write_text(text)


if __name__ == "__main__":
    main()
