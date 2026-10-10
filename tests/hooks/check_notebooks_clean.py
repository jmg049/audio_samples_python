#!/usr/bin/env python3
"""Fail if a tutorial notebook is committed with outputs or execution counts.

The notebooks are executed by the test suite and by the docs build, so the
committed copies stay output-free for clean diffs.
"""

import json
import sys


def main(paths: list[str]) -> int:
    bad = []
    for path in paths:
        with open(path, encoding="utf-8") as fh:
            nb = json.load(fh)
        for cell in nb.get("cells", []):
            if cell.get("cell_type") != "code":
                continue
            if cell.get("outputs") or cell.get("execution_count") is not None:
                bad.append(path)
                break
    for path in bad:
        print(f"{path}: clear outputs before committing "
              "(jupyter nbconvert --clear-output --inplace <file>)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
