#!/usr/bin/env bash
# Execute the tutorial notebooks headlessly. Each notebook asserts every number
# its text states, so a passing run means the teaching material is still true.
#
# pre-commit passes the changed notebooks; with no arguments all are run.
# Requires the extension to be built into the active environment
# (`maturin develop --release`) and the `test` dependency group (nbmake).
set -euo pipefail
cd "$(dirname "$0")/../.."
PY="${PYTHON:-python}"
if [ -x .venv/bin/python ]; then PY=.venv/bin/python; fi
if [ "$#" -eq 0 ]; then
    set -- docs/source/tutorials/*.ipynb
fi
"$PY" -m pytest -q --nbmake --nbmake-timeout=180 "$@"
