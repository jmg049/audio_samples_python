#!/usr/bin/env bash
# Compare the type stubs with the built extension, and type-check them.
#
#   mypy.stubtest   names, parameters and defaults of the stubs against the runtime
#   mypy, pyright   the stubs themselves and a typed usage file (tests/typing/usage.py)
#   pytest          reveal_type against the dtype each call really returns
#
# Requires the extension built into the active environment (`maturin develop --release`)
# and the `test` dependency group.
set -euo pipefail
cd "$(dirname "$0")/../.."
PY="${PYTHON:-python}"
if [ -x .venv/bin/python ]; then PY=.venv/bin/python; fi
"$PY" -m mypy.stubtest audio_samples --concise \
    --mypy-config-file pyproject.toml \
    --allowlist tests/typing/stubtest_allowlist.txt
"$PY" -m mypy -p audio_samples
"$PY" -m mypy tests/typing/usage.py
PYRIGHT="$(dirname "$PY")/pyright"
if [ ! -x "$PYRIGHT" ]; then PYRIGHT=pyright; fi
"$PYRIGHT" --pythonpath "$PY"
"$PY" -m pytest -q tests/test_stub_types.py tests/test_librosa_stubs.py
