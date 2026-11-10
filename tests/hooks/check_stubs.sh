#!/usr/bin/env bash
# Pre-commit and CI entry point for the type stubs: builds the extension with pyo3's
# introspection data and runs every stub check against that build. See scripts/stubs.sh.
set -euo pipefail
exec "$(dirname "$0")/../../scripts/stubs.sh" --check
