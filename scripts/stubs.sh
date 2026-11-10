#!/usr/bin/env bash
# Generate the reference stubs from the Rust bindings and check the hand-written stubs against them.
#
#   scripts/stubs.sh          regenerate stubs/generated/ and tests/typing/reveals/, then check
#   scripts/stubs.sh --check  regenerate into target/stubcheck/ and fail if stubs/generated/ differs
#                             (pre-commit and CI run this, through tests/hooks/check_stubs.sh)
#
# The build uses the `stubs` cargo feature (pyo3's experimental-inspect) and the `stubs` profile, in
# target/ unless CARGO_TARGET_DIR is set. Release wheels are built without the feature.
#
# Checks, all against this one fresh build:
#   stubs/generated/ is what the bindings generate (in --check mode)
#   tests/typing/check_against_rust.py   hand stubs agree with stubs/generated (names, parameters, types)
#   tests/typing/check_precision.py      every overload still gives what it gave (mypy and pyright)
#   mypy.stubtest                        hand stubs against the runtime, including Python-only modules
#   mypy, pyright                        the stubs and tests/typing/usage.py type-check
#   pytest                               reveal_type matches the dtype each call really returns
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$PWD
MODE=write
case "${1:-}" in
    --check) MODE=check ;;
    "") ;;
    *) echo "usage: scripts/stubs.sh [--check]" >&2; exit 2 ;;
esac

PY="${PYTHON:-python}"
if [ -x .venv/bin/python ]; then PY=$ROOT/.venv/bin/python; fi
BIN="$(dirname "$PY")"
MATURIN="$BIN/maturin"; [ -x "$MATURIN" ] || MATURIN=maturin
PYRIGHT="$BIN/pyright"; [ -x "$PYRIGHT" ] || PYRIGHT=pyright
TARGET="${CARGO_TARGET_DIR:-$ROOT/target}"
WORK=$ROOT/target/stubcheck
rm -rf "$WORK/generated" "$WORK/pkg"
mkdir -p "$WORK"

step() { printf '\n== %s\n' "$*"; }

step "build with pyo3 introspection and generate stubs"
"$MATURIN" generate-stubs --quiet --features stubs --profile stubs --target-dir "$TARGET" \
    -i "$PY" --out "$WORK/generated"
# maturin names the extension audio_samples.audio_samples (the package's own __init__.py loads it
# as audio_samples) and writes its stubs under that path.
GEN="$WORK/generated"
mv "$GEN/audio_samples/audio_samples" "$WORK/ext" && rm -rf "$GEN" && mkdir -p "$GEN" && mv "$WORK/ext" "$GEN/audio_samples"

if [ "$MODE" = check ]; then
    step "stubs/generated/ matches the bindings"
    if ! diff -ru stubs/generated "$GEN" > "$WORK/generated.diff"; then
        head -200 "$WORK/generated.diff"
        echo
        echo "The Rust bindings changed but stubs/generated/ was not regenerated (full diff: target/stubcheck/generated.diff)."
        echo "Run scripts/stubs.sh, update the hand-written stubs in audio_samples/ until it passes, and commit both."
        exit 1
    fi
else
    rm -rf stubs/generated
    mkdir -p stubs
    cp -r "$GEN" stubs/generated
    "$PY" tests/typing/gen_reveals.py
fi

step "hand-written stubs agree with the bindings"
"$PY" tests/typing/check_against_rust.py

step "no stub overload changed what it gives"
"$PY" tests/typing/gen_reveals.py --check
"$PY" tests/typing/check_precision.py

# The package with this build's extension, for the checks that import it.
SO="$TARGET/maturin/libaudio_samples.so"
[ -f "$SO" ] || { echo "built extension not found: $SO" >&2; exit 1; }
mkdir -p "$WORK/pkg"
tar -C "$ROOT" --exclude='__pycache__' --exclude='*.so' -cf - audio_samples | tar -C "$WORK/pkg" -xf -
SUFFIX=$("$PY" -c "import importlib.machinery as m; print(m.EXTENSION_SUFFIXES[0])")
cp "$SO" "$WORK/pkg/audio_samples/audio_samples$SUFFIX"
export PYTHONPATH="$WORK/pkg${PYTHONPATH:+:$PYTHONPATH}"
# -P: do not put the working directory, with its own audio_samples/ and whatever extension is
# installed there, ahead of this build on sys.path.
"$PY" -P -c 'import audio_samples, sys; sys.exit(not audio_samples.__file__.startswith(sys.argv[1]))' "$WORK/pkg" \
    || { echo "audio_samples is not imported from $WORK/pkg" >&2; exit 1; }

step "mypy.stubtest: stubs against the runtime"
"$PY" -P -m mypy.stubtest audio_samples --concise \
    --mypy-config-file pyproject.toml \
    --allowlist tests/typing/stubtest_allowlist.txt

step "mypy and pyright"
"$PY" -m mypy -p audio_samples
"$PY" -m mypy tests/typing/usage.py
"$PYRIGHT" --pythonpath "$PY"

step "reveal_type against the dtype each call returns"
"$PY" -P -m pytest -q -p no:cacheprovider tests/test_stub_types.py tests/test_librosa_stubs.py

echo
echo "stubs: all checks passed"
