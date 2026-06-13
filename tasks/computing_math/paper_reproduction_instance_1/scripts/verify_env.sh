#!/usr/bin/env bash
# Verify — computing_math/paper_reproduction_instance_1
# The baked software/.venv runs via software/python, has pip (so the agent can
# install torch at solve), and codebase.zip is unzippable.
set -uo pipefail
CANON="${CANON:-/media/user/data/agenthle/computing_math/paper_reproduction_instance_1/base}"
PY="$CANON/software/python"
fail(){ echo "[verify] FATAL: $*" >&2; exit 1; }
test -x "$PY" || fail "software/python not executable"
"$PY" -c "import sys;print('[verify] venv python', sys.version.split()[0])" || fail "software/.venv python will not run"
"$PY" -m pip --version >/dev/null 2>&1 || fail "software/.venv has no pip (agent could not install torch at solve)"
echo "[verify] venv pip available: $("$PY" -m pip --version 2>&1 | cut -d' ' -f1-2)"
Z="$CANON/input/codebase.zip"; test -f "$Z" && unzip -l "$Z" >/dev/null 2>&1 && echo "[verify] codebase.zip lists OK" || fail "codebase.zip missing/unreadable"
echo "[verify] PASS"
