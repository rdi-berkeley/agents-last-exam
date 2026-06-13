#!/usr/bin/env bash
# Verify — homework_grading (stdlib via software/python)
set -uo pipefail
CANON="${CANON:-/media/user/data/agenthle/education_info/homework_grading_numerical_pdes_instance_02/base}"
PY="$CANON/software/python"
fail(){ echo "[verify] FATAL: $*" >&2; exit 1; }
test -x "$PY" || fail "software/python not executable"
"$PY" - <<'PY' || fail "stdlib probe failed"
import sys, math, json, decimal, fractions, statistics
print("[verify] python", sys.version.split()[0], "stdlib (math/json/decimal/fractions/statistics) OK")
PY
echo "[verify] PASS"
