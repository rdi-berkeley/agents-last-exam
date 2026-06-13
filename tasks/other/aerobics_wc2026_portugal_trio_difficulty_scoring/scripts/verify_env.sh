#!/usr/bin/env bash
set -uo pipefail
CANON="${CANON:-/media/user/data/agenthle/other/aerobics_wc2026_portugal_trio_difficulty_scoring/variant_1}"
fail(){ echo "[verify] FATAL: $*" >&2; exit 1; }
/usr/bin/python - <<'PY' || fail "stdlib failed"
import json,math,csv,sys,itertools,fractions
print("[verify] python", sys.version.split()[0], "stdlib OK")
PY
test -d "$CANON/input" && echo "[verify] input present" || true
echo "[verify] PASS"
