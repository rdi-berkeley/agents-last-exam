#!/usr/bin/env bash
# Verify — power_10kv: the input/ uv project (python3.12, openpyxl+pandas) builds + imports
set -uo pipefail
CANON="${CANON:-/media/user/data/agenthle/engineering/power_10kv_feeder_reliability_001/base}"
RTE="$CANON/input"
fail(){ echo "[verify] FATAL: $*" >&2; exit 1; }
test -f "$RTE/pyproject.toml" || fail "input/pyproject.toml missing"
export UV_PROJECT_ENVIRONMENT="$CANON/output/.verify_venv"
export UV_CACHE_DIR="$CANON/output/.uv_cache"
if [ -f "$RTE/uv.lock" ]; then S="uv sync --locked --project $RTE"; else S="uv sync --project $RTE"; fi
echo "[verify] $S ..."
$S >/dev/null 2>&1 || $S || fail "uv sync failed"
"$UV_PROJECT_ENVIRONMENT/bin/python" - <<'PY' || fail "imports failed"
import openpyxl, pandas, sys
print("[verify] python", sys.version.split()[0], "openpyxl", openpyxl.__version__, "pandas", pandas.__version__)
PY
echo "[verify] PASS"
