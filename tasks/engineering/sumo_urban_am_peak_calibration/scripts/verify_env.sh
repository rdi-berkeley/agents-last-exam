#!/usr/bin/env bash
# Verify — engineering/sumo_urban_am_peak_calibration
# eclipse-sumo==1.26.0 ships SUMO binaries + python tools under its `sumo`
# package (exposed via sumo.SUMO_HOME); sumolib/traci live in $SUMO_HOME/tools.
set -uo pipefail
CANON="${CANON:-/media/user/data/agenthle/engineering/sumo_urban_am_peak_calibration/base}"
RTE="$CANON/input/runtime_env"
fail(){ echo "[verify] FATAL: $*" >&2; exit 1; }
test -f "$RTE/pyproject.toml" || fail "runtime_env/pyproject.toml missing"
export UV_PROJECT_ENVIRONMENT="$CANON/output/.verify_venv"
export UV_CACHE_DIR="$CANON/output/.uv_cache"
if [ -f "$RTE/uv.lock" ]; then S="uv sync --frozen --project $RTE"; else S="uv sync --project $RTE"; fi
echo "[verify] $S ..."
$S >/dev/null 2>&1 || $S || fail "uv sync failed (missing system lib?)"
PY="$UV_PROJECT_ENVIRONMENT/bin/python"
SUMO_HOME="$("$PY" -c 'import sumo; print(sumo.SUMO_HOME)' 2>/dev/null)" || fail "eclipse-sumo not importable"
test -x "$SUMO_HOME/bin/sumo" || fail "sumo binary missing"
VER="$("$SUMO_HOME/bin/sumo" --version 2>/dev/null | head -1)"; echo "[verify] $VER"
echo "$VER" | grep -q "1.26.0" || fail "sumo binary not 1.26.0 (got: $VER)"
PYTHONPATH="$SUMO_HOME/tools" "$PY" - <<'PY' || fail "sumolib/traci + jsonschema/lxml import failed"
import sumolib, traci, jsonschema, lxml.etree
print("[verify] sumolib/traci + jsonschema + lxml import OK")
PY
echo "[verify] PASS"
