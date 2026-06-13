#!/usr/bin/env bash
# Verify — mpc_control_building: EnergyPlus 22.1.0 runs + cvxpy RTENV builds/imports
set -uo pipefail
CANON="${CANON:-/media/user/data/agenthle/engineering/mpc_control_building_v1/base}"
RTE="$CANON/input/runtime_env"
fail(){ echo "[verify] FATAL: $*" >&2; exit 1; }
EP=/opt/energyplus-22.1.0/energyplus
VER="$("$EP" --version 2>/dev/null || true)"; echo "[verify] $VER"
echo "$VER" | grep -q "22.1.0" || fail "EnergyPlus not 22.1.0"
# EnergyPlus can parse the staged IDF (smoke: --help/--version proves the binary+libs run)
test -f "$CANON/input/SFH.idf" && echo "[verify] staged SFH.idf present" || fail "SFH.idf missing"
export UV_PROJECT_ENVIRONMENT="$CANON/output/.verify_venv" UV_CACHE_DIR="$CANON/output/.uv_cache"
S="uv sync --frozen --project $RTE"; [ -f "$RTE/uv.lock" ] || S="uv sync --project $RTE"
echo "[verify] $S ..."; $S >/dev/null 2>&1 || $S || fail "uv sync failed"
"$UV_PROJECT_ENVIRONMENT/bin/python" - <<'PY' || fail "imports failed"
import cvxpy, numpy, pandas, scipy, matplotlib
x=cvxpy.Variable(); p=cvxpy.Problem(cvxpy.Minimize((x-3)**2)); p.solve()
assert abs(x.value-3)<1e-4
print("[verify] cvxpy solve OK; numpy",numpy.__version__,"cvxpy",cvxpy.__version__)
PY
echo "[verify] PASS"
