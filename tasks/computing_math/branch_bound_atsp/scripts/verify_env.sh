#!/usr/bin/env bash
# Verify — computing_math/branch_bound_atsp : build the frozen runtime_env (proves all locked wheels install
# on this image) and import the headline packages. Runs as uid 1000; needs net.
set -uo pipefail
CANON="${CANON:-/media/user/data/agenthle/computing_math/branch_bound_atsp/base}"
RTE="$CANON/input/runtime_env"
fail(){ echo "[verify] FATAL: $*" >&2; exit 1; }
test -f "$RTE/pyproject.toml" || fail "runtime_env/pyproject.toml missing"
export UV_PROJECT_ENVIRONMENT="$CANON/output/.verify_venv"
export UV_CACHE_DIR="$CANON/output/.uv_cache"
# Frozen build when a lockfile is staged; otherwise resolve from pyproject.
if [ -f "$RTE/uv.lock" ]; then SYNC="uv sync --frozen --project $RTE"; else SYNC="uv sync --project $RTE"; fi
echo "[verify] building runtime_env ($SYNC) ..."
$SYNC >/dev/null 2>&1 || $SYNC || fail "uv sync failed (missing system lib?)"
"$UV_PROJECT_ENVIRONMENT/bin/python" - <<'PYEOF' || fail "imports failed"
import importlib
mods = "numpy scipy".split()
for m in mods:
    importlib.import_module(m); print("  import", m, "OK")
print("[verify] all headline imports OK")
PYEOF
echo "[verify] PASS"
