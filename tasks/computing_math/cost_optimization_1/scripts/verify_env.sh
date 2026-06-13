#!/usr/bin/env bash
# Verify — computing_math/cost_optimization_1 : build the frozen runtime_env (proves all locked wheels install
# on this image) and import the headline packages. Runs as uid 1000; needs net.
set -uo pipefail
CANON="${CANON:-/media/user/data/agenthle/computing_math/cost_optimization_1/base}"
RTE="$CANON/input/runtime_env"
fail(){ echo "[verify] FATAL: $*" >&2; exit 1; }
test -f "$RTE/pyproject.toml" || fail "runtime_env/pyproject.toml missing"
test -f "$RTE/uv.lock"        || fail "runtime_env/uv.lock missing"
export UV_PROJECT_ENVIRONMENT="$CANON/output/.verify_venv"
export UV_CACHE_DIR="$CANON/output/.uv_cache"
echo "[verify] building frozen runtime_env (uv sync --frozen) ..."
uv sync --frozen --project "$RTE" >/dev/null 2>&1 || uv sync --frozen --project "$RTE" || fail "uv sync --frozen failed (missing system lib?)"
uv run --frozen --project "$RTE" python - <<'PYEOF' || fail "imports failed"
import importlib
mods = "pandas".split()
for m in mods:
    importlib.import_module(m); print("  import", m, "OK")
print("[verify] all headline imports OK")
PYEOF
echo "[verify] PASS"
