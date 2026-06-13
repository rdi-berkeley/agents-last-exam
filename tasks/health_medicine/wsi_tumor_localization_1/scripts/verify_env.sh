#!/usr/bin/env bash
set -uo pipefail
CANON="${CANON:-/media/user/data/agenthle/health_medicine/wsi_tumor_localization_1/center_point}"
RTE="$CANON/input/runtime_env"
fail(){ echo "[verify] FATAL: $*" >&2; exit 1; }
test -f "$RTE/pyproject.toml" || fail "runtime_env/pyproject.toml missing"
export UV_PROJECT_ENVIRONMENT="$CANON/output/.verify_venv" UV_CACHE_DIR="$CANON/output/.uv_cache"
S="uv sync --frozen --project $RTE"; [ -f "$RTE/uv.lock" ] || S="uv sync --project $RTE"
echo "[verify] $S ..."; $S >/dev/null 2>&1 || $S || fail "uv sync failed"
"$UV_PROJECT_ENVIRONMENT/bin/python" - <<'PY' || fail "imports failed (openslide runtime lib?)"
import openslide, numpy, PIL, shapely
print("[verify] openslide", openslide.__version__, "+ numpy/PIL/shapely OK")
PY
echo "[verify] PASS"
