#!/usr/bin/env bash
# Environment verification probe — business_finance/digital_marketing_audience_segmentation_1
# Builds the frozen input/runtime_env and imports pandas, pyarrow, yaml.
# Runs as uid 1000; needs network (uv fetches locked wheels at solve time).
set -uo pipefail
CANON="${CANON:-/media/user/data/agenthle/business_finance/digital_marketing_audience_segmentation_1/base}"
RTE="$CANON/input/runtime_env"
fail() { echo "[verify] FATAL: $*" >&2; exit 1; }
test -f "$RTE/pyproject.toml" || fail "runtime_env/pyproject.toml missing"
test -f "$RTE/uv.lock"        || fail "runtime_env/uv.lock missing"
export UV_PROJECT_ENVIRONMENT="$CANON/output/.verify_venv"
export UV_CACHE_DIR="$CANON/output/.uv_cache"
echo "[verify] building frozen runtime_env with uv ..."
uv run --frozen --project "$RTE" python - <<'PYEOF' || fail "uv run / imports failed"
import pandas, pyarrow, yaml
print("pandas", pandas.__version__, "pyarrow", pyarrow.__version__)
import pandas as pd
df = pd.DataFrame({"a":[1,2,3]}); df.to_parquet("/tmp/_seg_probe.parquet"); _ = pd.read_parquet("/tmp/_seg_probe.parquet")
print("[verify] pandas+pyarrow parquet round-trip OK")
PYEOF
echo "[verify] PASS"
