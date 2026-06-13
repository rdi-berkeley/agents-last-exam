#!/usr/bin/env bash
# ============================================================================
# Environment verification probe
#   task : business_finance/digital_marketing_ab_test_analysis_1
#
# Proves the staged Python runtime builds and imports on the post-install image:
#   uv builds the frozen input/runtime_env and imports growthbook, pandas,
#   scipy, statsmodels, yaml (the libs an A/B-test analysis solver needs).
# Runs as uid 1000; requires network (uv fetches the locked wheels at solve time).
# ============================================================================
set -uo pipefail

CANON="${CANON:-/media/user/data/agenthle/business_finance/digital_marketing_ab_test_analysis_1/base}"
RTE="$CANON/input/runtime_env"
fail() { echo "[verify] FATAL: $*" >&2; exit 1; }

test -f "$RTE/pyproject.toml" || fail "runtime_env/pyproject.toml missing"
test -f "$RTE/uv.lock"        || fail "runtime_env/uv.lock missing"

export UV_PROJECT_ENVIRONMENT="$CANON/output/.verify_venv"
export UV_CACHE_DIR="$CANON/output/.uv_cache"
echo "[verify] building frozen runtime_env with uv ..."
uv run --frozen --project "$RTE" python - <<'PYEOF' || fail "uv run / imports failed"
import growthbook, pandas, scipy, statsmodels, yaml
print("growthbook", getattr(growthbook, "__version__", "?"))
print("pandas", pandas.__version__, "scipy", scipy.__version__, "statsmodels", statsmodels.__version__)
# core A/B stats the solver uses
from scipy import stats
import statsmodels.stats.proportion as ssp
_ = stats.norm.ppf(0.975)
_ = ssp.proportions_ztest([50, 40], [100, 100])
print("[verify] A/B stats API OK")
PYEOF

echo "[verify] PASS"
