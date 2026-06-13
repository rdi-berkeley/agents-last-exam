#!/usr/bin/env bash
# Verify — business_finance/ff5_public_reconstruction
#   1. software/browser launches Google Chrome (headless) and renders a page
#   2. the staged input/runtime_env builds via uv and imports the data-science
#      + scraping stack (pandas, numpy, statsmodels, yfinance, bs4, lxml, requests)
# Runs as uid 1000; needs network (uv fetches locked wheels at solve time).
set -uo pipefail
CANON="${CANON:-/media/user/data/agenthle/business_finance/ff5_public_reconstruction/base}"
SW="$CANON/software"; RTE="$CANON/input/runtime_env"
fail(){ echo "[verify] FATAL: $*" >&2; exit 1; }

# --- browser ---
test -x "$SW/browser" || fail "software/browser wrapper not executable"
VER="$("$SW/browser" --version 2>/dev/null)"; echo "[verify] browser: ${VER:-<none>}"
echo "$VER" | grep -qiE "chrom" || fail "software/browser did not report a Chrome/Chromium version"
# functional headless render (containers need --no-sandbox)
DOM="$("$SW/browser" --headless=new --no-sandbox --disable-gpu --dump-dom 'data:text/html,<h1>ale</h1>' 2>/dev/null)" || true
echo "$DOM" | grep -qi "ale" || fail "headless Chrome did not render a trivial page"
echo "[verify] headless Chrome render OK"

# --- python runtime ---
test -f "$RTE/pyproject.toml" || fail "runtime_env/pyproject.toml missing"
export UV_PROJECT_ENVIRONMENT="$CANON/output/.verify_venv"
export UV_CACHE_DIR="$CANON/output/.uv_cache"
echo "[verify] building frozen runtime_env with uv ..."
uv run --frozen --project "$RTE" python - <<'PYEOF' || fail "uv run / imports failed"
import pandas, numpy, statsmodels, yfinance, bs4, lxml, requests
print("pandas", pandas.__version__, "numpy", numpy.__version__, "statsmodels", statsmodels.__version__)
import statsmodels.api as sm, numpy as np
X = sm.add_constant(np.arange(10.0)); y = 2*np.arange(10.0)+1
r = sm.OLS(y, X).fit(); assert abs(r.params[1]-2) < 1e-6
print("[verify] OLS + scraping stack OK")
PYEOF
echo "[verify] PASS"
