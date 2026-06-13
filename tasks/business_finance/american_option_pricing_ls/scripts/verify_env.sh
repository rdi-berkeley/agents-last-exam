#!/usr/bin/env bash
# ============================================================================
# Environment verification probe
#   task : business_finance/american_option_pricing_ls
#
# Proves the post-install container can actually run what the task needs:
#   * the staged Python wrapper (software/python.sh) resolves and runs
#   * it imports numpy 1.26.4 and scipy 1.12.0 at the pinned versions
#   * core API the solver uses (scipy.stats.norm, numpy.linalg.cholesky,
#     numpy.polynomial least squares) is importable/callable
#
# Run as the kasm-user (uid 1000), the same identity the agent runs as.
# Requires network (uv fetches the locked wheels + managed CPython 3.10),
# matching the project policy that Python deps are provisioned by the wrapper.
# ============================================================================
set -euo pipefail

CANON="${CANON:-/media/user/data/agenthle/business_finance/american_option_pricing_ls/base}"
PY="$CANON/software/python.sh"

echo "[verify] wrapper: $PY"
test -x "$PY" || { echo "[verify] FATAL: python.sh not executable/missing" >&2; exit 1; }

"$PY" - <<'PYEOF'
import numpy, scipy
print("numpy", numpy.__version__)
print("scipy", scipy.__version__)
assert numpy.__version__ == "1.26.4", f"numpy version {numpy.__version__} != 1.26.4"
assert scipy.__version__ == "1.12.0", f"scipy version {scipy.__version__} != 1.12.0"

# core API exercised by a Longstaff-Schwartz solver
import numpy as np
from scipy.stats import norm
L = np.linalg.cholesky(np.array([[1.0, 0.3], [0.3, 1.0]]))
_ = norm.cdf(0.5), norm.pdf(0.5)
coef = np.polynomial.polynomial.polyfit(np.linspace(0, 1, 50), np.random.default_rng(0).normal(size=50), 3)
assert L.shape == (2, 2) and coef.shape[0] == 4
print("[verify] numpy/scipy API OK")
PYEOF

echo "[verify] PASS"
