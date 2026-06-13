#!/usr/bin/env bash
# Verify — computing_math/k8s_payment_api_root_cause_analysis (stdlib text analysis)
set -uo pipefail
CANON="${CANON:-/media/user/data/agenthle/computing_math/k8s_payment_api_root_cause_analysis/base}"
fail(){ echo "[verify] FATAL: $*" >&2; exit 1; }
/usr/bin/python - <<'PY' || fail "python stdlib failed"
import json,re,sys
print("[verify] python", sys.version.split()[0], "stdlib OK")
PY
for f in failing_pod.log deployment.yaml cluster_state.txt; do test -f "$CANON/input/$f" && echo "[verify] read input/$f ($(wc -l <"$CANON/input/$f") lines)" || fail "missing input/$f"; done
echo "[verify] PASS"
