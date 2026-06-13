#!/usr/bin/env bash
# Verify — business_finance/legal_ma_consistency_audit_01 (stdlib via software/python)
set -uo pipefail
CANON="${CANON:-/media/user/data/agenthle/business_finance/legal_ma_consistency_audit_01/base}"
PY="$CANON/software/python"
fail(){ echo "[verify] FATAL: $*" >&2; exit 1; }
test -x "$PY" || fail "software/python wrapper not executable"
"$PY" - <<'PYEOF' || fail "software/python stdlib probe failed"
import json, re, csv, difflib, collections, sys
print("python", sys.version.split()[0])
print("[verify] stdlib (json/re/csv/difflib/collections) OK")
PYEOF
M="$CANON/input/document_manifest.json"
test -f "$M" && "$PY" -c "import json;json.load(open('$M'));print('[verify] parsed document_manifest.json')" || fail "cannot parse staged document_manifest.json"
echo "[verify] PASS"
