#!/usr/bin/env bash
# Verify — business_finance/llm_ecosystem_privacy_audit_realdata_1 (stdlib via software/python)
set -uo pipefail
CANON="${CANON:-/media/user/data/agenthle/business_finance/llm_ecosystem_privacy_audit_realdata_1/base}"
PY="$CANON/software/python"
fail(){ echo "[verify] FATAL: $*" >&2; exit 1; }
test -x "$PY" || fail "software/python wrapper not executable"
"$PY" - <<'PYEOF' || fail "software/python stdlib probe failed"
import json, csv, re, collections, sys
print("python", sys.version.split()[0]); print("[verify] stdlib (json/csv/re/collections) OK")
PYEOF
T="$CANON/input/taxonomy.csv"
test -f "$T" && "$PY" -c "import csv;rows=list(csv.reader(open('$T')));print('[verify] read taxonomy.csv rows=',len(rows))" || fail "cannot read taxonomy.csv"
echo "[verify] PASS"
