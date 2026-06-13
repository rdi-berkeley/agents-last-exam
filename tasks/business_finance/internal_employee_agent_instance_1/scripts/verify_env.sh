#!/usr/bin/env bash
# Verify — business_finance/internal_employee_agent_instance_1 (system python3, stdlib)
set -uo pipefail
CANON="${CANON:-/media/user/data/agenthle/business_finance/internal_employee_agent_instance_1/base}"
fail(){ echo "[verify] FATAL: $*" >&2; exit 1; }
python3 - <<'PYEOF' || fail "python3 stdlib probe failed"
import json, uuid, re, sys
print("python3", sys.version.split()[0]); print("[verify] stdlib (json/uuid/re) OK")
PYEOF
S="$CANON/input/stubs.py"; test -f "$S" && python3 -m py_compile "$S" && echo "[verify] stubs.py compiles" || fail "stubs.py missing/failed compile"
Q="$CANON/input/queries.json"; test -f "$Q" && python3 -c "import json;json.load(open('$Q'));print('[verify] parsed queries.json')" || fail "cannot parse queries.json"
echo "[verify] PASS"
