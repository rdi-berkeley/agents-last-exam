#!/usr/bin/env bash
# Verify — business_finance/sse_northbound_programmatic_trading_01 (system python3, stdlib)
set -uo pipefail
CANON="${CANON:-/media/user/data/agenthle/business_finance/sse_northbound_programmatic_trading_01/base}"
fail(){ echo "[verify] FATAL: $*" >&2; exit 1; }
python3 - <<'PYEOF' || fail "python3 stdlib probe failed"
import json, re, csv, sys
print("python3", sys.version.split()[0]); print("[verify] stdlib OK")
PYEOF
QS="$CANON/input/question_set.json"; test -f "$QS" && python3 -c "import json;json.load(open('$QS'));print('[verify] parsed question_set.json')" || fail "cannot parse question_set.json"
ET="$CANON/input/extracted_text"; test -d "$ET" && [ -n "$(ls -A "$ET" 2>/dev/null)" ] && echo "[verify] extracted_text present ($(ls "$ET"|wc -l) files)" || fail "extracted_text mirror missing"
echo "[verify] PASS"
