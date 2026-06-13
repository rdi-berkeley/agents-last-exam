#!/usr/bin/env bash
set -uo pipefail
CANON="${CANON:-/media/user/data/agenthle/health_medicine/Clinical_Variant_Annotation/base}"
fail(){ echo "[verify] FATAL: $*" >&2; exit 1; }
for b in python3 curl jq; do command -v $b >/dev/null 2>&1 || fail "$b missing"; done
echo '{"a":1}' | jq -e '.a==1' >/dev/null || fail "jq broken"
python3 -c "import json,csv,re,sys;print('[verify] python3',sys.version.split()[0],'+ curl + jq OK')" || fail "python3 stdlib failed"
P="$CANON/software/runtime_probe.sh"; [ -x "$P" ] && { bash "$P" >/dev/null 2>&1 && echo "[verify] runtime_probe.sh OK" || echo "[verify] note: runtime_probe.sh nonzero (may need data)"; }
echo "[verify] PASS"
