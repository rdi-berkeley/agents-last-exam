#!/usr/bin/env bash
# Verify — legal/agora: software/python (-> /opt/cua-server/.venv) runs stdlib + reads staged docs
set -uo pipefail
CANON="${CANON:-/media/user/data/agenthle/legal/agora_governance_classify_instance_1/base}"
PY="$CANON/software/python"
fail(){ echo "[verify] FATAL: $*" >&2; exit 1; }
test -x "$PY" || fail "software/python not executable"
"$PY" - <<'PY' || fail "stdlib probe failed"
import json,re,sys,argparse,tempfile,dataclasses
print("[verify] python", sys.version.split()[0], "stdlib (json/re/argparse/dataclasses) OK")
PY
DI="$CANON/input/document_index.json"; test -f "$DI" && "$PY" -c "import json;json.load(open('$DI'));print('[verify] parsed document_index.json')" || fail "cannot parse document_index.json"
echo "[verify] PASS"
