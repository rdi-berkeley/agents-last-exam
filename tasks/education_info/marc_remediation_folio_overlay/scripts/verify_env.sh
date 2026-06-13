#!/usr/bin/env bash
# Verify — marc_remediation (python3.12 stdlib via software/python3.12)
set -uo pipefail
CANON="${CANON:-/media/user/data/agenthle/education_info/marc_remediation_folio_overlay/base}"
PY="$CANON/software/python3.12"
fail(){ echo "[verify] FATAL: $*" >&2; exit 1; }
test -x "$PY" || fail "software/python3.12 not executable"
"$PY" - <<'PY' || fail "stdlib probe failed"
import sys, json, re, csv, xml.etree.ElementTree, unicodedata
assert sys.version_info[:2]==(3,12), f"expected 3.12, got {sys.version.split()[0]}"
print("[verify] python", sys.version.split()[0], "stdlib (json/re/csv/xml/unicodedata) OK")
PY
echo "[verify] PASS"
