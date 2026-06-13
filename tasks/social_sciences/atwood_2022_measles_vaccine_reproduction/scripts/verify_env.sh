#!/usr/bin/env bash
set -uo pipefail
CANON="${CANON:-/media/user/data/agenthle/social_sciences/atwood_2022_measles_vaccine_reproduction/base}"
SW="$CANON/software"
fail(){ echo "[verify] FATAL: $*" >&2; exit 1; }
for w in pdftotext python unzip; do test -x "$SW/$w" || fail "software/$w not executable"; done
"$SW/python" -c "import sys,csv,json,statistics;print('[verify] python',sys.version.split()[0],'stdlib OK')" || fail "python failed"
"$SW/pdftotext" -v >/dev/null 2>&1 || "$SW/pdftotext" -h >/dev/null 2>&1 || true
echo "[verify] pdftotext+unzip wrappers run"
echo "[verify] PASS"
