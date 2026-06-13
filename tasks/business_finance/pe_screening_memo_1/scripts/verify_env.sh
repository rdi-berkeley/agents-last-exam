#!/usr/bin/env bash
# Verify — business_finance/pe_screening_memo_1 (baseline python/uv; reads staged docs)
set -uo pipefail
CANON="${CANON:-/media/user/data/agenthle/business_finance/pe_screening_memo_1/zscaler_fy2025}"
fail(){ echo "[verify] FATAL: $*" >&2; exit 1; }
python --version >/dev/null 2>&1 || fail "python missing"; echo "[verify] $(python --version 2>&1)"
uv --version >/dev/null 2>&1 || fail "uv missing"
# read a staged text mirror + memo template (the agent's real inputs)
T="$CANON/input/zscaler_10k_2025.txt"
test -f "$T" && python -c "print('[verify] read 10-K txt chars=', len(open('$T',encoding='utf-8',errors='ignore').read()))" || fail "cannot read staged 10-K txt"
M="$CANON/input/memo_template.md"; test -f "$M" && echo "[verify] memo_template.md present" || fail "memo_template.md missing"
# pdftotext (base) on a staged PDF
P="$CANON/input/zscaler_10k_2025.pdf"
if [ -f "$P" ]; then OUT=$(mktemp); pdftotext "$P" "$OUT" 2>/dev/null && [ -s "$OUT" ] && echo "[verify] pdftotext on 10-K PDF OK ($(wc -l <"$OUT") lines)" || echo "[verify] note: pdftotext produced little (txt mirror available anyway)"; fi
echo "[verify] PASS"
