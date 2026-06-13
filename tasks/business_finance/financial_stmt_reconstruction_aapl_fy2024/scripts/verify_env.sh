#!/usr/bin/env bash
# ============================================================================
# Environment verification probe
#   task : business_finance/financial_stmt_reconstruction_aapl_fy2024
#
# Proves the agent's real workflow is possible post-install:
#   * software/{python,pdftotext,grep} wrappers resolve and run
#   * pdftotext can actually extract text from the staged 10-K PDF
#   * python can parse JSON (stdlib) — the task's output is JSON
# Runs as uid 1000.
# ============================================================================
set -uo pipefail

CANON="${CANON:-/media/user/data/agenthle/business_finance/financial_stmt_reconstruction_aapl_fy2024/base}"
SW="$CANON/software"
fail() { echo "[verify] FATAL: $*" >&2; exit 1; }

for w in python pdftotext grep; do
  test -x "$SW/$w" || fail "wrapper software/$w not executable"
done

"$SW/python" -c 'import json,sys; json.dumps({"ok":1}); print("python", sys.version.split()[0])' || fail "software/python cannot run"
"$SW/grep" --version >/dev/null || fail "software/grep cannot run"

PDF="$CANON/input/aapl-2024-10k.pdf"
test -f "$PDF" || fail "staged 10-K PDF missing: $PDF"
OUT="$(mktemp)"
"$SW/pdftotext" "$PDF" "$OUT" 2>/dev/null || fail "pdftotext failed on staged PDF"
LINES=$(wc -l < "$OUT")
[ "$LINES" -gt 100 ] || fail "pdftotext extracted suspiciously little text ($LINES lines)"
echo "[verify] pdftotext extracted $LINES lines from the 10-K PDF"

echo "[verify] PASS"
