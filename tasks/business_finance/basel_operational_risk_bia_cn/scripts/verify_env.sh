#!/usr/bin/env bash
# ============================================================================
# Environment verification probe
#   task : business_finance/basel_operational_risk_bia_cn
#
# Proves the agent's real workflow is possible post-install:
#   * software/libreoffice wrapper resolves and reports a version
#   * LibreOffice headless can actually convert a staged .xlsx -> .csv
#     (the canonical way to inspect the workbooks without openpyxl)
# Runs as uid 1000.
# ============================================================================
set -euo pipefail

CANON="${CANON:-/media/user/data/agenthle/business_finance/basel_operational_risk_bia_cn/base}"
LO="$CANON/software/libreoffice"

echo "[verify] wrapper: $LO"
test -x "$LO" || { echo "[verify] FATAL: software/libreoffice not executable" >&2; exit 1; }
"$LO" --version | head -1

# Real capability: convert a staged workbook to CSV headless.
WORK="$(mktemp -d)"
export HOME="${HOME:-/home/kasm-user}"
SRC="$CANON/input/operational_risk_events.xlsx"
test -f "$SRC" || { echo "[verify] FATAL: input workbook missing: $SRC" >&2; exit 1; }

echo "[verify] converting $SRC -> csv (headless)"
"$LO" --headless --norestore --convert-to csv --outdir "$WORK" "$SRC" >/dev/null 2>&1 || true
CSV="$WORK/operational_risk_events.csv"
if [ -s "$CSV" ]; then
  echo "[verify] CSV produced: $(wc -l < "$CSV") lines"
else
  echo "[verify] FATAL: headless xlsx->csv conversion produced no output" >&2
  exit 1
fi

echo "[verify] PASS"
