#!/usr/bin/env bash
# Verify — business_finance/sec_10k_financial_parsing
# Runs the staged wrapper (uv --locked) and imports pdfplumber/pypdf/pydantic,
# then opens a staged filing PDF to prove the PDF stack actually works.
# Runs as uid 1000; needs network (uv fetches locked wheels at solve time).
set -uo pipefail
CANON="${CANON:-/media/user/data/agenthle/business_finance/sec_10k_financial_parsing/base}"
PY="$CANON/software/python_with_task_deps.sh"
fail(){ echo "[verify] FATAL: $*" >&2; exit 1; }
test -x "$PY" || fail "software/python_with_task_deps.sh not executable"
echo "[verify] building staged runtime via wrapper (uv --locked) ..."
"$PY" - <<'PYEOF' || fail "wrapper run / imports failed"
import pdfplumber, pypdf, pydantic
print("pdfplumber", pdfplumber.__version__, "pypdf", pypdf.__version__, "pydantic", pydantic.__version__)
import glob, os
pdfs = glob.glob(os.path.join(os.environ.get("CANON",""), "input", "filings", "**", "*.pdf"), recursive=True)
if pdfs:
    with pdfplumber.open(pdfs[0]) as pdf:
        n = len(pdf.pages); txt = (pdf.pages[0].extract_text() or "")
    print(f"[verify] opened staged filing {os.path.basename(pdfs[0])}: {n} pages, first-page chars={len(txt)}")
else:
    print("[verify] note: no staged filing PDF found; import-only check")
print("[verify] pdf stack OK")
PYEOF
echo "[verify] PASS"
