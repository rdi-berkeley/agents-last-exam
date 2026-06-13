#!/usr/bin/env bash
# Verify — computing_math/dit_pipeline_cfg_alignment_fid_256_001 (file-editing task; baseline python)
set -uo pipefail
CANON="${CANON:-/media/user/data/agenthle/computing_math/dit_pipeline_cfg_alignment_fid_256_001/base}"
fail(){ echo "[verify] FATAL: $*" >&2; exit 1; }
/usr/bin/python -c "import sys,ast;print('[verify] python',sys.version.split()[0])" || fail "python missing"
# solve time is file editing only: prove the staged python sources parse
for f in "$CANON"/input/*.py; do [ -f "$f" ] && { /usr/bin/python -c "import ast;ast.parse(open('$f').read())" && echo "[verify] parsed $(basename $f)" || fail "parse failed: $f"; }; done
echo "[verify] PASS"
