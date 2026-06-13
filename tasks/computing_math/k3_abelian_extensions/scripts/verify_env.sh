#!/usr/bin/env bash
# Verify — computing_math/k3_abelian_extensions : GAP runs via software/run_gap.sh
set -uo pipefail
CANON="${CANON:-/media/user/data/agenthle/computing_math/k3_abelian_extensions/h_4_4_4_m_1_8}"
GAP="$CANON/software/run_gap.sh"
fail(){ echo "[verify] FATAL: $*" >&2; exit 1; }
test -x "$GAP" || fail "software/run_gap.sh not executable"
# trivial computation through the wrapper: |S4| = 24, and an abelian-group invariant
OUT="$(printf 'Print(Size(SymmetricGroup(4)), "\\n"); Print(AbelianInvariants(AbelianGroup([4,4,4])), "\\n"); QUIT;\n' | "$GAP" -q 2>/dev/null)" || fail "GAP execution failed"
echo "[verify] GAP output: $(echo "$OUT" | tr '\n' ' ')"
echo "$OUT" | grep -q "24" || fail "GAP did not compute |S4|=24 (got: $OUT)"
echo "[verify] PASS"
