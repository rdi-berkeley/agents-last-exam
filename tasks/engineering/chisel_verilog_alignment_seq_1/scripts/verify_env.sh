#!/usr/bin/env bash
# Verify — chisel: yosys/firtool/sbt(+Java) all runnable via the software/ wrappers
set -uo pipefail
CANON="${CANON:-/media/user/data/agenthle/engineering/chisel_verilog_alignment_seq_1/base}"
SW="$CANON/software"
fail(){ echo "[verify] FATAL: $*" >&2; exit 1; }
chk(){ local want="$1"; shift; local out; out="$("$@" 2>&1 || true)"; echo "$out" | grep -q "$want"; }
chk "Yosys"   "$SW/yosys" --version   || fail "software/yosys not runnable"
echo "[verify] yosys: $("$SW/yosys" --version 2>/dev/null | head -1)"
chk "1.138.0" "$SW/firtool" --version || fail "software/firtool not 1.138.0"
echo "[verify] firtool: $("$SW/firtool" --version 2>/dev/null | head -1)"
"$SW/jq" --version >/dev/null 2>&1 || fail "software/jq not runnable"
java -version >/dev/null 2>&1 || fail "java missing"
# sbt: prove the launcher + Java work. `sbt --version` (a.k.a. script version) runs
# without a project and bootstraps the sbt launcher (network) — proves Java+sbt.
SBTOUT="$("$SW/sbt" --version 2>&1 || true)"
echo "$SBTOUT" | grep -qiE "sbt|script version|1\.9\.9" || fail "software/sbt did not run (Java/sbt issue): $(echo "$SBTOUT"|head -1)"
echo "[verify] sbt: $(echo "$SBTOUT" | grep -iE 'sbt|version' | head -1)"
echo "[verify] PASS"
