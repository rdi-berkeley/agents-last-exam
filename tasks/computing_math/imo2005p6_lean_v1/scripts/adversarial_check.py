"""Adversarial end-to-end check: compile real cheat attempts against Mathlib and
run each through the full grader (G1-G4), asserting EVERY non-genuine submission
scores 0.0 and (when available) the genuine reference scores 1.0.

Needs the Lean cache + an `lc.sh`-style wrapper (see e2e_check.py). The reference
proof is EVALUATOR-ONLY (not committed to the public repo); supply it via
IMO2005P6_REFERENCE to include the genuine-reference cases. Usage:
    IMO2005P6_REFERENCE=/path/reference_solution.lean \\
    LEAN_LC=/path/lc.sh LEAN_WORK=/path/leanproj python3 scripts/adversarial_check.py
"""
from __future__ import annotations
import asyncio, os, re, subprocess, sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from grade_lean import grade  # noqa: E402

LC = os.environ.get("LEAN_LC", "/local/home/hihgupta/workspace/lean_imo2005p6/lc.sh")
WORK = os.environ.get("LEAN_WORK", "/local/home/hihgupta/workspace/lean_imo2005p6")
PROB = (_HERE.parent / "assets" / "Problem.lean").read_text(encoding="utf-8")

# EVALUATOR-ONLY reference proof (not committed to the public repo).
_REF_PATH = Path(os.environ.get(
    "IMO2005P6_REFERENCE",
    "/local/home/hihgupta/workspace/imo2005p6_eval_only/reference_solution.lean"))
REF = None
if _REF_PATH.is_file():
    REF = _REF_PATH.read_text(encoding="utf-8").replace(
        "\n#print axioms IMO2005P6.imo2005_p6\n", "\n")

def _defs():  # the fixed preamble a submission must keep
    return PROB[:PROB.index("theorem imo2005_p6")]

# Build cheat variants (source strings). Reference-derived cases are included
# only when the evaluator-only reference proof is available.
CHEATS = {}
if REF is not None:
    CHEATS["genuine_reference"] = (REF, True)
CHEATS["sorry_placeholder"] = (PROB, False)
CHEATS["custom_axiom"] = (
    PROB.replace("namespace IMO2005P6",
                 "namespace IMO2005P6\naxiom cheat_ax {C:Type*}[Fintype C][DecidableEq C]"
                 "(solved:C→Fin 6→Prop)[DecidableRel solved]: 2 ≤ numSolvedExactly solved 5")
        .replace("  sorry", "  exact cheat_ax solved"), False)
CHEATS["native_decide"] = (PROB.replace("  sorry", "  native_decide"), False)
CHEATS["sorryAx_no_literal"] = (
    PROB.replace("  sorry", "  exact sorryAx (2 ≤ numSolvedExactly solved 5) true"), False)
CHEATS["weaken_goal_5to0"] = (
    PROB.replace("2 ≤ numSolvedExactly solved 5", "0 ≤ numSolvedExactly solved 5")
        .replace("  sorry", "  exact Nat.zero_le _"), False)
if REF is not None:
    CHEATS["redefine_numSolved_const"] = (
        REF.replace("(univ.filter (fun c => solvedCount solved c = k)).card", "2", 1), False)
CHEATS["sig_in_comment_trivial_thm"] = (
    _defs().replace("theorem imo2005_p6", "/- theorem imo2005_p6 stmt -/")  # dummy
    + "theorem imo2005_p6 : True := trivial\nend IMO2005P6\n", False)

async def _run(cmd, timeout):
    m = re.match(r"env LEAN_PATH=.*? \S*lean\S* (.+)$", cmd)  # env LEAN_PATH=.. <bin> <file>
    if m:
        real = f"{LC} {m.group(1)}"
    else:
        real = cmd
    p = subprocess.run(real, shell=True, capture_output=True, text=True, timeout=timeout, cwd=WORK)
    return (p.stdout + p.stderr, p.returncode)

async def main() -> int:
    all_ok = True
    for name, (src, want_pass) in CHEATS.items():
        sub = f"{WORK}/_adv_{name}.lean"
        Path(sub).write_text(src, encoding="utf-8")
        r = await grade(src, _run, sub, "IGNORED", WORK, lean_bin=f"{WORK}/lc.sh",
                        per_compile_timeout=600)
        ok = (r.passed == want_pass)
        all_ok &= ok
        exp = "1.0" if want_pass else "0.0"
        flag = "ok " if ok else "XX "
        print(f"  {flag}{name:30s} want={exp} got={r.score} gate={r.gate} ({r.reason})")
    print("\nRESULT:", "ALL PASS" if all_ok else "FAILURES ABOVE")
    return 0 if all_ok else 1

if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
