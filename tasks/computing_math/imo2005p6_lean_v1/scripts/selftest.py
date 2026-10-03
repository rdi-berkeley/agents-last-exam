"""Host self-test for the IMO 2005 P6 Lean task grader.

Exercises the pure decision function in ``grade_lean.py`` against a battery of
realistic submissions: every cheat / malformed submission must score 0.0, and
(when the evaluator-only reference proof is available via IMO2005P6_REFERENCE)
the genuine reference must score 1.0. Does NOT require Lean (it feeds synthetic
compile / axiom-print outputs); an end-to-end test that actually compiles the
reference proof against Mathlib lives in ``scripts/e2e_check.py``. The reference
proof is evaluator-only and not committed to the public repo, so reference-based
cases skip when it is absent.

Run:  python3 scripts/selftest.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from grade_lean import grade_from_outputs  # noqa: E402

_ASSETS = _HERE.parent / "assets"
PROB = (_ASSETS / "Problem.lean").read_text(encoding="utf-8")

# The reference proof is EVALUATOR-ONLY (not committed to the public repo). Load
# it from IMO2005P6_REFERENCE if available; reference-based cases skip if absent.
_REF_PATH = Path(os.environ.get(
    "IMO2005P6_REFERENCE",
    "/local/home/hihgupta/workspace/imo2005p6_eval_only/reference_solution.lean",
))
REF = _REF_PATH.read_text(encoding="utf-8") if _REF_PATH.is_file() else None

CLEAN_AXIOMS = (
    "'IMO2005P6.imo2005_p6' depends on axioms: [propext, Classical.choice, Quot.sound]"
)


def _case(name, src, comp_out, comp_exit, ax_out, ax_exit, want_pass):
    r = grade_from_outputs(src, comp_out, comp_exit, ax_out, ax_exit)
    ok = (r.passed == want_pass)
    flag = "ok " if ok else "XX "
    print(f"  {flag}{name}: score={r.score} gate={r.gate} ({r.reason})")
    return ok


def main() -> int:
    all_ok = True

    # --- PROB-based cases (no reference proof needed) — always run. ---

    # sorry submission → 0.0 (G2, and G4 backstop would see sorryAx)
    all_ok &= _case("sorry", PROB, "warning: declaration uses 'sorry'", 0,
                    "'IMO2005P6.imo2005_p6' depends on axioms: [sorryAx]", 0, want_pass=False)

    # custom-axiom cheat → 0.0
    cheat = PROB.replace("namespace IMO2005P6", "namespace IMO2005P6\naxiom cheat : True")
    cheat = cheat.replace("  sorry", "  trivial")
    all_ok &= _case("custom axiom", cheat, "", 0,
                    "'IMO2005P6.imo2005_p6' depends on axioms: [cheat]", 0, want_pass=False)

    # native_decide → 0.0
    nd = PROB.replace("  sorry", "  native_decide")
    all_ok &= _case("native_decide", nd, "", 0,
                    "'IMO2005P6.imo2005_p6' depends on axioms: [Lean.ofReduceBool]", 0, want_pass=False)

    # --- Reference-based cases need the EVALUATOR-ONLY reference proof, which is
    # not committed to the public repo. Skip gracefully if it is absent. ---
    if REF is None:
        print("  -- reference-based cases SKIPPED (set IMO2005P6_REFERENCE to the "
              "evaluator-only reference_solution.lean to run them)")
    else:
        # Genuine reference proof → 1.0
        all_ok &= _case("reference proof", REF, "", 0, CLEAN_AXIOMS, 0, want_pass=True)

        # vacuous restatement (5 -> 4) that "compiles clean" → 0.0 via G1
        vac = REF.replace("2 ≤ numSolvedExactly solved 5", "2 ≤ numSolvedExactly solved 4")
        all_ok &= _case("vacuous restatement", vac, "", 0, CLEAN_AXIOMS, 0, want_pass=False)

        # edited hypothesis (weaken hpair 2*n -> n) → 0.0 via G1
        weak = REF.replace("2 * n < 5 * (univ.filter", "n < 5 * (univ.filter")
        all_ok &= _case("weakened hypothesis", weak, "", 0, CLEAN_AXIOMS, 0, want_pass=False)

        # does not compile → 0.0
        all_ok &= _case("compile error", REF,
                        "solution.lean:10:2: error: unknown identifier 'foo'", 1, "", 0, want_pass=False)

        # G4 backstop: signature+source clean but the axiom set has an extra
        # (e.g. a hole hidden via a tactic that leaves sorryAx) → 0.0
        all_ok &= _case("hidden sorryAx (no literal)", REF, "", 0,
                        "'IMO2005P6.imo2005_p6' depends on axioms: [propext, sorryAx]", 0, want_pass=False)

    print("\nRESULT:", "ALL PASS" if all_ok else "FAILURES ABOVE")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
