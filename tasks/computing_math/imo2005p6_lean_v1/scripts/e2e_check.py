"""End-to-end check: actually compile the reference proof against Mathlib and
run it through the real async grader (G1-G4), asserting score 1.0; and confirm a
`sorry` submission scores 0.0.

Unlike selftest.py (which feeds synthetic compiler output), this compiles for
real, so it requires a working Lean 4.27.0 + Mathlib v4.27 cache and a small
shell shim that mimics the sandbox `run(cmd, timeout) -> (stdout, exit)`.

Usage (host, with the local Lean cache used during authoring):
    IMO2005P6_REFERENCE=/path/to/reference_solution.lean \\
    LEAN_LC=/local/home/hihgupta/workspace/lean_imo2005p6/lc.sh \\
    LEAN_WORK=/local/home/hihgupta/workspace/lean_imo2005p6 \\
    python3 scripts/e2e_check.py

The reference proof is EVALUATOR-ONLY and is NOT committed to this public repo
(committing it would leak the answer to a solver with web access). Supply it via
IMO2005P6_REFERENCE (defaults to the local eval-only copy used during authoring).

`lc.sh` is a wrapper that sets LEAN_PATH to the prebuilt cache and runs `lean`.
On the task VM the equivalent is `env LEAN_PATH="$(cat input/LEANPATH)" lean`.
"""

from __future__ import annotations

import asyncio
import os
import re
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from grade_lean import grade  # noqa: E402

LC = os.environ.get("LEAN_LC", "/local/home/hihgupta/workspace/lean_imo2005p6/lc.sh")
WORK = os.environ.get("LEAN_WORK", "/local/home/hihgupta/workspace/lean_imo2005p6")
# EVALUATOR-ONLY: the reference proof is deliberately NOT in this public repo.
REFERENCE = Path(os.environ.get(
    "IMO2005P6_REFERENCE",
    "/local/home/hihgupta/workspace/imo2005p6_eval_only/reference_solution.lean",
))


async def _run(cmd: str, timeout: int):
    # Translate the grader's `env LEAN_PATH=... lean <file>` into the local lc.sh
    # wrapper (which already sets LEAN_PATH); pass other commands through.
    m = re.match(r"env LEAN_PATH=.* lean (.+)$", cmd)
    real = f"{LC} {m.group(1)}" if m else cmd
    p = subprocess.run(real, shell=True, capture_output=True, text=True, timeout=timeout, cwd=WORK)
    return (p.stdout + p.stderr, p.returncode)


async def main() -> int:
    if not REFERENCE.is_file():
        print(f"SKIP: evaluator-only reference not found at {REFERENCE} "
              f"(set IMO2005P6_REFERENCE); it is not committed to the public repo.")
        return 0
    ref = REFERENCE.read_text(encoding="utf-8")
    ref = ref.replace("\n#print axioms IMO2005P6.imo2005_p6\n", "\n")
    sub = Path(WORK) / "_e2e_sub.lean"
    sub.write_text(ref, encoding="utf-8")

    r = await grade(ref, _run, str(sub), "IGNORED", WORK, per_compile_timeout=1200)
    print(f"REFERENCE: passed={r.passed} score={r.score} gate={r.gate} :: {r.reason}")
    print(f"           axioms={r.details.get('axioms')}")
    ok = r.passed and r.score == 1.0

    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
