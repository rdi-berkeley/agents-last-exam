"""Grader for the IMO 2005 P6 Lean-formalization task.

The agent submits ``solution.lean`` — the ``Problem.lean`` interface with the
`sorry` in ``imo2005_p6`` replaced by a real Lean 4 / Mathlib proof (plus any
auxiliary lemmas it likes). Grading is a layered, deterministic gate; the score
is 1.0 only if the submission is a genuine, complete, axiom-clean proof of the
*exact* required theorem, and 0.0 otherwise.

Layers (all must pass for score 1.0):

  G1. SIGNATURE PIN — the submission must contain the `imo2005_p6` declaration
      with byte-for-byte the required signature (name, binders, hypotheses,
      conclusion) and the two auxiliary definitions `solvedCount` /
      `numSolvedExactly` with their required bodies. This stops an agent from
      "proving" a weaker/vacuous restatement.

  G2. SOURCE SCAN — reject `sorry`, `admit`, `native_decide`, `axiom`, and other
      escape hatches at the source level (defense-in-depth; G4 is the real gate).

  G3. COMPILE — the file must compile against Mathlib with `lean` exiting 0 and
      no `error:` in its output. A `sorry` also emits only a warning, so G3
      alone is not sufficient — hence G4.

  G4. AXIOM AUDIT — append `#print axioms IMO2005P6.imo2005_p6`, recompile, and
      require the reported axiom set to be a SUBSET of the trusted whitelist
      {propext, Classical.choice, Quot.sound}. This rejects `sorryAx` (holes),
      any solver-declared `axiom`, and `Lean.ofReduceBool` (the `native_decide`
      compiler-trust axiom) — i.e. every way to close the goal without a real
      proof.

This module is import-safe on the host (no Lean required to import); the actual
compilation happens inside the sandbox via a caller-supplied ``run`` coroutine
in :func:`grade`, or synchronously in :func:`grade_local` for host self-tests.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

# The only axioms a genuine Mathlib proof is permitted to depend on.
AXIOM_WHITELIST = {"propext", "Classical.choice", "Quot.sound"}

# The fully-qualified name of the theorem the agent must prove.
TARGET_THEOREM = "IMO2005P6.imo2005_p6"

# Source-level escape hatches that HARD-FAIL G2 (case-insensitive word-ish match).
# `native_decide` and `axiom` are also caught by the axiom audit (G4), but we
# reject them early with a clear reason. Comments are stripped before scanning.
_BANNED_SOURCE = [
    r"\bsorry\b",
    r"\badmit\b",
    r"\bnative_decide\b",
    r"^\s*axiom\b",          # a new axiom declaration
    r"\baddDecl\b",          # metaprogramming that injects declarations
    r"\bimplemented_by\b",   # can smuggle native behavior
    r"\bofReduceBool\b",
    r"\blean_ofReduceBool\b",
]
_BANNED_RE = re.compile("|".join(_BANNED_SOURCE), re.IGNORECASE | re.MULTILINE)


# --- G1: the exact declarations the submission MUST contain, whitespace-normalized. ---
# We normalize runs of whitespace to a single space and strip, then require these
# substrings to be present. This pins the theorem's meaning without being brittle
# about indentation / line breaks.
_REQUIRED_SIGNATURE = (
    "theorem imo2005_p6 "
    "(solved : C → Fin 6 → Prop) [DecidableRel solved] "
    "(n : ℕ) (hn : n = Fintype.card C) "
    "(hpair : ∀ i j : Fin 6, i ≠ j → "
    "2 * n < 5 * (univ.filter (fun c => solved c i ∧ solved c j)).card) "
    "(hnall : ∀ c : C, ∃ p : Fin 6, ¬ solved c p) : "
    "2 ≤ numSolvedExactly solved 5"
)
_REQUIRED_SOLVEDCOUNT = (
    "abbrev solvedCount (solved : C → Fin 6 → Prop) [DecidableRel solved] (c : C) : ℕ := "
    "(univ.filter (fun p => solved c p)).card"
)
_REQUIRED_NUMSOLVED = (
    "abbrev numSolvedExactly (solved : C → Fin 6 → Prop) [DecidableRel solved] (k : ℕ) : ℕ := "
    "(univ.filter (fun c => solvedCount solved c = k)).card"
)
# The required namespace + variable line so the pinned names resolve as intended.
_REQUIRED_NAMESPACE = "namespace IMO2005P6"
_REQUIRED_VARIABLE = "variable {C : Type*} [Fintype C] [DecidableEq C]"


def _strip_comments(src: str) -> str:
    """Remove Lean line comments (``--``) and block comments (``/- … -/``)."""
    # Block comments (non-greedy, across lines). Lean block comments nest, but a
    # single non-nested pass is enough for scanning purposes here.
    src = re.sub(r"/-.*?-/", " ", src, flags=re.DOTALL)
    # Line comments.
    src = re.sub(r"--[^\n]*", " ", src)
    return src


def _normalize_ws(src: str) -> str:
    return re.sub(r"\s+", " ", src).strip()


@dataclass
class GradeResult:
    score: float
    passed: bool
    gate: str                      # which gate decided the outcome
    reason: str
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "score": self.score,
            "passed": self.passed,
            "gate": self.gate,
            "reason": self.reason,
            "details": self.details,
        }


def check_signature(src: str) -> tuple[bool, str]:
    """G1 — the submission pins the exact theorem + auxiliary definitions."""
    body = _strip_comments(src)
    norm = _normalize_ws(body)
    if _REQUIRED_NAMESPACE not in body:
        return False, "missing `namespace IMO2005P6`"
    if _normalize_ws(_REQUIRED_VARIABLE) not in norm:
        return False, "missing/edited `variable {C : Type*} [Fintype C] [DecidableEq C]`"
    if _normalize_ws(_REQUIRED_SOLVEDCOUNT) not in norm:
        return False, "missing/edited `solvedCount` definition"
    if _normalize_ws(_REQUIRED_NUMSOLVED) not in norm:
        return False, "missing/edited `numSolvedExactly` definition"
    if _normalize_ws(_REQUIRED_SIGNATURE) not in norm:
        return False, "missing/edited `imo2005_p6` signature"
    return True, "signature pinned OK"


def scan_source(src: str) -> tuple[bool, str]:
    """G2 — reject source-level escape hatches (comments stripped first)."""
    body = _strip_comments(src)
    m = _BANNED_RE.search(body)
    if m:
        return False, f"banned construct in source: {m.group(0)!r}"
    return True, "source scan OK"


def parse_axioms(print_axioms_output: str) -> set[str] | None:
    """Parse the `#print axioms` output into a set of axiom names.

    Recognizes both:
      "'<thm>' depends on axioms: [a, b, c]"
      "'<thm>' does not depend on any axioms"
    Returns None if no recognizable line is found.
    """
    if "does not depend on any axioms" in print_axioms_output:
        return set()
    m = re.search(r"depends on axioms:\s*\[([^\]]*)\]", print_axioms_output)
    if not m:
        return None
    inner = m.group(1).strip()
    if not inner:
        return set()
    return {a.strip() for a in inner.split(",") if a.strip()}


def check_axioms(print_axioms_output: str) -> tuple[bool, str, set[str] | None]:
    """G4 — the theorem depends only on whitelisted axioms."""
    axioms = parse_axioms(print_axioms_output)
    if axioms is None:
        return False, "could not parse `#print axioms` output", None
    extra = axioms - AXIOM_WHITELIST
    if extra:
        return False, f"disallowed axioms: {sorted(extra)}", axioms
    return True, "axiom audit OK", axioms


def compile_ok(compile_output: str, exit_code: int) -> tuple[bool, str]:
    """G3 — lean exits 0 and prints no `error:`."""
    if exit_code != 0:
        return False, f"lean exited with code {exit_code}"
    if re.search(r"^\S*error:", compile_output, re.MULTILINE) or "error:" in compile_output:
        first = next((ln for ln in compile_output.splitlines() if "error:" in ln), "error")
        return False, f"compile error: {first.strip()[:200]}"
    return True, "compile OK"


def grade_from_outputs(
    src: str,
    compile_output: str,
    compile_exit: int,
    axioms_output: str,
    axioms_exit: int,
) -> GradeResult:
    """Pure decision function given all the raw artifacts (host-testable)."""
    ok, reason = check_signature(src)
    if not ok:
        return GradeResult(0.0, False, "G1_signature", reason)

    ok, reason = scan_source(src)
    if not ok:
        return GradeResult(0.0, False, "G2_source_scan", reason)

    ok, reason = compile_ok(compile_output, compile_exit)
    if not ok:
        return GradeResult(0.0, False, "G3_compile", reason,
                           {"compile_output_tail": compile_output[-800:]})

    if axioms_exit != 0:
        return GradeResult(0.0, False, "G4_axioms",
                           f"axiom-print recompile exited {axioms_exit}",
                           {"axioms_output_tail": axioms_output[-800:]})

    ok, reason, axioms = check_axioms(axioms_output)
    if not ok:
        return GradeResult(0.0, False, "G4_axioms", reason,
                           {"axioms": sorted(axioms) if axioms else None,
                            "axioms_output_tail": axioms_output[-800:]})

    return GradeResult(1.0, True, "all", "genuine axiom-clean proof",
                       {"axioms": sorted(axioms)})


async def grade(
    src: str,
    run: Callable,
    solution_path: str,
    lean_path_env: str,
    workdir: str,
    lean_bin: str = "lean",
    per_compile_timeout: int = 600,
) -> GradeResult:
    """Async grader: `run` is a coroutine `run(cmd, timeout) -> (stdout, exit)`
    that executes a shell command inside the sandbox. Performs G1/G2 locally,
    then G3 (compile) and G4 (axiom audit) inside the sandbox. `lean_bin` is the
    lean executable (bare ``lean`` if on PATH, or an absolute path for a
    bind-mounted toolchain).
    """
    ok, reason = check_signature(src)
    if not ok:
        return GradeResult(0.0, False, "G1_signature", reason)
    ok, reason = scan_source(src)
    if not ok:
        return GradeResult(0.0, False, "G2_source_scan", reason)

    env = f"env LEAN_PATH={lean_path_env!r} {lean_bin}"

    # G3: compile the submission as-is.
    out, code = await run(f"{env} {solution_path!r}", per_compile_timeout)
    ok, reason = compile_ok(out, code)
    if not ok:
        return GradeResult(0.0, False, "G3_compile", reason,
                           {"compile_output_tail": out[-800:]})

    # G4: append `#print axioms` and recompile, then audit.
    ax_file = f"{workdir}/_axcheck.lean"
    append = f"\\n#print axioms {TARGET_THEOREM}\\n"
    mk = (
        f"cp {solution_path!r} {ax_file!r} && "
        f"printf '{append}' >> {ax_file!r}"
    )
    _out, _code = await run(mk, 60)
    ax_out, ax_code = await run(f"{env} {ax_file!r}", per_compile_timeout)
    if ax_code != 0:
        return GradeResult(0.0, False, "G4_axioms",
                           f"axiom-print recompile exited {ax_code}",
                           {"axioms_output_tail": ax_out[-800:]})
    ok, reason, axioms = check_axioms(ax_out)
    if not ok:
        return GradeResult(0.0, False, "G4_axioms", reason,
                           {"axioms": sorted(axioms) if axioms else None,
                            "axioms_output_tail": ax_out[-800:]})
    return GradeResult(1.0, True, "all", "genuine axiom-clean proof",
                       {"axioms": sorted(axioms)})
