# computing_math/imo2005p6_lean_v1

Formalize-and-prove task: the agent produces a complete, machine-checked
Lean 4 / Mathlib proof of a combinatorial extremal theorem (IMO 2005 Problem 6),
and is graded by an axiom audit that certifies the proof is genuine.

## What the agent does

- Receives `input/Problem.lean`: the exact theorem `imo2005_p6` (with fixed
  auxiliary definitions `solvedCount` / `numSolvedExactly`) and a `sorry`.
- Writes `output/solution.lean`: the same file with the `sorry` replaced by a
  real proof (auxiliary lemmas allowed before the theorem).

## The theorem

> In a competition with 6 problems where every pair of problems was solved by
> more than 2/5 of the contestants and nobody solved all 6, at least two
> contestants each solved exactly 5 problems.

Formalized over an arbitrary finite contestant type `C` with a decidable
`solved : C → Fin 6 → Prop`; `hpair` is the integer form of "> 2/5"
(`2*n < 5*|pair-solvers|`), `hnall` is "nobody solved all 6", goal is
`2 ≤ numSolvedExactly solved 5`.

## Grading (deterministic, layered — `scripts/grade_lean.py`)

| Gate | Check |
|------|-------|
| G1 | Submission keeps the **exact** `imo2005_p6` signature + `solvedCount`/`numSolvedExactly` definitions (whitespace-normalized substring pin). Blocks vacuous/weaker restatements. |
| G2 | Source scan rejects `sorry`, `admit`, `native_decide`, new `axiom`, `addDecl`, `implemented_by`, `ofReduceBool`. |
| G3 | `lean solution.lean` exits 0 with no `error:`. |
| G4 | `#print axioms IMO2005P6.imo2005_p6` ⊆ `{propext, Classical.choice, Quot.sound}`. Rejects `sorryAx` (holes), declared axioms, and `Lean.ofReduceBool` (from `native_decide`). |

Score = **1.0** iff all gates pass, else **0.0**.

Because the theorem is universally quantified over an arbitrary `C` and `solved`,
it cannot be closed by `decide` / `native_decide` (there is no finite object to
enumerate), so the only route to a passing score is a genuine proof — and the
axiom audit rejects the non-genuine routes.

## Solver isolation (anti-leakage)

Only `input/Problem.lean` (+ `LEANPATH`/`LEANBIN`) is staged into the solver
sandbox — the reference proof and grader are **evaluator-only** and never enter
it, and the reference is **not committed to this public repo**. The informal
IMO 2005 P6 solution is public (a famous olympiad problem); the protected asset
is the bespoke Lean proof, which is what a solver would otherwise copy. Run the
solve phase network-restricted where possible — the only network need is the
setup-time toolchain/cache fetch from upstream.

## Toolchain (self-contained — no data upload, no image changes)

The task fetches its own toolchain from **official upstream** at setup time
(the sandbox has outbound network, as in other ALE tasks that `pip install` /
download at solve time). `start()` (see `main.py:_setup_toolchain`):

1. installs Lean **v4.27.0** via `elan` from the official Lean releases;
2. creates a pinned Lake project requiring **Mathlib v4.27** and runs
   `lake exe cache get`, which downloads the *prebuilt* Mathlib `.olean` cache
   (~7,900 oleans) from Mathlib's official cache CDN in a few minutes — it does
   **not** build Mathlib from source (that would take hours);
3. discovers the olean search path and writes `input/LEANPATH` + `input/LEANBIN`
   pointer files (and puts `lean` on `PATH`).

There is therefore **no `software/` payload to upload and no custom image** — the
submission is code-only. (`SOFTWARE.md` + `scripts/build_software.sh` document an
optional offline fallback: pre-staging the cache as task-data `software/`, e.g.
if the pinned upstream cache is ever garbage-collected. Not used by default.)

Determinism note: Lean and Mathlib are version-pinned (`LEAN_VERSION`,
`MATHLIB_REV` in `main.py`), so every run resolves the same toolchain; the
dependency on the upstream cache endpoint is the same kind of network dependency
other accepted ALE tasks already have.

## Files

- `assets/Problem.lean` — the staged problem (statement + fixed defs + `sorry`).
- reference proof — **evaluator-only**: a complete, axiom-clean proof (32
  lemmas/definitions; compiles clean, axioms = `{propext, Classical.choice,
  Quot.sound}`). Delivered out-of-band via the ALE reference channel; **not
  committed to this public repo** and never staged in the sandbox.
- `main.py` — ALE task driver (`load` / `start` / `evaluate`).
- `scripts/grade_lean.py` — the layered grader (import-safe on host).
- `scripts/selftest.py` — host battery over synthetic artifacts (no Lean needed).
- `scripts/e2e_check.py` — compiles the reference for real and grades it (needs
  the Lean cache).

## Reproducing the checks

```bash
# Pure grader logic (no Lean needed):
python3 scripts/selftest.py            # → ALL PASS

# End-to-end with a real Mathlib compile (needs the cache + an `lc.sh`-style shim):
LEAN_LC=/path/to/lc.sh LEAN_WORK=/path/to/leanproj python3 scripts/e2e_check.py
# → REFERENCE: passed=True score=1.0 ; RESULT: PASS
```

## Difficulty

Last-exam candidate. Combinatorics is the hardest IMO genre to formalize, and
this problem is **not present in Compfiles or the Mathlib archive** (verified:
they carry 2005 P2/P3/P4/Q3/Q4 only), so there is no public Lean proof to copy.
The hard content is a multi-stage extremal / equality-case analysis over
`Finset` cardinalities that is delicate to mechanize. A local Sonnet-4.6
calibration run (2 h wall) scored 0/1, consistent with the <1% last-exam bar.
