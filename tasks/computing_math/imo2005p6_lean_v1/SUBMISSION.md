# Submission narrative — `computing_math/imo2005p6_lean_v1`

*Reviewer-facing summary for the Agents' Last Exam data-track submission. This
document argues why the task is in-scope, genuinely last-exam-hard, and
verifiable, and records provenance and the one infrastructure dependency.*

---

## 1. One-line description

Given a fixed Lean 4 theorem statement (with a `sorry`), the agent must produce
a **complete, machine-checked Lean 4 / Mathlib proof** that passes an automated
verifier — no `sorry`, no added axioms, no `native_decide`. The theorem is a
formalization of a hard combinatorial extremal result (IMO 2005 Problem 6).

## 2. Why this is a professional workflow, not a puzzle

The task's *deliverable* is a **verification artifact**, and the *work* is
formal-methods engineering — the same activity used in industry and research to
certify that a claim holds with zero trust in informal argument (protocol
correctness, cryptographic proofs, compiler/kernel verification, and the
Lean/Mathlib and Isabelle ecosystems generally). The agent is prompted as a
"formal-methods engineer producing a machine-checked verification artifact,"
reads a specification, works against a real toolchain (Lean 4.27.0 + Mathlib
v4.27), and iterates with the compiler — exactly the professional loop.

This mirrors the precedent already in the benchmark: `computing_math/cp_test_gen_1`
uses a competitive-programming problem as *substrate* for a real engineering task
(writing an adversarial test-case generator), not as a "solve this contest
problem" exercise. Here likewise, the mathematical theorem is substrate; the
graded skill is producing a rigorous, machine-verifiable formalization.

Formal mathematics is also economically and scientifically load-bearing: recent
formalization projects (PFR, Liquid Tensor Experiment), AI systems trained on
Mathlib (AlphaProof), and industrial verification all depend on precisely this
skill of turning a specification into a kernel-checked proof.

## 3. Why it is genuinely last-exam-hard

- **Novelty / no leakage.** IMO 2005 P6 is **not** formalized in Compfiles or the
  Mathlib archive (verified: they carry only 2005 P2/P3/P4/Q3/Q4). There is no
  public Lean proof to copy; the agent must construct one.
- **Hardest genre to mechanize.** Combinatorics is the genre where formalization
  most often fails — autoformalization pipelines and IMO-in-Lean efforts are
  dominated by algebra/number-theory; combinatorial extremal arguments with
  equality-case analysis are notoriously painful in `Finset` cardinality terms.
- **The proof itself is hard.** The argument is a multi-stage extremal /
  equality-case analysis over `Finset` cardinalities that is delicate to carry
  out rigorously; our reference proof is ~800 lines / 30+ lemmas. (The specific
  proof route is deliberately kept out of this public document — see §7.)
- **Cannot be brute-forced.** The theorem is universally quantified over an
  arbitrary finite contestant type, so `decide`/`native_decide` cannot enumerate
  it; only a genuine proof closes the goal, and the verifier rejects the
  non-genuine routes.

### Calibration (second-tier agent, 2 h wall — the ALE default)

Measured against **Sonnet 4.6** (the FAQ's second-tier reference), on AWS
Bedrock, with extended thinking, at the 2 h wall clock (via the Claude Code
harness):

- **Sonnet 4.6: 0 / 1 passing** (score 0.0; run hit the 2 h wall — `timeout`).
- **Opus 4.8 (top-tier):** an attempted run did not yield a usable data point —
  the Bedrock request timed out after 2 turns (infrastructure, not a genuine
  attempt), so we do not report an Opus pass rate.

We report this transparently rather than inflate it: it is **one** valid
second-tier trial, not a large-N pass rate. Two things make it a meaningful
last-exam signal nonetheless:

1. **The failure is at the formalization, not the mathematics.** From the run
   transcript, Sonnet 4.6 reconstructs essentially the *entire correct informal
   proof* in its reasoning — yet in 2 h it never produces a single compiling
   `.lean` file. The wall is translating a correct argument into a kernel-checked
   Mathlib proof.
2. **The intrinsic difficulty markers are strong and agent-independent:** the
   problem is unformalized in Compfiles and the Mathlib archive (no proof to
   copy), combinatorics is the genre where formalization most often fails, the
   reference proof is ~800 lines / 30+ lemmas, and the universally-quantified
   statement cannot be brute-forced.

*Honest caveat:* part of the observed 2 h failure is a behavioral pattern of the
Claude Code CLI on this workload (very long extended-thinking phases that crowd
out the write-compile-iterate loop), not purely task difficulty. A different
harness might reach the compile stage; we would expect it to then fail on the
Mathlib formalization for the reasons above. Additional trials / harnesses can be
run on request.

## 4. Verifiability / genuineness

The reference output (an **evaluator-only** file, delivered out-of-band via the
ALE reference channel and deliberately **not committed to this public repo** —
see §7) is a **real, complete proof**: it compiles against Mathlib v4.27 and
`#print axioms IMO2005P6.imo2005_p6` reports only the standard trusted axioms
`{propext, Classical.choice, Quot.sound}` — no `sorry`, no added axioms, no
`native_decide`.

Grading is a deterministic, layered gate (`scripts/grade_lean.py`), score 1.0
iff **all** pass, else 0.0:

| Gate | Check |
|------|-------|
| G1 | Submission keeps the exact `imo2005_p6` signature + `solvedCount`/`numSolvedExactly` definitions (blocks vacuous/weakened restatements). |
| G2 | Source scan rejects `sorry` / `admit` / `native_decide` / new `axiom`. |
| G3 | `lean solution.lean` compiles with no errors. |
| G4 | `#print axioms` ⊆ `{propext, Classical.choice, Quot.sound}` — rejects `sorryAx`, declared axioms, and `Lean.ofReduceBool`. |

The anti-cheat is a **whitelist** (rejects any axiom outside the trusted three,
including ones we did not anticipate) and fails closed on unparseable output.
An adversarial battery (`scripts/adversarial_check.py`) verifies that the
reference scores 1.0 and that sorry / custom-axiom / native_decide /
weakened-goal / redefined-definition / comment-hidden-signature submissions all
score 0.0 — including the case that evades the source scan but is caught by the
axiom audit. `scripts/selftest.py` covers the grader logic without a Lean
toolchain.

## 5. Provenance and licensing

- The **problem statement** is International Mathematical Olympiad 2005, Problem 6
  — a factual competition problem statement (not copyrightable subject matter).
- The **Lean formalization and the reference proof are our original work**,
  released under the submission's terms (dataset CC BY 4.0).
- The task depends on **Lean 4** (Apache-2.0) and **Mathlib** (Apache-2.0), both
  permissively licensed and compatible.

## 6. Toolchain: self-contained, fetched from upstream (no upload, no image)

This is the first Lean task in the benchmark, but it needs **no special
infrastructure and no data upload**. `start()` fetches its toolchain from
official upstream at setup time (the sandbox has outbound network, as used by
other accepted ALE tasks that `pip install` or download at solve time):

- `elan` installs Lean **v4.27.0** from the official Lean releases;
- `lake exe cache get` downloads the *prebuilt* **Mathlib v4.27** `.olean` cache
  (~7,900 oleans) from Mathlib's official cache CDN in a few minutes — no
  building from source.

Verified end-to-end in the `cpu-free-ubuntu` sandbox: the toolchain installs,
the cache downloads, and the reference proof compiles clean with
`#print axioms = {propext, Classical.choice, Quot.sound}`. The submission is
therefore **code-only** (this repo/PR) — there is no `software/` payload to place
in a data bucket and no custom image to build.

*Optional offline fallback:* `SOFTWARE.md` + `scripts/build_software.sh` document
pre-staging the same cache as task-data `software/` (≈ 8.1 GB), for a fully
hermetic/offline variant should the pinned upstream cache ever be
garbage-collected. Not used by default.

## 7. Conformance to the ALE add-task guide

Verified against `docs/ale-docs-site/pages/add-task.html`:
- Two-file package (`task_card.json` + `main.py`) with `load()`/`start()`/
  `evaluate()`; `evaluate()` returns `list[float]` in `[0,1]` and returns
  `[0.0]` (never raises) on missing output. ✓
- `taskId` matches the folder path (`computing_math/imo2005p6_lean_v1`). ✓
- `vm.snapshot: cpu-free-ubuntu`, `vm.timeout: 7200` — same as accepted tasks
  (e.g. `regex_synth_v1`); paired with `wall_time_s: 7200`. ✓
- Data model: self-contained (`REQUIRES_TASK_DATA = False`); `start()` fetches
  the toolchain from upstream and writes `input/` at setup — no task-data
  staging required. ✓
- **Reference hiding & solver isolation (leakage mitigation).** The bespoke Lean
  reference proof is **evaluator-only**: it is *not committed to this public
  repository* and is delivered out-of-band via the ALE reference channel. Only
  `input/Problem.lean` (statement + `sorry`, no proof, no hints) plus the
  `LEANPATH`/`LEANBIN` pointer files are ever staged into the solver sandbox —
  the reference and grader never are (the grader runs an in-VM compile + axiom
  audit, so it needs no staged reference file). The *informal* IMO 2005 P6
  solution is inherently public (a well-known olympiad problem); the protected
  asset is the machine-checked Lean artifact, which is what a solver would
  otherwise copy. We recommend the solve phase be run **network-restricted** —
  the only setup-time network need is fetching the pinned Lean toolchain +
  prebuilt Mathlib cache from official upstream, and for a fully hermetic variant
  that fetch can instead be pre-staged as evaluator-supplied task-data (see
  `SOFTWARE.md`).
- Submission channel: task idea (description, reference, rubric) via
  **agenthle.org/submit**, plus adding the task to a `selected_tasks/` list and
  opening a PR.

## 8. Files

```
task_card.json               metadata + prompt (formal-verification framing)
main.py                      load/start/evaluate (ALE task driver)
assets/Problem.lean          the staged task: statement + fixed defs + sorry
(reference proof)            EVALUATOR-ONLY — complete, axiom-clean proof;
                             delivered out-of-band, NOT committed to this repo
scripts/grade_lean.py        layered deterministic verifier (G1–G4)
scripts/selftest.py          grader unit battery (no Lean needed)
scripts/adversarial_check.py end-to-end anti-cheat battery (real compiles)
scripts/e2e_check.py         compiles the reference through the grader
scripts/build_software.sh    rebuild the software/ artifact
SOFTWARE.md                  software/ layout + upload instructions
README.md                    task overview
SUBMISSION.md                this document
```
