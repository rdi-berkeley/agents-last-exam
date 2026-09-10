import Mathlib
open Finset

/-!
# IMO 2005, Problem 6 — formalization task

In a mathematical competition, 6 problems were posed to the contestants. Every
pair of these problems was solved by *more than* 2/5 of the contestants, and no
contestant solved all 6 problems. Prove that there are at least two contestants
who each solved exactly 5 problems.

## Your task

Replace the `sorry` in `imo2005_p6` below with a complete Lean 4 / Mathlib proof.

## Formalization contract (do NOT change any of the following)

* `C` is the (finite, decidable-equality) type of contestants; `Fintype.card C`
  is the number `n` of contestants.
* `solved c p` means contestant `c` solved problem `p : Fin 6`.
* `numSolvedExactly solved k` is the number of contestants who solved exactly `k`
  of the 6 problems.
* `hpair` encodes "every pair `i ≠ j` was solved by more than 2/5 of the
  contestants": with `m` the number of contestants solving both `i` and `j`,
  the integer form of `m > 2n/5` is `2 * n < 5 * m`.
* `hnall` encodes "no contestant solved all 6 problems".
* The goal `2 ≤ numSolvedExactly solved 5` is "at least two contestants solved
  exactly 5 problems".

You MUST keep the signature of `imo2005_p6` **exactly** as given (same name, same
hypotheses, same conclusion, same auxiliary definitions `solvedCount` /
`numSolvedExactly`). You may add any number of auxiliary lemmas/definitions
before it. Your submission must compile against Mathlib with **no `sorry`, no new
`axiom`s, and no `native_decide`** (the grader rejects all of these).
-/

namespace IMO2005P6

variable {C : Type*} [Fintype C] [DecidableEq C]

/-- Number of problems contestant `c` solved. -/
abbrev solvedCount (solved : C → Fin 6 → Prop) [DecidableRel solved] (c : C) : ℕ :=
  (univ.filter (fun p => solved c p)).card

/-- Number of contestants who solved exactly `k` problems. -/
abbrev numSolvedExactly (solved : C → Fin 6 → Prop) [DecidableRel solved] (k : ℕ) : ℕ :=
  (univ.filter (fun c => solvedCount solved c = k)).card

/-- **IMO 2005, Problem 6.** In a competition with 6 problems where every pair of
problems was solved by more than 2/5 of the contestants and nobody solved all 6,
at least two contestants each solved exactly 5 problems. -/
theorem imo2005_p6
    (solved : C → Fin 6 → Prop) [DecidableRel solved]
    (n : ℕ) (hn : n = Fintype.card C)
    (hpair : ∀ i j : Fin 6, i ≠ j →
      2 * n < 5 * (univ.filter (fun c => solved c i ∧ solved c j)).card)
    (hnall : ∀ c : C, ∃ p : Fin 6, ¬ solved c p) :
    2 ≤ numSolvedExactly solved 5 := by
  sorry

end IMO2005P6
