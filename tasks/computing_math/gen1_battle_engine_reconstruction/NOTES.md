# gen1_battle_engine_reconstruction: maintainer notes

Self-contained: no baked image data, no task-data pull, no
`requiredSystemPackages`, no network. `start()` writes 2.98 MB of JSON inline.
Runs on `cpu-free-ubuntu` under any provider including Docker.

## Layout

```
assets/            build-time only, never staged onto the VM
  oracle_script.zig  the driver; imports pkmn/engine, emits one transcript
  build.zig(.zon)    driver build, with -Dshowdown and -Dstrip options
  build_corpus.py    builds data/, and refuses to ship a bad corpus
data/              generated, committed
  visible/           staged to input/corpus/, 638 scenarios over 48 families
  holdout/           graded set, host side only, 274 scenarios over 47 families
  protocol.json      engine's own `dump protocol`
  layout.json        engine's own `dump layout`
scripts/
  grade.py           host-side scoring, no sandbox dependency
  eval_runner.py     staged to the VM at eval time
```

## Rebuilding

The oracle is not committed as a binary. Rebuild it from the pinned upstream:

```
git clone https://github.com/pkmn/engine && cd engine
git checkout 78dc891c49788e6ec9007d0f02247d2e04a03d29
# Zig 0.16.0; the driver lives beside it and depends on it by relative path
zig build --prefix out -Doptimize=ReleaseFast          # cartridge mode
zig build --prefix out-sd -Doptimize=ReleaseFast -Dshowdown=true
GEN1_ORACLE=out/bin/oracle python3 assets/build_corpus.py data 6 20
```

`build_corpus.py` refuses to ship if:

- a transcript is shared across two **different** families, which would mean two
  mechanics cannot be told apart (measured: zero);
- a scenario ran fewer than two updates, meaning it never got past the switch-in.

Duplicates *within* a family are expected and collapse to one scenario: Focus
Energy and Leech Seed consume no rolls at all, so the tape cannot vary them.

```
uv run pytest tests/tasks/test_gen1_battle_engine_reconstruction.py
uv run ruff check tasks/computing_math/gen1_battle_engine_reconstruction
```

## Four defects the build caught

Each was invisible to inspection and found by a measurement or an assertion.

1. **Scenario families leaked after a faint.** At six Pokemon a side, the move
   script applies to whoever switches in, so a family started exercising the next
   member's moves. 65% of scenarios were affected at cap 12 and no shorter cap
   fixed it (15% leaked even at cap 4). One Pokemon a side fixes it structurally:
   a faint ends the battle. Verified by re-running the localized-bug ladder, which
   went from 20/24 clean families to 22/24, with the two false positives gone.
2. **The roll tape was a hidden puzzle.** The oracle generated its tape from a
   seed, so reproducing a scenario required reverse-engineering an incidental
   splitmix/xorshift with nothing to do with battle rules. Tapes are now published
   as hex and the oracle reads them; verified the file path reproduces the seed
   path bit for bit.
3. **The corpus split was unsolvable.** Splitting by family would have graded 24
   mechanics the agent had never seen an example of, and a move-specific effect
   cannot be inferred from never observing it. Split by tape within each family
   instead; a test asserts no graded family is undemonstrated.
4. **The corpus was 36 MB.** Every tape was 65,536 rolls while a 20-update battle
   uses a sliver. Cut to 1,024 after verifying no exhaustion panic and
   byte-identical transcripts across 144 scenarios, then folded each tape into the
   scenario that uses it, to avoid one staging round trip per tape. The roster
   later grew to 48 moves; the shipped corpus is 2.98 MB.

## Why the per-family diagnostic needs scenario-scoped families

Measured on the earlier, smaller corpus: a build with one localized bug scored
0.965 under this grader, which named the two broken families outright, while the
same build over a corpus of random full-length battles scored 0.542 and could say
nothing about which mechanic was wrong. A failure anywhere in a long battle is
attributed to the whole battle. That is why each scenario here is scoped to a
single move.

## Why the oracle is not shipped to the VM

Staging the 276 KB stripped binary so the agent can probe it freely is the more
natural design and it is hackable: a submission can shell out to it, or embed it,
and score without implementing anything. Deleting it before grading does not help.
The agent therefore gets data, not a callable engine. The cost is that it cannot
test a hypothesis against a fresh input.

## Variants

`load()` returns `base` only. More variants are cheap: a variant is a different
roster or a different tape set plus a rebuild. The generator space is unbounded,
so the held-out set can be reminted if it ever leaks.
