# Investigation: ale_claw `final_metrics` under-reports cost & drops the cache split

**Date:** 2026-06-03
**Status:** **Fixed & verified** in worktree `fix/ale-claw-cost-accounting` — branch `fix/ale-claw-cost-accounting`, tests passing, validated against 3 real runs (not yet committed/merged)
**Component:** `ale_run` orchestration / ale_claw artifact normalization (NOT the `harness/` agent code)
**Severity:** Medium — headline cost/token rollups are systematically wrong; correct data exists alongside them.

> **Verification addendum (2026-06-03) — see [§ Verification findings](#verification-findings-2026-06-03) below.**
> The token/cache diagnosis is confirmed. **But the proposed fix was incomplete on two
> points:** (1) `extra.ale_claw.usage.total_cost_usd` is **also wrong** (same dropped turn) —
> so Option A "copy from `extra.ale_claw.usage`" would leave cost ~9% under; (2) the
> reconciliation must happen **inside `finalize()`**, not after `parse_transcripts_into`,
> because of call ordering. Corrected fix recorded below.

---

## TL;DR

For ale_claw runs, the **raw per-call provider data is correct** (OpenRouter/LiteLLM
`api_result.json`): token identities hold, the prompt-cache read/write split is sane,
and the cache **discount is applied** (cold step costs ~10× per prompt-token vs warm
steps). 

But the **normalized headline aggregate** that downstream cost rollups read —
`trajectory.final_metrics` (mirrored into `run.json.usage`) — is wrong in two ways:

1. **Drops the final API turn** (and any helper/compaction/VLM calls) → under-reports
   total cost & tokens by one step every run (~7–8% on the demo).
2. **Reports `cache_read`/`cache_creation` as 0** even on a ~98% cache-served run.

The accurate aggregate already exists in `trajectory.extra["ale_claw"]["usage"]`. Both
symptoms have the **same root cause**: `final_metrics` is summed from *transcript-derived
per-step metrics*, which neither carry cache tokens nor include steps that bypass the
transcript message writer.

---

## Why it matters

Downstream cost/token aggregation reads `final_metrics`, not `extra.ale_claw.usage`.
E.g. `ale_run/agents/terminus_2/deployer.py:528` ("Aggregate token + cost usage,
**preferring `final_metrics`**") and `ale_run/orchestration/lifecycle.py:934` (dumps
`final_metrics`). So every consumer that rolls up spend or cache-hit rate for ale_claw
gets the under-counted numbers.

---

## Evidence (demo smoke, model `openrouter/anthropic/claude-sonnet-4.6`)

Run dir:
`.logs/smoke/smoke_ale_claw_hello/ale_claw_sonnet_or/openrouter-anthropic-claude-sonnet-4-6/demo__hello_win/v0/20260602_235107/`

Ground truth = per-call `origin_log/ale-claw/trajectories/.../turn_00X/00YY_api_result.json`:

| step | prompt | compl | cached | cache_write | cost |
|------|--------|-------|--------|-------------|--------|
| t0 | 7737 | 136 | 0 | 7734 | 0.0310515 |
| t2 | 8062 | 104 | 7734 | 327 | 0.00510945 |
| t3 | 8232 | 80 | 8061 | 170 | 0.0042588 |
| t4 | 8386 | 140 | 8231 | 154 | 0.0051498 |
| t5 | 8693 | 96 | 8385 | 307 | 0.00510975 |
| **sum** | **41110** | **556** | **32411** | **8692** | **0.0506793** |

Raw-layer checks (all PASS): `total == prompt+compl`; `cached ≤ prompt`,
`cache_write ≤ prompt`; `cost == upstream_inference_cost == prompt_cost+compl_cost`
(residual $0.00). Effective $/prompt-token: `3.75e-6 → 4.4e-7 → 3.7e-7 → 3.6e-7 → 4.2e-7`
→ cache discount is applied.

What `final_metrics` / `run.json.usage` report instead:
- `total_cost ≈ 0.04556955` (only **4** of 5 calls — t5 missing) vs raw `0.0506793`.
- `total_input_tokens = 32417` (sum of 4 prompts) vs raw 41110.
- `total_cache_read_tokens = 0`, `total_cache_creation_tokens = 0` (raw: 32411 / 8692).

Reproduces on the other two runs inspected (`20260602_234610`, `20260602_233543`) — same
off-by-one-call under-report each time.

Correct aggregate (already present): `trajectory.extra["ale_claw"]["usage"]` =
`{overall_input_tokens: 41110, output_tokens: 556, cache_read_input_tokens: 32411,
cache_write_input_tokens: 8692, uncached_input_tokens: 7, total_cost_usd: ...}` — internally
consistent (`7 + 32411 + 8692 = 41110`).

---

## Root cause

`final_metrics` is built by `TrajectoryBuilder` summing per-step `StepMetrics`
(`ale_run/base_interface/trajectory.py:298-302`):

```python
m.total_cache_read_tokens     += s.metrics.cache_read_tokens or 0
m.total_cache_creation_tokens += s.metrics.cache_creation_tokens or 0
```

Those per-step `StepMetrics` come from `ale_run/agents/ale_claw/transcript_to_trajectory.py`:

- **`_metrics_from_message_usage()` (line 204)** maps the transcript's per-message usage —
  `{"input", "output", "total", "cost"}` — into `StepMetrics`. It sets **only**
  `input_tokens / output_tokens / cost_usd`; `cache_read_tokens` and
  `cache_creation_tokens` are left `None` because **the transcript never carries them**.
  → `final_metrics` cache totals sum to 0. (This is acknowledged in the module docstring,
  lines 27-31.)

- The per-step metrics exist **only for assistant turns written to the transcript message
  log**. The final "done" assistant turn — and helper/compaction/VLM calls — are not
  emitted as transcript message-usage rows, so they never become Steps and never enter the
  `final_metrics` sum. → one (or more) calls dropped from the total.

- The **correct** totals are computed separately by `_aggregate_usage()` (line 265), which
  sources tokens from `state.json` ("incremented for EVERY yielded step … including
  helper / compaction / VLM calls") and the cache split by walking every per-turn
  `api_result.json`. Its output is parked in `extra["ale_claw"]["usage"]` — **never folded
  back into `final_metrics`.**

So: the accurate accounting is computed and stored; `final_metrics` just isn't populated
from it.

---

## Suggested fix (pick one; option A preferred)

**A. Reconcile `final_metrics` from `_aggregate_usage()` in `parse_artifacts`.** After
`parse_transcripts_into(...)`, overwrite `builder.trajectory.final_metrics`'s
total_input/output/cost and the two cache totals from `_aggregate_usage(work_dir)` (the
same numbers already placed in `extra.ale_claw.usage`). Smallest change, single source of
truth, fixes both symptoms at once. Touch points:
`ale_run/agents/ale_claw/deployer.py:parse_artifacts` (~line 377) +
`ale_run/agents/ale_claw/transcript_to_trajectory.py` (expose the aggregate / a setter).

**B. Enrich per-step `StepMetrics` with cache + capture the final turn.** Have the
translator read each turn's `api_result.json` and set `cache_read_tokens` /
`cache_creation_tokens` on the corresponding Step, and emit a Step for the final
assistant turn. More faithful per-step, but more code and still misses transcript-bypassing
helper calls (which only `state.json` sees) — so A is more complete.

Whatever the choice, keep `extra.ale_claw.usage` as the cross-check.

---

## Verification plan

1. **Unit:** feed a synthetic `work_dir` (transcript missing the last turn + `state.json` /
   `api_result.json` with cache split) to `parse_artifacts`; assert
   `final_metrics.total_cost_usd == sum(api_result costs)` and
   `final_metrics.total_cache_read_tokens == 32411` etc.
2. **Runtime:** a real ale_claw run (the demo is fine — it caches). After the fix, assert
   `final_metrics` == `extra.ale_claw.usage` on tokens/cost/cache, and that the per-call
   `api_result.json` sum matches `final_metrics.total_cost_usd`.
3. Re-run the read-only audit script (below) before/after.

---

## Scope / provenance

- **Pre-existing**, independent of the `audit/ale-claw-readability` refactor branch. The bug
  is in `ale_run` orchestration normalization (`deployer.parse_artifacts`,
  `transcript_to_trajectory.py`, `base_interface/trajectory.py`), **not** in the `harness/`
  agent code that the readability refactor touched. The Tier 1–3 refactors are
  behavior-preserving and do not affect this path.
- Raw provider accounting is **correct** — do not "fix" the api_result data or LiteLLM
  layer; only the normalization that builds `final_metrics` needs changing.

## Verification findings (2026-06-03)

Re-ran the read-only audit against
`.../demo__hello_win/v0/20260602_235107/` and read the live code. Results:

| Metric | `final_metrics` / `run.json.usage` | `extra.ale_claw.usage` | Ground truth (Σ `api_result.result.usage`) |
|--------|-----------------------------------|------------------------|---------------------------------------------|
| input tokens | 32417 ❌ | **41110 ✓** | 41110 |
| output tokens | 460 ❌ | **556 ✓** | 556 |
| cache_read | 0 ❌ | **32411 ✓** | 32411 |
| cache_creation | 0 ❌ | **8692 ✓** | 8692 |
| **cost_usd** | 0.04556955 ❌ | **0.04557 ❌** | **0.0506793** |

**New finding — `extra.ale_claw.usage` cost is also wrong.** The doc's Option A assumed
`extra.ale_claw.usage` is fully correct and proposed copying it into `final_metrics`. But
its `total_cost_usd` is sourced from `_aggregate_message_usage()`
(`transcript_to_trajectory.py:332`), which sums cost over **transcript assistant messages** —
the *same* source that drops the final turn. So both aggregates under-report cost by the
same dropped call (0.04557 vs 0.0506793, ~9% low). The doc's TL;DR line that left
`total_cost_usd: ...` blank was the tell — the cost value there was never pinned.
**Tokens and the cache split in `extra.ale_claw.usage` are correct** (sourced from
`state.json` + per-turn `api_result.json`); only its cost is wrong.

**Correction — `api_result.json` shape.** Usage lives at `result.usage`, **not** top-level
`usage`. The appendix repro below (`json.load(open(p))["usage"]`) therefore reads `{}` and
reports all-zeros — it never actually validated ground truth. The live code is right:
`_aggregate_cache_tokens` reads `data["result"]["usage"]` (`:325`). The corrected repro is
in the appendix.

**Correction — fix location.** `parse_artifacts` runs **before** `builder.finalize()`
(`orchestration/lifecycle.py:458` then `:479`). So `final_metrics` does not exist during
`parse_transcripts_into`, and `finalize()` rebuilds it from the per-step sum afterwards —
clobbering anything set earlier. Reconciliation must be applied **inside `finalize()`**.

### Corrected fix (supersedes Option A)

1. **`transcript_to_trajectory.py`** — sum cost from the per-turn `api_result.json`
   (`result.usage.cost`, the same files `_aggregate_cache_tokens` already walks) and prefer
   it over the transcript-message cost in `_aggregate_usage()`. Fixes
   `extra.ale_claw.usage.total_cost_usd` → 0.0506793.
2. **`base_interface/trajectory.py`** — add a generic, agent-agnostic
   `TrajectoryBuilder.override_final_metrics(**totals)` hook; `finalize()` applies the
   recorded overrides over the per-step sum for whichever keys are provided. (Generic: any
   deployer with richer artifacts than per-step `StepMetrics` can use it.)
3. **`transcript_to_trajectory.py` / `parse_transcripts_into`** — after computing the
   aggregate, call `builder.override_final_metrics(...)` mapping
   `overall_input_tokens / output_tokens / cache_read_input_tokens /
   cache_write_input_tokens / total_cost_usd` → the matching `FinalMetrics` fields. Guarded
   to no-op when there is no authoritative aggregate (keeps per-step sums for degraded runs).

Single source of truth (`_aggregate_usage`), fixes all three symptoms (cost, tokens, cache),
and respects the real call order.

## Appendix — repro (read-only)

```bash
cd <run_dir>/origin_log/ale-claw/trajectories/*/
python3 - <<'PY'
import json, glob
# NOTE: usage lives under result.usage, NOT top-level usage (see verification addendum).
calls=[(json.load(open(p)).get("result") or {}).get("usage") or {}
       for p in sorted(glob.glob("turn_*/[0-9]*_api_result.json"))]
tot=lambda k: sum(c.get(k,0) for c in calls)
ptd=lambda k: sum((c.get("prompt_tokens_details") or {}).get(k,0) or 0 for c in calls)
print("calls:", len(calls), "prompt:", tot("prompt_tokens"), "compl:", tot("completion_tokens"))
print("cached:", ptd("cached_tokens"), "cache_write:", ptd("cache_write_tokens"))
print("cost:", round(sum(c.get("cost",0) for c in calls),6))
PY
# Compare to: python -c "import json;d=json.load(open('<run_dir>/run.json'));print(d['usage'])"
# Verified ground truth (20260602_235107): calls=5 prompt=41110 compl=556
#                                            cached=32411 cache_write=8692 cost=0.0506793
```
