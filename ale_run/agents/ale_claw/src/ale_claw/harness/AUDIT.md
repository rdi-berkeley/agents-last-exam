# ale-claw harness — readability/refactor audit

<!-- Branch: audit/ale-claw-readability. Scope: harness/ treated as fully ALE-owned. -->
<!-- Behavior-preserving refactor only. No functional changes. -->

## Verdict

There is real, addressable readability debt, but it is **concentrated**, not pervasive.
The harness is clean on the basics (0 TODO/FIXME, only 17 `noqa`, strong docstrings in
most files). The debt clusters in three shapes:

1. A handful of **god-functions** (90–220 LOC each) doing 5+ phases inline.
2. Four **mega-modules** (>1,200 LOC) that aggregate unrelated concerns.
3. **Cross-file duplication** of a few core algorithms/utilities, in two cases with
   *diverging* copies (a real drift hazard).

The repo rule "no single-line single-call-site helpers" was respected — apparent
duplication that is intentional (e.g. `computer_handler.py` delegation overrides) is
left alone.

## Hard precondition

There are **no harness tests in ALE today**. Several high-value refactors touch
behavior-sensitive paths (the run loop, message conversion, compaction, GUI parsing).
Per `CLAUDE.md`/`AGENTS.md`, those require a Level-2 VM test (≥50 steps) before shipping.
**Build a test safety net first** (unit tests around the pure converters + one VM smoke),
or restrict early work to the zero-risk tiers below.

## Cross-cutting themes (ranked by value)

| # | Theme | Where | Value | Risk |
|---|---|---|---|---|
| 1 | **Duplicated tool-pairing repair** — `repair_orphaned_pairs` (canonical) and `repair_tool_use_result_pairing` (context) reimplement the same 3-pass algorithm, with **diverging** `SYNTHETIC_TOOL_RESULT_CONTENT` + `_SKIP_SYNTHESIS_STOP_REASONS` constants | canonical.py:689, context.py:583 | High (~300 LOC parallel, drift risk) | med |
| 2 | **God-functions to decompose into named phases** | `unified_loop._convert_input_to_messages` (218), `agent_loop.run` (211), `context.compact_messages` (168), `canonical.repair_orphaned_pairs` (168), `tools_web._fetch` (122), `transcript.group_step_output` (98), `subagent_gui.run_gui_subagent` (97), `session.convert_to_responses_api_items`/`_unnest_assistant_blocks`, `image_sanitization.sanitize_raw_image_bytes` (90) | High | med (behavior-sensitive) |
| 3 | **Scattered tool utilities + circular-import workaround** — `_is_windows_path`, `_normalize_path`, `_assert_within_workspace`, `_get_required_str`, `_run_async`, `_resolve_int/_timeout` copied/cross-imported across tools_fs/shell/web/fs_backends (fs_backends does a lazy import to dodge a cycle) | tools_*, fs_backends.py:108 | High | med |
| 4 | **Mega-module splits** — `session.py` (1282, 4 jobs: SessionManager / replay pipeline / memory-flush policy / prompt-report) is the clearest; also canonical.py (1314), context.py (1267), agent_loop.py (1273) | — | High | low-med (pure moves, import churn) |
| 5 | **Tool `call()` boilerplate** — 6 fs/shell/web BaseTool classes repeat validate→resolve→`_run_async(_execute)`→broad-except-log→`{success/error}` | tools_fs/shell/web | Med | med |
| 6 | **`actions`/`action` duality normalized 3×** independently | transcript.py:140, session.py:1069, memory_flush.py:210 | Med | med |
| 7 | **Subagent-family dups** — delegate-tool preamble/response (subagent_tools), `_accumulate_usage` (gui vs session), `_TERMINAL_STATUSES` re-listed, `<base>/subagents/<id>/transcript.jsonl` layout in 4 places, `_extract_text` ×3, `memory_write` schema (memory.py vs memory_flush.py), print-vs-logger inconsistency | subagent_*, memory* | Med | low-med |
| 8 | **Prose-as-code** — `prompt._build_delegation` (118 LOC) etc. are literal prompt text; hoist to module constants so logic ≠ prose | prompt.py | Med | low |
| 9 | **Image-url block / `function_call_output` shape** duplicated 5× | agent_loop.py, unified_loop.py | Med | low |

## Suggested phasing

**Tier 0 — test safety net (precondition for Tier 2+).** Unit tests for the pure
converters (`canonical_to_*`, `convert_to_responses_api_items`, repair functions, the
GUI text parser) + one ale_claw VM smoke under `tests/integration/`.

**Tier 1 — zero-risk hygiene (no behavior change, no tests needed).**
- Delete dead code: `context.py:93` `_model_candidates` (grep-confirm first), `session.py:759` unreachable `image` branch, `session.py:743` no-op ternary + dead `None` guard at 712.
- Normalize `typing.Union` → `|` (memory.py, subagent_tools.py); hoist in-method stdlib imports (milestone.py, fs_backends.py, analyze_image.py); name magic numbers (context.py truncation ratios, unified_loop `"call_1"`/`(1024,768)`).
- Trim war-story docstrings/comments to contract/invariant (session.py:884, memory_flush.py:102, build_tools docstring); standardize subagent `print`→logger; replace cryptic `"E —"`/`"F —"` module titles.

**Tier 2 — cross-file dedup (needs Tier 0 net).** Themes #1, #3, #6, #7, #9 — centralize the
repair algorithm + its constants; create `_paths.py`/`_tool_utils.py` (kills the circular
import); one `normalize_computer_actions`; share subagent utilities + `memory_write` schema.

**Tier 3 — god-function decomposition (behavior-sensitive, VM-gated).** Theme #2, plus the
tool `call()` template (#5). One function per PR, each behind its tests.

**Tier 4 — module splits (pure moves, do last).** Split `session.py` first (cleanest
boundaries), then canonical/context/agent_loop. Land after the in-place decompositions so
moves are mechanical.

## Flagged for owner — possible BUGS, NOT readability — ALL RESOLVED

All four items below have been confirmed with the owner and fixed; kept here as a
record. None remain open.

- ✅ **OpenAI-family detection** (`model/model_config.py`, `model/thinking.py`):
  `startswith("o")` matched any `o*` model (`ollama/…`, `openchat`) and mis-routed
  `openrouter/google/*`. Fixed to key off the provider segment
  (`"openai" in …` + narrow `startswith(("o1","o3","o4"))`).
- ✅ **`analyze_image` Windows-path routing**: `_is_remote_path` now accepts both
  separators (`^[A-Za-z]:[\\/]`), consistent with the upstream guard — `C:/foo` is
  routed remotely. (Was already fixed in tree by the time of this pass.)
- ✅ **`memory.py` path-traversal guard**: replaced the string-prefix compare with
  `Path.is_relative_to`, closing the `<base>` vs `<base>-evil` sibling false-negative.
- ✅ **`session.py` `contextTokens`**: Python attribute renamed to snake_case
  `context_tokens` (matches the docstring); the on-disk `state.json` key stays
  `"contextTokens"` for OpenClaw compatibility.

## Note on adapter files

`adapters/trajectory_saver.py` and `adapters/image_retention.py` intentionally copy/extend
SDK callback bodies. If the pinned `cua-agent` is bumped, re-verify these mirror the SDK.
Add an invariant comment to that effect (trajectory_saver.py:42).
