# Context

This directory handles how ALE Claw keeps a long-running agent session inside
the model's context window without losing the thread of the task.

## Key ideas

- **Replay:** rebuild the model-visible history from the transcript and session
  state when a run resumes.
- **Token estimation:** estimate prompt size before each model call so the
  harness can react before the context overflows.
- **Compaction:** summarize older history while keeping recent turns and message
  structure usable.
- **Transcript shaping:** keep the transcript well-formed for both the live
  agent loop and later trajectory parsing.

## How it fits into the loop

The main loop keeps adding messages, tool calls, screenshots, and results. When
that history grows too large, the harness first runs a memory flush, then
compacts older context into a smaller summary. The goal is not to preserve
every byte verbatim; it is to preserve enough task state for the agent to keep
working correctly.

## Compaction logic

Compaction in ALE Claw is both **proactive** and **reactive**:

- **Proactive:** before a model call, the harness estimates context size and
  marks the run for compaction if it is getting too large.
- **Reactive:** if the provider still rejects the request as too large, the
  harness compacts and retries.

The actual compaction flow is:

1. **Flush memory first.** Important facts should be written to durable memory
   before any lossy summary step happens.
2. **Preserve recent turns.** The harness keeps the most recent assistant turns
   verbatim instead of summarizing them, so the agent's current working state is
   not disturbed.
3. **Split the rest into budgeted history.** Older messages become
   "pruneable." The harness computes how much of that history can still fit in
   the context window after accounting for instructions, summary overhead, and
   the preserved recent turns.
4. **Keep some history, compact the older half.** The pruneable region is split
   by token share. The older portion becomes `to_compact`; the newer portion is
   kept verbatim if it still fits the budget.
5. **Prune iteratively if needed.** If the kept portion is still too large, the
   harness repeatedly moves its older half into `to_compact` until the kept
   remainder fits.
6. **Repair tool-call structure.** Splitting history can orphan tool
   call/result pairs, so the harness drops invalid results, removes duplicates,
   and inserts synthetic error results when a call no longer has a matching
   result.
7. **Summarize the compacted portion in chunks.** The older history is
   serialized into a text form and summarized chunk by chunk. Each chunk update
   builds on the previous summary so the final result becomes a running
   checkpoint of earlier work.
8. **Fall back safely if summarization is hard.** If full summarization fails,
   the harness retries with oversized messages excluded, then falls back again
   to a static summary marker rather than crashing the run.

After that, the run continues with:

- the compaction summary for older history
- a smaller kept verbatim history
- the most recent preserved turns unchanged

This design tries to protect the agent's near-term working context while still
making long-horizon runs fit inside finite model windows.

## Read these files first

- `context.py`: context-window thresholds and overflow behavior
- `token_estimation.py`: prompt-size estimation logic
- `compaction.py`: summary-and-rebuild flow
- `replay.py`: history restoration for resumed sessions
- `transcript.py`: transcript grouping and shaping utilities
