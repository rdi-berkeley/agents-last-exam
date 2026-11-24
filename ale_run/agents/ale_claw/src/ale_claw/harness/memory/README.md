# Memory

This directory handles ALE Claw's durable task memory: what the agent writes to
disk so useful information survives context compaction and long runs.

## Key ideas

- **Durable memory:** important task facts are stored on disk, not only in the
  live transcript.
- **Memory flush:** before compaction, the harness gives the agent a chance to
  save what matters.
- **Memory retrieval:** the agent can later search or read that saved material
  with `memory_search` and `memory_get`.
- **Policy:** flushes should happen early enough to avoid losing context, but
  not so often that they dominate the run.

## How it fits into the loop

Memory exists because compaction is lossy by design. Once older history is
summarized, details that were never written down may disappear. The memory
flush step reduces that risk by persisting useful observations before the
history is compressed.

## Read these files first

- `memory.py`: memory store and retrieval primitives
- `memory_flush.py`: the flush-turn implementation
- `memory_flush_policy.py`: when and why a flush should run
