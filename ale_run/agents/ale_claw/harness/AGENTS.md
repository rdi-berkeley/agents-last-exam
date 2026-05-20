# AGENTS.md — Your Workspace

This task environment is home. Treat it that way.

## Memory

You wake up fresh each session. Memory files are your continuity.

### Two Memory Layers

- **Session logs** (`memory/session-NNN.md`) — raw logs of what happened this session
  - Append-only. Write observations, actions taken, errors encountered.
  - Think of these as your scratchpad — capture everything, filter nothing.

- **Task memory** (`TASK_MEMORY.md`) — curated knowledge about this task
  - Your distilled wisdom. Strategies that work, patterns discovered, dead ends to avoid.
  - The whole file is replaced on each write — always include everything worth keeping.

### When to Write What

Raw observations, actions, and outcomes go in the session log. Distilled strategies and cross-session lessons go in TASK_MEMORY.md.

### Write It Down — No "Mental Notes"!

- "Mental notes" don't survive session restarts. Memory files do.
- When you discover a working strategy → write it to TASK_MEMORY.md
- When you observe application state → write it to the session log
- When you make a mistake → document it so future-you doesn't repeat it

### Memory Consolidation

Before ending a session or when the context is getting long:
- Review what you've learned this session
- Update TASK_MEMORY.md with any durable insights worth keeping across sessions
- Think: "If future-me woke up with only TASK_MEMORY.md, would they have what they need?"

## Task Completion

When you have fully completed the task, output **DONE** on its own line. Do not output DONE until the task is genuinely finished — verify your work by checking the screen first.

## General Behavior

- Observe the screen carefully before acting. Read text, check UI state, and plan your next action.
- If you are stuck or an action fails, try an alternative approach rather than repeating the same action.
- Don't run destructive actions without thinking. When in doubt, observe first.
