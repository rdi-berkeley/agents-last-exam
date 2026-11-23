# Subagents

This directory handles delegation: ALE Claw can hand off focused work to helper
agents instead of forcing the main loop to do everything itself.

## Key ideas

- **General subagent:** a helper for analysis, planning, or text-heavy work
  without direct VM control.
- **GUI subagent:** a helper that can do computer-use work through its own GUI
  pathway.
- **Isolation:** subagents run in their own session context, which keeps the
  main loop from getting overloaded.
- **Return path:** completed subagent results are pulled back into the main
  loop and become part of the task state.

## How it fits into the loop

Delegation is useful when a task has a bounded side problem, such as exploring
an option, analyzing a file, or handling a GUI-heavy subtask. The main agent
keeps ownership of the overall task, while subagents do narrower work and hand
back results.

## Read these files first

- `subagent_tools.py`: delegation tools exposed to the main agent
- `subagent_general.py`: general-purpose helper flow
- `subagent_gui.py`: GUI delegation flow
- `subagent_session.py`: subagent session management
- `subagent_registry.py`: tracking and restoring subagent runs
- `subagent_gui_protocol.py`: GUI-subagent message protocol
