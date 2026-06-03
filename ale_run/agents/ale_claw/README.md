# ALE Claw

**ALE Claw** is a computer-use agent for [ALE](https://github.com/rdi-berkeley/agents-last-exam),
built on the [OpenClaw](https://openclaw.ai/) agent architecture and the
[CUA](https://cua.ai) Computer-Use Agent SDK. It drives a test VM
(click, type, read/write files, run shell commands, browse the web) to complete
benchmark tasks, while managing its own conversation context the way a
long-horizon assistant does — canonical message history, tool-result
truncation, automatic compaction, durable memory, and subagent delegation.

It is ALE's first **native** deployer: the agent runs in-process in the ALE
host's Python interpreter (no subprocess, no container, not inside the VM) and
talks to the target machine through the CUA Computer SDK (`env.session.computer`).
Per-turn transcripts, `state.json`, and raw API result dumps are written to a
host tempdir and mirrored back into the run directory, then translated into an
ALE `Trajectory`.

## What's inside

The agent loop is an OpenClaw reproduction adapted for CUA's `ComputerAgent`
lifecycle. The pieces that make it more than a thin tool-calling loop:

- **Canonical context pipeline** — a single typed message format with sanitize
  passes (orphaned tool-pair repair, thinking-block handling, provider-specific
  ordering) before each API call (`canonical/`).
- **Budget-aware compaction** — when the context window fills, older history is
  chunked and summarized in place and the loop continues, no agent rebuild
  (`context.py` + `compaction.py`).
- **Durable memory + pre-compaction flush** — the agent persists task memory and
  a session log to disk; a flush turn runs before compaction so nothing
  important is lost (`memory*.py`).
- **Subagent delegation** — spawn focused workers: an async general subagent
  (its own session + compaction) and a blocking GUI subagent that relays
  vision→action through a second `ComputerAgent` (`subagent/`).
- **Tool suite** — file read/write/edit, shell exec, web search/fetch, image
  analysis, milestone screenshots, and memory tools (`tools/`).
- **Multi-provider via OpenRouter** — a unified Chat-Completions loop registered
  for `openrouter/*` plus image sanitization (resize/transcode) so screenshots
  fit provider limits (`unified_loop.py`, `image_sanitization.py`).

## Running it

ALE Claw runs as an ALE agent (`harness: ale_claw`). Point an agent config at it
and run an experiment:

```yaml
# configs/agents/ale_claw.yaml
harness: ale_claw
model: openrouter/anthropic/claude-sonnet-4.6
config:
  max_turns: 100
  thinking_level: "off"
```

```bash
export OPENROUTER_API_KEY=...
uv run python -m ale_run run experiments/my_experiment.yaml
```

For a programmatic/standalone construction, the deployer and its config are:

```python
from ale_run.agents.ale_claw import AleClawConfig, AleClawDeployer

cfg = AleClawConfig(
    model="openrouter/anthropic/claude-sonnet-4.6",
    max_turns=100,                  # OpenClaw max_steps
    thinking_level="off",           # off | low | medium | high
    disabled_tools=["web_search"],  # default; set to [] + export BRAVE_API_KEY to enable
)
```

The full kwarg surface is documented in `config.py`. Two knobs worth calling out:

- **`summary_model` / `gui_model` / `lightweight_model`** — route compaction,
  GUI subagent, and helper calls through cheaper sibling models to save cost on
  long runs. Default: all use `model`.
- **`thinking_level`** (`off | low | medium | high`) — Claude reasoning depth;
  defaults per-model. Variants exist for flush / compaction / vision / GUI.

API keys are read from the environment: litellm picks up `OPENROUTER_API_KEY`
(or `ANTHROPIC_API_KEY` / `OPENAI_API_KEY`) straight from `os.environ`, so export
the key for your provider before running — the deployer errors early if none is
set. Web search additionally needs `BRAVE_API_KEY` (and `web_search` removed from
`disabled_tools`).

## Layout

```
ale_run/agents/ale_claw/
├── config.py                   — AleClawConfig (standalone dataclass)
├── deployer.py                 — AleClawDeployer (install → launch → parse_artifacts)
├── transcript_to_trajectory.py — on-disk transcripts → ALE Trajectory (ATIF) steps
├── CLAUDE.md                   — dev workflow + code map for this harness
├── README.md                   — this file
└── harness/                    — the OpenClaw agent, in-tree and ALE-owned
    ├── AGENTS.md               — system-prompt context file
    ├── agent_loop.py           — OpenClawComputerAgent (the run loop)
    ├── session.py / replay.py  — session state + cross-run transcript replay
    ├── canonical/              — typed message format + sanitize passes
    ├── tools/                  — fs / shell / web tool implementations
    ├── subagent/               — general + GUI subagent engines
    ├── adapters/               — CUA SDK callback extensions
    └── … (context, compaction, memory, prompt, unified_loop, …)
```

## Provenance

The harness reproduces OpenClaw's agent-side architecture but is **fully
ALE-owned** — no vendored namespace, no submodule, no upstream sync. The
`upstream_version` field in `config.py` records the OpenClaw commit the design
was adapted from, for provenance only. Develop here directly; see `CLAUDE.md`
for the workflow and verification rules, and `harness/AGENTS.md` for the agent's
own system-prompt context.
