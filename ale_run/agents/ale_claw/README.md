# ALE Claw

ALE Claw is the in-tree reference harness for running a general computer-use
agent on [Agents' Last Exam](https://github.com/rdi-berkeley/agents-last-exam).
It is a Python-native harness inspired by the
[OpenClaw](https://docs.openclaw.ai/) agent architecture and uses the
[Cua Agent SDK](https://cua.ai/docs/cua/reference/agent-sdk) to drive the VM.
ALE Claw is built for ALE's long-horizon benchmark tasks, where the agent may
need to work across software, files, shell commands, browser workflows, and a
remote desktop in a single run.

ALE Claw is also intentionally minimal. One of its main ideas is that a small
harness with the right loop, tools, memory, and context management can still
perform strongly on ALE. The harness stays narrow by design so the model, not a
large product layer, does most of the work.

> **Why minimal?** The companion blog post,
> [*Does the Harness Matter?*](https://agents-last-exam.org/blogs/harness-matters),
> reports the evidence: with GPT-5.5 held fixed, ALE Claw reaches the same accuracy
> band as OpenClaw, Codex, Cursor, and Droid while using far fewer tokens, dollars,
> and minutes.

## What the harness does

ALE Claw runs a single action loop that repeats until the task is done or the
turn budget is exhausted.

On each turn, it:

1. Builds the model context from the transcript and prompt files.
2. Calls the model.
3. Executes requested tools or GUI actions on the VM.
4. Records the results.
5. Compacts old context when the history gets too large.
6. Flushes important information to disk-backed memory before compaction.

Out of the box, the harness supports:

- File tools: `read`, `write`, `edit`
- Shell execution: `exec`
- Web tools: `web_search`, `web_fetch`
- Vision: `analyze_image`
- GUI control: `computer`
- Memory lookup: `memory_search`, `memory_get`
- Delegation: `delegate_general`, `delegate_gui`

> Note: `web_search` is disabled by default because it needs `BRAVE_API_KEY`.

## Running it

Use `harness: ale_claw` in an ALE agent config:

```yaml
harness: ale_claw
model: openrouter/anthropic/claude-sonnet-4.6
config:
  max_turns: 100
  thinking_level: "off"
```

Then run an experiment:

```bash
export OPENROUTER_API_KEY=...
uv run python -m ale_run run experiments/my_experiment.yaml
```

At least one LLM provider key must be present in the environment
(`OPENROUTER_API_KEY` / `ANTHROPIC_API_KEY` / `OPENAI_API_KEY`) — or set
`base_url` + `api_key` in the config to point litellm at a custom
OpenAI-compatible gateway instead (see the config knobs below).

To enable `web_search`, also export `BRAVE_API_KEY` and remove
`web_search` from `disabled_tools`.

## The config knobs that matter first

The full config surface lives in `config.py`, but most users only need these:

- `model`: main model id in LiteLLM format
- `base_url` / `api_key`: point litellm at a custom OpenAI-compatible gateway
- `extra_generation_kwargs`: extra per-request params (gateway quirks)
- `max_turns`: hard cap on the action loop
- `thinking_level`: base reasoning level
- `disabled_tools`: tools to hide from the model
<!-- - `summary_model` / `auxiliary_model` / `gui_model`: helper models -->

Minimal direct usage looks like this:

```python
from ale_run.agents.ale_claw import AleClawConfig, AleClawDeployer

cfg = AleClawConfig(
    model="openrouter/anthropic/claude-sonnet-4.6",
    max_turns=100,
    thinking_level="off",
)
```

## Where to read the code

If you want to understand the harness quickly, read these files in order:

- `deployer.py`: ALE entry point
- `config.py`: runtime knobs
- `harness/agent_loop.py`: main loop
- `harness/prompt.py`: system prompt assembly
- `harness/tools/tools.py`: tool registry
- `harness/context/`: transcript replay and compaction
- `harness/memory/`: durable memory and memory flush
- `harness/subagent/`: delegation

## Directory map

```text
ale_run/agents/ale_claw/
├── config.py
├── deployer.py
├── transcript_to_trajectory.py
└── harness/
    ├── AGENTS.md
    ├── agent_loop.py
    ├── prompt.py
    ├── session.py
    ├── tools/
    ├── context/
    ├── memory/
    ├── subagent/
    ├── model/
    └── adapters/
```

## ALE Claw vs. OpenClaw

ALE Claw reuses the core ideas that make OpenClaw useful for long-horizon agent
work: a tool-driven action loop, context compaction, durable memory, and
subagents. The difference is scope. [OpenClaw](https://docs.openclaw.ai/) is a
broader agent platform; ALE Claw is the benchmark-oriented harness version
adapted to ALE's Python runtime and
[Cua](https://cua.ai/docs/cua/guide/get-started/what-is-cua)-based VM control.

ALE Claw keeps the parts that matter most for ALE evaluation:

- a single long-horizon action loop
- a typed tool surface for files, shell, web, vision, and GUI control
- context compaction for long runs
- durable memory
- subagent delegation

To stay focused on single-task benchmark evaluation, ALE Claw leaves out the
broader interactive-assistant features of the OpenClaw platform, such as
messaging integrations, real-time interaction surfaces, user account and
preference layers, and the larger gateway/server product surface.

That tradeoff is deliberate: ALE Claw is meant to show that a focused, minimal
harness can still perform strongly on ALE.
