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

The agent runs on the host and uses one `RemoteDesktopSession` for Linux and
Windows VM files, commands, and desktop actions. No VM or GUI MCP bridge is
started by ALE Claw. File paths passed to `analyze_image` refer to the VM;
optional `target: host` file operations are restricted to the configured host
workspace.

## Standalone

Install from the repository root with Python 3.12 or 3.13:

```bash
pip install ./ale_run/agents/ale_claw
ale-claw --model openai/gpt-4.1 "Read the files in the workspace and summarize them."
```

Set the API key for the chosen model. The default target is an existing Linux
CUA service at `http://127.0.0.1:8000`. Inside a container, this address points
to the container. For a different port or a remote desktop, pass the CUA HTTP
endpoint and target OS:

```bash
ale-claw --config settings.toml --sandbox-url http://desktop:5000 --os-type windows "Open the spreadsheet."
```

`settings.toml` accepts these optional connection and output settings and the
fields of `AleClawConfig` under `[agent]`. Command-line flags override the file.

```toml
sandbox_url = "http://127.0.0.1:8000"
os_type = "linux"
output_dir = "runs/example"

[agent]
model = "openai/gpt-4.1"
max_turns = 100
```

Each invocation uses a fresh session. By default the artifacts are saved under
`./ale-claw-runs/<run-id>/` on the machine running the agent. Use a separate
output directory per run when specifying one. Progress goes to stderr. When the agent returns a result, stdout contains
one JSON object with status, duration, error and transcript path. A turn limit or
persistent context overflow sets `error_type` to `step_limit` or `context_limit`.
`status_code` records the HTTP status when supplied by the provider exception.
Failures exit nonzero; startup errors are reported on stderr.

The same endpoint and artifact settings are available in Python:

```python
import asyncio
from ale_claw import AleClawConfig, run

result = asyncio.run(run("Read the workspace.", config=AleClawConfig(model="openai/gpt-4.1")))
print(result.status, result.transcript_path)
```

## Running in ALE

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

CLI and ALE runs both use LiteLLM's credentials for the selected provider.
You can also supply `base_url` and `api_key` for a custom gateway.

`web_search` requires `BRAVE_API_KEY` when invoked.

## The config knobs that matter first

The full config surface lives in `src/ale_claw/config.py`, but most users only need these:

- `model`: main model id in LiteLLM format
- `base_url` / `api_key`: point litellm at a custom OpenAI-compatible gateway
- `extra_generation_kwargs`: extra per-request params (gateway quirks)
- `max_turns`: hard cap on the action loop
- `thinking_level`: base reasoning level
- `disabled_tools`: tools to hide from the model
- `summary_model` / `auxiliary_model` / `gui_model`: optional helper models;
  GUI delegation defaults to the auxiliary model or the main model

The last permitted turn's tools run before the budget is checked. A run that
reaches the limit without the `DONE` signal is marked failed. Explicit gateway
credentials are passed to helper models only when they use the same provider.
Screenshots, tool failures, and delegated sessions are included in the
standard trajectory when their source artifacts are available.
GUI delegation owns the desktop until it finishes; concurrent GUI delegation
and direct desktop actions are rejected during that time. The run cancels
active subagents when it ends. Cancellation stops the client-side RPC wait;
an already-started VM shell process may continue on the guest.

Usage includes main, delegated, and auxiliary model calls. The trajectory
flags missing usage, cost, and cache data where the artifacts expose them;
known costs may include LiteLLM's reported response cost. The final total cost
is `null` if any call's cost is unavailable; known partial costs remain in the
trajectory's usage details.

## Where to read the code

- `src/ale_claw/runner.py`: shared session runner
- `src/ale_claw/config.py`: model and tool settings
- `src/ale_claw/harness/`: agent loop, tools, memory, context and delegation
- `deployer.py`: ALE executor integration
- `transcript_to_trajectory.py`: ALE trajectory export

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
