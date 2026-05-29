# terminus_2 (harbor) — Implementation Notes

Per-agent notes covering source/install decisions, CLI invocation, trajectory
schema, tool classification, and deployer internals. Read `README.md` first
for what the agent is and how to configure it.

## Source And Fork

- Upstream: [`harbor-framework/harbor`](https://github.com/harbor-framework/harbor)
  — terminus_2 is a Python class invoked by harbor's own `Trial` runner.
- Fork: [`cua-verse/harbor`](https://github.com/cua-verse/harbor) on branch
  `agenthle`. Installed via `uv tool install` — **no submodule** (the source
  is library-shaped, not a checked-out tree).

### Why a fork?

Upstream terminus_2 expects a `BaseEnvironment` (Docker/E2B/Daytona/…) and
takes the instruction as a `run(instruction=…)` kwarg — no CLI surface. The
ALE external runner needs a self-contained binary that runs inside the
sandbox, reads the prompt from a file, and exits when done. The fork adds:

| Path | Role |
|---|---|
| `src/harbor/environments/local_shell.py` | `LocalShellEnvironment` — subprocess/shutil-backed `BaseEnvironment` so terminus_2 drives the host it runs on, not a separate container |
| `src/harbor/agents/terminus_2/_cli.py` | `harbor-terminus2` argparse entry point — reads the prompt file, builds `Terminus2`, runs it, dumps `agent_context.json` |
| `pyproject.toml` (`[project.scripts]`) | `harbor-terminus2 = "harbor.agents.terminus_2._cli:main"` |

terminus_2 itself is untouched, so future upstream syncs only merge `main`
into `agenthle`.

## Operating Systems

**Linux only.** terminus_2's TmuxSession requires tmux + asciinema and a
POSIX environment. `install()` raises `NotImplementedError` if invoked
against a non-Linux sandbox.

## Install

`install()` handles full setup on a clean image (no pre-baked CLI required):

1. `apt-get install -y tmux asciinema git` (best-effort; skipped if already
   present). Hard-fails only if `tmux` is still unavailable afterward.
2. Bootstrap `uv` via `astral.sh/uv/install.sh` if missing.
3. `uv tool install --python 3.12 --reinstall --from
   'git+https://github.com/cua-verse/harbor.git@agenthle' harbor` — lays the
   CLI at `~/.local/bin/harbor-terminus2` inside an isolated uv-tool venv.
4. Verify `harbor-terminus2 --version`.
5. Create `<work_dir>/logs/agent/`.

The default ale-kasm image may bake the CLI; if `~/.local/bin/harbor-terminus2`
already exists the install step is skipped (the `noagents` smoke exercises the
self-install path).

## CLI Invocation

```bash
harbor-terminus2 \
  --prompt-file <work_dir>/prompt.txt \
  --model openrouter/anthropic/claude-sonnet-4.6 \
  --logs-dir <work_dir>/logs \
  --temperature 0.7 \
  --max-turns 100000
  # --api-base <url>      (optional LiteLLM base url override)
  # --no-recording        (when record_terminal_session is false)
```

The process is spawned in its own session (`start_new_session=True`) so the
deployer can kill the whole process group (incl. the tmux server) on timeout.

## Provider Routing

YAML always carries the OpenRouter-native `<vendor>/<model>` id; only
`provider:` flips. The translation lives in `Terminus2Config.litellm_model_id`.

| `provider:` | YAML `model:` example | Env var injected | Model id → LiteLLM |
|---|---|---|---|
| `openrouter` | `anthropic/claude-sonnet-4.6` | `OPENROUTER_API_KEY` | `openrouter/anthropic/claude-sonnet-4.6` |
| `direct` | `anthropic/claude-sonnet-4.6` | `ANTHROPIC_API_KEY` (prefix lookup) | `anthropic/claude-sonnet-4.6` |
| `direct` | `openai/gpt-4o` | `OPENAI_API_KEY` | `openai/gpt-4o` |

`_selected_api_key` picks exactly one env var; it is injected into the child
process env only and **never written to any file** under `work_dir` (which the
framework gathers to the host).

## Tool Classification

terminus_2 ships a single conceptual action; the CUA MCP bridge is
intentionally not wired in.

| Tool | Description | Linux | Windows |
|---|---|:---:|:---:|
| `bash_command` | `tmux send-keys` of `{keystrokes, duration_sec}` to the agent pane | ✓ | — |

**Bridge:** none. The agent runs inside the sandbox, so MCP would be redundant.

## Trajectory Schema

terminus_2 emits an ATIF `trajectory.json` under `<logs-dir>/agent/`. The
deployer's `parse_artifacts` maps it to ALE steps:

| ATIF field | ALE step |
|---|---|
| `step.source == "user"` | `source="user"`, message |
| `step.reasoning_content` (agent) | `source="agent"`, reasoning |
| `step.message` + `step.tool_calls[]` (agent) | `source="agent"`, message + ToolCall(name=function_name, arguments) |
| `step.observation.results[]` | `source="environment"`, Observation(ToolResult(tool_call_id=source_call_id, content)) |
| `step.metrics` | `StepMetrics` (input/output/cache_read tokens, cost) |
| `final_metrics` | preferred source for aggregate usage (in `trajectory.extra.terminus_2.usage`) |

## Pitfalls

- **`task_complete` semantics**: terminus_2 declares success via a
  double-confirmed `task_complete` flag inside its trajectory — there is no
  externally visible "final answer" string. The verifier scores by reading
  sandbox state after the run, like other ALE agents.
- **Smoke score is not an integration signal**: `demo_tool_smoke_test`-style
  tasks score how many distinct tools were exercised. terminus_2 has one tool
  (`bash_command`), so every action types into the same pane. Read the smoke
  as binary integration signal (process launches, prompt arrives, tmux works,
  trajectory parses), not as a tool-coverage score.
- **`max_turns` has no unlimited sentinel**: defaulted to `100_000` so the
  wall-clock `timeout_s` is the real cap.
- **tmux must outlive nothing**: on timeout the deployer kills the process
  group and runs `tmux kill-server` to reap any orphaned panes.
