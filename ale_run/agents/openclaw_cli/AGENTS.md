# OpenClaw CLI Runner — Integration Notes

Standalone deployer that drives OpenClaw via `openclaw agent --local`.
Implements `BaseAgentDeployer` directly (no inheritance chain) with three
entry points: `install()`, `launch()`, `parse_artifacts()`.

## Source

| Aspect | Value |
|---|---|
| Source | Fork tarball, installed via `npm install -g`. Falls back to `tarball_url` (GitHub Release) when `tarball_path` is not on disk. |
| CLI subcommand | `openclaw agent --local --agent main -m "$PROMPT" --json --timeout N --thinking high` |
| Output channel | **stderr** (verified empirically; `--json` does not honor stdout) |
| Plugin loading | Process-startup-tied; `~/.openclaw/extensions/cua/dist/index.cjs` |
| CUA bridge | Native OpenClaw plugin (LiteDesktopActionSpace), NOT MCP |

## Deployer Lifecycle

`OpenClawCliDeployer` extends `BaseAgentDeployer` directly — there is no
intermediate `OpenClawDeployer` or `ExternalAgentDeployer` base class.
Everything runs locally inside the sandbox.

| Method | Purpose |
|---|---|
| `install()` | Install CLI from tarball (local path or GitHub Release URL), build CUA plugin from source (local path or git clone from `cua_plugin_repo`), write config files |
| `launch(prompt)` | Spawn `openclaw agent --local`, poll until exit or timeout, copy session trajectory, return `AgentRunResult` |
| `parse_artifacts(work_dir, ...)` | Parse session `transcript.jsonl` and stderr JSON envelope into trajectory steps |

## CLI Invocation

```bash
NO_COLOR=1 openclaw agent --local \
  --agent main \
  --message "$PROMPT" \
  --json \
  --timeout 600 \
  --thinking high
```

The process is spawned with `start_new_session=True` (Linux) to isolate
the process group. On timeout, the deployer sends SIGTERM, waits a grace
period, then SIGKILL.

## --json Envelope Shape

```json
{
  "payloads": [{"text": "...", "mediaUrl": null}],
  "meta": {
    "durationMs": 33958,
    "agentMeta": {
      "sessionId": "<uuid>",
      "provider": "openrouter",
      "model": "openai/gpt-5.4",
      "usage": {"input": 19345, "output": 5, "total": 19350},
      "lastCallUsage": {"input": N, "output": N, "cacheRead": N, "cacheWrite": N}
    },
    "finalAssistantVisibleText": "...",
    "stopReason": "stop",
    "executionTrace": {
      "winnerProvider": "openrouter",
      "winnerModel": "openai/gpt-5.4",
      "runner": "embedded"
    }
  }
}
```

Stderr preamble lines (stripped by `_parse_stderr_json`):
- `[agent/embedded] session file repaired: ...`
- `[agent/embedded] embedded run agent end: ...`
- `[diagnostic] lane task error: ...`
- `[model-fallback/decision] ...`

## Install Details

### 1. CLI tarball

`install()` checks if `openclaw` is already on PATH. If not:
1. Look for a local tarball at `config.tarball_path` (default: `/opt/ale/openclaw-fork.tgz`)
2. If the local path does not exist and `config.tarball_url` is set, download the tarball from the GitHub Release URL via `curl`
3. Run `npm install -g --prefix ~/.local <tarball>`
4. Add `~/.local/bin` to PATH if needed

### 2. CUA plugin build

If `~/.openclaw/extensions/cua/dist/index.cjs` does not exist:
1. Look for plugin source at `config.cua_plugin_path` (default: `/opt/ale/openclaw-cua-plugin`)
2. If the local path does not exist and `config.cua_plugin_repo` is set, sparse-clone the plugin from the repo (branch `config.cua_plugin_branch`, default `agenthle`)
3. Build: `npm install --no-audit --no-fund` then `npm run build` (uses esbuild under the hood)
4. Install the built artifacts (`package.json`, `openclaw.plugin.json`, `dist/index.cjs`) to `~/.openclaw/extensions/cua/`

### 3. Config write

Write four config files — see "Config File Layout" below.

### 4. Pre-warm the plugin runtime-deps mirror

When `openclaw` is installed to a global npm prefix (Windows
`AppData\Roaming\npm`, or `~/.local` on Linux) it cannot write its bundled-plugin
runtime deps in place, so on first plugin-load it **mirrors the bundled plugins'
dist tree** (~60 plugins, thousands of files) into
`~/.openclaw/plugin-runtime-deps/openclaw-<ver>-<hash>/dist/extensions/` under a
filesystem lock. On Windows this recursive copy takes **minutes**. If it runs
inside the timed `agent --local` turn it emits no stdout/stderr until done and
blows the agent's wall budget — the silent Windows runtime hang.

The mirror is built once and reused (existing targets are skipped). `install()`
runs a cheap `openclaw models list` (loads the same provider plugins) to pay this
cost up front during the untimed install, so every subsequent `launch()` is fast.
Best-effort: failure is non-fatal (the agent turn would just pay the cost
itself). See `_prewarm_plugin_runtime_mirror()`.

## Config File Layout

Written by `_write_config()` during `install()`:

### `~/.openclaw/openclaw.json`

Agent model, timeout, plugin allow/deny, tool allow/deny, gateway mode
(`local`), heartbeat, vision model.

### `~/.openclaw/agents/main/agent/auth-profiles.json`

Provider credentials (OpenRouter or OpenAI API key).

### `~/.openclaw/exec-approvals.json`

YOLO mode: `security: "full"`, `ask: "off"`.

### `~/.openclaw/workspace/.openclaw/workspace-state.json`

Marks setup as completed (`setupCompletedAt`) to skip the bootstrap wizard.
**Must live under the agent *workspace* (`~/.openclaw/workspace/`), not
`~/.openclaw/state/`** — OpenClaw's embedded `agent --local` resolves bootstrap
state from the workspace copy. `_complete_workspace_bootstrap()` writes this,
removes `~/.openclaw/workspace/BOOTSTRAP.md`, and seeds `MEMORY.md` +
`memory/{today,yesterday}.md`. The runtime re-materializes the workspace (with a
fresh `BOOTSTRAP.md`) lazily on first run, so `launch()` re-asserts these markers
right before spawning the agent.

If bootstrap stays *pending*, `agent --local` enters the interactive "who am I?"
bootstrap conversation instead of the task. Under `--json` (which buffers all
output until the turn ends) this produces **zero stdout/stderr and never exits**
on a headless VM — the silent Windows runtime hang. (The earlier deployer wrote
`workspace-state.json` to the wrong path, `~/.openclaw/state/`, and deleted the
wrong `BOOTSTRAP.md`, so the skip never took effect.)

## Config Fields

| Field | Meaning |
|---|---|
| `tarball_path` | Path to fork tarball inside the sandbox (default: `/opt/ale/openclaw-fork.tgz`) |
| `tarball_url` | GitHub Release URL for the fork tarball — fallback when `tarball_path` is absent |
| `cua_plugin_path` | Path to CUA plugin source directory (default: `/opt/ale/openclaw-cua-plugin`) |
| `cua_plugin_repo` | Git URL to clone CUA plugin source from when `cua_plugin_path` is missing (default: `https://github.com/cua-verse/openclaw.git`) |
| `cua_plugin_branch` | Branch to clone for CUA plugin source (default: `agenthle`) |
| `model` | OpenRouter model slug (e.g. `openai/gpt-5.4`) |
| `thinking` | Reasoning depth: `off\|low\|medium\|high` |
| `vision_model` | Per-tool model override for image analysis |
| `tools_deny` | Tools removed from the schema |
| `plugins_allow` | Which plugins load at startup |
| `heartbeat_every` | `"never"` skips background loop |

## Tool Classification

### CUA Plugin Tools (14 — native, not MCP)

`screenshot`, `click`, `type`, `key`, `key_down`, `key_up`, `hold_key`,
`mouse_move`, `mouse_down`, `mouse_up`, `drag`, `scroll`, `wait`,
`cursor_position`.

### Denied by Default

| Tool | Reason |
|---|---|
| `web_search` | Requires Brave API key (not provisioned) |
| `web_fetch` | External network access |
| `image_generate` | External service |
| `video_generate` | External service |
| `music_generate` | External service |
| `memory_search` | Cross-session memory (not applicable) |
| `sessions_yield` | Gateway-coupled; raises error under `--local` |

### Gateway-Coupled Tools (unavailable under --local)

Five OpenClaw built-ins reach for the gateway WebSocket at runtime:
`sessions_list`, `sessions_history`, `sessions_spawn`, `sessions_send`,
`cron`. Under `agent --local` they raise `1006 abnormal closure`. The
default config denies them so the agent never sees them in the schema.

## Session Trajectory

Per-session JSONL at `~/.openclaw/agents/main/sessions/<sid>.jsonl`
contains Anthropic-style message events with `content[]` blocks:
- `type: "message"` with `message.role` + `message.content[]`
  (text, thinking, toolCall, tool_result blocks)
- `type: "tool_result"` with `output`, `call_id`, `is_error`

The deployer copies this to `transcript.jsonl` in the work dir after
the agent exits. `parse_artifacts()` then reads the transcript and the
stderr JSON envelope to populate the trajectory.

## Artifacts

`parse_artifacts()` reads:

```
<work_dir>/
├── transcript.jsonl    # session JSONL (copied from ~/.openclaw/agents/main/sessions/)
├── stderr.log          # contains the --json envelope + diagnostic preamble
├── stdout.log          # usually empty on success
└── prompt.txt          # the prompt sent to the agent
```

The method populates the `TrajectoryBuilder` with:
- One step per session event (assistant messages, tool calls, tool results)
- The full `--json` result envelope stored in `trajectory.extra["openclaw_cli"]`
- Usage/token counts from both the envelope and the session JSONL
