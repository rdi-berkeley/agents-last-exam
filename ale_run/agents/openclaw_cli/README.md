# OpenClaw CLI Agent

Standalone deployer for `openclaw agent --local`, using a fork tarball
and native CUA plugin (not MCP). Implements `BaseAgentDeployer` directly
with `install()` / `launch()` / `parse_artifacts()`.

```text
openclaw agent --local --agent main \
               --message "$PROMPT" --json --timeout N --thinking high
  -> OpenClaw built-in tools (read/write/exec/...)
  -> CUA plugin tools (LiteDesktopActionSpace)
     from ~/.openclaw/extensions/cua/
  -> CUA Computer Server :8000
  -> desktop / filesystem
```

Each task spawns a fresh openclaw process — no gateway, no port
management, no service lifecycle. Plugin registry is preloaded at
process startup.

## Architecture

OpenClaw is installed from a fork tarball (not public npm). The CUA
bridge is the native OpenClaw plugin (not MCP), built from source at
install time via `npm install` + esbuild (`npm run build`).

| | Path |
|---|---|
| Binary | `~/.local/bin/openclaw` (npm global install) |
| Config home | `~/.openclaw/` |
| CUA plugin | `~/.openclaw/extensions/cua/dist/index.cjs` |
| Auth profiles | `~/.openclaw/agents/main/agent/auth-profiles.json` |
| Sessions | `~/.openclaw/agents/main/sessions/<sid>.jsonl` |

## Provider — OpenRouter or Direct

Auth is configured via `auth-profiles.json`. Routing is **explicit**, set
by the `provider` config field (not inferred from which keys are in the
env):

- `provider: openrouter` (default) — openrouter auth profile keyed by
  `OPENROUTER_API_KEY`; the model is prefixed `openrouter/<model>`.
- `provider: direct` — the native provider is chosen by the model's
  vendor: an OpenAI model (`gpt-*` / `openai/...`) uses the `openai`
  provider + `OPENAI_API_KEY`; an Anthropic model (`claude-*` /
  `anthropic/...`) uses the `anthropic` provider + `ANTHROPIC_API_KEY`.
  `OPENROUTER_API_KEY` is dropped from the launch env so it can't override
  the chosen direct provider.

Missing the required key for the chosen provider is a hard error.

## Install

Three-stage install in `install()`:

1. **CLI**: If `openclaw` not on PATH, install from fork tarball at
   `config.tarball_path` (default: `/opt/ale/openclaw-fork.tgz`). If the
   local tarball is missing and `config.tarball_url` is set, download
   from the GitHub Release URL first. Install via `npm install -g`.
2. **CUA plugin**: If `~/.openclaw/extensions/cua/dist/index.cjs` not
   present, build from source at `config.cua_plugin_path`
   (default: `/opt/ale/openclaw-cua-plugin`). If the local source is
   missing and `config.cua_plugin_repo` is set, sparse-clone from the
   repo (branch `config.cua_plugin_branch`). Build via
   `npm install && npm run build` (esbuild). Artifacts installed:
   `package.json`, `openclaw.plugin.json`, `dist/index.cjs`.
3. **Config**: Write `openclaw.json`, `auth-profiles.json`,
   `exec-approvals.json`, `workspace-state.json`.

Both the tarball and plugin source can be baked into the sandbox image,
volume-mounted, or fetched at install time via the GitHub Release / git
clone fallbacks.

## Config

```yaml
agent:
  harness: openclaw_cli
  model: openai/gpt-5.4
  config:
    provider: openrouter   # or "direct" (native openai/anthropic)
    timeout_s: 1800
    thinking: high
    tarball_path: /opt/ale/openclaw-fork.tgz
    tarball_url: https://github.com/cua-verse/openclaw/releases/download/v0.1.0/openclaw-fork.tgz
    cua_plugin_path: /opt/ale/openclaw-cua-plugin
    cua_plugin_repo: https://github.com/cua-verse/openclaw.git
    cua_plugin_branch: agenthle
```

| Field | Meaning |
|---|---|
| `provider` | Routing: `openrouter` (default) or `direct` (native openai/anthropic) |
| `model` | Model slug. OpenRouter: `openai/gpt-5.4`, `anthropic/claude-sonnet-4-6`. Direct: `gpt-5.4` or `claude-sonnet-4-6` |
| `thinking` | Reasoning depth: `off\|low\|medium\|high` |
| `vision_model` | Per-tool model override for image analysis |
| `tools_deny` | Tools removed from the schema |
| `plugins_allow` | Which plugins load at startup |
| `heartbeat_every` | `"never"` skips background loop |
| `tarball_path` | Path to fork tarball inside sandbox |
| `tarball_url` | GitHub Release URL fallback for the fork tarball |
| `cua_plugin_path` | Path to CUA plugin source inside sandbox |
| `cua_plugin_repo` | Git URL to clone CUA plugin source (fallback when `cua_plugin_path` is missing) |
| `cua_plugin_branch` | Branch to clone for CUA plugin source (default: `agenthle`) |

## Output

`--json` writes a JSON envelope to **stderr** (not stdout). Key fields:

- `meta.finalAssistantVisibleText` -> final agent text
- `meta.agentMeta.sessionId` -> session id
- `meta.agentMeta.usage.{input,output,total}` -> token counts
- `meta.executionTrace.{winnerProvider,winnerModel}` -> actual provider/model

Session trajectory at `~/.openclaw/agents/main/sessions/<sid>.jsonl`
is copied to `transcript.jsonl` in the work dir for artifact gathering.

`parse_artifacts()` reads the transcript JSONL and stderr envelope to
populate the `TrajectoryBuilder` with agent steps, tool calls, tool
results, and usage metrics.

## Layout

```
ale_run/agents/openclaw_cli/
├── __init__.py     — re-exports OpenClawCliConfig, OpenClawCliDeployer
├── config.py       — OpenClawCliConfig (standalone dataclass)
├── deployer.py     — OpenClawCliDeployer (BaseAgentDeployer: install/launch/parse_artifacts)
├── README.md       — this file
└── AGENTS.md       — integration notes, tool classification, install details
```
