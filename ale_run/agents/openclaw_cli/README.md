# OpenClaw CLI Agent

Headless wrapper around `openclaw agent --local`, using a fork tarball
and native CUA plugin (not MCP).

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
install time.

| | Path |
|---|---|
| Binary | `~/.local/bin/openclaw` (npm global install) |
| Config home | `~/.openclaw/` |
| CUA plugin | `~/.openclaw/extensions/cua/dist/index.cjs` |
| Auth profiles | `~/.openclaw/agents/main/agent/auth-profiles.json` |
| Sessions | `~/.openclaw/agents/main/sessions/<sid>.jsonl` |

## Provider — OpenRouter or Direct

Auth is configured via `auth-profiles.json`. The deployer auto-detects
from available env vars:
- `OPENROUTER_API_KEY` → provider `openrouter`
- `OPENAI_API_KEY` fallback → provider `openai`

## Install

Three-stage install in `install()`:

1. **CLI**: If `openclaw` not on PATH, install from fork tarball at
   `config.tarball_path` (default: `/opt/ale/openclaw-fork.tgz`) via
   `npm install -g`
2. **CUA plugin**: If `~/.openclaw/extensions/cua/dist/index.cjs` not
   present, build from source at `config.cua_plugin_path`
   (default: `/opt/ale/openclaw-cua-plugin`) via `npm install && npm run build`
3. **Config**: Write `openclaw.json`, `auth-profiles.json`,
   `exec-approvals.json`, `workspace-state.json`

Both the tarball and plugin source must be baked into the sandbox image
or volume-mounted.

## Config

```yaml
agent:
  harness: openclaw_cli
  model: openai/gpt-5.4
  config:
    timeout_s: 1800
    thinking: high
    tarball_path: /opt/ale/openclaw-fork.tgz
    cua_plugin_path: /opt/ale/openclaw-cua-plugin
```

| Field | Meaning |
|---|---|
| `model` | OpenRouter model slug (e.g. `openai/gpt-5.4`, `anthropic/claude-sonnet-4-6`) |
| `thinking` | Reasoning depth: `off\|low\|medium\|high` |
| `vision_model` | Per-tool model override for image analysis |
| `tools_deny` | Tools removed from the schema |
| `plugins_allow` | Which plugins load at startup |
| `heartbeat_every` | `"never"` skips background loop |
| `tarball_path` | Path to fork tarball inside sandbox |
| `cua_plugin_path` | Path to CUA plugin source inside sandbox |

## Output

`--json` writes a JSON envelope to **stderr** (not stdout). Key fields:

- `meta.finalAssistantVisibleText` → final agent text
- `meta.agentMeta.sessionId` → session id
- `meta.agentMeta.usage.{input,output,total}` → token counts
- `meta.executionTrace.{winnerProvider,winnerModel}` → actual provider/model

Session trajectory at `~/.openclaw/agents/main/sessions/<sid>.jsonl`
is copied to `transcript.jsonl` in the work dir for artifact gathering.

## Layout

```
ale_run/agents/openclaw_cli/
├── __init__.py     — re-exports OpenClawCliConfig, OpenClawCliDeployer
├── config.py       — OpenClawCliConfig (BaseAgentConfig dataclass)
├── deployer.py     — OpenClawCliDeployer (install/launch/parse_artifacts)
├── README.md       — this file
└── AGENTS.md       — integration notes, tool classification
```
