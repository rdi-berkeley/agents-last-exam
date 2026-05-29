# Droid (Factory AI CLI) Integration Notes

## Source And Fork

- Upstream: Factory.ai's closed-source `droid` binary (no public source repo,
  no npm package — distributed only via the `app.factory.ai/cli` installer).
- Fork: **none** (closed-source). All ALE-specific behavior lives in
  `deployer.py` and the YAML config.
- Pinned version: **`0.116.0`**.

The `app.factory.ai/cli` shell installer downloads from
`https://downloads.factory.ai/factory-cli/releases/<VER>/<platform>/<arch>/droid[.exe]`.
Direct-download URLs are stable per version.

## Install

### Auto-install (sandbox deployer)

The `DroidDeployer.install()` method auto-installs if `droid` is not on
PATH. Two-stage fallback:

1. Official installer: `curl -fsSL https://app.factory.ai/cli | sh`
   (needs `xdg-utils` as a dep)
2. Pinned direct download:
   ```bash
   VER=0.116.0
   curl -fsSL "https://downloads.factory.ai/factory-cli/releases/${VER}/linux/x64/droid" \
     -o ~/.local/bin/droid && chmod +x ~/.local/bin/droid
   ```

The Linux installer drops `~/.local/bin/droid` (147 MB binary) and a
bundled ripgrep at `~/.factory/bin/rg`.

Use `linux/x64-baseline` if `/proc/cpuinfo` does not advertise `avx2`.

### Uninstall

```bash
rm -rf ~/.local/bin/droid ~/.factory
```

## CLI Invocation

Verified headless invocation (droid 0.116.0):

```bash
FACTORY_API_KEY=byok-noop \
  droid exec \
    -f /path/to/prompt.txt \
    -m anthropic/claude-sonnet-4-6 \
    --output-format stream-json \
    --skip-permissions-unsafe \
    --reasoning-effort medium \
    --cwd /home/kasm-user/work \
    --disabled-tools 'AskUser,squad-board,slack_post_message,...'
```

Required flags:

| Flag | Reason |
|---|---|
| `-f <prompt-file>` | UTF-8-safe prompt input |
| `-m <model-id>` | OpenRouter model slug; matches `customModels[0].model` |
| `--output-format stream-json` | NDJSON event stream the deployer parses |
| `--skip-permissions-unsafe` | Allows ALL tools; only fully-headless mode |
| `--cwd <dir>` | Pin the working dir |
| `--disabled-tools <csv>` | Strip Factory-cloud SaaS / interactive tools |

Optional:

| Flag | Reason |
|---|---|
| `--reasoning-effort {off\|none\|low\|medium\|high}` | Defaults per model |
| `--enabled-tools <csv>` | Whitelist; rarely needed |

## Auth (OpenRouter BYOK without a Factory account)

Even with BYOK, `droid exec` rejects sessions when no Factory
credentials are present. The check is satisfied by *any* non-empty
`FACTORY_API_KEY` env var (verified on 0.116.0). The deployer exports
`byok-noop` as a sentinel value so users never need a Factory account.

The OpenRouter API key goes into `~/.factory/settings.json`:

```json
{
  "customModels": [{
    "model": "anthropic/claude-sonnet-4-6",
    "displayName": "anthropic/claude-sonnet-4-6 [OpenRouter]",
    "baseUrl": "https://openrouter.ai/api/v1",
    "apiKey": "<OPENROUTER_API_KEY>",
    "provider": "generic-chat-completion-api",
    "maxOutputTokens": 128000
  }]
}
```

## Verified Event Shapes (droid 0.116.0, --output-format stream-json)

```
{"type":"system","subtype":"init","cwd","session_id","tools":[...],"model","reasoning_effort"}
{"type":"message","role":"user","id","text","timestamp","session_id"}
{"type":"message","role":"assistant","id","text","timestamp","session_id"}
{"type":"tool_call","id":"toolu_...","messageId","toolId","toolName","parameters":{...},"timestamp","session_id"}
{"type":"tool_result","id":"toolu_...","messageId","toolId","isError":bool,"value":"<text>","timestamp","session_id"}
{"type":"error","source":"agent_loop"|"cli","message","timestamp","session_id"}
{"type":"completion","finalText","numTurns","durationMs","usage":{...}}
```

Per-session origin log:
- `~/.factory/sessions/<cwd-encoded>/<session-id>.jsonl`
- `~/.factory/logs/droid-log-single.log`
- `~/.factory/logs/console.log`

## Tool Classification

### Disabled by default (Factory-cloud / interactive)

| Tool | Reason |
|---|---|
| `squad-board` | Factory cloud SaaS |
| `slack_post_message` | External comms |
| `store_agent_readiness_report` | Factory cloud |
| `GenerateDroid` | Factory cloud — sub-agent generation |
| `ProposeMission` | Factory cloud — mission lifecycle |
| `StartMissionRun` | Factory cloud |
| `EndFeatureRun` | Factory cloud |
| `DismissHandoffItems` | Factory cloud |
| `ExitSpecMode` | Factory cloud |
| `AskUser` | Interactive — would block headless runs |
| `WebSearch` | Factory cloud-routed — requires real Factory key |

### Kept enabled (24 tools, all verified)

Built-in: Read, Edit, Execute, Glob, Grep, LS, Create, ApplyPatch,
FetchUrl, TodoWrite, Skill, Task, ToolSearch, Write.

CUA MCP (14 tools via bridge): `cua___screenshot`, `cua___click`,
`cua___type`, `cua___key`, `cua___key_down`, `cua___key_up`,
`cua___hold_key`, `cua___mouse_move`, `cua___mouse_down`,
`cua___mouse_up`, `cua___drag`, `cua___scroll`, `cua___wait`,
`cua___cursor_position`.

## Known Issues

### Provider routing (byok_provider selection)

| `byok_provider` value | Endpoint | Status |
|---|---|---|
| `generic-chat-completion-api` | `/v1/chat/completions` | **Working** — verified with OpenRouter |
| `openai` | `/v1/chat/completions` then `/v1/responses` | ⚠️ OpenRouter doesn't implement Responses API; may retry forever |
| `anthropic` | `/v1/messages` | ⚠️ Empty-body retry loop on OpenRouter |

Use `generic-chat-completion-api` for OpenRouter.

### ARM64 not supported

The droid binary is x86_64-only. Docker images running on ARM64 hosts
(e.g. Apple Silicon Macs) will fail at install with
`Unsupported platform: linux-arm64`. Use an AMD64 machine or cross-arch
emulation.
