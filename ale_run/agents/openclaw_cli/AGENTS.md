# OpenClaw CLI Runner — Integration Notes

## Source

| Aspect | Value |
|---|---|
| Source | Fork tarball, installed via `npm install -g` |
| CLI subcommand | `openclaw agent --local --agent main -m "$PROMPT" --json --timeout N --thinking high` |
| Output channel | **stderr** (verified empirically; `--json` does not honor stdout) |
| Plugin loading | Process-startup-tied; `~/.openclaw/extensions/cua/dist/index.cjs` |
| CUA bridge | Native OpenClaw plugin (LiteDesktopActionSpace), NOT MCP |

## CLI Invocation

```bash
NO_COLOR=1 openclaw agent --local \
  --agent main \
  --message "$PROMPT" \
  --json \
  --timeout 600 \
  --thinking high
```

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

## Config File Layout

Written by `_write_config()` during `install()`:

### `~/.openclaw/openclaw.json`

Agent model, timeout, plugin allow/deny, tool allow/deny, gateway mode
(`local`), heartbeat, vision model.

### `~/.openclaw/agents/main/agent/auth-profiles.json`

Provider credentials (OpenRouter or OpenAI API key).

### `~/.openclaw/exec-approvals.json`

YOLO mode: `security: "full"`, `ask: "off"`.

### `~/.openclaw/state/workspace-state.json`

Marks setup as completed to skip the bootstrap wizard.

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
the agent exits.
