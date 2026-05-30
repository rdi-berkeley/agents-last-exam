# Cursor CLI Agent

Cursor CLI (`cursor-agent`) is Anysphere's closed-source binary. It
runs inside the sandbox as a one-shot process:

```text
cursor-agent -p --output-format stream-json --model <id>
             --force --approve-mcps --trust --sandbox disabled
  -> CUA MCP Server (stdio, auto-discovered from ~/.cursor/mcp.json)
  -> CUA Computer Server :8000
  -> desktop / filesystem
```

`cursor-agent` keeps its built-in tools enabled alongside the CUA MCP
tools. The CLI ships a bundled `node` runtime; the sandbox only needs
Node for the CUA MCP Server bridge.

## Architecture

Closed-source binary distributed by Cursor (Anysphere). Pre-built
versioned binary tree drops into a per-user install dir.

| | Path |
|---|---|
| Binary | `~/.local/bin/cursor-agent` |
| Config | `~/.cursor/` |
| MCP config | `~/.cursor/mcp.json` |
| CLI config | `~/.cursor/cli-config.json` |
| Auth | `~/.config/cursor/auth.json` |

## Provider — Cursor Backend Only

There is **no OpenRouter routing path** for Cursor CLI. The binary
validates keys against Cursor's own backend at `api.cursor.sh`. Use a
real Cursor account.

### Authentication: via auth.json, NOT CURSOR_API_KEY

`cursor-agent` authenticates exclusively via `~/.config/cursor/auth.json`
(OAuth tokens), **NOT** the `CURSOR_API_KEY` env var. The auth.json
contains `accessToken`, `refreshToken`, and `apiKey` fields from an
OAuth login flow (`cursor-agent login`).

The deployer receives the auth.json content via:
1. `CURSOR_AUTH_JSON` env var — raw JSON content (set by lifecycle env passthrough)
2. `CURSOR_AUTH_JSON_PATH` env var — path to a file on the host

The lifecycle materializes the file content into `CURSOR_AUTH_JSON`
automatically when only `CURSOR_AUTH_JSON_PATH` is set.

### Host setup (macOS)

On macOS, `cursor-agent login` stores tokens in the **Keychain**, not a
file. Extract them manually:

```bash
security unlock-keychain
ACCESS=$(security find-generic-password -s "cursor-access-token"  -a "cursor-user" -w)
REFRESH=$(security find-generic-password -s "cursor-refresh-token" -a "cursor-user" -w)
APIKEY=$(security find-generic-password -s "Cursor Safe Storage"  -a "Cursor Key"  -w)

mkdir -p /tmp/cursor_local
cat > /tmp/cursor_local/auth.json <<EOF
{"accessToken":"$ACCESS","refreshToken":"$REFRESH","apiKey":"$APIKEY"}
EOF
chmod 600 /tmp/cursor_local/auth.json
```

Then set `CURSOR_AUTH_JSON_PATH=/tmp/cursor_local/auth.json` in your
secret `.env` file.

## Install

The deployer auto-installs if `cursor-agent` is not on PATH:

```bash
curl -fsSL https://cursor.com/install | bash
```

## Config

```yaml
agent:
  harness: cursor_cli
  model: claude-4.6-sonnet-medium
  config:
    timeout_s: 600
    max_turns: 300
```

| Field | Meaning |
|---|---|
| `model` | Cursor model id passed via `--model`. Use `cursor-agent --list-models` to enumerate. |
| `timeout_s` | Wall-clock timeout |
| `max_turns` | Informational only; cursor-agent has no native turn cap |
| `disabled_tools` | Scope patterns injected into `cli-config.json` `permissions.deny` |

## Bridge

`install()` writes `~/.cursor/mcp.json` (auto-discovered, no CLI flag)
and `~/.cursor/cli-config.json` with permissive allow-list. Permissions
locked open via three layers:

1. CLI flags: `--force --approve-mcps --trust --sandbox disabled`
2. `cli-config.json` with `approvalMode: "unrestricted"` + `Mcp(*:*)` allow
3. No project-level `.cursor/cli.json` (deleted if present)

## Output Format

`--output-format stream-json` produces NDJSON:

- `system / init` — `model`, `session_id`, `permissionMode`
- `user` — echoed prompt content
- `assistant` — model text (`message.content[].text`)
- `tool_call / started|completed` — payload at `event.tool_call.<kind>ToolCall`
- `result / success` — final answer + `usage.{inputTokens,outputTokens,cacheReadTokens,cacheWriteTokens}`

Usage fields are camelCase (unlike Anthropic's snake_case).

## Layout

```
ale_run/agents/cursor_cli/
├── __init__.py     — re-exports CursorCliConfig, CursorCliDeployer
├── config.py       — CursorCliConfig (standalone dataclass)
├── deployer.py     — CursorCliDeployer (install/launch/parse_artifacts)
├── README.md       — this file
└── AGENTS.md       — integration notes, tool classification, known issues
```
