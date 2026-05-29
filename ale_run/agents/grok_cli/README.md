# Grok CLI External Agent

Grok CLI runs inside the sandbox as a one-shot process:

```text
grok --prompt "..." --format json --max-tool-rounds 400
  -> CUA MCP Server over stdio
  -> sandbox filesystem / desktop
```

Unlike Claude Code (`--bare`), Grok CLI keeps its built-in tools (bash, file
read/write, grep, delegation, scheduling, etc.) enabled alongside the CUA MCP
tools. This gives the agent 54 tools total (33 enabled, 21 disabled).

## Architecture

Grok CLI is a standalone Bun-compiled binary -- no Node.js runtime needed for
the CLI itself. Node.js is only required for the CUA MCP Server bridge and for
running the fork bundle.

- Linux: installed via the official install script (`curl | bash`), binary at
  `~/.grok/bin/grok`
- Fork bundle (Linux): downloaded from a GitHub Release URL (via `bundle_url`
  config), placed at `~/.grok/bin/grok-bundle.js`, launched via `bun <bundle>`
  (the bundle uses `bun:sqlite`, so Bun — not Node — runs it)
- Fork binary (Windows): the linux bundle can't load under the Windows Bun
  loader (`@opentui` externalization), so Windows downloads a self-contained
  native `grok.exe` (via `win_binary_url` config) to `~/.grok/bin/grok.exe`
  and launches it directly — no bundle, no Bun. It is compiled from the same
  fork tree (`bun build --compile src/index.ts`) and carries the identical
  OpenRouter fixes, so headless `--prompt` over OpenRouter works without
  ZodErrors.

Both the linux bundle and the Windows `grok.exe` are published as assets on
the fork release `cua-verse/grok-cli` `v0.1.1-agenthle`
(`grok-bundle.js` and `grok-x86_64-pc-windows.exe`).

## Providers

**Direct xAI** uses the native xAI API and requires `GROK_API_KEY`. Uses the
stock binary.

**OpenRouter** requires `OPENROUTER_API_KEY` and the fork bundle (set
`bundle_url` in config). The fork switches from `@ai-sdk/xai` to
`@ai-sdk/openai` with `compatibility: "compatible"` for custom base URLs,
forces the Chat Completions API, and passes model IDs through unchanged so
provider-prefixed names like `x-ai/grok-3` reach OpenRouter as-is. It also
fixes MCP `CallToolResult` serialization in NDJSON output.

### Fork Bundle

When `bundle_url` is set in `GrokCliConfig`, the deployer:
1. Downloads the bundle JS file via curl during `install()`
2. Places it at `~/.grok/bin/grok-bundle.js`
3. At launch, uses `node grok-bundle.js --prompt ...` instead of `grok --prompt ...`

The bundle provides:
- Provider switching (`@ai-sdk/openai` for custom base URLs)
- Image injection middleware (prevents 100-300K text token costs per screenshot)
- `disabledTools` support (removes tools before agent starts)
- Model info resolution for OpenRouter model IDs

### Reasoning Model Support via Image Middleware

Grok-4 reasoning models (`x-ai/grok-4.1-fast`, `x-ai/grok-4-fast`) work via
OpenRouter after the image injection middleware fix (fork commit `4fe2f86`).
The middleware wraps the Chat Completions model to extract base64 images from
MCP tool results and re-inject them as proper vision content, preventing
100-300K text token costs per screenshot.

### OpenRouter Model Names

OpenRouter uses `x-ai/grok-4.1-fast` (dots, not dashes). The direct xAI API
uses `grok-4-1-fast-reasoning`. The deployer's `native_to_openrouter_model()`
handles this translation automatically.

## Bridge

The deployer writes `~/.grok/user-settings.json` with MCP server config during
`install()`. Grok CLI uses a **different MCP config format** from Claude Code
-- an array of `McpServerConfig` objects under `mcp.servers`:

```json
{
  "mcp": {
    "servers": [
      {
        "id": "cua",
        "label": "CUA MCP Server",
        "enabled": true,
        "transport": "stdio",
        "command": "/usr/local/bin/node",
        "args": ["/home/user/cua_mcp_server/src/index.js"]
      }
    ]
  },
  "disabledTools": ["search_web", "paid_request", "..."]
}
```

The `disabledTools` array is read by the fork and removes those tools before
the agent starts. The deployer populates it from the `disabled_tools` field in
`GrokCliConfig` (defaults to `_DISABLED_TOOLS_OPENROUTER` -- 21 tools).

## Config

```yaml
agent:
  harness: grok_cli
  model: x-ai/grok-4.3
  config:
    timeout_s: 600
    max_tool_rounds: 400
    bundle_url: "https://github.com/cua-verse/grok-cli/releases/download/..."
```

| Field | Meaning |
|-------|---------|
| `model` | Model id passed to the CLI via `--model` |
| `max_tool_rounds` | Passed to `--max-tool-rounds` flag (-1 = unlimited, mapped to 100000) |
| `timeout_s` | Wall-clock budget for the episode |
| `disabled_tools` | Tool names to remove before the agent runs. Written to `user-settings.json` as `disabledTools`. Defaults to 22 tools unavailable via OpenRouter. |
| `bundle_url` | Linux: URL to download the fork bundle JS. Empty = use stock binary. |
| `win_binary_url` | Windows: URL of the fork `grok.exe` (carries the OpenRouter fixes). Empty = debug-only fallback to stock grok on PATH. |

## Logs

`--format json` produces NDJSON with these event types:
- `step_start` -- session/step metadata
- `text` -- assistant text content
- `tool_use` -- tool call AND result bundled in one event
- `step_finish` -- result with usage (inputTokens, outputTokens, costUsdTicks)
- `error` -- error messages

## Execution

The deployer spawns the grok command (or `node bundle.js`) as a subprocess
with stdin=DEVNULL, capturing stdout as NDJSON and stderr as a log file.
Polled for completion with a configurable timeout.

## Smoke Test

```bash
uv run python -m ale_run run experiments/smoke_grok_cli_docker.yaml
```
