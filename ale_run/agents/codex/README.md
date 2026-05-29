# Codex

OpenAI Codex CLI agent deployed via NPM in sandbox environments.

## Architecture

```
FRAMEWORK                                 SANDBOX
lifecycle.py                              cua-computer-server
  install()   -> npm install, write config.toml, verify codex --version
  launch()    -> codex exec --json (NDJSON pipe, stdin from prompt file)
                   +-- CUA MCP Server (stdio)
                        +-- CUA HTTP API (desktop/shell)
  parse_artifacts() <- transcript.jsonl, stderr.log
```

Codex runs as a headless CLI process with the prompt piped via stdin. The agent
reaches the sandbox desktop/filesystem through the CUA MCP Server configured as
a stdio transport in `~/.codex/config.toml`.

## Source Strategy

NPM package: `@openai/codex@0.114.0` (pinned version, installed via
`npm install -g`).

Optionally, the native `codex` binary shipped by the NPM package is replaced
with a patched build (downloaded from a configured GitHub Release URL) to fix
the `apply_patch` corruption bug on Windows. The Linux and Windows builds are
distinct release assets (musl ELF `codex` vs `codex.exe` for
`x86_64-pc-windows-msvc`); the deployer downloads `patched_binary_url` on
Linux and `patched_binary_url_windows` on Windows. Replacement is a no-op when
the matching URL is empty.

## Providers

| Provider | Auth | Model ID |
|---|---|---|
| `direct` | `OPENAI_API_KEY` | `gpt-5.4` |
| `openrouter` | `OPENROUTER_API_KEY` | `openai/gpt-5.4` |

OpenRouter routing is auto-detected when the model ID contains a `/` (e.g.
`openai/gpt-5.4`). Configuration is written to `~/.codex/config.toml`:

```toml
model_provider = "openrouter"

[model_providers.openrouter]
name = "openrouter"
base_url = "https://openrouter.ai/api/v1"
env_key = "OPENROUTER_API_KEY"
```

## Bridge

CUA MCP Server, stdio transport. Written to `~/.codex/config.toml`:

```toml
[mcp_servers.cua]
type = "stdio"
command = "/usr/local/bin/node"
args = ["/home/user/cua_mcp_server/src/index.js"]
```

## Config

Agent-specific fields beyond `BaseAgentConfig`:

| Field | Type | Default | Meaning |
|---|---|---|---|
| `sandbox_mode` | str | `"danger-full-access"` | Codex sandbox policy |
| `yolo` | bool | `true` | Bypass all approval prompts |
| `reasoning_effort` | str | `"high"` | Model reasoning effort hint |
| `codex_version` | str | `"0.114.0"` | NPM package version to install |
| `patched_binary_url` | str | `""` | GitHub Release URL for patched Linux native binary (`codex`, musl x86-64) |
| `patched_binary_url_windows` | str | `""` | GitHub Release URL for patched Windows binary (`codex.exe`, windows-msvc); used instead of `patched_binary_url` on Windows |

## Logs

Codex `--json` outputs NDJSON with event types:
- `thread.started` -- session ID
- `turn.started` / `turn.completed` -- turn boundaries with usage stats
- `item.started` / `item.completed` -- tool calls, messages, reasoning
- `error` -- critical failures

Item types: `agent_message`, `reasoning`, `command_execution`,
`mcp_tool_call`, `file_change`, `web_search`, `error`.

## Execution

- CLI: `codex exec --model <model> --json --dangerously-bypass-approvals-and-sandbox`
- Process: detached via `setsid` (Linux) / `Start-Process` (Windows)
- Polling: every 2s via Popen.poll()
- Timeout: terminates process, then kills after grace period
- Exit code: nonzero -> `status: "failed"`

## Patched Binary Paths

npm 11.x stopped hoisting platform deps so the nested copy is the one
`codex.js`'s `require.resolve` actually picks. Both paths are checked
and whichever exists gets replaced:

- Linux top-level: `/usr/local/lib/node_modules/@openai/codex-linux-x64/vendor/x86_64-unknown-linux-musl/codex/codex`
- Linux nested: `/usr/local/lib/node_modules/@openai/codex/node_modules/@openai/codex-linux-x64/vendor/x86_64-unknown-linux-musl/codex/codex`
- Windows top-level: `...\npm\node_modules\@openai\codex-win32-x64\vendor\x86_64-pc-windows-msvc\codex\codex.exe`
- Windows nested: `...\npm\node_modules\@openai\codex\node_modules\@openai\codex-win32-x64\vendor\x86_64-pc-windows-msvc\codex\codex.exe`

## Smoke Test

```bash
uv run python -m ale_run run experiments/smoke_codex_docker.yaml
```
