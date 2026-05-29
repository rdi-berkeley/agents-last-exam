# Codex -- Implementation Notes

Per-agent implementation details, test records, and tool compatibility for the
OpenAI Codex CLI agent deployer (agents-last-exam framework).

---

## 1. Source & Fork Strategy

| Aspect | Value |
|---|---|
| Source | NPM `@openai/codex@0.114.0` + optional patched native binary from GitHub Release URL |
| Install method | `npm install -g @openai/codex@0.114.0`, then optionally overwrite vendor binary with patched build |
| Fork patch (2026-05-01) | `codex-rs/arg0/src/lib.rs` (+39/-12): replace Windows `apply_patch.bat` shim with `apply_patch.exe` hardlink + add `apply_patch.exe`/`applypatch.exe` to argv0 dispatch |

The NPM package alone handles headless execution, OpenRouter routing, and
MCP server config. The fork is needed only to fix the Windows
`apply_patch` corruption bug. When no `patched_binary_url` is configured,
the binary-replacement step is silently skipped.

---

## 2. Install

### Commands

```bash
# Linux
npm install -g @openai/codex@0.114.0
```

### Binary Paths

| OS | Binary | Version command |
|---|---|---|
| Linux | `/usr/local/bin/codex` (symlink to npm module) | `codex --version` -> `codex-cli 0.114.0` |

### Required Environment

| Variable | Provider | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | direct | OpenAI API authentication |
| `OPENROUTER_API_KEY` | openrouter | OpenRouter API authentication |

### Bridge Files (written by `install()`)

- `~/.codex/config.toml` -- MCP server config + provider routing

### Prerequisites

- Node.js (for npm install + MCP server)
- Git (Codex requires a git repo as working directory)
- CUA MCP Server at sandbox's `mcp_server_dir`

---

## 3. CLI Invocation

```bash
# YOLO mode (default -- bypasses all prompts and sandbox)
cat prompt.txt | codex exec --model <model> --json \
  --dangerously-bypass-approvals-and-sandbox

# Full-auto mode (respects sandbox policy)
cat prompt.txt | codex exec --model <model> --json \
  --full-auto --sandbox danger-full-access
```

---

## 4. Output Format

NDJSON (one JSON object per line) on stdout:

```jsonl
{"type":"thread.started","thread_id":"019dd0cc-..."}
{"type":"turn.started"}
{"type":"item.completed","item":{"id":"item_0","type":"agent_message","text":"PONG"}}
{"type":"turn.completed","usage":{"input_tokens":10209,"cached_input_tokens":0,"output_tokens":28}}
```

### Event Types

| Event | Meaning |
|---|---|
| `thread.started` | Session created |
| `turn.started` / `turn.completed` | Turn boundaries; `turn.completed` includes token usage |
| `item.started` | Tool call or message began (may lack result) |
| `item.completed` | Tool call or message finished with result |
| `error` | Critical failure |

### Item Types (in `item.completed`)

| Type | Role | Description |
|---|---|---|
| `agent_message` | assistant | Model text response |
| `reasoning` | assistant | Internal reasoning trace |
| `command_execution` | tool | Shell command + output + exit code |
| `mcp_tool_call` | tool | MCP tool invocation with result/error |
| `file_change` | tool | File edit metadata |
| `web_search` | tool | Search query |
| `error` | system | Error item |

---

## 5. Config Fields

| Field | Type | Default | Meaning |
|---|---|---|---|
| `model` | str | `"openai/gpt-5.4"` | LLM model (OpenRouter format if contains `/`) |
| `timeout_s` | float | `600` | Wall-clock budget |
| `sandbox_mode` | str | `"danger-full-access"` | Codex sandbox policy |
| `yolo` | bool | `true` | Bypass all approval prompts |
| `reasoning_effort` | str | `"high"` | Model reasoning effort hint |
| `codex_version` | str | `"0.114.0"` | NPM package version to install |
| `patched_binary_url` | str | `""` | GitHub Release URL for patched Linux binary (`codex`) |
| `patched_binary_url_windows` | str | `""` | GitHub Release URL for patched Windows binary (`codex.exe`); used instead of `patched_binary_url` on Windows |

---

## 6. Known Issues

- **No prompt caching for Anthropic via codex+OpenRouter**: OpenRouter's
  `/v1/responses` translation layer drops `cache_control` for Anthropic models.
- **`apply_patch` on Windows**: Upstream `.bat` shim corruption -- resolved via
  patched binary when `patched_binary_url` is set.
- **Codex requires git repo**: The working directory must be a git repository.
  The deployer initializes one via `git init` if missing.
- **NDJSON BOM**: Output may include UTF-8 BOM prefix. The parser strips BOMs.
- **Orphaned MCP processes**: stdio MCP servers launched by Codex may survive
  after the parent is killed.

---

## 7. Migration Notes (agenthle -> agents-last-exam)

This deployer was migrated from `agenthle/orchestration/external/codex/`. Key
differences from the old framework:

- Uses `BaseAgentDeployer` (from `ale_run.base_interface`) instead of
  `ExternalAgentDeployer`
- Subprocess-based local execution instead of remote VM RPC
  (`_run_remote`, `_upload_file`, etc.)
- TrajectoryBuilder-based artifact parsing instead of InteractionLog
- Config is a simple dataclass (`CodexConfig`) instead of YAML-loader
- Registered via `_AGENT_FQNS` in `factory.py` instead of
  `register_agent()` call
