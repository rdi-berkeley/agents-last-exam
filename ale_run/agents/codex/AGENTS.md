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

## 4b. Tool Compatibility Matrix (demo/tool_smoke)

Probed by `demo/tool_smoke` (Linux) and `demo/tool_smoke_win` (Windows) on the
re-baked dev VMs, 2026-06-15, fork `codex-cli 0.0.0-agenthle-20260614`, a
direct-provider model. The task exercises every tool the agent is offered
and records pass/fail per tool. Result: **Linux 32/32 tested passed (2 untested),
Windows 26/28 passed (2 failed, 2 untested)** — all 14 GUI `mcp__cua.*` tools
pass on both OSes.

Legend: ✅ works · ❌ fails · ➖ untested (couldn't exercise) · — not offered on that OS

| Tool | Linux | Win | Note |
|---|---|---|---|
| `functions.exec_command` | ✅ | — | one-shot shell; **Win uses `shell_command`** (`unified_exec` off on Windows) |
| `functions.write_stdin` | ✅ | — | stdin to a persistent `unified_exec` session — Linux only |
| `functions.shell_command` | — | ✅ | Windows shell exec (replaces exec_command/write_stdin) |
| `functions.apply_patch` | ✅ | ✅ | **Win relies on the fork `apply_patch.exe` hardlink fix** |
| `functions.update_plan` | ✅ | ✅ | |
| `functions.view_image` | ✅ | ✅ | |
| `functions.list_mcp_resources` | ✅ | ✅ | |
| `functions.list_mcp_resource_templates` | ✅ | ✅ | |
| `functions.read_mcp_resource` | ➖ | ➖ | untested: MCP resource list empty, no URI to read |
| `functions.request_user_input` | ➖ | ➖ | untested: Plan-mode only + needs a human (headless) |
| `functions.spawn_agent` | ✅ | ✅ | multi_agent_v2 sub-agent (the target model accepts it; V1 must stay disabled) |
| `functions.wait_agent` | ✅ | ✅ | |
| `functions.interrupt_agent` | ✅ | ✅ | |
| `functions.list_agents` | ✅ | ✅ | |
| `functions.send_message` | ✅ | ❌ | **❌Win**: strict "no observable return content" (child DID receive `SEND_MESSAGE_OK`) — V2-messaging/test-rule artifact, not a transport bug |
| `functions.followup_task` | ✅ | ❌ | **❌Win**: same strict-return artifact (child returned `FOLLOWUP_TASK_OK`) |
| `functions.create_goal` | ✅ | — | goals tools not surfaced/exercised on Windows in this run |
| `functions.get_goal` | ✅ | — | |
| `functions.update_goal` | ✅ | — | |
| `mcp__cua.screenshot` | ✅ | ✅ | GUI via CUA MCP bridge |
| `mcp__cua.click` | ✅ | ✅ | |
| `mcp__cua.type` | ✅ | ✅ | (needs a clean desktop to verify visible effect) |
| `mcp__cua.scroll` | ✅ | ✅ | (needs a clean desktop to verify visible effect) |
| `mcp__cua.drag` | ✅ | ✅ | |
| `mcp__cua.key` / `key_down` / `key_up` / `hold_key` | ✅ | ✅ | |
| `mcp__cua.mouse_move` / `mouse_down` / `mouse_up` | ✅ | ✅ | |
| `mcp__cua.cursor_position` | ✅ | ✅ | |
| `mcp__cua.wait` | ✅ | ✅ | |
| `web.run` | ✅ | ✅ | |
| `multi_tool_use.parallel` | ✅ | ✅ | parallel tool-call wrapper |

Notes:
- Total tool count differs by OS (Linux 34, Windows 30) because `unified_exec`
  (and its `exec_command`/`write_stdin`) is off on Windows and the goals tools
  weren't offered there; Windows substitutes `shell_command`.
- The only true failures are the two Windows V2-messaging tools, and they're a
  strict scorer rule ("the call itself returned no observable payload") rather
  than a real breakage — the sub-agent did receive/complete the work.
- GUI (`mcp__cua.*`) tools all pass on both OSes; `type`/`scroll` only verify
  their visible effect on a clean desktop (leftover windows can hide it).

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
