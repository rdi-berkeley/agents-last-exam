# forgecode -- Agent Implementation Notes

Implementation reference for the `tailcallhq/forge` agent deployer in the
agents-last-exam framework. Read `README.md` first for what the agent is
and how to configure it; this file covers source/fork strategy, install
pinning, tool classification, and test records.

## 0. Conformance with BaseAgentDeployer

| Method | Status | Notes |
|---|---|---|
| `install()` | Implemented | Downloads forge binary from GitHub releases (`https://github.com/anysphere/forgecode/releases/`). Falls back to `cargo install` from source. Verifies with `forge --version`. Writes `permissions.yaml` (fully permissive) and `forge.toml` (model config, auto_dump). |
| `launch()` | Implemented | Runs `forge -p "<prompt>" --conversation-id <UUID> -C <dump_dir>`. Follows the droid deployer pattern for subprocess spawning, polling, timeout handling. Post-run materialises dump.json via `forge conversation dump <id>`. |
| `parse_artifacts()` | Implemented | Parses forge's `ConversationDump` JSON (`dump.json`). Falls back to transcript.jsonl if dump.json is missing. Maps text/tool/image entries to trajectory steps. |

## 1. Source strategy

| Aspect | Decision |
|---|---|
| Upstream | `github.com/tailcallhq/forgecode` |
| Releases | `github.com/anysphere/forgecode/releases/` |
| Install | `install()` downloads pre-built binary; falls back to `cargo install` from source |

## 2. Install

The deployer's `install()` method handles installation automatically:

1. Checks if `forge` is on PATH.
2. If not, downloads from GitHub releases:
   `https://github.com/anysphere/forgecode/releases/latest/download/forge-linux-x64`
3. Falls back to `cargo install --git https://github.com/anysphere/forgecode --bin forge forge_main`
4. Verifies with `forge --version`.
5. Writes config files:
   - `~/.forge/permissions.yaml` -- fully permissive (read/write/command/url)
   - `~/.forge/.forge.toml` -- `[session]` provider/model, `auto_dump = "json"`
   - Wipes `~/.forge/.credentials.json` to force env-var re-migration

### Where things land

| Path | Role |
|---|---|
| `~/.local/bin/forge` or `~/.cargo/bin/forge` | forge binary |
| `~/.forge/.forge.toml` | model routing, auto_dump config |
| `~/.forge/permissions.yaml` | permissive access rules |
| `~/.forge/logs/` | forge tracing logs |
| `~/.forge/.forge.db` | SQLite conversation store |
| `<work_dir>/prompt.txt` | task instruction |
| `<work_dir>/transcript.jsonl` | stdout capture |
| `<work_dir>/stderr.log` | stderr capture |
| `<work_dir>/forge.pid` | process pid |
| `<work_dir>/exit_code.txt` | exit code |
| `<work_dir>/dump_dir/` | CWD during run; auto_dump JSONs land here |
| `<work_dir>/dump.json` | materialised conversation dump |

## 3. Tool compatibility

forge ships a tool catalog defined in `crates/forge_domain/src/tools/catalog.rs`.
All tools execute directly inside the sandbox via Rust syscalls -- no Docker
layer or MCP bridge.

| Tool | Status | Notes |
|---|---|---|
| `read` | Enabled | Read files |
| `write` | Enabled | Create/overwrite files |
| `fs_search` | Enabled | Search for patterns in files |
| `remove` | Enabled | Delete files |
| `patch` | Enabled | Modify file content |
| `multi_patch` | Enabled | Multiple edits to a single file |
| `undo` | Enabled | Revert edits (context-dependent) |
| `shell` | Enabled | Run shell commands |
| `fetch` | Enabled | HTTP requests |
| `skill` | Enabled | Load skill information |
| `todo_write` | Enabled | Task-list bookkeeping |
| `todo_read` | Enabled | Retrieve todos |
| `task` | Enabled | Delegate to sub-agent |
| `followup` | Off by default | Interactive -- would block headless |
| `plan` | Off by default | muse-agent only |

## 4. Output schema

The canonical record is the `ConversationDump` JSON from
`forge conversation dump <id>`:

```
{
  "conversation": {
    "context": {
      "messages": [
        { "text": { "role": "User|Assistant|System", "content": "...",
                    "tool_calls": [...], "reasoning_details": [...] },
          "usage": { "prompt_tokens": ..., "completion_tokens": ...,
                     "cached_tokens": ..., "cost": ... } },
        { "tool": { "name": "shell", "call_id": "...",
                    "output": { "values": [{"text": "..."}, ...] } } },
        ...
      ]
    }
  },
  "related_conversations": [...]
}
```

Mapping to trajectory steps:

| Source | Trajectory Step |
|---|---|
| `text.role == "User"` | `source="user"` |
| `text.role == "System"` | `source="system"` |
| `text.reasoning_details[].text` | `source="agent"`, prefixed `[reasoning]` |
| `text.role == "Assistant"`, `content` | `source="agent"` |
| `text.tool_calls[]` | `source="agent"`, ToolCall |
| `tool.output.values[]` | `source="environment"`, Observation/ToolResult |

## 5. Provider routing

| `provider:` | Model example | Env var exported | `[session]` in forge.toml |
|---|---|---|---|
| `openrouter` | `anthropic/claude-sonnet-4-6` | `ANTHROPIC_API_KEY` = OR key, `ANTHROPIC_BASE_URL` = `https://openrouter.ai/api/v1` | `provider_id = "open_router"`, `model_id = "anthropic/claude-sonnet-4-6"` |
| `direct` | `anthropic/claude-sonnet-4-6` | `ANTHROPIC_API_KEY` | `provider_id = "anthropic"`, `model_id = "claude-sonnet-4-6"` |
| `direct` | `openai/gpt-4o` | `OPENAI_API_KEY` | `provider_id = "openai"`, `model_id = "gpt-4o"` |

## 6. Known gaps

- **Windows**: not supported. sandbox executor is Linux-only.
- **`Followup` tool**: forge can call `Followup` to ask the user a question;
  under benchmark prompts there is no human to answer.
- **`semantic_search` tool**: depends on `forge workspace sync`.
- **Sub-agent trajectories**: `related_conversations[]` are not folded into
  the main trajectory.
