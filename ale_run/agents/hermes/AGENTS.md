# Hermes Agent — Implementation Notes

Per-agent notes covering source/install decisions, CLI invocation, transcript
schema, tool classification, and deployer internals.

## Source And Fork

- Upstream: [`NousResearch/hermes-agent`](https://github.com/NousResearch/hermes-agent)
  (Apache-2.0; Python agent framework with a curl bash installer).
- Fork: [`cua-verse/hermes-agent`](https://github.com/cua-verse/hermes-agent)
  on branch ``agenthle`` (initial fork from upstream `main` @ `c8684254`,
  2026-04-30).  Submodule path: ``ale_run/agents/hermes/upstream``.
  Currently pinned at ``a1a77735``.

### Active fork patches (all on the `agenthle` branch)

| Commit | File | Purpose |
|---|---|---|
| `63cd14c4` | `tools/mcp_tool.py` | Register every dynamically-discovered MCP tool with `max_result_size_chars=float("inf")`.  Stops Layer-2 per-tool truncation from corrupting structured payloads (screenshots, large JSON results) at the 100K-char default. |
| `efca9dab` | `tools/tool_result_storage.py` | Skip tool messages whose content embeds a `data:image/...` URL from `enforce_turn_budget` Layer-3 candidates.  Prevents the 200K-char aggregate cap from truncating mid-base64. |
| `a1a77735` | `tools/mcp_tool.py` + `run_agent.py` | Forward MCP `ImageContent` blocks as a multimodal follow-up.  The handler persists images to `~/.hermes/mcp_images/` and emits a `_hermes_inline_images` marker; `AIAgent._expand_tool_image_followups` strips the marker and inserts a `user` message with `image_url` content parts right after the tool batch. |

Earlier patches (`e27f914f`, `a81d3662`) were intermediate steps for the
ImageContent path and are subsumed by `a1a77735`.

## Operating Systems

**Linux only.**  Upstream's installer reads ``OSTYPE`` and exits on Windows;
the docs explicitly say *"Native Windows is not supported. Please install
[WSL2]"*.  The deployer raises ``NotImplementedError`` if invoked against a
non-Linux sandbox.

## Install

The deployer's ``install()`` handles full setup on a clean ale-kasm image:

1. ``git clone --depth 1 -b agenthle https://github.com/cua-verse/hermes-agent.git ~/.hermes/hermes-agent``
2. ``uv pip install -e ".[all]"`` (with ``pip`` fallback)
3. ``npx playwright install --with-deps chromium`` (skipped if already cached)
4. Writes ``~/.hermes/.env`` (provider credentials)
5. Writes ``~/.hermes/config.yaml`` (model, compression, MCP servers, delegation)

Pre-requisites on the sandbox (already present on ale-kasm):
* `curl`, `git`, Node 22+
* `uv` (the deployer bootstraps it via `astral.sh/uv/install.sh` if missing)

## CLI Invocation

What ``hermes_runner.sh`` executes:

```bash
hermes chat \
  -q "$(cat ~/hermes_work/prompt.txt)" \
  -Q \
  --provider openrouter \
  --model "anthropic/claude-sonnet-4-6" \
  --max-turns 100000 \
  --yolo \
  --accept-hooks \
  --ignore-rules \
  --pass-session-id \
  --toolsets "terminal,file,skills,todo,memory,web,vision,browser,code_execution,delegation,cronjob,mcp-cua"
```

| Flag | Purpose |
|---|---|
| `-q --query <text>` | Single-query non-interactive mode. |
| `-Q --quiet` | Suppress banner, spinner, and tool previews. |
| `--yolo` | Bypass dangerous-command approval prompts. |
| `--accept-hooks` | Auto-approve unseen shell hooks. |
| `--ignore-rules` | Skip auto-injection of `AGENTS.md` / `SOUL.md` / `.cursorrules`. |
| `--max-turns N` | Hard turn cap (100000 so wall-clock timeout is the real cap). |
| `--pass-session-id` | Inject session id into system prompt for post-run lookup. |
| `--toolsets <csv>` | Exhaustive enabled list. |

## Transcript Recovery

Hermes persists sessions in SQLite at ``~/.hermes/state.db``. After the run,
the deployer finds the session id (preferring the printed ``session_id: ...``
from ``--pass-session-id``, falling back to the latest row whose
``started_at`` is at-or-after the run start marker) and exports it with
``hermes sessions export``.

### Session-export schema

```
session = {
  "id": str, "source": "cli", "model": str,
  "started_at": float, "ended_at": float, "end_reason": str,
  "message_count": int, "tool_call_count": int,
  "input_tokens": int, "output_tokens": int,
  "cache_read_tokens": int, "cache_write_tokens": int,
  "estimated_cost_usd": float | None, "actual_cost_usd": float | None,
  "api_call_count": int,
  "messages": [
    {
      "role": "user" | "assistant" | "tool" | "system",
      "content": str | None,
      "tool_calls": [...] | None,
      "tool_call_id": str | None, "tool_name": str | None,
      "reasoning_content": str | None,
    },
  ],
}
```

## Tool Classification

| Toolset | Status | Reason |
|---|:---:|---|
| `terminal`, `file`, `skills`, `todo`, `memory` | enabled | Core agent capabilities. |
| `web`, `vision`, `browser` | enabled | Network research + local Chromium via Playwright. |
| `code_execution`, `delegation`, `cronjob` | enabled | Long-run optimization. |
| `mcp-cua` (18 tools) | enabled | CUA bridge (screenshot, click, type, key, scroll, etc.). |
| `search`, `image_gen`, `rl`, `tts`, `moa` | disabled | Redundant or requires extra API keys. |
| `session_search`, `clarify`, `messaging` | disabled | No TTY / no gateway. |
| Platform-specific (`homeassistant`, `kanban`, `discord`, etc.) | disabled | Requires credentials not available in sandbox. |

## Pitfalls

- **No NDJSON output**: trajectories are SQLite-backed. Relies on `hermes sessions export --session-id` post-run.
- **MCP tools filtered by `mcp-<server>` toolset name**: deployer explicitly includes `mcp-cua` in toolsets.
- **Memory double-suppressed by `--ignore-rules`**: the in-session `memory` tool reports "not available". Cross-session value is zero anyway on fresh sandbox images.
- **`max_turns` has no unlimited sentinel**: `-1` would stop on turn 1. We use `100_000`.
- **`hermes chat -q` exit IS the run-end signal**: `KEEPALIVE_PREAMBLE` instructs the agent to keep the session alive for async work.
- **Compression config**: `context_length: 1M`, `threshold: 0.85`, `protect_last_n: 8`. Required for the multi-MB screenshot tool path.
