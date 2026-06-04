# Cursor CLI Agent (`cursor-agent`)

Anysphere's closed-source `cursor-agent` binary, run as a one-shot process
inside the sandbox VM:

```text
cursor-agent -p --model <id> --output-format stream-json
             --force --approve-mcps --trust --sandbox disabled
  → CUA MCP Server (stdio, auto-discovered from ~/.cursor/mcp.json)
  → CUA Computer Server (image-specific port; 5000 on the GCE images)
  → desktop / filesystem
```

The CLI ships its own bundled `node`, so the sandbox needs Node only for the
CUA MCP bridge, not for `cursor-agent` itself. Cursor's own built-in tools stay
enabled alongside the CUA MCP tools.

> **Cursor backend only.** `cursor-agent` validates credentials against Cursor's
> own backend — there is **no OpenRouter / BYOK path**. You need a real Cursor
> account. (This is why the agent presets are named `*_cursor`, not `*_or`.)

---

## Quick start

1. **Authenticate** — pick one (see [Authentication](#authentication)):
   - *Simplest:* put `CURSOR_API_KEY=crsr_...` in `secret/.env`.
   - *Recommended for batch / long tasks:* produce an `auth.json` and set
     `CURSOR_AUTH_JSON_PATH=/path/to/auth.json` in `secret/.env`.
2. **Agent preset** — `configs/agents/<name>_cursor.yaml`:
   ```yaml
   harness: cursor_cli
   model: composer-2.5        # omit for "auto" (cursor picks its Composer)
   ```
3. **Reference it from an experiment** and run as usual:
   ```yaml
   # my_exp.yaml
   agents: [configs/agents/cc_composer25_cursor.yaml]
   environment: configs/environments/environment.yaml
   tasks: selected_tasks/my_list.txt
   wall_time_s: 18000         # per-task wall-clock — set HERE, not in the agent
   ```
   ```bash
   uv run python -m ale_run run my_exp.yaml
   ```

The deployer auto-installs the pinned `cursor-agent` if absent (skips if the
pinned version is already baked), writes the MCP/CLI config, materialises the
auth, and launches. No manual VM setup needed.

---

## Authentication

`cursor-agent` reads credentials from a local file in the VM:

| OS | Path the deployer writes / cursor-agent reads |
|---|---|
| Linux | `~/.config/cursor/auth.json` |
| Windows | `%APPDATA%\Cursor\auth.json` |

You supply the credential to the run through `secret/.env` in **one** of three
ways (checked in this order by the lifecycle / deployer):

1. `CURSOR_AUTH_JSON` — the raw auth.json **content** (inline JSON string).
2. `CURSOR_AUTH_JSON_PATH` — a **host path** to an auth.json file. The lifecycle
   reads it and forwards the content into the VM automatically.
3. `CURSOR_API_KEY` — a `crsr_...` key. Used only if neither of the above is set.

### Two modes — and the trade-off

**Mode A — API key only (`CURSOR_API_KEY`)**
- Simplest: one env var, no login flow.
- ✅ Works for short tasks / smoke tests (initial connect authenticates fine;
  the run shows `apiKeySource:"env"`).
- ⚠️ **Fragile on long tasks.** `cursor-agent`'s session connection drops and
  goes into "Reconnecting…" — a known cursor-agent issue, independent of how you
  auth. With an API key **alone there is no refresh token, so it cannot
  re-authenticate on reconnect** and the run dies with
  `Authentication error … try logging out and back in`. The longer the task, the
  more likely it hits an unrecoverable reconnect. (Observed: ~half of long
  compute tasks failed this way.)

**Mode B — auth.json (OAuth) — recommended for batch / long runs**
- `auth.json` holds `accessToken` + `refreshToken` (an `apiKey` field may also be
  present but isn't required), produced by `cursor-agent login`.
- ✅ `cursor-agent` runs the session on `accessToken` and **silently renews it
  with `refreshToken`**, so it survives the reconnects that kill Mode A.
- With a valid `auth.json` you **do not need `CURSOR_API_KEY`** (the OAuth tokens
  are the credential). Tokens are **account-level, not machine-bound**, so an
  auth.json from any machine works on every VM.
- Caveat: `accessToken` expires (≈30 days personal / 24h enterprise-SSO); the
  `refreshToken` renews it, so just use a recently-logged-in auth.json.

### Producing an auth.json — one-time, on your own machine

Do this **on your laptop/workstation** (you need a browser to log in). It's
separate from the VM: the deployer installs cursor-agent *in the VM* for you;
here you install it *locally* only to log in and grab the tokens.

**Step 1 — install cursor-agent locally** (macOS / Linux, same one-liner):
```bash
curl -fsSL https://cursor.com/install | bash
# installs to ~/.local/bin/cursor-agent — make sure it's on PATH:
export PATH="$HOME/.local/bin:$PATH"
cursor-agent --version            # sanity check
```

**Step 2 — log in** (opens a browser; prints a URL to open):
```bash
cursor-agent login
# ✓ Logged in as you@example.com / Authentication tokens stored securely.
```

**Step 3 — get the `auth.json`** (just `accessToken` + `refreshToken`).

- **Linux:** login already wrote it as a plaintext file — nothing to do:
  ```bash
  cat ~/.config/cursor/auth.json   # {"accessToken":"...","refreshToken":"..."}  (0600)
  ```
- **macOS:** login stores tokens in the Keychain; this one-liner writes the same
  `auth.json`:
  ```bash
  mkdir -p ~/.config/cursor
  printf '{"accessToken":"%s","refreshToken":"%s"}\n' \
    "$(security find-generic-password -s cursor-access-token  -a cursor-user -w)" \
    "$(security find-generic-password -s cursor-refresh-token -a cursor-user -w)" \
    > ~/.config/cursor/auth.json && chmod 600 ~/.config/cursor/auth.json
  ```

Tokens are **account-level (not machine-bound)** — generate the file on whatever
machine is easiest and copy it to the host running experiments (`scp …`).

**Step 4 — wire it in** `secret/.env` (point at the file from Step 3):
```bash
export CURSOR_AUTH_JSON_PATH=$HOME/.config/cursor/auth.json
```
The lifecycle reads this file, ships its content into each VM, and the deployer
writes it to the VM's `~/.config/cursor/auth.json`. No `CURSOR_API_KEY` needed.

---

## Config (`CursorCliConfig`)

Agent config carries **only** these deployer-owned knobs:

```yaml
harness: cursor_cli
model: composer-2.5          # default "" = auto (deployer omits --model)
# provider: cursor           # fixed; cursor-agent has no OpenRouter/BYOK path
# disabled_tools: []         # permission deny patterns for cli-config.json
# cursor_version: "..."      # pinned cursor-agent version (install target)
```

| Field | Meaning |
|---|---|
| `model` | Cursor catalog id passed via `--model`. `""` ⇒ auto. Enumerate with `cursor-agent --list-models`. **Tier is part of the id** (e.g. `composer-2.5` = Standard, `composer-2.5-fast` = Fast: same model, ~6× price, lower latency only). |
| `provider` | Fixed `"cursor"` — documents that BYOK/OpenRouter is unsupported. The deployer does not branch on it. |
| `cursor_version` | Pinned version installed when `cursor-agent` is absent or mismatched. |
| `disabled_tools` | Patterns (`Shell(...)`, `Read(...)`, `Mcp(server,tool)`, …) injected into `cli-config.json` `permissions.deny`. |
---

## Install

The deployer handles install automatically:
- Pins `cursor_version` and downloads that exact build from
  `downloads.cursor.com/.../agent-cli-package.tar.gz`, extracting to
  `~/.local/share/cursor-agent/versions/<version>/`.
- **Skips** the download when the pinned version is already baked there
  (Linux: detected on PATH; Windows: detected by version dir).
- Falls back to `curl -fsSL https://cursor.com/install | bash` (latest-only) if
  the pinned download fails.

Manual install (for local testing) is the same `cursor.com/install` one-liner.

---

## Output format (`--output-format stream-json`, NDJSON)

| event | fields |
|---|---|
| `system` / `init` | `model`, `session_id`, `permissionMode`, `apiKeySource` |
| `user` | echoed prompt |
| `assistant` | model text at `message.content[].text` |
| `tool_call` `started`/`completed` | payload at `event.tool_call.<kind>ToolCall` |
| `result` / `success` | final answer + `usage.{inputTokens,outputTokens,cacheReadTokens,cacheWriteTokens}` (camelCase, unlike Anthropic's snake_case) + real `total_cost_usd` |

A long-task auth failure looks like `started → reconnecting → resuming →
{type:error} Authentication error` (see [Mode A](#two-modes--and-the-trade-off)).

---

## Permissions (locked fully open, three layers)

1. CLI flags: `--force --approve-mcps --trust --sandbox disabled`
2. `~/.cursor/cli-config.json`: `approvalMode: "unrestricted"` + `Mcp(*:*)` allow
3. Any project-level `.cursor/cli.json` is deleted if present

`install()` also writes `~/.cursor/mcp.json` (auto-discovered — no CLI flag) so
the CUA MCP bridge is wired without prompting.

---

## VM-side paths

| | Path (Linux) |
|---|---|
| Binary | `~/.local/bin/cursor-agent` → `…/versions/<ver>/cursor-agent` |
| Cursor config | `~/.cursor/` (`mcp.json`, `cli-config.json`) |
| Auth | `~/.config/cursor/auth.json` |

## Package layout

```
ale_run/agents/cursor_cli/
├── __init__.py   — re-exports CursorCliConfig, CursorCliDeployer
├── config.py     — CursorCliConfig (standalone dataclass)
├── deployer.py   — install / launch / parse_artifacts
├── README.md     — this file
└── AGENTS.md     — integration notes, tool classification, known issues
```
