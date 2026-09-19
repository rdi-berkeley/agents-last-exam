# Antigravity CLI Integration Notes

## Source

- Binary: `agy` — Google's Antigravity CLI (successor to Gemini CLI), a closed
  native Go binary. **No fork** (it cannot be patched like the JS gemini-cli).
- Install: official installer `https://antigravity.google/cli/install.sh` drops
  `~/.local/bin/agy`. Pinnable tarball via the updater manifest
  (`.../manifests/linux_amd64.json` → `storage.googleapis.com/antigravity-public/...`).
  The current ALE pin is `1.1.25`; its official URL and checksum are retained in
  the deployer so a newer latest-only manifest does not break reproducible installs.
- Auth: this integration uses Google OAuth via the host-login → token-file →
  in-sandbox-injection flow. `agy` also supports Gemini API keys; OpenRouter's
  OpenAI-compatible endpoint is not wire-compatible with agy's native Gemini
  requests without a translation proxy.

## Install

`AntigravityCliDeployer.install()` probes `agy --version` and installs the exact
configured release from Google's updater manifest when missing or mismatched.
It verifies the manifest SHA-512, then writes the OAuth credential, CUA MCP
config, and CLI policy settings. Linux targets `~/.local/bin/agy`; Windows
targets `%LOCALAPPDATA%\agy\bin\agy.exe`.

## Runtime

The deployer launches:

```bash
agy --print="<prompt>" --model "<display name>" --output-format stream-json \
  --dangerously-skip-permissions --add-dir <task_data_root>
```

- `--model` takes the `agy models` display names verbatim (e.g.
  `Gemini 3.7 Flash (High)`, `Claude Sonnet 4.6 (Thinking)`, `GPT-OSS 120B (Medium)`).
- Streaming JSON supplies per-generation usage, per-tool duration, and the final
  response in `transcript.jsonl`. ALE normalizes it into ATIF and one
  `metrics_summary.json` artifact whose `api_calls` shape matches the Codex and
  Claude telemetry summaries as far as agy's native fields permit.
- The deployer pins agy's native `--log-file` to `agy_cli.log` inside each task's
  work directory. This avoids the global `cli.log` symlink race under concurrent
  runs and ensures the native log is pulled with the task.
- Native `agy` omits cache-write tokens, request cost, and per-transport-retry
  tokens/latency. Availability flags retain those gaps.
- Auth at launch is **silent**: `agy` reads the injected
  `~/.gemini/antigravity-cli/antigravity-oauth-token` and refreshes it itself.

## Tool Surface

`agy` exposes a rich native toolset **plus** the CUA MCP tools. Unlike
gemini-cli, the CUA bridge must be declared in `agy`'s **native**
`~/.gemini/config/mcp_config.json` (NOT `settings.json`), or no GUI tools load.

The matrix below is the agent's own self-report from `demo/tool_smoke` on
`ale-ubuntu22` (Gemini 3.1 Pro): **36 tools identified, 33 passed, 0 failed,
3 untested**.

### Native `agy` tools (19 — all exercised, all passed)

| Tool | Classification | Notes |
|---|---|---|
| `run_command` | supported | VM shell execution. |
| `view_file`, `write_to_file` | supported | VM filesystem read/write. |
| `replace_file_content`, `multi_replace_file_content` | supported | In-place edits. |
| `list_dir`, `grep_search` | supported | File discovery / content search. |
| `search_web`, `read_url_content` | supported | Web access (internet is allowed). |
| `generate_image` | supported | Image generation. |
| `call_mcp_tool`, `list_resources`, `list_permissions` | supported | MCP meta / introspection. |
| `define_subagent`, `invoke_subagent`, `manage_subagents` | supported | Sub-agent orchestration. |
| `manage_task`, `schedule` | supported | Task list / scheduled work. |
| `send_message` | supported | Agent message channel. |

### CUA MCP GUI tools (14 — all exercised, all passed)

| Tool | Notes |
|---|---|
| `screenshot` | Desktop capture — the GUI→model image path (proven by `demo/seecheck`). |
| `click`, `mouse_move`, `mouse_down`, `mouse_up`, `drag` | Pointer actions. |
| `key`, `key_down`, `key_up`, `hold_key`, `type` | Keyboard actions. |
| `scroll`, `wait`, `cursor_position` | Scroll / pause / pointer query. |

> The CUA action tools require real arguments (e.g. `mouse_move` needs a
> `coordinate`). During the smoke test `agy`'s first degenerate probe calls
> returned `MCP -32602 Invalid arguments`; it then retried with proper args and
> all passed — so the errors are agent-side, not a bridge incompatibility.

### Untested (3)

| Tool | Reason |
|---|---|
| `ask_permission`, `ask_question` | Interactive — block headless; `agy` self-skips them. |
| `read_resource` | No MCP resources are exposed on the `cua` server. |

### Headless permissions

The deployer writes `toolPermission: always-proceed` and
`artifactReviewPolicy: always-proceed` to agy's CLI settings and launches with
`--dangerously-skip-permissions`. This avoids human-review prompts during a
task. The previous `tools.exclude` setting was removed because it belonged to a
different Gemini settings schema and was not enforced by `agy`.

## Validation (OS × provider)

| Task | Linux / QEMU | Linux / docker | Linux / gcloud | Windows / gcloud |
|---|---|---|---|---|
| `demo/seecheck` (GUI vision) | **1.0** (agy 1.1.25, Claude Sonnet 4.6 Thinking) | **1.0** | **1.0** | — |
| `demo/seecheck_win` (GUI vision) | n/a | n/a | n/a | **1.0** (3/3, with the cua mitigation) |
| `demo/tool_smoke` | — | **0.92** (33/36) | **0.92** (33/36) | n/a |
| `demo/tool_smoke_win` | n/a | n/a | n/a | **0.92** (34/37) — all 14 cua + native pass, incl. `grep_search` |

`demo/tool_smoke_win` (Claude Sonnet 4.6): all 14 cua GUI tools pass + the native
tools, including **`grep_search` (verified passing end-to-end)** — `install()`
puts Git-for-Windows' GNU grep (`…\Git\usr\bin\grep.exe`, baked into ale-win10)
on PATH (`grep (GNU grep) 3.0`). The remaining non-passes are expected and vary
by run: `read_resource` (cua exposes no MCP resources), `ask_question`
(interactive), `generate_image` (separate image-model quota), or a stray
`command_status`/`cua_scroll` model test-artifact. Note: Sonnet invokes the
cua tools via the `call_mcp_tool` wrapper, so they appear as
`call_mcp_tool__cua__screenshot` etc. (Gemini promotes them to bare names) — both
work. The thinking model is slow over 38 tools, so the run hit the 30-min wall
(`status=timeout`, score still real).

gcloud uses the operator's **active gcloud account** (compute access) when
`GCP_SA_KEY` is unset/missing — `gcloud_sa_key_path()` returns None and the
provider falls back to it. `output_path: local` needs no GCS key.

## Windows

The deployer supports Windows: it installs `agy.exe` via `install.ps1` into
`%LOCALAPPDATA%\agy\bin`, writes the OAuth token + MCP config under
`%USERPROFILE%\.gemini` (the Linux-generated token authenticates fine on
Windows), and launches `agy.exe --print="<prompt>"`. **agy's native tools work**
(`run_command`, file tools, web, etc.). `grep_search` shells out to `grep`, so
the deployer adds Git-for-Windows' GNU grep to `PATH` (or installs a BusyBox
fallback when Git is absent).

### cua GUI on Windows — an intermittent startup race (mitigated)

The CUA MCP tools loaded only **intermittently** on Windows (≈1 run in 4) — some
runs registered and used all 14, others registered none. Everything the deployer
controls is correct (verified on a live VM: `mcp_config.json` with the right
`node.exe`/`index.js` paths, the bridge, and its `node_modules` are all present;
the `GeminiDir "...not absolute, falling back to default"` log line is a red
herring — it appears even when cua works). The variance is a **race on the FIRST
agy run on a fresh VM**: agy's config-migration + auto-updater + a cold `node`
start (slow on Windows — Defender scan + ESM module load of the MCP SDK) race
with agy's MCP tool-discovery window, so cua sometimes isn't registered in time.

The deployer mitigates this in `install()` (all steps cost **no model quota**):

1. **Pre-warm the node bridge** — spawn it once so its modules are cached /
   Defender-scanned, making agy's spawn at launch fast.
2. **Prime agy** with `agy models` (a metadata call, *not* a `-p` generation
   turn) so first-run config-migration + the auto-updater are already done.
3. **Pass `--gemini_dir=<absolute>`** so config discovery is deterministic.

The deployer passes `--log-file=work_dir/agy_cli.log` and incrementally pulls it
as a hot artifact to keep this diagnosable. **Validated:** with the mitigation,
`demo/seecheck_win` (GUI vision) passes **3/3 = 1.0** on fresh ale-win10 gcloud
VMs — so cua now loads reliably on Windows.

### Account / quota note (important)

agy's quota is **per Google account AND per model**. If runs 429 with
*"Individual quota reached — please upgrade your subscription"*, the **active
account is a free tier**, not the intended plan. Check the active account in
`~/.gemini/google_accounts.json` (there is no agy CLI quota command; the web
Settings → Models tab shows usage). A different model can have separate quota —
e.g. when Gemini was exhausted, `Claude Sonnet 4.6 (Thinking)` still worked; the
Windows validation above used it. To switch accounts, delete
`~/.gemini/antigravity-cli/antigravity-oauth-token` and re-run the host login.

## Quota

Auth/routing is the operator's Google plan, not OpenRouter. The **free tier has
a limited rolling quota**: a normal task or a single ~36-tool smoke run
completes, but back-to-back heavy runs can exhaust it
(`RESOURCE_EXHAUSTED (429): Individual quota reached`, multi-day reset). Light
calls keep working while the heavy budget is depleted.
