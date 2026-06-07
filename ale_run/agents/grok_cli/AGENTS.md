# Grok CLI Integration Notes

> ## ✅ LINUX/OPENROUTER STATUS (verified 2026-06-06)
>
> Linux grok_cli over OpenRouter now works end to end. Verified on a Linux dev
> VM: `demo/tool_smoke` **8/8 (score 1.0)** and `demo/seecheck` **1.0** (reads a
> code off a screenshot). The two historical blockers below are resolved; the
> fix for both was to actually **run the fork bundle** (`bundle_url` is now a
> non-empty default — see below), launched via Bun.
>
> **1. OpenRouter model prefix — FIXED.** `native_to_openrouter_model()` used to
> force an `x-ai/` prefix on any model that didn't already start with `x-ai/`,
> so an OpenRouter slug like `anthropic/claude-sonnet-4-6` became the invalid
> `x-ai/anthropic/claude-sonnet-4-6` → OpenRouter 400 `not a valid model ID`,
> instant crash. Fixed in `config.py`: any model already in `provider/model`
> form (contains `/`) is passed through unchanged; only bare names get the
> `x-ai/` default.
>
> **2. Config did not wire the Linux fork bundle — FIXED (2026-06-06).** The
> root cause of "Linux grok scores 0.0" was a config gap, not (only) a code bug:
> `bundle_url` defaulted to `""`, so the deployer ran the **stock** Linux grok.
> The stock binary ZodErrors on OpenRouter's streamed `tool_call` deltas
> (partial chunks omit `id`/`type`/`function.name`), retries ~900× and times out
> with no output. Windows was already wired to the fork via `win_binary_url`;
> Linux was not. Fixed by defaulting `bundle_url` to the `v0.1.1-agenthle`
> `grok-bundle.js` (`_DEFAULT_BUNDLE_URL` in `config.py`, mirrored in
> `configs/agents/grok_cli.yaml`).
>
> **3. Fork bundle "Fatal: write after end" on background-process tools —
> RESOLVED.** A 2026-05-29 note had this OPEN: with the fork bundle, background
> tools (`bash background:true`, `process_logs`, `process_stop`) crashed the CLI
> mid-run. Re-verified 2026-06-06 with the `v0.1.1-agenthle` bundle launched via
> **Bun** (`bun <bundle>`, not `node`): `demo/tool_smoke` passes 8/8 with zero
> `write after end`. Considered resolved; reopen if it recurs on a real task.

## Source And Fork

- Upstream: `superagent-ai/grok-cli` (closed-source binary releases)
- Fork: `git@github.com:cua-verse/grok-cli.git`
- Branch: `agenthle`
- Current tested commit: `4fe2f86` (Image injection middleware + model info resolution)

The fork contains fixes across `src/grok/client.ts` and `src/agent/agent.ts`:

1. When a custom base URL is detected (`GROK_BASE_URL`), switch from
   `@ai-sdk/xai` to `@ai-sdk/openai` with `compatibility: "compatible"` mode.
   This handles non-standard streaming responses (OpenRouter omits `id`, `type`,
   `function.name` in first tool-call chunk).
2. Force Chat Completions API (`.chat(modelId)`) instead of default Responses
   API for custom endpoints.
3. Skip model name normalization for custom endpoints -- pass model IDs through
   unchanged so provider-prefixed names like `x-ai/grok-3` reach OpenRouter
   as-is.
4. MCP `CallToolResult` serialization fix in `src/agent/agent.ts` (`toToolResult`)
   -- objects with `{content: [...], isError?}` were previously serialized as
   `[object Object]` in NDJSON output.
5. `disabledTools` support in `user-settings.json` -- `filterDisabledTools()`
   removes listed tools before agent startup.
6. **Image injection middleware** (`injectImagesFromToolResults`) -- extracts
   base64 images from MCP tool results, replaces them with `[screenshot]`
   placeholders in tool messages, and injects pending images as user messages
   with proper `file` content type. Prevents 100-300K text token costs per
   screenshot via OpenRouter.
7. **`resolveModelInfoForCustomURL()`** -- reverse-maps OpenRouter model IDs
   (dots in versions like `grok-4.1-fast`) to native IDs (dashes like
   `grok-4-1-fast`) for context window and capability detection.

## Install

Stock binary install (pre-fork):

```bash
# Linux
curl -fsSL https://grok.com/install.sh | bash
```

Fork bundle (required for OpenRouter):

The deployer downloads a pre-built JS bundle from a GitHub Release via the
`bundle_url` config field (now a non-empty default). `install()` downloads the
bundle to `~/.grok/bin/grok-bundle.js` and `_build_argv()` launches it via
`bun grok-bundle.js --prompt ...` (the bundle uses `bun:sqlite`; Bun is
auto-installed) instead of the stock `grok` binary.

To build the bundle from source:

```bash
git clone https://github.com/cua-verse/grok-cli.git
cd grok-cli && git checkout agenthle
bun install && bun build --target=bun --outdir=dist \
  --entry-naming=grok-bundle.js --packages=bundle ./src/index.ts
```

Expected binary path: `~/.grok/bin/grok` (stock), `~/.grok/bin/grok-bundle.js` (fork).
Version command: `grok --version`.
Rollback: remove `grok-bundle.js` and clear `bundle_url` from config.

## CLI Invocation

```bash
# Stock binary
grok --prompt "$PROMPT" --format json --max-tool-rounds 400

# Fork bundle
node ~/.grok/bin/grok-bundle.js --prompt "$PROMPT" --format json --max-tool-rounds 400
```

Required flags:
- `--prompt`: non-interactive one-shot mode
- `--format json`: NDJSON structured output
- `--max-tool-rounds`: tool iteration limit (default 10 is too low)

Required env vars:
- OpenRouter: `OPENROUTER_API_KEY`, `GROK_BASE_URL=https://openrouter.ai/api/v1`
- Direct: `GROK_API_KEY`

## Output Format

NDJSON events, one JSON object per line:

```jsonl
{"type":"step_start","sessionID":"s1","stepNumber":0}
{"type":"text","text":"Let me help you with that."}
{"type":"tool_use","toolCall":{"id":"tc1","function":{"name":"bash","arguments":"{\"command\":\"echo hello\"}"}},"toolResult":{"success":true,"output":"hello"}}
{"type":"step_finish","finishReason":"stop","usage":{"inputTokens":1500,"outputTokens":200,"costUsdTicks":50000}}
{"type":"error","message":"something went wrong"}
```

## Tool Classification

Grok CLI exposes **54 native tools**. Of these, 21 are disabled via
`disabledTools` in `user-settings.json` (written by the deployer from the
`disabled_tools` config field, defaulting to `_DISABLED_TOOLS_OPENROUTER`).
The remaining **33 tools** are available to the agent at runtime. The fork's
`filterDisabledTools()` removes them before the agent starts, so the model
never sees them.

### Disabled Tools (21)

| Category | Tools | Reason |
|----------|-------|--------|
| xAI Responses API | `search_web`, `search_x`, `generate_image`, `generate_video` | Requires xAI-exclusive APIs |
| agent-desktop | `computer_snapshot`, `computer_screenshot`, `computer_click`, `computer_mouse_move`, `computer_type`, `computer_press`, `computer_scroll`, `computer_launch`, `computer_list_windows`, `computer_focus_window`, `computer_wait`, `computer_get` | Requires unpublished binary |
| Payment/wallet | `wallet_info`, `wallet_history`, `fetch_payment_info`, `paid_request` | No wallet on VMs; `paid_request` crashes headless mode |
| LSP | `lsp` | No language server on VMs |

## Known Issue: `needsApproval` Crashes Headless Mode

Any tool with `needsApproval: () => true` (e.g. `paid_request`) will crash the
headless NDJSON stream. The AI SDK emits a `tool-approval-request` event, but
the NDJSON emitter has no handler for it -- falls through to `default: break`,
nobody calls `respondToToolApproval()`, the stream terminates, and the process
exits silently (exit 0).

**Fix**: The fork adds `disabledTools` support in `user-settings.json`. The
deployer writes the `disabled_tools` list from config, so `paid_request`
and other problematic tools are removed before the agent starts.

## Known Issue: Delegate Sub-Agent Model ID

The `task` and `delegate` tools spawn sub-agents using the native model ID
(e.g. `grok-4-1-fast-reasoning`) rather than the OpenRouter-prefixed ID
(`x-ai/grok-4.1-fast`). OpenRouter rejects the native ID with "not a valid
model ID". This happens because the sub-agent creation path uses `this.modelId`
which was already normalized to native form, but doesn't re-apply the
custom-URL passthrough.

**Impact**: Sub-agent tool calls fail, but the parent agent continues normally.
Tasks that rely heavily on delegation may score lower.
