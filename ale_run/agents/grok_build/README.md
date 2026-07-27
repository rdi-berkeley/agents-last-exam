# Grok Build Agent

This harness runs xAI's official Grok Build CLI inside the ALE sandbox. It is
separate from the legacy `grok_cli` harness, which integrates the older
`superagent-ai/grok-cli` product.

## Runtime

- Official package: `@xai-official/grok`
- Default model: `grok-4.5`
- Headless entry point: `grok --prompt-file ... --output-format streaming-json`
- Authentication: `XAI_API_KEY`
- Internal state: isolated under the episode's `GROK_HOME`
- GUI access: ALE's existing CUA MCP bridge, registered as the `cua` server

The package version is pinned in `GrokBuildConfig.cli_version`. The deployer
installs the matching official platform package with npm and verifies both the
package metadata and native binary version. The large native binary is kept
outside the episode work directory so artifact gathering does not copy it.

On Windows, Grok's project cwd is a short per-run path under
`~/.ale-grok-build/cwd/`. Grok percent-encodes that cwd inside its session path;
using the full ALE run id would otherwise exceed the legacy Windows path limit
and prevent session trajectories from being gathered. `GROK_HOME`, transcripts,
and all ALE artifacts remain under the normal episode work directory.

## Headless Behavior

The harness passes:

- `--always-approve` so tool calls do not wait for an interactive user
- `--no-auto-update` so a benchmark run cannot change its own CLI version
- `--sandbox off` because the ALE VM is already the isolation boundary
- `--no-plan` because ALE episodes have no interactive plan-approval client
- `--output-format streaming-json` for incremental output and final usage

`transcript.jsonl` contains `thought`, `text`, `end`, and `error` events. Grok
Build does not emit tool calls on stdout. The harness therefore parses the
run-local session's `chat_history.jsonl` for model messages and tool results,
and uses `updates.jsonl` for tool status and fallback usage metadata.

When Grok exits, the harness also exports the primary session's chat, updates,
events, and MCP image results to fixed top-level files. Those files are hot
artifacts, so ALE's incremental puller performs a final bounded reconciliation
even if the nested session tree cannot be gathered. Large JSONL records are
spooled across range chunks without exposing or truncating partial records.

The final `end` event is the authoritative accounting source. Its
`input_tokens` field is uncached input, while `cache_read_input_tokens` is
recorded separately. Reported cost is persisted only when Grok Build supplies
the complete `total_cost_usd` field.

## Telemetry

Grok Build records native session events for each model loop and tool call. The
harness normalizes them into:

- `telemetry.jsonl`: the session's timestamped native events
- `telemetry_summary.json`: request/session IDs, aggregate API duration and
  usage, per-loop first-token and generation timing, plus native and MCP tool
  durations and outcomes

Unlike Kimi Code's JavaScript client, Grok Build runs its model transport in a
native binary, so Node/Undici OpenTelemetry injection cannot observe its HTTP
trace IDs or status codes. The harness uses Grok's own events rather than
adding a duplicate proxy or modifying upstream.

## MCP And Screenshots

The deployer writes this shape to `$GROK_HOME/config.toml`:

```toml
[mcp_servers.cua]
command = "/path/to/node"
args = ["/path/to/cua_mcp_server/src/index.js"]
env = { CUA_SERVER_URL = "http://127.0.0.1:5000" }
startup_timeout_sec = 60
tool_timeout_sec = 6000
```

Grok Build namespaces MCP tools as `cua__<tool>`. The model discovers and
invokes them through Grok's `search_tool` and `use_tool` wrappers.
The trajectory parser unwraps `use_tool` so ALE records the underlying CUA tool
name and arguments.

These are harness invariants rather than user-facing options. Grok Build's plan
approval is an interactive client flow, so `exit_plan_mode` cannot complete in
a headless episode even when `--always-approve` is set. The harness always
removes `ask_user_question`, `enter_plan_mode`, and `exit_plan_mode`.
`disabled_tools` only adds further exclusions.

MCP image results are stored as data URLs in Grok's chat history or its
run-local MCP spill files. The parser converts them to
`ImageSource(type="base64")`; the framework then writes them under the standard
`screenshots/` directory before serializing the trajectory.

## Custom Models

Set `base_url` to route Grok Build through another supported endpoint. The
harness writes a run-local custom model entry and keeps the key in the process
environment rather than `config.toml` or ALE's gathered `_spec.json`.

```yaml
harness: grok_build
model: kimi-k2.5

config:
  base_url: https://api.moonshot.ai/v1
  api_backend: chat_completions
  api_key: ${env:MOONSHOT_API_KEY}
  context_window: 262144
```

Supported `api_backend` values are `chat_completions`, `responses`, and
`messages`. Custom gateways must emit the selected protocol exactly. For
example, a Responses gateway that adds unknown SSE event types can be rejected
by Grok Build's strict event decoder.

`context_window` is optional custom-model metadata used to schedule automatic
context compaction. When it is omitted (or `null` in a programmatic config), the
harness does not write the field and Grok Build uses its catalog/provider
metadata. Direct xAI catalog models do not need this override.

`api_key` may be supplied directly through `${env:...}`, or `api_key_env` may
name the executor environment variable to read. In both cases the deployer
passes the resolved value through the private `ALE_GROK_BUILD_API_KEY` process
variable named by the generated model's `env_key`; the value is never written
to Grok's TOML.

The custom model credential does not replace authentication for Grok Build's
built-in xAI Imagine tools. `image_gen`, `image_edit`, `image_to_video`, and
`reference_to_video` still require a valid `XAI_API_KEY`. To use a custom model
endpoint and those tools together, provide both credentials; the harness keeps
both process-only.

## Upstream References

- https://docs.x.ai/build/overview
- https://docs.x.ai/build/cli/headless-scripting
- https://docs.x.ai/build/features/mcp-servers
- https://docs.x.ai/build/settings
- https://www.npmjs.com/package/@xai-official/grok
