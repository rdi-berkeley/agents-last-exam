# DeepSeek Harness

This deployer uses DeepSeek Harness's official Python SDK and bundled JSON-RPC runtime. It does not start the Web UI. The SDK owns one unattended session, streams canonical session notifications to `transcript.jsonl`, and returns after the whole agent becomes idle.

## Why the SDK entry is used

Upstream exposes two headless paths. `dsh --profile headless "task"` is a one-shot CLI whose stdout contains only the final response. `deepseek-harness-sdk` is the automation API: it returns the finish reason and the root session's canonical events, while its notification stream also includes descendants. ALE uses the SDK because these events can be converted to trajectory messages, tool calls, tool results, token usage, errors, and subagent metadata without scraping the browser or compressed session storage.

The version is pinned by `DeepSeekHarnessConfig.sdk_version`. Installation also brings in the exact same-version `deepseek-harness-runtime-bin` platform wheel, which contains a standalone executable and default Cordis composition. Node.js is not required.

## Configuration and authentication

| Field | Default | Purpose |
|---|---|---|
| `model` | `deepseek-v4-flash` | Model id passed in the SDK initialize request. |
| `provider` | `deepseek-official` | Provider route. The bundled composition registers this route. |
| `api_key` | `null` | Literal key. ALE keeps it out of gathered specs. |
| `api_key_env` | `DEEPSEEK_API_KEY` | Executor environment variable read when `api_key` is null. |
| `base_url` | `null` | Optional OpenAI-compatible API root exported as `DEEPSEEK_BASE_URL`. Null uses the official endpoint. |
| `max_tokens` | `null` | Optional positive output-token cap for the root session and in-process descendants. |
| `system_prompt` | `null` | Optional deployment persona. Null uses the bundled coding-agent default. |
| `sdk_version` | `0.1.0rc6` | Exact PyPI SDK/runtime release installed in the sandbox. |

The deployer removes ambient `DSH_CORDIS_CONFIG` and `DSH_RUNTIME_MODE` so an experiment always uses the pinned bundled runtime. Session storage is isolated under the episode work directory. Upstream telemetry is disabled.

## Permissions

The bundled SDK composition is intentionally unattended. It mounts local bash and filesystem tools directly and does not mount the interactive approval or sandbox-policy plugins used by the Web profile. ALE therefore exposes no approval or permission-mode config for this harness. The security boundary is the disposable ALE sandbox VM, equivalent to Claude Code's permission bypass and Codex's danger-full-access mode inside the same outer isolation.

## Current limits

- ALE support is Linux sandbox only. Upstream publishes manylinux x86-64 and arm64 runtime wheels, plus macOS arm64, but no Windows runtime wheel.
- The bundled SDK runtime does not include `@deepseek-ai/dsh-mcp-client`, so it cannot load ALE's CUA MCP bridge. This deployer is suitable for code and terminal tasks, not GUI tasks.
- The full npm headless profile does include the MCP client, but upstream currently projects non-text MCP results such as screenshots to placeholders in model-visible history. Adding the CUA server there would not provide reliable visual control.
- The raw transcript retains root and descendant notifications. ALE's normalized trajectory currently projects the root session and records the descendant count; the complete child event streams remain available in `transcript.jsonl`.
