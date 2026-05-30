# Gemini CLI External Agent

Gemini CLI runs inside the sandbox as a one-shot process:

```text
gemini -p - --output-format stream-json
  -> CUA MCP Server over stdio
  -> sandbox desktop / filesystem
```

The deployer enables the Gemini CLI built-in shell/file tools
(`run_shell_command`, `read_file`, `write_file`, `replace`, `list_directory`,
`glob`, `grep_search`, `read_many_files`, background-process tools) and adds
the CUA MCP tools through `~/.gemini/settings.json`. Only web /
persistent-state / interactive / tracker built-ins are listed in
`disabled_tools` and written to Gemini's `settings.tools.exclude`.

Headless runs use `gemini -p`, so there is no UI that can answer permission
prompts. The deployer writes `~/.gemini/agenthle_policy.toml` and passes
`--approval-mode yolo`; the fork also treats residual non-interactive
`ASK_USER` decisions as allowed for executable tools.

## Fork Requirement for OpenRouter

This deployer uses the `cua-verse/gemini-cli` fork (branch `agenthle`) instead
of the official `@google/gemini-cli` npm package. The fork adds:

- OpenRouter auth detection via `OPENROUTER_API_KEY`
- OpenAI-compatible streaming tool-call conversion, with tool-call fragments
  accumulated across the stream into one complete `functionCall`
- Tool-result linkage via `functionResponse.id` (matching the original
  `tool_call.id`)
- Tool-result forwarding regardless of turn role: gemini-cli packs tool
  results as their own turn with role `user` (Gemini's native convention),
  so the OpenAI converter must emit a `role:"tool"` message for any
  `functionResponse` part, not only `role:"function"` turns. Without this,
  native `read_file`/`write_file` results were silently dropped over
  OpenRouter and the model hallucinated file contents
- Correct tool schema forwarding to OpenRouter
- OpenRouter-safe compression model routing via `OPENROUTER_COMPRESSION_MODEL`

The default `npm_package` is the prebuilt fork release tarball
`v0.38.1-agenthle/google-gemini-cli-0.38.1.tgz` (set in `config.py`).
`github:cua-verse/gemini-cli#agenthle` (build-from-git) is the alternative.
Auto-install happens when `gemini` is not found on PATH.

> WARNING: the tool-result-role fix is currently UNCOMMITTED. It lives only in
> the local working tree of
> `upstream/packages/core/src/core/openRouterContentGenerator.ts`; it is NOT on
> HEAD (`416531f28`) and NOT on `origin/agenthle` — both committed refs still
> have the buggy `&& role === 'function'` guard, and that buggy guard is what
> ships in the published `v0.38.1-agenthle` tarball baked into the ale-kasm
> image. So neither the tarball nor `github:cua-verse/gemini-cli#agenthle`
> carries the fix yet. To deploy it: (1) commit the working-tree change to
> `agenthle` and push; then (2) rebuild + re-release the tarball
> (`npm run bundle` + `npm pack`) or switch `npm_package` to the github ref.
> Until then, native `read_file`/`write_file` results are dropped over
> OpenRouter and the model hallucinates file contents (verified: forced
> read_file smoke scores 0.0 over OpenRouter vs 0.5 on the `google` provider).

## Providers

Routing is **provider-driven** via the `provider` config field — it is NOT
inferred from which API keys happen to be in the environment.

**`openrouter`** (default): Requires `OPENROUTER_API_KEY` in the executor env
(hard error if missing). The deployer sets `OPENROUTER_COMPRESSION_MODEL` from
`config.compression_model` (default `google/gemini-3-flash-preview`) and clears
`GEMINI_API_KEY`/`GOOGLE_API_KEY` so the CLI takes the OpenRouter path. The
model id stays bare (`google/` prefix stripped); the fork maps `gemini-*` to
`google/gemini-*` on the OpenRouter request.

**`google`** (explicit opt-in): Requires `GEMINI_API_KEY` or `GOOGLE_API_KEY`
(hard error if neither is present). Uses Google's native API with the bare
model id; no compression-model routing.

## Config

```python
@dataclass
class GeminiCliConfig:
    model: str = "gemini-3.1-pro-preview"
    provider: str = "openrouter"  # "openrouter" | "google"
    approval_mode: str = "yolo"
    allowed_tools: tuple[str, ...] = _ALLOWED_TOOLS
    disabled_tools: tuple[str, ...] = _DISABLED_TOOLS
    npm_package: str = ".../v0.38.1-agenthle/google-gemini-cli-0.38.1.tgz"
    compression_model: str = "google/gemini-3-flash-preview"
```

## Smoke Test

```bash
uv run python -m ale_run run experiments/smoke_gemini_cli_docker.yaml
```

## Logs

The deployer always requests `stream-json`. The transcript is saved as
`transcript.jsonl` (raw NDJSON) and parsed into trajectory steps via
`parse_artifacts()`.
