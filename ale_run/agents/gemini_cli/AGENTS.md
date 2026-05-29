# Gemini CLI Integration Notes

## Source And Fork

- Upstream fork: `git@github.com:cua-verse/gemini-cli.git`
- Branch: `agenthle`
- npm install spec: `github:cua-verse/gemini-cli#agenthle`

The fork is based on Gemini CLI v0.38.1. OpenRouter support comes from the
upstream OpenRouter PR adapted to this release. ALE-specific fixes on top:

- Accumulate streaming OpenAI tool-call chunks before emitting a Gemini
  `functionCall`.
- Forward `parametersJsonSchema` to OpenRouter tools.
- Preserve `functionResponse.id` as OpenAI `tool_call_id`.
- Route `utility_compressor` requests to
  `google/gemini-3-flash-preview` by default.
- In non-interactive `yolo` mode, allow residual `ASK_USER` policy decisions
  for executable tools instead of throwing
  `requires user confirmation, which is not supported in non-interactive mode`.
  The `ask_user` tool itself is still denied by policy.

## Install

The deployer auto-installs via `npm install -g github:cua-verse/gemini-cli#agenthle`
when the `gemini` binary is not found on PATH. This is handled by
`GeminiCliDeployer.install()` in `deployer.py`.

For environments where GitHub source installs are unreliable (e.g. CPU-free
Ubuntu VMs hitting `ECONNRESET`), build a local tarball from the fork:

```bash
cd gemini-cli   # the cua-verse/gemini-cli checkout, branch agenthle
npm ci
npm run bundle
npm pack
```

Then place the resulting `.tgz` on PATH or pre-install before launching
the deployer.

## Runtime

The deployer launches:

```bash
gemini -p - --model <model> --output-format stream-json --approval-mode yolo --allowed-tools=...
```

For OpenRouter runs the executor env must contain `OPENROUTER_API_KEY`. The
deployer detects this and sets `OPENROUTER_COMPRESSION_MODEL` from
`config.compression_model` (default `google/gemini-3-flash-preview`) so
auto-compression is explicitly routed to an OpenRouter-supported model.

For direct Google API runs, set `GEMINI_API_KEY` in the executor env instead.

Because `gemini -p` is non-interactive, any tool policy result of `ASK_USER`
turns into a hard failure. The deployer therefore writes
`~/.gemini/agenthle_policy.toml` and references it from `settings.json`
`policyPaths`. The policy allows tools in `yolo` mode with
`allowRedirection = true` and denies `ask_user`.

## Tool Surface

Gemini CLI exposes built-in tools plus MCP-discovered tools. The deployer keeps
the VM-local built-ins needed for benchmark tasks and disables provider-side,
interactive, persistent-state, and non-benchmark tools via
`settings.tools.exclude`.

Kept Gemini built-ins:

| Tool | Classification | Notes |
|---|---|---|
| `run_shell_command` | supported | VM shell execution; required by demo tasks. |
| `list_background_processes`, `read_background_output` | supported | Gemini shell background helpers. |
| `read_file`, `write_file`, `list_directory` | supported | VM filesystem access. |
| `glob`, `grep_search`, `read_many_files` | supported | VM-local file discovery/search. |
| `replace` | supported | VM-local edit tool. |

Disabled Gemini built-ins:

| Tool | Reason |
|---|---|
| `google_web_search`, `web_fetch` | Provider-side web access, outside benchmark scope. |
| `save_memory`, `activate_skill`, `get_internal_docs`, `write_todos` | Persistent/local Gemini CLI state not backed by ALE task contract. |
| `ask_user` | Interactive; explicitly denied for headless runs. |
| `enter_plan_mode`, `exit_plan_mode`, `update_topic`, `complete_task` | Session-control tools not part of ALE runner contract. |
| `tracker_create_task`, `tracker_update_task`, `tracker_get_task`, `tracker_list_tasks`, `tracker_add_dependency`, `tracker_visualize` | Tracker backend is not provisioned on benchmark environments. |

CUA MCP tools are not disabled. Gemini sees them as MCP FQNs from the `cua`
server. The bridge exposes keyboard actions, pointer actions, `wait`,
`screenshot`, and `cursor_position`.

## Compact Pitfall

Gemini CLI auto-compression is a normal chat request, not a special compact
endpoint. On OpenRouter, compression must use a valid model. The fork routes
`utility_compressor` requests to `OPENROUTER_COMPRESSION_MODEL`
(`google/gemini-3-flash-preview` by default). Without the fork, compression
attempts may fail with `No endpoints found` on OpenRouter.
