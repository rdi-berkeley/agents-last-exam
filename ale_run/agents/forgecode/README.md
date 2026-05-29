# forgecode (tailcallhq/forge)

The [forgecode](https://github.com/tailcallhq/forgecode) Rust CLI adapted for
the agents-last-exam framework. forgecode is a terminal-native coding agent
with built-in `Read / Write / FsSearch / Patch / Shell / Fetch / ...` tools
that execute directly inside the sandbox via Rust syscalls. The `forge -p
"<prompt>"` mode runs a single task headlessly and exits.

## Architecture

```
agents-last-exam framework                          Sandbox
--------------------------                          -------
BaseAgentDeployer                                   
  install() ──────────────────────────────────────> forge --version
                                                     + write ~/.forge/permissions.yaml
                                                     + write ~/.forge/.forge.toml
  launch() ───────────────────────────────────────> PROMPT="$(cat prompt.txt)"
                                                     forge -p "$PROMPT" \
                                                     --conversation-id <UUID> \
                                                     -C <work_dir>/dump_dir
                                                       |
                                                       +-- tool calls execute in sandbox
                                                           (Read / Write / Shell / ...)
                                                           no Docker, no MCP bridge
  post-run ───────────────────────────────────────> forge conversation dump <UUID>
  parse_artifacts() <── dump.json + transcript.jsonl + stderr.log + exit_code.txt
```

There is **no Docker layer and no CUA MCP bridge**. forge runs unrestricted
in the sandbox (`restricted = false`, permissive `permissions.yaml`,
no `--sandbox`).

## Supported providers

| Provider | Config | API key | Model id format |
|---|---|---|---|
| OpenRouter (preferred) | `provider: openrouter` | `OPENROUTER_API_KEY` | `<vendor>/<model>` (e.g. `anthropic/claude-sonnet-4-6`) |
| Direct (Anthropic) | `provider: direct` | `ANTHROPIC_API_KEY` | `<vendor>/<model>` (e.g. `anthropic/claude-sonnet-4-6`) |
| Direct (OpenAI) | `provider: direct` | `OPENAI_API_KEY` | `<vendor>/<model>` (e.g. `openai/gpt-4o`) |

For OpenRouter, the deployer sets `ANTHROPIC_API_KEY` to the OpenRouter key
and `ANTHROPIC_BASE_URL` to `https://openrouter.ai/api/v1`. It also writes
`[session] provider_id = "open_router"` in `forge.toml`.

## Tool bridge

None. forge runs inside the sandbox and its built-in tools call libc/syscalls
directly.

## Configuration shape

In experiment YAML:

```yaml
agent:
  harness: forgecode
  model: anthropic/claude-sonnet-4-6
  config:
    timeout_s: 600
    temperature: 0.7
    provider: openrouter
```

| Field | Required | Meaning |
|---|---:|---|
| `harness` | yes | Must be `forgecode` |
| `model` | yes | Model id (OpenRouter-native `<vendor>/<model>` form) |
| `config.timeout_s` | no | Wall-clock budget (default 600s) |
| `config.temperature` | no | Sampling temperature (default 0.7) |
| `config.provider` | no | `openrouter` (default) or `direct` |

## Output artifacts

Written to `<work_dir>/`:

| File | Purpose |
|---|---|
| `dump.json` | `ConversationDump` JSON -- canonical record |
| `transcript.jsonl` | Raw stdout capture from `forge -p` |
| `stderr.log` | Stderr capture |
| `exit_code.txt` | Process exit code |
| `dump_dir/` | CWD during the run; auto_dump timestamped JSONs land here |

## OS support

Linux only (sandbox executor). forge itself supports macOS and Windows,
but the ALE framework targets Linux sandboxes for terminal-native agents.

## See also

- `AGENTS.md` (this dir) -- install details, tool matrix, output schema.
- forgecode upstream: <https://github.com/tailcallhq/forgecode>
- forgecode releases: <https://github.com/anysphere/forgecode/releases/>
