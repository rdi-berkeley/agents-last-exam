# terminus_2 (harbor) External Integration

The [harbor](https://github.com/harbor-framework/harbor) framework's
**terminus_2** agent, adapted for the ALE external runner. terminus_2 is a
tmux-driven, ReAct-style agent: each turn the LLM emits
`{analysis, plan, commands[]}` JSON; the agent feeds keystrokes into a tmux
pane on the sandbox it runs inside and feeds the resulting pane output back
into the next turn. It terminates on a double-confirmed `task_complete`
signal or on the outer timeout.

ALE runs it from the `cua-verse/harbor` fork on the `agenthle` branch, which
ships a thin `harbor-terminus2` CLI shim plus a `LocalShellEnvironment` so
the agent's tmux loop drives the same sandbox the CLI runs in.

## Operating Systems

**Linux only.** terminus_2's TmuxSession requires tmux + asciinema and a
POSIX environment. `install()` raises `NotImplementedError` on Windows.

| OS | Status |
|---|---|
| Linux (Ubuntu 22.04) | supported |
| Windows native | not supported (tmux/asciinema) |
| macOS | not used in benchmark |

## Architecture

```
ALE RUNNER (host)                    SANDBOX (ale-kasm / ubuntu22)
Terminus2Deployer
    install()  ──────────────→  apt install tmux asciinema
                                uv tool install harbor (fork @ agenthle)
                                  → ~/.local/bin/harbor-terminus2
    launch()   ──────────────→  harbor-terminus2 \
                                    --prompt-file prompt.txt --model <m> \
                                    --logs-dir <wd>/logs --temperature 0.7 \
                                    [--max-turns N] [--no-recording]
                                  └─ Terminus2 (Python ReAct loop)
                                     └─ LocalShellEnvironment
                                        └─ subprocess + tmux on this sandbox
    parse_artifacts() ←────────  logs/agent/trajectory.json + recording.cast
```

There is **no Docker layer and no CUA MCP bridge**. The single conceptual
action is `bash_command` = `tmux send-keys` of `{keystrokes, duration_sec}`.

## Supported Providers

YAML always carries the OpenRouter-native `<vendor>/<model>` id; only the
`provider:` field flips between modes. For openrouter mode the deployer
re-attaches the `openrouter/` prefix internally before invoking LiteLLM.

| `provider:` | API key | Model id passed to LiteLLM |
|---|---|---|
| `openrouter` (default) | `OPENROUTER_API_KEY` | `openrouter/<vendor>/<model>` |
| `direct` (anthropic) | `ANTHROPIC_API_KEY` | `<vendor>/<model>` (as-is) |
| `direct` (openai) | `OPENAI_API_KEY` | `<vendor>/<model>` (as-is) |

The deployer injects exactly one provider key into the launched process's
environment; it is never written to any file gathered to the host.

## Configuration

```yaml
agent:
  harness: terminus_2
  model: anthropic/claude-sonnet-4.6   # OpenRouter-native id
  config:
    timeout_s: 3600                    # outer wall-clock budget
    max_turns: 100000                  # cap LLM episodes
    provider: openrouter               # or "direct"
    record_terminal_session: true      # set false to skip asciinema
    api_base: null                     # optional LiteLLM base url override
    temperature: 0.7
```

## Output artifacts

Gathered under `<variant_dir>/debug/agent/`:

| File | Purpose |
|---|---|
| `logs/agent/trajectory.json` | ATIF trajectory; the canonical record |
| `logs/agent/recording.cast` | asciinema replay of the tmux pane (when enabled) |
| `logs/agent/terminus_2.pane` | final tmux pane contents |
| `logs/agent/agent_context.json` | LiteLLM token/cost totals |
| `stdout.log` | combined stdout/stderr of `harbor-terminus2` |
| `exit_code.txt` | process exit code |

`parse_artifacts` reads `logs/agent/trajectory.json` and converts it into ALE
trajectory steps.

## Smoke Test

```bash
uv run python -m ale_run run experiments/smoke_terminus_2_docker.yaml
```

## See Also

- Implementation notes: [AGENTS.md](AGENTS.md)
- Harbor fork: <https://github.com/cua-verse/harbor> (branch `agenthle`)
- Upstream: <https://github.com/harbor-framework/harbor>
