# CAR (Common Agent Runtime)

This deployer runs the [Common Agent Runtime](https://github.com/Parslee-ai/car-releases)
as the system under test on Agents' Last Exam. CAR is a deterministic execution
layer: the model proposes actions, and the runtime validates (preconditions,
policy, dependencies) and executes them. The benchmark measures CAR driving a
real OS sandbox over a long-horizon task.

## How it works

CAR runs **out-of-sandbox** (host-side, like `ale_claw`). The deployer:

1. Installs the `vm` (shell/fs) and, optionally, `cua` (GUI) stdio MCP bridges on
   the host (the same `_assets` bridges the other native agents use). Each bridge
   reaches the eval VM's cua-server via `CUA_SERVER_URL`.
2. Writes an MCP config and subprocesses the CAR headless runner:
   `car run-task --goal-file ... --mcp-config ... --transcript ... --model ...`.
3. CAR connects those bridges as tools through its connector manager, so every
   tool call is routed through CAR's validator / policy / eventlog, then drives
   its own propose -> validate -> execute loop until a done signal (a model turn
   with no tool calls) or `--max-turns`.
4. The runner writes a JSONL **transcript** (full per-turn content) which the
   deployer converts to an ATIF trajectory in `parse_artifacts`.

Why a transcript and not CAR's eventlog: CAR's engine eventlog is metadata-only
(action ids + durations, not tool names / parameters / outputs), so the runner
emits a separate content-rich transcript.

## Requirements

- The `car` CLI binary on PATH (or set `config.car_bin`). It must support
  `car run-task` (CAR >= the version that added the headless runner).
- An LLM API key in the operator's shell env, read by CAR directly:
  `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, or `GOOGLE_API_KEY`.
- A pinned `model:` (CAR's own catalog id, e.g. `claude-sonnet-4-6` — **not** the
  LiteLLM `provider/model` form the other harnesses use).

## Running it

```yaml
harness: car
model: claude-sonnet-4-6
config:
  max_turns: 100
  gui: true
```

```bash
export ANTHROPIC_API_KEY=...
uv run python -m ale_run run experiments/my_experiment.yaml   # agents: [ configs/agents/car.yaml ]
```

## The config knobs

The full surface is in `config.py`; the ones most users touch:

- `model`: CAR catalog model id (required)
- `max_turns`: hard cap on CAR's agent loop
- `car_bin`: path to the `car` binary if not on PATH
- `gui`: wire the cua GUI bridge in addition to the vm bridge
- `eventlog`: also write CAR's (metadata-only) engine journal for debugging

## The transcript schema

`car run-task` emits newline-delimited JSON, discriminated by `type`:

| `type`        | fields |
|---------------|--------|
| `run_start`   | `goal, model, max_turns, servers, tool_count` |
| `agent_turn`  | `turn, text, tool_calls:[{id,name,arguments}], usage:{input_tokens,output_tokens}` |
| `observation` | `turn, results:[{tool_call_id, content, is_error}]` |
| `run_end`     | `status, turns, answer, error` |

`transcript_to_trajectory.py` maps `agent_turn` -> `source="agent"` steps and
`observation` -> `source="environment"` steps; `run_start` / `run_end` land on
`trajectory.extra["car"]`.

## Where to read the code

- `deployer.py`: ALE entry point (install / launch / parse_artifacts)
- `config.py`: runtime knobs
- `transcript_to_trajectory.py`: transcript -> ATIF translator

The headless runner itself ships in the `car` CLI binary (the `car run-task`
subcommand), distributed via
[Parslee-ai/car-releases](https://github.com/Parslee-ai/car-releases).
