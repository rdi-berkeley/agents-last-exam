# Adding your own agent

Every agent in ALE is a small Python module that implements
[`BaseAgentDeployer`](../ale_run/base_interface/agent_deployer.py):

```python
class BaseAgentDeployer(abc.ABC):
    default_executor: ClassVar[str]
    supported_executors: ClassVar[frozenset[str]]
    hot_artifacts: ClassVar[tuple[str, ...]] = ()

    async def install(self) -> None: ...
    async def launch(self, prompt: str) -> AgentRunResult: ...

    @classmethod
    def parse_artifacts(cls, *, work_dir, config, run_result, builder): ...
```

`install` stages prerequisites (probes for the CLI binary, writes any
config files), `launch` spawns the agent and waits for it to finish,
and `parse_artifacts` converts the on-disk transcript into an ATIF
`Trajectory`. All substrate I/O goes through `self.executor` — the
deployer itself is agnostic to whether the agent runs on a VM, the
host, or in a container.

---

## Pre-installed agents

Registered in [`ale_run/orchestration/factory.py`](../ale_run/orchestration/factory.py)
under `_AGENT_FQNS`:

| Shortcut | Deployer | Executor | Notes |
|---|---|---|---|
| `claude_code` | [`ClaudeCodeDeployer`](../ale_run/agents/claude_code/deployer.py) | `sandbox` | Wraps the `@anthropic-ai/claude-code` CLI baked into the image |
| `ale_claw` | [`AleClawDeployer`](../ale_run/agents/ale_claw/deployer.py) | `local` | In-tree OpenClaw harness; runs on the framework host |

Reference both files when authoring your own — they cover the two main
deployment shapes.

---

## Three deployer flavors

The `executor` your deployer targets determines where the agent code
physically runs. Pick one based on the agent's runtime requirements.

### 1. Sandbox-resident (CLI baked into the image)

The agent CLI lives inside the VM image. Your deployer's `install()`
just probes the binary and writes any config; `launch()` spawns it
detached on the VM and polls a done-marker file.

**Use when:** your agent is a packaged CLI (Node, Go, Rust, Python
script with deps) that's easier to bake into the image than to install
at runtime.

**Reference:** [`ale_run/agents/claude_code/deployer.py`](../ale_run/agents/claude_code/deployer.py).

```python
class MyAgentDeployer(BaseAgentDeployer):
    default_executor = "sandbox"
    supported_executors = frozenset({"sandbox"})
    hot_artifacts = ("transcript.jsonl", "stderr.log")

    async def install(self) -> None:
        # 1. discover the binary on the VM
        # 2. probe --version
        # 3. write any config files via self.executor.write_file
        ...

    async def launch(self, prompt: str) -> AgentRunResult:
        # build a runner script (bash on linux, ps1 on windows)
        # spawn via self.executor.spawn_detached
        # poll self.executor.wait_marker(done_marker, pid, timeout)
        ...
```

**Image baking:** add your CLI to the image build script that produces
`ale-unified-v1` (or a fork). The `cua-server` already on the
image will dispatch your binary the same way it does `claude`.

### 2. Host-side harness (Python on the framework host)

The agent code runs in the same Python process as the runner, driving
the VM through `cb.RemoteDesktopSession` RPC. No CLI to install — the
deployer just `await`s the harness's `run()` and writes whatever logs
it produces.

**Use when:** your agent is a Python harness with custom planning,
tool-use, or model-routing logic; you want zero per-run boot cost on
the agent side; or you need the agent to inspect/manipulate the
framework's own state.

**Reference:** [`ale_run/agents/ale_claw/deployer.py`](../ale_run/agents/ale_claw/deployer.py).

```python
class MyHarnessDeployer(BaseAgentDeployer):
    default_executor = "local"
    supported_executors = frozenset({"local"})

    async def install(self) -> None:
        # nothing to install — the harness is imported, not spawned
        ...

    async def launch(self, prompt: str) -> AgentRunResult:
        # session = self.executor.env.session
        # result = await my_harness.run(prompt, session, self.config)
        # write transcript to self.executor.work_dir
        ...
```

API keys come from shell env (or `secret/.env`) — see how
`AleClawDeployer` reads `OPENROUTER_API_KEY` / `ANTHROPIC_API_KEY` via
litellm.

### 3. Docker wrapper (TODO)

The agent runs inside a host-side container; the deployer dispatches
via `docker exec`.

> **Status: stub.** [`ale_run/executors/docker.py`](../ale_run/executors/docker.py)
> declares the class but all methods raise `NotImplementedError`. Not
> usable today.

Intended shape: the deployer's `install()` would pull the image and
`docker run -d` a long-running container; `launch()` would
`docker exec` the agent inside it. Useful for agents with messy
system-level dependencies that you don't want to bake into the VM
image.

---

## Registering a new agent

Add an entry to [`ale_run/orchestration/factory.py`](../ale_run/orchestration/factory.py)
`_AGENT_FQNS`:

```python
_AGENT_FQNS: dict[str, str] = {
    "claude_code": "ale_run.agents.claude_code.deployer.ClaudeCodeDeployer",
    "ale_claw":    "ale_run.agents.ale_claw.deployer.AleClawDeployer",
    "my_agent":    "ale_run.agents.my_agent.deployer.MyAgentDeployer",
}
```

The factory expects the matching config class to live next to the
deployer at the same package, named `<DeployerStem>Config`:

```
ale_run/agents/my_agent/
├── __init__.py
├── deployer.py     class MyAgentDeployer(BaseAgentDeployer)
└── config.py       class MyAgentConfig          # standalone dataclass
```

Ship your agent's defaults as a preset file under `configs/agents/`. The
preset is a complete agent definition — `harness:`, `model:`, and a
`config:` block holding every agent/deployer knob:

```yaml
# configs/agents/my_agent.yaml
harness: my_agent
model: anthropic/claude-sonnet-4-6
config:
  max_turns: 30
```

Then wire it into an experiment by path. `agents:` is a list (list
several to run the agent matrix); the preset's `id` defaults to its
filename stem:

```yaml
agents:
  - configs/agents/my_agent.yaml
```

---

## What to log

ALE's eval pipeline ingests an ATIF `Trajectory`
([`ale_run/base_interface/trajectory.py`](../ale_run/base_interface/trajectory.py)).
Your `parse_artifacts` should:

- emit one `Step` per agent / tool / observation turn
- attach token usage to each step via `StepMetrics` (input/output/cache)
- record tool calls (`ToolCall`) and their results (`ToolResult`) so
  evaluators can grade tool use, not just final answers

If the run failed mid-stream, emit a single `source="system"` step
explaining the gap and return cleanly — don't raise from
`parse_artifacts`.

---

## Cost / safety knobs

Standard `BaseAgentConfig` fields available to every deployer:

- `model` — LLM id (`claude-opus-4-7`, `gpt-5`, etc.)
- `max_turns` — agent-turn cap
- `timeout_s` — wall-clock budget for the whole episode (incl. eval)
- `save_screenshots` — hint for vision-capable agents
- `api_keys` — explicit name→value bag (never auto-read from `os.environ`)

Add agent-specific fields on your `<X>Config` subclass — see
[`ClaudeCodeConfig`](../ale_run/agents/claude_code/config.py) (extends
with `max_budget_usd`, `disabled_tools`, `cli_version`, `base_url`).
