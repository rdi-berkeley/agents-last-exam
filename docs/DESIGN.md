# `ale` architecture

## One paragraph

`agent-last-exam` (`ale`) is the upper-layer-only replacement for the
agenthle orchestration code. Tasks stay in their current format
(``main.py`` with cua-bench decorators, ``DesktopSession`` as the VM RPC
surface, ``evaluate()`` returning a score). The framework above tasks is
rebuilt against OpenEnv's ``Environment`` abstraction: a single
``AgenthleEnv`` consumes any task at ``reset()`` time, delegates VM
lifecycle to a ``Provider``, and reports the task's score directly as
``observation.reward``. No new Task base class. No new Computer/session
wrapper. No Rubric layer.

## Layers

```
       AgenthleEnv  (OpenEnv Environment; one class for the whole benchmark)
              │
              ├── LoadedTask   (file-based: tasks/<path>/main.py + task_card.json)
              │       └── start_fn / evaluate_fn  (the existing agenthle decorator-tagged functions)
              │
              ├── Provider  (acquire / release / heartbeat / cancel_external)
              │       └── VMHandle
              │
              └── DesktopSession  (cua-bench Protocol; unchanged)
```

## Concrete contract

### What a task file looks like (unchanged from agenthle)

```python
# tasks/demo/hello/main.py
import cua_bench as cb
from tasks.linux_runtime import LinuxTaskConfig
from dataclasses import dataclass

@dataclass
class TaskConfig(LinuxTaskConfig):
    DOMAIN_NAME = "demo"
    TASK_NAME = "hello"
    # ... properties, to_metadata(), etc.

config = TaskConfig()

@cb.tasks_config(split="train")
def load(): return [cb.Task(description=..., metadata=..., computer=...)]

@cb.setup_task(split="train")
async def start(task, session: cb.DesktopSession): ...

@cb.evaluate_task(split="train")
async def evaluate(task, session: cb.DesktopSession) -> list[float]: ...
```

`task_card.json` sits next to `main.py` and declares VM resources:

```json
{"snapshot": "cpu-free-ubuntu", "vm": {"vcpus": 4, "memory_gb": 16}, "timeout_s": 600}
```

### What AgenthleEnv does

```
env = AgenthleEnv(provider=StubProvider())
obs = await env.reset_async(task_path="demo/hello", variant_index=0)
#   1. loader.load_task("demo/hello", 0)
#         → LoadedTask{cb_task, start_fn, evaluate_fn, task_card}
#   2. spec = lt.env_spec   (from task_card + cb_task.computer.setup_config.os_type)
#   3. vm = await provider.acquire(spec)
#   4. session = provider.open_session(vm)
#   5. await start_fn(cb_task, session)
#   6. return Observation(instruction=cb_task.description, done=False)

obs = await env.step_async(Submit())
#   1. scores = await evaluate_fn(cb_task, session)
#   2. reward = float(scores[0])
#   3. return Observation(done=True, reward=reward)

await env.close_async()
#      provider.release(vm)
```

That's the whole hot path. Five function calls in `reset`, three in
`step(Submit)`. Everything else (heartbeat, timeout, concurrency) is on
the Provider.

## Why these choices

| Decision | Why |
|---|---|
| One `AgenthleEnv` class, tasks-as-data | OpenEnv standard (echo / coding / opencode each one Environment). Maintaining 700+ Env subclasses is a waste; tasks differ in data, not behavior |
| Keep ``DesktopSession`` | The cua-bench Protocol works; reinventing it costs 200+ LOC and we lose accessibility-tree / PTY support |
| Keep ``evaluate() -> [float]`` | Migration cost of 700 tasks dominates any "cleaner" API; we accept the list[float] vestigial shape and pick element 0 |
| No Rubric | OpenEnv's `_apply_rubric` is overkill when tasks already return a score. We assign `reward = scores[0]` directly. Rubric can be reintroduced if/when we want composable scoring or ablations |
| Tasks at `tasks/<path>/main.py`, PEP 420 namespace | Mirrors agenthle's layout exactly. `from tasks.linux_runtime import LinuxTaskConfig` resolves identically. Zero migration friction |
| File-based loader, not registry | Discovery via filesystem (`tasks/<task_path>/main.py`) keeps task addition pure-data — no registry edits |

## Provider concrete impls (next slice)

| Provider | Status | Replaces |
|---|---|---|
| `StubProvider` (in tests) | ✅ tests/_stubs/ | — |
| `GCSDirectProvider` | TODO | `agenthle/scripts/web_console/lib/simprun` |
| `CuaHouseProvider` | TODO | `agenthle/agenthle/orchestration` |
| `LocalVMwareProvider` | TODO | none (new use case) |

Each implements: ``acquire``, ``release``, ``open_session``, ``heartbeat``,
``cancel_external``. Heartbeat is now a Provider concern, not a top-level
loop in `engine.py` with silent excepts.

## Agent + Deployer (next slice)

Out of scope for v0.1.0. Sketch:

```
BaseAgent.run(env: AgenthleEnv) -> EpisodeResult

NativeAgent(BaseAgent)         # orchestrator-side step loop
InstalledAgent(BaseAgent)      # in-VM CLI: claude-code / openclaw / ...
    └── InstalledAgentDeployer ABC  (install / launch / collect)
        └── 12 concrete deployers sharing a Toolchain mixin (Node/MCP)
```

Deployer becomes Agent-side, **not** env-side. One implementation per CLI,
shared by all Providers (no more simprun/cuahouse double-tracked code).
