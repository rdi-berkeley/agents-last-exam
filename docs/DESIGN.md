# `ale` architecture

`agent-last-exam` is the orchestration framework for evaluating
computer-use agents against task VMs. This doc is the
**single-page map**: what each module owns, how they fit, and the
exact lifecycle of one task × one agent.

Companion docs:
- `docs/AGENTS.md` — SOP for implementing a new agent deployer
- `docs/SESSION_API.md` — `cb.DesktopSession` / `computer.interface.*` surface

---

## One paragraph

ALE is built around three orthogonal abstractions: **`AgenthleEnv`**
(OpenEnv-shape, one class for every task), **`Provider`** (acquires +
releases the test VM), and **`Runtime`** (decides where + how an agent
deployer executes — `vm` / `local` / `docker`). A `Runner` reads a yaml
experiment spec, fans out (agent × task × variant) into RunUnits, and
for each one drives: `env.reset` → `executor.run_deployer` →
`gather work_dir` → `parse_artifacts` → `env.step(Submit)` → finalize
trajectory + run.json. Task files stay in agenthle's format unchanged.

---

## Overview picture

```
                                ┌────────────────────────────────────────────┐
                                │             experiment yaml                │
                                │   (provider, agents[], tasks[], runtime)   │
                                └─────────────────┬──────────────────────────┘
                                                  │ loader.load_experiment
                                                  ▼
                                          ┌──────────────┐
                                          │  Runner      │  ale/runner/runner.py
                                          │ (asyncio    │
                                          │ Semaphore N) │
                                          └──────┬───────┘
                                                  │ for unit in RunUnit[…]:
                                                  ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ lifecycle.run_one_unit(unit)                  ale/runner/lifecycle.py                  │
│  ┌──────────────────────────────────────────────────────────────────────────────────┐ │
│  │ ① resolve_agent(spec)                                                            │ │
│  │      → (deployer_cls, config, runtime_kind)   ale/runner/factory.py              │ │
│  │      validates: runtime_kind ∈ deployer_cls.supported_runtimes                   │ │
│  │                                                                                  │ │
│  │ ② env = ale.make(task_path, provider)                                            │ │
│  │      → AgenthleEnv  (one class for all tasks)   ale/core/env.py                  │ │
│  │                                                                                  │ │
│  │ ③ obs = await env.reset_async(variant_index)                                     │ │
│  │      provider.acquire(spec) → VMHandle                                           │ │
│  │      session = provider.open_session(vm)                                         │ │
│  │      await task.setup(session)        ── runs on VM                              │ │
│  │      return Observation(instruction=…)                                           │ │
│  │                                                                                  │ │
│  │ ④ runtime = make_runtime(kind, env, agent_name, run_id, host_origin_dir)         │ │
│  │      LocalRuntime  → work_dir = <run_dir>/origin_log/<agent>/                    │ │
│  │      VmRuntime     → work_dir = /home/user/.ale/<agent>/<run_id>  (VM path)      │ │
│  │      DockerRuntime → work_dir = <run_dir>/origin_log/<agent>/  (host;            │ │
│  │                       container bind-mounts it as /work)                         │ │
│  │                                                                                  │ │
│  │ ⑤ result = await EXECUTORS[kind].run_deployer(                                   │ │
│  │       deployer_cls, runtime, prompt, timeout_s)                                  │ │
│  │      LocalExecutor:  in-process — construct deployer, await install + launch     │ │
│  │      VmExecutor:     scp ale subtree to VM, python_exec _vm_entry bootstrap      │ │
│  │      DockerExecutor: docker run with bind mounts + entry script                  │ │
│  │      → AgentRunResult{status, transcript_path, exit_code, ...}                   │ │
│  │                                                                                  │ │
│  │ ⑥ local_work_dir = await executor.gather_to_host(runtime, dest=…)                │ │
│  │      vm:     session.read_bytes recursive pull from VM                           │ │
│  │      docker: no-op (work_dir was bind-mounted)                                   │ │
│  │      local:  no-op (already there)                                               │ │
│  │                                                                                  │ │
│  │ ⑦ _gather_task_output(env, dest=<run_dir>/output/)                               │ │
│  │      pull task.metadata['remote_output_dir'] from VM via env.session             │ │
│  │      (this is where the AGENT'S produced output lives; eval reads it)            │ │
│  │                                                                                  │ │
│  │ ⑧ deployer_cls.parse_artifacts(work_dir, config, run_result, builder)            │ │
│  │      pure classmethod, ALWAYS on host                                            │ │
│  │      reads files in work_dir → builder.add_step(source=…)                        │ │
│  │                                                                                  │ │
│  │ ⑨ final_obs = await env.step_async(Submit())                                     │ │
│  │      await task.evaluate(session) → reward                                       │ │
│  │      runs on VM via the framework's env.session                                  │ │
│  │                                                                                  │ │
│  │ ⑩ builder.finalize + RunWriter writes:                                           │ │
│  │      <run_dir>/{run.json, trajectory.json, eval_result.json, events.jsonl}       │ │
│  └──────────────────────────────────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────────────────────────────────────┘
                                                  │
                                                  ▼
                                          ┌──────────────┐
                                          │  UnitResult  │
                                          └──────────────┘
```

---

## Module map (what owns what)

```
ale/
├── __init__.py / registry.py     gym-style ale.register / ale.make
│
├── core/                         the task/env contract
│   ├── env.py                    AgenthleEnv (one class for all tasks)
│   ├── loader.py                 tasks/<path>/main.py → LoadedTask
│   ├── provider.py               Provider ABC + VMHandle
│   ├── types.py                  Action types (Submit/RunCommand/.../)
│   └── cmd_result.py             cua dict vs stub attr compat
│
├── providers/                    Provider impls (which VM, where?)
│   ├── gcs_direct.py             ephemeral VM via `gcloud compute instances create`
│   └── static.py                 pre-existing VM at a fixed endpoint (dev mode)
│
├── runtime/                      where + how the agent runs
│   ├── base.py                   AgentRuntime (passive context dataclass)
│   ├── executor.py               Executor ABC + EXECUTORS dict
│   ├── local.py / local_executor.py     in-process; trivial
│   ├── vm.py / vm_executor.py / _vm_entry.py
│   │                             scp ale subtree → cua python_exec bootstrap
│   ├── docker.py / docker_executor.py / _docker_entry.py
│   │                             docker run --network host + bind mount + entry
│   └── Dockerfile.native_base    ale/native-base:0.1.0 image
│
├── agents/                       agent deployers (per CLI / harness)
│   ├── base.py                   BaseAgentDeployer ABC + AgentRunResult etc.
│   ├── trajectory.py             ATIF-v1.0 Trajectory schema
│   ├── claude_code/              vm-runtime: @anthropic-ai/claude-code CLI
│   │   ├── pyproject.toml         (workspace member; no Python deps)
│   │   ├── config.py
│   │   └── deployer.py
│   └── ale_claw/                 local|docker-runtime: OpenClaw harness
│       ├── pyproject.toml         (cua-agent, cua-computer, litellm, ...)
│       ├── config.py
│       ├── deployer.py
│       ├── transcript_to_trajectory.py
│       └── harness/               vendored OpenClaw upstream
│
├── runner/                       yaml → matrix → execute
│   ├── spec.py                   ExperimentSpec / AgentSpec / RunUnit / ...
│   ├── loader.py                 yaml + ${env:X} substitution
│   ├── factory.py                resolve_agent(spec) → (cls, cfg, runtime_kind)
│   ├── lifecycle.py              run_one_unit (the orchestration above)
│   └── runner.py                 Runner (asyncio.Semaphore concurrency)
│
├── io/                           on-disk artifact layout
│   ├── run_writer.py             <run_dir>/{run.json, trajectory.json, events.jsonl}
│   └── artifact_mirror.py        GCS-bridge / cua-direct pull (used by Vm-style)
│
└── cli.py / __main__.py          python -m ale run <yaml> [--dry-run]
```

---

## Three orthogonal abstractions

### 1. `AgenthleEnv` — the task surface

One Python class for every benchmark task. OpenEnv-shape (`reset_async`,
`step_async`, `state`). Doesn't know about agents — just runs the task's
`setup` and `evaluate` against a session.

```
env = ale.make("demo/hello", provider=provider)
obs = await env.reset_async(variant_index=0)   # task.setup(session) on VM
# ... agent runs ...
obs = await env.step_async(Submit())            # task.evaluate(session) on VM
await env.close_async()                         # provider.release(vm)
```

**Owned by**: `ale/core/env.py`.
**Task files**: unchanged from agenthle's format
(`main.py` with `@cb.tasks_config / @cb.setup_task / @cb.evaluate_task`).

### 2. `Provider` — VM lifecycle

How to get + release a test VM. The session is built by the provider.

| Impl | Use |
|---|---|
| `StaticProvider` | pre-existing VM at a fixed endpoint (dev mode) |
| `GCSDirectProvider` | gcloud-create an ephemeral VM, wait for cua-server, release on close |

**Owned by**: `ale/providers/` + the ABC at `ale/core/provider.py`.

### 3. `AgentRuntime` + `Executor` — where the agent code runs

The deployer (which is just Python code) is **placed** by the framework
into one of three substrates:

| `runtime` kind | Where deployer runs | Executor does |
|---|---|---|
| `vm` | Inside the test VM's Python (via cua `python_exec`) | scp ale subtree → ship bootstrap fn → pull work_dir back |
| `local` | This Python process | direct `await deployer.install() / .launch()` |
| `docker` | Host docker container (`--network host`) | `docker run` + bind mounts + entry script |

**`AgentRuntime`** is a **passive dataclass** (work_dir, vm_endpoint, vm_os,
config) injected at deployer init. Deployer code uses **stdlib**
(subprocess, pathlib, json) — substrate differences are absorbed by
WHERE the deployer is constructed, not by what its code calls.

```
┌─────────────────────────────────────────────────────────────────┐
│ Deployer contract — 1 ClassVar + 3 methods, no env, no session  │
│                                                                 │
│   class MyDeployer(BaseAgentDeployer):                          │
│       supported_runtimes = frozenset({"local", "docker"})       │
│                                                                 │
│       def __init__(self, runtime: AgentRuntime): ...            │
│       async def install(self): ...                              │
│       async def launch(self, prompt) -> AgentRunResult: ...     │
│       @classmethod                                              │
│       def parse_artifacts(cls, *, work_dir, config, run_result, │
│                           builder) -> None: ...                 │
└─────────────────────────────────────────────────────────────────┘
```

**Owned by**: `ale/runtime/` (Runtime + Executor) and `ale/agents/base.py`
(the deployer ABC).

---

## On-disk run output

```
<exp_root>/<exp_name>/<agent_id>/<model_slug>/<task_slug>/v<i>/<ts>/
├── run.json              one-shot summary {agent, runtime, task, status, score, ...}
├── trajectory.json       ATIF-v1.0 (steps[source ∈ {user,agent,environment,system}])
├── eval_result.json      {eval_status, score, eval_duration_s, error}
├── events.jsonl          timestamped events: run_started, agent_run_started,
│                          agent_finished, artifact_gather_done,
│                          task_output_gather_done, run_completed
├── origin_log/<agent>/   deployer's work_dir (transcripts, scripts, etc.)
└── output/               task's remote_output_dir mirrored from VM
                          (= the AGENT'S produced output; what eval read)
```

---

## Concurrency model

- `Runner` uses `asyncio.Semaphore(concurrency)` to cap parallel units.
- Each unit has its own env + provider acquire + runtime + executor.
- `static` provider serves one VM, so concurrency MUST be 1 with it.
- `gcs_direct` provider creates per-acquire VMs, parallel safe.
- `local` runtime: in-process; API keys patched in `os.environ` per-launch
  — concurrent units with DIFFERENT keys race. Same-key batches are fine.
- `docker` runtime: each unit gets its own container with `--env-file`,
  no env race.

---

## Design choices (recap)

| Decision | Why |
|---|---|
| One `AgenthleEnv` class, tasks-as-data | OpenEnv standard. Tasks differ in data, not behavior — 700+ Env subclasses is waste. |
| Keep `cb.DesktopSession` | Works, has accessibility-tree + PTY support. Reinventing costs 200+ LOC. |
| Keep `evaluate() → [float]` | Migration cost of 700 tasks dominates any "cleaner" API. Pick element 0. |
| No Rubric layer | Tasks already return a score. Add Rubric only if we want composable scoring later. |
| Runtime is a passive context dataclass, not an API | Deployer uses stdlib; substrate differences absorbed in WHERE deployer is constructed, not in what it calls. (Less learning surface for new deployer authors.) |
| `supported_runtimes: frozenset[str]` ClassVar | Strings match yaml `runtime:` enum 1:1. Factory validates at spec-load. |
| Per-agent `pyproject.toml` + uv workspace | Agent deps self-contained. New agent = drop a folder. Docker images can `uv sync` agent-specific. |
| `parse_artifacts` is a classmethod (host-side) | Pure-fn translation; no need for an Executor to ship the parser. Same code path regardless of runtime kind. |
| `gather_to_host` separated from deployer | Mirroring is a runtime concern, not an agent concern (vm: cua pull; docker: bind mount = free; local: same path = free). |
| `_gather_task_output` in lifecycle (not executor) | Task output is always on VM regardless of agent runtime; framework owns env.session and does it once. |

---

## Reference impls

| Agent | Runtime | LOC (deployer + entry) | Notes |
|---|---|---|---|
| `claude_code` | `vm` only | ~350 | install verifies image-baked `/usr/local/bin/claude`; launch = setsid + done.marker poll on VM; parse stream-json transcript |
| `ale_claw` | `local` (default) / `docker` | ~360 + ~5000 vendored OpenClaw harness | install = sanity; launch builds session via `runtime.make_vm_session()` and runs the OpenClaw harness end-to-end |

Both produce identical-shape `<run_dir>/{run.json, trajectory.json, ...}`.

---

## Verified smoke matrix

All three combinations pass on the Linux dev VM `34.94.212.100`:

| Agent × Runtime | Duration | Origin files | Output mirror |
|---|---|---|---|
| `claude_code × vm` | 22.8s | 8 | `output/answer.txt` ✓ |
| `ale_claw × local` | 19.6s | 20 | `output/answer.txt` ✓ |
| `ale_claw × docker` | 57.0s (cold) | 24 | `output/answer.txt` ✓ |

Unit: `tests/smoke_runtime_validation.py` (6 cases) covers
`supported_runtimes` validation and default-pick policy.

---

## What's NOT here yet

- `CuaHouseProvider` (multi-tenant cua-house bridge) — TODO
- `LocalVMwareProvider` — TODO
- Docker base image registry push — local builds only for v1
- Cross-runtime agent (vm + docker support) — no driver yet
- Subagent trajectory extraction into `Trajectory.subagent_trajectories` — schema field exists; parsing deferred
- Trajectory chunking for very long episodes (`continued_trajectory_ref` field exists; splitting logic TBD)
