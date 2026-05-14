# agent-last-exam (`ale`)

Upper-layer-only replacement for agenthle orchestration. **Tasks stay in
the existing agenthle format** — `main.py` with `@cb.tasks_config` /
`@cb.setup_task` / `@cb.evaluate_task`, using `DesktopSession`. We only
redesign the bits above: Env / Provider / Agent / Runner.

## Layout

```
agent-last-exam/
├── ale/                       Framework (the only Python package we ship)
│   └── core/
│       ├── env.py             AgenthleEnv (one Env, OpenEnv-canonical)
│       ├── provider.py        Provider ABC + EnvSpec + VMHandle
│       ├── loader.py          File-based task discovery (tasks/<path>/main.py)
│       └── types.py           Actions + AgenthleObservation + AgenthleState
├── tasks/                     Task library (PEP 420 namespace package)
│   ├── common_config.py       GeneralTaskConfig (verbatim from agenthle)
│   ├── linux_runtime.py       LinuxTaskConfig (verbatim from agenthle)
│   └── demo/hello/
│       ├── main.py            Demo task in agenthle format
│       └── task_card.json     VM resource declaration
├── tests/_stubs/              In-process StubDesktopSession + StubProvider
└── docs/DESIGN.md             Architecture
```

Migration plan: `git mv agenthle/tasks/<foo>` → `agent-last-exam/tasks/<foo>`.
Task files are untouched.

## Why this shape

| Pain (old agenthle) | What this fixes it with |
|---|---|
| 12 deployers duplicated across `simprun` + `orchestration/external` | Deployer becomes an Agent concern, shared across all Providers (next slice) |
| simprun vs cuahouse double-tracked VM lifecycle | Single `Provider` ABC; one impl each, `acquire` / `release` / `heartbeat` / `cancel_external` |
| heartbeat / timeout / semaphore scattered with silent excepts | Central primitives on Provider — concrete impls choose semantics; framework doesn't ad-hoc |
| `evaluate()` reward computation lived inside the task | Same — we don't move it. Just feed the score straight to `observation.reward` |
| Single Env type doing too much in `task_env.py` | `AgenthleEnv` is OpenEnv-canonical `Environment`; everything outside it is the Env's API surface (reset/step) |

## Quick check

```bash
cd agent-last-exam
uv sync --extra dev
uv run python -c "from ale.core.env import AgenthleEnv; print('ok')"
uv run python tests/smoke_hello.py     # end-to-end with stub session
```

## v0.1.0 state

| Component | Status |
|-----------|--------|
| `AgenthleEnv` (one Env, OpenEnv-style) | ✅ |
| `Provider` ABC + `EnvSpec` + `VMHandle` | ✅ |
| File-based loader (`tasks/<path>/main.py`) | ✅ |
| Demo task + StubProvider/StubDesktopSession | ✅ |
| `GCSDirectProvider` (replaces simprun) | ⏳ next |
| `CuaHouseProvider` (replaces orchestration/external) | ⏳ next |
| Agent + Deployer (12 in-VM CLIs) | ⏳ next |
| Runner (replaces engine.py / cli.py) | ⏳ next |
