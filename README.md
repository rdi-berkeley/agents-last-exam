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
uv sync --extra dev --all-packages
uv run python -c "from ale.core.env import AgenthleEnv; print('ok')"
uv run python tests/smoke_hello.py        # in-process stub env
```

## First-time setup: environment config

Several env vars (LLM API keys, GCS service-account path, optional bucket
overrides) are required at run time. They never live in yaml.

```bash
cp .env.example .env       # gitignored
$EDITOR .env               # fill in OPENROUTER_API_KEY, ALE_GCS_SA_KEY_PATH, etc
source .env                # load into current shell
```

See `.env.example` for the full list grouped by purpose (LLM keys, GCS
data staging, artifact mirror). The framework reads from `os.environ`;
for docker / vm runtimes it propagates a fixed set of vars
(`ale/runtime/_env.py`) into the substrate.

## Running an experiment

```bash
source .env
uv run python -m ale run experiments/foo.yaml          # full run
uv run python -m ale run experiments/foo.yaml --dry-run     # show matrix only
uv run python -m ale run experiments/foo.yaml --force-rerun # bypass resume
uv run python -m ale run experiments/foo.yaml --agent cc_sonnet --task demo/hello
```

Re-running the same `name:` resumes by default (skips units with prior
`status in {completed, timeout}` under `<output.root>/<name>/`).
`--force-rerun` re-attempts everything.

## v0.2.0 state

| Component | Status |
|-----------|--------|
| `AgenthleEnv` (one Env, OpenEnv-style) | ✅ |
| `Provider` ABC + `EnvSpec` + `VMHandle` | ✅ |
| `GCSDirectProvider` + transient retry + exp backoff | ✅ |
| `StaticProvider` (point at existing VM) | ✅ |
| File-based loader + auto-discover | ✅ |
| **Framework data staging** (stage_input/eval/reference/upload_output) | ✅ |
| Runtime abstraction (vm / local / docker) + 3 Executors | ✅ |
| Agent deployers: `claude_code` (vm), `ale_claw` (local/docker) | ✅ |
| Runner + dual-sem concurrency (provision + run) | ✅ |
| Incremental log pull (vm runtime, JSONL-boundary safe) | ✅ |
| Cancel-safe gather + best-effort full pull on cancel/fail | ✅ |
| Resume / skip-completed by experiment `name` | ✅ |
| Phase + error category in run.json.termination | ✅ |
| `CuaHouseProvider` (replaces orchestration/external) | ⏳ |
| Rate-limit detector + circuit breaker | ⏳ |
| Atomic-write run.json / trajectory.json | ⏳ |
