# `ale.agents.ale_claw` — OpenClaw native deployer

First **native** deployer in ALE: the OpenClaw agent harness runs in the
ALE host's Python process (not subprocess, not docker, not in-VM). It
drives the test VM through `env.session.computer` (cua Computer SDK) and
writes its own per-turn transcripts + state.json + per-turn API result
dumps to a host tempdir we mirror back into the run directory.

The agent itself is unchanged from upstream `cua_bench/agents/openclaw/`
(branch `openclaw-cua` of `yixiao-huang/cua`, sha pinned in
`AleClawConfig.upstream_version`). The harness lives in `harness/` as
ALE-owned code — no `_vendor/` namespace, no `sys.modules` tricks. The
only edits to copied files were:

- delete `adapters/computer_agent.py` (historical alias)
- drop `OpenClawImageAwareComputerAgent` from `harness/__init__.py` and
  `harness/adapters/__init__.py`

The upstream wrapper class (`cua_bench/agents/openclaw_agent.py`) was
not copied; its `perform_task` body was inlined into
`AleClawDeployer.launch` so we have one procedural code path instead of
inheritance + 8 hook overrides.

## Layout

```
ale_run/agents/ale_claw/
├── __init__.py                 — re-exports AleClawConfig, AleClawDeployer
├── config.py                   — dataclass extending BaseAgentConfig
├── deployer.py                 — AleClawDeployer (work_dir_on_vm=False)
├── transcript_to_trajectory.py — on-disk transcripts → ATIF Steps
├── README.md                   — this file
└── harness/                    — OpenClaw harness (in-tree, ALE code)
    ├── AGENTS.md               — system-prompt context file
    ├── agent_loop.py           — OpenClawComputerAgent (run loop)
    ├── ... (32 modules)
    └── adapters/
```

## Configuration

```python
from ale.agents.ale_claw import AleClawConfig

cfg = AleClawConfig(
    model="openrouter/anthropic/claude-sonnet-4-20250514",
    openrouter_api_key=os.environ["OPENROUTER_API_KEY"],
    max_turns=100,         # OpenClaw max_steps
    timeout_s=3600,        # wall budget (asyncio.wait_for)
    disabled_tools=["web_search"],  # default; provide brave_api_key + clear list to enable
)
```

Full kwarg surface in `config.py` docstrings. Two model knobs to call out:

- `summary_model` / `gui_model` / `lightweight_model` — when set, OpenClaw
  routes compaction / vision / GUI subagent through cheaper siblings. Saves
  cost on long runs. Default: all use `model`.
- `thinking_level` (`off | low | medium | high`) — Claude reasoning depth.
  Defaults per-model via `harness.thinking.resolve_thinking_default`.

API keys are passed in explicitly. The deployer injects them into
`os.environ` (litellm reads from env) just-in-time around `agent.run()`,
restoring on exit. **Concurrency caveat**: with `concurrency > 1` and
DIFFERENT keys per unit, this races on the env. Same-key batches are
fine; different-key batches need subprocess isolation (v2).

## How it differs from agenthle's `ale-claw`

| | agenthle | ALE |
|---|---|---|
| Source layout | sparse git submodule + `_loader.py` (sys.modules trick) | in-tree `harness/`, no loader |
| Wrapper layer | `AleClawAgent(_UpstreamOpenClawAgent)` with 8 hook overrides | one procedural `launch()` body, no inheritance |
| Log format | `interaction_log.json` (InteractionLog/InteractionStep) | ALE `Trajectory` (ATIF) directly |
| `_resolve_workspace_root` | TASK_TAG / REMOTE_ROOT_DIR / TASK_CATEGORY env vars | `None` (permissive — full VM access) |
| Cross-run resume | `memory_base_dir` / `session_base_dir` per-run/explicit toggle | always per-run (v1); resume is a future flag |
| `_after_run_finally` writes log | yes | no — `collect()` translates transcript → Trajectory |

## Deferred for v2

- Cross-run memory + session resume (config flags + carry-over to subsequent runs)
- Subagent trajectory extraction into `Trajectory.subagent_trajectories`
- API key isolation across concurrent units (subprocess wrapping)
- Workspace root derivation from `task_path` (currently permissive)

See `docs/AGENTS.md` for the general deployer-author SOP and the Native
cookbook section that uses this deployer as the reference impl.
