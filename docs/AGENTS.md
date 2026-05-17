# How to add a new agent to ALE

This is the SOP for implementing a new agent deployer. Companion docs:

- `docs/DESIGN.md` — overall architecture (Env, Provider, Runtime, Deployer)
- `docs/SESSION_API.md` — `cb.DesktopSession` / `computer.interface.*` surface
- `ale/agents/claude_code/deployer.py` — in-VM (Phase 3 reference impl)
- `ale/agents/ale_claw/deployer.py` — host / docker native (Phase 2/4 reference impl)

> **Status (post Runtime refactor)**: 3 runtimes wired & smoke-verified on
> Linux dev VM 34.94.212.100 — `claude_code × vm` (22s), `ale_claw × local`
> (20s), `ale_claw × docker` (57s first / ~10s cached). Both deployers
> end-to-end on the same `BaseAgentDeployer` contract.

---

## 1. The contract — 1 ClassVar + 3 methods

```python
class BaseAgentDeployer(abc.ABC):
    supported_runtimes: ClassVar[frozenset[str]]   # subset of {"vm","local","docker"}

    def __init__(self, runtime: AgentRuntime):     # framework injects
        self.runtime = runtime
        self.config = runtime.config               # convenience alias

    @abc.abstractmethod
    async def install(self) -> None: ...           # stage prereqs

    @abc.abstractmethod
    async def launch(self, prompt: str) -> AgentRunResult: ...

    @classmethod
    @abc.abstractmethod
    def parse_artifacts(
        cls, *, work_dir, config, run_result, builder,
    ) -> None: ...                                 # always runs on host
```

That's the whole deployer surface. **No `session`, no `env`, no work_dir
method, no collect, no mirror_artifacts, no run.** The framework owns env
lifecycle, runtime construction, artifact gathering, trajectory finalize.

### What runtime gives you

```python
@dataclass
class AgentRuntime:
    work_dir: Path                                 # scratch dir framework created
    vm_endpoint: str                               # e.g. http://34.94.212.100:5000
    vm_os: Literal["linux","windows"]
    config: BaseAgentConfig                        # your config dataclass

    async def make_vm_session(self) -> cb.DesktopSession: ...   # ONE helper
```

Pure data. No API methods to learn — your deployer uses **stdlib**
(`subprocess`, `pathlib`, `json`) for execution. Where `self` happens to
live (VM Python / docker container / host process) is the framework's
concern, decided by the runtime kind.

---

## 2. Pick a runtime

| Runtime | Where deployer code runs | When to use |
|---|---|---|
| `vm` | Inside the eval VM (via `cua.python_exec`) | CLI-style agents that need to live next to eval files (claude-code, codex CLI) |
| `local` | This Python process | Lightweight host-side harnesses; fastest dev iteration (ale_claw default) |
| `docker` | A host docker container (`--network host`) | Same as local but with isolation: API-key env-bag, fs/dep sandboxing |

Declare the supported set on your deployer; yaml `runtime: <kind>`
validates against it (factory raises with the allowed set on mismatch):

```python
class MyAgentDeployer(BaseAgentDeployer):
    supported_runtimes = frozenset({"local", "docker"})       # ale_claw style
    # or:
    supported_runtimes = frozenset({"vm"})                    # claude_code style
```

---

## 3. File layout

```
ale/agents/<your_agent>/
├── __init__.py             — re-exports YourConfig, YourDeployer
├── pyproject.toml          — declares this agent's Python deps (uv workspace member)
├── config.py               — YourConfig(BaseAgentConfig); sets `name` ClassVar
├── deployer.py             — YourDeployer(BaseAgentDeployer)
└── (whatever else: helpers, vendored harness, Dockerfile if docker-only)
```

The framework auto-discovers via `ale/runner/factory.py:AGENT_REGISTRY`
— add a shortcut entry there pointing at your deployer + config classes
(or yaml callers can use the fqdn `ale.agents.<x>.deployer.YourDeployer`).

---

## 4. Per-agent dependencies

ALE root `pyproject.toml` carries only framework essentials
(`openenv-core`, `cua-bench`, `pydantic`, `httpx`, `anyio`). All
agent-specific deps live in `ale/agents/<your_agent>/pyproject.toml`:

```toml
[project]
name = "ale-my-agent"
version = "0.1.0"
requires-python = ">=3.12,<3.14"
dependencies = [
    "cua-agent",           # if you need cua's ComputerAgent SDK
    "litellm>=1.80",       # for LLM routing
    # any deps your agent harness imports
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
# Workspace member for dep-declaration only; actual code is in the parent
# `agent-last-exam` wheel.
only-include = []
sources = {}
bypass-selection = true
```

uv workspace at the root picks up `ale/agents/*/pyproject.toml` automatically.

Install in dev: `uv sync --all-packages --extra dev`.

---

## 5. Writing each method

### `install()`

Stage prereqs. The substrate is wherever the framework decided to place
you. Use stdlib only.

```python
async def install(self) -> None:
    import subprocess
    from pathlib import Path
    # self.runtime.work_dir exists; create whatever you need under it
    Path(self.runtime.work_dir).mkdir(parents=True, exist_ok=True)
    # Verify CLI is present (for vm-runtime: image-baked check)
    if not Path("/usr/local/bin/mycli").exists():
        raise RuntimeError("mycli missing on this image")
    # Or for local/docker: assume your pyproject's deps are installed
    # in this process and just sanity-check API keys.
```

### `launch(prompt)`

Spawn the agent. Block until done. Always return an `AgentRunResult`
(set `status="failed"` + `error=...` on internal errors; only raise if
*starting* failed).

```python
async def launch(self, prompt: str) -> AgentRunResult:
    import time, subprocess
    # write prompt to work_dir, spawn process, poll until done, classify outcome
    ...
    return AgentRunResult(
        status="completed",       # | "timeout" | "failed"
        exit_code=0,
        transcript_path=str(self.runtime.work_dir / "transcript.jsonl"),
        duration_s=time.monotonic() - t0,
    )
```

If your launch needs to drive the eval VM (host-runtime agents only):

```python
session = await self.runtime.make_vm_session()
await session.computer.interface.run_command("...")    # ssh/api into VM
```

### `parse_artifacts(...)` — host-side classmethod

```python
@classmethod
def parse_artifacts(
    cls, *, work_dir, config, run_result, builder,
) -> None:
    """Always runs on framework host AFTER gather pulls work_dir locally.
    Read files in work_dir, append Steps to builder."""
    transcript = work_dir / "transcript.jsonl"
    if not transcript.exists():
        builder.add_step(source="system", message="no transcript",
                         extra={"reason": "no_transcript"})
        return
    for line in transcript.read_text().splitlines():
        event = json.loads(line)
        # map to ATIF: source ∈ {"user","agent","environment","system"}
        builder.add_step(source="agent", message=..., tool_calls=...,
                         metrics=StepMetrics(input_tokens=..., output_tokens=...))
```

`builder` is a `TrajectoryBuilder` whose `add_step()` signature you can
see in `ale/agents/trajectory.py`. The framework seeds the leading
`source="user"` step (the instruction) before calling you.

---

## 6. Config

Subclass `BaseAgentConfig` (which gives `model`, `max_turns`, `timeout_s`,
`save_screenshots`, `api_keys`). Add your fields. **API keys are config
fields, never read from `os.environ` in the config itself** — caller
passes them explicitly. For docker/local-runtime agents, the executor
injects them into the substrate via `--env-file` (docker) or
`os.environ` patching (local) at runtime.

```python
@dataclass
class MyAgentConfig(BaseAgentConfig):
    name: ClassVar[str] = "my-agent"
    model: str = "anthropic/claude-sonnet-4.6"
    openrouter_api_key: str | None = None
    # ... agent-specific tunables ...

    def __post_init__(self):
        if not self.openrouter_api_key:
            raise ValueError("openrouter_api_key required")
```

---

## 7. Runtime-specific notes

### `vm` runtime (claude_code reference)

- Deployer runs INSIDE the test VM via `cua.python_exec`. Framework
  scp's `ale/runtime/` + `ale/agents/<your_agent>/` to
  `/home/user/.ale-src/` on the VM (idempotent hash-skip).
- `self.runtime.work_dir` is a VM path
  (e.g. `/home/user/.ale/<your_agent>/<run_id>/`).
- `subprocess.run("npm i -g ...")` executes ON the VM.
- `VmRuntime` provides image-baked path conventions: `node_exe`,
  `agent_bin_dir`, `cli_path("foo")`, etc.
- **No `from X import Y` in install/launch method body** — cua's
  `python_exec` source-generator lifts those to module level, before
  `sys.path.insert(0, '/home/user/.ale-src')` runs. Use
  `importlib.import_module` for any `ale.*` imports inside method bodies.
  (Framework-side static `from ...` is fine; only the function shipped
  to VM matters.) **`_vm_entry.py` is the framework's bootstrap; deployer
  code generally won't hit this restriction directly.**

### `local` runtime (ale_claw default)

- Deployer runs in this Python process.
- `self.runtime.work_dir` = `<run_dir>/origin_log/<agent>/` directly
  (no copy needed).
- Use `await self.runtime.make_vm_session()` to get a `cb.DesktopSession`
  pointing at the eval VM.
- API keys patched into `os.environ` only inside `launch()`'s scope
  (via a context manager); cleaned up on exit. **Concurrency caveat**:
  `concurrency > 1` with different API keys per unit races on
  `os.environ` — use `docker` runtime if you need true isolation.

### `docker` runtime (ale_claw isolation mode)

- Deployer runs in a host docker container started from `ale/native-base:0.1.0`.
- Build the base image once:
  ```bash
  docker build -t ale/native-base:0.1.0 \
    -f ale/runtime/Dockerfile.native_base ale/runtime/
  ```
- Container bind-mounts:
  - host `~/.cache/uv` → container `/root/.cache/uv` (uv cache, persists)
  - host repo parent dir → container `/projects` (so uv sync resolves
    path-pinned cua-* deps)
  - host `<run_dir>/origin_log/<agent>/` → container `/work`
    (this IS `self.runtime.work_dir`)
- API keys via `--env-file` (kept out of `docker inspect` and cmdline).
- Container entrypoint runs `uv sync --all-packages` then
  `python -m ale.runtime._docker_entry`. First run ~30-60s, cached ~5-10s.
- `--network host` so the container reaches the eval VM's public IP
  directly.

---

## 8. Registry + yaml

Add to `ale/runner/factory.py:AGENT_REGISTRY`:

```python
AGENT_REGISTRY: dict[str, tuple[str, str]] = {
    "claude_code": ("ale.agents.claude_code.deployer.ClaudeCodeDeployer",
                    "ale.agents.claude_code.config.ClaudeCodeConfig"),
    "ale_claw":    ("ale.agents.ale_claw.deployer.AleClawDeployer",
                    "ale.agents.ale_claw.config.AleClawConfig"),
    "my_agent":    ("ale.agents.my_agent.deployer.MyAgentDeployer",
                    "ale.agents.my_agent.config.MyAgentConfig"),
}
```

yaml usage:

```yaml
agents:
  - id: my_local
    class: my_agent
    # runtime: local         # implicit if single supported, or local preferred
    config:
      model: anthropic/claude-sonnet-4.6
      openrouter_api_key: ${env:OPENROUTER_API_KEY}

  - id: my_sandboxed
    class: my_agent
    runtime: docker          # explicit override
    config: {...}
```

The factory validates `runtime ∈ supported_runtimes` and raises with
the allowed set on mismatch.

---

## 9. Testing

Three layers, ordered by speed:

1. **Validation unit** — copy `tests/smoke_runtime_validation.py` and add
   cases for your deployer (auto-pick default, accept/reject runtimes).
   No LLM, no VM.

2. **Parser unit** — fabricate a fake on-disk `work_dir` with sample
   transcript files, call `YourDeployer.parse_artifacts(...)`, assert
   the trajectory builder's steps. Pattern in
   `tests/smoke_ale_claw_transcript.py`.

3. **Integration smoke** — full lifecycle against a real VM.
   Models: `tests/integration/runtime_smoke_ale_claw_local.py`,
   `runtime_smoke_ale_claw_docker.py`, `runtime_smoke_claude_code_vm.py`.
   Set `OPENROUTER_API_KEY` env var; run against Linux dev VM
   `34.94.212.100`.

---

## 10. Dos and don'ts

✅ Use stdlib (`subprocess`, `pathlib`, `json`) in install/launch — they
work the same way on host, in container, or in VM.

✅ Write everything to `self.runtime.work_dir`; framework handles
mirroring to host.

✅ Return `AgentRunResult` (don't raise) for agent-internal errors.

✅ Declare deps in `ale/agents/<your_agent>/pyproject.toml`, not in root.

✅ Use `importlib.import_module` for `ale.*` imports inside vm-runtime
deployer methods (cua python_exec gotcha).

❌ Don't take `session` / `env` as parameters — the contract gives you
`runtime` at init, that's it.

❌ Don't decide where `work_dir` lives — the runtime owns it.

❌ Don't read API keys from `os.environ` in config code. Take them as
explicit fields; the runtime puts them into the right place at run time.

❌ Don't write into `parse_artifacts`-only paths from install/launch
without using `self.runtime.work_dir` — the framework only gathers
work_dir, anything else is lost.

❌ Don't store the deployer's `cb.DesktopSession` on `self` — construct
fresh via `runtime.make_vm_session()` each launch (multiple session
clients on one cua-server is fine).
