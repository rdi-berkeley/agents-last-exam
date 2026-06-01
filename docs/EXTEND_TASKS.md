# Authoring your own task

A task in ALE is a Python package with two files:

```
tasks/<your_domain>/<your_task>/
├── task_card.json         metadata + VM resource spec
└── main.py                load() / start() / evaluate() hooks
```

`tasks/demo/hello/` (Linux) and `tasks/demo/hello_win/` (Windows) are the
canonical templates — copy one of them and edit. They cover the full
input / reference / software / output surface in ~250 lines.

---

## `task_card.json`

```json
{
  "taskId": "your_domain/your_task",
  "title":   "Short human-readable title",
  "summary": "One-paragraph description of what the agent must do.",
  "category": "your_domain",
  "vm": {
    "snapshot": "cpu-free-ubuntu",
    "vcpus": 4,
    "memory_gb": 16,
    "disk_gb": 200,
    "timeout_s": 1800
  }
}
```

| Field | Value |
|---|---|
| `taskId` | `<domain>/<task>` — must match the folder path |
| `category` | Top-level domain folder name |
| `vm.snapshot` | Which sandbox image to boot. Registered in [`ale_run/environments/images/`](../ale_run/environments/images/). Common values: `cpu-free-ubuntu`, `cpu-free` (Windows), `cpu-license`, `gpu-free`, `gpu-license` |
| `vm.{vcpus,memory_gb,disk_gb}` | Sizing hints — passed to the provider; resolved against the snapshot's capacity pool |
| `vm.timeout_s` | Per-variant wall budget (provisioning + setup + agent + eval) |

The framework reads `vm.snapshot` to pick the image; everything else
(network, machine family, fallback zones) comes from the env config your
experiment points at, e.g.
[`configs/environments/gcloud.yaml`](../configs/environments/gcloud.yaml).

---

## `main.py` — three required hooks

```python
import cua_bench as cb
from tasks.linux_runtime import LinuxTaskConfig    # or tasks.common_config.GeneralTaskConfig for Windows

VARIANTS: list[tuple[str, dict]] = [
    ("simple", {...}),
    ("hard",   {...}),
]

@dataclass
class TaskConfig(LinuxTaskConfig):
    DOMAIN_NAME: str = "your_domain"
    TASK_NAME: str = "your_task"
    VARIANT_NAME: str = "simple"
    # ... variant-specific fields here

@cb.tasks_config(split="train")
def load(): ...                                    # build one cb.Task per variant

@cb.setup_task(split="train")
async def start(task_cfg, session): ...            # stage inputs on the VM

@cb.evaluate_task(split="train")
async def evaluate(task_cfg, session) -> list[float]: ...
```

### `load()` — enumerate variants

Returns a list of `cb.Task` objects, one per variant. Each `cb.Task`
carries the task description (the prompt the agent sees) and metadata
the other hooks read back:

```python
@cb.tasks_config(split="train")
def load():
    out = []
    for variant_name, payload in VARIANTS:
        cfg = TaskConfig(VARIANT_NAME=variant_name, expected_payload=payload)
        out.append(cb.Task(
            description=cfg.task_description,
            metadata=cfg.to_metadata(),
            computer={"provider": "computer", "setup_config": {"os_type": cfg.OS_TYPE}},
        ))
    return out
```

### `start(task_cfg, session)` — stage the workspace

Runs once per variant on the freshly provisioned VM. Use it to:

- Create input/output directories on the sandbox.
- Write the prompt-side files the agent will read (`input/`).
- Drop helper scripts the agent may use (`software/`).
- Verify the reference is **not** visible yet (eval staging unlocks it later).

All sandbox I/O goes through `session` — see
[docs/SESSION_API.md](SESSION_API.md) for the full method list. Common
calls:

```python
await session.run_command("mkdir -p /path", check=False)
await session.write_file(path, content)
text = await session.read_file(path)
```

### `evaluate(task_cfg, session) -> list[float]`

Runs after the agent finishes. Returns a list of floats — one per
scoreable artifact (usually `[score]`). Standard scoring:

- `1.0` — exact match with the reference
- `0.0–0.5` — partial credit (line-level overlap, JSON-key match, etc.)
- `0.0` — no output, unparseable output, wrong format

Evaluate should **never raise** on missing output — return `[0.0]` and
let the framework log the absence. Raise only on infrastructure
failures (e.g. the sandbox is unreachable).

---

## Shipping task data

Two patterns, choose based on payload size.

### A. Inline staging in `start()` (small text)

What [`demo/hello`](../tasks/demo/hello/) does. Best for prompts,
small JSON inputs, helper scripts — anything under a few hundred KB.
Zero extra image work.

```python
async def start(task_cfg, session):
    meta = task_cfg.metadata
    await session.write_file(meta["input_request_path"], request_text)
    await session.write_file(meta["software_script_path"], helper_script)
```

### B. Pre-baked into the image (large binary data)

For datasets, benchmarks, model checkpoints, anything too heavy to
RPC-transfer per run. Set `requires_task_data=True` in the task config
and place files at:

```
<sandbox.task_data_root>/<domain>/<task>/<variant>/
├── input/
├── software/
└── reference.7z         # encrypted; decrypted JIT during evaluate
```

`<sandbox.task_data_root>` is `/media/user/data/ale-data` on Linux and
`E:\ale-data` on Windows (resolved through the image registry — see
[`ale_run/environments/images/`](../ale_run/environments/images/)).

Set `task_data_source: baked_in_sandbox` in the **environment** yaml
(`configs/environments/<env>.yaml`) if you ship your own image with these
files baked in. Otherwise point `task_data_source` at a `gs://<bucket>`
and `ale_run/environments/data_staging.py` will rsync from GCS per run.

---

## Conventions

- **Reference visibility:** the agent must not see `reference/` during
  `start()`. The framework writes the reference into the sandbox only at
  evaluate time. Always include a "reference correctly hidden" check at
  the end of `start()` — see the demo.
- **Idempotent setup:** `start()` may run against a reused dev VM. Wipe
  `output/` and `reference/` at the top so the run starts clean.
- **CRLF on Windows:** Windows tasks must use `\r\n` in expected text;
  CMD-style helpers emit CRLF. [`demo/hello_win`](../tasks/demo/hello_win/)
  shows the base64+PowerShell pattern for byte-exact file writes.
- **Variant params:** put per-variant data in `VARIANTS`, surface it
  through `TaskConfig`, and read it from `task_cfg.metadata` inside
  `start()` / `evaluate()`. Don't read globals.
- **Timeout budget:** allow ~2× the agent's wall time for slow tasks.
  GCP boot adds 3–5 minutes; eval adds whatever your scoring needs.

---

## Discovery

The framework discovers tasks by importing them by path — no central
registry. As soon as `tasks/<domain>/<task>/main.py` exists with the
three decorated hooks, the loader picks it up. To verify:

```bash
uv run python -m ale_run list | grep your_domain/your_task
```

To smoke-test the new task end-to-end, point a one-task experiment at
it:

```yaml
# tasks.yaml
- path: your_domain/your_task
  variants: [0]
```

```bash
uv run python -m ale_run run my_exp.yaml --task your_domain/your_task --dry-run
uv run python -m ale_run run my_exp.yaml --task your_domain/your_task
```

When the task is stable, add it to the appropriate curated list under
[`selected_tasks/`](../selected_tasks/) and open a PR.

---

## Submitting a workflow without writing code

Domain experts can submit task ideas (descriptions, reference files,
scoring rubric) through the contributor program at
[agents-last-exam.org/submit](https://agents-last-exam.org/submit). Qualifying
submissions become co-authored task packages — see
[agents-last-exam.org/rewards](https://agents-last-exam.org/rewards) for credit and
award details.
