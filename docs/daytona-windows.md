# Daytona Windows provider

Run the Windows CPU tasks on [Daytona](https://www.daytona.io) Windows sandboxes.
No local hypervisor, GPU passthrough, or cloud project is needed: each run unit
gets a fresh sandbox created from a Daytona snapshot that carries ALE's Windows
runtime and the task data, and the provider deletes it during normal cleanup.

Daytona has no Windows GPU sandboxes, so the three tasks in
`selected_tasks/gpu.txt` are not covered. The task lists are:

| List | Tasks | Notes |
|---|---:|---|
| `selected_tasks/daytona_windows_cpu.txt` | 33 | Every cpu-free Windows task |
| `selected_tasks/daytona_windows_cpu_noodoo.txt` | 32 | Without `business_finance/odoo`, which needs Odoo 19, PostgreSQL and a seeded template database that `env/packages-win` does not provide |

Daytona tiers cap vCPUs per sandbox and per organization (4 vCPU per sandbox
and 20 in total on the starter tier). Seven Windows tasks ask for 8 or 16 vCPU
and then run under-provisioned; the skeletal and butterfly animation tasks
still complete, the AR reconstruction tasks are the slowest.

## Host requirements

- Complete the [host evaluation setup](../env/README.md#host-evaluation-environment)
- A Daytona API key in `secret/.env` as `DAYTONA_API_KEY` (the key needs
  sandbox create/start/stop; without the delete scope the provider stops the
  sandbox and lets Daytona's auto-delete remove it)
- An `ssh` client and outbound TCP/22 to `ssh.app.daytona.io`: the cua
  computer-server inside the sandbox is reached through an SSH port-forward
  that the provider opens per sandbox
- 7-Zip (`7z`, `7zz` or `7za`) or `py7zr` on the host when building snapshots
  with reference archives

```bash
bash env/install_host_deps.sh
uv sync --all-packages --locked
bash env/verify_host_env.sh
```

## Build the runtime snapshot

Daytona sandboxes start from a snapshot. `env/daytona/build_snapshot.py` turns
a stock Daytona Windows Server 2025 snapshot into an ALE runtime snapshot, in
idempotent stages (the builder sandbox id is kept in
`env/daytona/builder_state.json`, which is gitignored):

```bash
export DAYTONA_API_KEY=...
uv run python env/daytona/build_snapshot.py all \
    --data /path/to/task-data \
    --tasks selected_tasks/daytona_windows_cpu.txt \
    --name ale-win-$(date +%Y%m%d)
```

`--data` is the task data root laid out as the published data packages are
(`tasks/<domain>/<task>/<variant>/{input,software,reference}`). The stages:

1. `create`: a builder sandbox from `--base-snapshot` (default `windows-large`).
2. `bootstrap`: `bootstrap_runtime.ps1` reproduces the `ale-win10` image
   conventions (`C:\Users\User` as a junction onto the Administrator profile,
   `E:` mapped onto `C:\drive-e`, Python 3.12 with the framework venv at
   `C:\ale-run\.venv`, Node 24, 7-Zip, the cua computer-server as the
   interactive scheduled task `CuaServer-ale` on port 5000).
3. `packages`: installs every `requiredSystemPackages` id of the selected tasks
   from `env/packages-win` (Chocolatey bootstrap first).
4. `data`: uploads `input/` and `software/` of each task to `E:\agenthle` and
   bakes `reference.7z` (password `ALE_REFERENCE_ARCHIVE_PASSWORD`) when the
   host has the task's `reference/`.
5. `verify`: runs each package's verify command and spot-checks the layout.
6. `snapshot`: stops the builder and creates the Daytona snapshot `--name`.

Optional follow-up scripts re-open the builder and re-snapshot under a new
name: `add_git.py` (Git for Windows; Claude Code's Bash tool needs git-bash),
`add_softwares.py` (Godot 4.6 and Blender 4.5 portable builds under
`C:\Softwares`), `add_hardened_python.py` (copies of the interpreter named
`alepy.exe` / `cuasrv.exe` so an agent's "kill every python" habit does not
take the in-box worker or the cua-server down; route it with `python:` in the
environment yaml), and `slim_snapshot.py` (frees disk; the builder's 50 GB disk
keeps about 10 GB free after the packages).

## Run

`configs/environments/daytona_windows.yaml` routes the `cpu-free` snapshot tag
to a Daytona snapshot. Set `daytona.snapshot` to the name you built, then start
from `daytona_windows_exp.yaml`:

```bash
uv run python -m ale_run run daytona_windows_exp.yaml --dry-run
uv run python -m ale_run run daytona_windows_exp.yaml
```

Provider settings (`daytona:` under a snapshot entry; client-wide keys may be
repeated on every entry, the last one wins):

| Key | Default | Meaning |
|---|---|---|
| `snapshot` | required | Daytona snapshot name |
| `api_key`, `api_url`, `target` | env / SDK defaults | Daytona client settings; `target` is the region |
| `ttl_minutes` | 600 | Auto-stop deadline per sandbox |
| `cpu`, `memory`, `disk` | snapshot shape | Hot-resize after creation (increase only; the tier may refuse) |
| `python` | image default | In-box interpreter for the sandbox worker (hardened snapshots: `C:\ale-py\alepy.exe`) |
| `disable_windows_update` | true | Windows Update reboots kill the cua transport mid-run |
| `isolate_task_data` | false | Delete the other tasks' directories under `E:\agenthle` right after creation |
| `hide_cua_console` | true | Run the cua-server through a hidden launcher so an agent's Alt+F4 cannot close it |
| `post_create_ps` | none | Extra PowerShell to run after the cua-server is up |
| `cua_ready_timeout_s` | 600 | How long to wait for the cua-server after creation |

`ALE_DAYTONA_CAPACITY_WAIT_S` (default 4 h) bounds how long `acquire` waits
for a free slot when the organization's vCPU cap is reached; set
`concurrency` to the number of sandboxes the tier allows. The provider also
runs a per-sandbox watchdog that reopens a dead SSH tunnel and restarts the
cua-server scheduled task when the sandbox stops answering.

### References staged from the host

A snapshot built without reference data (no `ALE_REFERENCE_ARCHIVE_PASSWORD`,
or references obtained later) can still be graded: set
`task_data_source: localref:/path/to/task-data/tasks` in the environment yaml.
Inputs and software stay baked in the snapshot; the hidden `reference/` is
zipped on the host and uploaded only after the agent has finished, so the
answers never exist in the sandbox during the run.

Runs that finished without a reference (`output_path: local` keeps their
deliverables under `<run_dir>/output/`) can be re-scored later in a fresh
sandbox with `env/daytona/regrade.py`, which writes `<run_dir>/regrade.json`:

```bash
uv run python env/daytona/regrade.py daytona_windows_exp.yaml --refs /path/to/task-data/tasks
```

## Self-hosted model gateways

`configs/agents/claude_code_gateway.yaml` shows the `claude_code` settings for
an Anthropic-compatible gateway that authenticates by header and implements the
Messages API only partially: `extra_headers` and `session_header` for
authentication and session affinity, `api_proxy` for an in-sandbox proxy that
converts PDF `document` blocks to text, calls the gateway non-streaming and
re-samples on malformed output or 5xx/429 before re-emitting SSE to the CLI,
`resume_on_api_error` to `--resume` the session when the CLI ends on an API
error, and `extra_env` to pin every Claude Code model slot to the served model.

## Known limitations

- Per-sandbox disk is fixed by the snapshot (50 GB on `windows-large`);
  Daytona has no disk resize API, so very large deliverables (the AR
  reconstruction tasks write several GB) can fill it.
- Sandboxes carry the data of every task they were built for;
  `isolate_task_data: true` removes the siblings so an agent cannot browse
  other tasks' inputs.
- The `qm9_mmff94_forcefield_survey_1` verifier rejects the published
  reference (`inconsistent delta_e in rank_1`) and cannot score any run.
