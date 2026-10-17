# Host and task environments

## Host evaluation environment

Run from the repository root on Ubuntu 22.04, 24.04, or 26.04, or Debian 12/13:

```bash
bash env/install_host_deps.sh
uv sync --all-packages --locked
bash env/verify_host_env.sh
```

Install `uv` and `python3` first. System package installation requires root or
sudo. Run `uv` as your normal user. The host script is idempotent and uses the
distribution's repositories; it does not change Docker, KVM, provider credentials,
VM images, or running experiments. Other distributions need an equivalent manual
installation of the tools listed in `host-deps.json`.

`tasks/pyproject.toml` declares host evaluator Python dependencies. The root
`uv.lock` is versioned and pins their resolved versions, including the required
NetworkX 3.6.1 route ordering. After changing any workspace dependency, update
the lock with `uv lock`, run verification, and include both files in the change.
Do not regenerate or upgrade dependencies during benchmark setup.

`host-deps.json` lists native packages, executable probes, and Python imports
that may be deferred until evaluation. FFmpeg provides both `ffmpeg` and
`ffprobe`; Bash validates submitted shell workflows, `findmnt` checks evaluator
scratch storage, and Rscript provides actual R syntax parsing. Native package
versions follow the supported distribution's security updates; they are not
pinned by `uv.lock`. The verifier reports their versions and exercises an H.264
encode, probe, and frame extraction to check the media path.

Verification checks the lockfile, installed-package consistency, evaluator
package versions, deferred imports, executable probes, and all task entry points
in `selected_tasks/full.txt`. It does not run agents, read task answers, or call
Judge APIs. To inspect another task list or save a report:

```bash
bash env/verify_host_env.sh --task-list selected_tasks/cpu.txt
bash env/verify_host_env.sh --json > host-dependencies.json
```

The runner repeats checks for the selected tasks in an isolated Python process
before its Judge preflight and before creating any sandbox. Missing tools,
incompatible versions, or broken imports abort the batch with setup instructions.
`--dry-run` only previews configuration and does not validate dependencies.

This verifies the host installation, not every task's full evaluator execution.
Guest applications and task-specific runtimes still come from the matching
published image/data revision or the per-task installation below. Use the release
provider allowlists; GPU tasks still require a compatible GPU guest. A new
evaluation dependency must be added to the relevant manifest and exercised in
the host or guest environment where it actually runs.

For local task-regression development, install `uv sync --all-packages --locked
--extra dev`. The dev extra includes libraries used by tests of guest scoring
helpers, such as DICOM, audio, and geometry packages. Those packages do not
replace the guest's native applications or pinned per-task environments.

## Per-task system environment

Each task declares the **system software** it needs in its `task_card.json`:

```json
"requiredSystemPackages": ["energyplus-22.1.0", "docker-ce"]
```

Every id maps to one idempotent installer in a package library:

- `env/packages-linux/<id>/` - `install.sh` + `meta.json` (Linux)
- `env/packages-win/<id>/` - `install.ps1` + `meta.json` (Windows Server 2022)

## Two entry points (no per-task scripts)

- `env/install_task_deps.sh <task_card.json>` - reads `requiredSystemPackages`
  and runs each package installer in order.
- `env/verify_task_env.sh <task_card.json> [task_base_dir]` - runs each
  package's `meta.json` `verify` command and validates declared
  `input/runtime_env` environments with `uv`.

## Policy

Installers provide **only system software/libraries** (apt packages,
version-pinned `/opt` (Linux) or `C:\Softwares` (Windows) binaries, docker
images). A task's **Python** packages come from its own staged
`input/runtime_env/` (`uv --frozen`) at solve time — they are never listed here.
Version-pinned tools install to a versioned path so multiple versions coexist;
task `software/` wrappers exec those exact paths.

## Licensed Windows software

Licensed packages use the same version pins and `meta.json` verification as
other system packages. Their installers and credentials are not committed to
this repository.

Each licensed package README records one reproducible provisioning sequence:

1. Purchase the product entitlement from the vendor.
2. Assign the entitlement to the account used for installation.
3. Sign in to the vendor account and download the pinned installer.
4. Install and activate the product on the target Windows machine.
5. Run the package's `meta.json` `verify` command.

Retain vendor installers securely for future reinstalls. Keep account
credentials, license keys, activation codes, and license files outside this
repository.
