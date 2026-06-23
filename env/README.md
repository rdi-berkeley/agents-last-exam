# Per-task system environment

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
