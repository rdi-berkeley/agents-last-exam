# Per-task environment: package library

Each Linux-only task declares the **system software** it needs in its
`task_card.json` under `requiredSystemPackages: [<id>, ...]`. Each `<id>` maps to
one idempotent installer at `env/packages/<id>/install.sh`. Running those
installers on the lean base image (`agentslastexam/ale-kasm:latest`) yields the
full environment the task needs.

## Model
- `env/packages/<id>/install.sh` — idempotent, root, installs ONE software
  component (apt package, version-pinned /opt binary, docker image, …) and
  self-checks at the end. Re-running is a no-op when already present.
- `env/packages/<id>/meta.json` — `{description, provides, verify}` (docs).
- `env/install_task_deps.sh <task_card.json>` — reads `requiredSystemPackages`
  and runs each package installer in order. This is the single entry point;
  per-task `scripts/install_deps.sh` is a thin shim that calls it.

## Policy (locked)
- Installers provide ONLY system software/libraries. A task's Python packages
  come from its own staged `input/runtime_env/` (`uv --frozen`, networked) at
  solve time — they are NOT packages here. So `requiredSystemPackages` lists
  things like `python-default-3.10`, `energyplus-22.1.0`, `docker-ce` — never
  numpy/pandas/etc.
- Version-pinned tools install under `/opt/<tool>-<version>/` so multiple
  versions coexist; the task `software/` wrappers exec those exact paths.

## Verification
Per-task `scripts/verify_env.sh` exercises the real task contract (runs the
software/ wrappers, imports the staged runtime, runs the binary). Package
installers also self-check. Test harness: see selected_tasks/INSTALL_DEPS_PROGRESS.md.
