"""Environments: OpenEnv-shaped VM-backed env + providers.

``ALEEnv`` (env.py) wraps a Provider + VMHandle + DesktopSession in the
OpenEnv ``Environment[Action, Observation, State]`` shape. step() is
intentionally absent — the tested agent drives itself inside the VM.

Helpers:
  - ``providers/``         Provider ABC + GcloudProvider + StaticProvider.
  - ``remote.py``          CUA HTTP primitives (run_remote, upload/download).
  - ``data_staging.py``    GCS-backed input/eval/reference staging onto the VM.
  - ``images.py``          ImageConfig + CapacityProfile dataclasses.
  - ``machine_types.py``   GCE machine-type parsing.
  - ``force_timeout.py``   In-VM sentinel for agent self-cancellation.

``ALEEnv`` is intentionally NOT re-exported at package level: importing
the env module pulls in the optional ``openenv`` dependency, and we want
``ale_run.environments.machine_types`` (and other pure-stdlib siblings)
to be importable without that. ``from ale_run.environments.env import ALEEnv``.
"""

__all__: list[str] = []
