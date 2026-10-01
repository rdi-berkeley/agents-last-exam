"""Logical image-family registry.

Each module under this package declares an :class:`Image` describing
one image family the framework knows how to provision against:

* identity (``name``, ``os``)
* VM-side path conventions (``work_dir_base`` / ``data_dir`` / ``node``
  / ``python`` / ``mcp_server_dir``) — splatted into
  :class:`SandboxHandle`
* provisioning defaults (``default_machine_type`` / ``gpu``) — read
  by Providers when sizing the substrate. Boot disk size comes from
  the underlying image itself; task data lives on the boot disk.

This is the framework's view of an image (what the deployer can rely
on without runtime discovery). The Provider-side, GCP-flavored
``GcloudImageSpec`` in :mod:`environments.capacity` is a different
concept — that one is per-deployment yaml config (zone, project,
network, image_name, ...).

Adding a new family = add a module here, declare ``IMAGE = Image(...)``,
register in ``_REGISTRY``. SandboxHandle / Providers / deployers never
hard-code an image-family literal — they consult :func:`get` /
:func:`registered`.

Base families:

  ``ale-kasm``             — linux (Docker, trycua/cua-ubuntu)
  ``ale-ubuntu22``         — linux (GCE VM)
  ``ale-ubuntu22-docker``  — linux (Docker, exported from the ale-ubuntu22 VM)
  ``ale-win10``            — windows (GCE VM)
  ``ale-win10-cpu-licensed`` — windows + licensed software (GCE VM)
  ``ale-win10-gpu-licensed`` — windows + licensed software + L4 (GCE VM)
  ``ale-win-server``       — windows + L4 GPU (GCE VM, g2 machine types)

Public v1.1 realizations use separate ``-v1-1`` keys. Their registration
does not change the legacy defaults or establish artifact availability.
"""
from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from importlib.util import find_spec
from typing import Literal


OS = Literal["linux", "windows"]


@dataclass(frozen=True)
class Image:
    """One logical image family — paths + provisioning defaults."""

    # ─── identity ───
    name: str                    # registry key, e.g. "ale-ubuntu22"
    os: OS

    # ─── sandbox-side paths (splatted into SandboxHandle) ───
    work_dir_base: str
    """Per-run scratch root, e.g. ``/home/user/.ale``."""

    task_data_root: str
    """Where staged task data lives (input/, reference/, output/, ...).
    e.g. ``/media/user/data/ale-data`` (linux) / ``E:\\ale-data`` (windows).
    Convention used by ``data_staging`` to build
    ``<task_data_root>/<domain>/<task>/<variant>/<subdir>``."""

    node: str
    python: str
    mcp_server_dir: str

    # ─── provisioning defaults (consumed by Providers) ───
    default_machine_type: str
    gpu: str | None = None

    # ─── docker realization (consumed by the docker provider) ───
    docker_image: str | None = None
    """The published container image ref that realizes this family under the
    docker provider (e.g. ``agentslastexam/ale-kasm:latest``). ``None`` for
    families that only exist as GCE images (ale-ubuntu22 / ale-win10)."""

    docker_entrypoint: str = "/dockerstartup/vnc_startup.sh"
    """Entrypoint the docker provider invokes (a container has no init, so the
    image must start cua-server itself). Default is the Kasm base image's startup
    script (ale-kasm); ale-ubuntu22-docker bakes its own at
    ``/dockerstartup/entrypoint.sh``."""

    # ─── cua-server port (image-specific; consumed by Providers + Executors) ───
    cua_server_port: int = 5000
    """Port the cua-server listens on inside this image. ``ale-kasm`` runs the
    cua-computer-server on its package default 8000; the GCE-backed families run
    it on 5000. The cua MCP bridge must be told this (it otherwise defaults to
    5000), so it is splatted into :class:`SandboxHandle` and used by Providers
    to build the cua-server URL + map ports."""

    def sandbox_paths(self) -> dict[str, object]:
        """Field dict for ``SandboxHandle(**image.sandbox_paths(), ...)``."""
        return {
            "work_dir_base":   self.work_dir_base,
            "task_data_root":  self.task_data_root,
            "node":            self.node,
            "python":          self.python,
            "mcp_server_dir":  self.mcp_server_dir,
            "cua_server_port": self.cua_server_port,
        }


_REGISTRY: dict[str, Image] = {}
for module_name in (
    "ale_kasm", "ale_ubuntu22", "ale_ubuntu22_docker", "ale_ubuntu22_docker_v1_1",
    "ale_ubuntu22_v1_1", "ale_win10", "ale_win10_v1_1", "ale_win_server",
):
    registered_image = import_module(f".{module_name}", __name__).IMAGE
    _REGISTRY[registered_image.name] = registered_image

for module_name in ("ale_win10_cpu_licensed", "ale_win10_gpu_licensed"):
    if find_spec(f"{__name__}.{module_name}") is not None:
        registered_image = import_module(f".{module_name}", __name__).IMAGE
        _REGISTRY[registered_image.name] = registered_image


def get(name: str) -> Image:
    """Look up an image family by name. Raise on unknown."""
    if name not in _REGISTRY:
        raise KeyError(
            f"unknown image family {name!r}; "
            f"registered: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name]


def registered() -> list[str]:
    return sorted(_REGISTRY)


__all__ = ["Image", "OS", "get", "registered"]
