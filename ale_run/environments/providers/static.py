"""StaticProvider — wraps an already-running VM by its cua-server endpoint.

No VM lifecycle of its own: ``acquire`` returns a handle pointing at the
configured URL, ``release`` is a no-op (unless ``cleanup_on_release`` runs a
shell snippet to scrub the VM between iterations).

Used for:

- **Image baking**: bring up a VM by hand, iterate on deployer code without
  paying the 3-5 min boot cost on every run.
- **Dev debug**: poke at a long-running staging VM, reproduce stuck cases.
- **Tests against a fixed scratch VM**: stable IP, deterministic state.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Any

from .provider import EnvSpec, Provider, ReleaseMode, VMHandle

logger = logging.getLogger(__name__)


# ======================================================================
# Config
# ======================================================================


@dataclasses.dataclass(frozen=True)
class StaticProviderConfig:
    """Pin to a pre-existing VM."""

    endpoint: str
    """Full cua-server URL, e.g. ``http://1.2.3.4:5000``."""

    os: str = "linux"
    """OS of the pinned VM."""

    vm_id: str = "static"
    """Informational id — shows up in logs / run.json."""

    cleanup_on_release: bool = False
    """If True, run :attr:`cleanup_script` on the VM during ``release``.
    The VM itself is never destroyed."""

    cleanup_script: str | None = None
    """Shell snippet executed via ``run_remote`` on release. Only runs
    when ``cleanup_on_release`` is True."""


def _build_provider_config(raw: dict[str, Any]) -> StaticProviderConfig:
    return StaticProviderConfig(
        endpoint=str(raw["endpoint"]),
        os=str(raw.get("os") or "linux"),
        vm_id=str(raw.get("vm_id") or "static"),
        cleanup_on_release=bool(raw.get("cleanup_on_release", False)),
        cleanup_script=raw.get("cleanup_script"),
    )


# ======================================================================
# Provider
# ======================================================================


class StaticProvider(Provider):
    """Provider impl that skips ``gcloud create / delete``."""

    def __init__(self, config: StaticProviderConfig | dict[str, Any]):
        if isinstance(config, dict):
            config = _build_provider_config(config)
        self._cfg = config

    async def acquire(
        self,
        spec: EnvSpec,
        *,
        exclude_profiles: set[str] | None = None,
    ) -> VMHandle:
        # StaticProvider has no capacity-profile concept; the exclude
        # argument is accepted for ABC parity but ignored.
        _ = exclude_profiles
        return VMHandle(
            id=self._cfg.vm_id,
            endpoint=self._cfg.endpoint,
            os=self._cfg.os,
            metadata={"static": True, "snapshot": spec.snapshot},
        )

    async def release(self, vm: VMHandle, *, mode: ReleaseMode = "keep") -> None:
        if not self._cfg.cleanup_on_release or not self._cfg.cleanup_script:
            return
        from ..remote import RemoteVMConfig, run_remote

        cfg = RemoteVMConfig(server_url=vm.endpoint, os_type=vm.os)
        try:
            run_remote(cfg, self._cfg.cleanup_script, timeout=120)
        except Exception as e:
            logger.warning("StaticProvider cleanup failed: %s", e)

    def open_session(self, vm: VMHandle) -> Any:
        from cua_bench.computers.remote import RemoteDesktopSession
        from .gcloud import _init_computer_skip_wait

        session = RemoteDesktopSession(api_url=vm.endpoint, os_type=vm.os)
        _init_computer_skip_wait(session)
        return session
