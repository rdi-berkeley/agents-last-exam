"""StaticProvider — wraps an already-running VM by its cua-server endpoint.

No VM lifecycle of its own: ``acquire`` returns a handle pointing at the
configured URL, ``release`` is a no-op (unless ``cleanup_on_release`` runs a
shell snippet to scrub the VM between iterations).

Used for:

- **Image baking**: bring up a VM by hand, iterate on deployer code without
  paying the 3-5 min boot cost on every run.
- **Dev debug**: poke at a long-running staging VM, reproduce stuck cases.
- **Tests against a fixed scratch VM**: stable IP, deterministic state.

Typical wiring::

    provider = StaticProvider(StaticProviderConfig(
        endpoint="http://34.55.x.x:5000",
        os="linux",
    ))
    env = ale.make("demo/hello", provider=provider)
"""
from __future__ import annotations

import dataclasses
import logging
from typing import TYPE_CHECKING

from ale.core.provider import EnvSpec, Provider, ReleaseMode, VMHandle

if TYPE_CHECKING:
    import cua_bench as cb

logger = logging.getLogger(__name__)


# =============================================================================
# Config
# =============================================================================

@dataclasses.dataclass(frozen=True)
class StaticProviderConfig:
    """Pin to a pre-existing VM."""

    endpoint: str
    """Full cua-server URL, e.g. ``http://1.2.3.4:5000``."""

    os: str = "linux"
    """OS of the pinned VM. Defaults to ``"linux"``."""

    vm_id: str = "static"
    """Informational id — shows up in logs / run.json."""

    cleanup_on_release: bool = False
    """If True, run :attr:`cleanup_script` on the VM during ``release``.
    The VM itself is never destroyed."""

    cleanup_script: str | None = None
    """Shell snippet executed via ``session.run_command`` on release. Only
    runs when ``cleanup_on_release`` is True."""


# =============================================================================
# Provider
# =============================================================================

class StaticProvider(Provider):
    """Provider impl that skips ``gcloud create / delete``."""

    def __init__(self, config: StaticProviderConfig):
        self._cfg = config

    async def acquire(self, spec: EnvSpec) -> VMHandle:
        # spec is informational only — we don't pick / create anything.
        logger.info(
            "static: returning fixed handle for %s (os=%s, spec.snapshot=%s)",
            self._cfg.endpoint, self._cfg.os, spec.snapshot,
        )
        return VMHandle(
            id=self._cfg.vm_id,
            endpoint=self._cfg.endpoint,
            os=self._cfg.os,                                # type: ignore[arg-type]
            metadata={
                "backend": "static",
                "spec_snapshot": spec.snapshot,
                "release_default": "keep",
            },
        )

    async def release(
        self, vm: VMHandle, *, mode: ReleaseMode = "keep",
    ) -> None:
        """Default: no-op. The VM is yours to manage; we never destroy it."""
        if not self._cfg.cleanup_on_release:
            logger.info("static: release no-op for %s", vm.endpoint)
            return
        if not self._cfg.cleanup_script:
            logger.warning(
                "static: cleanup_on_release=True but no cleanup_script; skipping"
            )
            return
        try:
            session = self.open_session(vm)
            await session.run_command(self._cfg.cleanup_script)
            logger.info("static: cleanup script ran on %s", vm.endpoint)
        except Exception as exc:                # noqa: BLE001
            logger.warning("static: cleanup script failed on %s: %s", vm.endpoint, exc)

    def open_session(self, vm: VMHandle) -> "cb.DesktopSession":
        import cua_bench as cb
        return cb.computers.remote.RemoteDesktopSession(
            api_url=vm.endpoint,
            os_type=vm.os,
            provider_type="computer",
            headless=True,
            ephemeral=False,                    # this VM is long-lived; tell cua-bench
        )
