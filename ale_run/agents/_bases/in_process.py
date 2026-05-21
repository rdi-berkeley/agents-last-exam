"""Base class for deployers whose agent code is an in-process Python
harness, not a remote CLI.

Used by AleClaw (OpenClaw harness) and any future agent of the same
shape. The deployer code runs in the framework's Python process
(``runtime: local``) OR in a host docker container (``runtime: docker``);
the eval VM is driven via ``runtime.make_vm_session()``.

The base provides a minimal :meth:`install` that sanity-checks:

* required Python modules import cleanly (catch packaging breaks before
  burning a VM acquire)
* at least one of a declared set of env-var API keys is present (catch
  missing-creds early)

``launch`` and ``parse_artifacts`` are agent-specific — the harness call
shapes vary too much across in-process agents to lift to a base.
"""
from __future__ import annotations

import importlib
import logging
import os
from typing import ClassVar

from ..base import BaseAgentDeployer

logger = logging.getLogger(__name__)


class InProcessHostDeployer(BaseAgentDeployer):
    """Common contract for in-process Python harness deployers.

    Subclass declares :attr:`required_modules` and
    :attr:`api_key_alternatives`; ``install`` raises early on either
    failing. Override :meth:`_extra_install` for further checks (e.g.
    workspace dir prep) — by default ``work_dir`` is created on the
    host so the launch path can write straight into it.
    """

    supported_runtimes: ClassVar[frozenset[str]] = frozenset({"local", "docker"})

    required_modules: ClassVar[tuple[str, ...]] = ()
    """Modules ``install`` will :func:`importlib.import_module` to fail
    fast on packaging breaks. Absolute FQNs (``"ale_run.agents.foo.bar"``)
    or relative-from-deployer-package (``".harness.agent_loop"``) — the
    latter is resolved against the deployer subclass's parent package
    (i.e. the directory the deployer lives in)."""

    api_key_alternatives: ClassVar[tuple[str, ...]] = ()
    """Env-var names; ``install`` requires at least one to be set when
    non-empty. Empty tuple skips the check (agent has no API key needs)."""

    async def install(self) -> None:
        # Relative-imports anchor: the deployer subclass's parent package
        # (the directory containing deployer.py). E.g. for a deployer
        # whose ``__module__`` is ``"ale_run.agents.ale_claw.deployer"``,
        # ``.harness`` resolves under ``ale_run.agents.ale_claw``.
        deployer_pkg = type(self).__module__.rsplit(".", 1)[0]
        for mod in self.required_modules:
            try:
                if mod.startswith("."):
                    importlib.import_module(mod, package=deployer_pkg)
                else:
                    importlib.import_module(mod)
            except ImportError as e:
                raise RuntimeError(
                    f"{type(self).__name__}: failed to import {mod!r}: {e}"
                ) from e
        if self.api_key_alternatives and not any(
            os.environ.get(k) for k in self.api_key_alternatives
        ):
            raise RuntimeError(
                f"{type(self).__name__}: no LLM API key in env — set one of "
                f"{', '.join(self.api_key_alternatives)}"
            )
        # work_dir is host-side here (LocalRuntime / DockerRuntime),
        # so the framework Python process can mkdir straight on it.
        await self.runtime.mkdir(self.runtime.work_dir)
        await self._extra_install()
        logger.info(
            "%s: install ok (model=%s, work_dir=%s, runtime=%s)",
            type(self).__name__,
            getattr(self.config, "model", "?"),
            self.runtime.work_dir,
            self.runtime.kind,
        )

    async def _extra_install(self) -> None:
        """Hook for subclass-specific install steps."""
        return None
