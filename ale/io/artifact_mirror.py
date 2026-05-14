"""ArtifactMirror — pull large work_dir + output_dir from VM to local disk.

Two transport paths:

1. **GCS bridge (primary)** — fast and reliable for big trees.
   - VM-side: ``gsutil cp -r <vm_path>/* gs://bucket/<run_id>/<sub>/``
   - Local:   ``gsutil cp -r gs://bucket/<run_id>/<sub>/* <local_dest>/``
   - Used when ``gcs_bucket`` is configured AND VM has gsutil + auth.

2. **CUA direct (fallback)** — slow but always available.
   - Walks the VM dir via ``session.list_dir`` + ``session.read_bytes``.
   - Triggered automatically on any GCS failure when
     ``fallback_to_cua`` is True (default).

Config is env-var driven by default; explicit ``ArtifactMirrorConfig``
overrides:

    ``ALE_ARTIFACT_GCS_BUCKET``        bucket name (no ``gs://`` prefix)
    ``ALE_ARTIFACT_GCS_KEY_FILE``      service-account JSON path (local-side)
    ``ALE_ARTIFACT_GCS_VM_KEY_FILE``   service-account JSON path on the VM
                                       (presumed pre-staged, e.g. via image bake)

If ``ALE_ARTIFACT_GCS_BUCKET`` is unset, the mirror skips GCS entirely
and always uses CUA direct.
"""
from __future__ import annotations

import asyncio
import dataclasses
import logging
import os
import shlex
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ale.core.cmd_result import cmd_ok, cmd_stderr, cmd_stdout

if TYPE_CHECKING:
    import cua_bench as cb

logger = logging.getLogger(__name__)


# =============================================================================
# Config
# =============================================================================

@dataclasses.dataclass(frozen=True)
class ArtifactMirrorConfig:
    """Where to mirror artifacts to + how."""

    local_root: Path
    """Run dir (e.g. ``.logs/.../<ts>/``). All ``dest_rel`` paths land under here."""

    run_id: str
    """Unique id used as GCS prefix. Typically the RunWriter's run_id."""

    gcs_bucket: str | None = None
    """Bucket name (no ``gs://`` prefix). None → always CUA direct."""

    gcs_local_key_file: str | None = None
    """Service-account JSON on the local machine for ``gsutil cp gs://...``."""

    gcs_vm_key_file: str | None = None
    """Path to a service-account JSON **on the VM** for ``gsutil cp ... gs://...``.
    Presumed pre-staged (baked into the image or uploaded by Runner)."""

    fallback_to_cua: bool = True
    """If GCS fails (network / auth / missing gsutil), fall back to cua direct."""

    @classmethod
    def from_env(
        cls,
        *,
        local_root: Path,
        run_id: str,
    ) -> "ArtifactMirrorConfig":
        return cls(
            local_root=local_root,
            run_id=run_id,
            gcs_bucket=os.environ.get("ALE_ARTIFACT_GCS_BUCKET") or None,
            gcs_local_key_file=os.environ.get("ALE_ARTIFACT_GCS_KEY_FILE") or None,
            gcs_vm_key_file=os.environ.get("ALE_ARTIFACT_GCS_VM_KEY_FILE") or None,
        )


# =============================================================================
# Mirror
# =============================================================================

class ArtifactMirror:
    """Pulls VM directories to local disk via GCS-bridge or CUA direct."""

    def __init__(self, config: ArtifactMirrorConfig):
        self._cfg = config

    async def pull_dir(
        self,
        session: "cb.DesktopSession",
        vm_path: str,
        dest_rel: str,
    ) -> dict[str, Any]:
        """Mirror ``vm_path`` → ``<local_root>/<dest_rel>/``.

        Returns a small status dict:
          ``{"transport": "gcs"|"cua"|"skipped", "files": int, "error": str|None}``
        """
        local_dir = self._cfg.local_root / dest_rel
        local_dir.mkdir(parents=True, exist_ok=True)

        # Pre-flight: does the VM dir even exist?
        try:
            if not await session.exists(vm_path):
                logger.info("artifact_mirror: vm_path %s missing, skipping", vm_path)
                return {"transport": "skipped", "files": 0, "error": None}
        except Exception as exc:                          # noqa: BLE001
            logger.warning("artifact_mirror: exists(%s) failed: %s", vm_path, exc)

        # Try GCS bridge first if configured.
        if self._cfg.gcs_bucket:
            try:
                files = await self._pull_via_gcs(session, vm_path, dest_rel)
                return {"transport": "gcs", "files": files, "error": None}
            except Exception as exc:                      # noqa: BLE001
                if not self._cfg.fallback_to_cua:
                    raise
                logger.warning(
                    "artifact_mirror: GCS bridge failed for %s, falling back to cua: %s",
                    vm_path, exc,
                )

        # CUA direct.
        files = await self._pull_via_cua(session, vm_path, local_dir)
        return {"transport": "cua", "files": files, "error": None}

    # -------------------------------------------------------------- GCS bridge

    def _gs_url(self, dest_rel: str) -> str:
        return f"gs://{self._cfg.gcs_bucket}/{self._cfg.run_id}/{dest_rel}"

    async def _pull_via_gcs(
        self,
        session: "cb.DesktopSession",
        vm_path: str,
        dest_rel: str,
    ) -> int:
        gs_url = self._gs_url(dest_rel)

        # 1. VM-side push: vm_path/* → gs://bucket/<run_id>/<dest_rel>/
        vm_env = ""
        if self._cfg.gcs_vm_key_file:
            vm_env = (
                f"GOOGLE_APPLICATION_CREDENTIALS="
                f"{shlex.quote(self._cfg.gcs_vm_key_file)} "
            )
        push = f"{vm_env}gsutil -m -q cp -r {shlex.quote(vm_path)} {shlex.quote(gs_url)}"
        cr = await session.run_command(push, timeout=600)
        if not cmd_ok(cr):
            raise RuntimeError(
                f"vm gsutil push failed: {cmd_stderr(cr)[:500] or cmd_stdout(cr)[:300]}"
            )

        # 2. Local pull: gs://bucket/<run_id>/<dest_rel>/ → local_dir
        local_dir = self._cfg.local_root / dest_rel
        local_dir.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        if self._cfg.gcs_local_key_file:
            env["GOOGLE_APPLICATION_CREDENTIALS"] = self._cfg.gcs_local_key_file
        argv = ["gsutil", "-m", "-q", "cp", "-r", gs_url + "/*", str(local_dir) + "/"]
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        _, stderr = await asyncio.wait_for(proc.communicate(), timeout=600)
        if proc.returncode != 0:
            raise RuntimeError(
                f"local gsutil pull failed (rc={proc.returncode}): "
                f"{stderr.decode(errors='replace')[:500]}"
            )
        # File count is approximate (recursive walk).
        return sum(1 for p in local_dir.rglob("*") if p.is_file())

    # ------------------------------------------------------------- CUA direct

    async def _pull_via_cua(
        self,
        session: "cb.DesktopSession",
        vm_path: str,
        local_dir: Path,
    ) -> int:
        """Recursive walk via session.list_dir / read_bytes. Slow but reliable."""
        count = 0
        try:
            entries = await session.list_dir(vm_path)
        except Exception as exc:                          # noqa: BLE001
            logger.warning("artifact_mirror[cua]: list_dir(%s) failed: %s", vm_path, exc)
            return 0
        for name in entries:
            remote = self._vm_join(vm_path, name)
            local = local_dir / name
            # Try as file first. If directory, recurse.
            try:
                data = await session.read_bytes(remote)
                local.parent.mkdir(parents=True, exist_ok=True)
                local.write_bytes(data)
                count += 1
            except Exception:                             # noqa: BLE001
                # Probably a directory or unreadable special file. Try recurse.
                if await session.exists(remote):
                    local.mkdir(parents=True, exist_ok=True)
                    count += await self._pull_via_cua(session, remote, local)
        return count

    @staticmethod
    def _vm_join(parent: str, name: str) -> str:
        sep = "\\" if "\\" in parent or parent[1:2] == ":" else "/"
        if parent.endswith(sep):
            return parent + name
        return parent + sep + name
