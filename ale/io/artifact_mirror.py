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
import base64
import dataclasses
import json
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
# Per-file size cap + retry (hardcoded — too trivial to expose)
# =============================================================================

# Files larger than this skip the in-memory read_bytes path and instead
# get head + tail dumps via `dd` so a runaway transcript doesn't OOM the
# cua-server RPC or the local Python process. 50MB is the simprun-era
# observed breaking point for the SSE response body buffer.
_MAX_FILE_BYTES = 50 * 1024 * 1024
_HEAD_TAIL_BYTES = 25 * 1024 * 1024     # 25MB each side when truncating

_READ_RETRIES = 3
_READ_BACKOFF_S = (1.0, 3.0, 9.0)


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
        """Recursive walk via ``session.list_dir`` + ``session.read_bytes``.

        Each file read is 3-retry with 1/3/9s backoff. Files larger than
        ``_MAX_FILE_BYTES`` (50MB) are NOT read via ``read_bytes`` (would
        OOM); instead we dump head+tail via ``dd`` and write a
        ``.truncated`` marker JSON next to it.

        On the recurse-on-failure path: we only recurse when ``read_bytes``
        is suspected to have failed because the entry is a directory.
        Genuine read errors (after 3 retries + size check) get an
        ``.unreadable`` marker, NOT a recurse attempt.
        """
        count = 0
        try:
            entries = await session.list_dir(vm_path)
        except Exception as exc:                          # noqa: BLE001
            logger.warning("artifact_mirror[cua]: list_dir(%s) failed: %s", vm_path, exc)
            return 0
        for name in entries:
            remote = self._vm_join(vm_path, name)
            local = local_dir / name
            # Probe: is this entry a directory? (cheap RPC vs guessing).
            if await _is_dir(session, remote):
                local.mkdir(parents=True, exist_ok=True)
                count += await self._pull_via_cua(session, remote, local)
                continue
            # Regular file: size-gated + retried read.
            try:
                await _pull_one_file(session, remote, local)
                count += 1
            except Exception as exc:                       # noqa: BLE001
                logger.warning(
                    "artifact_mirror[cua]: read failed for %s after retries: %s",
                    remote, exc,
                )
                local.parent.mkdir(parents=True, exist_ok=True)
                local.with_suffix(local.suffix + ".unreadable").write_text(
                    json.dumps({"vm_path": remote, "error": str(exc)}, indent=2)
                )
        return count

    @staticmethod
    def _vm_join(parent: str, name: str) -> str:
        sep = "\\" if "\\" in parent or parent[1:2] == ":" else "/"
        if parent.endswith(sep):
            return parent + name
        return parent + sep + name


# =============================================================================
# Per-file pull helpers (size cap + retry)
# =============================================================================

async def _is_dir(session: "cb.DesktopSession", remote: str) -> bool:
    """Probe whether ``remote`` is a directory. Conservative: returns False
    on any error so the caller falls through to the file-read path
    (which has its own error handling)."""
    try:
        cr = await session.run_command(
            f"if [ -d {shlex.quote(remote)} ]; then echo Y; else echo N; fi",
            timeout=10,
        )
    except Exception:                                   # noqa: BLE001
        return False
    if not cmd_ok(cr):
        return False
    return (cmd_stdout(cr) or "").strip() == "Y"


async def _stat_size_bytes(session: "cb.DesktopSession", remote: str) -> int:
    """Return file size in bytes, or -1 if missing / stat failed."""
    try:
        cr = await session.run_command(
            f"stat -c%s {shlex.quote(remote)} 2>/dev/null || echo -1",
            timeout=10,
        )
    except Exception:                                   # noqa: BLE001
        return -1
    try:
        return int((cmd_stdout(cr) or "-1").strip())
    except ValueError:
        return -1


async def _pull_one_file(
    session: "cb.DesktopSession",
    remote: str,
    local: Path,
) -> None:
    """Pull a single file with size cap + retry.

    Raises on final failure so caller can write an ``.unreadable`` marker.
    """
    size = await _stat_size_bytes(session, remote)

    if size < 0:
        # Treat as non-existent — caller may have raced with rotation.
        raise FileNotFoundError(f"stat failed / missing: {remote}")

    local.parent.mkdir(parents=True, exist_ok=True)

    if size > _MAX_FILE_BYTES:
        # Too big for single read_bytes — head + tail via dd.
        await _pull_head_tail(session, remote, local, size)
        return

    # Normal path: read whole file, 3-retry with backoff.
    last_err: Exception | None = None
    for attempt in range(_READ_RETRIES):
        try:
            data = await session.read_bytes(remote)
            local.write_bytes(data)
            return
        except Exception as exc:                        # noqa: BLE001
            last_err = exc
            if attempt < _READ_RETRIES - 1:
                await asyncio.sleep(_READ_BACKOFF_S[attempt])
    raise RuntimeError(f"read_bytes failed after {_READ_RETRIES} attempts: {last_err}")


async def _pull_head_tail(
    session: "cb.DesktopSession",
    remote: str,
    local: Path,
    total_size: int,
) -> None:
    """For oversized files: dump first 25MB + last 25MB via ``dd``.

    Writes the merged blob to ``local`` with a clear sentinel in between,
    plus a ``<local>.truncated`` marker JSON with the size breakdown.
    """
    head_b64 = await _dd_b64_segment(session, remote, skip=0, count=_HEAD_TAIL_BYTES)
    tail_skip = max(0, total_size - _HEAD_TAIL_BYTES)
    tail_b64 = await _dd_b64_segment(session, remote, skip=tail_skip, count=_HEAD_TAIL_BYTES)

    head_bytes = base64.b64decode(head_b64) if head_b64 else b""
    tail_bytes = base64.b64decode(tail_b64) if tail_b64 else b""

    sentinel = (
        f"\n\n--- ALE_TRUNCATED ({total_size} bytes total; "
        f"{len(head_bytes)} head + {len(tail_bytes)} tail kept) ---\n\n"
    ).encode()
    local.write_bytes(head_bytes + sentinel + tail_bytes)
    local.with_suffix(local.suffix + ".truncated").write_text(json.dumps({
        "vm_path": remote,
        "vm_size_bytes": total_size,
        "kept_head_bytes": len(head_bytes),
        "kept_tail_bytes": len(tail_bytes),
        "max_file_bytes": _MAX_FILE_BYTES,
    }, indent=2))
    logger.warning(
        "artifact_mirror[cua]: truncated %s (%.1f MB) → head/tail %.0f MB each",
        remote, total_size / 1024 / 1024, _HEAD_TAIL_BYTES / 1024 / 1024,
    )


async def _dd_b64_segment(
    session: "cb.DesktopSession",
    remote: str,
    *,
    skip: int,
    count: int,
) -> str:
    """``dd if=remote bs=1 skip=N count=M | base64 -w0`` — single round-trip.

    Returns the base64-encoded segment (no trailing newline). Raises on
    failure with last gsutil-style stderr.
    """
    last_err = ""
    for attempt in range(_READ_RETRIES):
        try:
            cr = await session.run_command(
                f"dd if={shlex.quote(remote)} bs=1 skip={skip} count={count} "
                f"2>/dev/null | base64 -w0",
                timeout=120,
            )
        except Exception as exc:                        # noqa: BLE001
            last_err = f"{type(exc).__name__}: {exc}"
        else:
            if cmd_ok(cr):
                return (cmd_stdout(cr) or "").strip()
            last_err = (cmd_stderr(cr) or cmd_stdout(cr) or "").strip()[:300]
        if attempt < _READ_RETRIES - 1:
            await asyncio.sleep(_READ_BACKOFF_S[attempt])
    raise RuntimeError(f"dd segment failed after {_READ_RETRIES} attempts: {last_err}")
