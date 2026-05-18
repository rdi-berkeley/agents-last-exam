"""GCSDirectProvider — gcloud-backed ephemeral VM provider.

Replaces the old ``simprun`` path: each ``acquire`` shells out to
``gcloud compute instances create`` to bring up a fresh VM from a baked
image, polls ``cua-computer-server:5000/status`` until ready, and returns
a :class:`VMHandle`. ``open_session`` wraps the VM in cua-bench's
:class:`RemoteDesktopSession`.

Scope cuts vs ``simprun/vm.py`` (560 LOC → ~320 LOC):
- **Single zone / single image**. No capacity-profile failover, no zone
  fallback list. Quota / stockout errors raise immediately (operator's
  cue to switch zones).
- **Transient gcloud errors retry with exponential backoff** (15s, 30s,
  60s; max 3 attempts). Patterns ported from simprun.vm._is_transient_error.
- **No force-timeout file plumbing**. ``cancel_external`` is left as a stub
  for the Runner layer (next slice) to implement on top of this Provider.

The image is expected to have ``cua-computer-server`` baked in and started
by the image's systemd unit on boot — no startup-script injection here.
"""
from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import os
import time
import uuid
from typing import TYPE_CHECKING, Any

import httpx

from ale.core.provider import EnvSpec, Provider, ReleaseMode, VMHandle

if TYPE_CHECKING:
    import cua_bench as cb

logger = logging.getLogger(__name__)


# cua-computer-server listens here on the baked image.
CUA_SERVER_PORT = 5000


# =============================================================================
# gcloud transient-error retry policy (ported from simprun/vm.py)
# =============================================================================

# Substrings (lowercased) in `gcloud` stderr that justify a retry. These are
# brief network / API flutters where the next attempt typically succeeds.
_GCP_RETRYABLE_TRANSIENT = (
    "ratelimitexceeded",
    "503",
    "service unavailable",
    "connection reset",
    "connection refused",
    "timed out",
    "deadline exceeded",
)

# Substrings indicating the *zone* is out of capacity (or quota is hit). These
# are NOT transient — retrying the same zone is futile. simprun handles them
# by switching capacity profile / zone; we surface the error so the operator
# can move to a different zone.
_GCP_ZONE_CAPACITY = (
    "quota",
    "resource_exhausted",
    "stockout",
    "insufficient",
    "does not have enough resources",
    "not enough resources",
    "zone does not have enough",
    "cpus_per_vm_family",
)

_GCP_MAX_RETRIES = 3
_GCP_RETRY_BASE_DELAY_S = 15.0  # 15s, 30s, 60s


def _classify_gcloud_error(stderr: str) -> str:
    """Return ``"transient"`` / ``"capacity"`` / ``"fatal"``."""
    lower = stderr.lower()
    if any(pat in lower for pat in _GCP_ZONE_CAPACITY):
        return "capacity"
    if any(pat in lower for pat in _GCP_RETRYABLE_TRANSIENT):
        return "transient"
    return "fatal"


# =============================================================================
# Provider config
# =============================================================================

@dataclasses.dataclass(frozen=True)
class GCSDirectConfig:
    """Static config for the GCS direct provider.

    Everything here is per-Provider-instance, not per-task. Per-task knobs
    live in :class:`EnvSpec` (snapshot, vcpus, memory_gb, ...).
    """

    project: str
    zone: str = "us-west1-b"
    machine_type: str = "e2-standard-4"
    network: str = "agenthle-vpc"
    """Default to ``agenthle-vpc`` — its firewall has port 5000 (cua-server)
    open to 0.0.0.0/0. The standard ``default`` network does NOT, so VMs
    there will look "alive" via SSH but unreachable for cua probing."""
    subnet: str | None = "agenthle-vpc"
    service_account: str | None = None
    scopes: tuple[str, ...] = (
        "https://www.googleapis.com/auth/cloud-platform",
    )
    instance_prefix: str = "ale"
    boot_disk_gb: int = 50
    data_disk_gb: int = 200
    data_disk_type: str = "pd-balanced"
    # Map from snapshot tag → GCE image (or family). The task's task_card.json
    # picks a snapshot; this map translates it to a real image.
    images: dict[str, str] = dataclasses.field(default_factory=dict)
    # Per-zone hint for snapshot selection (most snapshots are global, this is
    # informational; reserved for future per-image zone overrides).

    # Readiness polling
    ready_timeout_s: float = 600.0
    ready_poll_interval_s: float = 10.0
    ready_stable_successes: int = 2

    def resolve_image(self, snapshot: str) -> str:
        if snapshot in self.images:
            return self.images[snapshot]
        # Fall through: assume the snapshot tag IS a valid image family /
        # image name. This lets simple setups skip the map entirely.
        return snapshot


# =============================================================================
# Provider
# =============================================================================

class GCSDirectProvider(Provider):
    """Direct gcloud-based VM lifecycle."""

    def __init__(self, config: GCSDirectConfig):
        if not config.project:
            raise ValueError("GCSDirectConfig.project is required")
        self._cfg = config

    # ---------------------------------------------------------------- acquire

    async def acquire(self, spec: EnvSpec) -> VMHandle:
        instance_name = self._instance_name(spec)
        image = self._cfg.resolve_image(spec.snapshot)

        # 1. gcloud create.
        instance_meta = await self._gcloud_create(instance_name, spec, image)

        # 2. Pull the external IP off the create response.
        external_ip = self._extract_external_ip(instance_meta)
        if not external_ip:
            # Best-effort describe fallback — sometimes create returns before
            # the access config is attached.
            external_ip = await self._describe_external_ip(instance_name)
        if not external_ip:
            await self._best_effort_delete(instance_name)
            raise RuntimeError(
                f"gcloud created {instance_name} but no external IP became available"
            )

        # 3. Wait for cua-computer-server.
        try:
            await self._wait_cua_ready(external_ip, spec.os)
        except Exception:
            await self._best_effort_delete(instance_name)
            raise

        return VMHandle(
            id=instance_name,
            endpoint=f"http://{external_ip}:{CUA_SERVER_PORT}",
            os=spec.os,
            metadata={
                "backend": "gcs_direct",
                "project": self._cfg.project,
                "zone": self._cfg.zone,
                "image": image,
                "external_ip": external_ip,
                "snapshot": spec.snapshot,
            },
        )

    def _instance_name(self, spec: EnvSpec) -> str:
        # GCE names: max 63 chars, lowercase + digits + hyphen.
        ts = int(time.time())
        short = uuid.uuid4().hex[:6]
        snap = spec.snapshot.replace("_", "-").lower()
        name = f"{self._cfg.instance_prefix}-{snap}-{ts}-{short}"
        return name[:63].rstrip("-")

    async def _gcloud_create(
        self, name: str, spec: EnvSpec, image: str,
    ) -> dict[str, Any]:
        """Create one VM, retrying transient gcloud errors with exp backoff.

        Capacity / quota errors are NOT retried — surface them so the
        operator switches zone or profile.
        """
        data_disk = f"{name}-data"
        args = [
            "gcloud", "compute", "instances", "create", name,
            f"--project={self._cfg.project}",
            f"--zone={self._cfg.zone}",
            f"--machine-type={self._cfg.machine_type}",
            f"--image={image}",
            f"--boot-disk-size={self._cfg.boot_disk_gb}GB",
            f"--network={self._cfg.network}",
            (
                f"--create-disk=name={data_disk},size={self._cfg.data_disk_gb}GB,"
                f"type={self._cfg.data_disk_type},auto-delete=yes"
            ),
            "--format=json",
            "--quiet",
        ]
        if self._cfg.subnet:
            args.append(f"--subnetwork={self._cfg.subnet}")
        if self._cfg.service_account:
            args.append(f"--service-account={self._cfg.service_account}")
        if self._cfg.scopes:
            args.append("--scopes=" + ",".join(self._cfg.scopes))

        last_err = ""
        for attempt in range(1, _GCP_MAX_RETRIES + 1):
            logger.info(
                "gcloud create: %s (attempt %d/%d)",
                name, attempt, _GCP_MAX_RETRIES,
            )
            try:
                stdout, _ = await self._run_gcloud(args)
            except RuntimeError as exc:
                stderr_text = str(exc)
                last_err = stderr_text
                kind = _classify_gcloud_error(stderr_text)
                if kind == "capacity":
                    # No point retrying same zone — operator must move zones.
                    raise
                if kind == "transient" and attempt < _GCP_MAX_RETRIES:
                    delay = _GCP_RETRY_BASE_DELAY_S * (2 ** (attempt - 1))
                    logger.warning(
                        "gcloud create transient error (attempt %d/%d): %s "
                        "— retrying in %.0fs",
                        attempt, _GCP_MAX_RETRIES,
                        stderr_text[:200].strip(), delay,
                    )
                    await asyncio.sleep(delay)
                    continue
                # fatal, or last attempt
                raise

            try:
                parsed = json.loads(stdout)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"gcloud create returned non-JSON: {stdout[:500]}"
                ) from exc
            # `--format=json` returns a list with one element.
            return parsed[0] if isinstance(parsed, list) and parsed else {}
        raise RuntimeError(
            f"gcloud create exhausted {_GCP_MAX_RETRIES} attempts: {last_err}"
        )

    @staticmethod
    def _extract_external_ip(meta: dict[str, Any]) -> str | None:
        for nic in meta.get("networkInterfaces", []) or []:
            for ac in nic.get("accessConfigs", []) or []:
                ip = ac.get("natIP")
                if ip:
                    return ip
        return None

    async def _describe_external_ip(self, name: str) -> str | None:
        args = [
            "gcloud", "compute", "instances", "describe", name,
            f"--project={self._cfg.project}",
            f"--zone={self._cfg.zone}",
            "--format=json",
        ]
        # Brief settle delay.
        await asyncio.sleep(5.0)
        try:
            stdout, _ = await self._run_gcloud(args)
            return self._extract_external_ip(json.loads(stdout))
        except Exception:        # noqa: BLE001
            return None

    async def _wait_cua_ready(self, ip: str, os_type: OS) -> None:
        """Probe cua-computer-server by issuing a real ``run_command`` via ``/cmd``.

        Matches agenthle ``simprun.vm.wait_cua_ready``: POST a tiny ``echo ok``
        command, parse the first SSE event, accept when N consecutive
        invocations report ``success`` + ``return_code==0``. This proves the
        cmd pipeline works, not just that the HTTP frontend is up.
        """
        url = f"http://{ip}:{CUA_SERVER_PORT}/cmd"
        probe_cmd = "echo ok" if os_type == "linux" else "cmd /c echo ok"
        payload = {"command": "run_command", "params": {"command": probe_cmd}}
        deadline = time.monotonic() + self._cfg.ready_timeout_s
        consecutive = 0
        last_err = ""
        async with httpx.AsyncClient(timeout=10.0) as client:
            while time.monotonic() < deadline:
                ok, last_err = await self._probe_cua(client, url, payload)
                if ok:
                    consecutive += 1
                    if consecutive >= self._cfg.ready_stable_successes:
                        logger.info("cua-computer-server ready at %s", url)
                        return
                else:
                    consecutive = 0
                await asyncio.sleep(self._cfg.ready_poll_interval_s)
        raise TimeoutError(
            f"cua-computer-server not ready at {url} after "
            f"{self._cfg.ready_timeout_s}s (last error: {last_err})"
        )

    @staticmethod
    async def _probe_cua(
        client: httpx.AsyncClient, url: str, payload: dict[str, Any],
    ) -> tuple[bool, str]:
        """One ``POST /cmd`` probe. Returns (ok, error_summary)."""
        try:
            async with client.stream("POST", url, json=payload) as resp:
                if resp.status_code != 200:
                    return False, f"http_status={resp.status_code}"
                # First SSE ``data: {...}`` line carries the response.
                async for raw_line in resp.aiter_lines():
                    if not raw_line:
                        continue
                    if not raw_line.startswith("data:"):
                        continue
                    try:
                        data = json.loads(raw_line[5:].strip())
                    except json.JSONDecodeError as exc:
                        return False, f"bad_sse_payload: {exc}"
                    if not data.get("success"):
                        return False, str(data.get("error") or data)[:200]
                    rc = int(data.get("return_code", data.get("returncode", 0)) or 0)
                    return rc == 0, f"return_code={rc}" if rc != 0 else ""
                return False, "no SSE data"
        except (httpx.HTTPError, OSError) as exc:
            return False, f"{type(exc).__name__}: {exc}"

    # ---------------------------------------------------------------- release

    async def release(
        self, vm: VMHandle, *, mode: ReleaseMode = "delete",
    ) -> None:
        if mode == "keep":
            logger.info("gcloud release[keep]: %s", vm.id)
            return
        verb = "delete" if mode == "delete" else "stop"
        args = [
            "gcloud", "compute", "instances", verb, vm.id,
            f"--project={self._cfg.project}",
            f"--zone={self._cfg.zone}",
            "--quiet",
        ]
        logger.info("gcloud %s: %s", verb, vm.id)
        try:
            await self._run_gcloud(args)
        except Exception as exc:                   # noqa: BLE001
            # Best-effort: log and swallow. The Runner can re-poll inventory.
            logger.warning("gcloud %s failed for %s: %s", verb, vm.id, exc)

    async def _best_effort_delete(self, name: str) -> None:
        try:
            await self._run_gcloud([
                "gcloud", "compute", "instances", "delete", name,
                f"--project={self._cfg.project}",
                f"--zone={self._cfg.zone}",
                "--quiet",
            ])
        except Exception as exc:                   # noqa: BLE001
            logger.warning("best-effort delete failed for %s: %s", name, exc)

    # ----------------------------------------------------------- open_session

    def open_session(self, vm: VMHandle) -> "cb.DesktopSession":
        import cua_bench as cb
        return cb.computers.remote.RemoteDesktopSession(
            api_url=vm.endpoint,
            os_type=vm.os,
            provider_type="computer",
            headless=True,
            ephemeral=True,
        )

    # ------------------------------------------------------- gcloud subprocess

    async def _run_gcloud(
        self, args: list[str], *, timeout: float = 600.0,
    ) -> tuple[str, str]:
        """Run a gcloud command. Returns ``(stdout, stderr)``; raises on non-zero exit."""
        env = os.environ.copy()
        proc = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        try:
            stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.communicate()
            raise
        stdout = stdout_b.decode("utf-8", errors="replace")
        stderr = stderr_b.decode("utf-8", errors="replace")
        if proc.returncode != 0:
            raise RuntimeError(
                f"gcloud {args[2]} {args[3]} exit={proc.returncode}: "
                f"{stderr[:1000].strip()}"
            )
        return stdout, stderr
