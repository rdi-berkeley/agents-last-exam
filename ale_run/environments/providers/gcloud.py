"""GcloudProvider — ephemeral GCE VMs via ``gcloud compute instances``.

Split of concerns:

* **Framework facts (hardcoded, top of file)**: default machines, the
  C→N2 machine fallback, retry tuning, error classification.
* **Deployment knobs (yaml ``provider.config`` → :class:`GcloudProviderConfig`)**:
  project, service_account_key, instance_prefix, network/subnet, and the
  ``snapshots`` map (logical tag → image + optional gpu + zones).

A task asks for a logical snapshot (``cpu-free`` / ``gpu-free`` / ...);
the provider resolves it via the yaml ``snapshots`` map to a GCE image +
zone list, picks a machine (task-card ``vm.machineType`` override, else
a default, with C→N2 fallback), and tries the zones in order on capacity
errors. Boot disk size comes from the image's baked size; disk *type* is
derived from the machine family (c4/m4/x4 → hyperdisk-balanced, else pd-ssd).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import random
import re
import time
from dataclasses import dataclass, field as dataclass_field
from pathlib import Path
from typing import Any

import requests

from ...base_interface import SandboxSpec, Provider, ReleaseMode, SandboxHandle

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration — framework facts (hardcoded). Snapshot→image + zones are
# deployment-specific and live in the yaml profile (GcloudProviderConfig).
# ============================================================================

# Default machine when a task_card declares no ``vm.machineType``. CPU
# falls back C→N2 (see _machine_chain); GPU has no machine fallback.
_DEFAULT_CPU_MACHINE = "c4-standard-8"
_DEFAULT_GPU_MACHINE = "g2-standard-8"

# VM-create retry tuning.
_GCP_MAX_RETRIES_TRANSIENT = 3
_GCP_TRANSIENT_BASE_DELAY = 15          # seconds, exponential backoff
_CUA_READY_STABLE_SUCCESSES = 2         # consecutive /status oks before "ready"

# stderr substring → error class. transient = retry same zone; zone =
# move to next zone (capacity/quota); anything else = fail fast.
_GCP_RETRYABLE_TRANSIENT = [
    "ratelimitexceeded",
    # transient HTTP 5xx from the GCE API itself (not capacity): retry w/ backoff.
    # 503 was already covered; 500/502 (e.g. "Error 502 (Server Error)") were not,
    # so they previously failed fast with no backoff and no zone fallback.
    "500", "502", "503", "service unavailable", "bad gateway",
    "internal error", "backend error",
    "connection reset", "connection refused", "timed out", "deadline exceeded",
]
_GCP_RETRYABLE_ZONE = [
    "quota", "resource_exhausted", "cpus_per_vm_family", "stockout",
    "insufficient", "does not have enough resources",
    "not enough resources", "zone does not have enough",
]


# ============================================================================
# cua-server SSE helper — local copy (SandboxHandle hasn't been built yet
# at the point where we poll /status, so we can't use its API).
# ============================================================================

def _read_cua_sse_event(resp: requests.Response) -> dict[str, Any] | None:
    """Stream a cua-server response until the first ``data:`` line; parse JSON.

    Mirrors the private helper in base_interface/sandbox.py. Kept inline here
    because GcloudProvider needs to poll the cua-server's ``/status`` endpoint
    BEFORE acquire() returns a SandboxHandle — there's no SandboxHandle yet,
    so the public API isn't available. Timeout is enforced at the call site
    via ``requests.post(stream=True, timeout=...)``."""
    for line in resp.iter_lines(decode_unicode=False):
        if not line:
            continue
        if line.startswith(b"data:"):
            payload = line[len(b"data:"):].strip()
            if payload.startswith(b"\xef\xbb\xbf"):  # strip BOM
                payload = payload[3:]
            try:
                return json.loads(payload)
            except json.JSONDecodeError as e:
                logger.debug("SSE parse failed: %s -- raw=%s", e, payload[:200])
                return None
    return None


# ============================================================================
# GCE machine-type parsing (was ``environments/machine_types.py``)
# ============================================================================

_FAMILY_MEM_PER_VCPU: dict[str, float] = {
    "e2-standard": 4.0,
    "e2-highmem": 8.0,
    "e2-highcpu": 1.0,
    "n1-standard": 3.75,
    "n1-highmem": 6.5,
    "n1-highcpu": 0.9,
    "n2-standard": 4.0,
    "n2-highmem": 8.0,
    "n2-highcpu": 1.0,
    "n2d-standard": 4.0,
    "n2d-highmem": 8.0,
    "n2d-highcpu": 1.0,
    "c2-standard": 4.0,
    "c2d-standard": 4.0,
    "c2d-highmem": 8.0,
    "c2d-highcpu": 2.0,
    "c3-standard": 4.0,
    "c3-highmem": 8.0,
    "c3-highcpu": 2.0,
    "c4-standard": 3.75,
    "c4-highmem": 7.75,
    "c4-highcpu": 2.0,
    "c4a-standard": 4.0,
    "c4a-highmem": 8.0,
    "c4a-highcpu": 2.0,
    "c4d-standard": 4.0,
    "c4d-highmem": 8.0,
    "c4d-highcpu": 2.0,
    "n4-standard": 4.0,
    "n4-highmem": 8.0,
    "n4-highcpu": 2.0,
    "m1-megamem": 14.9,
    "m1-ultramem": 24.0,
    "g2-standard": 4.0,
    "a2-highgpu": 85.0,
    "a2-megagpu": 85.0,
}

_CUSTOM_RE = re.compile(r"^(?P<family>[a-z]\w+)-custom-(?P<vcpus>\d+)-(?P<mem_mb>\d+)$")
_STANDARD_RE = re.compile(r"^(?P<family>[a-z]\w+-\w+)-(?P<vcpus>\d+)$")
_ACCEL_RE = re.compile(r"^(?P<family>[a-z]\w+-\w+)-(?P<count>\d+)g$")


@dataclass(frozen=True)
class _GCEShape:
    vcpus: int
    memory_gb: int
    machine_type: str


def _parse_gce_machine_type(machine_type: str | None) -> _GCEShape | None:
    if not machine_type:
        return None
    mt = machine_type.strip().lower()
    m = _CUSTOM_RE.match(mt)
    if m:
        vcpus = int(m.group("vcpus"))
        mem_mb = int(m.group("mem_mb"))
        return _GCEShape(vcpus=vcpus, memory_gb=max(1, mem_mb // 1024), machine_type=machine_type)
    m = _ACCEL_RE.match(mt)
    if m:
        family = m.group("family")
        count = int(m.group("count"))
        mem_ratio = _FAMILY_MEM_PER_VCPU.get(family)
        if mem_ratio is not None:
            vcpus = count * 12
            return _GCEShape(
                vcpus=vcpus, memory_gb=max(1, int(count * mem_ratio)),
                machine_type=machine_type,
            )
        logger.warning("unknown accelerator family %r in %r", family, machine_type)
        return None
    m = _STANDARD_RE.match(mt)
    if m:
        family = m.group("family")
        vcpus = int(m.group("vcpus"))
        mem_ratio = _FAMILY_MEM_PER_VCPU.get(family)
        if mem_ratio is not None:
            return _GCEShape(
                vcpus=vcpus, memory_gb=max(1, int(vcpus * mem_ratio)),
                machine_type=machine_type,
            )
        logger.warning("unknown GCE family %r in %r", family, machine_type)
        return None
    logger.warning("unparseable GCE machine type: %r", machine_type)
    return None


def _is_accelerator_machine_type(machine_type: str) -> bool:
    return _ACCEL_RE.match(machine_type.strip().lower()) is not None


# ============================================================================
# Machine-type fallback chain
# ============================================================================


def _cpu_family_fallback(machine_type: str) -> str | None:
    """C-family → N2 fallback, keeping the ``type-size`` suffix.

    ``c4-standard-8`` → ``n2-standard-8``. Returns None for non-C families
    (they're already a fallback tier, or GPU). This is the only machine
    fallback we do: C runs out of stock far more often than N.
    """
    gen, _, rest = machine_type.partition("-")
    if gen[:1].lower() == "c" and rest:
        return f"n2-{rest}"
    return None


def _machine_chain(machine_type: str, *, is_gpu: bool) -> tuple[str, ...]:
    """Ordered machine types to try for one VM.

    * GPU: just the requested G-family machine — no machine-type fallback.
    * CPU: the requested machine, then its N2 fallback (``c*`` → ``n2-*``).

    Zone fallback (across the snapshot's zone list) is orthogonal and
    applied by the caller for each machine in this chain.
    """
    if is_gpu:
        return (machine_type,)
    fb = _cpu_family_fallback(machine_type)
    return (machine_type, fb) if fb else (machine_type,)


# ============================================================================
# Label sanitization
# ============================================================================

_LABEL_WHITESPACE_RE = re.compile(r"\s+")
_LABEL_VALUE_INVALID_RE = re.compile(r"[^a-z0-9_-]+")


def sanitize_label_value(value: str, max_length: int = 63) -> str:
    sanitized = _LABEL_WHITESPACE_RE.sub("_", str(value).lower())
    sanitized = _LABEL_VALUE_INVALID_RE.sub("", sanitized)
    return sanitized[:max_length] or "unknown"


# ======================================================================
# Provider config
# ======================================================================


@dataclass(frozen=True)
class SnapshotConfig:
    """What a logical snapshot tag maps to (yaml ``snapshots.<tag>``)."""

    image: str          # GCE image name (= framework Image-registry family)
    gpu: str | None     # accelerator type; None for CPU snapshots
    zones: tuple[str, ...]   # zones to try, in order, on capacity errors

    @property
    def os(self) -> str:
        return "windows" if "win" in self.image.lower() else "linux"


@dataclass(frozen=True)
class GcloudProviderConfig:
    """gcloud provider config (yaml ``provider.config``).

        project / service_account_key   GCP creds (required)
        instance_prefix                 VM name prefix
        network / subnet                VPC the VMs attach to (must expose
                                        port 5000)
        snapshots                       dict[snapshot tag → SnapshotConfig]

    Machine-type selection (task_card override → default → C→N2 fallback)
    and boot-disk-type are framework facts hardcoded in the provider.
    """

    project: str
    service_account_key: str
    instance_prefix: str = "ale"
    network: str = "default"
    subnet: str = "default"
    gcs_sa_key: str = ""
    """Host path to a GCS SA key. Injected into each VM (its baked gsutil is
    unauthenticated) so in-VM staging + gs:// pulls work; the key's project_id
    also bills requester-pays buckets via gsutil ``-u``. Mirrors the docker
    provider's same-named knob."""
    snapshots: dict[str, SnapshotConfig] = dataclass_field(default_factory=dict)


def _build_snapshot_config(raw: Any) -> SnapshotConfig:
    if not isinstance(raw, dict):
        raise TypeError(f"snapshot entry must be a mapping, got {type(raw).__name__}")
    image = raw.get("image")
    if not image:
        raise KeyError(f"snapshot entry missing required `image`: {raw!r}")
    zones = tuple(raw.get("zones") or ())
    if not zones:
        raise KeyError(f"snapshot {image!r} missing required `zones`")
    return SnapshotConfig(image=str(image), gpu=raw.get("gpu"), zones=zones)


def _build_provider_config(raw: dict[str, Any]) -> GcloudProviderConfig:
    snapshots = {
        str(tag): _build_snapshot_config(entry)
        for tag, entry in (raw.get("snapshots") or {}).items()
    }
    return GcloudProviderConfig(
        project=str(raw["project"]),
        service_account_key=str(raw["service_account_key"]),
        instance_prefix=str(raw.get("instance_prefix") or "ale"),
        network=str(raw.get("network") or "default"),
        subnet=str(raw.get("subnet") or "default"),
        gcs_sa_key=str(raw.get("gcs_sa_key") or ""),
        snapshots=snapshots,
    )


# ======================================================================
# Free helpers (ported from simprun/vm.py)
# ======================================================================


def generate_vm_name(
    prefix: str,
    *,
    snapshot: str,
    task_id: str = "",
    harness: str = "",
    model_tag: str = "",
) -> str:
    """Generate a GCP-safe VM name: ``<prefix>-<task-or-snapshot>-<hash8>``.

    Matches simprun's naming so leftover VMs are greppable by task. When
    ``task_id`` is set we use the task slug (40 chars max) — same as
    simprun. When it's empty (legacy callers, smoke tests) we fall back
    to the snapshot tag so the name still encodes something meaningful.
    ``harness`` / ``model_tag`` are only mixed into the hash seed for
    collision avoidance in batch runs; they don't appear in the name.
    """
    if task_id:
        body = re.sub(r"[^a-z0-9]", "-", task_id.lower()).strip("-")[:40]
    else:
        body = re.sub(r"[^a-z0-9]", "-", snapshot.lower()).strip("-")[:30]
    seed = f"{prefix}:{task_id}:{harness}:{model_tag}:{snapshot}:{time.time()}:{random.random()}"
    h = hashlib.sha256(seed.encode()).hexdigest()[:8]
    name = f"{prefix}-{body}-{h}"
    return name[:63]


async def _run_gcloud(*args: str, project: str) -> tuple[int, str, str]:
    cmd = ["gcloud", *args, f"--project={project}"]
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout_b, stderr_b = await proc.communicate()
    return (
        proc.returncode or 0,
        stdout_b.decode(errors="replace"),
        stderr_b.decode(errors="replace"),
    )


def _is_transient_error(stderr: str) -> bool:
    lower = stderr.lower()
    return any(pat in lower for pat in _GCP_RETRYABLE_TRANSIENT)


def _is_zone_capacity_error(stderr: str) -> bool:
    lower = stderr.lower()
    return any(pat in lower for pat in _GCP_RETRYABLE_ZONE)


def _boot_disk_type(machine_type: str) -> str:
    family = machine_type.split("-")[0].lower()
    if family in ("c4", "m4", "x4"):
        return "hyperdisk-balanced"
    return "pd-ssd"


def _build_create_args(
    *,
    name: str,
    image: str,
    gpu: str | None,
    os_type: str,
    network: str,
    subnet: str,
    machine_type: str,
    zone: str,
    label_str: str,
    project: str,
    boot_disk_type: str,
) -> list[str]:
    """``gcloud compute instances create`` argv.

    Boot disk size is NOT passed — we use the GCE image's baked size.
    """
    args = [
        "compute",
        "instances",
        "create",
        name,
        f"--zone={zone}",
        f"--machine-type={machine_type}",
        f"--image={image}",
        f"--image-project={project}",
        f"--boot-disk-type={boot_disk_type}",
        f"--network={network}",
        f"--subnet={subnet}",
        "--tags=ale-run",
        f"--labels={label_str}",
        "--format=json",
    ]
    if gpu:
        if not _is_accelerator_machine_type(machine_type):
            args.append(f"--accelerator=type={gpu},count=1")
        args.append("--maintenance-policy=TERMINATE")
    if os_type == "windows":
        args.append("--enable-display-device")
    return args


async def _try_create_in_zone(
    *,
    name: str,
    image: str,
    gpu: str | None,
    os_type: str,
    network: str,
    subnet: str,
    machine_type: str,
    zone: str,
    label_str: str,
    project: str,
) -> tuple[bool, str, str, str]:
    args = _build_create_args(
        name=name,
        image=image,
        gpu=gpu,
        os_type=os_type,
        network=network,
        subnet=subnet,
        machine_type=machine_type,
        zone=zone,
        label_str=label_str,
        project=project,
        boot_disk_type=_boot_disk_type(machine_type),
    )
    last_stderr = ""
    for attempt in range(1, _GCP_MAX_RETRIES_TRANSIENT + 1):
        logger.info(
            "Creating VM %s with machine=%s in %s (attempt %d/%d)",
            name, machine_type, zone, attempt, _GCP_MAX_RETRIES_TRANSIENT,
        )
        rc, stdout, stderr = await _run_gcloud(*args, project=project)
        if rc == 0:
            return True, stdout, "", zone
        last_stderr = stderr

        if _is_zone_capacity_error(stderr):
            logger.warning(
                "machine=%s exhausted in %s: %s",
                machine_type, zone, stderr[:300],
            )
            return False, "", last_stderr, zone

        if attempt < _GCP_MAX_RETRIES_TRANSIENT and _is_transient_error(stderr):
            delay = _GCP_TRANSIENT_BASE_DELAY * (2 ** (attempt - 1))
            logger.warning(
                "VM create transient error (attempt %d/%d): %s — retrying in %ds",
                attempt,
                _GCP_MAX_RETRIES_TRANSIENT,
                stderr[:200],
                delay,
            )
            await asyncio.sleep(delay)
            continue

        return False, "", last_stderr, zone
    return False, "", last_stderr, zone


def _extract_external_ip(inst: dict) -> str | None:
    for iface in inst.get("networkInterfaces", []):
        for ac in iface.get("accessConfigs", []):
            ip = ac.get("natIP")
            if ip:
                return ip
    return None


async def _poll_for_ip(name: str, zone: str, project: str, timeout: float = 120) -> str:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        rc, stdout, _ = await _run_gcloud(
            "compute",
            "instances",
            "describe",
            name,
            f"--zone={zone}",
            "--format=json",
            project=project,
        )
        if rc == 0:
            try:
                inst = json.loads(stdout)
                ip = _extract_external_ip(inst)
                if ip:
                    return ip
            except (json.JSONDecodeError, KeyError):
                pass
        await asyncio.sleep(5)
    raise RuntimeError(f"Timed out waiting for external IP on {name}")


def _probe_cua(cua_url: str, payload: dict) -> tuple[bool, str]:
    try:
        with requests.post(
            f"{cua_url}/cmd",
            json=payload,
            timeout=10,
            stream=True,
        ) as resp:
            if resp.status_code != 200:
                return False, f"status={resp.status_code}"
            data = _read_cua_sse_event(resp)
        if _cua_command_succeeded(data):
            return True, ""
        return False, _summarize_cua_response(data)
    except Exception as e:
        return False, str(e)


async def wait_cua_ready(
    cua_url: str,
    os_type: str,
    timeout: float = 600,
    poll_interval: float = 10,
) -> bool:
    cmd = "echo ok" if os_type == "linux" else "cmd /c echo ok"
    payload = {"command": "run_command", "params": {"command": cmd}}
    deadline = time.monotonic() + timeout
    last_err = ""
    successes = 0
    while time.monotonic() < deadline:
        ok, err = await asyncio.to_thread(_probe_cua, cua_url, payload)
        if ok:
            successes += 1
            if successes >= _CUA_READY_STABLE_SUCCESSES:
                logger.info("CUA server ready at %s", cua_url)
                return True
        else:
            successes = 0
            last_err = err
            logger.debug("CUA not ready at %s: %s", cua_url, last_err)
        await asyncio.sleep(poll_interval)
    logger.error("CUA server at %s did not become ready within %ss: %s", cua_url, timeout, last_err)
    return False


def _cua_command_succeeded(data: dict | None) -> bool:
    if not data or data.get("success") is not True:
        return False
    return int(data.get("return_code", data.get("returncode", 0)) or 0) == 0


def _summarize_cua_response(data: dict | None) -> str:
    if data is None:
        return "no SSE response"
    err = data.get("error") or data.get("stderr") or data.get("message")
    if err:
        return str(err)[:300]
    return json.dumps(data, default=str)[:300]


async def _delete_vm(name: str, zone: str, project: str) -> bool:
    logger.info("Deleting VM %s", name)
    rc, _, stderr = await _run_gcloud(
        "compute",
        "instances",
        "delete",
        name,
        f"--zone={zone}",
        "--quiet",
        project=project,
    )
    if rc != 0:
        logger.error("Failed to delete VM %s: %s", name, stderr)
        return False
    logger.info("VM %s deleted", name)
    return True


async def _stop_vm(name: str, zone: str, project: str) -> bool:
    logger.info("Stopping VM %s", name)
    rc, _, stderr = await _run_gcloud(
        "compute",
        "instances",
        "stop",
        name,
        f"--zone={zone}",
        "--quiet",
        project=project,
    )
    if rc != 0:
        logger.error("Failed to stop VM %s: %s", name, stderr)
        return False
    logger.info("VM %s stopped", name)
    return True


# ======================================================================
# Session bring-up: replicates simprun TaskEnv's _init_computer_skip_wait.
# wait_cua_ready already confirmed the CUA server is healthy, so we skip
# the fragile Computer.wait_for_ready() that breaks under concurrency.
# ======================================================================


def _init_computer_skip_wait(session: Any) -> None:
    from computer import Computer
    from computer.interface.factory import InterfaceFactory

    computer = Computer(
        os_type=session._os_type,
        use_host_computer_server=True,
        api_host=session._api_host,
        api_port=session._api_port,
        noVNC_port=session._vnc_port,
    )

    interface = InterfaceFactory.create_interface_for_os(
        os=session._os_type,
        ip_address=session._api_host,
        api_port=session._api_port,
    )
    computer._interface = interface
    computer._original_interface = interface
    computer._initialized = True

    session._computer = computer
    session._initialized = True


# ======================================================================
# Provider
# ======================================================================


class GcloudProvider(Provider):
    """Provider backed by ``gcloud compute instances create / delete``."""

    def __init__(self, config: GcloudProviderConfig | dict[str, Any]):
        if isinstance(config, dict):
            config = _build_provider_config(config)
        self._cfg = config

    @property
    def config(self) -> GcloudProviderConfig:
        return self._cfg

    # ------------------------------------------------------------------ acquire

    async def acquire(self, spec: SandboxSpec) -> SandboxHandle:
        snap = self._cfg.snapshots.get(spec.snapshot)
        if snap is None:
            raise KeyError(
                f"snapshot {spec.snapshot!r} not in provider config "
                f"(known: {sorted(self._cfg.snapshots)})"
            )
        if spec.os and spec.os != snap.os:
            logger.warning(
                "os mismatch for %s: task declares %r but image %r is %r",
                spec.snapshot, spec.os, snap.image, snap.os,
            )

        is_gpu = snap.gpu is not None
        zones = snap.zones

        # Machine fallback: task-card override (or default) → N2 (CPU only).
        base_machine = spec.machine_type or (
            _DEFAULT_GPU_MACHINE if is_gpu else _DEFAULT_CPU_MACHINE
        )
        machines = _machine_chain(base_machine, is_gpu=is_gpu)

        name = generate_vm_name(
            self._cfg.instance_prefix,
            snapshot=spec.snapshot,
            task_id=spec.task_id,
            harness=spec.harness,
            model_tag=spec.model_tag,
        )
        label_str = ",".join(
            f"{k}={sanitize_label_value(v)}"
            for k, v in {"purpose": "ale-run", "snapshot": spec.snapshot}.items()
        )

        logger.info(
            "VM %s candidates: machines=%s zones=%s",
            name, list(machines), list(zones),
        )

        last_stderr = ""
        used_zone = zones[0]
        used_machine = machines[0]
        stdout = ""

        # machine × zone, in order: each machine tried across all zones first.
        for machine in machines:
            for zone in zones:
                ok, out, stderr, used_zone = await _try_create_in_zone(
                    name=name,
                    image=snap.image,
                    gpu=snap.gpu,
                    os_type=snap.os,
                    network=self._cfg.network,
                    subnet=self._cfg.subnet,
                    machine_type=machine,
                    zone=zone,
                    label_str=label_str,
                    project=self._cfg.project,
                )
                if ok:
                    stdout = out
                    used_machine = machine
                    break
                last_stderr = stderr
                if not _is_zone_capacity_error(stderr):
                    raise RuntimeError(f"gcloud instances create failed: {stderr}")
            if stdout:
                break
        else:
            raise RuntimeError(
                f"gcloud instances create failed for all machines/zones: {last_stderr}"
            )

        try:
            instances = json.loads(stdout)
            inst = instances[0] if isinstance(instances, list) else instances
        except (json.JSONDecodeError, IndexError, KeyError) as e:
            raise RuntimeError(
                f"Failed to parse gcloud output: {e}\nstdout: {stdout[:500]}"
            ) from e

        external_ip = _extract_external_ip(inst)
        if not external_ip:
            external_ip = await _poll_for_ip(
                name, used_zone, self._cfg.project, timeout=120,
            )

        # GCE image name == framework Image-registry family name. Fetch it
        # early so the cua-server URL uses the image's declared port instead
        # of a hard-coded literal.
        from ..images import get as get_image

        image = get_image(snap.image)

        cua_url = f"http://{external_ip}:{image.cua_server_port}"
        logger.info(
            "VM %s created via %s in %s at %s",
            name, used_machine, used_zone, cua_url,
        )

        ready = await wait_cua_ready(cua_url, snap.os)
        if not ready:
            raise RuntimeError(f"CUA server at {cua_url} did not become ready")

        # The VM's baked gsutil is unauthenticated. Inject the GCS SA key so
        # in-VM staging (stage_reference) and gs:// pulls authenticate; surface
        # the key path + billing project via metadata so gsbucket's _gsutil
        # adds `-o gs_service_key_file` + `-u <project>`.
        gcs_key_path, gcs_user_project = "", ""
        if self._cfg.gcs_sa_key:
            try:
                gcs_key_path, gcs_user_project = await self._inject_gcs_credentials(
                    cua_url, image.os, self._cfg.gcs_sa_key,
                )
            except Exception as e:  # noqa: BLE001
                logger.error("gcloud: GCS credential injection failed on %s: %s", name, e)

        return SandboxHandle(
            id=name,
            endpoint=cua_url,
            os=image.os,
            **image.sandbox_paths(),
            metadata={
                "zone": used_zone,
                "project": self._cfg.project,
                "machine_type": used_machine,
                "external_ip": external_ip,
                "image": image.name,
                "snapshot": spec.snapshot,
                "gcs_key_path": gcs_key_path,
                "gcs_user_project": gcs_user_project,
            },
        )

    @staticmethod
    async def _inject_gcs_credentials(
        cua_url: str, os_type: str, host_key_path: str,
    ) -> tuple[str, str]:
        """Push the GCS SA key into the VM and return (vm_key_path, project_id).

        gsbucket's ``_gsutil`` reads ``metadata['gcs_key_path']`` and adds
        ``-o Credentials:gs_service_key_file=<path>`` so the VM's gsutil
        authenticates as the SA (instead of anonymous). The SA key's
        ``project_id`` bills requester-pays buckets via ``-u``.
        """
        from cua_bench.computers.remote import RemoteDesktopSession

        key = Path(host_key_path).expanduser()
        data = key.read_bytes()
        try:
            project_id = json.loads(data).get("project_id", "") or ""
        except (json.JSONDecodeError, AttributeError):
            project_id = ""

        is_win = os_type != "linux"
        dest = r"C:\agenthle\gcs-reader.json" if is_win else "/tmp/agenthle/gcs-reader.json"
        mk = (
            r'cmd /c if not exist C:\agenthle mkdir C:\agenthle'
            if is_win else "mkdir -p /tmp/agenthle"
        )

        session = RemoteDesktopSession(api_url=cua_url, os_type=os_type)
        _init_computer_skip_wait(session)
        await session.run_command(mk, check=False)
        await session.write_bytes(dest, data)
        logger.info("gcloud: injected GCS SA key -> %s (project=%s)", dest, project_id)
        return dest, project_id

    # ------------------------------------------------------------------ release

    async def release(self, vm: SandboxHandle, *, mode: ReleaseMode = "delete") -> None:
        zone = vm.metadata.get("zone")
        if not zone:
            logger.warning("VM %s has no zone in metadata; skipping %s", vm.id, mode)
            return
        if mode == "delete":
            await _delete_vm(vm.id, zone, self._cfg.project)
        elif mode == "stop":
            await _stop_vm(vm.id, zone, self._cfg.project)
        elif mode == "keep":
            logger.info("VM %s kept alive (mode=keep)", vm.id)
        else:
            raise ValueError(f"unknown release mode: {mode!r}")

    # ------------------------------------------------------------------ session

    def open_session(self, vm: SandboxHandle) -> Any:
        from cua_bench.computers.remote import RemoteDesktopSession

        session = RemoteDesktopSession(
            api_url=vm.endpoint,
            os_type=vm.os,
        )
        _init_computer_skip_wait(session)
        return session


def gcloud_sa_key_path(config: GcloudProviderConfig) -> Path | None:
    p = Path(config.service_account_key).expanduser() if config.service_account_key else None
    return p if (p and p.exists()) else None
