"""GcloudProvider — ephemeral GCE VMs via ``gcloud compute instances``.

Ported from simprun/vm.py. The free helpers (``_try_create_in_zone``,
``_delete_vm``, ``wait_cua_ready``, ...) remain at module scope; the class
wraps them in the Provider ABC.

The provider config (from yaml ``provider.config``) carries:

  project: str                       # GCP project id
  service_account_key: str           # path to SA JSON; used by `gcloud auth`
  zone: str                          # primary zone
  machine_type: str                  # default machine type
  network: str                       # VPC name
  subnet: str                        # subnet within network
  boot_disk_gb: int                  # boot disk size (default 200)
  data_disk_gb: int                  # extra data disk size (default 200)
  data_disk_type: str                # pd-balanced | pd-ssd | hyperdisk-balanced | auto
  instance_prefix: str               # prefix for generated VM names
  images:                            # snapshot tag → image_name mapping
    cpu-free-ubuntu: agenthle-ubuntu-0505
    cpu-free: agenthle-dev-cpu-free-0505
    ...
"""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import json
import logging
import random
import re
import time
from pathlib import Path
from typing import Any

import requests

from ..images import CapacityProfile, ImageConfig, PoolEntry, capacity_profiles_for
from ..machine_types import is_accelerator_machine_type
from ..remote import _read_first_sse_event
from ...base_interface import EnvSpec, Provider, ReleaseMode, EnvHandle

logger = logging.getLogger(__name__)

_GCP_RETRYABLE_TRANSIENT = [
    "ratelimitexceeded",
    "503",
    "service unavailable",
    "connection reset",
    "connection refused",
    "timed out",
    "deadline exceeded",
]

_GCP_RETRYABLE_ZONE = [
    "quota",
    "resource_exhausted",
    "cpus_per_vm_family",
    "stockout",
    "insufficient",
    "does not have enough resources",
    "not enough resources",
    "zone does not have enough",
]

_GCP_MAX_RETRIES_TRANSIENT = 3
_GCP_TRANSIENT_BASE_DELAY = 15

_CUA_READY_STABLE_SUCCESSES = 2

_LABEL_WHITESPACE_RE = re.compile(r"\s+")
_LABEL_VALUE_INVALID_RE = re.compile(r"[^a-z0-9_-]+")


def sanitize_label_value(value: str, max_length: int = 63) -> str:
    sanitized = _LABEL_WHITESPACE_RE.sub("_", str(value).lower())
    sanitized = _LABEL_VALUE_INVALID_RE.sub("", sanitized)
    return sanitized[:max_length] or "unknown"


# ======================================================================
# Provider config
# ======================================================================


# Default values applied per snapshot when the snapshot entry doesn't override.
@dataclasses.dataclass(frozen=True)
class ProviderDefaults:
    machine_type: str = "e2-highmem-8"
    network: str = "default"
    subnet: str = "default"
    boot_disk_gb: int = 200
    data_disk_gb: int = 200
    data_disk_type: str = "auto"
    zone: str = "us-central1-a"
    fallback_zones: tuple[str, ...] = ()
    os: str = "linux"
    gpu: str | None = None


@dataclasses.dataclass(frozen=True)
class SnapshotConfig:
    """Per-snapshot full image config — mirrors simprun/images.json shape."""

    image_name: str
    zone: str
    os: str
    gpu: str | None
    boot_disk_gb: int
    default_machine_type: str
    network: str
    subnet: str
    data_disk_gb: int
    data_disk_type: str
    fallback_zones: tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class GcloudProviderConfig:
    """gcloud provider config.

    Top-level keys:
        project / service_account_key   GCP creds (required)
        instance_prefix                 VM name prefix
        defaults                        ProviderDefaults — fallback per snapshot
        images                          dict[snapshot -> SnapshotConfig]
        capacity_pools                  dict[snapshot -> tuple[PoolEntry, ...]]
                                        Empty tuple = image-default profile only.
    """

    project: str
    service_account_key: str
    instance_prefix: str = "ale"
    defaults: ProviderDefaults = dataclasses.field(default_factory=ProviderDefaults)
    images: dict[str, SnapshotConfig] = dataclasses.field(default_factory=dict)
    capacity_pools: dict[str, tuple[PoolEntry, ...]] = dataclasses.field(default_factory=dict)


def _build_defaults(raw: dict[str, Any] | None) -> ProviderDefaults:
    raw = raw or {}
    return ProviderDefaults(
        machine_type=str(raw.get("machine_type") or "e2-highmem-8"),
        network=str(raw.get("network") or "default"),
        subnet=str(raw.get("subnet") or "default"),
        boot_disk_gb=int(raw.get("boot_disk_gb") or 200),
        data_disk_gb=int(raw.get("data_disk_gb") or 200),
        data_disk_type=str(raw.get("data_disk_type") or "auto"),
        zone=str(raw.get("zone") or "us-central1-a"),
        fallback_zones=tuple(raw.get("fallback_zones") or ()),
        os=str(raw.get("os") or "linux"),
        gpu=raw.get("gpu"),
    )


def _build_snapshot_config(raw: Any, defaults: ProviderDefaults) -> SnapshotConfig:
    """Accept both shorthand (``image_name`` string) and full dict form.

    Any field unset in a snapshot block falls back to ``defaults``.
    """
    if isinstance(raw, str):
        raw = {"image_name": raw}
    if not isinstance(raw, dict):
        raise TypeError(f"snapshot entry must be string or dict, got {type(raw).__name__}")
    image_name = raw.get("image_name")
    if not image_name:
        raise KeyError("snapshot entry missing required field `image_name`")
    return SnapshotConfig(
        image_name=str(image_name),
        zone=str(raw.get("zone") or defaults.zone),
        os=str(raw.get("os") or defaults.os),
        gpu=raw.get("gpu", defaults.gpu),
        boot_disk_gb=int(raw.get("boot_disk_gb") or defaults.boot_disk_gb),
        default_machine_type=str(raw.get("machine_type") or defaults.machine_type),
        network=str(raw.get("network") or defaults.network),
        subnet=str(raw.get("subnet") or defaults.subnet),
        data_disk_gb=int(raw.get("data_disk_gb") or defaults.data_disk_gb),
        data_disk_type=str(raw.get("data_disk_type") or defaults.data_disk_type),
        fallback_zones=tuple(raw.get("fallback_zones") or defaults.fallback_zones),
    )


def _build_pool_entry(raw: dict[str, Any]) -> PoolEntry:
    return PoolEntry(
        name=str(raw["name"]),
        machine_family=str(raw["machine_family"]),
        default_vcpus=int(raw["default_vcpus"]),
        max_vcpus=int(raw["max_vcpus"]),
        zones=tuple(raw.get("zones") or ()),
        boot_disk_type=str(raw.get("boot_disk_type") or "auto"),
        data_disk_type=str(raw.get("data_disk_type") or "auto"),
        priority=int(raw.get("priority", 100)),
    )


def _build_provider_config(raw: dict[str, Any]) -> GcloudProviderConfig:
    defaults = _build_defaults(raw.get("defaults"))

    images: dict[str, SnapshotConfig] = {}
    for snap, entry in (raw.get("images") or {}).items():
        images[str(snap)] = _build_snapshot_config(entry, defaults)

    pools: dict[str, tuple[PoolEntry, ...]] = {}
    for snap, entries in (raw.get("capacity_pools") or {}).items():
        sorted_pool = sorted(
            (_build_pool_entry(e) for e in (entries or [])),
            key=lambda p: p.priority,
        )
        pools[str(snap)] = tuple(sorted_pool)

    return GcloudProviderConfig(
        project=str(raw["project"]),
        service_account_key=str(raw["service_account_key"]),
        instance_prefix=str(raw.get("instance_prefix") or "ale"),
        defaults=defaults,
        images=images,
        capacity_pools=pools,
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


def _resolve_disk_type(machine_type: str, disk_type: str | None) -> str:
    if not disk_type or disk_type == "auto":
        return _boot_disk_type(machine_type)
    return disk_type


def _build_create_args(
    *,
    name: str,
    image_cfg: ImageConfig,
    machine_type: str,
    zone: str,
    label_str: str,
    project: str,
    boot_disk_type: str,
    data_disk_type: str,
    data_disk_gb: int,
) -> list[str]:
    args = [
        "compute",
        "instances",
        "create",
        name,
        f"--zone={zone}",
        f"--machine-type={machine_type}",
        f"--image={image_cfg.image_name}",
        f"--image-project={project}",
        f"--boot-disk-size={image_cfg.boot_disk_gb}GB",
        f"--boot-disk-type={boot_disk_type}",
        f"--network={image_cfg.network}",
        f"--subnet={image_cfg.subnet}",
        "--tags=ale-run",
        f"--labels={label_str}",
        "--format=json",
    ]
    data_disk_name = name[:58] + "-data"
    args.append(
        f"--create-disk=name={data_disk_name},size={data_disk_gb}GB,"
        f"type={data_disk_type},auto-delete=yes"
    )
    if image_cfg.gpu:
        if not is_accelerator_machine_type(machine_type):
            args.append(f"--accelerator=type={image_cfg.gpu},count=1")
        args.append("--maintenance-policy=TERMINATE")
    if image_cfg.os_type == "windows":
        args.append("--enable-display-device")
    return args


async def _try_create_in_zone(
    *,
    name: str,
    image_cfg: ImageConfig,
    profile: CapacityProfile,
    zone: str,
    label_str: str,
    project: str,
    data_disk_gb: int,
) -> tuple[bool, str, str, str]:
    boot_disk_type = _resolve_disk_type(profile.machine_type, profile.boot_disk_type)
    data_disk_type = _resolve_disk_type(profile.machine_type, profile.data_disk_type)
    args = _build_create_args(
        name=name,
        image_cfg=image_cfg,
        machine_type=profile.machine_type,
        zone=zone,
        label_str=label_str,
        project=project,
        boot_disk_type=boot_disk_type,
        data_disk_type=data_disk_type,
        data_disk_gb=data_disk_gb,
    )
    last_stderr = ""
    for attempt in range(1, _GCP_MAX_RETRIES_TRANSIENT + 1):
        logger.info(
            "Creating VM %s with profile %s in %s (machine=%s attempt %d/%d)",
            name,
            profile.name,
            zone,
            profile.machine_type,
            attempt,
            _GCP_MAX_RETRIES_TRANSIENT,
        )
        rc, stdout, stderr = await _run_gcloud(*args, project=project)
        if rc == 0:
            return True, stdout, "", zone
        last_stderr = stderr

        if _is_zone_capacity_error(stderr):
            logger.warning(
                "Capacity profile %s exhausted in %s: %s",
                profile.name,
                zone,
                stderr[:300],
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
            data = _read_first_sse_event(resp, read_timeout=10)
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

    async def acquire(
        self,
        spec: EnvSpec,
        *,
        exclude_profiles: set[str] | None = None,
    ) -> EnvHandle:
        snap_cfg = self._cfg.images.get(spec.snapshot)
        if snap_cfg is None:
            raise KeyError(
                f"snapshot {spec.snapshot!r} not in provider images map "
                f"(known: {sorted(self._cfg.images)})"
            )
        if spec.os and snap_cfg.os and spec.os != snap_cfg.os:
            logger.warning(
                "os mismatch for %s: task declares %r but snapshot %r is %r",
                spec.snapshot, spec.os, spec.snapshot, snap_cfg.os,
            )

        image_cfg = self._build_image_cfg(snap_cfg, spec.snapshot)
        pool = self._cfg.capacity_pools.get(spec.snapshot, ())
        profiles = capacity_profiles_for(
            image_cfg,
            machine_type_override=None,
            pool=pool,
        )
        if exclude_profiles:
            profiles = tuple(p for p in profiles if p.name not in exclude_profiles)
        if not profiles:
            raise RuntimeError(
                f"no capacity profile resolved for snapshot={spec.snapshot!r} "
                f"(after excluding {sorted(exclude_profiles or ())})"
            )

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

        last_stderr = ""
        used_zone = image_cfg.zone
        used_profile = profiles[0]
        stdout = ""

        logger.info(
            "VM %s capacity candidates: %s",
            name,
            ", ".join(f"{p.name}:{p.machine_type}[{','.join(p.zones)}]" for p in profiles),
        )

        for profile in profiles:
            for zone in profile.zones:
                ok, out, stderr, used_zone = await _try_create_in_zone(
                    name=name,
                    image_cfg=image_cfg,
                    profile=profile,
                    zone=zone,
                    label_str=label_str,
                    project=self._cfg.project,
                    data_disk_gb=snap_cfg.data_disk_gb,
                )
                if ok:
                    stdout = out
                    used_profile = profile
                    break
                last_stderr = stderr
                if not _is_zone_capacity_error(stderr):
                    raise RuntimeError(f"gcloud instances create failed: {stderr}")
            if stdout:
                break
        else:
            raise RuntimeError(f"gcloud instances create failed in all profiles: {last_stderr}")

        try:
            instances = json.loads(stdout)
            inst = instances[0] if isinstance(instances, list) else instances
        except (json.JSONDecodeError, IndexError, KeyError) as e:
            raise RuntimeError(f"Failed to parse gcloud output: {e}\nstdout: {stdout[:500]}") from e

        external_ip = _extract_external_ip(inst)
        if not external_ip:
            external_ip = await _poll_for_ip(name, used_zone, self._cfg.project, timeout=120)

        cua_url = f"http://{external_ip}:5000"
        logger.info("VM %s created via %s in %s at %s", name, used_profile.name, used_zone, cua_url)

        ready = await wait_cua_ready(cua_url, image_cfg.os_type)
        if not ready:
            raise RuntimeError(f"CUA server at {cua_url} did not become ready")

        return EnvHandle(
            id=name,
            endpoint=cua_url,
            os=image_cfg.os_type,
            metadata={
                "zone": used_zone,
                "project": self._cfg.project,
                "machine_type": used_profile.machine_type,
                "capacity_profile": used_profile.name,
                "external_ip": external_ip,
            },
        )

    # ------------------------------------------------------------------ release

    async def release(self, vm: EnvHandle, *, mode: ReleaseMode = "delete") -> None:
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

    def open_session(self, vm: EnvHandle) -> Any:
        from cua_bench.computers.remote import RemoteDesktopSession

        session = RemoteDesktopSession(
            api_url=vm.endpoint,
            os_type=vm.os,
        )
        _init_computer_skip_wait(session)
        return session

    # ------------------------------------------------------------------ helpers

    def _build_image_cfg(self, snap_cfg: SnapshotConfig, snapshot: str) -> ImageConfig:
        return ImageConfig(
            category=snapshot,
            image_name=snap_cfg.image_name,
            default_machine_type=snap_cfg.default_machine_type,
            zone=snap_cfg.zone,
            os_type=snap_cfg.os,
            gpu=snap_cfg.gpu,
            boot_disk_gb=snap_cfg.boot_disk_gb,
            network=snap_cfg.network,
            subnet=snap_cfg.subnet,
            fallback_zones=snap_cfg.fallback_zones,
        )


def gcloud_sa_key_path(config: GcloudProviderConfig) -> Path | None:
    p = Path(config.service_account_key).expanduser() if config.service_account_key else None
    return p if (p and p.exists()) else None
