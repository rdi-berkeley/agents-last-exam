"""AliyunProvider — ephemeral Alibaba Cloud ECS instances via ``aliyun ecs
RunInstances``.

Mirror of :mod:`ale_run.environments.providers.aws` (and, through it,
:mod:`ale_run.environments.providers.gcloud`), adapted to Alibaba Cloud ECS:

* **Framework facts (hardcoded, top of file)**: default instance types, the
  C→G instance fallback, retry tuning, error classification.
* **Deployment knobs (yaml ``provider.config`` → :class:`AliyunProviderConfig`)**:
  region, security group, optional key pair + instance RAM role, instance_prefix,
  internet bandwidth, system-disk category, and the ``snapshots`` map (logical
  tag → custom image + optional gpu + zones).

A task asks for a logical snapshot (``cpu-free-ubuntu`` / ...); the provider
resolves it via the yaml ``snapshots`` map to a custom-image id + a zone list,
picks an instance type (task-card ``vm.machineType`` override, else a default,
with C→G family fallback), and tries the zones in order on capacity errors.
Each zone is resolved to a vSwitch in that zone within the security group's VPC.
Root disk size comes from the image's baked system-disk snapshot (no override).

Why this looks like aws.py but isn't shared: the lifecycle, retry ordering
(instance-type × zone), readiness poll, Windows-resolution and DesktopSession
bring-up are identical in *shape*, so the genuinely cloud-agnostic helpers
(``wait_cua_ready``, ``_init_computer_skip_wait``, ``sanitize_label_value``,
``_SET_RES_PY``) are imported from gcloud.py rather than duplicated. Everything
that shells out to ``aliyun`` lives here.

Credentials: the provider uses the ambient Alibaba Cloud credential chain
(``aliyun configure`` profile / ``ALIBABA_CLOUD_ACCESS_KEY_ID`` +
``ALIBABA_CLOUD_ACCESS_KEY_SECRET`` env vars), so — like the aws provider — no
key is named in the yaml. The in-box ``ossutil`` authenticates to OSS via the
instance **RAM role** attached at launch (``ram_role_name``), the Alibaba
analogue of AWS's IAM instance profile — so there is nothing to inject for
oss:// task data / output.
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
from typing import Any

from ...base_interface import SandboxSpec, Provider, ReleaseMode, SandboxHandle
# Cloud-agnostic helpers shared with the gcloud/aws providers.
from .gcloud import (
    wait_cua_ready,
    _init_computer_skip_wait,
    sanitize_label_value,
    _SET_RES_PY,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration — framework facts (hardcoded). Snapshot→image + zones are
# deployment-specific and live in the yaml profile (AliyunProviderConfig).
# ============================================================================

# Default instance when a task_card declares no ``vm.machineType``. CPU falls
# back C→G (see _instance_chain); GPU has no instance fallback.
_DEFAULT_CPU_INSTANCE = "ecs.g7.2xlarge"          # 8 vCPU / 32 GiB, general (UEFI ✓)
# GPU tasks are NOT supported on Alibaba (the imported UEFI image's GPU driver
# can't be made to bind to Alibaba's vGPU — see the PR for the full analysis).
# This default is the closest launchable type (A10 vGPU vWS, UEFI-capable) and
# is kept only so the wiring stays complete; a GPU run will not produce a working
# GPU. Must be a UEFI-boot type — the common BIOS-only ``ecs.gn7i-c8g1`` is
# rejected outright against the UEFI image.
_DEFAULT_GPU_INSTANCE = "ecs.vgn7i-vws-m8.2xlarge"   # A10 vGPU vWS, UEFI-capable

# Launch retry tuning.
_ALIYUN_MAX_RETRIES_TRANSIENT = 3
_ALIYUN_TRANSIENT_BASE_DELAY = 15          # seconds, exponential backoff
# Delete retry tuning — a just-created box rejects DeleteInstance while
# ``Initializing``; retry through that window so we never leak a billable box.
_DELETE_MAX_RETRIES = 6
_DELETE_RETRY_DELAY = 12                   # seconds

# error-code/message substring → error class. transient = retry same zone;
# zone = move to next zone (capacity); anything else = fail fast.
#
# IMPORTANT: classify on the API **ErrorCode** (extracted by _error_code), NOT
# the raw stderr. The aliyun CLI wraps every API failure as
# ``ERROR: SDK.ServerError`` regardless of the underlying code, so matching the
# wrapper string (e.g. a naive "servererror" substring) would mark EVERY error —
# even a hard ``InvalidAccountStatus.NotEnoughBalance`` or ``InvalidImageId`` —
# transient and burn the full retry budget on it. These patterns are matched
# against the real ``ErrorCode:`` value; genuine transport errors (which carry no
# ErrorCode line) are caught by _TRANSIENT_TRANSPORT against the full stderr.
_ALIYUN_RETRYABLE_TRANSIENT = [          # matched against ErrorCode
    "throttling", "requestlimitexceeded", "requestthrottled", "rate exceeded",
    "internalerror", "serviceunavailable", "servicetimeout",
]
_TRANSIENT_TRANSPORT = [                  # matched against full stderr (no ErrorCode)
    "connection reset", "connection refused", "timed out", "timeout",
    "could not connect", "connection aborted", "i/o timeout", "no such host",
]
_ALIYUN_RETRYABLE_ZONE = [               # matched against ErrorCode + message
    "operationdenied.nostock", "nostock", "out of stock",
    "resource.notenough", "resourcenotavailable", "operationdenied.noschedule",
    "zonenotsupported", "zone_not_sufficient", "zonenotsupportedforspotinstance",
    "no capacity", "not have capacity", "instancediskcategorylimitexceeded",
]


# ============================================================================
# Provider config
# ============================================================================


@dataclass(frozen=True)
class SnapshotConfig:
    """What a logical snapshot tag maps to (yaml ``snapshots.<tag>``).

    Exactly parallel to aws/gcloud's SnapshotConfig: ``image`` is the
    **image-family name** (the same registry key gcloud/aws use — ``ale-ubuntu22``
    / ``ale-win10`` — from :mod:`ale_run.environments.images`), NOT a raw
    ``m-...`` image id. The provider resolves that family to a concrete custom
    image id at acquire time by looking up a self-owned image tagged
    ``ale:image-family=<name>`` (override with an explicit ``image_id:`` when you
    want to pin one). One ``ale-win10.py`` registry entry thus serves gcloud
    (GCE image name), aws (family→AMI lookup) and aliyun (family→image lookup).

    ``zones`` holds **Alibaba zone ids** (e.g. ``cn-hangzhou-i``) to try in order
    on capacity errors — the ECS analogue of aws's AZ list. The provider maps
    each zone to a vSwitch in that zone within the security group's VPC.
    """

    image: str               # image-family name (registry key), e.g. "ale-win10"
    gpu: str | None          # truthy → use a GPU instance type; None for CPU
    zones: tuple[str, ...]   # zone ids to try, in order, on capacity errors
    image_id: str | None = None   # optional explicit image id override (m-...)
    resolution: tuple[int, int] | None = None
    """Windows display resolution (w, h) forced after boot. See gcloud's
    SnapshotConfig for the full rationale; Linux ignores it."""

    @property
    def os(self) -> str:
        # image is the family name (e.g. ale-win10), so this is stable even when
        # an explicit image_id override is set.
        return "windows" if "win" in self.image.lower() else "linux"


@dataclass(frozen=True)
class AliyunProviderConfig:
    """aliyun provider config (yaml ``provider.config``).

        region                    Alibaba region id, e.g. ``cn-hangzhou`` (hardcoded
                                  in the yaml, like gcloud's zones)
        security_group            the firewall — a security-group NAME (``ale-sandbox``)
                                  or ``sg-...`` id. The provider resolves the name to
                                  its id AND the VPC it lives in, which scopes the
                                  zone→vSwitch lookup. The gcloud analogue is ``network``.
        instance_prefix           instance Name prefix
        key_name                  ECS key pair name (optional; we reach the box via
                                  cua-server, not SSH, so usually unset)
        ram_role_name             instance RAM role granting in-box OSS access
                                  (optional; only for oss:// task data / output) —
                                  the Alibaba analogue of AWS's IAM instance profile
        internet_max_bandwidth_out  outbound public bandwidth in Mbit/s. > 0 makes
                                  ECS auto-assign a public IP (default 100); 0 keeps
                                  the box private (then the private IP is used)
        system_disk_category      system-disk type for the boot volume (default
                                  ``cloud_essd``); size comes from the image
        instance_charge_type      ``PostPaid`` (pay-as-you-go, default) or ``PrePaid``
        snapshots                 dict[snapshot tag → SnapshotConfig]

    Credentials are NOT here — the provider uses the ambient Alibaba Cloud
    credential chain (``aliyun configure`` / ``ALIBABA_CLOUD_ACCESS_KEY_*`` env),
    so unlike gcloud no project/key is named in the yaml. Everything is a
    hardcoded convention (resolved by name/tag at run time); nothing needs
    ``${env:...}``.
    """

    region: str
    security_group: str = "ale-sandbox"
    instance_prefix: str = "ale"
    key_name: str = ""
    ram_role_name: str = ""
    internet_max_bandwidth_out: int = 100
    system_disk_category: str = "cloud_essd"
    instance_charge_type: str = "PostPaid"
    output_to_bucket: bool = False
    """Set by the config loader when ``output_path`` is an ``oss://`` bucket. It
    turns a missing ``ram_role_name`` from a tolerated skip into a hard error —
    without the role the in-box ossutil can't upload, so a bucket output would
    silently lose every run's files."""
    snapshots: dict[str, SnapshotConfig] = dataclass_field(default_factory=dict)

    @property
    def associate_public_ip(self) -> bool:
        return self.internet_max_bandwidth_out > 0


def _build_snapshot_config(raw: Any) -> SnapshotConfig:
    if not isinstance(raw, dict):
        raise TypeError(f"snapshot entry must be a mapping, got {type(raw).__name__}")
    image = raw.get("image")
    if not image:
        raise KeyError(
            f"snapshot entry missing required `image` (image-family name, "
            f"e.g. ale-win10): {raw!r}"
        )
    zones = tuple(raw.get("zones") or ())
    if not zones:
        raise KeyError(f"snapshot {image!r} missing required `zones` (zone ids)")
    return SnapshotConfig(
        image=str(image), gpu=raw.get("gpu"), zones=zones,
        image_id=(str(raw["image_id"]) if raw.get("image_id") else None),
        resolution=_parse_resolution(raw.get("resolution"), image),
    )


def _parse_resolution(raw: Any, image: str) -> tuple[int, int] | None:
    if raw is None:
        return None
    if isinstance(raw, str):
        parts = raw.lower().replace(" ", "").split("x")
    elif isinstance(raw, (list, tuple)):
        parts = list(raw)
    else:
        raise TypeError(
            f"snapshot {image!r} resolution must be [w, h] or 'WxH', got {raw!r}"
        )
    if len(parts) != 2:
        raise ValueError(f"snapshot {image!r} resolution must have 2 values, got {raw!r}")
    try:
        return (int(parts[0]), int(parts[1]))
    except (TypeError, ValueError):
        raise ValueError(f"snapshot {image!r} resolution values must be ints, got {raw!r}")


def _build_provider_config(raw: dict[str, Any]) -> AliyunProviderConfig:
    snapshots = {
        str(tag): _build_snapshot_config(entry)
        for tag, entry in (raw.get("snapshots") or {}).items()
    }
    return AliyunProviderConfig(
        region=str(raw["region"]),
        security_group=str(raw.get("security_group") or "ale-sandbox"),
        instance_prefix=str(raw.get("instance_prefix") or "ale"),
        key_name=str(raw.get("key_name") or ""),
        ram_role_name=str(raw.get("ram_role_name") or ""),
        internet_max_bandwidth_out=int(raw.get("internet_max_bandwidth_out", 100)),
        system_disk_category=str(raw.get("system_disk_category") or "cloud_essd"),
        instance_charge_type=str(raw.get("instance_charge_type") or "PostPaid"),
        output_to_bucket=bool(raw.get("output_to_bucket", False)),
        snapshots=snapshots,
    )


# ============================================================================
# Instance-type fallback chain
# ============================================================================


def _cpu_family_fallback(instance_type: str) -> str | None:
    """C-family → G fallback, keeping the size suffix.

    ``ecs.c7.2xlarge`` → ``ecs.g7.2xlarge``. Returns None for non-C families.
    This is the only instance fallback we do: compute-optimized capacity stocks
    out far more often than general-purpose. The ``ecs.`` prefix and the size
    suffix are preserved; only the family letter ``c`` → ``g`` changes.
    """
    m = re.fullmatch(r"(ecs\.)c(\w*?)(\..+)", instance_type)
    if m:
        return f"{m.group(1)}g{m.group(2)}{m.group(3)}"
    return None


def _instance_chain(instance_type: str, *, is_gpu: bool) -> tuple[str, ...]:
    """Ordered instance types to try for one box.

    * GPU: just the requested gn-family type — no fallback.
    * CPU: the requested type, then its G fallback (``ecs.c*`` → ``ecs.g*``).
    """
    if is_gpu:
        return (instance_type,)
    fb = _cpu_family_fallback(instance_type)
    return (instance_type, fb) if fb else (instance_type,)


# ============================================================================
# Naming
# ============================================================================


def generate_instance_name(
    prefix: str,
    *,
    snapshot: str,
    task_id: str = "",
    harness: str = "",
    model_tag: str = "",
) -> str:
    """``<prefix>-<task-or-snapshot>-<hash8>`` instance name (mirror of aws)."""
    if task_id:
        body = re.sub(r"[^a-z0-9]", "-", task_id.lower()).strip("-")[:40]
    else:
        body = re.sub(r"[^a-z0-9]", "-", snapshot.lower()).strip("-")[:30]
    seed = f"{prefix}:{task_id}:{harness}:{model_tag}:{snapshot}:{time.time()}:{random.random()}"
    h = hashlib.sha256(seed.encode()).hexdigest()[:8]
    # ECS InstanceName must start with a letter and be ≤ 128 chars.
    name = f"{prefix}-{body}-{h}"[:128]
    return name if name[:1].isalpha() else f"a{name}"[:128]


# ============================================================================
# aliyun CLI wrapper + error classification
# ============================================================================


async def _run_aliyun(product: str, action: str, *args: str) -> tuple[int, str, str]:
    """Invoke ``aliyun <product> <action> [args...]``.

    The aliyun CLI prints the raw JSON API response to stdout on success, so —
    unlike the aws CLI — no ``--output json`` flag is needed. ``--region`` /
    ``--RegionId`` are passed by the caller (region is on every ECS action).
    """
    cmd = ["aliyun", product, action, *args]
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


def _error_code(stderr: str) -> str:
    """Extract the API ``ErrorCode`` from the aliyun CLI's error block.

    The CLI prints e.g. ``ERROR: SDK.ServerError\\nErrorCode: <code>\\n...``; we
    classify on ``<code>`` so the generic ``SDK.ServerError`` wrapper never
    drives retry decisions. Returns "" when no ErrorCode line is present (a pure
    transport error)."""
    m = re.search(r"ErrorCode:\s*([A-Za-z0-9_.\-]+)", stderr)
    return m.group(1).lower() if m else ""


def _is_transient_error(stderr: str) -> bool:
    code = _error_code(stderr)
    if code:
        return any(pat in code for pat in _ALIYUN_RETRYABLE_TRANSIENT)
    # No ErrorCode line → a transport/network failure; scan the raw stderr.
    lower = stderr.lower()
    return any(pat in lower for pat in _TRANSIENT_TRANSPORT)


def _is_zone_capacity_error(stderr: str) -> bool:
    code = _error_code(stderr)
    haystack = code if code else stderr.lower()
    return any(pat in haystack for pat in _ALIYUN_RETRYABLE_ZONE)


# ============================================================================
# RunInstances argv
# ============================================================================


def _build_run_args(
    *,
    name: str,
    image_id: str,
    instance_type: str,
    zone_id: str,
    vswitch_id: str,
    security_group_id: str,
    cfg: AliyunProviderConfig,
    snapshot_tag: str,
    ram_role_name: str,
) -> list[str]:
    """``aliyun ecs RunInstances`` argv.

    System-disk size is NOT passed — we use the image's baked system-disk
    snapshot size. ``InternetMaxBandwidthOut > 0`` makes ECS auto-assign a
    public IP; tags mark the box for fleet cleanup (mirror of aws's tags).
    ``ram_role_name`` is the *effective* role (resolved + existence-checked by
    the caller), not ``cfg.ram_role_name`` — empty means attach none.
    """
    args = [
        "--RegionId", cfg.region,
        "--ZoneId", zone_id,
        "--ImageId", image_id,
        "--InstanceType", instance_type,
        "--Amount", "1",
        "--VSwitchId", vswitch_id,
        "--SecurityGroupId", security_group_id,
        "--InstanceName", name,
        "--InstanceChargeType", cfg.instance_charge_type,
        "--SystemDisk.Category", cfg.system_disk_category,
        "--InternetMaxBandwidthOut", str(cfg.internet_max_bandwidth_out),
        # tags: purpose=ale-run + snapshot=<tag> (fleet-cleanup selectors).
        "--Tag.1.Key", "purpose", "--Tag.1.Value", "ale-run",
        "--Tag.2.Key", "snapshot",
        "--Tag.2.Value", sanitize_label_value(snapshot_tag),
    ]
    if cfg.key_name:
        args += ["--KeyPairName", cfg.key_name]
    if ram_role_name:
        args += ["--RamRoleName", ram_role_name]
    return args


async def _try_run_in_zone(
    *,
    name: str,
    image_id: str,
    instance_type: str,
    zone_id: str,
    vswitch_id: str,
    security_group_id: str,
    cfg: AliyunProviderConfig,
    snapshot_tag: str,
    ram_role_name: str,
) -> tuple[bool, str, str]:
    """Returns (ok, stdout, stderr). On capacity error returns ok=False without
    retrying (caller moves to the next zone); transient errors retry here."""
    args = _build_run_args(
        name=name,
        image_id=image_id,
        instance_type=instance_type,
        zone_id=zone_id,
        vswitch_id=vswitch_id,
        security_group_id=security_group_id,
        cfg=cfg,
        snapshot_tag=snapshot_tag,
        ram_role_name=ram_role_name,
    )
    last_stderr = ""
    for attempt in range(1, _ALIYUN_MAX_RETRIES_TRANSIENT + 1):
        logger.info(
            "Launching %s type=%s in %s (attempt %d/%d)",
            name, instance_type, zone_id, attempt, _ALIYUN_MAX_RETRIES_TRANSIENT,
        )
        rc, stdout, stderr = await _run_aliyun("ecs", "RunInstances", *args)
        if rc == 0:
            return True, stdout, ""
        last_stderr = stderr

        if _is_zone_capacity_error(stderr):
            logger.warning(
                "type=%s capacity-exhausted in %s: %s",
                instance_type, zone_id, stderr[:300],
            )
            return False, "", last_stderr

        if _is_transient_error(stderr) and attempt < _ALIYUN_MAX_RETRIES_TRANSIENT:
            delay = _ALIYUN_TRANSIENT_BASE_DELAY * (2 ** (attempt - 1))
            logger.warning(
                "RunInstances transient error (attempt %d/%d): %s — retrying in %ds",
                attempt, _ALIYUN_MAX_RETRIES_TRANSIENT, stderr[:200], delay,
            )
            await asyncio.sleep(delay)
            continue

        return False, "", last_stderr
    return False, "", last_stderr


# ============================================================================
# describe / lifecycle helpers
# ============================================================================


def _instance_id_from_run(stdout: str) -> str:
    """RunInstances returns ``{"InstanceIdSets":{"InstanceIdSet":["i-..."]}}``."""
    data = json.loads(stdout)
    return data["InstanceIdSets"]["InstanceIdSet"][0]


def _first_ip(block: Any) -> str | None:
    """Pull the first ip out of aliyun's ``{"IpAddress": ["1.2.3.4"]}`` shape."""
    if isinstance(block, dict):
        ips = block.get("IpAddress")
        if isinstance(ips, list) and ips:
            return str(ips[0])
    return None


async def _wait_running_with_ip(
    instance_id: str, cfg: AliyunProviderConfig, *, want_public: bool,
    timeout: float = 240,
) -> str:
    """Poll DescribeInstances until the instance is ``Running`` and has an IP.

    Returns the public IP (want_public) or the private IP. Raises on timeout.
    Public IPs auto-assigned via InternetMaxBandwidthOut land in
    ``PublicIpAddress.IpAddress``; the VPC private IP is in
    ``VpcAttributes.PrivateIpAddress.IpAddress``."""
    deadline = time.monotonic() + timeout
    last_state = "?"
    nudged = False        # whether we've issued a one-shot StartInstance
    while time.monotonic() < deadline:
        rc, stdout, stderr = await _run_aliyun(
            "ecs", "DescribeInstances",
            "--RegionId", cfg.region,
            "--InstanceIds", json.dumps([instance_id]),
        )
        if rc == 0:
            try:
                inst = json.loads(stdout)["Instances"]["Instance"][0]
                last_state = inst.get("Status", "?")
                if last_state == "Running":
                    if want_public:
                        ip = _first_ip(inst.get("PublicIpAddress"))
                    else:
                        ip = _first_ip(
                            (inst.get("VpcAttributes") or {}).get("PrivateIpAddress")
                        )
                    if ip:
                        return ip
                # ``Deleted`` is the only terminal failure. A freshly-created ECS
                # instance briefly reports ``Stopped`` right after RunInstances
                # before its auto-start transitions it Starting→Running, so we do
                # NOT treat Stopped as terminal (doing so killed healthy boots).
                # If it lingers Stopped, nudge it once with StartInstance in case
                # auto-start didn't fire, then keep polling until Running/timeout.
                elif last_state == "Deleted":
                    raise RuntimeError(
                        f"instance {instance_id} entered Deleted before becoming reachable"
                    )
                elif last_state == "Stopped" and not nudged:
                    nudged = True
                    await _run_aliyun(
                        "ecs", "StartInstance",
                        "--RegionId", cfg.region, "--InstanceId", instance_id,
                    )
            except (KeyError, IndexError, json.JSONDecodeError):
                pass
        await asyncio.sleep(5)
    raise RuntimeError(
        f"timed out waiting for {instance_id} to be Running with an IP "
        f"(last state={last_state})"
    )


async def _delete(instance_id: str, cfg: AliyunProviderConfig) -> bool:
    """Force-delete the instance (``--Force true`` releases a Running box without
    a separate stop).

    A box still in its post-RunInstances ``Initializing`` window rejects
    DeleteInstance with ``IncorrectInstanceStatus.Initializing``; since a failed
    delete leaks a billable instance, retry through that window. Delete is the
    last safety net against billing leaks, so we also retry throttle/transport
    (transient) errors — only a genuine hard error (bad id, auth) breaks early."""
    logger.info("Deleting instance %s", instance_id)
    last_stderr = ""
    for attempt in range(_DELETE_MAX_RETRIES):
        rc, _, stderr = await _run_aliyun(
            "ecs", "DeleteInstance",
            "--RegionId", cfg.region,
            "--InstanceId", instance_id,
            "--Force", "true",
        )
        if rc == 0:
            return True
        last_stderr = stderr
        code = _error_code(stderr)
        # retry the Initializing window + any transient (throttle/transport) error;
        # bail only on a hard error we can't recover from.
        if "incorrectinstancestatus" not in code and not _is_transient_error(stderr):
            break
        await asyncio.sleep(_DELETE_RETRY_DELAY)
    logger.error("Failed to delete %s: %s", instance_id, last_stderr)
    return False


async def _stop(instance_id: str, cfg: AliyunProviderConfig) -> bool:
    logger.info("Stopping instance %s", instance_id)
    rc, _, stderr = await _run_aliyun(
        "ecs", "StopInstance",
        "--RegionId", cfg.region,
        "--InstanceId", instance_id,
    )
    if rc != 0:
        logger.error("Failed to stop %s: %s", instance_id, stderr)
        return False
    return True


def _parse_supported_modes(out: str) -> list[tuple[int, int]]:
    """Extract ``(w, h)`` modes from the resolution script's
    ``failed:<rc> supported=[(800, 600), ...]`` line."""
    marker = "supported="
    i = out.find(marker)
    if i < 0:
        return []
    import ast
    try:
        val = ast.literal_eval(out[i + len(marker):].strip())
        return [(int(a), int(b)) for a, b in val]
    except (ValueError, SyntaxError, TypeError):
        return []


# ============================================================================
# Provider
# ============================================================================


class AliyunProvider(Provider):
    """Provider backed by ``aliyun ecs RunInstances / DeleteInstance``."""

    def __init__(self, config: AliyunProviderConfig | dict[str, Any]):
        if isinstance(config, dict):
            config = _build_provider_config(config)
        self._cfg = config
        self._image_cache: dict[str, str] = {}    # family name → image id
        self._vswitch_cache: dict[str, str] = {}  # zone id → vswitch id
        self._sg: tuple[str, str] | None = None   # (sg_id, vpc_id), resolved once
        self._ram_role: str | None = None         # effective role name, resolved once

    @property
    def config(self) -> AliyunProviderConfig:
        return self._cfg

    # ----------------------------------------------------------- resolution

    async def _effective_ram_role(self) -> str:
        """The RAM role to actually attach — resolved + existence-checked once.

        Lets the config ship a default ``ram_role_name`` (e.g. ``ale-sandbox``)
        that a user who only runs with ``output_path: null``/``local`` never has
        to create: if the role doesn't exist we simply don't attach it (a bare
        ``--RamRoleName`` for a missing role is a hard ``InvalidRamRole.NotFound``
        at RunInstances). BUT if ``output_path`` is an ``oss://`` bucket the role
        is mandatory (the in-box ossutil needs it to upload), so a missing role
        is a fail-fast error instead of a silent skip."""
        if self._ram_role is not None:
            return self._ram_role
        name = self._cfg.ram_role_name
        if not name:
            self._ram_role = ""
            return ""
        rc, _, _ = await _run_aliyun("ram", "GetRole", "--RoleName", name)
        if rc == 0:
            self._ram_role = name
        elif self._cfg.output_to_bucket:
            raise RuntimeError(
                f"output_path is an oss:// bucket but RAM role {name!r} does not "
                f"exist — create it (see the Alibaba setup docs) so the sandbox "
                f"can upload output, or unset output_path"
            )
        else:
            logger.warning(
                "RAM role %r not found — launching without it (output_path isn't "
                "a bucket, so no in-box OSS access is needed)", name,
            )
            self._ram_role = ""
        return self._ram_role

    async def _resolve_image(self, snap: SnapshotConfig) -> str:
        """Resolve a snapshot's image to a concrete custom-image id.

        Order: explicit ``image_id:`` override → cached → look up a self-owned
        image tagged ``ale:image-family=<snap.image>`` (newest wins). This is the
        Alibaba analogue of aws resolving a family to an AMI.
        """
        if snap.image_id:
            return snap.image_id
        family = snap.image
        if family in self._image_cache:
            return self._image_cache[family]
        rc, out, err = await _run_aliyun(
            "ecs", "DescribeImages",
            "--RegionId", self._cfg.region,
            "--ImageOwnerAlias", "self",
            "--Tag.1.Key", "ale:image-family",
            "--Tag.1.Value", family,
        )
        image_id = ""
        if rc == 0:
            try:
                images = json.loads(out)["Images"]["Image"]
                # newest first by CreationTime (ISO-8601 strings sort lexically).
                images.sort(key=lambda im: im.get("CreationTime", ""), reverse=True)
                if images:
                    image_id = str(images[0]["ImageId"])
            except (KeyError, IndexError, json.JSONDecodeError):
                image_id = ""
        if not image_id:
            raise RuntimeError(
                f"no image tagged ale:image-family={family!r} in {self._cfg.region} "
                f"(import one, tag it, or set an explicit `image_id:` on the snapshot). "
                f"{(err or '').strip()[:200]}"
            )
        self._image_cache[family] = image_id
        return image_id

    async def _resolve_security_group(self) -> tuple[str, str]:
        """Resolve ``config.security_group`` (a name or ``sg-...`` id) to its id
        AND the VPC it lives in. The VPC scopes the zone→vSwitch lookup, so the
        yaml only names the firewall (like gcloud names ``network``) — no
        ids/env-vars. Resolved once and cached."""
        if self._sg is not None:
            return self._sg
        ref = self._cfg.security_group
        sel = (["--SecurityGroupId", ref] if ref.startswith("sg-")
               else ["--SecurityGroupName", ref])
        rc, out, err = await _run_aliyun(
            "ecs", "DescribeSecurityGroups",
            "--RegionId", self._cfg.region, *sel,
        )
        sg = None
        if rc == 0:
            try:
                groups = json.loads(out)["SecurityGroups"]["SecurityGroup"]
                if groups:
                    sg = (str(groups[0]["SecurityGroupId"]), str(groups[0]["VpcId"]))
            except (KeyError, IndexError, json.JSONDecodeError):
                sg = None
        if sg is None or not sg[0]:
            raise RuntimeError(
                f"security group {ref!r} not found in {self._cfg.region} "
                f"(create it — see the Alibaba setup docs). {(err or '').strip()[:200]}"
            )
        self._sg = sg
        return self._sg

    async def _vswitch_for_zone(self, zone_id: str, vpc_id: str) -> str:
        """Resolve a zone id to a vSwitch id in that zone within ``vpc_id`` (the
        security group's VPC). Prefers vSwitches tagged ``project=ale``; falls
        back to any vSwitch in the zone/VPC. Cached per zone."""
        if zone_id in self._vswitch_cache:
            return self._vswitch_cache[zone_id]
        base = [
            "--RegionId", self._cfg.region,
            "--ZoneId", zone_id,
            "--VpcId", vpc_id,
        ]
        rc, out, err = await _run_aliyun("ecs", "DescribeVSwitches", *base)
        vswitch = ""
        if rc == 0:
            try:
                vsws = json.loads(out)["VSwitches"]["VSwitch"]
                tagged = [
                    v for v in vsws
                    if any(
                        t.get("TagKey") == "project" and t.get("TagValue") == "ale"
                        for t in ((v.get("Tags") or {}).get("Tag") or [])
                    )
                ]
                pick = (tagged or vsws)
                pick.sort(key=lambda v: v.get("VSwitchId", ""))
                if pick:
                    vswitch = str(pick[0]["VSwitchId"])
            except (KeyError, IndexError, json.JSONDecodeError):
                vswitch = ""
        if not vswitch:
            raise RuntimeError(
                f"no vSwitch in zone {zone_id} (vpc={vpc_id}): {(err or '').strip()[:200]}"
            )
        self._vswitch_cache[zone_id] = vswitch
        return vswitch

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
                "os mismatch for %s: task declares %r but image %r looks %r",
                spec.snapshot, spec.os, snap.image, snap.os,
            )

        is_gpu = snap.gpu is not None
        zones = snap.zones

        base_instance = spec.machine_type or (
            _DEFAULT_GPU_INSTANCE if is_gpu else _DEFAULT_CPU_INSTANCE
        )
        instances = _instance_chain(base_instance, is_gpu=is_gpu)

        # Resolve the image-family name (e.g. "ale-win10") to a concrete image id,
        # and the security-group name to its id + the VPC that scopes vSwitches.
        image_id = await self._resolve_image(snap)
        sg_id, vpc_id = await self._resolve_security_group()
        # RAM role to attach: the configured name if it exists, else "" (skip) —
        # unless output_path is a bucket, in which case a missing role raises.
        ram_role = await self._effective_ram_role()

        name = generate_instance_name(
            self._cfg.instance_prefix,
            snapshot=spec.snapshot,
            task_id=spec.task_id,
            harness=spec.harness,
            model_tag=spec.model_tag,
        )

        logger.info(
            "instance %s candidates: image=%s types=%s zones=%s",
            name, image_id, list(instances), list(zones),
        )

        last_stderr = ""
        stdout = ""
        used_zone = zones[0]
        used_vswitch = ""
        used_instance = instances[0]

        # instance-type × zone, in order: each type tried across all zones. Each
        # zone is resolved to a vSwitch in that zone within the configured VPC.
        for instance_type in instances:
            for zone_id in zones:
                try:
                    vswitch = await self._vswitch_for_zone(zone_id, vpc_id)
                except Exception as e:  # noqa: BLE001
                    last_stderr = f"no vSwitch for zone {zone_id}: {e}"
                    logger.warning("%s", last_stderr)
                    continue
                ok, out, stderr = await _try_run_in_zone(
                    name=name,
                    image_id=image_id,
                    instance_type=instance_type,
                    zone_id=zone_id,
                    vswitch_id=vswitch,
                    security_group_id=sg_id,
                    cfg=self._cfg,
                    snapshot_tag=spec.snapshot,
                    ram_role_name=ram_role,
                )
                if ok:
                    stdout = out
                    used_vswitch = vswitch
                    used_zone = zone_id
                    used_instance = instance_type
                    break
                last_stderr = stderr
                if not _is_zone_capacity_error(stderr):
                    raise RuntimeError(f"aliyun RunInstances failed: {stderr}")
            if stdout:
                break
        else:
            raise RuntimeError(
                f"aliyun RunInstances failed for all types/zones: {last_stderr}"
            )

        try:
            instance_id = _instance_id_from_run(stdout)
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            raise RuntimeError(
                f"failed to parse RunInstances output: {e}\nstdout: {stdout[:500]}"
            ) from e

        # Instance now exists. Until the handle is returned, ANY failure leaks
        # it (the caller's cleanup only runs once it holds the handle), so
        # delete-on-failure here before propagating.
        try:
            public_ip = await _wait_running_with_ip(
                instance_id, self._cfg,
                want_public=self._cfg.associate_public_ip,
            )

            # snap.image IS the registry family name (same key gcloud/aws use), so
            # this resolves the in-box paths/port directly — one registry entry
            # serves all providers.
            from ..images import get as get_image
            image = get_image(snap.image)

            cua_url = f"http://{public_ip}:{image.cua_server_port}"
            logger.info(
                "instance %s (%s) launched as %s in %s (%s) at %s",
                name, instance_id, used_instance, used_zone, used_vswitch, cua_url,
            )

            # Windows first-boot on an imported image (driver init + login +
            # cua autostart) can exceed the 10-min Linux default; give it 20.
            cua_timeout = 1200 if snap.os == "windows" else 600
            ready = await wait_cua_ready(cua_url, snap.os, timeout=cua_timeout)
            if not ready:
                raise RuntimeError(f"CUA server at {cua_url} did not become ready")

            if image.os == "windows" and snap.resolution is not None:
                await self._set_windows_resolution(cua_url, snap.resolution)

            return SandboxHandle(
                id=instance_id,
                endpoint=cua_url,
                os=image.os,
                **image.sandbox_paths(),
                metadata={
                    "region": self._cfg.region,
                    "instance_id": instance_id,
                    "instance_type": used_instance,
                    "zone": used_zone,
                    "vswitch": used_vswitch,
                    "public_ip": public_ip,
                    "image": image.name,
                    "image_id": image_id,
                    "snapshot": spec.snapshot,
                    "name": name,
                },
            )
        except BaseException:
            logger.warning(
                "acquire: post-launch failure on %s (%s) — deleting to avoid leak",
                name, instance_id,
            )
            try:
                await _delete(instance_id, self._cfg)
            except Exception as de:  # noqa: BLE001
                logger.error("acquire: could not delete leaked %s: %s", instance_id, de)
            raise

    @staticmethod
    async def _set_windows_resolution(
        cua_url: str, resolution: tuple[int, int],
    ) -> None:
        """Force the Windows framebuffer to ``resolution`` (w, h), falling back to
        the largest supported mode if the exact one isn't available.

        Like the aws provider, an imported Windows image's display adapter may
        expose a different (often smaller) mode set than GCE. Hard-failing the
        whole run over that is wrong (it's a capability difference, not a
        misconfig), so we pick the best supported mode ≤ the request (else the
        largest available) and LOG it loudly — a logged, root-caused choice, not
        a silent mask. Only raise if the adapter reports no settable modes at
        all."""
        from cua_bench.computers.remote import RemoteDesktopSession

        w, h = resolution
        remote_path = r"C:\agenthle\_set_resolution.py"
        session = RemoteDesktopSession(api_url=cua_url, os_type="windows")
        _init_computer_skip_wait(session)
        await session.run_command(
            r"cmd /c if not exist C:\agenthle mkdir C:\agenthle", check=False,
        )
        await session.write_file(remote_path, _SET_RES_PY)

        async def _try(tw: int, th: int) -> str:
            res = await session.run_command(f'python "{remote_path}" {tw} {th}', check=False)
            return (res.get("stdout") or "").strip() if isinstance(res, dict) else ""

        out = await _try(w, h)
        if "set_ok" in out:
            logger.info("aliyun: set Windows resolution to %dx%d", w, h)
            return

        # Parse the adapter's supported modes ("...supported=[(800, 600), ...]")
        # and retry with the best fit instead of aborting the run.
        modes = _parse_supported_modes(out)
        if modes:
            below = [m for m in modes if m[0] <= w and m[1] <= h]
            best = max(below or modes, key=lambda m: m[0] * m[1])
            out2 = await _try(best[0], best[1])
            if "set_ok" in out2:
                logger.warning(
                    "aliyun: requested Windows resolution %dx%d unsupported on this "
                    "display adapter (supported=%s) — using %dx%d instead",
                    w, h, modes, best[0], best[1],
                )
                return
        err = out or (out and "no output")
        raise RuntimeError(
            f"aliyun: could not set any Windows resolution (requested {w}x{h}); "
            f"adapter output: {err or 'no output'}"
        )

    # ------------------------------------------------------------------ release

    async def release(self, vm: SandboxHandle, *, mode: ReleaseMode = "delete") -> None:
        instance_id = vm.metadata.get("instance_id") or vm.id
        if mode == "delete":
            await _delete(instance_id, self._cfg)
        elif mode == "stop":
            await _stop(instance_id, self._cfg)
        elif mode == "keep":
            logger.info("instance %s kept alive (mode=keep)", instance_id)
        else:
            raise ValueError(f"unknown release mode: {mode!r}")

    # ------------------------------------------------------------------ session

    def open_session(self, vm: SandboxHandle) -> Any:
        from cua_bench.computers.remote import RemoteDesktopSession

        session = RemoteDesktopSession(api_url=vm.endpoint, os_type=vm.os)
        _init_computer_skip_wait(session)
        return session
