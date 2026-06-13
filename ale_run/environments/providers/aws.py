"""AwsProvider — ephemeral EC2 instances via ``aws ec2 run-instances``.

Mirror of :mod:`ale_run.environments.providers.gcloud`, adapted to EC2:

* **Framework facts (hardcoded, top of file)**: default instance types, the
  C→M instance fallback, retry tuning, error classification.
* **Deployment knobs (yaml ``provider.config`` → :class:`AwsProviderConfig`)**:
  region, subnet(s), security group(s), optional key pair + IAM instance
  profile, instance_prefix, and the ``snapshots`` map (logical tag → AMI +
  optional gpu + subnets/AZs).

A task asks for a logical snapshot (``cpu-free-ubuntu`` / ...); the provider
resolves it via the yaml ``snapshots`` map to an AMI + subnet list, picks an
instance type (task-card ``vm.machineType`` override, else a default, with
C→M family fallback), and tries the subnets in order on capacity errors. Root
volume size comes from the AMI's baked snapshot (no override).

Why this looks like gcloud.py but isn't shared: the lifecycle, retry ordering
(instance-type × subnet), readiness poll, Windows-resolution and DesktopSession
bring-up are identical in *shape*, so the genuinely cloud-agnostic helpers
(``wait_cua_ready``, ``_init_computer_skip_wait``, ``sanitize_label_value``,
``_set_windows_resolution`` machinery) are imported from gcloud.py rather than
duplicated. Everything that shells out to ``aws`` lives here.

Credentials: unlike GCE (whose baked gsutil is anonymous, so the provider
pushes an SA key into each VM), EC2 instances authenticate via an **IAM
instance profile** attached at launch — the in-box ``aws`` CLI is then
authenticated with nothing to inject. So there is no AWS counterpart to
gcloud's ``_inject_gcs_credentials``; set ``iam_instance_profile`` and the
s3bucket data backend just works.
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
# Cloud-agnostic helpers shared with the gcloud provider.
from .gcloud import (
    wait_cua_ready,
    _init_computer_skip_wait,
    sanitize_label_value,
    _SET_RES_PY,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration — framework facts (hardcoded). Snapshot→AMI + subnets are
# deployment-specific and live in the yaml profile (AwsProviderConfig).
# ============================================================================

# Default instance when a task_card declares no ``vm.machineType``. CPU falls
# back C→M (see _instance_chain); GPU has no instance fallback. Both are Nitro
# families (required to boot an imported non-AWS image with ENA/NVMe).
_DEFAULT_CPU_INSTANCE = "m6i.2xlarge"     # 8 vCPU / 32 GiB
# g6 = NVIDIA L4, matching the GCE image's `nvidia-l4-vws` (L4) driver — the
# closest AWS family, so the baked driver has the best chance of binding.
_DEFAULT_GPU_INSTANCE = "g6.2xlarge"      # 8 vCPU / 32 GiB / 1x L4

# Launch retry tuning.
_AWS_MAX_RETRIES_TRANSIENT = 3
_AWS_TRANSIENT_BASE_DELAY = 15          # seconds, exponential backoff

# stderr substring → error class. transient = retry same subnet; subnet =
# move to next subnet (capacity); anything else = fail fast.
_AWS_RETRYABLE_TRANSIENT = [
    "requestlimitexceeded", "throttling", "rate exceeded",
    "serverinternal", "internalerror", "internal error",
    "service unavailable", "serviceunavailable", "unavailable",
    "connection reset", "connection refused", "timed out", "timeout",
    "could not connect", "connection aborted",
]
_AWS_RETRYABLE_SUBNET = [
    "insufficientinstancecapacity", "insufficient capacity",
    "instancelimitexceeded", "vcpulimitexceeded",
    "not have capacity", "no capacity",
]


# ============================================================================
# Provider config
# ============================================================================


@dataclass(frozen=True)
class SnapshotConfig:
    """What a logical snapshot tag maps to (yaml ``snapshots.<tag>``).

    Exactly parallel to gcloud's SnapshotConfig: ``image`` is the **image-family
    name** (the same registry key gcloud uses — ``ale-ubuntu22`` / ``ale-win10``
    — from :mod:`ale_run.environments.images`), NOT a raw ``ami-...``. The
    provider resolves that family to a concrete AMI id at acquire time by
    looking up an AMI tagged ``ale:image-family=<name>`` (override with an
    explicit ``ami:`` when you want to pin one). One ``ale-win10.py`` registry
    entry thus serves gcloud (GCE image name) and aws (family→AMI lookup) alike.

    ``zones`` holds **Availability-Zone names** (e.g. ``us-east-1a``) to try in
    order on capacity errors — the EC2 analogue of gcloud's zone list. The
    provider maps each AZ to a subnet in that AZ within the configured VPC.
    """

    image: str               # image-family name (registry key), e.g. "ale-win10"
    gpu: str | None          # truthy → use a GPU instance type; None for CPU
    zones: tuple[str, ...]   # AZ names to try, in order, on capacity errors
    ami: str | None = None   # optional explicit AMI id override (ami-...)
    resolution: tuple[int, int] | None = None
    """Windows display resolution (w, h) forced after boot. See gcloud's
    SnapshotConfig for the full rationale; Linux ignores it."""
    tenancy: str = "default"
    """EC2 placement tenancy: ``default`` (shared) or ``dedicated``. Per-snapshot
    so one env can mix Linux (``default``) with Windows-10-client snapshots
    (``dedicated`` — Win10 client BYOL is not licensed for shared tenancy)."""

    @property
    def os(self) -> str:
        # image is the family name (e.g. ale-win10), so this is stable even when
        # an explicit ami override is set.
        return "windows" if "win" in self.image.lower() else "linux"


@dataclass(frozen=True)
class AwsProviderConfig:
    """aws provider config (yaml ``provider.config``).

        region                       AWS region (required)
        vpc                          VPC id to resolve AZ→subnet within
                                     (required when zones are AZ names; the
                                     provider picks a subnet in each AZ here)
        security_group_ids           SGs to attach (must expose tcp:5000)
        instance_prefix              instance Name-tag prefix
        key_name                     EC2 key pair (optional; we reach the box
                                     via cua-server, not SSH, so usually unset)
        iam_instance_profile         instance profile name granting in-box S3
                                     access (optional; needed for s3:// data)
        associate_public_ip          assign a public IPv4 (default True)
        snapshots                    dict[snapshot tag → SnapshotConfig]

    region + vpc are provider-wide (like gcloud's project/network); the per-AZ
    capacity-fallback list lives per-snapshot (``snapshots.<tag>.zones``),
    mirroring gcloud. Instance-type selection (task_card override → default →
    C→M fallback) and root-volume sizing (AMI default) are framework facts here.
    """

    region: str
    vpc: str = ""
    security_group_ids: tuple[str, ...] = ()
    instance_prefix: str = "ale"
    key_name: str = ""
    iam_instance_profile: str = ""
    associate_public_ip: bool = True
    snapshots: dict[str, SnapshotConfig] = dataclass_field(default_factory=dict)


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
        raise KeyError(f"snapshot {image!r} missing required `zones` (AZ names)")
    return SnapshotConfig(
        image=str(image), gpu=raw.get("gpu"), zones=zones,
        ami=(str(raw["ami"]) if raw.get("ami") else None),
        resolution=_parse_resolution(raw.get("resolution"), image),
        tenancy=str(raw.get("tenancy") or "default"),
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


def _build_provider_config(raw: dict[str, Any]) -> AwsProviderConfig:
    snapshots = {
        str(tag): _build_snapshot_config(entry)
        for tag, entry in (raw.get("snapshots") or {}).items()
    }
    sgs = raw.get("security_group_ids") or raw.get("security_groups") or ()
    if isinstance(sgs, str):
        sgs = (sgs,)
    return AwsProviderConfig(
        region=str(raw["region"]),
        vpc=str(raw.get("vpc") or ""),
        security_group_ids=tuple(sgs),
        instance_prefix=str(raw.get("instance_prefix") or "ale"),
        key_name=str(raw.get("key_name") or ""),
        iam_instance_profile=str(raw.get("iam_instance_profile") or ""),
        associate_public_ip=bool(raw.get("associate_public_ip", True)),
        snapshots=snapshots,
    )


# ============================================================================
# Instance-type fallback chain
# ============================================================================


def _cpu_family_fallback(instance_type: str) -> str | None:
    """C-family → M fallback, keeping the size suffix.

    ``c6i.2xlarge`` → ``m6i.2xlarge``. Returns None for non-C families. This is
    the only instance fallback we do: compute-optimized capacity stocks out far
    more often than general-purpose.
    """
    fam, _, size = instance_type.partition(".")
    if fam[:1].lower() == "c" and size:
        return f"m{fam[1:]}.{size}"
    return None


def _instance_chain(instance_type: str, *, is_gpu: bool) -> tuple[str, ...]:
    """Ordered instance types to try for one box.

    * GPU: just the requested G-family type — no fallback.
    * CPU: the requested type, then its M fallback (``c*`` → ``m*``).
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
    """``<prefix>-<task-or-snapshot>-<hash8>`` Name tag (mirror of gcloud)."""
    if task_id:
        body = re.sub(r"[^a-z0-9]", "-", task_id.lower()).strip("-")[:40]
    else:
        body = re.sub(r"[^a-z0-9]", "-", snapshot.lower()).strip("-")[:30]
    seed = f"{prefix}:{task_id}:{harness}:{model_tag}:{snapshot}:{time.time()}:{random.random()}"
    h = hashlib.sha256(seed.encode()).hexdigest()[:8]
    return f"{prefix}-{body}-{h}"[:128]


# ============================================================================
# aws CLI wrapper + error classification
# ============================================================================


async def _run_aws(*args: str, region: str) -> tuple[int, str, str]:
    cmd = ["aws", *args, "--region", region, "--output", "json"]
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
    return any(pat in lower for pat in _AWS_RETRYABLE_TRANSIENT)


def _is_subnet_capacity_error(stderr: str) -> bool:
    lower = stderr.lower()
    return any(pat in lower for pat in _AWS_RETRYABLE_SUBNET)


# ============================================================================
# run-instances argv
# ============================================================================


def _build_run_args(
    *,
    name: str,
    image: str,
    instance_type: str,
    subnet: str,
    security_group_ids: tuple[str, ...],
    key_name: str,
    iam_instance_profile: str,
    associate_public_ip: bool,
    tenancy: str,
    snapshot_tag: str,
) -> list[str]:
    """``aws ec2 run-instances`` argv.

    Root volume size is NOT passed — we use the AMI's baked snapshot size.
    """
    tags = (
        f"ResourceType=instance,Tags=["
        f"{{Key=Name,Value={name}}},"
        f"{{Key=purpose,Value=ale-run}},"
        f"{{Key=snapshot,Value={sanitize_label_value(snapshot_tag)}}}]"
    )
    args = [
        "ec2", "run-instances",
        "--image-id", image,
        "--instance-type", instance_type,
        "--count", "1",
        "--subnet-id", subnet,
        "--tag-specifications", tags,
    ]
    if security_group_ids:
        args += ["--security-group-ids", *security_group_ids]
    if associate_public_ip:
        args.append("--associate-public-ip-address")
    if key_name:
        args += ["--key-name", key_name]
    if iam_instance_profile:
        args += ["--iam-instance-profile", f"Name={iam_instance_profile}"]
    if tenancy and tenancy != "default":
        args += ["--placement", f"Tenancy={tenancy}"]
    return args


async def _try_run_in_subnet(
    *,
    name: str,
    image: str,
    instance_type: str,
    subnet: str,
    cfg: AwsProviderConfig,
    tenancy: str,
    snapshot_tag: str,
) -> tuple[bool, str, str]:
    """Returns (ok, stdout, stderr). On capacity error returns ok=False without
    retrying (caller moves to the next subnet); transient errors retry here."""
    args = _build_run_args(
        name=name,
        image=image,
        instance_type=instance_type,
        subnet=subnet,
        security_group_ids=cfg.security_group_ids,
        key_name=cfg.key_name,
        iam_instance_profile=cfg.iam_instance_profile,
        associate_public_ip=cfg.associate_public_ip,
        tenancy=tenancy,
        snapshot_tag=snapshot_tag,
    )
    last_stderr = ""
    for attempt in range(1, _AWS_MAX_RETRIES_TRANSIENT + 1):
        logger.info(
            "Launching %s type=%s in %s (attempt %d/%d)",
            name, instance_type, subnet, attempt, _AWS_MAX_RETRIES_TRANSIENT,
        )
        rc, stdout, stderr = await _run_aws(*args, region=cfg.region)
        if rc == 0:
            return True, stdout, ""
        last_stderr = stderr

        if _is_subnet_capacity_error(stderr):
            logger.warning(
                "type=%s capacity-exhausted in %s: %s",
                instance_type, subnet, stderr[:300],
            )
            return False, "", last_stderr

        if _is_transient_error(stderr) and attempt < _AWS_MAX_RETRIES_TRANSIENT:
            delay = _AWS_TRANSIENT_BASE_DELAY * (2 ** (attempt - 1))
            logger.warning(
                "run-instances transient error (attempt %d/%d): %s — retrying in %ds",
                attempt, _AWS_MAX_RETRIES_TRANSIENT, stderr[:200], delay,
            )
            await asyncio.sleep(delay)
            continue

        return False, "", last_stderr
    return False, "", last_stderr


# ============================================================================
# describe / lifecycle helpers
# ============================================================================


def _instance_id_from_run(stdout: str) -> str:
    data = json.loads(stdout)
    return data["Instances"][0]["InstanceId"]


async def _wait_running_with_ip(
    instance_id: str, region: str, *, want_public: bool, timeout: float = 240,
) -> str:
    """Poll describe-instances until the instance is ``running`` and has an IP.

    Returns the public IP (want_public) or the private IP. Raises on timeout."""
    deadline = time.monotonic() + timeout
    last_state = "?"
    while time.monotonic() < deadline:
        rc, stdout, stderr = await _run_aws(
            "ec2", "describe-instances", "--instance-ids", instance_id,
            region=region,
        )
        if rc == 0:
            try:
                inst = json.loads(stdout)["Reservations"][0]["Instances"][0]
                last_state = inst.get("State", {}).get("Name", "?")
                if last_state == "running":
                    ip = (
                        inst.get("PublicIpAddress") if want_public
                        else inst.get("PrivateIpAddress")
                    )
                    if ip:
                        return ip
                if last_state in ("terminated", "shutting-down", "stopped"):
                    raise RuntimeError(
                        f"instance {instance_id} entered {last_state} before becoming reachable"
                    )
            except (KeyError, IndexError, json.JSONDecodeError):
                pass
        await asyncio.sleep(5)
    raise RuntimeError(
        f"timed out waiting for {instance_id} to be running with an IP "
        f"(last state={last_state})"
    )


async def _terminate(instance_id: str, region: str) -> bool:
    logger.info("Terminating instance %s", instance_id)
    rc, _, stderr = await _run_aws(
        "ec2", "terminate-instances", "--instance-ids", instance_id, region=region,
    )
    if rc != 0:
        logger.error("Failed to terminate %s: %s", instance_id, stderr)
        return False
    return True


async def _stop(instance_id: str, region: str) -> bool:
    logger.info("Stopping instance %s", instance_id)
    rc, _, stderr = await _run_aws(
        "ec2", "stop-instances", "--instance-ids", instance_id, region=region,
    )
    if rc != 0:
        logger.error("Failed to stop %s: %s", instance_id, stderr)
        return False
    return True


# ============================================================================
# Provider
# ============================================================================


class AwsProvider(Provider):
    """Provider backed by ``aws ec2 run-instances / terminate-instances``."""

    def __init__(self, config: AwsProviderConfig | dict[str, Any]):
        if isinstance(config, dict):
            config = _build_provider_config(config)
        self._cfg = config
        self._ami_cache: dict[str, str] = {}     # family name → ami id
        self._subnet_cache: dict[str, str] = {}  # az name → subnet id

    @property
    def config(self) -> AwsProviderConfig:
        return self._cfg

    # ----------------------------------------------------------- resolution

    async def _resolve_ami(self, snap: SnapshotConfig) -> str:
        """Resolve a snapshot's image to a concrete AMI id.

        Order: explicit ``ami:`` override → cached → look up an AMI owned by us
        tagged ``ale:image-family=<snap.image>`` (newest wins). This is the AWS
        analogue of gcloud resolving a snapshot's image name to a GCE image.
        """
        if snap.ami:
            return snap.ami
        family = snap.image
        if family in self._ami_cache:
            return self._ami_cache[family]
        rc, out, err = await _run_aws(
            "ec2", "describe-images", "--owners", "self",
            "--filters", f"Name=tag:ale:image-family,Values={family}",
            "--query", "reverse(sort_by(Images,&CreationDate))[0].ImageId",
            region=self._cfg.region,
        )
        ami = (out or "").strip().strip('"')
        if rc != 0 or not ami or ami == "None":
            raise RuntimeError(
                f"no AMI tagged ale:image-family={family!r} in {self._cfg.region} "
                f"(register one, tag it, or set an explicit `ami:` on the snapshot). "
                f"{(err or '').strip()[:200]}"
            )
        self._ami_cache[family] = ami
        return ami

    async def _subnet_for_az(self, az: str) -> str:
        """Resolve an AZ name to a subnet id in that AZ within the configured VPC.

        Prefers subnets tagged ``project=ale``; falls back to any subnet in the
        AZ/VPC. Cached per AZ. The VPC scopes the lookup so we don't pick a
        stray subnet from another network."""
        if az in self._subnet_cache:
            return self._subnet_cache[az]
        filters = [f"Name=availability-zone,Values={az}"]
        if self._cfg.vpc:
            filters.append(f"Name=vpc-id,Values={self._cfg.vpc}")
        rc, out, err = await _run_aws(
            "ec2", "describe-subnets", "--filters", *filters,
            "--query",
            "sort_by(Subnets, &SubnetId)[?Tags[?Key=='project' && Value=='ale']].SubnetId "
            "| [0]",
            region=self._cfg.region,
        )
        subnet = (out or "").strip().strip('"')
        if not subnet or subnet == "None":
            # fall back to any subnet in the AZ/VPC
            rc, out, err = await _run_aws(
                "ec2", "describe-subnets", "--filters", *filters,
                "--query", "Subnets[0].SubnetId", region=self._cfg.region,
            )
            subnet = (out or "").strip().strip('"')
        if rc != 0 or not subnet or subnet == "None":
            raise RuntimeError(
                f"no subnet in AZ {az} (vpc={self._cfg.vpc or 'any'}): "
                f"{(err or '').strip()[:200]}"
            )
        self._subnet_cache[az] = subnet
        return subnet

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
                "os mismatch for %s: task declares %r but AMI %r looks %r",
                spec.snapshot, spec.os, snap.image, snap.os,
            )

        is_gpu = snap.gpu is not None
        azs = snap.zones

        base_instance = spec.machine_type or (
            _DEFAULT_GPU_INSTANCE if is_gpu else _DEFAULT_CPU_INSTANCE
        )
        instances = _instance_chain(base_instance, is_gpu=is_gpu)

        # Resolve the image-family name (e.g. "ale-win10") to a concrete AMI id.
        ami_id = await self._resolve_ami(snap)

        name = generate_instance_name(
            self._cfg.instance_prefix,
            snapshot=spec.snapshot,
            task_id=spec.task_id,
            harness=spec.harness,
            model_tag=spec.model_tag,
        )

        logger.info(
            "instance %s candidates: ami=%s types=%s azs=%s",
            name, ami_id, list(instances), list(azs),
        )

        last_stderr = ""
        stdout = ""
        used_az = azs[0]
        used_subnet = ""
        used_instance = instances[0]

        # instance-type × AZ, in order: each type tried across all AZs. Each AZ
        # is resolved to a subnet in that AZ within the configured VPC.
        for instance_type in instances:
            for az in azs:
                try:
                    subnet = await self._subnet_for_az(az)
                except Exception as e:  # noqa: BLE001
                    last_stderr = f"no subnet for AZ {az}: {e}"
                    logger.warning("%s", last_stderr)
                    continue
                ok, out, stderr = await _try_run_in_subnet(
                    name=name,
                    image=ami_id,
                    instance_type=instance_type,
                    subnet=subnet,
                    cfg=self._cfg,
                    tenancy=snap.tenancy,
                    snapshot_tag=spec.snapshot,
                )
                if ok:
                    stdout = out
                    used_subnet = subnet
                    used_az = az
                    used_instance = instance_type
                    break
                last_stderr = stderr
                if not _is_subnet_capacity_error(stderr):
                    raise RuntimeError(f"aws run-instances failed: {stderr}")
            if stdout:
                break
        else:
            raise RuntimeError(
                f"aws run-instances failed for all types/AZs: {last_stderr}"
            )

        try:
            instance_id = _instance_id_from_run(stdout)
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            raise RuntimeError(
                f"failed to parse run-instances output: {e}\nstdout: {stdout[:500]}"
            ) from e

        # Instance now exists. Until the handle is returned, ANY failure leaks
        # it (the caller's cleanup only runs once it holds the handle), so
        # terminate-on-failure here before propagating.
        try:
            public_ip = await _wait_running_with_ip(
                instance_id, self._cfg.region,
                want_public=self._cfg.associate_public_ip,
            )

            # snap.image IS the registry family name (same key gcloud uses), so
            # this resolves the in-box paths/port directly — one registry entry
            # serves both providers.
            from ..images import get as get_image
            image = get_image(snap.image)

            cua_url = f"http://{public_ip}:{image.cua_server_port}"
            logger.info(
                "instance %s (%s) launched as %s in %s (%s) at %s",
                name, instance_id, used_instance, used_az, used_subnet, cua_url,
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
                    "az": used_az,
                    "subnet": used_subnet,
                    "public_ip": public_ip,
                    "image": image.name,
                    "ami": ami_id,
                    "snapshot": spec.snapshot,
                    "name": name,
                },
            )
        except BaseException:
            logger.warning(
                "acquire: post-launch failure on %s (%s) — terminating to avoid leak",
                name, instance_id,
            )
            try:
                await _terminate(instance_id, self._cfg.region)
            except Exception as te:  # noqa: BLE001
                logger.error("acquire: could not terminate leaked %s: %s", instance_id, te)
            raise

    @staticmethod
    async def _set_windows_resolution(
        cua_url: str, resolution: tuple[int, int],
    ) -> None:
        """Force the Windows framebuffer to ``resolution`` (w, h). Same machinery
        as the gcloud provider; raises (no silent fallback) if unsupported."""
        from cua_bench.computers.remote import RemoteDesktopSession

        w, h = resolution
        remote_path = r"C:\agenthle\_set_resolution.py"
        session = RemoteDesktopSession(api_url=cua_url, os_type="windows")
        _init_computer_skip_wait(session)
        await session.run_command(
            r"cmd /c if not exist C:\agenthle mkdir C:\agenthle", check=False,
        )
        await session.write_file(remote_path, _SET_RES_PY)
        res = await session.run_command(
            f'python "{remote_path}" {w} {h}', check=False,
        )
        out = (res.get("stdout") or "").strip() if isinstance(res, dict) else ""
        if "set_ok" in out:
            logger.info("aws: set Windows resolution to %dx%d", w, h)
            return
        err = out or (res.get("stderr") if isinstance(res, dict) else "") or "no output"
        raise RuntimeError(f"aws: could not set Windows resolution {w}x{h} — {err}")

    # ------------------------------------------------------------------ release

    async def release(self, vm: SandboxHandle, *, mode: ReleaseMode = "delete") -> None:
        instance_id = vm.metadata.get("instance_id") or vm.id
        region = vm.metadata.get("region") or self._cfg.region
        if mode == "delete":
            await _terminate(instance_id, region)
        elif mode == "stop":
            await _stop(instance_id, region)
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
