"""Image + capacity-profile primitives for the gcloud provider.

Ported from simprun/config.py. Drops:
  - module-level IMAGE_MAP / CAPACITY_POOL_MAP (those were JSON-driven globals
    in simprun); agenthle-public reads image info from the ProviderSpec config.
  - GCP_PROJECT constant (lives in the provider config now).
  - REPO_ROOT / ENV_FILE / GCP_KEY_FILE constants (orchastration.config_loader
    handles env files).

Retains the ImageConfig + CapacityProfile dataclasses and ``resolve()`` /
``capacity_profiles_for()`` helpers used by GcloudProvider.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from .machine_types import (
    family_of,
    is_accelerator_machine_type,
    parse_gce_machine_type,
    round_up_vcpus,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ImageConfig:
    category: str
    image_name: str
    default_machine_type: str
    zone: str
    os_type: str
    gpu: str | None
    boot_disk_gb: int
    network: str = "default"
    subnet: str = "default"
    fallback_zones: tuple[str, ...] = ()


@dataclass(frozen=True)
class CapacityProfile:
    name: str
    machine_type: str
    zones: tuple[str, ...]
    boot_disk_type: str = "auto"
    data_disk_type: str = "auto"
    priority: int = 100


@dataclass(frozen=True)
class PoolEntry:
    """A capacity-pool entry: family + zone-set with a default size and a cap."""

    name: str
    machine_family: str
    default_vcpus: int
    max_vcpus: int
    zones: tuple[str, ...]
    boot_disk_type: str = "auto"
    data_disk_type: str = "auto"
    priority: int = 100


def image_zones(image_cfg: ImageConfig) -> tuple[str, ...]:
    return (image_cfg.zone, *image_cfg.fallback_zones)


def capacity_profiles_for(
    image_cfg: ImageConfig,
    *,
    machine_type_override: str | None = None,
    pool: tuple[PoolEntry, ...] = (),
) -> tuple[CapacityProfile, ...]:
    """Compute the prioritized list of CapacityProfiles to try for this image.

    With ``pool=()`` (the default in agenthle-public) the return value is a
    single image-default profile, optionally preceded by an override profile
    derived from ``machine_type_override``.
    """
    target_vcpus: int | None = None
    override_concrete_mt: str | None = None
    if machine_type_override:
        shape = parse_gce_machine_type(machine_type_override)
        if shape is None:
            raise ValueError(
                f"Invalid vm.machineType override {machine_type_override!r}: "
                "could not parse as a standard GCE machine type"
            )
        target_vcpus = shape.vcpus
        if "-custom-" in machine_type_override.strip().lower():
            override_concrete_mt = machine_type_override
        elif is_accelerator_machine_type(machine_type_override):
            override_concrete_mt = machine_type_override
        else:
            override_family = family_of(machine_type_override)
            if override_family is None:
                raise ValueError(
                    f"Invalid vm.machineType override {machine_type_override!r}: "
                    "could not extract family"
                )
            rounded = round_up_vcpus(override_family, target_vcpus)
            if rounded is None:
                raise ValueError(
                    f"Invalid vm.machineType override {machine_type_override!r}: "
                    f"family {override_family!r} cannot host {target_vcpus} vCPUs"
                )
            override_concrete_mt = f"{override_family}-{rounded}"

    profiles: list[CapacityProfile] = []

    if override_concrete_mt is not None:
        profiles.append(
            CapacityProfile(
                name=f"task-card-{override_concrete_mt}",
                priority=-1000,
                machine_type=override_concrete_mt,
                zones=image_zones(image_cfg),
            )
        )

    for entry in pool:
        if target_vcpus is None:
            chosen_vcpus: int | None = entry.default_vcpus
        else:
            if target_vcpus > entry.max_vcpus:
                continue
            chosen_vcpus = round_up_vcpus(entry.machine_family, target_vcpus)
            if chosen_vcpus is None or chosen_vcpus > entry.max_vcpus:
                continue
        zones = entry.zones or image_zones(image_cfg)
        profiles.append(
            CapacityProfile(
                name=f"{entry.name}-{chosen_vcpus}",
                priority=entry.priority,
                machine_type=f"{entry.machine_family}-{chosen_vcpus}",
                zones=zones,
                boot_disk_type=entry.boot_disk_type,
                data_disk_type=entry.data_disk_type,
            )
        )

    if not profiles:
        profiles.append(
            CapacityProfile(
                name=f"image-default-{image_cfg.default_machine_type}",
                priority=0,
                machine_type=image_cfg.default_machine_type,
                zones=image_zones(image_cfg),
            )
        )

    seen: set[tuple[str, str, str, tuple[str, ...]]] = set()
    deduped: list[CapacityProfile] = []
    for profile in profiles:
        key = (
            profile.machine_type,
            profile.boot_disk_type,
            profile.data_disk_type,
            profile.zones,
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(profile)

    return tuple(deduped)


# ======================================================================
# VM-side data layout (matches the agenthle on-VM convention)
# ======================================================================


def vm_data_root(os_type: str) -> str:
    if os_type == "linux":
        return "/media/user/data/agenthle"
    return "E:\\agenthle"


def vm_task_dir(os_type: str, domain: str, task: str, variant: str) -> str:
    root = vm_data_root(os_type)
    if os_type == "linux":
        return f"{root}/{domain}/{task}/{variant}"
    return f"{root}\\{domain}\\{task}\\{variant}"


def vm_subdir(os_type: str, domain: str, task: str, variant: str, subdir: str) -> str:
    base = vm_task_dir(os_type, domain, task, variant)
    if os_type == "linux":
        return f"{base}/{subdir}"
    return f"{base}\\{subdir}"


def gcs_task_prefix(bucket: str, domain: str, task: str, variant: str) -> str:
    return f"{bucket.rstrip('/')}/{domain}/{task}/{variant}"
