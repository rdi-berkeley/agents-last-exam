"""GCE machine type parsing, family memory ratios, and valid-size tables.

Ported verbatim from simprun/machine_types.py.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)


_FAMILY_MEM_PER_VCPU: dict[str, float] = {
    "e2-standard": 4.0,
    "e2-highmem": 8.0,
    "e2-highcpu": 1.0,
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


_VALID_VCPUS: dict[str, frozenset[int]] = {
    "c4-standard": frozenset({2, 4, 8, 16, 32, 48, 96, 192}),
    "c4-highmem": frozenset({2, 4, 8, 16, 32, 48, 96, 192}),
    "c4-highcpu": frozenset({2, 4, 8, 16, 32, 48, 96, 192}),
    "n2-standard": frozenset({2, 4, 8, 16, 32, 48, 64, 80, 96, 128}),
    "n2-highmem": frozenset({2, 4, 8, 16, 32, 48, 64, 80, 96, 128}),
    "n2-highcpu": frozenset({2, 4, 8, 16, 32, 48, 64, 80, 96}),
    "e2-standard": frozenset({2, 4, 8, 16, 32}),
    "e2-highmem": frozenset({2, 4, 8, 16}),
    "e2-highcpu": frozenset({2, 4, 8, 16, 32}),
    "g2-standard": frozenset({4, 8, 12, 16, 24, 32, 48, 96}),
}


_CUSTOM_RE = re.compile(r"^(?P<family>[a-z]\w+)-custom-(?P<vcpus>\d+)-(?P<mem_mb>\d+)$")
_STANDARD_RE = re.compile(r"^(?P<family>[a-z]\w+-\w+)-(?P<vcpus>\d+)$")
_ACCEL_RE = re.compile(r"^(?P<family>[a-z]\w+-\w+)-(?P<count>\d+)g$")


@dataclass(frozen=True)
class GCEShape:
    vcpus: int
    memory_gb: int
    machine_type: str


def parse_gce_machine_type(machine_type: str | None) -> GCEShape | None:
    if not machine_type:
        return None

    mt = machine_type.strip().lower()

    m = _CUSTOM_RE.match(mt)
    if m:
        vcpus = int(m.group("vcpus"))
        mem_mb = int(m.group("mem_mb"))
        return GCEShape(vcpus=vcpus, memory_gb=max(1, mem_mb // 1024), machine_type=machine_type)

    m = _ACCEL_RE.match(mt)
    if m:
        family = m.group("family")
        count = int(m.group("count"))
        mem_ratio = _FAMILY_MEM_PER_VCPU.get(family)
        if mem_ratio is not None:
            vcpus = count * 12
            return GCEShape(
                vcpus=vcpus, memory_gb=max(1, int(count * mem_ratio)), machine_type=machine_type
            )
        logger.warning("unknown accelerator family %r in %r", family, machine_type)
        return None

    m = _STANDARD_RE.match(mt)
    if m:
        family = m.group("family")
        vcpus = int(m.group("vcpus"))
        mem_ratio = _FAMILY_MEM_PER_VCPU.get(family)
        if mem_ratio is not None:
            return GCEShape(
                vcpus=vcpus, memory_gb=max(1, int(vcpus * mem_ratio)), machine_type=machine_type
            )
        logger.warning("unknown GCE family %r in %r", family, machine_type)
        return None

    logger.warning("unparseable GCE machine type: %r", machine_type)
    return None


def family_of(machine_type: str) -> str | None:
    m = _STANDARD_RE.match(machine_type.strip().lower())
    if not m:
        return None
    return m.group("family")


def is_accelerator_machine_type(machine_type: str) -> bool:
    return _ACCEL_RE.match(machine_type.strip().lower()) is not None


def is_valid_gce_size(family: str, vcpus: int) -> bool:
    sizes = _VALID_VCPUS.get(family)
    return bool(sizes and vcpus in sizes)


def round_up_vcpus(family: str, target_vcpus: int) -> int | None:
    sizes = _VALID_VCPUS.get(family)
    if not sizes:
        return None
    candidates = [s for s in sizes if s >= target_vcpus]
    if not candidates:
        return None
    return min(candidates)
