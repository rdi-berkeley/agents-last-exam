"""Select native replay workers within guest affinity, quota and memory headroom."""

import json
import os
from pathlib import Path

REPLAY_SECONDS = 6600
WORKER_MEMORY_MIB = 4096
COORDINATOR_MEMORY_MIB = 4096
REPLAY_WORKERS = 5


class ReplayBudgetExceeded(RuntimeError):
    def __init__(self, resource, detail):
        self.resource = resource
        super().__init__(detail)


def replay_resources(cpus=None, *, proc=Path("/proc"), cgroups=Path("/sys/fs/cgroup")):
    available = sorted(os.sched_getaffinity(0))
    memory = next(
        int(line.split()[1]) * 1024
        for line in (proc / "meminfo").read_text().splitlines()
        if line.startswith("MemAvailable:")
    )
    capacity = len(available)
    relative = next(
        line.split("::", 1)[1]
        for line in (proc / "self/cgroup").read_text().splitlines()
        if line.startswith("0::")
    )
    group = cgroups / relative.lstrip("/")
    while True:
        if (group / "cpu.max").exists():
            quota, period = (group / "cpu.max").read_text().split()
            if quota != "max":
                capacity = min(capacity, int(quota) // int(period))
        if (group / "memory.max").exists():
            limit = (group / "memory.max").read_text().strip()
            if limit != "max":
                current = int((group / "memory.current").read_text())
                memory = min(memory, max(0, int(limit) - current))
        if group == cgroups:
            break
        group = group.parent
    capacity = min(
        REPLAY_WORKERS,
        capacity,
        (memory - (COORDINATOR_MEMORY_MIB + 2048) * 1024**2) // (WORKER_MEMORY_MIB * 1024**2),
    )
    selected = tuple(available[: max(0, capacity)] if cpus is None else cpus)
    if (
        not 1 <= len(selected) <= capacity
        or any(type(cpu) is not int for cpu in selected)
        or len(set(selected)) != len(selected)
        or not set(selected).issubset(available)
    ):
        raise ValueError("Invalid native evaluator CPU allocation or insufficient memory headroom")
    return {"cpus": selected, "available_cpus": available, "available_memory_bytes": memory}


if __name__ == "__main__":
    import sys

    print(json.dumps(replay_resources(json.loads(sys.argv[1]) if len(sys.argv) > 1 else None)))
