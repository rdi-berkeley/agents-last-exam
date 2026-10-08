"""Bound independent native processes inside a delegated aggregate resource group."""

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


class NativeWorkers:
    def __init__(self, cpus, memory_mib=2048):
        self.cpus = tuple(cpus)
        if (
            not 1 <= len(self.cpus) <= 2
            or len(set(self.cpus)) != len(self.cpus)
            or not set(self.cpus).issubset(os.sched_getaffinity(0))
        ):
            raise ValueError("Native workers require one or two distinct allocated CPUs")
        self.memory_bytes = memory_mib * 1024**2
        relative = Path("/proc/self/cgroup").read_text().strip().split("::", 1)[1]
        self.root = Path("/sys/fs/cgroup") / relative.lstrip("/")
        aggregate_memory = (self.root / "memory.max").read_text().strip()
        quota, period = (self.root / "cpu.max").read_text().split()
        required_memory = len(self.cpus) * self.memory_bytes + 1024**3
        if aggregate_memory != "max" and int(aggregate_memory) < required_memory:
            raise ValueError("Aggregate memory does not cover workers and coordinator")
        if quota != "max" and int(quota) / int(period) < len(self.cpus):
            raise ValueError("Aggregate CPU quota would serialize the allocated workers")
        coordinator = self.root / "coordinator"
        coordinator.mkdir()
        (coordinator / "cgroup.procs").write_text(str(os.getpid()))
        (self.root / "cgroup.subtree_control").write_text("+cpu +memory")
        (coordinator / "memory.max").write_text(str(1024**3))
        (coordinator / "memory.swap.max").write_text("0")

    def start(self, command, destination, cpu):
        destination = Path(destination)
        group = self.root / destination.name
        group.mkdir()
        (group / "memory.max").write_text(str(self.memory_bytes))
        (group / "memory.swap.max").write_text("0")
        (group / "cpu.max").write_text("100000 100000")
        log = destination.with_name(destination.name + "-launcher.log").open("w")
        environment = {
            **os.environ,
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        }
        process = subprocess.Popen(
            [
                sys.executable,
                str(Path(__file__).with_name("native_worker.py")),
                str(group),
                str(cpu),
                *command,
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            env=environment,
        )
        return {
            "process": process,
            "group": group,
            "log": log,
            "started": time.monotonic(),
            "cpu": cpu,
        }

    def finish(self, worker):
        worker["log"].close()
        group = worker["group"]
        usage = dict(line.split() for line in (group / "cpu.stat").read_text().splitlines())
        events = dict(line.split() for line in (group / "memory.events").read_text().splitlines())
        result = {
            "cpu_s": int(usage["usage_usec"]) / 1e6,
            "peak_memory_bytes": int((group / "memory.peak").read_text()),
            "memory_limit_bytes": self.memory_bytes,
            "oom_kills": int(events["oom_kill"]),
        }
        group.rmdir()
        return result

    def stop(self, worker):
        (worker["group"] / "cgroup.kill").write_text("1")
        worker["process"].wait(timeout=10)

    def aggregate_usage(self):
        usage = dict(line.split() for line in (self.root / "cpu.stat").read_text().splitlines())
        return {
            "aggregate_cpu_s": int(usage["usage_usec"]) / 1e6,
            "aggregate_peak_memory_bytes": int((self.root / "memory.peak").read_text()),
            "aggregate_memory_limit_bytes": (len(self.cpus) * self.memory_bytes + 1024**3),
            "allocated_cpus": list(self.cpus),
            "worker_memory_limit_bytes": self.memory_bytes,
        }


def run_cases(
    family,
    output,
    review_path,
    positions,
    checkpoint,
    record,
    workers,
    deadline,
    max_iterations,
    *,
    case_script=None,
    immutable=None,
    poll_seconds=0.25,
):
    pending = [(position, checkpoint, 0) for position in positions]
    active = {}
    failures = []
    previous_hash = {}
    if checkpoint:
        metadata = json.loads(Path(checkpoint).with_suffix(".json").read_text())
        previous_hash = {position: metadata["native_state_sha256"] for position in positions}
    output = Path(output)
    last_receipt = 0
    script = Path(case_script) if case_script else Path(__file__).with_name("native_replay_case.py")
    while pending or active:
        remaining = deadline - time.monotonic()
        if remaining < 120 and pending:
            failures.append(
                "Family runtime exhausted before pending attempts; checkpoints retained"
            )
            pending.clear()
        if not failures:
            for cpu in workers.cpus:
                if cpu in active or not pending:
                    continue
                position, restart, attempt = pending.pop(0)
                destination = output / f"pos_{position}m-attempt{attempt}"
                if destination.exists():
                    raise FileExistsError(
                        f"Native attempt must have fresh writable state: {destination}"
                    )
                if restart and destination in Path(restart).resolve().parents:
                    raise ValueError("Worker output overlaps its immutable restart source")
                command = [
                    sys.executable,
                    str(script),
                    str(family),
                    str(destination),
                    str(review_path),
                    "--position",
                    str(position),
                    "--runtime",
                    str(min(3570, int(remaining - 60))),
                    "--max-iterations",
                    str(max_iterations),
                ]
                if restart:
                    command.extend(["--restart", str(restart)])
                try:
                    worker = workers.start(command, destination, cpu)
                except (OSError, ValueError) as error:
                    failures.append(f"pos_{position}m worker launch: {error}")
                    pending.clear()
                    break
                worker.update(
                    position=position,
                    restart=str(restart) if restart else None,
                    destination=destination,
                    attempt=attempt,
                    command=command,
                )
                active[cpu] = worker
                last_receipt = 0
        for cpu, worker in list(active.items()):
            process = worker["process"]
            if time.monotonic() >= deadline and process.poll() is None:
                workers.stop(worker)
                worker["coordinator_timeout"] = True
            if process.poll() is None:
                continue
            del active[cpu]
            destination = worker["destination"]
            terminal_path = destination / "terminal.json"
            try:
                terminal = json.loads(terminal_path.read_text()) if terminal_path.exists() else {}
            except (OSError, ValueError) as error:
                terminal = {}
                failures.append(f"pos_{worker['position']}m terminal receipt: {error}")
            measured = workers.finish(worker)
            wall_elapsed = time.monotonic() - worker["started"]
            attempt_record = {
                "position": worker["position"],
                "directory": str(destination),
                "native_exit_code": terminal.get("native_exit_code"),
                "worker_exit_code": process.returncode,
                "elapsed_s": terminal.get("elapsed_s", wall_elapsed),
                "elapsed_source": "native_terminal"
                if "elapsed_s" in terminal
                else "worker_wall_fallback",
                "wall_elapsed_s": wall_elapsed,
                "restart": worker["restart"],
                "cpu": cpu,
                **measured,
                "coordinator_timeout": worker.get("coordinator_timeout", False),
            }
            record["attempts"].append(attempt_record)
            last_receipt = 0
            if immutable and any(file_digest(path) != digest for path, digest in immutable.items()):
                failures.append("Common checkpoint changed during native branching")
            position = worker["position"]
            if process.returncode == 0 and terminal.get("native_exit_code") == 0:
                if not terminal.get("source_fact_audit", {}).get("passed") or not terminal.get(
                    "extraction", {}
                ).get("native_replay_and_reader_completed"):
                    failures.append(
                        f"Native pos_{position}m missing source audit or physical reader"
                    )
                else:
                    record["cases"][f"pos_{position}m"] = {
                        "directory": str(destination),
                        "terminal_sha256": file_digest(terminal_path),
                        "extraction": terminal["extraction"],
                    }
            elif process.returncode == 124 and terminal.get("native_exit_code") == 124:
                latest = destination / "restart/latest"
                try:
                    metadata = json.loads(latest.with_suffix(".json").read_text())
                    digest = file_digest(latest.with_suffix(".rest"))
                    if (
                        digest != metadata["native_state_sha256"]
                        or metadata["input_hashes"] != record["input_hashes"]
                    ):
                        raise ValueError("Continuation checkpoint identity differs")
                    if previous_hash.get(position) == digest:
                        raise TimeoutError("Native continuation made no committed progress")
                    previous_hash[position] = digest
                    if not failures:
                        pending.append((position, latest, worker["attempt"] + 1))
                except (ValueError, OSError, KeyError, TimeoutError) as error:
                    failures.append(f"pos_{position}m continuation: {error}")
            else:
                failures.append(
                    f"Native pos_{position}m worker exit {process.returncode}: {terminal.get('source_audit_error', terminal.get('reader_error', 'native execution incomplete'))}"
                )
            if failures:
                pending.clear()
        if time.monotonic() - last_receipt >= 45 or not active:
            record["native_compute_s"] = sum(row["elapsed_s"] for row in record["attempts"])
            record["worker_cpu_s"] = sum(row["cpu_s"] for row in record["attempts"])
            record["elapsed_s"] = time.monotonic() - record["started_monotonic"]
            record["active_workers"] = [
                {
                    "position": row["position"],
                    "cpu": slot,
                    "pid": row["process"].pid,
                    "directory": str(row["destination"]),
                }
                for slot, row in active.items()
            ]
            (output / "family-replay.json").write_text(json.dumps(record, indent=2))
            last_receipt = time.monotonic()
        if active:
            time.sleep(poll_seconds)
    if failures:
        raise RuntimeError("; ".join(failures))
