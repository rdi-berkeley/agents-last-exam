"""QemuProvider: one local QEMU VM per run, packaged in Docker.

The provider keeps a read-only qcow2 base image on the host, creates a tiny
qcow2 overlay for each run, and boots that overlay with QEMU inside the
``agentslastexam/ale-qemu`` runner container. The guest exposes the same cua-server
API as GCE sandboxes, so the rest of ALE uses an ordinary ``SandboxHandle``.

This first implementation is intentionally CPU-only and expects task data to
already be baked into the guest image. GPU passthrough and ``local:`` task-data
staging require separate host and guest plumbing and fail explicitly.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import random
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ...base_interface import SandboxSpec, Provider, ReleaseMode, SandboxHandle

logger = logging.getLogger(__name__)

_DEFAULT_DOCKER_IMAGE = "agentslastexam/ale-qemu:0.1.0"
_DEFAULT_CUA_PORT = 5000
_DEFAULT_NOVNC_PORT = 8006
_CONTAINER_PREFIX = "ale-qemu"


@dataclass(frozen=True)
class QemuSnapshotConfig:
    """One logical snapshot's local QEMU realization."""

    image: str
    qcow2: str
    runtime_root: Path
    image_cache_dir: Path
    docker_image: str = _DEFAULT_DOCKER_IMAGE
    vcpus: int = 0
    memory_gb: int = 0
    shm_size: str = "2g"
    bind_address: str = "127.0.0.1"
    ready_timeout_s: int = 900
    readiness_poll_interval_s: float = 5


@dataclass(frozen=True)
class QemuProviderConfig:
    """Provider config containing one entry per task-card snapshot tag."""

    snapshots: dict[str, QemuSnapshotConfig]


def _expand_path(value: str) -> Path:
    return Path(value).expanduser().resolve()


def _build_snapshot_config(raw: dict[str, Any]) -> QemuSnapshotConfig:
    qcow2 = str(raw.get("qcow2") or "").strip()
    if not qcow2:
        raise KeyError("qemu snapshot config missing required field `qcow2`")
    root = _expand_path(str(raw.get("root") or "~/.cache/ale/qemu"))
    runtime_root = _expand_path(str(raw.get("runtime_root") or root / "runtime"))
    image_cache_dir = _expand_path(str(raw.get("image_cache_dir") or root / "images"))
    vcpus = int(raw.get("vcpus") or 0)
    memory_gb = int(raw.get("memory_gb") or 0)
    if vcpus < 0:
        raise ValueError(f"qemu vcpus must be >= 0, got {vcpus}")
    if memory_gb < 0:
        raise ValueError(f"qemu memory_gb must be >= 0, got {memory_gb}")
    return QemuSnapshotConfig(
        image=str(raw["image"]),
        qcow2=qcow2,
        runtime_root=runtime_root,
        image_cache_dir=image_cache_dir,
        docker_image=str(raw.get("docker_image") or _DEFAULT_DOCKER_IMAGE),
        vcpus=vcpus,
        memory_gb=memory_gb,
        shm_size=str(raw.get("shm_size") or "2g"),
        bind_address=str(raw.get("bind_address") or "127.0.0.1"),
        ready_timeout_s=int(raw.get("ready_timeout_s") or 900),
        readiness_poll_interval_s=float(raw.get("readiness_poll_interval_s") or 5),
    )


def _build_provider_config(raw: dict[str, Any]) -> QemuProviderConfig:
    snapshots_raw = raw.get("snapshots")
    if not isinstance(snapshots_raw, dict) or not snapshots_raw:
        raise TypeError("qemu provider config requires a non-empty `snapshots` mapping")
    return QemuProviderConfig(
        snapshots={
            str(tag): _build_snapshot_config(dict(snapshot_raw))
            for tag, snapshot_raw in snapshots_raw.items()
        }
    )


def _generate_container_name(spec: SandboxSpec) -> str:
    source = spec.task_id or spec.snapshot
    body = re.sub(r"[^a-z0-9]", "-", source.lower()).strip("-")[:40]
    seed = (
        f"{spec.snapshot}:{spec.task_id}:{spec.harness}:{spec.model_tag}:"
        f"{time.time()}:{random.random()}"
    )
    suffix = hashlib.sha256(seed.encode()).hexdigest()[:8]
    return f"{_CONTAINER_PREFIX}-{body or 'sandbox'}-{suffix}"


async def _run_process(*args: str) -> tuple[int, str, str]:
    try:
        proc = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(f"required command not found: {args[0]}") from exc
    stdout_b, stderr_b = await proc.communicate()
    return (
        proc.returncode or 0,
        stdout_b.decode(errors="replace").strip(),
        stderr_b.decode(errors="replace").strip(),
    )


async def _run_docker(*args: str, check: bool = True) -> tuple[int, str, str]:
    rc, stdout, stderr = await _run_process("docker", *args)
    if check and rc != 0:
        rendered = " ".join(("docker", *args))
        raise RuntimeError(
            f"command failed ({rc}): {rendered}\nstdout:\n{stdout}\nstderr:\n{stderr}"
        )
    return rc, stdout, stderr


async def _get_host_port(container_name: str, internal_port: int) -> int:
    _, stdout, _ = await _run_docker(
        "inspect",
        "--format",
        f'{{{{(index (index .NetworkSettings.Ports "{internal_port}/tcp") 0).HostPort}}}}',
        container_name,
    )
    return int(stdout)


async def _wait_for_container_exit(container_name: str) -> int:
    while True:
        rc, stdout, _ = await _run_docker(
            "inspect",
            "--format",
            "{{.State.Running}} {{.State.ExitCode}}",
            container_name,
            check=False,
        )
        if rc != 0:
            return -1
        running, _, exit_code = stdout.partition(" ")
        if running.strip().lower() != "true":
            return int(exit_code.strip() or 0)
        await asyncio.sleep(2)


async def _wait_cua_or_container_exit(
    *,
    container_name: str,
    cua_url: str,
    os_type: str,
    timeout: float,
    poll_interval: float,
) -> bool:
    from .gcloud import wait_cua_ready

    ready_task = asyncio.create_task(
        wait_cua_ready(
            cua_url,
            os_type,
            timeout=timeout,
            poll_interval=poll_interval,
        )
    )
    exit_task = asyncio.create_task(_wait_for_container_exit(container_name))
    done, pending = await asyncio.wait(
        {ready_task, exit_task},
        return_when=asyncio.FIRST_COMPLETED,
    )
    for task in pending:
        task.cancel()
    await asyncio.gather(*pending, return_exceptions=True)

    if exit_task in done:
        exit_code = exit_task.result()
        _, logs, log_error = await _run_docker(
            "logs",
            "--tail",
            "120",
            container_name,
            check=False,
        )
        detail = logs or log_error or "no container logs"
        raise RuntimeError(
            f"QEMU container {container_name} exited with status {exit_code} "
            f"before CUA became ready; container logs:\n{detail[-6000:]}"
        )
    return ready_task.result()


class QemuProvider(Provider):
    """Provider backed by one ephemeral Docker-packaged QEMU VM per run."""

    def __init__(self, config: QemuProviderConfig | dict[str, Any]):
        if isinstance(config, dict):
            config = _build_provider_config(config)
        self._cfg = config
        self._image_locks: dict[str, asyncio.Lock] = {}
        self._preflight_lock = asyncio.Lock()
        self._preflight_done = False

    @property
    def config(self) -> QemuProviderConfig:
        return self._cfg

    async def acquire(self, spec: SandboxSpec) -> SandboxHandle:
        snapshot = self._cfg.snapshots.get(spec.snapshot)
        if snapshot is None:
            raise KeyError(
                f"snapshot {spec.snapshot!r} not in qemu provider config "
                f"(known: {sorted(self._cfg.snapshots)})"
            )
        if spec.gpu:
            raise RuntimeError(
                "qemu provider does not support GPU tasks yet. PCIe passthrough "
                "requires an IOMMU-enabled host, a dedicated VFIO-bound GPU, "
                "QEMU vfio-pci arguments, and matching guest drivers."
            )

        from ..images import get as get_image

        image = get_image(snapshot.image)
        if spec.os and spec.os != image.os:
            logger.warning(
                "os mismatch for %s: task declares %r but image %r is %r",
                spec.snapshot,
                spec.os,
                snapshot.image,
                image.os,
            )

        await self._preflight()
        base_qcow2 = await self._resolve_qcow2(snapshot)
        vcpus, memory_gb = self._resolve_shape(snapshot, spec)
        name = _generate_container_name(spec)
        slot_root = snapshot.runtime_root / "slots" / name
        storage_dir = slot_root / "storage"
        storage_dir.mkdir(parents=True, exist_ok=False)

        try:
            cua_internal_port = image.cua_server_port or _DEFAULT_CUA_PORT
            await self._create_overlay(
                snapshot=snapshot,
                base_qcow2=base_qcow2,
                storage_dir=storage_dir,
            )
            await self._start_container(
                snapshot=snapshot,
                name=name,
                base_qcow2=base_qcow2,
                storage_dir=storage_dir,
                cua_port=cua_internal_port,
                vcpus=vcpus,
                memory_gb=memory_gb,
            )

            cua_port = await _get_host_port(name, cua_internal_port)
            novnc_port = await _get_host_port(name, _DEFAULT_NOVNC_PORT)
            client_host = (
                "127.0.0.1" if snapshot.bind_address in {"0.0.0.0", "::"} else snapshot.bind_address
            )
            cua_url = f"http://{client_host}:{cua_port}"

            ready = await _wait_cua_or_container_exit(
                container_name=name,
                cua_url=cua_url,
                os_type=image.os,
                timeout=snapshot.ready_timeout_s,
                poll_interval=snapshot.readiness_poll_interval_s,
            )
            if not ready:
                _, logs, log_error = await _run_docker("logs", "--tail", "80", name, check=False)
                detail = logs or log_error or "no container logs"
                raise RuntimeError(
                    f"CUA server at {cua_url} did not become ready; "
                    f"container logs:\n{detail[-4000:]}"
                )

            logger.info(
                "QEMU sandbox %s ready: image=%s vcpus=%d memory=%dG cua=%s",
                name,
                snapshot.image,
                vcpus,
                memory_gb,
                cua_url,
            )
            return SandboxHandle(
                id=name,
                endpoint=cua_url,
                os=image.os,
                **image.sandbox_paths(),
                metadata={
                    "provider": "qemu",
                    "container_name": name,
                    "slot_root": str(slot_root),
                    "base_qcow2": str(base_qcow2),
                    "overlay_qcow2": str(storage_dir / "data.qcow2"),
                    "cua_port": cua_port,
                    "novnc_port": novnc_port,
                    "novnc_url": f"http://{client_host}:{novnc_port}",
                    "image": image.name,
                    "snapshot": spec.snapshot,
                    "machine_type": spec.machine_type,
                    "vcpus": vcpus,
                    "memory_gb": memory_gb,
                },
            )
        except BaseException:
            await _run_docker("rm", "-f", name, check=False)
            await asyncio.to_thread(shutil.rmtree, slot_root, True)
            raise

    async def release(
        self,
        sandbox: SandboxHandle,
        *,
        mode: ReleaseMode = "delete",
    ) -> None:
        if mode == "keep":
            logger.info("QEMU sandbox %s kept alive", sandbox.id)
            return
        if mode == "stop":
            logger.info("Stopping QEMU sandbox %s", sandbox.id)
            await _run_docker("stop", sandbox.id, check=False)
            return
        if mode != "delete":
            raise ValueError(f"unknown release mode: {mode!r}")

        logger.info("Deleting QEMU sandbox %s", sandbox.id)
        await _run_docker("rm", "-f", sandbox.id, check=False)
        slot_root = sandbox.metadata.get("slot_root")
        if slot_root:
            await asyncio.to_thread(shutil.rmtree, slot_root, True)

    def open_session(self, sandbox: SandboxHandle) -> Any:
        from cua_bench.computers.remote import RemoteDesktopSession
        from .gcloud import _init_computer_skip_wait

        session = RemoteDesktopSession(
            api_url=sandbox.endpoint,
            os_type=sandbox.os,
        )
        _init_computer_skip_wait(session)
        return session

    async def _preflight(self) -> None:
        if self._preflight_done:
            return
        async with self._preflight_lock:
            if self._preflight_done:
                return
            if shutil.which("docker") is None:
                raise RuntimeError("qemu provider requires Docker, but `docker` is not installed")
            if not Path("/dev/kvm").exists():
                raise RuntimeError(
                    "qemu provider requires /dev/kvm. Enable hardware or nested "
                    "virtualization on the host."
                )
            await _run_docker("version", "--format", "{{.Server.Version}}")
            self._preflight_done = True

    async def _resolve_qcow2(self, snapshot: QemuSnapshotConfig) -> Path:
        source = snapshot.qcow2
        if not source.startswith("gs://"):
            path = _expand_path(source)
            if not path.is_file():
                raise FileNotFoundError(f"qemu base image not found: {path}")
            return path

        filename = source.rstrip("/").rsplit("/", 1)[-1]
        if not filename:
            raise ValueError(f"invalid qemu qcow2 GCS URI: {source!r}")
        destination = snapshot.image_cache_dir / filename
        lock = self._image_locks.setdefault(source, asyncio.Lock())
        async with lock:
            if destination.is_file() and destination.stat().st_size > 0:
                return destination

            destination.parent.mkdir(parents=True, exist_ok=True)
            partial = destination.with_name(f"{destination.name}.partial")
            if shutil.which("gcloud"):
                command = ("gcloud", "storage", "cp", source, str(partial))
            elif shutil.which("gsutil"):
                command = ("gsutil", "-m", "cp", source, str(partial))
            else:
                raise RuntimeError("downloading a gs:// qcow2 requires `gcloud` or `gsutil`")

            logger.info("Downloading QEMU base image %s to %s", source, destination)
            rc, stdout, stderr = await _run_process(*command)
            if rc != 0:
                raise RuntimeError(
                    f"qcow2 download failed ({rc}): {' '.join(command)}\n"
                    f"stdout:\n{stdout}\nstderr:\n{stderr}"
                )
            if not partial.is_file() or partial.stat().st_size == 0:
                raise RuntimeError(f"qcow2 download produced no file: {partial}")
            os.replace(partial, destination)
            return destination

    @staticmethod
    def _resolve_shape(
        snapshot: QemuSnapshotConfig,
        spec: SandboxSpec,
    ) -> tuple[int, int]:
        from .gcloud import _DEFAULT_CPU_MACHINE, _parse_gce_machine_type

        shape = _parse_gce_machine_type(spec.machine_type or _DEFAULT_CPU_MACHINE)
        vcpus = snapshot.vcpus or (shape.vcpus if shape else 4)
        memory_gb = snapshot.memory_gb or (shape.memory_gb if shape else 8)
        return vcpus, memory_gb

    @staticmethod
    async def _create_overlay(
        *,
        snapshot: QemuSnapshotConfig,
        base_qcow2: Path,
        storage_dir: Path,
    ) -> None:
        await _run_docker(
            "run",
            "--rm",
            "--entrypoint",
            "qemu-img",
            "-v",
            f"{base_qcow2}:/images/base.qcow2:ro",
            "-v",
            f"{storage_dir}:/storage",
            snapshot.docker_image,
            "create",
            "-f",
            "qcow2",
            "-F",
            "qcow2",
            "-b",
            "/images/base.qcow2",
            "/storage/data.qcow2",
        )

    @staticmethod
    async def _start_container(
        *,
        snapshot: QemuSnapshotConfig,
        name: str,
        base_qcow2: Path,
        storage_dir: Path,
        cua_port: int,
        vcpus: int,
        memory_gb: int,
    ) -> None:
        await _run_docker(
            "run",
            "-d",
            "--name",
            name,
            "--device=/dev/kvm",
            "--cap-add",
            "NET_ADMIN",
            f"--shm-size={snapshot.shm_size}",
            "-v",
            f"{base_qcow2}:/images/base.qcow2:ro",
            "-v",
            f"{storage_dir}:/storage",
            "-p",
            f"{snapshot.bind_address}:0:{cua_port}",
            "-p",
            f"{snapshot.bind_address}:0:{_DEFAULT_NOVNC_PORT}",
            "-e",
            f"RAM_SIZE={memory_gb}G",
            "-e",
            f"CPU_CORES={vcpus}",
            "-e",
            "CPU_MODEL=host",
            "-e",
            "HV=N",
            "-e",
            "VM_NET_IP=172.30.0.2",
            snapshot.docker_image,
        )
        await _run_docker(
            "exec",
            name,
            "iptables",
            "-t",
            "nat",
            "-A",
            "POSTROUTING",
            "-d",
            "172.30.0.2/32",
            "-o",
            "docker",
            "-j",
            "MASQUERADE",
            check=False,
        )


__all__ = [
    "QemuProvider",
    "QemuProviderConfig",
    "QemuSnapshotConfig",
]
