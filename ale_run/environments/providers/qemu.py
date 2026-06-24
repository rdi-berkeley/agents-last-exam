"""Local full-OS sandboxes backed by QEMU packaged in Docker.

The host provider owns guest-disk caching and one VM lifecycle per ALE run.
The runner container only supplies QEMU, networking, noVNC, and process
supervision. Guest operating systems stay in separately distributed qcow2
images and expose the same CUA API as the cloud providers.
"""

from __future__ import annotations

import asyncio
import hashlib
import ipaddress
import json
import logging
import os
import re
import shutil
import stat
import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Literal, cast

from ...base_interface import Provider, ReleaseMode, SandboxHandle, SandboxSpec

logger = logging.getLogger(__name__)

RunnerPullPolicy = Literal["always", "missing", "never"]

_DEFAULT_RUNNER_IMAGE = "agentslastexam/ale-qemu:0.2.0"
_DEFAULT_CUA_PORT = 5000
_DEFAULT_NOVNC_PORT = 8006
_CONTAINER_PREFIX = "ale-qemu"
_SUPPORTED_PULL_POLICIES = frozenset({"always", "missing", "never"})
_REMOTE_SOURCE_RE = re.compile(r"^[a-z][a-z0-9+.-]*://", re.IGNORECASE)
_HF_DISK_MANIFEST_FORMAT = "ale-qemu-disk-parts-v1"
_HF_DISK_MANIFEST_SUFFIX = ".manifest.json"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SLOT_CLEANUP_ATTEMPTS = 3
_SLOT_CLEANUP_RETRY_S = 0.25


@dataclass(frozen=True, slots=True)
class QemuSnapshotConfig:
    """Local QEMU realization of one task-card snapshot."""

    image: str
    disk_source: str
    runtime_root: Path
    image_cache_dir: Path
    runner_image: str = _DEFAULT_RUNNER_IMAGE
    runner_pull_policy: RunnerPullPolicy = "missing"
    hf_revision: str | None = None
    vcpus: int = 0
    memory_gb: int = 0
    shm_size: str = "2g"
    bind_address: str = "127.0.0.1"
    ready_timeout_s: int = 900
    readiness_poll_interval_s: float = 5


@dataclass(frozen=True, slots=True)
class QemuProviderConfig:
    """Provider configuration keyed by task-card snapshot tag."""

    snapshots: dict[str, QemuSnapshotConfig]
    gcs_sa_key: str = ""
    """Optional host path to a GCS service-account key injected into each guest."""


@dataclass(frozen=True, slots=True)
class _HuggingFaceSource:
    repo_id: str
    filename: str


@dataclass(frozen=True, slots=True)
class _HuggingFaceDiskPart:
    filename: str
    size: int
    sha256: str


@dataclass(frozen=True, slots=True)
class _HuggingFaceDiskManifest:
    filename: str
    size: int
    sha256: str
    parts: tuple[_HuggingFaceDiskPart, ...]


def _expand_path(value: str) -> Path:
    return Path(value).expanduser().resolve()


def _parse_hf_source(source: str) -> _HuggingFaceSource:
    """Parse ``hf://<owner>/<repo>/<path>`` into Hub coordinates."""
    body = source.removeprefix("hf://").strip("/")
    parts = body.split("/")
    if len(parts) < 3 or any(not part for part in parts):
        raise ValueError(
            f"invalid QEMU disk source {source!r}; expected hf://<owner>/<dataset>/<path>"
        )
    return _HuggingFaceSource(
        repo_id="/".join(parts[:2]),
        filename="/".join(parts[2:]),
    )


def _validate_bind_address(value: str) -> str:
    try:
        address = ipaddress.ip_address(value)
    except ValueError as exc:
        raise ValueError(f"qemu bind_address must be an IP address, got {value!r}") from exc
    if address.version != 4:
        raise ValueError("qemu provider currently supports IPv4 bind_address values only")
    return value


def _validate_hf_relative_path(value: str, *, field: str) -> str:
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"invalid {field} in QEMU disk manifest: {value!r}")
    return value


def _parse_hf_disk_manifest(raw: Any) -> _HuggingFaceDiskManifest:
    if not isinstance(raw, dict) or raw.get("format") != _HF_DISK_MANIFEST_FORMAT:
        raise ValueError(f"QEMU disk manifest must use format {_HF_DISK_MANIFEST_FORMAT!r}")

    filename = _validate_hf_relative_path(str(raw.get("filename") or ""), field="filename")
    size = int(raw.get("size") or 0)
    sha256 = str(raw.get("sha256") or "")
    parts_raw = raw.get("parts")
    if size <= 0:
        raise ValueError("QEMU disk manifest size must be positive")
    if not _SHA256_RE.fullmatch(sha256):
        raise ValueError("QEMU disk manifest sha256 must be 64 lowercase hex characters")
    if not isinstance(parts_raw, list) or not parts_raw:
        raise ValueError("QEMU disk manifest requires a non-empty parts list")

    parts: list[_HuggingFaceDiskPart] = []
    part_filenames: set[str] = set()
    for index, part_raw in enumerate(parts_raw):
        if not isinstance(part_raw, dict):
            raise ValueError(f"QEMU disk manifest part {index} must be a mapping")
        part_filename = _validate_hf_relative_path(
            str(part_raw.get("filename") or ""),
            field=f"parts[{index}].filename",
        )
        part_size = int(part_raw.get("size") or 0)
        part_sha256 = str(part_raw.get("sha256") or "")
        if part_size <= 0:
            raise ValueError(f"QEMU disk manifest part {index} size must be positive")
        if not _SHA256_RE.fullmatch(part_sha256):
            raise ValueError(
                f"QEMU disk manifest part {index} sha256 must be 64 lowercase hex characters"
            )
        if part_filename == filename or part_filename in part_filenames:
            raise ValueError(f"QEMU disk manifest part {index} has a duplicate or unsafe filename")
        part_filenames.add(part_filename)
        parts.append(
            _HuggingFaceDiskPart(
                filename=part_filename,
                size=part_size,
                sha256=part_sha256,
            )
        )

    if sum(part.size for part in parts) != size:
        raise ValueError("QEMU disk manifest part sizes do not match total size")
    return _HuggingFaceDiskManifest(
        filename=filename,
        size=size,
        sha256=sha256,
        parts=tuple(parts),
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _build_snapshot_config(raw: dict[str, Any]) -> QemuSnapshotConfig:
    disk_source = str(raw.get("disk_source") or "").strip()
    if not disk_source:
        raise KeyError("qemu snapshot config missing required field `disk_source`")

    root = _expand_path(str(raw.get("root") or "~/.cache/ale/qemu"))
    runtime_root = _expand_path(str(raw.get("runtime_root") or root / "runtime"))
    image_cache_dir = _expand_path(str(raw.get("image_cache_dir") or root / "images"))
    runner_pull_policy = str(raw.get("runner_pull_policy") or "missing")
    if runner_pull_policy not in _SUPPORTED_PULL_POLICIES:
        raise ValueError(
            "qemu runner_pull_policy must be one of "
            f"{sorted(_SUPPORTED_PULL_POLICIES)}, got {runner_pull_policy!r}"
        )

    vcpus = int(raw.get("vcpus") or 0)
    memory_gb = int(raw.get("memory_gb") or 0)
    ready_timeout_s = int(raw.get("ready_timeout_s") or 900)
    readiness_poll_interval_s = float(raw.get("readiness_poll_interval_s") or 5)
    if vcpus < 0:
        raise ValueError(f"qemu vcpus must be >= 0, got {vcpus}")
    if memory_gb < 0:
        raise ValueError(f"qemu memory_gb must be >= 0, got {memory_gb}")
    if ready_timeout_s <= 0:
        raise ValueError(f"qemu ready_timeout_s must be > 0, got {ready_timeout_s}")
    if readiness_poll_interval_s <= 0:
        raise ValueError(
            f"qemu readiness_poll_interval_s must be > 0, got {readiness_poll_interval_s}"
        )

    hf_revision = str(raw.get("hf_revision") or "").strip() or None
    if hf_revision and not disk_source.startswith("hf://"):
        raise ValueError("qemu hf_revision is valid only for an hf:// disk_source")

    return QemuSnapshotConfig(
        image=str(raw["image"]),
        disk_source=disk_source,
        runtime_root=runtime_root,
        image_cache_dir=image_cache_dir,
        runner_image=str(raw.get("runner_image") or _DEFAULT_RUNNER_IMAGE),
        runner_pull_policy=cast(RunnerPullPolicy, runner_pull_policy),
        hf_revision=hf_revision,
        vcpus=vcpus,
        memory_gb=memory_gb,
        shm_size=str(raw.get("shm_size") or "2g"),
        bind_address=_validate_bind_address(str(raw.get("bind_address") or "127.0.0.1")),
        ready_timeout_s=ready_timeout_s,
        readiness_poll_interval_s=readiness_poll_interval_s,
    )


def _cache_destination(snapshot: QemuSnapshotConfig, source: str) -> Path:
    if source.startswith("hf://"):
        filename = _parse_hf_source(source).filename
        if filename.endswith(_HF_DISK_MANIFEST_SUFFIX):
            filename = filename[: -len(_HF_DISK_MANIFEST_SUFFIX)]
        return snapshot.image_cache_dir / filename
    filename = source.rstrip("/").rsplit("/", 1)[-1]
    if not filename:
        raise ValueError(f"invalid QEMU disk source: {source!r}")
    return snapshot.image_cache_dir / filename


def _build_provider_config(raw: dict[str, Any]) -> QemuProviderConfig:
    snapshots_raw = raw.get("snapshots")
    if not isinstance(snapshots_raw, dict) or not snapshots_raw:
        raise TypeError("qemu provider config requires a non-empty `snapshots` mapping")

    snapshots = {
        str(tag): _build_snapshot_config(dict(snapshot_raw))
        for tag, snapshot_raw in snapshots_raw.items()
    }
    cache_owners: dict[Path, str] = {}
    for snapshot in snapshots.values():
        if not snapshot.disk_source.startswith(("hf://", "gs://")):
            continue
        destination = _cache_destination(snapshot, snapshot.disk_source)
        owner = snapshot.disk_source
        if snapshot.disk_source.startswith("hf://"):
            owner = f"{owner}@{snapshot.hf_revision or 'main'}"
        previous = cache_owners.setdefault(destination, owner)
        if previous != owner:
            raise ValueError(
                f"qemu disk identities {previous!r} and {owner!r} "
                f"would share cache path {destination}"
            )
    gcs_sa_key = str(raw.get("gcs_sa_key") or "").strip()
    if gcs_sa_key:
        gcs_sa_key = str(Path(gcs_sa_key).expanduser().resolve())
    return QemuProviderConfig(
        snapshots=snapshots,
        gcs_sa_key=gcs_sa_key,
    )


def _generate_container_name(spec: SandboxSpec) -> str:
    source = spec.task_id or spec.snapshot
    body = re.sub(r"[^a-z0-9]", "-", source.lower()).strip("-")[:40]
    return f"{_CONTAINER_PREFIX}-{body or 'sandbox'}-{uuid.uuid4().hex[:8]}"


def _run_locked(lock_path: Path, operation: Callable[[], Path]) -> Path:
    try:
        import fcntl
    except ImportError as exc:
        raise RuntimeError("qemu provider requires a Linux host") from exc

    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            return operation()
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


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


async def _remove_container(container_name: str) -> None:
    rc, _, remove_error = await _run_docker("rm", "-f", container_name, check=False)
    if rc == 0:
        return
    inspect_rc, _, inspect_error = await _run_docker(
        "inspect",
        container_name,
        check=False,
    )
    if inspect_rc != 0 and "no such" in f"{remove_error}\n{inspect_error}".lower():
        return
    raise RuntimeError(
        f"failed to remove QEMU container {container_name}; preserving its disk: "
        f"{remove_error or inspect_error}"
    )


def _remove_slot_root(slot_root: Path) -> None:
    def make_writable_and_retry(
        operation: Callable[[str], object],
        path: str,
        error: BaseException,
    ) -> None:
        try:
            failed_path = Path(path)
            os.chmod(failed_path, failed_path.stat().st_mode | stat.S_IRWXU)
            parent = failed_path.parent
            os.chmod(parent, parent.stat().st_mode | stat.S_IRWXU)
            operation(path)
        except OSError:
            raise error

    last_error: OSError | None = None
    for attempt in range(1, _SLOT_CLEANUP_ATTEMPTS + 1):
        try:
            shutil.rmtree(slot_root, onexc=make_writable_and_retry)
        except FileNotFoundError:
            return
        except OSError as exc:
            last_error = exc
        if not slot_root.exists():
            return
        if attempt < _SLOT_CLEANUP_ATTEMPTS:
            time.sleep(_SLOT_CLEANUP_RETRY_S)

    detail = f": {last_error}" if last_error else ""
    raise RuntimeError(
        f"failed to remove QEMU runtime slot after {_SLOT_CLEANUP_ATTEMPTS} attempts: "
        f"{slot_root}{detail}"
    )


class QemuProvider(Provider):
    """One ephemeral Docker-packaged QEMU VM per ALE run."""

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
                "qemu provider is currently CPU-only. GPU passthrough requires "
                "an IOMMU-enabled host, a dedicated VFIO-bound GPU, runner "
                "device wiring, and matching guest drivers."
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
        base_qcow2 = await self._resolve_disk(snapshot)
        vcpus, memory_gb = self._resolve_shape(snapshot, spec)
        name = _generate_container_name(spec)
        slot_root = snapshot.runtime_root / "slots" / name
        storage_dir = slot_root / "storage"
        exchange_dir = slot_root / "exchange"
        storage_dir.mkdir(parents=True, exist_ok=False)
        exchange_dir.mkdir()
        container_started = False

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
                task_id=spec.task_id,
                base_qcow2=base_qcow2,
                storage_dir=storage_dir,
                exchange_dir=exchange_dir,
                cua_port=cua_internal_port,
                vcpus=vcpus,
                memory_gb=memory_gb,
            )
            container_started = True
            await self._configure_hairpin_nat(name)

            cua_port = await _get_host_port(name, cua_internal_port)
            novnc_port = await _get_host_port(name, _DEFAULT_NOVNC_PORT)
            client_host = (
                "127.0.0.1" if snapshot.bind_address == "0.0.0.0" else snapshot.bind_address
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
                _, logs, log_error = await _run_docker(
                    "logs",
                    "--tail",
                    "80",
                    name,
                    check=False,
                )
                detail = logs or log_error or "no container logs"
                raise RuntimeError(
                    f"CUA server at {cua_url} did not become ready; "
                    f"container logs:\n{detail[-4000:]}"
                )

            metadata = {
                "provider": "qemu",
                "container_name": name,
                "slot_root": str(slot_root),
                "base_qcow2": str(base_qcow2),
                "overlay_qcow2": str(storage_dir / "data.qcow2"),
                "exchange_host_dir": str(exchange_dir),
                "exchange_guest_share": (
                    "//172.30.0.1/Data" if image.os == "linux" else r"\\host.lan\Data"
                ),
                "cua_port": cua_port,
                "novnc_port": novnc_port,
                "novnc_url": f"http://{client_host}:{novnc_port}",
                "image": image.name,
                "snapshot": spec.snapshot,
                "machine_type": spec.machine_type,
                "vcpus": vcpus,
                "memory_gb": memory_gb,
                "runner_image": snapshot.runner_image,
            }
            sandbox = SandboxHandle(
                id=name,
                endpoint=cua_url,
                os=image.os,
                **image.sandbox_paths(),
                metadata=metadata,
            )
            if self._cfg.gcs_sa_key:
                gcs_key_path, gcs_user_project = await self._inject_gcs_credentials(
                    sandbox,
                    self._cfg.gcs_sa_key,
                )
                metadata["gcs_key_path"] = gcs_key_path
                metadata["gcs_user_project"] = gcs_user_project

            logger.info(
                "QEMU sandbox %s ready: image=%s vcpus=%d memory=%dG cua=%s",
                name,
                snapshot.image,
                vcpus,
                memory_gb,
                cua_url,
            )
            return sandbox
        except BaseException:
            try:
                if container_started:
                    await _remove_container(name)
                await asyncio.to_thread(_remove_slot_root, slot_root)
            except Exception:
                logger.exception(
                    "failed to clean up QEMU sandbox %s after acquire failure; "
                    "slot preserved at %s",
                    name,
                    slot_root,
                )
            raise

    @staticmethod
    async def _inject_gcs_credentials(
        sandbox: SandboxHandle,
        host_key_path: str,
    ) -> tuple[str, str]:
        """Copy a GCS service-account key into the guest."""
        key = Path(host_key_path)
        if not key.is_file():
            raise FileNotFoundError(f"gcs_sa_key not found: {host_key_path}")
        data = await asyncio.to_thread(key.read_bytes)
        try:
            parsed = json.loads(data)
            project_id = parsed.get("project_id", "") if isinstance(parsed, dict) else ""
        except json.JSONDecodeError:
            project_id = ""

        if sandbox.is_linux:
            destination = "/tmp/agenthle/gcs-reader.json"
            prepare = "mkdir -p /tmp/agenthle"
            protect = f"chmod 600 {destination}"
        else:
            destination = r"C:\agenthle\gcs-reader.json"
            prepare = r"cmd /c if not exist C:\agenthle mkdir C:\agenthle"
            protect = ""

        result = await sandbox.run_command(prepare)
        if result.returncode != 0:
            raise RuntimeError(
                f"failed to prepare QEMU GCS credential directory: "
                f"{(result.stderr or result.stdout or '')[:300]}"
            )
        await sandbox.write_file(destination, data)
        if protect:
            result = await sandbox.run_command(protect)
            if result.returncode != 0:
                raise RuntimeError(
                    f"failed to protect QEMU GCS credential: "
                    f"{(result.stderr or result.stdout or '')[:300]}"
                )
        logger.info(
            "QEMU: injected GCS SA key -> %s (project=%s)",
            destination,
            project_id,
        )
        return destination, str(project_id or "")

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
            await _run_docker("stop", sandbox.id)
            return
        if mode != "delete":
            raise ValueError(f"unknown release mode: {mode!r}")

        logger.info("Deleting QEMU sandbox %s", sandbox.id)
        await _remove_container(sandbox.id)
        slot_root = sandbox.metadata.get("slot_root")
        if slot_root:
            await asyncio.to_thread(_remove_slot_root, Path(slot_root))

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
            kvm = Path("/dev/kvm")
            if not kvm.exists():
                raise RuntimeError(
                    "qemu provider requires /dev/kvm. Enable hardware or nested "
                    "virtualization on the host."
                )
            await _run_docker("version", "--format", "{{.Server.Version}}")
            self._preflight_done = True

    async def _resolve_disk(self, snapshot: QemuSnapshotConfig) -> Path:
        source = snapshot.disk_source
        if source.startswith("hf://"):
            return await self._fetch_hf_disk(snapshot)
        if source.startswith("gs://"):
            return await self._fetch_gcs_disk(snapshot)
        if _REMOTE_SOURCE_RE.match(source):
            raise ValueError(
                f"unsupported QEMU disk source scheme in {source!r}; "
                "supported sources are hf://, gs://, and local paths"
            )
        path = _expand_path(source)
        if not path.is_file():
            raise FileNotFoundError(f"QEMU base disk not found: {path}")
        if path.stat().st_size == 0:
            raise ValueError(f"QEMU base disk is empty: {path}")
        return path

    async def _fetch_gcs_disk(self, snapshot: QemuSnapshotConfig) -> Path:
        source = snapshot.disk_source
        destination = _cache_destination(snapshot, source)
        lock = self._image_locks.setdefault(source, asyncio.Lock())
        async with lock:
            lock_path = (
                snapshot.image_cache_dir
                / ".locks"
                / (hashlib.sha256(source.encode()).hexdigest() + ".lock")
            )

            def download() -> Path:
                destination.parent.mkdir(parents=True, exist_ok=True)
                partial = destination.with_name(f"{destination.name}.partial")
                sidecar = destination.with_name(f"{destination.name}.ale-source.json")
                if shutil.which("gcloud"):
                    describe_command = (
                        "gcloud",
                        "storage",
                        "objects",
                        "describe",
                        source,
                        "--format=json",
                    )
                    describe_result = subprocess.run(
                        describe_command,
                        text=True,
                        capture_output=True,
                        check=False,
                    )
                    if describe_result.returncode != 0:
                        raise RuntimeError(
                            f"QEMU disk metadata lookup failed "
                            f"({describe_result.returncode}): "
                            f"{' '.join(describe_command)}\n"
                            f"stdout:\n{describe_result.stdout}\n"
                            f"stderr:\n{describe_result.stderr}"
                        )
                    try:
                        remote_raw = json.loads(describe_result.stdout)
                        remote_size = int(remote_raw["size"])
                        generation = str(remote_raw["generation"])
                    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                        raise RuntimeError(f"invalid GCS metadata for QEMU disk {source}") from exc
                    remote_metadata = {
                        "source": source,
                        "generation": generation,
                        "etag": str(remote_raw.get("etag") or ""),
                        "crc32c": str(remote_raw.get("crc32c_hash") or ""),
                        "size": remote_size,
                    }
                    versioned_source = f"{source}#{generation}"
                    command = ("gcloud", "storage", "cp", versioned_source, str(partial))
                elif shutil.which("gsutil"):
                    describe_command = ("gsutil", "stat", source)
                    describe_result = subprocess.run(
                        describe_command,
                        text=True,
                        capture_output=True,
                        check=False,
                    )
                    if describe_result.returncode != 0:
                        raise RuntimeError(
                            f"QEMU disk metadata lookup failed "
                            f"({describe_result.returncode}): "
                            f"{' '.join(describe_command)}\n"
                            f"stdout:\n{describe_result.stdout}\n"
                            f"stderr:\n{describe_result.stderr}"
                        )
                    fields: dict[str, str] = {}
                    for line in describe_result.stdout.splitlines():
                        key, separator, value = line.strip().partition(":")
                        if separator:
                            fields[key] = value.strip()
                    try:
                        remote_size = int(fields["Content-Length"])
                        generation = fields["Generation"]
                    except (KeyError, ValueError) as exc:
                        raise RuntimeError(f"invalid GCS metadata for QEMU disk {source}") from exc
                    remote_metadata = {
                        "source": source,
                        "generation": generation,
                        "etag": fields.get("ETag", ""),
                        "crc32c": fields.get("Hash (crc32c)", ""),
                        "size": remote_size,
                    }
                    versioned_source = f"{source}#{generation}"
                    command = ("gsutil", "-m", "cp", versioned_source, str(partial))
                else:
                    raise RuntimeError(
                        "downloading a gs:// QEMU disk requires `gcloud` or `gsutil`"
                    )

                try:
                    cached_metadata = json.loads(sidecar.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    cached_metadata = {}
                if (
                    destination.is_file()
                    and destination.stat().st_size == remote_size
                    and cached_metadata == remote_metadata
                ):
                    return destination

                logger.info("Downloading QEMU base disk %s to %s", source, destination)
                partial.unlink(missing_ok=True)
                result = subprocess.run(
                    command,
                    text=True,
                    capture_output=True,
                    check=False,
                )
                if result.returncode != 0:
                    raise RuntimeError(
                        f"QEMU disk download failed ({result.returncode}): "
                        f"{' '.join(command)}\nstdout:\n{result.stdout}\n"
                        f"stderr:\n{result.stderr}"
                    )
                if not partial.is_file() or partial.stat().st_size != remote_size:
                    actual_size = partial.stat().st_size if partial.is_file() else 0
                    raise RuntimeError(
                        f"QEMU disk download size mismatch for {source}: "
                        f"expected {remote_size}, got {actual_size}"
                    )
                os.replace(partial, destination)
                sidecar_tmp = sidecar.with_suffix(f"{sidecar.suffix}.tmp")
                sidecar_tmp.write_text(
                    json.dumps(remote_metadata, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                os.replace(sidecar_tmp, sidecar)
                return destination

            return await asyncio.to_thread(_run_locked, lock_path, download)

    async def _fetch_hf_disk(self, snapshot: QemuSnapshotConfig) -> Path:
        source = snapshot.disk_source
        parsed = _parse_hf_source(source)
        lock = self._image_locks.setdefault(source, asyncio.Lock())
        async with lock:
            lock_identity = f"{source}@{snapshot.hf_revision or 'main'}"
            lock_path = (
                snapshot.image_cache_dir
                / ".locks"
                / (hashlib.sha256(lock_identity.encode()).hexdigest() + ".lock")
            )

            def download() -> Path:
                try:
                    from huggingface_hub import (
                        get_hf_file_metadata,
                        hf_hub_download,
                        hf_hub_url,
                    )
                    from huggingface_hub.errors import EntryNotFoundError
                except ImportError as exc:
                    raise RuntimeError(
                        "downloading an hf:// QEMU disk requires `huggingface-hub`; "
                        "install the project dependencies with `uv sync`"
                    ) from exc

                snapshot.image_cache_dir.mkdir(parents=True, exist_ok=True)
                artifact_filename = parsed.filename
                if not artifact_filename.endswith(_HF_DISK_MANIFEST_SUFFIX):
                    manifest_filename = f"{artifact_filename}{_HF_DISK_MANIFEST_SUFFIX}"
                    manifest_url = hf_hub_url(
                        repo_id=parsed.repo_id,
                        filename=manifest_filename,
                        repo_type="dataset",
                        revision=snapshot.hf_revision,
                    )
                    try:
                        metadata = get_hf_file_metadata(manifest_url)
                        artifact_filename = manifest_filename
                    except EntryNotFoundError:
                        artifact_url = hf_hub_url(
                            repo_id=parsed.repo_id,
                            filename=artifact_filename,
                            repo_type="dataset",
                            revision=snapshot.hf_revision,
                        )
                        metadata = get_hf_file_metadata(artifact_url)
                else:
                    artifact_url = hf_hub_url(
                        repo_id=parsed.repo_id,
                        filename=artifact_filename,
                        repo_type="dataset",
                        revision=snapshot.hf_revision,
                    )
                    metadata = get_hf_file_metadata(artifact_url)

                if artifact_filename.endswith(_HF_DISK_MANIFEST_SUFFIX):
                    manifest_path = Path(
                        hf_hub_download(
                            repo_id=parsed.repo_id,
                            filename=artifact_filename,
                            repo_type="dataset",
                            revision=snapshot.hf_revision,
                            local_dir=snapshot.image_cache_dir,
                        )
                    )
                    try:
                        manifest = _parse_hf_disk_manifest(
                            json.loads(manifest_path.read_text(encoding="utf-8"))
                        )
                    except (OSError, json.JSONDecodeError) as exc:
                        raise ValueError(f"invalid QEMU disk manifest {source}: {exc}") from exc
                    destination = snapshot.image_cache_dir / manifest.filename
                    expected_size = manifest.size
                    expected_sha256 = manifest.sha256
                else:
                    manifest = None
                    destination = snapshot.image_cache_dir / artifact_filename
                    expected_size = metadata.size
                    expected_sha256 = None

                sidecar = destination.with_name(f"{destination.name}.ale-source.json")

                def write_sidecar() -> None:
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    sidecar_tmp = sidecar.with_suffix(f"{sidecar.suffix}.tmp")
                    sidecar_tmp.write_text(
                        json.dumps(
                            {
                                "source": source,
                                "revision": snapshot.hf_revision or "main",
                                "artifact": artifact_filename,
                                "commit_hash": metadata.commit_hash,
                                "etag": metadata.etag,
                                "size": expected_size,
                                "sha256": expected_sha256,
                            },
                            indent=2,
                            sort_keys=True,
                        )
                        + "\n",
                        encoding="utf-8",
                    )
                    os.replace(sidecar_tmp, sidecar)

                source_metadata: dict[str, Any] = {}
                if sidecar.exists():
                    try:
                        source_metadata = json.loads(sidecar.read_text(encoding="utf-8"))
                    except (OSError, json.JSONDecodeError):
                        source_metadata = {}
                if (
                    destination.is_file()
                    and expected_size is not None
                    and destination.stat().st_size == expected_size
                ):
                    if (
                        source_metadata.get("source") == source
                        and source_metadata.get("revision") == (snapshot.hf_revision or "main")
                        and source_metadata.get("artifact", artifact_filename) == artifact_filename
                        and source_metadata.get("commit_hash") == metadata.commit_hash
                        and source_metadata.get("etag") == metadata.etag
                        and source_metadata.get("sha256") == expected_sha256
                    ):
                        return destination
                    if (
                        source_metadata.get("etag") == metadata.etag
                        and source_metadata.get("sha256") == expected_sha256
                    ):
                        write_sidecar()
                        return destination
                    if expected_sha256 and _sha256_file(destination) == expected_sha256:
                        write_sidecar()
                        return destination
                    if manifest is None and not sidecar.exists():
                        write_sidecar()
                        return destination

                logger.info(
                    "Downloading QEMU base disk %s revision=%s to %s",
                    source,
                    snapshot.hf_revision or "main",
                    destination,
                )
                if manifest is not None:
                    return self._assemble_hf_disk(
                        snapshot=snapshot,
                        parsed=parsed,
                        metadata=metadata,
                        manifest=manifest,
                        destination=destination,
                        write_sidecar=write_sidecar,
                        hf_hub_download=hf_hub_download,
                    )

                destination.parent.mkdir(parents=True, exist_ok=True)
                result = Path(
                    hf_hub_download(
                        repo_id=parsed.repo_id,
                        filename=artifact_filename,
                        repo_type="dataset",
                        revision=snapshot.hf_revision,
                        local_dir=snapshot.image_cache_dir,
                        force_download=destination.exists(),
                    )
                )
                if not result.is_file() or result.stat().st_size == 0:
                    raise RuntimeError(f"QEMU disk download produced no file: {result}")
                if expected_size is not None and result.stat().st_size != expected_size:
                    raise RuntimeError(
                        f"QEMU disk size mismatch for {source}: "
                        f"expected {expected_size}, got {result.stat().st_size}"
                    )
                write_sidecar()
                return result

            return await asyncio.to_thread(_run_locked, lock_path, download)

    @staticmethod
    def _assemble_hf_disk(
        *,
        snapshot: QemuSnapshotConfig,
        parsed: _HuggingFaceSource,
        metadata: Any,
        manifest: _HuggingFaceDiskManifest,
        destination: Path,
        write_sidecar: Callable[[], None],
        hf_hub_download: Callable[..., str],
    ) -> Path:
        partial = destination.with_name(f"{destination.name}.partial")
        state_path = partial.with_name(f"{partial.name}.json")
        completed_parts = 0
        completed_bytes = 0

        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            state = {}
        if (
            partial.is_file()
            and state.get("source") == snapshot.disk_source
            and state.get("revision") == (snapshot.hf_revision or "main")
            and state.get("commit_hash") == metadata.commit_hash
            and state.get("manifest_sha256") == manifest.sha256
        ):
            completed_parts = int(state.get("completed_parts") or 0)
            if 0 <= completed_parts <= len(manifest.parts):
                completed_bytes = sum(part.size for part in manifest.parts[:completed_parts])
                if partial.stat().st_size >= completed_bytes:
                    with partial.open("r+b") as partial_file:
                        partial_file.truncate(completed_bytes)
                else:
                    completed_parts = 0
                    completed_bytes = 0
            else:
                completed_parts = 0
        if completed_parts == 0:
            partial.unlink(missing_ok=True)
            state_path.unlink(missing_ok=True)
        else:
            for part in manifest.parts[:completed_parts]:
                (snapshot.image_cache_dir / part.filename).unlink(missing_ok=True)

        destination.parent.mkdir(parents=True, exist_ok=True)
        with partial.open("ab") as output:
            for index, part in enumerate(
                manifest.parts[completed_parts:],
                start=completed_parts,
            ):
                part_path = Path(
                    hf_hub_download(
                        repo_id=parsed.repo_id,
                        filename=part.filename,
                        repo_type="dataset",
                        revision=snapshot.hf_revision,
                        local_dir=snapshot.image_cache_dir,
                    )
                )
                if (
                    not part_path.is_file()
                    or part_path.stat().st_size != part.size
                    or _sha256_file(part_path) != part.sha256
                ):
                    part_path.unlink(missing_ok=True)
                    part_path = Path(
                        hf_hub_download(
                            repo_id=parsed.repo_id,
                            filename=part.filename,
                            repo_type="dataset",
                            revision=snapshot.hf_revision,
                            local_dir=snapshot.image_cache_dir,
                            force_download=True,
                        )
                    )
                if (
                    not part_path.is_file()
                    or part_path.stat().st_size != part.size
                    or _sha256_file(part_path) != part.sha256
                ):
                    raise RuntimeError(f"QEMU disk part failed verification: {part.filename}")

                with part_path.open("rb") as part_file:
                    shutil.copyfileobj(part_file, output, length=8 * 1024 * 1024)
                output.flush()
                os.fsync(output.fileno())
                completed_bytes += part.size
                state_tmp = state_path.with_suffix(f"{state_path.suffix}.tmp")
                state_tmp.write_text(
                    json.dumps(
                        {
                            "source": snapshot.disk_source,
                            "revision": snapshot.hf_revision or "main",
                            "commit_hash": metadata.commit_hash,
                            "manifest_sha256": manifest.sha256,
                            "completed_parts": index + 1,
                            "completed_bytes": completed_bytes,
                        },
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                os.replace(state_tmp, state_path)
                part_path.unlink(missing_ok=True)

        if partial.stat().st_size != manifest.size:
            raise RuntimeError(
                f"QEMU disk assembly size mismatch: expected {manifest.size}, "
                f"got {partial.stat().st_size}"
            )
        if _sha256_file(partial) != manifest.sha256:
            partial.unlink(missing_ok=True)
            state_path.unlink(missing_ok=True)
            raise RuntimeError("QEMU disk assembly sha256 mismatch")
        os.replace(partial, destination)
        state_path.unlink(missing_ok=True)
        write_sidecar()
        return destination

    @staticmethod
    def _resolve_shape(
        snapshot: QemuSnapshotConfig,
        spec: SandboxSpec,
    ) -> tuple[int, int]:
        from ..images import get as get_image
        from .gcloud import _parse_gce_machine_type

        default_machine_type = get_image(snapshot.image).default_machine_type
        shape = _parse_gce_machine_type(spec.machine_type or default_machine_type)
        vcpus = snapshot.vcpus or spec.vcpus or (shape.vcpus if shape else 4)
        memory_gb = snapshot.memory_gb or spec.memory_gb or (shape.memory_gb if shape else 8)
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
            f"--pull={snapshot.runner_pull_policy}",
            "--entrypoint",
            "qemu-img",
            "--mount",
            f"type=bind,src={base_qcow2},dst=/images/base.qcow2,readonly",
            "--mount",
            f"type=bind,src={storage_dir},dst=/storage",
            snapshot.runner_image,
            "create",
            "-f",
            "qcow2",
            "-F",
            "qcow2",
            "-b",
            "/images/base.qcow2",
            "/storage/data.qcow2",
        )
        overlay = storage_dir / "data.qcow2"
        if not overlay.is_file() or overlay.stat().st_size == 0:
            raise RuntimeError(f"qemu-img did not create overlay disk: {overlay}")

    @staticmethod
    async def _start_container(
        *,
        snapshot: QemuSnapshotConfig,
        name: str,
        task_id: str,
        base_qcow2: Path,
        storage_dir: Path,
        exchange_dir: Path,
        cua_port: int,
        vcpus: int,
        memory_gb: int,
    ) -> None:
        run_args = [
            "run",
            "-d",
            f"--pull={snapshot.runner_pull_policy}",
            "--name",
            name,
            "--label",
            "io.agents-last-exam.provider=qemu",
            "--label",
            f"io.agents-last-exam.task={task_id or 'unknown'}",
            "--device=/dev/kvm",
            "--cap-add",
            "NET_ADMIN",
            f"--shm-size={snapshot.shm_size}",
            "--mount",
            f"type=bind,src={base_qcow2},dst=/images/base.qcow2,readonly",
            "--mount",
            f"type=bind,src={storage_dir},dst=/storage",
            "--mount",
            f"type=bind,src={exchange_dir},dst=/shared",
            "--publish",
            f"{snapshot.bind_address}:0:{cua_port}",
            "--publish",
            f"{snapshot.bind_address}:0:{_DEFAULT_NOVNC_PORT}",
            "--env",
            f"RAM_SIZE={memory_gb}G",
            "--env",
            f"CPU_CORES={vcpus}",
            "--env",
            "CPU_MODEL=host",
            "--env",
            "HV=N",
            "--env",
            "VM_NET_IP=172.30.0.2",
            snapshot.runner_image,
        ]
        await _run_docker(*run_args)

    @staticmethod
    async def _configure_hairpin_nat(name: str) -> None:
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
        )


__all__ = [
    "QemuProvider",
    "QemuProviderConfig",
    "QemuSnapshotConfig",
]
