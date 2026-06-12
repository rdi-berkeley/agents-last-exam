"""DockerProvider — ephemeral Docker containers via ``docker run``.

Each ``acquire`` launches a fresh container from the ale-kasm image,
maps a random host port to the container's cua-server (port 8000),
waits for the server to become healthy, and returns a SandboxHandle
whose ``endpoint`` is ``http://localhost:<host_port>``.

Concurrent tasks each get their own container + port — no proxy layer
required. Docker's native port mapping IS the routing mechanism.

Container naming follows the same ``<prefix>-<task-slug>-<hash8>``
scheme as GCE VMs for log greppability.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import random
import re
import time
from dataclasses import dataclass
from typing import Any

from ...base_interface import SandboxSpec, Provider, ReleaseMode, SandboxHandle

logger = logging.getLogger(__name__)

_CUA_INTERNAL_PORT = 8000
_VNC_INTERNAL_PORT = 6901
_CUA_READY_TIMEOUT = 120
_CUA_READY_POLL_INTERVAL = 3
_CUA_READY_STABLE_SUCCESSES = 2

_DEFAULT_CONTAINER_REF = "agentslastexam/ale-kasm:latest"  # fallback if an Image has no docker_image
_CONTAINER_PREFIX = "ale"                                  # container name prefix (not configurable)
_KASM_ENTRYPOINT = "/dockerstartup/vnc_startup.sh"


_GCS_KEY_CONTAINER_PATH = "/etc/agenthle/gcs-reader.json"
# System-wide boto config so gsutil authenticates regardless of which user the
# container runs as (kasm-user on ale-kasm, user on ale-ubuntu22-docker, ...).
# boto reads /etc/boto.cfg before any per-user ~/.boto, so this is user-agnostic.
_BOTO_CONFIG_CONTAINER_PATH = "/etc/boto.cfg"


@dataclass(frozen=True)
class DockerProviderConfig:
    """Docker provider config (the per-snapshot ``docker:`` block).

    image       Image NAME (registry key, e.g. "ale-kasm"). The container ref to
                boot + the cua-server port + sandbox paths are read from that
                Image's registry entry — nothing image-specific is configured here.
    shm_size    Shared memory size (default: 512m).
    cpus        CPU limit per container (0 = unlimited).
    memory      Memory limit per container (e.g. "8g", "" = unlimited).
    gcs_sa_key  Host path to a GCS service-account JSON key. NOT a provider knob —
                the loader injects it from the env-yaml top level (it travels with
                ``task_data_source: gs://…``). When set, it is copied into each
                container and gsutil/boto configured so gs:// staging works; the
                key's ``project_id`` also bills requester-pays buckets.
    privileged  Run containers with ``--privileged``. Needed for tasks whose eval
                runs nested Docker (openroad, minikube/k8s, compose) — the baked
                image starts a fresh inner dockerd. Under ROOTLESS docker this is
                user-namespace-bounded (container root maps to the unprivileged
                host user), so it does not expose the host kernel.
    image_ref   Override the container ref to boot (default: the Image entry's
                ``docker_image``). Lets one env config pin a specific tag, e.g.
                a DinD-capable build, without editing the Image registry.
    """

    image: str = "ale-kasm"
    shm_size: str = "512m"
    cpus: float = 0
    memory: str = ""
    gcs_sa_key: str = ""
    privileged: bool = False
    image_ref: str = ""


def _build_provider_config(raw: dict[str, Any]) -> DockerProviderConfig:
    gcs_sa = raw.get("gcs_sa_key") or ""
    if gcs_sa:
        from pathlib import Path as _P
        gcs_sa = str(_P(gcs_sa).expanduser().resolve())
    return DockerProviderConfig(
        image=str(raw.get("image") or "ale-kasm"),
        shm_size=str(raw.get("shm_size") or "512m"),
        cpus=float(raw.get("cpus") or 0),
        memory=str(raw.get("memory") or ""),
        gcs_sa_key=gcs_sa,
        privileged=bool(raw.get("privileged") or False),
        image_ref=str(raw.get("image_ref") or ""),
    )


def _generate_container_name(
    prefix: str,
    *,
    snapshot: str,
    task_id: str = "",
    harness: str = "",
    model_tag: str = "",
) -> str:
    if task_id:
        body = re.sub(r"[^a-z0-9]", "-", task_id.lower()).strip("-")[:40]
    else:
        body = re.sub(r"[^a-z0-9]", "-", snapshot.lower()).strip("-")[:30]
    seed = f"{prefix}:{task_id}:{harness}:{model_tag}:{snapshot}:{time.time()}:{random.random()}"
    h = hashlib.sha256(seed.encode()).hexdigest()[:8]
    return f"{prefix}-{body}-{h}"


async def _run_docker(*args: str) -> tuple[int, str, str]:
    cmd = ["docker", *args]
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout_b, stderr_b = await proc.communicate()
    return (
        proc.returncode or 0,
        stdout_b.decode(errors="replace").strip(),
        stderr_b.decode(errors="replace").strip(),
    )


async def _get_host_port(container_name: str, internal_port: int) -> int:
    """Inspect the container to find the host port mapped to internal_port."""
    rc, stdout, stderr = await _run_docker(
        "inspect",
        "--format",
        f'{{{{(index (index .NetworkSettings.Ports "{internal_port}/tcp") 0).HostPort}}}}',
        container_name,
    )
    if rc != 0:
        raise RuntimeError(
            f"Failed to inspect container {container_name}: {stderr}"
        )
    return int(stdout.strip())


async def _wait_cua_ready(
    cua_url: str,
    timeout: float = _CUA_READY_TIMEOUT,
    poll_interval: float = _CUA_READY_POLL_INTERVAL,
) -> bool:
    """Poll cua-server until it responds healthily."""
    from .gcloud import wait_cua_ready
    return await wait_cua_ready(
        cua_url, os_type="linux",
        timeout=timeout, poll_interval=poll_interval,
    )


def _sa_key_project_id(host_key_path: str) -> str:
    """project_id field of a GCS service-account JSON key (``""`` on error).

    Used as the billing/user project for requester-pays buckets: gsutil needs
    an explicit ``-u <project>`` (no boto-config equivalent works), and the
    injected SA key already names the project it can bill. Surfaced to
    data-staging via ``SandboxHandle.metadata["gcs_user_project"]``.
    """
    import json as _json
    from pathlib import Path as _P
    try:
        return _json.loads(_P(host_key_path).read_text()).get("project_id", "") or ""
    except Exception:
        return ""


class DockerProvider(Provider):
    """Provider backed by ``docker run`` / ``docker rm``."""

    def __init__(self, config: DockerProviderConfig | dict[str, Any]):
        if isinstance(config, dict):
            config = _build_provider_config(config)
        self._cfg = config

    @property
    def config(self) -> DockerProviderConfig:
        return self._cfg

    async def acquire(self, spec: SandboxSpec) -> SandboxHandle:
        name = _generate_container_name(
            _CONTAINER_PREFIX,
            snapshot=spec.snapshot,
            task_id=spec.task_id,
            harness=spec.harness,
            model_tag=spec.model_tag,
        )

        # Everything image-specific comes from the Image entry named by `image`:
        # the container ref to boot, the in-container cua-server port (8000 on
        # ale-kasm, 5000 on the ubuntu22 export), and the sandbox paths.
        from ..images import get as get_image
        image = get_image(self._cfg.image)
        container_ref = self._cfg.image_ref or image.docker_image or _DEFAULT_CONTAINER_REF
        cua_internal_port = image.cua_server_port

        run_args = [
            "run", "-d",
            "--name", name,
            "-p", f"0:{cua_internal_port}",
            "-p", f"0:{_VNC_INTERNAL_PORT}",
            f"--shm-size={self._cfg.shm_size}",
            # The kasm desktop entrypoint boots the X/VNC display *and* the
            # cua-server via custom_startup.sh. Some images ship a no-op CMD
            # ("sleep infinity") that never starts it, so we always invoke the
            # kasm startup script explicitly.
            "--entrypoint", _KASM_ENTRYPOINT,
        ]
        if self._cfg.privileged:
            # Needed for tasks whose eval runs nested Docker; under rootless
            # docker the privileged container's root is still user-ns-bounded.
            run_args.append("--privileged")
        if self._cfg.cpus > 0:
            run_args.extend(["--cpus", str(self._cfg.cpus)])
        if self._cfg.memory:
            run_args.extend(["--memory", self._cfg.memory])
        run_args.append(container_ref)
        run_args.append("--wait")

        logger.info("Creating Docker container %s from %s", name, container_ref)
        rc, stdout, stderr = await _run_docker(*run_args)
        if rc != 0:
            raise RuntimeError(
                f"docker run failed for {name}: {stderr}"
            )
        logger.info("Container %s started (id=%s)", name, stdout[:12])

        cua_port = await _get_host_port(name, cua_internal_port)
        vnc_port = await _get_host_port(name, _VNC_INTERNAL_PORT)
        cua_url = f"http://localhost:{cua_port}"

        logger.info(
            "Container %s ports: cua=%d vnc=%d",
            name, cua_port, vnc_port,
        )

        ready = await _wait_cua_ready(cua_url)
        if not ready:
            logger.error("CUA server not ready in %s, cleaning up", name)
            await _run_docker("rm", "-f", name)
            raise RuntimeError(
                f"CUA server at {cua_url} (container {name}) did not become ready"
            )

        gcs_user_project = ""
        if self._cfg.gcs_sa_key:
            await self._inject_gcs_credentials(name, self._cfg.gcs_sa_key)
            # gs://ale-data-public is requester-pays: gsutil needs an explicit
            # -u <billing-project>. Derive it from the SA key's own project_id
            # — same source as the injected identity, so the project we bill
            # always matches the SA we authenticate as. Surfaced so data-staging
            # adds the flag (see gsbucket).
            gcs_user_project = _sa_key_project_id(self._cfg.gcs_sa_key)

        return SandboxHandle(
            id=name,
            endpoint=cua_url,
            os=image.os,
            **image.sandbox_paths(),
            metadata={
                "container_name": name,
                "container_id": stdout[:12],
                "cua_port": cua_port,
                "vnc_port": vnc_port,
                "image": self._cfg.image,
                "snapshot": spec.snapshot,
                "gcs_user_project": gcs_user_project,
            },
        )

    async def release(
        self, sandbox: SandboxHandle, *, mode: ReleaseMode = "delete",
    ) -> None:
        name = sandbox.id
        if mode == "delete":
            logger.info("Removing container %s", name)
            rc, _, stderr = await _run_docker("rm", "-f", name)
            if rc != 0:
                logger.error("Failed to remove container %s: %s", name, stderr)
        elif mode == "stop":
            logger.info("Stopping container %s", name)
            rc, _, stderr = await _run_docker("stop", name)
            if rc != 0:
                logger.error("Failed to stop container %s: %s", name, stderr)
        elif mode == "keep":
            logger.info("Container %s kept alive (mode=keep)", name)
        else:
            raise ValueError(f"unknown release mode: {mode!r}")

    @staticmethod
    async def _inject_gcs_credentials(container_name: str, host_key_path: str) -> None:
        """Copy SA key into container and write a boto config for gsutil."""
        from pathlib import Path as _P

        key_path = _P(host_key_path)
        if not key_path.exists():
            raise FileNotFoundError(f"gcs_sa_key not found: {host_key_path}")

        project_id = _sa_key_project_id(host_key_path)

        rc, _, stderr = await _run_docker(
            "exec", "-u", "root", container_name,
            "mkdir", "-p", str(_P(_GCS_KEY_CONTAINER_PATH).parent),
        )
        rc, _, stderr = await _run_docker(
            "cp", host_key_path, f"{container_name}:{_GCS_KEY_CONTAINER_PATH}",
        )
        if rc != 0:
            raise RuntimeError(f"Failed to copy SA key into {container_name}: {stderr}")
        await _run_docker(
            "exec", "-u", "root", container_name,
            "chmod", "644", _GCS_KEY_CONTAINER_PATH,
        )

        boto_content = (
            "[Credentials]\n"
            f"gs_service_key_file = {_GCS_KEY_CONTAINER_PATH}\n\n"
            "[Boto]\n"
            "https_validate_certificates = True\n\n"
            "[GSUtil]\n"
            f"default_project_id = {project_id}\n"
        )
        # Write a system-wide /etc/boto.cfg (as root) rather than a per-user
        # ~/.boto: the container user is image-specific (kasm-user, user, ...),
        # and hardcoding one user's home silently broke gsutil auth on every
        # other image family. /etc/boto.cfg is read regardless of $HOME.
        rc, _, stderr = await _run_docker(
            "exec", "-u", "root", container_name,
            "bash", "-c",
            f"cat > {_BOTO_CONFIG_CONTAINER_PATH} << 'BOTOEOF'\n{boto_content}BOTOEOF",
        )
        if rc != 0:
            raise RuntimeError(
                f"Failed to write {_BOTO_CONFIG_CONTAINER_PATH} into "
                f"{container_name}: {stderr}"
            )
        logger.info("Injected GCS credentials into %s", container_name)

    def open_session(self, sandbox: SandboxHandle) -> Any:
        from cua_bench.computers.remote import RemoteDesktopSession
        from .gcloud import _init_computer_skip_wait

        session = RemoteDesktopSession(
            api_url=sandbox.endpoint,
            os_type=sandbox.os,
        )
        _init_computer_skip_wait(session)
        return session
