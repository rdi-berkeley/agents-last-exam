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
from dataclasses import dataclass, field as dataclass_field
from typing import Any

from ...base_interface import SandboxSpec, Provider, ReleaseMode, SandboxHandle

logger = logging.getLogger(__name__)

_CUA_INTERNAL_PORT = 8000
_VNC_INTERNAL_PORT = 6901
_CUA_READY_TIMEOUT = 120
_CUA_READY_POLL_INTERVAL = 3
_CUA_READY_STABLE_SUCCESSES = 2

_DEFAULT_IMAGE = "agentslastexam/ale-kasm:latest"
_KASM_ENTRYPOINT = "/dockerstartup/vnc_startup.sh"


_GCS_KEY_CONTAINER_PATH = "/etc/agenthle/gcs-reader.json"
# System-wide boto config so gsutil authenticates regardless of which user the
# container runs as (kasm-user on ale-kasm, user on ale-ubuntu22-docker, ...).
# boto reads /etc/boto.cfg before any per-user ~/.boto, so this is user-agnostic.
_BOTO_CONFIG_CONTAINER_PATH = "/etc/boto.cfg"


@dataclass(frozen=True)
class DockerProviderConfig:
    """Docker provider config (yaml ``provider.config``).

    image               Docker image to run (default: agentslastexam/ale-kasm:latest)
    container_prefix    Name prefix for containers
    extra_env           Extra env vars to inject into containers
    shm_size            Shared memory size (default: 512m)
    cpus                CPU limit per container (0 = unlimited)
    memory              Memory limit per container (e.g. "8g", "" = unlimited)
    gcs_sa_key          Host path to a GCS service-account JSON key.
                        When set, copied into each container and gsutil
                        boto is configured so task_data_source=gs://...
                        works inside the sandbox. The key's ``project_id``
                        doubles as the billing/user project for requester-pays
                        buckets (gsutil ``-u``) — same source as the identity,
                        so the two can't drift (see ``_sa_key_project_id``).
    """

    image: str = _DEFAULT_IMAGE
    image_family: str = "ale-kasm"
    container_prefix: str = "ale"
    extra_env: dict[str, str] = dataclass_field(default_factory=dict)
    shm_size: str = "512m"
    cpus: float = 0
    memory: str = ""
    gcs_sa_key: str = ""


def _build_provider_config(raw: dict[str, Any]) -> DockerProviderConfig:
    gcs_sa = raw.get("gcs_sa_key") or ""
    if gcs_sa:
        from pathlib import Path as _P
        gcs_sa = str(_P(gcs_sa).expanduser().resolve())
    return DockerProviderConfig(
        image=str(raw.get("image") or _DEFAULT_IMAGE),
        image_family=str(raw.get("image_family") or "ale-kasm"),
        container_prefix=str(raw.get("container_prefix") or "ale"),
        extra_env=dict(raw.get("extra_env") or {}),
        shm_size=str(raw.get("shm_size") or "512m"),
        cpus=float(raw.get("cpus") or 0),
        memory=str(raw.get("memory") or ""),
        gcs_sa_key=gcs_sa,
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
            self._cfg.container_prefix,
            snapshot=spec.snapshot,
            task_id=spec.task_id,
            harness=spec.harness,
            model_tag=spec.model_tag,
        )

        # The cua-server's in-container port is image-specific (8000 on
        # ale-kasm, 5000 on GCE families). Read it from the Image registry
        # rather than hard-coding, so the published port + handle URL stay
        # correct for any image family.
        from ..images import get as get_image
        image = get_image(self._cfg.image_family)
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
        if self._cfg.cpus > 0:
            run_args.extend(["--cpus", str(self._cfg.cpus)])
        if self._cfg.memory:
            run_args.extend(["--memory", self._cfg.memory])
        for k, v in self._cfg.extra_env.items():
            run_args.extend(["-e", f"{k}={v}"])
        run_args.append(self._cfg.image)
        run_args.append("--wait")

        logger.info("Creating Docker container %s from %s", name, self._cfg.image)
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
