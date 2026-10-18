"""Daytona provider for ALE's Linux container images."""
from __future__ import annotations

import asyncio
import json
import logging
import math
import shlex
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aiohttp import ClientSession, ClientTimeout, WSMsgType, web

from ...base_interface import Provider, ReleaseMode, SandboxHandle, SandboxSpec

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DaytonaProviderConfig:
    image: str = "ale-ubuntu22-docker-v1-1"
    image_ref: str = ""
    snapshot: str = ""
    target: str = "us"
    disk_gb: int = 200
    cpu: int = 0
    memory_gb: int = 0
    resolution: tuple[int, int] = (1024, 768)
    ready_timeout_s: int = 300
    create_timeout_s: int = 1800
    auto_stop_minutes: int = 60
    ttl_minutes: int = 0
    gcs_sa_key: str = ""


def _build_provider_config(raw: dict[str, Any]) -> DaytonaProviderConfig:
    from .gcloud import _parse_resolution

    image = str(raw.get("image") or "ale-ubuntu22-docker-v1-1")
    cfg = DaytonaProviderConfig(
        image=image,
        image_ref=str(raw.get("image_ref") or ""),
        snapshot=str(raw.get("snapshot") or ""),
        target=str(raw.get("target") or "us"),
        disk_gb=int(raw.get("disk_gb", 200)),
        cpu=int(raw.get("cpu", 0)),
        memory_gb=int(raw.get("memory_gb", 0)),
        resolution=_parse_resolution(raw.get("resolution"), image) or (1024, 768),
        ready_timeout_s=int(raw.get("ready_timeout_s", 300)),
        create_timeout_s=int(raw.get("create_timeout_s", 1800)),
        auto_stop_minutes=int(raw.get("auto_stop_minutes", 60)),
        ttl_minutes=int(raw.get("ttl_minutes", 0)),
        gcs_sa_key=str(raw.get("gcs_sa_key") or ""),
    )
    if min(cfg.disk_gb, cfg.ready_timeout_s, cfg.create_timeout_s) < 1:
        raise ValueError("Daytona disk and timeout settings must be positive")
    if min(cfg.cpu, cfg.memory_gb, cfg.auto_stop_minutes, cfg.ttl_minutes) < 0:
        raise ValueError("Daytona resource and lifecycle settings cannot be negative")
    if min(cfg.resolution) < 1:
        raise ValueError("Daytona display dimensions must be positive")
    return cfg


class _PreviewProxy:
    """A loopback-only, authenticated HTTP/WebSocket proxy to a private preview."""

    def __init__(self, remote_url: str, token: str):
        self._remote_url = remote_url.rstrip("/")
        self._token = token
        self._runner: web.AppRunner | None = None
        self._client: ClientSession | None = None
        self.endpoint = ""

    async def start(self) -> str:
        self._client = ClientSession(timeout=ClientTimeout(total=None), auto_decompress=False)
        app = web.Application(client_max_size=0)
        app.router.add_route("*", "/{path:.*}", self._forward)
        self._runner = web.AppRunner(app, access_log=None)
        await self._runner.setup()
        site = web.TCPSite(self._runner, "127.0.0.1", 0)
        await site.start()
        assert site._server is not None
        port = site._server.sockets[0].getsockname()[1]
        self.endpoint = f"http://127.0.0.1:{port}"
        return self.endpoint

    async def _forward(self, request: web.Request) -> web.StreamResponse:
        assert self._client is not None
        url = self._remote_url + request.rel_url.raw_path_qs
        headers = {
            key: value for key, value in request.headers.items()
            if key.lower() not in {"host", "connection", "content-length", "transfer-encoding",
                                   "x-daytona-preview-token"}
        }
        headers["X-Daytona-Preview-Token"] = self._token
        if request.headers.get("Upgrade", "").lower() == "websocket":
            upstream = await self._client.ws_connect(url, headers={
                k: v for k, v in headers.items() if not k.lower().startswith("sec-websocket-")
                and k.lower() != "upgrade"
            }, max_msg_size=0, heartbeat=30)
            downstream = web.WebSocketResponse(max_msg_size=0, heartbeat=30)
            await downstream.prepare(request)

            async def relay(source: Any, target: Any) -> None:
                async for msg in source:
                    if msg.type == WSMsgType.TEXT:
                        await target.send_str(msg.data)
                    elif msg.type == WSMsgType.BINARY:
                        await target.send_bytes(msg.data)
                    elif msg.type in (WSMsgType.CLOSE, WSMsgType.CLOSED, WSMsgType.ERROR):
                        break
                await target.close()

            try:
                await asyncio.gather(relay(upstream, downstream), relay(downstream, upstream))
            finally:
                await upstream.close()
                await downstream.close()
            return downstream
        async with self._client.request(
            request.method, url, headers=headers, data=await request.read(), allow_redirects=False,
        ) as upstream:
            response_headers = {
                key: value for key, value in upstream.headers.items()
                if key.lower() not in {"connection", "content-length", "transfer-encoding"}
            }
            response = web.StreamResponse(status=upstream.status, headers=response_headers)
            await response.prepare(request)
            try:
                async for chunk in upstream.content.iter_any():
                    await response.write(chunk)
                await response.write_eof()
            except (ConnectionResetError, BrokenPipeError):
                pass
            return response

    async def close(self) -> None:
        if self._runner is not None:
            await self._runner.cleanup()
        if self._client is not None:
            await self._client.close()


class DaytonaProvider(Provider):
    """Launch private Daytona containers from the registered ALE image."""

    def __init__(self, config: DaytonaProviderConfig | dict[str, Any]):
        self._cfg = _build_provider_config(config) if isinstance(config, dict) else config
        self._active: dict[str, tuple[Any, Any, _PreviewProxy]] = {}

    @property
    def config(self) -> DaytonaProviderConfig:
        return self._cfg

    async def acquire(self, spec: SandboxSpec) -> SandboxHandle:
        try:
            from daytona import (
                AsyncDaytona,
                CreateSandboxFromImageParams,
                CreateSandboxFromSnapshotParams,
                DaytonaConfig,
                Resources,
            )
        except ImportError as exc:
            raise RuntimeError("Daytona provider requires `uv sync --extra daytona`") from exc
        from ..images import get as get_image
        from .gcloud import _DEFAULT_CPU_MACHINE, _parse_gce_machine_type, wait_cua_ready

        image = get_image(self._cfg.image)
        if spec.os != "linux" or image.os != "linux" or spec.gpu:
            raise ValueError("Daytona container provider supports Linux CPU tasks only")
        image_ref = self._cfg.image_ref or image.docker_image
        if not image_ref:
            raise ValueError(f"Image {image.name!r} has no Docker realization")
        shape = _parse_gce_machine_type(spec.machine_type or _DEFAULT_CPU_MACHINE)
        cpu = self._cfg.cpu or spec.vcpus or (shape.vcpus if shape else 0)
        memory = self._cfg.memory_gb or spec.memory_gb or (shape.memory_gb if shape else 0)
        if not cpu or not memory:
            raise ValueError("Could not determine Daytona task CPU/memory")
        client = AsyncDaytona(DaytonaConfig(target=self._cfg.target))
        name = f"ale-v11-{uuid.uuid4().hex}"
        cloud = None
        proxy = None
        try:
            common = {
                "name": name,
                "os_user": "user",
                "env_vars": {"LANG": "C.UTF-8", "LC_ALL": "C.UTF-8",
                          "ALE_SCREEN_RESOLUTION": f"{self._cfg.resolution[0]}x{self._cfg.resolution[1]}",
                          "ALE_ENABLE_DIND": "0"},
                "labels": {"project": "ale", "provider": "ale-v11"},
                "public": False,
                "ephemeral": True,
                "auto_stop_interval": self._cfg.auto_stop_minutes,
                "auto_delete_interval": 0,
                "ttl_minutes": self._cfg.ttl_minutes,
                "target": self._cfg.target,
            }
            if self._cfg.snapshot:
                source = await client.snapshot.get(self._cfg.snapshot)
                if (source.cpu, source.mem, source.disk) != (
                    math.ceil(cpu), math.ceil(memory), self._cfg.disk_gb,
                ):
                    raise ValueError("Daytona snapshot resources do not match the task shape")
                if source.image_name != image_ref:
                    raise ValueError("Daytona snapshot image does not match the pinned image")
                params = CreateSandboxFromSnapshotParams(snapshot=self._cfg.snapshot, **common)
            else:
                params = CreateSandboxFromImageParams(
                    image=image_ref,
                    resources=Resources(cpu=math.ceil(cpu), memory=math.ceil(memory),
                                        disk=self._cfg.disk_gb),
                    **common,
                )
            cloud = await client.create(params, timeout=self._cfg.create_timeout_s)
            result = await cloud.process.exec(
                f"nohup {shlex.quote(image.docker_entrypoint)} >/tmp/ale-entrypoint.log 2>&1 </dev/null &",
                timeout=30,
            )
            if result.exit_code:
                raise RuntimeError(f"Daytona image entrypoint could not start (rc={result.exit_code})")
            preview = await cloud.get_preview_link(image.cua_server_port)
            proxy = _PreviewProxy(preview.url, preview.token)
            endpoint = await proxy.start()
            if not await wait_cua_ready(endpoint, "linux", timeout=self._cfg.ready_timeout_s,
                                        poll_interval=3):
                raise RuntimeError("Daytona CUA server did not become ready")
            metadata: dict[str, Any] = {
                "provider": "daytona", "image": image.name, "snapshot": spec.snapshot,
                "resource_limits": {"cpu_quota": cpu, "memory_limit_bytes": memory * 1024**3,
                                    "disk_gb": self._cfg.disk_gb},
                "image_provenance": {"provider": "daytona", "image": image.name,
                                     "image_ref": image_ref,
                                     "daytona_snapshot": self._cfg.snapshot or None,
                                     "image_revision": image_ref.rsplit("@", 1)[-1]
                                     if "@" in image_ref else None},
            }
            sandbox = SandboxHandle(id=cloud.id, endpoint=endpoint, os=image.os,
                                    **image.sandbox_paths(), metadata=metadata)
            sandbox.upload_transport = cloud.fs.upload_file_stream
            if self._cfg.gcs_sa_key:
                path = Path(self._cfg.gcs_sa_key).expanduser()
                key = path.read_bytes()
                key_path = f"{image.work_dir_base}/daytona-gcs-reader.json"
                await sandbox.mkdir(image.work_dir_base)
                await sandbox.write_file(key_path, key)
                chmod = await sandbox.run_command(f"chmod 600 {shlex.quote(key_path)}")
                if chmod.returncode:
                    raise RuntimeError("Cannot secure Daytona GCS credentials")
                metadata["gcs_key_path"] = key_path
                metadata["gcs_user_project"] = json.loads(key).get("project_id", "")
            self._active[cloud.id] = (client, cloud, proxy)
            return sandbox
        except BaseException:
            if proxy:
                await proxy.close()
            if cloud is None:
                try:
                    cloud = await client.get(name, request_timeout=20)
                except Exception as exc:  # noqa: BLE001
                    logger.debug("Daytona sandbox %s was not available for cleanup: %s", name, exc)
            if cloud is not None:
                try:
                    await cloud.delete(timeout=60)
                except Exception:
                    logger.exception("Failed to delete Daytona sandbox %s", name)
            await client.close()
            raise

    async def release(self, sandbox: SandboxHandle, *, mode: ReleaseMode = "delete") -> None:
        if mode not in ("delete", "stop", "keep"):
            raise ValueError(f"unknown release mode: {mode!r}")
        active = self._active.pop(sandbox.id, None)
        if active is None:
            raise RuntimeError(f"Daytona sandbox {sandbox.id} is not owned by this provider")
        client, cloud, proxy = active
        try:
            if mode == "delete":
                await cloud.delete(timeout=60)
            elif mode == "stop":
                await cloud.stop(timeout=60)
        finally:
            await proxy.close()
            await client.close()

    def open_session(self, sandbox: SandboxHandle) -> Any:
        from cua_bench.computers.remote import RemoteDesktopSession

        from .gcloud import _init_computer_skip_wait

        session = RemoteDesktopSession(api_url=sandbox.endpoint, os_type=sandbox.os)
        _init_computer_skip_wait(session)
        return session
