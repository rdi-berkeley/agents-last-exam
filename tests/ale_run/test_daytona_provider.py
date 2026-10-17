from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import aiohttp
import pytest
from aiohttp import web

from ale_run.base_interface import SandboxSpec, TaskDataSpec
from ale_run.environments.providers.daytona import DaytonaProvider, _PreviewProxy
from ale_run.environments.task_data import local_host
from ale_run.orchestration.config_loader import _build_per_snapshot_env
from ale_run.orchestration.factory import build_provider


@pytest.mark.asyncio
async def test_private_preview_proxy_forwards_http_and_websocket():
    token = "preview-secret"
    seen = []

    async def handler(request):
        seen.append(request.headers.get("X-Daytona-Preview-Token"))
        if request.path == "/ws":
            ws = web.WebSocketResponse()
            await ws.prepare(request)
            msg = await ws.receive()
            await ws.send_str(msg.data.upper())
            await ws.close()
            return ws
        return web.Response(body=await request.read(), status=201)

    app = web.Application()
    app.router.add_route("*", "/{path:.*}", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    proxy = _PreviewProxy(f"http://127.0.0.1:{port}", token)
    endpoint = await proxy.start()
    try:
        async with aiohttp.ClientSession() as client:
            async with client.post(endpoint + "/cmd", data=b"command") as response:
                assert response.status == 201
                assert await response.read() == b"command"
            async with client.ws_connect(endpoint + "/ws") as ws:
                await ws.send_str("hello")
                assert (await ws.receive()).data == "HELLO"
        assert seen == [token, token]
        assert token not in endpoint
    finally:
        await proxy.close()
        await runner.cleanup()


@pytest.mark.asyncio
async def test_provider_sizes_task_and_deletes_on_release(monkeypatch):
    daytona = pytest.importorskip("daytona")

    from ale_run.environments.providers import gcloud

    cloud = SimpleNamespace(
        id="sandbox-id",
        process=SimpleNamespace(exec=AsyncMock(return_value=SimpleNamespace(exit_code=0))),
        fs=SimpleNamespace(upload_file_stream=AsyncMock()),
        get_preview_link=AsyncMock(return_value=SimpleNamespace(url="https://preview", token="secret")),
        delete=AsyncMock(), stop=AsyncMock(),
    )
    client = SimpleNamespace(create=AsyncMock(return_value=cloud), close=AsyncMock())
    monkeypatch.setattr(daytona, "AsyncDaytona", lambda config: client)
    monkeypatch.setattr(gcloud, "wait_cua_ready", AsyncMock(return_value=True))
    proxy = SimpleNamespace(start=AsyncMock(return_value="http://127.0.0.1:1234"), close=AsyncMock())
    monkeypatch.setattr("ale_run.environments.providers.daytona._PreviewProxy", lambda *args: proxy)
    provider = DaytonaProvider({"disk_gb": 200})
    handle = await provider.acquire(SandboxSpec(snapshot="cpu-free-ubuntu", vcpus=4, memory_gb=16))
    params = client.create.await_args.args[0]
    assert (params.resources.cpu, params.resources.memory, params.resources.disk) == (4, 16, 200)
    assert params.public is False
    assert params.image.endswith("sha256:0b223c5042bedfa78f37f17881f6b410aa3d745310b667e5dff62637363c5085")
    assert "secret" not in repr(handle)
    from dataclasses import asdict

    assert "upload_transport" not in asdict(handle)
    await handle.upload_local_file("/local", "/remote")
    cloud.fs.upload_file_stream.assert_awaited_once_with("/local", "/remote")
    await provider.release(handle)
    cloud.delete.assert_awaited_once()
    proxy.close.assert_awaited_once()
    client.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_acquire_failure_cleans_remote_sandbox(monkeypatch):
    daytona = pytest.importorskip("daytona")

    cloud = SimpleNamespace(
        id="sandbox-id", process=SimpleNamespace(exec=AsyncMock(side_effect=RuntimeError("failed"))),
        delete=AsyncMock(),
    )
    client = SimpleNamespace(create=AsyncMock(return_value=cloud), close=AsyncMock())
    monkeypatch.setattr(daytona, "AsyncDaytona", lambda config: client)
    with pytest.raises(RuntimeError, match="failed"):
        await DaytonaProvider({}).acquire(SandboxSpec(snapshot="cpu-free-ubuntu", vcpus=4, memory_gb=16))
    cloud.delete.assert_awaited_once()
    client.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_snapshot_resource_mismatch_fails_before_sandbox_creation(monkeypatch):
    daytona = pytest.importorskip("daytona")

    source = SimpleNamespace(cpu=4, mem=8, disk=200, image_name="unused")
    client = SimpleNamespace(
        snapshot=SimpleNamespace(get=AsyncMock(return_value=source)),
        create=AsyncMock(), get=AsyncMock(side_effect=LookupError()), close=AsyncMock(),
    )
    monkeypatch.setattr(daytona, "AsyncDaytona", lambda config: client)
    provider = DaytonaProvider({"snapshot": "test"})
    with pytest.raises(ValueError, match="resources"):
        await provider.acquire(SandboxSpec(snapshot="cpu-free-ubuntu", vcpus=4, memory_gb=16))
    client.create.assert_not_awaited()
    client.close.assert_awaited_once()


def test_environment_routes_daytona():
    raw = {"snapshots": {"cpu-free-ubuntu": {"provider": "daytona", "image": "ale-ubuntu22-docker-v1-1", "daytona": {"disk_gb": 200}}}}
    env = _build_per_snapshot_env(raw, "test.yaml")
    assert env.kind_for("cpu-free-ubuntu") == "daytona"
    assert isinstance(build_provider(env.provider_specs["daytona"]), DaytonaProvider)


@pytest.mark.asyncio
async def test_daytona_local_data_stages_reference_only_after_solver(tmp_path, monkeypatch):
    base = tmp_path / "demo" / "hello" / "base"
    (base / "input").mkdir(parents=True)
    (base / "input" / "value.txt").write_text("input")
    (base / "reference.7z").write_bytes(b"encrypted")
    uploads = []
    async def upload(local, remote):
        uploads.append((Path(local).name, remote))
    sandbox = SimpleNamespace(
        metadata={"provider": "daytona"}, is_linux=True,
        task_data_root="/data", work_dir_base="/work", python="python",
        mkdir=AsyncMock(), rm=AsyncMock(), upload_local_file=upload,
        run_command=AsyncMock(return_value=subprocess.CompletedProcess([], 0, "", "")),
    )
    task = TaskDataSpec(domain_name="demo", task_name="hello", variant_name="base")
    monkeypatch.setattr("ale_run.environments.task_data.baked_in_sandbox.stage_reference", AsyncMock(return_value={"staged": ["reference"]}))
    source = "local:" + str(tmp_path)
    await local_host.stage_input(sandbox, task, source=source)
    assert all(name != "reference.7z" for name, _ in uploads)
    await local_host.stage_reference(sandbox, task, source=source)
    assert uploads[-1][0] == "reference.7z"
