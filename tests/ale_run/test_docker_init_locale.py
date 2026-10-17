import asyncio
from unittest.mock import AsyncMock

import pytest

from ale_run.base_interface import SandboxSpec
from ale_run.environments.providers import docker


@pytest.mark.parametrize("ready", [True, False])
def test_init_locale_local_bindings_and_cleanup(monkeypatch, ready):
    calls = []

    async def run_docker(*args):
        calls.append(args)
        return 0, "test-container-id", ""

    monkeypatch.setattr(docker, "_run_docker", run_docker)
    monkeypatch.setattr(
        docker, "_get_image_provenance", AsyncMock(return_value={"provider": "docker"})
    )
    monkeypatch.setattr(docker, "_get_host_port", AsyncMock(return_value=12345))
    monkeypatch.setattr(docker, "_wait_cua_ready", AsyncMock(return_value=ready))
    monkeypatch.setattr(docker, "_get_resource_limits", AsyncMock(return_value={}))
    provider = docker.DockerProvider({"image": "ale-ubuntu22-docker", "cpus": 2, "memory": "2g"})
    spec = SandboxSpec(snapshot="test", task_id="review-only")
    if ready:
        handle = asyncio.run(provider.acquire(spec))
        asyncio.run(provider.release(handle))
    else:
        with pytest.raises(RuntimeError, match="did not become ready"):
            asyncio.run(provider.acquire(spec))
    args = calls[0]
    assert "--init" in args
    assert "LANG=C.UTF-8" in args and "LC_ALL=C.UTF-8" in args
    ports = [args[index + 1] for index, value in enumerate(args) if value == "-p"]
    assert len(ports) == 2 and all(port.startswith("127.0.0.1:0:") for port in ports)
    assert args[args.index("--cpus") + 1] == "2.0"
    assert args[args.index("--memory") + 1] == "2g"
    assert calls[-1][:2] == ("rm", "-f")
