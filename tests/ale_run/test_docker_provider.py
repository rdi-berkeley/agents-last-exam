import json
from unittest.mock import AsyncMock

import pytest

from ale_run.base_interface import SandboxSpec
from ale_run.environments import images
from ale_run.environments.providers import docker


@pytest.mark.asyncio
@pytest.mark.parametrize("image_name", ["ale-kasm", "ale-ubuntu22-docker-v1-1"])
@pytest.mark.parametrize("ready", [True, False])
async def test_acquire_uses_one_loopback_address_family(monkeypatch, image_name, ready):
    image = images.get(image_name)
    commands = AsyncMock(
        side_effect=[
            (0, "container-id", ""),
            (0, "32809", ""),
            (0, "32808", ""),
            (0, json.dumps({"NanoCpus": 4_000_000_000, "Memory": 15 * 1024 ** 3}), ""),
        ]
    )
    readiness = AsyncMock(return_value=ready)
    monkeypatch.setattr(docker, "_run_docker", commands)
    monkeypatch.setattr(docker, "_wait_cua_ready", readiness)
    provider = docker.DockerProvider({"image": image_name})
    spec = SandboxSpec(snapshot="cpu-free-ubuntu", machine_type="c4-standard-4")

    if ready:
        sandbox = await provider.acquire(spec)
        assert sandbox.endpoint == "http://127.0.0.1:32809"
        assert sandbox.metadata["cua_port"] == 32809
        assert sandbox.metadata["vnc_port"] == 32808
        assert sandbox.metadata["resource_limits"] == {
            "cpu_quota": 4.0,
            "memory_limit_bytes": 15 * 1024 ** 3,
        }
        assert commands.await_count == 4
    else:
        with pytest.raises(RuntimeError, match=r"127\.0\.0\.1:32809.*did not become ready"):
            await provider.acquire(spec)
        assert commands.await_args_list[-1].args[:2] == ("rm", "-f")

    readiness.assert_awaited_once_with("http://127.0.0.1:32809")
    arguments = commands.await_args_list[0].args
    published = [arguments[index + 1] for index, value in enumerate(arguments) if value == "-p"]
    assert published == [f"127.0.0.1:0:{image.cua_server_port}", "127.0.0.1:0:6901"]
    assert arguments[arguments.index("--cpus") + 1] == "4.0"
    assert arguments[arguments.index("--memory") + 1] == "15g"
