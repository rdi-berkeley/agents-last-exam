import json
from unittest.mock import AsyncMock

import pytest

from ale_run.base_interface import SandboxSpec
from ale_run.environments import images
from ale_run.environments.providers import docker
from ale_run.orchestration.config_loader import _build_per_snapshot_env


_IMAGE_ID = "sha256:" + "1" * 64
_REPO_DIGEST = "sha256:" + "2" * 64


@pytest.mark.asyncio
@pytest.mark.parametrize("image_name", ["ale-kasm", "ale-ubuntu22-docker-v1-1"])
@pytest.mark.parametrize("ready", [True, False])
async def test_acquire_uses_one_loopback_address_family(monkeypatch, image_name, ready):
    image = images.get(image_name)
    selected_digest = image.docker_image.rsplit("@", 1)[-1] if "@" in image.docker_image else None
    commands = AsyncMock(
        side_effect=[
            (0, "container-id", ""),
            (0, _IMAGE_ID, ""),
            (0, json.dumps([f"example/image@{selected_digest}"] if selected_digest else []), ""),
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
        assert sandbox.metadata["image_provenance"] == {
            "provider": "docker",
            "image": image_name,
            "image_revision": selected_digest or _IMAGE_ID,
            "image_ref": image.docker_image,
            "image_id": _IMAGE_ID,
        }
        assert commands.await_count == 6
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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("container_ref", "expected_revision", "recorded_revision"),
    [
        ("example/image:latest", None, _IMAGE_ID),
        ("example/image:latest", _IMAGE_ID, _IMAGE_ID),
        ("example/image:latest", _REPO_DIGEST, _REPO_DIGEST),
        (f"example/image:v1@{_REPO_DIGEST}", None, _REPO_DIGEST),
        (f"example/image:v1@{_REPO_DIGEST}", _IMAGE_ID, _REPO_DIGEST),
    ],
)
async def test_acquire_verifies_revision_and_preserves_selected_digest(
    monkeypatch, container_ref, expected_revision, recorded_revision,
):
    commands = AsyncMock(side_effect=[
        (0, "container-id", ""),
        (0, _IMAGE_ID, ""),
        (0, json.dumps([f"example/other@{'sha256:' + '3' * 64}", f"example/image@{_REPO_DIGEST}"]), ""),
        (0, "32809", ""),
        (0, "32808", ""),
        (0, json.dumps({"NanoCpus": 4_000_000_000, "Memory": 1024 ** 3}), ""),
    ])
    monkeypatch.setattr(docker, "_run_docker", commands)
    monkeypatch.setattr(docker, "_wait_cua_ready", AsyncMock(return_value=True))
    provider = docker.DockerProvider({
        "image": "ale-kasm", "image_ref": container_ref, "image_revision": expected_revision,
    })

    sandbox = await provider.acquire(SandboxSpec(snapshot="cpu-free"))

    assert json.loads(json.dumps(sandbox.metadata["image_provenance"])) == {
        "provider": "docker", "image": "ale-kasm", "image_revision": recorded_revision,
        "image_ref": container_ref, "image_id": _IMAGE_ID,
    }
    assert commands.await_args_list[0].args[-2] == container_ref
    assert commands.await_args_list[2].args == (
        "image", "inspect", "--format", "{{json .RepoDigests}}", _IMAGE_ID,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["expected_pin", "selected_digest", "inspect", "image_inspect"])
async def test_acquire_cleans_up_when_image_identity_cannot_be_verified(monkeypatch, failure):
    config = {"image": "ale-kasm", "image_ref": "example/image:latest"}
    if failure == "expected_pin":
        config["image_revision"] = "sha256:" + "9" * 64
    if failure == "selected_digest":
        config["image_ref"] = f"example/image@{_REPO_DIGEST}"
    responses = [(0, "container-id", "")]
    if failure == "inspect":
        responses.append((1, "", "inspect unavailable"))
    else:
        responses.append((0, _IMAGE_ID, ""))
        responses.append(
            (1, "", "image inspect unavailable") if failure == "image_inspect" else (0, "[]", "")
        )
    responses.append((0, "removed", ""))
    commands = AsyncMock(side_effect=responses)
    readiness = AsyncMock(return_value=True)
    monkeypatch.setattr(docker, "_run_docker", commands)
    monkeypatch.setattr(docker, "_wait_cua_ready", readiness)

    with pytest.raises(RuntimeError, match="mismatch|unavailable"):
        await docker.DockerProvider(config).acquire(SandboxSpec(snapshot="cpu-free"))

    assert commands.await_args_list[-1].args[:2] == ("rm", "-f")
    readiness.assert_not_awaited()


@pytest.mark.parametrize("revision", ["latest", "sha256:short", 0, f"example/image@{_REPO_DIGEST}"])
def test_config_rejects_invalid_image_revision(revision):
    with pytest.raises(ValueError, match="docker image_revision must be a sha256"):
        docker.DockerProvider({"image_revision": revision})


def test_loader_forwards_docker_image_revision():
    environment = _build_per_snapshot_env({
        "snapshots": {
            "cpu-free": {
                "provider": "docker",
                "image": "ale-kasm",
                "docker": {"image_ref": "example/image:latest", "image_revision": _REPO_DIGEST},
            },
        },
    }, "environment.yaml")

    provider = docker.DockerProvider(environment.provider_specs["docker"].config)

    assert provider.config.image_revision == _REPO_DIGEST
