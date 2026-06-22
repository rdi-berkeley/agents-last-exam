from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from ale_run.base_interface import SandboxSpec
from ale_run.environments.output_pull import _docker_container
from ale_run.environments.providers import qemu as qemu_module
from ale_run.environments.providers.qemu import QemuProvider
from ale_run.orchestration.config_loader import load_experiment


def _provider_config(tmp_path: Path, base_qcow2: Path) -> dict:
    return {
        "snapshots": {
            "cpu-free": {
                "image": "ale-win10",
                "disk_source": str(base_qcow2),
                "root": str(tmp_path / "qemu"),
                "vcpus": 4,
                "memory_gb": 8,
            }
        }
    }


def test_loader_builds_qemu_snapshot_map(tmp_path: Path) -> None:
    agent = tmp_path / "agent.yaml"
    agent.write_text("harness: dummy\nmodel: test\n", encoding="utf-8")
    environment = tmp_path / "environment.yaml"
    environment.write_text(
        """
snapshots:
  cpu-free:
    provider: qemu
    image: ale-win10
    qemu:
      disk_source: gs://ale-data-public/images/ale-win10.qcow2
task_data_source: baked_in_sandbox
output_path: local
""",
        encoding="utf-8",
    )
    experiment = tmp_path / "experiment.yaml"
    experiment.write_text(
        f"""
name: qemu-test
agent: {agent}
environment: {environment}
tasks:
  - path: demo/hello
""",
        encoding="utf-8",
    )

    spec = load_experiment(experiment)

    assert spec.environment.snapshot_kind == {"cpu-free": "qemu"}
    provider = spec.environment.provider_specs["qemu"]
    assert provider.config["snapshots"]["cpu-free"] == {
        "image": "ale-win10",
        "disk_source": "gs://ale-data-public/images/ale-win10.qcow2",
    }


def test_loader_rejects_qemu_without_disk_source(tmp_path: Path) -> None:
    agent = tmp_path / "agent.yaml"
    agent.write_text("harness: dummy\nmodel: test\n", encoding="utf-8")
    environment = tmp_path / "environment.yaml"
    environment.write_text(
        """
snapshots:
  cpu-free:
    provider: qemu
    image: ale-win10
    qemu: {}
""",
        encoding="utf-8",
    )
    experiment = tmp_path / "experiment.yaml"
    experiment.write_text(
        f"""
name: qemu-test
agent: {agent}
environment: {environment}
tasks:
  - path: demo/hello
""",
        encoding="utf-8",
    )

    with pytest.raises(KeyError, match="qemu.disk_source"):
        load_experiment(experiment)


def test_provider_rejects_invalid_runner_pull_policy(
    tmp_path: Path,
) -> None:
    base_qcow2 = tmp_path / "ale-win10.qcow2"
    base_qcow2.write_bytes(b"qcow2")
    config = _provider_config(tmp_path, base_qcow2)
    config["snapshots"]["cpu-free"]["runner_pull_policy"] = "sometimes"

    with pytest.raises(ValueError, match="runner_pull_policy"):
        QemuProvider(config)


def test_provider_rejects_hf_revisions_sharing_one_cache_path(
    tmp_path: Path,
) -> None:
    source = "hf://agents-last-exam/ale-images-qcow2/ale-win10.qcow2"
    config = {
        "snapshots": {
            "first": {
                "image": "ale-win10",
                "disk_source": source,
                "hf_revision": "revision-a",
                "root": str(tmp_path / "qemu"),
            },
            "second": {
                "image": "ale-win10",
                "disk_source": source,
                "hf_revision": "revision-b",
                "root": str(tmp_path / "qemu"),
            },
        }
    }

    with pytest.raises(ValueError, match="share cache path"):
        QemuProvider(config)


@pytest.mark.asyncio
async def test_acquire_creates_overlay_and_returns_guest_handle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_qcow2 = tmp_path / "ale-win10.qcow2"
    base_qcow2.write_bytes(b"qcow2")
    provider = QemuProvider(_provider_config(tmp_path, base_qcow2))
    provider._preflight_done = True

    docker_calls: list[tuple[str, ...]] = []

    async def fake_run_docker(
        *args: str,
        check: bool = True,
    ) -> tuple[int, str, str]:
        _ = check
        docker_calls.append(args)
        if args[:4] == ("run", "--rm", "--pull=missing", "--entrypoint"):
            storage_mount = next(
                arg
                for arg in args
                if arg.startswith("type=bind,src=") and arg.endswith(",dst=/storage")
            )
            storage_dir = Path(
                storage_mount.removeprefix("type=bind,src=").removesuffix(",dst=/storage")
            )
            (storage_dir / "data.qcow2").write_bytes(b"overlay")
        if args[:2] == ("inspect", "--format"):
            if args[2] == "{{.State.Running}} {{.State.ExitCode}}":
                return (0, "true 0", "")
            internal_port = args[2]
            return (0, "15000" if "5000/tcp" in internal_port else "18000", "")
        return (0, "container-id", "")

    async def fake_wait_ready(
        cua_url: str,
        os_type: str,
        timeout: float,
        poll_interval: float,
    ) -> bool:
        assert cua_url == "http://127.0.0.1:15000"
        assert os_type == "windows"
        assert timeout == 900
        assert poll_interval == 5
        return True

    monkeypatch.setattr(qemu_module, "_run_docker", fake_run_docker)
    monkeypatch.setattr(
        "ale_run.environments.providers.gcloud.wait_cua_ready",
        fake_wait_ready,
    )

    sandbox = await provider.acquire(
        SandboxSpec(snapshot="cpu-free", os="windows", task_id="demo/hello")
    )

    assert sandbox.endpoint == "http://127.0.0.1:15000"
    assert sandbox.metadata["provider"] == "qemu"
    assert sandbox.metadata["vcpus"] == 4
    assert sandbox.metadata["memory_gb"] == 8
    assert sandbox.metadata["novnc_url"] == "http://127.0.0.1:18000"
    assert sandbox.metadata["runner_image"] == "agentslastexam/ale-qemu:0.2.0"
    assert any(
        call[:5] == ("run", "--rm", "--pull=missing", "--entrypoint", "qemu-img")
        for call in docker_calls
    )
    start_call = next(call for call in docker_calls if call[:2] == ("run", "-d"))
    assert "--pull=missing" in start_call
    assert "--device=/dev/kvm" in start_call
    assert f"type=bind,src={base_qcow2},dst=/images/base.qcow2,readonly" in start_call
    assert "RAM_SIZE=8G" in start_call
    assert "CPU_CORES=4" in start_call
    assert start_call[-1] == "agentslastexam/ale-qemu:0.2.0"


@pytest.mark.asyncio
async def test_acquire_fails_immediately_when_qemu_container_exits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_qcow2 = tmp_path / "ale-win10.qcow2"
    base_qcow2.write_bytes(b"qcow2")
    provider = QemuProvider(_provider_config(tmp_path, base_qcow2))
    provider._preflight_done = True

    async def fake_run_docker(
        *args: str,
        check: bool = True,
    ) -> tuple[int, str, str]:
        _ = check
        if args[:4] == ("run", "--rm", "--pull=missing", "--entrypoint"):
            storage_mount = next(
                arg
                for arg in args
                if arg.startswith("type=bind,src=") and arg.endswith(",dst=/storage")
            )
            storage_dir = Path(
                storage_mount.removeprefix("type=bind,src=").removesuffix(",dst=/storage")
            )
            (storage_dir / "data.qcow2").write_bytes(b"overlay")
        if args[:2] == ("inspect", "--format"):
            if args[2] == "{{.State.Running}} {{.State.ExitCode}}":
                return (0, "false 15", "")
            return (0, "15000" if "5000/tcp" in args[2] else "18000", "")
        if args[:2] == ("logs", "--tail"):
            return (0, "qemu: could not initialize KVM", "")
        return (0, "", "")

    async def never_ready(
        cua_url: str,
        os_type: str,
        timeout: float,
        poll_interval: float,
    ) -> bool:
        _ = (cua_url, os_type, timeout, poll_interval)
        await qemu_module.asyncio.sleep(60)
        return False

    monkeypatch.setattr(qemu_module, "_run_docker", fake_run_docker)
    monkeypatch.setattr(
        "ale_run.environments.providers.gcloud.wait_cua_ready",
        never_ready,
    )

    with pytest.raises(RuntimeError, match=r"(?s)exited with status 15.*initialize KVM"):
        await provider.acquire(SandboxSpec(snapshot="cpu-free", os="windows", task_id="demo/hello"))


@pytest.mark.asyncio
async def test_gpu_task_fails_before_preflight(tmp_path: Path) -> None:
    base_qcow2 = tmp_path / "ale-win10.qcow2"
    base_qcow2.write_bytes(b"qcow2")
    provider = QemuProvider(_provider_config(tmp_path, base_qcow2))

    with pytest.raises(RuntimeError, match="CPU-only"):
        await provider.acquire(
            SandboxSpec(
                snapshot="cpu-free",
                os="windows",
                gpu="nvidia-l4-vws",
            )
        )


@pytest.mark.asyncio
async def test_release_removes_container_and_slot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_qcow2 = tmp_path / "ale-win10.qcow2"
    base_qcow2.write_bytes(b"qcow2")
    provider = QemuProvider(_provider_config(tmp_path, base_qcow2))
    slot_root = tmp_path / "qemu" / "runtime" / "slots" / "ale-qemu-test"
    slot_root.mkdir(parents=True)
    calls: list[tuple[str, ...]] = []

    async def fake_run_docker(
        *args: str,
        check: bool = True,
    ) -> tuple[int, str, str]:
        _ = check
        calls.append(args)
        return (0, "", "")

    monkeypatch.setattr(qemu_module, "_run_docker", fake_run_docker)

    from ale_run.base_interface import SandboxHandle

    sandbox = SandboxHandle(
        id="ale-qemu-test",
        endpoint="http://127.0.0.1:15000",
        os="windows",
        work_dir_base=r"C:\Users\User\.ale",
        task_data_root=r"E:\agenthle",
        node=r"C:\node.exe",
        python=r"C:\python.exe",
        mcp_server_dir=r"C:\cua_mcp_server",
        metadata={"provider": "qemu", "slot_root": str(slot_root)},
    )

    await provider.release(sandbox)

    assert ("rm", "-f", "ale-qemu-test") in calls
    assert not slot_root.exists()


def test_parse_hf_source_splits_repo_and_path() -> None:
    parsed = qemu_module._parse_hf_source(
        "hf://agents-last-exam/ale-images-qcow2/ale-ubuntu22.qcow2"
    )
    assert parsed.repo_id == "agents-last-exam/ale-images-qcow2"
    assert parsed.filename == "ale-ubuntu22.qcow2"


def test_parse_hf_source_rejects_short_source() -> None:
    with pytest.raises(ValueError, match="hf://"):
        qemu_module._parse_hf_source("hf://agents-last-exam/ale-images-qcow2")


def test_parse_hf_disk_manifest_validates_total_size() -> None:
    with pytest.raises(ValueError, match="part sizes"):
        qemu_module._parse_hf_disk_manifest(
            {
                "format": "ale-qemu-disk-parts-v1",
                "filename": "ale-win10.qcow2",
                "size": 7,
                "sha256": "0" * 64,
                "parts": [
                    {
                        "filename": "ale-win10.qcow2.parts/part-00000",
                        "size": 6,
                        "sha256": "1" * 64,
                    }
                ],
            }
        )


@pytest.mark.asyncio
async def test_resolve_hf_disk_downloads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = {
        "snapshots": {
            "cpu-free-ubuntu": {
                "image": "ale-ubuntu22",
                "disk_source": "hf://agents-last-exam/ale-images-qcow2/ale-ubuntu22.qcow2",
                "hf_revision": "deadbeef",
                "root": str(tmp_path / "qemu"),
            }
        }
    }
    provider = QemuProvider(config)
    snapshot = provider.config.snapshots["cpu-free-ubuntu"]
    captured: dict[str, object] = {}

    class Metadata:
        size = 5
        commit_hash = "deadbeef"
        etag = "etag"

    def fake_hf_hub_url(*, repo_id, filename, repo_type, revision):
        captured.update(
            repo_id=repo_id,
            filename=filename,
            repo_type=repo_type,
            revision=revision,
        )
        return "https://example.test/disk"

    def fake_get_hf_file_metadata(url):
        captured.update(url=url)
        return Metadata()

    def fake_hf_hub_download(
        *,
        repo_id,
        filename,
        repo_type,
        revision,
        local_dir,
        force_download,
    ):
        captured.update(
            repo_id=repo_id,
            filename=filename,
            repo_type=repo_type,
            revision=revision,
            force_download=force_download,
        )
        dest = Path(local_dir) / filename
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"qcow2")
        return str(dest)

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "hf_hub_url", fake_hf_hub_url)
    monkeypatch.setattr(
        huggingface_hub,
        "get_hf_file_metadata",
        fake_get_hf_file_metadata,
    )
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_hf_hub_download)

    resolved = await provider._resolve_disk(snapshot)

    assert resolved.is_file()
    assert captured["repo_id"] == "agents-last-exam/ale-images-qcow2"
    assert captured["filename"] == "ale-ubuntu22.qcow2"
    assert captured["repo_type"] == "dataset"
    assert captured["revision"] == "deadbeef"
    assert captured["force_download"] is False
    assert (resolved.parent / f"{resolved.name}.ale-source.json").is_file()


@pytest.mark.asyncio
async def test_resolve_hf_manifest_assembles_and_verifies_disk(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = "hf://agents-last-exam/ale-images-qcow2/ale-win10.qcow2.manifest.json"
    provider = QemuProvider(
        {
            "snapshots": {
                "cpu-free": {
                    "image": "ale-win10",
                    "disk_source": source,
                    "hf_revision": "deadbeef",
                    "root": str(tmp_path / "qemu"),
                }
            }
        }
    )
    snapshot = provider.config.snapshots["cpu-free"]
    part_data = {
        "ale-win10.qcow2.parts/part-00000": b"abc",
        "ale-win10.qcow2.parts/part-00001": b"def",
    }
    disk = b"".join(part_data.values())
    manifest = {
        "format": "ale-qemu-disk-parts-v1",
        "filename": "ale-win10.qcow2",
        "size": len(disk),
        "sha256": hashlib.sha256(disk).hexdigest(),
        "parts": [
            {
                "filename": filename,
                "size": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
            for filename, data in part_data.items()
        ],
    }

    class Metadata:
        size = 1
        commit_hash = "deadbeef"
        etag = "manifest-etag"

    def fake_hf_hub_download(
        *,
        filename,
        local_dir,
        force_download=False,
        **kwargs,
    ):
        _ = (force_download, kwargs)
        destination = Path(local_dir) / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        if filename.endswith(".manifest.json"):
            destination.write_text(json.dumps(manifest), encoding="utf-8")
        else:
            destination.write_bytes(part_data[filename])
        return str(destination)

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "hf_hub_url", lambda **kwargs: "test-url")
    monkeypatch.setattr(
        huggingface_hub,
        "get_hf_file_metadata",
        lambda url: Metadata(),
    )
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_hf_hub_download)

    resolved = await provider._resolve_disk(snapshot)

    assert resolved.name == "ale-win10.qcow2"
    assert resolved.read_bytes() == disk
    assert not any((snapshot.image_cache_dir / filename).exists() for filename in part_data)
    sidecar = json.loads(resolved.with_name(f"{resolved.name}.ale-source.json").read_text())
    assert sidecar["sha256"] == manifest["sha256"]


def test_assemble_hf_disk_resumes_completed_parts(tmp_path: Path) -> None:
    source = "hf://owner/repo/disk.qcow2.manifest.json"
    provider = QemuProvider(
        {
            "snapshots": {
                "cpu-free": {
                    "image": "ale-win10",
                    "disk_source": source,
                    "hf_revision": "deadbeef",
                    "root": str(tmp_path / "qemu"),
                }
            }
        }
    )
    snapshot = provider.config.snapshots["cpu-free"]
    destination = snapshot.image_cache_dir / "disk.qcow2"
    destination.parent.mkdir(parents=True)
    partial = destination.with_name(f"{destination.name}.partial")
    partial.write_bytes(b"abc-extra")
    state_path = partial.with_name(f"{partial.name}.json")
    state_path.write_text(
        json.dumps(
            {
                "source": source,
                "revision": "deadbeef",
                "commit_hash": "deadbeef",
                "manifest_sha256": hashlib.sha256(b"abcdef").hexdigest(),
                "completed_parts": 1,
            }
        ),
        encoding="utf-8",
    )
    completed_part = snapshot.image_cache_dir / "disk.qcow2.parts/part-00000"
    completed_part.parent.mkdir(parents=True)
    completed_part.write_bytes(b"abc")
    remaining_part = snapshot.image_cache_dir / "disk.qcow2.parts/part-00001"
    calls: list[str] = []

    def fake_download(*, filename, **kwargs):
        _ = kwargs
        calls.append(filename)
        remaining_part.write_bytes(b"def")
        return str(remaining_part)

    class Metadata:
        commit_hash = "deadbeef"

    manifest = qemu_module._HuggingFaceDiskManifest(
        filename="disk.qcow2",
        size=6,
        sha256=hashlib.sha256(b"abcdef").hexdigest(),
        parts=(
            qemu_module._HuggingFaceDiskPart(
                filename="disk.qcow2.parts/part-00000",
                size=3,
                sha256=hashlib.sha256(b"abc").hexdigest(),
            ),
            qemu_module._HuggingFaceDiskPart(
                filename="disk.qcow2.parts/part-00001",
                size=3,
                sha256=hashlib.sha256(b"def").hexdigest(),
            ),
        ),
    )

    resolved = provider._assemble_hf_disk(
        snapshot=snapshot,
        parsed=qemu_module._parse_hf_source(source),
        metadata=Metadata(),
        manifest=manifest,
        destination=destination,
        write_sidecar=lambda: None,
        hf_hub_download=fake_download,
    )

    assert resolved.read_bytes() == b"abcdef"
    assert calls == ["disk.qcow2.parts/part-00001"]
    assert not completed_part.exists()
    assert not state_path.exists()


@pytest.mark.asyncio
async def test_resolve_hf_disk_adopts_matching_existing_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = "hf://agents-last-exam/ale-images-qcow2/ale-win10.qcow2"
    provider = QemuProvider(
        {
            "snapshots": {
                "cpu-free": {
                    "image": "ale-win10",
                    "disk_source": source,
                    "hf_revision": "deadbeef",
                    "root": str(tmp_path / "qemu"),
                }
            }
        }
    )
    snapshot = provider.config.snapshots["cpu-free"]
    destination = snapshot.image_cache_dir / "ale-win10.qcow2"
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"qcow2")

    class Metadata:
        size = 5
        commit_hash = "deadbeef"
        etag = "etag"

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "hf_hub_url", lambda **kwargs: "test-url")
    monkeypatch.setattr(
        huggingface_hub,
        "get_hf_file_metadata",
        lambda url: Metadata(),
    )

    def fail_download(**kwargs):
        raise AssertionError("matching existing disk should not be downloaded")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fail_download)

    resolved = await provider._resolve_disk(snapshot)

    assert resolved == destination
    sidecar = destination.with_name(f"{destination.name}.ale-source.json")
    assert json.loads(sidecar.read_text())["commit_hash"] == "deadbeef"


@pytest.mark.asyncio
async def test_acquire_removes_container_when_network_setup_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_qcow2 = tmp_path / "ale-win10.qcow2"
    base_qcow2.write_bytes(b"qcow2")
    provider = QemuProvider(_provider_config(tmp_path, base_qcow2))
    provider._preflight_done = True
    calls: list[tuple[str, ...]] = []

    async def fake_run_docker(
        *args: str,
        check: bool = True,
    ) -> tuple[int, str, str]:
        calls.append(args)
        if args[:4] == ("run", "--rm", "--pull=missing", "--entrypoint"):
            storage_mount = next(
                arg
                for arg in args
                if arg.startswith("type=bind,src=") and arg.endswith(",dst=/storage")
            )
            storage_dir = Path(
                storage_mount.removeprefix("type=bind,src=").removesuffix(",dst=/storage")
            )
            (storage_dir / "data.qcow2").write_bytes(b"overlay")
        if args[:2] == ("exec", "ale-qemu-test"):
            raise RuntimeError("iptables failed")
        return (0, "", "")

    monkeypatch.setattr(qemu_module, "_run_docker", fake_run_docker)
    monkeypatch.setattr(
        qemu_module,
        "_generate_container_name",
        lambda spec: "ale-qemu-test",
    )

    with pytest.raises(RuntimeError, match="iptables failed"):
        await provider.acquire(
            SandboxSpec(
                snapshot="cpu-free",
                os="windows",
                task_id="demo/hello",
            )
        )

    assert ("rm", "-f", "ale-qemu-test") in calls
    assert not (tmp_path / "qemu" / "runtime" / "slots" / "ale-qemu-test").exists()


def test_qemu_outer_container_is_not_guest_filesystem() -> None:
    from ale_run.base_interface import SandboxHandle

    sandbox = SandboxHandle(
        id="ale-qemu-test",
        endpoint="http://127.0.0.1:15000",
        os="windows",
        work_dir_base=r"C:\Users\User\.ale",
        task_data_root=r"E:\agenthle",
        node=r"C:\node.exe",
        python=r"C:\python.exe",
        mcp_server_dir=r"C:\cua_mcp_server",
        metadata={"provider": "qemu", "container_name": "ale-qemu-test"},
    )

    assert _docker_container(sandbox) is None
