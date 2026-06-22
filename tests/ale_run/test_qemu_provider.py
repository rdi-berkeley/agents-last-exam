from __future__ import annotations

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
                "qcow2": str(base_qcow2),
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
      qcow2: gs://ale-data-public/images/ale-win10.qcow2
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
        "qcow2": "gs://ale-data-public/images/ale-win10.qcow2",
    }


def test_loader_rejects_qemu_without_qcow2(tmp_path: Path) -> None:
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

    with pytest.raises(KeyError, match="qemu.qcow2"):
        load_experiment(experiment)


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
    assert any(
        call[:5] == ("run", "--rm", "--entrypoint", "qemu-img", "-v") for call in docker_calls
    )
    start_call = next(call for call in docker_calls if call[:2] == ("run", "-d"))
    assert "--device=/dev/kvm" in start_call
    assert f"{base_qcow2}:/images/base.qcow2:ro" in start_call
    assert "RAM_SIZE=8G" in start_call
    assert "CPU_CORES=4" in start_call
    assert start_call[-1] == "agentslastexam/ale-qemu:0.1.0"


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

    with pytest.raises(RuntimeError, match="does not support GPU"):
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
