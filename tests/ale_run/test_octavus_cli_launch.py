import asyncio
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import yaml

from ale_run.agents.octavus_cli import OctavusCliConfig, OctavusCliDeployer
from ale_run.agents.octavus_cli import deployer as deployer_module


def _deployer(tmp_path):
    deployer = OctavusCliDeployer(
        SimpleNamespace(
            config=OctavusCliConfig(),
            env={},
            work_dir=str(tmp_path),
            sandbox=SimpleNamespace(task_data_root=str(tmp_path)),
        )
    )
    deployer._octoagent_path = "octoagent"
    return deployer


@pytest.mark.parametrize("cancel", [False, True])
async def test_launch_reaps_and_waits_for_probe(monkeypatch, tmp_path, cancel):
    deployer = _deployer(tmp_path)
    child = Mock(pid=123, returncode=0)
    child.poll.return_value = None if cancel else 0
    probe = Mock(pid=456)
    monkeypatch.setattr(deployer_module.subprocess, "Popen", Mock(return_value=child))
    monkeypatch.setattr(deployer, "_start_oom_probe", Mock(return_value=probe))
    reap = Mock()
    monkeypatch.setattr(deployer, "_reap", reap)

    async def cancelled_sleep(_seconds):
        raise asyncio.CancelledError

    if cancel:
        monkeypatch.setattr(deployer_module.asyncio, "sleep", cancelled_sleep)
        with pytest.raises(asyncio.CancelledError):
            await deployer.launch("do it")
        assert reap.call_args_list[0].args == (child,)
    else:
        result = await deployer.launch("do it")
        assert result.status == "completed"
    assert reap.call_args.args == (probe,)
    assert reap.call_args.kwargs == {"force": True}
    probe.wait.assert_called_once_with(timeout=deployer_module._TERM_GRACE_S)


async def test_cli_spawn_failure_does_not_start_probe(monkeypatch, tmp_path):
    deployer = _deployer(tmp_path)
    probe_start = Mock()
    monkeypatch.setattr(deployer, "_start_oom_probe", probe_start)
    monkeypatch.setattr(
        deployer_module.subprocess, "Popen", Mock(side_effect=OSError("missing CLI"))
    )
    with pytest.raises(OSError, match="missing CLI"):
        await deployer.launch("do it")
    probe_start.assert_not_called()


async def test_probe_start_failure_does_not_fail_run(monkeypatch, tmp_path):
    deployer = _deployer(tmp_path)
    child = Mock(pid=123, returncode=0)
    child.poll.return_value = 0
    popen = Mock(side_effect=[child, OSError("no probe")])
    monkeypatch.setattr(deployer_module.subprocess, "Popen", popen)
    result = await deployer.launch("do it")
    assert result.status == "completed"
    assert popen.call_count == 2


async def test_probe_cleanup_failure_does_not_mask_result(monkeypatch, tmp_path):
    deployer = _deployer(tmp_path)
    child = Mock(pid=123, returncode=0)
    child.poll.return_value = 0
    probe = Mock(pid=456)
    probe.wait.side_effect = subprocess.TimeoutExpired("probe", 3)
    monkeypatch.setattr(deployer_module.subprocess, "Popen", Mock(return_value=child))
    monkeypatch.setattr(deployer, "_start_oom_probe", Mock(return_value=probe))
    monkeypatch.setattr(deployer, "_reap", Mock())
    result = await deployer.launch("do it")
    assert result.status == "completed"


async def test_launch_discards_stale_probe_and_result_artifacts(monkeypatch, tmp_path):
    deployer = _deployer(tmp_path)
    for name in (
        "octoagent.result.json",
        "octoagent.stdout.log",
        "octoagent.stderr.log",
        "octoagent.pid",
        "octoagent.oom_probe.log",
        "octoagent.oom_probe.sh",
    ):
        (tmp_path / name).write_text("stale")
    child = Mock(pid=123, returncode=0)
    child.poll.return_value = 0
    monkeypatch.setattr(deployer_module.subprocess, "Popen", Mock(return_value=child))
    monkeypatch.setattr(deployer, "_start_oom_probe", Mock(return_value=None))

    result = await deployer.launch("do it")

    assert result.status == "completed"
    assert (tmp_path / "octoagent.pid").read_text() == "123"
    assert (tmp_path / "octoagent.stdout.log").read_text() == ""
    assert not (tmp_path / "octoagent.result.json").exists()
    assert not (tmp_path / "octoagent.oom_probe.log").exists()
    assert not (tmp_path / "octoagent.oom_probe.sh").exists()


def test_probe_script_has_valid_shell_syntax(tmp_path):
    script = tmp_path / "probe.sh"
    script.write_text(deployer_module._OOM_PROBE_SH)
    subprocess.run(["bash", "-n", str(script)], check=True)


def test_probe_file_write_failure_is_best_effort(tmp_path):
    probe = OctavusCliDeployer._start_oom_probe(
        tmp_path / "absent" / "probe.sh",
        tmp_path / "probe.log",
    )
    assert probe is None


def test_real_probe_is_detached_and_reaped(tmp_path):
    log = tmp_path / "probe.log"
    probe = OctavusCliDeployer._start_oom_probe(tmp_path / "probe.sh", log)
    assert probe is not None
    try:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if log.exists() and "oom_probe start" in log.read_text():
                break
            time.sleep(0.01)
        assert log.exists()
        assert "oom_probe start" in log.read_text()
        assert deployer_module.os.getpgid(probe.pid) == probe.pid
    finally:
        OctavusCliDeployer._reap(probe, force=True)
        probe.wait(timeout=5)
    assert probe.returncode is not None


def test_preset_and_config_pin_the_same_cli_version(tmp_path):
    preset = Path(__file__).resolve().parents[2] / "configs/agents/octavus_cli.yaml"
    config = OctavusCliConfig(**yaml.safe_load(preset.read_text())["config"])
    assert config.cli_version == OctavusCliConfig().cli_version == "@octavus/agent@1.0.10"
    assert _deployer(tmp_path).version == config.cli_version
    assert "octoagent.oom_probe.log" in OctavusCliDeployer.hot_artifacts
