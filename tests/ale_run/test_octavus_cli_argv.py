"""OctavusCliDeployer._build_argv construction.

The deployer uses the CLI's built-in default platform unless the caller pins a
platform_url to point the run at a different deployment. These tests lock that in:
the --platform-url flag appears iff platform_url is set, and the prompt stays last.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from ale_run.agents.octavus_cli import OctavusCliConfig, OctavusCliDeployer


def _deployer() -> OctavusCliDeployer:
    deployer = OctavusCliDeployer(SimpleNamespace(config=SimpleNamespace(), env={}))  # type: ignore[arg-type]
    deployer._octoagent_path = "octoagent"
    return deployer


def test_build_argv_omits_platform_url_by_default() -> None:
    cfg = OctavusCliConfig()
    argv = _deployer()._build_argv(cfg, workdir="/wd", prompt="do it")

    assert "--platform-url" not in argv
    assert argv[-1] == "do it"


def test_build_argv_includes_platform_url_when_set() -> None:
    cfg = OctavusCliConfig(platform_url="https://platform.example")
    argv = _deployer()._build_argv(cfg, workdir="/wd", prompt="do it")

    assert argv[argv.index("--platform-url") + 1] == "https://platform.example"
    assert argv[-1] == "do it"


@pytest.mark.parametrize("thinking", [None, "", "off", "low", "medium", "high", "max"])
def test_build_argv_thinking(thinking) -> None:
    argv = _deployer()._build_argv(
        OctavusCliConfig(thinking=thinking),
        workdir="/wd",
        prompt="do it",
    )
    if thinking:
        assert argv[argv.index("--thinking") + 1] == thinking
    else:
        assert "--thinking" not in argv
    assert argv[-1] == "do it"


@pytest.mark.parametrize("thinking", ["typo", "HIGH", False, 0, [], {}])
def test_config_rejects_invalid_thinking(thinking) -> None:
    with pytest.raises(ValueError, match="thinking must be"):
        OctavusCliConfig(thinking=thinking)


def test_build_argv_combines_overrides_without_exposing_credentials() -> None:
    cfg = OctavusCliConfig(
        platform_url="https://platform.example",
        operator_url="wss://operator.example",
        model="provider/primary",
        backup_model="provider/backup",
        thinking="high",
        capabilities={"memory": False},
        record=True,
        record_visibility="public",
        chrome_path="/bin/chrome",
        api_key="secret-test-key",
    )
    argv = _deployer()._build_argv(cfg, workdir="/wd", prompt="do it")
    assert argv == [
        "octoagent",
        "run",
        "--json",
        "--workdir",
        "/wd",
        "--platform-url",
        cfg.platform_url,
        "--operator-url",
        cfg.operator_url,
        "--model",
        cfg.model,
        "--backup-model",
        cfg.backup_model,
        "--thinking",
        "high",
        "--capability",
        "memory=off",
        "--record-public",
        "--chrome-path",
        "/bin/chrome",
        "do it",
    ]
    assert cfg.api_key not in argv
