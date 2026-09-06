"""OctavusCliDeployer._build_argv construction.

The deployer uses the CLI's built-in default platform unless the caller pins a
platform_url to point the run at a different deployment. These tests lock that in:
the --platform-url flag appears iff platform_url is set, and the prompt stays last.
"""
from __future__ import annotations

from types import SimpleNamespace

from ale_run.agents.octavus_cli import OctavusCliConfig, OctavusCliDeployer


def _deployer() -> OctavusCliDeployer:
    d = OctavusCliDeployer(SimpleNamespace(config=SimpleNamespace(), env={}))  # type: ignore[arg-type]
    d._octoagent_path = "octoagent"
    return d


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
