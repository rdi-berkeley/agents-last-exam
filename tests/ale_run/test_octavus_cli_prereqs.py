"""OctavusCliDeployer computer-prereq + python-interpreter pinning behavior.

Regression tests for a `computer-use__label` break on the ALE box. The AT-SPI
`label` driver runs as `python3`; the ALE image prepends a gi-less venv to PATH,
so a bare `python3` could not import the AT-SPI2 bindings (which are installed
only for /usr/bin/python3) and label died with "AT-SPI2 not available" while
screenshots (scrot, no gi) still worked. The fix pins the driver to the system
interpreter (_build_env) and the prereq guard probes / installs against that same
interpreter (_ensure_computer_prereqs).
"""
from __future__ import annotations

import os
import subprocess
from types import SimpleNamespace

import ale_run.agents.octavus_cli.deployer as deployer_mod
from ale_run.agents.octavus_cli.deployer import OctavusCliDeployer


def _make_deployer() -> OctavusCliDeployer:
    # _ensure_computer_prereqs never touches self; a stub executor carrying a
    # `config` attribute is all BaseAgentDeployer.__init__ needs.
    return OctavusCliDeployer(SimpleNamespace(config=SimpleNamespace(), env={}))  # type: ignore[arg-type]


def _fake_run(atspi_rc: int, apt_calls: list[list[str]]):
    """subprocess.run stand-in: the `python3` AT-SPI probe returns atspi_rc; any
    `bash -c` (the apt-get install) is recorded and reported successful."""
    def _run(argv, *_args, **_kwargs):
        if argv[:1] == [deployer_mod._SYSTEM_PYTHON]:
            return subprocess.CompletedProcess(argv, atspi_rc)
        apt_calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    return _run


async def test_installs_when_atspi_bindings_missing(monkeypatch) -> None:
    monkeypatch.setattr(deployer_mod.shutil, "which", lambda _cmd: "/usr/bin/x")
    apt_calls: list[list[str]] = []
    monkeypatch.setattr(deployer_mod.subprocess, "run", _fake_run(atspi_rc=1, apt_calls=apt_calls))

    await _make_deployer()._ensure_computer_prereqs()

    assert apt_calls, "apt-get must run when the AT-SPI2 python bindings do not import"
    assert "apt-get install" in apt_calls[0][2]


async def test_skips_when_tools_and_atspi_present(monkeypatch) -> None:
    monkeypatch.setattr(deployer_mod.shutil, "which", lambda _cmd: "/usr/bin/x")
    apt_calls: list[list[str]] = []
    monkeypatch.setattr(deployer_mod.subprocess, "run", _fake_run(atspi_rc=0, apt_calls=apt_calls))

    await _make_deployer()._ensure_computer_prereqs()

    assert not apt_calls, "apt-get must be skipped when both tools and AT-SPI bindings are present"


async def test_installs_when_screenshot_tools_missing(monkeypatch) -> None:
    # A missing screenshot/automation tool must install even if AT-SPI imports.
    monkeypatch.setattr(deployer_mod.shutil, "which", lambda _cmd: None)
    apt_calls: list[list[str]] = []
    monkeypatch.setattr(deployer_mod.subprocess, "run", _fake_run(atspi_rc=0, apt_calls=apt_calls))

    await _make_deployer()._ensure_computer_prereqs()

    assert apt_calls, "apt-get must run when scrot/xdotool/ffmpeg are missing"


def test_build_env_pins_system_python_first_on_path(monkeypatch) -> None:
    # The MCP's `python3` (AT-SPI driver) must resolve to the gi-carrying system
    # interpreter, not a venv the image prepends to PATH.
    monkeypatch.setenv("PATH", "/opt/cua-server/.venv/bin:/usr/local/bin:/usr/bin:/bin")
    monkeypatch.setattr(deployer_mod.os.path, "isfile", lambda p: p == deployer_mod._SYSTEM_PYTHON)
    cfg = SimpleNamespace(display=":0", api_key=None)

    env = _make_deployer()._build_env(cfg)  # type: ignore[arg-type]

    assert env["PATH"].split(os.pathsep)[0] == deployer_mod._SYSTEM_BIN
    assert env["DISPLAY"] == ":0"
