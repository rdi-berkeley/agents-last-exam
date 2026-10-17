from __future__ import annotations

import asyncio
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from ale_run.agents import _bootstrap
from ale_run.agents.zcode import deployer as module
from ale_run.agents.zcode.config import ZCodeConfig
from ale_run.agents.zcode.deployer import ZCodeDeployer


@pytest.fixture
def deployer(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    return ZCodeDeployer(
        SimpleNamespace(
            config=ZCodeConfig(api_key="test-private-key"),
            work_dir=str(tmp_path / "work"),
            env={},
            sandbox=SimpleNamespace(is_linux=False),
        )
    )


async def test_windows_install_builds_pinned_source_and_isolates_provider_files(
    deployer,
    tmp_path,
    monkeypatch,
):
    node = tmp_path / "runtime/node.exe"
    npm = node.parent / "node_modules/npm/bin/npm-cli.js"
    npm.parent.mkdir(parents=True)
    npm.touch()
    bootstrap = AsyncMock(return_value=(str(node), "npm.cmd"))
    monkeypatch.setattr(_bootstrap, "ensure_node_npm", bootstrap)
    monkeypatch.setattr(module, "ensure_cli_node", AsyncMock(side_effect=AssertionError("POSIX")))
    monkeypatch.setattr(module, "write_private_json", Mock(side_effect=AssertionError("POSIX")))
    monkeypatch.setattr(module.shutil, "which", lambda name: "git.exe" if name == "git" else None)
    runtime = AsyncMock(return_value=str(node))
    pnpm = AsyncMock(return_value="pnpm.cjs")
    clone = AsyncMock()
    install = AsyncMock()

    async def build(pnpm, repo):
        cli = repo / module._CLI_DIST_RELPATH
        cli.parent.mkdir(parents=True, exist_ok=True)
        cli.touch()
        builtin = cli.parent / "provider/zcode-builtin.json"
        builtin.parent.mkdir(exist_ok=True)
        builtin.write_text(
            json.dumps({"config": {"modelConfigRules": {"builtinProviderModelRules": []}}}),
            encoding="utf-8",
        )

    monkeypatch.setattr(deployer, "_ensure_windows_node", runtime)
    monkeypatch.setattr(deployer, "_ensure_pnpm", pnpm)
    monkeypatch.setattr(deployer, "_clone_or_update", clone)
    monkeypatch.setattr(deployer, "_pnpm_install", install)
    monkeypatch.setattr(deployer, "_pnpm_build", build)

    await deployer.install()
    repo = Path.home() / ".zcode_build/ZCode"
    clone.assert_awaited_once_with(repo, deployer.config.cli_ref)
    install.assert_awaited_once_with("pnpm.cjs", repo)
    pnpm.assert_awaited_once_with(str(npm))
    assert deployer._cli_path.is_file()
    personal = Path(deployer._provider_env[module._PERSONAL_PROVIDER_CONFIG_FILE_ENV])
    assert not personal.is_relative_to(deployer.executor.work_dir)
    value = json.loads(personal.read_text())
    rule = value["config"]["providerConfigRules"]["providerRules"][0]
    assert rule["config"]["access"]["apiKey"] == deployer.config.api_key
    environment = deployer._build_env(deployer.config)
    assert environment["HOME"] == environment["USERPROFILE"] == deployer._home_dir
    assert environment["PATH"].split(os.pathsep)[0] == str(node.parent)
    await deployer.install()
    assert deployer._home_dir != str(personal.parent)


async def test_windows_missing_api_key_fails_before_bootstrap(deployer, monkeypatch):
    deployer.config.api_key = None
    bootstrap = AsyncMock()
    monkeypatch.setattr(_bootstrap, "ensure_node_npm", bootstrap)
    with pytest.raises(RuntimeError, match="config.api_key is required"):
        await deployer.install()
    bootstrap.assert_not_awaited()


@pytest.mark.parametrize("cached", [False, True])
async def test_windows_old_node_uses_verified_private_runtime(deployer, monkeypatch, cached):
    monkeypatch.setattr(module.platform, "machine", lambda: "AMD64")
    runtime = Path.home() / ".zcode_build/node/node-v24.14.0-win-x64/node.exe"
    if cached:
        runtime.parent.mkdir(parents=True)
        runtime.touch()
    command = AsyncMock(
        side_effect=[
            subprocess.CompletedProcess([], 0, "v24.12.0\n", ""),
            subprocess.CompletedProcess([], 0, "v24.14.0\n", ""),
        ]
    )
    download = Mock()
    monkeypatch.setattr(deployer, "_run_command", command)
    monkeypatch.setattr(module, "_download_windows_node", download)
    assert await deployer._ensure_windows_node("baked-node.exe") == str(runtime)
    assert command.await_args_list[-1].args[0] == [str(runtime), "--version"]
    assert download.call_count == int(not cached)
    if not cached:
        download.assert_called_once_with(runtime.parent.parent, runtime.parent.name)
    assert os.environ["PATH"].split(os.pathsep)[0] == str(runtime.parent)


async def test_windows_node_rejects_wrong_cached_version(deployer, monkeypatch):
    monkeypatch.setattr(
        deployer,
        "_run_command",
        AsyncMock(return_value=subprocess.CompletedProcess([], 0, "v22.0.0", "")),
    )
    monkeypatch.setattr(module, "_download_windows_node", Mock())
    with pytest.raises(RuntimeError, match="private Windows Node version mismatch"):
        await deployer._ensure_windows_node("node.exe")


@pytest.mark.parametrize("failure", [None, "checksum", "npm"])
def test_windows_node_archive_verified_before_publish(tmp_path, monkeypatch, failure):
    directory = "node-v24.14.0-win-x64"
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr(f"{directory}/node.exe", "node")
        if failure != "npm":
            bundle.writestr(f"{directory}/node_modules/npm/bin/npm-cli.js", "npm")
    payload = archive.getvalue()
    digest = "0" * 64 if failure == "checksum" else hashlib.sha256(payload).hexdigest()
    downloads = []

    def urlopen(url, *, timeout):
        downloads.append(url)
        assert timeout in {60, 120}
        if url.endswith("SHASUMS256.txt"):
            return io.BytesIO(f"{digest}  {directory}.zip\n".encode())
        return io.BytesIO(payload)

    monkeypatch.setattr(module.urllib.request, "urlopen", urlopen)
    cache = tmp_path / "disk-cache"
    if failure:
        with pytest.raises(RuntimeError, match="checksum mismatch|missing Node or npm"):
            module._download_windows_node(cache, directory)
        assert not (cache / directory).exists()
    else:
        module._download_windows_node(cache, directory)
        assert (cache / directory / "node.exe").read_text() == "node"
        assert (cache / directory / "node_modules/npm/bin/npm-cli.js").is_file()
    assert all(url.startswith("https://nodejs.org/dist/v24.14.0/") for url in downloads)
    assert not list(cache.glob(".download-*"))


@pytest.mark.parametrize("cached_version", [None, "10.0.0", module._PNPM_VERSION])
async def test_windows_pnpm_is_private_pinned_and_not_a_cmd_shim(
    deployer,
    monkeypatch,
    cached_version,
):
    pnpm = Path.home() / ".zcode_build/tools/node_modules/pnpm/bin/pnpm.cjs"
    if cached_version:
        pnpm.parent.mkdir(parents=True)
        pnpm.touch()
    commands = []

    async def command(argv, **kwargs):
        commands.append(argv)
        if "install" in argv:
            pnpm.parent.mkdir(parents=True, exist_ok=True)
            pnpm.touch()
        version = cached_version if len(commands) == 1 and cached_version else module._PNPM_VERSION
        return subprocess.CompletedProcess(argv, 0, version + "\n", "")

    monkeypatch.setattr(deployer, "_run_command", command)
    assert await deployer._ensure_pnpm("npm-cli.js") == str(pnpm)
    installs = [argv for argv in commands if "install" in argv]
    assert len(installs) == int(cached_version != module._PNPM_VERSION)
    if installs:
        assert installs[0][0] == "npm-cli.js"
        assert f"pnpm@{module._PNPM_VERSION}" in installs[0]
        assert "--prefix" in installs[0]
        assert "-g" not in installs[0]


async def test_windows_clone_enables_longpaths_before_checkout(deployer, tmp_path, monkeypatch):
    command = AsyncMock(return_value=subprocess.CompletedProcess([], 0, "test-sha\n", ""))
    monkeypatch.setattr(deployer, "_run_command", command)
    await deployer._clone_or_update(tmp_path / "repo", deployer.config.cli_ref)
    commands = [call.args[0] for call in command.await_args_list]
    config_index = next(index for index, argv in enumerate(commands) if "core.longpaths" in argv)
    checkout_index = next(index for index, argv in enumerate(commands) if "checkout" in argv)
    assert config_index < checkout_index
    assert deployer.version == "src:test-sha"


@pytest.mark.parametrize("returncode", [0, 7])
async def test_windows_command_captures_utf8_without_posix_helpers(
    deployer,
    monkeypatch,
    returncode,
):
    monkeypatch.setattr(module, "run_command", AsyncMock(side_effect=AssertionError("POSIX")))
    monkeypatch.setattr(module, "wait_process", AsyncMock(side_effect=AssertionError("POSIX")))
    result = await deployer._run_command(
        [
            sys.executable,
            "-c",
            "import sys; sys.stdout.buffer.write('ok \\u4e2d'.encode('utf-8')); "
            f"sys.stderr.write('diagnostic'); sys.exit({returncode})",
        ]
    )
    assert result.returncode == returncode
    assert result.stdout == "ok \u4e2d"
    assert result.stderr == "diagnostic"


async def test_windows_command_runs_javascript_with_node_without_shell(deployer, monkeypatch):
    deployer._node_path = "node.exe"
    process = Mock(returncode=0)
    process.poll.return_value = 0
    popen = Mock(return_value=process)
    monkeypatch.setattr(module.subprocess, "Popen", popen)
    await deployer._run_command(["path with spaces/pnpm.cjs", "--version"])
    assert popen.call_args.args[0] == ["node.exe", "path with spaces/pnpm.cjs", "--version"]
    assert not popen.call_args.kwargs.get("shell")
    assert not popen.call_args.kwargs.get("start_new_session")
    assert not popen.call_args.kwargs.get("creationflags")


@pytest.mark.parametrize("cancel", [False, True])
async def test_windows_wait_reaps_on_timeout_or_cancellation(deployer, monkeypatch, cancel):
    monkeypatch.setattr(module, "wait_process", AsyncMock(side_effect=AssertionError("POSIX")))
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        operation = asyncio.create_task(deployer._wait_process(process, timeout=0.1))
        if cancel:
            await asyncio.sleep(0.02)
            operation.cancel()
        with pytest.raises(asyncio.CancelledError if cancel else TimeoutError):
            await operation
        assert process.poll() is not None
    finally:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=3)


@pytest.mark.parametrize("returncode", [0, 7])
async def test_windows_launch_bounds_command_line_and_preserves_containment(
    deployer,
    tmp_path,
    monkeypatch,
    returncode,
):
    deployer._node_path = "node.exe"
    deployer._cli_path = tmp_path / "cli with spaces/zcode.cjs"
    deployer._provider_env = {}
    prompt = 'quoted "prompt" & %PATH% \u4e2d\n' * 3000
    process = Mock(pid=12345, returncode=returncode)
    process.poll.return_value = returncode

    def popen(argv, **kwargs):
        kwargs["stderr"].write(deployer.config.api_key.encode())
        assert prompt not in argv
        assert len(subprocess.list2cmdline(argv)) < 2000
        assert kwargs["start_new_session"] is False
        assert not kwargs.get("creationflags")
        assert not kwargs.get("shell")
        assert argv[4:] == [
            "--mode",
            "yolo",
            "--no-browser",
            "--output-format",
            "stream-json",
            "--cwd",
            deployer.executor.work_dir,
        ]
        prompt_path = Path(kwargs["env"]["ALE_ZCODE_PROMPT_FILE"])
        assert prompt_path.read_text(encoding="utf-8") == prompt
        return process

    monkeypatch.setattr(module.subprocess, "Popen", popen)
    result = await deployer.launch(prompt)
    assert result.status == ("completed" if returncode == 0 else "failed")
    assert result.exit_code == returncode
    assert Path(result.transcript_path).is_file()
    assert Path(deployer.executor.work_dir, "zcode.pid").read_text() == "12345"
    if returncode:
        assert deployer.config.api_key not in result.error
        assert "[REDACTED]" in result.error


async def test_windows_prompt_loader_roundtrips_real_node_argv(deployer, tmp_path, monkeypatch):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is needed to execute the pinned CLI's CJS loading contract")
    cli = tmp_path / "fake cli.cjs"
    cli.write_text("console.log(JSON.stringify(process.argv.slice(2)));", encoding="utf-8")
    deployer._node_path = node
    deployer._cli_path = cli
    deployer._provider_env = {}
    monkeypatch.setattr(module, "wait_process", AsyncMock(side_effect=AssertionError("POSIX")))
    prompt = 'quotes " \\ & %PATH%\n\u4e2d' * 3000
    result = await deployer.launch(prompt)
    assert result.status == "completed"
    argv = json.loads(Path(result.transcript_path).read_text(encoding="utf-8"))
    assert argv[:4] == ["-p", prompt, "--mode", "yolo"]


async def test_linux_keeps_shared_runtime_and_environment(deployer, monkeypatch):
    deployer.executor.sandbox.is_linux = True
    command = AsyncMock(return_value=subprocess.CompletedProcess([], 0, "ok", ""))
    wait = AsyncMock(return_value=0)
    monkeypatch.setattr(module, "run_command", command)
    monkeypatch.setattr(module, "wait_process", wait)
    await deployer._run_command(["pnpm", "--version"], timeout=30)
    command.assert_awaited_once_with(["pnpm", "--version"], timeout=30)
    process = Mock()
    await deployer._wait_process(process)
    wait.assert_awaited_once_with(process, timeout=None, grace_s=module._TERM_GRACE_S)
    deployer._provider_env = {}
    deployer._home_dir = "/isolated-linux-home"
    monkeypatch.setenv("USERPROFILE", "unchanged")
    assert deployer._build_env(deployer.config)["USERPROFILE"] == "unchanged"
