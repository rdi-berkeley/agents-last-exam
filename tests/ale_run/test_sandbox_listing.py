from __future__ import annotations

import ast
import base64
import json
import ntpath
import shlex
import stat
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from ale_run.agents.dummy.config import DummyConfig
from ale_run.base_interface import SandboxHandle
from ale_run.base_interface import sandbox as sandbox_interface
from ale_run.executors import sandbox as sandbox_executor
from ale_run.executors.sandbox import SandboxExecutor


@pytest.fixture(params=["linux", "windows"])
def sandbox(request: pytest.FixtureRequest) -> SandboxHandle:
    return SandboxHandle(
        id="listing-test",
        endpoint="http://invalid.invalid",
        os=request.param,
        work_dir_base="/home/user/.ale" if request.param == "linux" else r"C:\Users\User\.ale",
        task_data_root="/data",
        node="node",
        python="python",
        mcp_server_dir="cua_mcp_server",
    )


@pytest.mark.parametrize("stdout", ["", " \r\n", "[]", " []\r\n", None])
@pytest.mark.asyncio
async def test_successful_empty_listing_and_gather_remain_successful(
    sandbox: SandboxHandle,
    stdout: str | None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = Mock(return_value={"success": True, "return_code": 0, "stdout": stdout})
    monkeypatch.setattr(sandbox_interface, "_post_cmd", transport)
    download = AsyncMock()
    monkeypatch.setattr(sandbox_executor, "_download_with_retry", download)

    assert await sandbox.list_dir(sandbox.work_dir_base) == []
    executor = SandboxExecutor(
        config=DummyConfig(), work_dir=sandbox.work_dir_base, sandbox=sandbox
    )
    report = await executor.gather_dir(src=sandbox.work_dir_base, dst=tmp_path)

    assert report.transport == "cua"
    assert report.files == report.bytes == 0
    assert report.error is None
    download.assert_not_awaited()
    assert transport.call_count == 2


@pytest.mark.parametrize(
    ("response", "expected_code", "expected_detail", "attempts"),
    [
        (
            {"success": True, "return_code": 1, "stdout": "", "stderr": "Access denied"},
            1,
            "Access denied",
            1,
        ),
        (
            {
                "success": True,
                "return_code": 2,
                "stdout": "partial\tf\t3\n",
                "stderr": "Missing descendant",
            },
            2,
            "Missing descendant",
            1,
        ),
        (
            {"success": True, "return_code": 1, "stdout": "[]", "stderr": "Listing failed"},
            1,
            "Listing failed",
            1,
        ),
        ({"success": False, "error": "Remote command rejected"}, 1, "Remote command rejected", 1),
        ({"success": True, "return_code": 1, "stderr": None}, 1, "no stderr", 1),
        (None, -1, "transport error", 3),
    ],
)
@pytest.mark.asyncio
async def test_failed_listing_propagates_into_gather_error(
    sandbox: SandboxHandle,
    response: dict | None,
    expected_code: int,
    expected_detail: str,
    attempts: int,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = Mock(return_value=response)
    monkeypatch.setattr(sandbox_interface, "_post_cmd", transport)
    download = AsyncMock()
    monkeypatch.setattr(sandbox_executor, "_download_with_retry", download)

    with pytest.raises(RuntimeError) as failure:
        await sandbox.list_dir(sandbox.work_dir_base)

    message = str(failure.value)
    assert sandbox.work_dir_base in message
    assert f"rc={expected_code}" in message
    assert expected_detail in message
    assert transport.call_count == attempts

    transport.reset_mock()
    executor = SandboxExecutor(
        config=DummyConfig(), work_dir=sandbox.work_dir_base, sandbox=sandbox
    )
    report = await executor.gather_dir(src=sandbox.work_dir_base, dst=tmp_path)

    assert report.files == report.bytes == 0
    assert report.error == message
    assert transport.call_count == attempts
    download.assert_not_awaited()


def test_listing_error_diagnostic_is_bounded(
    sandbox: SandboxHandle,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = Mock(
        return_value={
            "success": True,
            "return_code": 1,
            "stderr": "detail:" + "x" * 10_000,
        }
    )
    monkeypatch.setattr(sandbox_interface, "_post_cmd", transport)

    with pytest.raises(RuntimeError) as failure:
        sandbox_interface._list_dir_sync(sandbox, sandbox.work_dir_base)

    message = str(failure.value)
    assert "detail:" in message
    assert len(message) <= len(sandbox.work_dir_base) + 1100


@pytest.mark.asyncio
async def test_successful_listing_and_download_are_unchanged(
    sandbox: SandboxHandle,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = sandbox.work_dir_base
    stdout = "logs\td\t0\nlogs/transcript.jsonl\tf\t3\n"
    expected_relative = "logs/transcript.jsonl"
    if not sandbox.is_linux:
        stdout = json.dumps(
            [
                {"relpath": "logs", "is_dir": True, "size": 0},
                {"relpath": "logs/transcript.jsonl", "is_dir": False, "size": 3},
            ]
        )
        expected_relative = r"logs\transcript.jsonl"
    transport = Mock(return_value={"success": True, "return_code": 0, "stdout": stdout})
    monkeypatch.setattr(sandbox_interface, "_post_cmd", transport)
    downloaded = []

    async def download(handle: SandboxHandle, source: str, destination: Path) -> bool:
        assert handle is sandbox
        downloaded.append(source)
        destination.write_bytes(b"raw")
        return True

    monkeypatch.setattr(sandbox_executor, "_download_with_retry", download)
    assert await sandbox.list_dir(root) == [
        {"relpath": "logs", "is_dir": True, "size": 0},
        {"relpath": expected_relative, "is_dir": False, "size": 3},
    ]
    executor = SandboxExecutor(config=DummyConfig(), work_dir=root, sandbox=sandbox)
    report = await executor.gather_dir(src=root, dst=tmp_path)

    separator = "/" if sandbox.is_linux else "\\"
    assert downloaded == [root + separator + "logs" + separator + "transcript.jsonl"]
    assert (tmp_path / "logs/transcript.jsonl").read_bytes() == b"raw"
    assert report.files == 1
    assert report.bytes == 3
    assert report.error is None


def test_linux_native_listing_preserves_missing_empty_and_quoted_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sandbox = SandboxHandle(
        id="native-listing-test",
        endpoint="http://invalid.invalid",
        os="linux",
        work_dir_base=str(tmp_path),
        task_data_root="/data",
        node="node",
        python="python",
        mcp_server_dir="cua_mcp_server",
    )

    def local_command(
        handle: SandboxHandle, command: str, timeout: float
    ) -> subprocess.CompletedProcess:
        assert handle is sandbox
        return subprocess.run(command, shell=True, capture_output=True, text=True, timeout=timeout)

    monkeypatch.setattr(sandbox_interface, "_run_remote_sync", local_command)
    root = tmp_path / "agent's logs"
    assert sandbox_interface._list_dir_sync(sandbox, str(root)) == []
    root.mkdir()
    assert sandbox_interface._list_dir_sync(sandbox, str(root)) == []
    (root / "transcript.jsonl").write_bytes(b"raw")
    assert sandbox_interface._list_dir_sync(sandbox, str(root)) == [
        {"relpath": "transcript.jsonl", "is_dir": False, "size": 3},
    ]


def windows_listing_program(sandbox, monkeypatch, root):
    transport = Mock(return_value=subprocess.CompletedProcess("listing", 0, "[]", ""))
    monkeypatch.setattr(sandbox_interface, "_run_remote_sync", transport)
    assert sandbox_interface._list_dir_sync(sandbox, root) == []
    transport.assert_called_once()
    assert transport.call_args.kwargs == {"timeout": 30}
    command = transport.call_args.args[1]
    assert "powershell" not in command.lower()
    inline = ast.parse(shlex.split(command)[-1])
    encoded = next(
        node.args[0].value
        for node in ast.walk(inline)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "b64decode"
    )
    return base64.b64decode(encoded).decode()


@pytest.mark.parametrize(
    "root", [r"c:/Users/User/a/../logs", r"\\server\share\logs", r"\\?\C:\logs"]
)
def test_windows_walk_uses_extended_root_and_relative_json(sandbox, monkeypatch, capsys, root):
    if sandbox.is_linux:
        pytest.skip("Windows-only command")
    program = windows_listing_program(sandbox, monkeypatch, root)
    walked = []

    def walk(path, *, onerror):
        walked.append(path)
        assert path.startswith("\\\\?\\")
        yield path, ["nested"], ["long-" + "x" * 260 + ".jsonl"]
        yield ntpath.join(path, "nested"), [], ["quoted ' &\tname.jsonl"]

    def remote_stat(path):
        is_directory = path == walked[0] or path.endswith("nested") if walked else True
        return SimpleNamespace(st_mode=stat.S_IFDIR if is_directory else stat.S_IFREG, st_size=7)

    with monkeypatch.context() as remote:
        remote.setattr(sandbox_interface.os, "path", ntpath)
        remote.setattr(sandbox_interface.os, "stat", remote_stat)
        remote.setattr(sandbox_interface.os, "walk", walk)
        exec(compile(program, "native-listing", "exec"), {})
    entries = json.loads(capsys.readouterr().out)
    assert len(walked) == 1
    assert len(entries) == 3
    assert entries[0] == {"relpath": "nested", "is_dir": True, "size": 0}
    assert entries[-1]["relpath"] == "nested\\quoted ' &\tname.jsonl"
    assert all(not ntpath.isabs(entry["relpath"]) for entry in entries)


@pytest.mark.parametrize(
    "condition",
    ["missing", "not_directory", "empty", "root_denied", "walk_denied", "file_stat_failed"],
)
def test_windows_walk_missing_empty_and_errors(sandbox, monkeypatch, capsys, condition):
    if sandbox.is_linux:
        pytest.skip("Windows-only command")
    program = windows_listing_program(sandbox, monkeypatch, r"C:\logs")

    def remote_stat(path):
        if condition == "missing":
            raise FileNotFoundError(path)
        if condition == "root_denied":
            raise PermissionError(path)
        if condition == "file_stat_failed" and path.endswith("broken"):
            raise FileNotFoundError(path)
        mode = stat.S_IFREG if condition == "not_directory" else stat.S_IFDIR
        return SimpleNamespace(st_mode=mode, st_size=0)

    def walk(path, *, onerror):
        if condition == "walk_denied":
            onerror(PermissionError("denied descendant"))
        yield path, [], ["broken"] if condition == "file_stat_failed" else []

    with monkeypatch.context() as remote:
        remote.setattr(sandbox_interface.os, "path", ntpath)
        remote.setattr(sandbox_interface.os, "stat", remote_stat)
        remote.setattr(sandbox_interface.os, "walk", walk)
        if condition in {"root_denied", "walk_denied", "file_stat_failed"}:
            with pytest.raises(OSError):
                exec(compile(program, "native-listing", "exec"), {})
        else:
            try:
                exec(compile(program, "native-listing", "exec"), {})
            except SystemExit as exit_result:
                assert exit_result.code in (None, 0)
    output = capsys.readouterr().out
    if condition in {"missing", "not_directory", "empty"}:
        assert json.loads(output) == []
    else:
        assert output == ""


@pytest.mark.parametrize(
    "relative",
    [r"C:\outside", r"\\?\C:\outside", r"..\outside", r"nested\..\..\outside", r"C:outside", "."],
)
@pytest.mark.asyncio
async def test_windows_listing_rejects_nonrelative_entries(
    sandbox, monkeypatch, tmp_path, relative
):
    if sandbox.is_linux:
        pytest.skip("Windows-only protocol")
    payload = json.dumps([{"relpath": relative, "is_dir": False, "size": 1}])
    monkeypatch.setattr(
        sandbox_interface,
        "_post_cmd",
        Mock(return_value={"success": True, "return_code": 0, "stdout": payload}),
    )
    download = AsyncMock()
    monkeypatch.setattr(sandbox_executor, "_download_with_retry", download)
    executor = SandboxExecutor(
        config=DummyConfig(), work_dir=sandbox.work_dir_base, sandbox=sandbox
    )
    report = await executor.gather_dir(src=sandbox.work_dir_base, dst=tmp_path)
    assert report.error
    download.assert_not_awaited()


@pytest.mark.parametrize(
    "source,expected",
    [
        (r"C:/logs/nested/../file.bin", r"\\?\C:\logs\file.bin"),
        (r"\\server\share\logs\file.bin", r"\\?\UNC\server\share\logs\file.bin"),
        (r"\\?\C:\logs\file.bin", r"\\?\C:\logs\file.bin"),
        (r"\\?\UNC\server\share\file.bin", r"\\?\UNC\server\share\file.bin"),
        (r"logs\file.bin", r"logs\file.bin"),
        (r"C:relative.bin", r"C:relative.bin"),
    ],
)
def test_download_windows_extended_paths_keep_binary_chunks(
    sandbox, monkeypatch, tmp_path, source, expected
):
    transport = Mock(
        side_effect=[
            {"success": True, "content_b64": base64.b64encode(b"\x00\xff").decode()},
            {"success": True, "content_b64": base64.b64encode(b"\r").decode()},
        ]
    )
    monkeypatch.setattr(sandbox_interface, "_post_cmd", transport)
    monkeypatch.setattr(sandbox_interface, "_DOWNLOAD_CHUNK_BYTES", 2)
    target = tmp_path / "raw.bin"
    assert sandbox_interface._download_to_local_sync(sandbox, source, str(target), 30)
    assert target.read_bytes() == b"\x00\xff\r"
    assert [call.args[1]["params"]["offset"] for call in transport.call_args_list] == [0, 2]
    assert all(
        call.args[1]["params"]["path"] == (source if sandbox.is_linux else expected)
        for call in transport.call_args_list
    )


@pytest.mark.parametrize(
    "payload",
    [
        "not-json",
        "null",
        "{}",
        "[{}]",
        '[{"relpath":"file","is_dir":false,"size":-1}]',
        '[{"relpath":"file","is_dir":false,"size":true}]',
    ],
)
def test_windows_listing_rejects_invalid_payload(sandbox, monkeypatch, payload):
    if sandbox.is_linux:
        pytest.skip("Windows-only protocol")
    monkeypatch.setattr(
        sandbox_interface,
        "_post_cmd",
        Mock(return_value={"success": True, "return_code": 0, "stdout": payload}),
    )
    with pytest.raises(RuntimeError):
        sandbox_interface._list_dir_sync(sandbox, sandbox.work_dir_base)


@pytest.mark.parametrize("failure", [None, {"success": False, "error": "denied"}])
def test_download_failure_preserves_existing_destination(sandbox, monkeypatch, tmp_path, failure):
    monkeypatch.setattr(sandbox_interface, "_DOWNLOAD_CHUNK_BYTES", 2)
    monkeypatch.setattr(
        sandbox_interface,
        "_post_cmd",
        Mock(side_effect=[{"success": True, "content_b64": "YWI="}, failure]),
    )
    target = tmp_path / "raw.bin"
    target.write_bytes(b"original")
    assert not sandbox_interface._download_to_local_sync(
        sandbox, r"C:\logs\raw.bin", str(target), 30
    )
    assert target.read_bytes() == b"original"
