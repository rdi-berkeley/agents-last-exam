import os
import subprocess
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from ale_run.environments.task_data import baked_in_sandbox, gsbucket


@pytest.fixture
def sandbox():
    return SimpleNamespace(
        is_linux=True,
        metadata={},
        mkdir=AsyncMock(),
        rm=AsyncMock(),
        run_command=AsyncMock(return_value=SimpleNamespace(returncode=0, stdout="", stderr="")),
    )


@pytest.fixture(autouse=True)
def task_paths(monkeypatch):
    monkeypatch.setattr(gsbucket, "task_subdir", lambda *args: "/tasks/domain/task/base")
    monkeypatch.setattr(
        gsbucket, "prepare_task_subdir", AsyncMock(return_value="/tasks/domain/task/base")
    )
    monkeypatch.setattr(gsbucket, "_gcs_prefix", lambda source, task: source + "/domain/task/base")
    monkeypatch.setattr(gsbucket, "_restore_symlinks", AsyncMock())


@pytest.mark.asyncio
async def test_encrypted_reference_downloads_before_decryption(monkeypatch, sandbox):
    monkeypatch.setattr(gsbucket, "_gcs_exists", AsyncMock(return_value=True))
    decrypt = AsyncMock(return_value={"staged": ["reference"], "decrypted_from": "reference.7z"})
    monkeypatch.setattr(baked_in_sandbox, "stage_reference", decrypt)
    result = await gsbucket.stage_reference(sandbox, None, source="gs://ale-data-public/v1.1")
    assert "cp gs://ale-data-public/v1.1/domain/task/base/reference.7z" in sandbox.run_command.call_args.args[0]
    sandbox.rm.assert_awaited_once_with(["/tasks/domain/task/base/reference.7z"])
    decrypt.assert_awaited_once()
    assert result["source"] == "gs://ale-data-public/v1.1"


@pytest.mark.asyncio
async def test_encrypted_download_error_never_decrypts_stale_archive(monkeypatch, sandbox):
    monkeypatch.setattr(gsbucket, "_gcs_exists", AsyncMock(return_value=True))
    sandbox.run_command.return_value.returncode = 1
    decrypt = AsyncMock()
    monkeypatch.setattr(baked_in_sandbox, "stage_reference", decrypt)
    with pytest.raises(RuntimeError, match="download failed"):
        await gsbucket.stage_reference(sandbox, None, source="gs://ale-data-public/v1.1")
    decrypt.assert_not_awaited()


@pytest.mark.asyncio
async def test_private_plaintext_reference_remains_supported(monkeypatch, sandbox):
    monkeypatch.setattr(gsbucket, "_gcs_exists", AsyncMock(side_effect=[False, True]))
    result = await gsbucket.stage_reference(sandbox, None, source="gs://private/v1.1")
    assert result["staged"] == ["reference"]
    assert "rsync" in sandbox.run_command.call_args_list[0].args[0]


@pytest.mark.asyncio
async def test_versioned_inputs_do_not_reuse_unversioned_baked_data(monkeypatch, sandbox):
    baked = AsyncMock(return_value=True)
    monkeypatch.setattr(gsbucket, "_has_baked_files", baked)
    monkeypatch.setattr(gsbucket, "_gcs_exists", AsyncMock(return_value=True))
    result = await gsbucket.stage_input(sandbox, None, source="gs://ale-data-public/v1.1")
    assert result["staged"] == ["input", "software"]
    baked.assert_not_awaited()
    assert sandbox.rm.await_count == 2


@pytest.mark.asyncio
async def test_legacy_source_keeps_baked_shortcut(monkeypatch, sandbox):
    monkeypatch.setattr(gsbucket, "_has_baked_files", AsyncMock(return_value=True))
    result = await gsbucket.stage_input(sandbox, None, source="gs://ale-data-public")
    assert result["staged"] == ["input(baked)", "software(baked)"]
    sandbox.rm.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("is_linux", [True, False], ids=["linux", "windows"])
async def test_gcs_exists_success_preserves_auth_and_exit_status(sandbox, is_linux):
    sandbox.is_linux = is_linux
    sandbox.metadata = {"gcs_user_project": "billing-project", "gcs_key_path": "/injected/key.json"}
    assert await gsbucket._gcs_exists(sandbox, "gs://bucket/input") is True
    command = sandbox.run_command.await_args.args[0]
    assert "gsutil -u billing-project -o Credentials:gs_service_key_file=/injected/key.json ls" in command
    assert ">" not in command
    assert "$null" not in command
    assert sandbox.run_command.await_args.kwargs == {"timeout": 30}
    if not is_linux:
        assert command.startswith("powershell -NoProfile -Command")
        assert "$ErrorActionPreference = 'Stop'" in command
        assert command.endswith('; exit $LASTEXITCODE"')


@pytest.mark.asyncio
@pytest.mark.parametrize("is_linux", [True, False], ids=["linux", "windows"])
@pytest.mark.parametrize("stderr", [
    "CommandException: One or more URLs matched no objects.\n",
    "No URLs matched: gs://bucket/input\n",
    "CommandException: No URLs matched: gs://bucket/input\r\n",
    "NotFoundException: 404 No such object: bucket/input\n",
    "NotFoundException: 404 gs://bucket/input does not exist.\n",
])
async def test_gcs_exists_confirmed_missing(sandbox, is_linux, stderr):
    sandbox.is_linux = is_linux
    sandbox.run_command.return_value = subprocess.CompletedProcess([], 1, "", stderr)
    assert await gsbucket._gcs_exists(sandbox, "gs://bucket/input") is False


LOOKUP_FAILURES = [
    (1, "", "AccessDeniedException: 403 Permission denied"),
    (1, "", "AuthenticationException: invalid_grant"),
    (1, "", "SocketError: DNS resolution failed"),
    (1, "", "ServiceException: 503 Service unavailable"),
    (127, "", "gsutil: command not found"),
    (1, "", "The term 'gsutil' is not recognized as the name of a cmdlet"),
    (124, "", "command timed out"),
    (-1, "", "transport error"),
    (1, "", ""),
    (1, "", "HTTPError: 404 authentication endpoint not found"),
    (1, "", "BucketNotFoundException: 404 gs://bucket bucket does not exist."),
    (1, "", "BucketNotFoundException: 404 The specified bucket does not exist."),
    (1, "", "NotFoundException: 404 The specified bucket does not exist."),
    (1, "", "AccessDeniedException: 404 Permission denied"),
    (1, "", "NotFoundException: 404 Permission denied"),
    (1, "", "NotFoundException: 404 Not found"),
    (1, "", "NotFoundException: 404 No such object: different/input"),
    (2, "", "CommandException: One or more URLs matched no objects."),
    (1, "", "AccessDeniedException: 403\nCommandException: One or more URLs matched no objects."),
    (1, "", "No URLs matched: gs://different/input"),
    (1, "gs://bucket/input/partial", "CommandException: One or more URLs matched no objects."),
    (1, "network failure on stdout", ""),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("is_linux", [True, False], ids=["linux", "windows"])
@pytest.mark.parametrize("returncode,stdout,stderr", LOOKUP_FAILURES)
async def test_gcs_exists_lookup_failures_raise(sandbox, is_linux, returncode, stdout, stderr):
    sandbox.is_linux = is_linux
    sandbox.run_command.return_value = subprocess.CompletedProcess([], returncode, stdout, stderr)
    with pytest.raises(RuntimeError, match="gsutil ls failed") as caught:
        await gsbucket._gcs_exists(sandbox, "gs://bucket/input")
    assert f"rc={returncode}" in str(caught.value)
    assert (stderr or stdout or "no diagnostic output") in str(caught.value)


@pytest.mark.asyncio
async def test_windows_probe_quotes_single_quote_in_url(sandbox):
    sandbox.is_linux = False
    assert await gsbucket._gcs_exists(sandbox, "gs://bucket/it's/input") is True
    command = sandbox.run_command.await_args.args[0]
    assert "ls 'gs://bucket/it''s/input'; exit $LASTEXITCODE" in command


@pytest.mark.asyncio
@pytest.mark.parametrize("is_linux", [True, False], ids=["linux", "windows"])
@pytest.mark.parametrize("error", [
    TimeoutError("probe timed out"),
    subprocess.TimeoutExpired("gsutil ls", 30),
    ConnectionError("connection reset"),
    FileNotFoundError("missing gsutil"),
])
async def test_gcs_exists_command_exceptions_raise(sandbox, is_linux, error):
    sandbox.is_linux = is_linux
    sandbox.run_command.side_effect = error
    with pytest.raises(RuntimeError, match=type(error).__name__):
        await gsbucket._gcs_exists(sandbox, "gs://bucket/input")


@pytest.mark.asyncio
@pytest.mark.parametrize("stderr", [
    "network timeout; access_token=fake-access-token",
    b"network timeout; access_token=fake-access-token",
])
async def test_gcs_exists_timeout_preserves_sanitized_stderr(sandbox, stderr):
    sandbox.run_command.side_effect = subprocess.TimeoutExpired("gsutil ls", 30, stderr=stderr)
    with pytest.raises(RuntimeError, match="network timeout") as caught:
        await gsbucket._gcs_exists(sandbox, "gs://bucket/input")
    assert "fake-access-token" not in str(caught.value)
    assert "access_token=[REDACTED]" in str(caught.value)


@pytest.mark.asyncio
@pytest.mark.parametrize("stderr,missing", [
    ("CommandException: One or more URLs matched no objects.", True),
    ("AccessDeniedException: 403 Permission denied", False),
    ("SocketError: DNS resolution failed", False),
])
async def test_linux_probe_captures_real_subprocess_stderr(
    sandbox, monkeypatch, tmp_path, stderr, missing,
):
    executable = tmp_path / "gsutil"
    executable.write_text("#!/bin/sh\nprintf '%s\\n' \"$PROBE_STDERR\" >&2\nexit 1\n")
    executable.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ["PATH"])
    monkeypatch.setenv("PROBE_STDERR", stderr)

    async def run_command(command, *, timeout):
        return subprocess.run(command, shell=True, text=True, capture_output=True, timeout=timeout)

    sandbox.run_command.side_effect = run_command
    if missing:
        assert await gsbucket._gcs_exists(sandbox, "gs://bucket/input") is False
    else:
        with pytest.raises(RuntimeError) as caught:
            await gsbucket._gcs_exists(sandbox, "gs://bucket/input")
        assert stderr in str(caught.value)


@pytest.mark.asyncio
@pytest.mark.parametrize("as_exception", [True, False])
async def test_gcs_exists_redacts_credentials_but_preserves_diagnostics(sandbox, as_exception):
    secrets = ["fake-bearer-token", "fake-refresh-token", "fake-client-secret", "fake-signature"]
    diagnostic = (
        "AuthenticationException: 401 invalid credentials\n"
        f"Authorization: Bearer {secrets[0]}\n"
        f'{{"refresh_token": "{secrets[1]}", "client_secret": "{secrets[2]}"}}\n'
        f"https://storage.example/object?X-Goog-Signature={secrets[3]}&other=value\n"
        '"private_key": "-----BEGIN PRIVATE KEY-----\\nfake-key-material\\n'
        '-----END PRIVATE KEY-----\\n"'
    )
    if as_exception:
        sandbox.run_command.side_effect = TimeoutError(diagnostic)
    else:
        sandbox.run_command.return_value = subprocess.CompletedProcess([], 1, "", diagnostic)
    with pytest.raises(RuntimeError) as caught:
        await gsbucket._gcs_exists(sandbox, "gs://bucket/input")
    message = str(caught.value)
    assert "401 invalid credentials" in message
    assert "[REDACTED]" in message
    assert all(secret not in message for secret in [*secrets, "fake-key-material"])
    if as_exception:
        assert caught.value.__suppress_context__


@pytest.mark.asyncio
@pytest.mark.parametrize("is_linux", [True, False], ids=["linux", "windows"])
@pytest.mark.parametrize("error", [
    subprocess.CompletedProcess([], 1, "", "AccessDeniedException: 403 denied"),
    subprocess.CompletedProcess([], 1, "", "SocketError: connection failed"),
    TimeoutError("probe timed out"),
])
async def test_versioned_input_probe_error_never_removes_destination(
    sandbox, monkeypatch, tmp_path, is_linux, error,
):
    sandbox.is_linux = is_linux
    destination = tmp_path / "input"
    destination.mkdir()
    valid_input = destination / "valid.txt"
    valid_input.write_text("existing valid input")
    monkeypatch.setattr(gsbucket, "prepare_task_subdir", AsyncMock(return_value=str(tmp_path)))
    if isinstance(error, Exception):
        sandbox.run_command.side_effect = error
    else:
        sandbox.run_command.return_value = error
    with pytest.raises(RuntimeError, match="gsutil ls failed"):
        await gsbucket.stage_input(sandbox, None, source="gs://bucket/v1.1")
    sandbox.rm.assert_not_awaited()
    gsbucket._restore_symlinks.assert_not_awaited()
    assert valid_input.read_text() == "existing valid input"


@pytest.mark.asyncio
@pytest.mark.parametrize("is_linux", [True, False], ids=["linux", "windows"])
async def test_versioned_refresh_probes_before_each_removal(sandbox, is_linux):
    sandbox.is_linux = is_linux

    async def check_probe_before_removal(paths):
        subdir = paths[0].rsplit("/" if is_linux else "\\", 1)[-1]
        command = sandbox.run_command.await_args.args[0]
        assert " ls " in command
        assert f"/domain/task/base/{subdir}" in command

    sandbox.rm.side_effect = check_probe_before_removal
    result = await gsbucket.stage_input(sandbox, None, source="gs://bucket/v1.1")
    assert result["staged"] == ["input", "software"]
    assert sandbox.rm.await_count == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("is_linux", [True, False], ids=["linux", "windows"])
async def test_missing_optional_software_still_creates_empty_directory(sandbox, is_linux):
    sandbox.is_linux = is_linux

    async def run_command(command, *, timeout):
        if " ls " in command and "/software" in command:
            return subprocess.CompletedProcess(
                [], 1, "", "CommandException: One or more URLs matched no objects.",
            )
        return subprocess.CompletedProcess([], 0, "", "")

    sandbox.run_command.side_effect = run_command
    result = await gsbucket.stage_input(sandbox, None, source="gs://bucket/v1.1")
    assert result["staged"] == ["input"]
    sandbox.mkdir.assert_any_await(
        gsbucket.join(sandbox, "/tasks/domain/task/base", "software")
    )
    assert sandbox.rm.await_count == 2
    assert gsbucket._restore_symlinks.await_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("is_linux", [True, False], ids=["linux", "windows"])
@pytest.mark.parametrize("failed_probe", ["archive", "plaintext"])
@pytest.mark.parametrize("diagnostic", [
    "AccessDeniedException: 403 denied",
    "BucketNotFoundException: 404 gs://bucket bucket does not exist.",
    "NotFoundException: 404 Permission denied",
])
async def test_reference_lookup_error_never_removes_or_decrypts(
    sandbox, monkeypatch, is_linux, failed_probe, diagnostic,
):
    sandbox.is_linux = is_linux
    failure = subprocess.CompletedProcess([], 1, "", diagnostic)
    sandbox.run_command.side_effect = (
        [failure] if failed_probe == "archive" else [
            subprocess.CompletedProcess(
                [], 1, "", "CommandException: One or more URLs matched no objects.",
            ),
            failure,
        ]
    )
    decrypt = AsyncMock()
    monkeypatch.setattr(baked_in_sandbox, "stage_reference", decrypt)
    with pytest.raises(RuntimeError) as caught:
        await gsbucket.stage_reference(sandbox, None, source="gs://bucket/v1.1")
    assert diagnostic in str(caught.value)
    sandbox.rm.assert_not_awaited()
    decrypt.assert_not_awaited()
    gsbucket._restore_symlinks.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("is_linux", [True, False], ids=["linux", "windows"])
async def test_missing_reference_still_skips_without_removal(sandbox, is_linux):
    sandbox.is_linux = is_linux
    sandbox.run_command.return_value = subprocess.CompletedProcess(
        [], 1, "", "CommandException: One or more URLs matched no objects.",
    )
    assert await gsbucket.stage_reference(sandbox, None, source="gs://bucket/v1.1") == {
        "skipped": True, "reason": "no_reference_on_gcs",
    }
    assert sandbox.run_command.await_count == 2
    sandbox.rm.assert_not_awaited()
