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
        run_command=AsyncMock(return_value=SimpleNamespace(returncode=0, stderr="")),
    )


@pytest.fixture(autouse=True)
def task_paths(monkeypatch):
    monkeypatch.setattr(gsbucket, "task_subdir", lambda *args: "/tasks/domain/task/base")
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
