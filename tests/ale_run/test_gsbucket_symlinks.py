import base64
import hashlib
import json
import os
import shlex
import stat
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from ale_run.base_interface import TaskDataSpec
from ale_run.environments.task_data import gsbucket


SOURCE = "gs://ale-data-public"
TASK = "life_sciences/tcga_brca_deg_analysis/base"
LINKS = {
    "runtime_env/.venv/bin/python": (
        "/home/user/.local/share/uv/python/cpython-3.14-linux-x86_64-gnu/bin/python3.14"
    ),
    "runtime_env/.venv/bin/python3": "python",
    "runtime_env/.venv/bin/python3.14": "python",
    "runtime_env/.venv/lib64": "lib",
}
HASHES = {
    "runtime_env/.venv/bin/python": "dd504fd25652beaeeb7164a35665015c24138ea7cae14387d0156827c0338429",
    "runtime_env/.venv/bin/python3": "11a4a60b518bf24989d481468076e5d5982884626aed9faeb35b8576fcd223e1",
    "runtime_env/.venv/bin/python3.14": "11a4a60b518bf24989d481468076e5d5982884626aed9faeb35b8576fcd223e1",
    "runtime_env/.venv/lib64": "76b5a357391276b282a516f54f48ef3c207f46d8192dc58c208d5183d38415f8",
}


def metadata_record(url, payload, marker="true"):
    properties = [
        ("Content-Length", str(len(payload)), 1),
        ("Content-Type", "application/octet-stream", 1),
        ("Metadata", "", 1),
        ("goog-reserved-file-is-symlink", marker, 2),
        ("Hash (md5)", base64.b64encode(hashlib.md5(payload).digest()).decode(), 1),
        ("ETag", "fixture", 1),
        ("Generation", "123456", 1),
        ("Metageneration", "1", 1),
    ]
    return (
        url
        + ":\n"
        + "".join(
            (" " * (indent * 4) + name + ":").ljust(28) + value + "\n"
            for name, value, indent in properties
        )
    )


@pytest.fixture
def restore_fixture(tmp_path, monkeypatch):
    binary = tmp_path / "bin"
    binary.mkdir()
    executable = binary / "gsutil"
    executable.write_text(
        f"#!{sys.executable}\n"
        "import json, os, pathlib, sys\n"
        "with open(os.environ['STUB_CALLS'], 'a') as log:\n"
        "    log.write(json.dumps(sys.argv[1:]) + '\\n')\n"
        "assert sys.argv[-2] == 'stat'\n"
        "assert sys.argv[-1] == os.environ['STUB_PREFIX'] + '/**'\n"
        "sys.stdout.write(pathlib.Path(os.environ['STUB_METADATA']).read_text())\n"
        "if os.environ.get('STUB_EMPTY'):\n"
        "    sys.stderr.write(os.environ.get('STUB_EMPTY_ERROR', ''))\n"
        "    sys.exit(int(os.environ.get('STUB_EMPTY_RC', '1')))\n"
        "if os.environ.get('STUB_FAILURE'):\n"
        "    sys.stderr.write('fixture metadata failure')\n"
        "    sys.exit(1)\n"
    )
    executable.chmod(0o755)
    prefix = f"{SOURCE}/{TASK}/input"
    metadata = tmp_path / "metadata.txt"
    metadata.write_text("")
    calls = tmp_path / "calls.jsonl"
    destination = tmp_path / "data with ' quote" / TASK / "input"
    destination.mkdir(parents=True)
    monkeypatch.setenv("PATH", str(binary) + os.pathsep + os.environ["PATH"])
    monkeypatch.setenv("STUB_PREFIX", prefix)
    monkeypatch.setenv("STUB_METADATA", str(metadata))
    monkeypatch.setenv("STUB_CALLS", str(calls))

    async def run_command(command, *, timeout):
        return subprocess.run(command, shell=True, text=True, capture_output=True, timeout=timeout)

    sandbox = SimpleNamespace(
        is_linux=True,
        python=sys.executable,
        metadata={"gcs_user_project": "billing-project", "gcs_key_path": "/injected/reader.json"},
        run_command=AsyncMock(side_effect=run_command),
    )
    return SimpleNamespace(
        sandbox=sandbox,
        prefix=prefix,
        destination=destination,
        metadata=metadata,
        calls=calls,
    )


def placeholders(fixture, links=LINKS):
    records = []
    for relative, target in links.items():
        path = fixture.destination / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(target)
        records.append(metadata_record(f"{fixture.prefix}/{relative}", target.encode()))
    fixture.metadata.write_text("".join(records))


@pytest.mark.asyncio
async def test_restores_all_four_native_targets_and_is_idempotent(restore_fixture):
    fixture = restore_fixture
    placeholders(fixture)
    library = fixture.destination / "runtime_env/.venv/lib"
    library.mkdir()
    library.chmod(0o750)
    library_stat = library.stat()
    regular = fixture.destination / "ordinary.txt"
    regular.write_text("python")
    regular.chmod(0o640)
    before = regular.stat()
    with fixture.metadata.open("a") as metadata:
        metadata.write(metadata_record(f"{fixture.prefix}/ordinary.txt", b"python", "false"))
    for _ in range(2):
        await gsbucket._restore_symlinks(fixture.sandbox, fixture.prefix, str(fixture.destination))
        for relative, target in LINKS.items():
            restored = fixture.destination / relative
            assert os.readlink(restored) == target
            assert hashlib.sha256(os.readlink(restored).encode()).hexdigest() == HASHES[relative]
            assert stat.S_IMODE(restored.lstat().st_mode) == 0o777
    assert regular.read_text() == "python"
    assert regular.stat().st_ino == before.st_ino
    assert stat.S_IMODE(regular.stat().st_mode) == 0o640
    assert library.stat() == library_stat
    calls = [json.loads(line) for line in fixture.calls.read_text().splitlines()]
    assert (
        calls
        == [
            [
                "-u",
                "billing-project",
                "-o",
                "Credentials:gs_service_key_file=/injected/reader.json",
                "stat",
                fixture.prefix + "/**",
            ]
        ]
        * 2
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("target", ["../not present", "space ' quote\nsecond line", "lib"])
async def test_targets_are_literal_and_never_followed(restore_fixture, target):
    fixture = restore_fixture
    placeholders(fixture, {"link": target})
    await gsbucket._restore_symlinks(fixture.sandbox, fixture.prefix, str(fixture.destination))
    assert os.readlink(fixture.destination / "link") == target


@pytest.mark.asyncio
@pytest.mark.parametrize("marker", ["false", "", "True"])
async def test_only_true_metadata_restores_links(restore_fixture, marker):
    fixture = restore_fixture
    placeholders(fixture, {"link": "missing"})
    fixture.metadata.write_text(metadata_record(f"{fixture.prefix}/link", b"missing", marker))
    await gsbucket._restore_symlinks(fixture.sandbox, fixture.prefix, str(fixture.destination))
    assert (fixture.destination / "link").is_symlink() == (marker == "True")


@pytest.mark.asyncio
async def test_metadata_failure_never_converts_any_placeholders(restore_fixture, monkeypatch):
    fixture = restore_fixture
    placeholders(fixture)
    monkeypatch.setenv("STUB_FAILURE", "1")
    with pytest.raises(RuntimeError, match="metadata listing failed"):
        await gsbucket._restore_symlinks(fixture.sandbox, fixture.prefix, str(fixture.destination))
    assert all(not (fixture.destination / path).is_symlink() for path in LINKS)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mutation",
    [
        "wrong-payload",
        "missing",
        "directory",
        "fifo",
        "hardlink",
        "wrong-link",
        "missing-hash",
    ],
)
async def test_invalid_placeholder_fails_before_any_restore(restore_fixture, mutation):
    fixture = restore_fixture
    placeholders(fixture)
    path = fixture.destination / "runtime_env/.venv/lib64"
    if mutation == "wrong-payload":
        path.write_text("bad")
    elif mutation == "missing-hash":
        fixture.metadata.write_text(fixture.metadata.read_text().replace("Hash (md5):", "Other:"))
    elif mutation == "hardlink":
        os.link(path, fixture.destination / "foreign")
    else:
        path.unlink()
        if mutation == "directory":
            path.mkdir()
        elif mutation == "fifo":
            os.mkfifo(path)
        elif mutation == "wrong-link":
            path.symlink_to("elsewhere")
    with pytest.raises(RuntimeError, match="GCS symlink restore failed"):
        await gsbucket._restore_symlinks(fixture.sandbox, fixture.prefix, str(fixture.destination))
    assert not (fixture.destination / "runtime_env/.venv/bin/python").is_symlink()


@pytest.mark.asyncio
@pytest.mark.parametrize("payload", [b"", b"a\x00b", b"\xff", b"a" * 4096, b"a" * 4097])
async def test_invalid_target_bytes_are_not_replaced(restore_fixture, payload):
    fixture = restore_fixture
    path = fixture.destination / "link"
    path.write_bytes(payload)
    fixture.metadata.write_text(metadata_record(f"{fixture.prefix}/link", payload))
    with pytest.raises(RuntimeError, match="GCS symlink restore failed"):
        await gsbucket._restore_symlinks(fixture.sandbox, fixture.prefix, str(fixture.destination))
    assert path.read_bytes() == payload


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "relative", ["../reference/gold", "/outside", "a/../link", "a//link", "./link"]
)
async def test_object_path_traversal_is_rejected(restore_fixture, relative):
    fixture = restore_fixture
    fixture.metadata.write_text(metadata_record(f"{fixture.prefix}/{relative}", b"missing"))
    with pytest.raises(RuntimeError, match="Unsafe or duplicate"):
        await gsbucket._restore_symlinks(fixture.sandbox, fixture.prefix, str(fixture.destination))


@pytest.mark.asyncio
async def test_out_of_prefix_metadata_is_rejected(restore_fixture):
    fixture = restore_fixture
    fixture.metadata.write_text(metadata_record(f"{SOURCE}/{TASK}/reference/gold", b"missing"))
    with pytest.raises(RuntimeError, match="outside requested prefix"):
        await gsbucket._restore_symlinks(fixture.sandbox, fixture.prefix, str(fixture.destination))


@pytest.mark.asyncio
async def test_duplicate_object_is_rejected_before_restore(restore_fixture):
    fixture = restore_fixture
    placeholders(fixture)
    fixture.metadata.write_text(fixture.metadata.read_text() * 2)
    with pytest.raises(RuntimeError, match="Unsafe or duplicate"):
        await gsbucket._restore_symlinks(fixture.sandbox, fixture.prefix, str(fixture.destination))
    assert all(not (fixture.destination / path).is_symlink() for path in LINKS)


@pytest.mark.asyncio
@pytest.mark.parametrize("location", ["ancestor", "root", "nested"])
async def test_symlinked_parent_never_reads_or_changes_external_file(restore_fixture, location):
    fixture = restore_fixture
    outside = fixture.destination.parent / "outside"
    outside.mkdir()
    external = outside / "link"
    external.write_text("missing")
    before = external.stat()
    if location == "nested":
        (fixture.destination / "nested").symlink_to(outside, target_is_directory=True)
        relative = "nested/link"
    else:
        fixture.destination.rmdir()
        fixture.destination.symlink_to(outside, target_is_directory=True)
        relative = "link"
        if location == "ancestor":
            nested = outside / "nested"
            nested.mkdir()
            external.rename(nested / "link")
            external = nested / "link"
            before = external.stat()
            fixture.destination /= "nested"
    fixture.metadata.write_text(metadata_record(f"{fixture.prefix}/{relative}", b"missing"))
    with pytest.raises(RuntimeError, match="GCS symlink restore failed"):
        await gsbucket._restore_symlinks(fixture.sandbox, fixture.prefix, str(fixture.destination))
    assert external.stat() == before
    assert external.read_text() == "missing"


@pytest.fixture
def staging_sandbox(monkeypatch):
    exists = gsbucket._gcs_exists

    async def plaintext_exists(sandbox, source):
        if source.endswith("/reference.7z"):
            return False
        return await exists(sandbox, source)

    monkeypatch.setattr(gsbucket, "_gcs_exists", plaintext_exists)
    return SimpleNamespace(
        is_linux=True,
        python="/opt/ale-run/.venv/bin/python",
        task_data_root="/data",
        metadata={"gcs_user_project": "billing-project", "gcs_key_path": "/injected/reader.json"},
        mkdir=AsyncMock(),
        rm=AsyncMock(),
        exists=AsyncMock(return_value=False),
        list_dir=AsyncMock(return_value=[{"is_dir": False}]),
        run_command=AsyncMock(return_value=subprocess.CompletedProcess([], 0, "", "")),
    )


@pytest.fixture
def task_data():
    return TaskDataSpec(
        requires_task_data=True,
        domain_name="life_sciences",
        task_name="tcga_brca_deg_analysis",
        variant_name="base",
    )


@pytest.mark.asyncio
async def test_fresh_input_preserves_auth_order_and_reference_isolation(staging_sandbox, task_data):
    sandbox = staging_sandbox
    result = await gsbucket.stage_input(sandbox, task_data, source=SOURCE)
    assert result == {"staged": ["input", "software"], "source": SOURCE}
    commands = [call.args[0] for call in sandbox.run_command.await_args_list]
    assert len(commands) == 7
    assert "rsync -r" in commands[1]
    assert shlex.split(commands[2])[:2] == [sandbox.python, "-c"]
    assert "rsync -r" in commands[4]
    assert "chmod +x" in commands[5]
    assert shlex.split(commands[6])[:2] == [sandbox.python, "-c"]
    assert not any("reference" in command for command in commands)
    for index in (0, 1, 3, 4):
        assert "gsutil -u billing-project -o Credentials:gs_service_key_file=" in commands[index]
    sandbox.mkdir.assert_any_await(f"/data/{TASK}/output")
    sandbox.rm.assert_not_awaited()


@pytest.mark.asyncio
async def test_baked_input_never_contacts_gcs_or_restores_links(staging_sandbox, task_data):
    sandbox = staging_sandbox
    sandbox.exists.return_value = True
    result = await gsbucket.stage_input(sandbox, task_data, source=SOURCE)
    assert result["staged"] == ["input(baked)", "software(baked)"]
    sandbox.run_command.assert_not_awaited()


@pytest.mark.asyncio
async def test_missing_input_creates_empty_dirs_without_restore(staging_sandbox, task_data):
    sandbox = staging_sandbox
    sandbox.run_command.return_value = subprocess.CompletedProcess([], 1, "", "")
    assert (await gsbucket.stage_input(sandbox, task_data, source=SOURCE))["staged"] == []
    assert sandbox.run_command.await_count == 2
    sandbox.mkdir.assert_any_await(f"/data/{TASK}/input")
    sandbox.mkdir.assert_any_await(f"/data/{TASK}/software")


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["stage_input", "stage_reference"])
async def test_failed_rsync_does_not_restore(staging_sandbox, task_data, phase):
    sandbox = staging_sandbox
    sandbox.run_command.side_effect = [
        subprocess.CompletedProcess([], 0, "", ""),
        subprocess.CompletedProcess([], 1, "", "failed copy"),
    ]
    with pytest.raises(RuntimeError, match="failed copy"):
        await getattr(gsbucket, phase)(sandbox, task_data, source=SOURCE)
    assert sandbox.run_command.await_count == 2


@pytest.mark.asyncio
async def test_reference_wipes_and_normalizes_before_restore(staging_sandbox, task_data):
    sandbox = staging_sandbox
    result = await gsbucket.stage_reference(sandbox, task_data, source=SOURCE)
    assert result == {"staged": ["reference"], "source": SOURCE}
    sandbox.rm.assert_awaited_once_with([f"/data/{TASK}/reference"])
    commands = [call.args[0] for call in sandbox.run_command.await_args_list]
    assert len(commands) == 4
    assert "rsync -r" in commands[1]
    assert "chmod -R 777" in commands[2]
    assert shlex.split(commands[3])[-2] == "-o"
    assert shlex.split(commands[3])[3:5] == [
        f"{SOURCE}/{TASK}/reference",
        f"/data/{TASK}/reference",
    ]


@pytest.mark.asyncio
async def test_missing_reference_preserves_existing_skip(staging_sandbox, task_data):
    sandbox = staging_sandbox
    sandbox.run_command.return_value = subprocess.CompletedProcess([], 1, "", "")
    assert await gsbucket.stage_reference(sandbox, task_data, source=SOURCE) == {
        "skipped": True,
        "reason": "no_reference_on_gcs",
    }
    sandbox.rm.assert_not_awaited()
    assert sandbox.run_command.await_count == 1


@pytest.mark.asyncio
async def test_restore_failure_is_propagated(staging_sandbox, task_data):
    sandbox = staging_sandbox
    sandbox.run_command.side_effect = [
        subprocess.CompletedProcess([], 0, "", ""),
        subprocess.CompletedProcess([], 0, "", ""),
        subprocess.CompletedProcess([], 1, "", "restore failed"),
    ]
    with pytest.raises(RuntimeError, match="GCS symlink restore failed"):
        await gsbucket.stage_input(sandbox, task_data, source=SOURCE)
    assert sandbox.run_command.await_count == 3


@pytest.mark.asyncio
async def test_windows_staging_commands_remain_unchanged(staging_sandbox, task_data):
    sandbox = staging_sandbox
    sandbox.is_linux = False
    sandbox.task_data_root = "E:\\ale-data"
    assert (await gsbucket.stage_input(sandbox, task_data, source=SOURCE))["staged"] == [
        "input",
        "software",
    ]
    assert (await gsbucket.stage_reference(sandbox, task_data, source=SOURCE))["staged"] == [
        "reference",
    ]
    commands = [call.args[0] for call in sandbox.run_command.await_args_list]
    assert len(commands) == 6
    assert all(command.startswith("powershell -NoProfile") for command in commands)
    assert all(" -c " not in command for command in commands)


def test_gsutil_keeps_ambient_boto_auth_without_provider_key():
    assert gsbucket._gsutil(SimpleNamespace(metadata={})) == "gsutil"


@pytest.mark.asyncio
async def test_unmarked_target_text_and_directory_object_stay_unchanged(restore_fixture):
    fixture = restore_fixture
    path = fixture.destination / "ordinary"
    path.write_text("python")
    metadata = metadata_record(f"{fixture.prefix}/ordinary", b"python")
    metadata += metadata_record(f"{fixture.prefix}/directory/", b"")
    fixture.metadata.write_text(
        "\n".join(
            line for line in metadata.splitlines() if "goog-reserved-file-is-symlink" not in line
        )
        + "\n"
    )
    await gsbucket._restore_symlinks(fixture.sandbox, fixture.prefix, str(fixture.destination))
    assert not path.is_symlink()
    assert path.read_text() == "python"


@pytest.mark.asyncio
async def test_external_target_bytes_and_mode_are_not_touched(restore_fixture):
    fixture = restore_fixture
    external = fixture.destination.parent / "external"
    external.write_bytes(b"not placeholder data")
    external.chmod(0o600)
    before = external.stat()
    placeholders(fixture, {"link": str(external)})
    await gsbucket._restore_symlinks(fixture.sandbox, fixture.prefix, str(fixture.destination))
    assert external.stat() == before
    assert os.readlink(fixture.destination / "link") == str(external)


@pytest.mark.asyncio
async def test_installed_sdk_encoder_and_stat_schema(restore_fixture):
    sdk = os.environ.get("ALE_TEST_GCLOUD_SDK_ROOT")
    if not sdk:
        pytest.skip("Set ALE_TEST_GCLOUD_SDK_ROOT to validate installed SDK transport APIs")
    fixture = restore_fixture
    script = r"""
import base64
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
from unittest.mock import patch

sdk, destination, prefix, encoded = sys.argv[1:]
sys.path[:0] = [sdk + '/lib', sdk + '/lib/third_party', sdk + '/platform/gsutil']
from googlecloudsdk.command_lib.storage import symlink_util
runpy.run_path(sdk + '/platform/gsutil/gsutil.py', run_name='sdk_schema_test')
from gslib.bucket_listing_ref import BucketListingObject
from gslib.storage_url import StorageUrlFromString
from gslib.third_party.storage_apitools import storage_v1_messages as messages
from gslib.utils.ls_helper import PrintFullInfoAboutObject

for relative, target in json.loads(encoded).items():
    placeholder = Path(destination) / relative
    placeholder.parent.mkdir(parents=True, exist_ok=True)
    link = placeholder.with_name(placeholder.name + '.source')
    link.symlink_to(target)
    try:
        with patch.object(symlink_util, 'get_symlink_placeholder_path',
                          return_value=str(placeholder)):
            symlink_util.get_symlink_placeholder_file(str(link))
    finally:
        link.unlink()
    payload = placeholder.read_bytes()
    assert payload == target.encode('utf-8')
    metadata = {}
    symlink_util.update_custom_metadata_dict_with_symlink_attributes(metadata, True)
    obj = messages.Object(
        size=len(payload), contentType='application/octet-stream', etag='fixture',
        md5Hash=base64.b64encode(hashlib.md5(payload).digest()).decode(),
        metadata=messages.Object.MetadataValue(additionalProperties=[
            messages.Object.MetadataValue.AdditionalProperty(key=key, value=value)
            for key, value in metadata.items()
        ]), generation=123456, metageneration=1,
    )
    reference = BucketListingObject(StorageUrlFromString(prefix + '/' + relative), root_object=obj)
    PrintFullInfoAboutObject(reference, incl_acl=False)
"""
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            "-c",
            script,
            sdk,
            str(fixture.destination),
            fixture.prefix,
            json.dumps(LINKS),
        ],
        text=True,
        capture_output=True,
        timeout=30,
        env={**os.environ, "CLOUDSDK_CORE_DISABLE_FILE_LOGGING": "true"},
    )
    assert result.returncode == 0, result.stderr
    fixture.metadata.write_text(result.stdout)
    await gsbucket._restore_symlinks(fixture.sandbox, fixture.prefix, str(fixture.destination))
    for relative, target in LINKS.items():
        assert os.readlink(fixture.destination / relative) == target


@pytest.mark.asyncio
@pytest.mark.parametrize("object_count", [0, 20_000])
async def test_installed_sdk_stat_uses_list_pages_not_object_gets(
    restore_fixture,
    monkeypatch,
    object_count,
):
    sdk = os.environ.get("ALE_TEST_GCLOUD_SDK_ROOT")
    if not sdk:
        pytest.skip("Set ALE_TEST_GCLOUD_SDK_ROOT to validate installed SDK list requests")
    fixture = restore_fixture
    script = r"""
import base64
from contextlib import redirect_stderr
import hashlib
import io
import json
import logging
import runpy
import sys
from types import SimpleNamespace
from unittest.mock import patch

sdk, prefix, object_count, encoded = sys.argv[1:]
object_count = int(object_count)
links = list(json.loads(encoded).items())
sys.path.insert(0, sdk + '/platform/gsutil')
runpy.run_path(sdk + '/platform/gsutil/gsutil.py', run_name='sdk_list_test')
from gslib.commands.stat import StatCommand
from gslib.gcs_json_api import GcsJsonApi
from gslib.third_party.storage_apitools import storage_v1_messages as messages
from gslib.utils.constants import NUM_OBJECTS_PER_LIST_PAGE
from gslib.utils import text_util

object_prefix = prefix.split('/', 3)[3] + '/'
pages = []
printed = 0
selected_output = []

def list_page(request, global_params):
    assert request.bucket == 'ale-data-public'
    assert request.prefix == object_prefix
    assert request.delimiter is None
    assert not request.versions
    assert request.userProject == 'billing-project'
    fields = set(global_params.fields.split(','))
    assert {'items/name', 'items/metadata', 'items/size', 'items/md5Hash',
            'nextPageToken'} <= fields
    assert request.maxResults == NUM_OBJECTS_PER_LIST_PAGE
    start = int(request.pageToken or 0)
    assert start == len(pages) * NUM_OBJECTS_PER_LIST_PAGE
    stop = min(start + request.maxResults, object_count)
    pages.append(request.pageToken)
    objects = []
    for index in range(start, stop):
        relative, target = links[index] if index < len(links) else (
            'nested/ordinary-' + str(index), 'ordinary fixture',
        )
        payload = target.encode()
        metadata = messages.Object.MetadataValue()
        if index < len(links):
            metadata.additionalProperties = [
                messages.Object.MetadataValue.AdditionalProperty(
                    key='goog-reserved-file-is-symlink', value='true',
                )
            ]
        objects.append(messages.Object(
            name=object_prefix + relative, size=len(payload), metadata=metadata,
            md5Hash=base64.b64encode(hashlib.md5(payload).digest()).decode(),
            etag='fixture', generation=index + 1,
        ))
    return messages.Objects(items=objects, nextPageToken=(
        str(stop) if stop < object_count else None
    ))

def capture_line(message, *args, **kwargs):
    global printed
    if message.startswith('gs://'):
        printed += 1
    if printed <= len(links):
        selected_output.append(message + '\n')

api = object.__new__(GcsJsonApi)
api.user_project = 'billing-project'
api.api_client = SimpleNamespace(objects=SimpleNamespace(List=list_page))
command = object.__new__(StatCommand)
command.args = [prefix + '/**']
command.gsutil_api = api
command.project_id = 'billing-project'
command.logger = logging.getLogger()
command.logger.setLevel(logging.INFO)
errors = io.StringIO()
with patch.object(api, 'GetObjectMetadata', side_effect=AssertionError('object GET')) as gets, \
     patch.object(api, '_GetObjectMetadataHelper', side_effect=AssertionError('hash GET')) as hashes, \
     patch.object(api, 'GetBucket', side_effect=AssertionError('bucket GET')) as buckets, \
     patch.object(text_util, 'print_to_fd', capture_line), redirect_stderr(errors):
    returncode = command.RunCommand()
gets.assert_not_called()
hashes.assert_not_called()
buckets.assert_not_called()
assert printed == object_count
assert len(pages) == max(1, (object_count + NUM_OBJECTS_PER_LIST_PAGE - 1)
                        // NUM_OBJECTS_PER_LIST_PAGE)
assert returncode == (0 if object_count else 1)
assert errors.getvalue() == ('' if object_count else 'No URLs matched: ' + prefix + '/**\n')
print(json.dumps({'list_requests': len(pages), 'object_gets': gets.call_count,
                  'hash_gets': hashes.call_count, 'objects': printed,
                  'metadata': ''.join(selected_output), 'stderr': errors.getvalue()}))
"""
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            "-c",
            script,
            sdk,
            fixture.prefix,
            str(object_count),
            json.dumps(LINKS),
        ],
        text=True,
        capture_output=True,
        timeout=60,
        env={**os.environ, "BOTO_CONFIG": os.devnull, "CLOUDSDK_CORE_DISABLE_FILE_LOGGING": "true"},
    )
    assert result.returncode == 0, result.stderr
    evidence = json.loads(result.stdout)
    assert evidence["objects"] == object_count
    assert evidence["object_gets"] == evidence["hash_gets"] == 0
    assert evidence["list_requests"] == (20 if object_count else 1)
    if object_count:
        placeholders(fixture)
    else:
        monkeypatch.setenv("STUB_EMPTY", "1")
        monkeypatch.setenv("STUB_EMPTY_ERROR", evidence["stderr"])
    fixture.metadata.write_text(evidence["metadata"])
    await gsbucket._restore_symlinks(fixture.sandbox, fixture.prefix, str(fixture.destination))
    if object_count:
        for relative, target in LINKS.items():
            assert os.readlink(fixture.destination / relative) == target
    else:
        assert list(fixture.destination.iterdir()) == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "extra_error,returncode",
    [
        ("", 1),
        ("AccessDenied: no permission\n", 1),
        ("", 2),
        ("network error\n", 1),
    ],
)
async def test_empty_match_is_distinct_from_failed_listing(
    restore_fixture,
    monkeypatch,
    extra_error,
    returncode,
):
    fixture = restore_fixture
    monkeypatch.setenv("STUB_EMPTY", "1")
    monkeypatch.setenv("STUB_EMPTY_RC", str(returncode))
    monkeypatch.setenv(
        "STUB_EMPTY_ERROR",
        extra_error + f"No URLs matched: {fixture.prefix}/**\n",
    )
    if extra_error or returncode != 1:
        with pytest.raises(RuntimeError, match="metadata listing failed"):
            await gsbucket._restore_symlinks(
                fixture.sandbox,
                fixture.prefix,
                str(fixture.destination),
            )
    else:
        await gsbucket._restore_symlinks(fixture.sandbox, fixture.prefix, str(fixture.destination))


@pytest.mark.asyncio
async def test_partial_listing_then_no_matches_is_not_treated_as_empty(
    restore_fixture, monkeypatch
):
    fixture = restore_fixture
    placeholders(fixture)
    monkeypatch.setenv("STUB_EMPTY", "1")
    monkeypatch.setenv("STUB_EMPTY_ERROR", f"No URLs matched: {fixture.prefix}/**\n")
    with pytest.raises(RuntimeError, match="metadata listing failed"):
        await gsbucket._restore_symlinks(fixture.sandbox, fixture.prefix, str(fixture.destination))
    assert all(not (fixture.destination / relative).is_symlink() for relative in LINKS)


@pytest.mark.asyncio
async def test_successful_empty_software_rsync_still_counts_as_staged(
    restore_fixture,
    staging_sandbox,
    task_data,
    monkeypatch,
):
    fixture = restore_fixture
    sandbox = staging_sandbox
    software = fixture.destination.parent / "software"
    software.mkdir()
    prefix = f"{SOURCE}/{TASK}/software"
    monkeypatch.setenv("STUB_PREFIX", prefix)
    monkeypatch.setenv("STUB_EMPTY", "1")
    monkeypatch.setenv("STUB_EMPTY_ERROR", f"No URLs matched: {prefix}/**\n")
    sandbox.python = sys.executable
    sandbox.task_data_root = str(fixture.destination.parents[3])
    sandbox.exists.side_effect = [True, False]

    async def run_command(command, *, timeout):
        if shlex.split(command)[0] == sandbox.python:
            return await fixture.sandbox.run_command(command, timeout=timeout)
        return subprocess.CompletedProcess([], 0, "", "")

    sandbox.run_command.side_effect = run_command
    assert await gsbucket.stage_input(sandbox, task_data, source=SOURCE) == {
        "staged": ["input(baked)", "software"],
        "source": SOURCE,
    }
    commands = [call.args[0] for call in sandbox.run_command.await_args_list]
    assert len(commands) == 4
    assert "rsync -r" in commands[1]
    assert shlex.split(commands[1])[-2:] == [prefix, str(software)]
    assert "chmod +x" in commands[2]
    assert shlex.split(commands[3])[:2] == [sandbox.python, "-c"]
    assert list(software.iterdir()) == []
