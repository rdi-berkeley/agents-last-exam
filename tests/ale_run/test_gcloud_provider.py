import json
import shlex
import subprocess
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from ale_run.base_interface import SandboxSpec
from ale_run.environments.providers import gcloud
from ale_run.environments.providers.gcloud import (
    _build_create_args,
    _build_snapshot_config,
)
from ale_run.orchestration.config_loader import _build_per_snapshot_env


def test_snapshot_boot_disk_size_is_optional() -> None:
    default = _build_snapshot_config({
        "image": "ale-ubuntu22",
        "zones": ["us-central1-a"],
    })
    expanded = _build_snapshot_config({
        "image": "ale-ubuntu22",
        "zones": ["us-central1-a"],
        "boot_disk_size_gb": 100,
    })

    assert default.boot_disk_size_gb is None
    assert expanded.boot_disk_size_gb == 100


def test_create_args_include_boot_disk_size_override() -> None:
    args = _build_create_args(
        name="ale-test",
        image="ale-ubuntu22",
        gpu=None,
        os_type="linux",
        network="default",
        subnet="default",
        machine_type="c4-standard-4",
        zone="us-central1-a",
        label_str="purpose=ale-run",
        project="test-project",
        boot_disk_type="hyperdisk-balanced",
        boot_disk_size_gb=100,
    )

    assert "--boot-disk-size=100GB" in args


@pytest.fixture
def acquisition(monkeypatch):
    state = SimpleNamespace(
        instance={
            "networkInterfaces": [{"accessConfigs": [{"natIP": "192.0.2.1"}]}],
            "disks": [
                {"boot": False, "source": "projects/test-project/zones/test-zone/disks/data"},
                {"boot": True, "source": "projects/test-project/zones/test-zone/disks/boot"},
            ],
        },
        disk={
            "sourceImageId": "1234567890123456789",
            "sourceImage": "projects/test-project/global/images/ale-ubuntu22-v1-1",
        },
        disk_rc=0,
        config={
            "project": "test-project",
            "service_account_key": "unused-key.json",
            "snapshots": {
                "cpu-free": {"image": "ale-ubuntu22-v1-1", "zones": ["test-zone"]},
            },
        },
    )

    async def run_gcloud(*args, project):
        assert project == "test-project"
        if args[:3] == ("compute", "instances", "create"):
            assert "--image=ale-ubuntu22-v1-1" in args
            return 0, json.dumps([state.instance]), ""
        if args[:3] == ("compute", "disks", "describe"):
            assert args[3] == "boot"
            assert "--zone=test-zone" in args
            return state.disk_rc, json.dumps(state.disk), "disk lookup failed"
        if args[:3] == ("compute", "instances", "delete"):
            return 0, "", ""
        raise AssertionError(f"unexpected gcloud command: {args}")

    state.commands = AsyncMock(side_effect=run_gcloud)
    state.readiness = AsyncMock(return_value=True)
    monkeypatch.setattr(gcloud, "_run_gcloud", state.commands)
    monkeypatch.setattr(gcloud, "wait_cua_ready", state.readiness)
    return state


@pytest.mark.asyncio
@pytest.mark.parametrize("expected_revision", [None, "1234567890123456789", 1234567890123456789])
async def test_acquire_records_actual_boot_image_revision(acquisition, expected_revision):
    acquisition.config["snapshots"]["cpu-free"]["image_revision"] = expected_revision
    provider = gcloud.GcloudProvider(acquisition.config)

    sandbox = await provider.acquire(SandboxSpec(snapshot="cpu-free"))

    provenance = sandbox.metadata["image_provenance"]
    assert json.loads(json.dumps(provenance)) == {
        "provider": "gcloud",
        "image": "ale-ubuntu22-v1-1",
        "image_revision": "1234567890123456789",
        "source_image": acquisition.disk["sourceImage"],
    }
    assert acquisition.commands.await_count == 2
    acquisition.readiness.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["mismatch", "missing_id", "invalid_id", "no_boot", "lookup"])
async def test_acquire_cleans_up_when_image_identity_cannot_be_verified(acquisition, failure):
    expected_error = ""
    if failure == "mismatch":
        acquisition.config["snapshots"]["cpu-free"]["image_revision"] = "9999999999999999999"
        expected_error = "image_revision mismatch"
    elif failure == "missing_id":
        acquisition.disk.pop("sourceImageId")
        expected_error = "no sourceImageId"
    elif failure == "invalid_id":
        acquisition.disk["sourceImageId"] = "ale-ubuntu22-v1-1"
        expected_error = "invalid sourceImageId"
    elif failure == "no_boot":
        acquisition.instance["disks"] = []
        expected_error = "requires an acquired boot disk"
    else:
        acquisition.disk_rc = 1
        expected_error = "disk lookup failed"

    with pytest.raises(RuntimeError, match=expected_error):
        await gcloud.GcloudProvider(acquisition.config).acquire(SandboxSpec(snapshot="cpu-free"))

    assert acquisition.commands.await_args_list[-1].args[:3] == ("compute", "instances", "delete")
    acquisition.readiness.assert_not_awaited()


@pytest.mark.parametrize("revision", ["latest", "projects/project/global/images/name", 0, -1, True])
def test_snapshot_rejects_non_numeric_image_revision(revision):
    with pytest.raises(ValueError, match="image_revision must be a numeric GCE image ID"):
        _build_snapshot_config({
            "image": "ale-ubuntu22-v1-1", "zones": ["test-zone"], "image_revision": revision,
        })


def test_loader_forwards_gcloud_image_revision():
    environment = _build_per_snapshot_env({
        "snapshots": {
            "cpu-free": {
                "provider": "gcloud",
                "image": "ale-ubuntu22-v1-1",
                "gcloud": {
                    "project": "test-project",
                    "service_account_key": "unused-key.json",
                    "zones": ["test-zone"],
                    "image_revision": "1234567890123456789",
                },
            },
        },
    }, "environment.yaml")

    provider = gcloud.GcloudProvider(environment.provider_specs["gcloud"].config)

    assert provider.config.snapshots["cpu-free"].image_revision == "1234567890123456789"


@pytest.fixture
def expansion_session(monkeypatch):
    from cua_bench.computers import remote

    state = SimpleNamespace(
        growpart_rc=0,
        resize_rc=0,
        disk_bytes=200 * 1024**3,
        partition_bytes=200 * 1024**3 - 2 * 1024**2,
        filesystem_bytes=195 * 1024**3,
        fstype="ext4",
        legacy_lsblk=False,
    )

    async def run_command(command, *, check):
        assert check is False
        executable, option, script = shlex.split(command)
        assert (executable, option) == ("bash", "-lc")
        stubs = f"""
findmnt() {{
  case "$*" in
    '-n -o SOURCE /') echo /dev/sda1 ;;
    '-n -o FSTYPE /') echo {state.fstype} ;;
    *) return 99 ;;
  esac
}}
lsblk() {{
  case "$*" in
    '-no PKNAME /dev/sda1') echo sda ;;
    '-no PARTN /dev/sda1')
      if {str(state.legacy_lsblk).lower()}; then return 1; fi
      echo 1 ;;
    '-bdno SIZE /dev/sda') echo {state.disk_bytes} ;;
    '-bno SIZE /dev/sda1'|'-bdno SIZE /dev/sda1') echo {state.partition_bytes} ;;
    '-dno START /dev/sda1')
      if {str(state.legacy_lsblk).lower()}; then return 1; fi
      echo 2048 ;;
    *) return 99 ;;
  esac
}}
sudo() {{
  case "$*" in
    'growpart /dev/sda 1') return {state.growpart_rc} ;;
    'resize2fs /dev/sda1'|'xfs_growfs /') return {state.resize_rc} ;;
    *) return 99 ;;
  esac
}}
df() {{
  case "$*" in
    '-B1 --output=size /') printf 'Size\\n{state.filesystem_bytes}\\n' ;;
    '-h /') echo expanded ;;
    *) return 99 ;;
  esac
}}
readlink() {{
  test "$*" = '-f /dev/sda1' || return 99
  echo /dev/sda1
}}
cat() {{
  case "$*" in
    /sys/class/block/sda1/partition) echo 1 ;;
    /sys/class/block/sda1/start) echo 2048 ;;
    *) return 99 ;;
  esac
}}
"""
        result = subprocess.run(
            ["bash", "-c", stubs + script], capture_output=True, text=True, timeout=5,
        )
        return {
            "return_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }

    state.command = AsyncMock(side_effect=run_command)
    monkeypatch.setattr(
        remote, "RemoteDesktopSession", lambda **kwargs: SimpleNamespace(run_command=state.command),
    )
    monkeypatch.setattr(gcloud, "_init_computer_skip_wait", lambda session: None)
    return state


@pytest.mark.parametrize("growpart_rc", [0, 1])
@pytest.mark.parametrize("fstype", ["ext4", "xfs"])
async def test_linux_root_expansion_accepts_grown_or_already_full_disk(
    expansion_session, growpart_rc, fstype,
):
    expansion_session.growpart_rc = growpart_rc
    expansion_session.fstype = fstype

    await gcloud.GcloudProvider._expand_linux_root_disk("http://unused")

    expansion_session.command.assert_awaited_once()


async def test_linux_root_expansion_supports_legacy_lsblk(expansion_session):
    expansion_session.legacy_lsblk = True

    await gcloud.GcloudProvider._expand_linux_root_disk("http://unused")


@pytest.mark.parametrize("growpart_rc", [0, 1, 2, 127])
@pytest.mark.parametrize("disk_gb", [101, 200])
async def test_linux_root_expansion_rejects_partition_left_at_old_size(
    expansion_session, growpart_rc, disk_gb,
):
    expansion_session.growpart_rc = growpart_rc
    expansion_session.disk_bytes = disk_gb * 1024**3
    expansion_session.partition_bytes = 100 * 1024**3
    expansion_session.filesystem_bytes = 98 * 1024**3

    with pytest.raises(RuntimeError, match="failed to expand Linux root disk"):
        await gcloud.GcloudProvider._expand_linux_root_disk("http://unused")


@pytest.mark.parametrize("failure", ["growpart", "resize", "small_filesystem", "unsupported"])
async def test_linux_root_expansion_rejects_errors_even_when_partition_is_full(
    expansion_session, failure,
):
    if failure == "growpart":
        expansion_session.growpart_rc = 2
    elif failure == "resize":
        expansion_session.resize_rc = 1
    elif failure == "small_filesystem":
        expansion_session.filesystem_bytes = 100 * 1024**3
    else:
        expansion_session.fstype = "btrfs"

    with pytest.raises(RuntimeError, match="failed to expand Linux root disk"):
        await gcloud.GcloudProvider._expand_linux_root_disk("http://unused")


async def test_acquire_deletes_vm_on_expansion_failure(acquisition, monkeypatch):
    acquisition.config["snapshots"]["cpu-free"]["boot_disk_size_gb"] = 200
    expand = AsyncMock(side_effect=RuntimeError("failed to expand Linux root disk"))
    monkeypatch.setattr(gcloud.GcloudProvider, "_expand_linux_root_disk", expand)

    with pytest.raises(RuntimeError, match="failed to expand Linux root disk"):
        await gcloud.GcloudProvider(acquisition.config).acquire(SandboxSpec(snapshot="cpu-free"))

    expand.assert_awaited_once_with("http://192.0.2.1:5000")
    assert acquisition.commands.await_args_list[-1].args[:3] == ("compute", "instances", "delete")
