import json
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
