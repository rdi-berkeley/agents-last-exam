import json
from unittest.mock import AsyncMock

import pytest

from ale_run.base_interface import SandboxSpec
from ale_run.environments.providers import gcloud


@pytest.mark.parametrize(
    "case",
    [
        "regional",
        "path",
        "typo",
        "permission",
        "invalid",
        "lookup",
        "bad_json",
        "wrong_machine",
        "wrong_zone",
        "wrong_name",
        "not_list",
        "not_mapping",
    ],
)
async def test_retry_only_for_confirmed_regional_machine_absence(monkeypatch, case):
    attempts = []
    standard = "Machine type with name 'c4-standard-8' does not exist in zone 'zone-a'."
    errors = {
        "path": "The resource 'projects/test/zones/zone-a/machineTypes/c4-standard-8' was not found",
        "permission": "Permission denied for resource.machineType c4-standard-8 in zone-a",
        "invalid": "Invalid value for field 'resource.machineType': unsupported CPU count",
        "wrong_machine": standard.replace("c4-standard-8", "c4-standard-80"),
        "wrong_zone": standard.replace("zone-a", "zone-aa"),
    }

    async def create(**kwargs):
        attempts.append((kwargs["machine_type"], kwargs["zone"]))
        if len(attempts) == 1:
            return False, "", errors.get(case, standard), kwargs["zone"]
        return (
            True,
            json.dumps(
                [
                    {
                        "networkInterfaces": [{"accessConfigs": [{"natIP": "192.0.2.1"}]}],
                    }
                ]
            ),
            "",
            kwargs["zone"],
        )

    listed = {
        "typo": [],
        "wrong_name": [{"name": "c4-standard-80"}],
        "not_list": {"name": "c4-standard-8"},
        "not_mapping": [None],
    }.get(case, [{"name": "c4-standard-8"}])
    lookup = AsyncMock(
        return_value=(
            1 if case == "lookup" else 0,
            "{" if case == "bad_json" else json.dumps(listed),
            "access denied",
        )
    )
    monkeypatch.setattr(gcloud, "_try_create_in_zone", create)
    monkeypatch.setattr(gcloud, "_run_gcloud", lookup)
    monkeypatch.setattr(gcloud, "wait_cua_ready", AsyncMock(return_value=True))
    monkeypatch.setattr(
        gcloud,
        "_get_image_provenance",
        AsyncMock(
            return_value={
                "provider": "gcloud",
                "image_revision": "123",
            }
        ),
    )
    provider = gcloud.GcloudProvider(
        {
            "project": "test",
            "service_account_key": "unused",
            "snapshots": {"cpu": {"image": "ale-ubuntu22-v1-1", "zones": ["zone-a", "zone-b"]}},
        }
    )
    if case in ("regional", "path"):
        handle = await provider.acquire(SandboxSpec(snapshot="cpu"))
        assert attempts == [("c4-standard-8", "zone-a"), ("c4-standard-8", "zone-b")]
        assert handle.metadata["image_provenance"]["image_revision"] == "123"
        assert handle.metadata["zone"] == "zone-b"
        assert handle.metadata["machine_type"] == "c4-standard-8"
        assert provider.config.snapshots["cpu"].zones == ("zone-a", "zone-b")
    else:
        with pytest.raises(RuntimeError):
            await provider.acquire(SandboxSpec(snapshot="cpu"))
        assert len(attempts) == 1
    if case in ("permission", "invalid", "wrong_machine", "wrong_zone"):
        lookup.assert_not_awaited()
    else:
        lookup.assert_awaited_once_with(
            "compute",
            "machine-types",
            "list",
            "--filter=name=c4-standard-8",
            "--limit=1",
            "--format=json",
            project="test",
        )


async def test_capacity_exhaustion_preserves_machine_then_zone_fallback(monkeypatch):
    attempts = []

    async def create(**kwargs):
        attempts.append((kwargs["machine_type"], kwargs["zone"]))
        return False, "", "ZONE_RESOURCE_EXHAUSTED", kwargs["zone"]

    monkeypatch.setattr(gcloud, "_try_create_in_zone", create)
    lookup = AsyncMock()
    monkeypatch.setattr(gcloud, "_run_gcloud", lookup)
    provider = gcloud.GcloudProvider(
        {
            "project": "test",
            "service_account_key": "unused",
            "snapshots": {"cpu": {"image": "ale-ubuntu22-v1-1", "zones": ["zone-a", "zone-b"]}},
        }
    )
    with pytest.raises(RuntimeError, match="all machines/zones"):
        await provider.acquire(SandboxSpec(snapshot="cpu"))
    assert attempts == [
        ("c4-standard-8", "zone-a"),
        ("c4-standard-8", "zone-b"),
        ("n2-standard-8", "zone-a"),
        ("n2-standard-8", "zone-b"),
    ]
    lookup.assert_not_awaited()
