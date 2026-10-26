from __future__ import annotations

import pytest

from ale_run.environments import task_data as task_data_pkg
from ale_run.environments.providers import daytona
from ale_run.environments.task_data import local_ref
from ale_run.orchestration.config_loader import (
    _build_per_snapshot_env,
    _validate_provider_required,
)


def _raw(**daytona_knobs):
    return {
        "snapshots": {
            "cpu-free": {
                "provider": "daytona",
                "image": "ale-win10",
                "daytona": {"snapshot": "ale-win-20261025", "api_key": "k", **daytona_knobs},
            },
        },
        "task_data_source": "baked_in_sandbox",
        "output_path": "local",
    }


def test_per_snapshot_env_routes_daytona_snapshot():
    spec = _build_per_snapshot_env(_raw(python=r"C:\ale-py\alepy.exe", ttl_minutes=90), "env.yaml")
    provider = spec.provider_specs["daytona"]
    assert provider.kind == "daytona"
    assert provider.config["api_key"] == "k"
    assert provider.config["ttl_minutes"] == 90
    route = provider.config["snapshots"]["cpu-free"]
    assert route == {"image": "ale-win10", "snapshot": "ale-win-20261025", "python": r"C:\ale-py\alepy.exe"}
    assert spec.snapshot_kind["cpu-free"] == "daytona"


def test_per_snapshot_env_requires_daytona_snapshot_name():
    raw = _raw()
    del raw["snapshots"]["cpu-free"]["daytona"]["snapshot"]
    with pytest.raises(KeyError, match="daytona.snapshot"):
        _build_per_snapshot_env(raw, "env.yaml")


def test_validate_provider_required_rejects_flat_daytona_config():
    with pytest.raises(KeyError, match="snapshots"):
        _validate_provider_required("daytona", {"api_key": "k"}, "env.yaml")


def test_build_provider_config_coerces_types():
    cfg = daytona._build_provider_config({
        "api_key": "k",
        "ttl_minutes": "120",
        "isolate_task_data": 1,
        "post_create_ps": "Write-Output hi",
        "labels": {"team": 7},
        "snapshots": {
            "cpu-free": {"image": "ale-win10", "snapshot": "snap", "cpu": "8", "memory": 32},
        },
    })
    assert cfg.ttl_minutes == 120
    assert cfg.isolate_task_data is True
    assert cfg.disable_windows_update is True
    assert cfg.post_create_ps == ("Write-Output hi",)
    assert cfg.labels == {"team": "7"}
    route = cfg.snapshots["cpu-free"]
    assert (route.cpu, route.memory, route.disk, route.python) == (8, 32, None, None)


def test_build_provider_config_requires_snapshot_name():
    with pytest.raises(KeyError, match="daytona.snapshot"):
        daytona._build_provider_config({"snapshots": {"cpu-free": {"image": "ale-win10"}}})


def test_localref_source_selects_local_ref_backend():
    assert task_data_pkg.select("localref:/data/tasks") is local_ref
