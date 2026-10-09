import hashlib
import json
from pathlib import Path
import re

import pytest
import yaml

from ale_run.environments.images import get
from ale_run.environments.providers.qemu import _build_snapshot_config


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def assets():
    return json.loads((ROOT / "releases/v1.1/assets.json").read_text())


def test_selected_task_lists_match_release(assets):
    assert set(assets["selected_task_lists"]) == {
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "selected_tasks").rglob("*.txt")
    }
    for path, entry in assets["selected_task_lists"].items():
        payload = (ROOT / path).read_bytes()
        tasks = [line.strip() for line in payload.decode().splitlines() if line.strip() and not line.lstrip().startswith("#")]
        assert hashlib.sha256(payload).hexdigest() == entry["sha256"]
        assert len(tasks) == entry["count"]
    assert assets["selected_task_lists"]["selected_tasks/full.txt"]["count"] == 151
    assert assets["selected_task_lists"]["selected_tasks/cpu.txt"]["count"] == 146
    assert assets["selected_task_lists"]["selected_tasks/docker_support.txt"]["count"] == 102


def test_manifest_describes_one_public_release(assets):
    assert assets["version"] == "1.1"
    assert "revision" not in assets
    for entry in assets["huggingface"].values():
        assert entry["tag"] == "v1.1"
        assert re.fullmatch(r"[0-9a-f]{40}", entry["revision"])


@pytest.mark.parametrize("tag,platform", [("cpu-free-ubuntu", "linux"), ("cpu-free", "windows")])
def test_shipped_qemu_profile_pins_complete_release(assets, tag, platform):
    profile = yaml.safe_load((ROOT / "configs/environments/qemu.yaml").read_text())
    snapshot = profile["snapshots"][tag]
    config = _build_snapshot_config({"image": snapshot["image"], **snapshot["qemu"]})
    assert config.image_revision == assets["huggingface"]["images"]["revision"]
    assert re.fullmatch(r"[0-9a-f]{40}", config.image_revision)
    assert config.disk_source.endswith("/" + assets["images"][platform]["filename"])
    assert config.image == assets["images"][platform]["gcloud"]["name"]
    assert profile["task_data_source"] == "baked_in_sandbox"


def test_gcloud_profiles_use_versioned_images(assets):
    profile = yaml.safe_load((ROOT / "configs/environments/environment_gcloud.yaml").read_text())
    for snapshot, platform in (("cpu-free-ubuntu", "linux"), ("cpu-free", "windows"), ("gpu-free", "windows")):
        assert profile["snapshots"][snapshot]["image"] == assets["images"][platform]["gcloud"]["name"]


@pytest.mark.parametrize(
    "filename,snapshots",
    [
        ("docker.yaml", {"cpu-free-ubuntu"}),
        ("environment_aliyun.yaml", {"cpu-free-ubuntu", "cpu-free"}),
        ("environment_aws.yaml", {"cpu-free-ubuntu", "cpu-free"}),
        ("environment_gcloud.yaml", {"cpu-free-ubuntu", "cpu-free", "gpu-free"}),
        ("qemu.yaml", {"cpu-free-ubuntu", "cpu-free"}),
        ("static_win_dev.yaml", set()),
    ],
)
def test_shipped_profiles_only_expose_supported_routes(filename, snapshots):
    profile = yaml.safe_load((ROOT / "configs/environments" / filename).read_text())
    assert set(profile.get("snapshots", {})) == snapshots
    for snapshot in profile.get("snapshots", {}).values():
        get(snapshot["image"])
    if "image" in profile:
        get(profile["image"])
    assert profile["output_path"] is None


def test_docker_pins_published_image_and_matching_data(assets):
    profile = yaml.safe_load((ROOT / "configs/environments/docker.yaml").read_text())
    image = get(profile["snapshots"]["cpu-free-ubuntu"]["image"])
    assert image.docker_image == assets["docker"]["pinned_image"]
    assert image.docker_image == (
        assets["docker"]["image"] + "@" + assets["docker"]["registry_digest"]
    )
    assert profile["task_data_source"] == "local:task-data-v1.1"
    assert re.fullmatch(r"[0-9a-f]{64}", assets["task_data"]["archive_sha256"])
    assert re.fullmatch(r"[0-9a-f]{40}", assets["huggingface"]["archive"]["revision"])
    assert assets["docker"]["publication_status"] == "published"
    assert re.fullmatch(r"sha256:[0-9a-f]{64}", assets["docker"]["registry_digest"])
    assert re.fullmatch(r"sha256:[0-9a-f]{64}", assets["docker"]["config_digest"])


def test_manifest_preserves_gates_and_has_no_private_paths(assets):
    assert assets["huggingface"]["reference"]["gated"] == "manual"
    assert assets["huggingface"]["archive"]["gated"] == "manual"
    text = json.dumps(assets)
    assert "gcs_all" not in assets["task_data"]
    for forbidden in ("/home/allennie", "private-vm-transfer", "Bearer ", "api_key", "password"):
        assert forbidden not in text
    assert assets["task_data"]["selected_tasks"] == 151
    assert assets["task_data"]["public_reference_format"] == "header-encrypted reference.7z"
