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
    for path, entry in assets["selected_task_lists"].items():
        payload = (ROOT / path).read_bytes()
        tasks = [line.strip() for line in payload.decode().splitlines() if line.strip() and not line.lstrip().startswith("#")]
        assert hashlib.sha256(payload).hexdigest() == entry["sha256"]
        assert len(tasks) == entry["count"]
    assert assets["selected_task_lists"]["selected_tasks/full.txt"]["count"] == 152
    assert assets["selected_task_lists"]["selected_tasks/docker_support.txt"]["count"] == 99


@pytest.mark.parametrize("tag,platform", [("cpu-free-ubuntu", "linux"), ("cpu-free", "windows")])
def test_shipped_qemu_profile_pins_complete_release(assets, tag, platform):
    profile = yaml.safe_load((ROOT / "configs/environments/qemu.yaml").read_text())
    snapshot = profile["snapshots"][tag]
    config = _build_snapshot_config({"image": snapshot["image"], **snapshot["qemu"]})
    assert config.hf_revision == assets["huggingface"]["images"]["revision"]
    assert re.fullmatch(r"[0-9a-f]{40}", config.hf_revision)
    assert config.disk_source.endswith("/" + assets["images"][platform]["filename"])
    assert config.image == assets["images"][platform]["gcloud"]["name"]
    assert profile["task_data_source"] == "baked_in_sandbox"


def test_gcloud_free_profiles_use_versioned_images(assets):
    profile = yaml.safe_load((ROOT / "configs/environments/environment_gcloud.yaml").read_text())
    for snapshot, platform in (("cpu-free-ubuntu", "linux"), ("cpu-free", "windows"), ("gpu-free", "windows")):
        assert profile["snapshots"][snapshot]["image"] == assets["images"][platform]["gcloud"]["name"]
    for snapshot in ("cpu-license", "gpu-license"):
        assert profile["snapshots"][snapshot]["image"] == "ale-win10"


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
    for forbidden in ("/home/allennie", "private-vm-transfer", "Bearer ", "api_key", "password"):
        assert forbidden not in text
    assert assets["task_data"]["selected_tasks"] == 152
    assert assets["task_data"]["public_reference_format"] == "header-encrypted reference.7z"
