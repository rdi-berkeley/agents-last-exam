from dataclasses import replace
import re

import pytest

from ale_run.environments.images import get, registered
from ale_run.orchestration.config_loader import load_experiment


@pytest.mark.parametrize("base", ["ale-ubuntu22", "ale-win10"])
def test_versioned_public_image_preserves_runtime_contract(base):
    original = get(base)
    candidate = get(f"{base}-v1-1")
    assert candidate == replace(original, name=f"{base}-v1-1")
    assert original.name == base
    assert candidate.sandbox_paths() == original.sandbox_paths()


def test_versioned_docker_does_not_move_latest():
    original = get("ale-ubuntu22-docker")
    candidate = get("ale-ubuntu22-docker-v1-1")
    assert original.docker_image == "agentslastexam/ale-ubuntu22-docker:latest"
    assert re.fullmatch(
        r"agentslastexam/ale-ubuntu22-docker:v1\.1@sha256:[0-9a-f]{64}",
        candidate.docker_image,
    )
    assert candidate.sandbox_paths() == original.sandbox_paths()
    assert candidate.docker_entrypoint == original.docker_entrypoint
    assert len(registered()) == len(set(registered()))


def test_unknown_release_does_not_fall_back():
    with pytest.raises(KeyError, match="unknown image family"):
        get("ale-ubuntu22-v99-0")


@pytest.mark.parametrize("provider", ["qemu", "gcloud", "docker"])
def test_loader_preserves_explicit_release_selection(tmp_path, provider):
    image = "ale-ubuntu22-docker-v1-1" if provider == "docker" else "ale-ubuntu22-v1-1"
    knobs = {
        "qemu": "      disk_source: /persistent/ale-ubuntu22-v1.1.qcow2\n",
        "gcloud": (
            "      project: test-project\n      zones: [us-east1-c]\n"
            "      service_account_key: /credentials/test.json\n"
        ),
        "docker": "      memory_gb: 8\n",
    }
    agent = tmp_path / "agent.yaml"
    agent.write_text("harness: dummy\nmodel: test\n")
    environment = tmp_path / "environment.yaml"
    environment.write_text(
        f"snapshots:\n  cpu-free-ubuntu:\n    provider: {provider}\n"
        f"    image: {image}\n    {provider}:\n{knobs[provider]}"
        "task_data_source: baked_in_sandbox\noutput_path: local\n"
    )
    experiment = tmp_path / "experiment.yaml"
    experiment.write_text(
        f"name: release-test\nagent: {agent}\nenvironment: {environment}\n"
        "tasks:\n  - path: demo/hello\nwall_time_s: 18000\n"
    )
    specification = load_experiment(experiment)
    configuration = specification.environment.provider_specs[provider].config
    selected = (
        configuration if provider == "docker" else configuration["snapshots"]["cpu-free-ubuntu"]
    )
    assert selected["image"] == image
    assert specification.wall_time_s == 18000
