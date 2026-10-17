import pytest

from tasks.computing_math.k8s_migration_1.scripts.verify_k8s_migration import score_cicd


@pytest.mark.parametrize("command", [
    "docker run --rm aquasec/trivy:0.70.0 image webapp:latest",
    "docker run --rm -v /var/run/docker.sock:/var/run/docker.sock aquasec/trivy:0.70.0 image webapp:latest",
    'docker run --rm -v "$PWD/reports:/reports" docker.io/aquasec/trivy:0.70.0 image --format json --output /reports/image.json --exit-code 1 "$image"',
    "docker run --rm aquasec/trivy@sha256:abc image webapp:latest",
])
def test_official_scanner_container_entrypoint(tmp_path, command):
    workflow = tmp_path / "deploy.yml"
    workflow.write_text("jobs:\n  security:\n    steps:\n      - run: |\n          " + command + "\n")
    assert score_cicd(workflow)[1]["security"]


@pytest.mark.parametrize("command", [
    "echo docker run aquasec/trivy:0.70.0 image webapp:latest",
    "docker pull aquasec/trivy:0.70.0",
    "docker run --rm other/trivy:0.70.0 image webapp:latest",
    "docker run --entrypoint=echo aquasec/trivy:0.70.0 image webapp:latest",
    "docker run --entrypoint echo aquasec/trivy:0.70.0 image webapp:latest",
])
def test_image_mentions_and_replaced_entrypoints_are_not_scans(tmp_path, command):
    workflow = tmp_path / "deploy.yml"
    workflow.write_text("jobs:\n  security:\n    steps:\n      - run: |\n          " + command + "\n")
    assert not score_cicd(workflow)[1]["security"]
