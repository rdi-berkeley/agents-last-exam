import pytest
import yaml

from tasks.computing_math.k8s_migration_1.scripts.verify_k8s_migration import score_cicd


@pytest.mark.parametrize("command", [
    "docker run --rm backend:${{ github.sha }} pytest -q tests",
    "docker run --rm -e CI=true frontend:${{ github.sha }} npm test -- --runInBand",
    "docker run --rm backend:${{github.sha}} sh -c 'pip install pytest && pytest -q tests'",
    "./software/docker run --rm backend:${{ github.sha }} sh -c 'pip install pytest && pytest -q tests'",
    "docker run --rm backend:latest bash -lc 'python3 -m pytest'",
    "docker run --rm frontend:latest sh -ec 'npm test'",
    'docker run --rm "frontend:${{ github.sha }}" npm test',
    "docker run --rm backend:${{ matrix.version }} pytest",
])
def test_workflow_values_and_container_shells_preserve_test_commands(tmp_path, command):
    path = tmp_path / "deploy.yml"
    path.write_text(yaml.safe_dump({"jobs": {"test": {"steps": [{"run": command}]}}}))
    _, detail = score_cicd(path)
    assert detail["test"]
    assert not detail["build"]
    assert not detail["security"]


@pytest.mark.parametrize("command", [
    "echo 'docker run backend:${{ github.sha }} pytest'",
    "docker run backend:${{ github.sha }} echo pytest",
    "docker run backend:${{ github.sha }} sh -c 'echo pytest'",
    "docker run backend:${{ github.sha }} sh -c 'pip install pytest'",
    "docker run --entrypoint=echo backend:latest sh -c 'pytest'",
    "docker run --entrypoint echo backend:latest sh -c 'pytest'",
    "echo ${{ 'pytest; npm test' }}",
    "docker run backend:latest ${{ 'pytest' }}",
])
def test_mentions_opaque_commands_and_overridden_shells_do_not_count(tmp_path, command):
    path = tmp_path / "deploy.yml"
    path.write_text(yaml.safe_dump({"jobs": {"test": {"steps": [{"run": command}]}}}))
    assert not score_cicd(path)[1]["test"]
