import json
from types import SimpleNamespace

import pytest

from tasks.computing_math.k8s_migration_1.scripts import verify_k8s_migration as verifier


@pytest.fixture
def configuration(tmp_path):
    source = tmp_path / "submission"
    source.mkdir()
    (source / "main.tf").write_text('output "name" { value = "test" }')
    return source


@pytest.mark.parametrize("valid", [False, True])
def test_native_validation_and_original_files_are_preserved(monkeypatch, configuration, tmp_path, valid):
    original = (configuration / "main.tf").read_bytes()
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        copied = type(configuration)(command[1].split("=", 1)[1])
        assert copied != configuration
        assert (copied / "main.tf").read_bytes() == original
        if command[2] == "init":
            (copied / ".terraform.lock.hcl").write_text("generated")
            assert "-backend=false" in command
            return SimpleNamespace(returncode=0, stdout="Initialized", stderr="")
        return SimpleNamespace(returncode=0 if valid else 1, stdout=json.dumps({"valid": valid, "diagnostics": []}), stderr="")

    monkeypatch.setattr(verifier.subprocess, "run", run)
    assert verifier.validate_terraform(configuration, "/trusted/terraform", tmp_path / "scratch")["valid"] is valid
    assert [command[2] for command in calls] == ["init", "validate"]
    assert not (configuration / ".terraform.lock.hcl").exists()
    assert (configuration / "main.tf").read_bytes() == original


@pytest.mark.parametrize("message", [
    "registry.terraform.io does not have a provider named scott-howard-535/minikube",
    "provider registry registry.terraform.io does not\nhave a provider named registry.terraform.io/scott-howard-535/minikube",
    "no available releases match the given constraints",
    "Error: Argument or block definition required",
])
def test_confirmed_configuration_error_fails_validation(monkeypatch, configuration, tmp_path, message):
    monkeypatch.setattr(verifier.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=1, stdout="", stderr=message))
    assert not verifier.validate_terraform(configuration, "terraform", tmp_path / "scratch")["valid"]


@pytest.mark.parametrize("message", [
    "Could not connect to registry: no such host",
    "Provider query timeout",
    "503 Service Unavailable",
    "Unknown unexpected provider registry error",
    "Failed to request discovery document; no available releases match",
])
def test_unavailable_registry_never_becomes_candidate_zero(monkeypatch, configuration, tmp_path, message):
    monkeypatch.setattr(verifier.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=1, stdout="", stderr=message))
    with pytest.raises(RuntimeError, match="initialization unavailable"):
        verifier.validate_terraform(configuration, "terraform", tmp_path / "scratch")


def test_missing_configuration_fails_without_invoking_terraform(monkeypatch, tmp_path):
    def unexpected(*args, **kwargs):
        pytest.fail("Missing input must not invoke Terraform")
    monkeypatch.setattr(verifier.subprocess, "run", unexpected)
    assert not verifier.validate_terraform(tmp_path / "missing", "terraform", tmp_path / "scratch")["valid"]


@pytest.mark.parametrize("valid", [True, False, None])
def test_main_uses_native_result_and_preserves_infrastructure_failure(monkeypatch, tmp_path, capsys, valid):
    monkeypatch.setattr(verifier, "parse_args", lambda: SimpleNamespace(
        output_dir=str(tmp_path / "output"), eval_tmp_dir=str(tmp_path),
        helm_version="unused", kubectl="kubectl", terraform="terraform"
    ))
    monkeypatch.setattr(verifier, "hard_gate_reason", lambda path: None)
    monkeypatch.setattr(verifier, "ensure_helm", lambda *args: "helm")
    monkeypatch.setattr(verifier, "render_helm", lambda *args: (True, [], ""))
    monkeypatch.setattr(verifier, "score_terraform", lambda *args: (1.0, {}))
    monkeypatch.setattr(verifier, "score_cluster", lambda *args: {
        name: {"score": 0} for name in ("pods", "services", "health", "hpa", "netpol")
    })

    def native(*args):
        if valid is None:
            raise RuntimeError("registry unavailable")
        return {"valid": valid}

    monkeypatch.setattr(verifier, "validate_terraform", native)
    assert verifier.main() == (1 if valid is None else 0)
    payload = json.loads(capsys.readouterr().out)
    if valid is None:
        assert payload["score"] is None
        assert payload["infrastructure_error"] == "registry unavailable"
    else:
        assert payload["breakdown"]["terraform_validity"]["score"] == float(valid)
