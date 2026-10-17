import json
import subprocess
from copy import deepcopy
from types import SimpleNamespace

import pytest

from tasks.computing_math.k8s_migration_1.scripts import verify_k8s_migration as verifier


@pytest.fixture
def resources():
    documents = []
    for role, kind, service_port, target_port in (
        ("frontend", "Deployment", 80, 8080),
        ("backend", "Deployment", 5000, 5000),
        ("db", "StatefulSet", 5432, 5432),
    ):
        labels = {"app": "webapp", "tier": role, "instance": "submission"}
        documents.append({
            "kind": kind,
            "metadata": {"name": role, "labels": labels},
            "spec": {"template": {"metadata": {"labels": labels}, "spec": {
                "containers": [{"ports": [{"name": "application", "containerPort": target_port}]}],
            }}},
        })
        documents.append({
            "kind": "Service",
            "metadata": {"name": role, "labels": {"app": "webapp"}},
            "spec": {
                "type": "NodePort" if role == "frontend" else "ClusterIP",
                "selector": labels.copy(),
                "ports": [{"name": "traffic", "port": service_port, "targetPort": "application"}],
            },
        })
    return documents


def test_renamed_services_are_selected_by_workload_labels(resources):
    matches = verifier.workload_services(resources)
    assert {role: items[0]["service"]["metadata"]["name"] for role, items in matches.items()} == {
        "frontend": "frontend", "backend": "backend", "db": "db",
    }
    assert matches["frontend"][0]["ports"] == [{"port": 80, "target_port": 8080, "name": "traffic"}]
    assert verifier.score_config_correctness(resources)[1]["service_types"]


@pytest.mark.parametrize("role", ["frontend", "backend", "db"])
@pytest.mark.parametrize("failure", ["absent", "wrong-selector", "empty-selector", "shared-selector", "wrong-type", "bad-target", "udp"])
def test_service_names_cannot_mask_wrong_workloads_or_ports(resources, role, failure):
    service = next(document for document in resources if document["kind"] == "Service" and document["metadata"]["name"] == role)
    service["metadata"]["name"] = role + "-service"
    if failure == "absent":
        resources.remove(service)
    elif failure == "wrong-selector":
        service["spec"]["selector"]["tier"] = "unrelated"
    elif failure == "empty-selector":
        service["spec"]["selector"] = {}
    elif failure == "shared-selector":
        service["spec"]["selector"] = {"app": "webapp"}
    elif failure == "wrong-type":
        service["spec"]["type"] = "ExternalName"
    elif failure == "bad-target":
        service["spec"]["ports"][0]["targetPort"] = "unknown"
    else:
        service["spec"]["ports"][0]["protocol"] = "UDP"
    assert not verifier.score_config_correctness(resources)[1]["service_types"]


def test_headless_and_regular_database_services_are_allowed(resources):
    headless = deepcopy(resources[-1])
    headless["metadata"]["name"] = "db-headless"
    headless["spec"]["clusterIP"] = "None"
    resources.append(headless)
    assert len(verifier.workload_services(resources)["db"]) == 2
    assert verifier.score_config_correctness(resources)[1]["service_types"]


@pytest.mark.parametrize("failure", [None, "no-addresses", "unready", "wrong-port", "udp", "wrong-name"])
def test_live_endpoints_must_match_the_discovered_service_ports(resources, failure):
    endpoints = []
    for role, port in (("frontend", 8080), ("backend", 5000), ("db", 5432)):
        endpoints.append({"metadata": {"name": role}, "subsets": [{
            "addresses": [{"ip": "10.0.0.1"}], "ports": [{"name": "traffic", "port": port}],
        }]})
    subset = endpoints[1]["subsets"][0]
    if failure in {"no-addresses", "unready"}:
        subset["addresses"] = []
        subset["notReadyAddresses"] = [{"ip": "10.0.0.2"}]
    elif failure == "wrong-port":
        subset["ports"][0]["port"] = 5432
    elif failure == "udp":
        subset["ports"][0]["protocol"] = "UDP"
    elif failure == "wrong-name":
        endpoints[1]["metadata"]["name"] = "backend-service"
    cluster = SimpleNamespace(get=lambda kind, *args: {"items": endpoints if kind == "endpoints" else resources})
    score, detail = verifier.score_services_live(cluster)
    assert score == (1.0 if failure is None else 0.0)
    assert detail["populated"]["frontend"] == ["frontend"]


@pytest.mark.parametrize("returncode,body,expected", [(0, {"code": 200, "body": {"status": "healthy"}}, 1), (1, {}, 0), (0, {"code": 503, "body": {"status": "healthy"}}, 0), (0, {"code": 200, "body": {"status": "unhealthy"}}, 0), (0, "invalid", 0)])
def test_health_uses_selected_backend_and_still_requires_real_success(resources, returncode, body, expected):
    calls = []

    def run(*arguments):
        calls.append(arguments)
        return SimpleNamespace(returncode=returncode, stdout=json.dumps(body), stderr="connection failed" if returncode else "")

    cluster = SimpleNamespace(get=lambda *args: {"items": resources}, run=run)
    score, detail = verifier.score_health_live(cluster)
    assert score == expected
    assert "http://backend:5000/health" in calls[0][-1]
    assert detail["probes"][0]["stdout"] == json.dumps(body)


def test_live_receipt_retains_failed_queries_and_successful_cleanup(tmp_path, monkeypatch):
    cluster = verifier.Cluster("/task/software/kubectl", "context-a", "namespace-a")
    monkeypatch.setattr(verifier, "discover_cluster", lambda kubectl: cluster)

    def run(command, **kwargs):
        return SimpleNamespace(returncode=7 if "endpoints" in command else 0, stdout="probe output\n", stderr="probe stderr\n")

    monkeypatch.setattr(verifier.subprocess, "run", run)

    def check(current):
        result = current.run("get", "endpoints")
        current.run("delete", "pod", "probe", "--ignore-not-found")
        return 0.0, {"returncode": result.returncode}

    for name in ("pods", "services", "health", "hpa", "netpol"):
        monkeypatch.setattr(verifier, "score_" + name + "_live", check)
    receipt = tmp_path / "live-probes.json"
    results = verifier.score_cluster(cluster.kubectl, receipt)
    saved = json.loads(receipt.read_text())
    assert saved["results"] == results
    assert len(saved["observations"]) == 10
    assert saved["observations"][0] == {
        "command": [cluster.kubectl, "--context", "context-a", "--namespace", "namespace-a", "get", "endpoints"],
        "returncode": 7, "stdout": "probe output\n", "stderr": "probe stderr\n",
    }
    assert saved["observations"][-1]["returncode"] == 0
    assert saved["observations"][-1]["command"][-4:] == ["delete", "pod", "probe", "--ignore-not-found"]


def test_cluster_discovery_failure_is_persisted(tmp_path, monkeypatch):
    def fail(kubectl):
        raise RuntimeError("no task context")

    monkeypatch.setattr(verifier, "discover_cluster", fail)
    receipt = tmp_path / "live-probes.json"
    results = verifier.score_cluster("/task/software/kubectl", receipt)
    assert all(result["score"] == 0 for result in results.values())
    assert json.loads(receipt.read_text())["discovery_error"] == "no task context"


def test_probe_timeout_retains_partial_output(monkeypatch):
    def fail(command, **kwargs):
        raise subprocess.TimeoutExpired(command, 30, output=b"partial probe\n", stderr=b"timeout detail\n")

    monkeypatch.setattr(verifier.subprocess, "run", fail)
    cluster = verifier.Cluster("kubectl", "chosen", "app")
    with pytest.raises(subprocess.TimeoutExpired):
        cluster.run("get", "endpoints")
    assert cluster.observations[0]["stdout"] == "partial probe\n"
    assert cluster.observations[0]["stderr"] == "timeout detail\n"


@pytest.mark.parametrize("gate", [None, "missing-chart", "render-failure"])
def test_main_persists_exact_score_json_on_success_and_hard_gates(tmp_path, monkeypatch, capsys, gate):
    monkeypatch.setattr(verifier, "parse_args", lambda: SimpleNamespace(output_dir=str(tmp_path / "output"), eval_tmp_dir=str(tmp_path), helm_version="unused", kubectl="task-kubectl", terraform="task-terraform"))
    monkeypatch.setattr(verifier, "hard_gate_reason", lambda path: "missing-chart" if gate == "missing-chart" else None)
    monkeypatch.setattr(verifier, "ensure_helm", lambda *args: "unused")
    monkeypatch.setattr(verifier, "render_helm", lambda *args: (gate != "render-failure", [], "render failed"))
    monkeypatch.setattr(verifier, "score_cluster", lambda *args: {name: {"score": 0, "detail": {"reason": "control"}} for name in ("pods", "services", "health", "hpa", "netpol")})
    assert verifier.main() == 0
    assert (tmp_path / "verification.json").read_text() == capsys.readouterr().out
