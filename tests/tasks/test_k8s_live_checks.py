import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from tasks.computing_math.k8s_migration_1 import main
from tasks.computing_math.k8s_migration_1.scripts import verify_k8s_migration as verifier


def test_discovery_and_queries_use_task_wrapper_context_and_namespace(monkeypatch):
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        if command[1:] == ["config", "current-context"]:
            return SimpleNamespace(returncode=0, stdout="minikube-task\n", stderr="")
        if "--all-namespaces" in command:
            payload = {"items": [{"metadata": {"namespace": "chosen", "name": role}} for role in ("frontend", "backend")]}
        else:
            payload = {"items": []}
        return SimpleNamespace(returncode=0, stdout=json.dumps(payload), stderr="")

    monkeypatch.setattr(verifier.subprocess, "run", run)
    cluster = verifier.discover_cluster("/task/software/kubectl")
    assert cluster.namespace == "chosen"
    cluster.get("hpa")
    assert calls[-1] == ["/task/software/kubectl", "--context", "minikube-task", "--namespace", "chosen", "get", "hpa", "-o", "json"]
    assert "--context" in calls[1]


@pytest.mark.parametrize("phase,owner,expected", [("Succeeded", "Job", 1), ("Succeeded", "ReplicaSet", 0), ("Failed", "Job", 0), ("Pending", "Job", 0)])
def test_only_successful_job_pods_are_ignored(phase, owner, expected):
    items = [{"metadata": {"name": role + "-123"}, "status": {"phase": "Running"}} for role in ("frontend", "backend", "db")]
    items.append({"metadata": {"name": "hook", "ownerReferences": [{"kind": owner}]}, "status": {"phase": phase}})
    cluster = SimpleNamespace(get=lambda *args: {"items": items})
    assert verifier.score_pods_live(cluster)[0] == expected


@pytest.mark.parametrize("value,resource,target,expected", [(0, "cpu", "backend", 1), (70, "cpu", "backend", 1), (None, "cpu", "backend", 0), ("<unknown>", "cpu", "backend", 0), (20, "memory", "backend", 0), (20, "cpu", "other", 0)])
def test_hpa_requires_actual_backend_cpu_reading(value, resource, target, expected):
    cluster = SimpleNamespace(get=lambda *args: {"items": [{"spec": {"scaleTargetRef": {"kind": "Deployment", "name": target}}, "status": {"currentMetrics": [{"type": "Resource", "resource": {"name": resource, "current": {"averageUtilization": value}}}]}}]})
    assert verifier.score_hpa_live(cluster, wait_seconds=0)[0] == expected


@pytest.mark.parametrize("addresses,expected", [([{"ip": "10.0.0.1"}], 1), ([], 0)])
def test_services_need_ready_endpoints_not_just_cluster_ips(monkeypatch, addresses, expected):
    monkeypatch.setattr(verifier, "workload_services", lambda docs: {
        role: [{"service": {"metadata": {"name": role + "-service"}}, "ports": [{"target_port": 80, "name": ""}]}]
        for role in ("frontend", "backend", "db")
    })
    cluster = SimpleNamespace(get=lambda *args: {"items": [{"metadata": {"name": role + "-service"}, "subsets": [{"addresses": addresses, "ports": [{"port": 80}]}]} for role in ("frontend", "backend", "db")]})
    assert verifier.score_services_live(cluster)[0] == expected


@pytest.mark.parametrize("code,status,expected", [(200, "healthy", 1), (200, "unhealthy", 0), (503, "healthy", 0)])
def test_live_health_requires_successful_response(monkeypatch, code, status, expected):
    monkeypatch.setattr(verifier, "workload_services", lambda docs: {
        "backend": [{"service": {"metadata": {"name": "backend-service"}}, "ports": [{"port": 5000}]}]
    })
    calls = []

    def run(*args):
        calls.append(args)
        return SimpleNamespace(returncode=0, stdout=json.dumps({"code": code, "body": {"status": status}}), stderr="")

    assert verifier.score_health_live(SimpleNamespace(run=run, get=lambda *args: {"items": []}))[0] == expected
    assert calls[0][:3] == ("exec", "deployment/backend", "--")
    assert "backend-service:5000/health" in calls[0][-1]


@pytest.mark.parametrize("backend,frontend,returncode,expected", [("REACHED", "BLOCKED", 0, 1), ("BLOCKED", "BLOCKED", 0, 0), ("REACHED", "REACHED", 0, 0), ("REACHED", "BLOCKED", 1, 0), ("REACHED", "DNS_FAILURE", 0, 0)])
def test_isolation_requires_successful_positive_control_and_cleans_probes(backend, frontend, returncode, expected):
    calls = []
    pods = [{"metadata": {"name": role + "-123", "labels": {"app": "webapp", "tier": role}}, "status": {"phase": "Running"}, "spec": {"containers": [{"image": "original-backend:local"}]}} for role in ("frontend", "backend")]

    def run(*args, **kwargs):
        calls.append(args)
        marker = backend if "--labels=app=webapp,tier=backend" in args else frontend
        return SimpleNamespace(returncode=returncode, stdout=marker + "\n", stderr="")

    score, _ = verifier.score_netpol_live(SimpleNamespace(get=lambda *args: {"items": pods}, run=run))
    assert score == expected
    assert [call[0] for call in calls] == ["run", "delete", "run", "delete"]
    assert "--image=original-backend:local" in calls[0]
    assert "socket.gethostbyname" in calls[0][-1]
    assert "except socket.timeout" in calls[0][-1]


@pytest.mark.asyncio
async def test_task_evaluation_always_enables_task_local_cluster_checks():
    metadata = main.config.to_metadata()
    session = SimpleNamespace(interface=SimpleNamespace(create_dir=AsyncMock()), write_file=AsyncMock(), run_command=AsyncMock(return_value={"return_code": 0, "stdout": '{"score":1,"mode":"live"}'}))
    assert await main.evaluate(SimpleNamespace(metadata=metadata), session) == [1]
    command = session.run_command.await_args.args[0]
    assert "--kubectl " + metadata["software_entries"]["kubectl"] in command
    assert "--live-cluster" not in command
