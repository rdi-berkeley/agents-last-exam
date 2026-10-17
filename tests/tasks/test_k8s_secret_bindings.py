from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tasks.computing_math.k8s_migration_1.scripts import verify_k8s_migration as verifier


def resources(container):
    return [
        {"kind": "Deployment", "metadata": {"name": "backend"},
         "spec": {"template": {"spec": {"containers": [container]}}}},
        {"kind": "Secret", "metadata": {"name": "credentials"},
         "data": {"DB_PASSWORD": "Zml4dHVyZQ==", "SECRET_KEY": "Zml4dHVyZQ==", "API_KEY": "Zml4dHVyZQ=="}},
        {"kind": "ConfigMap", "metadata": {"name": "plaintext"},
         "data": {"DB_PASSWORD": "fixture"}},
    ]


def secret_score(documents):
    return verifier.score_config_correctness(documents)[1]["secret_key_ref_for_sensitive"]


@pytest.mark.parametrize("field", ["DB_PASSWORD", "SECRET_KEY", "API_KEY"])
def test_explicit_and_bulk_secret_bindings_are_equivalent(field):
    bulk = resources({"envFrom": [{"secretRef": {"name": "credentials"}}]})
    explicit = resources({"env": [{"name": field, "valueFrom": {
        "secretKeyRef": {"name": "credentials", "key": field}}}]})
    assert secret_score(bulk) and secret_score(explicit)


@pytest.mark.parametrize("explicit", [True, False])
@pytest.mark.parametrize("defect", ["missing", "wrong_namespace", "wrong_key"])
def test_secret_must_resolve_to_delivered_resource(explicit, defect):
    documents = resources({"env": [{"name": "DB_PASSWORD", "valueFrom": {
        "secretKeyRef": {"name": "credentials", "key": "DB_PASSWORD"}}}]} if explicit else
        {"envFrom": [{"secretRef": {"name": "credentials"}}]})
    if defect == "missing":
        documents.pop(1)
    elif defect == "wrong_namespace":
        documents[1]["metadata"]["namespace"] = "elsewhere"
    else:
        documents[1]["data"] = {"unrelated": "Zml4dHVyZQ=="}
    assert not secret_score(documents)


def test_explicit_environment_overrides_bulk_secret():
    documents = resources({"envFrom": [{"secretRef": {"name": "credentials"}}],
                           "env": [{"name": "API_KEY", "value": "fixture"}]})
    assert not secret_score(documents)


@pytest.mark.parametrize("secret_last", [True, False])
def test_bulk_source_precedence_is_respected(secret_last):
    sources = [{"configMapRef": {"name": "plaintext"}}, {"secretRef": {"name": "credentials"}}]
    if not secret_last:
        sources.reverse()
    assert secret_score(resources({"envFrom": sources})) == secret_last


def test_prefix_changes_actual_environment_names():
    documents = resources({"envFrom": [{"secretRef": {"name": "credentials"}, "prefix": "OTHER_"}]})
    assert not secret_score(documents)
    documents[1]["data"] = {"PASSWORD": "Zml4dHVyZQ=="}
    documents[0]["spec"]["template"]["spec"]["containers"][0]["envFrom"][0]["prefix"] = "DB_"
    assert secret_score(documents)


def hpa_payload(value):
    metric = {"type": ""} if value is None else {
        "type": "Resource", "resource": {"name": "cpu", "current": {"averageUtilization": value}}
    }
    return {"items": [{"spec": {"scaleTargetRef": {"kind": "Deployment", "name": "backend"}},
                       "status": {"currentMetrics": [metric]}}]}


def test_hpa_accepts_measurement_after_initial_readiness_delay(monkeypatch):
    cluster = SimpleNamespace(get=Mock(side_effect=[hpa_payload(None), hpa_payload(None), hpa_payload(0)]))
    sleep = Mock()
    monkeypatch.setattr(verifier.time, "sleep", sleep)
    score, detail = verifier.score_hpa_live(cluster, wait_seconds=10)
    assert score == 1 and detail["cpu_utilization"] == [0]
    assert cluster.get.call_count == 3 and sleep.call_count == 2


def test_hpa_never_receiving_metrics_still_fails_with_bounded_wait(monkeypatch):
    cluster = SimpleNamespace(get=Mock(return_value=hpa_payload(None)))
    sleep = Mock()
    monkeypatch.setattr(verifier.time, "sleep", sleep)
    assert verifier.score_hpa_live(cluster, wait_seconds=10)[0] == 0
    assert cluster.get.call_count == 3 and sleep.call_count == 2


def test_hpa_wrong_workload_does_not_gain_credit_or_wait(monkeypatch):
    payload = deepcopy(hpa_payload(10))
    payload["items"][0]["spec"]["scaleTargetRef"]["name"] = "unrelated"
    cluster = SimpleNamespace(get=Mock(return_value=payload))
    sleep = Mock()
    monkeypatch.setattr(verifier.time, "sleep", sleep)
    assert verifier.score_hpa_live(cluster)[0] == 0
    sleep.assert_not_called()
