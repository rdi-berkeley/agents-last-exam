from textwrap import dedent

import pytest
import yaml

from tasks.computing_math.k8s_migration_1.scripts import verify_k8s_migration as verifier


@pytest.fixture
def workflow_file(tmp_path):
    path = tmp_path / ".github" / "workflows" / "deploy.yml"
    path.parent.mkdir(parents=True)
    return path


def score_job(workflow_file, steps, runner="ubuntu-latest", **settings):
    workflow = {"jobs": {"release": {"runs-on": runner, "steps": steps, **settings}}}
    workflow_file.write_text(yaml.safe_dump(workflow))
    return verifier.score_cicd(workflow_file)


OPERATIONS = "helm upgrade --install webapp ./chart\nkubectl rollout status deployment/backend\n"


@pytest.mark.parametrize("runner", ["ubuntu-latest", "windows-latest", "macos-14", ["ubuntu-24.04"], {"labels": "ubuntu-latest"}])
def test_hosted_jobs_without_any_cluster_setup_lose_both_stage_credits(workflow_file, runner):
    score, detail = score_job(workflow_file, [
        {"uses": "actions/checkout@v4"},
        {"uses": "azure/setup-helm@v4"},
        {"run": "docker build .\npytest\ntrivy image webapp\n" + OPERATIONS},
    ], runner)
    assert score == 0.6
    assert detail["build"] and detail["test"] and detail["security"]
    assert not detail["deploy"] and not detail["verify"]
    assert detail["deploy_missing_cluster_jobs"] == ["release"]
    assert detail["verify_missing_cluster_jobs"] == ["release"]


@pytest.mark.parametrize("runner", ["self-hosted", ["self-hosted", "linux", "webapp-minikube"], "local-minikube", {"group": "dedicated", "labels": "self-hosted"}])
def test_local_runner_keeps_existing_cluster_prerequisite(workflow_file, runner):
    _, detail = score_job(workflow_file, [{"run": OPERATIONS}], runner)
    assert detail["deploy"] and detail["verify"]


@pytest.mark.parametrize("setup", [
    "minikube start --driver=docker",
    "kind create cluster --name ci",
    "k3d cluster create ci",
    "aws eks update-kubeconfig --name production --region us-east-1",
    "az aks get-credentials --resource-group production --name webapp",
    "gcloud container clusters get-credentials webapp --zone us-central1-a",
    "oc login https://cluster.example --token=$TOKEN",
    "mkdir -p ~/.kube\nprintf '%s' \"$REMOTE_KUBECONFIG\" > ~/.kube/config",
    "cp downloaded/remote-config ~/.kube/config",
    "kubectl config set-cluster remote --server=https://cluster.example",
])
def test_hosted_job_has_its_own_setup_before_operations(workflow_file, setup):
    _, detail = score_job(workflow_file, [{"run": setup}, {"run": OPERATIONS}])
    assert detail["deploy"] and detail["verify"]


@pytest.mark.parametrize("action", [
    "azure/k8s-set-context@v4", "azure/aks-set-context@v4",
    "google-github-actions/get-gke-credentials@v2", "helm/kind-action@v1",
    "medyagh/setup-minikube@latest", "engineerd/setup-kind@v0.6.2",
])
def test_cluster_setup_actions_are_distinct_from_tool_installation(workflow_file, action):
    _, detail = score_job(workflow_file, [{"uses": action}, {"run": OPERATIONS}])
    assert detail["deploy"] and detail["verify"]


@pytest.mark.parametrize("step", [
    {"uses": "azure/setup-kubectl@v4"},
    {"uses": "actions/setup-node@v4"},
    {"uses": "aws-actions/configure-aws-credentials@v4"},
    {"uses": "actions/download-artifact@v4", "with": {"name": "images"}},
    {"run": "echo 'minikube start; aws eks update-kubeconfig --name webapp'"},
    {"run": "minikube status"},
    {"run": "kubectl config current-context"},
    {"run": "npm ci"},
])
def test_unrelated_actions_and_commands_do_not_supply_cluster_access(workflow_file, step):
    _, detail = score_job(workflow_file, [step, {"run": OPERATIONS}])
    assert not detail["deploy"] and not detail["verify"]


def test_setup_after_operations_does_not_retroactively_grant_credit(workflow_file):
    _, detail = score_job(workflow_file, [{"run": OPERATIONS}, {"run": "minikube start"}])
    assert not detail["deploy"] and not detail["verify"]


def test_installing_cluster_after_failed_deploy_does_not_populate_it(workflow_file):
    _, detail = score_job(workflow_file, [{"run": "helm upgrade --install webapp ./chart\nminikube start\nkubectl rollout status deployment/backend"}])
    assert not detail["deploy"] and not detail["verify"]
    assert detail["verify_empty_cluster_jobs"] == ["release"]


def test_step_environment_does_not_leak_into_later_step(workflow_file):
    _, detail = score_job(workflow_file, [
        {"run": "helm upgrade --install webapp ./chart", "env": {"KUBECONFIG": "/configured/remote"}},
        {"run": "kubectl rollout status deployment/backend"},
    ])
    assert detail["deploy"] and not detail["verify"]


def test_explicit_remote_job_configuration_is_preserved(workflow_file):
    _, detail = score_job(workflow_file, [{"run": OPERATIONS}], env={"KUBECONFIG": "/configured/remote"})
    assert detail["deploy"] and detail["verify"]


def test_inline_kubeconfig_is_scoped_to_its_command(workflow_file):
    _, detail = score_job(workflow_file, [{"run": "KUBECONFIG=/configured/remote helm upgrade --install webapp ./chart\nkubectl rollout status deployment/backend"}])
    assert detail["deploy"] and not detail["verify"]


def test_exported_kubeconfig_is_scoped_to_its_run_step(workflow_file):
    _, detail = score_job(workflow_file, [
        {"run": "export KUBECONFIG=/configured/remote\nhelm upgrade --install webapp ./chart"},
        {"run": "kubectl rollout status deployment/backend"},
    ])
    assert detail["deploy"] and not detail["verify"]


def test_explicit_remote_flags_are_scoped_to_their_commands(workflow_file):
    _, detail = score_job(workflow_file, [{"run": "helm upgrade --install webapp ./chart --kubeconfig=/configured/remote\nkubectl rollout status deployment/backend"}])
    assert detail["deploy"] and not detail["verify"]
    _, detail = score_job(workflow_file, [{"run": "helm upgrade --install webapp ./chart --kube-apiserver=https://cluster.example\nkubectl --kubeconfig=/configured/remote rollout status deployment/backend"}])
    assert detail["deploy"] and detail["verify"]


def test_dependency_alone_does_not_transfer_local_cluster(workflow_file):
    workflow_file.write_text(yaml.safe_dump({"jobs": {
        "create": {"runs-on": "ubuntu-latest", "steps": [{"run": "minikube start"}]},
        "deploy": {"runs-on": "ubuntu-latest", "needs": "create", "steps": [{"run": "helm upgrade --install webapp ./chart"}]},
        "verify": {"runs-on": "ubuntu-latest", "needs": "deploy", "steps": [{"run": "kubectl rollout status deployment/backend"}]},
    }}))
    _, detail = verifier.score_cicd(workflow_file)
    assert not detail["deploy"] and not detail["verify"]


def test_remote_kubeconfig_artifact_is_explicitly_installed_in_each_job(workflow_file):
    steps = [
        {"uses": "actions/download-artifact@v4", "with": {"name": "remote-kubeconfig", "path": "downloaded"}},
        {"run": "mkdir -p ~/.kube\ncp downloaded/config ~/.kube/config"},
        {"run": OPERATIONS},
    ]
    _, detail = score_job(workflow_file, steps)
    assert detail["deploy"] and detail["verify"]


@pytest.mark.parametrize("invocation", ["bash scripts/connect.sh", "./scripts/connect.sh", ". scripts/connect.sh", "source scripts/connect.sh"])
def test_actual_submitted_setup_script_is_read_without_execution(workflow_file, tmp_path, monkeypatch, invocation):
    script = tmp_path / "scripts" / "connect.sh"
    script.parent.mkdir()
    script.write_text("aws eks update-kubeconfig --name webapp\ntouch must-not-exist\n")

    def reject_execution(*args, **kwargs):
        pytest.fail("Submitted setup must only be parsed")

    monkeypatch.setattr(verifier.subprocess, "run", reject_execution)
    _, detail = score_job(workflow_file, [{"run": invocation}, {"run": OPERATIONS}])
    assert detail["deploy"] and detail["verify"]
    assert not (tmp_path / "must-not-exist").exists()


def test_setup_script_can_contain_the_deploy_and_verify_commands(workflow_file, tmp_path):
    script = tmp_path / "release.sh"
    script.write_text("minikube start --driver=docker\n" + OPERATIONS)
    _, detail = score_job(workflow_file, [{"run": "bash release.sh"}])
    assert detail["deploy"] and detail["verify"]


def test_shell_script_without_extension_can_supply_setup(workflow_file, tmp_path):
    (tmp_path / "connect").write_text("aws eks update-kubeconfig --name webapp\n")
    _, detail = score_job(workflow_file, [{"run": "bash connect\n" + OPERATIONS}])
    assert detail["deploy"] and detail["verify"]


def test_inline_shell_can_supply_setup_without_execution(workflow_file):
    _, detail = score_job(workflow_file, [{"run": "bash -c 'minikube start --driver=docker'\n" + OPERATIONS}])
    assert detail["deploy"] and detail["verify"]


def test_script_working_directory_and_bounded_nested_setup(workflow_file, tmp_path):
    directory = tmp_path / "scripts"
    directory.mkdir()
    (directory / "setup.sh").write_text("bash remote.sh\n")
    (directory / "remote.sh").write_text("gcloud container clusters get-credentials webapp --zone us-central1-a\n")
    _, detail = score_job(workflow_file, [{"working-directory": "scripts", "run": "bash setup.sh\n" + OPERATIONS}])
    assert detail["deploy"] and detail["verify"]


@pytest.mark.parametrize("contents", ["echo 'minikube start'\n", "bash setup.sh\n", "echo no-setup\n"])
def test_unconfigured_or_recursive_scripts_cannot_earn_access(workflow_file, tmp_path, contents):
    (tmp_path / "setup.sh").write_text(contents)
    _, detail = score_job(workflow_file, [{"run": "bash setup.sh\n" + OPERATIONS}])
    assert not detail["deploy"] and not detail["verify"]


def test_missing_script_name_is_not_itself_setup_evidence(workflow_file):
    _, detail = score_job(workflow_file, [{"run": "bash setup-kubernetes.sh\n" + OPERATIONS}])
    assert not detail["deploy"] and not detail["verify"]


def test_script_reads_do_not_escape_submitted_tree(workflow_file, tmp_path):
    external = tmp_path.parent / (tmp_path.name + "-external.sh")
    external.write_text("minikube start\n")
    (tmp_path / "setup.sh").symlink_to(external)
    try:
        _, detail = score_job(workflow_file, [{"run": "bash setup.sh\n" + OPERATIONS}])
        assert not detail["deploy"] and not detail["verify"]
    finally:
        external.unlink()


def test_earlier_kimi_shape_preserves_deploy_but_rejects_fresh_verify(workflow_file):
    workflow_file.write_text(dedent("""\
        jobs:
          deploy:
            runs-on: ubuntu-latest
            steps:
              - run: |
                  minikube start --driver=docker
                  docker build -t webapp:latest .
                  minikube image load webapp:latest
                  helm upgrade --install webapp ./chart
          verify:
            runs-on: ubuntu-latest
            needs: deploy
            steps:
              - uses: actions/checkout@v4
              - run: kubectl rollout status deployment/backend
    """))
    _, detail = verifier.score_cicd(workflow_file)
    assert detail["deploy"] and not detail["verify"]


@pytest.mark.parametrize("setup", [
    {"uses": "medyagh/setup-minikube@latest"},
    {"uses": "helm/kind-action@v1"},
    {"run": "minikube start"},
    {"run": "kind create cluster"},
])
def test_new_local_cluster_in_separate_verify_job_is_empty(workflow_file, setup):
    workflow_file.write_text(yaml.safe_dump({"jobs": {
        "deploy": {"runs-on": "ubuntu-latest", "steps": [setup, {"run": OPERATIONS}]},
        "verify": {"runs-on": "ubuntu-latest", "needs": "deploy", "steps": [setup, {"run": "kubectl rollout status deployment/backend"}]},
    }}))
    _, detail = verifier.score_cicd(workflow_file)
    assert detail["deploy"]
    assert detail["verify"]
    workflow = yaml.safe_load(workflow_file.read_text())
    workflow["jobs"]["deploy"]["steps"][-1]["run"] = "helm upgrade --install webapp ./chart"
    workflow_file.write_text(yaml.safe_dump(workflow))
    _, detail = verifier.score_cicd(workflow_file)
    assert detail["deploy"] and not detail["verify"]
    assert detail["verify_empty_cluster_jobs"] == ["verify"]


@pytest.mark.parametrize("deployment", ["helm upgrade --install webapp ./chart", "kubectl apply -f manifests/", "kubectl apply -k overlays/test"])
def test_verify_job_can_redeploy_to_its_own_new_cluster(workflow_file, deployment):
    _, detail = score_job(workflow_file, [{"run": "minikube start\n" + deployment + "\nkubectl rollout status deployment/backend"}])
    assert detail["verify"]


def test_remote_verify_without_local_creation_remains_valid(workflow_file):
    _, detail = score_job(workflow_file, [{"run": "aws eks update-kubeconfig --name deployed\nkubectl rollout status deployment/backend"}])
    assert detail["verify"]


@pytest.mark.parametrize("access", [
    {"run": "aws eks update-kubeconfig --name deployed"},
    {"uses": "google-github-actions/get-gke-credentials@v2"},
])
def test_switching_to_existing_remote_cluster_after_local_setup(workflow_file, access):
    _, detail = score_job(workflow_file, [{"run": "minikube start"}, access, {"run": "kubectl rollout status deployment/backend"}])
    assert detail["verify"]


def test_explicit_remote_verify_environment_is_not_assumed_local(workflow_file):
    _, detail = score_job(workflow_file, [{"run": "minikube start"}, {
        "run": "kubectl rollout status deployment/backend", "env": {"KUBECONFIG": "/configured/remote"},
    }])
    assert detail["verify"]
