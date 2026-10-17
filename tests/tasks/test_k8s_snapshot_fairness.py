from textwrap import dedent

import pytest

from tasks.computing_math.k8s_migration_1.scripts import verify_k8s_migration as verifier


CODEX_SNAPSHOT = dedent("""\
    Captured at: 2026-10-11T05:19:10Z
    Command: software/kubectl -n webapp get pods -l app=webapp -o wide

    NAME                         READY   STATUS    RESTARTS   AGE     IP             NODE
    backend-7c5db5654d-gsfgv     1/1     Running   0          5m10s   10.244.120.71  minikube
    backend-7c5db5654d-mnc96     1/1     Running   0          5m10s   10.244.120.72  minikube
    db-0                         1/1     Running   0          2m43s   10.244.120.76  minikube
    frontend-74d64ccf5c-2c9v2    1/1     Running   0          5m10s   10.244.120.73  minikube
    frontend-74d64ccf5c-rw72z    1/1     Running   0          5m10s   10.244.120.74  minikube
""")

RESOURCE_TABLES = dedent("""\
    NAME                        TYPE        CLUSTER-IP       EXTERNAL-IP   PORT(S)        AGE
    service/backend-service     ClusterIP   10.103.142.234   <none>        5000/TCP       4m
    service/db-service          ClusterIP   None             <none>        5432/TCP       4m
    service/frontend-service    NodePort    10.106.89.51      <none>        80:30080/TCP   4m

    NAME                        READY   UP-TO-DATE   AVAILABLE   AGE
    deployment.apps/backend     2/2     2            2           4m
    deployment.apps/frontend    2/2     2            2           4m

    NAME                                   DESIRED   CURRENT   READY   AGE
    replicaset.apps/backend-765f5fd98       2         2         2       4m

    NAME                   READY   AGE
    statefulset.apps/db    1/1     4m

    NAME                      REFERENCE            TARGETS   MINPODS   MAXPODS   REPLICAS   AGE
    backend-hpa               Deployment/backend   0%/70%    2         5         2          4m

    NAME                                STATUS   VOLUME      CAPACITY   ACCESS MODES
    persistentvolumeclaim/db-data-db-0   Bound    pvc-123     5Gi        RWO

    NAME                                       POD-SELECTOR         AGE
    networkpolicy.networking.k8s.io/db-policy   app=webapp,tier=db   4m

    NAME                           CLASS   HOSTS          ADDRESS        PORTS   AGE
    ingress.networking.k8s.io/app   nginx   webapp.local   192.168.49.2   80      4m

    $ kubectl top pods -l app=webapp
    NAME                         CPU(cores)   MEMORY(bytes)
    backend-765f5fd98-sqdbn      1m           49Mi
    db-0                         2m           70Mi
""")

KIMI_SNAPSHOT = dedent("""\
    $ kubectl get pods -l app=webapp -o wide
    NAME                         READY   STATUS    RESTARTS   AGE   IP             NODE
    backend-765f5fd98-sqdbn      1/1     Running   0          4m    10.244.120.72  minikube
    backend-765f5fd98-v86vr      1/1     Running   0          4m    10.244.120.74  minikube
    db-0                         1/1     Running   0          4m    10.244.120.75  minikube
    frontend-5f5df8478c-dhcfp    1/1     Running   0          4m    10.244.120.71  minikube
    frontend-5f5df8478c-k47s9    1/1     Running   0          4m    10.244.120.73  minikube

    $ kubectl get all -l app=webapp
    NAME                             READY   STATUS    RESTARTS   AGE
    pod/backend-765f5fd98-sqdbn      1/1     Running   0          4m
    pod/backend-765f5fd98-v86vr      1/1     Running   0          4m
    pod/db-0                         1/1     Running   0          4m
    pod/frontend-5f5df8478c-dhcfp    1/1     Running   0          4m
    pod/frontend-5f5df8478c-k47s9    1/1     Running   0          4m
""") + RESOURCE_TABLES


@pytest.mark.parametrize(
    "snapshot,expected_rows",
    [(CODEX_SNAPSHOT, 5), (KIMI_SNAPSHOT, 10)],
    ids=["codex-capture-preamble", "kimi-get-all-and-metrics"],
)
def test_historical_snapshot_formats_accept_healthy_pods(tmp_path, snapshot, expected_rows):
    path = tmp_path / "pods.txt"
    path.write_text(snapshot)

    score, detail = verifier.score_pods_snapshot(path)

    assert score == 1.0
    assert detail["rows"] == expected_rows
    assert detail["all_running"] is True


@pytest.mark.parametrize(
    "snapshot,expected_rows",
    [
        ("NAME READY STATUS RESTARTS AGE\nbackend-123 2/2 Running 1 (20s ago) 5m\n", 1),
        (
            "NAMESPACE NAME READY STATUS RESTARTS AGE\n"
            "webapp backend-123 1/1 Running 0 5m\n"
            "webapp db-0 1/1 Running 0 5m\n",
            2,
        ),
        (
            "NAMESPACE NAME READY STATUS RESTARTS AGE\n"
            "webapp pod/backend-123 1/1 Running 0 5m\n"
            "NAMESPACE NAME TYPE CLUSTER-IP EXTERNAL-IP PORT(S) AGE\n"
            "webapp service/backend-service ClusterIP 10.0.0.1 <none> 5000/TCP 5m\n"
            "NAME READY STATUS RESTARTS AGE\n"
            "pod/db-0 1/1 Running 0 5m\n",
            2,
        ),
        ("backend-123 1/1 Running 0 5m\nwebapp pod/db-0 1/1 Running 0 5m\n", 2),
        (
            "  # Capture of application pods\n"
            "  Captured at: 2026-10-11T05:19:10Z\n"
            "  Command: kubectl get pods\n"
            "  All application pods are ready.\n"
            "  NAME\tREADY\tSTATUS\tRESTARTS\tAGE\n"
            "  backend-123\t1/1\tRunning\t0\t5m\n"
            "  Additional resource metrics follow.\n" + RESOURCE_TABLES,
            1,
        ),
    ],
    ids=["sidecar-and-restarts", "namespaces", "mixed-tables", "no-headers", "notes-and-tabs"],
)
def test_supported_pod_table_layouts(tmp_path, snapshot, expected_rows):
    path = tmp_path / "pods.txt"
    path.write_text(snapshot)

    score, detail = verifier.score_pods_snapshot(path)

    assert score == 1.0
    assert detail["rows"] == expected_rows


@pytest.mark.parametrize("prefix", ["", "webapp ", "pod/", "webapp pod/"])
@pytest.mark.parametrize(
    "ready,status",
    [
        ("0/1", "Pending"),
        ("0/1", "CrashLoopBackOff"),
        ("1/1", "Terminating"),
        ("0/1", "ContainerCreating"),
        ("0/1", "ImagePullBackOff"),
        ("0/1", "Init:0/2"),
        ("0/1", "Init:CrashLoopBackOff"),
        ("1/1", "Unknown"),
    ],
)
def test_nonrunning_pod_is_not_hidden_by_good_rows_or_noise(
    tmp_path, prefix, ready, status
):
    path = tmp_path / "pods.txt"
    path.write_text(
        CODEX_SNAPSHOT + RESOURCE_TABLES + f"{prefix}broken-pod {ready} {status} 0 5m\n"
    )

    score, detail = verifier.score_pods_snapshot(path)

    assert score == 0.0
    assert detail["rows"] == 6
    assert detail["all_running"] is False


@pytest.mark.parametrize("prefix", ["", "webapp ", "pod/", "webapp pod/"])
@pytest.mark.parametrize("ready", ["0/1", "1/2", "0/0", "2/1"])
def test_readiness_is_diagnostic_only_for_running_pods(tmp_path, prefix, ready):
    path = tmp_path / "pods.txt"
    path.write_text(
        CODEX_SNAPSHOT + RESOURCE_TABLES + f"{prefix}backend-123 {ready} Running 0 5m\n"
    )

    score, detail = verifier.score_pods_snapshot(path)

    assert score == 1.0
    assert detail["rows"] == 6
    assert detail["all_running"] is True
    assert detail["all_ready"] is False


@pytest.mark.parametrize(
    "snapshot",
    [
        "",
        "\n\n",
        "NAME READY STATUS RESTARTS AGE\n",
        "NAMESPACE NAME READY STATUS RESTARTS AGE\n",
        "No resources found in webapp namespace.\n",
        "Captured at: 2026-10-11T05:19:10Z\nCommand: kubectl get pods\nAll pods Running.\n",
        RESOURCE_TABLES,
        "NAME READY UP-TO-DATE AVAILABLE AGE\nbackend 2/2 2 2 5m\n",
        "NAME READY AGE\ndb 1/1 5m\n",
        "NAME CPU(cores) MEMORY(bytes)\nbackend-123 1m 49Mi\n",
    ],
    ids=[
        "empty", "whitespace", "header", "namespace-header", "no-resources", "notes",
        "other-resources", "deployment", "statefulset", "metrics",
    ],
)
def test_snapshot_without_pod_records_cannot_pass(tmp_path, snapshot):
    path = tmp_path / "pods.txt"
    path.write_text(snapshot)

    score, detail = verifier.score_pods_snapshot(path)

    assert score == 0.0
    assert detail == {"file": True, "rows": 0}


def test_missing_snapshot_cannot_pass(tmp_path):
    score, detail = verifier.score_pods_snapshot(tmp_path / "pods.txt")

    assert score == 0.0
    assert detail == {"file": False}


def test_successful_admission_jobs_do_not_fail_healthy_application_pods(tmp_path):
    path = tmp_path / "pods.txt"
    jobs = "ingress-nginx ingress-nginx-admission-create-wk8gn 0/1 Completed 2 17m\ningress-nginx ingress-nginx-admission-patch-whz8p 0/1 Completed 3 17m\n"
    path.write_text(CODEX_SNAPSHOT + jobs)
    score, detail = verifier.score_pods_snapshot(path)
    assert score == 1
    assert detail["completed"] == 2
    path.write_text(jobs)
    assert verifier.score_pods_snapshot(path)[0] == 0


@pytest.mark.parametrize("command", ['"${TOOLS}/docker" build .', '"$TOOLS/docker" build .', '/task/software/docker build .', 'docker build .'])
def test_ci_shell_quoting_preserves_real_commands(tmp_path, command):
    path = tmp_path / "deploy.yml"
    path.write_text('jobs:\n  deploy:\n    steps:\n      - run: |\n          ' + command + '\n          pytest\n          "$TOOLS/trivy" image webapp\n          "$TOOLS/helm" upgrade --install webapp ./chart\n          "$TOOLS/kubectl" -n webapp rollout status deployment/backend\n')
    score, detail = verifier.score_cicd(path)
    assert score == 1
    assert all(detail.values())


def test_ci_descriptions_comments_and_echoes_do_not_count_as_commands(tmp_path):
    path = tmp_path / "deploy.yml"
    path.write_text('jobs:\n  build:\n    steps:\n      - name: docker build pytest npm test trivy helm upgrade rollout status\n        run: |\n          # docker build .\n          echo "docker build; pytest; trivy; helm upgrade; kubectl rollout status"\n')
    assert verifier.score_cicd(path)[0] == 0


@pytest.mark.parametrize("command", ['"$TOOLS/docker" run --rm "backend:$TAG" pytest -q', '"$TOOLS/docker" run --rm -e CI=true "frontend:$TAG" npm test -- --runInBand', 'python3 -m pytest'])
def test_ci_recognizes_container_tests_and_real_trivy_action(tmp_path, command):
    path = tmp_path / "deploy.yml"
    path.write_text('jobs:\n  test:\n    steps:\n      - run: |\n          ' + command + '\n      - uses: aquasecurity/trivy-action@master\n        with:\n          image-ref: webapp:latest\n')
    _, detail = verifier.score_cicd(path)
    assert detail["test"] and detail["security"]
    assert not detail["build"]


def test_echo_in_container_does_not_count_as_test_execution(tmp_path):
    path = tmp_path / "deploy.yml"
    path.write_text('jobs:\n  test:\n    steps:\n      - run: docker run --rm image echo "pytest"\n')
    assert not verifier.score_cicd(path)[1]["test"]


@pytest.mark.parametrize("runner,setup,expected", [
    ("ubuntu-latest", "", False),
    ("ubuntu-24.04", "", False),
    ("[self-hosted, linux, dedicated-minikube]", "", True),
    ("ubuntu-latest", "      - uses: azure/k8s-set-context@v4\n", True),
    ("ubuntu-latest", "      - run: kubectl config set-cluster remote --server=https://cluster.example\n", True),
    ("ubuntu-latest", "      - run: kubectl --kubeconfig=/configured/remote rollout status deployment/backend\n", True),
    ("ubuntu-latest", "      - run: kubectl rollout status deployment/backend\n        env:\n          KUBECONFIG: /configured/remote\n", True),
])
def test_fresh_verify_job_cannot_inherit_another_jobs_local_cluster(tmp_path, runner, setup, expected):
    path = tmp_path / "deploy.yml"
    path.write_text(
        "jobs:\n"
        "  deploy:\n"
        "    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - run: |\n"
        "          minikube start --driver=docker\n"
        "          helm upgrade --install webapp ./chart\n"
        "  verify:\n"
        "    needs: deploy\n"
        f"    runs-on: {runner}\n"
        "    steps:\n"
        "      - uses: actions/checkout@v4\n"
        + setup +
        "      - run: kubectl rollout status deployment/backend\n"
    )
    _, detail = verifier.score_cicd(path)
    assert detail["verify"] is expected
    assert detail["deploy"]
    if not expected:
        assert detail["verify_missing_cluster_jobs"] == ["verify"]


def test_deploy_and_verify_in_same_hosted_job_keep_cluster_state(tmp_path):
    path = tmp_path / "deploy.yml"
    path.write_text(
        "jobs:\n"
        "  deployment:\n"
        "    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - run: |\n"
        "          minikube start --driver=docker\n"
        "          helm upgrade --install webapp ./chart\n"
        "          kubectl rollout status deployment/backend\n"
    )
    _, detail = verifier.score_cicd(path)
    assert detail["deploy"] and detail["verify"]
