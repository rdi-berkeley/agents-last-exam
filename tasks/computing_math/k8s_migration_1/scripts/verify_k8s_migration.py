"""Evaluator for computing_math/k8s_migration_1.

Runs entirely on the VM. Reads the agent's `output/` tree, renders the Helm
chart with a pinned `helm` binary, and emits a JSON score payload on stdout.

Scoring (weighted sum, [0.0, 1.0]):

Hard gates (set score = 0.0 if any triggered):
  - helm/webapp-chart/Chart.yaml missing or invalid YAML
  - helm/webapp-chart/values.yaml missing
  - Plaintext secrets in Helm templates (`stringData:` in any Secret manifest)
  - `helm template` fails to render

Static subset (70%):
  1. Helm completeness           (20%)  present/8 required K8s kinds after render
  2. Configuration correctness   (25%)  12-item checklist on rendered manifests
  3. Terraform validity          (10%)  native validation and 4 structural checks
  4. CI/CD completeness          (10%)  5 stages in deploy.yml
  5. Report completeness         ( 5%)  4 files in verification/

Live subset (30%): application pods Running, populated service endpoints,
healthy backend, reported HPA CPU metrics, and frontend-to-db isolation with
a successful backend-to-db control. All checks use the task-local kubectl
wrapper, a captured context, and the discovered application namespace.
"""

import argparse
import hashlib
import json
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import yaml


HELM_VERSION_DEFAULT = "v3.14.0"
HELM_TARBALL_URL = (
    "https://get.helm.sh/helm-{ver}-linux-amd64.tar.gz"
)
HELM_TARBALL_SHA256 = {
    # upstream-published sha256 for linux-amd64 tarball
    "v3.14.0": "f43e1c3387de24547506ab05d24e5309c0ce0b228c23bd8aa64e9ec4b8206651",
}

REQUIRED_HELM_KINDS = {
    "frontend_deployment": ("Deployment", "frontend"),
    "backend_deployment": ("Deployment", "backend"),
    "db_statefulset": ("StatefulSet", "db"),
    "configmap": ("ConfigMap", None),
    "secret": ("Secret", None),
    "hpa": ("HorizontalPodAutoscaler", None),
    "networkpolicy": ("NetworkPolicy", None),
    "ingress": ("Ingress", None),
}

CI_STAGES = ("build", "test", "security", "deploy", "verify")

REQUIRED_VERIFICATION_FILES = (
    "pods.txt",
    "services.txt",
    "helm-status.txt",
    "health-check.txt",
)

# Component weights (sum to 1.0)
W_HELM = 0.20
W_CONFIG = 0.25
W_TERRAFORM = 0.10
W_CI = 0.10
W_REPORT = 0.05
# Live/snapshot sub-checks weights (sum to 0.30)
W_PODS = 0.12
W_SVC = 0.045
W_HEALTH = 0.045
W_HPA = 0.045
W_NETPOL = 0.045


def log(msg: str) -> None:
    print(msg, file=sys.stderr)


# ---------------------------------------------------------------------------
# Helm bootstrap
# ---------------------------------------------------------------------------


def ensure_helm(bin_dir: Path, version: str) -> str:
    """Return absolute path to a helm binary at `version`, installing if missing."""
    existing = shutil.which("helm")
    if existing:
        return existing
    bin_dir.mkdir(parents=True, exist_ok=True)
    target = bin_dir / "helm"
    if target.exists():
        return str(target)
    url = HELM_TARBALL_URL.format(ver=version)
    expected_sha = HELM_TARBALL_SHA256.get(version)
    with tempfile.TemporaryDirectory() as td:
        tarball = Path(td) / "helm.tar.gz"
        log(f"downloading helm {version} from {url}")
        urllib.request.urlretrieve(url, tarball)
        if expected_sha is not None:
            digest = hashlib.sha256(tarball.read_bytes()).hexdigest()
            if digest != expected_sha:
                raise RuntimeError(
                    f"helm tarball sha256 mismatch: got {digest} expected {expected_sha}"
                )
        subprocess.run(
            ["tar", "-xzf", str(tarball), "-C", td],
            check=True,
        )
        src = Path(td) / "linux-amd64" / "helm"
        shutil.copy2(src, target)
        target.chmod(0o755)
    return str(target)


# ---------------------------------------------------------------------------
# Hard gates
# ---------------------------------------------------------------------------


def hard_gate_reason(chart_dir: Path) -> str | None:
    chart_yaml = chart_dir / "Chart.yaml"
    values_yaml = chart_dir / "values.yaml"
    if not chart_yaml.is_file():
        return "hard_gate: Chart.yaml missing"
    try:
        parsed = yaml.safe_load(chart_yaml.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        return f"hard_gate: Chart.yaml invalid YAML ({exc.__class__.__name__})"
    if not isinstance(parsed, dict) or not parsed.get("name"):
        return "hard_gate: Chart.yaml missing required fields"
    if not values_yaml.is_file():
        return "hard_gate: values.yaml missing"
    templates_dir = chart_dir / "templates"
    if templates_dir.is_dir():
        for tmpl in templates_dir.rglob("*.yaml"):
            text = tmpl.read_text(encoding="utf-8", errors="replace")
            if re.search(r"^\s*kind:\s*Secret\b", text, re.MULTILINE) and \
               re.search(r"^\s*stringData\s*:", text, re.MULTILINE):
                return f"hard_gate: plaintext secrets in {tmpl.name} (stringData)"
    return None


# ---------------------------------------------------------------------------
# Helm render
# ---------------------------------------------------------------------------


def render_helm(helm_bin: str, chart_dir: Path) -> tuple[bool, list[dict], str]:
    proc = subprocess.run(
        [helm_bin, "template", "webapp", str(chart_dir)],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        return False, [], proc.stderr.strip()
    docs: list[dict] = []
    for doc in yaml.safe_load_all(proc.stdout):
        if isinstance(doc, dict) and doc.get("kind"):
            docs.append(doc)
    return True, docs, ""


# ---------------------------------------------------------------------------
# Static scoring
# ---------------------------------------------------------------------------


def score_helm_completeness(docs: list[dict]) -> tuple[float, dict]:
    present = {}
    found = {
        (doc.get("kind"), (doc.get("metadata") or {}).get("name"))
        for doc in docs
    }
    kinds_by_kind = {doc.get("kind") for doc in docs}
    for label, (kind, name) in REQUIRED_HELM_KINDS.items():
        if name is None:
            present[label] = kind in kinds_by_kind
        else:
            present[label] = (kind, name) in found
    score = sum(1 for v in present.values() if v) / len(REQUIRED_HELM_KINDS)
    return score, present


def _match_resource(docs: list[dict], kind: str, name: str | None = None) -> dict | None:
    for d in docs:
        if d.get("kind") != kind:
            continue
        if name is not None and (d.get("metadata") or {}).get("name") != name:
            continue
        return d
    return None


def _container_limits(dep: dict | None) -> dict | None:
    if not dep:
        return None
    spec = dep.get("spec") or {}
    template = spec.get("template") or {}
    pod_spec = template.get("spec") or {}
    for c in pod_spec.get("containers") or []:
        res = c.get("resources") or {}
        limits = res.get("limits")
        if limits:
            return {str(k): str(v) for k, v in limits.items()}
    return None


def _container_probes(dep: dict | None) -> tuple[bool, bool]:
    if not dep:
        return False, False
    spec = dep.get("spec") or {}
    template = spec.get("template") or {}
    pod_spec = template.get("spec") or {}
    r_ok = l_ok = False
    for c in pod_spec.get("containers") or []:
        if c.get("readinessProbe"):
            r_ok = True
        if c.get("livenessProbe"):
            l_ok = True
    return r_ok, l_ok


def _all_containers_in_docs(docs: list[dict]):
    for d in docs:
        if d.get("kind") not in {"Deployment", "StatefulSet"}:
            continue
        pod_spec = ((d.get("spec") or {}).get("template") or {}).get("spec") or {}
        for c in pod_spec.get("containers") or []:
            yield d, c


def workload_services(docs: list[dict]) -> dict[str, list[dict]]:
    workloads = {
        role: _match_resource(docs, kind, role) or {}
        for role, kind in (("frontend", "Deployment"), ("backend", "Deployment"), ("db", "StatefulSet"))
    }
    templates = {role: workload.get("spec", {}).get("template", {}) for role, workload in workloads.items()}
    matches = {role: [] for role in workloads}
    for service in docs:
        if service.get("kind") != "Service":
            continue
        spec = service.get("spec") or {}
        selector = spec.get("selector") or {}
        if not selector:
            continue
        roles = [role for role, template in templates.items() if all(
            template.get("metadata", {}).get("labels", {}).get(key) == value
            for key, value in selector.items()
        )]
        if len(roles) != 1:
            continue
        role = roles[0]
        named_ports = {
            port.get("name"): port.get("containerPort")
            for container in templates[role].get("spec", {}).get("containers", [])
            for port in container.get("ports", [])
            if port.get("name") and port.get("protocol", "TCP") == "TCP"
        }
        ports = []
        for port in spec.get("ports", []):
            target = port.get("targetPort", port.get("port"))
            if isinstance(target, str):
                target = named_ports.get(target)
            if port.get("protocol", "TCP") == "TCP" and all(
                type(number) is int and 0 < number < 65536
                for number in (port.get("port"), target)
            ):
                ports.append({"port": port["port"], "target_port": target, "name": port.get("name", "")})
        if ports and service.get("metadata", {}).get("name"):
            matches[role].append({"service": service, "ports": ports})
    return matches


def score_config_correctness(docs: list[dict]) -> tuple[float, dict]:
    checks: dict[str, bool] = {}

    frontend = _match_resource(docs, "Deployment", "frontend")
    backend = _match_resource(docs, "Deployment", "backend")
    db = _match_resource(docs, "StatefulSet", "db")
    hpa = _match_resource(docs, "HorizontalPodAutoscaler")
    netpol = _match_resource(docs, "NetworkPolicy")
    ingress = _match_resource(docs, "Ingress")

    # 1. frontend limits 256Mi/500m
    lim = _container_limits(frontend)
    checks["frontend_limits_256Mi_500m"] = bool(
        lim and lim.get("memory") == "256Mi" and lim.get("cpu") == "500m"
    )

    # 2. backend limits 512Mi/1000m
    lim = _container_limits(backend)
    checks["backend_limits_512Mi_1000m"] = bool(
        lim and lim.get("memory") == "512Mi" and lim.get("cpu") in {"1000m", "1"}
    )

    # 3. readinessProbe (backend)
    r_ok, l_ok = _container_probes(backend)
    checks["backend_readiness_probe"] = r_ok
    # 4. livenessProbe (backend)
    checks["backend_liveness_probe"] = l_ok

    # 5. HPA avg 70% CPU
    hpa_70 = False
    if hpa:
        for m in (hpa.get("spec") or {}).get("metrics") or []:
            if m.get("type") == "Resource":
                res = m.get("resource") or {}
                tgt = res.get("target") or {}
                if res.get("name") == "cpu" and tgt.get("averageUtilization") == 70:
                    hpa_70 = True
                    break
    checks["hpa_70_cpu"] = hpa_70

    # 6. NetworkPolicy podSelector scoped to db tier
    np_db = False
    if netpol:
        selector = ((netpol.get("spec") or {}).get("podSelector") or {}).get(
            "matchLabels"
        ) or {}
        if selector.get("tier") == "database" or selector.get("tier") == "db":
            np_db = True
    checks["netpol_selects_db"] = np_db

    sensitive_env = {"DB_PASSWORD", "SECRET_KEY", "API_KEY", "POSTGRES_PASSWORD"}
    resources = {
        (document.get("kind"), document.get("metadata", {}).get("namespace", "default"),
         document.get("metadata", {}).get("name")): document
        for document in docs if document.get("kind") in {"Secret", "ConfigMap"}
    }
    used_secret_ref = set()
    env_value_plain = False
    for _dep, c in _all_containers_in_docs(docs):
        namespace = _dep.get("metadata", {}).get("namespace", "default")
        resolved = {}
        for source in c.get("envFrom") or []:
            for reference_field, kind in (("secretRef", "Secret"), ("configMapRef", "ConfigMap")):
                reference = source.get(reference_field) or {}
                resource = resources.get((kind, namespace, reference.get("name")), {})
                for key, value in (resource.get("data") or {}).items():
                    name = source.get("prefix", "") + key
                    if name in sensitive_env:
                        resolved[name] = kind == "Secret" and isinstance(value, str)
        for env in c.get("env") or []:
            name = env.get("name")
            if name in sensitive_env:
                vf = env.get("valueFrom") or {}
                reference = vf.get("secretKeyRef") or {}
                secret = resources.get(("Secret", namespace, reference.get("name")), {})
                resolved[name] = "value" not in env and isinstance(
                    (secret.get("data") or {}).get(reference.get("key")), str
                )
        used_secret_ref.update(name for name, valid in resolved.items() if valid)
        env_value_plain = env_value_plain or any(not valid for valid in resolved.values())
    checks["secret_key_ref_for_sensitive"] = bool(
        used_secret_ref and not env_value_plain
    )

    # 8. PVC request 5Gi (StatefulSet volumeClaimTemplate)
    pvc_5gi = False
    if db:
        for vct in (db.get("spec") or {}).get("volumeClaimTemplates") or []:
            req = (((vct.get("spec") or {}).get("resources") or {}).get(
                "requests"
            ) or {})
            if str(req.get("storage")) == "5Gi":
                pvc_5gi = True
                break
    checks["pvc_5gi"] = pvc_5gi

    # 9. replica counts: frontend=2, backend min=2 (Deployment.replicas OR HPA.minReplicas)
    frontend_replicas = (frontend or {}).get("spec", {}).get("replicas")
    backend_replicas = (backend or {}).get("spec", {}).get("replicas")
    backend_min_hpa = (hpa or {}).get("spec", {}).get("minReplicas")
    checks["replicas_counts"] = (
        frontend_replicas == 2 and (backend_replicas == 2 or backend_min_hpa == 2)
    )

    # 10. Ingress path /api
    api_path = False
    if ingress:
        for rule in (ingress.get("spec") or {}).get("rules") or []:
            paths = ((rule.get("http") or {}).get("paths") or [])
            for p in paths:
                if p.get("path") == "/api":
                    api_path = True
                    break
    checks["ingress_api_path"] = api_path

    # 11. frontend svc NodePort, backend/db ClusterIP
    services = workload_services(docs)
    svc_ok = all(
        any(item["service"]["spec"].get("type", "ClusterIP") == expected for item in services[role])
        for role, expected in (("frontend", "NodePort"), ("backend", "ClusterIP"), ("db", "ClusterIP"))
    )
    checks["service_types"] = svc_ok

    # 12. consistent labels (every workload carries app=webapp)
    labels_ok = True
    for d in docs:
        if d.get("kind") not in {
            "Deployment",
            "StatefulSet",
            "Service",
            "ConfigMap",
            "Secret",
            "HorizontalPodAutoscaler",
            "NetworkPolicy",
            "Ingress",
        }:
            continue
        labels = ((d.get("metadata") or {}).get("labels") or {})
        if labels.get("app") != "webapp":
            labels_ok = False
            break
    checks["consistent_labels"] = labels_ok

    score = sum(1 for v in checks.values() if v) / len(checks)
    return score, checks


TF_REQUIRED_ADDONS = ("ingress", "metrics-server", "storage-provisioner")


def score_terraform(tf_dir: Path) -> tuple[float, dict]:
    if not tf_dir.is_dir():
        return 0.0, {"present": False}
    all_tf = "\n".join(
        p.read_text(encoding="utf-8", errors="replace")
        for p in sorted(tf_dir.rglob("*.tf"))
    )
    subs = {
        "required_providers": bool(re.search(r"required_providers\s*{", all_tf)),
        "variable_descriptions": (
            bool(re.search(r"variable\s+\"[^\"]+\"\s*{", all_tf))
            and all_tf.count("description") >= 1
            and _variables_have_descriptions(all_tf)
        ),
        "output_blocks": bool(re.search(r"^\s*output\s+\"", all_tf, re.MULTILINE)),
        "minikube_config": (
            ("calico" in all_tf)
            and all(addon in all_tf for addon in TF_REQUIRED_ADDONS)
        ),
    }
    score = sum(1 for v in subs.values() if v) / len(subs)
    return score, subs


def _variables_have_descriptions(tf_text: str) -> bool:
    blocks = re.findall(
        r"variable\s+\"[^\"]+\"\s*{([^}]*)}",
        tf_text,
        re.DOTALL,
    )
    if not blocks:
        return False
    return all(re.search(r"\bdescription\b", b) for b in blocks)


def validate_terraform(tf_dir: Path, terraform: str, scratch: Path) -> dict:
    if not tf_dir.is_dir():
        return {"valid": False, "reason": "Terraform directory missing"}
    scratch.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="terraform-", dir=scratch) as work:
        copied = Path(work) / "configuration"
        shutil.copytree(tf_dir, copied, ignore=shutil.ignore_patterns(".terraform", "*.tfstate", "*.tfstate.*"))
        initialized = subprocess.run(
            [terraform, f"-chdir={copied}", "init", "-backend=false", "-input=false", "-no-color"],
            capture_output=True, text=True, timeout=180,
        )
        if initialized.returncode:
            diagnostic = initialized.stdout + initialized.stderr
            unavailable = ("failed to request discovery document", "could not connect to registry",
                           "timeout", "no such host", "network is unreachable", "tls handshake",
                           "502 bad gateway", "503 service unavailable", "429 too many requests")
            invalid = ("does not have a provider named", "no available releases match",
                       "invalid provider source", "invalid version constraint",
                       "invalid required_version", "unsupported terraform core version",
                       "argument or block definition required", "unclosed configuration block",
                       "invalid single-argument block definition", "duplicate required providers")
            lowered = " ".join(diagnostic.lower().split())
            if not any(marker in lowered for marker in unavailable) and any(marker in lowered for marker in invalid):
                return {"valid": False, "phase": "init", "diagnostic": diagnostic[-8192:]}
            raise RuntimeError("Terraform initialization unavailable: " + diagnostic[-2000:])
        result = subprocess.run(
            [terraform, f"-chdir={copied}", "validate", "-json"],
            capture_output=True, text=True, timeout=60,
        )
        payload = json.loads(result.stdout)
        if type(payload.get("valid")) is not bool or result.returncode not in (0, 1):
            raise RuntimeError("Terraform validation lacks a valid execution receipt")
        if payload["valid"] != (result.returncode == 0):
            raise RuntimeError("Terraform validity and exit code disagree")
        diagnostics = payload.get("diagnostics", [])
        if any("failed to load plugin schemas" in item.get("summary", "").lower() for item in diagnostics):
            raise RuntimeError("Terraform provider schema could not be loaded")
        return {"valid": payload["valid"], "phase": "validate", "diagnostics": diagnostics}


def ci_shell_commands(script: str, root: Path, directory: Path, depth: int = 0) -> list[list[str]]:
    commands = []
    script = re.sub(r"\$\{\{.*?\}\}", "__workflow_expression__", script, flags=re.DOTALL)
    for line in script.replace("\\\n", " ").splitlines():
        try:
            lexer = shlex.shlex(line, posix=True, punctuation_chars=";&|()")
            lexer.whitespace_split = True
            tokens = list(lexer)
        except ValueError:
            continue
        command = []
        for token in [*tokens, ";"]:
            if not (token and all(character in ";&|()" for character in token)):
                command.append(token)
                continue
            environment_prefix = []
            while command and (command[0] in {"sudo", "env", "then", "do"} or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*=.*", command[0])):
                prefix = command.pop(0)
                if prefix.startswith("KUBECONFIG="):
                    environment_prefix.append(prefix)
            if command:
                executable = command[0].rsplit("/", 1)[-1]
                commands.append([executable, *command[1:], *environment_prefix])
                script_name = command[0]
                shell_invocation = executable in {"bash", "sh", "source", "."}
                if shell_invocation:
                    script_name = next((argument for argument in command[1:] if not argument.startswith("-")), "")
                inline_script = executable in {"bash", "sh"} and any(argument in {"-c", "-lc", "-ec"} for argument in command[1:])
                if executable == "cd" and len(command) == 2:
                    directory = (directory / command[1]).resolve()
                if depth < 3 and inline_script:
                    commands.extend(ci_shell_commands(script_name, root, directory, depth + 1))
                elif depth < 3 and (shell_invocation or script_name.endswith(".sh")) and script_name:
                    script_name = script_name.replace("${GITHUB_WORKSPACE}/", "").replace("$GITHUB_WORKSPACE/", "")
                    if root.name == "output" and script_name.startswith("output/"):
                        script_name = script_name[len("output/"):]
                    candidate = (directory / script_name).resolve()
                    if candidate.is_relative_to(root) and candidate.is_file() and candidate.stat().st_size <= 65536:
                        commands.extend(ci_shell_commands(candidate.read_text(encoding="utf-8", errors="replace"), root, directory, depth + 1))
            command = []
    return commands


def score_cicd(ci_file: Path) -> tuple[float, dict]:
    if not ci_file.is_file():
        return 0.0, {stage: False for stage in CI_STAGES}
    try:
        workflow = yaml.safe_load(ci_file.read_text(encoding="utf-8", errors="replace"))
    except yaml.YAMLError:
        return 0.0, {stage: False for stage in CI_STAGES}
    commands = []
    actions = []
    stage_jobs = {"deploy": set(), "verify": set()}
    accessible_jobs = {"deploy": set(), "verify": set()}
    empty_cluster_jobs = set()
    jobs = workflow.get("jobs", {}) if isinstance(workflow, dict) else {}
    if not isinstance(jobs, dict):
        return 0.0, {stage: False for stage in CI_STAGES}
    root = ci_file.parent.resolve()
    if root.name == "workflows" and root.parent.name == ".github":
        root = root.parent.parent
    for job_name, job in jobs.items():
        if not isinstance(job, dict) or not isinstance(job.get("steps", []), list):
            continue
        runner = job.get("runs-on", "")
        labels = runner.get("labels", []) if isinstance(runner, dict) else runner
        labels = [labels] if isinstance(labels, str) else labels
        hosted = isinstance(labels, list) and "self-hosted" not in labels and any(
            isinstance(label, str) and label.startswith(("ubuntu-", "windows-", "macos-")) for label in labels
        )
        environments = [workflow.get("env") or {}, job.get("env") or {}]
        cluster_ready = not hosted or any(environment.get("KUBECONFIG") for environment in environments if isinstance(environment, dict))
        local_cluster_created = False
        local_workload_deployed = False
        defaults = job.get("defaults", workflow.get("defaults", {})) or {}
        default_directory = (defaults.get("run") or {}).get("working-directory", ".")
        for step in job.get("steps", []):
            if not isinstance(step, dict):
                continue
            action = str(step.get("uses", ""))
            actions.append(action)
            if action.lower().startswith(("azure/k8s-set-context@", "azure/aks-set-context@", "google-github-actions/get-gke-credentials@")):
                local_cluster_created = False
            local_cluster_created = local_cluster_created or action.lower().startswith((
                "helm/kind-action@", "medyagh/setup-minikube@", "engineerd/setup-kind@", "absaoss/k3d-action@",
            ))
            if action.lower().startswith((
                "azure/k8s-set-context@", "azure/aks-set-context@",
                "google-github-actions/get-gke-credentials@", "helm/kind-action@",
                "medyagh/setup-minikube@", "engineerd/setup-kind@", "absaoss/k3d-action@",
            )):
                cluster_ready = True
            environment = step.get("env") or {}
            step_ready = cluster_ready or (isinstance(environment, dict) and bool(environment.get("KUBECONFIG")))
            script = step.get("run", "")
            if not isinstance(script, str):
                continue
            directory = (root / str(step.get("working-directory", default_directory))).resolve()
            for command in ci_shell_commands(script, root, directory):
                commands.append(command)
                local_cluster_created = local_cluster_created or (
                    command[:2] == ["minikube", "start"]
                    or command[:3] in (["kind", "create", "cluster"], ["k3d", "cluster", "create"])
                )
                setup = (
                    command[:2] in (["minikube", "start"], ["oc", "login"])
                    or command[:3] in (["kind", "create", "cluster"], ["k3d", "cluster", "create"], ["aws", "eks", "update-kubeconfig"], ["az", "aks", "get-credentials"])
                    or command[:4] == ["gcloud", "container", "clusters", "get-credentials"]
                    or (command[:2] == ["kubectl", "config"] and "set-cluster" in command and any(token.startswith("--server") for token in command))
                )
                destination = command[-1]
                kubeconfig_destination = destination.endswith("/.kube/config") or destination in {"$KUBECONFIG", "${KUBECONFIG}"}
                if kubeconfig_destination and (
                    command[0] in {"cp", "mv", "install"}
                    or (command[0] in {"echo", "printf", "base64", "tee"} and (">" in command or command[0] == "tee"))
                ):
                    setup = True
                if setup:
                    cluster_ready = step_ready = True
                    if not (command[:2] == ["minikube", "start"] or command[:3] in (["kind", "create", "cluster"], ["k3d", "cluster", "create"])):
                        local_cluster_created = False
                if command[0] == "export" and any(token.startswith("KUBECONFIG=") for token in command[1:]):
                    step_ready = True
                explicit_access = any(
                    token in {"--kubeconfig", "--server", "--kube-apiserver"}
                    or token.startswith(("--kubeconfig=", "--server=", "--kube-apiserver=", "KUBECONFIG="))
                    for token in command
                )
                stage = None
                if command[0] == "helm" and len(command) > 1 and command[1] in {"upgrade", "install"}:
                    stage = "deploy"
                elif command[0] == "kubectl" and any(command[index:index + 2] == ["rollout", "status"] for index in range(1, len(command))):
                    stage = "verify"
                if stage is not None:
                    stage_jobs[stage].add(job_name)
                    if step_ready or explicit_access:
                        remote_environment = any(
                            isinstance(value, dict) and value.get("KUBECONFIG")
                            for value in [*environments, environment]
                        )
                        if stage == "verify" and hosted and local_cluster_created and not local_workload_deployed and not (explicit_access or remote_environment):
                            empty_cluster_jobs.add(job_name)
                        else:
                            accessible_jobs[stage].add(job_name)
                            if stage == "deploy":
                                local_workload_deployed = True
                if local_cluster_created and command[0] == "kubectl" and any(
                    token in {"apply", "create"} for token in command[1:]
                ) and any(token in {"-f", "--filename", "-k", "--kustomize"} or token.startswith(("--filename=", "--kustomize=")) for token in command[1:]):
                    local_workload_deployed = True
    for command in list(commands):
        if command[:2] != ["docker", "run"]:
            continue
        remaining = command[2:]
        while remaining and remaining[0].startswith("-"):
            option = remaining.pop(0)
            if option in {"-e", "--env", "-v", "--volume", "-w", "--workdir", "--name", "--network", "--user", "-u"} and remaining:
                remaining.pop(0)
        if len(remaining) > 1:
            image = remaining[0].split("@", 1)[0].split(":", 1)[0]
            if image in {"aquasec/trivy", "docker.io/aquasec/trivy"} and not any(
                token.split("=", 1)[0] == "--entrypoint" for token in command[2:]
            ):
                commands.append(["trivy", *remaining[1:]])
            nested = [remaining[1].rsplit("/", 1)[-1], *remaining[2:]]
            commands.append(nested)
            if nested[0] in {"sh", "bash"} and any(flag in {"-c", "-lc", "-ec"} for flag in nested[1:]) and not any(
                token.split("=", 1)[0] == "--entrypoint" for token in command[2:]
            ):
                commands.extend(ci_shell_commands(shlex.join(nested), root, root))
    results = {
        "build": any(command[:2] == ["docker", "build"] for command in commands),
        "test": any(command[0] == "pytest" or command[:2] == ["npm", "test"] or (command[0] in {"python", "python3"} and command[1:3] == ["-m", "pytest"]) for command in commands),
        "security": any(command[0] == "trivy" for command in commands) or any(action.startswith("aquasecurity/trivy-action@") for action in actions),
        "deploy": bool(accessible_jobs["deploy"]),
        "verify": bool(accessible_jobs["verify"]),
    }
    score = sum(1 for value in results.values() if value) / len(CI_STAGES)
    for stage in stage_jobs:
        missing = sorted(stage_jobs[stage] - accessible_jobs[stage] - (empty_cluster_jobs if stage == "verify" else set()))
        if missing:
            results[f"{stage}_missing_cluster_jobs"] = missing
    if empty_cluster_jobs - accessible_jobs["verify"]:
        results["verify_empty_cluster_jobs"] = sorted(empty_cluster_jobs - accessible_jobs["verify"])
    return score, results

def score_report(verification_dir: Path) -> tuple[float, dict]:
    present = {
        name: (verification_dir / name).is_file()
        for name in REQUIRED_VERIFICATION_FILES
    }
    score = sum(1 for v in present.values() if v) / len(present)
    return score, present


# ---------------------------------------------------------------------------
# Snapshot-derived live sub-checks
# ---------------------------------------------------------------------------


def score_pods_snapshot(path: Path) -> tuple[float, dict]:
    if not path.is_file():
        return 0.0, {"file": False}
    text = path.read_text(encoding="utf-8", errors="replace")
    pod_row = re.compile(
        r"^(?:[a-z0-9][a-z0-9-]*\s+)?"
        r"(?P<name>(?:pod/)?[a-z0-9][a-z0-9.-]*)\s+"
        r"(?P<ready>\d+)/(?P<total>\d+)\s+"
        r"(?P<status>[A-Za-z][A-Za-z0-9_:/-]*)(?:\s+.*)?$"
    )
    rows: list[dict] = []
    for line in text.splitlines():
        match = pod_row.fullmatch(line.strip())
        if match is None:
            continue
        ready_count = int(match["ready"])
        total_count = int(match["total"])
        rows.append({
            "name": match["name"],
            "status": match["status"],
            "ready": ready_count == total_count and total_count > 0,
        })
    if not rows:
        return 0.0, {"file": True, "rows": 0}
    active = [row for row in rows if row["status"] != "Completed"]
    all_running = bool(active) and all(row["status"] == "Running" for row in active)
    all_ready = bool(active) and all(row["ready"] for row in active)
    return (1.0 if all_running else 0.0), {
        "file": True,
        "rows": len(rows),
        "completed": len(rows) - len(active),
        "all_running": all_running,
        "all_ready": all_ready,
    }


def score_services_snapshot(path: Path) -> tuple[float, dict]:
    if not path.is_file():
        return 0.0, {"file": False}
    text = path.read_text(encoding="utf-8", errors="replace")
    required = {"frontend-service", "backend-service", "db-service"}
    found = {svc for svc in required if svc in text}
    # require each line for found service to have a non-<none> CLUSTER-IP
    ok = len(found) == len(required)
    if ok:
        for line in text.splitlines():
            for svc in required:
                if line.startswith(svc):
                    cols = line.split()
                    # NAME TYPE CLUSTER-IP EXTERNAL-IP ...
                    if len(cols) < 3 or cols[2] in {"<none>", "<pending>", "-"}:
                        ok = False
                        break
    return (1.0 if ok else 0.0), {
        "file": True,
        "found": sorted(found),
        "all_endpoints": ok,
    }


def score_health_snapshot(path: Path) -> tuple[float, dict]:
    if not path.is_file():
        return 0.0, {"file": False}
    text = path.read_text(encoding="utf-8", errors="replace")
    ok = ("200" in text) and ("healthy" in text.lower())
    return (1.0 if ok else 0.0), {"file": True, "ok": ok}


@dataclass
class Cluster:
    kubectl: str
    context: str
    namespace: str
    observations: list[dict] = field(default_factory=list)

    def run(self, *arguments, timeout=30):
        command = [self.kubectl, "--context", self.context, "--namespace", self.namespace, *arguments]
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
        except (OSError, subprocess.TimeoutExpired) as error:
            observation = {"command": command, "error": str(error)}
            if isinstance(error, subprocess.TimeoutExpired):
                for name, output in (("stdout", error.stdout), ("stderr", error.stderr)):
                    observation[name] = output.decode(errors="replace") if isinstance(output, bytes) else output
            self.observations.append(observation)
            raise
        self.observations.append({"command": command, "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr})
        return result

    def get(self, *arguments):
        result = self.run("get", *arguments, "-o", "json")
        if result.returncode:
            raise RuntimeError(f"kubectl get failed: {result.stderr[:200]}")
        return json.loads(result.stdout)


def discover_cluster(kubectl: str) -> Cluster:
    current = subprocess.run([kubectl, "config", "current-context"], capture_output=True, text=True, timeout=30)
    if current.returncode or not current.stdout.strip():
        raise RuntimeError("Task-local kubectl has no active context")
    context = current.stdout.strip()
    result = subprocess.run(
        [kubectl, "--context", context, "get", "deployments", "--all-namespaces", "-l", "app=webapp", "-o", "json"],
        capture_output=True, text=True, timeout=30,
    )
    if result.returncode:
        raise RuntimeError(f"Cannot discover application namespace: {result.stderr[:200]}")
    namespaces = {}
    for deployment in json.loads(result.stdout).get("items", []):
        metadata = deployment.get("metadata", {})
        namespaces.setdefault(metadata.get("namespace", "default"), set()).add(metadata.get("name"))
    matching = [namespace for namespace, names in namespaces.items() if {"frontend", "backend"} <= names]
    if len(matching) != 1:
        raise RuntimeError("Expected one namespace with frontend and backend deployments")
    return Cluster(kubectl, context, matching[0])


def score_pods_live(cluster: Cluster) -> tuple[float, dict]:
    pods = cluster.get("pods", "-l", "app=webapp").get("items", [])
    active = [pod for pod in pods if not (
        pod.get("status", {}).get("phase") == "Succeeded"
        and any(owner.get("kind") == "Job" for owner in pod.get("metadata", {}).get("ownerReferences", [])))]
    names = [pod.get("metadata", {}).get("name", "") for pod in active]
    required = all(any(name == role or name.startswith(role + "-") for name in names) for role in ("frontend", "backend", "db"))
    running = required and all(pod.get("status", {}).get("phase") == "Running" for pod in active)
    return float(running), {"pods": names, "all_running": running, "completed_jobs": len(pods) - len(active)}


def score_services_live(cluster: Cluster) -> tuple[float, dict]:
    services = workload_services(cluster.get("deployments,statefulsets,services", "-l", "app=webapp").get("items", []))
    endpoints = {item.get("metadata", {}).get("name"): item for item in cluster.get("endpoints").get("items", [])}
    populated = {role: [] for role in services}
    for role, candidates in services.items():
        for candidate in candidates:
            name = candidate["service"]["metadata"]["name"]
            subsets = endpoints.get(name, {}).get("subsets", [])
            if all(any(
                subset.get("addresses") and any(
                    endpoint_port.get("port") == port["target_port"]
                    and endpoint_port.get("protocol", "TCP") == "TCP"
                    and endpoint_port.get("name", "") == port["name"]
                    for endpoint_port in subset.get("ports", [])
                ) for subset in subsets
            ) for port in candidate["ports"]):
                populated[role].append(name)
    return float(all(populated.values())), {"populated": populated}


def score_health_live(cluster: Cluster) -> tuple[float, dict]:
    services = workload_services(cluster.get("deployments,statefulsets,services", "-l", "app=webapp").get("items", []))
    probes = []
    for candidate in services["backend"]:
        name = candidate["service"]["metadata"]["name"]
        for port in candidate["ports"]:
            if port["port"] != 5000:
                continue
            url = f"http://{name}:{port['port']}/health"
            script = f"import json,urllib.request; response=urllib.request.urlopen({url!r},timeout=5); print(json.dumps({{'code':response.status,'body':json.load(response)}}))"
            result = cluster.run("exec", "deployment/backend", "--", "python", "-c", script)
            probe = {"url": url, "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
            probes.append(probe)
            if result.returncode:
                continue
            try:
                payload = json.loads(result.stdout)
            except ValueError:
                continue
            healthy = isinstance(payload, dict) and payload.get("code") == 200 and isinstance(payload.get("body"), dict) and payload["body"].get("status") == "healthy"
            if healthy:
                return 1.0, {"healthy": True, "probes": probes}
    return 0.0, {"healthy": False, "probes": probes}


def score_hpa_live(cluster: Cluster, *, wait_seconds: int = 180) -> tuple[float, dict]:
    readings = []
    for attempt in range(wait_seconds // 5 + 1):
        hpas = cluster.get("hpa", "-l", "app=webapp").get("items", [])
        backend_hpas = [hpa for hpa in hpas if
                        hpa.get("spec", {}).get("scaleTargetRef", {}).get("kind") == "Deployment"
                        and hpa.get("spec", {}).get("scaleTargetRef", {}).get("name") == "backend"]
        for hpa in backend_hpas:
            for metric in hpa.get("status", {}).get("currentMetrics", []):
                resource = metric.get("resource", {})
                value = resource.get("current", {}).get("averageUtilization")
                if metric.get("type") == "Resource" and resource.get("name") == "cpu" and type(value) is int and value >= 0:
                    readings.append(value)
        if readings or not backend_hpas or attempt == wait_seconds // 5:
            break
        time.sleep(5)
    return float(bool(readings)), {"cpu_utilization": readings}


def score_netpol_live(cluster: Cluster) -> tuple[float, dict]:
    pods = cluster.get("pods", "-l", "app=webapp").get("items", [])
    selected = {}
    for role in ("frontend", "backend"):
        selected[role] = next((pod for pod in pods if pod.get("metadata", {}).get("name", "").startswith(role + "-") and pod.get("status", {}).get("phase") == "Running"), None)
    if not all(selected.values()):
        return 0.0, {"reason": "Running frontend and backend required for network probes"}
    image = selected["backend"]["spec"]["containers"][0]["image"]
    script = (
        "import socket\n"
        "address=socket.gethostbyname('db-service')\n"
        "try:\n"
        "    connection=socket.create_connection((address,5432),timeout=4)\n"
        "    connection.close()\n"
        "    print('REACHED')\n"
        "except socket.timeout:\n"
        "    print('BLOCKED')\n"
    )
    outcomes = {}
    for role in ("backend", "frontend"):
        name = f"eval-netpol-{uuid.uuid4().hex[:12]}"
        labels = selected[role]["metadata"].get("labels", {})
        label_text = ",".join(f"{key}={value}" for key, value in labels.items() if key != "pod-template-hash")
        try:
            result = cluster.run("run", name, f"--image={image}", "--image-pull-policy=IfNotPresent", "--restart=Never", "--rm", "-i", "--pod-running-timeout=45s", f"--labels={label_text}", "--command", "--", "python", "-c", script, timeout=60)
            markers = [line.strip() for line in result.stdout.splitlines() if line.strip() in {"REACHED", "BLOCKED"}]
            outcomes[role] = markers[0] if result.returncode == 0 and len(markers) == 1 else "PROBE_FAILED"
        finally:
            cluster.run("delete", "pod", name, "--ignore-not-found", "--wait=false")
    return float(outcomes == {"backend": "REACHED", "frontend": "BLOCKED"}), outcomes


def score_cluster(kubectl: str, receipt_path: Path | None = None) -> dict:
    checks = {"pods": score_pods_live, "services": score_services_live, "health": score_health_live, "hpa": score_hpa_live, "netpol": score_netpol_live}
    try:
        cluster = discover_cluster(kubectl)
    except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired) as error:
        results = {name: {"score": 0.0, "detail": {"reason": str(error)}} for name in checks}
        if receipt_path is not None:
            receipt_path.write_text(json.dumps({"results": results, "discovery_error": str(error)}), encoding="utf-8")
        return results
    results = {}
    for name, check in checks.items():
        try:
            score, detail = check(cluster)
        except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired) as error:
            score, detail = 0.0, {"reason": str(error)}
        results[name] = {"score": score, "detail": detail, "context": cluster.context, "namespace": cluster.namespace}
        if receipt_path is not None:
            receipt_path.write_text(json.dumps({"results": results, "observations": cluster.observations}), encoding="utf-8")
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, help="agent's output/ root")
    parser.add_argument(
        "--eval-tmp-dir",
        default="/media/user/data/agenthle/computing_math/k8s_migration_1/base/eval_data/_eval_tmp",
        help="evaluator scratch/cache (e.g. bootstrap helm binary)",
    )
    parser.add_argument("--helm-version", default=HELM_VERSION_DEFAULT)
    parser.add_argument("--kubectl", required=True, help="task-local kubectl wrapper")
    parser.add_argument("--terraform", required=True, help="task-local Terraform executable")
    parser.add_argument("--json-only", action="store_true", help="suppress debug log on stdout")
    return parser.parse_args()


def emit_payload(payload: dict, tmp_dir: Path) -> None:
    serialized = json.dumps(payload)
    (tmp_dir / "verification.json").write_text(serialized + "\n", encoding="utf-8")
    print(serialized)


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    tmp_dir = Path(args.eval_tmp_dir).resolve()
    tmp_dir.mkdir(parents=True, exist_ok=True)

    breakdown: dict[str, object] = {"weights": {
        "helm": W_HELM, "config": W_CONFIG, "terraform": W_TERRAFORM,
        "cicd": W_CI, "report": W_REPORT,
        "pods": W_PODS, "services": W_SVC, "health": W_HEALTH,
        "hpa": W_HPA, "netpol": W_NETPOL,
    }}

    chart_dir = output_dir / "helm" / "webapp-chart"
    gate = hard_gate_reason(chart_dir)
    if gate is not None:
        payload = {
            "score": 0.0,
            "hard_gate": gate,
            "breakdown": breakdown,
        }
        emit_payload(payload, tmp_dir)
        return 0

    helm_bin = ensure_helm(tmp_dir / "bin", args.helm_version)
    rendered_ok, docs, render_err = render_helm(helm_bin, chart_dir)
    if not rendered_ok:
        payload = {
            "score": 0.0,
            "hard_gate": f"hard_gate: helm template failed ({render_err[:200]})",
            "breakdown": breakdown,
        }
        emit_payload(payload, tmp_dir)
        return 0

    helm_score, helm_detail = score_helm_completeness(docs)
    config_score, config_detail = score_config_correctness(docs)
    tf_score, tf_detail = score_terraform(output_dir / "terraform")
    try:
        tf_validation = validate_terraform(output_dir / "terraform", args.terraform, tmp_dir)
    except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired) as error:
        emit_payload({"score": None, "infrastructure_error": str(error)}, tmp_dir)
        return 1
    tf_detail["native_validation"] = tf_validation
    if not tf_validation["valid"]:
        tf_score = 0.0
    ci_score, ci_detail = score_cicd(output_dir / ".github" / "workflows" / "deploy.yml")
    report_score, report_detail = score_report(output_dir / "verification")

    live = score_cluster(args.kubectl, tmp_dir / "live-probes.json")

    total = (
        helm_score * W_HELM
        + config_score * W_CONFIG
        + tf_score * W_TERRAFORM
        + ci_score * W_CI
        + report_score * W_REPORT
        + live["pods"]["score"] * W_PODS
        + live["services"]["score"] * W_SVC
        + live["health"]["score"] * W_HEALTH
        + live["hpa"]["score"] * W_HPA
        + live["netpol"]["score"] * W_NETPOL
    )

    breakdown.update({
        "helm_completeness": {"score": helm_score, "detail": helm_detail},
        "config_correctness": {"score": config_score, "detail": config_detail},
        "terraform_validity": {"score": tf_score, "detail": tf_detail},
        "cicd_completeness": {"score": ci_score, "detail": ci_detail},
        "report_completeness": {"score": report_score, "detail": report_detail},
        **{f"{name}_live": result for name, result in live.items()},
    })

    payload = {
        "score": round(total, 6),
        "hard_gate": None,
        "mode": "live",
        "breakdown": breakdown,
    }
    emit_payload(payload, tmp_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
