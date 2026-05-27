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
  3. Terraform validity          (10%)  4 structural checks on *.tf
  4. CI/CD completeness          (10%)  5 stages in deploy.yml
  5. Report completeness         ( 5%)  4 files in verification/

Snapshot-derived live subset (30%, capped by snapshot evidence):
  (a) Pods Running               (12%)  verification/pods.txt — all rows Running
  (b) Services have endpoints    (4.5%) verification/services.txt — expected svcs
  (c) Health endpoint returns 200(4.5%) verification/health-check.txt
  (d) HPA TARGETS not <unknown>  (4.5%) live-cluster only — 0 in snapshot mode
  (e) Frontend→db blocked        (4.5%) live-cluster only — 0 in snapshot mode

Checks (d) and (e) only score when `--live-cluster` is set and a reachable
Minikube cluster already exists. Stage 2 default = snapshot mode.
"""

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request
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

CI_STAGES = {
    "build": re.compile(r"docker\s+build", re.IGNORECASE),
    "test": re.compile(r"(pytest|\bnpm\s+test\b)", re.IGNORECASE),
    "security": re.compile(r"trivy", re.IGNORECASE),
    "deploy": re.compile(r"helm\s+(upgrade|install)", re.IGNORECASE),
    "verify": re.compile(r"rollout\s+status", re.IGNORECASE),
}

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

    # 7. secretKeyRef for sensitive env vars
    sensitive_env = {"DB_PASSWORD", "SECRET_KEY", "POSTGRES_PASSWORD"}
    used_secret_ref = set()
    env_value_plain = False
    for _dep, c in _all_containers_in_docs(docs):
        for env in c.get("env") or []:
            name = env.get("name")
            if name in sensitive_env:
                vf = env.get("valueFrom") or {}
                if "secretKeyRef" in vf:
                    used_secret_ref.add(name)
                elif "value" in env:
                    env_value_plain = True
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
    svc_types = {}
    for d in docs:
        if d.get("kind") != "Service":
            continue
        name = (d.get("metadata") or {}).get("name") or ""
        svc_types[name] = (d.get("spec") or {}).get("type") or "ClusterIP"
    svc_ok = (
        svc_types.get("frontend-service") == "NodePort"
        and svc_types.get("backend-service", "ClusterIP") == "ClusterIP"
        and svc_types.get("db-service", "ClusterIP") == "ClusterIP"
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


def score_cicd(ci_file: Path) -> tuple[float, dict]:
    if not ci_file.is_file():
        return 0.0, {stage: False for stage in CI_STAGES}
    text = ci_file.read_text(encoding="utf-8", errors="replace")
    results = {stage: bool(pattern.search(text)) for stage, pattern in CI_STAGES.items()}
    score = sum(1 for v in results.values() if v) / len(results)
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
    rows: list[dict] = []
    for line in text.splitlines():
        if not line.strip() or line.startswith(("$", "#")):
            continue
        if line.lstrip().startswith("NAME"):
            continue
        cols = line.split()
        if len(cols) < 3:
            continue
        # columns: NAME READY STATUS ...
        status = cols[2]
        rows.append({"name": cols[0], "status": status})
    if not rows:
        return 0.0, {"file": True, "rows": 0}
    all_running = all(r["status"] == "Running" for r in rows)
    return (1.0 if all_running else 0.0), {
        "file": True,
        "rows": len(rows),
        "all_running": all_running,
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


def score_hpa_live(live: bool) -> tuple[float, dict]:
    if not live:
        return 0.0, {"reason": "live-only; not scored in snapshot mode"}
    proc = subprocess.run(
        ["kubectl", "get", "hpa", "-o",
         "jsonpath={range .items[*]}{.status.currentMetrics}{\"\\n\"}{end}"],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        return 0.0, {"reason": "kubectl get hpa failed", "stderr": proc.stderr[:200]}
    ok = bool(proc.stdout.strip()) and "<nil>" not in proc.stdout
    return (1.0 if ok else 0.0), {"raw": proc.stdout[:200]}


def score_netpol_live(live: bool) -> tuple[float, dict]:
    if not live:
        return 0.0, {"reason": "live-only; not scored in snapshot mode"}
    # Run a curl pod in the frontend tier and expect it to fail reaching db-service:5432.
    cmd = [
        "kubectl", "run", "netpol-probe",
        "--image=busybox:1.36", "--restart=Never", "--rm", "-i",
        "--labels=app=webapp,tier=frontend", "--",
        "sh", "-c", "nc -w 3 -z db-service 5432 && echo REACHED || echo BLOCKED",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 and "BLOCKED" not in proc.stdout:
        return 0.0, {"reason": "kubectl run probe failed", "stderr": proc.stderr[:200]}
    blocked = "BLOCKED" in proc.stdout and "REACHED" not in proc.stdout
    return (1.0 if blocked else 0.0), {"raw": proc.stdout[:200]}


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
    parser.add_argument(
        "--live-cluster",
        action="store_true",
        help="score HPA TARGETS and NetworkPolicy against a running cluster",
    )
    parser.add_argument("--json-only", action="store_true", help="suppress debug log on stdout")
    return parser.parse_args()


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
        print(json.dumps(payload))
        return 0

    helm_bin = ensure_helm(tmp_dir / "bin", args.helm_version)
    rendered_ok, docs, render_err = render_helm(helm_bin, chart_dir)
    if not rendered_ok:
        payload = {
            "score": 0.0,
            "hard_gate": f"hard_gate: helm template failed ({render_err[:200]})",
            "breakdown": breakdown,
        }
        print(json.dumps(payload))
        return 0

    helm_score, helm_detail = score_helm_completeness(docs)
    config_score, config_detail = score_config_correctness(docs)
    tf_score, tf_detail = score_terraform(output_dir / "terraform")
    ci_score, ci_detail = score_cicd(output_dir / ".github" / "workflows" / "deploy.yml")
    report_score, report_detail = score_report(output_dir / "verification")

    pods_score, pods_detail = score_pods_snapshot(output_dir / "verification" / "pods.txt")
    svc_score, svc_detail = score_services_snapshot(output_dir / "verification" / "services.txt")
    health_score, health_detail = score_health_snapshot(
        output_dir / "verification" / "health-check.txt"
    )
    hpa_score, hpa_detail = score_hpa_live(args.live_cluster)
    netpol_score, netpol_detail = score_netpol_live(args.live_cluster)

    total = (
        helm_score * W_HELM
        + config_score * W_CONFIG
        + tf_score * W_TERRAFORM
        + ci_score * W_CI
        + report_score * W_REPORT
        + pods_score * W_PODS
        + svc_score * W_SVC
        + health_score * W_HEALTH
        + hpa_score * W_HPA
        + netpol_score * W_NETPOL
    )

    breakdown.update({
        "helm_completeness": {"score": helm_score, "detail": helm_detail},
        "config_correctness": {"score": config_score, "detail": config_detail},
        "terraform_validity": {"score": tf_score, "detail": tf_detail},
        "cicd_completeness": {"score": ci_score, "detail": ci_detail},
        "report_completeness": {"score": report_score, "detail": report_detail},
        "pods_snapshot": {"score": pods_score, "detail": pods_detail},
        "services_snapshot": {"score": svc_score, "detail": svc_detail},
        "health_snapshot": {"score": health_score, "detail": health_detail},
        "hpa_live": {"score": hpa_score, "detail": hpa_detail, "mode": "live" if args.live_cluster else "skipped"},
        "netpol_live": {"score": netpol_score, "detail": netpol_detail, "mode": "live" if args.live_cluster else "skipped"},
    })

    payload = {
        "score": round(total, 6),
        "hard_gate": None,
        "mode": "live" if args.live_cluster else "snapshot",
        "breakdown": breakdown,
    }
    print(json.dumps(payload))
    return 0


if __name__ == "__main__":
    sys.exit(main())
