"""computing_math/k8s_migration_1 — AgentHLE Kubernetes-migration task.

The agent migrates a Docker Compose 3-tier web app (React + Flask + PostgreSQL)
to a Minikube deployment. Expected deliverables under `output/`: Helm chart,
Terraform config, GitHub Actions pipeline, and live verification snapshots.

Scoring is performed on the VM by `scripts/verify_k8s_migration.py`, which
renders the chart with a pinned Helm binary and produces a weighted score
(hard gates drop to 0.0; static 70%, snapshot-derived live 21%, live-only
(d)+(e) another 9% when run against a real cluster).
"""

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import cua_bench as cb

from tasks.common_setup import BaseTaskSetup
from tasks.linux_runtime import LinuxTaskConfig

_setup = BaseTaskSetup()

logger = logging.getLogger(__name__)

DOMAIN_NAME = "computing_math"
TASK_NAME = "k8s_migration_1"
VARIANT_NAME = "base"

SCRIPTS_DIR = Path(__file__).resolve().parent / "scripts"
VERIFY_SCRIPT_NAME = "verify_k8s_migration.py"
EVAL_TMP_SUBDIR = "_eval_tmp"
RUNTIME_STATE_SUBDIR = "_runtime_state"
SOFTWARE_ENTRY_NAMES = (
    "docker",
    "minikube",
    "kubectl",
    "helm",
    "terraform",
    "trivy",
)

ALLOWED_OUTPUT_DIRS = {
    "output",
    "output_test_pos",
    "output_test_neg",
    "output_admin_pos",
    "output_admin_neg",
}

MINIKUBE_START_CMD = (
    "minikube start --driver=docker --cpus=4 --memory=8192 "
    "--kubernetes-version=v1.29.0 --cni=calico "
    "--addons=ingress,metrics-server,storage-provisioner"
)


def _read_script(name: str) -> str:
    return (SCRIPTS_DIR / name).read_text(encoding="utf-8")


async def _run_command(
    session: cb.DesktopSession,
    command: str,
    *,
    timeout: float | None = None,
    check: bool = False,
) -> dict:
    try:
        if timeout is not None:
            return await session.run_command(command, timeout=timeout, check=check)
        return await session.run_command(command, check=check)
    except TypeError:
        return await session.run_command(command, check=check)


@dataclass
class K8sMigrationTaskConfig(LinuxTaskConfig):
    DOMAIN_NAME: str = DOMAIN_NAME
    TASK_NAME: str = TASK_NAME
    VARIANT_NAME: str = VARIANT_NAME

    @property
    def output_dir_name(self) -> str:
        return (self.REMOTE_OUTPUT_DIR or "output").strip().strip("/")

    @property
    def remote_output_dir(self) -> str:
        return f"{self.task_dir}/{self.output_dir_name}"

    @property
    def software_readme(self) -> str:
        return f"{self.software_dir}/README.txt"

    @property
    def eval_dir(self) -> str:
        # Task-local evaluator scratch root. Not part of the canonical staged
        # tree (input/ reference/ software/...); used here for runtime-only
        # state (eval_tmp + runtime_state subdirs below).
        return f"{self.task_dir}/eval_data"

    @property
    def eval_tmp_dir(self) -> str:
        return f"{self.eval_dir}/{EVAL_TMP_SUBDIR}"

    @property
    def runtime_state_dir(self) -> str:
        return f"{self.eval_dir}/{RUNTIME_STATE_SUBDIR}"

    @property
    def software_entries(self) -> dict[str, str]:
        return {
            name: f"{self.software_dir}/{name}" for name in SOFTWARE_ENTRY_NAMES
        }

    @property
    def input_readme(self) -> str:
        return f"{self.input_dir}/README.md"

    @property
    def input_requirements(self) -> str:
        return f"{self.input_dir}/requirements.md"

    @property
    def input_compose(self) -> str:
        return f"{self.input_dir}/app/docker-compose.yml"

    @property
    def task_description(self) -> str:
        return f"""\
You are migrating a Docker Compose 3-tier web application (React frontend,
Flask backend, PostgreSQL) to a production-grade Kubernetes deployment on a
local Minikube cluster.

## Working Directory
Task root on this VM:
- `{self.task_dir}`

Agent-visible inputs:
- Compose source + app code: `{self.input_dir}/app/`
- Architecture spec:        `{self.input_requirements}`
- Task README:              `{self.input_readme}`

Use the task-local software entry points under `{self.software_dir}` rather
than raw PATH commands:
- Docker:     `{self.software_entries["docker"]}`
- Minikube:   `{self.software_entries["minikube"]}`
- kubectl:    `{self.software_entries["kubectl"]}`
- Helm:       `{self.software_entries["helm"]}`
- Terraform:  `{self.software_entries["terraform"]}`
- Trivy:      `{self.software_entries["trivy"]}`

Write all deliverables here:
- `{self.remote_output_dir}`

## Required Deliverables

1. Helm chart at `{self.remote_output_dir}/helm/webapp-chart/`:
   - `Chart.yaml` (valid YAML, with `name`)
   - `values.yaml`
   - Templates covering 8 Kubernetes resources:
     Deployment/frontend (2 replicas, limits 256Mi/500m),
     Deployment/backend  (2 replicas, limits 512Mi/1000m, readiness + liveness
       probes on `/health`),
     StatefulSet/db      (1 replica, PVC request 5Gi),
     ConfigMap, Secret (base64 only — no plaintext), HPA (min=2, max=5, 70% CPU),
     NetworkPolicy (only backend tier may reach db tier), Ingress
     (`/` → frontend, `/api` → backend). Every resource must carry the label
     `app=webapp`.

2. Terraform config at `{self.remote_output_dir}/terraform/`:
   - `required_providers` block
   - every `variable` has a `description`
   - at least one `output` block
   - Minikube resource references `cni=calico` and addons
     `ingress`, `metrics-server`, `storage-provisioner`

3. GitHub Actions workflow at
   `{self.remote_output_dir}/.github/workflows/deploy.yml` with 5 stages:
   `build` (`docker build`), `test` (`pytest` + `npm test`), `security`
   (`trivy`), `deploy` (`helm upgrade`), `verify` (`rollout status`).

4. Live verification text snapshots under
   `{self.remote_output_dir}/verification/` — `pods.txt`, `services.txt`,
   `helm-status.txt`, `health-check.txt` — captured from the running cluster.

## Expected Minikube Startup

```
{self.software_entries["minikube"]} start --driver=docker --cpus=4 --memory=8192 --kubernetes-version=v1.29.0 --cni=calico --addons=ingress,metrics-server,storage-provisioner
```

The `software/minikube`, `software/kubectl`, and `software/helm` wrappers
already point their kube state and caches at task-local data-disk paths under
`{self.runtime_state_dir}` so the root partition does not fill. Calico is
required for NetworkPolicy enforcement; metrics-server is required for HPA to
report a CPU target.

## Success Criteria

- `Chart.yaml` parses as YAML and declares a `name` (hard gate otherwise).
- `values.yaml` is present (hard gate otherwise).
- Secrets must carry only base64-encoded values (no plaintext) (hard gate otherwise).
- `helm template` renders the chart without error (hard gate otherwise).
- Static checks (Helm resource coverage, config correctness, Terraform, CI/CD,
  verification files) contribute up to 70% of the score.
- Verification snapshots (pod status / services / health endpoint) contribute
  up to another ~21%. Full credit requires a running Minikube cluster.
"""

    def to_metadata(self) -> dict:
        metadata = super().to_metadata()
        metadata.update(
            {
                "output_dir_name": self.output_dir_name,
                "software_readme": self.software_readme,
                "software_entries": self.software_entries,
                "input_readme": self.input_readme,
                "input_requirements": self.input_requirements,
                "input_compose": self.input_compose,
                "eval_tmp_dir": self.eval_tmp_dir,
                "runtime_state_dir": self.runtime_state_dir,
                "minikube_start_cmd": MINIKUBE_START_CMD,
            }
        )
        return metadata


config = K8sMigrationTaskConfig()


@cb.tasks_config(split="train")
def load():
    cfg = K8sMigrationTaskConfig()
    return [
        cb.Task(
            description=cfg.task_description,
            metadata=cfg.to_metadata(),
            computer={"provider": "computer", "setup_config": {"os_type": "linux"}},
        )
    ]


@cb.setup_task(split="train")
async def start(task_cfg, session: cb.DesktopSession):
    await _setup(task_cfg, session)


@cb.evaluate_task(split="train")
async def evaluate(task_cfg, session: cb.DesktopSession) -> list[float]:
    meta = task_cfg.metadata
    eval_tmp_dir = meta["eval_tmp_dir"]
    await session.makedirs(eval_tmp_dir)
    verify_path = f"{eval_tmp_dir}/{VERIFY_SCRIPT_NAME}"
    await session.write_file(verify_path, _read_script(VERIFY_SCRIPT_NAME))

    live_cluster = os.environ.get("K8S_LIVE_CLUSTER") == "1"
    live_flag = " --live-cluster" if live_cluster else ""

    cmd = (
        f'python3.12 "{verify_path}" '
        f'--output-dir "{meta["remote_output_dir"]}" '
        f'--eval-tmp-dir "{eval_tmp_dir}" '
        f'--json-only{live_flag}'
    )
    result = await _run_command(session, cmd, timeout=600.0, check=False)
    stdout = (result.get("stdout") or "").strip()
    if result.get("return_code") != 0 and not stdout:
        logger.error("verifier crashed: %s", (result.get("stderr") or "")[:400])
        return [0.0]
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError:
        logger.error(
            "verifier stdout was not JSON: stdout=%r stderr=%r",
            stdout[:400],
            (result.get("stderr") or "")[:400],
        )
        return [0.0]
    score = float(payload.get("score", 0.0))
    logger.info(
        "evaluation: score=%.3f mode=%s hard_gate=%s",
        score,
        payload.get("mode"),
        payload.get("hard_gate"),
    )
    return [score]


if __name__ == "__main__":
    for task in load():
        print(task.description)
