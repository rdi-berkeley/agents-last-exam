"""Linux task definition for Kubernetes payment-api root-cause analysis."""

from __future__ import annotations

import logging
import os
import posixpath
import sys
from pathlib import Path
from typing import Any

import cua_bench as cb

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tasks.common_setup import BaseTaskSetup  # noqa: E402
from tasks.linux_runtime import LinuxTaskConfig  # noqa: E402

_setup = BaseTaskSetup()

SCRIPTS_DIR = Path(__file__).resolve().parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from score_root_cause_analysis import ScoreResult, score_report  # noqa: E402

logger = logging.getLogger(__name__)

DOMAIN_NAME = "computing_math"
TASK_NAME = "k8s_payment_api_root_cause_analysis"
TASK_ID = f"{DOMAIN_NAME}/{TASK_NAME}"
VARIANT_NAME = "base"
CANONICAL_OUTPUT_DIR_NAMES = {"output", "output_test_pos", "output_test_neg"}


def _normalize_output_dir_name(raw: str) -> str:
    normalized = posixpath.normpath(raw.replace("\\", "/"))
    if normalized not in CANONICAL_OUTPUT_DIR_NAMES:
        raise ValueError(
            "REMOTE_OUTPUT_DIR must normalize to one of: "
            + ", ".join(sorted(CANONICAL_OUTPUT_DIR_NAMES))
        )
    return normalized


def _as_text(payload: Any) -> str:
    if isinstance(payload, bytes):
        return payload.decode("utf-8-sig")
    return str(payload)


class K8sRootCauseAnalysisConfig(LinuxTaskConfig):
    DOMAIN_NAME: str = DOMAIN_NAME
    TASK_NAME: str = TASK_NAME
    VARIANT_NAME: str = VARIANT_NAME

    def __init__(self, *, remote_output_dir: str = "output") -> None:
        super().__init__(
            DOMAIN_NAME=DOMAIN_NAME,
            TASK_NAME=TASK_NAME,
            VARIANT_NAME=VARIANT_NAME,
            OS_TYPE="linux",
            REMOTE_OUTPUT_DIR=remote_output_dir,
        )

    @property
    def output_dir_name(self) -> str:
        return _normalize_output_dir_name(self.REMOTE_OUTPUT_DIR)

    @property
    def remote_output_dir(self) -> str:
        return f"{self.task_dir}/{self.output_dir_name}"

    @property
    def task_prompt_file(self) -> str:
        return f"{self.input_dir}/task_prompt.md"

    @property
    def cluster_state_file(self) -> str:
        return f"{self.input_dir}/cluster_state.txt"

    @property
    def deployment_file(self) -> str:
        return f"{self.input_dir}/deployment.yaml"

    @property
    def failing_pod_log_file(self) -> str:
        return f"{self.input_dir}/failing_pod.log"

    @property
    def output_report(self) -> str:
        return f"{self.remote_output_dir}/root_cause_analysis.json"

    @property
    def reference_report(self) -> str:
        return f"{self.reference_dir}/root_cause_analysis.json"

    @property
    def task_description(self) -> str:
        return f"""You are the on-call SRE for a Kubernetes production incident.

## Task Directory
`{self.task_dir}`

## Visible Inputs
- Task prompt: `{self.task_prompt_file}`
- Cluster state capture: `{self.cluster_state_file}`
- Current Deployment and HPA YAML: `{self.deployment_file}`
- Crashing pod log: `{self.failing_pod_log_file}`

## Your Task
Analyze the supplied static evidence and write a complete root-cause analysis
report for the `payment-api` incident, including why the HPA reports memory
utilization above target. Ground every root cause in literal
evidence from the visible files, list affected Kubernetes resources, propose a
prioritized remediation plan, and state whether rollback is safe.

## Required Output
Write exactly one JSON file:

```text
{self.output_report}
```

The JSON must include these top-level keys:

```text
incident_metadata, root_causes, affected_resources, remediation_plan, summary
```

### Schema details

**`incident_metadata`** — object with at least:
- `safe_to_rollback` (boolean): whether rolling back to the previous ReplicaSet revision is safe.

**`root_causes`** — array of objects, each with at least:
- `type` (string): short classification label, e.g. `"OOMKilled"`, `"liveness_probe_too_aggressive"`, `"missing_metrics_port_env"`.
- `severity` (string): one of `"critical"`, `"high"`, `"medium"`, `"low"`.
- `description` (string): explanation of the root cause.
- `evidence` (array of strings): each item must be a verbatim text string copied from one of the visible input files (not a file-name citation — the actual text that appears in the file).

**`affected_resources`** — array of objects, each with `kind`, `name`, and `namespace`.

**`remediation_plan`** — array of objects, each with at least an `action` (string).

**`summary`** — string summarizing the incident and recommended path forward.

## Constraints
- Treat `{self.input_dir}` as read-only.
- Keep generated files inside `{self.remote_output_dir}`.
- Do not use external web sources or invent resource names, metrics, commit SHAs, or timestamps.
"""

    def to_metadata(self) -> dict[str, Any]:
        metadata = super().to_metadata()
        metadata.update(
            {
                "task_id": TASK_ID,
                "variant_name": VARIANT_NAME,
                "output_dir_name": self.output_dir_name,
                "task_prompt_file": self.task_prompt_file,
                "cluster_state_file": self.cluster_state_file,
                "deployment_file": self.deployment_file,
                "failing_pod_log_file": self.failing_pod_log_file,
                "output_report": self.output_report,
                "reference_report": self.reference_report,
                "canonical_gcs_root": f"gs://ale-data-all/{TASK_ID}/{VARIANT_NAME}/",
            }
        )
        return metadata


@cb.tasks_config(split="train")
def load():
    cfg = K8sRootCauseAnalysisConfig(remote_output_dir=os.environ.get("REMOTE_OUTPUT_DIR", "output"))
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
    try:
        agent_report = await session.read_bytes(meta["output_report"])
        cluster_state = _as_text(await session.read_file(meta["cluster_state_file"]))
        deployment_yaml = _as_text(await session.read_file(meta["deployment_file"]))
        failing_pod_log = _as_text(await session.read_file(meta["failing_pod_log_file"]))
    except Exception as exc:
        logger.error("unable to read evaluation artifacts: %s", exc)
        return [0.0]

    result: ScoreResult = score_report(
        agent_report,
        cluster_state=cluster_state,
        deployment_yaml=deployment_yaml,
        failing_pod_log=failing_pod_log,
    )
    logger.info("score=%.4f details=%s", result.score, result.to_dict())
    return [float(result.score)]


if __name__ == "__main__":
    for task in load():
        print(task.description)
