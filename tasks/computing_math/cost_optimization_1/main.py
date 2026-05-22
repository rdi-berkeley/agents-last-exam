"""Linux task definition for computing_math/cost_optimization_1."""

from __future__ import annotations

import logging
import os
import posixpath
import sys
from pathlib import Path
from types import SimpleNamespace

try:
    import cua_bench as cb
except ModuleNotFoundError:  # pragma: no cover - local fallback for direct scorer imports

    class _FallbackTask:
        def __init__(self, description, metadata, computer):
            self.description = description
            self.metadata = metadata
            self.computer = computer

    def _identity_decorator(*args, **kwargs):
        def _wrap(fn):
            return fn

        return _wrap

    cb = SimpleNamespace(
        Task=_FallbackTask,
        DesktopSession=object,
        tasks_config=_identity_decorator,
        setup_task=_identity_decorator,
        evaluate_task=_identity_decorator,
    )

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tasks.common_setup import BaseTaskSetup  # noqa: E402
from tasks.linux_runtime import LinuxTaskConfig  # noqa: E402

_setup = BaseTaskSetup()

SCRIPTS_DIR = Path(__file__).resolve().parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from score_outputs import EXPECTED_SUMMARY_COLUMNS, score_output_bundle  # noqa: E402

logger = logging.getLogger(__name__)

DOMAIN_NAME = "computing_math"
TASK_NAME = "cost_optimization_1"
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


class CostOptimizationConfig(LinuxTaskConfig):
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
    def dashboard_file(self) -> str:
        return f"{self.input_dir}/aws_billing_dashboard.png"

    @property
    def usage_csv(self) -> str:
        return f"{self.input_dir}/resource_usage.csv"

    @property
    def pricing_csv(self) -> str:
        return f"{self.input_dir}/aws_pricing_reference.csv"

    @property
    def task_description_file(self) -> str:
        return f"{self.input_dir}/task_description.txt"

    @property
    def runtime_env_dir(self) -> str:
        return f"{self.input_dir}/runtime_env"

    @property
    def runtime_pyproject(self) -> str:
        return f"{self.runtime_env_dir}/pyproject.toml"

    @property
    def runtime_lockfile(self) -> str:
        return f"{self.runtime_env_dir}/uv.lock"

    @property
    def python_wrapper(self) -> str:
        return f"{self.software_dir}/python_cost_optimization.sh"

    @property
    def output_report(self) -> str:
        return f"{self.remote_output_dir}/optimization_report.json"

    @property
    def output_summary(self) -> str:
        return f"{self.remote_output_dir}/savings_summary.csv"

    @property
    def reference_report(self) -> str:
        return f"{self.reference_dir}/optimization_report.json"

    @property
    def reference_manifest(self) -> str:
        return f"{self.reference_dir}/source_manifest.json"

    @property
    def task_description(self) -> str:
        return (
            "You are an AWS cost optimization analyst working on Linux.\n\n"
            f"## Task Directory\n`{self.task_dir}`\n\n"
            "## Visible Inputs\n"
            f"- Task brief: `{self.task_description_file}`\n"
            f"- Dashboard image: `{self.dashboard_file}`\n"
            f"- Structured usage data: `{self.usage_csv}`\n"
            f"- Pricing table: `{self.pricing_csv}`\n"
            f"- Python runtime manifest: `{self.runtime_pyproject}`\n\n"
            f"- Python runtime lockfile: `{self.runtime_lockfile}`\n\n"
            "## Your Task\n"
            "1. Read the task brief and inspect the dashboard image directly.\n"
            "2. Use the CSV inputs plus the dashboard-only findings to identify wasteful spend.\n"
            "3. Apply the EC2, RDS, S3, NAT Gateway, Elastic IP, and CloudWatch optimization rules described in the brief.\n"
            "4. Write `optimization_report.json` with an `executive_summary` object and a `recommendations` array.\n"
            "5. Write `savings_summary.csv` with columns:\n"
            "   `resource_id,resource_type,name,action,current_cost,projected_cost,monthly_savings`\n\n"
            "## Environment\n"
            f"- Treat `{self.input_dir}` as read-only.\n"
            f"- Use `{self.python_wrapper}` if you want a task-local Python environment with pandas.\n"
            "- The benchmark only exposes the visible input/software/output surface while you are solving the task.\n"
            f"- Write final deliverables only under `{self.remote_output_dir}`.\n"
        )

    def to_metadata(self) -> dict:
        metadata = super().to_metadata()
        metadata.update(
            {
                "task_id": TASK_ID,
                "variant_name": VARIANT_NAME,
                "output_dir_name": self.output_dir_name,
                "dashboard_file": self.dashboard_file,
                "usage_csv": self.usage_csv,
                "pricing_csv": self.pricing_csv,
                "task_description_file": self.task_description_file,
                "runtime_env_dir": self.runtime_env_dir,
                "runtime_pyproject": self.runtime_pyproject,
                "runtime_lockfile": self.runtime_lockfile,
                "python_wrapper": self.python_wrapper,
                "output_report": self.output_report,
                "output_summary": self.output_summary,
                "reference_report": self.reference_report,
                "reference_manifest": self.reference_manifest,
                "expected_summary_columns": EXPECTED_SUMMARY_COLUMNS,
                "canonical_gcs_root": f"gs://ale-data-all/{TASK_ID}/{VARIANT_NAME}/",
            }
        )
        return metadata


@cb.tasks_config(split="train")
def load():
    cfg = CostOptimizationConfig(remote_output_dir=os.environ.get("REMOTE_OUTPUT_DIR", "output"))
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
        agent_summary = await session.read_bytes(meta["output_summary"])
    except Exception as exc:
        logger.error("unable to read agent outputs: %s", exc)
        return [0.0]

    try:
        reference_report = await session.read_bytes(meta["reference_report"])
        reference_manifest = await session.read_bytes(meta["reference_manifest"])
    except Exception as exc:
        logger.error("unable to read evaluator reference data: %s", exc)
        return [0.0]

    result = score_output_bundle(
        agent_report_json=agent_report,
        agent_summary_csv=agent_summary,
        reference_report_json=reference_report,
        source_manifest_json=reference_manifest,
    )
    logger.info("score=%.4f reason=%s details=%s", result.score, result.reason, result.details)
    return [float(result.score)]


if __name__ == "__main__":
    for task in load():
        print(task.description)
