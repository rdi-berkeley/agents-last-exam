"""ALE task: computing_math/incident_rca.

Patterned on tasks/demo/hello/main.py. The framework auto-stages input/ + reference/; start()
ensures the output dir + sanity-checks input; evaluate() reads output/rca.json + the hidden
reference and scores via scripts/score_incident_rca.py. No LLM judge.
"""
from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import cua_bench as cb

from tasks.linux_runtime import LinuxTaskConfig

SCRIPTS_DIR = Path(__file__).parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from score_incident_rca import score_submission  # noqa: E402

logger = logging.getLogger(__name__)

DOMAIN_NAME = "computing_math"
TASK_NAME = "incident_rca"
VARIANT_NAME = "base"


@dataclass
class TaskConfig(LinuxTaskConfig):
    DOMAIN_NAME: str = DOMAIN_NAME
    TASK_NAME: str = TASK_NAME
    VARIANT_NAME: str = VARIANT_NAME

    @property
    def deploy_file(self) -> str:
        return f"{self.input_dir}/deploy/deployment.yaml"

    @property
    def logs_dir(self) -> str:
        return f"{self.input_dir}/logs"

    @property
    def readme_file(self) -> str:
        return f"{self.input_dir}/README.md"

    @property
    def rca_output(self) -> str:
        return f"{self.remote_output_dir}/rca.json"

    @property
    def ground_truth_reference(self) -> str:
        return f"{self.reference_dir}/ground_truth.json"

    @property
    def task_description(self) -> str:
        return (
            "You are the on-call SRE for an Ubuntu-based Kubernetes cluster.\n\n"
            "The `payments-api` service returns 503 'no healthy upstream' and its pods are Running "
            f"but never Ready. Using `{self.deploy_file}` and the files under `{self.logs_dir}`, "
            f"diagnose the ONE true root cause and write `{self.rca_output}`:\n"
            '  {"root_cause": "<token>", "fix": "<token>", "evidence": "the single most decisive line"}\n\n'
            f"Choose `root_cause` and `fix` from the exact token lists in `{self.readme_file}`. The "
            "logs contain distractors (benign warnings and a recovered blip); identify the real "
            f"cause, not a red herring. Write your answer to exactly `{self.rca_output}`."
        )

    def to_metadata(self) -> dict:
        m = super().to_metadata()
        m.update(
            {
                "readme_file": self.readme_file,
                "rca_output": self.rca_output,
                "ground_truth_reference": self.ground_truth_reference,
            }
        )
        return m


@cb.tasks_config(split="train")
def load():
    cfg = TaskConfig()
    return [
        cb.Task(
            description=cfg.task_description,
            metadata=cfg.to_metadata(),
            computer={"provider": "computer", "setup_config": {"os_type": cfg.OS_TYPE}},
        )
    ]


@cb.setup_task(split="train")
async def start(task_cfg, session: cb.DesktopSession):
    meta = task_cfg.metadata
    await session.run_command(f"mkdir -p {meta['remote_output_dir']!r}", check=False)
    try:
        await session.read_file(meta["readme_file"])
    except Exception as exc:
        raise RuntimeError(f"staged input missing: {meta['readme_file']} unreadable ({exc})")
    logger.info("[incident_rca] output dir ready; input staged")


@cb.evaluate_task(split="train")
async def evaluate(task_cfg, session: cb.DesktopSession) -> list[float]:
    meta = task_cfg.metadata
    try:
        rca_bytes = await session.read_bytes(meta["rca_output"])
    except Exception as exc:
        logger.info("[incident_rca] output unreadable at %s: %s", meta["rca_output"], exc)
        return [0.0]
    try:
        reference_bytes = await session.read_bytes(meta["ground_truth_reference"])
        score, report = score_submission(rca_bytes, reference_bytes)
    except Exception as exc:
        logger.exception("[incident_rca] evaluation failed: %s", exc)
        return [0.0]
    logger.info("[incident_rca] eval_report=%s", json.dumps(report, sort_keys=True))
    return [score]
