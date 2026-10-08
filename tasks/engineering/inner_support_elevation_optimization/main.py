"""Source-preserving open-source support-elevation task."""

from dataclasses import dataclass
import asyncio
import json
from pathlib import Path
import time

import cua_bench as cb

from tasks.common_setup import BaseTaskSetup
from tasks.engineering.inner_support_elevation_optimization.native_task import (
    evaluate_delivery,
    replay_submission,
    setup_native,
)
from tasks.engineering.inner_support_elevation_optimization.source_review import (
    review_submission,
)
from tasks.engineering.inner_support_elevation_optimization.verification import (
    EvaluationUnavailableError,
)
from tasks.linux_runtime import LinuxTaskConfig

ASSETS = Path(__file__).with_name("assets")


@dataclass
class InnerSupportElevationConfig(LinuxTaskConfig):
    DOMAIN_NAME: str = "engineering"
    TASK_NAME: str = "inner_support_elevation_optimization"
    VARIANT_NAME: str = "base"

    @property
    def task_description(self):
        return (
            (ASSETS / "instruction.md")
            .read_text()
            .replace("{input_dir}", self.input_dir)
            .replace("{output_dir}", self.remote_output_dir)
        )


@cb.tasks_config(split="train")
def load():
    config = InnerSupportElevationConfig()
    return [
        cb.Task(
            description=config.task_description,
            metadata={
                **config.to_metadata(),
                "release_status": "native_entrypoint_and_agent_review_passed",
            },
            computer={"provider": "computer", "setup_config": {"os_type": "linux"}},
        )
    ]


@cb.setup_task(split="train")
async def start(task_cfg, session: cb.DesktopSession):
    await BaseTaskSetup()(task_cfg, session)
    await setup_native(task_cfg.metadata, session)


@cb.evaluate_task(split="train")
async def evaluate(task_cfg, session: cb.DesktopSession) -> list[float]:
    deadline = time.monotonic() + 7100
    review = await review_submission(task_cfg.metadata, session)
    evidence = review["evidence_directory"]
    if not review["replay_eligible"]:
        findings = review["blocking_findings"]
        status = "source_noncompliant"
        gate = {
            "score": 0.0,
            "success": False,
            "evaluation_status": status,
            "source_review_outcome": review["outcome"],
            "findings": findings,
            "engineering_caveats": review["engineering_caveats"],
            "evidence_directory": evidence,
            "native_replay_completed": False,
            "full_task_acceptance": False,
        }
        (Path(evidence) / "evaluation-gate.json").write_text(json.dumps(gate, indent=2))
        return [0.0]
    replay = await replay_submission(task_cfg.metadata, session, review, deadline=deadline - 420)
    try:
        result = await asyncio.wait_for(
            evaluate_delivery(task_cfg.metadata, session, review, replay),
            timeout=max(1, deadline - time.monotonic()),
        )
    except TimeoutError as error:
        raise EvaluationUnavailableError(
            "Native delivery assessment exceeded the evaluator time budget"
        ) from error
    return [float(result["score"])]
