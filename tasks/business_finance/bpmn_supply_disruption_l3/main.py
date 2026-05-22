"""AgentHLE task: business_finance/bpmn_supply_disruption_l3."""

from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

try:
    import cua_bench as cb
except ModuleNotFoundError:  # pragma: no cover - local import fallback only

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

# cua_bench loads task modules via exec_module without always pre-registering
# them in sys.modules; dataclass needs this for string annotation handling.
if __name__ not in sys.modules:
    sys.modules[__name__] = sys.modules.get(__name__, type(sys)(__name__))

from tasks.business_finance.bpmn_supply_disruption_l3.scripts.score_output_bundle import \
    score_output_bundle
from tasks.common_setup import BaseTaskSetup
from tasks.linux_runtime import LinuxTaskConfig

_setup = BaseTaskSetup()

logger = logging.getLogger(__name__)


async def _missing(session: cb.DesktopSession, path: str, *, label: str) -> bool:
    if await session.exists(path):
        return False
    logger.error("Missing %s: %s", label, path)
    return True


@dataclass
class BpmnSupplyDisruptionConfig(LinuxTaskConfig):
    DOMAIN_NAME: str = "business_finance"
    TASK_NAME: str = "bpmn_supply_disruption_l3"
    VARIANT_NAME: str = "base"
    REMOTE_OUTPUT_DIR: str = os.environ.get("REMOTE_OUTPUT_DIR", "output")

    @property
    def task_prompt_file(self) -> str:
        return f"{self.input_dir}/task_prompt.md"

    @property
    def starter_project_dir(self) -> str:
        return f"{self.input_dir}/starter_project"

    @property
    def starter_bpmn(self) -> str:
        return f"{self.starter_project_dir}/original_process.bpmn20.xml"

    @property
    def starter_scenario(self) -> str:
        return f"{self.starter_project_dir}/disruption_scenario_L3.md"

    @property
    def starter_rules(self) -> str:
        return f"{self.starter_project_dir}/business_rules_L3.md"

    @property
    def starter_org(self) -> str:
        return f"{self.starter_project_dir}/org_hierarchy.json"

    @property
    def starter_scenarios(self) -> str:
        return f"{self.starter_project_dir}/test_scenarios_L3.json"

    @property
    def starter_compose(self) -> str:
        return f"{self.starter_project_dir}/docker-compose.yml"

    @property
    def starter_diagram(self) -> str:
        return f"{self.starter_project_dir}/original_process_diagram.png"

    @property
    def software_readme(self) -> str:
        return f"{self.software_dir}/README.txt"

    @property
    def output_test_pos_dir(self) -> str:
        return f"{self.task_dir}/output_test_pos"

    @property
    def output_test_neg_dir(self) -> str:
        return f"{self.task_dir}/output_test_neg"

    @property
    def agent_output_dir(self) -> str:
        return f"{self.task_dir}/output"

    @property
    def output_bpmn(self) -> str:
        return f"{self.remote_output_dir}/modified_process.bpmn20.xml"

    @property
    def output_structural(self) -> str:
        return f"{self.remote_output_dir}/structural_changes.json"

    @property
    def output_rules(self) -> str:
        return f"{self.remote_output_dir}/business_rules_compliance.json"

    @property
    def output_results(self) -> str:
        return f"{self.remote_output_dir}/test_results.json"

    @property
    def task_description(self) -> str:
        return f"""\
You are working on a Linux VM with Docker, Docker Compose, Python, and curl available.

## Your Task
Redesign LY Juice Company's monthly production scheduling workflow in Flowable BPMN 6.5.0 so it handles a compound disruption:
1. raw material supply shortage
2. incoming material quality hold

## Visible Inputs
- task prompt: `{self.task_prompt_file}`
- starter project directory: `{self.starter_project_dir}`
- original BPMN: `{self.starter_bpmn}`
- disruption scenario: `{self.starter_scenario}`
- business rules: `{self.starter_rules}`
- org hierarchy: `{self.starter_org}`
- scenario suite: `{self.starter_scenarios}`
- Flowable stack file: `{self.starter_compose}`
- original diagram: `{self.starter_diagram}`

## What You Must Do
1. Read the staged task materials.
2. Modify the BPMN while preserving original anchored elements and IDs.
3. Use process definition key `monthlyProductionScheduling_modified_L3`.
4. Start or reuse the local Flowable stack from `{self.starter_compose}`.
5. Deploy the modified BPMN and validate it against the provided scenario definitions.
6. Write the final output bundle to `{self.agent_output_dir}`.

## Runtime Notes
- Flowable REST API base URL: `http://localhost:8080/flowable-task/process-api/`
- Flowable credentials: `admin` / `test`
- For runtime testing, set role assignee variables to `admin`
- Ensure every gateway or assignee expression only references variables that already exist
- Keep new manual work as `userTask` nodes and use `${{...}}` expressions for gateway logic

## Required Output Files
- `{self.agent_output_dir}/modified_process.bpmn20.xml`
- `{self.agent_output_dir}/structural_changes.json`
- `{self.agent_output_dir}/business_rules_compliance.json`
- `{self.agent_output_dir}/test_results.json`

You may also write `{self.agent_output_dir}/deployment_log.json` as an optional diagnostic artifact.

## Evaluation Method
Your output is scored by a **static structural evaluator** that parses the BPMN XML directly — it does NOT re-run Flowable. The evaluator checks structural topology, test results, compliance cross-validation, anti-gaming, data flow, and role coupling. Each section is weighted; partial credit is given. Read the full evaluation details and **required output schemas** (including the exact JSON format for `test_results.json`) in `{self.task_prompt_file}`.

## Data Flow Topology Rule
Every new task's `in_*` form property must have a matching `out_*` producer on a **topological predecessor** in the sequence-flow graph. The evaluator traces BFS paths, not Flowable's global variable scope.

## Important Constraints
- Do not modify the staged input files
- Keep successful exception handling paths rejoined into the main workflow
- Send last-resort escalation to a separate terminal end event
- Keep the JSON artifacts consistent with the BPMN you actually produced
"""

    def to_metadata(self) -> dict:
        metadata = super().to_metadata()
        metadata.update(
            {
                "task_prompt_file": self.task_prompt_file,
                "starter_project_dir": self.starter_project_dir,
                "starter_bpmn": self.starter_bpmn,
                "starter_scenario": self.starter_scenario,
                "starter_rules": self.starter_rules,
                "starter_org": self.starter_org,
                "starter_scenarios": self.starter_scenarios,
                "starter_compose": self.starter_compose,
                "starter_diagram": self.starter_diagram,
                "software_readme": self.software_readme,
                "output_test_pos_dir": self.output_test_pos_dir,
                "output_test_neg_dir": self.output_test_neg_dir,
                "agent_output_dir": self.agent_output_dir,
                "output_bpmn": self.output_bpmn,
                "output_structural": self.output_structural,
                "output_rules": self.output_rules,
                "output_results": self.output_results,
                "canonical_gcs_root": (
                    f"gs://ale-data-all/{self.DOMAIN_NAME}/{self.TASK_NAME}/{self.VARIANT_NAME}/"
                ),
            }
        )
        return metadata


config = BpmnSupplyDisruptionConfig()


@cb.tasks_config(split="train")
def load():
    return [
        cb.Task(
            description=config.task_description,
            metadata=config.to_metadata(),
            computer={"provider": "computer", "setup_config": {"os_type": "linux"}},
        )
    ]


@cb.setup_task(split="train")
async def start(task_cfg, session: cb.DesktopSession):
    await _setup(task_cfg, session)


@cb.evaluate_task(split="train")
async def evaluate(task_cfg, session: cb.DesktopSession) -> list[float]:
    meta = task_cfg.metadata
    required_outputs = [
        ("output_bpmn", "modified BPMN"),
        ("output_structural", "structural changes JSON"),
        ("output_rules", "business rules JSON"),
        ("output_results", "test results JSON"),
    ]

    for key, label in required_outputs:
        if not await session.exists(meta[key]):
            logger.error("Missing %s at %s", label, meta[key])
            return [0.0]

    with tempfile.TemporaryDirectory(prefix="bpmn_supply_disruption_eval_") as tmp_dir:
        tmp = Path(tmp_dir)
        local_bpmn = tmp / "modified_process.bpmn20.xml"
        local_structural = tmp / "structural_changes.json"
        local_rules = tmp / "business_rules_compliance.json"
        local_results = tmp / "test_results.json"

        try:
            local_bpmn.write_bytes(await session.read_bytes(meta["output_bpmn"]))
            local_structural.write_bytes(await session.read_bytes(meta["output_structural"]))
            local_rules.write_bytes(await session.read_bytes(meta["output_rules"]))
            local_results.write_bytes(await session.read_bytes(meta["output_results"]))
            result = score_output_bundle(
                bpmn_path=local_bpmn,
                structural_path=local_structural,
                rules_path=local_rules,
                results_path=local_results,
            )
        except Exception as exc:
            logger.exception("Evaluation failed: %s", exc)
            return [0.0]

    logger.info("evaluation=%s", json.dumps(result, sort_keys=True)[:2000])
    return [float(result.get("score", 0.0))]
