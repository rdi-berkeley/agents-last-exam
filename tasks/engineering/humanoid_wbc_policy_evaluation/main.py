"""humanoid_wbc_policy_evaluation — evaluate Unitree G1 WBC policy rollouts."""

from __future__ import annotations

import logging
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cua_bench as cb

if __name__ not in sys.modules:
    sys.modules[__name__] = sys.modules.get(__name__, type(sys)(__name__))

from tasks.common_setup import BaseTaskSetup
from tasks.linux_runtime import LinuxTaskConfig
from tasks.engineering.humanoid_wbc_policy_evaluation.scripts.score_outputs import score_report


_setup = BaseTaskSetup()

logger = logging.getLogger(__name__)

VARIANTS = [("base", "8 humanoid whole-body-control policy evaluation cases")]


async def _missing(session: cb.DesktopSession, path: str, *, label: str, tag: str) -> bool:
    if (await session.file_exists(path) or await session.directory_exists(path)):
        return False
    logger.error("[%s] Missing %s: %s", tag, label, path)
    return True


@dataclass
class HumanoidWbcPolicyEvaluationConfig(LinuxTaskConfig):
    DOMAIN_NAME: str = "engineering"
    TASK_NAME: str = "humanoid_wbc_policy_evaluation"
    VARIANT_NAME: str = ""
    VARIANT_LABEL: str = ""

    @property
    def input_task_brief(self) -> str:
        return f"{self.input_dir}/task_brief.md"

    @property
    def input_policy_cases(self) -> str:
        return f"{self.input_dir}/policy_cases.json"

    @property
    def input_output_schema(self) -> str:
        return f"{self.input_dir}/output_schema.json"

    @property
    def input_runtime_env_dir(self) -> str:
        return f"{self.input_dir}/runtime_env"

    @property
    def input_runtime_mjlab_zip(self) -> str:
        return f"{self.input_runtime_env_dir}/mjlab.zip"

    @property
    def input_runtime_env_export(self) -> str:
        return f"{self.input_runtime_env_dir}/mjlab_env_export.txt"

    @property
    def input_runtime_motions_zip(self) -> str:
        return f"{self.input_runtime_env_dir}/motions-1.zip"

    @property
    def input_runtime_policies_zip(self) -> str:
        return f"{self.input_runtime_env_dir}/policies.zip"

    @property
    def output_report(self) -> str:
        return f"{self.remote_output_dir}/policy_evaluation_report.json"

    @property
    def output_visual_demos_dir(self) -> str:
        return f"{self.remote_output_dir}/visual_demos"

    @property
    def reference_expected_verdicts(self) -> str:
        return f"{self.reference_dir}/expected_verdicts.json"

    @property
    def task_description(self) -> str:
        return f"""\
You are evaluating whole-body-control policy rollouts for a Unitree G1 humanoid in mjlab.

## Variant
`{self.VARIANT_NAME}`: {self.VARIANT_LABEL}

## Input Files
- Task brief: `{self.input_task_brief}`
- Policy case list: `{self.input_policy_cases}`
- Required JSON output schema: `{self.input_output_schema}`
- mjlab runtime archive and exported package list: `{self.input_runtime_env_dir}`
- Offline motion and checkpoint archives: `{self.input_runtime_env_dir}/motions-1.zip`
  and `{self.input_runtime_env_dir}/policies.zip`

## What You Must Do
1. Inspect the 8 policy cases in `{self.input_policy_cases}`.
2. Use mjlab and the listed `play_command` or `video_command` values to observe
   each policy against its reference motion. The runtime archive, motion files,
   and policy checkpoints are staged under `{self.input_runtime_env_dir}` if you
   need to install mjlab locally on the VM.
3. For each case, save a visible motion demo artifact. The supplied
   `video_command` adds mjlab's existing `--video` flag; mjlab writes those
   videos under the run/checkpoint log directory, so copy or rename the
   recorded artifact into `{self.output_visual_demos_dir}`.
4. Classify every case as exactly one of `successful`, `nearly_successful`, or
   `failed`, using the definitions in `{self.input_task_brief}`.
5. Save the final JSON report at `{self.output_report}` and one visible motion
   demo artifact per case under `{self.output_visual_demos_dir}`.

## Output Requirements
- The final answer must include `policy_evaluation_report.json` and a
  `visual_demos/` directory.
- It must validate against `{self.input_output_schema}`.
- Include all 8 case IDs exactly once.
- Preserve the exact `case_id`, `motion`, `mjlab_task`, `motion_file`, and `checkpoint_file`
  values from `{self.input_policy_cases}`.
- Each evaluation item must include `evidence.visual_demo_path`, pointing to a
  visible playback artifact under `visual_demos/`. Accepted formats are
  `.mp4`, `.webm`, `.gif`, and `.html`.
- Do not write final answers outside `{self.remote_output_dir}`.
"""

    def to_metadata(self) -> dict:
        metadata = super().to_metadata()
        metadata.update(
            {
                "variant_label": self.VARIANT_LABEL,
                "input_task_brief": self.input_task_brief,
                "input_policy_cases": self.input_policy_cases,
                "input_output_schema": self.input_output_schema,
                "input_runtime_env_dir": self.input_runtime_env_dir,
                "input_runtime_mjlab_zip": self.input_runtime_mjlab_zip,
                "input_runtime_env_export": self.input_runtime_env_export,
                "input_runtime_motions_zip": self.input_runtime_motions_zip,
                "input_runtime_policies_zip": self.input_runtime_policies_zip,
                "output_report": self.output_report,
                "output_visual_demos_dir": self.output_visual_demos_dir,
                "reference_expected_verdicts": self.reference_expected_verdicts,
                "canonical_gcs_root": (
                    f"gs://ale-data-all/{self.DOMAIN_NAME}/{self.TASK_NAME}/{self.VARIANT_NAME}/"
                ),
            }
        )
        return metadata


@cb.tasks_config(split="train")
def load():
    return [
        cb.Task(
            description=HumanoidWbcPolicyEvaluationConfig(
                VARIANT_NAME=variant_name,
                VARIANT_LABEL=variant_label,
            ).task_description,
            metadata=HumanoidWbcPolicyEvaluationConfig(
                VARIANT_NAME=variant_name,
                VARIANT_LABEL=variant_label,
            ).to_metadata(),
            computer={"provider": "computer", "setup_config": {"os_type": "linux"}},
        )
        for variant_name, variant_label in VARIANTS
    ]


@cb.setup_task(split="train")
async def start(task_cfg, session: cb.DesktopSession):
    await _setup(task_cfg, session)


@cb.evaluate_task(split="train")
async def evaluate(task_cfg, session: cb.DesktopSession) -> list[float]:
    meta = task_cfg.metadata
    tag = meta["variant_name"]

    if not (await session.file_exists(meta["reference_expected_verdicts"]) or await session.directory_exists(meta["reference_expected_verdicts"])):
        logger.error("[%s] Missing hidden reference: %s", tag, meta["reference_expected_verdicts"])
        return [0.0]

    if not (await session.file_exists(meta["remote_output_dir"]) or await session.directory_exists(meta["remote_output_dir"])):
        logger.error("[%s] Missing output directory: %s", tag, meta["remote_output_dir"])
        return [0.0]

    entries = sorted(await session.list_dir(meta["remote_output_dir"]))
    if entries != ["policy_evaluation_report.json", "visual_demos"]:
        logger.error(
            "[%s] Output directory contents must be exactly "
            "['policy_evaluation_report.json', 'visual_demos'], found %s",
            tag,
            entries,
        )
        return [0.0]
    if not (await session.file_exists(meta["output_visual_demos_dir"]) or await session.directory_exists(meta["output_visual_demos_dir"])):
        logger.error("[%s] Missing visual demos directory: %s", tag, meta["output_visual_demos_dir"])
        return [0.0]

    with tempfile.TemporaryDirectory(prefix="humanoid_wbc_policy_eval_") as tmp_dir:
        tmp = Path(tmp_dir)
        local_output_dir = tmp / "output"
        local_output_dir.mkdir()
        local_visual_dir = local_output_dir / "visual_demos"
        local_visual_dir.mkdir()
        local_report = tmp / "policy_evaluation_report.json"
        local_reference = tmp / "expected_verdicts.json"
        local_report_in_output = local_output_dir / "policy_evaluation_report.json"
        local_report.write_bytes(await session.read_bytes(meta["output_report"]))
        local_report_in_output.write_bytes(local_report.read_bytes())
        local_reference.write_bytes(await session.read_bytes(meta["reference_expected_verdicts"]))

        try:
            import json

            report = json.loads(local_report.read_text(encoding="utf-8"))
            for item in report.get("evaluations", []):
                demo_path = item.get("evidence", {}).get("visual_demo_path")
                if not isinstance(demo_path, str):
                    continue
                if not demo_path.startswith("visual_demos/") or ".." in demo_path:
                    continue
                remote_demo_path = f"{meta['remote_output_dir']}/{demo_path}"
                local_demo_path = local_output_dir / demo_path
                local_demo_path.parent.mkdir(parents=True, exist_ok=True)
                local_demo_path.write_bytes(await session.read_bytes(remote_demo_path))
        except Exception as exc:
            logger.error("[%s] Failed to collect visual demo artifacts: %s", tag, exc)
            return [0.0]

        result = score_report(local_report_in_output, local_reference, local_output_dir)
        for diagnostic in result.diagnostics:
            logger.info("[%s] %s", tag, diagnostic)
        return [result.score]
