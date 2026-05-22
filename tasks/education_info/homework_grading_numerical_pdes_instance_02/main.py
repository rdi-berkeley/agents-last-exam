"""Linux task definition for education_info/homework_grading_numerical_pdes_instance_02."""

from __future__ import annotations

import json
import logging
import os
import posixpath
import sys
import tempfile
from pathlib import Path

import cua_bench as cb

from tasks.common_setup import BaseTaskSetup
from tasks.linux_runtime import LinuxTaskConfig

_setup = BaseTaskSetup()

SCRIPTS_DIR = Path(__file__).resolve().parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from score_outputs import score_submission  # noqa: E402

logger = logging.getLogger(__name__)

DOMAIN_NAME = "education_info"
TASK_NAME = "homework_grading_numerical_pdes_instance_02"
TASK_ID = f"{DOMAIN_NAME}/{TASK_NAME}"
VARIANT_NAME = "base"
CANONICAL_OUTPUT_DIR_NAMES = {"output", "output_test_pos", "output_test_neg"}
FIXTURE_OUTPUT_DIR_NAMES = {"output_test_pos", "output_test_neg"}
REQUIRED_OUTPUT_FILES = [
    "grades.csv",
    "error_tags.csv",
    "per_student_feedback.json",
    "common_mistakes_summary.md",
    "grader_manifest.json",
]


def _canonical_output_dir_name(path: str) -> str:
    normalized = posixpath.normpath(path.replace("\\", "/"))
    if normalized not in CANONICAL_OUTPUT_DIR_NAMES:
        raise ValueError(
            "REMOTE_OUTPUT_DIR must normalize to one of: "
            + ", ".join(sorted(CANONICAL_OUTPUT_DIR_NAMES))
        )
    return normalized


class HomeworkGradingNumericalPDEsConfig(LinuxTaskConfig):
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
        return _canonical_output_dir_name(self.REMOTE_OUTPUT_DIR)

    @property
    def remote_output_dir(self) -> str:
        return f"{self.task_dir}/{self.output_dir_name}"

    @property
    def task_prompt_file(self) -> str:
        return f"{self.input_dir}/TASK_PROMPT.md"

    @property
    def input_readme(self) -> str:
        return f"{self.input_dir}/README.md"

    @property
    def released_dir(self) -> str:
        return f"{self.input_dir}/released"

    @property
    def released_protocol(self) -> str:
        return f"{self.released_dir}/grading_protocol.md"

    @property
    def released_rubric(self) -> str:
        return f"{self.released_dir}/rubric.json"

    @property
    def released_solution_key(self) -> str:
        return f"{self.released_dir}/solution_key.md"

    @property
    def released_submissions_dir(self) -> str:
        return f"{self.released_dir}/submissions"

    @property
    def starter_project_dir(self) -> str:
        return f"{self.input_dir}/starter_project"

    @property
    def starter_readme(self) -> str:
        return f"{self.starter_project_dir}/README.md"

    @property
    def starter_contract(self) -> str:
        return f"{self.starter_project_dir}/output_contract.json"

    @property
    def starter_scaffold(self) -> str:
        return f"{self.starter_project_dir}/grading_scaffold.py"

    @property
    def starter_run_script(self) -> str:
        return f"{self.starter_project_dir}/run_submission.py"

    @property
    def software_readme(self) -> str:
        return f"{self.software_dir}/README.txt"

    @property
    def reference_spec(self) -> str:
        return f"{self.reference_dir}/evaluation_spec.json"

    @property
    def reference_scores(self) -> str:
        return f"{self.reference_dir}/gold_scores.csv"

    @property
    def reference_tags(self) -> str:
        return f"{self.reference_dir}/gold_error_tags.csv"

    @property
    def reference_feedback_requirements(self) -> str:
        return f"{self.reference_dir}/feedback_requirements.json"

    @property
    def reference_summary_requirements(self) -> str:
        return f"{self.reference_dir}/summary_requirements.json"

    @property
    def reference_manifest(self) -> str:
        return f"{self.reference_dir}/source_manifest.json"

    @property
    def task_description(self) -> str:
        return (
            "You are grading synthetic graduate numerical-PDE homework submissions on Linux.\n\n"
            f"## Task Directory\n`{self.task_dir}`\n\n"
            "## Visible Inputs\n"
            f"- Task prompt: `{self.task_prompt_file}`\n"
            f"- Bundle README: `{self.input_readme}`\n"
            f"- Released materials: `{self.released_dir}`\n"
            f"- Starter scaffold: `{self.starter_project_dir}`\n\n"
            "## Your Task\n"
            "1. Read the grading protocol, rubric, solution key, and the five student submissions.\n"
            "2. Grade each student submission part by part.\n"
            "3. Assign rubric-aligned error tags where appropriate.\n"
            "4. Write concise per-student feedback and a short common-mistakes summary.\n"
            "5. Save exactly these files under the active output directory:\n"
            "   - `grades.csv`\n"
            "   - `error_tags.csv`\n"
            "   - `per_student_feedback.json`\n"
            "   - `common_mistakes_summary.md`\n"
            "   - `grader_manifest.json`\n\n"
            "## Rules\n"
            f"- Treat `{self.input_dir}` as read-only.\n"
            f"- Do not read or modify evaluator-only files under `{self.reference_dir}`.\n"
            "- Keep student IDs stable across every output file.\n"
            f"- Use the task-local `{self.software_dir}/python` wrapper if you want to script the grading workflow (it `exec`s the preinstalled system Python).\n"
            f"- Write final deliverables only under `{self.remote_output_dir}`.\n"
        )

    def to_metadata(self) -> dict:
        metadata = super().to_metadata()
        metadata.update(
            {
                "task_id": TASK_ID,
                "variant_name": VARIANT_NAME,
                "output_dir_name": self.output_dir_name,
                "task_prompt_file": self.task_prompt_file,
                "input_readme": self.input_readme,
                "released_dir": self.released_dir,
                "released_protocol": self.released_protocol,
                "released_rubric": self.released_rubric,
                "released_solution_key": self.released_solution_key,
                "released_submissions_dir": self.released_submissions_dir,
                "starter_project_dir": self.starter_project_dir,
                "starter_readme": self.starter_readme,
                "starter_contract": self.starter_contract,
                "starter_scaffold": self.starter_scaffold,
                "starter_run_script": self.starter_run_script,
                "software_readme": self.software_readme,
                "reference_spec": self.reference_spec,
                "reference_scores": self.reference_scores,
                "reference_tags": self.reference_tags,
                "reference_feedback_requirements": self.reference_feedback_requirements,
                "reference_summary_requirements": self.reference_summary_requirements,
                "reference_manifest": self.reference_manifest,
                "required_output_files": REQUIRED_OUTPUT_FILES,
                "canonical_gcs_root": f"gs://ale-data-all/{TASK_ID}/{VARIANT_NAME}/",
            }
        )
        return metadata


@cb.tasks_config(split="train")
def load():
    cfg = HomeworkGradingNumericalPDEsConfig(
        remote_output_dir=os.environ.get("REMOTE_OUTPUT_DIR", "output")
    )
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
    output_dir_name = str(meta["output_dir_name"])
    with tempfile.TemporaryDirectory(prefix="homework_grading_numerical_pdes_") as tmpdir:
        tmp_root = Path(tmpdir)
        submission_dir = tmp_root / "submission"
        reference_dir = tmp_root / "reference"
        submission_dir.mkdir()
        reference_dir.mkdir()

        missing_outputs = []
        for filename in meta["required_output_files"]:
            remote_file = f"{meta['remote_output_dir']}/{filename}"
            if not await session.exists(remote_file):
                missing_outputs.append(filename)
                continue
            (submission_dir / filename).write_bytes(await session.read_bytes(remote_file))

        if missing_outputs:
            if output_dir_name in FIXTURE_OUTPUT_DIR_NAMES:
                raise RuntimeError(
                    "fixture output is missing at evaluation time: " + ", ".join(missing_outputs)
                )
            logger.info("agent output is missing required files: %s", ", ".join(missing_outputs))
            return [0.0]

        for key, remote_file in [
            ("evaluation_spec.json", meta["reference_spec"]),
            ("gold_scores.csv", meta["reference_scores"]),
            ("gold_error_tags.csv", meta["reference_tags"]),
            ("feedback_requirements.json", meta["reference_feedback_requirements"]),
            ("summary_requirements.json", meta["reference_summary_requirements"]),
            ("source_manifest.json", meta["reference_manifest"]),
        ]:
            (reference_dir / key).write_bytes(await session.read_bytes(remote_file))

        report = score_submission(submission_dir=submission_dir, reference_dir=reference_dir)
        logger.info("evaluation report=%s", json.dumps(report, sort_keys=True))
        return [float(report["score"])]


if __name__ == "__main__":
    for task in load():
        print(task.description)
