"""Ubuntu-native TCGA LUAD KRAS survival benchmark."""

from __future__ import annotations

import json
import logging
import posixpath
import sys
from pathlib import Path

import cua_bench as cb
from tasks.common_setup import BaseTaskSetup
from tasks.linux_runtime import LinuxTaskConfig

_setup = BaseTaskSetup()

SCRIPTS_DIR = Path(__file__).parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from score_outputs import REQUIRED_FILES, score_submission  # noqa: E402

logger = logging.getLogger(__name__)

TASK_NAME = "healthcare_tcga_luad_survival_kras"
VARIANT_NAME = "base"
DOMAIN_NAME = "health_medicine"
CANONICAL_OUTPUT_DIR_NAMES = {"output", "output_test_pos", "output_test_neg"}


def _canonical_output_dir_name(path: str) -> str:
    normalized = posixpath.normpath(path.replace("\\", "/"))
    if normalized not in CANONICAL_OUTPUT_DIR_NAMES:
        raise ValueError(
            "OUTPUT_SUBDIR must normalize to one of: output, output_test_pos, output_test_neg"
        )
    return normalized


class TaskConfig(LinuxTaskConfig):
    DOMAIN_NAME: str = DOMAIN_NAME
    TASK_NAME: str = TASK_NAME
    VARIANT_NAME: str = VARIANT_NAME

    @property
    def output_dir_name(self) -> str:
        return _canonical_output_dir_name(self.OUTPUT_SUBDIR)

    @property
    def output_dir(self) -> str:
        return f"{self.task_dir}/{self.output_dir_name}"

    @property
    def output_files(self) -> dict[str, str]:
        return {name: f"{self.output_dir}/{name}" for name in REQUIRED_FILES}

    @property
    def task_spec_file(self) -> str:
        return f"{self.input_dir}/task_spec.json"

    @property
    def task_description_file(self) -> str:
        return f"{self.input_dir}/task_description.txt"

    @property
    def output_requirements_file(self) -> str:
        return f"{self.input_dir}/output_requirements.txt"

    @property
    def runtime_notes_file(self) -> str:
        return f"{self.input_dir}/runtime_notes.txt"

    @property
    def software_readme_file(self) -> str:
        return f"{self.software_dir}/README.txt"

    @property
    def reference_cohort_file(self) -> str:
        return f"{self.reference_dir}/reference_outputs/cohort.csv"

    @property
    def reference_cox_file(self) -> str:
        return f"{self.reference_dir}/reference_outputs/cox_results.json"

    @property
    def evaluation_contract_file(self) -> str:
        return f"{self.reference_dir}/evaluation_contract.json"

    @property
    def task_description(self) -> str:
        return f"""You are a bioinformatics analyst performing a TCGA-LUAD KRAS survival analysis on Linux.

Task directory:
- `{self.task_dir}`

Visible files and directories:
- Task specification: `{self.task_spec_file}`
- Detailed instructions: `{self.task_description_file}`
- Output contract: `{self.output_requirements_file}`
- Runtime notes: `{self.runtime_notes_file}`
- Software notes: `{self.software_readme_file}`

What you need to do:
1. Use public TCGA-LUAD data from the GDC Data Portal.
2. Download the clinical data and STAR-count expression data for the LUAD cohort.
3. Extract KRAS (`ENSG00000133703`) expression using the `tpm_unstranded` metric (not raw counts).
4. When a patient has multiple primary-tumor samples, keep only the one whose sample submitter_id is lexicographically smallest.
5. Merge expression with survival and clinical covariates, and stratify patients into high/low KRAS groups using the cohort median.
6. Fit a Kaplan-Meier analysis, a log-rank test, and a Cox proportional hazards model with KRAS group, age, and stage grouping.
7. Save every required output under `{self.output_dir}`.

Required output files:
- `cohort.csv`
- `km_plot.png`
- `cox_results.json`
- `analysis.R`

Do not modify staged inputs or evaluator-only directories.
Do not ask for confirmation. Execute directly.
"""

    def to_metadata(self) -> dict:
        metadata = super().to_metadata()
        metadata.update(
            {
                "task_dir": self.task_dir,
                "input_dir": self.input_dir,
                "software_dir": self.software_dir,
                "output_dir": self.output_dir,
                "output_dir_name": self.output_dir_name,
                "output_files": self.output_files,
                "task_spec_file": self.task_spec_file,
                "task_description_file": self.task_description_file,
                "output_requirements_file": self.output_requirements_file,
                "runtime_notes_file": self.runtime_notes_file,
                "software_readme_file": self.software_readme_file,
                "reference_cohort_file": self.reference_cohort_file,
                "reference_cox_file": self.reference_cox_file,
                "evaluation_contract_file": self.evaluation_contract_file,
                "canonical_gcs_root": (f"gs://ale-data-all/{DOMAIN_NAME}/{TASK_NAME}/{VARIANT_NAME}/"),
            }
        )
        return metadata


config = TaskConfig(DOMAIN_NAME=DOMAIN_NAME, TASK_NAME=TASK_NAME, VARIANT_NAME=VARIANT_NAME)


@cb.tasks_config(split="train")
def load():
    return [
        cb.Task(
            description=config.task_description,
            metadata=config.to_metadata(),
            computer={"provider": "computer", "setup_config": {"os_type": config.OS_TYPE}},
        )
    ]


@cb.setup_task(split="train")
async def start(task_cfg, session: cb.DesktopSession):
    await _setup(task_cfg, session)


async def _read_required_output_files(session: cb.DesktopSession, output_files: dict[str, str]):
    payloads: dict[str, bytes] = {}
    missing: list[str] = []
    for name, path in output_files.items():
        if not (await session.file_exists(path) or await session.directory_exists(path)):
            missing.append(name)
            continue
        payloads[name] = await session.read_bytes(path)
    return payloads, missing


@cb.evaluate_task(split="train")
async def evaluate(task_cfg, session: cb.DesktopSession) -> list[float]:
    meta = task_cfg.metadata
    outputs, missing = await _read_required_output_files(session, meta["output_files"])
    if missing:
        logger.info("Missing output files: %s", missing)
        return [0.0]

    for ref_key in ("reference_cohort_file", "reference_cox_file", "evaluation_contract_file"):
        ref_path = meta[ref_key]
        if not (await session.file_exists(ref_path) or await session.directory_exists(ref_path)):
            raise RuntimeError(f"evaluator-controlled reference missing: {ref_key}={ref_path}")

    reference_cohort = await session.read_bytes(meta["reference_cohort_file"])
    reference_cox = await session.read_bytes(meta["reference_cox_file"])
    evaluation_contract = await session.read_bytes(meta["evaluation_contract_file"])
    report = score_submission(
        outputs,
        reference_cohort_csv=reference_cohort,
        reference_cox_json=reference_cox,
        evaluation_contract_json=evaluation_contract,
    )
    logger.info("Evaluation report: %s", json.dumps(report.to_dict(), sort_keys=True))
    return [report.score]
