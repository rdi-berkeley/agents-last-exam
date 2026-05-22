"""ecg_rhythm_conduction_ptbxl — PTB-XL rhythm/conduction multilabel classification."""

import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cua_bench as cb
from tasks.common_config import GeneralTaskConfig
from tasks.common_setup import BaseTaskSetup


_setup = BaseTaskSetup()

SCRIPTS_DIR = Path(__file__).parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from score_predictions import ScoreResult, score_prediction_tables

logger = logging.getLogger(__name__)

EXPECTED_LABELS = [
    "SR",
    "STACH",
    "SBRAD",
    "SARRH",
    "AFIB",
    "AFLT",
    "PACE",
    "CLBBB",
    "CRBBB",
    "LAFB",
    "1AVB",
    "IRBBB",
    "NORM",
]

VARIANTS = [
    (
        "ptbxl_2263",
        "Single PTB-XL-derived rhythm/conduction benchmark slice with 2,263 anonymized ECG records.",
    )
]


@dataclass
class ECGRhythmConductionConfig(GeneralTaskConfig):
    DOMAIN_NAME: str = "health_medicine"

    TASK_NAME: str = "ecg_rhythm_conduction_ptbxl"
    VARIANT_NAME: str = ""
    VARIANT_LABEL: str = ""

    @property
    def input_dir(self) -> str:
        return rf"{self.task_dir}\input"

    @property
    def records_dir(self) -> str:
        return rf"{self.input_dir}\records"

    @property
    def record_ids_file(self) -> str:
        return rf"{self.input_dir}\record_ids.csv"

    @property
    def label_codebook_file(self) -> str:
        return rf"{self.input_dir}\label_codebook.csv"

    @property
    def template_output_file(self) -> str:
        return rf"{self.input_dir}\template_output.csv"

    @property
    def output_file(self) -> str:
        return rf"{self.remote_output_dir}\predictions.csv"

    @property
    def reference_file(self) -> str:
        return rf"{self.reference_dir}\gold_standard.csv"

    @property
    def open_input_cmd(self) -> str:
        return rf"{self.software_dir}\open_input_cmd.bat"

    @property
    def install_baseline_packages_cmd(self) -> str:
        return rf"{self.software_dir}\install_baseline_python_packages.bat"

    @property
    def task_description(self) -> str:
        label_columns = ", ".join(EXPECTED_LABELS)
        return f"""\
You are a computational precision health researcher working on staged ECG waveform data.

## Variant
`{self.VARIANT_NAME}`: {self.VARIANT_LABEL}

## Your Task
Build a multilabel classifier for rhythm and conduction findings from the staged WFDB ECG records.

## Input Files
- WFDB records directory: `{self.records_dir}`
- Required ECG ID list: `{self.record_ids_file}`
- Label glossary: `{self.label_codebook_file}`
- Output template: `{self.template_output_file}`

## Software
- Task-local command prompt helper: `{self.open_input_cmd}`
- Baseline package installer: `{self.install_baseline_packages_cmd}`
- You may use the system `python` on the VM and install additional open-source Python packages if needed

## What You Must Do
1. Read the staged ECG records from `{self.records_dir}`.
2. Use `{self.record_ids_file}` as the exact required output row set.
3. Train or fit any CPU-feasible multilabel workflow you judge appropriate.
4. Write one CSV exactly to:
   `{self.output_file}`

## Output Requirements
- The file must be named exactly `predictions.csv`
- The columns must be exactly:
  `ecg_id, {label_columns}`
- Include exactly the ECG IDs from `{self.record_ids_file}`
- Every label value must be binary: `0` or `1`
- Do not save extra files into `{self.remote_output_dir}`

## Important Constraints
- Treat this as a multilabel problem; labels are not guaranteed to be mutually exclusive
- Use the staged task files as the source of truth for row set and schema
- Do not modify evaluator-owned directories such as `reference/`, `output_test_pos/`, or `output_test_neg/`
"""

    def to_metadata(self) -> dict[str, Any]:
        metadata = super().to_metadata()
        metadata.update(
            {
                "variant_label": self.VARIANT_LABEL,
                "input_dir": self.input_dir,
                "records_dir": self.records_dir,
                "record_ids_file": self.record_ids_file,
                "label_codebook_file": self.label_codebook_file,
                "template_output_file": self.template_output_file,
                "output_file": self.output_file,
                "reference_file": self.reference_file,
                "open_input_cmd": self.open_input_cmd,
                "install_baseline_packages_cmd": self.install_baseline_packages_cmd,
                "expected_labels": list(EXPECTED_LABELS),
                "score_threshold": 0.70,
            }
        )
        return metadata


def _cfg_for_variant(spec: tuple[str, str]) -> ECGRhythmConductionConfig:
    return ECGRhythmConductionConfig(
        VARIANT_NAME=spec[0],
        VARIANT_LABEL=spec[1],
    )


@cb.tasks_config(split="train")
def load():
    return [
        cb.Task(
            description=_cfg_for_variant(spec).task_description,
            metadata=_cfg_for_variant(spec).to_metadata(),
            computer={"provider": "computer", "setup_config": {"os_type": "windows"}},
        )
        for spec in VARIANTS
    ]


@cb.setup_task(split="train")
async def start(task_cfg, session: cb.DesktopSession):
    await _setup(task_cfg, session)


def _log_score(tag: str, result: ScoreResult) -> None:
    logger.info(
        "[%s] score=%.6f macro_f1=%.6f passed=%s reason=%s",
        tag,
        result.score,
        result.macro_f1,
        result.passed,
        result.reason,
    )


@cb.evaluate_task(split="train")
async def evaluate(task_cfg, session: cb.DesktopSession) -> list[float]:
    meta = task_cfg.metadata
    tag = meta["variant_name"]
    output_file = meta["output_file"]
    reference_file = meta["reference_file"]
    threshold = float(meta["score_threshold"])

    if not await session.exists(output_file):
        logger.error("[%s] Missing output file: %s", tag, output_file)
        return [0.0]
    if not await session.exists(reference_file):
        logger.error("[%s] Missing reference file: %s", tag, reference_file)
        return [0.0]

    try:
        output_csv = await session.read_file(output_file)
        reference_csv = await session.read_file(reference_file)
    except Exception as exc:
        logger.error("[%s] Failed to read output/reference CSV: %s", tag, exc)
        return [0.0]

    result = score_prediction_tables(
        agent_csv=output_csv,
        reference_csv=reference_csv,
        expected_labels=EXPECTED_LABELS,
        threshold=threshold,
    )
    _log_score(tag, result)
    logger.info("[%s] details=%s", tag, json.dumps(result.to_dict(), ensure_ascii=True))
    return [result.score]
