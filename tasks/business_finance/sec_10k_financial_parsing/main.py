"""Task definition for business_finance/sec_10k_financial_parsing."""

from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cua_bench as cb
from tasks.common_setup import BaseTaskSetup
from tasks.linux_runtime import LinuxTaskConfig

_setup = BaseTaskSetup()

logger = logging.getLogger(__name__)

DOMAIN_NAME = "business_finance"
TASK_NAME = "sec_10k_financial_parsing"
TASK_ID = f"{DOMAIN_NAME}/{TASK_NAME}"
VARIANT_NAME = "base"
EVAL_TMP_DIR = "/tmp/agenthle_eval/sec_10k_financial_parsing"

SCRIPTS_DIR = Path(__file__).parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


def _read_script(name: str) -> str:
    return (SCRIPTS_DIR / name).read_text(encoding="utf-8")


def _extract_json_payload(stdout: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    stripped = stdout.strip()
    for start in [idx for idx, char in enumerate(stripped) if char == "{"]:
        try:
            payload, end = decoder.raw_decode(stripped[start:])
        except json.JSONDecodeError:
            continue
        if stripped[start + end :].strip():
            continue
        if isinstance(payload, dict):
            return payload
    raise ValueError("No JSON payload found in verifier stdout")


@dataclass
class Sec10KFinancialParsingConfig(LinuxTaskConfig):
    DOMAIN_NAME: str = DOMAIN_NAME
    TASK_NAME: str = TASK_NAME
    VARIANT_NAME: str = VARIANT_NAME

    @property
    def reference_dir(self) -> str:
        return super().reference_dir

    @property
    def output_test_pos_dir(self) -> str:
        return f"{self.task_dir}/output_test_pos"

    @property
    def output_test_neg_dir(self) -> str:
        return f"{self.task_dir}/output_test_neg"

    @property
    def remote_output_dir(self) -> str:
        if self.REMOTE_OUTPUT_DIR == "output_test_pos":
            return self.output_test_pos_dir
        if self.REMOTE_OUTPUT_DIR == "output_test_neg":
            return self.output_test_neg_dir
        return super().remote_output_dir

    @property
    def filings_dir(self) -> str:
        return f"{self.input_dir}/filings"

    @property
    def filing_manifest(self) -> str:
        return f"{self.filings_dir}/manifest.json"

    @property
    def schema_file(self) -> str:
        return f"{self.input_dir}/schema/extraction_schema.json"

    @property
    def normalization_rules(self) -> str:
        return f"{self.input_dir}/schema/normalization_rules.txt"

    @property
    def questions_file(self) -> str:
        return f"{self.input_dir}/questions.json"

    @property
    def validation_dir(self) -> str:
        return f"{self.input_dir}/validation"

    @property
    def runtime_manifest(self) -> str:
        return f"{self.input_dir}/runtime_env/pyproject.toml"

    @property
    def runtime_lock(self) -> str:
        return f"{self.input_dir}/runtime_env/uv.lock"

    @property
    def python_wrapper(self) -> str:
        return f"{self.software_dir}/python_with_task_deps.sh"

    @property
    def extraction_output_dir(self) -> str:
        return f"{self.remote_output_dir}/extractions"

    @property
    def raw_extractions_output_dir(self) -> str:
        return f"{self.remote_output_dir}/raw_extractions"

    @property
    def qa_output_file(self) -> str:
        return f"{self.remote_output_dir}/qa_answers.json"

    @property
    def run2_output_dir(self) -> str:
        return f"{self.remote_output_dir}/run2_extractions"

    @property
    def reference_ground_truth_dir(self) -> str:
        return f"{self.reference_dir}/ground_truth"

    @property
    def reference_baselines_dir(self) -> str:
        return f"{self.reference_dir}/raw_text_baselines"

    @property
    def reference_qa_file(self) -> str:
        return f"{self.reference_dir}/qa_ground_truth.json"

    @property
    def validation_manifest_file(self) -> str:
        return f"{self.reference_dir}/validation_manifest.json"

    @property
    def cross_validation_expectations_file(self) -> str:
        return f"{self.reference_dir}/cross_validation_expectations.json"

    @property
    def task_description(self) -> str:
        return f"""\
You are working on a Linux VM with a fixed corpus of SEC 10-K PDFs.

Visible files and tools:
- Filing corpus: `{self.filings_dir}`
- Filing metadata: `{self.filing_manifest}`
- Output schema: `{self.schema_file}`
- Normalization rules: `{self.normalization_rules}`
- Analytical questions: `{self.questions_file}`
- Deterministic validation subset: `{self.validation_dir}`
- Python wrapper for the staged task runtime: `{self.python_wrapper}`

Your job:
1. Parse every filing in `input/filings/`.
2. Save one normalized extraction JSON per filing to `{self.extraction_output_dir}`.
3. Save one raw extraction text file per filing to `{self.raw_extractions_output_dir}`.
4. Answer the analytical questions and save them to `{self.qa_output_file}`.
5. Re-run the extraction workflow on the fixed validation subset in `input/validation/`
   and save those second-pass extraction JSONs to `{self.run2_output_dir}`.

Important requirements:
- Follow the staged schema and normalization rules exactly.
- Use raw USD integers for monetary fields and 2-decimal floats for EPS.
- Do not write outside `{self.remote_output_dir}`.
- Do not modify files under `input/`.
- The validation rerun should reflect a second deterministic pass over the same
  visible validation filings.
"""

    def to_metadata(self) -> dict[str, Any]:
        metadata = super().to_metadata()
        metadata.update(
            {
                "task_id": TASK_ID,
                "eval_tmp_dir": EVAL_TMP_DIR,
                "reference_dir": self.reference_dir,
                "reference_gcs_prefix": f"gs://ale-data-all/{TASK_ID}/{VARIANT_NAME}/reference",
                "filings_dir": self.filings_dir,
                "filing_manifest": self.filing_manifest,
                "schema_file": self.schema_file,
                "normalization_rules": self.normalization_rules,
                "questions_file": self.questions_file,
                "validation_dir": self.validation_dir,
                "runtime_manifest": self.runtime_manifest,
                "runtime_lock": self.runtime_lock,
                "python_wrapper": self.python_wrapper,
                "extraction_output_dir": self.extraction_output_dir,
                "raw_extractions_output_dir": self.raw_extractions_output_dir,
                "qa_output_file": self.qa_output_file,
                "run2_output_dir": self.run2_output_dir,
                "reference_ground_truth_dir": self.reference_ground_truth_dir,
                "reference_baselines_dir": self.reference_baselines_dir,
                "reference_qa_file": self.reference_qa_file,
                "validation_manifest_file": self.validation_manifest_file,
                "cross_validation_expectations_file": self.cross_validation_expectations_file,
                "output_test_pos_dir": self.output_test_pos_dir,
                "output_test_neg_dir": self.output_test_neg_dir,
            }
        )
        return metadata


config = Sec10KFinancialParsingConfig()


@cb.tasks_config(split="train")
def load():
    cfg = Sec10KFinancialParsingConfig()
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
    tag = meta["variant_name"]

    await session.makedirs(meta["eval_tmp_dir"])
    required_reference_paths = [
        meta["reference_dir"],
        meta["reference_ground_truth_dir"],
        meta["reference_baselines_dir"],
        meta["reference_qa_file"],
        meta["validation_manifest_file"],
        meta["cross_validation_expectations_file"],
    ]
    missing_reference_paths = [
        path for path in required_reference_paths if not await session.exists(path)
    ]
    if missing_reference_paths:
        logger.error("[%s] staged evaluator reference missing: %s", tag, missing_reference_paths)
        return [0.0]

    verifier_path = f'{meta["eval_tmp_dir"]}/score_outputs.py'
    await session.write_file(verifier_path, _read_script("score_outputs.py"))

    command = (
        f'python "{verifier_path}" '
        f'--predictions "{meta["extraction_output_dir"]}" '
        f'--ground-truth "{meta["reference_ground_truth_dir"]}" '
        f'--raw-extractions "{meta["raw_extractions_output_dir"]}" '
        f'--baselines "{meta["reference_baselines_dir"]}" '
        f'--qa-predictions "{meta["qa_output_file"]}" '
        f'--qa-ground-truth "{meta["reference_qa_file"]}" '
        f'--run2-extractions "{meta["run2_output_dir"]}" '
        f'--validation-manifest "{meta["validation_manifest_file"]}" '
        f'--cross-validation-expectations "{meta["cross_validation_expectations_file"]}"'
    )

    result = await session.run_command(command, check=False)
    stdout = result.get("stdout", "") if isinstance(result, dict) else ""
    stderr = result.get("stderr", "") if isinstance(result, dict) else ""
    rc = result.get("return_code", 1) if isinstance(result, dict) else 1

    if stderr:
        logger.info("[%s] verifier stderr: %s", tag, stderr.strip()[:2000])

    if rc != 0:
        logger.error(
            "[%s] verifier failed rc=%s stdout=%s stderr=%s", tag, rc, stdout[:1000], stderr[:1000]
        )
        return [0.0]

    try:
        payload = _extract_json_payload(stdout)
    except Exception as exc:
        logger.error("[%s] failed to parse verifier payload: %s stdout=%s", tag, exc, stdout[:1500])
        return [0.0]

    score = float(payload.get("score", 0.0))
    logger.info("[%s] score=%.4f components=%s", tag, score, payload.get("component_scores"))
    return [score]
