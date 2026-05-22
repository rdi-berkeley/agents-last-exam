"""Linux task entrypoint for healthcare_variant_annotation_pipeline."""

from __future__ import annotations

import json
import logging
import os
import posixpath
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

try:
    import cua_bench as cb
except ModuleNotFoundError:  # pragma: no cover - local fallback only

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

if __name__ not in sys.modules:
    sys.modules[__name__] = sys.modules.get(__name__, type(sys)(__name__))

from tasks.common_setup import BaseTaskSetup
from tasks.linux_runtime import LinuxTaskConfig


_setup = BaseTaskSetup()

SCRIPTS_DIR = Path(__file__).resolve().parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from score_outputs import ScoreResult, score_submission_texts  # noqa: E402

logger = logging.getLogger(__name__)

DOMAIN_NAME = "health_medicine"
TASK_NAME = "healthcare_variant_annotation_pipeline"
TASK_ID = f"{DOMAIN_NAME}/{TASK_NAME}"
VARIANT_NAME = "base"
CANONICAL_OUTPUT_DIR_NAMES = {"output", "output_test_pos", "output_test_neg"}


def _canonical_output_dir_name(path: str) -> str:
    normalized = posixpath.normpath(path.replace("\\", "/"))
    if normalized not in CANONICAL_OUTPUT_DIR_NAMES:
        raise ValueError(
            "REMOTE_OUTPUT_DIR must normalize to one of: output, output_test_pos, output_test_neg"
        )
    return normalized


def _decode_text(payload: Any) -> str:
    return payload.decode("utf-8") if isinstance(payload, bytes) else str(payload)


@dataclass
class HealthcareVariantAnnotationConfig(LinuxTaskConfig):
    DOMAIN_NAME: str = DOMAIN_NAME
    TASK_NAME: str = TASK_NAME
    VARIANT_NAME: str = VARIANT_NAME

    @property
    def output_dir_name(self) -> str:
        return _canonical_output_dir_name(self.REMOTE_OUTPUT_DIR)

    @property
    def remote_output_dir(self) -> str:
        return f"{self.task_dir}/{self.output_dir_name}"

    @property
    def variants_file(self) -> str:
        return f"{self.input_dir}/variants_to_annotate.tsv"

    @property
    def snapshots_file(self) -> str:
        return f"{self.input_dir}/annotation_snapshots/vep_batch_responses.jsonl"

    @property
    def source_manifest_file(self) -> str:
        return f"{self.input_dir}/source_manifest.json"

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
    def annotated_output_file(self) -> str:
        return f"{self.remote_output_dir}/annotated_variants.tsv"

    @property
    def reportable_output_file(self) -> str:
        return f"{self.remote_output_dir}/reportable_variants.tsv"

    @property
    def pipeline_output_file(self) -> str:
        return f"{self.remote_output_dir}/pipeline.py"

    @property
    def run_log_output_file(self) -> str:
        return f"{self.remote_output_dir}/run_log.json"

    @property
    def reference_annotated_file(self) -> str:
        return f"{self.reference_dir}/hidden_truth/annotated_variants.tsv"

    @property
    def reference_reportable_file(self) -> str:
        return f"{self.reference_dir}/hidden_truth/reportable_variants.tsv"

    @property
    def task_description(self) -> str:
        return f"""You are working on a Linux clinical-variant annotation task.

Task directory:
- `{self.task_dir}`

Visible inputs:
- Variant list: `{self.variants_file}`
- Raw VEP snapshots: `{self.snapshots_file}`
- Rule / provenance manifest: `{self.source_manifest_file}`
- Task-local runtime manifest: `{self.runtime_pyproject}`
- Task-local runtime lockfile: `{self.runtime_lockfile}`

Your job:
1. Work from a writable directory under `{self.remote_output_dir}`.
2. Read the 50 staged variants in order.
3. Parse the raw VEP snapshot JSON lines and derive, for each variant:
   - `gene`
   - `consequence`
   - `max_population_af`
   - `clinvar_significance`
   - `is_reportable`
4. Filter the reportable subset exactly according to `{self.source_manifest_file}`.
5. Write these files under `{self.remote_output_dir}`:
   - `annotated_variants.tsv`
   - `reportable_variants.tsv`
   - `pipeline.py`
   - `run_log.json`

Output rules:
- `annotated_variants.tsv` and `reportable_variants.tsv` must use tab-separated columns with this exact header:
  `variant_id chrom pos ref alt gene consequence max_population_af clinvar_significance is_reportable`
- Missing values: use `NA` for missing numeric fields (`max_population_af` when no frequency data exists) and `not_reported` for missing categorical fields (`clinvar_significance` when no ClinVar annotation is available). Do not use `.` or leave cells empty.
- `is_reportable` must be `yes` or `no` (lowercase)
- `reportable_variants.tsv` must contain only rows where `is_reportable=yes`
- `pipeline.py` should be your solver-authored pipeline implementation
- `run_log.json` must be valid UTF-8 JSON with at least these keys:
  `{{"status": "<any non-empty string>", "annotated_variants_written": <int>, "reportable_variants_written": <int>}}`
- Treat `{self.input_dir}` as read-only source data
- Do not use hidden evaluator-owned files

Recommended setup:
- You may work with the system `python`, or create a task-local environment from `{self.runtime_env_dir}` with:
  `uv sync --frozen --project input/runtime_env`

Do not ask for confirmation. Execute directly.
"""

    def to_metadata(self) -> dict[str, Any]:
        metadata = super().to_metadata()
        metadata.pop("software_dir", None)
        metadata.update(
            {
                "task_id": TASK_ID,
                "output_dir_name": self.output_dir_name,
                "variants_file": self.variants_file,
                "snapshots_file": self.snapshots_file,
                "source_manifest_file": self.source_manifest_file,
                "runtime_env_dir": self.runtime_env_dir,
                "runtime_pyproject": self.runtime_pyproject,
                "runtime_lockfile": self.runtime_lockfile,
                "annotated_output_file": self.annotated_output_file,
                "reportable_output_file": self.reportable_output_file,
                "pipeline_output_file": self.pipeline_output_file,
                "run_log_output_file": self.run_log_output_file,
                "reference_annotated_file": self.reference_annotated_file,
                "reference_reportable_file": self.reference_reportable_file,
                "canonical_gcs_root": f"gs://ale-data-all/{DOMAIN_NAME}/{TASK_NAME}/{VARIANT_NAME}/",
            }
        )
        return metadata


config = HealthcareVariantAnnotationConfig()


@cb.tasks_config(split="train")
def load():
    cfg = HealthcareVariantAnnotationConfig(
        REMOTE_OUTPUT_DIR=os.environ.get("REMOTE_OUTPUT_DIR", "output")
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
    try:
        annotated_text = _decode_text(await session.read_bytes(meta["annotated_output_file"]))
        reportable_text = _decode_text(await session.read_bytes(meta["reportable_output_file"]))
        pipeline_text = _decode_text(await session.read_bytes(meta["pipeline_output_file"]))
        run_log_text = _decode_text(await session.read_bytes(meta["run_log_output_file"]))
        reference_annotated_text = _decode_text(
            await session.read_bytes(meta["reference_annotated_file"])
        )
        reference_reportable_text = _decode_text(
            await session.read_bytes(meta["reference_reportable_file"])
        )
    except Exception as exc:
        logger.error("Failed to read required submission/reference files: %s", exc)
        return [0.0]

    try:
        result: ScoreResult = score_submission_texts(
            annotated_tsv_text=annotated_text,
            reportable_tsv_text=reportable_text,
            pipeline_py_text=pipeline_text,
            run_log_text=run_log_text,
            reference_annotated_tsv_text=reference_annotated_text,
            reference_reportable_tsv_text=reference_reportable_text,
        )
    except Exception as exc:
        logger.error("Scoring crashed for %s: %s", meta["variant_name"], exc)
        return [0.0]

    logger.info(
        "[%s] score=%.4f points=%.2f/%.2f passed=%s valid=%s reason=%s",
        meta["variant_name"],
        result.score,
        result.total_points,
        result.max_points,
        result.passed,
        result.valid,
        result.reason,
    )
    logger.info("[%s] scoring_details=%s", meta["variant_name"], json.dumps(result.to_dict()))
    return [result.score]


if __name__ == "__main__":
    for task in load():
        print(task.description)
