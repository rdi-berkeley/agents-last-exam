"""Ubuntu-native TCGA BRCA differential expression benchmark."""

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

TASK_NAME = "tcga_brca_deg_analysis"
VARIANT_NAME = "base"
DOMAIN_NAME = "life_sciences"
CANONICAL_OUTPUT_DIR_NAMES = {"output", "output_test_pos", "output_test_neg"}


def _canonical_output_dir_name(path: str) -> str:
    normalized = posixpath.normpath(path.replace("\\", "/"))
    if normalized not in CANONICAL_OUTPUT_DIR_NAMES:
        raise ValueError(
            "REMOTE_OUTPUT_DIR must normalize to one of: output, output_test_pos, output_test_neg"
        )
    return normalized


class TaskConfig(LinuxTaskConfig):
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
    def output_files(self) -> dict[str, str]:
        return {name: f"{self.remote_output_dir}/{name}" for name in REQUIRED_FILES}

    @property
    def runtime_env_dir(self) -> str:
        return f"{self.input_dir}/runtime_env"

    @property
    def truth_file(self) -> str:
        return f"{self.input_dir}/truth_breast_degs.csv"

    @property
    def reference_truth_file(self) -> str:
        return f"{self.reference_dir}/truth_breast_degs.csv"

    @property
    def gold_deg_results_file(self) -> str:
        return f"{self.reference_dir}/gold_output/deg_results_all.csv"

    @property
    def task_description(self) -> str:
        return f"""You are a bioinformatics analyst performing a TCGA Breast Cancer differential gene expression analysis on Linux.

Task directory:
- `{self.task_dir}`

Visible input files:
- Expression matrix: `{self.input_dir}/expression_matrix.tsv.gz`
- Clinical annotations: `{self.input_dir}/clinical_matrix.tsv`
- Benchmark gene list: `{self.truth_file}`
- Runtime dependency manifest: `{self.runtime_env_dir}/pyproject.toml`
- Requirements file: `{self.input_dir}/requirements.txt`
- Manifest template: `{self.input_dir}/run_manifest_template.json`
- Detailed task brief: `{self.input_dir}/task_description.txt`
- Output contract: `{self.input_dir}/output_requirements.txt`

Recommended setup:
- Work in `{self.task_dir}`
- Install the Python analysis packages into your own environment from the staged input, for example:
  `uv sync --project "{self.runtime_env_dir}"`
  or `pip install -r "{self.input_dir}/requirements.txt"`

Analysis requirements:
1. Match expression matrix sample columns with clinical rows.
2. Compare `Primary Tumor` against `Solid Tissue Normal` using the clinical `sample_type` field.
3. Filter genes where more than 80% of matched samples have expression equal to zero.
4. Run Welch's t-test for each remaining gene.
5. Compute `log2FC = mean(tumor) - mean(normal)`.
6. Apply Benjamini-Hochberg correction.
7. Mark significant DEGs using `abs(log2FC) > 1` and `padj < 0.05`.
8. Generate PCA and volcano plots.
9. Benchmark significant DEGs against `{self.truth_file}`.

Required output directory:
- Write every deliverable under `{self.remote_output_dir}`

Required output files:
- `sample_summary.txt`
- `pca_plot.png`
- `deg_results_all.csv`
- `deg_results_significant.csv`
- `volcano_plot.png`
- `benchmark_summary.csv`
- `run_manifest.json`

Do not modify input files or any non-output task directories.
Do not ask for confirmation. Execute directly.
"""

    def to_metadata(self) -> dict:
        metadata = super().to_metadata()
        metadata.update(
            {
                "task_dir": self.task_dir,
                "data_task_dir": self.data_task_dir,
                "input_dir": self.input_dir,
                "runtime_env_dir": self.runtime_env_dir,
                "output_dir_name": self.output_dir_name,
                "output_files": self.output_files,
                "reference_truth_file": self.reference_truth_file,
                "gold_deg_results_file": self.gold_deg_results_file,
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
        if not await session.exists(path):
            missing.append(name)
            continue
        payloads[name] = await session.read_bytes(path)
    return payloads, missing


@cb.evaluate_task(split="train")
async def evaluate(task_cfg, session: cb.DesktopSession) -> list[float]:
    meta = task_cfg.metadata
    try:
        outputs, missing = await _read_required_output_files(session, meta["output_files"])
        if missing:
            logger.info("Missing output files: %s", missing)
            return [0.0]
        reference_truth = await session.read_bytes(meta["reference_truth_file"])
        gold_deg_results = await session.read_bytes(meta["gold_deg_results_file"])
        report = score_submission(
            outputs,
            reference_truth_csv=reference_truth,
            gold_deg_results_all_csv=gold_deg_results,
        )
        logger.info("Evaluation report: %s", json.dumps(report.to_dict(), sort_keys=True))
        return [report.score]
    except Exception as exc:
        logger.exception("Evaluation failed unexpectedly: %s", exc)
        return [0.0]
