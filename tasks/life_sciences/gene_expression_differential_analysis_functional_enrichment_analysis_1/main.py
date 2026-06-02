"""Linux benchmark task for BRCA differential expression plus KEGG enrichment."""

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

DOMAIN_NAME = "life_sciences"
TASK_NAME = "gene_expression_differential_analysis_functional_enrichment_analysis_1"
VARIANT_NAME = "base"
CANONICAL_OUTPUT_DIR_NAMES = {"output", "output_test_pos", "output_test_neg"}


def _normalize_output_dir_name(path: str) -> str:
    normalized = posixpath.normpath(path.replace("\\", "/").strip())
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
        return _normalize_output_dir_name(self.REMOTE_OUTPUT_DIR)

    @property
    def remote_output_dir(self) -> str:
        return f"{self.task_dir}/{self.output_dir_name}"

    @property
    def output_files(self) -> dict[str, str]:
        return {name: f"{self.remote_output_dir}/{name}" for name in REQUIRED_FILES}

    @property
    def task_brief_file(self) -> str:
        return f"{self.input_dir}/task_description.txt"

    @property
    def count_matrix_file(self) -> str:
        return f"{self.input_dir}/BRCA_selected_samples_counts.tsv"

    @property
    def metadata_file(self) -> str:
        return f"{self.input_dir}/BRCA_selected_samples_metadata.tsv"

    @property
    def analysis_spec_file(self) -> str:
        return f"{self.input_dir}/analysis_spec.json"

    @property
    def output_contract_file(self) -> str:
        return f"{self.input_dir}/output_contract.json"

    @property
    def gene_map_file(self) -> str:
        return f"{self.input_dir}/gene_id_gene_symbol_map.tsv"

    @property
    def enrichment_config_file(self) -> str:
        return f"{self.input_dir}/enrichment_config.json"

    @property
    def requirements_file(self) -> str:
        return f"{self.input_dir}/requirements.txt"

    @property
    def runtime_env_dir(self) -> str:
        return f"{self.input_dir}/runtime_env"

    @property
    def reference_deg_file(self) -> str:
        return f"{self.reference_dir}/BRCA_deseq2_results.tsv"

    @property
    def reference_up_file(self) -> str:
        return f"{self.reference_dir}/BRCA_upregulated_genes_kegg_enrichment.tsv"

    @property
    def reference_down_file(self) -> str:
        return f"{self.reference_dir}/BRCA_downregulated_genes_kegg_enrichment.tsv"

    @property
    def task_description(self) -> str:
        return f"""You are a bioinformatics analyst working on a Linux VM.

Task directory:
- `{self.task_dir}`

Visible input files:
- Count matrix: `{self.count_matrix_file}`
- Sample metadata: `{self.metadata_file}`
- Analysis spec: `{self.analysis_spec_file}`
- Output contract: `{self.output_contract_file}`
- Detailed task brief: `{self.task_brief_file}`
- Gene-id to gene-symbol map: `{self.gene_map_file}`
- Enrichment config: `{self.enrichment_config_file}`
- Runtime dependency manifest: `{self.runtime_env_dir}/pyproject.toml`
- Requirements file: `{self.requirements_file}`

What you must do:
1. Read the count matrix, metadata, analysis spec, and output contract first.
2. Use the metadata `condition` column to compare `tumor` against `normal`.
3. Run differential gene expression analysis with `pydeseq2` using the design formula `~ batch + condition`.
4. Use `{self.gene_map_file}` to populate the required gene-symbol output fields.
5. Classify each gene as `upregulated`, `downregulated`, or `no significant` using the benchmark rule:
   - `log2FoldChange > 1` and `padj < 0.05` for `upregulated`
   - `log2FoldChange < -1` and `padj < 0.05` for `downregulated`
   - otherwise `no significant`
6. Read `{self.enrichment_config_file}` and run KEGG enrichment separately on the upregulated and downregulated gene sets with `gseapy`.
7. Write exactly these files under `{self.remote_output_dir}`:
   - `BRCA_deseq2_results.tsv`
   - `BRCA_upregulated_genes_kegg_enrichment.tsv`
   - `BRCA_downregulated_genes_kegg_enrichment.tsv`

Helpful notes:
- The first column of the count matrix is `gene_id`.
- The first column of the metadata file is `sample_id`.
- The sample order in the metadata matches the count-matrix sample columns exactly.
- The benchmark standardizes the enrichment target as `KEGG_2021_Human` on `Human`.
- Outbound network access to Enrichr is allowed and expected for the enrichment step.
- You may install the Python dependencies without modifying `input/`, for example:
  `uv venv .venv && uv pip install --python .venv/bin/python -r {self.requirements_file}`

Do not modify files under `input/`.
Do not write outputs anywhere except `{self.remote_output_dir}`.
Do not ask for confirmation. Execute directly.
"""

    def to_metadata(self) -> dict:
        metadata = super().to_metadata()
        metadata.update(
            {
                "task_dir": self.task_dir,
                "data_task_dir": self.data_task_dir,
                "input_dir": self.input_dir,
                "count_matrix_file": self.count_matrix_file,
                "metadata_file": self.metadata_file,
                "analysis_spec_file": self.analysis_spec_file,
                "output_contract_file": self.output_contract_file,
                "task_brief_file": self.task_brief_file,
                "gene_map_file": self.gene_map_file,
                "enrichment_config_file": self.enrichment_config_file,
                "requirements_file": self.requirements_file,
                "runtime_env_dir": self.runtime_env_dir,
                "output_dir_name": self.output_dir_name,
                "output_files": self.output_files,
                "reference_deg_file": self.reference_deg_file,
                "reference_up_file": self.reference_up_file,
                "reference_down_file": self.reference_down_file,
                "canonical_gcs_root": f"gs://ale-data-all/{DOMAIN_NAME}/{TASK_NAME}/{VARIANT_NAME}/",
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
    try:
        outputs, missing = await _read_required_output_files(session, meta["output_files"])
        if missing:
            logger.info("Missing output files: %s", missing)
            return [0.0]

        reference_deg = await session.read_bytes(meta["reference_deg_file"])
        reference_up = await session.read_bytes(meta["reference_up_file"])
        reference_down = await session.read_bytes(meta["reference_down_file"])

        report = score_submission(
            output_payloads=outputs,
            reference_deg_payload=reference_deg,
            reference_up_payload=reference_up,
            reference_down_payload=reference_down,
        )
        logger.info("Evaluation report: %s", json.dumps(report.to_dict(), sort_keys=True))
        return [report.score]
    except Exception as exc:
        logger.exception("Evaluation failed unexpectedly: %s", exc)
        return [0.0]
