"""Pseudotime differential expression benchmark (Ubuntu-native, Python+R)."""

import hashlib
import json
import logging

import cua_bench as cb
from tasks.common_setup import BaseTaskSetup
from tasks.linux_runtime import LinuxTaskConfig
from tasks.life_sciences.pseudotime_de.scripts.score_outputs import (
    REQUIRED_FILES,
    ReferenceValidationError,
    score_submission,
)

_setup = BaseTaskSetup()

logger = logging.getLogger(__name__)

TASK_NAME = "pseudotime_de"
VARIANT_NAME = "base"
DOMAIN_NAME = "life_sciences"
ESTIMATOR_ID = "palantir-tradeseq-v2-20260910"
REFERENCE_FILENAME = "degs_palantir_tradeseq_v2_20260910.csv"
REFERENCE_SHA256 = "9d56773eb07f243be6b2ca40efa176b4f4eb9b08974f4c1bc7a267b3a5ce45a1"


class TaskConfig(LinuxTaskConfig):
    DOMAIN_NAME: str = DOMAIN_NAME
    TASK_NAME: str = TASK_NAME
    VARIANT_NAME: str = VARIANT_NAME

    @property
    def output_file(self) -> str:
        return f"{self.remote_output_dir}/de_genes.csv"

    @property
    def reference_file(self) -> str:
        return f"{self.reference_dir}/{REFERENCE_FILENAME}"

    @property
    def output_files(self) -> dict[str, str]:
        return {name: f"{self.remote_output_dir}/{name}" for name in REQUIRED_FILES}

    @property
    def task_description(self) -> str:
        return f"""\
You are a computational biologist performing pseudotime differential expression analysis on a bone-marrow scRNA-seq dataset on Linux.

Task directory:
- `{self.task_dir}`

Input files:
- scRNA-seq counts: `{self.input_dir}/marrow_sample_scseq_counts.h5ad`
- Detailed task brief: `{self.input_dir}/task_description.txt`
- Estimator specification: `{self.input_dir}/estimator_contract.json`
- Python dependency manifest: `{self.input_dir}/runtime_env/pyproject.toml`
- R version manifest: `{self.input_dir}/runtime_env/R-versions.json`

Recommended setup:
- Work in `{self.task_dir}`
- Use Python 3.11.15 and the exact staged package versions, for example:
  `uv venv --python 3.11.15 "{self.remote_output_dir}/.venv"`
  followed by `uv pip install --python "{self.remote_output_dir}/.venv/bin/python" -r "{self.input_dir}/runtime_env/pyproject.toml"`
- Use R 4.5.3 with the versions in the R manifest, including tradeSeq 1.24.0, edgeR 4.8.2, mgcv 1.9-4, and SingleCellExperiment 1.32.0. Do not silently substitute newer versions.
- Set PYTHONHASHSEED=0 before Python starts, NumPy seed 0, Scanpy n_jobs=1, and one numeric-library thread per worker. Use no more than four CPU workers.

Analysis pipeline:
1. Read `input/marrow_sample_scseq_counts.h5ad` (4142 cells x 16106 genes, raw counts).
2. Follow estimator `palantir-tradeseq-v2-20260910` in estimator_contract.json. Preserve original cell/gene order and raw counts. On a float64 working copy, normalize_total(target_sum=10000, exclude_highly_expressed=False), natural log1p, select 1500 highly variable genes (flavor='cell_ranger'), and PCA50 on the HVG mask only (zero-centered, ARPACK, random_state=0, float64). No additional filtering, imputation, or unit-variance scaling.
3. Run Palantir: diffusion maps (n_components=5, knn=30, alpha=0, seed=0, kernel_backend='scanpy'), default eigengap multiscale space, neighbors and UMAP with random_state=0 as specified. Then `palantir.core.run_palantir` with start cell `Run5_164698952452459`, terminal states DC=`Run5_131097901611291`, Mono=`Run5_134936662236454`, Ery=`Run4_200562869397916`, num_waypoints=500, knn=30, seed=20, scale_components=True, use_early_cell_as_start=True, max_iterations=25, n_jobs=1. Use returned PResults; do not additionally normalize pseudotime or undo its fate-probability truncation below 0.01. 500 is the requested waypoint count, not the final cardinality.
4. Export the raw-count expression matrix (all genes, not only the 1500 HVGs), pseudotime, fate probabilities, and UMAP from Python to CSV/MTX files under output/. Keep cell identifiers aligned across exports.
5. In R, assemble a SingleCellExperiment from the exported files. Use its raw-count assay with explicit pseudotime and cellWeights for fitGAM; the SCE-input fitGAM method assumes slingshot rather than Palantir.
6. Keep DC and Ery lineages by their terminal-state identities (columns 1 and 3 only if ordered DC, Mono, Ery); drop cells with zero weight in both; renormalize weights row-wise. Use the Palantir pseudotime for both retained lineages.
7. Compute one shared full-gene TMM log-offset on the retained cells using edgeR. Set R RNGkind('Mersenne-Twister','Inversion','Rejection') and R seed 27 immediately before fitGAM(nknots=6), using family='nb', sce=TRUE, parallel=FALSE and default mgcv::gam.control(), without covariates, conditions, or observation weights. Memory-bounded gene batching is allowed with the same full-count offset, cell order, seed-reset lineage assignment, knots and design for every batch; fit all 16106 genes, not just HVGs.
8. Run patternTest(l2fc=log2(1.5)) with global=TRUE, pairwise=FALSE, nPoints=12, eigenThresh=0.01. Set effective p=1 for a non-TRUE tradeSeq converged flag, nonfinite coefficients/covariance, or nonfinite test p-value; record these failures. A batch exception is not permission to skip its genes. Apply one global BH correction across all tested genes (n=16106), never per batch; retain genes with finite padj < 0.05. The effect-size threshold is supplied to patternTest via l2fc=log2(1.5); do not apply a separate absolute logFC or fcMedian filter.

Required output:
- Save to: `{self.remote_output_dir}/de_genes.csv`
- Format: UTF-8 CSV with exactly one column `gene` containing HGNC symbols. Optional UTF-8 BOM, standard CSV quoting, LF/CRLF line endings, blank lines, row order, and duplicate symbols do not affect scoring. Header case and surrounding whitespace are ignored; symbol case is significant. Symbols must be nonempty and contain no embedded whitespace, control characters, commas, or quotes. Empty gene fields, malformed CSV/UTF-8, extra columns (including a row index), and other headers are invalid.
- Scoring: Let A be the deduplicated submitted symbol set, G the hidden reference symbol set, and TP = |A intersect G|. Precision = TP / |A|; recall = TP / |G|. Score is 1.0 only when BOTH precision >= 0.80 AND recall >= 0.80; otherwise 0.0. Missing, empty, or invalid output scores 0.0. Additional nonreference symbols reduce precision; duplicates cannot improve either metric. Metrics are compared before rounding.

Do not modify input files or any non-output task directories.
Do not ask for confirmation. Execute directly.
"""

    def to_metadata(self) -> dict:
        metadata = super().to_metadata()
        metadata.update(
            {
                "output_file": self.output_file,
                "output_files": self.output_files,
                "reference_file": self.reference_file,
                "reference_sha256": REFERENCE_SHA256,
                "estimator_id": ESTIMATOR_ID,
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
            computer={
                "provider": "computer",
                "setup_config": {"os_type": config.OS_TYPE},
            },
        )
    ]


@cb.setup_task(split="train")
async def start(task_cfg, session: cb.DesktopSession):
    await _setup(task_cfg, session)


@cb.evaluate_task(split="train")
async def evaluate(task_cfg, session: cb.DesktopSession) -> list[float]:
    meta = task_cfg.metadata
    if not await session.file_exists(meta["reference_file"]):
        raise RuntimeError(f"evaluator-controlled reference missing: {meta['reference_file']}")
    reference_csv = await session.read_bytes(meta["reference_file"])
    if hashlib.sha256(reference_csv).hexdigest() != REFERENCE_SHA256:
        raise ReferenceValidationError(
            f"Evaluator-controlled reference hash mismatch for {ESTIMATOR_ID}"
        )

    output_files = meta["output_files"]
    payloads: dict[str, bytes] = {}
    for name, path in output_files.items():
        if await session.file_exists(path):
            payloads[name] = await session.read_bytes(path)

    report = score_submission(payloads, reference_csv=reference_csv)
    logger.info("Evaluation report: %s", json.dumps(report.to_dict(), sort_keys=True))
    return [report.score]
