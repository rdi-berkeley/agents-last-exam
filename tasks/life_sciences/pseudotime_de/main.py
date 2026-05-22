"""Pseudotime differential expression benchmark (Ubuntu-native, Python+R)."""

import json
import logging
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

TASK_NAME = "pseudotime_de"
VARIANT_NAME = "base"
DOMAIN_NAME = "life_sciences"


class TaskConfig(LinuxTaskConfig):
    DOMAIN_NAME: str = DOMAIN_NAME
    TASK_NAME: str = TASK_NAME
    VARIANT_NAME: str = VARIANT_NAME

    @property
    def output_file(self) -> str:
        return f"{self.remote_output_dir}/de_genes.csv"

    @property
    def reference_file(self) -> str:
        return f"{self.reference_dir}/degs.csv"

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
- Python dependency manifest: `{self.input_dir}/runtime_env/pyproject.toml`

Recommended setup:
- Work in `{self.task_dir}`
- Install the Python analysis packages from the staged manifest, for example:
  `uv sync --project "{self.input_dir}/runtime_env"`
  or `pip install anndata scanpy palantir pandas "numpy<2" scipy scikit-learn fa2-modified leidenalg`
- R packages `SingleCellExperiment` and `tradeSeq` must be available via `Rscript`.

Analysis pipeline:
1. Read `input/marrow_sample_scseq_counts.h5ad` (4142 cells x 16106 genes, raw counts).
2. Preprocess: normalize per cell, log-transform, select 1500 highly variable genes (flavor='cell_ranger'), PCA.
3. Run Palantir: diffusion maps (n_components=5), multiscale space, neighbors, UMAP, then `palantir.core.run_palantir` with start cell `Run5_164698952452459`, terminal states DC=`Run5_131097901611291`, Mono=`Run5_134936662236454`, Ery=`Run4_200562869397916`, num_waypoints=500.
4. Export expression matrix, pseudotime, fate probabilities, and UMAP from Python to CSV/MTX files.
5. In R, assemble a SingleCellExperiment from the exported files.
6. Keep DC (column 1) and Ery (column 3) lineages; drop cells with zero weight in both; renormalize weights row-wise.
7. Set R seed 27, run fitGAM(nknots=6), then patternTest(l2fc=log2(1.5)).
8. Apply BH correction; filter genes with padj < 0.05.

Required output:
- Save to: `{self.remote_output_dir}/de_genes.csv`
- Format: CSV with a single column `gene` containing HGNC symbols (one per row, deduplicated, order-independent).
- Success criterion: at least 80% symbol-set overlap with a hidden gold-standard gene list.

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
    output_files = meta["output_files"]
    payloads: dict[str, bytes] = {}
    missing: list[str] = []
    for name, path in output_files.items():
        if not await session.exists(path):
            missing.append(name)
            continue
        payloads[name] = await session.read_bytes(path)

    if missing:
        logger.info("Missing output files: %s", missing)
        return [0.0]

    if not await session.exists(meta["reference_file"]):
        raise RuntimeError(f"evaluator-controlled reference missing: {meta['reference_file']}")

    reference_csv = await session.read_bytes(meta["reference_file"])
    report = score_submission(payloads, reference_csv=reference_csv)
    logger.info("Evaluation report: %s", json.dumps(report.to_dict(), sort_keys=True))
    return [report.score]
