"""Ubuntu-native MERFISH image decoding + segmentation benchmark.

The agent receives a starfish-format MERFISH experiment (one FOV, 8 rounds,
2 channels, 2048x2048) and must produce decoded transcripts, a labeled cell
segmentation, a cell-by-gene count matrix, and a summary quality-metrics file.

Scoring is entirely local: the evaluator reads the agent's four output files
and the hidden reference CSV from the VM via session.read_bytes and runs the
structural + correlation + spatial-concordance gates in-process.
"""

import logging
import posixpath
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cua_bench as cb

from tasks.common_setup import BaseTaskSetup
from tasks.linux_runtime import LinuxTaskConfig

if __name__ not in sys.modules:
    sys.modules[__name__] = sys.modules.get(__name__, type(sys)(__name__))

_setup = BaseTaskSetup()

SCRIPTS_DIR = Path(__file__).resolve().parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from score_merfish import ScoreResult, score  # noqa: E402

logger = logging.getLogger(__name__)

DOMAIN_NAME = "life_sciences"
TASK_NAME = "merfish_image_decoding_segmentation_1"
VARIANT_NAME = "base"
CANONICAL_OUTPUT_DIR_NAMES = {"output", "output_test_pos", "output_test_neg"}


def _canonical_output_dir_name(path: str) -> str:
    normalized = posixpath.normpath(path.replace("\\", "/"))
    if normalized not in CANONICAL_OUTPUT_DIR_NAMES:
        raise ValueError(
            "REMOTE_OUTPUT_DIR must normalize to one of: output, output_test_pos, output_test_neg"
        )
    return normalized


@dataclass
class MerfishTaskConfig(LinuxTaskConfig):
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
    def input_experiment(self) -> str:
        return f"{self.input_dir}/experiment.json"

    @property
    def input_codebook(self) -> str:
        return f"{self.input_dir}/codebook.json"

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
    def python_entrypoint(self) -> str:
        return f"{self.software_dir}/merfish_runtime.sh"

    @property
    def decoded_file(self) -> str:
        return f"{self.remote_output_dir}/decoded_transcripts.csv"

    @property
    def segmentation_file(self) -> str:
        return f"{self.remote_output_dir}/segmentation.tiff"

    @property
    def cell_by_gene_file(self) -> str:
        return f"{self.remote_output_dir}/cell_by_gene.csv"

    @property
    def metrics_file(self) -> str:
        return f"{self.remote_output_dir}/quality_metrics.json"

    @property
    def reference_file(self) -> str:
        return f"{self.reference_dir}/benchmark_results.csv"

    @property
    def task_description(self) -> str:
        return f"""\
You are working on an Ubuntu VM to run the full MERFISH image-analysis pipeline
for one field of view of a human U2OS cell population stained against a
130-gene panel.

Task directory:
- `{self.task_dir}`

Inputs (all under `{self.input_dir}`; treat as read-only):
- `experiment.json` plus `primary_images.json`, `nuclei.json` and the per-FOV
  manifests `primary_images-fov_000.json` / `nuclei-fov_000.json`. These
  describe a starfish Experiment with one FOV, 8 hybridization rounds, and
  2 color channels.
- 16 primary fluorescence TIFFs
  `primary_images-fov_000-Z0-H<h>-C<c>.tiff` (h in 0..7, c in 0..1); each is
  2048x2048.
- 1 DAPI nuclear stain TIFF `nuclei-fov_000-Z0-H0-C0.tiff`.
- `codebook.json` mapping 130 target genes and 10 Blank controls to 16-bit
  MHD4 (Hamming-distance-4 with 1-bit error correction) binary codewords.
- `experiment.json` `extras.scale_factors` gives per-(round, channel)
  normalization scalars.
- `output_contract.json` lists the exact output files, columns, and dtypes
  your pipeline must produce.
- `runtime_env/pyproject.toml`, `runtime_env/uv.lock`, and
  `runtime_env/python-version` pin the Python 3.11 environment that
  starfish 0.3.4 and Cellpose 3.1.1 need. The canonical agent-facing entry
  point `{self.python_entrypoint}` behaves like a Python interpreter, keeps
  its `uv` cache and resolved environment under `{self.remote_output_dir}`,
  and uses `{self.runtime_pyproject}` (with the staged `uv.lock`) as the
  frozen runtime manifest. You can invoke it directly
  (`{self.python_entrypoint} your_script.py`) or materialize the env
  yourself with `uv sync --frozen --project {self.runtime_env_dir}` and
  then run the resulting `.venv/bin/python`.

Recommended workflow:
1. Load the starfish Experiment and sanity-check the image and codebook.
2. Preprocess: high-pass filter or background-subtract, normalize intensity
   using the provided `scale_factors`, and confirm the pre-registered tiles
   are still aligned.
3. Run starfish BlobDetector on each round/channel and extract 16-bit
   intensity traces per spot.
4. Decode with an MHD4-aware decoder (starfish `CheckAll`) with 1-bit error
   correction; filter by magnitude and distance thresholds.
5. Run Cellpose on the DAPI image with a pretrained nuclei model and a
   diameter appropriate for U2OS (roughly 30-50 px).
6. Assign decoded transcripts to the nearest segmented cell; mark transcripts
   outside every cell as extracellular and exclude them from the
   cell-by-gene matrix.
7. Build the cell-by-gene count matrix for the 130 real genes (exclude
   blanks) and compute the quality metrics listed in `output_contract.json`.

Outputs must go into `{self.remote_output_dir}`:
- `decoded_transcripts.csv` with columns `gene, x, y, is_exact, total_magnitude`.
- `segmentation.tiff` - 2048x2048 `uint16` labeled mask (background=0).
- `cell_by_gene.csv` - leading `cell_id` column followed by the 130 real genes
  in codebook order; integer counts.
- `quality_metrics.json` - scalar keys `total_decoded_transcripts`,
  `blank_rate`, `exact_match_fraction`, `n_cells`, `assigned_fraction`,
  `mean_transcripts_per_cell`.

All coordinates (`x`, `y`) in `decoded_transcripts.csv` must be pixel
coordinates within the 2048x2048 FOV so they line up with the segmentation
mask.
"""

    def to_metadata(self) -> dict[str, Any]:
        metadata = super().to_metadata()
        metadata.update(
            {
                "task_id": f"{self.DOMAIN_NAME}/{self.TASK_NAME}",
                "output_dir_name": self.output_dir_name,
                "input_experiment": self.input_experiment,
                "input_codebook": self.input_codebook,
                "runtime_env_dir": self.runtime_env_dir,
                "runtime_pyproject": self.runtime_pyproject,
                "runtime_lockfile": self.runtime_lockfile,
                "python_entrypoint": self.python_entrypoint,
                "decoded_file": self.decoded_file,
                "segmentation_file": self.segmentation_file,
                "cell_by_gene_file": self.cell_by_gene_file,
                "metrics_file": self.metrics_file,
                "reference_file": self.reference_file,
                "canonical_gcs_root": f"gs://ale-data-all/{self.DOMAIN_NAME}/{self.TASK_NAME}/{self.VARIANT_NAME}/",
            }
        )
        return metadata


config = MerfishTaskConfig()


@cb.tasks_config(split="train")
def load():
    cfg = MerfishTaskConfig()
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
        decoded_bytes = await session.read_bytes(meta["decoded_file"])
        seg_bytes = await session.read_bytes(meta["segmentation_file"])
        cbg_bytes = await session.read_bytes(meta["cell_by_gene_file"])
        metrics_bytes = await session.read_bytes(meta["metrics_file"])
    except Exception as exc:
        logger.error("missing one or more agent output files: %s", exc)
        return [0.0]

    try:
        ref_bytes = await session.read_bytes(meta["reference_file"])
        codebook_bytes = await session.read_bytes(meta["input_codebook"])
    except Exception as exc:
        logger.error("missing evaluator-side staged file: %s", exc)
        return [0.0]

    try:
        result: ScoreResult = score(
            decoded_csv_bytes=decoded_bytes,
            segmentation_tiff_bytes=seg_bytes,
            cell_by_gene_csv_bytes=cbg_bytes,
            quality_metrics_json_bytes=metrics_bytes,
            reference_csv_bytes=ref_bytes,
            codebook_json_bytes=codebook_bytes,
        )
    except Exception as exc:
        logger.error("scoring crashed: %s", exc)
        return [0.0]

    if result.hard_failed:
        logger.info("[%s] hard_fail reason=%s", meta["variant_name"], result.hard_fail_reason)
        return [0.0]

    logger.info(
        "[%s] score=%.4f pearson=%s spatial=%s n_cells_seg=%s assigned=%s",
        meta["variant_name"],
        result.score,
        result.pearson_r,
        result.spatial_concordance,
        result.structural.get("n_cells_segmentation"),
        result.structural.get("assigned_fraction"),
    )
    for name, comp in result.components.items():
        logger.info("  %s=%.4f  %s", name, comp.score, comp.detail)
    return [float(result.score)]


if __name__ == "__main__":
    for task in load():
        print(task.description)
