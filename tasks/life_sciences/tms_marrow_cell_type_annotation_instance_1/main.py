"""Ubuntu-native marrow cell-type annotation benchmark."""

import logging
import posixpath
from dataclasses import dataclass

import cua_bench as cb
from tasks.common_setup import BaseTaskSetup
from tasks.linux_runtime import LinuxTaskConfig
from tasks.life_sciences.tms_marrow_cell_type_annotation_instance_1.eval import (
    CANONICAL_ALLOWED_LABELS,
    evaluate_prediction_submission,
)

logger = logging.getLogger(__name__)
CANONICAL_OUTPUT_DIR_NAMES = {"output", "output_test_pos", "output_test_neg"}


def _canonical_output_dir_name(path: str) -> str:
    normalized = posixpath.normpath(path.replace("\\", "/"))
    if normalized not in CANONICAL_OUTPUT_DIR_NAMES:
        raise ValueError(
            "REMOTE_OUTPUT_DIR must normalize to one of: output, output_test_pos, output_test_neg"
        )
    return normalized


@dataclass
class TaskConfig(LinuxTaskConfig):
    DOMAIN_NAME: str = "life_sciences"
    VARIANT_NAME: str = "base"

    PASS_MACRO_F1: float = 0.75
    PASS_ACCURACY: float = 0.85
    EXPECTED_CELL_COUNT: int = 14517
    EXPECTED_LABEL_COUNT: int = 21

    INPUT_H5AD_FILE: str = "tms_marrow_unlabeled.h5ad"
    INPUT_NOTE_FILE: str = "input.txt"
    ALLOWED_LABELS_FILE: str = "allowed_labels.txt"
    PREDICTIONS_FILE: str = "predictions.csv"
    GROUND_TRUTH_FILE: str = "ground_truth.csv"

    @property
    def data_input_dir(self) -> str:
        return f"{self.data_task_dir}/input"

    @property
    def data_reference_dir(self) -> str:
        return f"{self.data_task_dir}/reference"

    @property
    def data_output_dir(self) -> str:
        return f"{self.data_task_dir}/output"

    @property
    def reference_dir(self) -> str:
        return self.data_reference_dir

    @property
    def input_h5ad_path(self) -> str:
        return f"{self.input_dir}/{self.INPUT_H5AD_FILE}"

    @property
    def input_note_path(self) -> str:
        return f"{self.input_dir}/{self.INPUT_NOTE_FILE}"

    @property
    def allowed_labels_path(self) -> str:
        return f"{self.input_dir}/{self.ALLOWED_LABELS_FILE}"

    @property
    def runtime_env_dir(self) -> str:
        return f"{self.input_dir}/runtime_env"

    @property
    def visible_predictions_path(self) -> str:
        return f"{self.task_dir}/output/{self.PREDICTIONS_FILE}"

    @property
    def remote_output_dir(self) -> str:
        output_dir_name = _canonical_output_dir_name(self.REMOTE_OUTPUT_DIR)
        if output_dir_name in {"output_test_pos", "output_test_neg"}:
            return f"{self.data_task_dir}/{output_dir_name}"
        return self.data_output_dir

    @property
    def predictions_path(self) -> str:
        return f"{self.remote_output_dir}/{self.PREDICTIONS_FILE}"

    @property
    def ground_truth_path(self) -> str:
        return f"{self.data_reference_dir}/{self.GROUND_TRUTH_FILE}"

    @property
    def task_description(self) -> str:
        return f"""You are given an unlabeled Smart-seq2 mouse bone marrow single-cell dataset and must annotate every cell with one label from a fixed 21-class ontology.

Task directory:
- `{self.task_dir}`

Visible inputs:
- AnnData matrix: `{self.input_h5ad_path}`
- Allowed labels: `{self.allowed_labels_path}`
- Agent-facing runtime manifest: `{self.runtime_env_dir}/pyproject.toml`
- Agent-facing runtime lockfile: `{self.runtime_env_dir}/uv.lock`
- Task note: `{self.input_note_path}`

Recommended setup:
- Work on Linux in `{self.task_dir}`
- If you need Scanpy / AnnData tooling, install from the staged runtime with:
  `uv sync --frozen --project "{self.runtime_env_dir}"`

Required output:
- Write exactly one UTF-8 CSV to `{self.visible_predictions_path}`
- Keep the header exactly `cell_id,predicted_cell_type`
- Emit exactly one row for every input cell
- Use only labels from `{self.allowed_labels_path}`

Relevant metadata available in the AnnData object includes:
- `age`
- `mouse.id`
- `sex`
- `n_genes`
- `n_counts`

This task is graded against hidden reference labels. A submission counts as successful only if it is format-valid and achieves both:
- macro F1 >= {self.PASS_MACRO_F1}
- overall accuracy >= {self.PASS_ACCURACY}

Do not ask for confirmation. Execute directly.
"""

    def to_metadata(self) -> dict:
        metadata = super().to_metadata()
        metadata.update(
            {
                "input_h5ad_path": self.input_h5ad_path,
                "input_note_path": self.input_note_path,
                "allowed_labels_path": self.allowed_labels_path,
                "runtime_env_dir": self.runtime_env_dir,
                "output_dir_name": _canonical_output_dir_name(self.REMOTE_OUTPUT_DIR),
                "visible_predictions_path": self.visible_predictions_path,
                "predictions_path": self.predictions_path,
                "ground_truth_path": self.ground_truth_path,
                "pass_macro_f1": self.PASS_MACRO_F1,
                "pass_accuracy": self.PASS_ACCURACY,
                "expected_cell_count": self.EXPECTED_CELL_COUNT,
                "expected_label_count": self.EXPECTED_LABEL_COUNT,
            }
        )
        return metadata


config = TaskConfig(TASK_NAME="tms_marrow_cell_type_annotation_instance_1")


@cb.tasks_config(split="train")
def load():
    return [
        cb.Task(
            description=config.task_description,
            metadata=config.to_metadata(),
            computer={"provider": "computer", "setup_config": {"os_type": config.OS_TYPE}},
        )
    ]


_setup = BaseTaskSetup()


@cb.setup_task(split="train")
async def start(task_cfg, session: cb.DesktopSession):
    await _setup(task_cfg, session)


@cb.evaluate_task(split="train")
async def evaluate(task_cfg, session: cb.DesktopSession) -> list[float]:
    meta = task_cfg.metadata
    if not (await session.file_exists(meta["predictions_path"]) or await session.directory_exists(meta["predictions_path"])):
        logger.error("agent missing output: %s", meta["predictions_path"])
        return [0.0]

    if not (await session.file_exists(meta["ground_truth_path"]) or await session.directory_exists(meta["ground_truth_path"])):
        raise RuntimeError(
            f"evaluator-controlled reference missing: {meta['ground_truth_path']}"
        )

    prediction_bytes = await session.read_bytes(meta["predictions_path"])
    ground_truth_bytes = await session.read_bytes(meta["ground_truth_path"])
    result = evaluate_prediction_submission(
        prediction_bytes,
        ground_truth_bytes,
        allowed_labels=CANONICAL_ALLOWED_LABELS,
        pass_macro_f1=meta["pass_macro_f1"],
        pass_accuracy=meta["pass_accuracy"],
    )
    if not result["valid"]:
        logger.info("Evaluation failed validation: %s", result["error"])
        return [0.0]
    logger.info(
        "Evaluation metrics: macro_f1=%.6f accuracy=%.6f passes=%s",
        result["macro_f1"],
        result["accuracy"],
        result["passes"],
    )
    return [1.0 if result["passes"] else 0.0]
