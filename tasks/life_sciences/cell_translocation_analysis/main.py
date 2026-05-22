"""AgentHLE task: cell translocation analysis from microscopy images."""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

try:
    import cua_bench as cb
except ModuleNotFoundError:  # pragma: no cover - local import fallback only

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

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tasks.common_setup import BaseTaskSetup  # noqa: E402
from tasks.linux_runtime import LinuxTaskConfig  # noqa: E402

SCRIPTS_DIR = Path(__file__).resolve().parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from score_outputs import CSV_NAMES, ScoreResult, score_output_bundle  # noqa: E402

_setup = BaseTaskSetup()

logger = logging.getLogger(__name__)

DOMAIN_NAME = "life_sciences"
TASK_NAME = "cell_translocation_analysis"
TASK_ID = f"{DOMAIN_NAME}/{TASK_NAME}"
VARIANT_NAME = "base"
ALLOWED_OUTPUT_DIRS = {"output", "output_test_pos", "output_test_neg"}


def _canonical_output_dir_name(path: str) -> str:
    normalized = path.replace("\\", "/").strip("/")
    if normalized not in ALLOWED_OUTPUT_DIRS:
        raise ValueError(
            "REMOTE_OUTPUT_DIR must be one of: " + ", ".join(sorted(ALLOWED_OUTPUT_DIRS))
        )
    return normalized


def _as_text(payload: Any) -> str:
    return payload.decode("utf-8") if isinstance(payload, bytes) else str(payload)


async def _run_command(
    session: cb.DesktopSession,
    command: str,
    *,
    check: bool = False,
    timeout: float | None = None,
) -> dict[str, Any]:
    try:
        if timeout is not None:
            return await session.run_command(command, check=check, timeout=timeout)
        return await session.run_command(command, check=check)
    except TypeError:
        return await session.run_command(command, check=check)


class CellTranslocationConfig(LinuxTaskConfig):
    def __init__(self, remote_output_dir: str = "output") -> None:
        super().__init__(
            DOMAIN_NAME=DOMAIN_NAME,
            TASK_NAME=TASK_NAME,
            VARIANT_NAME=VARIANT_NAME,
            OS_TYPE="linux",
            REMOTE_OUTPUT_DIR=remote_output_dir,
        )

    @property
    def images_dir(self) -> str:
        return f"{self.input_dir}/images"

    @property
    def metadata_file(self) -> str:
        return f"{self.images_dir}/Translocation_doses_and_controls.csv"

    @property
    def instructions_file(self) -> str:
        return f"{self.input_dir}/task_instructions.md"

    @property
    def runtime_env_dir(self) -> str:
        return f"{self.input_dir}/runtime_env"

    @property
    def runtime_pyproject(self) -> str:
        return f"{self.runtime_env_dir}/pyproject.toml"

    @property
    def runtime_lock(self) -> str:
        return f"{self.runtime_env_dir}/uv.lock"

    @property
    def python_wrapper(self) -> str:
        return f"{self.software_dir}/python_cell_translocation.sh"

    @property
    def remote_output_dir(self) -> str:
        output_dir_name = _canonical_output_dir_name(self.REMOTE_OUTPUT_DIR)
        return f"{self.task_dir}/{output_dir_name}"

    @property
    def answer_output(self) -> str:
        return f"{self.remote_output_dir}/answer.json"

    @property
    def answer_reference(self) -> str:
        return f"{self.reference_dir}/answer.json"

    def output_csv_path(self, name: str) -> str:
        return f"{self.remote_output_dir}/{name}"

    def reference_csv_path(self, name: str) -> str:
        return f"{self.reference_dir}/{name}"

    @property
    def task_description(self) -> str:
        return f"""You are analyzing two-channel cell microscopy images for GFP nuclear translocation.

Task directory:
- `{self.task_dir}`

Visible inputs:
- Paired TIFF images: `{self.images_dir}`
- Dose and control metadata: `{self.metadata_file}`
- Detailed task instructions: `{self.instructions_file}`
- Optional Python dependency manifest: `{self.runtime_pyproject}`
- Optional Python wrapper: `{self.python_wrapper}`

Recommended setup:
- Work on Linux in `{self.task_dir}`.
- If you want the staged Python image-analysis stack, run scripts through `{self.python_wrapper}`. The wrapper stores dependency state under the writable output directory.

Your task:
1. Pair the DNA channel (`w1`) and GFP channel (`w2`) images by well.
2. Segment nuclei from the DNA channel and derive cell/cytoplasm compartments.
3. Measure GFP intensity, channel correlation, location, and ratio-style features for the relevant objects.
4. Use the dose/control metadata to classify positive translocation and identify the minimum effective dose.
5. Save exactly these files under `{self.remote_output_dir}`:
   - `Cells.csv`
   - `Cytoplasm.csv`
   - `Nuclei.csv`
   - `answer.json`

`answer.json` must be a JSON object with numeric keys `minimum_dose` and `positive_percentage`.

Do not modify files under `input/`. Write final results only under `{self.remote_output_dir}`.
"""

    def to_metadata(self) -> dict[str, Any]:
        metadata = super().to_metadata()
        metadata.update(
            {
                "task_id": TASK_ID,
                "images_dir": self.images_dir,
                "metadata_file": self.metadata_file,
                "instructions_file": self.instructions_file,
                "runtime_env_dir": self.runtime_env_dir,
                "runtime_pyproject": self.runtime_pyproject,
                "runtime_lock": self.runtime_lock,
                "python_wrapper": self.python_wrapper,
                "answer_output": self.answer_output,
                "answer_reference": self.answer_reference,
                "output_csvs": {name: self.output_csv_path(name) for name in CSV_NAMES},
                "reference_csvs": {name: self.reference_csv_path(name) for name in CSV_NAMES},
                "output_dir_name": _canonical_output_dir_name(self.REMOTE_OUTPUT_DIR),
                "canonical_gcs_root": f"gs://ale-data-all/{TASK_ID}/{self.VARIANT_NAME}/",
            }
        )
        return metadata


config = CellTranslocationConfig(remote_output_dir=os.environ.get("REMOTE_OUTPUT_DIR", "output"))


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


def _log_score(result: ScoreResult) -> None:
    logger.info(
        "[%s] score=%.6f passed=%s reason=%s hard_gate=%s",
        TASK_NAME,
        result.score,
        result.passed,
        result.reason,
        result.hard_gate,
    )
    logger.info("[%s] details=%s", TASK_NAME, json.dumps(result.to_dict(), ensure_ascii=False))


@cb.evaluate_task(split="train")
async def evaluate(task_cfg, session: cb.DesktopSession) -> list[float]:
    meta = task_cfg.metadata
    required_outputs = [meta["answer_output"], *meta["output_csvs"].values()]
    missing_outputs = [path for path in required_outputs if not await session.exists(path)]
    if missing_outputs:
        logger.error("[%s] missing outputs: %s", TASK_NAME, missing_outputs)
        return [0.0]

    evaluator_paths = [meta["answer_reference"], *meta["reference_csvs"].values()]
    missing_refs = [path for path in evaluator_paths if not await session.exists(path)]
    if missing_refs:
        raise RuntimeError(
            f"evaluator-controlled references missing: {missing_refs}"
        )

    output_csvs = {
        name: _as_text(await session.read_file(path))
        for name, path in meta["output_csvs"].items()
    }
    reference_csvs = {
        name: _as_text(await session.read_file(path))
        for name, path in meta["reference_csvs"].items()
    }
    result = score_output_bundle(
        answer_json=_as_text(await session.read_file(meta["answer_output"])),
        reference_answer_json=_as_text(await session.read_file(meta["answer_reference"])),
        output_csvs=output_csvs,
        reference_csvs=reference_csvs,
    )

    _log_score(result)
    return [result.score]


if __name__ == "__main__":
    for task in load():
        print(task.description)
