"""AgentHLE task: spatial transcriptomics spatial domain identification."""

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
except ModuleNotFoundError:  # pragma: no cover - local fallback for import checks

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

# cua_bench loads task modules via exec_module without always pre-registering
# them in sys.modules; dataclass needs this for annotation handling.
if __name__ not in sys.modules:
    sys.modules[__name__] = sys.modules.get(__name__, type(sys)(__name__))

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tasks.common_setup import BaseTaskSetup  # noqa: E402
from tasks.linux_runtime import LinuxTaskConfig  # noqa: E402

_setup = BaseTaskSetup()

SCRIPTS_DIR = Path(__file__).resolve().parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from score_spatial_domains import (
    REQUIRED_PNG,
    SLICE_CONFIG,
    ScoreResult,
    score_output_bundle,
)  # noqa: E402

logger = logging.getLogger(__name__)

DOMAIN_NAME = "life_sciences"
TASK_NAME = "spatial_transcriptomics_spatial_domain_identification"
TASK_ID = f"{DOMAIN_NAME}/{TASK_NAME}"
VARIANT_NAME = "base"
ALLOWED_OUTPUT_DIRS = {
    "output",
    "output_test_pos",
    "output_test_neg",
    "output_admin_pos",
    "output_admin_neg",
}
ADMIN_REPLAY_OUTPUT_DIRS = ALLOWED_OUTPUT_DIRS - {"output"}


def _canonical_output_dir_name(path: str) -> str:
    normalized = posixpath.normpath(path.replace("\\", "/")).strip("/")
    if normalized not in ALLOWED_OUTPUT_DIRS:
        raise ValueError(
            "REMOTE_OUTPUT_DIR must normalize to one of: " + ", ".join(sorted(ALLOWED_OUTPUT_DIRS))
        )
    return normalized


def _as_text(payload: Any) -> str:
    return payload.decode("utf-8") if isinstance(payload, bytes) else str(payload)


@dataclass
class SpatialDomainConfig(LinuxTaskConfig):
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
    def data_dir(self) -> str:
        return f"{self.input_dir}/data"

    @property
    def slice_config_file(self) -> str:
        return f"{self.data_dir}/slice_config.csv"

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
    def task_prompt_file(self) -> str:
        return f"{self.input_dir}/task_prompt.md"

    @property
    def software_uv(self) -> str:
        return f"{self.software_dir}/uv"

    @property
    def software_python(self) -> str:
        return f"{self.software_dir}/python3.12"

    @property
    def summary_file(self) -> str:
        return f"{self.remote_output_dir}/summary.csv"

    @property
    def manifest_file(self) -> str:
        return f"{self.remote_output_dir}/manifest.json"

    @property
    def umap_png(self) -> str:
        return f"{self.remote_output_dir}/{REQUIRED_PNG}"

    @property
    def task_description(self) -> str:
        replay_note = ""
        if self.output_dir_name in ADMIN_REPLAY_OUTPUT_DIRS:
            replay_note = (
                "\n\nAdmin replay note: this task instance points at an evaluator fixture directory. "
                "Formal solve runs use the normal `output/` directory and do not expose evaluator "
                "fixture directories to the solving agent."
            )
        return f"""\
You are given 12 human DLPFC 10x Visium slices and must identify spatial domains on every slice.

Task directory:
- `{self.task_dir}`

Visible inputs:
- Slice data directories: `{self.data_dir}/<slice_id>/`
- Per-slice target cluster counts: `{self.slice_config_file}`
- Optional Python runtime manifest: `{self.runtime_pyproject}`
- Optional Python lockfile: `{self.runtime_lock}`
- Baseline uv wrapper: `{self.software_uv}`
- Baseline Python 3.12 wrapper: `{self.software_python}`
- Detailed task prompt: `{self.task_prompt_file}`

Recommended setup:
- Work on Linux in `{self.task_dir}`.
- If you need the staged spatial transcriptomics stack, run:
  `{self.software_uv} sync --frozen --python 3.12 --project {self.runtime_env_dir}`

Required outputs under `{self.remote_output_dir}`:
- `summary.csv` with exact header `slice_id,n_clusters_pred` (one row per slice)
- `manifest.json` — must contain `"has_graph": true`, `"has_embedding": true`, `"has_clustering": true`, an integer `"seed"`, and a nonempty string `"method"`
- `per_slice/<slice_id>_labels.csv` for all 12 slices with exact header `barcode,predicted_label`
- `{REQUIRED_PNG}`

Rules:
- Use the exact per-slice target cluster counts from `{self.slice_config_file}`.
- Treat `{self.input_dir}` as read-only.
- Keep all solver-created files under `{self.remote_output_dir}`.
- Do not use hidden evaluator-owned directories in your workflow.
{replay_note}
"""

    def label_file(self, slice_id: str) -> str:
        return f"{self.remote_output_dir}/per_slice/{slice_id}_labels.csv"

    def annotation_file(self, slice_id: str) -> str:
        return f"{self.reference_dir}/manual_annotations/{slice_id}_manual_annotations.tsv"

    def to_metadata(self) -> dict[str, Any]:
        metadata = super().to_metadata()
        metadata.update(
            {
                "task_id": TASK_ID,
                "output_dir_name": self.output_dir_name,
                "data_dir": self.data_dir,
                "slice_config_file": self.slice_config_file,
                "runtime_env_dir": self.runtime_env_dir,
                "runtime_pyproject": self.runtime_pyproject,
                "runtime_lock": self.runtime_lock,
                "software_uv": self.software_uv,
                "software_python": self.software_python,
                "task_prompt_file": self.task_prompt_file,
                "summary_file": self.summary_file,
                "manifest_file": self.manifest_file,
                "umap_png": self.umap_png,
                "label_files": {
                    slice_id: self.label_file(slice_id) for slice_id, _ in SLICE_CONFIG
                },
                "annotation_files": {
                    slice_id: self.annotation_file(slice_id) for slice_id, _ in SLICE_CONFIG
                },
                "canonical_gcs_root": f"gs://ale-data-all/{TASK_ID}/{self.VARIANT_NAME}/",
            }
        )
        return metadata


config = SpatialDomainConfig(REMOTE_OUTPUT_DIR=os.environ.get("REMOTE_OUTPUT_DIR", "output"))


@cb.tasks_config(split="train")
def load():
    cfg = SpatialDomainConfig(REMOTE_OUTPUT_DIR=os.environ.get("REMOTE_OUTPUT_DIR", "output"))
    return [
        cb.Task(
            description=cfg.task_description,
            metadata=cfg.to_metadata(),
            computer={"provider": "computer", "setup_config": {"os_type": cfg.OS_TYPE}},
        )
    ]


@cb.setup_task(split="train")
async def start(task_cfg, session: cb.DesktopSession):
    await _setup(task_cfg, session)


@cb.evaluate_task(split="train")
async def evaluate(task_cfg, session: cb.DesktopSession) -> list[float]:
    meta = task_cfg.metadata

    required_output_paths = [
        meta["summary_file"],
        meta["manifest_file"],
        meta["umap_png"],
        *meta["label_files"].values(),
    ]
    missing_outputs = [path for path in required_output_paths if not await session.exists(path)]
    if missing_outputs:
        logger.error("missing output files: %s", missing_outputs)
        return [0.0]

    missing_refs = [
        path for path in meta["annotation_files"].values() if not await session.exists(path)
    ]
    if missing_refs:
        raise RuntimeError(
            f"evaluator-controlled reference annotations missing: {missing_refs}"
        )

    per_slice_labels = {
        slice_id: _as_text(await session.read_bytes(path))
        for slice_id, path in meta["label_files"].items()
    }
    reference_annotations = {
        slice_id: _as_text(await session.read_bytes(path))
        for slice_id, path in meta["annotation_files"].items()
    }
    result: ScoreResult = score_output_bundle(
        summary_csv=_as_text(await session.read_bytes(meta["summary_file"])),
        manifest_json=_as_text(await session.read_bytes(meta["manifest_file"])),
        per_slice_labels=per_slice_labels,
        reference_annotations=reference_annotations,
        umap_png=await session.read_bytes(meta["umap_png"]),
    )

    logger.info("[%s] evaluation=%s", TASK_NAME, json.dumps(result.to_dict(), sort_keys=True))
    return [float(result.score)]


if __name__ == "__main__":
    for task in load():
        print(task.description)
