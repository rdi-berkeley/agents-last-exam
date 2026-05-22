"""AgentHLE task: sanding_performance_scoring_instance_1."""

from __future__ import annotations

import importlib.util
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from types import ModuleType
from typing import Any

import cua_bench as cb

from tasks.common_config import GeneralTaskConfig
from tasks.common_setup import BaseTaskSetup



_setup = BaseTaskSetup()

logger = logging.getLogger(__name__)

DOMAIN_NAME = "engineering"
TASK_NAME = "sanding_performance_scoring_instance_1"
TASK_ID = f"{DOMAIN_NAME}/{TASK_NAME}"
VARIANT_NAME = "base"
SCRIPTS_DIR = Path(__file__).resolve().parent / "scripts"
REMOTE_ROOT = r"E:\agenthle"


def _remote_child(base: str, *parts: str) -> str:
    return str(PureWindowsPath(base, *parts))


def _load_verifier() -> ModuleType:
    module_path = SCRIPTS_DIR / "verify_sanding_scores.py"
    spec = importlib.util.spec_from_file_location("verify_sanding_scores", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load verifier from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


async def _read_remote_text(session: cb.DesktopSession, path: str) -> str:
    data = await session.read_file(path)
    if isinstance(data, bytes):
        return data.decode("utf-8-sig", errors="replace")
    return str(data)


@dataclass
class SandingPerformanceScoringConfig(GeneralTaskConfig):
    DOMAIN_NAME: str = DOMAIN_NAME
    TASK_NAME: str = TASK_NAME
    VARIANT_NAME: str = VARIANT_NAME
    REMOTE_ROOT_DIR: str = os.environ.get("REMOTE_ROOT_DIR", REMOTE_ROOT)
    REMOTE_OUTPUT_DIR: str = os.environ.get("REMOTE_OUTPUT_DIR", "output")

    @property
    def input_dir(self) -> str:
        return _remote_child(self.task_dir, "input")

    @property
    def output_file(self) -> str:
        return _remote_child(self.remote_output_dir, "sanding_scores.csv")

    @property
    def reference_file(self) -> str:
        return _remote_child(self.reference_dir, "sanding_scores.csv")

    @property
    def output_test_pos_dir(self) -> str:
        return _remote_child(self.task_dir, "output_test_pos")

    @property
    def output_test_neg_dir(self) -> str:
        return _remote_child(self.task_dir, "output_test_neg")

    @property
    def task_description(self) -> str:
        return f"""\
You are completing a sanding-performance image scoring task on a Windows VM.

Input files are staged in:
`{self.input_dir}`

Read `algorithm_spec.md` in that directory. It defines how to use the reference images
`panel_painted.png` and `panel_unpainted.png` to score:
- `panel_sanded_01.png`
- `panel_sanded_02.png`
- `panel_sanded_03.png`

Use RGB image values and follow the formulas in `algorithm_spec.md`.

Write exactly one output file:
`{self.output_file}`

The CSV must have this header:
`panel_id,quantity_pct,uniformity_pct,composite_pct`

It must contain exactly one row for each sanded panel:
- `panel_sanded_01`
- `panel_sanded_02`
- `panel_sanded_03`

Round the three numeric metric columns to one decimal place. Do not modify the files
under `input/`.
"""

    def to_metadata(self) -> dict[str, Any]:
        metadata = super().to_metadata()
        metadata.update(
            {
                "task_id": TASK_ID,
                "task_dir": self.task_dir,
                "input_dir": self.input_dir,
                "output_file": self.output_file,
                "reference_file": self.reference_file,
                "output_test_pos_dir": self.output_test_pos_dir,
                "output_test_neg_dir": self.output_test_neg_dir,
                "canonical_gcs_root": (
                    "gs://ale-data-all/engineering/sanding_performance_scoring_instance_1/base/"
                ),
            }
        )
        return metadata


config = SandingPerformanceScoringConfig()


@cb.tasks_config(split="train")
def load():
    cfg = SandingPerformanceScoringConfig()
    return [
        cb.Task(
            description=cfg.task_description,
            metadata=cfg.to_metadata(),
            computer={"provider": "computer", "setup_config": {"os_type": "windows"}},
        )
    ]


@cb.setup_task(split="train")
async def start(task_cfg, session: cb.DesktopSession):
    await _setup(task_cfg, session)


@cb.evaluate_task(split="train")
async def evaluate(task_cfg, session: cb.DesktopSession) -> list[float]:
    meta = task_cfg.metadata
    output_file = meta["output_file"]
    reference_file = meta["reference_file"]

    if not await session.file_exists(output_file):
        logger.error("candidate output missing: %s", output_file)
        return [0.0]
    if not await session.file_exists(reference_file):
        logger.error("reference output missing: %s", reference_file)
        return [0.0]

    candidate_text = await _read_remote_text(session, output_file)
    reference_text = await _read_remote_text(session, reference_file)
    verifier = _load_verifier()
    payload = verifier.score_csv_text(candidate_text, reference_text)
    logger.info(
        "sanding score verifier payload: %s", json.dumps(payload, ensure_ascii=False)[:2000]
    )
    return [float(payload.get("score", 0.0))]
