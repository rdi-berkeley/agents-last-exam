"""abb_irb6700_asset_to_urdf_instance_1 — reconstruct an ABB IRB6700 URDF."""

from __future__ import annotations

import json
import logging
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cua_bench as cb

# cua_bench may exec task modules before they are registered in sys.modules.
if __name__ not in sys.modules:
    sys.modules[__name__] = sys.modules.get(__name__, type(sys)(__name__))

from tasks.common_setup import BaseTaskSetup
from tasks.linux_runtime import LinuxTaskConfig
from tasks.engineering.abb_irb6700_asset_to_urdf_instance_1.scripts.score_submission_urdf import (
    evaluate_files,
)


_setup = BaseTaskSetup()

logger = logging.getLogger(__name__)

VARIANTS = [("base", "ABB IRB6700 URDF reconstruction")]


async def _missing(session: cb.DesktopSession, path: str, *, label: str, tag: str) -> bool:
    if (await session.file_exists(path) or await session.directory_exists(path)):
        return False
    logger.error("[%s] Missing %s: %s", tag, label, path)
    return True


@dataclass
class AbbIrb6700UrdfConfig(LinuxTaskConfig):
    DOMAIN_NAME: str = "engineering"
    TASK_NAME: str = "abb_irb6700_asset_to_urdf_instance_1"
    VARIANT_NAME: str = ""
    VARIANT_LABEL: str = ""

    @property
    def input_meshes_dir(self) -> str:
        return f"{self.input_dir}/meshes"

    @property
    def input_metadata_dir(self) -> str:
        return f"{self.input_dir}/metadata"

    @property
    def input_task_brief(self) -> str:
        return f"{self.input_dir}/task_brief.md"

    @property
    def input_link_manifest(self) -> str:
        return f"{self.input_metadata_dir}/link_manifest.json"

    @property
    def input_joint_manifest(self) -> str:
        return f"{self.input_metadata_dir}/joint_manifest.json"

    @property
    def input_tree_hint(self) -> str:
        return f"{self.input_metadata_dir}/kinematic_tree_hint.json"

    @property
    def input_joint_limits(self) -> str:
        return f"{self.input_metadata_dir}/joint_limits.csv"

    @property
    def input_mimic_rules(self) -> str:
        return f"{self.input_metadata_dir}/mimic_rules.json"

    @property
    def output_submission_urdf(self) -> str:
        return f"{self.remote_output_dir}/submission.urdf"

    @property
    def reference_urdf(self) -> str:
        return f"{self.reference_dir}/abb_irb6700_200_260.urdf"

    @property
    def reference_joint_table(self) -> str:
        return f"{self.reference_dir}/gold_joint_table.csv"

    @property
    def reference_link_table(self) -> str:
        return f"{self.reference_dir}/gold_link_mesh_table.csv"

    @property
    def reference_pose_manifest(self) -> str:
        return f"{self.reference_dir}/joint_manifest.json"

    @property
    def task_description(self) -> str:
        return f"""\
You are reconstructing a robot URDF on a Linux VM.

## Variant
`{self.VARIANT_NAME}`: {self.VARIANT_LABEL}

## Input Files
- Mesh assets: `{self.input_meshes_dir}`
- Structural metadata: `{self.input_metadata_dir}`
- Task brief: `{self.input_task_brief}`
- Reference meshes under the staged `meshes/` directory.

## What You Must Do
1. Read `{self.input_task_brief}`.
2. Use the mesh assets and metadata under `{self.input_dir}` to reconstruct the ABB IRB6700 robot as a URDF.
3. Save exactly one final file at `{self.output_submission_urdf}`.

## Output Requirements
- The file must be named `submission.urdf`.
- The file must be valid XML/URDF.
- Preserve the required link names, joint names, kinematic tree, joint limits, mimic rules, and auxiliary frames.
- Do not place the final answer anywhere outside `{self.remote_output_dir}`.
"""

    def to_metadata(self) -> dict:
        metadata = super().to_metadata()
        metadata.pop("software_dir", None)
        metadata.update(
            {
                "variant_label": self.VARIANT_LABEL,
                "input_meshes_dir": self.input_meshes_dir,
                "input_metadata_dir": self.input_metadata_dir,
                "input_task_brief": self.input_task_brief,
                "input_link_manifest": self.input_link_manifest,
                "input_joint_manifest": self.input_joint_manifest,
                "input_tree_hint": self.input_tree_hint,
                "input_joint_limits": self.input_joint_limits,
                "input_mimic_rules": self.input_mimic_rules,
                "output_submission_urdf": self.output_submission_urdf,
                "reference_urdf": self.reference_urdf,
                "reference_joint_table": self.reference_joint_table,
                "reference_link_table": self.reference_link_table,
                "reference_pose_manifest": self.reference_pose_manifest,
                "canonical_gcs_root": (
                    f"gs://ale-data-all/{self.DOMAIN_NAME}/{self.TASK_NAME}/{self.VARIANT_NAME}/"
                ),
            }
        )
        return metadata


@cb.tasks_config(split="train")
def load():
    return [
        cb.Task(
            description=AbbIrb6700UrdfConfig(
                VARIANT_NAME=variant_name,
                VARIANT_LABEL=variant_label,
            ).task_description,
            metadata=AbbIrb6700UrdfConfig(
                VARIANT_NAME=variant_name,
                VARIANT_LABEL=variant_label,
            ).to_metadata(),
            computer={"provider": "computer", "setup_config": {"os_type": "linux"}},
        )
        for variant_name, variant_label in VARIANTS
    ]


@cb.setup_task(split="train")
async def start(task_cfg, session: cb.DesktopSession):
    await _setup(task_cfg, session)


@cb.evaluate_task(split="train")
async def evaluate(task_cfg, session: cb.DesktopSession) -> list[float]:
    meta = task_cfg.metadata
    tag = meta["variant_name"]

    required_hidden_paths = [
        ("reference_urdf", "hidden reference urdf"),
        ("reference_joint_table", "hidden joint table"),
        ("reference_link_table", "hidden link table"),
        ("reference_pose_manifest", "hidden pose manifest"),
    ]
    for key, label in required_hidden_paths:
        if not (await session.file_exists(meta[key]) or await session.directory_exists(meta[key])):
            logger.error("[%s] Missing %s at %s", tag, label, meta[key])
            return [0.0]

    if not (await session.file_exists(meta["remote_output_dir"]) or await session.directory_exists(meta["remote_output_dir"])):
        logger.error("[%s] Missing output directory: %s", tag, meta["remote_output_dir"])
        return [0.0]

    entries = sorted(await session.list_dir(meta["remote_output_dir"]))
    if entries != ["submission.urdf"]:
        logger.error(
            "[%s] Output directory contents must be exactly ['submission.urdf'], found %s",
            tag,
            entries,
        )
        return [0.0]

    if not (await session.file_exists(meta["output_submission_urdf"]) or await session.directory_exists(meta["output_submission_urdf"])):
        logger.error(
            "[%s] Missing output submission.urdf at %s", tag, meta["output_submission_urdf"]
        )
        return [0.0]

    with tempfile.TemporaryDirectory(prefix="abb_irb6700_eval_") as tmp_dir:
        tmp = Path(tmp_dir)
        local_output = tmp / "submission.urdf"
        local_reference_dir = tmp / "reference"
        local_reference_dir.mkdir()
        try:
            local_output.write_bytes(await session.read_bytes(meta["output_submission_urdf"]))
            for key, filename in [
                ("reference_urdf", "abb_irb6700_200_260.urdf"),
                ("reference_joint_table", "gold_joint_table.csv"),
                ("reference_link_table", "gold_link_mesh_table.csv"),
                ("reference_pose_manifest", "joint_manifest.json"),
            ]:
                (local_reference_dir / filename).write_bytes(await session.read_bytes(meta[key]))
            result = evaluate_files(
                output_file=local_output,
                reference_dir=local_reference_dir,
            )
        except Exception as exc:
            logger.exception("[%s] Evaluation failed: %s", tag, exc)
            return [0.0]

    logger.info("[%s] evaluation=%s", tag, json.dumps(result, sort_keys=True))
    return [float(result.get("score", 0.0))]
