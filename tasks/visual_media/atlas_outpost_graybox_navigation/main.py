"""atlas_outpost_graybox_navigation — Windows GPU graybox level-design task."""

from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path, PureWindowsPath
from typing import Any

import cua_bench as cb

from tasks.common_config import GeneralTaskConfig
from tasks.common_setup import BaseTaskSetup

_setup = BaseTaskSetup()

if __name__ not in sys.modules:
    sys.modules[__name__] = sys.modules.get(__name__, type(sys)(__name__))

logger = logging.getLogger(__name__)

DOMAIN_NAME = "visual_media"
TASK_NAME = "atlas_outpost_graybox_navigation"
REMOTE_EVAL_TMP_DIR = rf"C:\Users\User\AppData\Local\Temp\agenthle_eval\{TASK_NAME}"
SCRIPTS_DIR = Path(__file__).resolve().parent / "scripts"

VARIANTS = [
    ("base", "Atlas Outpost graybox navigation blockout"),
]


def _read_script_bytes(name: str) -> bytes:
    return (SCRIPTS_DIR / name).read_bytes()


def _win_join(*parts: str) -> str:
    return str(PureWindowsPath(*parts))


def _safe_remote_name(path: str) -> str:
    return (
        path.replace("\\", "_").replace("/", "_").replace(":", "").replace(" ", "_").strip("_")
        or "output"
    )


def _extract_json_payload(stdout: str) -> dict[str, Any]:
    text = (stdout or "").strip()
    if text:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
    for line in reversed([line.strip() for line in stdout.splitlines() if line.strip()]):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    raise ValueError("No JSON payload found in verifier stdout")


@dataclass
class AtlasOutpostConfig(GeneralTaskConfig):
    DOMAIN_NAME: str = DOMAIN_NAME
    TASK_NAME: str = TASK_NAME
    VARIANT_NAME: str = "base"
    VARIANT_LABEL: str = ""
    OS_TYPE: str = "windows"
    REMOTE_OUTPUT_DIR: str = field(
        default_factory=lambda: os.environ.get("REMOTE_OUTPUT_DIR", "output")
    )

    @property
    def task_dir(self) -> str:
        return _win_join(self.REMOTE_ROOT_DIR, self.DOMAIN_NAME, self.TASK_NAME, self.VARIANT_NAME)

    @property
    def input_dir(self) -> str:
        return _win_join(self.task_dir, "input")

    @property
    def software_dir(self) -> str:
        return _win_join(self.task_dir, "software")

    @property
    def remote_output_dir(self) -> str:
        return _win_join(self.task_dir, self.REMOTE_OUTPUT_DIR)

    @property
    def reference_dir(self) -> str:
        return _win_join(self.task_dir, "reference")

    @property
    def input_prompt(self) -> str:
        return _win_join(self.input_dir, "TASK_PROMPT.md")

    @property
    def input_blend(self) -> str:
        return _win_join(self.input_dir, "blender", "atlas_outpost_blockout.blend")

    @property
    def input_gltf(self) -> str:
        return _win_join(self.input_dir, "exports", "atlas_outpost_blockout.gltf")

    @property
    def input_scene(self) -> str:
        return _win_join(self.input_dir, "godot", "scenes", "atlas_outpost_validation.tscn")

    @property
    def output_blend(self) -> str:
        return _win_join(self.remote_output_dir, "blender", "atlas_outpost_blockout.blend")

    @property
    def output_gltf(self) -> str:
        return _win_join(self.remote_output_dir, "exports", "atlas_outpost_blockout.gltf")

    @property
    def output_bin(self) -> str:
        return _win_join(self.remote_output_dir, "exports", "atlas_outpost_blockout.bin")

    @property
    def output_scene(self) -> str:
        return _win_join(self.remote_output_dir, "godot", "scenes", "atlas_outpost_validation.tscn")

    @property
    def output_handoff(self) -> str:
        return _win_join(self.remote_output_dir, "docs", "layout_handoff.md")

    @property
    def blender_launcher(self) -> str:
        return _win_join(self.software_dir, "open_blender.bat")

    @property
    def godot_launcher(self) -> str:
        return _win_join(self.software_dir, "open_godot.bat")

    @property
    def task_description(self) -> str:
        return f"""\
You are working on a Windows GPU VM.

## Variant
`{self.VARIANT_NAME}`: {self.VARIANT_LABEL}

## Your Task
Finish the Atlas Outpost graybox level-design starter project. This is a
level-blockout workflow: author route geometry, re-export the glTF, wire the
Godot validation scene, and write the visible handoff metrics.

## Input Files
- Task prompt: `{self.input_prompt}`
- Starter project root: `{self.input_dir}`
- Blender blockout: `{self.input_blend}`
- Exported glTF starter: `{self.input_gltf}`
- Godot validation scene: `{self.input_scene}`
- Optional Blender launcher: `{self.blender_launcher}`
- Optional Godot launcher: `{self.godot_launcher}`

## What You Must Do
1. Read `{self.input_prompt}` and the visible specs under `{self.input_dir}`.
2. Complete the graybox route layout in the project files.
3. Re-export `exports/atlas_outpost_blockout.gltf` and any referenced sidecar files.
4. Update `godot/scenes/atlas_outpost_validation.tscn` with the required review markers.
5. Fill `docs/layout_handoff.md` with the visible route metrics.
6. Save the completed project under `{self.remote_output_dir}` using the same relative paths as the starter.

## Required Output Paths
- `{self.output_blend}`
- `{self.output_gltf}`
- `{self.output_bin}` if referenced by the exported glTF
- `{self.output_scene}`
- `{self.output_handoff}`

## Important
- Keep your final deliverables under `{self.remote_output_dir}`.
- Do not modify `input\\` or any non-output data.
- Do not rely on internet access.
"""

    def to_metadata(self) -> dict[str, Any]:
        metadata = super().to_metadata()
        metadata.update(
            {
                "variant_label": self.VARIANT_LABEL,
                "task_dir": self.task_dir,
                "input_dir": self.input_dir,
                "input_prompt": self.input_prompt,
                "input_blend": self.input_blend,
                "input_gltf": self.input_gltf,
                "input_scene": self.input_scene,
                "output_blend": self.output_blend,
                "output_gltf": self.output_gltf,
                "output_bin": self.output_bin,
                "output_scene": self.output_scene,
                "output_handoff": self.output_handoff,
                "blender_launcher": self.blender_launcher,
                "godot_launcher": self.godot_launcher,
                "eval_tmp_dir": REMOTE_EVAL_TMP_DIR,
                "canonical_gcs_root": f"gs://ale-data-all/{DOMAIN_NAME}/{TASK_NAME}/{self.VARIANT_NAME}/",
            }
        )
        return metadata


def _cfg_for_variant(variant_name: str, variant_label: str) -> AtlasOutpostConfig:
    return AtlasOutpostConfig(VARIANT_NAME=variant_name, VARIANT_LABEL=variant_label)


@cb.tasks_config(split="train")
def load():
    return [
        cb.Task(
            description=_cfg_for_variant(variant_name, variant_label).task_description,
            metadata=_cfg_for_variant(variant_name, variant_label).to_metadata(),
            computer={
                "provider": "computer",
                "setup_config": {"os_type": _cfg_for_variant(variant_name, variant_label).OS_TYPE},
            },
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

    required_outputs = [
        meta["output_blend"],
        meta["output_gltf"],
        meta["output_scene"],
        meta["output_handoff"],
    ]
    missing = []
    for path in required_outputs:
        if not (await session.file_exists(path) or await session.directory_exists(path)):
            missing.append(path)
    if missing:
        logger.error("[%s] Missing required output paths: %s", tag, missing)
        return [0.0]

    await session.interface.create_dir(meta["eval_tmp_dir"])
    remote_score_script = _win_join(meta["eval_tmp_dir"], "remote_score.py")
    remote_result = _win_join(
        meta["eval_tmp_dir"],
        f'score_{tag}_{_safe_remote_name(meta["remote_output_dir"])}.json',
    )
    await session.write_bytes(remote_score_script, _read_script_bytes("remote_score.py"))

    command = (
        f'python -B "{remote_score_script}" '
        f'--submission-dir "{meta["remote_output_dir"]}" '
        f'--out "{remote_result}"'
    )
    result = await session.run_command(command, check=False)

    stdout = result.get("stdout", "")
    stderr = result.get("stderr", "")
    if stderr:
        logger.info("[%s] remote scorer stderr: %s", tag, stderr[:4000])

    payload: dict[str, Any] | None = None
    if (await session.file_exists(remote_result) or await session.directory_exists(remote_result)):
        try:
            payload = json.loads((await session.read_bytes(remote_result)).decode("utf-8"))
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error("[%s] Failed to read remote score result: %s", tag, exc)

    if payload is None:
        try:
            payload = _extract_json_payload(stdout)
        except Exception as exc:
            logger.error(
                "[%s] remote scorer returned no parseable payload: %s stdout=%s stderr=%s",
                tag,
                exc,
                stdout[:4000],
                stderr[:4000],
            )
            return [0.0]

    score = float(payload.get("score", 0.0))
    logger.info(
        "[%s] score=%.4f raw_total_score=%s passes=%s",
        tag,
        score,
        payload.get("raw_total_score"),
        payload.get("passes"),
    )
    return [score]
