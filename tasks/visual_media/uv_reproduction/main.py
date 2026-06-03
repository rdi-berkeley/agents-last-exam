from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import textwrap
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Any

import cua_bench as cb

from tasks.common_config import GeneralTaskConfig
from tasks.common_setup import BaseTaskSetup
from tasks.visual_media.uv_reproduction.scripts.local_soft_eval import \
    run_local_soft_eval

_setup = BaseTaskSetup()

logger = logging.getLogger(__name__)

TASK_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = TASK_DIR / "scripts"
WORKFLOW = "uv_reproduction"
REMOTE_PYTHON = os.environ.get(
    "BLENDER_TASK_REMOTE_PYTHON",
    r"C:\Users\User\AppData\Local\Programs\Python\Python312\python.exe",
)
REMOTE_BLENDER = os.environ.get(
    "BLENDER_TASK_REMOTE_BLENDER", r"C:\Program Files\Blender Foundation\Blender 5.0\blender.exe"
)
REMOTE_TEMP_ROOT = os.environ.get(
    "BLENDER_TASK_REMOTE_EVAL_ROOT",
    r"C:\Users\User\AppData\Local\Temp\agenthle_eval\uv_reproduction",
)

VARIANTS = [
    {
        "task_tag": "anime_singing_girl_uv_reproduction",
        "display_name": "Anime Singing Girl UV Reproduction",
        "remote_task_dir_name": "uv_reproduction_anime_singing_girl",
        "task_shape": "single_part",
        "submission_contract": "object_bundle",
        "mode": "material_reproduction",
        "input_obj_name": "singing_raw.obj",
        "input_mtl_name": None,
        "reference_image_dir_name": "reference_images",
        "submission_obj_name": "singing.obj",
        "submission_mtl_name": "material.mtl",
        "reference_mtl_name": "singing.mtl",
        "reference_texture_subpath": ("objects", "textures"),
        "requires_material_files": True,
        "requires_uv": True,
        "requires_mtl": True,
        "requires_basecolor_texture": True,
        "requires_color_match_gate": True,
        "color_match_gate_threshold": 0.9,
    },
    {
        "task_tag": "samurai_uv_reproduction",
        "display_name": "Samurai UV Reproduction",
        "remote_task_dir_name": "uv_reproduction_samurai",
        "task_shape": "single_part",
        "submission_contract": "object_bundle",
        "mode": "material_reproduction",
        "input_obj_name": "samurai.obj",
        "input_mtl_name": "samurai.mtl",
        "reference_image_dir_name": "reference_images",
        "submission_obj_name": "samurai.obj",
        "submission_mtl_name": "material.mtl",
        "reference_mtl_name": "samurai.mtl",
        "reference_texture_subpath": ("objects", "textures"),
        "requires_material_files": True,
        "requires_uv": True,
        "requires_mtl": True,
        "requires_basecolor_texture": True,
        "requires_color_match_gate": True,
        "color_match_gate_threshold": 0.9,
    },
    {
        "task_tag": "uv_reproduction_muji_paper_bag",
        "display_name": "MUJI Paper Bag UV Reproduction",
        "remote_task_dir_name": "uv_reproduction_muji_paper_bag",
        "task_shape": "single_part",
        "submission_contract": "object_bundle",
        "mode": "material_reproduction",
        "input_obj_name": "muji_paper_bag_raw.obj",
        "input_mtl_name": None,
        "reference_image_dir_name": "reference_images",
        "reference_obj_name": "muji_paper_bag.obj",
        "submission_obj_name": "muji_paper_bag.obj",
        "submission_mtl_name": "material.mtl",
        "requires_material_files": True,
        "requires_uv": True,
        "requires_mtl": True,
        "requires_basecolor_texture": True,
        "requires_color_match_gate": True,
        "color_match_gate_threshold": 0.9,
    },
    {
        "task_tag": "stylized_panda",
        "display_name": "Stylized Panda UV + Material Reproduction",
        "remote_task_dir_name": "uv_material_reproduction_stylized_panda",
        "task_shape": "single_part",
        "submission_contract": "object_bundle",
        "mode": "material_reproduction",
        "input_obj_name": "panda_raw.obj",
        "input_mtl_name": None,
        "reference_image_dir_name": "reference_images",
        "reference_obj_name": "panda.obj",
        "submission_obj_name": "panda.obj",
        "submission_mtl_name": "material.mtl",
        "requires_material_files": True,
        "requires_uv": True,
        "requires_mtl": True,
        "requires_basecolor_texture": True,
        "requires_color_match_gate": True,
        "color_match_gate_threshold": 0.9,
    },
    {
        "task_tag": "uv_material_reproduction_leather_backpack",
        "display_name": "Leather Backpack UV + Material Reproduction",
        "remote_task_dir_name": "uv_material_reproduction_leather_backpack",
        "task_shape": "single_part",
        "submission_contract": "object_bundle",
        "mode": "material_reproduction",
        "input_obj_name": "leather_backpack_raw.obj",
        "input_mtl_name": None,
        "reference_image_dir_name": "reference_images",
        "reference_obj_name": "leather_backpack.obj",
        "submission_obj_name": "leather_backpack.obj",
        "submission_mtl_name": "material.mtl",
        "requires_material_files": True,
        "requires_uv": True,
        "requires_mtl": True,
        "requires_basecolor_texture": True,
        "requires_color_match_gate": True,
        "color_match_gate_threshold": 0.9,
    },
    {
        "task_tag": "uv_material_reproduction_old_sci_fi_gun",
        "display_name": "Old Sci-Fi Gun UV + Material Reproduction",
        "remote_task_dir_name": "uv_material_reproduction_old_sci_fi_gun",
        "task_shape": "single_part",
        "submission_contract": "object_bundle",
        "mode": "material_reproduction",
        "input_obj_name": "old_sci_fi_gun_raw.obj",
        "input_mtl_name": None,
        "reference_image_dir_name": "reference_images",
        "reference_obj_name": "old_sci_fi_gun.obj",
        "submission_obj_name": "old_sci_fi_gun.obj",
        "submission_mtl_name": "material.mtl",
        "requires_material_files": True,
        "requires_uv": True,
        "requires_mtl": True,
        "requires_basecolor_texture": True,
        "requires_color_match_gate": True,
        "color_match_gate_threshold": 0.9,
    },
    {
        "task_tag": "uv_material_reproduction_skatetron",
        "display_name": "Skatetron UV + Material Reproduction",
        "remote_task_dir_name": "uv_material_reproduction_skatetron",
        "task_shape": "single_part",
        "submission_contract": "object_bundle",
        "mode": "material_reproduction",
        "input_obj_name": "skatetron_raw.obj",
        "input_mtl_name": None,
        "reference_image_dir_name": "reference_images",
        "reference_obj_name": "skatetron.obj",
        "submission_obj_name": "skatetron.obj",
        "submission_mtl_name": "material.mtl",
        "requires_material_files": True,
        "requires_uv": True,
        "requires_mtl": True,
        "requires_basecolor_texture": True,
        "requires_color_match_gate": True,
        "color_match_gate_threshold": 0.9,
    },
    {
        "task_tag": "uv_material_reproduction_war_robot",
        "display_name": "War Robot UV + Material Reproduction",
        "remote_task_dir_name": "uv_material_reproduction_war_robot",
        "task_shape": "single_part",
        "submission_contract": "object_bundle",
        "mode": "material_reproduction",
        "input_obj_name": "war_robot_raw.obj",
        "input_mtl_name": None,
        "reference_image_dir_name": "reference_images",
        "reference_obj_name": "war_robot.obj",
        "submission_obj_name": "war_robot.obj",
        "submission_mtl_name": "material.mtl",
        "requires_material_files": True,
        "requires_uv": True,
        "requires_mtl": True,
        "requires_basecolor_texture": True,
        "requires_color_match_gate": True,
        "color_match_gate_threshold": 0.9,
    },
    {
        "task_tag": "secret_labs_microscope",
        "display_name": "Secret Labs Microscope UV + Material Reproduction",
        "remote_task_dir_name": "uv_material_reproduction_secret_labs_microscope",
        "task_shape": "single_part",
        "submission_contract": "object_bundle",
        "mode": "material_reproduction",
        "input_obj_name": "microscope_raw.obj",
        "input_mtl_name": None,
        "reference_image_dir_name": "reference_images",
        "reference_obj_name": "microscope.obj",
        "submission_obj_name": "microscope.obj",
        "submission_mtl_name": "material.mtl",
        "requires_material_files": True,
        "requires_uv": True,
        "requires_mtl": True,
        "requires_basecolor_texture": True,
        "requires_color_match_gate": True,
        "color_match_gate_threshold": 0.9,
    },
    {
        "task_tag": "secret_labs_full_scene",
        "display_name": "Secret Labs Full Scene UV + Material Reproduction",
        "remote_task_dir_name": "uv_material_reproduction_secret_labs_full_scene",
        "task_shape": "multi_part",
        "submission_contract": "single_scene",
        "mode": "material_reproduction",
        "input_obj_name": "",
        "input_mtl_name": None,
        "reference_image_dir_name": "scene_reference_images",
        "reference_manifest_name": "manifest.json",
        "scene_reference_dir_name": "scene_reference_images",
        "part_reference_dir_name": "parts",
        "submission_obj_name": "",
        "submission_scene_name": "final.blend",
        "requires_material_files": True,
        "requires_uv": True,
        "requires_mtl": True,
        "requires_basecolor_texture": True,
        "requires_color_match_gate": True,
        "color_match_gate_threshold": 0.9,
        "multipart_sample_count": 5,
        "multipart_sample_seed": "uv_reproduction_multipart_eval_v1",
    },
]
PACKAGE_DIR_NAMES = {item["task_tag"]: item["remote_task_dir_name"] for item in VARIANTS}
VARIANT_TAGS = [item["task_tag"] for item in VARIANTS]
VARIANTS_BY_TAG = {item["task_tag"]: item for item in VARIANTS}


def _ps_quote(text: str) -> str:
    return text.replace("'", "''")


def _cmd_quote(text: str) -> str:
    return '"' + text.replace('"', '""') + '"'


def _remote_child(base: str, *parts: str) -> str:
    path = PureWindowsPath(base)
    for part in parts:
        if part:
            path = path / part
    return str(path)


@dataclass
class UVTaskConfig(GeneralTaskConfig):
    DOMAIN_NAME: str = "visual_media"

    TASK_NAME: str = "uv_reproduction"
    VARIANT_NAME: str = ""
    TASK_TAG: str = ""
    REMOTE_TASK_DIR_NAME: str = ""
    DISPLAY_NAME: str = ""
    TASK_SHAPE: str = "single_part"
    SUBMISSION_CONTRACT: str = "object_bundle"
    MODE: str = "contract_only"
    INPUT_OBJ_NAME: str = ""
    INPUT_MTL_NAME: str | None = None
    REFERENCE_IMAGE_DIR_NAME: str = "reference_images"
    REFERENCE_OBJ_NAME: str | None = None
    REFERENCE_MANIFEST_NAME: str = "manifest.json"
    SCENE_REFERENCE_DIR_NAME: str = "scene_reference_images"
    PART_REFERENCE_DIR_NAME: str = "parts"
    REFERENCE_MTL_NAME: str = "material.mtl"
    REFERENCE_TEXTURE_SUBPATH: tuple[str, ...] = ("objects", "textures")
    SUBMISSION_OBJ_NAME: str = ""
    SUBMISSION_MTL_NAME: str | None = None
    SUBMISSION_SCENE_NAME: str = "final.blend"
    REQUIRES_MATERIAL_FILES: bool = False
    REQUIRES_UV: bool = True
    REQUIRES_MTL: bool = False
    REQUIRES_BASECOLOR_TEXTURE: bool = False
    REQUIRES_COLOR_MATCH_GATE: bool = False
    COLOR_MATCH_GATE_THRESHOLD: float = 0.0
    MULTIPART_SAMPLE_RATIO: float = 1.0
    MULTIPART_SAMPLE_COUNT: int = 5
    MULTIPART_SAMPLE_SEED: str = "uv_reproduction_multipart_eval_v1"

    @property
    def input_dir(self) -> str:
        return _remote_child(self.task_dir, "input")

    @property
    def task_dir(self) -> str:
        return _remote_child(
            self.REMOTE_ROOT_DIR, self.DOMAIN_NAME, self.TASK_NAME, self.REMOTE_TASK_DIR_NAME
        )

    @property
    def input_obj(self) -> str:
        if not self.INPUT_OBJ_NAME:
            return ""
        return _remote_child(self.input_dir, self.INPUT_OBJ_NAME)

    @property
    def input_mtl(self) -> str | None:
        if not self.INPUT_MTL_NAME:
            return None
        return _remote_child(self.input_dir, self.INPUT_MTL_NAME)

    @property
    def reference_image_dir(self) -> str:
        return _remote_child(self.input_dir, self.REFERENCE_IMAGE_DIR_NAME)

    @property
    def scene_reference_dir(self) -> str:
        return _remote_child(self.input_dir, self.SCENE_REFERENCE_DIR_NAME)

    @property
    def part_reference_dir(self) -> str:
        return _remote_child(self.input_dir, self.PART_REFERENCE_DIR_NAME)

    @property
    def input_manifest(self) -> str:
        return _remote_child(self.input_dir, "manifest.json")

    @property
    def output_submission_dir(self) -> str:
        return _remote_child(self.remote_output_dir, "submission")

    @property
    def output_texture_dir(self) -> str:
        return _remote_child(self.output_submission_dir, "textures")

    @property
    def output_obj(self) -> str:
        if not self.SUBMISSION_OBJ_NAME:
            return ""
        return _remote_child(self.output_submission_dir, self.SUBMISSION_OBJ_NAME)

    @property
    def output_mtl(self) -> str | None:
        if not self.SUBMISSION_MTL_NAME:
            return None
        return _remote_child(self.output_submission_dir, self.SUBMISSION_MTL_NAME)

    @property
    def reference_obj(self) -> str:
        name = self.REFERENCE_OBJ_NAME or self.SUBMISSION_OBJ_NAME
        return _remote_child(self.reference_dir, "objects", name)

    @property
    def reference_manifest(self) -> str:
        return _remote_child(self.reference_dir, self.REFERENCE_MANIFEST_NAME)

    @property
    def output_scene(self) -> str:
        return _remote_child(self.output_submission_dir, self.SUBMISSION_SCENE_NAME)

    @property
    def reference_mtl(self) -> str | None:
        if self.MODE != "material_reproduction":
            return None
        return _remote_child(self.reference_dir, "objects", self.REFERENCE_MTL_NAME)

    @property
    def reference_texture_dir(self) -> str | None:
        if self.MODE == "material_reproduction":
            return _remote_child(self.reference_dir, *self.REFERENCE_TEXTURE_SUBPATH)
        return None

    @property
    def reference_image_render_dir(self) -> str:
        # Pre-rendered reference views (front/back/left/right/top_front/bottom_front).
        # Consumed by the single-part hard eval when the full set is staged; otherwise
        # the evaluator renders the reference OBJ live.
        return _remote_child(self.reference_dir, "images")

    @property
    def task_description(self) -> str:
        if self.TASK_SHAPE == "multi_part":
            return textwrap.dedent(f"""\
                You are a 3D artist using Blender.

                Your task is to recreate the UV unwrap and textured material appearance for every required named part in the provided asset set.

                Agent-visible inputs:
                - Overall scene reference renders: `{self.scene_reference_dir}`
                - Per-part detail renders and metadata folders: `{self.part_reference_dir}`
                - Input manifest listing the required object names and part folders: `{self.input_manifest}`

                Required submission:
                - one Blender scene file at `{self.output_scene}`

                Use the exact object names required by the input manifest.
                Your scene must contain all required parts with the expected final UV and material results.
                """)
        if self.MODE == "material_reproduction":
            material_sidecar = self.input_mtl or "None"
            return textwrap.dedent(f"""\
                You are a 3D artist using Blender.

                Your task is to reproduce the UV unwrap and textured material appearance of the provided model.

                Official input:
                - White model OBJ: `{self.input_obj}`
                - Optional input material sidecar: `{material_sidecar}`
                - Reference renders: `{self.reference_image_dir}`

                Required submission:
                - `{self.output_obj}`
                - `{self.output_mtl}`
                - textures referenced by your material under `{self.output_texture_dir}`
                """)
        return textwrap.dedent(f"""\
            You are a 3D artist using Blender.

            Your task is to reproduce the target UV configuration for the provided whole-object mesh.

            Official input:
            - Mesh: `{self.input_obj}`
            - Material sidecar: `{self.input_mtl}`
            - Agent-visible reference images: `{self.reference_image_dir}`

            Required submission:
            - Export a UV-bearing OBJ named `{self.SUBMISSION_OBJ_NAME}`
            - Save it to `{self.output_obj}`

            Only the UV layout is evaluated for this variant. Do not add material assets unless the variant input already includes them.
            """)

    def to_metadata(self) -> dict[str, Any]:
        data = super().to_metadata()
        data.pop("software_dir", None)
        data.update(
            {
                "task_tag": self.TASK_TAG,
                "display_name": self.DISPLAY_NAME,
                "task_shape": self.TASK_SHAPE,
                "submission_contract": self.SUBMISSION_CONTRACT,
                "mode": self.MODE,
                "task_dir": self.task_dir,
                "remote_output_dir": self.remote_output_dir,
                "reference_dir": self.reference_dir,
                "input_dir": self.input_dir,
                "input_obj": self.input_obj,
                "input_mtl": self.input_mtl,
                "reference_image_dir": self.reference_image_dir,
                "scene_reference_dir": self.scene_reference_dir,
                "part_reference_dir": self.part_reference_dir,
                "input_manifest": self.input_manifest,
                "output_submission_dir": self.output_submission_dir,
                "output_texture_dir": self.output_texture_dir,
                "output_obj": self.output_obj,
                "output_mtl": self.output_mtl,
                "output_scene": self.output_scene,
                "reference_obj": self.reference_obj,
                "reference_manifest": self.reference_manifest,
                "reference_mtl": self.reference_mtl,
                "reference_texture_dir": self.reference_texture_dir,
                "reference_image_render_dir": self.reference_image_render_dir,
                "remote_task_dir_name": self.REMOTE_TASK_DIR_NAME,
                "requires_material_files": self.REQUIRES_MATERIAL_FILES,
                "requires_uv": self.REQUIRES_UV,
                "requires_mtl": self.REQUIRES_MTL,
                "requires_basecolor_texture": self.REQUIRES_BASECOLOR_TEXTURE,
                "requires_color_match_gate": self.REQUIRES_COLOR_MATCH_GATE,
                "color_match_gate_threshold": self.COLOR_MATCH_GATE_THRESHOLD,
                "multipart_sample_ratio": self.MULTIPART_SAMPLE_RATIO,
                "multipart_sample_count": self.MULTIPART_SAMPLE_COUNT,
                "multipart_sample_seed": self.MULTIPART_SAMPLE_SEED,
            }
        )
        return data


def _variant_to_cfg(spec: dict[str, Any]) -> UVTaskConfig:
    return UVTaskConfig(
        VARIANT_NAME=spec["remote_task_dir_name"],
        TASK_TAG=spec["task_tag"],
        REMOTE_TASK_DIR_NAME=spec["remote_task_dir_name"],
        DISPLAY_NAME=spec["display_name"],
        TASK_SHAPE=spec.get("task_shape", "single_part"),
        SUBMISSION_CONTRACT=spec.get("submission_contract", "object_bundle"),
        MODE=spec["mode"],
        INPUT_OBJ_NAME=spec["input_obj_name"],
        INPUT_MTL_NAME=spec.get("input_mtl_name"),
        REFERENCE_IMAGE_DIR_NAME=spec["reference_image_dir_name"],
        REFERENCE_OBJ_NAME=spec.get("reference_obj_name"),
        REFERENCE_MANIFEST_NAME=spec.get("reference_manifest_name", "manifest.json"),
        SCENE_REFERENCE_DIR_NAME=spec.get("scene_reference_dir_name", "scene_reference_images"),
        PART_REFERENCE_DIR_NAME=spec.get("part_reference_dir_name", "parts"),
        REFERENCE_MTL_NAME=spec.get("reference_mtl_name", "material.mtl"),
        REFERENCE_TEXTURE_SUBPATH=tuple(
            spec.get("reference_texture_subpath", ("objects", "textures"))
        ),
        SUBMISSION_OBJ_NAME=spec["submission_obj_name"],
        SUBMISSION_MTL_NAME=spec.get("submission_mtl_name"),
        SUBMISSION_SCENE_NAME=spec.get("submission_scene_name", "final.blend"),
        REQUIRES_MATERIAL_FILES=spec.get("requires_material_files", False),
        REQUIRES_UV=spec.get("requires_uv", True),
        REQUIRES_MTL=spec.get("requires_mtl", False),
        REQUIRES_BASECOLOR_TEXTURE=spec.get("requires_basecolor_texture", False),
        REQUIRES_COLOR_MATCH_GATE=spec.get("requires_color_match_gate", False),
        COLOR_MATCH_GATE_THRESHOLD=float(spec.get("color_match_gate_threshold", 0.0)),
        MULTIPART_SAMPLE_RATIO=float(spec.get("multipart_sample_ratio", 1.0)),
        MULTIPART_SAMPLE_COUNT=int(spec.get("multipart_sample_count", 5)),
        MULTIPART_SAMPLE_SEED=str(
            spec.get("multipart_sample_seed", "uv_reproduction_multipart_eval_v1")
        ),
    )


@cb.tasks_config(split="train")
def load():
    tasks = []
    for spec in VARIANTS:
        cfg = _variant_to_cfg(spec)
        tasks.append(
            cb.Task(
                description=cfg.task_description,
                metadata=cfg.to_metadata(),
                computer={"provider": "computer", "setup_config": {"os_type": "windows"}},
            )
        )
    return tasks


@cb.setup_task(split="train")
async def start(task_cfg, session: cb.DesktopSession):
    await _setup(task_cfg, session)


def _obj_has_uv_text(content: str) -> bool:
    return any(line.startswith("vt ") for line in content.splitlines())


async def _upload_scripts(session: cb.DesktopSession, remote_scripts_dir: str) -> None:
    await session.interface.create_dir(remote_scripts_dir)
    for name in [
        "remote_hard_eval.py",
        "blender_render_material_views.py",
        "blender_render_multipart_material_views.py",
    ]:
        await session.write_file(
            _remote_child(remote_scripts_dir, name),
            (SCRIPTS_DIR / name).read_text(encoding="utf-8"),
        )


async def _launch_remote_job(
    session: cb.DesktopSession, *, remote_scripts_dir: str, args: list[str]
) -> None:
    script_path = _remote_child(remote_scripts_dir, "remote_hard_eval.py")
    stdout_path = _remote_child(remote_scripts_dir, "job_stdout.txt")
    stderr_path = _remote_child(remote_scripts_dir, "job_stderr.txt")
    argv = " ".join(_cmd_quote(v) for v in [script_path, *args])
    ps = (
        "$ErrorActionPreference='Stop'; "
        "$wd='" + _ps_quote(remote_scripts_dir) + "'; "
        "$py='" + _ps_quote(REMOTE_PYTHON) + "'; "
        "$stdout='" + _ps_quote(stdout_path) + "'; "
        "$stderr='" + _ps_quote(stderr_path) + "'; "
        "Set-Location -LiteralPath $wd; "
        "if (Test-Path -LiteralPath $stdout) { Remove-Item -LiteralPath $stdout -Force -ErrorAction SilentlyContinue }; "
        "if (Test-Path -LiteralPath $stderr) { Remove-Item -LiteralPath $stderr -Force -ErrorAction SilentlyContinue }; "
        f"$argLine='{_ps_quote(argv)}'; "
        "Write-Output $argLine | Out-Null; "
        "Start-Process -FilePath $py -ArgumentList $argLine -WorkingDirectory $wd "
        "-RedirectStandardOutput $stdout -RedirectStandardError $stderr -WindowStyle Hidden"
    )
    await session.run_command(f'powershell -NoProfile -Command "{ps}"', check=False)


async def _wait_for_file(
    session: cb.DesktopSession, path: str, timeout_sec: float = 1800.0, poll_sec: float = 10.0
) -> bool:
    deadline = asyncio.get_event_loop().time() + timeout_sec
    while asyncio.get_event_loop().time() < deadline:
        try:
            if (await session.file_exists(path) or await session.directory_exists(path)):
                return True
        except Exception:
            pass
        await asyncio.sleep(poll_sec)
    return False


@cb.evaluate_task(split="train")
async def evaluate(task_cfg, session: cb.DesktopSession) -> list[float]:
    meta = task_cfg.metadata
    if meta["task_shape"] == "single_part":
        if not (await session.file_exists(meta["output_obj"]) or await session.directory_exists(meta["output_obj"])):
            return [0.0]
    else:
        if not (await session.file_exists(meta["output_scene"]) or await session.directory_exists(meta["output_scene"])):
            return [0.0]

    if meta["task_shape"] == "single_part" and meta["mode"] == "contract_only":
        try:
            content = await session.read_file(meta["output_obj"])
            return [1.0 if _obj_has_uv_text(content) else 0.0]
        except Exception:
            logger.warning("Temporary UV contract evaluation failed", exc_info=True)
            return [0.0]

    remote_eval_dir = _remote_child(REMOTE_TEMP_ROOT, meta["task_tag"])
    remote_scripts_dir = _remote_child(remote_eval_dir, "scripts")
    remote_results_dir = _remote_child(remote_eval_dir, "results")
    remote_report = _remote_child(remote_results_dir, "hard_eval_report.json")
    await session.interface.create_dir(remote_eval_dir)
    await session.run_command(
        "powershell -NoProfile -Command "
        + repr(
            f"$ErrorActionPreference='Stop'; "
            f"if (Test-Path -LiteralPath '{remote_results_dir}') "
            f"{{ Remove-Item -LiteralPath '{remote_results_dir}' -Recurse -Force -ErrorAction SilentlyContinue }}; "
            f"New-Item -ItemType Directory -Path '{remote_results_dir}' -Force | Out-Null"
        ),
        check=False,
    )
    await _upload_scripts(session, remote_scripts_dir)
    remote_args = [
        "--output-dir",
        remote_results_dir,
        "--requires-uv",
        "1" if meta.get("requires_uv", True) else "0",
        "--requires-mtl",
        "1" if meta.get("requires_mtl", False) else "0",
        "--requires-basecolor-texture",
        "1" if meta.get("requires_basecolor_texture", False) else "0",
        "--requires-color-match-gate",
        "1" if meta.get("requires_color_match_gate", False) else "0",
        "--color-match-gate-threshold",
        str(meta.get("color_match_gate_threshold", 0.0)),
    ]
    if meta["task_shape"] == "single_part":
        remote_args.extend(
            [
                "--task-shape",
                "single_part",
                "--reference-obj",
                meta["reference_obj"],
                "--reference-mtl",
                meta["reference_mtl"],
                "--reference-texture-dir",
                meta["reference_texture_dir"],
                "--reference-images-dir",
                meta["reference_image_render_dir"],
                "--candidate-obj",
                meta["output_obj"],
                "--candidate-mtl",
                meta["output_mtl"],
                "--candidate-texture-dir",
                meta["output_texture_dir"],
                "--renderer-script",
                _remote_child(remote_scripts_dir, "blender_render_material_views.py"),
            ]
        )
    else:
        remote_args.extend(
            [
                "--task-shape",
                "multi_part",
                "--reference-manifest",
                meta["reference_manifest"],
                "--input-manifest",
                meta["input_manifest"],
                "--reference-dir",
                meta["reference_dir"],
                "--candidate-scene",
                meta["output_scene"],
                "--multipart-sample-count",
                str(meta.get("multipart_sample_count", 5)),
                "--multipart-sample-seed",
                str(meta.get("multipart_sample_seed", "uv_reproduction_multipart_eval_v1")),
                "--renderer-script",
                _remote_child(remote_scripts_dir, "blender_render_multipart_material_views.py"),
            ]
        )
    await _launch_remote_job(session, remote_scripts_dir=remote_scripts_dir, args=remote_args)
    if not await _wait_for_file(session, remote_report):
        return [0.0]
    report = json.loads((await session.read_bytes(remote_report)).decode("utf-8"))
    hard_score = float(report.get("score", 0.0))
    hard_fail = False
    if meta["task_shape"] == "multi_part" and report.get("missing_parts"):
        hard_fail = True
    gate = (report.get("metrics") or {}).get("gate") or {}
    if gate and not all(bool(value) for value in gate.values()):
        hard_fail = True
    frame_pairs = report.get("frame_pairs", [])
    local_tmp = TASK_DIR / ".tmp_soft_eval" / meta["task_tag"]
    if local_tmp.exists():
        shutil.rmtree(local_tmp)
    local_tmp.mkdir(parents=True, exist_ok=True)
    localized_pairs = []
    try:
        for idx, pair in enumerate(frame_pairs):
            ref_path = local_tmp / f"reference_{idx}_{pair['view']}.png"
            cand_path = local_tmp / f"candidate_{idx}_{pair['view']}.png"
            ref_path.write_bytes(await session.read_bytes(pair["reference_image"]))
            cand_path.write_bytes(await session.read_bytes(pair["candidate_image"]))
            localized_pairs.append(
                {
                    "view": pair["view"],
                    "reference_image": str(ref_path),
                    "candidate_image": str(cand_path),
                }
            )
        soft_score = run_local_soft_eval(localized_pairs, local_tmp)
    except Exception:
        logger.warning("Local soft eval failed; falling back to hard-only", exc_info=True)
        soft_score = hard_score
    final_score = hard_score if hard_fail else (0.3 * hard_score + 0.7 * soft_score)
    logger.info(
        "uv_reproduction eval task=%s hard=%.4f soft=%.4f final=%.4f",
        meta["task_tag"],
        hard_score,
        soft_score,
        final_score,
    )
    return [float(final_score)]


if __name__ == "__main__":
    print(json.dumps({"workflow": WORKFLOW, "variants": VARIANT_TAGS}, indent=2))
