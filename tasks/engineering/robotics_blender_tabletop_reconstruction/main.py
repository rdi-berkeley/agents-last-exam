"""AgentHLE task: robotics_blender_tabletop_reconstruction."""

from __future__ import annotations

import base64
import json
import logging
import os
import textwrap
from pathlib import Path, PureWindowsPath
from typing import Any

import openai
import cua_bench as cb

from tasks.common_config import GeneralTaskConfig
from tasks.common_setup import BaseTaskSetup

logger = logging.getLogger(__name__)

TASK_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = TASK_DIR / "scripts"
WORKFLOW = "robotics_blender_tabletop_reconstruction"
EVAL_TMP_DIR = r"C:\Users\User\AppData\Local\Temp\agenthle_eval\robotics_blender_tabletop_reconstruction"
PASS_THRESHOLD = 0.70

SPATIAL_TOL_M = 0.02
ROTATION_TOL_DEG = 8.0
MATERIAL_COLOR_TOL = 0.12
MATERIAL_PROP_TOL = 0.15
LIGHT_POS_TOL_M = 0.5
LIGHT_ENERGY_TOL_FRAC = 0.30
MIN_BLEND_BYTES = 10_000
MIN_PNG_BYTES = 5_000
MIN_OBJ_BYTES = 100

VLM_JUDGE_MODEL = "gpt-5.5"
VLM_FALLBACK_SCORE = 0.5
_ENV_FILE = Path(__file__).resolve().parents[3] / ".env"


def _read_script(name: str) -> str:
    return (SCRIPTS_DIR / name).read_text(encoding="utf-8")


def _remote_child(base: str, *parts: str) -> str:
    path = PureWindowsPath(base)
    for part in parts:
        if part:
            path = path / part
    return str(path)


async def _run_command(
    session: cb.DesktopSession,
    command: str,
    *,
    timeout: float | None = None,
    check: bool = False,
) -> dict[str, Any]:
    try:
        if timeout is not None:
            return await session.run_command(command, timeout=timeout, check=check)
        return await session.run_command(command, check=check)
    except TypeError:
        return await session.run_command(command, check=check)


async def _read_text(session: cb.DesktopSession, path: str) -> str:
    try:
        return await session.read_file(path)
    except Exception:
        data = await session.read_bytes(path)
        return data.decode("utf-8")


def _angle_error_deg(a: float, b: float) -> float:
    delta = abs(a - b) % 360.0
    return min(delta, 360.0 - delta)


def _pass_fraction(passes: list[bool]) -> float:
    if not passes:
        return 0.0
    return sum(1.0 for item in passes if item) / len(passes)


def _spatial_score(reference_spec: dict[str, Any], metrics: dict[str, Any]) -> float:
    observed = metrics.get("objects", {})
    checks: list[bool] = []
    for obj in reference_spec["objects"]:
        candidate = observed.get(obj["name"])
        for axis, ref_val in enumerate(obj["position"]):
            checks.append(
                candidate is not None
                and abs(candidate["location"][axis] - ref_val) < SPATIAL_TOL_M
            )
    return _pass_fraction(checks)


def _rotation_score(reference_spec: dict[str, Any], metrics: dict[str, Any]) -> float:
    observed = metrics.get("objects", {})
    checks: list[bool] = []
    for obj in reference_spec["objects"]:
        candidate = observed.get(obj["name"])
        for axis, ref_val in enumerate(obj["rotation_deg"]):
            if obj["name"] == "potted_meat_can" and axis == 2:
                continue
            checks.append(
                candidate is not None
                and _angle_error_deg(candidate["rotation_deg"][axis], ref_val) < ROTATION_TOL_DEG
            )
    return _pass_fraction(checks)


def _material_score(reference_spec: dict[str, Any], metrics: dict[str, Any]) -> float:
    observed = metrics.get("objects", {})
    checks: list[bool] = []
    for obj in reference_spec["objects"]:
        candidate = observed.get(obj["name"])
        material = None if candidate is None else candidate.get("material")
        for idx, ref_val in enumerate(obj["material"]["base_color_rgb"]):
            checks.append(
                material is not None
                and abs(material["base_color_rgb"][idx] - ref_val) < MATERIAL_COLOR_TOL
            )
        checks.append(
            material is not None
            and abs(material["roughness"] - obj["material"]["roughness"]) < MATERIAL_PROP_TOL
        )
        checks.append(
            material is not None
            and abs(material["metallic"] - obj["material"]["metallic"]) < MATERIAL_PROP_TOL
        )
    return _pass_fraction(checks)


def _export_score(
    metrics: dict[str, Any],
    file_sizes: dict[str, int] | None = None,
) -> float:
    exports = metrics.get("exports", {})
    sizes = file_sizes or {}
    checks = [
        exports.get("verification_render", False)
        and sizes.get("verification_render", 0) >= MIN_PNG_BYTES,
        exports.get("mustard_bottle", False)
        and sizes.get("mustard_bottle", 0) >= MIN_OBJ_BYTES,
        exports.get("mug", False)
        and sizes.get("mug", 0) >= MIN_OBJ_BYTES,
        exports.get("potted_meat_can", False)
        and sizes.get("potted_meat_can", 0) >= MIN_OBJ_BYTES,
        exports.get("full_scene", False)
        and sizes.get("full_scene", 0) >= MIN_OBJ_BYTES,
    ]
    return _pass_fraction(checks)


def _light_match_score(reference_light: dict[str, Any], candidate_light: dict[str, Any]) -> float:
    checks = [
        abs(candidate_light["location"][0] - reference_light["position"][0]) < LIGHT_POS_TOL_M,
        abs(candidate_light["location"][1] - reference_light["position"][1]) < LIGHT_POS_TOL_M,
        abs(candidate_light["location"][2] - reference_light["position"][2]) < LIGHT_POS_TOL_M,
        abs(candidate_light["energy"] - reference_light["energy"])
        <= (LIGHT_ENERGY_TOL_FRAC * reference_light["energy"]),
    ]
    return _pass_fraction(checks)


def _lighting_score(reference_spec: dict[str, Any], metrics: dict[str, Any]) -> float:
    candidate_lights = metrics.get("lights", [])
    if len(candidate_lights) < 3:
        return 0.0
    ref_lights = [
        reference_spec["lighting"]["key_light"],
        reference_spec["lighting"]["fill_light"],
        reference_spec["lighting"]["rim_light"],
    ]
    scores: list[float] = []
    for ref_light in ref_lights:
        same_type = [
            cand for cand in candidate_lights
            if cand.get("type", "").upper() == ref_light["type"].upper()
        ]
        if not same_type:
            scores.append(0.0)
            continue
        scores.append(max(_light_match_score(ref_light, cand) for cand in same_type))
    return sum(scores) / len(scores)


_VLM_PROMPT = """\
Compare these two top-down renders of a tabletop scene with objects.
Image 1 is the REFERENCE (ground truth). Image 2 is the CANDIDATE (to evaluate).

Score each dimension from 0.0 to 1.0:
- object_count: Same number and types of distinct objects on the table?
- spatial_layout: Objects in similar relative positions?
- camera_perspective: Same viewing angle and framing?
- material_fidelity: Similar colors, textures, surface appearance on the objects?
- lighting_quality: Similar brightness, shadow direction, highlight distribution?
- table_surface: Table appearance similar (color, texture, proportion)?

Return ONLY a JSON object with exactly these six keys and float values, nothing else."""


def _load_openai_key() -> str:
    try:  # load secret/eval_time/*.env so the OpenAI judge key is present
        from tasks.utils.evaluation import load_eval_env

        load_eval_env()
    except Exception:
        pass
    key = os.environ.get("OPENAI_API_KEY")
    if key:
        return key
    if _ENV_FILE.exists():
        for line in _ENV_FILE.read_text().splitlines():
            line = line.strip()
            if line.startswith("OPENAI_API_KEY"):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise RuntimeError("OPENAI_API_KEY not found in environment or .env")


def _call_vlm_judge(ref_b64: str, cand_b64: str) -> float:
    client = openai.OpenAI(api_key=_load_openai_key())
    resp = client.chat.completions.create(
        model=VLM_JUDGE_MODEL,
        max_tokens=256,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": _VLM_PROMPT},
                    {
                        "type": "text",
                        "text": "Reference render (ground truth):",
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{ref_b64}",
                        },
                    },
                    {
                        "type": "text",
                        "text": "Candidate render:",
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{cand_b64}",
                        },
                    },
                ],
            }
        ],
    )
    text = resp.choices[0].message.content.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    scores = json.loads(text)
    vals = [float(v) for v in scores.values()]
    return sum(vals) / len(vals) if vals else 0.0


async def _vlm_render_score(
    session: cb.DesktopSession,
    reference_render_path: str,
    candidate_render_path: str,
) -> float:
    try:
        if not (await session.file_exists(candidate_render_path) or await session.directory_exists(candidate_render_path)):
            return 0.0
        ref_bytes = await session.read_bytes(reference_render_path)
        cand_bytes = await session.read_bytes(candidate_render_path)
        ref_b64 = base64.standard_b64encode(ref_bytes).decode()
        cand_b64 = base64.standard_b64encode(cand_bytes).decode()
        return _call_vlm_judge(ref_b64, cand_b64)
    except Exception as exc:
        logger.warning("VLM render comparison failed (%s); using fallback", exc)
        return VLM_FALLBACK_SCORE


def _final_score(
    reference_spec: dict[str, Any],
    metrics: dict[str, Any],
    vlm_score: float,
    file_sizes: dict[str, int] | None = None,
) -> dict[str, float]:
    spatial = _spatial_score(reference_spec, metrics)
    rotation = _rotation_score(reference_spec, metrics)
    material = _material_score(reference_spec, metrics)
    export = _export_score(metrics, file_sizes)
    lighting = _lighting_score(reference_spec, metrics)
    final = (
        0.15 * spatial
        + 0.10 * rotation
        + 0.10 * material
        + 0.05 * export
        + 0.15 * lighting
        + 0.45 * vlm_score
    )
    return {
        "spatial_score": spatial,
        "rotation_score": rotation,
        "material_score": material,
        "export_score": export,
        "lighting_score": lighting,
        "vlm_render_score": vlm_score,
        "final_score": final,
        "pass_threshold": PASS_THRESHOLD,
    }


class RoboticsBlenderTaskConfig(GeneralTaskConfig):
    def __init__(
        self,
        *,
        REMOTE_OUTPUT_DIR: str | None = None,
        REMOTE_ROOT_DIR: str | None = None,
        DOMAIN_NAME: str = "engineering",
        TASK_NAME: str = "robotics_blender_tabletop_reconstruction",
        OS_TYPE: str = "windows",
    ) -> None:
        super().__init__(
            REMOTE_OUTPUT_DIR=REMOTE_OUTPUT_DIR or os.environ.get("REMOTE_OUTPUT_DIR", "output"),
            REMOTE_ROOT_DIR=REMOTE_ROOT_DIR or os.environ.get("REMOTE_ROOT_DIR", r"E:\agenthle"),
            DOMAIN_NAME=DOMAIN_NAME,
            TASK_NAME=TASK_NAME,
            OS_TYPE=OS_TYPE,
            VARIANT_NAME="base",
        )

    @property
    def task_dir(self) -> str:
        return _remote_child(self.REMOTE_ROOT_DIR, self.DOMAIN_NAME, self.TASK_NAME, self.VARIANT_NAME)

    @property
    def input_dir(self) -> str:
        return _remote_child(self.task_dir, "input")

    @property
    def export_spec_path(self) -> str:
        return _remote_child(self.input_dir, "export_spec.json")

    @property
    def floor_plan_path(self) -> str:
        return _remote_child(self.input_dir, "floor_plan.png")

    @property
    def reference_render_path(self) -> str:
        return _remote_child(self.input_dir, "reference_render.png")

    @property
    def meshes_dir(self) -> str:
        return _remote_child(self.input_dir, "meshes")

    @property
    def reference_scene_spec(self) -> str:
        return _remote_child(self.reference_dir, "scene_spec.json")

    @property
    def reference_scene_file(self) -> str:
        return _remote_child(self.reference_dir, "reference_scene.blend")

    @property
    def output_scene(self) -> str:
        return _remote_child(self.remote_output_dir, "scene.blend")

    @property
    def output_render(self) -> str:
        return _remote_child(self.remote_output_dir, "verification_render.png")

    @property
    def output_full_scene_obj(self) -> str:
        return _remote_child(self.remote_output_dir, "full_scene.obj")

    @property
    def output_mustard_obj(self) -> str:
        return _remote_child(self.remote_output_dir, "mustard_bottle.obj")

    @property
    def output_mug_obj(self) -> str:
        return _remote_child(self.remote_output_dir, "mug.obj")

    @property
    def output_potted_meat_obj(self) -> str:
        return _remote_child(self.remote_output_dir, "potted_meat_can.obj")

    @property
    def blender_wrapper(self) -> str:
        return _remote_child(self.software_dir, "run_blender_portable.ps1")

    @property
    def blender_download_script(self) -> str:
        return _remote_child(self.software_dir, "download_blender_portable.ps1")

    @property
    def task_description(self) -> str:
        return textwrap.dedent(
            f"""\
            You are a robotics-focused 3D scene artist using Blender.

            Your task is to reconstruct a tabletop robotic-manipulation workspace from staged assets.

            Agent-visible inputs:
            - Export contract: `{_remote_child(self.input_dir, 'export_spec.json')}`
            - Floor plan: `{_remote_child(self.input_dir, 'floor_plan.png')}`
            - Reference render: `{_remote_child(self.input_dir, 'reference_render.png')}`
            - Mesh assets: `{_remote_child(self.input_dir, 'meshes')}`

            Software:
            - Launch Blender through the staged task-local wrapper: `{self.blender_wrapper}`

            Required outputs under `{self.remote_output_dir}`:
            - `scene.blend`
            - `verification_render.png`
            - `mustard_bottle.obj`
            - `mug.obj`
            - `potted_meat_can.obj`
            - `full_scene.obj`

            Requirements:
            - Use the floor plan for explicit spatial values and camera setup
            - Infer rotations, materials, and lighting from the reference render
            - Keep the coordinate system z-up and 1 Blender unit = 1 meter
            - Save the rendered image as `verification_render.png`
            """
        )

    def to_metadata(self) -> dict[str, Any]:
        data = super().to_metadata()
        data.update(
            {
                "variant_name": "base",
                "task_dir": self.task_dir,
                "input_dir": self.input_dir,
                "export_spec_path": self.export_spec_path,
                "floor_plan_path": self.floor_plan_path,
                "reference_render_path": self.reference_render_path,
                "meshes_dir": self.meshes_dir,
                "reference_scene_spec": self.reference_scene_spec,
                "reference_scene_file": self.reference_scene_file,
                "output_scene": self.output_scene,
                "output_render": self.output_render,
                "output_full_scene_obj": self.output_full_scene_obj,
                "output_mustard_obj": self.output_mustard_obj,
                "output_mug_obj": self.output_mug_obj,
                "output_potted_meat_obj": self.output_potted_meat_obj,
                "blender_wrapper": self.blender_wrapper,
                "blender_download_script": self.blender_download_script,
                "pass_threshold": PASS_THRESHOLD,
                "canonical_gcs_root": "gs://ale-data-all/engineering/robotics_blender_tabletop_reconstruction/base/",
            }
        )
        return data


@cb.tasks_config(split="train")
def load():
    cfg = RoboticsBlenderTaskConfig()
    return [
        cb.Task(
            description=cfg.task_description,
            metadata=cfg.to_metadata(),
            computer={"provider": "computer", "setup_config": {"os_type": "windows"}},
        )
    ]


_setup = BaseTaskSetup()


@cb.setup_task(split="train")
async def start(task_cfg, session: cb.DesktopSession):
    await _setup(task_cfg, session)


@cb.evaluate_task(split="train")
async def evaluate(task_cfg, session: cb.DesktopSession) -> list[float]:
    meta = task_cfg.metadata
    if not (await session.file_exists(meta["reference_scene_spec"]) or await session.directory_exists(meta["reference_scene_spec"])):
        logger.error("Missing hidden reference facts: %s", meta["reference_scene_spec"])
        return [0.0]
    if not (await session.file_exists(meta["output_scene"]) or await session.directory_exists(meta["output_scene"])):
        logger.error("Missing candidate scene: %s", meta["output_scene"])
        return [0.0]

    await session.interface.create_dir(EVAL_TMP_DIR)
    output_tag = PureWindowsPath(meta["remote_output_dir"]).name or "output"
    helper_path = _remote_child(EVAL_TMP_DIR, f"extract_scene_metrics_{output_tag}.py")
    metrics_path = _remote_child(EVAL_TMP_DIR, f"candidate_metrics_{output_tag}.json")
    await session.write_file(helper_path, _read_script("extract_scene_metrics.py"))

    command = (
        "powershell -NoProfile -ExecutionPolicy Bypass "
        f'-File "{meta["blender_wrapper"]}" '
        f'-b "{meta["output_scene"]}" '
        f'--python "{helper_path}" '
        f'-- --output-json "{metrics_path}" '
        f'--render-path "{meta["output_render"]}" '
        f'--mustard-path "{meta["output_mustard_obj"]}" '
        f'--mug-path "{meta["output_mug_obj"]}" '
        f'--potted-meat-path "{meta["output_potted_meat_obj"]}" '
        f'--full-scene-path "{meta["output_full_scene_obj"]}"'
    )
    result = await _run_command(session, command, timeout=2400.0, check=False)
    if result.get("return_code", 1) != 0:
        logger.error("Blender metrics extraction failed: %s", result.get("stderr", "")[:400])
        return [0.0]
    if not (await session.file_exists(metrics_path) or await session.directory_exists(metrics_path)):
        logger.error("Expected metrics file missing: %s", metrics_path)
        return [0.0]

    reference_spec = json.loads(await _read_text(session, meta["reference_scene_spec"]))
    metrics = json.loads(await _read_text(session, metrics_path))

    file_size_map = {
        "verification_render": meta["output_render"],
        "mustard_bottle": meta["output_mustard_obj"],
        "mug": meta["output_mug_obj"],
        "potted_meat_can": meta["output_potted_meat_obj"],
        "full_scene": meta["output_full_scene_obj"],
    }
    file_sizes: dict[str, int] = {}
    for key, path in file_size_map.items():
        try:
            data = await session.read_bytes(path)
            file_sizes[key] = len(data)
        except Exception:
            file_sizes[key] = 0

    vlm_score = await _vlm_render_score(
        session, meta["reference_render_path"], meta["output_render"]
    )

    score_payload = _final_score(reference_spec, metrics, vlm_score, file_sizes)
    logger.info("Evaluation summary: %s", json.dumps(score_payload, ensure_ascii=False))
    return [float(score_payload["final_score"])]
