"""AgentHLE task: blender_character_reconstruction_from_multiview_01."""

from __future__ import annotations

import json
import logging
import math
import os
import textwrap
from io import BytesIO
from pathlib import Path, PureWindowsPath
from typing import Any

import cua_bench as cb
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
from skimage.metrics import structural_similarity

from tasks.common_config import GeneralTaskConfig
from tasks.common_setup import BaseTaskSetup

_setup = BaseTaskSetup()

logger = logging.getLogger(__name__)

TASK_ID = "visual_media/blender_character_reconstruction_from_multiview_01"
TASK_NAME = "blender_character_reconstruction_from_multiview_01"
VARIANT_NAME = "base"
TASK_DIR = Path(__file__).resolve().parent
EVAL_TMP_DIR = r"C:\Users\User\AppData\Local\Temp\agenthle_eval\blender_character_reconstruction_from_multiview_01"
BLENDER_INSTALL_DIR = r"C:\Softwares\Blender-5.0.1"
BLENDER_EXE = rf"{BLENDER_INSTALL_DIR}\blender.exe"
BACKGROUND_GRAY = 0.52
MASK_THRESHOLD = 0.06
OBJ_SAMPLE_LIMIT_DEFAULT = 6000


def _remote_child(base: str, *parts: str) -> str:
    path = PureWindowsPath(base)
    for part in parts:
        if part:
            path = path / part
    return str(path)


def _ps_quote(text: str) -> str:
    return text.replace("'", "''")


def _as_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    return str(value)


def _clamp01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


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


async def _read_bytes(session: cb.DesktopSession, path: str) -> bytes:
    try:
        data = await session.read_bytes(path)
        if isinstance(data, bytes):
            return data
        return bytes(data)
    except Exception:
        data = await session.read_file(path)
        if isinstance(data, bytes):
            return data
        return data.encode("utf-8")


async def _read_json(session: cb.DesktopSession, path: str) -> dict[str, Any]:
    return json.loads((await _read_bytes(session, path)).decode("utf-8"))


async def _remote_file_size(session: cb.DesktopSession, path: str) -> int | None:
    ps = (
        f"$p = '{_ps_quote(path)}'; "
        "if (Test-Path -LiteralPath $p) { "
        "(Get-Item -LiteralPath $p).Length "
        "} else { "
        "Write-Output '__MISSING__' "
        "}"
    )
    result = await _run_command(session, f'powershell -NoProfile -Command "{ps}"', check=False)
    stdout = _as_text(result.get("stdout", "")).strip()
    if not stdout or "__MISSING__" in stdout:
        return None
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if line.isdigit():
            return int(line)
    return None


def _load_gray_image(raw: bytes) -> np.ndarray:
    image = Image.open(BytesIO(raw)).convert("L")
    return np.asarray(image, dtype=np.float32) / 255.0


def _foreground_mask(gray: np.ndarray) -> np.ndarray:
    return np.abs(gray - BACKGROUND_GRAY) > MASK_THRESHOLD


def _mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    intersection = float(np.logical_and(mask_a, mask_b).sum())
    union = float(np.logical_or(mask_a, mask_b).sum())
    if union <= 0.0:
        return 1.0
    return intersection / union


def _render_similarity(reference_png: bytes, candidate_png: bytes) -> dict[str, float]:
    ref = _load_gray_image(reference_png)
    cand = _load_gray_image(candidate_png)
    if ref.shape != cand.shape:
        raise RuntimeError(f"render shape mismatch: reference={ref.shape} candidate={cand.shape}")
    ssim = float(structural_similarity(ref, cand, data_range=1.0))
    iou = _mask_iou(_foreground_mask(ref), _foreground_mask(cand))
    combined = (0.7 * ssim) + (0.3 * iou)
    return {
        "ssim": ssim,
        "mask_iou": iou,
        "combined": float(combined),
    }


def _sample_vertices(vertices: np.ndarray, limit: int) -> np.ndarray:
    if len(vertices) <= limit:
        return vertices
    step = max(1, len(vertices) // limit)
    sampled = vertices[::step]
    return sampled[:limit]


def _parse_obj_vertices(raw: bytes) -> np.ndarray:
    vertices: list[tuple[float, float, float]] = []
    for line in raw.decode("utf-8", errors="ignore").splitlines():
        if not line.startswith("v "):
            continue
        parts = line.split()
        if len(parts) < 4:
            continue
        try:
            vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
        except ValueError:
            continue
    if not vertices:
        raise RuntimeError("OBJ has no readable vertices")
    return np.asarray(vertices, dtype=np.float64)


def _bbox_diagonal(vertices: np.ndarray) -> float:
    mins = np.min(vertices, axis=0)
    maxs = np.max(vertices, axis=0)
    return float(np.linalg.norm(maxs - mins))


def _geometry_similarity(
    reference_obj: bytes,
    candidate_obj: bytes,
    sample_limit: int,
) -> dict[str, float]:
    ref_vertices = _sample_vertices(_parse_obj_vertices(reference_obj), sample_limit)
    cand_vertices = _sample_vertices(_parse_obj_vertices(candidate_obj), sample_limit)
    reference_scale = max(_bbox_diagonal(ref_vertices), 1e-6)
    ref_tree = cKDTree(ref_vertices)
    cand_tree = cKDTree(cand_vertices)
    ref_to_cand = cand_tree.query(ref_vertices, k=1)[0]
    cand_to_ref = ref_tree.query(cand_vertices, k=1)[0]
    chamfer = float((ref_to_cand.mean() + cand_to_ref.mean()) * 0.5)
    normalized = chamfer / reference_scale
    score = float(math.exp(-14.0 * normalized))
    return {
        "sample_limit": float(sample_limit),
        "reference_scale": reference_scale,
        "mean_chamfer": chamfer,
        "normalized_chamfer": normalized,
        "score": score,
    }


def _bbox_similarity(scale_guide: dict[str, Any], metrics: dict[str, Any]) -> dict[str, float]:
    ref_box = scale_guide["computed_bounding_box"]
    ref_center = np.asarray(ref_box["center"], dtype=np.float64)
    ref_extent = np.asarray(ref_box["extent"], dtype=np.float64)
    cand_center = np.asarray(metrics["bbox_center"], dtype=np.float64)
    cand_extent = np.asarray(metrics["bbox_extent"], dtype=np.float64)
    center_offset = float(np.linalg.norm(cand_center - ref_center))
    center_limit = float(
        scale_guide["allowed_bounding_box_ranges"]["max_center_offset_world_units"]
    )
    center_score = _clamp01(1.0 - (center_offset / max(center_limit * 4.0, 1e-6)))

    extent_ratios = cand_extent / np.maximum(ref_extent, 1e-6)
    extent_scores = [_clamp01(1.0 - (abs(float(ratio) - 1.0) / 0.35)) for ratio in extent_ratios]
    score = (0.4 * center_score) + (0.6 * (sum(extent_scores) / len(extent_scores)))
    return {
        "center_offset": center_offset,
        "center_score": center_score,
        "extent_ratio_x": float(extent_ratios[0]),
        "extent_ratio_y": float(extent_ratios[1]),
        "extent_ratio_z": float(extent_ratios[2]),
        "score": float(score),
    }


async def _find_first_existing_path(
    session: cb.DesktopSession, candidates: list[str]
) -> str | None:
    for candidate in candidates:
        if (await session.file_exists(candidate) or await session.directory_exists(candidate)):
            return candidate
    return None


async def _blend_can_open(
    session: cb.DesktopSession,
    *,
    blender_exe: str,
    blend_path: str,
) -> tuple[bool, str]:
    command = (
        f'cmd /c ""{blender_exe}" '
        f'--background --factory-startup "{blend_path}" '
        '--python-expr "print(\'BLEND_OPEN_CHECK_OK\')""'
    )
    result = await _run_command(session, command, timeout=3600.0, check=False)
    stdout = _as_text(result.get("stdout", ""))
    stderr = _as_text(result.get("stderr", ""))
    ok = result.get("return_code", 1) == 0 and "BLEND_OPEN_CHECK_OK" in stdout
    return ok, (stdout + "\n" + stderr).strip()


class BlenderCharacterTaskConfig(GeneralTaskConfig):
    def __init__(
        self,
        *,
        REMOTE_OUTPUT_DIR: str | None = None,
        REMOTE_ROOT_DIR: str | None = None,
        DOMAIN_NAME: str = "visual_media",
        TASK_NAME: str = TASK_NAME,
        OS_TYPE: str = "windows",
    ) -> None:
        super().__init__(
            REMOTE_OUTPUT_DIR=REMOTE_OUTPUT_DIR or os.environ.get("REMOTE_OUTPUT_DIR", "output"),
            REMOTE_ROOT_DIR=REMOTE_ROOT_DIR or os.environ.get("REMOTE_ROOT_DIR", r"E:\agenthle"),
            DOMAIN_NAME=DOMAIN_NAME,
            TASK_NAME=TASK_NAME,
            OS_TYPE=OS_TYPE,
            VARIANT_NAME=VARIANT_NAME,
        )

    @property
    def input_dir(self) -> str:
        return _remote_child(self.task_dir, "input")

    @property
    def output_submission_dir(self) -> str:
        return _remote_child(self.remote_output_dir, "submission")

    @property
    def output_blend(self) -> str:
        return _remote_child(self.output_submission_dir, "final.blend")

    @property
    def output_glb(self) -> str:
        return _remote_child(self.output_submission_dir, "reconstructed_character.glb")

    @property
    def output_obj(self) -> str:
        return _remote_child(self.output_submission_dir, "reconstructed_character.obj")

    @property
    def output_report(self) -> str:
        return _remote_child(self.output_submission_dir, "modeling_report.md")

    @property
    def reference_blend(self) -> str:
        return _remote_child(self.reference_dir, "final.blend")

    @property
    def reference_glb(self) -> str:
        return _remote_child(self.reference_dir, "reconstructed_character.glb")

    @property
    def reference_obj(self) -> str:
        return _remote_child(self.reference_dir, "reconstructed_character.obj")

    @property
    def reference_report(self) -> str:
        return _remote_child(self.reference_dir, "modeling_report.md")

    @property
    def scale_guide(self) -> str:
        return _remote_child(self.reference_dir, "scale_orientation_guide.json")

    @property
    def validation_views(self) -> str:
        return _remote_child(self.reference_dir, "validation_views.json")

    @property
    def evaluation_config(self) -> str:
        return _remote_child(self.reference_dir, "evaluation_config.json")

    @property
    def software_launcher(self) -> str:
        return _remote_child(self.software_dir, "open_blender.bat")

    @property
    def task_description(self) -> str:
        return textwrap.dedent(f"""\
            You are a 3D artist using Blender on a Windows VM.

            Reconstruct the provided stylized character from the staged multiview references.

            Agent-visible input:
            - Modeling brief: `{_remote_child(self.input_dir, "modeling_brief.md")}`
            - Scale/orientation guide: `{_remote_child(self.input_dir, "scale_orientation_guide.json")}`
            - Validation camera guide: `{_remote_child(self.input_dir, "validation_views.json")}`
            - Views: `view_front.png`, `view_side.png`, `view_back.png`, `view_front_three_quarter.png`, `view_back_three_quarter.png`
            - Blender launcher: `{self.software_launcher}`

            Required submission under `{self.output_submission_dir}`:
            - `final.blend`
            - `reconstructed_character.glb`
            - `reconstructed_character.obj`
            - `modeling_report.md`

            Requirements:
            - Preserve the major full-body silhouette, proportions, and component structure.
            - Match the coordinate and scale guidance from `scale_orientation_guide.json`.
            - Treat the five staged views as the visual target.
            - Write the required deliverables only under the designated `output/submission/` path.
            """)

    def to_metadata(self) -> dict[str, Any]:
        data = super().to_metadata()
        data.update(
            {
                "task_id": TASK_ID,
                "input_dir": self.input_dir,
                "output_submission_dir": self.output_submission_dir,
                "output_blend": self.output_blend,
                "output_glb": self.output_glb,
                "output_obj": self.output_obj,
                "output_report": self.output_report,
                "reference_blend": self.reference_blend,
                "reference_glb": self.reference_glb,
                "reference_obj": self.reference_obj,
                "reference_report": self.reference_report,
                "scale_guide": self.scale_guide,
                "validation_views": self.validation_views,
                "evaluation_config": self.evaluation_config,
                "software_launcher": self.software_launcher,
                "blender_exe": BLENDER_EXE,
            }
        )
        return data


config = BlenderCharacterTaskConfig()


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


@cb.evaluate_task(split="train")
async def evaluate(task_cfg, session: cb.DesktopSession) -> list[float]:
    meta = task_cfg.metadata
    scale_guide = await _read_json(session, meta["scale_guide"])
    views_config = await _read_json(session, meta["validation_views"])
    evaluation_config = await _read_json(session, meta["evaluation_config"])

    selected = {
        "name": "submission",
        "blend": meta["output_blend"],
        "glb": meta["output_glb"],
        "obj": meta["output_obj"],
        "report": meta["output_report"],
    }

    sizes = {}
    for key in ["blend", "glb", "obj", "report"]:
        size = await _remote_file_size(session, selected[key])
        sizes[key] = size
        if size is None or size <= 0:
            logger.error("Missing or empty required submission file: %s", selected[key])
            return [0.0]

    blender_exe = meta["blender_exe"]
    if not (await session.file_exists(blender_exe) or await session.directory_exists(blender_exe)):
        raise RuntimeError(
            f"Blender missing from VM snapshot at {blender_exe}; "
            "Stage 1 snapshot must preinstall Blender (do not patch from evaluate())"
        )

    blend_ok, blend_log = await _blend_can_open(
        session,
        blender_exe=blender_exe,
        blend_path=selected["blend"],
    )
    if not blend_ok:
        logger.error("Submitted blend did not open cleanly. log=%s", blend_log[-800:])
        return [0.0]

    await session.interface.create_dir(EVAL_TMP_DIR)
    helper_path = _remote_child(EVAL_TMP_DIR, "blender_eval_submission.py")
    metrics_path = _remote_child(EVAL_TMP_DIR, "submission_metrics.json")
    render_dir = _remote_child(EVAL_TMP_DIR, "renders")
    await session.write_file(
        helper_path,
        (TASK_DIR / "scripts" / "blender_eval_submission.py").read_text(encoding="utf-8"),
    )

    eval_command = (
        f'cmd /c ""{blender_exe}" '
        "--background --factory-startup "
        f'--python "{helper_path}" '
        f'-- --submission-obj "{selected["obj"]}" '
        f'--views-config "{meta["validation_views"]}" '
        f'--output-json "{metrics_path}" '
        f'--render-dir "{render_dir}""'
    )
    eval_result = await _run_command(session, eval_command, timeout=3600.0, check=False)
    if eval_result.get("return_code", 1) != 0:
        logger.error(
            "Blender render helper failed. stdout_tail=%s stderr_tail=%s",
            _as_text(eval_result.get("stdout", ""))[-800:],
            _as_text(eval_result.get("stderr", ""))[-800:],
        )
        return [0.0]
    if not (await session.file_exists(metrics_path) or await session.directory_exists(metrics_path)):
        logger.error("Expected metrics output missing: %s", metrics_path)
        return [0.0]

    metrics = await _read_json(session, metrics_path)
    if metrics.get("mesh_object_count", 0) <= 0 or metrics.get("vertex_count", 0) <= 0:
        logger.error("Imported OBJ was empty or unreadable: %s", metrics)
        return [0.0]

    geometry = _geometry_similarity(
        await _read_bytes(session, meta["reference_obj"]),
        await _read_bytes(session, selected["obj"]),
        int(evaluation_config.get("geometry_downsample_limit", OBJ_SAMPLE_LIMIT_DEFAULT)),
    )
    bbox = _bbox_similarity(scale_guide, metrics)

    view_scores: list[dict[str, Any]] = []
    per_view = views_config.get("per_view_cameras", [])
    for view in per_view:
        output_name = PureWindowsPath(str(view["output_image_path"])).name
        reference_path = _remote_child(meta["reference_dir"], output_name)
        candidate_path = _remote_child(render_dir, output_name)
        if not (await session.file_exists(reference_path) or await session.directory_exists(reference_path)) or not (await session.file_exists(candidate_path) or await session.directory_exists(candidate_path)):
            logger.error(
                "Missing render pair for %s. ref_exists=%s cand_exists=%s",
                output_name,
                (await session.file_exists(reference_path) or await session.directory_exists(reference_path)),
                (await session.file_exists(candidate_path) or await session.directory_exists(candidate_path)),
            )
            return [0.0]
        similarity = _render_similarity(
            await _read_bytes(session, reference_path),
            await _read_bytes(session, candidate_path),
        )
        similarity["view_name"] = str(view["view_name"])
        view_scores.append(similarity)

    render_score = float(sum(item["combined"] for item in view_scores) / max(len(view_scores), 1))
    weights = evaluation_config["weights"]
    final_score = (
        float(weights["geometry_score"]) * geometry["score"]
        + float(weights["render_score"]) * render_score
        + float(weights["bbox_score"]) * bbox["score"]
    )

    payload = {
        "sizes": sizes,
        "blend_open_ok": blend_ok,
        "geometry": geometry,
        "bbox": bbox,
        "metrics": metrics,
        "view_scores": view_scores,
        "render_score": render_score,
        "final_score": final_score,
        "pass_threshold": float(evaluation_config["pass_threshold"]),
    }
    logger.info("Evaluation payload: %s", json.dumps(payload, ensure_ascii=False))
    return [float(final_score)]
