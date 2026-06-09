from __future__ import annotations

import logging
import math
import os
import textwrap
from dataclasses import dataclass
from pathlib import PureWindowsPath
from typing import Any

import cua_bench as cb
import numpy as np

from tasks.common_config import GeneralTaskConfig
from tasks.common_setup import BaseTaskSetup

_setup = BaseTaskSetup()

try:
    from scipy.spatial import cKDTree
except Exception:  # pragma: no cover - scipy is expected, but keep a safe fallback
    cKDTree = None

logger = logging.getLogger(__name__)

TASK_ID = "visual_media/human_mesh_animation_reproduction"
TASK_NAME = "human_mesh_animation_reproduction"
VARIANT_NAME = "base"
FRAME_START = 1
FRAME_END = 60
FRAME_COUNT = 60
FPS = 30
BLENDER_VERSION = "4.3"
APPROVED_BLENDER_RUNTIME_VERSION = "5.0.1"
APPROVED_BLENDER_RUNTIME_MAJOR = "5.0"
SURFACE_SAMPLE_COUNT_TARGET = 100_000
VERTEX_SAMPLE_FALLBACK_TARGET = 4_096
CHAMFER_PAIRWISE_FALLBACK_TARGET = 20_000


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


def _frame_name(frame_number: int) -> str:
    return f"frame_{frame_number:04d}.obj"


def _sample_vertices(
    vertices: np.ndarray,
    target: int = VERTEX_SAMPLE_FALLBACK_TARGET,
) -> np.ndarray:
    if len(vertices) <= target:
        return vertices
    step = max(1, len(vertices) // target)
    sampled = vertices[::step]
    return sampled[:target]


def _pairwise_min_distances(src: np.ndarray, dst: np.ndarray, chunk_size: int = 256) -> np.ndarray:
    mins = np.empty(len(src), dtype=np.float64)
    for start in range(0, len(src), chunk_size):
        chunk = src[start : start + chunk_size]
        diffs = chunk[:, None, :] - dst[None, :, :]
        dists = np.sqrt(np.sum(diffs * diffs, axis=2))
        mins[start : start + len(chunk)] = dists.min(axis=1)
    return mins


@dataclass(frozen=True)
class ObjMesh:
    vertices: np.ndarray
    triangles: np.ndarray


async def _read_obj_mesh(session: cb.DesktopSession, path: str) -> ObjMesh:
    raw = await session.read_file(path)
    vertices: list[tuple[float, float, float]] = []
    triangles: list[tuple[int, int, int]] = []
    for line in _as_text(raw).splitlines():
        if line.startswith("v "):
            parts = line.split()
            if len(parts) < 4:
                continue
            try:
                vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
            except ValueError:
                continue
            continue

        if not line.startswith("f "):
            continue

        tokens = line.split()[1:]
        if len(tokens) < 3:
            continue

        face_indices: list[int] = []
        for token in tokens:
            first = token.split("/")[0]
            if not first:
                face_indices = []
                break
            try:
                idx = int(first)
            except ValueError:
                face_indices = []
                break

            if idx > 0:
                zero_based = idx - 1
            elif idx < 0:
                zero_based = len(vertices) + idx
            else:
                face_indices = []
                break

            if zero_based < 0 or zero_based >= len(vertices):
                face_indices = []
                break
            face_indices.append(zero_based)

        if len(face_indices) < 3:
            continue

        # Fan triangulation keeps evaluation faithful for OBJ polygons.
        anchor = face_indices[0]
        for i in range(1, len(face_indices) - 1):
            triangles.append((anchor, face_indices[i], face_indices[i + 1]))

    if not vertices:
        raise RuntimeError(f"OBJ has no readable vertices: {path}")

    triangle_array = (
        np.asarray(triangles, dtype=np.int32) if triangles else np.empty((0, 3), dtype=np.int32)
    )
    return ObjMesh(vertices=np.asarray(vertices, dtype=np.float64), triangles=triangle_array)


def _surface_triangle_data(mesh: ObjMesh) -> tuple[np.ndarray, np.ndarray]:
    if len(mesh.triangles) == 0:
        return (
            np.empty((0, 3, 3), dtype=np.float64),
            np.empty(0, dtype=np.float64),
        )

    triangle_vertices = mesh.vertices[mesh.triangles]
    edge1 = triangle_vertices[:, 1] - triangle_vertices[:, 0]
    edge2 = triangle_vertices[:, 2] - triangle_vertices[:, 0]
    areas = 0.5 * np.linalg.norm(np.cross(edge1, edge2), axis=1)
    valid = areas > 1e-12
    return triangle_vertices[valid], areas[valid]


def _sample_surface_points(
    mesh: ObjMesh,
    rng: np.random.Generator,
    target_count: int,
) -> tuple[np.ndarray | None, str]:
    triangle_vertices, areas = _surface_triangle_data(mesh)
    if len(triangle_vertices) == 0:
        return None, "vertex_fallback_no_faces"

    total_area = float(areas.sum())
    if total_area <= 1e-12:
        return None, "vertex_fallback_degenerate_faces"

    triangle_indices = rng.choice(
        len(triangle_vertices),
        size=target_count,
        replace=True,
        p=areas / total_area,
    )
    chosen = triangle_vertices[triangle_indices]

    u = rng.random(target_count)
    v = rng.random(target_count)
    sqrt_u = np.sqrt(u)
    bary_a = 1.0 - sqrt_u
    bary_b = sqrt_u * (1.0 - v)
    bary_c = sqrt_u * v
    samples = (
        chosen[:, 0] * bary_a[:, None]
        + chosen[:, 1] * bary_b[:, None]
        + chosen[:, 2] * bary_c[:, None]
    )
    return samples, "surface"


def _chamfer_distance(ref_points: np.ndarray, candidate_points: np.ndarray) -> float | None:
    if len(ref_points) == 0 or len(candidate_points) == 0:
        return None
    if cKDTree is not None:
        ref_tree = cKDTree(ref_points)
        cand_tree = cKDTree(candidate_points)
        ref_to_cand = cand_tree.query(ref_points, k=1)[0]
        cand_to_ref = ref_tree.query(candidate_points, k=1)[0]
    else:
        ref_to_cand = _pairwise_min_distances(ref_points, candidate_points)
        cand_to_ref = _pairwise_min_distances(candidate_points, ref_points)
    return float((ref_to_cand.mean() + cand_to_ref.mean()) * 0.5)


def _normalized_chamfer_score(mean_ncd: float) -> float:
    return float(max(0.0, min(1.0, math.exp(-40.0 * mean_ncd))))


@dataclass
class HumanMeshAnimationReproductionConfig(GeneralTaskConfig):
    DOMAIN_NAME: str = "visual_media"
    TASK_NAME: str = "human_mesh_animation_reproduction"
    VARIANT_NAME: str = VARIANT_NAME
    OS_TYPE: str = "windows"

    @property
    def task_dir(self) -> str:
        return _remote_child(
            self.REMOTE_ROOT_DIR, self.DOMAIN_NAME, self.TASK_NAME, self.VARIANT_NAME
        )

    @property
    def input_dir(self) -> str:
        return _remote_child(self.task_dir, "input")

    @property
    def input_obj(self) -> str:
        return _remote_child(self.input_dir, "character.obj")

    @property
    def input_mtl(self) -> str:
        return _remote_child(self.input_dir, "character.mtl")

    @property
    def reference_video(self) -> str:
        return _remote_child(self.input_dir, "reference.mp4")

    @property
    def output_submission_dir(self) -> str:
        return _remote_child(self.remote_output_dir, "submission")

    @property
    def output_blend(self) -> str:
        return _remote_child(self.output_submission_dir, "final.blend")

    @property
    def output_mesh_seq_dir(self) -> str:
        return _remote_child(self.output_submission_dir, "mesh_seq")

    @property
    def reference_blend(self) -> str:
        return _remote_child(self.reference_dir, "final.blend")

    @property
    def reference_mesh_seq_dir(self) -> str:
        return _remote_child(self.reference_dir, "mesh_seq")

    @property
    def reference_contract(self) -> str:
        return _remote_child(self.reference_dir, "reference_contract.json")

    @property
    def software_notes(self) -> str:
        return _remote_child(self.software_dir, "README.txt")

    @property
    def blender_launcher_candidates(self) -> list[str]:
        env_candidates = [
            os.environ.get("BLENDER_50_EXECUTABLE", "").strip(),
            os.environ.get("BLENDER_43_EXECUTABLE", "").strip(),
            os.environ.get("BLENDER_TASK_REMOTE_BLENDER", "").strip(),
        ]
        default_candidates = [
            rf"C:\Program Files\Blender Foundation\Blender {APPROVED_BLENDER_RUNTIME_MAJOR}\blender.exe",
            rf"C:\Program Files\Blender Foundation\Blender {BLENDER_VERSION}\blender.exe",
            rf"C:\Program Files\Blender Foundation\Blender {BLENDER_VERSION} LTS\blender.exe",
            rf"D:\Blender\Blender {APPROVED_BLENDER_RUNTIME_MAJOR}\blender.exe",
            rf"D:\Blender\Blender {BLENDER_VERSION}\blender.exe",
            _remote_child(
                self.software_dir,
                rf"Blender {APPROVED_BLENDER_RUNTIME_MAJOR}",
                "blender.exe",
            ),
            _remote_child(self.software_dir, rf"Blender {BLENDER_VERSION}", "blender.exe"),
            _remote_child(self.software_dir, "blender.exe"),
        ]
        deduped: list[str] = []
        for candidate in env_candidates + default_candidates:
            if candidate and candidate not in deduped:
                deduped.append(candidate)
        return deduped

    @property
    def task_description(self) -> str:
        return textwrap.dedent(f"""\
            You are a 3D artist using Blender {BLENDER_VERSION} on a Windows VM.

            Rig and animate the provided human mesh so it matches the motion shown in the reference video.

            Agent-visible inputs:
            - Mesh: `{self.input_obj}`
            - Material sidecar: `{self.input_mtl}`
            - Reference motion video: `{self.reference_video}`

            Required submission:
            - Save the Blender scene to `{self.output_blend}`
            - Export exactly {FRAME_COUNT} OBJ frames to `{self.output_mesh_seq_dir}`
            - Use frame names `{_frame_name(FRAME_START)}` through `{_frame_name(FRAME_END)}`

            Requirements:
            - Animate at {FPS} fps.
            - Export frames {FRAME_START}..{FRAME_END} inclusive.
            - Preserve the character identity, proportions, topology, and overall geometry while reproducing the motion.
            """)

    def to_metadata(self) -> dict[str, Any]:
        metadata = super().to_metadata()
        metadata.update(
            {
                "task_id": TASK_ID,
                "task_dir": self.task_dir,
                "input_dir": self.input_dir,
                "input_obj": self.input_obj,
                "input_mtl": self.input_mtl,
                "reference_video": self.reference_video,
                "output_submission_dir": self.output_submission_dir,
                "output_blend": self.output_blend,
                "output_mesh_seq_dir": self.output_mesh_seq_dir,
                "reference_blend": self.reference_blend,
                "reference_mesh_seq_dir": self.reference_mesh_seq_dir,
                "reference_contract": self.reference_contract,
                "software_notes": self.software_notes,
                "frame_start": FRAME_START,
                "frame_end": FRAME_END,
                "frame_count": FRAME_COUNT,
                "fps": FPS,
                "blender_version_required": BLENDER_VERSION,
                "blender_runtime_version_approved": APPROVED_BLENDER_RUNTIME_VERSION,
                "blender_launcher_candidates": list(self.blender_launcher_candidates),
            }
        )
        return metadata


config = HumanMeshAnimationReproductionConfig()


@cb.tasks_config(split="train")
def load():
    return [
        cb.Task(
            description=config.task_description,
            metadata=config.to_metadata(),
            computer={"provider": "computer", "setup_config": {"os_type": config.OS_TYPE}},
        )
    ]


async def _path_missing(
    session: cb.DesktopSession,
    path: str,
    *,
    tag: str,
    label: str,
) -> bool:
    if (await session.file_exists(path) or await session.directory_exists(path)):
        return False
    logger.error("[%s] Missing staged %s at %s", tag, label, path)
    return True


async def _find_first_existing_path(
    session: cb.DesktopSession,
    candidates: list[str],
) -> str | None:
    for path in candidates:
        if (await session.file_exists(path) or await session.directory_exists(path)):
            return path
    return None


async def _remote_file_size(session: cb.DesktopSession, path: str) -> int | None:
    ps = (
        f"$p = '{_ps_quote(path)}'; "
        "if (Test-Path -LiteralPath $p) { "
        "(Get-Item -LiteralPath $p).Length "
        "} else { "
        "Write-Output '__MISSING__' "
        "}"
    )
    result = await session.run_command(f'powershell -NoProfile -Command "{ps}"', check=False)
    stdout = _as_text(result.get("stdout", "")).strip()
    if not stdout or "__MISSING__" in stdout:
        return None
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if line.isdigit():
            return int(line)
    return None


async def _remote_obj_count(session: cb.DesktopSession, directory: str) -> int | None:
    ps = (
        f"$p = '{_ps_quote(directory)}'; "
        "if (Test-Path -LiteralPath $p) { "
        "(Get-ChildItem -LiteralPath $p -Filter '*.obj' -File | Measure-Object).Count "
        "} else { "
        "Write-Output '__MISSING__' "
        "}"
    )
    result = await session.run_command(f'powershell -NoProfile -Command "{ps}"', check=False)
    stdout = _as_text(result.get("stdout", "")).strip()
    if not stdout or "__MISSING__" in stdout:
        return None
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if line.isdigit():
            return int(line)
    return None


async def _blender_can_open_blend(
    session: cb.DesktopSession,
    *,
    blender_path: str,
    blend_path: str,
) -> tuple[bool, str, str]:
    ps = (
        f"$exe = '{_ps_quote(blender_path)}'; "
        f"$blend = '{_ps_quote(blend_path)}'; "
        "& $exe --background --factory-startup $blend --python-expr "
        "\"print('BLEND_OPEN_CHECK_OK')\""
    )
    result = await session.run_command(
        f'powershell -NoProfile -Command "{ps}"',
        check=False,
    )
    stdout = _as_text(result.get("stdout", ""))
    stderr = _as_text(result.get("stderr", ""))
    opened = "Read blend:" in stdout
    marker = "BLEND_OPEN_CHECK_OK" in stdout
    return_code = result.get("return_code", 1)
    ok = bool(opened and marker and return_code == 0)
    return ok, stdout, stderr


def _bbox_diagonal(vertices: np.ndarray) -> float:
    mins = np.min(vertices, axis=0)
    maxs = np.max(vertices, axis=0)
    return float(np.linalg.norm(maxs - mins))


def _frame_rng(frame_number: int) -> np.random.Generator:
    return np.random.default_rng(20260407 + frame_number)


def _frame_eval_points(
    mesh: ObjMesh,
    *,
    frame_number: int,
) -> tuple[np.ndarray, str, int]:
    rng = _frame_rng(frame_number)
    sample_count = SURFACE_SAMPLE_COUNT_TARGET
    if cKDTree is None:
        # The pure NumPy fallback is O(n*m), so cap points to keep Stage 2 runnable.
        sample_count = CHAMFER_PAIRWISE_FALLBACK_TARGET

    surface_points, mode = _sample_surface_points(mesh, rng, sample_count)
    if surface_points is not None:
        return surface_points, mode, sample_count

    fallback_target = (
        VERTEX_SAMPLE_FALLBACK_TARGET if cKDTree is not None else CHAMFER_PAIRWISE_FALLBACK_TARGET
    )
    return _sample_vertices(mesh.vertices, fallback_target), mode, fallback_target


def _submission_layouts(meta: dict[str, Any]) -> list[dict[str, str]]:
    layouts = [
        {
            "name": "submission",
            "blend": meta["output_blend"],
            "mesh_seq_dir": meta["output_mesh_seq_dir"],
        }
    ]
    root_blend = _remote_child(meta["remote_output_dir"], "final.blend")
    root_mesh_seq = _remote_child(meta["remote_output_dir"], "mesh_seq")
    if root_blend != meta["output_blend"]:
        layouts.append(
            {
                "name": "output_root_fallback",
                "blend": root_blend,
                "mesh_seq_dir": root_mesh_seq,
            }
        )
    return layouts


@cb.setup_task(split="train")
async def start(task_cfg, session: cb.DesktopSession):
    await _setup(task_cfg, session)


@cb.evaluate_task(split="train")
async def evaluate(task_cfg, session: cb.DesktopSession):
    meta = task_cfg.metadata
    tag = meta["variant_name"]

    selected_layout = _submission_layouts(meta)[0]
    for layout in _submission_layouts(meta):
        if (await session.file_exists(layout["blend"]) or await session.directory_exists(layout["blend"])) or (await session.file_exists(layout["mesh_seq_dir"]) or await session.directory_exists(layout["mesh_seq_dir"])):
            selected_layout = layout
            break

    blend_path = selected_layout["blend"]
    mesh_seq_dir = selected_layout["mesh_seq_dir"]

    if not (await session.file_exists(blend_path) or await session.directory_exists(blend_path)):
        logger.error("[%s] Missing submitted final.blend at %s", tag, blend_path)
        return [0.0]

    blend_size = await _remote_file_size(session, blend_path)
    if blend_size is None or blend_size <= 0:
        logger.error("[%s] Submitted final.blend is empty or unreadable at %s", tag, blend_path)
        return [0.0]

    candidate_count = await _remote_obj_count(session, mesh_seq_dir)
    if candidate_count != FRAME_COUNT:
        logger.error(
            "[%s] mesh_seq must contain exactly %d OBJ files, observed=%s at %s",
            tag,
            FRAME_COUNT,
            candidate_count,
            mesh_seq_dir,
        )
        return [0.0]

    reference_count = await _remote_obj_count(session, meta["reference_mesh_seq_dir"])
    if reference_count != FRAME_COUNT:
        logger.error(
            "[%s] hidden reference mesh_seq is incomplete, expected=%d observed=%s",
            tag,
            FRAME_COUNT,
            reference_count,
        )
        return [0.0]

    blender_path = await _find_first_existing_path(
        session, list(meta["blender_launcher_candidates"])
    )
    if blender_path is None:
        logger.warning(
            "[%s] Blender %s is still unavailable during evaluate(); blend validation is limited to "
            "existence and non-zero file size on this VM.",
            tag,
            BLENDER_VERSION,
        )
    else:
        blend_open_ok, blend_stdout, blend_stderr = await _blender_can_open_blend(
            session,
            blender_path=blender_path,
            blend_path=blend_path,
        )
        if not blend_open_ok:
            logger.error(
                "[%s] Submitted final.blend could not be opened by approved Blender runtime %s at %s. "
                "stdout_tail=%s stderr_tail=%s",
                tag,
                meta["blender_runtime_version_approved"],
                blender_path,
                blend_stdout[-400:],
                blend_stderr[-400:],
            )
            return [0.0]
    if cKDTree is None:
        logger.warning(
            "[%s] scipy.spatial.cKDTree is unavailable; falling back to %d-point symmetric Chamfer "
            "to keep Stage 2 evaluation tractable.",
            tag,
            CHAMFER_PAIRWISE_FALLBACK_TARGET,
        )

    try:
        input_mesh = await _read_obj_mesh(session, meta["input_obj"])
    except Exception as exc:
        logger.error("[%s] Failed to parse input character OBJ for normalization: %s", tag, exc)
        return [0.0]

    normalization_d = _bbox_diagonal(input_mesh.vertices)
    if normalization_d <= 1e-9:
        logger.error(
            "[%s] Input character OBJ has a degenerate bbox diagonal at %s", tag, meta["input_obj"]
        )
        return [0.0]

    normalized_distances: list[float] = []
    sample_modes: dict[str, int] = {}
    sample_counts: dict[int, int] = {}
    for frame_number in range(FRAME_START, FRAME_END + 1):
        filename = _frame_name(frame_number)
        candidate_path = _remote_child(mesh_seq_dir, filename)
        reference_path = _remote_child(meta["reference_mesh_seq_dir"], filename)

        if not (await session.file_exists(candidate_path) or await session.directory_exists(candidate_path)):
            logger.error("[%s] Missing candidate frame %s", tag, candidate_path)
            return [0.0]
        if not (await session.file_exists(reference_path) or await session.directory_exists(reference_path)):
            logger.error("[%s] Missing reference frame %s", tag, reference_path)
            return [0.0]

        try:
            candidate_mesh = await _read_obj_mesh(session, candidate_path)
            reference_mesh = await _read_obj_mesh(session, reference_path)
        except Exception as exc:
            logger.error("[%s] Failed to parse OBJ frame %s: %s", tag, filename, exc)
            return [0.0]

        reference_points, reference_mode, point_count = _frame_eval_points(
            reference_mesh,
            frame_number=frame_number,
        )
        candidate_points, candidate_mode, candidate_point_count = _frame_eval_points(
            candidate_mesh,
            frame_number=frame_number,
        )
        sample_modes[reference_mode] = sample_modes.get(reference_mode, 0) + 1
        sample_modes[candidate_mode] = sample_modes.get(candidate_mode, 0) + 1
        sample_counts[point_count] = sample_counts.get(point_count, 0) + 1
        sample_counts[candidate_point_count] = sample_counts.get(candidate_point_count, 0) + 1

        chamfer = _chamfer_distance(reference_points, candidate_points)
        if chamfer is None:
            logger.error("[%s] Chamfer distance could not be computed for %s", tag, filename)
            return [0.0]

        normalized_distances.append(float(chamfer / normalization_d))

    mean_ncd = float(sum(normalized_distances) / len(normalized_distances))
    final_score = _normalized_chamfer_score(mean_ncd)
    logger.info(
        "[%s] layout=%s blend_size=%s obj_count=%s normalization_d=%.8f "
        "surface_target=%d sample_modes=%s sample_counts=%s mean_ncd=%.8f final_score=%.6f",
        tag,
        selected_layout["name"],
        blend_size,
        candidate_count,
        normalization_d,
        SURFACE_SAMPLE_COUNT_TARGET,
        sample_modes,
        sample_counts,
        mean_ncd,
        final_score,
    )
    return [final_score]
