from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import textwrap
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath

import cua_bench as cb

from tasks.common_config import GeneralTaskConfig
from tasks.common_setup import BaseTaskSetup
from tasks.visual_media.object_generation.scripts.local_soft_eval import \
    run_local_soft_eval

_setup = BaseTaskSetup()

logger = logging.getLogger(__name__)

TASK_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = TASK_DIR / "scripts"
WORKFLOW = "object_generation"
REMOTE_PYTHON = os.environ.get(
    "BLENDER_TASK_REMOTE_PYTHON",
    r"C:\Users\User\AppData\Local\Programs\Python\Python312\python.exe",
)
REMOTE_BLENDER = os.environ.get(
    "BLENDER_TASK_REMOTE_BLENDER", r"C:\Program Files\Blender Foundation\Blender 5.0\blender.exe"
)
REMOTE_TEMP_ROOT = os.environ.get(
    "BLENDER_TASK_REMOTE_EVAL_ROOT",
    r"C:\Users\User\AppData\Local\Temp\agenthle_eval\object_generation",
)
VARIANT_CATALOG_PATH = TASK_DIR / "variant_catalog.json"


def _load_variants() -> list[dict]:
    raw = json.loads(VARIANT_CATALOG_PATH.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise RuntimeError(f"Expected list variant catalog: {VARIANT_CATALOG_PATH}")
    variants: list[dict] = []
    for item in raw:
        if not isinstance(item, dict):
            raise RuntimeError(f"Expected dict variant entry: {item!r}")
        variants.append(item)
    return variants


VARIANTS = _load_variants()
VARIANTS_BY_TAG = {item["task_tag"]: item for item in VARIANTS}


def _remote_child(base: str, *parts: str) -> str:
    path = PureWindowsPath(base)
    for part in parts:
        if part:
            path = path / part
    return str(path)


def _ps_quote(text: str) -> str:
    return text.replace("'", "''")


def _ps_literal(text: str) -> str:
    return f"'{_ps_quote(text)}'"


@dataclass
class ObjectGenerationTaskConfig(GeneralTaskConfig):
    DOMAIN_NAME: str = "visual_media"

    TASK_NAME: str = "object_generation"
    VARIANT_NAME: str = ""
    TASK_TAG: str = ""
    DISPLAY_NAME: str = ""
    REMOTE_TASK_DIR_NAME: str = ""
    MULTIPART_SAMPLE_RATIO: float = 0.20
    MULTIPART_MIN_SAMPLES: int = 3
    MULTIPART_MAX_SAMPLES: int = 8

    @property
    def task_dir(self) -> str:
        return _remote_child(self.REMOTE_ROOT_DIR, self.DOMAIN_NAME, self.REMOTE_TASK_DIR_NAME)

    @property
    def input_dir(self) -> str:
        return _remote_child(self.task_dir, "input")

    @property
    def partial_blend(self) -> str:
        return _remote_child(self.input_dir, "partial.blend")

    @property
    def input_manifest(self) -> str:
        return _remote_child(self.input_dir, "manifest.json")

    @property
    def scene_reference_dir(self) -> str:
        return _remote_child(self.input_dir, "scene_reference_images")

    @property
    def input_parts_dir(self) -> str:
        return _remote_child(self.input_dir, "parts")

    @property
    def output_submission_dir(self) -> str:
        return _remote_child(self.remote_output_dir, "submission")

    @property
    def output_scene(self) -> str:
        return _remote_child(self.output_submission_dir, "final.blend")

    @property
    def output_objects_dir(self) -> str:
        return _remote_child(self.output_submission_dir, "objects")

    @property
    def reference_manifest(self) -> str:
        return _remote_child(self.reference_dir, "manifest.json")

    @property
    def evaluation_config(self) -> str:
        return _remote_child(self.reference_dir, "evaluation_config.json")

    @property
    def task_description(self) -> str:
        return textwrap.dedent(f"""\
            You are a 3D artist using Blender.

            Your task is to complete one specific static object variant from a partial Blender scene.

            Selected variant:
            - `{self.DISPLAY_NAME}` (`{self.TASK_TAG}`)

            Agent-visible inputs:
            - Partial Blender scene: `{self.partial_blend}` containing the preserved scaffold and unchanged parts for this variant
            - Input manifest: `{self.input_manifest}` with the required exported part/object naming for this variant
            - Whole-object reference images: `{self.scene_reference_dir}` showing the full target object from multiple views
            - Missing-part detail/context references: `{self.input_parts_dir}` showing the missing geometry and placement cues

            Required submission:
            - Save the finished Blender scene to `{self.output_scene}`
            - Export the final object bundle as OBJ parts into `{self.output_objects_dir}`
            - Export one OBJ for every required part/object name listed in the input manifest

            Requirements:
            - Reconstruct the missing geometry for this selected variant only
            - Keep alignment with the staged partial scaffold and reference views
            - Preserve the expected part/object naming from the input manifest
            - Produce a coherent final object with a convincing final rendered look
            """)

    def to_metadata(self) -> dict:
        data = super().to_metadata()
        data.pop("software_dir", None)
        data.update(
            {
                "task_tag": self.TASK_TAG,
                "display_name": self.DISPLAY_NAME,
                "remote_task_dir_name": self.REMOTE_TASK_DIR_NAME,
                "input_dir": self.input_dir,
                "partial_blend": self.partial_blend,
                "input_manifest": self.input_manifest,
                "scene_reference_dir": self.scene_reference_dir,
                "input_parts_dir": self.input_parts_dir,
                "output_submission_dir": self.output_submission_dir,
                "output_scene": self.output_scene,
                "output_objects_dir": self.output_objects_dir,
                "reference_manifest": self.reference_manifest,
                "evaluation_config": self.evaluation_config,
                "multipart_sample_ratio": self.MULTIPART_SAMPLE_RATIO,
                "multipart_min_samples": self.MULTIPART_MIN_SAMPLES,
                "multipart_max_samples": self.MULTIPART_MAX_SAMPLES,
            }
        )
        return data


def _build_config(spec: dict) -> ObjectGenerationTaskConfig:
    return ObjectGenerationTaskConfig(
        VARIANT_NAME=spec["remote_task_dir_name"],
        TASK_TAG=spec["task_tag"],
        DISPLAY_NAME=spec["display_name"],
        REMOTE_TASK_DIR_NAME=spec["remote_task_dir_name"],
        MULTIPART_SAMPLE_RATIO=spec["multipart_sample_ratio"],
        MULTIPART_MIN_SAMPLES=spec["multipart_min_samples"],
        MULTIPART_MAX_SAMPLES=spec["multipart_max_samples"],
    )


@cb.tasks_config(split="train")
def load():
    tasks = []
    for spec in VARIANTS:
        cfg = _build_config(spec)
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


async def _upload_scripts(session: cb.DesktopSession, remote_scripts_dir: str) -> None:
    await session.interface.create_dir(remote_scripts_dir)
    for name in ["remote_hard_eval.py", "blender_render_object_views.py"]:
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
    argv = ", ".join(_ps_literal(v) for v in [script_path, *args])
    ps = (
        "$ErrorActionPreference='Stop'; "
        f"$wd={_ps_literal(remote_scripts_dir)}; "
        f"$py={_ps_literal(REMOTE_PYTHON)}; "
        f"$stdout={_ps_literal(stdout_path)}; "
        f"$stderr={_ps_literal(stderr_path)}; "
        f"$env:BLENDER_BINARY={_ps_literal(REMOTE_BLENDER)}; "
        "Set-Location -LiteralPath $wd; "
        "if (Test-Path -LiteralPath $stdout) { Remove-Item -LiteralPath $stdout -Force -ErrorAction SilentlyContinue }; "
        "if (Test-Path -LiteralPath $stderr) { Remove-Item -LiteralPath $stderr -Force -ErrorAction SilentlyContinue }; "
        f"Start-Process -FilePath $py -ArgumentList @({argv}) -WorkingDirectory $wd "
        "-RedirectStandardOutput $stdout -RedirectStandardError $stderr -WindowStyle Hidden"
    )
    await session.run_command(f'powershell -NoProfile -Command "{ps}"', check=False)


async def _wait_for_file(
    session: cb.DesktopSession, path: str, timeout_sec: float = 3600.0, poll_sec: float = 10.0
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


def _safe_local_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_") or "item"


async def _materialize_frame_pairs(
    session: cb.DesktopSession,
    frame_pairs: list[dict],
    local_tmp_dir: Path,
) -> list[dict]:
    local_pairs: list[dict] = []
    assets_dir = local_tmp_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    for idx, pair in enumerate(frame_pairs):
        label = pair.get("view", f"pair_{idx}")
        ref_remote = pair["reference_image"]
        cand_remote = pair["candidate_image"]
        ref_local = assets_dir / f"{idx:02d}_{_safe_local_name(label)}_ref.png"
        cand_local = assets_dir / f"{idx:02d}_{_safe_local_name(label)}_cand.png"
        ref_local.write_bytes(await session.read_bytes(ref_remote))
        cand_local.write_bytes(await session.read_bytes(cand_remote))
        local_pair = dict(pair)
        local_pair["reference_image"] = str(ref_local)
        local_pair["candidate_image"] = str(cand_local)
        local_pairs.append(local_pair)
    return local_pairs


@cb.evaluate_task(split="train")
async def evaluate(task_cfg, session: cb.DesktopSession) -> list[float]:
    meta = task_cfg.metadata
    if not (await session.file_exists(meta["output_scene"]) or await session.directory_exists(meta["output_scene"])):
        return [0.0]

    remote_eval_dir = _remote_child(REMOTE_TEMP_ROOT, meta["task_tag"])
    remote_scripts_dir = _remote_child(remote_eval_dir, "scripts")
    remote_results_dir = _remote_child(remote_eval_dir, "results")
    remote_report = _remote_child(remote_results_dir, "hard_eval_report.json")
    await session.interface.create_dir(remote_eval_dir)
    await session.interface.create_dir(remote_results_dir)
    await _upload_scripts(session, remote_scripts_dir)
    await _launch_remote_job(
        session,
        remote_scripts_dir=remote_scripts_dir,
        args=[
            "--reference-manifest",
            meta["reference_manifest"],
            "--evaluation-config",
            meta["evaluation_config"],
            "--reference-dir",
            meta["reference_dir"],
            "--candidate-scene",
            meta["output_scene"],
            "--candidate-objects-dir",
            meta["output_objects_dir"],
            "--output-dir",
            remote_results_dir,
            "--sample-ratio",
            str(meta["multipart_sample_ratio"]),
            "--min-samples",
            str(meta["multipart_min_samples"]),
            "--max-samples",
            str(meta["multipart_max_samples"]),
            "--renderer-script",
            _remote_child(remote_scripts_dir, "blender_render_object_views.py"),
        ],
    )
    if not await _wait_for_file(session, remote_report):
        logger.error("[%s] Timed out waiting for %s", meta["task_tag"], remote_report)
        return [0.0]

    report = json.loads((await session.read_bytes(remote_report)).decode("utf-8"))
    hard_score = float(report.get("score", 0.0))
    judge_weight = 0.70
    try:
        eval_cfg = json.loads((await session.read_bytes(meta["evaluation_config"])).decode("utf-8"))
        judge_weight = float(eval_cfg.get("score_weights", {}).get("judge_score", judge_weight))
    except Exception:
        pass

    local_tmp_dir = TASK_DIR / ".tmp_soft_eval" / meta["task_tag"]
    frame_pairs = report.get("frame_pairs", [])
    soft_score = 0.0
    try:
        local_pairs = await _materialize_frame_pairs(session, frame_pairs, local_tmp_dir)
        soft_score = run_local_soft_eval(local_pairs, local_tmp_dir)
    except Exception as exc:
        logger.warning("[%s] local soft eval failed: %s", meta["task_tag"], exc)
        soft_score = 0.0

    final_score = (1.0 - judge_weight) * hard_score + judge_weight * soft_score
    logger.info(
        "[%s] hard=%.4f soft=%.4f final=%.4f", meta["task_tag"], hard_score, soft_score, final_score
    )
    return [float(final_score)]
