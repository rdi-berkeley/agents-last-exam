import asyncio
import json
import logging
import os
import subprocess
import textwrap
from dataclasses import dataclass, field
from pathlib import Path, PureWindowsPath
from typing import Any

import cua_bench as cb

from tasks.common_config import GeneralTaskConfig
from tasks.common_setup import BaseTaskSetup
from tasks.utils.evaluation import (llm_vision_binary_questions_sync,
                              resolve_llm_judge_model)

_setup = BaseTaskSetup()

logger = logging.getLogger(__name__)

TASK_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = TASK_DIR / "scripts"
CATALOG_PATH = TASK_DIR / "variant_catalog.json"
TASK_WORKFLOW = "high_to_low_modeling"
REMOTE_PYTHON = os.environ.get(
    "BLENDER_TASK_REMOTE_PYTHON",
    r"C:\Users\User\AppData\Local\Programs\Python\Python312\python.exe",
)
REMOTE_BLENDER = os.environ.get(
    "BLENDER_TASK_REMOTE_BLENDER", r"C:\Program Files\Blender Foundation\Blender 5.0\blender.exe"
)
REMOTE_TEMP_ROOT = os.environ.get(
    "BLENDER_TASK_REMOTE_EVAL_ROOT", r"C:\Users\User\AppData\Local\Temp\agenthle_eval\high_to_low"
)
SOFT_EVAL_MODEL = resolve_llm_judge_model(
    env_var="BLENDER_TASK_SOFT_EVAL_MODEL",
    default="gpt-4.1-mini",
)


def _ps_quote(text: str) -> str:
    return text.replace("'", "''")


def _ps_literal(text: str) -> str:
    return f"'{_ps_quote(text)}'"


def _remote_child(base: str, *parts: str) -> str:
    path = PureWindowsPath(base)
    for part in parts:
        path = path / part
    return str(path)


@dataclass(frozen=True)
class VariantSpec:
    task_tag: str
    display_name: str
    source_package: str
    source_object_name: str
    input_obj_name: str
    submission_obj_name: str
    local_package_root: str | None = None
    source_face_count: int | None = None
    software_blend_name: str | None = None


def _load_variants() -> list[VariantSpec]:
    payload = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    return [VariantSpec(**entry) for entry in payload]


VARIANTS = _load_variants()


@dataclass
class HighToLowTaskConfig(GeneralTaskConfig):
    REMOTE_ROOT_DIR: str = os.environ.get("REMOTE_ROOT_DIR", r"E:\agenthle")
    variant: VariantSpec = field(default_factory=lambda: VARIANTS[0])
    DOMAIN_NAME: str = "visual_media"

    TASK_NAME: str = "high_to_low_modeling"
    VARIANT_NAME: str = ""

    def __post_init__(self) -> None:
        self.VARIANT_NAME = self.variant.task_tag

    @property
    def input_dir(self) -> str:
        return rf"{self.task_dir}\input"

    @property
    def software_blend(self) -> str | None:
        if not self.variant.software_blend_name:
            return None
        return rf"{self.software_dir}\{self.variant.software_blend_name}"

    @property
    def input_obj(self) -> str:
        return rf"{self.input_dir}\{self.variant.input_obj_name}"

    @property
    def evaluation_config(self) -> str:
        return rf"{self.reference_dir}\evaluation_config.json"

    @property
    def reference_manifest(self) -> str:
        return rf"{self.reference_dir}\manifest.json"

    @property
    def output_submission_dir(self) -> str:
        return rf"{self.remote_output_dir}\submission"

    @property
    def output_obj(self) -> str:
        return rf"{self.output_submission_dir}\{self.variant.submission_obj_name}"

    @property
    def task_description(self) -> str:
        asset_lines = [
            f"- Display name: `{self.variant.display_name}`",
            f"- Source package: `{self.variant.source_package}`",
            f"- Source object: `{self.variant.source_object_name}`",
        ]
        if self.variant.source_face_count:
            asset_lines.append(f"- Source triangle count: `{self.variant.source_face_count}`")
        context_block = ""
        if self.software_blend:
            context_block = f"\nOptional provenance/context:\n- Source high-poly Blender file: `{self.software_blend}`\n"
        return textwrap.dedent(f"""\
            You are a 3D artist using Blender.

            Your task is to convert a high-poly source object into a clean low-poly model.

            Asset:
            {chr(10).join(asset_lines)}

            Official input:
            - High-poly OBJ: `{self.input_obj}`
            {context_block}\

            Required submission:
            - Export a low-poly OBJ named `{self.variant.submission_obj_name}`
            - Save it to `{self.output_obj}`
            - Keep the model aligned to the original world space

            Evaluation:
            - Hard eval runs remotely with Blender and geometry/image metrics
            - Soft eval runs locally on a small set of evidence sheets
            - Final score combines hard and soft scores
            """)

    def to_metadata(self) -> dict[str, Any]:
        data = super().to_metadata()
        if not self.software_blend:
            data.pop("software_dir", None)
        data.update(
            {
                "variant_name": self.VARIANT_NAME,
                "input_dir": self.input_dir,
                "input_obj": self.input_obj,
                "evaluation_config": self.evaluation_config,
                "reference_manifest": self.reference_manifest,
                "output_submission_dir": self.output_submission_dir,
                "output_obj": self.output_obj,
                "source_package": self.variant.source_package,
                "source_object_name": self.variant.source_object_name,
                "source_face_count": self.variant.source_face_count,
                "local_package_root": self.variant.local_package_root,
            }
        )
        if self.software_blend:
            data["software_blend"] = self.software_blend
        return data


@cb.tasks_config(split="train")
def load():
    tasks = []
    for variant in VARIANTS:
        cfg = HighToLowTaskConfig(variant=variant)
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
    await session.makedirs(remote_scripts_dir)
    for name in ["run_lowpoly_evaluation.py", "render_parts_dataset.py"]:
        await session.write_file(
            _remote_child(remote_scripts_dir, name),
            (SCRIPTS_DIR / name).read_text(encoding="utf-8"),
        )


async def _launch_remote_job(
    session: cb.DesktopSession, *, remote_scripts_dir: str, args: list[str]
) -> None:
    script_path = _remote_child(remote_scripts_dir, "run_lowpoly_evaluation.py")
    stdout_path = _remote_child(remote_scripts_dir, "job_stdout.txt")
    stderr_path = _remote_child(remote_scripts_dir, "job_stderr.txt")
    argv = subprocess.list2cmdline([script_path, *args])
    ps = (
        "$ErrorActionPreference='Stop'; "
        f"$wd={_ps_literal(remote_scripts_dir)}; "
        f"$py={_ps_literal(REMOTE_PYTHON)}; "
        f"$argv={_ps_literal(argv)}; "
        f"$stdout={_ps_literal(stdout_path)}; "
        f"$stderr={_ps_literal(stderr_path)}; "
        f"$env:BLENDER_BINARY={_ps_literal(REMOTE_BLENDER)}; "
        "Set-Location -LiteralPath $wd; "
        "if (Test-Path -LiteralPath $stdout) { Remove-Item -LiteralPath $stdout -Force -ErrorAction SilentlyContinue }; "
        "if (Test-Path -LiteralPath $stderr) { Remove-Item -LiteralPath $stderr -Force -ErrorAction SilentlyContinue }; "
        "Start-Process -FilePath $py -ArgumentList $argv -WorkingDirectory $wd "
        "-RedirectStandardOutput $stdout -RedirectStandardError $stderr -WindowStyle Hidden"
    )
    await session.run_command(f'powershell -NoProfile -Command "{ps}"', check=False)


async def _wait_for_file(
    session: cb.DesktopSession, path: str, timeout_sec: float = 1800.0, poll_sec: float = 10.0
) -> bool:
    deadline = asyncio.get_event_loop().time() + timeout_sec
    while asyncio.get_event_loop().time() < deadline:
        try:
            if await session.exists(path):
                return True
        except Exception:
            pass
        await asyncio.sleep(poll_sec)
    return False


def _run_local_soft_eval(
    *,
    overlay_sheet: Path,
    heatmap_sheet: Path,
    silhouette_sheet: Path,
    metrics: dict[str, Any],
    report_path: Path | None = None,
) -> float | None:
    if not os.environ.get("OPENAI_API_KEY"):
        return None
    summary = json.dumps(
        {
            "triangle_ratio": metrics.get("triangle_ratio"),
            "distance_score": metrics.get("distance_score"),
            "silhouette_score": metrics.get("silhouette_score"),
            "alignment_score": metrics.get("alignment_score"),
            "mesh_health_score": metrics.get("mesh_health_score"),
            "geometry_score": metrics.get("geometry_score"),
        },
        ensure_ascii=False,
        indent=2,
    )
    prompt_context = (
        "You are an expert 3D artist evaluating whether a lowpoly model preserves the highpoly shape and silhouette "
        "while using significantly fewer polygons. "
        "You will receive exactly three evidence images in this order: "
        "(1) overlay sheet: gray highpoly base render with green lowpoly wire overlay, "
        "(2) error heatmap sheet: projected distance errors on the lowpoly surface, "
        "(3) silhouette comparison sheet: high silhouette, low silhouette, and their diff for multiple views. "
        f"Key metrics:\n{summary}\n\n"
        "Judge each question independently and only use the provided evidence and metrics."
    )
    questions = [
        "Does the candidate preserve the overall highpoly shape well enough to pass?",
        "Does the candidate preserve the main silhouettes across the sampled views well enough to pass?",
        "Does the candidate achieve a meaningful lowpoly reduction rather than effectively submitting the highpoly again?",
        "Are obvious visual artifacts or shape collapses absent enough for the result to pass?",
    ]
    data = llm_vision_binary_questions_sync(
        prompt_context=prompt_context,
        questions=questions,
        image_bytes_list=[
            overlay_sheet.read_bytes(),
            heatmap_sheet.read_bytes(),
            silhouette_sheet.read_bytes(),
        ],
        model=SOFT_EVAL_MODEL,
        temperature=0,
    )
    if report_path is not None:
        report_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return max(0.0, min(1.0, float(data["final_score"])))


@cb.evaluate_task(split="train")
async def evaluate(task_cfg, session: cb.DesktopSession) -> list[float]:
    meta = task_cfg.metadata
    if not await session.exists(meta["output_obj"]):
        return [0.0]

    task_tag = meta["variant_name"]
    remote_eval_dir = _remote_child(REMOTE_TEMP_ROOT, task_tag)
    remote_scripts_dir = _remote_child(remote_eval_dir, "scripts")
    remote_results_dir = _remote_child(remote_eval_dir, "results")
    remote_report = _remote_child(remote_results_dir, "final_report.json")
    await session.makedirs(remote_eval_dir)
    if await session.exists(remote_results_dir):
        await session.remove_file(remote_results_dir)
    await session.makedirs(remote_results_dir)
    await _upload_scripts(session, remote_scripts_dir)
    await _launch_remote_job(
        session,
        remote_scripts_dir=remote_scripts_dir,
        args=[
            "--high-obj",
            meta["input_obj"],
            "--low-obj",
            meta["output_obj"],
            "--output-dir",
            remote_results_dir,
            "--evaluation-config",
            meta["evaluation_config"],
            "--skip-judge",
        ],
    )
    if not await _wait_for_file(session, remote_report):
        return [0.0]

    report = json.loads(await session.read_file(remote_report))
    hard_score = float(report.get("geometry_score", 0.0))
    judge_weight = 0.25
    try:
        cfg = json.loads(await session.read_file(meta["evaluation_config"]))
        judge_weight = float(cfg.get("score_weights", {}).get("judge_score", judge_weight))
    except Exception:
        pass

    local_tmp = TASK_DIR / ".tmp_soft_eval" / task_tag
    local_tmp.mkdir(parents=True, exist_ok=True)
    evidence = report.get("evidence_paths", {})
    soft_score = hard_score
    try:
        overlay = local_tmp / "overlay_sheet.png"
        heatmap = local_tmp / "heatmap_sheet.png"
        silhouette = local_tmp / "silhouette_sheet.png"
        overlay.write_bytes(await session.read_bytes(evidence["overlay_sheet"]))
        heatmap.write_bytes(await session.read_bytes(evidence["heatmap_sheet"]))
        silhouette.write_bytes(await session.read_bytes(evidence["silhouette_sheet"]))
        maybe_soft_score = _run_local_soft_eval(
            overlay_sheet=overlay,
            heatmap_sheet=heatmap,
            silhouette_sheet=silhouette,
            metrics=report.get("metrics", {}),
            report_path=local_tmp / "soft_eval_report.json",
        )
        if maybe_soft_score is not None:
            soft_score = maybe_soft_score
    except Exception:
        logger.warning("Local soft eval failed; falling back to hard-only", exc_info=True)
        soft_score = hard_score

    final_score = (1.0 - judge_weight) * hard_score + judge_weight * soft_score
    logger.info(
        "high_to_low eval hard=%.4f soft=%.4f final=%.4f", hard_score, soft_score, final_score
    )
    return [float(final_score)]
