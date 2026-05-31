from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import textwrap
from dataclasses import dataclass, field
from pathlib import Path, PureWindowsPath
from typing import Any

import cua_bench as cb
from tasks.common_config import GeneralTaskConfig
from tasks.common_setup import BaseTaskSetup
from tasks.utils.evaluation import llm_multimodal_binary_questions_sync, resolve_llm_judge_model

logger = logging.getLogger(__name__)

TASK_FAMILY_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = TASK_FAMILY_DIR / "scripts"
REMOTE_SCRIPT_FILES = (
    "common.py",
    "evaluate_submission.py",
    "run_evaluation.py",
    "run_judge_evaluation.py",
    "render_parts_dataset.py",
    "render_eval_dataset.py",
)
REMOTE_PYTHON = os.environ.get(
    "ZBRUSH_REMOTE_PYTHON",
    r"C:\Users\User\AppData\Local\Programs\Python\Python312\python.exe",
)
REMOTE_BLENDER = os.environ.get(
    "ZBRUSH_REMOTE_BLENDER",
    r"C:\Program Files\Blender Foundation\Blender 5.0\blender.exe",
)
SOFT_SAMPLE_PARTS = 3
OPENAI_MODEL = resolve_llm_judge_model(
    env_var="ZBRUSH_SOFT_EVAL_MODEL",
    default="gpt-4.1-mini",
)


@dataclass(frozen=True)
class ProjectSpec:
    task_tag: str
    display_name: str
    input_project_file: str


VARIANTS = [
    ProjectSpec(
        task_tag="japanese_samurai_full_body",
        display_name="Japanese Samurai Full Body",
        input_project_file="Samurai-Armor-partial.zpr",
    ),
    ProjectSpec(
        task_tag="japanese_samurai_with_weapons",
        display_name="Japanese Samurai With Weapons",
        input_project_file="Samural-weapons-partial.zpr",
    ),
]


ARCHIVED_BUCKET_ONLY_VARIANTS = [
    ProjectSpec(
        task_tag="courtyard_scene",
        display_name="Courtyard Scene",
        input_project_file="courtyard_scene_partial.zpr",
    ),
]


SPEC_BY_TAG = {spec.task_tag: spec for spec in VARIANTS}


def _remote_child(base: str, *parts: str) -> str:
    path = PureWindowsPath(base)
    for part in parts:
        path = path / part
    return str(path)


def _ps_quote(text: str) -> str:
    return text.replace("'", "''")


def _ps_literal(text: str) -> str:
    return f"'{_ps_quote(text)}'"


def _resolve_remote_path(base_dir: str, path_value: str) -> str:
    path = PureWindowsPath(path_value)
    return str(path if path.is_absolute() else PureWindowsPath(base_dir) / path)


def _normalize_weights(raw: dict[str, Any] | None) -> dict[str, float]:
    default = {
        "completeness_score": 0.20,
        "geometry_score": 0.25,
        "render_score": 0.20,
        "mesh_health_score": 0.10,
        "judge_score": 0.25,
    }
    weights = dict(default)
    if raw:
        for key in weights:
            if key in raw:
                weights[key] = float(raw[key])
    total = sum(weights.values())
    return {key: value / total for key, value in weights.items()} if total > 0 else default


def _load_manifest_entries(data: dict[str, Any] | list[Any]) -> list[dict[str, Any]]:
    if isinstance(data, dict) and "parts" in data:
        return data["parts"]
    if isinstance(data, list):
        return data
    raise RuntimeError("Unexpected manifest format")


def _hard_score_from_report(report: dict[str, Any]) -> float:
    summary = report.get("summary", {})
    weights = _normalize_weights(report.get("weights"))
    hard_keys = [
        "completeness_score",
        "geometry_score",
        "render_score",
        "mesh_health_score",
    ]
    hard_weight = sum(weights[key] for key in hard_keys)
    if hard_weight <= 1e-12:
        return 0.0
    value = sum(weights[key] * float(summary.get(f"mean_{key}", 0.0)) for key in hard_keys)
    return float(value / hard_weight)


async def _upload_remote_eval_bundle(session: cb.DesktopSession, remote_scripts_dir: str) -> None:
    await session.interface.create_dir(remote_scripts_dir)
    for name in REMOTE_SCRIPT_FILES:
        local_path = SCRIPTS_DIR / name
        await session.write_file(_remote_child(remote_scripts_dir, name), local_path.read_text(encoding="utf-8"))


async def _launch_remote_python_job(session: cb.DesktopSession, *, remote_scripts_dir: str, script_name: str, args: list[str]) -> None:
    script_path = _remote_child(remote_scripts_dir, script_name)
    stdout_path = _remote_child(remote_scripts_dir, "job_stdout.txt")
    stderr_path = _remote_child(remote_scripts_dir, "job_stderr.txt")
    argv = ", ".join(_ps_literal(value) for value in [script_path, *args])
    ps = (
        "$ErrorActionPreference='Stop'; "
        f"$wd={_ps_literal(remote_scripts_dir)}; "
        f"$py={_ps_literal(REMOTE_PYTHON)}; "
        f"$stdout={_ps_literal(stdout_path)}; "
        f"$stderr={_ps_literal(stderr_path)}; "
        f"$env:BLENDER_BINARY={_ps_literal(REMOTE_BLENDER)}; "
        f"$env:ZBRUSH_HELPER_PYTHON={_ps_literal(REMOTE_PYTHON)}; "
        "Set-Location -LiteralPath $wd; "
        "if (Test-Path -LiteralPath $stdout) { Remove-Item -LiteralPath $stdout -Force -ErrorAction SilentlyContinue }; "
        "if (Test-Path -LiteralPath $stderr) { Remove-Item -LiteralPath $stderr -Force -ErrorAction SilentlyContinue }; "
        f"Start-Process -FilePath $py -ArgumentList @({argv}) -WorkingDirectory $wd "
        "-RedirectStandardOutput $stdout -RedirectStandardError $stderr -WindowStyle Hidden"
    )
    await session.run_command(f'powershell -NoProfile -Command "{ps}"', check=False)


async def _wait_for_remote_file(
    session: cb.DesktopSession,
    *,
    path: str,
    poll_interval_sec: float = 20.0,
    timeout_sec: float = 7200.0,
) -> bool:
    deadline = asyncio.get_event_loop().time() + timeout_sec
    while asyncio.get_event_loop().time() < deadline:
        try:
            if (await session.file_exists(path) or await session.directory_exists(path)):
                return True
        except Exception:
            pass
        await asyncio.sleep(poll_interval_sec)
    return False


async def _sample_reference_views(
    session: cb.DesktopSession,
    reference_manifest_path: str,
    candidate_manifest: dict[str, Any],
) -> list[dict[str, Any]]:
    manifest = _load_manifest_entries(json.loads((await session.read_bytes(reference_manifest_path)).decode("utf-8")))
    candidate_parts = candidate_manifest.get("parts", [])
    if not manifest or not candidate_parts:
        return []
    manifest_by_name = {entry["name"]: entry for entry in manifest}
    package_root = str(PureWindowsPath(reference_manifest_path).parent.parent)
    samples: list[dict[str, Any]] = []
    for candidate_entry in candidate_parts[:SOFT_SAMPLE_PARTS]:
        entry = manifest_by_name.get(candidate_entry["name"])
        if not entry:
            continue
        meta_path = _resolve_remote_path(package_root, entry["meta"])
        meta = json.loads((await session.read_bytes(meta_path)).decode("utf-8"))
        views = meta.get("angles") or meta.get("views") or []
        if not views:
            continue
        candidate_angles = candidate_entry.get("angles") or candidate_entry.get("views") or []
        if not candidate_angles:
            continue
        candidate_azimuth = int(candidate_angles[0].get("azimuth_deg", 0))
        view = next(
            (item for item in views if int(item.get("azimuth_deg", 0)) == candidate_azimuth),
            None,
        )
        if view is None:
            continue
        detail_path = view.get("reference_detail_image") or view.get("detail_image")
        context_path = view.get("reference_context_image") or view.get("context_image")
        if not detail_path or not context_path:
            continue
        samples.append(
            {
                "name": entry["name"],
                "azimuth_deg": candidate_azimuth,
                "reference_detail_remote": _resolve_remote_path(package_root, detail_path),
                "reference_context_remote": _resolve_remote_path(package_root, context_path),
            }
        )
    return samples


async def _collect_remote_image_pairs(
    session: cb.DesktopSession,
    candidate_manifest: dict[str, Any],
    sampled_parts: list[dict[str, Any]],
    local_tmp_dir: Path,
) -> list[dict[str, Any]]:
    candidate_parts = {part["name"]: part for part in candidate_manifest.get("parts", [])}
    pairs: list[dict[str, Any]] = []
    local_tmp_dir.mkdir(parents=True, exist_ok=True)
    for sample in sampled_parts:
        part = candidate_parts.get(sample["name"])
        if not part:
            continue
        angles = part.get("angles") or part.get("views") or []
        if not angles:
            continue
        angle = angles[0]
        candidate_detail_remote = angle.get("detail_image") or angle.get("reference_detail_image")
        candidate_context_remote = angle.get("context_image") or angle.get("reference_context_image")
        if not candidate_detail_remote or not candidate_context_remote:
            continue

        local_reference_detail = local_tmp_dir / f"{sample['name']}__reference_detail.png"
        local_reference_context = local_tmp_dir / f"{sample['name']}__reference_context.png"
        local_candidate_detail = local_tmp_dir / f"{sample['name']}__candidate_detail.png"
        local_candidate_context = local_tmp_dir / f"{sample['name']}__candidate_context.png"

        local_reference_detail.write_bytes(await session.read_bytes(sample["reference_detail_remote"]))
        local_reference_context.write_bytes(await session.read_bytes(sample["reference_context_remote"]))
        local_candidate_detail.write_bytes(await session.read_bytes(candidate_detail_remote))
        local_candidate_context.write_bytes(await session.read_bytes(candidate_context_remote))
        pairs.append(
            {
                "name": sample["name"],
                "reference_detail": str(local_reference_detail),
                "reference_context": str(local_reference_context),
                "candidate_detail": str(local_candidate_detail),
                "candidate_context": str(local_candidate_context),
            }
        )
    return pairs


def _image_to_data_url(path: str) -> str:
    raw = Path(path).read_bytes()
    return f"data:image/png;base64,{base64.b64encode(raw).decode('utf-8')}"


def _local_soft_score(image_pairs: list[dict[str, Any]], *, report_path: Path | None = None) -> float:
    if not image_pairs:
        return 0.0
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        logger.warning("OPENAI_API_KEY missing; soft score falls back to 0.0")
        return 0.0
    content: list[dict[str, Any]] = []
    for pair in image_pairs:
        content.append({"type": "text", "text": f"Part: {pair['name']}"})
        for label, key in [
            ("reference detail", "reference_detail"),
            ("candidate detail", "candidate_detail"),
            ("reference context", "reference_context"),
            ("candidate context", "candidate_context"),
        ]:
            content.append({"type": "text", "text": label})
            content.append({"type": "image_url", "image_url": {"url": _image_to_data_url(pair[key])}})
    prompt_context = textwrap.dedent(
        """
        You are evaluating a 3D reconstruction task from sampled render pairs.
        For each part, you will see four images in order:
        1. reference detail
        2. candidate detail
        3. reference context
        4. candidate context

        Judge each question independently using only YES or NO.
        """
    ).strip()
    questions = [
        "Does the candidate match the reference shape well enough to pass?",
        "Does the candidate preserve visible detail well enough to pass?",
        "Does the candidate preserve the structure and placement relationship well enough to pass?",
    ]
    parsed = llm_multimodal_binary_questions_sync(
        prompt_context=prompt_context,
        questions=questions,
        content=content,
        model=OPENAI_MODEL,
        max_tokens=32,
        temperature=0,
        api_key=api_key,
    )
    if report_path is not None:
        report_path.write_text(json.dumps(parsed, ensure_ascii=False, indent=2), encoding="utf-8")
    score = float(parsed.get("final_score", 0.0))
    return max(0.0, min(1.0, score))


@dataclass
class ZBrushTaskConfig(GeneralTaskConfig):
    spec: ProjectSpec = field(default=None)
    DOMAIN_NAME: str = field(init=False, default="game")

    TASK_NAME: str = field(init=False, default="zbrush_3d_modeling")
    VARIANT_NAME: str = field(init=False, default="")

    def __post_init__(self):
        if self.spec is not None:
            object.__setattr__(self, "VARIANT_NAME", self.spec.task_tag)

    @property
    def input_dir(self) -> str:
        return _remote_child(self.task_dir, "input")

    @property
    def input_project_path(self) -> str:
        return _remote_child(self.input_dir, self.spec.input_project_file)

    @property
    def output_project_path(self) -> str:
        return _remote_child(self.output_dir, self.spec.input_project_file)

    @property
    def candidate_objects_dir(self) -> str:
        return _remote_child(self.output_dir, "output-test", "objects")

    @property
    def reference_manifest_path(self) -> str:
        return _remote_child(self.reference_dir, "manifest.json")

    @property
    def reference_objects_dir(self) -> str:
        return _remote_child(self.reference_dir, "objects")

    @property
    def evaluation_config_path(self) -> str:
        return _remote_child(self.reference_dir, "evaluation_config.json")

    @property
    def task_description(self) -> str:
        description = f"""\
You are a 3D artist working in ZBrush on a constrained partial-completion task.

## Task
Reconstruct the missing parts of **{self.spec.display_name}**.

Canonical variant root:
- `{self.task_dir}`

Benchmark inputs:
- part reference images: `{self.input_dir}`
- partial ZBrush project: `{self.input_project_path}`

## Inputs
- Per-part reference images in `{self.input_dir}`
- Partial ZBrush project in:
  `{self.input_project_path}`
"""
        description += f"""## What You Must Do
1. Open the partial ZBrush project at `{self.input_project_path}`
2. Use the reference images in `{self.input_dir}` to reconstruct or refine the missing parts
3. Keep the model in the same global coordinate system as the provided partial project
4. Export the full object set as individual OBJ files to:
   `{self.candidate_objects_dir}`
5. Use the exact part names for OBJ filenames
6. Save your working ZBrush project to `{self.output_project_path}`
"""
        return description

    def to_metadata(self) -> dict[str, Any]:
        metadata = super().to_metadata()
        metadata.update(
            {
                "variant_name": self.spec.task_tag,
                "project_name": self.spec.display_name,
                "input_project_file": self.spec.input_project_file,
                "input_dir": self.input_dir,
                "input_project_path": self.input_project_path,
                "output_project_path": self.output_project_path,
                "candidate_objects_dir": self.candidate_objects_dir,
                "reference_manifest_path": self.reference_manifest_path,
                "reference_objects_dir": self.reference_objects_dir,
                "evaluation_config_path": self.evaluation_config_path,
            }
        )
        return metadata


def _build_config(spec: ProjectSpec) -> ZBrushTaskConfig:
    return ZBrushTaskConfig(spec=spec)


@cb.tasks_config(split="train")
def load():
    tasks = []
    for spec in VARIANTS:
        config = _build_config(spec)
        tasks.append(
            cb.Task(
                description=config.task_description,
                metadata=config.to_metadata(),
                computer={"provider": "computer", "setup_config": {"os_type": "windows"}},
            )
        )
    return tasks


_setup = BaseTaskSetup()


@cb.setup_task(split="train")
async def start(task_cfg, session: cb.DesktopSession):
    await _setup(task_cfg, session)


@cb.evaluate_task(split="train")
async def evaluate(task_cfg, session: cb.DesktopSession) -> list[float]:
    meta = task_cfg.metadata
    task_tag = meta["variant_name"]
    remote_job_root = _remote_child(r"C:\Users\User\AppData\Local\Temp", "agenthle_eval", "zbrush", task_tag)
    remote_scripts_dir = _remote_child(remote_job_root, "scripts")
    remote_eval_output_dir = _remote_child(remote_job_root, "results")

    await session.interface.create_dir(remote_eval_output_dir)
    await _upload_remote_eval_bundle(session, remote_scripts_dir)
    await _launch_remote_python_job(
        session,
        remote_scripts_dir=remote_scripts_dir,
        script_name="evaluate_submission.py",
        args=[
            "--mode",
            "full",
            "--reference-manifest",
            meta["reference_manifest_path"],
            "--reference-objects-dir",
            meta["reference_objects_dir"],
            "--candidate-objects-dir",
            meta["candidate_objects_dir"],
            "--output-dir",
            remote_eval_output_dir,
            "--evaluation-config",
            meta["evaluation_config_path"],
            "--skip-judge",
        ],
    )

    report_json = _remote_child(remote_eval_output_dir, "evaluation_report.json")
    if not await _wait_for_remote_file(session, path=report_json):
        logger.error("[%s] Remote evaluator timed out waiting for %s", task_tag, report_json)
        return [0.0]

    try:
        report = json.loads((await session.read_bytes(report_json)).decode("utf-8"))
    except Exception as exc:
        logger.error("[%s] Could not read remote evaluation report: %s", task_tag, exc)
        return [0.0]

    hard_score = _hard_score_from_report(report)
    soft_score = 0.0
    try:
        candidate_manifest_path = _remote_child(remote_eval_output_dir, "candidate_renders", "candidate_manifest.json")
        if (await session.file_exists(candidate_manifest_path) or await session.directory_exists(candidate_manifest_path)):
            candidate_manifest = json.loads((await session.read_bytes(candidate_manifest_path)).decode("utf-8"))
            sampled_parts = await _sample_reference_views(
                session,
                meta["reference_manifest_path"],
                candidate_manifest,
            )
            local_tmp_dir = TASK_FAMILY_DIR / ".tmp_soft_eval" / task_tag
            image_pairs = await _collect_remote_image_pairs(session, candidate_manifest, sampled_parts, local_tmp_dir)
            soft_score = _local_soft_score(image_pairs, report_path=local_tmp_dir / "soft_eval_report.json")
    except Exception as exc:
        logger.warning("[%s] Local soft eval failed: %s", task_tag, exc)
        soft_score = 0.0

    weights = _normalize_weights(report.get("weights"))
    final_score = (1.0 - weights["judge_score"]) * hard_score + weights["judge_score"] * soft_score
    logger.info("[%s] hard=%.4f soft=%.4f final=%.4f", task_tag, hard_score, soft_score, final_score)
    return [float(final_score)]
