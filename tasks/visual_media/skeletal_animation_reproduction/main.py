from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import textwrap
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from tempfile import TemporaryDirectory
from typing import Any

import cua_bench as cb

from tasks.common_config import GeneralTaskConfig
from tasks.common_setup import BaseTaskSetup
from tasks.visual_media.skeletal_animation_reproduction._shared_eval import (
    compute_final_score,
    compute_local_metrics,
    evenly_spaced_positions,
    extract_reference_frames,
    reference_framing_targets,
    write_eval_bundle,
)
from tasks.visual_media.skeletal_animation_reproduction.scripts.local_soft_eval import (
    run_local_soft_eval,
)

logger = logging.getLogger(__name__)

TASK_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = TASK_DIR / "scripts"
TASK_WORKFLOW = "skeletal_animation_reproduction"

REMOTE_PYTHON = os.environ.get(
    "BLENDER_TASK_REMOTE_PYTHON",
    r"C:\Users\User\AppData\Local\Programs\Python\Python312\python.exe",
)
REMOTE_BLENDER = os.environ.get(
    "BLENDER_TASK_REMOTE_BLENDER",
    r"C:\Program Files\Blender Foundation\Blender 5.0\blender.exe",
)
REMOTE_TEMP_ROOT = os.environ.get(
    "BLENDER_TASK_REMOTE_EVAL_ROOT",
    r"C:\Users\User\AppData\Local\Temp\agenthle_eval\skeletal_animation_reproduction",
)


@dataclass(frozen=True)
class VariantSpec:
    task_tag: str
    display_name: str
    remote_task_dir_name: str
    input_obj_name: str
    input_mtl_name: str
    hidden_reference_fbx_name: str
    motion_scope: str
    required_bone_names: tuple[str, ...] = ()


VARIANTS = [
    VariantSpec(
        task_tag="singing_anime_character",
        display_name="Singing Anime Character",
        remote_task_dir_name="skeletal_animation_reproduction_singing_anime_character",
        input_obj_name="Singing.obj",
        input_mtl_name="Singing.mtl",
        hidden_reference_fbx_name="Singing.fbx",
        motion_scope="body_motion",
        required_bone_names=(
            "pelvis",
            "spine",
            "chest",
            "neck",
            "head",
            "clavicle.L",
            "upper_arm.L",
            "lower_arm.L",
            "hand.L",
            "clavicle.R",
            "upper_arm.R",
            "lower_arm.R",
            "hand.R",
            "thigh.L",
            "shin.L",
            "foot.L",
            "thigh.R",
            "shin.R",
            "foot.R",
        ),
    ),
    VariantSpec(
        task_tag="spinosaurus",
        display_name="Spinosaurus Skeletal Animation Reproduction",
        remote_task_dir_name="skeletal_animation_reproduction_spinosaurus",
        input_obj_name="spinosaurus.obj",
        input_mtl_name="spinosaurus.mtl",
        hidden_reference_fbx_name="spinosaurus.fbx",
        motion_scope="full_body_motion",
        required_bone_names=(
            "hips",
            "chest",
            "spine_base",
            "neck",
            "head",
            "shoulder.L",
            "shoulder.R",
            "front_leg.L",
            "front_leg.R",
            "front_foot.L",
            "front_foot.R",
            "hind_leg.L",
            "hind_leg.R",
            "hind_shin.L",
            "hind_shin.R",
            "hind_foot.L",
            "hind_foot.R",
            "tail_base",
        ),
    ),
    VariantSpec(
        task_tag="skeletal_animation_reproduction_white_cyborg_idle",
        display_name="White Cyborg Idle Skeletal Animation Reproduction",
        remote_task_dir_name="skeletal_animation_reproduction_white_cyborg_idle",
        input_obj_name="white_cyborg.obj",
        input_mtl_name="white_cyborg.mtl",
        hidden_reference_fbx_name="white_cyborg.fbx",
        motion_scope="body_motion",
    ),
    VariantSpec(
        task_tag="skeletal_animation_reproduction_white_cyborg_walk",
        display_name="White Cyborg Walk Skeletal Animation Reproduction",
        remote_task_dir_name="skeletal_animation_reproduction_white_cyborg_walk",
        input_obj_name="white_cyborg.obj",
        input_mtl_name="white_cyborg.mtl",
        hidden_reference_fbx_name="white_cyborg.fbx",
        motion_scope="full_body_motion",
    ),
    VariantSpec(
        task_tag="skeletal_animation_reproduction_white_cyborg_run",
        display_name="White Cyborg Run Skeletal Animation Reproduction",
        remote_task_dir_name="skeletal_animation_reproduction_white_cyborg_run",
        input_obj_name="white_cyborg.obj",
        input_mtl_name="white_cyborg.mtl",
        hidden_reference_fbx_name="white_cyborg.fbx",
        motion_scope="full_body_motion",
    ),
]
SPEC_BY_TAG = {spec.task_tag: spec for spec in VARIANTS}


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
class SkeletalAnimationTaskConfig(GeneralTaskConfig):
    DOMAIN_NAME: str = "visual_media"

    TASK_NAME: str = "skeletal_animation_reproduction"
    VARIANT_NAME: str = ""
    TASK_TAG: str = ""
    DISPLAY_NAME: str = ""
    REMOTE_TASK_DIR_NAME: str = ""
    INPUT_OBJ_NAME: str = ""
    INPUT_MTL_NAME: str = ""
    HIDDEN_REFERENCE_FBX_NAME: str = ""
    MOTION_SCOPE: str = "body_motion"
    REQUIRED_BONE_NAMES: tuple[str, ...] = ()

    @property
    def input_dir(self) -> str:
        return rf"{self.task_dir}\input"

    @property
    def task_dir(self) -> str:
        return _remote_child(self.data_root, self.DOMAIN_NAME, self.REMOTE_TASK_DIR_NAME)

    @property
    def input_obj(self) -> str:
        return rf"{self.input_dir}\{self.INPUT_OBJ_NAME}"

    @property
    def input_mtl(self) -> str:
        return rf"{self.input_dir}\{self.INPUT_MTL_NAME}"

    @property
    def reference_video(self) -> str:
        return rf"{self.input_dir}\reference.mp4"

    @property
    def clean_reference_video(self) -> str:
        return rf"{self.reference_dir}\reference_clean.mp4"

    @property
    def output_submission_dir(self) -> str:
        return rf"{self.output_dir}\submission"

    @property
    def output_blend(self) -> str:
        return rf"{self.output_submission_dir}\final.blend"

    @property
    def output_preview_video(self) -> str:
        return rf"{self.output_submission_dir}\preview.mp4"

    @property
    def hidden_reference_fbx(self) -> str:
        return rf"{self.reference_dir}\{self.HIDDEN_REFERENCE_FBX_NAME}"

    @property
    def skeleton_package(self) -> str:
        return rf"{self.reference_dir}\package.json"

    @property
    def evaluation_config(self) -> str:
        return rf"{self.reference_dir}\evaluation_config.json"

    @property
    def manifest(self) -> str:
        return rf"{self.reference_dir}\manifest.json"

    @property
    def task_description(self) -> str:
        bone_section = ""
        if self.REQUIRED_BONE_NAMES:
            bone_list = ", ".join(self.REQUIRED_BONE_NAMES)
            bone_section = (
                f"\n"
                f"Required bone naming:\n"
                f"Your armature MUST use the following bone names exactly (case-sensitive):\n"
                f"  {bone_list}\n"
                f"You may add extra bones, but all {len(self.REQUIRED_BONE_NAMES)} listed above must be present with these exact names.\n"
            )
        return (
            textwrap.dedent(f"""\
            You are a 3D artist using Blender.

            Your task is to rig and animate the provided character so that it reproduces the body motion shown in the reference video.

            Official input:
            - Unrigged mesh: `{self.input_obj}`
            - Material sidecar: `{self.input_mtl}`
            - Reference animation video: `{self.reference_video}`

            Required submission:
            - Save a Blender scene named `final.blend` to `{self.output_blend}`
            - Render and save a matching preview video named `preview.mp4` to `{self.output_preview_video}`
            """)
            + bone_section
            + textwrap.dedent(f"""\
            Notes:
            - Focus on {self.MOTION_SCOPE.replace('_', ' ')}.
            - Evaluation will compare your `preview.mp4` against a hidden clean reference and also replay your `.blend`.
            - Your `.blend` must contain the actual rig and animation that produce the submitted preview.
            """)
        )

    def to_metadata(self) -> dict[str, Any]:
        data = super().to_metadata()
        data.pop("software_dir", None)
        data.update(
            {
                "task_tag": self.TASK_TAG,
                "display_name": self.DISPLAY_NAME,
                "input_dir": self.input_dir,
                "input_obj": self.input_obj,
                "input_mtl": self.input_mtl,
                "reference_video": self.reference_video,
                "clean_reference_video": self.clean_reference_video,
                "output_submission_dir": self.output_submission_dir,
                "output_blend": self.output_blend,
                "output_preview_video": self.output_preview_video,
                "hidden_reference_fbx": self.hidden_reference_fbx,
                "skeleton_package": self.skeleton_package,
                "evaluation_config": self.evaluation_config,
                "manifest": self.manifest,
                "motion_scope": self.MOTION_SCOPE,
                "visible_required_bone_names": list(self.REQUIRED_BONE_NAMES),
            }
        )
        return data


def _build_config(spec: VariantSpec) -> SkeletalAnimationTaskConfig:
    return SkeletalAnimationTaskConfig(
        VARIANT_NAME=spec.remote_task_dir_name,
        TASK_TAG=spec.task_tag,
        DISPLAY_NAME=spec.display_name,
        REMOTE_TASK_DIR_NAME=spec.remote_task_dir_name,
        INPUT_OBJ_NAME=spec.input_obj_name,
        INPUT_MTL_NAME=spec.input_mtl_name,
        HIDDEN_REFERENCE_FBX_NAME=spec.hidden_reference_fbx_name,
        MOTION_SCOPE=spec.motion_scope,
        REQUIRED_BONE_NAMES=spec.required_bone_names,
    )


def _assert_visible_required_bone_contract(
    meta: dict[str, Any], skeleton_package: dict[str, Any]
) -> None:
    hidden_required = {
        str(item) for item in skeleton_package.get("required_bones", []) if str(item).strip()
    }
    visible_required = {
        str(item) for item in meta.get("visible_required_bone_names", []) if str(item).strip()
    }
    if not hidden_required:
        return
    missing_from_prompt = sorted(hidden_required - visible_required)
    if missing_from_prompt:
        raise RuntimeError(
            "Task implementation bug: evaluator package requires bone names that are not "
            f"exposed in the agent-visible task contract for {meta.get('task_tag')}: "
            f"{missing_from_prompt}"
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


_setup = BaseTaskSetup()


@cb.setup_task(split="train")
async def start(task_cfg, session: cb.DesktopSession):
    await _setup(task_cfg, session)


async def _upload_scripts(session: cb.DesktopSession, remote_scripts_dir: str) -> None:
    await session.interface.create_dir(remote_scripts_dir)
    for name in ["remote_render_eval.py", "blender_render_submission.py"]:
        await session.write_file(
            _remote_child(remote_scripts_dir, name),
            (SCRIPTS_DIR / name).read_text(encoding="utf-8"),
        )


async def _launch_remote_job(
    session: cb.DesktopSession, *, remote_scripts_dir: str, args: list[str]
) -> None:
    script_path = _remote_child(remote_scripts_dir, "remote_render_eval.py")
    stdout_path = _remote_child(remote_scripts_dir, "job_stdout.txt")
    stderr_path = _remote_child(remote_scripts_dir, "job_stderr.txt")
    argv = ", ".join("'" + _ps_quote(v) + "'" for v in [script_path, *args])
    ps = (
        "$ErrorActionPreference='Stop'; "
        f"$wd='{_ps_quote(remote_scripts_dir)}'; "
        f"$py='{_ps_quote(REMOTE_PYTHON)}'; "
        f"$env:BLENDER_BINARY='{_ps_quote(REMOTE_BLENDER)}'; "
        f"$stdout='{_ps_quote(stdout_path)}'; "
        f"$stderr='{_ps_quote(stderr_path)}'; "
        "Set-Location -LiteralPath $wd; "
        "if (Test-Path -LiteralPath $stdout) { Remove-Item -LiteralPath $stdout -Force -ErrorAction SilentlyContinue }; "
        "if (Test-Path -LiteralPath $stderr) { Remove-Item -LiteralPath $stderr -Force -ErrorAction SilentlyContinue }; "
        f"Start-Process -FilePath $py -ArgumentList @({argv}) -WorkingDirectory $wd "
        "-RedirectStandardOutput $stdout -RedirectStandardError $stderr -WindowStyle Hidden"
    )
    await session.run_command(f'powershell -NoProfile -Command "{ps}"', check=False)


async def _wait_for_file(
    session: cb.DesktopSession,
    path: str,
    timeout_sec: float = 1800.0,
    poll_sec: float = 10.0,
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


def _read_session_text(payload: Any) -> str:
    if isinstance(payload, str):
        return payload
    content = getattr(payload, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(payload, bytes):
        return payload.decode("utf-8")
    return str(payload)


async def _download_reference_package(
    session: cb.DesktopSession,
    *,
    reference_dir: str,
    local_dir: Path,
    skeleton_package: dict[str, Any],
) -> tuple[Path, Path]:
    local_reference_dir = local_dir / "reference"
    local_reference_dir.mkdir(parents=True, exist_ok=True)

    local_clean_reference_video = local_reference_dir / "reference_clean.mp4"
    local_clean_reference_video.write_bytes(
        await session.read_bytes(_remote_child(reference_dir, "reference_clean.mp4"))
    )

    (local_reference_dir / "package.json").write_text(
        json.dumps(skeleton_package, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    for pose_state in skeleton_package.get("pose_states", []):
        rel_gt_path = str(pose_state.get("gt_pose_image", "")).strip()
        if not rel_gt_path:
            continue
        remote_gt_path = _remote_child(reference_dir, *PureWindowsPath(rel_gt_path).parts)
        local_gt_path = local_reference_dir / Path(rel_gt_path)
        local_gt_path.parent.mkdir(parents=True, exist_ok=True)
        local_gt_path.write_bytes(await session.read_bytes(remote_gt_path))

    return local_reference_dir, local_clean_reference_video


@cb.evaluate_task(split="train")
async def evaluate(task_cfg, session: cb.DesktopSession) -> list[float]:
    meta = task_cfg.metadata
    if not (await session.file_exists(meta["output_blend"]) or await session.directory_exists(meta["output_blend"])):
        return [0.0]
    if not (await session.file_exists(meta["output_preview_video"]) or await session.directory_exists(meta["output_preview_video"])):
        return [0.0]
    with TemporaryDirectory(prefix=f"skeletal_eval_{meta['task_tag']}_") as tmp_dir:
        local_tmp = Path(tmp_dir)

        static_config = json.loads(
            _read_session_text(await session.read_file(meta["evaluation_config"]))
        )
        skeleton_package = json.loads(
            _read_session_text(await session.read_file(meta["skeleton_package"]))
        )
        _assert_visible_required_bone_contract(meta, skeleton_package)
        local_reference_dir, local_clean_reference_video = await _download_reference_package(
            session,
            reference_dir=meta["reference_dir"],
            local_dir=local_tmp,
            skeleton_package=skeleton_package,
        )

        pre_sample_positions = evenly_spaced_positions(int(static_config.get("sample_count", 10)))
        reference_frame_dir = local_tmp / "reference_frames"
        reference_frame_dir.mkdir(parents=True, exist_ok=True)
        reference_frame_paths = extract_reference_frames(
            video_path=local_clean_reference_video,
            sample_positions=pre_sample_positions,
            output_dir=reference_frame_dir,
        )
        runtime_config = dict(static_config)
        runtime_config["reference_framing"] = reference_framing_targets(reference_frame_paths)
        runtime_config["skeleton_package"] = skeleton_package

        remote_eval_dir = _remote_child(REMOTE_TEMP_ROOT, meta["variant_name"])
        remote_scripts_dir = _remote_child(remote_eval_dir, "scripts")
        remote_results_dir = _remote_child(remote_eval_dir, "results")
        remote_report = _remote_child(remote_results_dir, "render_report.json")
        remote_runtime_config = _remote_child(remote_eval_dir, "runtime_evaluation_config.json")
        await session.interface.create_dir(remote_eval_dir)
        await _upload_scripts(session, remote_scripts_dir)
        await session.write_file(
            remote_runtime_config, json.dumps(runtime_config, ensure_ascii=False, indent=2)
        )
        await _launch_remote_job(
            session,
            remote_scripts_dir=remote_scripts_dir,
            args=[
                "--blend",
                meta["output_blend"],
                "--output-dir",
                remote_results_dir,
                "--renderer-script",
                _remote_child(remote_scripts_dir, "blender_render_submission.py"),
                "--evaluation-config",
                remote_runtime_config,
            ],
        )
        if not await _wait_for_file(session, remote_report):
            return [0.0]

        report = json.loads(_read_session_text(await session.read_file(remote_report)))
        if not report.get("validity_gate_passed", False):
            final_report = {
                "validity_gate_passed": False,
                "gate_fail_reasons": report.get("gate_fail_reasons", []),
                "video_match_score": 0.0,
                "replay_consistency_score": 0.0,
                "minimal_skeleton_score": 0.0,
                "vlm_score": 0.0,
                "final_score": 0.0,
                "metrics": report,
                "evidence_paths": {},
            }
            (local_tmp / "final_report.json").write_text(
                json.dumps(final_report, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            return [0.0]

        candidate_render_dir = local_tmp / "candidate"
        candidate_render_dir.mkdir(parents=True, exist_ok=True)
        localized_view_paths: dict[str, list[Path]] = {}
        for view, remote_paths in report.get("view_paths", {}).items():
            local_paths: list[Path] = []
            for idx, remote_path in enumerate(remote_paths):
                local_path = candidate_render_dir / f"{view}_{idx:02d}.png"
                local_path.write_bytes(await session.read_bytes(remote_path))
                local_paths.append(local_path)
            localized_view_paths[view] = local_paths

        localized_pose_states = []
        for pose_state in report.get("pose_states", []):
            remote_path = pose_state.get("image_path")
            if not remote_path:
                continue
            pose_name = str(pose_state.get("name", f"pose_{len(localized_pose_states):02d}"))
            local_path = candidate_render_dir / f"{pose_name}_candidate.png"
            local_path.write_bytes(await session.read_bytes(remote_path))
            item = dict(pose_state)
            item["image_path"] = str(local_path)
            localized_pose_states.append(item)
        report["pose_states"] = localized_pose_states

        sample_positions = [float(item) for item in report.get("sample_positions", [])]
        if sample_positions != pre_sample_positions:
            reference_frame_dir = local_tmp / "reference_frames"
            if reference_frame_dir.exists():
                shutil.rmtree(reference_frame_dir)
            reference_frame_dir.mkdir(parents=True, exist_ok=True)
            reference_frame_paths = extract_reference_frames(
                video_path=local_clean_reference_video,
                sample_positions=sample_positions,
                output_dir=reference_frame_dir,
            )

        preview_frame_dir = local_tmp / "preview_frames"
        preview_frame_dir.mkdir(parents=True, exist_ok=True)
        local_preview_video = local_tmp / "preview.mp4"
        local_preview_video.write_bytes(await session.read_bytes(meta["output_preview_video"]))
        preview_frame_paths = extract_reference_frames(
            video_path=local_preview_video,
            sample_positions=sample_positions,
            output_dir=preview_frame_dir,
            prefix="preview",
        )

        pose_reference_dir = local_tmp / "reference_pose_frames"
        pose_reference_dir.mkdir(parents=True, exist_ok=True)
        pose_positions = [
            float(item.get("sample_position", 0.0))
            for item in skeleton_package.get("pose_states", [])
        ]
        reference_pose_paths = extract_reference_frames(
            video_path=local_clean_reference_video,
            sample_positions=pose_positions,
            output_dir=pose_reference_dir,
            prefix="reference_pose",
        )
        pose_state_rows = []
        package_root = local_reference_dir
        for idx, pose_state in enumerate(skeleton_package.get("pose_states", [])):
            candidate_pose = next(
                (
                    item
                    for item in localized_pose_states
                    if item.get("name") == pose_state.get("name")
                ),
                None,
            )
            if candidate_pose is None or idx >= len(reference_pose_paths):
                continue
            pose_state_rows.append(
                {
                    "name": str(pose_state.get("name", f"pose_{idx:02d}")),
                    "reference_pose": reference_pose_paths[idx],
                    "gt_pose": package_root / str(pose_state.get("gt_pose_image", "")),
                    "candidate_pose": Path(str(candidate_pose["image_path"])),
                }
            )

        hard_metrics = compute_local_metrics(
            reference_frame_paths=reference_frame_paths,
            preview_frame_paths=preview_frame_paths,
            replay_frame_paths=localized_view_paths.get("front", []),
            skeleton_package=skeleton_package,
            package_root=package_root,
            validity_payload=report,
        )

        bundle = write_eval_bundle(
            output_dir=local_tmp / "bundle",
            reference_frame_paths=reference_frame_paths,
            preview_frame_paths=preview_frame_paths,
            replay_frame_paths=localized_view_paths.get("front", []),
            pose_state_rows=pose_state_rows,
        )

        vlm_score = float(
            (
                float(hard_metrics["video_match_score"])
                + float(hard_metrics["replay_consistency_score"])
                + float(hard_metrics["minimal_skeleton_score"])
            )
            / 3.0
        )
        if os.environ.get("OPENAI_API_KEY"):
            try:
                vlm_score = run_local_soft_eval(
                    reference_sheet=bundle["reference_sheet"],
                    preview_sheet=bundle["preview_sheet"],
                    replay_sheet=bundle["replay_sheet"],
                    pose_state_sheet=bundle["pose_state_sheet"],
                )
            except Exception:
                logger.warning(
                    "Local VLM judge failed; falling back to machine-score mean", exc_info=True
                )

        final_score = compute_final_score(
            validity_gate=bool(hard_metrics["validity_gate_passed"]),
            video_match_score=float(hard_metrics["video_match_score"]),
            replay_consistency_score=float(hard_metrics["replay_consistency_score"]),
            minimal_skeleton_score=float(hard_metrics["minimal_skeleton_score"]),
            vlm_score=float(vlm_score),
        )

        final_report = {
            "validity_gate_passed": hard_metrics["validity_gate_passed"],
            "gate_fail_reasons": hard_metrics.get("gate_fail_reasons", []),
            "video_match_score": hard_metrics["video_match_score"],
            "replay_consistency_score": hard_metrics["replay_consistency_score"],
            "minimal_skeleton_score": hard_metrics["minimal_skeleton_score"],
            "vlm_score": vlm_score,
            "final_score": final_score,
            "metrics": hard_metrics,
            "evidence_paths": {key: str(value) for key, value in bundle.items()},
        }
        (local_tmp / "final_report.json").write_text(
            json.dumps(final_report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info(
            "skeletal_animation_reproduction eval task=%s validity=%s video_match=%.4f replay=%.4f skeleton=%.4f vlm=%.4f final=%.4f",
            meta["variant_name"],
            hard_metrics["validity_gate_passed"],
            hard_metrics["video_match_score"],
            hard_metrics["replay_consistency_score"],
            hard_metrics["minimal_skeleton_score"],
            vlm_score,
            final_score,
        )
        return [float(final_score)]


if __name__ == "__main__":
    print(
        json.dumps(
            {"workflow": TASK_WORKFLOW, "variants": [spec.task_tag for spec in VARIANTS]},
            ensure_ascii=False,
            indent=2,
        )
    )
