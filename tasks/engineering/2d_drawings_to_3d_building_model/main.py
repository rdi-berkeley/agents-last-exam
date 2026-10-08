"""Betonwerk Katzenberger — 3D architectural model from drawings.

Single-variant task. Agent receives 6 PDFs + a 3D snapshot + a footprint-only
`base_model.{obj,3dm}` and builds the full Betonwerk Katzenberger building
(workshop Hall + residential Tower) in Blender, then exports `model.obj` and
`model.blend`. Evaluation renders 14 canonical views of the agent's OBJ with
Blender on the VM and scores them against the frozen reference with a
local image-only multimodal LLM judge.

Z-up, millimeters. See `tmp/base/eval_config.json` for the per-instance
judge questions, units, and up-axis.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shlex
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import cua_bench as cb

from tasks.linux_runtime import LinuxTaskConfig
from tasks.common_setup import BaseTaskSetup


_setup = BaseTaskSetup()
logger = logging.getLogger(__name__)

SCRIPTS_DIR = Path(__file__).parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from score_outputs import DEFAULT_VIEW_NAMES, evaluate_renders  # noqa: E402

DOMAIN_NAME = "engineering"
TASK_NAME = "2d_drawings_to_3d_building_model"
VARIANT_NAME = "base"
VARIANT_LABEL = "Betonwerk Katzenberger — workshop hall + residential tower"

REMOTE_BLENDER_CANDIDATES: tuple[str, ...] = (
    "/opt/blender-4.5.1/blender",
    "/usr/local/bin/blender",
    "/usr/bin/blender",
)
REMOTE_EVAL_ROOT = os.environ.get(
    "D2T3B_REMOTE_EVAL_ROOT",
    "/home/user/.cache/ale-evaluation/2d_drawings_to_3d_building_model",
)
RENDER_SCRIPT_FILES = ("render_human_views.py", "detect_floors.py", "validate_native.py")


def _remote_child(base: str, *parts: str) -> str:
    path = PurePosixPath(base)
    for part in parts:
        path = path / part
    return str(path)


@dataclass
class DrawingsTo3DBuildingConfig(LinuxTaskConfig):
    DOMAIN_NAME: str = DOMAIN_NAME
    TASK_NAME: str = TASK_NAME
    VARIANT_NAME: str = VARIANT_NAME

    @property
    def input_dir(self) -> str:
        return f"{self.task_dir}/input"

    @property
    def output_obj(self) -> str:
        return f"{self.remote_output_dir}/model.obj"

    @property
    def output_blend(self) -> str:
        return f"{self.remote_output_dir}/model.blend"

    @property
    def reference_eval_config(self) -> str:
        return f"{self.reference_dir}/eval_config.json"

    @property
    def reference_renders_dir(self) -> str:
        return f"{self.reference_dir}/reference_renders_linux"

    @property
    def task_description(self) -> str:
        return f"""\
You are a BIM modeler using Blender to build the Betonwerk Katzenberger
building in 3D, given the architectural drawings of the project.

The building is a Bavarian concrete-plant complex with two main parts:
  - a wide, low workshop Hall housing three prefabricated workshop modules
  - a tall narrow residential Tower (Hostel) attached at one end

## Inputs
All input materials are under:
  {self.input_dir}

Required files:
- README.md — natural-language task brief (open this first)
- output_contract.json — machine-readable spec (file names, formats,
  coordinate convention, required geometry components)
- architectural_drawings/*.pdf — 6 drawings (Hall plans + Tower plan +
  Hostel elevation/section + Diagram)
- 3D Snapshot.png — rendered preview of the finished building
- base_model.obj — footprint-only positioning anchor

## Software
Blender 4.5 is installed at /opt/blender-4.5.1/blender. GUI and scripting
workflows are both allowed; rendering and evaluation work on a CPU.

## What You Must Do
1. Read README.md and output_contract.json.
2. Open the PDFs and import base_model.obj into Blender. It gives the Hall footprint
   and axis orientation. Keep the dimensions, orientation and ground-floor elevation.
3. Build the geometry (Hall + Tower + facades + structural framing +
   modules + mezzanines + glass curtain wall).
4. Export to {self.output_obj} (Wavefront OBJ ASCII) and
   {self.output_blend} (Blender native). Both must contain the same
   evaluated geometry within 1 mm, including applied modifiers and instances.
   Mesh organization, object names and triangulation are unrestricted.
   Pack linked geometry so the saved project reopens independently.

## Hard Constraints
- Do not output any `.dwg` (evaluator cannot read DWG; forbidden).
- Units = millimeters. +X east, +Y north, +Z up; ground-floor elevation is z=0.
- Common X/Y translation of the complete model is allowed. Evaluation centers
  the views independently; absolute horizontal origin is not scored. Relative
  placement of Hall, Tower, structure and workshop modules remains required.
- The evaluator will render 14 canonical views from your model.obj and
  compare them to a hidden reference using a multimodal LLM judge.
- The eight equally weighted criteria are Hall/Tower proportions, floor levels,
  Tower placement, the solid north Hall facade, the open south structural frame,
  the Tower curtain-wall mullions, workshop-module placement, and the four
  elevation silhouettes. Materials and textures are not scored.
"""

    def to_metadata(self) -> dict:
        metadata = super().to_metadata()
        metadata.update(
            {
                "variant_name": self.VARIANT_NAME,
                "input_dir": self.input_dir,
                "remote_output_dir": self.remote_output_dir,
                "output_obj": self.output_obj,
                "output_blend": self.output_blend,
                "reference_eval_config": self.reference_eval_config,
                "reference_renders_dir": self.reference_renders_dir,
            }
        )
        return metadata


@cb.tasks_config(split="train")
def load():
    return [
        cb.Task(
            description=DrawingsTo3DBuildingConfig().task_description,
            metadata=DrawingsTo3DBuildingConfig().to_metadata(),
            computer={
                "provider": "computer",
                "setup_config": {"os_type": "linux"},
            },
        )
    ]


@cb.setup_task(split="train")
async def start(task_cfg, session: cb.DesktopSession):
    await _setup(task_cfg, session)
    for name in ("README.md", "output_contract.json"):
        await session.write_file(
            _remote_child(task_cfg.metadata["input_dir"], name),
            (Path(__file__).parent / "assets" / name).read_text(),
        )


async def _log_missing(session, remote_path: str, *, tag: str, label: str) -> bool:
    if not (await session.file_exists(remote_path) or await session.directory_exists(remote_path)):
        logger.error(f"[{tag}] missing {label}: {remote_path}")
        return True
    return False


async def _upload_render_scripts(session: cb.DesktopSession, remote_scripts_dir: str) -> None:
    await session.interface.create_dir(remote_scripts_dir)
    for name in RENDER_SCRIPT_FILES:
        await session.write_file(
            _remote_child(remote_scripts_dir, name),
            (SCRIPTS_DIR / name).read_text(encoding="utf-8"),
        )


async def _reset_remote_dir(session: cb.DesktopSession, path: str) -> None:
    await session.run_command(f"rm -rf -- {shlex.quote(path)}")
    await session.interface.create_dir(path)


async def _resolve_remote_blender(session: cb.DesktopSession) -> str | None:
    """Locate the Linux Blender runtime."""
    override = os.environ.get("D2T3B_REMOTE_BLENDER") or os.environ.get("BLENDER_TASK_REMOTE_BLENDER")
    if override:
        return override
    for path in REMOTE_BLENDER_CANDIDATES:
        if await session.file_exists(path):
            return path
    listing = await session.run_command("command -v blender")
    for line in (listing.get("stdout") or "").splitlines():
        candidate = line.strip()
        if candidate.startswith("/"):
            return candidate
    return None


async def _launch_remote_blender_render(
    session: cb.DesktopSession,
    *,
    blender_bin: str,
    remote_scripts_dir: str,
    output_obj: str,
    output_blend: str,
    candidate_dir: str,
    render_units: str,
    source_up_axis: str,
    cand_res: int,
    cand_samples: int,
) -> None:
    render_script = _remote_child(remote_scripts_dir, "render_human_views.py")
    stdout_path = _remote_child(remote_scripts_dir, "blender_stdout.txt")
    stderr_path = _remote_child(remote_scripts_dir, "blender_stderr.txt")
    exit_path = _remote_child(candidate_dir, "exit-code")
    validate_script = _remote_child(remote_scripts_dir, "validate_native.py")
    report_path = _remote_child(candidate_dir, "native-report.json")
    validator = shlex.join([
        "timeout", "--kill-after=10", "300", blender_bin, "--background",
        "--factory-startup", "--disable-autoexec", "--python-exit-code", "1",
        "--python", validate_script, "--", "--obj", output_obj,
        "--blend", output_blend, "--report", report_path,
    ])
    renderer = shlex.join([
        "timeout", "--kill-after=10", "900", blender_bin, "--background",
        "--factory-startup", "--disable-autoexec", "--python-exit-code", "1",
        "--python", render_script, "--", "--obj", output_obj,
        "--out", candidate_dir, "--units", render_units,
        "--source-up-axis", source_up_axis, "--res", str(cand_res),
        "--samples", str(cand_samples),
    ])
    script = (
        f"({validator} && {renderer}) > {shlex.quote(stdout_path)} "
        f"2> {shlex.quote(stderr_path)}\nrender_status=$?\n"
        f"printf '%s' \"$render_status\" > {shlex.quote(exit_path)}"
    )
    launcher = (
        "import subprocess; "
        f"process = subprocess.Popen(['sh', '-c', {script!r}], "
        "stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, "
        "close_fds=True, start_new_session=True); print(process.pid)"
    )
    launched = await session.run_command("python3 -c " + shlex.quote(launcher))
    if not launched.get("stdout", "").strip().isdigit():
        raise RuntimeError(f"Could not launch building renderer: {launched}")


async def _wait_for_render_outputs(
    session: cb.DesktopSession,
    *,
    candidate_dir: str,
    view_names: list[str],
    timeout_sec: float = 1250.0,
    poll_sec: float = 10.0,
) -> bool:
    deadline = asyncio.get_event_loop().time() + timeout_sec
    expected = [_remote_child(candidate_dir, f"{name}.png") for name in view_names]
    exit_path = _remote_child(candidate_dir, "exit-code")
    report_path = _remote_child(candidate_dir, "native-report.json")
    while asyncio.get_event_loop().time() < deadline:
        if await session.file_exists(exit_path):
            status = (await session.read_bytes(exit_path)).strip()
            if not await session.file_exists(report_path):
                raise RuntimeError(f"Building evaluator exited {status!r} without a native validation report")
            try:
                report = json.loads(await session.read_bytes(report_path))
            except (OSError, ValueError) as error:
                raise RuntimeError(f"Building evaluator exited {status!r} without a readable native report") from error
            if not isinstance(report, dict) or type(report.get("valid")) is not bool:
                raise RuntimeError("Building evaluator produced an invalid native report")
            if status != b"0":
                if report["valid"] is False and status in {b"1", b"2"}:
                    return False
                raise RuntimeError(f"Building evaluator failed after native validation: exit {status!r}")
            if not report["valid"]:
                raise RuntimeError("Building evaluator reported success with invalid native geometry")
            for path in expected:
                if not await session.file_exists(path):
                    raise RuntimeError(f"Building evaluator exited successfully but did not produce {path}")
            return True
        await asyncio.sleep(poll_sec)
    raise RuntimeError(f"Building evaluator did not finish within {timeout_sec:g} seconds")


async def _read_text_if_exists(session: cb.DesktopSession, path: str) -> str:
    try:
        if not (await session.file_exists(path) or await session.directory_exists(path)):
            return ""
        return (await session.read_bytes(path)).decode("utf-8", errors="replace")
    except Exception:
        return ""


async def _download_render_pngs(
    session: cb.DesktopSession,
    *,
    remote_dir: str,
    local_dir: Path,
    view_names: list[str],
) -> None:
    local_dir.mkdir(parents=True, exist_ok=True)
    for name in view_names:
        png_bytes = await session.read_bytes(_remote_child(remote_dir, f"{name}.png"))
        (local_dir / f"{name}.png").write_bytes(png_bytes)


@cb.evaluate_task(split="train")
async def evaluate(task_cfg, session: cb.DesktopSession) -> list[float]:
    """Render candidate views on the VM with Blender; judge locally with the VLM."""
    meta = task_cfg.metadata
    tag = meta["variant_name"]
    output_dir = meta["remote_output_dir"]
    output_obj = meta["output_obj"]
    output_blend = meta["output_blend"]
    ref_config_remote = meta["reference_eval_config"]
    ref_renders_remote = meta["reference_renders_dir"]

    logger.info(
        f"[{tag}] starting CPU Blender evaluation; "
        f"Blender renders remotely, VLM judge runs locally (output_dir={output_dir})"
    )

    blender_bin = await _resolve_remote_blender(session)
    if not blender_bin:
        raise RuntimeError("Blender is missing from the Linux task environment")
    logger.info(f"[{tag}] using remote Blender: {blender_bin}")

    for path, label in [
        (output_obj, "output OBJ"),
        (output_blend, "output BLEND"),
    ]:
        if await _log_missing(session, path, tag=tag, label=label):
            return [0.0]

    for path in (ref_config_remote, ref_renders_remote):
        if not (await session.file_exists(path) or await session.directory_exists(path)):
            raise RuntimeError(f"Missing building reference: {path}")

    listing = await session.run_command(f"find {shlex.quote(output_dir)} -type f -iname '*.dwg' -print")
    if listing.get("stdout", "").strip():
        logger.error(f"[{tag}] forbidden .dwg files in output: {listing['stdout'][:200]}")
        return [0.0]

    try:
        config_bytes = await session.read_bytes(ref_config_remote)
        variant_config = json.loads(config_bytes.decode("utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Could not load building eval_config: {exc}") from exc

    render_units = variant_config.get("render_units", "mm")
    source_up_axis = variant_config.get("render_source_up_axis", "Z")
    view_names = list(variant_config.get("view_names") or DEFAULT_VIEW_NAMES)

    remote_eval_dir = _remote_child(REMOTE_EVAL_ROOT, tag)
    remote_scripts_dir = _remote_child(remote_eval_dir, "scripts")
    remote_candidate_dir = _remote_child(remote_eval_dir, "candidate_renders")

    await session.interface.create_dir(remote_eval_dir)
    await _reset_remote_dir(session, remote_candidate_dir)
    await _upload_render_scripts(session, remote_scripts_dir)

    cand_res = int(os.environ.get("CAND_RES", os.environ.get("D2T3B_CAND_RES", "1024")))
    cand_samples = int(os.environ.get("CAND_SAMPLES", os.environ.get("D2T3B_CAND_SAMPLES", "32")))

    await _launch_remote_blender_render(
        session,
        blender_bin=blender_bin,
        remote_scripts_dir=remote_scripts_dir,
        output_obj=output_obj,
        output_blend=output_blend,
        candidate_dir=remote_candidate_dir,
        render_units=render_units,
        source_up_axis=source_up_axis,
        cand_res=cand_res,
        cand_samples=cand_samples,
    )

    try:
        valid = await _wait_for_render_outputs(
            session,
            candidate_dir=remote_candidate_dir,
            view_names=view_names,
        )
    except RuntimeError:
        stderr = await _read_text_if_exists(
            session, _remote_child(remote_scripts_dir, "blender_stderr.txt")
        )
        stdout = await _read_text_if_exists(
            session, _remote_child(remote_scripts_dir, "blender_stdout.txt")
        )
        logger.error(
            f"[{tag}] Building evaluator failed. stdout={stdout[-2000:]} stderr={stderr[-2000:]}"
        )
        raise
    if not valid:
        logger.info(f"[{tag}] submitted native geometry is invalid or differs from the OBJ")
        return [0.0]

    with tempfile.TemporaryDirectory(prefix=f"betonwerk_{tag}_") as scratch:
        scratch_p = Path(scratch)
        local_ref_renders = scratch_p / "reference_renders"
        local_cand_renders = scratch_p / "candidate_renders"
        local_config = scratch_p / "eval_config.json"
        local_config.write_bytes(config_bytes)

        await _download_render_pngs(
            session,
            remote_dir=ref_renders_remote,
            local_dir=local_ref_renders,
            view_names=view_names,
        )
        await _download_render_pngs(
            session,
            remote_dir=remote_candidate_dir,
            local_dir=local_cand_renders,
            view_names=view_names,
        )

        loop = asyncio.get_running_loop()

        def persist_judge_report(report: dict) -> None:
            payload = json.dumps(report, ensure_ascii=False)
            asyncio.run_coroutine_threadsafe(
                session.write_file(_remote_child(remote_eval_dir, "judge-report.json"), payload),
                loop,
            ).result()

        score_report = await asyncio.to_thread(
            evaluate_renders,
            reference_render_dir=local_ref_renders,
            candidate_render_dir=local_cand_renders,
            config_path=local_config,
            on_report=persist_judge_report,
        )

    logger.info(
        "[%s] judge score: %s (%s/%s); full replies saved to %s/judge-report.json",
        tag, score_report["score"], score_report["yes_count"],
        score_report["question_count"], remote_eval_dir,
    )
    return [float(score_report["score"])]
