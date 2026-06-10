"""Betonwerk Katzenberger — 3D architectural model from drawings.

Single-variant task. Agent receives 6 PDFs + a 3D snapshot + a footprint-only
`base_model.{obj,3dm}` and builds the full Betonwerk Katzenberger building
(workshop Hall + residential Tower) in Rhino 8, then exports `model.obj` and
`model.3dm`. Evaluation renders 14 canonical views of the agent's OBJ with
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
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath

import cua_bench as cb

from tasks.common_config import GeneralTaskConfig
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

REQUIRED_VM_NAME = "agenthle-dev-gpu-licensed"
REMOTE_BLENDER_CANDIDATES: tuple[str, ...] = (
    r"C:\Program Files\Blender Foundation\Blender 4.2\blender.exe",
    r"C:\Program Files\Blender Foundation\Blender 5.0\blender.exe",
)
REMOTE_EVAL_ROOT = os.environ.get(
    "D2T3B_REMOTE_EVAL_ROOT",
    r"C:\Users\User\AppData\Local\Temp\agenthle_eval\2d_drawings_to_3d_building_model",
)
RENDER_SCRIPT_FILES = ("render_human_views.py", "detect_floors.py")


def _remote_child(base: str, *parts: str) -> str:
    path = PureWindowsPath(base)
    for part in parts:
        path = path / part
    return str(path)


@dataclass
class DrawingsTo3DBuildingConfig(GeneralTaskConfig):
    DOMAIN_NAME: str = DOMAIN_NAME
    TASK_NAME: str = TASK_NAME
    VARIANT_NAME: str = VARIANT_NAME

    @property
    def input_dir(self) -> str:
        return rf"{self.task_dir}\input"

    @property
    def output_obj(self) -> str:
        return rf"{self.remote_output_dir}\model.obj"

    @property
    def output_3dm(self) -> str:
        return rf"{self.remote_output_dir}\model.3dm"

    @property
    def reference_eval_config(self) -> str:
        return rf"{self.reference_dir}\eval_config.json"

    @property
    def reference_renders_dir(self) -> str:
        return rf"{self.reference_dir}\reference_renders_v1"

    @property
    def software_shortcut(self) -> str:
        return rf"{self.software_dir}\Rhino 8.lnk"

    @property
    def task_description(self) -> str:
        return f"""\
You are a BIM modeler using Rhino 8 to build the Betonwerk Katzenberger
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
- base_model.obj + base_model.3dm — footprint-only positioning anchor

## Software
Open Rhino 8 via:
  {self.software_shortcut}

## What You Must Do
1. Read README.md and output_contract.json.
2. Open the PDFs and base_model.3dm in Rhino. The base_model defines the
   world coordinate frame — do not move the origin or rotate it.
3. Build the geometry (Hall + Tower + facades + structural framing +
   modules + mezzanines + glass curtain wall).
4. Export to {self.output_obj} (Wavefront OBJ ASCII) and
   {self.output_3dm} (Rhino 5.0+ native). Both must contain the same
   geometry.

## Hard Constraints
- Do not output any `.dwg` (evaluator cannot read DWG; forbidden).
- Units = millimeters. Origin from base_model. +X east, +Y north, +Z up.
- The evaluator will render 14 canonical views from your model.obj and
  compare them to a hidden reference using a multimodal LLM judge.
"""

    def to_metadata(self) -> dict:
        metadata = super().to_metadata()
        metadata.update(
            {
                "variant_name": self.VARIANT_NAME,
                "input_dir": self.input_dir,
                "remote_output_dir": self.remote_output_dir,
                "output_obj": self.output_obj,
                "output_3dm": self.output_3dm,
                "reference_eval_config": self.reference_eval_config,
                "reference_renders_dir": self.reference_renders_dir,
                "software_shortcut": self.software_shortcut,
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
                "setup_config": {"os_type": "windows"},
            },
        )
    ]


@cb.setup_task(split="train")
async def start(task_cfg, session: cb.DesktopSession):
    await _setup(task_cfg, session)


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
    await session.run_command(f'cmd /c if exist "{path}" rmdir /s /q "{path}"', check=False)
    await session.interface.create_dir(path)


async def _resolve_remote_blender(session: cb.DesktopSession) -> str | None:
    """Pick Blender on the task VM. Requires a GPU host (agenthle-dev-gpu-licensed)."""
    override = os.environ.get("D2T3B_REMOTE_BLENDER") or os.environ.get("BLENDER_TASK_REMOTE_BLENDER")
    if override:
        return override
    for path in REMOTE_BLENDER_CANDIDATES:
        if await session.file_exists(path):
            return path
    listing = await session.run_command(
        r'for /d %D in ("C:\Program Files\Blender Foundation\*") do @if exist "%D\blender.exe" echo %D\blender.exe'
    )
    for line in (listing.get("stdout") or "").splitlines():
        candidate = line.strip()
        if candidate.lower().endswith("blender.exe"):
            return candidate
    return None


async def _launch_remote_blender_render(
    session: cb.DesktopSession,
    *,
    blender_bin: str,
    remote_scripts_dir: str,
    output_obj: str,
    candidate_dir: str,
    render_units: str,
    source_up_axis: str,
    cand_res: int,
    cand_samples: int,
) -> None:
    render_script = _remote_child(remote_scripts_dir, "render_human_views.py")
    stdout_path = _remote_child(remote_scripts_dir, "blender_stdout.txt")
    stderr_path = _remote_child(remote_scripts_dir, "blender_stderr.txt")
    await session.run_command(
        f'cmd /c if exist "{stdout_path}" del /f /q "{stdout_path}"',
        check=False,
    )
    await session.run_command(
        f'cmd /c if exist "{stderr_path}" del /f /q "{stderr_path}"',
        check=False,
    )
    cmd = (
        f'"{blender_bin}" --background --python "{render_script}" -- '
        f'--obj "{output_obj}" --out "{candidate_dir}" '
        f'--units {render_units} --source-up-axis {source_up_axis} '
        f'--res {cand_res} --samples {cand_samples} '
        f'1> "{stdout_path}" 2> "{stderr_path}"'
    )
    await session.run_command(cmd, check=False)


async def _wait_for_render_outputs(
    session: cb.DesktopSession,
    *,
    candidate_dir: str,
    view_names: list[str],
    timeout_sec: float = 2400.0,
    poll_sec: float = 10.0,
) -> bool:
    deadline = asyncio.get_event_loop().time() + timeout_sec
    expected = [_remote_child(candidate_dir, f"{name}.png") for name in view_names]
    while asyncio.get_event_loop().time() < deadline:
        ready = True
        for path in expected:
            if not (await session.file_exists(path) or await session.directory_exists(path)):
                ready = False
                break
        if ready:
            return True
        await asyncio.sleep(poll_sec)
    return False


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
    output_3dm = meta["output_3dm"]
    ref_config_remote = meta["reference_eval_config"]
    ref_renders_remote = meta["reference_renders_dir"]

    logger.info(
        f"[{tag}] starting evaluation on GPU VM ({REQUIRED_VM_NAME}); "
        f"Blender renders remotely, VLM judge runs locally (output_dir={output_dir})"
    )

    blender_bin = await _resolve_remote_blender(session)
    if not blender_bin:
        logger.error(
            f"[{tag}] Blender not found on VM — evaluator requires {REQUIRED_VM_NAME} "
            "(EEVEE headless needs the L4 GPU display stack; CPU dev VMs are unsupported)"
        )
        return [0.0]
    logger.info(f"[{tag}] using remote Blender: {blender_bin}")

    for path, label in [
        (output_obj, "output OBJ"),
        (output_3dm, "output 3DM"),
        (ref_config_remote, "reference eval_config.json"),
        (ref_renders_remote, "reference renders dir"),
    ]:
        if await _log_missing(session, path, tag=tag, label=label):
            return [0.0]

    listing = await session.run_command(
        f'powershell -Command "Get-ChildItem -Path \'{output_dir}\' -Filter *.dwg -Recurse | Select-Object -ExpandProperty FullName"'
    )
    if listing.get("stdout", "").strip():
        logger.error(f"[{tag}] forbidden .dwg files in output: {listing['stdout'][:200]}")
        return [0.0]

    try:
        config_bytes = await session.read_bytes(ref_config_remote)
        variant_config = json.loads(config_bytes.decode("utf-8"))
    except Exception as exc:
        logger.error(f"[{tag}] could not load eval_config: {exc}")
        return [0.0]

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
        candidate_dir=remote_candidate_dir,
        render_units=render_units,
        source_up_axis=source_up_axis,
        cand_res=cand_res,
        cand_samples=cand_samples,
    )

    if not await _wait_for_render_outputs(
        session,
        candidate_dir=remote_candidate_dir,
        view_names=view_names,
    ):
        stderr = await _read_text_if_exists(
            session, _remote_child(remote_scripts_dir, "blender_stderr.txt")
        )
        stdout = await _read_text_if_exists(
            session, _remote_child(remote_scripts_dir, "blender_stdout.txt")
        )
        logger.error(
            f"[{tag}] Blender render timed out or incomplete. stdout={stdout[-2000:]} stderr={stderr[-2000:]}"
        )
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

        score_report = evaluate_renders(
            reference_render_dir=local_ref_renders,
            candidate_render_dir=local_cand_renders,
            config_path=local_config,
        )

    logger.info(f"[{tag}] judge report: {json.dumps(score_report, ensure_ascii=False)[:600]}")
    return [float(score_report["score"])]