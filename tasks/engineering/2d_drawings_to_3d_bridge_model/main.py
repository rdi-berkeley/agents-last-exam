"""Bridge the Gap — add a bridge to the existing site model.

Single-variant task. Agent receives a design poster + a 3D snapshot + an
existing 3D Site model (terrain + surrounding buildings) and must extend
that site model with a bridge structure shown in the posters. The output
must contain BOTH the unchanged site AND the new bridge. Evaluation
renders 12 canonical views of the agent's OBJ with Blender and scores
them against the frozen reference with an image-only multimodal LLM judge.

Y-up, meters. See `tmp/base/eval_config.json` for the per-instance judge
questions, units, and up-axis.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cua_bench as cb

from tasks.common_config import GeneralTaskConfig
from tasks.common_setup import BaseTaskSetup


_setup = BaseTaskSetup()
logger = logging.getLogger(__name__)

SCRIPTS_DIR = Path(__file__).parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from score_outputs import evaluate_renders  # noqa: E402

DOMAIN_NAME = "engineering"
TASK_NAME = "2d_drawings_to_3d_bridge_model"
VARIANT_NAME = "base"
VARIANT_LABEL = "Bridge the Gap — add a bridge to the existing site model"


@dataclass
class DrawingsTo3DBridgeConfig(GeneralTaskConfig):
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
You are a BIM modeler using Rhino 8 to add a bridge design to an existing
3D site model, given the design posters and the starting site geometry.

This is NOT a build-from-scratch task. The site already exists — you must
place the bridge into it and preserve the rest of the site unchanged.

## Inputs
All input materials are under:
  {self.input_dir}

Required files:
- README.md — natural-language task brief (open this first)
- output_contract.json — machine-readable spec (file names, formats,
  coordinate convention, required geometry components)
- reference_documents/Posters.pdf — design posters with plans, sections,
  axonometric of the bridge
- 3D Snapshot.png — rendered preview of the finished site + bridge
- site_model/3D Model_Site.{{obj,3dm,dwg}} — the EXISTING site you extend

## Software
Open Rhino 8 via:
  {self.software_shortcut}

## What You Must Do
1. Read README.md and output_contract.json.
2. Open site_model/3D Model_Site.3dm in Rhino. It defines the world
   coordinate frame — do not move, rotate, or rescale it.
3. Add the bridge: deck, structural supports (piers/abutments),
   railings/parapet, and a plausible connection to the existing terrain
   at both ends. Use Posters.pdf for form, deck level, support locations.
4. Export to {self.output_obj} (Wavefront OBJ ASCII) and
   {self.output_3dm} (Rhino 5.0+ native). Both must contain the original
   site AND the new bridge geometry.

## Hard Constraints
- Do not output any `.dwg` (evaluator cannot read DWG; forbidden).
- Units = meters. World coordinates inherited from the supplied site model.
- The site geometry in your output must remain visually identical to the
  input site model — the evaluator scores you on that preservation in
  addition to bridge correctness.
- The evaluator will render 12 canonical views from your model.obj and
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
            description=DrawingsTo3DBridgeConfig().task_description,
            metadata=DrawingsTo3DBridgeConfig().to_metadata(),
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


@cb.evaluate_task(split="train")
async def evaluate(task_cfg, session: cb.DesktopSession) -> list[float]:
    """Score the agent's bridge+site model via the shared image-only judge pipeline."""
    meta = task_cfg.metadata
    tag = meta["variant_name"]
    output_dir = meta["remote_output_dir"]
    output_obj = meta["output_obj"]
    output_3dm = meta["output_3dm"]
    ref_config_remote = meta["reference_eval_config"]
    ref_renders_remote = meta["reference_renders_dir"]

    logger.info(f"[{tag}] starting evaluation (output_dir={output_dir})")

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

    render_units = variant_config.get("render_units", "m")
    source_up_axis = variant_config.get("render_source_up_axis", "Y")

    with tempfile.TemporaryDirectory(prefix=f"bridge_{tag}_") as scratch:
        scratch_p = Path(scratch)
        local_obj = scratch_p / "agent_model.obj"
        local_obj.write_bytes(await session.read_bytes(output_obj))

        local_ref_renders = scratch_p / "reference_renders"
        local_ref_renders.mkdir()
        ref_listing = await session.run_command(
            f'powershell -Command "Get-ChildItem -Path \'{ref_renders_remote}\' -Filter *.png | Select-Object -ExpandProperty Name"'
        )
        for name in (ref_listing.get("stdout") or "").splitlines():
            name = name.strip()
            if not name:
                continue
            png_bytes = await session.read_bytes(rf"{ref_renders_remote}\{name}")
            (local_ref_renders / name).write_bytes(png_bytes)

        local_config = scratch_p / "eval_config.json"
        local_config.write_bytes(config_bytes)

        cand_dir = scratch_p / "candidate_renders"
        cand_dir.mkdir()
        blender = os.environ.get("BLENDER_BIN", str(Path.home() / "blender" / "blender-4.2.4-linux-x64" / "blender"))
        render_script = SCRIPTS_DIR / "render_human_views.py"
        # Candidate must render at the SAME settings as the frozen reference so
        # both reach the judge as pixel-comparable images. The judge downsamples
        # everything to <=1024 anyway (score_outputs._JUDGE_MAX_EDGE), so we
        # render natively at 1024 instead of supersampling from 2048 — the judge
        # sees the same resolution either way, and native 1024 is ~4x faster
        # (critical for this 1.5M-vertex ArchiCAD mesh). The frozen reference is
        # rendered at these same 1024/32 settings. Env vars override for tuning.
        cand_res = int(os.environ.get("CAND_RES", "1024"))
        cand_samples = int(os.environ.get("CAND_SAMPLES", "32"))
        cmd = [
            blender, "--background", "--python", str(render_script), "--",
            "--obj", str(local_obj),
            "--out", str(cand_dir),
            "--units", render_units,
            "--source-up-axis", source_up_axis,
            "--res", str(cand_res),
            "--samples", str(cand_samples),
        ]
        import subprocess
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        if result.returncode != 0:
            logger.error(f"[{tag}] Blender render failed: {result.stderr[-2000:]}")
            return [0.0]

        score_report = evaluate_renders(
            reference_render_dir=local_ref_renders,
            candidate_render_dir=cand_dir,
            config_path=local_config,
        )

    logger.info(f"[{tag}] judge report: {json.dumps(score_report, ensure_ascii=False)[:600]}")
    return [float(score_report["score"])]
