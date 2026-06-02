"""mold-flow — AgentHLE Task: Moldex3D Injection Molding Simulation

The agent is given a Moldex3D project (mesh + material pre-configured), a process
specification JSON, and a results template JSON. The agent must:
  1. Configure process parameters in Moldex3D according to the spec
  2. Run the full analysis (Fill → Pack → Cool → Warp)
  3. Extract simulation results and fill in the template

Variants: 12 injection molding projects with identical evaluation logic.
"""

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import cua_bench as cb
from tasks.common_config import GeneralTaskConfig
from tasks.common_setup import BaseTaskSetup


_setup = BaseTaskSetup()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Variants — one tuple per task instance
# Format: (task_tag, moldex3d_project_folder_name, has_cad_reference)
# ---------------------------------------------------------------------------
VARIANTS = [
    # Existing 3 variants
    ("230057", "4穴230057", True),
    ("220089", "吸嘴电子烟16穴（220089）", True),
    ("240167", "SH14U(拨动开关)(Exported)", True),
    # New 9 variants (added 2026-03-25)
    ("MDB230048", "MDB230048", False),
    ("MDB230058", "MDB230058", False),
    ("MDB230113", "MDB230113", False),
    ("MDB230114", "MDB230114", False),
    ("MDB230159", "V12(发热顶盖)(8穴", False),
    ("MDB230160", "MDB230160", False),
    ("MDB230161", "MDB230161", False),
    ("MDB230177", "MDB230177", False),
    ("MDB240023", "MDB240023", False),
]


#################################################################
############################# Config ############################
#################################################################


@dataclass
class MoldFlowTaskConfig(GeneralTaskConfig):
    """Configuration for a single Moldex3D mold-flow analysis task."""

    DOMAIN_NAME: str = "engineering"

    TASK_NAME: str = "mold-flow"
    VARIANT_NAME: str = ""
    MDX_PROJECT_NAME: str = ""
    HAS_CAD_REFERENCE: bool = False

    @property
    def input_dir(self) -> str:
        return rf"{self.task_dir}\input"

    @property
    def input_project(self) -> str:
        """The clean Moldex3D project in input/ (mesh+material, no process/analysis)."""
        return rf"{self.input_dir}\{self.MDX_PROJECT_NAME}"

    @property
    def software_shortcut(self) -> str:
        return rf"{self.software_dir}\MDXStudio.lnk"

    @property
    def output_project(self) -> str:
        """The Moldex3D project copied to output/ (agent works here)."""
        return rf"{self.remote_output_dir}\{self.MDX_PROJECT_NAME}"

    @property
    def process_spec_file(self) -> str:
        """Process specification JSON that the agent must follow."""
        return rf"{self.input_dir}\process_spec.json"

    @property
    def results_template_file(self) -> str:
        """JSON template with null fields for the agent to fill."""
        return rf"{self.input_dir}\results_template.json"

    @property
    def agent_results_file(self) -> str:
        """Agent's filled results JSON in output/."""
        return rf"{self.remote_output_dir}\results.json"

    @property
    def reference_results_file(self) -> str:
        """Ground-truth results JSON in reference/."""
        return rf"{self.reference_dir}\results.json"

    @property
    def reference_process_file(self) -> str:
        """Ground-truth process parameters in reference/ for optional verification."""
        return rf"{self.reference_dir}\process_reference.json"

    @property
    def cad_file(self) -> str:
        """Original part geometry (.x_t Parasolid) for reference."""
        if not self.HAS_CAD_REFERENCE:
            return ""
        return rf"{self.input_dir}\{self.VARIANT_NAME}.x_t"

    @property
    def task_description(self) -> str:
        cad_input = (
            f"- **Part geometry** (.x_t Parasolid): `{self.cad_file}` (for visual reference only)\n"
            if self.HAS_CAD_REFERENCE
            else ""
        )
        cad_step = (
            "4. Use the optional `.x_t` CAD file as a visual cross-check if helpful\n"
            if self.HAS_CAD_REFERENCE
            else ""
        )
        return f"""\
You are a CAE (Computer-Aided Engineering) engineer performing injection molding simulation using Moldex3D.

## Your Task
Configure process parameters and run a mold-flow analysis for project **{self.VARIANT_NAME}** using the Moldex3D software.

## Input Files
- **Moldex3D project** (mesh + material already configured, NO process parameters or analysis results):
  `{self.input_project}`
- **Process specification**: `{self.process_spec_file}`
  This JSON file contains all the injection molding parameters you must configure.
- **Results template**: `{self.results_template_file}`
  This JSON file has fields with null values that you must reproduce in `{self.agent_results_file}`.
{cad_input}

## What You Must Do
1. Open Moldex3D from `{self.software_shortcut}` and load the project at:
   `{self.input_project}`
2. Read the process specification JSON carefully
3. Configure ALL process parameters in Moldex3D's Process Wizard:
   - Melt temperature, mold temperature
   - Injection flow rate profile
   - V/P switch point (volume filled %)
   - Packing pressure profile and time
   - Cooling time, coolant temperature, coolant flow rate
4. Set up the analysis sequence: **Fill → Pack → Cool (Transient) → Warp**
{cad_step}5. Run the full analysis (this may take 1-2 hours)
6. After analysis completes, extract the simulation results from the Moldex3D result viewer
7. Create `{self.agent_results_file}` using the same JSON structure as `{self.results_template_file}` and fill in all null fields with the simulation values
8. Save any working project artifacts under `{self.remote_output_dir}` if you choose to create a writable copy

## Results to Extract
After analysis completes, fill in these fields in results.json:
- **Max injection pressure** (MPa)
- **V/P switch pressure** (MPa)
- **Runner pressure drop** (MPa)
- **Max clamping force** (ton)
- **Filling time** (sec)
- **Estimated cooling time** (sec)
- **Cycle time** (sec)
- **Part weight** per cavity (g)
- **Part volume** per cavity (cc)

## Evaluation
Your work will be evaluated by comparing your results.json against the reference values.
Each field within ±1% tolerance of the reference scores 1 point.
Final score = fraction of correct fields.

## Important
- Treat `{self.input_project}` as the staged clean baseline
- Do not modify or rename the staged input project in place
- SAVE your project when finished
- The results template at `{self.results_template_file}` shows the exact JSON structure expected
"""

    def to_metadata(self) -> dict:
        metadata = super().to_metadata()
        metadata.update(
            {
                "input_dir": self.input_dir,
                "mdx_project_name": self.MDX_PROJECT_NAME,
                "software_shortcut": self.software_shortcut,
                "input_project": self.input_project,
                "output_project": self.output_project,
                "process_spec_file": self.process_spec_file,
                "results_template_file": self.results_template_file,
                "agent_results_file": self.agent_results_file,
                "reference_results_file": self.reference_results_file,
                "reference_process_file": self.reference_process_file,
                "cad_file": self.cad_file,
                "has_cad_reference": self.HAS_CAD_REFERENCE,
            }
        )
        return metadata


# ---------------------------------------------------------------------------
# Local scripts directory
# ---------------------------------------------------------------------------
SCRIPTS_DIR = Path(__file__).parent / "scripts"


def _read_script(name: str) -> str:
    """Read a script file from the local scripts/ directory."""
    return (SCRIPTS_DIR / name).read_text(encoding="utf-8")


async def _log_missing_path(
    session: cb.DesktopSession,
    path: str,
    *,
    tag: str,
    label: str,
) -> bool:
    if (await session.file_exists(path) or await session.directory_exists(path)):
        return False
    logger.error("[%s] Missing %s: %s", tag, label, path)
    return True


#################################################################
###################### Task Registration ########################
#################################################################


@cb.tasks_config(split="train")
def load():
    """Register all Moldex3D mold-flow task variants."""
    return [
        cb.Task(
            description=MoldFlowTaskConfig(
                VARIANT_NAME=tag,
                MDX_PROJECT_NAME=project_name,
                HAS_CAD_REFERENCE=has_cad_reference,
            ).task_description,
            metadata=MoldFlowTaskConfig(
                VARIANT_NAME=tag,
                MDX_PROJECT_NAME=project_name,
                HAS_CAD_REFERENCE=has_cad_reference,
            ).to_metadata(),
            computer={
                "provider": "computer",
                "setup_config": {"os_type": "windows"},
            },
        )
        for tag, project_name, has_cad_reference in VARIANTS
    ]


#################################################################
######################## Initialization #########################
#################################################################


@cb.setup_task(split="train")
async def start(task_cfg, session: cb.DesktopSession):
    await _setup(task_cfg, session)


#################################################################
########################## Evaluation ###########################
#################################################################


@cb.evaluate_task(split="train")
async def evaluate(task_cfg, session: cb.DesktopSession) -> list[float]:
    """Score the agent's mold-flow analysis results.

    Pipeline:
      1. Upload evaluation scripts to a temp location on the VM
      2. Gate check: does results.json exist and is it valid?
      3. Primary score: compare results.json vs reference
      4. (Optional) Secondary: compare .pro process config
      5. Return weighted score

    Test mode:
      If results.json already exists in the output dir, skip to scoring.
    """
    meta = task_cfg.metadata
    task_tag = meta["variant_name"]
    output_dir = meta["remote_output_dir"]
    agent_results = meta["agent_results_file"]
    ref_results = meta["reference_results_file"]
    ref_process = meta["reference_process_file"]
    output_project = meta["output_project"]

    logger.info(f"[{task_tag}] Starting evaluation (output_dir={output_dir})")

    if await _log_missing_path(
        session,
        ref_results,
        tag=task_tag,
        label="reference results JSON",
    ):
        return [0.0]

    # ── 0. Upload evaluation scripts ──────────────────────────────────────────
    tmp_scripts = r"C:\Users\User\AppData\Local\Temp\mold_flow_eval_scripts"
    await session.interface.create_dir(tmp_scripts)

    for script_name in ["verify_results.py", "verify_process.py"]:
        script_content = _read_script(script_name)
        remote_path = rf"{tmp_scripts}\{script_name}"
        await session.write_file(remote_path, script_content)

    logger.info(f"[{task_tag}] Evaluation scripts uploaded to {tmp_scripts}")

    # ── 1. Gate check: does results.json exist? ───────────────────────────────
    if not (await session.file_exists(agent_results) or await session.directory_exists(agent_results)):
        logger.error(f"[{task_tag}] results.json not found at {agent_results}")
        return [0.0]

    # ── 2. Primary score: compare results.json vs reference ───────────────────
    logger.info(f"[{task_tag}] Scoring: Comparing results vs reference...")

    verify_script = rf"{tmp_scripts}\verify_results.py"
    result = await session.run_command(
        f'python "{verify_script}" --agent "{agent_results}" --ref "{ref_results}"',
    )

    score = 0.0
    try:
        verify_data = json.loads(result["stdout"])
        score = float(verify_data.get("score", 0.0))
        logger.info(
            f"[{task_tag}] Results score={score:.4f}, "
            f"matched={verify_data.get('matched_fields', '?')}/{verify_data.get('total_fields', '?')}, "
            f"details={verify_data.get('field_details', {})}"
        )
    except Exception as e:
        logger.error(f"[{task_tag}] Could not parse verify output: {e}")
        logger.error(f"[{task_tag}] stdout: {result.get('stdout', '')[:500]}")
        logger.error(f"[{task_tag}] stderr: {result.get('stderr', '')[:500]}")

    # ── 3. Optional: process config check for partial credit ──────────────────
    if score < 0.5 and (await session.file_exists(ref_process) or await session.directory_exists(ref_process)):
        logger.info(f"[{task_tag}] Low score — checking process config for partial credit...")
        process_script = rf"{tmp_scripts}\verify_process.py"

        # Find the .pro file in the output project
        pro_file = ""
        try:
            process_dir = rf"{output_project}\Process"
            if (await session.file_exists(process_dir) or await session.directory_exists(process_dir)):
                files = await session.list_dir(process_dir)
                pro_files = [f for f in files if f.endswith(".pro")]
                if pro_files:
                    pro_file = rf"{process_dir}\{pro_files[0]}"
        except Exception as e:
            logger.warning(f"[{task_tag}] Could not find .pro file: {e}")

        if pro_file:
            result = await session.run_command(
                f'python "{process_script}" --agent-pro "{pro_file}" --ref "{ref_process}"',
            )
            try:
                process_data = json.loads(result["stdout"])
                process_score = float(process_data.get("score", 0.0))
                logger.info(f"[{task_tag}] Process config score={process_score:.4f}")
                # Give partial credit: 30% weight to process config if results score is low
                if process_score > 0:
                    score = max(score, 0.3 * process_score)
                    logger.info(f"[{task_tag}] Adjusted score with process credit: {score:.4f}")
            except Exception as e:
                logger.warning(f"[{task_tag}] Could not parse process output: {e}")

    logger.info(f"[{task_tag}] Final score: {score:.4f}")
    return [score]
