"""Project Migration — 10 variants.

Given a Cubase project with missing/invalid VST plugins, replace all missing
VSTs with functionally equivalent ones from the available plugins on the
target system, then export per-track stems and a full mixdown.

Evaluation: Stem completeness & quality (0.20) + Timbral similarity (0.80).
"""

import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path

if __name__ not in sys.modules:
    sys.modules[__name__] = sys.modules.get(__name__, type(sys)(__name__))

import cua_bench as cb
from tasks.common_config import GeneralTaskConfig
from tasks.common_setup import BaseTaskSetup
from utils.evaluation import EvaluationContext, llm_vision_judge

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROJECT_FILE = "project.cpr"
AVAILABLE_VSTS = "available_vsts.txt"
MIGRATED_PROJECT = "migrated_project.cpr"
STEMS_DIR = "stems"
OVERVIEW_SCREENSHOT = "overview.png"

# Scoring weights
W_STEM_QUALITY = 0.20
W_TIMBRE = 0.80

# MFCC cosine similarity threshold: >= this -> full credit
TIMBRE_THRESHOLD = 0.70

# ---------------------------------------------------------------------------
# Variants — (task_tag,)
# Each variant has its own input/ and reference/ in GCS at:
#   gs://ale-data-all/visual_media/project_migration/<task_tag>/input/
#   gs://ale-data-all/visual_media/project_migration/<task_tag>/reference/
# ---------------------------------------------------------------------------
#   Stage 2 decision: only the 5 variants with fully exported reference stems
#   are registered here. The 5 pending variants (freehold_battle_music,
#   geralt_of_rivia, kingdoms_will_burn, pokemon_go_medley,
#   undertale_symphonic_suite) lack reference data and are deferred until
#   their original VSTs can be sourced to re-render reference stems.
VARIANTS = [
    ("celeste_symphonic_suite",),
    ("eora",),
    ("hollow_knight_symphonic_suite",),
    ("twilight_princess_credits",),
    ("undertale_medley",),
]


REMOTE_SCORE_SCRIPT = Path(__file__).parent / "scripts" / "score_audio_remote.py"


# ---------------------------------------------------------------------------
# Task config
# ---------------------------------------------------------------------------
@dataclass
class TaskConfig(GeneralTaskConfig):
    DOMAIN_NAME: str = "visual_media"

    TASK_NAME: str = "project_migration"
    VARIANT_NAME: str = ""

    @property
    def input_dir(self) -> str:
        return rf"{self.task_dir}\input"

    @property
    def reference_stems_dir(self) -> str:
        return rf"{self.reference_dir}\{STEMS_DIR}"

    @property
    def task_description(self) -> str:
        return f"""\
Goal: Migrate a Cubase project to this computer by replacing all missing/invalid VST \
plugins with equivalent ones from the available plugins, then export stems and a mixdown.

Software:
- Cubase: {self.software_dir}\\Cubase.lnk

Input:
- Cubase project: {self.input_dir}\\{PROJECT_FILE}

Before you begin:
1. Generate the list of VST3 plugins installed on this VM and save it to \
`{self.remote_output_dir}\\{AVAILABLE_VSTS}`. In PowerShell:
   `$roots = @('C:\\Program Files\\Common Files\\VST3','C:\\Program Files (x86)\\Common Files\\VST3'); \
$items = @(); foreach ($r in $roots) {{ if (Test-Path $r) {{ $items += Get-ChildItem -Path $r -Recurse -Filter *.vst3 -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Name }} }}; \
($items | Sort-Object -Unique) | Out-File -FilePath '{self.remote_output_dir}\\{AVAILABLE_VSTS}' -Encoding ascii`
2. Open the Cubase project at `{self.input_dir}\\{PROJECT_FILE}` using the Cubase shortcut \
at `{self.software_dir}\\Cubase.lnk` (e.g. double-click the .lnk, or \
`Start-Process '{self.software_dir}\\Cubase.lnk'`).

Steps:
1. When the "Missing Plugins" dialog appears, note all unavailable VST plugins.
2. For each track with a missing VST instrument or effect:
   a. Open the track's instrument or insert slot.
   b. Replace the missing VST with a functionally equivalent one from the available \
plugins listed in `{self.remote_output_dir}\\{AVAILABLE_VSTS}`.
   c. Choose a preset or patch that best approximates the original sound character \
(e.g., replace a missing string library with an available string patch).
3. Play back the project and verify all tracks produce audible, artifact-free audio.
4. Export stems: for each track/channel as organized in the project (do NOT split or \
reorganize tracks — export exactly as the project defines them), solo it and export \
audio mixdown (WAV, 44.1 kHz or higher) to {self.remote_output_dir}\\{STEMS_DIR}\\<track_name>.wav
5. Save the migrated project to {self.remote_output_dir}\\{MIGRATED_PROJECT}
6. Take a screenshot of the Cubase MixConsole or arrange view showing all tracks with \
their replaced plugins visible:
   save_milestone_screenshot(path="{self.remote_output_dir}\\{OVERVIEW_SCREENSHOT}", \
description="Cubase project with replaced plugins")

Output files (all saved to {self.remote_output_dir}):
- {MIGRATED_PROJECT}: Cubase project with all missing VSTs replaced.
- {STEMS_DIR}/: Per-track solo WAV exports (one per track).
- {OVERVIEW_SCREENSHOT}: Screenshot of MixConsole or arrange view.
"""

    def to_metadata(self) -> dict:
        metadata = super().to_metadata()
        metadata.pop("software_dir", None)
        metadata.update(
            {
                "input_dir": self.input_dir,
                "reference_stems_dir": self.reference_stems_dir,
            }
        )
        return metadata


# ---------------------------------------------------------------------------
# load
# ---------------------------------------------------------------------------
@cb.tasks_config(split="train")
def load():
    """Register all project migration variants."""
    tasks = []
    for (tag,) in VARIANTS:
        cfg = TaskConfig(VARIANT_NAME=tag)
        tasks.append(
            cb.Task(
                description=cfg.task_description,
                metadata=cfg.to_metadata(),
                computer={
                    "provider": "computer",
                    "setup_config": {"os_type": cfg.OS_TYPE},
                },
            )
        )
    return tasks


# ---------------------------------------------------------------------------
# start
# ---------------------------------------------------------------------------
_setup = BaseTaskSetup()


@cb.setup_task(split="train")
async def start(task_cfg, session: cb.DesktopSession):
    await _setup(task_cfg, session)


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------
@cb.evaluate_task(split="train")
async def evaluate(task_cfg, session: cb.DesktopSession) -> list[float]:
    """Score: Gates (project + stems + screenshot),
    then stem completeness & quality (0.20) + timbral similarity (0.80).

    WAV analysis runs on the remote VM via score_audio_remote.py to avoid
    downloading large audio files.
    """
    try:
        meta = task_cfg.metadata
        tag = meta["variant_name"]
        output_dir = meta["remote_output_dir"]
        ref_dir = meta["reference_dir"]
        reference_stems_dir = meta["reference_stems_dir"]

        if not await session.exists(ref_dir):
            logger.error(f"[{tag}] reference_dir missing: {ref_dir}")
            return [0.0]

        if not await session.exists(output_dir):
            logger.warning(f"Output directory not found: {output_dir}")
            return [0.0]

        output_files = await session.list_dir(output_dir)

        async with EvaluationContext(
            task_tag=tag,
            mode="custom",
            output_dir=None,
            target_path=output_dir,
        ) as ctx:

            # -----------------------------------------------------------
            # Gate 1: migrated_project.cpr exists
            # -----------------------------------------------------------
            if MIGRATED_PROJECT not in output_files:
                logger.warning(f"{MIGRATED_PROJECT} not found — gate fail")
                ctx.log_evaluation(
                    identifier="gate_project",
                    score=0.0,
                    error=f"{MIGRATED_PROJECT} not found",
                )
                ctx.finalize(num_output_files=len(output_files))
                return [0.0]

            ctx.log_evaluation(identifier="gate_project", score=1.0)

            # -----------------------------------------------------------
            # Gate 2: stems/ directory exists with WAV files
            # -----------------------------------------------------------
            stems_output_dir = os.path.join(output_dir, STEMS_DIR)
            if not await session.exists(stems_output_dir):
                logger.warning("stems/ directory not found — gate fail")
                ctx.log_evaluation(
                    identifier="gate_stems",
                    score=0.0,
                    error="stems/ directory not found",
                )
                ctx.finalize(num_output_files=len(output_files))
                return [0.0]

            stem_files_list = await session.list_dir(stems_output_dir)
            wav_stems = [f for f in stem_files_list if f.lower().endswith(".wav")]
            if not wav_stems:
                logger.warning("No WAV stems found — gate fail")
                ctx.log_evaluation(
                    identifier="gate_stems",
                    score=0.0,
                    error="No WAV files in stems/",
                )
                ctx.finalize(num_output_files=len(output_files))
                return [0.0]

            # -----------------------------------------------------------
            # Gate 3: screenshot shows Cubase project
            # -----------------------------------------------------------
            if OVERVIEW_SCREENSHOT not in output_files:
                logger.warning(f"{OVERVIEW_SCREENSHOT} not found — gate fail")
                ctx.log_evaluation(
                    identifier="gate_screenshot",
                    score=0.0,
                    error="Screenshot not found",
                )
                ctx.finalize(num_output_files=len(output_files))
                return [0.0]

            screenshot_path = os.path.join(output_dir, OVERVIEW_SCREENSHOT)
            screenshot_bytes = await session.read_bytes(screenshot_path)

            prompt_gate = (
                "You are evaluating a DAW (Digital Audio Workstation) screenshot.\n\n"
                "Does this image show a Cubase project (MixConsole or arrange view) "
                "with audio/instrument tracks visible and VST plugin assignments "
                "or instrument names shown?\n"
                "(Look for: track list, instrument/plugin names, mixer channels, "
                "transport bar, Cubase UI elements)\n"
                'Answer with ONLY "YES" or "NO".'
            )

            eval_gate = await llm_vision_judge(
                prompt=prompt_gate,
                image_bytes=screenshot_bytes,
                reference_image_bytes=None,
                return_details=True,
                max_tokens=10,
                eval_context=ctx,
                identifier="gate_screenshot",
            )
            if eval_gate["score"] == 0.0:
                logger.warning("Screenshot gate failed — does not show Cubase project")
                ctx.finalize(num_output_files=len(output_files))
                return [0.0]

            # -----------------------------------------------------------
            # Gate: reference stems exist
            # -----------------------------------------------------------
            if not await session.exists(reference_stems_dir):
                logger.error("Reference stems directory not found — cannot score")
                ctx.finalize(num_output_files=len(output_files))
                return [0.0]

            # -----------------------------------------------------------
            # Remote scoring: upload script, run on VM, pull JSON result
            # -----------------------------------------------------------
            remote_script = rf"{output_dir}\__score_audio_remote.py"
            remote_result = rf"{output_dir}\__eval_result.json"

            await session.write_bytes(remote_script, REMOTE_SCORE_SCRIPT.read_bytes())

            cmd = (
                f'python "{remote_script}" '
                f'--agent-stems-dir "{stems_output_dir}" '
                f'--ref-stems-dir "{reference_stems_dir}" '
                f'--result-path "{remote_result}"'
            )
            logger.info(f"[{tag}] Running remote scoring: {cmd}")
            await session.run_command(cmd, timeout=300)

            result_bytes = await session.read_bytes(remote_result)
            result = json.loads(result_bytes)

            if "error" in result:
                logger.error(f"Remote scoring error: {result['error']}")
                ctx.finalize(num_output_files=len(output_files))
                return [0.0]

            # -----------------------------------------------------------
            # Gate check: non-silent stems from remote result
            # -----------------------------------------------------------
            num_non_silent = result["num_non_silent"]
            if num_non_silent == 0:
                logger.warning("No non-silent stems found — gate fail")
                ctx.log_evaluation(
                    identifier="gate_stems",
                    score=0.0,
                    error="No non-silent WAV stems",
                )
                ctx.finalize(num_output_files=len(output_files))
                return [0.0]

            ctx.log_evaluation(
                identifier="gate_stems",
                score=1.0,
                num_stems=num_non_silent,
            )

            # -----------------------------------------------------------
            # Score 1: Stem completeness & quality (weight 0.20)
            # -----------------------------------------------------------
            num_valid = result["num_valid"]
            num_expected = result["num_expected"]

            stem_score_raw = num_valid / num_expected if num_expected > 0 else 0.0
            stem_score = W_STEM_QUALITY * stem_score_raw
            ctx.add_score(stem_score)
            ctx.log_evaluation(
                identifier="stem_completeness_quality",
                score=stem_score,
                valid_stems=num_valid,
                expected_stems=num_expected,
                raw_score=round(stem_score_raw, 4),
            )
            logger.info(
                f"Stem completeness & quality: {num_valid}/{num_expected} "
                f"= {stem_score_raw:.4f} (weighted: {stem_score:.4f})"
            )

            # -----------------------------------------------------------
            # Score 2: Timbral similarity (weight 0.80)
            # -----------------------------------------------------------
            avg_timbre = result["avg_timbre"]
            timbre_score = W_TIMBRE * avg_timbre
            ctx.add_score(timbre_score)
            ctx.log_evaluation(
                identifier="timbral_similarity",
                score=timbre_score,
                avg_similarity=round(avg_timbre, 4),
                num_pairs=len(result["matches"]),
            )
            logger.info(
                f"Timbral similarity: avg={avg_timbre:.4f} " f"(weighted: {timbre_score:.4f})"
            )

            for m in result["matches"]:
                if m["agent"] is None:
                    logger.info(f"  Timbre: ref='{m['ref']}' — no match")
                else:
                    logger.info(
                        f"  Timbre: ref='{m['ref']}' agent='{m['agent']}' "
                        f"similarity={m.get('timbre_similarity', 0):.4f}"
                    )

            # -----------------------------------------------------------
            # Finalize
            # -----------------------------------------------------------
            ctx.finalize(
                num_agent_stems=num_non_silent,
                num_ref_stems=num_expected,
                num_output_files=len(output_files),
            )

            total = ctx.total_score
            logger.info(
                f"Final score: {total:.4f} " f"(stem={stem_score:.3f} timbre={timbre_score:.3f})"
            )
            return [total]

    except Exception as e:
        logger.error(f"Evaluation error: {e}")

    return [0.0]
