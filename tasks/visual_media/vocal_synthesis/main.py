"""Vocal Synthesis — 10 recovered variants.

Reproduce a virtual singer's dry vocal by manually tuning ACE Studio parameters
(voice preset, pitch, vibrato, breath, dynamics, etc.). The source audio may
include effects (reverb, delay, EQ, etc.); the agent must listen through them
and output only the dry vocal.

Evaluation: Mel-Spectrogram MSE mapped to [0, 1] via exp(-alpha * mse).

Variants: hbc, mhahq, wxytdtd, ydc, zzrhxh, soul, chorus,
french, opera, world_music.
"""

import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path

# Workaround: cua_bench loads main.py via exec_module without registering
# in sys.modules, which causes @dataclass to fail. Register ourselves.
if __name__ not in sys.modules:
    sys.modules[__name__] = sys.modules.get(__name__, type(sys)(__name__))

import cua_bench as cb

from tasks.common_config import GeneralTaskConfig
from tasks.common_setup import BaseTaskSetup
from tasks.utils.evaluation import EvaluationContext, llm_vision_judge

_setup = BaseTaskSetup()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DOMAIN_NAME = "visual_media"
TASK_NAME = "vocal_synthesis"

SOURCE_VOCAL = "source_vocal.wav"
REPRODUCED_VOCAL = "reproduced_vocal.wav"
REPRODUCED_VOCAL_GT = "reproduced_vocal_ground_truth.wav"
OVERVIEW_SCREENSHOT = "overview.png"

# Mel-spectrogram parameters
MEL_SR = 22050
MEL_N_FFT = 2048
MEL_HOP_LENGTH = 512
MEL_N_MELS = 128

# MSE-to-score mapping: score = 1 / (1 + ALPHA * mse^2)
# Calibrated so small tuning differences score high, large differences (effects/wrong song) score low
ALPHA = 0.002105

# ---------------------------------------------------------------------------
# Variants — (task_tag,)
# Each variant has its own input/ and reference/ in GCS at:
#   gs://ale-data-all/visual_media/vocal_synthesis/<task_tag>/input/
#   gs://ale-data-all/visual_media/vocal_synthesis/<task_tag>/reference/
# ---------------------------------------------------------------------------
VARIANTS = [
    ("hbc",),
    ("mhahq",),
    ("wxytdtd",),
    ("ydc",),
    ("zzrhxh",),
    ("soul",),
    ("chorus",),
    ("french",),
    ("opera",),
    ("world_music",),
]


REMOTE_SCORE_SCRIPT = Path(__file__).parent / "scripts" / "score_audio_remote.py"


# ---------------------------------------------------------------------------
# Task config
# ---------------------------------------------------------------------------
@dataclass
class TaskConfig(GeneralTaskConfig):
    DOMAIN_NAME: str = "visual_media"

    TASK_NAME: str = "vocal_synthesis"
    VARIANT_NAME: str = ""  # Set per variant

    @property
    def task_dir(self) -> str:
        """Use the canonical benchmark runtime root on Windows data disk."""
        return rf"E:\agenthle\visual_media\vocal_synthesis\{self.VARIANT_NAME}"

    @property
    def input_dir(self) -> str:
        return rf"{self.task_dir}\input"

    @property
    def reproduced_vocal_gt_path(self) -> str:
        return rf"{self.reference_dir}\{REPRODUCED_VOCAL_GT}"

    @property
    def task_description(self) -> str:
        return f"""\
Goal: Act as a human vocal tuner in ACE Studio. You will reproduce the dry vocal from \
a source audio by manually selecting voice presets, entering notes and lyrics, and \
adjusting synthesis parameters (pitch bend, vibrato, breath, dynamics, tension, gender, etc.).

This is NOT voice cloning — you must tune the synthesizer by hand, the same way a human \
vocal tuner would. The source audio may include effects (reverb, delay, EQ, etc.); \
you must listen through the effects and reproduce only the underlying dry vocal. \
Do NOT add any effects to your output.

## Instructions

1. Listen to the source vocal at: {self.input_dir}\\{SOURCE_VOCAL}
   Identify the melody, lyrics, rhythm, and vocal timbre characteristics. \
Note: the source audio may contain multiple vocals (e.g., main vocal and harmonies, \
or different singers), and even a single singer may use different timbres across sections \
(e.g., soft/falsetto vs. powerful/chest voice). You must identify all distinct voices and \
timbre variations, using separate tracks with appropriate voice presets as needed to \
reproduce them all.
2. Open ACE Studio and create a new blank project.
3. Select an appropriate voice preset from the built-in voice library that best matches \
the source vocal's timbre.
4. Enter the melody as MIDI notes in the piano roll, matching pitch and timing from the source.
5. Input the lyrics aligned to the corresponding notes.
6. Adjust vocal parameters — pitch bend curves, vibrato, breath, dynamics, tension, gender, \
etc. — to match the nuances of the source vocal.
7. Render/export the project as dry vocal (no effects) to: {self.remote_output_dir}\\{REPRODUCED_VOCAL}
8. Take a screenshot of the ACE Studio project window showing the piano roll with notes \
and lyrics visible, and save it as: {self.remote_output_dir}\\{OVERVIEW_SCREENSHOT}

## Output files (saved to {self.remote_output_dir}):
- {REPRODUCED_VOCAL}: Your render — the reproduced dry vocal (no effects).
- {OVERVIEW_SCREENSHOT}: Screenshot of the ACE Studio project with piano roll and notes visible.

Export as WAV at 44100 Hz or 48000 Hz sample rate for best evaluation accuracy.
"""

    def to_metadata(self) -> dict:
        metadata = super().to_metadata()
        metadata.update(
            {
                "input_dir": self.input_dir,
                "reproduced_vocal_gt_path": self.reproduced_vocal_gt_path,
            }
        )
        return metadata


# ---------------------------------------------------------------------------
# load
# ---------------------------------------------------------------------------
@cb.tasks_config(split="train")
def load():
    """Register all vocal synthesis variants."""
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
@cb.setup_task(split="train")
async def start(task_cfg, session: cb.DesktopSession):
    await _setup(task_cfg, session)


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------
@cb.evaluate_task(split="train")
async def evaluate(task_cfg, session: cb.DesktopSession) -> list[float]:
    """Score: Gate (WAV exists, >= 50KB, non-silent), then Mel-Spectrogram MSE.

    WAV analysis runs on the remote VM via score_audio_remote.py to avoid
    downloading large audio files.
    """
    try:
        meta = task_cfg.metadata
        tag = meta["variant_name"]
        output_dir = meta["remote_output_dir"]
        ref_dir = meta["reference_dir"]
        reproduced_vocal_gt_path = meta["reproduced_vocal_gt_path"]

        if not (await session.file_exists(ref_dir) or await session.directory_exists(ref_dir)):
            logger.error(f"[{tag}] reference_dir missing: {ref_dir}")
            return [0.0]

        if not (await session.file_exists(output_dir) or await session.directory_exists(output_dir)):
            logger.warning(f"Output directory not found: {output_dir}")
            return [0.0]

        async with EvaluationContext(
            task_tag=tag,
            mode="custom",
            output_dir=None,
            target_path=output_dir,
        ) as ctx:

            # ---------------------------------------------------------------
            # Gate: reproduced_vocal.wav exists
            # ---------------------------------------------------------------
            filepath = os.path.join(output_dir, REPRODUCED_VOCAL)
            if not (await session.file_exists(filepath) or await session.directory_exists(filepath)):
                logger.warning(f"{REPRODUCED_VOCAL} not found — gate fail")
                ctx.log_evaluation(
                    identifier="gate_reproduced",
                    score=0.0,
                    error=f"{REPRODUCED_VOCAL} not found",
                )
                ctx.finalize()
                return [0.0]

            # ---------------------------------------------------------------
            # Gate: screenshot shows ACE Studio project
            # ---------------------------------------------------------------
            screenshot_path = os.path.join(output_dir, OVERVIEW_SCREENSHOT)
            if not (await session.file_exists(screenshot_path) or await session.directory_exists(screenshot_path)):
                logger.warning(f"{OVERVIEW_SCREENSHOT} not found — gate fail")
                ctx.log_evaluation(
                    identifier="gate_screenshot",
                    score=0.0,
                    error=f"{OVERVIEW_SCREENSHOT} not found",
                )
                ctx.finalize()
                return [0.0]

            screenshot_bytes = await session.read_bytes(screenshot_path)

            prompt_screenshot = (
                "You are evaluating a vocal synthesis project screenshot.\n\n"
                "Does this image show an ACE Studio (or similar vocal synthesis DAW) "
                "project window with vocal tracks visible? The view may be a "
                "timeline/arrangement view or a piano roll view.\n"
                "(Look for: vocal track lanes with MIDI note blocks or waveforms, "
                "lyrics/text on tracks, transport bar, timeline with measure numbers)\n"
                'Answer with ONLY "YES" or "NO".'
            )

            eval_screenshot = await llm_vision_judge(
                prompt=prompt_screenshot,
                image_bytes=screenshot_bytes,
                reference_image_bytes=None,
                return_details=True,
                max_tokens=10,
                eval_context=ctx,
                identifier="gate_screenshot",
            )
            if eval_screenshot["score"] == 0.0:
                logger.warning("Screenshot gate failed — does not show ACE Studio project")
                ctx.finalize()
                return [0.0]

            # ---------------------------------------------------------------
            # Gate: ground truth exists
            # ---------------------------------------------------------------
            if not (await session.file_exists(reproduced_vocal_gt_path) or await session.directory_exists(reproduced_vocal_gt_path)):
                logger.error("Reproduced vocal ground truth not found")
                ctx.finalize()
                return [0.0]

            # ---------------------------------------------------------------
            # Remote scoring: upload script, run on VM, pull JSON result
            # ---------------------------------------------------------------
            remote_script = rf"{output_dir}\__score_audio_remote.py"
            remote_result = rf"{output_dir}\__eval_result.json"

            await session.write_bytes(remote_script, REMOTE_SCORE_SCRIPT.read_bytes())

            cmd = (
                f'python "{remote_script}" '
                f'--agent-wav "{filepath}" '
                f'--ref-wav "{reproduced_vocal_gt_path}" '
                f'--result-path "{remote_result}"'
            )
            logger.info(f"[{tag}] Running remote scoring: {cmd}")
            await session.run_command(cmd, timeout=180)

            result_bytes = await session.read_bytes(remote_result)
            result = json.loads(result_bytes)

            if "error" in result:
                logger.error(f"Remote scoring error: {result['error']}")
                ctx.finalize()
                return [0.0]

            # ---------------------------------------------------------------
            # WAV gate checks from remote result
            # ---------------------------------------------------------------
            file_size = result["file_size"]
            rms_db = result["rms_db"]

            if file_size < 51200:
                logger.warning(f"{REPRODUCED_VOCAL} too small ({file_size} B) — gate fail")
                ctx.log_evaluation(
                    identifier="gate_reproduced",
                    score=0.0,
                    error=f"{REPRODUCED_VOCAL} < 50 KB",
                )
                ctx.finalize()
                return [0.0]

            if rms_db <= -60.0:
                logger.warning(f"{REPRODUCED_VOCAL} is silent (RMS={rms_db:.1f} dB) — gate fail")
                ctx.log_evaluation(
                    identifier="gate_reproduced",
                    score=0.0,
                    error=f"{REPRODUCED_VOCAL} silent (RMS={rms_db:.1f} dB)",
                )
                ctx.finalize()
                return [0.0]

            ctx.log_evaluation(identifier="gate_reproduced", score=1.0)

            # ---------------------------------------------------------------
            # Score: Mel-Spectrogram MSE (from remote result)
            # ---------------------------------------------------------------
            mel_mse = result["mel_mse"] if result["mel_mse"] is not None else float("inf")
            score = result["score"]
            ctx.add_score(score)
            ctx.log_evaluation(
                identifier="reproduction_quality",
                score=score,
                mel_mse=round(mel_mse, 6),
            )
            logger.info(f"Reproduction: MSE={mel_mse:.6f} score={score:.4f}")

            # ---------------------------------------------------------------
            # Finalize
            # ---------------------------------------------------------------
            ctx.finalize(mel_mse=round(mel_mse, 6))

            total = ctx.total_score
            logger.info(f"Final score: {total:.4f}")
            return [total]

    except Exception as e:
        logger.error(f"Evaluation error: {e}")

    return [0.0]
