"""Timbre Synthesis — 10 variants.

Given an audio clip of a synth patch, reverse-engineer the timbre by programming
a software synthesizer from scratch, then render a specified chord progression.
No MIDI files are used; evaluation is purely audio-domain.

Evaluation: Mel-Spectrogram MSE (0.90) + LLM screenshot judge (0.10).
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
from tasks.utils.evaluation import EvaluationContext, llm_vision_judge

_setup = BaseTaskSetup()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
REFERENCE_PATCH = "reference_patch.wav"
TASK_BRIEF = "task_brief.json"
RENDERED_CHORDS = "rendered_chords.wav"
GROUND_TRUTH = "ground_truth.wav"
OVERVIEW_SCREENSHOT = "overview.png"

# Mel-spectrogram parameters
MEL_SR = 22050
MEL_N_FFT = 2048
MEL_HOP_LENGTH = 512
MEL_N_MELS = 128

# MSE-to-score mapping: score = 1 / (1 + ALPHA * mse^2)
ALPHA = 0.002105

# Scoring weights
W_MEL = 0.90
W_SCREENSHOT = 0.10

# ---------------------------------------------------------------------------
# Variants — (task_tag,)
# Each variant has its own input/ and reference/ in GCS at:
#   gs://ale-data-all/visual_media/timbre_synthesis/<task_tag>/input/
#   gs://ale-data-all/visual_media/timbre_synthesis/<task_tag>/reference/
# ---------------------------------------------------------------------------
VARIANTS = [
    ("analog_dream_1",),
    ("ethernal_earth_1",),
    ("straylight_1",),
    ("straylight_2",),
    ("pharlight_1",),
    ("pharlight_2",),
    ("massive_x_1",),
    ("massive_x_2",),
    ("retrologue_1",),
    ("padshop_1",),
]


REMOTE_SCORE_SCRIPT = Path(__file__).parent / "scripts" / "score_audio_remote.py"


# ---------------------------------------------------------------------------
# Task config
# ---------------------------------------------------------------------------
@dataclass
class TaskConfig(GeneralTaskConfig):
    DOMAIN_NAME: str = "visual_media"

    TASK_NAME: str = "timbre_synthesis"
    VARIANT_NAME: str = ""

    @property
    def input_dir(self) -> str:
        return rf"{self.task_dir}\input"

    @property
    def ground_truth_path(self) -> str:
        return rf"{self.reference_dir}\{GROUND_TRUTH}"

    @property
    def task_description(self) -> str:
        return f"""\
Goal: Reverse-engineer a synthesizer patch from a reference audio clip, then use it \
to render a specified chord progression.

You will listen to a reference audio clip that demonstrates a specific synthesizer timbre \
(e.g., a pad, lead, pluck, or bass sound). Your task is to recreate that timbre by \
programming a software synthesizer from scratch — adjusting oscillators, filters, \
envelopes, LFOs, effects, and modulation parameters until the sound closely matches \
the reference. Then render a chord progression specified in a JSON file.

No MIDI files are involved. You must input notes directly into a DAW piano roll and \
export audio only.

## Instructions

1. Listen to the reference patch audio at: {self.input_dir}\\{REFERENCE_PATCH}
   Identify the timbral characteristics: waveform type, filter type/cutoff/resonance, \
envelope shape (attack, decay, sustain, release), modulation, effects (reverb, delay, \
chorus, etc.).
2. Read the chord progression specification at: {self.input_dir}\\{TASK_BRIEF}
   This JSON file contains a list of chords, each with a chord symbol and exact note \
names (with octaves). All chords use the same fixed parameters: tempo = 120 BPM, \
time signature = 4/4, duration = 1 bar per chord.
3. Open the DAW using the canonical entry point staged under this variant's task directory: {self.task_dir}\\software\\launch_cubase.bat — this launches Cubase 15 Pro. From inside Cubase, pick any available software synthesizer (HALion Sonic, Retrologue, Padshop, or any Native Instruments VST host loaded as a VST3).
4. Program a patch from the default/init state to approximate the reference timbre. \
You are free to choose whichever synthesizer best suits the sound.
5. Input the chord progression into the DAW piano roll at 120 BPM, 4/4 time, \
with each chord lasting exactly 1 bar. Use the exact note names specified in the JSON.
6. Render/export the result as: {self.remote_output_dir}\\{RENDERED_CHORDS}
7. Take an "overview" screenshot of your DAW project (arrangement view with the \
tracks and clips you used, piano roll with notes entered, or the synthesizer plugin \
UI with its configured parameters) and save as: \
{self.remote_output_dir}\\{OVERVIEW_SCREENSHOT}

## Output files (saved to {self.remote_output_dir}):
- {RENDERED_CHORDS}: The chord progression rendered with your recreated patch (stereo WAV).
- {OVERVIEW_SCREENSHOT}: Screenshot of your DAW project or the synthesizer plugin UI.

"""

    def to_metadata(self) -> dict:
        metadata = super().to_metadata()
        metadata.update(
            {
                "input_dir": self.input_dir,
                "ground_truth_path": self.ground_truth_path,
            }
        )
        return metadata


# ---------------------------------------------------------------------------
# load
# ---------------------------------------------------------------------------
@cb.tasks_config(split="train")
def load():
    """Register all timbre synthesis variants."""
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
    """Score: Gates (WAV + screenshot) then weighted Mel-MSE + screenshot quality.

    WAV analysis runs on the remote VM via score_audio_remote.py to avoid
    downloading large audio files.
    """
    try:
        meta = task_cfg.metadata
        tag = meta["variant_name"]
        output_dir = meta["remote_output_dir"]
        ground_truth_path = meta["ground_truth_path"]

        if not await session.exists(output_dir):
            logger.warning(f"Output directory not found: {output_dir}")
            return [0.0]

        async with EvaluationContext(
            task_tag=tag,
            mode="custom",
            output_dir=None,
            target_path=output_dir,
        ) as ctx:

            # ---------------------------------------------------------------
            # Gate 1: rendered_chords.wav exists
            # ---------------------------------------------------------------
            filepath = os.path.join(output_dir, RENDERED_CHORDS)
            if not await session.exists(filepath):
                logger.warning(f"{RENDERED_CHORDS} not found — gate fail")
                ctx.log_evaluation(
                    identifier="gate_audio",
                    score=0.0,
                    error=f"{RENDERED_CHORDS} not found",
                )
                ctx.finalize()
                return [0.0]

            # ---------------------------------------------------------------
            # Gate 2: screenshot exists and shows synthesizer UI
            # ---------------------------------------------------------------
            screenshot_path = os.path.join(output_dir, OVERVIEW_SCREENSHOT)
            if not await session.exists(screenshot_path):
                logger.warning(f"{OVERVIEW_SCREENSHOT} not found — gate fail")
                ctx.log_evaluation(
                    identifier="gate_screenshot",
                    score=0.0,
                    error=f"{OVERVIEW_SCREENSHOT} not found",
                )
                ctx.finalize()
                return [0.0]

            screenshot_bytes = await session.read_bytes(screenshot_path)

            prompt_gate = (
                "You are evaluating a music-production workflow screenshot.\n\n"
                "Does this image show either (a) a software synthesizer / VST "
                "plugin interface with visible parameters such as oscillators, "
                "filters, envelopes, LFOs, or effects, OR (b) a DAW "
                "(e.g., Cubase, Ableton, FL Studio, Logic, Reaper) arrangement / "
                "piano-roll / mixer view that contains one or more tracks with "
                "audio or MIDI clips?\n"
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
                logger.warning("Screenshot gate failed — does not show synthesizer interface")
                ctx.finalize()
                return [0.0]

            # ---------------------------------------------------------------
            # Gate: ground truth exists
            # ---------------------------------------------------------------
            if not await session.exists(ground_truth_path):
                logger.error("Ground truth not found")
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
                f'--ref-wav "{ground_truth_path}" '
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
                logger.warning(f"{RENDERED_CHORDS} too small ({file_size} B) — gate fail")
                ctx.log_evaluation(
                    identifier="gate_audio",
                    score=0.0,
                    error=f"{RENDERED_CHORDS} < 50 KB",
                )
                ctx.finalize()
                return [0.0]

            if rms_db <= -60.0:
                logger.warning(f"{RENDERED_CHORDS} is silent (RMS={rms_db:.1f} dB) — gate fail")
                ctx.log_evaluation(
                    identifier="gate_audio",
                    score=0.0,
                    error=f"{RENDERED_CHORDS} silent (RMS={rms_db:.1f} dB)",
                )
                ctx.finalize()
                return [0.0]

            ctx.log_evaluation(identifier="gate_audio", score=1.0)

            # ---------------------------------------------------------------
            # Score 1: Mel-Spectrogram MSE (weight 0.90)
            # ---------------------------------------------------------------
            mel_mse = result["mel_mse"] if result["mel_mse"] is not None else float("inf")
            mel_score = result["score"]
            ctx.log_evaluation(
                identifier="mel_spectrogram",
                score=mel_score,
                mel_mse=round(mel_mse, 6),
            )
            logger.info(f"Mel-spectrogram: MSE={mel_mse:.6f} score={mel_score:.4f}")

            # ---------------------------------------------------------------
            # Score 2: Screenshot quality (weight 0.10)
            # ---------------------------------------------------------------
            prompt_quality = (
                "You are evaluating a music-production workflow screenshot.\n\n"
                "Does this screenshot provide evidence that the user has actually "
                "worked on the project — for example a DAW arrangement containing "
                "multiple tracks with audio or MIDI clips, a piano roll with notes "
                "entered, or a synthesizer plugin whose knobs/sliders are clearly "
                "not all in their default/init positions?\n"
                'Answer with ONLY "YES" or "NO".'
            )

            eval_quality = await llm_vision_judge(
                prompt=prompt_quality,
                image_bytes=screenshot_bytes,
                reference_image_bytes=None,
                return_details=True,
                max_tokens=10,
                eval_context=ctx,
                identifier="screenshot_quality",
            )
            screenshot_score = eval_quality["score"]
            logger.info(f"Screenshot quality: {screenshot_score:.1f}")

            # ---------------------------------------------------------------
            # Final weighted score
            # ---------------------------------------------------------------
            total = W_MEL * mel_score + W_SCREENSHOT * screenshot_score
            ctx.add_score(total)
            ctx.finalize(
                mel_mse=round(mel_mse, 6),
                mel_score=round(mel_score, 4),
                screenshot_score=round(screenshot_score, 1),
            )

            logger.info(f"Final score: {total:.4f}")
            return [total]

    except Exception as e:
        logger.error(f"Evaluation error: {e}")

    return [0.0]
