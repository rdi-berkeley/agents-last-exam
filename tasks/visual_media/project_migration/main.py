"""Five complete source-only Ardour migrations, scored at 20% delivery / 80% timbre."""

from dataclasses import dataclass
import json
from pathlib import Path
import shlex
import uuid

import cua_bench as cb

from tasks.common_setup import BaseTaskSetup
from tasks.linux_runtime import LinuxTaskConfig
from tasks.utils.evaluation import EvaluationContext, JudgeInfrastructureError, llm_vision_judge
from tasks.visual_media.project_migration.runtime import run_evaluator, stage


VARIANTS = [
    ("celeste_symphonic_suite",),
    ("eora",),
    ("hollow_knight_symphonic_suite",),
    ("twilight_princess_credits",),
    ("undertale_medley",),
]
W_STEM_QUALITY = 0.20
W_TIMBRE = 0.80


@dataclass
class TaskConfig(LinuxTaskConfig):
    DOMAIN_NAME: str = "visual_media"
    TASK_NAME: str = "project_migration"
    VARIANT_NAME: str = ""

    @property
    def reference_stems_dir(self):
        return f"{self.reference_dir}/stems"

    @property
    def task_description(self):
        return f"""Repair this complete original music project in Ardour on Linux.
Input: {self.input_dir}/project/project.ardour. This is a faithful source-only
conversion of the original Cubase input, with original notes (including muted
editable material), tempo, meter, controller state, routing and original plugin
state in source-state/. Missing plugin placeholders are deliberately unresolved.
Do not delete, transpose or rearrange music to make an instrument fit.
Original source clues are in project/conversion-manifest.json (plugin names,
IDs, route mappings and state-file hashes), source-state/native-dawproject.xml,
source-state/original-track-archive.xml.gz and source-state/plugins/ (original
presets/state). These describe the original input, not chosen replacements.

Inventory installed LV2/LADSPA/VST plugins and sound libraries into
{self.remote_output_dir}/available_plugins.txt. Replace unavailable active
instruments/effects with installed open-source equivalents and choose their
presets, articulations and controller adapters to retain the original sound.
Disabled effects do not require replacement. No particular plugin or GM program
is an accepted answer by itself. Evaluation compares timbre against the original
reference stems, which are provisioned separately for evaluation.
Functional equivalence does not require the original proprietary library or
identical waveforms; ordinary library variation is allowed by the public rubric.
Keep original editable event data and track/channel organization. Instrument
adapters may translate controls without deleting the original data.

Save the self-contained editable project and all media/plugin state as
{self.remote_output_dir}/migrated_project/migrated_project.ardour.
Export the COMPLETE song from time zero, at 44.1kHz or higher, as
{self.remote_output_dir}/mixdown.wav and one solo WAV per original reference
track/channel in {self.remote_output_dir}/stems/<track_name>.wav. The common
reference/delivery/native stem tap is the instrument channel output with its
Inserts/Strip (instrument and track insert effects), before downstream group or
master processing. In Ardour, export that instrument route's audio output ports.
Keep all original routing and group/master effects in the editable project and
mixdown.wav; do not remove them to make stems. Preserve complete effect tails
at each export tap; do not substitute short excerpts. The public native_audio.py
helper implements this stem tap with full=True on a separate project copy;
the stock ardour6-export master output is appropriate for the full mix.
Capture {self.remote_output_dir}/overview.png showing Ardour tracks and plugins.

Evaluation: native save/reopen, original editable music/routing, required files
and a valid full mix are delivery gates. The score is 20% complete, finite,
unclipped matched stems plus 80% timbral/performance-character similarity.
Missing/invalid stems remain in the denominator. Audio is judged against the
original references using the fixed public rubric at {self.software_dir}/rubric.txt.
Three 12-second windows are fixed from each reference's active timeline at
25/50/75% quantiles (100ms RMS bins above -60dB relative to its loudest bin;
duplicates merged). Windows are averaged per active stem, then across stems.
Each window compares both your delivered audio and a fresh native solo render of
your project against the reference in one audio request with one reference
description and two ratings; its timbre score is the lower of the two.
Constant overall gain is ignored; changing dynamics and wrong attacks, sustain,
articulation or instruments are not. Proven PCM equality/common positive gain
within analytic integer quantization bounds receives full timbre credit.
Controller setters may lose redundant messages; effective values must remain.
Native controller rounding up to 5/1920 quarter notes is reported and accepted;
bank/program, sustain and stateful-message order is retained. Evaluator runtime,
missing reference or audio-provider failures are unscored infrastructure errors.
"""

    def to_metadata(self):
        metadata = super().to_metadata()
        metadata["reference_stems_dir"] = self.reference_stems_dir
        return metadata


@cb.tasks_config(split="train")
def load():
    tasks = []
    for (tag,) in VARIANTS:
        config = TaskConfig(VARIANT_NAME=tag)
        tasks.append(
            cb.Task(
                description=config.task_description,
                metadata=config.to_metadata(),
                computer={"provider": "computer", "setup_config": {"os_type": "linux"}},
            )
        )
    return tasks


_setup = BaseTaskSetup()


@cb.setup_task(split="train")
async def start(task_cfg, session: cb.DesktopSession):
    await _setup(task_cfg, session)
    await stage(task_cfg.metadata, session)


@cb.evaluate_task(split="train")
async def evaluate(task_cfg, session: cb.DesktopSession) -> list[float]:
    try:
        from tasks.visual_media.project_migration.audio_judge import score_delivery
    except ImportError as exc:
        raise JudgeInfrastructureError(
            "Install task requirements-eval.txt on the evaluator host"
        ) from exc
    meta = task_cfg.metadata
    destination, software = await stage(meta, session, evaluating=True)
    evidence = destination + "/evidence"
    command = shlex.join(
        [
            "timeout",
            "7200",
            "python3",
            f"{software}/evaluate_remote.py",
            "--output",
            meta["remote_output_dir"],
            "--baseline",
            f"{destination}/project/project.ardour",
            "--references",
            meta["reference_stems_dir"],
            "--evidence",
            evidence,
        ]
    )
    result = await run_evaluator(session, command, evidence)
    if result.get("return_code", 1):
        raise JudgeInfrastructureError(
            "Linux migration evaluator process failed; inspect remote receipt"
        )
    assessment = json.loads(await session.read_bytes(evidence + "/assessment.json"))
    if assessment.get("infrastructure_error"):
        raise JudgeInfrastructureError(assessment["infrastructure_error"])
    async with EvaluationContext(
        task_tag=meta["variant_name"],
        mode="custom",
        output_dir=None,
        target_path=meta["remote_output_dir"],
    ) as context:
        context.log_evaluation(
            identifier="native_and_delivery_gates",
            score=float(assessment["gate_passed"]),
            assessment=assessment,
        )
        if not assessment["gate_passed"]:
            context.finalize()
            return [0.0]
        overview = await llm_vision_judge(
            prompt="Does this screenshot show Ardour with music tracks and plugin assignments visible? Answer only YES or NO. Do not require a particular theme, layout or instrument.",
            image_bytes=await session.read_bytes(meta["remote_output_dir"] + "/overview.png"),
            reference_image_bytes=None,
            return_details=True,
            max_tokens=10,
            eval_context=context,
            identifier="ardour_overview",
        )
        if overview["score"] == 0:
            context.finalize()
            return [0.0]
        root = Path(__file__).parent / ".evaluation" / uuid.uuid4().hex
        root.mkdir(parents=True)
        for stem in assessment["delivery"]["stems"]:
            for passage in stem["passages"]:
                for field in ("reference_path", "candidate_path", "native_path"):
                    if field not in passage:
                        raise JudgeInfrastructureError("Evaluator-native audio passage is missing")
                    remote = passage[field]
                    path = root / Path(remote).name
                    path.write_bytes(await session.read_bytes(remote))
                    passage[field] = str(path)
        scored = await score_delivery(assessment["delivery"], root)
        context.log_evaluation(
            identifier="audio_semantic_rubric", score=scored["weighted_score"], result=scored
        )
        context.add_score(scored["weighted_score"])
        await session.write_bytes(evidence + "/scored.json", json.dumps(scored, indent=2).encode())
        context.finalize()
        return [context.total_score]
