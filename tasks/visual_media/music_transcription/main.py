"""Music Transcription — 6 canonical variants.

Transcribe a recorded piece into sheet music using any music notation software,
export PDF score and MIDI with correct instrument assignments.
Evaluation: MIDI pitch/rhythm F1, dynamics correlation, instrument
assignment accuracy, and LLM-judged score layout quality.

Variants: Dorico Prelude, Fugue 16, Iconica, Liebestraume, Triumphant, Unshaken.
"""

import io
import json
import logging
import os
import sys
from dataclasses import dataclass

import fitz  # PyMuPDF

# Workaround: cua_bench loads main.py via exec_module without registering
# in sys.modules, which causes @dataclass to fail. Register ourselves.
if __name__ not in sys.modules:
    sys.modules[__name__] = sys.modules.get(__name__, type(sys)(__name__))

import cua_bench as cb
import pretty_midi

from tasks.common_config import GeneralTaskConfig
from tasks.common_setup import BaseTaskSetup
from utils.evaluation import EvaluationContext, llm_vision_judge

_setup = BaseTaskSetup()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DOMAIN_NAME = "visual_media"
TASK_NAME = "music_transcription"
REMOTE_TASK_CATEGORY = rf"{DOMAIN_NAME}\{TASK_NAME}"

TASK_BRIEF_FILE = "task_brief.json"
REFERENCE_SONG_MP3 = "reference_song.mp3"
REFERENCE_MIDI_FILE = "reference.mid"
REFERENCE_SCORE_PDF = "reference_score.pdf"

TRANSCRIPTION_PDF = "transcription.pdf"
TRANSCRIPTION_MIDI = "transcription.mid"
OVERVIEW_SCREENSHOT = "overview.png"

# ---------------------------------------------------------------------------
# Variants — (task_tag,)
# Each variant has its own input/ and reference/ in GCS at:
#   gs://ale-data-all/visual_media/music_transcription/<task_tag>/input/
#   gs://ale-data-all/visual_media/music_transcription/<task_tag>/reference/
# ---------------------------------------------------------------------------
VARIANTS = [
    ("dorico_prelude",),
    ("fugue_16",),
    ("iconica",),
    ("liebestraume",),
    ("triumphant",),
    ("unshaken",),
]


# ---------------------------------------------------------------------------
# MIDI comparison helpers
# ---------------------------------------------------------------------------
@dataclass
class TrackInfo:
    """Metadata for a single MIDI track/instrument."""

    name: str
    program: int | None  # None for drums
    is_drum: bool
    notes: list  # list[pretty_midi.Note]
    control_changes: list  # list[pretty_midi.ControlChange]


def _extract_tracks_full(midi_bytes: bytes) -> list[TrackInfo]:
    """Extract all tracks with full metadata from MIDI bytes."""
    try:
        midi_data = pretty_midi.PrettyMIDI(io.BytesIO(midi_bytes))
    except Exception as e:
        logger.warning(f"Failed to parse MIDI: {e}")
        return []

    tracks: list[TrackInfo] = []
    for inst in midi_data.instruments:
        name = inst.name.strip() if inst.name else f"Track_{inst.program}"
        if inst.is_drum:
            name = f"Drums_{name}"
        tracks.append(
            TrackInfo(
                name=name,
                program=None if inst.is_drum else int(inst.program),
                is_drum=inst.is_drum,
                notes=list(inst.notes),
                control_changes=list(inst.control_changes),
            )
        )
    return tracks


def _get_min_duration(midi_bytes: bytes) -> float:
    """Find the shortest note duration in the MIDI file for quantization."""
    try:
        midi_data = pretty_midi.PrettyMIDI(io.BytesIO(midi_bytes))
    except Exception:
        raise ValueError("Failed to parse reference MIDI bytes.")

    min_dur = float("inf")
    for inst in midi_data.instruments:
        for note in inst.notes:
            dur = note.end - note.start
            if dur > 0.01 and dur < min_dur:
                min_dur = dur

    if min_dur == float("inf"):
        raise ValueError("No valid notes found in reference MIDI.")

    return min_dur


def _quantize(t: float, resolution: float) -> float:
    """Quantize a time value to the nearest grid step."""
    return round(t / resolution) * resolution


def _note_set(notes: list[pretty_midi.Note], resolution: float) -> set[tuple[int, float]]:
    """Convert notes to a set of (pitch, quantized_onset) tuples."""
    return {(n.pitch, _quantize(n.start, resolution)) for n in notes}


def _duration_pairs(notes: list[pretty_midi.Note], resolution: float) -> list[tuple[float, float]]:
    """Return (quantized_onset, quantized_duration) pairs for rhythm comparison."""
    return [(_quantize(n.start, resolution), _quantize(n.end - n.start, resolution)) for n in notes]


def compute_pitch_f1(
    agent_notes: list[pretty_midi.Note],
    ref_notes: list[pretty_midi.Note],
    resolution: float,
) -> float:
    """F1 score on (pitch, quantized_onset) pairs."""
    if not ref_notes:
        return 1.0 if not agent_notes else 0.0
    if not agent_notes:
        return 0.0

    agent_set = _note_set(agent_notes, resolution)
    ref_set = _note_set(ref_notes, resolution)

    tp = len(agent_set & ref_set)
    precision = tp / len(agent_set) if agent_set else 0.0
    recall = tp / len(ref_set) if ref_set else 0.0

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compute_rhythm_f1(
    agent_notes: list[pretty_midi.Note],
    ref_notes: list[pretty_midi.Note],
    resolution: float,
) -> float:
    """F1 score on (quantized_onset, quantized_duration) pairs."""
    if not ref_notes:
        return 1.0 if not agent_notes else 0.0
    if not agent_notes:
        return 0.0

    agent_durations = set(_duration_pairs(agent_notes, resolution))
    ref_durations = set(_duration_pairs(ref_notes, resolution))

    tp = len(agent_durations & ref_durations)
    precision = tp / len(agent_durations) if agent_durations else 0.0
    recall = tp / len(ref_durations) if ref_durations else 0.0

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _sample_cc_curve(
    cc_events: list[tuple[float, int]],
    sample_times: list[float],
) -> list[float]:
    """Sample a CC curve at given times using step interpolation."""
    if not cc_events:
        return [64.0] * len(sample_times)
    sorted_events = sorted(cc_events, key=lambda x: x[0])
    result: list[float] = []
    cc_idx = 0
    current_val = float(sorted_events[0][1])
    for t in sample_times:
        while cc_idx < len(sorted_events) and sorted_events[cc_idx][0] <= t:
            current_val = float(sorted_events[cc_idx][1])
            cc_idx += 1
        result.append(current_val)
    return result


def compute_dynamics_correlation(
    agent_notes: list[pretty_midi.Note],
    ref_notes: list[pretty_midi.Note],
    agent_ccs: list[pretty_midi.ControlChange],
    ref_ccs: list[pretty_midi.ControlChange],
    resolution: float,
) -> float:
    """Dynamics similarity via Spearman rank correlation on velocities and CC curves."""
    import numpy as np
    from scipy.stats import spearmanr

    scores: list[float] = []

    # 1. Velocity rank correlation on matched notes
    ref_vel_map: dict[tuple[int, float], int] = {}
    for n in ref_notes:
        key = (n.pitch, _quantize(n.start, resolution))
        ref_vel_map[key] = n.velocity

    agent_vels: list[int] = []
    ref_vels: list[int] = []
    for n in agent_notes:
        key = (n.pitch, _quantize(n.start, resolution))
        if key in ref_vel_map:
            agent_vels.append(n.velocity)
            ref_vels.append(ref_vel_map[key])

    if len(agent_vels) >= 3 and len(set(ref_vels)) > 1:
        corr, _ = spearmanr(agent_vels, ref_vels)
        if not np.isnan(corr):
            scores.append(max(0.0, (corr + 1.0) / 2.0))

    # 2. CC curve correlation (CC7=volume, CC11=expression)
    for cc_num in [7, 11]:
        agent_cc = [(cc.time, cc.value) for cc in agent_ccs if cc.number == cc_num]
        ref_cc = [(cc.time, cc.value) for cc in ref_ccs if cc.number == cc_num]

        if len(ref_cc) >= 2 and len(agent_cc) >= 2:
            max_time = max(
                max(t for t, _ in agent_cc),
                max(t for t, _ in ref_cc),
            )
            if max_time <= 0:
                continue
            num_samples = max(3, min(100, int(max_time / resolution)))
            sample_times = [i * max_time / num_samples for i in range(num_samples)]
            agent_vals = _sample_cc_curve(agent_cc, sample_times)
            ref_vals = _sample_cc_curve(ref_cc, sample_times)

            if len(set(ref_vals)) > 1:
                corr, _ = spearmanr(agent_vals, ref_vals)
                if not np.isnan(corr):
                    scores.append(max(0.0, (corr + 1.0) / 2.0))

    if not scores:
        return 1.0  # No dynamics data — benefit of the doubt

    return sum(scores) / len(scores)


def _match_tracks_hungarian(
    agent_tracks: list[TrackInfo],
    ref_tracks: list[TrackInfo],
    resolution: float,
) -> list[tuple[TrackInfo, TrackInfo | None, float]]:
    """Match agent tracks to reference tracks via Hungarian algorithm on pitch F1."""
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    n_ref = len(ref_tracks)
    n_agent = len(agent_tracks)

    if n_ref == 0:
        return []
    if n_agent == 0:
        return [(ref, None, 0.0) for ref in ref_tracks]

    cost_matrix = np.ones((n_ref, n_agent), dtype=float)
    for i, ref in enumerate(ref_tracks):
        for j, agent in enumerate(agent_tracks):
            f1 = compute_pitch_f1(agent.notes, ref.notes, resolution)
            cost_matrix[i, j] = 1.0 - f1

    ref_indices, agent_indices = linear_sum_assignment(cost_matrix)

    assignment_map: dict[int, int] = {}
    for ri, ai in zip(ref_indices, agent_indices):
        assignment_map[ri] = ai

    result: list[tuple[TrackInfo, TrackInfo | None, float]] = []
    for i, ref in enumerate(ref_tracks):
        if i in assignment_map:
            ai = assignment_map[i]
            f1 = 1.0 - cost_matrix[i, ai]
            result.append((ref, agent_tracks[ai], f1))
        else:
            result.append((ref, None, 0.0))

    return result


def compare_midi_unified(
    agent_midi_bytes: bytes,
    ref_midi_bytes: bytes,
    expected_instruments: list[dict],
    resolution: float,
) -> tuple[float, float, float, float, list[dict]]:
    """Unified MIDI comparison using content-based Hungarian matching.

    Returns (avg_pitch_f1, avg_rhythm_f1, dynamics_correlation,
             instrument_assignment_score, per_track_details).
    """
    agent_tracks = _extract_tracks_full(agent_midi_bytes)
    ref_tracks = _extract_tracks_full(ref_midi_bytes)

    if not ref_tracks:
        logger.warning("Reference MIDI has no tracks")
        return 0.0, 0.0, 0.0, 0.0, []

    matched = _match_tracks_hungarian(agent_tracks, ref_tracks, resolution)

    details: list[dict] = []
    pitch_scores: list[float] = []
    rhythm_scores: list[float] = []
    dynamics_scores: list[float] = []
    instrument_correct = 0
    instrument_total = 0

    for ref_track, agent_track, match_pitch_f1 in matched:
        agent_notes = agent_track.notes if agent_track else []
        ref_notes = ref_track.notes
        agent_ccs = agent_track.control_changes if agent_track else []
        ref_ccs = ref_track.control_changes

        pitch_f1 = match_pitch_f1
        rhythm_f1 = compute_rhythm_f1(agent_notes, ref_notes, resolution)
        dynamics_corr = compute_dynamics_correlation(
            agent_notes,
            ref_notes,
            agent_ccs,
            ref_ccs,
            resolution,
        )

        pitch_scores.append(pitch_f1)
        rhythm_scores.append(rhythm_f1)
        dynamics_scores.append(dynamics_corr)

        ref_idx = ref_tracks.index(ref_track)
        exp_program = None
        if ref_idx < len(expected_instruments):
            exp_program = expected_instruments[ref_idx].get("gm_program")

        agent_program = agent_track.program if agent_track else None
        is_program_correct = False
        if exp_program is not None:
            instrument_total += 1
            is_program_correct = agent_program is not None and agent_program == exp_program
            if is_program_correct:
                instrument_correct += 1

        details.append(
            {
                "ref_track": ref_track.name,
                "agent_track": agent_track.name if agent_track else "(unmatched)",
                "pitch_f1": round(pitch_f1, 4),
                "rhythm_f1": round(rhythm_f1, 4),
                "dynamics_corr": round(dynamics_corr, 4),
                "agent_notes": len(agent_notes),
                "ref_notes": len(ref_notes),
                "expected_program": exp_program,
                "actual_program": agent_program,
                "program_correct": is_program_correct,
            }
        )

    avg_pitch = sum(pitch_scores) / len(pitch_scores) if pitch_scores else 0.0
    avg_rhythm = sum(rhythm_scores) / len(rhythm_scores) if rhythm_scores else 0.0
    avg_dynamics = sum(dynamics_scores) / len(dynamics_scores) if dynamics_scores else 1.0
    instrument_score = instrument_correct / instrument_total if instrument_total > 0 else 0.0

    return avg_pitch, avg_rhythm, avg_dynamics, instrument_score, details


# ---------------------------------------------------------------------------
# Task config
# ---------------------------------------------------------------------------
@dataclass
class TaskConfig(GeneralTaskConfig):
    DOMAIN_NAME: str = "visual_media"

    TASK_NAME: str = "music_transcription"
    VARIANT_NAME: str = ""  # Set per variant

    @property
    def input_dir(self) -> str:
        return rf"{self.task_dir}\input"

    @property
    def task_brief_path(self) -> str:
        return rf"{self.input_dir}\{TASK_BRIEF_FILE}"

    @property
    def reference_song_mp3_path(self) -> str:
        return rf"{self.input_dir}\{REFERENCE_SONG_MP3}"

    @property
    def reference_midi_path(self) -> str:
        return rf"{self.reference_dir}\{REFERENCE_MIDI_FILE}"

    @property
    def reference_score_pdf_path(self) -> str:
        return rf"{self.reference_dir}\{REFERENCE_SCORE_PDF}"

    @property
    def task_description(self) -> str:
        return f"""\
Goal: Transcribe a recorded piece into musical notation, export both a PDF score and a MIDI file with correct instrument assignments. You may use any music notation software available on the system.

Read the task specification from task_brief.json at {self.task_brief_path}. It contains:
- title: the title of the piece
- composer: the composer of the piece
- tempo_bpm: the tempo of the song
- instruments: list of instruments to transcribe, each with name, clef, and GM program number

Input audio files are available in the input directory:
- {self.reference_song_mp3_path} — MP3 recording of the target song

Steps:
1. Listen to the reference audio to identify all instrumental parts.
2. Open a music notation software and create a new project.
3. Fill in the Title and Composer from task_brief.json in the project info.
4. Add players/instruments matching those listed in task_brief.json.
5. Configure the time signature, key signature, and tempo marking based on what you hear.
6. Transcribe all notes, rhythms, dynamics, articulations, and expression markings for each part.
7. Format the score layout professionally (proper spacing, alignment, readable note density).
8. Configure playback: assign each instrument to the correct General MIDI (GM) program number as specified in task_brief.json.
9. Export outputs:
   - Export a PDF of the full score. Save it as {self.remote_output_dir}\\{TRANSCRIPTION_PDF}
   - Export MIDI: save as {self.remote_output_dir}\\{TRANSCRIPTION_MIDI}
10. Take ONE final screenshot showing the notation software with the score visible (showing some instrument staves):
    save_milestone_screenshot(path="{self.remote_output_dir}\\{OVERVIEW_SCREENSHOT}", description="Overview of the transcribed score")

Output files (all saved to {self.remote_output_dir}):
- {TRANSCRIPTION_PDF}: Exported PDF of the complete score with all parts.
- {TRANSCRIPTION_MIDI}: Exported MIDI file. Each track must have correct MIDI Program Change messages matching the GM program numbers in task_brief.json.
- {OVERVIEW_SCREENSHOT}: Screenshot of the notation software showing the transcribed score with instrument staves visible.
"""

    def to_metadata(self) -> dict:
        metadata = super().to_metadata()
        metadata.update(
            {
                "input_dir": self.input_dir,
                "task_brief_path": self.task_brief_path,
                "reference_song_mp3_path": self.reference_song_mp3_path,
                "reference_midi_path": self.reference_midi_path,
                "reference_score_pdf_path": self.reference_score_pdf_path,
            }
        )
        return metadata


# ---------------------------------------------------------------------------
# load
# ---------------------------------------------------------------------------
@cb.tasks_config(split="train")
def load():
    """Register the recovered music transcription variants."""
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


def _pdf_to_png(pdf_bytes: bytes, page: int = 0, dpi: int = 150) -> bytes:
    """Convert the first page of a PDF to PNG bytes."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = doc[page].get_pixmap(matrix=mat)
    png_bytes = pix.tobytes("png")
    doc.close()
    return png_bytes


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------
@cb.evaluate_task(split="train")
async def evaluate(task_cfg, session: cb.DesktopSession) -> list[float]:
    """Score: Gates (PDF + screenshot + MIDI exist),
    then pitch 0.30 + rhythm 0.30 + dynamics 0.20 + instrument 0.10 + layout 0.10."""
    try:
        meta = task_cfg.metadata
        tag = meta["variant_name"]
        output_dir = meta["remote_output_dir"]
        task_brief_path = meta["task_brief_path"]
        reference_midi_path = meta["reference_midi_path"]
        reference_score_pdf_path = meta["reference_score_pdf_path"]

        # Read task brief. Reference fixtures are pre-staged on the VM
        # under `reference/` by Stage 1; `evaluate()` does not fetch them.
        brief_bytes = await session.read_bytes(task_brief_path)
        brief = json.loads(brief_bytes.decode("utf-8"))
        instruments = brief["instruments"]
        song_title = brief.get("title", "")
        song_composer = brief.get("composer", "")

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
            # Gate 1: PDF export
            # -----------------------------------------------------------
            if TRANSCRIPTION_PDF not in output_files:
                logger.warning(f"PDF file {TRANSCRIPTION_PDF} not found — gate fail")
                ctx.log_evaluation(identifier="gate_pdf", score=0.0, error="PDF not found")
                ctx.finalize(num_output_files=len(output_files))
                return [0.0]

            pdf_path = os.path.join(output_dir, TRANSCRIPTION_PDF)
            pdf_bytes = await session.read_bytes(pdf_path)
            if len(pdf_bytes) < 1024:
                logger.warning("PDF file too small — likely invalid")
                ctx.log_evaluation(identifier="gate_pdf", score=0.0, error="PDF too small")
                ctx.finalize(num_output_files=len(output_files))
                return [0.0]

            ctx.log_evaluation(identifier="gate_pdf", score=1.0)

            # -----------------------------------------------------------
            # Gate 2: Screenshot shows notation software UI
            # -----------------------------------------------------------
            if OVERVIEW_SCREENSHOT not in output_files:
                logger.warning(f"Screenshot {OVERVIEW_SCREENSHOT} not found — gate fail")
                ctx.log_evaluation(
                    identifier="gate_screenshot",
                    score=0.0,
                    error="Screenshot not found",
                )
                ctx.finalize(num_output_files=len(output_files))
                return [0.0]

            screenshot_path = os.path.join(output_dir, OVERVIEW_SCREENSHOT)
            screenshot_bytes = await session.read_bytes(screenshot_path)

            prompt_screenshot = (
                "You are evaluating a music notation software screenshot.\n\n"
                "Does this image show a music notation software interface with "
                "professional score notation actively being edited or viewed "
                "(showing instrument staves with notes)?\n"
                "(Look for: notation staves with notes, a music notation "
                "application window, score layout elements like clefs, key "
                "signatures, time signatures. It does not need to show the "
                "entire page, just a clear view of the working score)\n"
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
                logger.warning("Screenshot gate failed — does not show notation software")
                ctx.finalize(num_output_files=len(output_files))
                return [0.0]

            # -----------------------------------------------------------
            # Gate 3: MIDI file exists
            # -----------------------------------------------------------
            if TRANSCRIPTION_MIDI not in output_files:
                logger.warning(f"MIDI file {TRANSCRIPTION_MIDI} not found")
                ctx.log_evaluation(identifier="midi_missing", score=0.0, error="MIDI not found")
                ctx.finalize(num_output_files=len(output_files))
                return [0.0]

            # -----------------------------------------------------------
            # All gates passed — scoring
            # -----------------------------------------------------------
            agent_midi_path = os.path.join(output_dir, TRANSCRIPTION_MIDI)
            agent_midi_bytes = await session.read_bytes(agent_midi_path)

            ref_midi_bytes = None
            if await session.exists(reference_midi_path):
                ref_midi_bytes = await session.read_bytes(reference_midi_path)

            pitch_score_raw = 0.0
            rhythm_score_raw = 0.0
            dynamics_score_raw = 0.0
            instrument_score_raw = 0.0
            track_details: list[dict] = []

            if ref_midi_bytes:
                min_note_sec = _get_min_duration(ref_midi_bytes)

                if "tempo_bpm" not in brief:
                    raise ValueError("Task brief must contain 'tempo_bpm'")
                bpm = brief["tempo_bpm"]

                sixteenth_sec = (60.0 / bpm) * 0.25
                resolution_sec = min(min_note_sec, sixteenth_sec)

                (
                    avg_pitch_f1,
                    avg_rhythm_f1,
                    avg_dynamics,
                    inst_score,
                    track_details,
                ) = compare_midi_unified(
                    agent_midi_bytes,
                    ref_midi_bytes,
                    expected_instruments=instruments,
                    resolution=resolution_sec,
                )
                pitch_score_raw = avg_pitch_f1
                rhythm_score_raw = avg_rhythm_f1
                dynamics_score_raw = avg_dynamics
                instrument_score_raw = inst_score

            # Pitch (0.30)
            pitch_score = 0.30 * pitch_score_raw
            ctx.add_score(pitch_score)
            ctx.log_evaluation(
                identifier="pitch_accuracy",
                score=pitch_score,
                pitch_f1=round(pitch_score_raw, 4),
            )

            # Rhythm (0.30)
            rhythm_score = 0.30 * rhythm_score_raw
            ctx.add_score(rhythm_score)
            ctx.log_evaluation(
                identifier="rhythm_accuracy",
                score=rhythm_score,
                rhythm_f1=round(rhythm_score_raw, 4),
            )

            # Dynamics (0.20)
            dynamics_score = 0.20 * dynamics_score_raw
            ctx.add_score(dynamics_score)
            ctx.log_evaluation(
                identifier="dynamics",
                score=dynamics_score,
                dynamics_corr=round(dynamics_score_raw, 4),
            )

            # Instrument assignment (0.10)
            instrument_score = 0.10 * instrument_score_raw
            ctx.add_score(instrument_score)
            ctx.log_evaluation(
                identifier="instrument_assignment",
                score=instrument_score,
                assignment_rate=round(instrument_score_raw, 4),
            )

            # Per-track detail logging
            logger.info(
                f"MIDI comparison (Hungarian): pitch_f1={pitch_score_raw:.4f} "
                f"rhythm_f1={rhythm_score_raw:.4f} dynamics={dynamics_score_raw:.4f} "
                f"instrument_rate={instrument_score_raw:.4f}"
            )
            for d in track_details:
                prog_status = "OK" if d.get("program_correct") else "MISS"
                logger.info(
                    f"  [{prog_status}] Ref '{d['ref_track']}' <-> Agent "
                    f"'{d['agent_track']}': pitch_f1={d['pitch_f1']:.4f}, "
                    f"rhythm_f1={d['rhythm_f1']:.4f}, dynamics={d['dynamics_corr']:.4f} "
                    f"(agent:{d['agent_notes']}, ref:{d['ref_notes']}) "
                    f"program: expected={d['expected_program']} "
                    f"actual={d['actual_program']}"
                )

            # Score layout quality (0.10) — LLM vision judge
            layout_score = 0.0
            if await session.exists(reference_score_pdf_path):
                ref_pdf_bytes = await session.read_bytes(reference_score_pdf_path)

                prompt_layout = (
                    "You are evaluating a music score PDF.\n\n"
                    "Compare these two music score images:\n"
                    "1. First image: The agent's exported PDF score.\n"
                    "2. Second image: The reference score.\n\n"
                    "Evaluate the professional engraving quality of the first image:\n"
                    "- Proper spacing between staves\n"
                    "- Balanced page layout\n"
                    "- Readable note density\n"
                    "- Clean alignment of barlines, notes, and text\n"
                    "- Appropriate use of clefs, key signatures, and time signatures\n"
                    f'- The score should contain a title related to "{song_title}" '
                    f'and a composer related to "{song_composer}" '
                    f"(variations in formatting, full names, or subtitles are acceptable).\n\n"
                    "Does the first image show professional-quality score layout "
                    "comparable to the reference?\n"
                    'Answer with ONLY "YES" or "NO".'
                )

                agent_pdf_png = _pdf_to_png(pdf_bytes)
                ref_pdf_png = _pdf_to_png(ref_pdf_bytes)

                eval_layout = await llm_vision_judge(
                    prompt=prompt_layout,
                    image_bytes=agent_pdf_png,
                    reference_image_bytes=ref_pdf_png,
                    return_details=True,
                    max_tokens=10,
                    eval_context=ctx,
                    identifier="score_layout",
                )
                layout_score = 0.10 * eval_layout["score"]
            else:
                prompt_layout_single = (
                    "You are evaluating a music score PDF.\n\n"
                    "Does this score show professional engraving quality?\n"
                    "- Proper spacing between staves\n"
                    "- Balanced page layout\n"
                    "- Readable note density\n"
                    "- Clean alignment of barlines, notes, and text\n"
                    "- Appropriate clefs, key signatures, and time signatures\n"
                    f'- The score should contain a title related to "{song_title}" '
                    f'and a composer related to "{song_composer}" '
                    f"(variations in formatting, full names, or subtitles are acceptable).\n\n"
                    "Does it meet professional engraving standards?\n"
                    'Answer with ONLY "YES" or "NO".'
                )

                agent_pdf_png_single = _pdf_to_png(pdf_bytes)

                eval_layout = await llm_vision_judge(
                    prompt=prompt_layout_single,
                    image_bytes=agent_pdf_png_single,
                    reference_image_bytes=None,
                    return_details=True,
                    max_tokens=10,
                    eval_context=ctx,
                    identifier="score_layout",
                )
                layout_score = 0.10 * eval_layout["score"]

            ctx.add_score(layout_score)

            # Finalize
            ctx.finalize(
                num_instruments=len(instruments),
                num_output_files=len(output_files),
            )

            logger.info(
                f"Final score: {ctx.total_score:.4f} "
                f"(pitch={pitch_score:.2f} rhythm={rhythm_score:.2f} "
                f"dynamics={dynamics_score:.2f} instrument={instrument_score:.2f} "
                f"layout={layout_score:.2f})"
            )
            return [ctx.total_score]

    except Exception as e:
        logger.error(f"Evaluation error: {e}")

    return [0.0]
