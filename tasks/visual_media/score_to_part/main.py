"""Score to Part — 6 active variants.

Extract individual player parts from a full orchestral score PDF.
The agent opens the PDF in notation software, isolates each printed part,
and exports per-part MIDI files and PDF scores.

Evaluation: per-part MIDI pitch/rhythm F1, dynamics correlation,
instrument assignment accuracy, and LLM-judged part PDF layout quality.

Variants: Dorico Prelude, Fugue 16, Iconica, Liebestraume, Triumphant, Unshaken.
(preces_and_responses removed — source data unrecoverable, empty shells on source VM.)
"""

import io
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
from tasks.utils.evaluation import EvaluationContext, llm_vision_judge

_setup = BaseTaskSetup()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DOMAIN_NAME = "visual_media"
TASK_NAME = "score_to_part"

FULL_SCORE_PDF = "full_score.pdf"
OVERVIEW_SCREENSHOT = "overview.png"
PARTS_SUBDIR = "parts"
REF_MIDI_SUBDIR = "midi"
REF_SCORES_SUBDIR = "scores"

# ---------------------------------------------------------------------------
# Variants — (task_tag,)
# Each variant has its own input/ and reference/ in GCS at:
#   gs://ale-data-all/visual_media/score_to_part/<task_tag>/input/   (full_score.pdf)
#   gs://ale-data-all/visual_media/score_to_part/<task_tag>/reference/midi/
#   gs://ale-data-all/visual_media/score_to_part/<task_tag>/reference/scores/
# ---------------------------------------------------------------------------
VARIANTS = [
    ("dorico_prelude",),
    ("fugue_16",),
    ("iconica",),
    ("liebestraume",),
    ("triumphant",),
    ("unshaken",),
]


async def _run_command(session, command: str, timeout: float | None = None):
    """Run a shell command across DesktopSession variants.

    Some session implementations accept `timeout=...`; others do not.
    """
    try:
        if timeout is not None:
            return await session.run_command(command, timeout=timeout)
        return await session.run_command(command)
    except TypeError:
        return await session.run_command(command)


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


def _get_tempo_bpm(midi_bytes: bytes) -> float:
    """Extract tempo from MIDI file. Returns 120 BPM as fallback."""
    try:
        midi_data = pretty_midi.PrettyMIDI(io.BytesIO(midi_bytes))
        tempos = midi_data.get_tempo_changes()[1]
        if len(tempos) > 0:
            return float(tempos[0])
    except Exception:
        pass
    return 120.0


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


def _match_agent_to_ref(
    agent_parts: dict[str, TrackInfo],
    ref_parts: dict[str, TrackInfo],
    resolution: float,
) -> list[tuple[str, TrackInfo, TrackInfo | None]]:
    """Match agent parts to reference parts.

    First tries exact filename match, then falls back to Hungarian matching
    on pitch F1 for unmatched parts.
    """
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    matched: list[tuple[str, TrackInfo, TrackInfo | None]] = []
    unmatched_ref_names: list[str] = []
    unmatched_agent_names: list[str] = []

    # Pass 1: exact filename match
    for ref_name, ref_track in ref_parts.items():
        if ref_name in agent_parts:
            matched.append((ref_name, ref_track, agent_parts[ref_name]))
        else:
            unmatched_ref_names.append(ref_name)

    for agent_name in agent_parts:
        if agent_name not in ref_parts:
            unmatched_agent_names.append(agent_name)

    # Pass 2: Hungarian matching on remaining
    if unmatched_ref_names and unmatched_agent_names:
        n_ref = len(unmatched_ref_names)
        n_agent = len(unmatched_agent_names)
        cost_matrix = np.ones((n_ref, n_agent), dtype=float)
        for i, rn in enumerate(unmatched_ref_names):
            for j, an in enumerate(unmatched_agent_names):
                f1 = compute_pitch_f1(agent_parts[an].notes, ref_parts[rn].notes, resolution)
                cost_matrix[i, j] = 1.0 - f1

        ref_indices, agent_indices = linear_sum_assignment(cost_matrix)
        matched_agent_set: set[int] = set()
        for ri, ai in zip(ref_indices, agent_indices):
            f1 = 1.0 - cost_matrix[ri, ai]
            if f1 > 0.05:  # Only accept non-trivial matches
                matched.append(
                    (
                        unmatched_ref_names[ri],
                        ref_parts[unmatched_ref_names[ri]],
                        agent_parts[unmatched_agent_names[ai]],
                    )
                )
                matched_agent_set.add(ai)
            else:
                matched.append((unmatched_ref_names[ri], ref_parts[unmatched_ref_names[ri]], None))

        # Remaining unmatched ref parts
        matched_ref_set = set(ref_indices)
        for i, rn in enumerate(unmatched_ref_names):
            if i not in matched_ref_set:
                matched.append((rn, ref_parts[rn], None))
    else:
        # Only unmatched refs remain (no agent parts to match)
        for rn in unmatched_ref_names:
            matched.append((rn, ref_parts[rn], None))

    return matched


def _pdf_to_png(pdf_bytes: bytes, page: int = 0, dpi: int = 150) -> bytes:
    """Convert the first page of a PDF to PNG bytes."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = doc[page].get_pixmap(matrix=mat)
    png_bytes = pix.tobytes("png")
    doc.close()
    return png_bytes


# ---------------------------------------------------------------------------
# Task config
# ---------------------------------------------------------------------------
@dataclass
class TaskConfig(GeneralTaskConfig):
    DOMAIN_NAME: str = "visual_media"

    TASK_NAME: str = "score_to_part"
    VARIANT_NAME: str = ""  # Set per variant

    @property
    def task_dir(self) -> str:
        """Use the canonical benchmark runtime root on Windows data disk."""
        return rf"E:\agenthle\visual_media\score_to_part\{self.VARIANT_NAME}"

    @property
    def input_dir(self) -> str:
        return rf"{self.task_dir}\input"

    @property
    def full_score_pdf_path(self) -> str:
        return rf"{self.input_dir}\{FULL_SCORE_PDF}"

    @property
    def parts_output_dir(self) -> str:
        return rf"{self.remote_output_dir}\{PARTS_SUBDIR}"

    @property
    def reference_midi_dir(self) -> str:
        return rf"{self.reference_dir}\{REF_MIDI_SUBDIR}"

    @property
    def reference_scores_dir(self) -> str:
        return rf"{self.reference_dir}\{REF_SCORES_SUBDIR}"

    @property
    def task_description(self) -> str:
        return f"""\
Goal: Extract individual player parts from a full orchestral score PDF. \
Open the PDF in music notation software, identify each printed part, and export \
each part as a separate MIDI file and PDF score. You may use any music notation \
software available on the system.

Input:
- Full orchestral score PDF: {self.full_score_pdf_path}
  This PDF contains the full score with notation, dynamics, articulations, and
  printed part names. Read the score to identify part names, tempo, key signature,
  and any part-doubling layout such as one player switching instruments. Cues like
  "to Contrabassoon" indicate a change within the same printed part rather than a
  new separate part.

Steps:
1. Open the full score PDF in music notation software (e.g. MuseScore, Dorico, Sibelius).
2. Identify all printed player parts in the score.
3. For each player part:
   a. Isolate that part using the software's part extraction/layout features, or by
      selecting the staves that belong to the same printed part.
   b. Export the part as a MIDI file to {self.parts_output_dir}\\<part_name>.mid
      - Filename: lowercase, spaces replaced with underscores (e.g. violin_i.mid, horn_f_1.mid).
      - The MIDI must contain ONLY that printed part's notes with the correct GM
        program number for the exported part.
      - If one printed part doubles instruments, keep it as a single exported part
        rather than splitting it into separate files per instrument identity. Text
        cues like "to Contrabassoon" should be treated as an in-part instrument
        change, not as a new output part.
   c. Export the part as a PDF score to {self.parts_output_dir}\\<part_name>.pdf
      - Same naming convention as the MIDI file.
4. Take a screenshot showing the notation software with the part extraction or export workflow:
   save_milestone_screenshot(path="{self.remote_output_dir}\\{OVERVIEW_SCREENSHOT}", \
description="Part extraction workflow in notation software")

Output files (all saved to {self.remote_output_dir}):
- {PARTS_SUBDIR}/<part_name>.mid: One MIDI file per printed player part.
- {PARTS_SUBDIR}/<part_name>.pdf: One PDF score per printed player part.
- {OVERVIEW_SCREENSHOT}: Screenshot of the notation software showing the workflow.
"""

    def to_metadata(self) -> dict:
        metadata = super().to_metadata()
        metadata.update(
            {
                "input_dir": self.input_dir,
                "full_score_pdf_path": self.full_score_pdf_path,
                "parts_output_dir": self.parts_output_dir,
                "reference_midi_dir": self.reference_midi_dir,
                "reference_scores_dir": self.reference_scores_dir,
            }
        )
        return metadata


# ---------------------------------------------------------------------------
# load
# ---------------------------------------------------------------------------
@cb.tasks_config(split="train")
def load():
    """Register the recovered score-to-part variants."""
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
    """Score: Gates (parts exist + screenshot + PDFs),
    then pitch 0.30 + rhythm 0.30 + dynamics 0.20 + instrument 0.10 + layout 0.10."""
    try:
        meta = task_cfg.metadata
        tag = meta["variant_name"]
        output_dir = meta["remote_output_dir"]
        ref_dir = meta["reference_dir"]
        parts_output_dir = meta["parts_output_dir"]
        ref_midi_dir = meta["reference_midi_dir"]

        if not (await session.file_exists(ref_dir) or await session.directory_exists(ref_dir)):
            logger.error(f"[{tag}] reference_dir missing: {ref_dir}")
            return [0.0]

        # Discover expected player parts from reference parts
        if not (await session.file_exists(ref_midi_dir) or await session.directory_exists(ref_midi_dir)):
            logger.error(f"[{tag}] Reference parts dir not found: {ref_midi_dir}")
            return [0.0]

        ref_files = await session.list_dir(ref_midi_dir)
        ref_midi_names = [f for f in ref_files if f.lower().endswith(".mid")]
        expected_count = len(ref_midi_names)

        if expected_count == 0:
            logger.error(f"[{tag}] No reference MIDI files found")
            return [0.0]

        logger.info(f"[{tag}] Found {expected_count} reference parts: {ref_midi_names}")

        if not (await session.file_exists(output_dir) or await session.directory_exists(output_dir)):
            logger.warning(f"Output directory not found: {output_dir}")
            return [0.0]

        async with EvaluationContext(
            task_tag=tag,
            mode="custom",
            output_dir=None,
            target_path=output_dir,
        ) as ctx:

            # -----------------------------------------------------------
            # Gate 1: parts/ directory exists with .mid files
            # -----------------------------------------------------------
            if not (await session.file_exists(parts_output_dir) or await session.directory_exists(parts_output_dir)):
                logger.warning("parts/ directory not found — gate fail")
                ctx.log_evaluation(
                    identifier="gate_parts_dir",
                    score=0.0,
                    error="parts/ directory not found",
                )
                ctx.finalize(num_output_files=0)
                return [0.0]

            output_parts_files = await session.list_dir(parts_output_dir)
            agent_midi_names = [f for f in output_parts_files if f.lower().endswith(".mid")]

            if len(agent_midi_names) == 0:
                logger.warning("No .mid files in parts/ — gate fail")
                ctx.log_evaluation(
                    identifier="gate_midi_count",
                    score=0.0,
                    error="No MIDI files found",
                )
                ctx.finalize(num_output_files=len(output_parts_files))
                return [0.0]

            if len(agent_midi_names) < expected_count * 0.5:
                logger.warning(
                    f"Only {len(agent_midi_names)}/{expected_count} MIDI files "
                    f"(<50%) — gate fail"
                )
                ctx.log_evaluation(
                    identifier="gate_midi_count",
                    score=0.0,
                    error=f"Only {len(agent_midi_names)}/{expected_count} parts",
                )
                ctx.finalize(num_output_files=len(output_parts_files))
                return [0.0]

            ctx.log_evaluation(
                identifier="gate_midi_count",
                score=1.0,
                agent_count=len(agent_midi_names),
                expected_count=expected_count,
            )

            # -----------------------------------------------------------
            # Gate 2: Screenshot shows notation software UI
            # -----------------------------------------------------------
            output_files = await session.list_dir(output_dir)
            if OVERVIEW_SCREENSHOT not in output_files:
                logger.warning(f"Screenshot {OVERVIEW_SCREENSHOT} not found — gate fail")
                ctx.log_evaluation(
                    identifier="gate_screenshot",
                    score=0.0,
                    error="Screenshot not found",
                )
                ctx.finalize(num_output_files=len(output_parts_files))
                return [0.0]

            screenshot_path = os.path.join(output_dir, OVERVIEW_SCREENSHOT)
            screenshot_bytes = await session.read_bytes(screenshot_path)

            prompt_screenshot = (
                "You are evaluating a music notation software screenshot.\n\n"
                "Does this image show a music notation software interface with "
                "score notation being edited or viewed "
                "(showing printed part staves with notes)?\n"
                "(Look for: notation staves with notes, a music notation "
                "application window, score layout elements like clefs, key "
                "signatures, time signatures.)\n"
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
                ctx.finalize(num_output_files=len(output_parts_files))
                return [0.0]

            # -----------------------------------------------------------
            # Gate 3 & 4: PDF files exist and are valid
            # -----------------------------------------------------------
            agent_pdf_names = [f for f in output_parts_files if f.lower().endswith(".pdf")]

            if len(agent_pdf_names) < expected_count * 0.5:
                logger.warning(
                    f"Only {len(agent_pdf_names)}/{expected_count} PDF files " f"(<50%) — gate fail"
                )
                ctx.log_evaluation(
                    identifier="gate_pdf_count",
                    score=0.0,
                    error=f"Only {len(agent_pdf_names)}/{expected_count} PDFs",
                )
                ctx.finalize(num_output_files=len(output_parts_files))
                return [0.0]

            for pdf_name in agent_pdf_names:
                pdf_path = os.path.join(parts_output_dir, pdf_name)
                pdf_bytes = await session.read_bytes(pdf_path)
                if len(pdf_bytes) < 1024:
                    logger.warning(f"PDF {pdf_name} too small ({len(pdf_bytes)}B)")
                    ctx.log_evaluation(
                        identifier="gate_pdf_size",
                        score=0.0,
                        error=f"{pdf_name} < 1KB",
                    )
                    ctx.finalize(num_output_files=len(output_parts_files))
                    return [0.0]

            ctx.log_evaluation(
                identifier="gate_pdf_count",
                score=1.0,
                pdf_count=len(agent_pdf_names),
            )

            # -----------------------------------------------------------
            # All gates passed — per-part MIDI scoring
            # -----------------------------------------------------------

            # Load all reference MIDIs
            ref_parts: dict[str, TrackInfo] = {}
            ref_midi_bytes_map: dict[str, bytes] = {}
            for ref_name in ref_midi_names:
                stem = os.path.splitext(ref_name)[0]
                ref_path = os.path.join(ref_midi_dir, ref_name)
                ref_bytes = await session.read_bytes(ref_path)
                ref_midi_bytes_map[stem] = ref_bytes
                tracks = _extract_tracks_full(ref_bytes)
                if tracks:
                    # Each reference file should have exactly one track
                    ref_parts[stem] = tracks[0]

            # Load all agent MIDIs
            agent_parts: dict[str, TrackInfo] = {}
            for agent_name in agent_midi_names:
                stem = os.path.splitext(agent_name)[0]
                agent_path = os.path.join(parts_output_dir, agent_name)
                agent_bytes = await session.read_bytes(agent_path)
                tracks = _extract_tracks_full(agent_bytes)
                if tracks:
                    agent_parts[stem] = tracks[0]

            # Determine quantization resolution from first reference MIDI
            first_ref_bytes = next(iter(ref_midi_bytes_map.values()))
            min_note_sec = _get_min_duration(first_ref_bytes)
            bpm = _get_tempo_bpm(first_ref_bytes)
            sixteenth_sec = (60.0 / bpm) * 0.25
            resolution_sec = min(min_note_sec, sixteenth_sec)

            # Match agent parts to reference parts
            matches = _match_agent_to_ref(agent_parts, ref_parts, resolution_sec)

            pitch_scores: list[float] = []
            rhythm_scores: list[float] = []
            dynamics_scores: list[float] = []
            instrument_correct = 0
            instrument_total = 0
            track_details: list[dict] = []

            for ref_name, ref_track, agent_track in matches:
                agent_notes = agent_track.notes if agent_track else []
                ref_notes = ref_track.notes
                agent_ccs = agent_track.control_changes if agent_track else []
                ref_ccs = ref_track.control_changes

                pitch_f1 = compute_pitch_f1(agent_notes, ref_notes, resolution_sec)
                rhythm_f1 = compute_rhythm_f1(agent_notes, ref_notes, resolution_sec)
                dynamics_corr = compute_dynamics_correlation(
                    agent_notes, ref_notes, agent_ccs, ref_ccs, resolution_sec
                )

                pitch_scores.append(pitch_f1)
                rhythm_scores.append(rhythm_f1)
                dynamics_scores.append(dynamics_corr)

                # Instrument assignment: compare GM program
                ref_program = ref_track.program
                agent_program = agent_track.program if agent_track else None
                is_correct = False
                if ref_program is not None:
                    instrument_total += 1
                    is_correct = agent_program is not None and agent_program == ref_program
                    if is_correct:
                        instrument_correct += 1

                track_details.append(
                    {
                        "ref_part": ref_name,
                        "agent_part": (agent_track.name if agent_track else "(unmatched)"),
                        "pitch_f1": round(pitch_f1, 4),
                        "rhythm_f1": round(rhythm_f1, 4),
                        "dynamics_corr": round(dynamics_corr, 4),
                        "agent_notes": len(agent_notes),
                        "ref_notes": len(ref_notes),
                        "expected_program": ref_program,
                        "actual_program": agent_program,
                        "program_correct": is_correct,
                    }
                )

            avg_pitch = sum(pitch_scores) / len(pitch_scores) if pitch_scores else 0.0
            avg_rhythm = sum(rhythm_scores) / len(rhythm_scores) if rhythm_scores else 0.0
            avg_dynamics = sum(dynamics_scores) / len(dynamics_scores) if dynamics_scores else 1.0
            instrument_score_raw = (
                instrument_correct / instrument_total if instrument_total > 0 else 0.0
            )

            # Pitch (0.30)
            pitch_score = 0.30 * avg_pitch
            ctx.add_score(pitch_score)
            ctx.log_evaluation(
                identifier="pitch_accuracy",
                score=pitch_score,
                pitch_f1=round(avg_pitch, 4),
            )

            # Rhythm (0.30)
            rhythm_score = 0.30 * avg_rhythm
            ctx.add_score(rhythm_score)
            ctx.log_evaluation(
                identifier="rhythm_accuracy",
                score=rhythm_score,
                rhythm_f1=round(avg_rhythm, 4),
            )

            # Dynamics (0.20)
            dynamics_score = 0.20 * avg_dynamics
            ctx.add_score(dynamics_score)
            ctx.log_evaluation(
                identifier="dynamics",
                score=dynamics_score,
                dynamics_corr=round(avg_dynamics, 4),
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
                f"[{tag}] Per-part MIDI comparison: "
                f"pitch_f1={avg_pitch:.4f} rhythm_f1={avg_rhythm:.4f} "
                f"dynamics={avg_dynamics:.4f} instrument={instrument_score_raw:.4f}"
            )
            for d in track_details:
                prog_status = "OK" if d.get("program_correct") else "MISS"
                logger.info(
                    f"  [{prog_status}] Ref '{d['ref_part']}' <-> Agent "
                    f"'{d['agent_part']}': pitch_f1={d['pitch_f1']:.4f}, "
                    f"rhythm_f1={d['rhythm_f1']:.4f}, "
                    f"dynamics={d['dynamics_corr']:.4f} "
                    f"(agent:{d['agent_notes']}, ref:{d['ref_notes']}) "
                    f"program: expected={d['expected_program']} "
                    f"actual={d['actual_program']}"
                )

            # Score layout quality (0.10) — LLM vision judge on part PDFs
            layout_score_raw = 0.0
            layout_checked = 0
            layout_passed = 0

            # Sample up to 3 part PDFs for layout evaluation
            pdfs_to_check = agent_pdf_names[:3]
            for pdf_name in pdfs_to_check:
                pdf_path = os.path.join(parts_output_dir, pdf_name)
                pdf_bytes = await session.read_bytes(pdf_path)

                prompt_layout = (
                    "You are evaluating a music score image.\n\n"
                    "Does this image show music notation that is related to "
                    "a printed player-part score?\n"
                    "- Contains music notation (notes, rests, staves)\n"
                    "- Has clefs, key signatures, or time signatures\n"
                    "- Appears to be a readable, properly formatted score\n"
                    "(It does not need to be perfect — just recognizable as "
                    "a music score with notation)\n"
                    'Answer with ONLY "YES" or "NO".'
                )

                pdf_png = _pdf_to_png(pdf_bytes)

                eval_layout = await llm_vision_judge(
                    prompt=prompt_layout,
                    image_bytes=pdf_png,
                    reference_image_bytes=None,
                    return_details=True,
                    max_tokens=10,
                    eval_context=ctx,
                    identifier=f"layout_{pdf_name}",
                )
                layout_checked += 1
                if eval_layout["score"] > 0.0:
                    layout_passed += 1

            if layout_checked > 0:
                layout_score_raw = layout_passed / layout_checked
            layout_score = 0.10 * layout_score_raw
            ctx.add_score(layout_score)
            ctx.log_evaluation(
                identifier="score_layout",
                score=layout_score,
                layout_rate=round(layout_score_raw, 4),
                pdfs_checked=layout_checked,
                pdfs_passed=layout_passed,
            )

            # Finalize
            ctx.finalize(
                num_expected_parts=expected_count,
                num_agent_midi=len(agent_midi_names),
                num_agent_pdf=len(agent_pdf_names),
                num_output_files=len(output_parts_files),
            )

            final = ctx.total_score
            logger.info(
                f"[{tag}] Final score: {final:.4f} "
                f"(pitch={pitch_score:.2f} rhythm={rhythm_score:.2f} "
                f"dynamics={dynamics_score:.2f} instrument={instrument_score:.2f} "
                f"layout={layout_score:.2f})"
            )
            return [final]

    except Exception as e:
        logger.error(f"Evaluation error: {e}")

    return [0.0]
